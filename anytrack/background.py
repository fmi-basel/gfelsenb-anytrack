from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Sequence, Dict, Any
import numpy as np
import cv2
import math

@dataclass
class BackgroundModel:
    image: np.ndarray  # uint8 grayscale background


# Debug utilities
def _resize_max_side(gray: np.ndarray, max_side: int) -> np.ndarray:
    if max_side is None or max_side <= 0:
        return gray
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return gray
    s = max_side / float(m)
    nh = max(1, int(round(h * s)))
    nw = max(1, int(round(w * s)))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)


def _emit_debug(
    debug_hook: Optional[Callable[[str, Dict[str, Any]], None]],
    event: str,
    payload: Dict[str, Any],
):
    if debug_hook is None:
        return
    try:
        debug_hook(event, payload)
    except Exception:
        # Debugging should never break background building
        return

def _read_gray(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def _read_gray_small(cap: cv2.VideoCapture, idx: int, max_side: int = 96) -> Optional[np.ndarray]:
    """Read frame idx, convert to grayscale, and downsample so max(h,w)==max_side."""
    g = _read_gray(cap, idx)
    if g is None:
        return None
    h, w = g.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return g
    scale = max_side / float(m)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)

def _sample_indices(n_frames: int, k: int) -> np.ndarray:
    if k <= 1:
        return np.array([0], dtype=int)
    return np.linspace(0, max(0, n_frames - 1), k).astype(int)


def _pca2(x: np.ndarray) -> np.ndarray:
    """Project rows of x to 2D using PCA via SVD. Returns (n,2)."""
    # x: (n, d)
    x = x.astype(np.float32, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    # SVD on the (n,d) matrix; take first 2 right-singular vectors
    # For small n, full_matrices=False is efficient.
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    pcs = vt[:2].T  # (d,2)
    return x @ pcs


def _kmeans_pp_init(x2: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ initialization. Returns centers (k,2)."""
    n = x2.shape[0]
    centers = np.empty((k, 2), dtype=np.float32)
    # first center
    centers[0] = x2[rng.integers(0, n)]
    # distances to nearest center
    d2 = np.sum((x2 - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        # choose next center proportional to distance^2
        total = float(d2.sum())
        if total <= 1e-12:
            centers[i] = x2[rng.integers(0, n)]
            continue
        probs = d2 / total
        idx = int(rng.choice(n, p=probs))
        centers[i] = x2[idx]
        d2 = np.minimum(d2, np.sum((x2 - centers[i]) ** 2, axis=1))
    return centers


def _kmeans2d(
    x2: np.ndarray,
    k: int,
    iters: int = 50,
    restarts: int = 2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run k-means in 2D. Returns (centers, labels)."""
    x2 = x2.astype(np.float32, copy=False)
    n = x2.shape[0]
    k = int(max(1, min(k, n)))
    best_inertia = float("inf")
    best_centers = None
    best_labels = None

    base_rng = np.random.default_rng(seed)
    for _ in range(max(1, restarts)):
        rng = np.random.default_rng(int(base_rng.integers(0, 2**31 - 1)))
        centers = _kmeans_pp_init(x2, k, rng)

        labels = np.zeros((n,), dtype=np.int32)
        for _it in range(max(1, iters)):
            # assign
            dists = ((x2[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (n,k)
            new_labels = dists.argmin(axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels) and _it > 0:
                break
            labels = new_labels

            # update
            for j in range(k):
                m = labels == j
                if not np.any(m):
                    # re-seed empty cluster
                    centers[j] = x2[rng.integers(0, n)]
                else:
                    centers[j] = x2[m].mean(axis=0)

        inertia = float(((x2 - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
            best_labels = labels.copy()

    return best_centers, best_labels


def pca_kmeans_indices(
    cap: cv2.VideoCapture,
    n_frames: int,
    k: int,
    max_candidates: int = 400,
    max_side: int = 96,
    iters: int = 50,
    restarts: int = 2,
    seed: int = 0,
) -> np.ndarray:
    """Select k diverse frame indices using PCA(2D)+k-means on downsampled images.

    Steps:
      1) choose up to `max_candidates` candidate frames uniformly across the video
      2) embed each candidate as a flattened, downsampled grayscale vector
      3) PCA -> 2D
      4) k-means -> k clusters
      5) pick the candidate closest to each cluster center
    """
    k = int(max(1, k))
    n = int(max(1, n_frames))

    cand_n = int(min(max_candidates, n))
    cand_idxs = _sample_indices(n, cand_n)

    vecs = []
    kept = []
    for idx in cand_idxs:
        g = _read_gray_small(cap, int(idx), max_side=max_side)
        if g is None:
            continue
        v = g.astype(np.float32).ravel()
        # normalize per-image brightness to reduce illumination dominance
        v = v - float(v.mean())
        s = float(v.std())
        if s > 1e-6:
            v = v / s
        vecs.append(v)
        kept.append(int(idx))

    if len(kept) == 0:
        return _sample_indices(n, k)
    if len(kept) <= k:
        return np.array(sorted(set(kept)), dtype=int)

    X = np.stack(vecs, axis=0)
    X2 = _pca2(X)  # (m,2)

    centers, labels = _kmeans2d(X2, k=k, iters=iters, restarts=restarts, seed=seed)

    # For each center, pick closest point (ensure uniqueness)
    chosen = []
    used = set()
    dmat = ((X2[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (m,k)
    for j in range(centers.shape[0]):
        order = np.argsort(dmat[:, j])
        pick = None
        for ii in order:
            idx = kept[int(ii)]
            if idx not in used:
                pick = idx
                break
        if pick is not None:
            used.add(pick)
            chosen.append(pick)

    # If duplicates reduced count, fill with farthest-from-selected frames
    if len(chosen) < min(k, len(kept)):
        chosen_set = set(chosen)
        # score candidates by distance to nearest chosen (in PCA space)
        if len(chosen) == 0:
            remaining = kept
        else:
            chosen_mask = np.array([idx in chosen_set for idx in kept], dtype=bool)
            sel = X2[chosen_mask]
            rem = X2[~chosen_mask]
            rem_idxs = [idx for idx in kept if idx not in chosen_set]
            # nearest distance
            dnear = ((rem[:, None, :] - sel[None, :, :]) ** 2).sum(axis=2).min(axis=1)
            order = np.argsort(-dnear)  # farthest first
            remaining = [rem_idxs[int(i)] for i in order]
        for idx in remaining:
            if len(chosen) >= min(k, len(kept)):
                break
            if idx not in chosen_set:
                chosen.append(idx)
                chosen_set.add(idx)

    return np.array(sorted(chosen), dtype=int)

def mean_background(
    cap: cv2.VideoCapture,
    n_frames: int,
    k: int,
    idxs: Optional[np.ndarray] = None,
    debug_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    debug_stride: int = 1,
    debug_max_side: int = 0,
) -> np.ndarray:
    idxs = _sample_indices(n_frames, k) if idxs is None else np.asarray(idxs, dtype=int)
    acc = None
    count = 0
    for i in idxs:
        g = _read_gray(cap, int(i))
        if g is None:
            continue
        if acc is None:
            acc = g.astype(np.float32)
        else:
            acc += g.astype(np.float32)
        count += 1
        if debug_hook is not None and (count == 1 or count % max(1, int(debug_stride)) == 0):
            bg_est = (acc / count).clip(0, 255).astype(np.uint8)
            _emit_debug(
                debug_hook,
                "frame",
                {
                    "method": "mean",
                    "k": int(k),
                    "step": int(count),
                    "steps": int(len(idxs)),
                    "frame_idx": int(i),
                    "frame": _resize_max_side(g, int(debug_max_side)) if debug_max_side else g,
                    "bg": _resize_max_side(bg_est, int(debug_max_side)) if debug_max_side else bg_est,
                },
            )
    if acc is None or count == 0:
        raise RuntimeError("Failed to build background: no frames read.")
    bg = (acc / count).clip(0, 255).astype(np.uint8)
    return bg

def median_background(
    cap: cv2.VideoCapture,
    n_frames: int,
    k: int,
    idxs: Optional[np.ndarray] = None,
    debug_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    debug_stride: int = 1,
    debug_max_side: int = 0,
) -> np.ndarray:
    idxs = _sample_indices(n_frames, k) if idxs is None else np.asarray(idxs, dtype=int)
    frames = []
    for i in idxs:
        g = _read_gray(cap, int(i))
        if g is not None:
            frames.append(g)
            if debug_hook is not None and (len(frames) == 1 or len(frames) % max(1, int(debug_stride)) == 0):
                # quick proxy during build: mean of collected frames
                tmp = (np.mean(np.stack(frames, axis=0).astype(np.float32), axis=0)).clip(0, 255).astype(np.uint8)
                _emit_debug(
                    debug_hook,
                    "frame",
                    {
                        "method": "median",
                        "k": int(k),
                        "step": int(len(frames)),
                        "steps": int(len(idxs)),
                        "frame_idx": int(i),
                        "frame": _resize_max_side(g, int(debug_max_side)) if debug_max_side else g,
                        "bg": _resize_max_side(tmp, int(debug_max_side)) if debug_max_side else tmp,
                    },
                )
    if not frames:
        raise RuntimeError("Failed to build background: no frames read.")
    stack = np.stack(frames, axis=0)
    bg = np.median(stack, axis=0).astype(np.uint8)
    return bg

def mean_excluding_foreground(
    cap: cv2.VideoCapture,
    n_frames: int,
    k: int,
    fg_thresh: int = 25,
    init_k: int = 11,
    idxs: Optional[np.ndarray] = None,
    debug_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    debug_stride: int = 1,
    debug_max_side: int = 0,
) -> np.ndarray:
    # Build iteratively: initialize bg robustly (median of a small subset), then refine while
    # excluding large-difference pixels from accumulation.
    idxs = _sample_indices(n_frames, k) if idxs is None else np.asarray(idxs, dtype=int)
    if len(idxs) == 0:
        raise RuntimeError("No indices provided for background build.")

    # --- Robust init (Option A): median of a small, evenly-spaced subset ---
    init_k_eff = int(max(1, min(int(init_k), len(idxs))))
    if init_k_eff == 1:
        init_pos = np.array([0], dtype=int)
    else:
        init_pos = np.linspace(0, len(idxs) - 1, init_k_eff).astype(int)
    init_idxs = [int(idxs[p]) for p in init_pos]
    init_set = set(init_idxs)

    init_frames = []
    first_init_frame = None
    for fi in init_idxs:
        g = _read_gray(cap, fi)
        if g is None:
            continue
        if first_init_frame is None:
            first_init_frame = g
        init_frames.append(g)

    if not init_frames:
        # Fallback: try the first index
        g0 = _read_gray(cap, int(idxs[0]))
        if g0 is None:
            raise RuntimeError("Failed to read any frames for background init.")
        bg0 = g0.astype(np.uint8)
        first_init_frame = g0
    else:
        bg0 = np.median(np.stack(init_frames, axis=0), axis=0).astype(np.uint8)

    # Start accumulators from bg0 so it contributes as one pseudo-frame (not an over-weighted first frame).
    acc = bg0.astype(np.float32)
    cnt = np.ones_like(bg0, dtype=np.float32)
    bg = bg0.copy()

    # Emit init info for debugging
    _emit_debug(debug_hook, "init", {"init_k": int(init_k_eff), "init_indices": [int(x) for x in init_idxs]})

    # Debug: show initial model
    if debug_hook is not None:
        _emit_debug(
            debug_hook,
            "frame",
            {
                "method": "mean_excluding_fg",
                "k": int(k),
                "step": 1,
                "steps": int(1 + (len(idxs) - len(init_set))),
                "frame_idx": int(init_idxs[0]) if init_idxs else int(idxs[0]),
                "frame": _resize_max_side(first_init_frame if first_init_frame is not None else bg, int(debug_max_side)) if debug_max_side else (first_init_frame if first_init_frame is not None else bg),
                "bg": _resize_max_side(bg, int(debug_max_side)) if debug_max_side else bg,
            },
        )

    # --- Refinement: iterate over remaining sampled frames ---
    step = 1
    steps_total = int(1 + (len(idxs) - len(init_set)))

    for i in idxs:
        ii = int(i)
        if ii in init_set:
            continue
        g = _read_gray(cap, ii)
        if g is None:
            continue
        diff = cv2.absdiff(g, bg)
        mask = (diff < fg_thresh).astype(np.float32)  # 1 where likely background
        acc += g.astype(np.float32) * mask
        cnt += mask
        bg = (acc / np.maximum(cnt, 1.0)).clip(0, 255).astype(np.uint8)

        step += 1
        if debug_hook is not None and step % max(1, int(debug_stride)) == 0:
            _emit_debug(
                debug_hook,
                "frame",
                {
                    "method": "mean_excluding_fg",
                    "k": int(k),
                    "step": int(step),
                    "steps": int(steps_total),
                    "frame_idx": int(ii),
                    "frame": _resize_max_side(g, int(debug_max_side)) if debug_max_side else g,
                    "bg": _resize_max_side(bg, int(debug_max_side)) if debug_max_side else bg,
                },
            )

    return bg

def mog2_background(
    cap: cv2.VideoCapture,
    n_frames: int,
    k: int,
    history: int = 200,
    idxs: Optional[np.ndarray] = None,
    debug_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    debug_stride: int = 1,
    debug_max_side: int = 0,
) -> np.ndarray:
    # Use MOG2 to estimate background; sampling k frames spaced across video.
    idxs = _sample_indices(n_frames, k) if idxs is None else np.asarray(idxs, dtype=int)
    bs = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=16, detectShadows=False)
    last = None
    for i in idxs:
        g = _read_gray(cap, int(i))
        if g is None:
            continue
        last = g
        bs.apply(g, learningRate=0.01)
        if debug_hook is not None:
            # use current internal bg if available
            bg_est = bs.getBackgroundImage()
            if bg_est is None:
                bg_est = g
            if bg_est.ndim == 3:
                bg_est = cv2.cvtColor(bg_est, cv2.COLOR_BGR2GRAY)
            step = int(np.where(idxs == i)[0][0]) + 1 if hasattr(idxs, "__len__") else 0
            if step % max(1, int(debug_stride)) == 0:
                _emit_debug(
                    debug_hook,
                    "frame",
                    {
                        "method": "mog2",
                        "k": int(k),
                        "step": int(step),
                        "steps": int(len(idxs)),
                        "frame_idx": int(i),
                        "frame": _resize_max_side(g, int(debug_max_side)) if debug_max_side else g,
                        "bg": _resize_max_side(bg_est.astype(np.uint8), int(debug_max_side)) if debug_max_side else bg_est.astype(np.uint8),
                    },
                )
    if last is None:
        raise RuntimeError("Failed to build MOG2 background: no frames read.")
    bg = bs.getBackgroundImage()
    if bg is None:
        # fallback
        return mean_background(cap, n_frames, k, idxs)
    if bg.ndim == 3:
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    return bg.astype(np.uint8)

def choose_k_by_convergence(
    cap: cv2.VideoCapture,
    n_frames: int,
    k_min: int,
    k_max: int,
    eps: float,
    builder: Callable[[cv2.VideoCapture, int, int, Optional[np.ndarray], Optional[Callable[[str, Dict[str, Any]], None]], int, int], np.ndarray],
    indices_selector: Optional[Callable[[cv2.VideoCapture, int, int], np.ndarray]] = None,
    debug_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    debug_stride: int = 1,
    debug_max_side: int = 0,
) -> Tuple[np.ndarray, int, float]:
    # Increase k until the background stabilizes.
    prev = None
    chosen = k_min
    last_diff = float("inf")
    for k in range(k_min, k_max + 1, max(5, (k_max - k_min)//10 or 5)):
        idxs = indices_selector(cap, n_frames, k) if indices_selector is not None else None
        _emit_debug(
            debug_hook,
            "trial",
            {"k": int(k), "indices": [int(x) for x in np.asarray(idxs if idxs is not None else _sample_indices(n_frames, k), dtype=int).tolist()]},
        )
        bg = builder(cap, n_frames, k, idxs, debug_hook=debug_hook, debug_stride=debug_stride, debug_max_side=debug_max_side)
        if prev is not None:
            last_diff = float(np.mean(np.abs(bg.astype(np.float32) - prev.astype(np.float32))))
            if last_diff < eps:
                chosen = k
                return bg, chosen, last_diff
        prev = bg
        chosen = k
    return prev, chosen, last_diff

def remove_ghosts_inpaint(bg: np.ndarray, area_min: int = 20, area_max: int = 600) -> np.ndarray:
    # Heuristic: ghost flies often appear as small sharp blobs in the background.
    # Detect them as high-frequency structures relative to a blurred version, then inpaint.
    blur = cv2.GaussianBlur(bg, (0, 0), 3)
    hf = cv2.absdiff(bg, blur)
    _, m = cv2.threshold(hf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(bg, dtype=np.uint8)
    for c in cnts:
        a = cv2.contourArea(c)
        if area_min <= a <= area_max:
            cv2.drawContours(mask, [c], -1, 255, -1)
    if mask.max() == 0:
        return bg
    out = cv2.inpaint(bg, mask, 3, cv2.INPAINT_TELEA)
    return out

def build_background(
    video_path: str,
    method: str = "mean_excluding_fg",
    k_min: int = 30,
    k_max: int = 300,
    converge_eps: float = 2.5,
    fg_thresh: int = 25,
    inpaint_ghosts: bool = True,
    ghost_area_min: int = 20,
    ghost_area_max: int = 600,
    sampling: str = "uniform",  # "uniform" or "pca_kmeans"
    pca_max_candidates: int = 400,
    pca_max_side: int = 96,
    kmeans_iters: int = 50,
    kmeans_restarts: int = 2,
    kmeans_seed: int = 0,
    debug: bool = False,
    debug_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    debug_stride: int = 1,
    debug_max_side: int = 0,
) -> BackgroundModel:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n_frames <= 0:
        cap.release()
        raise RuntimeError("Video has 0 frames (or could not be probed).")

    sampling = (sampling or "uniform").lower()
    if sampling == "pca_kmeans":
        def indices_selector(c: cv2.VideoCapture, nf: int, kk: int) -> np.ndarray:
            return pca_kmeans_indices(
                c,
                nf,
                kk,
                max_candidates=pca_max_candidates,
                max_side=pca_max_side,
                iters=kmeans_iters,
                restarts=kmeans_restarts,
                seed=kmeans_seed,
            )
    else:
        indices_selector = None

    if not debug:
        debug_hook = None

    if method == "mean":
        builder = mean_background
        bg, k, _ = choose_k_by_convergence(
            cap, n_frames, k_min, k_max, converge_eps, builder,
            indices_selector=indices_selector,
            debug_hook=debug_hook, debug_stride=debug_stride, debug_max_side=debug_max_side,
        )
    elif method == "median":
        builder = median_background
        bg, k, _ = choose_k_by_convergence(
            cap, n_frames, k_min, k_max, converge_eps, builder,
            indices_selector=indices_selector,
            debug_hook=debug_hook, debug_stride=debug_stride, debug_max_side=debug_max_side,
        )
    elif method == "mog2":
        def builder2(c, n, k, idxs, debug_hook=None, debug_stride=1, debug_max_side=0):
            return mog2_background(c, n, k, idxs=idxs, debug_hook=debug_hook, debug_stride=debug_stride, debug_max_side=debug_max_side)
        bg, k, _ = choose_k_by_convergence(
            cap, n_frames, k_min, k_max, converge_eps, builder2,
            indices_selector=indices_selector,
            debug_hook=debug_hook, debug_stride=debug_stride, debug_max_side=debug_max_side,
        )
    else:
        def builder3(c, n, k, idxs, debug_hook=None, debug_stride=1, debug_max_side=0):
            return mean_excluding_foreground(
                c,
                n,
                k,
                fg_thresh=fg_thresh,
                init_k=11,
                idxs=idxs,
                debug_hook=debug_hook,
                debug_stride=debug_stride,
                debug_max_side=debug_max_side,
            )
        bg, k, _ = choose_k_by_convergence(
            cap, n_frames, k_min, k_max, converge_eps, builder3,
            indices_selector=indices_selector,
            debug_hook=debug_hook, debug_stride=debug_stride, debug_max_side=debug_max_side,
        )

    cap.release()

    if inpaint_ghosts:
        bg = remove_ghosts_inpaint(bg, area_min=ghost_area_min, area_max=ghost_area_max)
    return BackgroundModel(image=bg)
