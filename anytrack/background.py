from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Sequence, Dict, Any, List
import numpy as np
import cv2
import math

@dataclass
class BackgroundModel:
    """Background model containing average, GMM, and arena detection results."""
    average: np.ndarray  # uint8 grayscale - temporal average
    gmm: np.ndarray  # uint8 grayscale - GMM background (brighter component)
    thresholds: Optional[np.ndarray] = None  # float32, NaN where 1-component used
    arena_mask: Optional[np.ndarray] = None  # uint8 {0,255} - combined arena mask
    arena_circles: Optional[List[Tuple[float, float, float]]] = None  # [(cx, cy, r), ...]

    @property
    def image(self) -> np.ndarray:
        """Backward compatibility - returns GMM background."""
        return self.gmm


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


# ============================================================================
# GMM Background Modeling Functions
# ============================================================================

def _gmm_two_gaussian_threshold(weights: np.ndarray, means: np.ndarray, variances: np.ndarray) -> float:
    """Calculate intersection threshold between two 1D Gaussians in a 2-component GMM.

    Returns threshold between the two means if possible, else midpoint.
    """
    # Sort by mean
    order = np.argsort(means)
    w1, w2 = weights[order]
    m1, m2 = means[order]
    v1, v2 = variances[order]
    s1, s2 = np.sqrt(v1), np.sqrt(v2)

    # Solve: w1 * N(x|m1,s1^2) = w2 * N(x|m2,s2^2)
    # Quadratic: a x^2 + b x + c = 0
    a = 1/(2*v1) - 1/(2*v2)
    b = m2/v2 - m1/v1
    c = (m1*m1)/(2*v1) - (m2*m2)/(2*v2) + np.log((w2*s1)/(w1*s2))

    if abs(a) < 1e-12:
        # Linear case
        if abs(b) < 1e-12:
            return float((m1 + m2) / 2.0)
        x = -c / b
        return float(np.clip(x, 0, 255))

    disc = b*b - 4*a*c
    if disc < 0:
        return float((m1 + m2) / 2.0)

    r1 = (-b + np.sqrt(disc)) / (2*a)
    r2 = (-b - np.sqrt(disc)) / (2*a)

    lo, hi = min(m1, m2), max(m1, m2)
    candidates = [r for r in (r1, r2) if lo <= r <= hi]
    if candidates:
        x = candidates[0] if len(candidates) == 1 else min(candidates, key=lambda r: abs(r - (m1+m2)/2))
        return float(np.clip(x, 0, 255))

    return float((m1 + m2) / 2.0)


def sample_frames_uniformly(
    video_path: str,
    n_samples: int,
    debug_hook: Optional[Callable] = None,
) -> np.ndarray:
    """Sample n_samples frames uniformly from video.

    Returns: (N, H, W) uint8 grayscale array

    Debug events:
    - "sampling_progress": {"step": int, "total": int, "frame_idx": int}
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total <= 0:
        cap.release()
        raise ValueError("Could not read frame count from video.")

    idxs = np.linspace(0, n_total - 1, n_samples, dtype=int)

    frames = []
    for step, idx in enumerate(idxs, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise IOError(f"Failed to read frame at index {idx}.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

        _emit_debug(debug_hook, "sampling_progress", {
            "step": step,
            "total": n_samples,
            "frame_idx": int(idx)
        })

    cap.release()
    samples = np.stack(frames, axis=0).astype(np.uint8)  # (N, H, W)
    return samples


def compute_average_background(
    samples: np.ndarray,
    debug_hook: Optional[Callable] = None,
) -> np.ndarray:
    """Compute temporal average of sampled frames.

    Returns: (H, W) uint8

    Debug events:
    - "average_complete": {"average": np.ndarray}
    """
    avg_bg = samples.mean(axis=0)  # (H, W) float
    avg_bg_uint8 = np.clip(avg_bg, 0, 255).astype(np.uint8)

    _emit_debug(debug_hook, "average_complete", {"average": avg_bg_uint8})

    return avg_bg_uint8


def _fit_gmm_background_sklearn(
    samples: np.ndarray,
    bic_improvement: float = 10.0,
    lowp: float = 120.0,
    min_std: float = 10.0,
    reg_covar: float = 1e-3,
    random_state: int = 0,
    debug_hook: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference per-pixel GMM via sklearn (kept for validation; the vectorized
    ``fit_gmm_background`` below is the default). Fit 1-2 components with BIC.

    Returns:
        gmm_bg: (H, W) uint8 - brighter component mean
        thresholds: (H, W) float32 - NaN where 1-comp used

    Debug events:
    - "gmm_progress": {"pixel": int, "total": int, "percent": float}  # Every 5%
    - "gmm_complete": {"gmm": np.ndarray, "thresholds": np.ndarray}
    """
    from sklearn.mixture import GaussianMixture

    N, H, W = samples.shape
    P = H * W
    vals = samples.reshape(N, P).astype(np.float32)  # (N, P)

    bg = np.zeros(P, dtype=np.float32)
    thr = np.full(P, np.nan, dtype=np.float32)

    last_percent = -1
    for p in range(P):
        X = vals[:, p].reshape(-1, 1)

        # Skip pixels with very low variation
        if np.unique(X).size < 5 or X.std() < min_std:
            bg[p] = float(X.mean())
            thr[p] = np.nan
            continue

        # Fit 1-component GMM
        g1 = GaussianMixture(n_components=1, covariance_type="full",
                             reg_covar=reg_covar, random_state=random_state)
        g1.fit(X)
        bic1 = g1.bic(X)

        # Fit 2-component GMM
        g2 = GaussianMixture(n_components=2, covariance_type="full",
                             reg_covar=reg_covar, random_state=random_state)
        g2.fit(X)
        bic2 = g2.bic(X)

        # Require improvement to use 2 components
        use_two = (bic2 + bic_improvement) < bic1

        if not use_two:
            bg[p] = float(g1.means_[0, 0])
        else:
            weights = g2.weights_
            means = g2.means_.flatten()
            variances = np.array([g2.covariances_[k, 0, 0] for k in range(2)], dtype=np.float32)

            # Background = brighter component mean
            bg[p] = float(np.max(means))
            if bg[p] < lowp:
                bg[p] = lowp
            thr[p] = _gmm_two_gaussian_threshold(weights, means, variances)

        # Emit progress every 5%
        percent = (p + 1) * 100.0 / P
        if int(percent / 5) > int(last_percent / 5):
            _emit_debug(debug_hook, "gmm_progress", {
                "pixel": p + 1,
                "total": P,
                "percent": percent
            })
            last_percent = percent

    bg_img = bg.reshape(H, W)
    thr_img = thr.reshape(H, W)

    bg_img_uint8 = np.clip(bg_img, 0, 255).astype(np.uint8)

    _emit_debug(debug_hook, "gmm_complete", {
        "gmm": bg_img_uint8,
        "thresholds": thr_img
    })

    return bg_img_uint8, thr_img


# ---------------------------------------------------------------------------
# Vectorized per-pixel GMM (numpy EM over all pixels at once).
# Same model + BIC selection as the sklearn reference above, but fits every
# pixel simultaneously instead of a Python loop of per-pixel sklearn fits.
# ---------------------------------------------------------------------------

def _norm_pdf(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var)


def _em2_1d(X: np.ndarray, reg_covar: float, max_iter: int = 100, tol: float = 1e-4):
    """Vectorized 1-D two-component Gaussian EM over the rows of X (Pa, N)."""
    Pa, N = X.shape
    Xd = X.astype(np.float64)
    mu1 = np.percentile(Xd, 25.0, axis=1)
    mu2 = np.percentile(Xd, 75.0, axis=1)
    mu2 = np.where(mu2 <= mu1, mu1 + 1.0, mu2)          # keep components distinct
    v0 = Xd.var(axis=1) + reg_covar
    v1 = v0.copy()
    v2 = v0.copy()
    w1 = np.full(Pa, 0.5)
    w2 = np.full(Pa, 0.5)
    eps = 1e-300
    prev_ll = None
    for _ in range(max_iter):
        p1 = w1[:, None] * _norm_pdf(Xd, mu1[:, None], v1[:, None])
        p2 = w2[:, None] * _norm_pdf(Xd, mu2[:, None], v2[:, None])
        s = p1 + p2 + eps
        r1 = p1 / s
        r2 = p2 / s
        N1 = r1.sum(axis=1)
        N2 = r2.sum(axis=1)
        N1c = np.maximum(N1, 1e-8)
        N2c = np.maximum(N2, 1e-8)
        mu1 = (r1 * Xd).sum(axis=1) / N1c
        mu2 = (r2 * Xd).sum(axis=1) / N2c
        v1 = (r1 * (Xd - mu1[:, None]) ** 2).sum(axis=1) / N1c + reg_covar
        v2 = (r2 * (Xd - mu2[:, None]) ** 2).sum(axis=1) / N2c + reg_covar
        w1 = N1 / N
        w2 = N2 / N
        ll = float(np.log(s).sum())
        if prev_ll is not None and abs(ll - prev_ll) <= tol * Pa:
            break
        prev_ll = ll
    p1 = w1[:, None] * _norm_pdf(Xd, mu1[:, None], v1[:, None])
    p2 = w2[:, None] * _norm_pdf(Xd, mu2[:, None], v2[:, None])
    logL2 = np.log(p1 + p2 + eps).sum(axis=1)
    return mu1, mu2, v1, v2, w1, w2, logL2


def fit_gmm_background(
    samples: np.ndarray,
    bic_improvement: float = 10.0,
    lowp: float = 120.0,
    min_std: float = 10.0,
    reg_covar: float = 1e-3,
    random_state: int = 0,
    debug_hook: Optional[Callable] = None,
    max_iter: int = 100,
    chunk: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a per-pixel 1-2 component GMM (BIC model selection), vectorized.

    Numpy EM over all pixels at once instead of a per-pixel sklearn fit — same
    model and selection rule, orders of magnitude faster. Returns:
        gmm_bg:     (H, W) uint8   — brighter-component mean (clamped to lowp)
        thresholds: (H, W) float32 — component midpoint where 2 comps used, else NaN

    Pixels with <5 unique values or std < min_std are treated as background
    (their mean), matching the reference implementation. The per-pixel threshold
    is advisory and not used by the tracker.
    """
    N, H, W = samples.shape
    P = H * W
    vals = samples.reshape(N, P).astype(np.float32).T   # (P, N)
    bg = np.empty(P, dtype=np.float32)
    thr = np.full(P, np.nan, dtype=np.float32)
    ln_n = float(np.log(N))
    k1, k2 = 2, 5

    for s0 in range(0, P, chunk):
        X = vals[s0:s0 + chunk]                          # (Pb, N)
        mean = X.mean(axis=1)
        var1 = X.var(axis=1)
        std = np.sqrt(var1)
        Xs = np.sort(X, axis=1)
        n_unique = 1 + (np.diff(Xs, axis=1) != 0).sum(axis=1)
        active = ~((n_unique < 5) | (std < min_std))

        bg_c = mean.astype(np.float32).copy()            # skipped / 1-comp -> mean
        thr_c = np.full(X.shape[0], np.nan, dtype=np.float32)

        if active.any():
            Xa = X[active]
            muA = mean[active].astype(np.float64)
            vA = (var1[active] + reg_covar).astype(np.float64)
            logL1 = (-0.5 * N * np.log(2.0 * np.pi * vA)
                     - ((Xa - muA[:, None]) ** 2).sum(axis=1) / (2.0 * vA))
            mu1, mu2, v1, v2, w1, w2, logL2 = _em2_1d(Xa, reg_covar, max_iter)
            use_two = (-2.0 * logL2 + k2 * ln_n + bic_improvement) < (-2.0 * logL1 + k1 * ln_n)
            bright = np.maximum(mu1, mu2)
            bg_a = np.where(use_two, np.maximum(bright, lowp), muA)
            thr_a = np.where(use_two, (mu1 + mu2) / 2.0, np.nan)
            bg_c[active] = bg_a.astype(np.float32)
            thr_c[active] = thr_a.astype(np.float32)

        bg[s0:s0 + chunk] = bg_c
        thr[s0:s0 + chunk] = thr_c

    bg_img = np.clip(bg.reshape(H, W), 0, 255).astype(np.uint8)
    thr_img = thr.reshape(H, W)
    _emit_debug(debug_hook, "gmm_complete", {"gmm": bg_img, "thresholds": thr_img})
    return bg_img, thr_img


def detect_arenas(
    bg_uint8: np.ndarray,
    min_area_frac: float = 0.01,
    expected_n: int = 4,
    debug_hook: Optional[Callable] = None,
) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
    """Detect circular arenas using Otsu threshold + connected components.

    Returns:
        circles: [(cx, cy, r), ...] sorted by area descending
        arena_mask: (H, W) uint8 {0,255}

    Debug events:
    - "arenas_detected": {"circles": list, "mask": np.ndarray}
    """
    H, W = bg_uint8.shape[:2]

    # 1) Otsu threshold
    ret, bw = cv2.threshold(bg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2) Connected components
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    # 3) Filter by area
    min_area = int(min_area_frac * H * W)
    comps = []
    for lab in range(1, nlab):  # Skip background label 0
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask = (labels == lab).astype(np.uint8) * 255
        comps.append((area, mask))

    if len(comps) < expected_n:
        # Not enough arenas found - return empty
        _emit_debug(debug_hook, "arenas_detected", {
            "circles": [],
            "mask": np.zeros((H, W), dtype=np.uint8)
        })
        return [], np.zeros((H, W), dtype=np.uint8)

    # 4) Keep N largest components
    comps.sort(key=lambda x: x[0], reverse=True)
    comps = comps[:expected_n]

    # 5) Fit circles and combine mask
    circles = []
    arena_mask = np.zeros((H, W), dtype=np.uint8)

    for area, comp_mask in comps:
        arena_mask = cv2.bitwise_or(arena_mask, comp_mask)

        # Fit circle using contour points
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(cnts, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        circles.append((float(cx), float(cy), float(r)))

    _emit_debug(debug_hook, "arenas_detected", {
        "circles": circles,
        "mask": arena_mask
    })

    return circles, arena_mask


def blur_arenas(
    bg_uint8: np.ndarray,
    arena_mask: np.ndarray,
    sigma: float = 3.0,
) -> np.ndarray:
    """Apply Gaussian blur only inside arena mask.

    Returns: (H, W) uint8 - bg with arenas blurred
    """
    mask = arena_mask.astype(bool)
    blurred = cv2.GaussianBlur(bg_uint8, (0, 0), sigmaX=sigma)

    out = bg_uint8.copy()
    out[mask] = blurred[mask]
    return out


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


def build_background(
    video_path: str,
    n_samples: int = 100,
    bic_improvement: float = 10.0,
    min_std: float = 10.0,
    reg_covar: float = 1e-3,
    lowp: float = 120.0,
    arena_detection: bool = True,
    arena_min_area_frac: float = 0.01,
    arena_blur_sigma: float = 3.0,
    debug: bool = False,
    debug_hook: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    debug_stride: int = 1,  # Kept for API compatibility, not used
    debug_max_side: int = 0,  # Kept for API compatibility, not used
) -> BackgroundModel:
    """Build background using GMM approach.

    Steps:
    1. Sample n_samples frames uniformly
    2. Compute average background
    3. Fit per-pixel GMM → GMM background + thresholds
    4. Detect arenas → circles + mask (if arena_detection=True)
    5. Blur GMM within arenas (if sigma > 0)

    Returns BackgroundModel with all results.
    """
    if not debug:
        debug_hook = None

    # 1. Sample frames uniformly
    samples = sample_frames_uniformly(video_path, n_samples, debug_hook)

    # 2. Compute average background
    avg_bg = compute_average_background(samples, debug_hook)

    # 3. Fit GMM background
    gmm_bg, thresholds = fit_gmm_background(
        samples,
        bic_improvement=bic_improvement,
        lowp=lowp,
        min_std=min_std,
        reg_covar=reg_covar,
        debug_hook=debug_hook
    )

    # 4. Arena detection (optional)
    arena_circles = None
    arena_mask = None
    if arena_detection:
        arena_circles, arena_mask = detect_arenas(
            gmm_bg,
            min_area_frac=arena_min_area_frac,
            expected_n=4,
            debug_hook=debug_hook
        )

        # 5. Blur GMM within arenas (if arenas detected and sigma > 0)
        if arena_blur_sigma > 0 and arena_mask is not None and arena_circles:
            gmm_bg = blur_arenas(gmm_bg, arena_mask, arena_blur_sigma)
            _emit_debug(debug_hook, "blur_complete", {"blurred": gmm_bg})

    return BackgroundModel(
        average=avg_bg,
        gmm=gmm_bg,
        thresholds=thresholds,
        arena_mask=arena_mask,
        arena_circles=arena_circles,
    )


def build_background_image(video_path, cfg) -> np.ndarray:
    """Build the model background image using the config's GMM/arena params.

    Thin wrapper around :func:`build_background` with the standard config
    mapping, so ROI detection and QC share one background definition.
    """
    return build_background(
        str(video_path),
        n_samples=cfg.gmm_n_samples,
        bic_improvement=cfg.gmm_bic_improvement,
        min_std=cfg.gmm_min_std,
        reg_covar=cfg.gmm_reg_covar,
        lowp=cfg.gmm_lowp,
        arena_detection=cfg.arena_detection_enabled,
        arena_min_area_frac=cfg.arena_min_area_frac,
        arena_blur_sigma=cfg.arena_blur_sigma,
    ).image
