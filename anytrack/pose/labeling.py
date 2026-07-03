"""Headless core for the quick-label GUI (Milestone B2).

Everything display-independent lives here so it is unit-testable without a Tk
display or heavy deps:

* **sampling**   — pick a diverse, per-ROI-balanced subset of frames to label;
* **seeding**    — initial keypoint positions from the tracked ellipse, so the
  human *corrects* points instead of placing five from scratch;
* **store**      — a native, self-contained JSON label store (autosave + resume);
* **conversion** — display <-> crop-local <-> full-frame helpers;
* **export**     — a long-format manifest DataFrame + an optional ``.slp`` export
  (lazy ``sleap_io`` import; only needed for the SLEAP training handoff in B3).

The Tk shell (:mod:`anytrack.pose.label_gui`) is a thin layer over this module.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..coordinates import crop_to_full
from ..cropper import extract_crop
from .skeleton import Skeleton, DEFAULT_SKELETON


# --- frame sampling ----------------------------------------------------------

def sample_frames(
    df: pd.DataFrame,
    n: int = 100,
    strategy: str = "diversity",
    per_roi: bool = True,
    seed: int = 0,
    roi_col: str = "roi",
    frame_col: str = "frame",
) -> pd.DataFrame:
    """Pick a subset of ``df`` rows (one per crop) to hand-label.

    ``strategy``: ``"uniform"`` (evenly spaced in time), ``"random"`` (seeded),
    or ``"diversity"`` (stratify across ellipse angle x area so the labeled set
    spans real pose variety, not near-duplicate frames). With ``per_roi`` the
    budget is split evenly across ROIs. Deterministic for a fixed ``seed``.
    """
    if df is None or df.empty or frame_col not in df.columns:
        return df.iloc[0:0] if isinstance(df, pd.DataFrame) else pd.DataFrame()

    valid = df
    coord_cols = [c for c in ("x", "y") if c in df.columns]
    if coord_cols:
        dropped = df.dropna(subset=coord_cols)
        if not dropped.empty:
            valid = dropped

    rng = np.random.default_rng(int(seed))
    rois = list(valid[roi_col].unique()) if (per_roi and roi_col in valid.columns) else [None]
    per = max(1, int(math.ceil(n / max(1, len(rois)))))

    picks: List[pd.DataFrame] = []
    for r in rois:
        sub = valid if r is None else valid[valid[roi_col] == r]
        if not sub.empty:
            picks.append(_sample_one(sub, per, strategy, rng, frame_col))
    if not picks:
        return valid.iloc[0:0]

    out = pd.concat(picks, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=int(seed))
    sort_cols = [c for c in (roi_col, frame_col) if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def _sample_one(sub: pd.DataFrame, k: int, strategy: str, rng, frame_col: str) -> pd.DataFrame:
    m = len(sub)
    if m <= k:
        return sub
    if strategy == "uniform":
        idx = np.unique(np.linspace(0, m - 1, k).round().astype(int))
        return sub.iloc[idx]
    if strategy == "random":
        return sub.iloc[np.sort(rng.choice(m, size=k, replace=False))]
    return _diversity_sample(sub, k, rng)


def _diversity_sample(sub: pd.DataFrame, k: int, rng) -> pd.DataFrame:
    """Round-robin over (angle x area) strata to maximize pose coverage."""
    m = len(sub)
    ang = (sub["angle_deg"].to_numpy(dtype=float) if "angle_deg" in sub.columns
           else np.zeros(m))
    area = (sub["area"].to_numpy(dtype=float) if "area" in sub.columns else np.zeros(m))
    ang = np.nan_to_num(ang, nan=0.0) % 180.0
    abin = np.clip((ang / 30.0).astype(int), 0, 5)          # 6 angle bins
    area = np.nan_to_num(area, nan=0.0)
    if np.std(area) > 0:
        qs = np.quantile(area, [1 / 3, 2 / 3])
        arbin = np.digitize(area, qs)                        # 3 area bins
    else:
        arbin = np.zeros(m, dtype=int)
    key = abin * 3 + arbin

    groups: Dict[int, List[int]] = {}
    for i, kk in enumerate(key):
        groups.setdefault(int(kk), []).append(i)
    gkeys = list(groups.keys())
    rng.shuffle(gkeys)
    for g in gkeys:
        rng.shuffle(groups[g])

    chosen: List[int] = []
    pos = {g: 0 for g in gkeys}
    while len(chosen) < k:
        progressed = False
        for g in gkeys:
            if pos[g] < len(groups[g]):
                chosen.append(groups[g][pos[g]])
                pos[g] += 1
                progressed = True
                if len(chosen) >= k:
                    break
        if not progressed:
            break
    return sub.iloc[np.sort(chosen)]


# --- ellipse seeding ---------------------------------------------------------

def crop_origin(cx: float, cy: float, size: int) -> Tuple[int, int]:
    """Full-frame top-left of a ``size`` crop centered on ``(cx, cy)``.

    Matches :func:`anytrack.cropper.extract_crop` exactly so crop-local points
    map back to the same full-frame pixels the pixels were cut from.
    """
    half = size // 2
    return int(round(cx)) - half, int(round(cy)) - half


def seed_from_ellipse(row: Any, skeleton: Skeleton, crop_size: int) -> Dict[str, "Keypoint"]:
    """Initial crop-local keypoints from the tracked ellipse.

    ``thorax`` -> centroid; ``head``/``abdomen_tip`` -> the two ends of the major
    axis (which end is the head is ambiguous from the ellipse alone — the human
    fixes/swaps); wings -> lateral offsets toward the abdomen. Any skeleton node
    without a rule defaults to the crop center. All points clamped in-bounds.
    """
    size = int(crop_size)
    cx = float(_get(row, "x", 0.0))
    cy = float(_get(row, "y", 0.0))
    x0, y0 = crop_origin(cx, cy, size)
    tx, ty = cx - x0, cy - y0                                # thorax crop-local (~center)

    ang = math.radians(float(_get(row, "angle_deg", 0.0) or 0.0))
    major = float(_get(row, "major", size * 0.4) or size * 0.4)
    minor = float(_get(row, "minor", size * 0.2) or size * 0.2)
    ux, uy = math.cos(ang), math.sin(ang)                    # major-axis direction
    px, py = -uy, ux                                         # perpendicular

    rules = {
        "thorax": (tx, ty),
        "head": (tx + 0.5 * major * ux, ty + 0.5 * major * uy),
        "abdomen_tip": (tx - 0.5 * major * ux, ty - 0.5 * major * uy),
        "wingL": (tx - 0.25 * major * ux - 0.5 * minor * px,
                  ty - 0.25 * major * uy - 0.5 * minor * py),
        "wingR": (tx - 0.25 * major * ux + 0.5 * minor * px,
                  ty - 0.25 * major * uy + 0.5 * minor * py),
    }
    out: Dict[str, Keypoint] = {}
    for node in skeleton.nodes:
        x, y = rules.get(node, (tx, ty))
        out[node] = Keypoint(
            x=float(min(max(0.0, x), size)),
            y=float(min(max(0.0, y), size)),
            visible=True,
        )
    return out


def _get(row: Any, key: str, default: float) -> float:
    """Fetch ``key`` from a Series/dict, tolerating missing/NaN."""
    try:
        v = row[key]
    except (KeyError, IndexError, TypeError):
        return default
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return v


# --- display <-> crop-local conversion ---------------------------------------

def display_to_crop(x_disp: float, y_disp: float, zoom: float) -> Tuple[float, float]:
    return x_disp / zoom, y_disp / zoom


def crop_to_display(x_crop: float, y_crop: float, zoom: float) -> Tuple[float, float]:
    return x_crop * zoom, y_crop * zoom


# --- node display colors -----------------------------------------------------

# Built-in fly5 colors (fallback when a node isn't in the config spec). Kept in
# sync with the ``pose_node_colors`` config default; the config string wins.
DEFAULT_NODE_COLORS: Dict[str, str] = {
    "head": "#ff5252", "thorax": "#ffd740", "abdomen_tip": "#40c4ff",
    "wingL": "#69f0ae", "wingR": "#e040fb",
}
# Distinct-color cycle for skeleton nodes with no explicit/default color.
_FALLBACK_COLORS = ["#ff8a65", "#ba68c8", "#4dd0e1", "#aed581", "#f06292", "#7986cb"]


def parse_node_colors(spec: str) -> Dict[str, str]:
    """Parse a ``"name:#hex,name:#hex"`` config string into a mapping.

    Whitespace-tolerant; a missing ``#`` is added; malformed entries are
    skipped so a bad config never crashes the GUI.
    """
    out: Dict[str, str] = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, _, color = part.partition(":")
        name, color = name.strip(), color.strip()
        if name and color:
            out[name] = color if color.startswith("#") else "#" + color
    return out


def resolve_node_colors(nodes: Iterable[str], spec: str = "") -> Dict[str, str]:
    """Map every node to a hex color: config ``spec`` first, then the built-in
    fly5 defaults, then a distinct-color cycle — so no node is ever uncolored."""
    override = parse_node_colors(spec)
    out: Dict[str, str] = {}
    for i, node in enumerate(nodes):
        if node in override:
            out[node] = override[node]
        elif node in DEFAULT_NODE_COLORS:
            out[node] = DEFAULT_NODE_COLORS[node]
        else:
            out[node] = _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]
    return out


def contrast_fg(hex_color: str) -> str:
    """Return ``"#000000"`` or ``"#ffffff"`` for legible text on ``hex_color``."""
    h = (hex_color or "").lstrip("#")
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except (ValueError, IndexError):
        return "#ffffff"
    return "#000000" if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else "#ffffff"


# --- label store -------------------------------------------------------------

@dataclass
class Keypoint:
    x: float                    # crop-local pixels
    y: float
    visible: bool = True
    score: float = 1.0          # 1.0 for human labels; predictions carry model conf


@dataclass
class FrameLabel:
    roi: str
    frame: int
    track_id: int
    t_s: float
    cx: float                   # full-frame centroid (the crop center)
    cy: float
    points: Dict[str, Keypoint] = field(default_factory=dict)
    bad_frame: bool = False
    labeled: bool = False        # True once a human has confirmed this frame

    def origin(self, size: int) -> Tuple[int, int]:
        return crop_origin(self.cx, self.cy, size)

    def to_dict(self) -> dict:
        return {
            "roi": self.roi, "frame": int(self.frame), "track_id": int(self.track_id),
            "t_s": float(self.t_s), "cx": float(self.cx), "cy": float(self.cy),
            "bad_frame": bool(self.bad_frame), "labeled": bool(self.labeled),
            "points": {k: [p.x, p.y, bool(p.visible), float(p.score)]
                       for k, p in self.points.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrameLabel":
        pts = {k: Keypoint(x=float(v[0]), y=float(v[1]),
                           visible=bool(v[2]) if len(v) > 2 else True,
                           score=float(v[3]) if len(v) > 3 else 1.0)
               for k, v in d.get("points", {}).items()}
        return cls(
            roi=str(d["roi"]), frame=int(d["frame"]), track_id=int(d.get("track_id", 0)),
            t_s=float(d.get("t_s", float("nan"))), cx=float(d["cx"]), cy=float(d["cy"]),
            points=pts, bad_frame=bool(d.get("bad_frame", False)),
            labeled=bool(d.get("labeled", False)),
        )


@dataclass
class LabelStore:
    """A self-contained, resume-able set of hand-labeled frames.

    Points are stored **crop-local** (the natural labeling frame). Full-frame
    coordinates for export are derived on demand via the crop origin, so the
    store round-trips exactly and stays independent of the source video's
    on-disk location.
    """
    skeleton: Skeleton
    video_path: str
    crop_size: int
    frames: List[FrameLabel] = field(default_factory=list)

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_tracks(
        cls,
        video_path: str,
        df: pd.DataFrame,
        skeleton: Optional[Skeleton] = None,
        crop_size: int = 96,
        n: int = 100,
        strategy: str = "diversity",
        seed: int = 0,
        seed_ellipse: bool = True,
    ) -> "LabelStore":
        """Sample frames from a tracks table and build a store with seeded points."""
        skeleton = skeleton or DEFAULT_SKELETON
        picks = sample_frames(df, n=n, strategy=strategy, seed=seed)
        frames: List[FrameLabel] = []
        for _, row in picks.iterrows():
            pts = seed_from_ellipse(row, skeleton, crop_size) if seed_ellipse else {}
            frames.append(FrameLabel(
                roi=str(row.get("roi", "roi")),
                frame=int(row["frame"]),
                track_id=int(row.get("track_id", 0)) if "track_id" in row else 0,
                t_s=float(row.get("t_s", float("nan"))),
                cx=float(row["x"]), cy=float(row["y"]),
                points=pts,
            ))
        return cls(skeleton=skeleton, video_path=str(video_path),
                   crop_size=int(crop_size), frames=frames)

    # ---- stats --------------------------------------------------------------

    def n_labeled(self) -> int:
        return sum(1 for f in self.frames if f.labeled and not f.bad_frame)

    def __len__(self) -> int:
        return len(self.frames)

    # ---- (de)serialization --------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": "anytrack-labels/1",
            "skeleton": self.skeleton.to_dict(),
            "video_path": self.video_path,
            "crop_size": int(self.crop_size),
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LabelStore":
        return cls(
            skeleton=Skeleton.from_dict(d["skeleton"]),
            video_path=str(d.get("video_path", "")),
            crop_size=int(d.get("crop_size", 96)),
            frames=[FrameLabel.from_dict(f) for f in d.get("frames", [])],
        )

    def save(self, path) -> Path:
        """Atomically write the store to JSON (safe for autosave)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path) -> "LabelStore":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---- export -------------------------------------------------------------

    def to_manifest_df(self, only_labeled: bool = True) -> pd.DataFrame:
        """Long table (one row per frame x node) with full-frame + crop coords."""
        rows: List[Dict[str, Any]] = []
        for fl in self.frames:
            if only_labeled and (not fl.labeled or fl.bad_frame):
                continue
            x0, y0 = fl.origin(self.crop_size)
            for node in self.skeleton.nodes:
                kp = fl.points.get(node)
                if kp is None:
                    continue
                xf, yf = crop_to_full(kp.x, kp.y, x0, y0)
                rows.append({
                    "roi": fl.roi, "frame": fl.frame, "track_id": fl.track_id,
                    "t_s": fl.t_s, "keypoint": node,
                    "x_full": xf, "y_full": yf, "x_crop": kp.x, "y_crop": kp.y,
                    "visible": bool(kp.visible), "score": float(kp.score),
                })
        return pd.DataFrame.from_records(rows)

    def export_slp(self, path, only_labeled: bool = True) -> Path:
        """Export to a SLEAP ``.slp`` file for training (lazy ``sleap_io`` import).

        Keypoints are written **full-frame against the source video**, so a
        centered-instance trainer (B3) crops around the anchor itself — matching
        the crop-centered inference in B4. Invisible/absent nodes are written as
        NaN, SLEAP's convention for missing points.
        """
        try:
            import sleap_io as sio
        except ImportError as e:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "export_slp requires 'sleap-io'. Install the pose extra:\n"
                "    uv pip install 'anytrack[pose]'   (or)   uv pip install sleap-io"
            ) from e

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # NOTE: sio.Skeleton(nodes=...) mutates the passed list in place (str ->
        # Node objects), so keep our own stable list of string names for the
        # dict lookups and hand sio a throwaway copy.
        node_names = [str(n) for n in self.skeleton.nodes]
        skel = sio.Skeleton(
            nodes=list(node_names),
            edges=[(a, b) for a, b in self.skeleton.edges],
            symmetries=[set(s) for s in self.skeleton.symmetries],
            name=self.skeleton.name,
        )
        video = sio.Video.from_filename(self.video_path)

        lfs = []
        for fl in self.frames:
            if only_labeled and (not fl.labeled or fl.bad_frame):
                continue
            x0, y0 = fl.origin(self.crop_size)
            pts = np.full((len(node_names), 2), np.nan, dtype="float64")
            for i, node in enumerate(node_names):
                kp = fl.points.get(node)
                if kp is not None and kp.visible:
                    pts[i] = crop_to_full(kp.x, kp.y, x0, y0)
            inst = sio.Instance.from_numpy(pts, skeleton=skel)
            lfs.append(sio.LabeledFrame(video=video, frame_idx=int(fl.frame), instances=[inst]))

        labels = sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[skel])
        labels.save(str(path))
        return path


# --- pixel-crop extraction (for the GUI display) -----------------------------

def extract_label_crops(
    video_path,
    frames: Iterable[FrameLabel],
    crop_size: int,
    pad: str = "background",
    context: int = 0,
    show_progress: bool = False,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Sequential-decode the source video into per-anchor context windows.

    Returns ``{anchor_pos: {video_frame: crop}}`` keyed by each anchor's
    **position** in ``frames`` (not its frame index — two ROIs can share a frame
    index). For each anchor, crops are taken for frames
    ``[anchor - context, anchor + context]`` (clamped to the video) **using the
    anchor's centroid** (a fixed crop box), so the fly visibly translates through
    the window and the anchor's labels overlay consistently on every neighbor.

    One pass over the video (like :func:`anytrack.cropper.export_crops`) rather
    than per-frame seeks, so it stays codec-robust; ``context=0`` yields just the
    anchor crop. Crops are grayscale ``crop_size x crop_size`` uint8.
    """
    import cv2

    frames = list(frames)
    if not frames:
        return {}

    # video frame -> [(anchor_pos, anchor_FrameLabel)] whose window covers it.
    contrib: Dict[int, List[Tuple[int, FrameLabel]]] = {}
    for i, fl in enumerate(frames):
        af = int(fl.frame)
        for f in range(max(0, af - context), af + context + 1):
            contrib.setdefault(f, []).append((i, fl))
    max_needed = max(contrib)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=max_needed + 1, desc="  extracting crops",
                        bar_format="  {desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                        leave=True)
        except Exception:
            pbar = None

    out: Dict[int, Dict[int, np.ndarray]] = {i: {} for i in range(len(frames))}
    fidx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if fidx in contrib:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                pad_value = float(np.median(gray)) if pad == "background" else None
                for pos, fl in contrib[fidx]:
                    crop, _, _, _ = extract_crop(gray, fl.cx, fl.cy, crop_size,
                                                 pad=pad, pad_value=pad_value)
                    out[pos][fidx] = crop
            fidx += 1
            if pbar is not None:
                pbar.update(1)
            if fidx > max_needed:
                break
    finally:
        cap.release()
        if pbar is not None:
            pbar.close()
    return out
