"""Pose quality-control (Milestone B5): flags, confidence plot, skeleton overlay.

Consumes the long pose table (:data:`~anytrack.pose.pipeline.POSE_COLUMNS`) and
produces:

* **per-instance flags** — low confidence, missing keypoints, wing L/R swap, and
  head/tail flips (see :func:`compute_pose_flags`);
* a **keypoint-confidence-over-time** plot;
* **skeleton overlay** drawing helpers used by ``qc.render_overlay``.

Kept separate from ``qc.py`` so the classical QC path carries no pose logic; it
is imported lazily only when a pose table is present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

POSE_FLAG_COLUMNS = ["flag_low_conf", "flag_missing", "flag_wing_swap", "flag_headtail"]


def _hex_to_bgr(hex_color: str) -> tuple:
    h = (hex_color or "#ffffff").lstrip("#")
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except (ValueError, IndexError):
        return (255, 255, 255)
    return (b, g, r)


def compute_pose_flags(pose_df: pd.DataFrame, skeleton, cfg) -> pd.DataFrame:
    """Per-instance pose failure flags: one row per ``(roi, track_id, frame)``.

    - ``flag_low_conf``: any present keypoint scores below ``cfg.pose_conf_min``.
    - ``flag_missing``:  any keypoint has NaN coordinates (below peak threshold).
    - ``flag_wing_swap``: wings on the same side of the body axis, or wingL on
      the opposite side from its per-track majority (an L/R label swap).
    - ``flag_headtail``: body axis (head→abdomen) reverses by more than
      ``cfg.pose_headtail_flip_deg`` between consecutive frames of a track.
    """
    base_cols = ["roi", "track_id", "frame", "t_s"] + POSE_FLAG_COLUMNS
    if pose_df is None or pose_df.empty:
        return pd.DataFrame(columns=base_cols)

    conf_min = float(getattr(cfg, "pose_conf_min", 0.2))
    flip_cos = float(np.cos(np.radians(float(getattr(cfg, "pose_headtail_flip_deg", 120.0)))))
    nodes = list(skeleton.nodes)

    idx = ["roi", "track_id", "frame"]
    piv = pose_df.pivot_table(index=idx, columns="keypoint",
                              values=["x_full", "y_full", "score"], aggfunc="first").sort_index()
    ts = pose_df.groupby(idx)["t_s"].first().reindex(piv.index)
    n = len(piv)

    def arr(val: str, node: str) -> np.ndarray:
        return piv[(val, node)].to_numpy(dtype=float) if (val, node) in piv.columns \
            else np.full(n, np.nan)

    xs = {node: arr("x_full", node) for node in nodes}
    ys = {node: arr("y_full", node) for node in nodes}
    ss = {node: arr("score", node) for node in nodes}
    scoremat = np.stack([ss[node] for node in nodes], axis=1)
    xmat = np.stack([xs[node] for node in nodes], axis=1)
    ymat = np.stack([ys[node] for node in nodes], axis=1)

    # low confidence: any PRESENT keypoint below threshold (NaN -> handled by missing)
    sc = np.where(np.isfinite(scoremat), scoremat, np.inf)
    flag_low_conf = sc.min(axis=1) < conf_min
    flag_missing = ~np.isfinite(xmat).all(axis=1) | ~np.isfinite(ymat).all(axis=1)

    flag_wing_swap = np.zeros(n, dtype=bool)
    flag_headtail = np.zeros(n, dtype=bool)
    if {"head", "thorax", "abdomen_tip", "wingL", "wingR"} <= set(nodes):
        ext_frac = float(getattr(cfg, "pose_wing_extend_frac", 0.15))

        def P(node):
            return np.stack([xs[node], ys[node]], axis=1)
        head, thorax, abd = P("head"), P("thorax"), P("abdomen_tip")
        wl, wr = P("wingL"), P("wingR")
        fwd = head - abd                                   # body axis (tail -> head)
        blen = np.linalg.norm(fwd, axis=1)                 # body length
        # unit left-normal to the body axis; NaN where the axis is degenerate.
        with np.errstate(invalid="ignore", divide="ignore"):
            nx = np.where(blen > 0, -fwd[:, 1] / blen, np.nan)
            ny = np.where(blen > 0, fwd[:, 0] / blen, np.nan)
        # signed lateral offset of each wing from the thorax (>0 one side, <0 other)
        lat_l = (wl[:, 0] - thorax[:, 0]) * nx + (wl[:, 1] - thorax[:, 1]) * ny
        lat_r = (wr[:, 0] - thorax[:, 0]) * nx + (wr[:, 1] - thorax[:, 1]) * ny
        conf_l = np.nan_to_num(ss["wingL"], nan=0.0) >= conf_min
        conf_r = np.nan_to_num(ss["wingR"], nan=0.0) >= conf_min
        # only judge a wing when it is CLEARLY extended (folded wings sit on the
        # midline where L/R is genuinely ambiguous -> never a "swap").
        ext_l = np.isfinite(lat_l) & (np.abs(lat_l) > ext_frac * blen) & conf_l
        ext_r = np.isfinite(lat_r) & (np.abs(lat_r) > ext_frac * blen) & conf_r

        keys = piv.index.to_frame(index=False)
        track_key = keys["roi"].astype(str) + "|" + keys["track_id"].astype(str)
        for key in pd.unique(track_key):
            ii = np.where((track_key == key).to_numpy())[0]        # sorted by frame
            # a wing extended to the opposite side from its per-track majority is a swap
            el, er = ext_l[ii], ext_r[ii]
            if el.any():
                exp_l = np.sign(np.median(lat_l[ii][el])) or 1.0
                flag_wing_swap[ii] |= el & (np.sign(lat_l[ii]) != exp_l)
            if er.any():
                exp_r = np.sign(np.median(lat_r[ii][er])) or 1.0
                flag_wing_swap[ii] |= er & (np.sign(lat_r[ii]) != exp_r)
            # head/tail: body axis reverses sharply between consecutive frames
            fv = fwd[ii]
            prev = np.roll(fv, 1, axis=0)
            dot = (fv * prev).sum(axis=1)
            nrm = np.linalg.norm(fv, axis=1) * np.linalg.norm(prev, axis=1)
            cos = np.divide(dot, nrm, out=np.full_like(dot, np.nan), where=nrm > 0)
            flip = np.isfinite(cos) & (cos < flip_cos)
            if len(flip):
                flip[0] = False                                    # no previous frame
            flag_headtail[ii] |= flip

    out = piv.index.to_frame(index=False)
    out["t_s"] = ts.to_numpy()
    out["flag_low_conf"] = flag_low_conf
    out["flag_missing"] = flag_missing
    out["flag_wing_swap"] = flag_wing_swap
    out["flag_headtail"] = flag_headtail
    return out[base_cols]


def pose_flag_summary(flags: pd.DataFrame) -> Dict[str, Any]:
    """Overall + per-ROI pose-flag rates for the report/summary."""
    if flags is None or flags.empty:
        return {}
    n = len(flags)
    overall = {c: round(float(flags[c].mean()), 4) for c in POSE_FLAG_COLUMNS if c in flags.columns}
    overall["n_instances"] = int(n)
    return overall


def plot_pose_confidence(pose_df: pd.DataFrame, cfg, out_dir: Path) -> List[Path]:
    """Per-keypoint mean confidence over time (one line per node, node colors)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    if pose_df is None or pose_df.empty:
        return []
    from .skeleton import get_skeleton
    from .labeling import resolve_node_colors

    sk = get_skeleton(cfg)
    nodes = list(sk.nodes)
    colors = resolve_node_colors(nodes, getattr(cfg, "pose_node_colors", ""))
    x = "t_s" if "t_s" in pose_df.columns and pose_df["t_s"].notna().any() else "frame"
    g = pose_df.groupby([x, "keypoint"])["score"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 4))
    for node in nodes:
        gg = g[g["keypoint"] == node].sort_values(x)
        if not gg.empty:
            ax.plot(gg[x], gg["score"], lw=0.8, label=node, color=colors[node])
    ax.axhline(float(getattr(cfg, "pose_conf_min", 0.2)), ls="--", lw=1, color="#888",
               label=f"conf_min={getattr(cfg, 'pose_conf_min', 0.2)}")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("keypoint confidence")
    ax.set_xlabel(x)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=min(6, len(nodes) + 1), loc="lower right")
    ax.set_title("Pose keypoint confidence over time (mean across ROIs)")
    fig.tight_layout()
    path = out_dir / "qc_pose_confidence.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return [path]


def prepare_pose_overlay(pose_df: pd.DataFrame, skeleton, cfg, sx: float, sy: float) -> Dict[str, Any]:
    """Precompute per-frame scaled keypoints + colors/edges for the overlay."""
    from .labeling import resolve_node_colors
    nodes = list(skeleton.nodes)
    hexc = resolve_node_colors(nodes, getattr(cfg, "pose_node_colors", ""))
    by_frame: Dict[int, List[Dict[str, tuple]]] = {}
    for (f, _roi, _tid), g in pose_df.groupby(["frame", "roi", "track_id"]):
        inst = {r.keypoint: (r.x_full * sx, r.y_full * sy, r.score)
                for r in g.itertuples(index=False)}
        by_frame.setdefault(int(f), []).append(inst)
    return {
        "by_frame": by_frame,
        "node_bgr": {node: _hex_to_bgr(hexc[node]) for node in nodes},
        "edges": skeleton.edge_indices(),
        "nodes": nodes,
        "conf_min": float(getattr(cfg, "pose_conf_min", 0.2)),
    }


def draw_pose(frame, fidx: int, prep: Dict[str, Any], thickness: int = 1, radius: int = 3) -> None:
    """Draw the skeleton for all instances at ``fidx`` onto ``frame`` (in place)."""
    import cv2
    insts = prep["by_frame"].get(int(fidx))
    if not insts:
        return
    nodes, cmin = prep["nodes"], prep["conf_min"]
    for inst in insts:
        for ai, bi in prep["edges"]:
            pa, pb = inst.get(nodes[ai]), inst.get(nodes[bi])
            if (pa and pb and pa[2] >= cmin and pb[2] >= cmin
                    and np.isfinite(pa[0]) and np.isfinite(pb[0])):
                cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                         (210, 210, 210), thickness, cv2.LINE_AA)
        for name, (x, y, s) in inst.items():
            if s >= cmin and np.isfinite(x) and np.isfinite(y):
                cv2.circle(frame, (int(x), int(y)), radius,
                           prep["node_bgr"].get(name, (255, 255, 255)), -1, cv2.LINE_AA)
