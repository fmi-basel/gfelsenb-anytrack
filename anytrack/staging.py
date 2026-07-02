"""
Local staging of source videos.

Copies a source video off slow/network storage to a local scratch directory
before processing, so the many reads/seeks the pipeline performs (frame
sampling for the background model, FFmpeg ROI extraction, per-frame decode)
hit a fast local disk instead of the network. Staging is skipped when the
source already lives on the same local device as the scratch directory, and a
(size, mtime) cache means re-running on the same clip reuses the copy.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Tuple

from platformdirs import user_cache_dir


def stage_dir_for(cfg) -> Path:
    """Local scratch directory for staged videos (created if missing)."""
    d = getattr(cfg, "stage_dir", "") or str(Path(user_cache_dir("anytrack")) / "staged")
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def should_stage(src: Path, stage_dir: Path, mode: str) -> bool:
    """Decide whether to copy ``src`` locally.

    ``"auto"`` stages only when the source is on a *different* device than the
    scratch dir (i.e. not already on this local disk) — a robust, cross-platform
    stand-in for "is this remote?" that avoids pointless copies of local files.
    """
    if mode == "never":
        return False
    if mode == "always":
        return True
    try:
        return os.stat(src).st_dev != os.stat(stage_dir).st_dev
    except OSError:
        return False


def _cache_name(src: Path, st: os.stat_result) -> str:
    # Key on size + mtime so an edited source re-copies but re-runs reuse.
    return f"{src.stem}__{st.st_size}__{int(st.st_mtime)}{src.suffix}"


def stage_video(src, cfg) -> Tuple[Path, bool]:
    """Copy ``src`` into the local scratch dir if warranted.

    Returns ``(path_to_use, staged)``: the local copy (``staged=True``) or the
    original path (``staged=False``) when staging is disabled, unnecessary, or
    fails. Never raises — falls back to the source on any error.
    """
    src = Path(src)
    mode = getattr(cfg, "stage_video_locally", "auto")
    try:
        stage_dir = stage_dir_for(cfg)
    except OSError:
        return src, False
    if not should_stage(src, stage_dir, mode):
        return src, False
    try:
        st = src.stat()
    except OSError:
        return src, False

    dst = stage_dir / _cache_name(src, st)
    if dst.exists() and dst.stat().st_size == st.st_size:
        return dst, True  # cache hit — reuse the existing copy

    tmp = dst.with_name(dst.name + ".partial")
    try:
        # copyfile (data only) — copy2/copystat raises "Operation not permitted"
        # on network shares, and we don't need the source's metadata for a
        # scratch copy (the cache key already encodes size + mtime).
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)        # atomic publish
    except OSError:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return src, False           # e.g. disk full -> just use the source
    return dst, True


def unstage(local_path, cfg) -> None:
    """Delete a staged copy if ``cleanup_staged`` is set (otherwise cache it)."""
    if not getattr(cfg, "cleanup_staged", False):
        return
    try:
        Path(local_path).unlink()
    except OSError:
        pass
