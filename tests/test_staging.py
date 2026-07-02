"""Tests for anytrack.staging (local video staging)."""
from __future__ import annotations

from pathlib import Path

import pytest

from anytrack.config import AnyTrackConfig
from anytrack.staging import should_stage, stage_video, unstage, stage_dir_for


def _cfg(tmp_path, mode="auto", cleanup=False):
    return AnyTrackConfig(
        stage_video_locally=mode,
        stage_dir=str(tmp_path / "stage"),
        cleanup_staged=cleanup,
    )


def _src(tmp_path, name="clip.avi", content=b"video-bytes-0123456789"):
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_should_stage_modes(tmp_path):
    src = _src(tmp_path)
    sd = stage_dir_for(_cfg(tmp_path))
    assert should_stage(src, sd, "never") is False
    assert should_stage(src, sd, "always") is True


def test_auto_skips_when_same_device(tmp_path):
    # src and stage_dir are both under tmp_path -> same device -> auto skips.
    src = _src(tmp_path)
    local, staged = stage_video(src, _cfg(tmp_path, mode="auto"))
    assert staged is False
    assert local == src


def test_never_returns_source(tmp_path):
    src = _src(tmp_path)
    local, staged = stage_video(src, _cfg(tmp_path, mode="never"))
    assert staged is False and local == src


def test_always_copies_and_caches(tmp_path):
    src = _src(tmp_path)
    cfg = _cfg(tmp_path, mode="always")

    local, staged = stage_video(src, cfg)
    assert staged is True
    assert local != src
    assert Path(local).exists()
    assert Path(local).read_bytes() == src.read_bytes()
    assert Path(local).parent == stage_dir_for(cfg)

    # Second call is a cache hit: same destination, not re-copied.
    mtime_before = Path(local).stat().st_mtime_ns
    local2, staged2 = stage_video(src, cfg)
    assert staged2 is True and local2 == local
    assert Path(local2).stat().st_mtime_ns == mtime_before


def test_edited_source_recopies(tmp_path):
    src = _src(tmp_path)
    cfg = _cfg(tmp_path, mode="always")
    local, _ = stage_video(src, cfg)

    # Change size -> different cache key -> a new staged file.
    src.write_bytes(b"different-and-longer-video-bytes-xxxxx")
    local2, staged2 = stage_video(src, cfg)
    assert staged2 is True
    assert Path(local2).name != Path(local).name
    assert Path(local2).read_bytes() == src.read_bytes()


def test_unstage_respects_cleanup_flag(tmp_path):
    src = _src(tmp_path)
    keep = _cfg(tmp_path, mode="always", cleanup=False)
    local, _ = stage_video(src, keep)
    unstage(local, keep)
    assert Path(local).exists()  # kept (cache)

    clean = _cfg(tmp_path, mode="always", cleanup=True)
    local2, _ = stage_video(src, clean)
    unstage(local2, clean)
    assert not Path(local2).exists()  # deleted
