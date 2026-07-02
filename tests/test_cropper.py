"""Tests for anytrack.cropper (Milestone A5)."""
from __future__ import annotations

import numpy as np
import cv2
import pandas as pd
import pytest

from anytrack.config import AnyTrackConfig
from anytrack.cropper import extract_crop, CropBatcher, export_crops, main as crop_export_main
from anytrack.models import VideoAsset, CircleROI


def _ramp(h=100, w=100):
    # Distinct value per pixel so we can check placement/padding precisely.
    return np.arange(h * w, dtype=np.uint16).reshape(h, w).astype(np.uint8)


def test_extract_crop_fully_inside():
    frame = _ramp()
    crop, x0, y0, oob = extract_crop(frame, cx=50, cy=50, size=20)
    assert crop.shape == (20, 20)
    assert not oob
    assert (x0, y0) == (40, 40)
    # Crop must equal the corresponding frame slice.
    np.testing.assert_array_equal(crop, frame[40:60, 40:60])


def test_extract_crop_size_always_exact_at_corner():
    frame = _ramp()
    crop, x0, y0, oob = extract_crop(frame, cx=2, cy=2, size=20, pad="background", pad_value=0)
    assert crop.shape == (20, 20)
    assert oob
    assert (x0, y0) == (-8, -8)
    # The valid region sits at offset (8,8); padded region is 0.
    assert crop[0, 0] == 0
    np.testing.assert_array_equal(crop[8:, 8:], frame[0:12, 0:12])


def test_extract_crop_edge_padding_replicates():
    frame = _ramp()
    crop, _, _, oob = extract_crop(frame, cx=0, cy=50, size=20, pad="edge")
    assert crop.shape == (20, 20)
    assert oob
    # Left half is replicated from column 0.
    assert crop[10, 0] == crop[10, 9]


def test_extract_crop_fully_outside_is_padded():
    frame = _ramp()
    crop, _, _, oob = extract_crop(frame, cx=-100, cy=-100, size=16, pad="background", pad_value=7)
    assert crop.shape == (16, 16)
    assert oob
    assert np.all(crop == 7)


def test_crop_batcher():
    b = CropBatcher(batch_size=3)
    for i in range(2):
        b.add(np.zeros((8, 8), np.uint8), {"i": i})
    assert not b.ready() and len(b) == 2
    b.add(np.ones((8, 8), np.uint8), {"i": 2})
    assert b.ready()
    batch, metas = b.pop()
    assert batch.shape == (3, 8, 8)
    assert [m["i"] for m in metas] == [0, 1, 2]
    assert len(b) == 0


def _write_video(path, n=10, h=64, w=64):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h), isColor=True)
    if not writer.isOpened():
        writer.release()
        return False
    for i in range(n):
        gray = np.full((h, w), 200, np.uint8)
        cv2.circle(gray, (32, 32), 5, 20, -1)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    writer.release()
    return True


def test_export_crops_writes_pngs_and_manifest(tmp_path):
    video_path = tmp_path / "vid.avi"
    if not _write_video(video_path):
        pytest.skip("No MJPG encoder available")

    n = 10
    df = pd.DataFrame({
        "roi": ["r0"] * n,
        "frame": np.arange(n),
        "x": [32.0] * n,
        "y": [32.0] * n,
    })
    video = VideoAsset(
        video_path=video_path,
        timing_csv_path=video_path.with_suffix(".csv"),
        rois=[CircleROI(name="r0", cx=32, cy=32, r=31)],
    )
    cfg = AnyTrackConfig(crop_size=32, crop_pad_mode="background")
    out_dir = tmp_path / "crops"

    manifest = export_crops(video, df, cfg, out_dir=out_dir, every_n=2)

    # every_n=2 -> frames 0,2,4,6,8 = 5 crops
    assert len(manifest) == 5
    assert set(manifest["frame"]) == {0, 2, 4, 6, 8}
    assert (out_dir / "manifest.parquet").exists()
    for _, row in manifest.iterrows():
        p = out_dir / row["path"]
        assert p.exists()
        crop = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        assert crop.shape == (32, 32)
        # object centered -> x_crop/y_crop near size/2
        assert abs(row["x_crop"] - 16) <= 1
        assert not row["out_of_bounds"]


def test_crop_export_cli(tmp_path):
    """The anytrack-crop-export CLI runs A5 over a saved tracks table."""
    video_path = tmp_path / "vid.avi"
    if not _write_video(video_path):
        pytest.skip("No MJPG encoder available")

    n = 6
    df = pd.DataFrame({
        "roi": ["r0"] * n,
        "frame": np.arange(n),
        "x": [32.0] * n,
        "y": [32.0] * n,
    })
    tracks = tmp_path / "tracks.parquet"
    df.to_parquet(tracks, index=False)
    out_dir = tmp_path / "cli_crops"

    rc = crop_export_main([
        "--video", str(video_path),
        "--tracks", str(tracks),
        "--out-dir", str(out_dir),
        "--crop-size", "32",
    ])
    assert rc == 0

    manifest_path = out_dir / "manifest.parquet"
    assert manifest_path.exists()
    manifest = pd.read_parquet(manifest_path)
    assert len(manifest) == n
    for rel in manifest["path"]:
        assert (out_dir / rel).exists()
