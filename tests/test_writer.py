"""Tests for anytrack.writer (Milestone A6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anytrack.writer import (
    write_dataframe,
    write_tracks,
    default_output_path,
    TrackWriter,
)


def _df(n=5, start=0):
    return pd.DataFrame({
        "roi": ["r0"] * n,
        "frame": np.arange(start, start + n, dtype=np.int64),
        "x": np.arange(start, start + n, dtype=np.float64) + 0.5,
        "y": np.arange(start, start + n, dtype=np.float64) + 1.5,
    })


def test_parquet_roundtrip(tmp_path):
    df = _df()
    p = write_tracks(df, tmp_path / "out.parquet")
    assert p.exists()
    back = pd.read_parquet(p)
    pd.testing.assert_frame_equal(df, back)


def test_csv_roundtrip(tmp_path):
    df = _df()
    p = write_dataframe(df, tmp_path / "out.csv")
    back = pd.read_csv(p)
    np.testing.assert_allclose(back["x"], df["x"])
    assert list(back["roi"]) == list(df["roi"])


def test_format_inference_and_override(tmp_path):
    df = _df()
    # Suffix-inferred CSV, then explicit override to parquet at a .dat path.
    write_dataframe(df, tmp_path / "a.csv")
    assert (tmp_path / "a.csv").read_text().startswith("roi,frame,x,y")
    p = write_dataframe(df, tmp_path / "b.dat", fmt="parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(p), df)


def test_unsupported_format_raises(tmp_path):
    with pytest.raises(ValueError):
        write_dataframe(_df(), tmp_path / "x.csv", fmt="hdf5")


def test_default_output_path(tmp_path):
    class Cfg:
        output_format = "parquet"
        output_dir = str(tmp_path)

    class Vid:
        video_path = tmp_path / "movie.avi"

    p = default_output_path(Cfg(), Vid(), kind="tracks")
    assert p == tmp_path / "movie_tracks.parquet"

    Cfg.output_format = "csv"
    Cfg.output_dir = ""
    p2 = default_output_path(Cfg(), Vid(), kind="pose")
    assert p2 == tmp_path / "movie_pose.csv"  # falls back to video dir


def test_streaming_parquet(tmp_path):
    p = tmp_path / "stream.parquet"
    with TrackWriter(p) as w:
        w.write(_df(5, start=0))
        w.write(None)          # ignored
        w.write(pd.DataFrame())  # ignored
        w.write(_df(5, start=5))
    back = pd.read_parquet(p)
    assert len(back) == 10
    assert back["frame"].tolist() == list(range(10))


def test_streaming_csv(tmp_path):
    p = tmp_path / "stream.csv"
    with TrackWriter(p) as w:
        w.write(_df(3, start=0))
        w.write(_df(3, start=3))
    back = pd.read_csv(p)
    assert len(back) == 6
    # Header written exactly once.
    assert p.read_text().count("roi,frame,x,y") == 1
