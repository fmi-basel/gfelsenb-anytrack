"""
Unified result output.

Wraps (does not replace) ``TrackingResult.to_dataframe`` and the GUI's CSV
export: given a dataframe, write it as parquet (default; ``pyarrow`` is already
a dependency) or CSV. :class:`TrackWriter` offers optional streaming for large
offline runs so the whole result need not be held in memory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

PathLike = Union[str, Path]


def _infer_fmt(path: Path, fmt: Optional[str]) -> str:
    if fmt:
        fmt = fmt.lower()
        if fmt not in ("parquet", "csv"):
            raise ValueError(f"Unsupported output format: {fmt!r}")
        return fmt
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return "parquet"
    if suffix == ".csv":
        return "csv"
    return "parquet"


def write_dataframe(df: pd.DataFrame, path: PathLike, fmt: Optional[str] = None) -> Path:
    """Write ``df`` to ``path`` as parquet or CSV (format inferred from suffix)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = _infer_fmt(path, fmt)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:  # csv
        df.to_csv(path, index=False)
    return path


# Semantic wrappers so call sites read clearly and can diverge later if needed.
def write_tracks(df: pd.DataFrame, path: PathLike, fmt: Optional[str] = None) -> Path:
    """Write the per-frame centroid table."""
    return write_dataframe(df, path, fmt)


def write_pose(df: pd.DataFrame, path: PathLike, fmt: Optional[str] = None) -> Path:
    """Write the long-format per-keypoint pose table (Milestone B)."""
    return write_dataframe(df, path, fmt)


def write_diagnostics(df: pd.DataFrame, path: PathLike, fmt: Optional[str] = None) -> Path:
    """Write the per-frame diagnostics table."""
    return write_dataframe(df, path, fmt)


def default_output_path(cfg, video, kind: str = "tracks") -> Path:
    """Compute a default output path from ``cfg.output_dir``/``cfg.output_format``.

    Falls back to the video's own directory when ``output_dir`` is unset.
    """
    fmt = (getattr(cfg, "output_format", "parquet") or "parquet").lower()
    ext = "csv" if fmt == "csv" else "parquet"
    out_dir = getattr(cfg, "output_dir", "") or ""
    base = Path(out_dir) if out_dir else Path(video.video_path).parent
    stem = Path(video.video_path).stem
    return base / f"{stem}_{kind}.{ext}"


class TrackWriter:
    """Streaming writer for large runs.

    ``parquet`` appends row-group tables via ``pyarrow.parquet.ParquetWriter``
    (all chunks must share a schema). ``csv`` appends rows, writing the header
    only on the first chunk. Use as a context manager::

        with TrackWriter(path) as w:
            for chunk in chunks:
                w.write(chunk)
    """

    def __init__(self, path: PathLike, fmt: Optional[str] = None):
        self.path = Path(path)
        self.fmt = _infer_fmt(self.path, fmt)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._pq_writer = None
        self._csv_started = False

    def write(self, df: Optional[pd.DataFrame]) -> None:
        if df is None or df.empty:
            return
        if self.fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._pq_writer is None:
                self._pq_writer = pq.ParquetWriter(str(self.path), table.schema)
            self._pq_writer.write_table(table)
        else:  # csv
            df.to_csv(
                self.path,
                index=False,
                mode="w" if not self._csv_started else "a",
                header=not self._csv_started,
            )
            self._csv_started = True

    def close(self) -> None:
        if self._pq_writer is not None:
            self._pq_writer.close()
            self._pq_writer = None

    def __enter__(self) -> "TrackWriter":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
