"""
tqdm-based CLI progress rendering + formatted console output.

:class:`TqdmProgress` is a ``progress_hook`` callable that turns pipeline events
into tqdm bars and status lines. It understands both the fast path
(``status`` / ``preprocessing`` / ``tracking`` events) and the legacy path
(``started`` / ``progress``), and degrades to plain prints if tqdm is missing.
The small ``header``/``step``/``ok``/``info`` helpers give the CLIs a consistent,
readable layout.
"""
from __future__ import annotations

from typing import Optional

try:  # tqdm is a declared dependency, but never let its absence break a run.
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

_RULE = "─" * 60
_BAR_FMT = "  {desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


def header(title: str) -> None:
    """Print a boxed section header."""
    print(f"\n{_RULE}\n {title}\n{_RULE}")


def step(msg: str) -> None:
    """Announce a pipeline step."""
    print(f"\n▶ {msg}")


def ok(msg: str) -> None:
    """Report a completed step."""
    print(f"  ✓ {msg}")


def info(msg: str) -> None:
    """Print an indented detail line."""
    print(f"  {msg}")


def progress_iter(iterable, desc: str, total: Optional[int] = None, enable: bool = True):
    """Wrap ``iterable`` in a tqdm bar when enabled (and available)."""
    if enable and tqdm is not None:
        return tqdm(iterable, desc=f"  {desc}", total=total, bar_format=_BAR_FMT, leave=True)
    return iterable


class TqdmProgress:
    """A ``progress_hook`` that renders pipeline progress as tqdm bars.

    Pass the instance as ``progress_hook`` to ``TrackingSession.run`` (or the
    trackers). Call :meth:`close` when done. All rendering is best-effort:
    exceptions in the hook never propagate into the pipeline.
    """

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.bar = None

    def _close_bar(self) -> None:
        if self.bar is not None:
            try:
                self.bar.close()
            finally:
                self.bar = None

    def _open_bar(self, total: Optional[int], desc: str) -> None:
        self._close_bar()
        if tqdm is not None and not self.quiet:
            self.bar = tqdm(total=total, desc=f"  {desc}", bar_format=_BAR_FMT, leave=True)

    def _set(self, n: int) -> None:
        if self.bar is not None:
            self.bar.n = n
            self.bar.refresh()

    def __call__(self, event: str, payload: dict) -> None:
        try:
            self._dispatch(event, payload or {})
        except Exception:  # pragma: no cover - progress must never break a run
            pass

    def _dispatch(self, event: str, payload: dict) -> None:
        if event == "status":
            stage = payload.get("stage")
            if stage == "preprocessing":
                step("Preprocess (FFmpeg single-pass decode) …")
            elif stage == "tracking":
                step("Track")
            return

        if event == "started":  # legacy per-frame tracking
            total = int(payload.get("total_frames", 0)) or None
            step("Track")
            self._open_bar(total, "frames")
            return

        if event == "tracking":
            st = payload.get("status")
            if st == "starting":
                n = int(payload.get("n_rois", 0)) or None
                w = payload.get("n_workers")
                self._open_bar(n, f"ROIs (×{w} workers)")
            elif st == "progress":
                self._set(int(payload.get("completed", 0)))
            elif st == "complete":
                if self.bar is not None and self.bar.total:
                    self._set(self.bar.total)
                self._close_bar()
            return

        if event == "progress":  # legacy per-frame update
            self._set(int(payload.get("frame_count", 0)))
            return

        if event in ("done", "cancelled", "error"):
            self._close_bar()
            return

    def close(self) -> None:
        self._close_bar()
