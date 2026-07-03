"""
tqdm-based CLI progress rendering + formatted console output.

:class:`TqdmProgress` is a ``progress_hook`` callable that turns pipeline events
into tqdm bars and status lines. It understands both the fast path
(``status`` / ``preprocessing`` / ``tracking`` events) and the legacy path
(``started`` / ``progress``), and degrades to plain prints if tqdm is missing.
The small ``header``/``step``/``ok``/``info`` helpers give the CLIs a consistent,
readable layout.

:class:`BatchProgress` is the compact, animated one-line-per-video renderer for
``--video DIR`` runs: an animated spinner, a per-stage color, a live ROI
progress bar, and a ticking elapsed clock, all repainted in place by a
background thread so the line feels alive even during blocking, event-less
stages (ROI detection). :func:`batch_banner` / :func:`batch_summary` frame the
run with an opener and an aggregate footer.
"""
from __future__ import annotations

import sys
import threading
import time as _time
from typing import Optional

try:  # tqdm is a declared dependency, but never let its absence break a run.
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

_RULE = "─" * 60
_BAR_FMT = "  {desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

# ANSI colors for the compact batch-mode progress lines. One color per stage so
# ROI detection / preprocessing / tracking are distinguishable at a glance.
_ANSI = {
    "roi": "\033[33m",         # yellow  — ROI detection
    "preprocess": "\033[36m",  # cyan    — preprocessing / decode
    "track": "\033[32m",       # green   — tracking
    "pose": "\033[35m",        # magenta — pose
    "qc": "\033[34m",          # blue    — QC / crops
    "done": "\033[32m",        # green
    "error": "\033[31m",       # red
}
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"

# Braille spinner (10 frames, smooth) + shaded blocks for the progress bar.
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_BAR_FILL, _BAR_EMPTY = "█", "░"

# Short labels for the frame-based per-stage bars ("frames" events).
_STAGE_LABEL = {
    "roi": "ROI detect",
    "preprocess": "decode",
    "track": "track",
    "pose": "pose",
    "qc": "QC",
}


def _fmt_dt(s: float) -> str:
    """Human runtime: ``1m 05s`` past a minute, else ``4.2s``."""
    return f"{int(s // 60)}m {int(s % 60):02d}s" if s >= 60 else f"{s:.1f}s"


def _fmt_clock(s: float) -> str:
    """Stopwatch ``m:ss`` for the live per-video elapsed counter."""
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"


def _mini_bar(frac: float, width: int = 12) -> str:
    """A ``████░░░░`` progress bar for a 0..1 fraction."""
    frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
    fill = int(round(frac * width))
    return _BAR_FILL * fill + _BAR_EMPTY * (width - fill)


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
        self._frames_stage: Optional[str] = None
        # The tracking frame counter is polled from a background thread, so bar
        # mutations must be serialized against the main thread's event handling.
        self._lock = threading.Lock()

    def _close_bar(self) -> None:
        if self.bar is not None:
            try:
                self.bar.close()
            finally:
                self.bar = None
        self._frames_stage = None

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
            with self._lock:
                self._dispatch(event, payload or {})
        except Exception:  # pragma: no cover - progress must never break a run
            pass

    def _dispatch(self, event: str, payload: dict) -> None:
        if event == "frames":  # frame-based bar per stage (roi/preprocess/track)
            stage = payload.get("stage")
            if stage != self._frames_stage:
                total = int(payload.get("total", 0)) or None
                self._open_bar(total, _STAGE_LABEL.get(stage, stage or "frames"))
                self._frames_stage = stage
            self._set(int(payload.get("n", 0)))
            return

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
            # Frame-granular progress arrives via "frames" (stage=track); the ROI
            # start/complete events just bound the bar so it finishes cleanly.
            st = payload.get("status")
            if st == "complete":
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
        with self._lock:
            self._close_bar()


def batch_banner(n: int, mode: str) -> None:
    """Opener for a batch run: ``anytrack · N videos · fast mode``."""
    print(f"\n{_RULE}")
    print(f"  {_BOLD}anytrack{_RESET} {_DIM}·{_RESET} {n} videos {_DIM}·{_RESET} {mode} mode")
    print(_RULE)


def batch_summary(n_ok: int, n_total: int, total_rows: int, dt: float) -> None:
    """Aggregate footer: ✓/✗, videos done, total rows, wall-clock runtime."""
    all_ok = n_ok >= n_total
    color = _ANSI["done"] if all_ok else _ANSI["error"]
    mark = "✓" if all_ok else "✗"
    print(f"\n{_RULE}")
    print(f"  {color}{mark}{_RESET} {_BOLD}{n_ok}/{n_total}{_RESET} videos "
          f"{_DIM}·{_RESET} {total_rows:,} rows {_DIM}·{_RESET} {_fmt_dt(dt)}")
    print(_RULE)


class _LiveLine:
    """Per-video progress state + event handling, with no rendering of its own.

    Shared by the single-line :class:`BatchProgress` and the multi-line
    :class:`BatchProgressGroup` slots so both interpret pipeline events and format
    a line identically. :meth:`render` returns the live content (spinner + index +
    name + stage bar/label + clock); :meth:`result_line` the committed ✓/✗ result.
    """

    _NAME_W = 26

    def __init__(self, idx: int, total: int, name: str):
        self.idx = idx
        self.total = total
        self.name = name
        self.phase_key = "roi"
        self.phase_text = "starting…"
        self.fn = 0        # frames processed in the current stage
        self.ftotal = 0    # frames expected (0 → no bar, show phase text)
        self.t0 = _time.monotonic()

    def _set_phase(self, key: str, text: str) -> None:
        self.phase_key = key
        self.phase_text = text
        self.fn = self.ftotal = 0

    def _apply(self, event: str, payload: dict) -> None:
        """Update state from a pipeline event (no rendering)."""
        if event == "frames":
            stage = payload.get("stage")
            if stage:
                self.phase_key = stage
            self.fn = int(payload.get("n", 0))
            self.ftotal = int(payload.get("total", 0))
        elif event == "status":
            stage = payload.get("stage")
            if stage == "preprocessing":
                self._set_phase("preprocess", "preprocessing…")
            elif stage == "tracking":
                self._set_phase("track", "tracking…")
        elif event == "started":     # legacy per-frame tracking
            self._set_phase("track", "tracking…")
        elif event == "tracking":
            st = payload.get("status")
            if st == "starting":
                self._set_phase("track", "tracking…")
            elif st == "complete" and self.ftotal:
                self.fn = self.ftotal
        elif event in ("pose", "pose_done"):
            self._set_phase("pose", "pose…")

    def render(self, spin_char: str) -> str:
        color = _ANSI.get(self.phase_key, "")
        name = self.name[:self._NAME_W].ljust(self._NAME_W)
        if self.ftotal > 0:
            frac = self.fn / self.ftotal
            label = _STAGE_LABEL.get(self.phase_key, self.phase_key)
            phase = f"{label:<10} {color}{_mini_bar(frac, 16)}{_RESET} {int(frac * 100):3d}%"
        else:
            phase = f"{color}{self.phase_text}{_RESET}"
        clock = _fmt_clock(_time.monotonic() - self.t0)
        return (f"{color}{spin_char}{_RESET} {_DIM}[{self.idx}/{self.total}]{_RESET} "
                f"{name} {phase} {_DIM}{clock}{_RESET}")

    def result_line(self, summary: str, ok: bool) -> str:
        color = _ANSI["done"] if ok else _ANSI["error"]
        mark = "✓" if ok else "✗"
        name = self.name[:self._NAME_W].ljust(self._NAME_W)
        return (f"{color}{mark}{_RESET} {_DIM}[{self.idx}/{self.total}]{_RESET} "
                f"{name} {_DIM}{summary}{_RESET}")


class BatchProgress(_LiveLine):
    """Animated, compact one-line-per-video progress for a single in-flight video.

    Doubles as a ``progress_hook``: pipeline events set the current stage, and a
    daemon thread repaints the line ~10×/s so the spinner and elapsed clock keep
    moving even through blocking, event-less stages (ROI detection). Each stage
    has its own color and tracking shows a live frame bar. Off a TTY, the
    animation is skipped and only the final result line is printed. For
    *concurrent* videos use :class:`BatchProgressGroup` instead.
    """

    def __init__(self, idx: int, total: int, name: str):
        super().__init__(idx, total, name)
        self._tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._lock = threading.Lock()
        self._spin = 0
        self._finished = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self._tty:
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()

    def _tick(self) -> None:
        while not self._stop.wait(0.1):
            with self._lock:
                self._spin = (self._spin + 1) % len(_SPINNER)
                self._paint()

    def _paint(self) -> None:
        """Repaint the live line (lock held, TTY only)."""
        if not self._tty or self._finished:
            return
        sys.stdout.write(f"\r\033[K  {self.render(_SPINNER[self._spin])}")
        sys.stdout.flush()

    def phase(self, key: str, text: str) -> None:
        """Set the current phase label + color, clearing any frame bar."""
        with self._lock:
            self._set_phase(key, text)
            self._paint()

    def __call__(self, event: str, payload: dict) -> None:
        try:
            with self._lock:
                self._apply(event, payload or {})
                self._paint()
        except Exception:  # pragma: no cover - progress must never break a run
            pass

    def finish(self, summary: str, ok: bool = True) -> None:
        """Stop the animation and replace the live line with a result."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        with self._lock:
            self._finished = True
            line = f"  {self.result_line(summary, ok)}"
            if self._tty:
                sys.stdout.write(f"\r\033[K{line}\n")
                sys.stdout.flush()
            else:
                print(line)


class BatchProgressGroup:
    """Multi-line live progress for *concurrent* videos (Phase 4 concurrency).

    Owns a bottom "live region" of one line per in-flight video. :meth:`acquire`
    hands out a :class:`_Slot` (same ``progress_hook`` / ``phase`` / ``finish``
    interface as :class:`BatchProgress`, but threadless — the group's single
    painter thread redraws every slot ~10×/s). :meth:`_commit` (via
    ``slot.finish``) prints a video's final line as permanent scrollback above the
    shrinking live region. Off a TTY, only committed result lines are printed.
    """

    def __init__(self, total: int):
        self.total = total
        self._tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._lock = threading.Lock()
        self._slots: list = []      # active _Slot, in acquisition order
        self._drawn = 0             # live-region lines currently on screen
        self._spin = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self._tty:
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()

    def acquire(self, idx: int, name: str) -> "_Slot":
        slot = _Slot(self, idx, name)
        with self._lock:
            self._slots.append(slot)
        return slot

    def _tick(self) -> None:
        while not self._stop.wait(0.1):
            with self._lock:
                self._spin = (self._spin + 1) % len(_SPINNER)
                self._repaint()

    def _repaint(self) -> None:
        """Redraw the whole live region in place (lock held, TTY only)."""
        if not self._tty:
            return
        if self._drawn:
            sys.stdout.write(f"\033[{self._drawn}A")   # up to the top of the region
        spin = _SPINNER[self._spin]
        for slot in self._slots:
            sys.stdout.write(f"\r\033[K  {slot.render(spin)}\n")
        self._drawn = len(self._slots)
        sys.stdout.flush()

    def _commit(self, slot: "_Slot", summary: str, ok: bool) -> None:
        """Print a finished video's line as permanent scrollback, drop its slot."""
        with self._lock:
            line = f"  {slot.result_line(summary, ok)}"
            if not self._tty:
                print(line)
                self._remove(slot)
                return
            if self._drawn:
                sys.stdout.write(f"\033[{self._drawn}A\r")  # top of region, col 0
            sys.stdout.write("\033[J")                       # clear region to end of screen
            sys.stdout.write(line + "\n")                    # permanent result line
            self._remove(slot)
            self._drawn = 0
            self._repaint()                                  # redraw survivors below it

    def _remove(self, slot: "_Slot") -> None:
        try:
            self._slots.remove(slot)
        except ValueError:
            pass

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)


class _Slot(_LiveLine):
    """One video's line inside a :class:`BatchProgressGroup` (threadless)."""

    def __init__(self, group: BatchProgressGroup, idx: int, name: str):
        super().__init__(idx, group.total, name)
        self._group = group

    def phase(self, key: str, text: str) -> None:
        with self._group._lock:
            self._set_phase(key, text)

    def __call__(self, event: str, payload: dict) -> None:
        try:
            with self._group._lock:
                self._apply(event, payload or {})
        except Exception:  # pragma: no cover
            pass

    def finish(self, summary: str, ok: bool = True) -> None:
        self._group._commit(self, summary, ok)
