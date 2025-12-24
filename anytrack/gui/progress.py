from __future__ import annotations

import queue
import threading
from typing import Callable, Optional
import tkinter as tk
from tkinter import ttk

import ttkbootstrap as tb

import time


class ETAEstimator:
    """Estimate remaining time (seconds) using a smoothed EMA of per-frame time.

    Uses a slower-adapting EMA (lower alpha) for more stable estimates,
    plus a warmup period before reporting values.

    Usage:
      eta = ETAEstimator()
      eta.start()
      eta.update(frames_done)
      s = eta.eta_seconds(total_frames, frames_done)
      eta.format_seconds(s)
    """

    def __init__(self, alpha: float = 0.05, warmup_frames: int = 10):
        """
        Args:
            alpha: EMA smoothing factor (lower = smoother, 0.05 recommended)
            warmup_frames: Minimum frames before reporting estimates
        """
        self.alpha = float(alpha)
        self.warmup_frames = warmup_frames
        self.start_t: float | None = None
        self.last_t: float | None = None
        self.last_count: int = 0
        self.ema_per_frame: float | None = None
        self._update_count: int = 0

    def start(self) -> None:
        now = time.monotonic()
        self.start_t = now
        self.last_t = now
        self.last_count = 0
        self.ema_per_frame = None
        self._update_count = 0

    def update(self, frames_done: int) -> None:
        now = time.monotonic()
        if self.last_t is None:
            self.start()
            return
        delta = frames_done - self.last_count
        if delta <= 0:
            return
        dt = now - self.last_t
        if dt <= 0:
            return
        per_frame = dt / delta
        self._update_count += 1

        if self.ema_per_frame is None:
            self.ema_per_frame = per_frame
        else:
            # Use higher alpha during warmup for faster initial convergence
            a = min(0.3, self.alpha * 3) if self._update_count < self.warmup_frames else self.alpha
            self.ema_per_frame = a * per_frame + (1.0 - a) * self.ema_per_frame
        self.last_t = now
        self.last_count = frames_done

    def is_warmed_up(self) -> bool:
        """Return True if enough samples have been collected for stable estimates."""
        return self._update_count >= self.warmup_frames

    def eta_seconds(self, total_frames: int, frames_done: int) -> float | None:
        if self.ema_per_frame is None or not self.is_warmed_up():
            return None
        remaining = max(0, int(total_frames) - int(frames_done))
        return remaining * self.ema_per_frame

    def fps(self) -> float | None:
        """Return current estimated FPS (frames per second)."""
        if self.ema_per_frame is None or self.ema_per_frame <= 0 or not self.is_warmed_up():
            return None
        return 1.0 / self.ema_per_frame

    def format_seconds(self, s: float | None) -> str:
        if s is None:
            return "--:--"
        s = int(round(s))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        if h > 0:
            return f"{h:d}:{m:02d}:{sec:02d}"
        return f"{m:02d}:{sec:02d}"


def make_progress_hook(q: queue.Queue) -> Callable[[str, dict], None]:
    """Return a thread-safe progress_hook that enqueues (event, payload) tuples.

    Worker threads should call: progress_hook("progress", {...})
    """
    def hook(event: str, payload: dict):
        try:
            if q.full():
                try:
                    q.get_nowait()
                except Exception:
                    pass
            q.put_nowait((event, payload))
        except Exception:
            # best-effort: don't raise in worker thread
            return

    return hook


class TrackingProgressDialog:
    """Modal progress dialog for tracking.

    Usage:
      q = queue.Queue()
      cancel_event = threading.Event()
      dialog = TrackingProgressDialog(parent, q, cancel_event)
      # Worker should use make_progress_hook(q) to report events.
    """

    def __init__(
        self,
        parent: tk.Widget,
        q: queue.Queue,
        cancel_event: threading.Event,
        title: Optional[str] = "Tracking",
        modal: bool = True,
    ) -> None:
        self.parent = parent
        self.q = q
        self.cancel_event = cancel_event
        self.title = title
        self.modal = modal

        self._top = tk.Toplevel(parent)
        self._top.title(self.title)
        self._top.resizable(False, False)
        self._top.transient(parent)

        frm = tb.Frame(self._top, padding=12)
        frm.pack(fill="both", expand=True)

        # Use monospace font for stable alignment of changing numbers
        self._status = tb.Label(frm, text="Starting...", font=("TkFixedFont", 11))
        self._status.pack(fill="x", pady=(0, 8))

        self._bar = ttk.Progressbar(frm, orient="horizontal", length=400, mode="determinate")
        self._bar.pack(fill="x", pady=(0, 8))
        self._bar['value'] = 0

        btn_row = tb.Frame(frm)
        btn_row.pack(fill="x")
        btn_row.columnconfigure(0, weight=1)

        self._cancel_btn = tb.Button(btn_row, text="Cancel", bootstyle="danger", command=self._on_cancel)
        self._cancel_btn.pack(side="right")

        # internal state
        self._closed = False
        self._grabbed = False

        # center over parent
        self._center_over_parent()

        if self.modal:
            try:
                self._top.grab_set()
                self._grabbed = True
            except Exception:
                self._grabbed = False

    def _center_over_parent(self):
        try:
            self.parent.update_idletasks()
            pw = self.parent.winfo_width()
            ph = self.parent.winfo_height()
            px = self.parent.winfo_rootx()
            py = self.parent.winfo_rooty()
            self._top.update_idletasks()
            w = self._top.winfo_width() or 420
            h = self._top.winfo_height() or 120
            x = px + max(0, (pw - w) // 2)
            y = py + max(0, (ph - h) // 2)
            self._top.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

    def _on_cancel(self):
        # User requested cancellation: set the cancel_event and disable the button.
        try:
            self.cancel_event.set()
        except Exception:
            pass
        try:
            self._cancel_btn.configure(state="disabled", text="Cancelling…")
        except Exception:
            pass
        # also offer an event for the poller
        try:
            if self.q is not None:
                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except Exception:
                        pass
                self.q.put_nowait(("cancel_request", {}))
        except Exception:
            pass

    def update_progress(self, percent: float, frame: int | None = None, t_s: float | None = None, text: Optional[str] = None):
        """Update the progress bar and status text. percent in 0.0..1.0."""
        try:
            pv = max(0.0, min(1.0, float(percent or 0.0)))
            self._bar['value'] = pv * 100.0
        except Exception:
            pass
        if text is None:
            if frame is not None and t_s is not None:
                text = f"Frame: {frame}  time: {t_s:.2f}s  ({pv*100:.1f}%)"
            elif frame is not None:
                text = f"Frame: {frame}  ({pv*100:.1f}%)"
            else:
                text = f"{pv*100:.1f}%"
        try:
            self._status.configure(text=text)
        except Exception:
            pass

    def set_error(self, message: str):
        try:
            self._status.configure(text=message, foreground="#b00020")
        except Exception:
            pass

    def close(self):
        if self._closed:
            return
        try:
            if self._grabbed:
                try:
                    self._top.grab_release()
                except Exception:
                    pass
            self._top.destroy()
        finally:
            self._closed = True
