"""FFmpeg must not inherit the terminal's stdin.

If ffmpeg's stdin is the controlling TTY it flips the terminal into raw mode to
watch for its interactive 'q' key; when anytrack later kills/tears down that
process (watchdog timeout, FIFO teardown) ffmpeg can't restore the mode and the
shell is left broken until `stty sane`. Every ffmpeg/ffprobe launch therefore
passes ``stdin=subprocess.DEVNULL``. These tests lock that in — deterministically
(captured kwargs) and, when ffmpeg is present, via the underlying tty mechanism.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

import anytrack.preprocess as pp


class _FakeProc:
    """Minimal stand-in for a finished ffmpeg process (no stdout/stderr lines)."""
    def __init__(self, cmd, **kw):
        _FakeProc.last_kwargs = kw
        self.stdout = iter([])
        self.stderr = iter([])
        self.returncode = 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass

    def poll(self):
        return 0


def test_run_ffmpeg_with_progress_detaches_stdin(monkeypatch):
    monkeypatch.setattr(pp.subprocess, "Popen", _FakeProc)
    pp._run_ffmpeg_with_progress(["ffmpeg", "-i", "x", "y"], total_frames=0, progress_hook=None)
    assert _FakeProc.last_kwargs.get("stdin") is subprocess.DEVNULL


def test_probe_codec_detaches_stdin(monkeypatch):
    rec = {}

    def fake_run(cmd, **kw):
        rec["kw"] = kw
        return subprocess.CompletedProcess(cmd, 0, stdout="mpeg4\n", stderr="")

    monkeypatch.setattr(pp, "get_ffprobe_path", lambda: "ffprobe")
    monkeypatch.setattr(pp.subprocess, "run", fake_run)
    pp._probe_codec("clip.avi")
    assert rec["kw"].get("stdin") is subprocess.DEVNULL


def test_hwaccel_probe_detaches_stdin(monkeypatch):
    rec = {}

    def fake_run(cmd, **kw):
        rec["kw"] = kw
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(pp, "get_ffmpeg_path", lambda: "ffmpeg")
    monkeypatch.setattr(pp.subprocess, "run", fake_run)
    pp._hwaccel_probe_ok("clip.avi", ["-hwaccel", "videotoolbox"])
    assert rec["kw"].get("stdin") is subprocess.DEVNULL


@pytest.mark.skipif(not hasattr(__import__("shutil"), "which")
                    or __import__("shutil").which("ffmpeg") is None,
                    reason="ffmpeg not installed")
@pytest.mark.skipif(sys.platform == "win32", reason="pty is POSIX-only")
def test_devnull_stdin_leaves_terminal_intact_when_ffmpeg_killed(tmp_path):
    """The mechanism: an ffmpeg killed mid-run corrupts a TTY on its stdin, but
    a DEVNULL stdin leaves it untouched."""
    import os
    import pty
    import termios
    import time

    src = tmp_path / "t.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi",
                    "-i", "testsrc=size=64x64:rate=10:duration=3",
                    "-pix_fmt", "yuv420p", str(src)],
                   stdin=subprocess.DEVNULL, capture_output=True)
    if not src.exists() or src.stat().st_size == 0:
        pytest.skip("ffmpeg could not synthesize a test clip")

    def termios_intact_after_kill(use_devnull: bool) -> bool:
        master, slave = pty.openpty()
        try:
            before = termios.tcgetattr(slave)
            p = subprocess.Popen(["ffmpeg", "-i", str(src), "-f", "null", "-"],
                                 stdin=(subprocess.DEVNULL if use_devnull else slave),
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.4)          # let ffmpeg switch the tty to raw mode
            p.kill()
            p.wait()
            return termios.tcgetattr(slave) == before
        finally:
            os.close(master)
            os.close(slave)

    assert termios_intact_after_kill(use_devnull=False) is False   # tty stdin → broken
    assert termios_intact_after_kill(use_devnull=True) is True     # DEVNULL → intact
