"""
Preflight input validation for anytrack.

Checks that each input is actually processable *before* a (long) tracking run:
the video opens and its frames decode, and the per-frame timing CSV is present
and parseable — plus soft checks (frame↔timing-row mismatch, time going
backwards) that don't block a run but usually mean a staging mistake. Point it
at a single video or a whole directory; it exits non-zero if anything fails
(``--strict`` also fails on warnings), so it drops straight into a batch script
or a CI gate::

    anytrack-validate --video /path/to/dir
    anytrack-validate --video clip.avi --timing clip.csv --deep

This is the same "is this input usable?" question ``anytrack-run`` answers
internally when it discovers a directory, exposed as its own fast, side-effect-
free command so you can check a staging folder without kicking off tracking.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from .video_select import discover_videos, VIDEO_EXTS

# Check outcomes. WARN = usable but suspicious (fails only under --strict).
PASS, FAIL, WARN = "pass", "fail", "warn"


@dataclass
class Check:
    """One named validation result: ``name`` + a PASS/FAIL/WARN ``status``."""
    name: str
    status: str
    detail: str = ""


@dataclass
class FileReport:
    """All checks for one (video, timing) input pair."""
    video: Path
    timing: Optional[Path]
    checks: List[Check] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    @property
    def warned(self) -> bool:
        return any(c.status == WARN for c in self.checks)

    def is_ok(self, strict: bool = False) -> bool:
        """Usable? Never with a FAIL; and never with a WARN under ``strict``."""
        if self.failed:
            return False
        return not (strict and self.warned)


def _human_size(p: Path) -> str:
    """File size like ``1.4 GB`` (best effort; empty on stat failure)."""
    try:
        b = float(p.stat().st_size)
    except OSError:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return f"{b:.0f} {unit}" if unit == "B" else f"{b:.1f} {unit}"
        b /= 1024
    return ""


def _check_video(path: Path, deep: bool) -> Tuple[List[Check], Optional[int]]:
    """Video-side checks. Returns ``(checks, frame_count)`` (count None on failure).

    Opens the file once and, because container metadata can lie, actually
    ``read()``s frame 0 to prove the stream decodes. ``deep`` additionally seeks
    to and decodes the last frame (catches truncated files) at the cost of a seek.
    """
    import cv2
    checks: List[Check] = []
    p = Path(path)
    if not p.exists():
        checks.append(Check("video file exists", FAIL, "not found"))
        return checks, None
    checks.append(Check("video file exists", PASS, _human_size(p)))

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        cap.release()
        checks.append(Check("video opens", FAIL, "cv2 cannot open (bad or unsupported codec?)"))
        return checks, None

    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    checks.append(Check("video opens", PASS, f"{n} frames · {fps:.4g} fps · {w}×{h}"))

    if n <= 0:
        checks.append(Check("frame count > 0", FAIL, "0 frames reported"))
    if w <= 0 or h <= 0:
        checks.append(Check("frame size", WARN, "0×0 reported (metadata missing)"))

    ok_first, _ = cap.read()
    checks.append(Check("first frame decodes", PASS if ok_first else FAIL,
                        "" if ok_first else "read() failed on frame 0"))

    if deep and ok_first and n > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
        ok_last, _ = cap.read()
        checks.append(Check("last frame decodes", PASS if ok_last else WARN,
                            "" if ok_last else "read() failed near the end (truncated file?)"))
    cap.release()
    return checks, (n if n > 0 else None)


def _check_timing(timing: Path) -> Tuple[List[Check], Optional[int]]:
    """Timing-CSV checks. Returns ``(checks, row_count)`` (count None on failure).

    Loads via the same :func:`anytrack.io.load_timing_csv` the pipeline uses, so
    a CSV that passes here is one the run can actually consume (right columns,
    parseable), and flags any ``dt_s < 0`` (clock ran backwards → bad kinematics).
    """
    checks: List[Check] = []
    if not timing.exists():
        checks.append(Check("timing CSV exists", FAIL, f"not found: {timing.name}"))
        return checks, None
    checks.append(Check("timing CSV exists", PASS, timing.name))
    try:
        from .io import load_timing_csv
        df = load_timing_csv(timing)
    except Exception as e:  # missing 'frame'/'dt_s' columns, unreadable CSV, …
        checks.append(Check("timing CSV parses", FAIL, str(e)))
        return checks, None
    rows = len(df)
    checks.append(Check("timing CSV parses", PASS if rows else FAIL,
                        f"{rows} rows" if rows else "0 rows"))
    if rows:
        neg = int((df["dt_s"] < 0).sum())
        if neg:
            checks.append(Check("timing monotonic", WARN, f"{neg} row(s) with dt_s < 0"))
    return checks, (rows or None)


def validate_input(video, timing=None, deep: bool = False) -> FileReport:
    """Run every check for one input and return its :class:`FileReport`.

    ``timing`` defaults to ``<video>.csv`` (the pipeline convention). When both
    the video frame count and the timing row count are known, a final
    frame↔timing-row check compares them (mismatch → WARN).
    """
    video = Path(video)
    timing = Path(timing) if timing else video.with_suffix(".csv")
    rep = FileReport(video=video, timing=timing)

    vchecks, frames = _check_video(video, deep)
    rep.checks.extend(vchecks)
    tchecks, rows = _check_timing(timing)
    rep.checks.extend(tchecks)

    if frames is not None and rows is not None:
        if frames == rows:
            rep.checks.append(Check("frame/timing match", PASS, f"{frames} frames = {rows} rows"))
        else:
            rep.checks.append(Check(
                "frame/timing match", WARN,
                f"{frames} video frames vs {rows} timing rows (Δ{rows - frames:+d})"))
    return rep


# ── formatted output ───────────────────────────────────────────────────────

def _sym(status: str) -> Tuple[str, str]:
    """(symbol, ANSI color) for a check status."""
    from .cli_progress import _ANSI
    if status == PASS:
        return "✓", _ANSI["done"]
    if status == WARN:
        return "⚠", _ANSI["roi"]     # yellow
    return "✗", _ANSI["error"]


def _print_report(rep: FileReport) -> None:
    from .cli_progress import step, _RESET, _DIM
    step(rep.video.name)
    for c in rep.checks:
        sym, col = _sym(c.status)
        detail = f"   {_DIM}{c.detail}{_RESET}" if c.detail else ""
        print(f"  {col}{sym}{_RESET} {c.name}{detail}")


def _print_summary(reports: List[FileReport], strict: bool) -> None:
    from .cli_progress import _RULE, _ANSI, _RESET, _BOLD, _DIM
    n = len(reports)
    n_ok = sum(1 for r in reports if r.is_ok(strict))
    n_fail = sum(1 for r in reports if r.failed)
    n_warn = sum(1 for r in reports if r.warned and not r.failed)
    all_ok = n_ok == n
    col = _ANSI["done"] if all_ok else _ANSI["error"]
    parts = [f"{_BOLD}{n_ok}/{n}{_RESET} valid"]
    if n_fail:
        parts.append(f"{_ANSI['error']}{n_fail} failed{_RESET}")
    if n_warn:
        parts.append(f"{_ANSI['roi']}{n_warn} with warnings{_RESET}"
                     + (" (fatal under --strict)" if strict else ""))
    print(f"\n{_RULE}")
    print(f"  {col}{'✓' if all_ok else '✗'}{_RESET} " + f" {_DIM}·{_RESET} ".join(parts))
    print(_RULE)


def _report_dict(rep: FileReport, strict: bool) -> dict:
    return {
        "video": str(rep.video),
        "timing": str(rep.timing) if rep.timing else None,
        "ok": rep.is_ok(strict),
        "checks": [{"name": c.name, "status": c.status, "detail": c.detail}
                   for c in rep.checks],
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        prog="anytrack-validate",
        description="Preflight-check that input videos (+ their timing CSVs) are "
                    "readable before a tracking run. Exits non-zero if any input fails.",
    )
    ap.add_argument("--video", required=True, type=Path,
                    help="A video file or a directory of videos to validate.")
    ap.add_argument("--timing", type=Path, default=None,
                    help="Timing CSV for a single --video file (default: <video>.csv). "
                         "Ignored for a directory — each video pairs with its own <stem>.csv.")
    ap.add_argument("--deep", action="store_true",
                    help="Also seek to and decode the last frame (catches truncated files); slower.")
    ap.add_argument("--strict", action="store_true",
                    help="Treat warnings (e.g. a frame/timing-row mismatch) as failures.")
    ap.add_argument("--json", action="store_true",
                    help="Emit a JSON report to stdout instead of the formatted view.")
    args = ap.parse_args(argv)

    videos = discover_videos(args.video)
    single = not Path(args.video).is_dir()
    if not videos:
        msg = f"no videos found in {args.video} (looked for {', '.join(VIDEO_EXTS)})"
        print(json.dumps({"error": msg, "reports": []}) if args.json else msg)
        return 1

    reports = [validate_input(v, args.timing if (single and args.timing) else None,
                              deep=args.deep)
               for v in videos]

    if args.json:
        print(json.dumps({"reports": [_report_dict(r, args.strict) for r in reports]}, indent=2))
    else:
        from .cli_progress import header
        header(f"anytrack-validate · {len(reports)} input(s)")
        for r in reports:
            _print_report(r)
        _print_summary(reports, args.strict)

    return 0 if all(r.is_ok(args.strict) for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
