"""Discover, validate, and interactively select videos for batch processing.

Both ``anytrack-run`` and ``anytrack-label`` accept a **directory** for
``--video``: we discover the videos in it, validate each (openable + whatever
the caller needs, e.g. a timing CSV or a tracks table), then ask whether to
process all of them (default yes) and, if not, show a checkbox selector.

A single explicit file path bypasses all of this (unchanged behavior).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Tuple

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".m4v")

# Frames decode but the container reports 0 frames: an unfinalized recording
# (interrupted recorder, no index). Seeking in such files silently lands on
# frame 0, so they must be remuxed before processing, not merely let through.
UNINDEXED_REASON = "unindexed — needs remux"

# validator(path) -> (ok, reason). reason is shown next to the video either way.
Validator = Callable[[Path], Tuple[bool, str]]


def discover_videos(path, exts=VIDEO_EXTS) -> List[Path]:
    """Videos directly under ``path`` if it is a directory, else ``[path]``."""
    p = Path(path)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in exts)
    return [p]


def validate_openable(path) -> Tuple[bool, str]:
    """Default validator: the file opens and reports >0 frames (metadata only)."""
    import cv2
    p = Path(path)
    if not p.exists():
        return False, "not found"
    cap = cv2.VideoCapture(str(p))
    ok = cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else 0
    decodes = ok and n <= 0 and cap.read()[0]
    cap.release()
    if not ok:
        return False, "cannot open"
    if n <= 0:
        return False, UNINDEXED_REASON if decodes else "0 frames"
    return True, f"{n} frames"


def _confirm_all(n: int) -> bool:
    """Ask whether to process all; Enter / y / yes → True (default yes)."""
    try:
        ans = input(f"Process all {n} videos? [Y/n] ").strip().lower()
    except EOFError:
        return True
    return ans in ("", "y", "yes")


def _text_select(labels: List[str]) -> List[int]:
    """Non-TTY fallback: numbered list + comma/range input (blank = all)."""
    print("Select videos — comma-separated numbers/ranges (e.g. 1,3-5); blank = all:")
    for i, lab in enumerate(labels, 1):
        print(f"  {i}. {lab}")
    try:
        raw = input("> ").strip()
    except EOFError:
        return list(range(len(labels)))
    if not raw:
        return list(range(len(labels)))
    picked = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            a, _, b = part.partition("-")
            if a.strip().isdigit() and b.strip().isdigit():
                picked.update(range(int(a) - 1, int(b)))
        elif part.isdigit():
            picked.add(int(part) - 1)
    return sorted(i for i in picked if 0 <= i < len(labels))


def _curses_select(labels: List[str]) -> List[int]:
    """Interactive checkbox selector (arrows move, Space toggles, Enter confirms)."""
    import curses

    def _run(stdscr):
        curses.curs_set(0)
        # Inherit the terminal's own background instead of painting it black, so
        # the selector blends in (no "clear to black" on light themes).
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, -1, -1)
            stdscr.bkgd(" ", curses.color_pair(1))
        except Exception:
            pass
        checked = [True] * len(labels)     # default: all selected
        pos = 0
        head = "↑/↓ move · Space toggle · a all · n none · Enter confirm · q cancel"
        while True:
            stdscr.erase()
            stdscr.addstr(0, 0, head[:max(0, curses.COLS - 1)])
            for i, lab in enumerate(labels):
                mark = "[x]" if checked[i] else "[ ]"
                line = f"{mark} {lab}"[:max(0, curses.COLS - 1)]
                stdscr.addstr(i + 2, 0, line, curses.A_REVERSE if i == pos else curses.A_NORMAL)
            stdscr.refresh()
            k = stdscr.getch()
            if k in (curses.KEY_UP, ord("k")):
                pos = (pos - 1) % len(labels)
            elif k in (curses.KEY_DOWN, ord("j")):
                pos = (pos + 1) % len(labels)
            elif k == ord(" "):
                checked[pos] = not checked[pos]
            elif k == ord("a"):
                checked = [True] * len(labels)
            elif k == ord("n"):
                checked = [False] * len(labels)
            elif k in (10, 13, curses.KEY_ENTER):
                return [i for i, c in enumerate(checked) if c]
            elif k in (ord("q"), 27):           # q / Esc
                return []
    return curses.wrapper(_run)


def checkbox_select(labels: List[str]) -> List[int]:
    """Return selected indices via a checkbox TUI (curses) or a numbered fallback."""
    if not labels:
        return []
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _curses_select(labels)
        except Exception:
            pass
    return _text_select(labels)


def resolve_videos(path, validator: Validator = validate_openable) -> List[Path]:
    """Discover → validate → confirm/select. Returns the chosen valid videos.

    A single file path is returned as-is (the caller does its own validation, so
    single-video runs are unchanged). A directory triggers discovery, per-video
    validation (invalid ones are listed and skipped), a "process all?" prompt
    (default yes), and a checkbox selector on "no".
    """
    p = Path(path)
    if not p.is_dir():
        return [p]

    found = discover_videos(p)
    if not found:
        print(f"no videos found in {p}")
        return []

    valid: List[Path] = []
    labels: List[str] = []
    unindexed: List[Path] = []
    print(f"found {len(found)} video(s) in {p}:")
    for v in found:
        ok, reason = validator(v)
        print(f"  {'✓' if ok else '✗'} {v.name}  ({reason})")
        if ok:
            valid.append(v)
            labels.append(v.name)
        elif reason == UNINDEXED_REASON:
            unindexed.append(v)
    if unindexed:
        print(
            "\nhint: the unindexed video(s) look like unfinalized recordings (the\n"
            "recorder was interrupted before writing the AVI index). The frames are\n"
            "intact; rebuild the index losslessly with ffmpeg, then rerun:\n"
        )
        for v in unindexed:
            fixed = v.with_suffix(".fixed" + v.suffix)
            print(f"  ffmpeg -i {v} -c copy {fixed} && mv {fixed} {v}")
        print()
    if not valid:
        print("no valid videos to process.")
        return []
    if len(valid) == 1 or _confirm_all(len(valid)):
        return valid
    idx = checkbox_select(labels)
    chosen = [valid[i] for i in idx]
    print(f"selected {len(chosen)}/{len(valid)} video(s):")
    for c in chosen:
        print(f"  • {c.name}")
    return chosen
