"""
The ``anytrack`` command: a thin subcommand dispatcher.

``anytrack <sub> ...`` routes to the identically-named ``anytrack-<sub>`` console
script, so ``anytrack run --video X`` behaves exactly like ``anytrack-run
--video X``. This is the single front door to the whole tool; the individual
``anytrack-*`` scripts remain for direct/scripted use.

``anytrack`` no longer starts the GUI — the GUI is now an explicit ``anytrack
gui`` (equivalently ``anytrack-gui``). Targets are resolved lazily, so a
subcommand only imports its own (sometimes heavy — tkinter, torch, matplotlib)
dependencies; ``anytrack run`` never pulls in the GUI stack.
"""
from __future__ import annotations

import importlib
import sys
from typing import List, Optional

# Subcommand → "module:function". Each entry mirrors the anytrack-<sub> console
# script in pyproject.toml; keep the two in sync when adding a command.
SUBCOMMANDS = {
    "run":         "anytrack.run:main",
    "validate":    "anytrack.validate:main",
    "qc":          "anytrack.qc:main",
    "bg":          "anytrack.cli:bg_main",
    "roi":         "anytrack.cli:roi_main",
    "bench":       "anytrack.benchmark:main",
    "crop-export": "anytrack.cropper:main",
    "label":       "anytrack.pose.label_gui:main",
    "train":       "anytrack.pose.train:main",
    "debug":       "anytrack.gui.debug_app:main",
    "arena-debug": "anytrack.gui.arena_debug:main",
    "gui":         "anytrack.cli:gui_main",
}


def _print_usage() -> None:
    width = max(len(k) for k in SUBCOMMANDS)
    print("usage: anytrack <command> [args...]\n\ncommands:")
    for name in SUBCOMMANDS:
        print(f"  {name:<{width}}  → anytrack-{name}")
    print("\nRun 'anytrack <command> --help' for a command's options.")
    print("The GUI no longer starts by default — use 'anytrack gui'.")


def main(argv: Optional[List[str]] = None) -> int:
    """Route ``anytrack <sub> ...`` to the ``anytrack-<sub>`` entry point.

    Rewrites ``sys.argv`` to drop the subcommand token and delegates to the
    target's ``main()``, so each target parses its own args unchanged (some read
    ``sys.argv`` directly rather than taking an ``argv`` argument). ``anytrack``
    with no command — or ``-h``/``--help``/``help`` — prints the command list.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        _print_usage()
        return 0

    sub = argv[0]
    if sub not in SUBCOMMANDS:
        print(f"anytrack: unknown command {sub!r}\n")
        _print_usage()
        return 2

    module_path, func_name = SUBCOMMANDS[sub].split(":")
    fn = getattr(importlib.import_module(module_path), func_name)
    # Look like a direct `anytrack-<sub>` invocation so the target's own argparse
    # (prog name, sys.argv reads) works exactly as if called standalone.
    sys.argv = [f"anytrack-{sub}"] + argv[1:]
    rc = fn()
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
