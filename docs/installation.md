# Installation

## Requirements

- **Python ≥ 3.10**
- **FFmpeg** and **ffprobe** on your `PATH` — used for decoding and sampling frames.
  Verify with `ffmpeg -version`. Install via your platform's package manager
  (`brew install ffmpeg`, `apt install ffmpeg`, `conda install -c conda-forge ffmpeg`).
- A working [`uv`](https://docs.astral.sh/uv/) is recommended (Astral's package
  manager). Plain `pip` also works.

## Install with uv

From a clone of the repository:

```bash
git clone https://github.com/fmi-basel/gfelsenb-anytrack.git
cd gfelsenb-anytrack
uv venv
uv pip install -e .
```

The console scripts (`anytrack`, `anytrack-run`, …) are installed into the venv.
Activate it (`source .venv/bin/activate`) or prefix commands with `uv run`, e.g.
`uv run anytrack run --video ...`.

## Optional extras

| Extra | Install | Adds |
|-------|---------|------|
| `pose` | `uv pip install -e '.[pose]'` | `sleap-io` — the SLEAP label/export bridge (Milestone B). |
| `sam`  | `uv pip install -e '.[sam]'`  | `segment-anything` — the optional SAM odor-port backend. |
| `docs` | `uv pip install -e '.[docs]'` | `mkdocs-material` — build/preview this documentation. |

!!! note "Pose backend"
    The pose *training/inference* backend (`sleap-nn` + `torch`) is **not** declared
    as a dependency: it is prerelease and needs Python ≥ 3.11, which would constrain
    the base install. Install it directly into your environment when you need it:

    ```bash
    uv pip install --prerelease=allow 'sleap-nn[torch]'
    ```

## Verify

```bash
uv run anytrack            # lists subcommands
uv run anytrack-run --help # options for the main pipeline
uv run python -m pytest    # run the test suite (use python -m pytest, not pytest)
```
