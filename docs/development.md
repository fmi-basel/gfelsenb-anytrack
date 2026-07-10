# Development

## Setup & tests

```bash
uv venv
uv pip install -e '.[docs]'      # add pose/sam extras as needed
uv run python -m pytest          # use `python -m pytest`, NOT bare `pytest`
```

A handful of tests need a sample video and are skipped unless `ANYTRACK_TEST_VIDEO`
points at one.

## Repository layout

| Path | What |
|------|------|
| `anytrack/` | The package (pipeline, CLI, GUIs, pose). |
| `tests/` | Test suite. |
| `docs/` | This documentation (MkDocs Markdown source). |
| `notes/` | **Dev-only** internal analysis HTMLs — see below. |
| `mkdocs.yml` | Docs site config. |
| `.github/workflows/docs.yml` | Builds + deploys the docs to GitHub Pages. |

Key modules: `run.py` (orchestration), `roi.py` (arena detection), `preprocess.py`
+ `background.py` (backgrounds), `detector.py` (per-frame detection), `tracker.py`
(linking), `kinematics.py`, `odor_port.py`, `qc.py`, `writer.py`/`session.py`
(output), `tracking_fast.py` (streaming path), `config.py`.

## Branch model

- **`dev`** — active development; carries `notes/` (internal analysis HTMLs).
- **`main`** — the released branch; **`notes/` is removed here** so releases stay
  clean and clones stay small.

Because `notes/` lives only on `dev`, the two branches deliberately diverge. Keep
that in mind when merging (see the release procedure).

## Documentation

The docs are a [MkDocs](https://www.mkdocs.org/) site with the Material theme.

```bash
uv pip install -e '.[docs]'
mkdocs serve          # live preview at http://127.0.0.1:8000
mkdocs build --strict # build into ./site (CI uses --strict)
```

`.github/workflows/docs.yml` builds and deploys the site to **GitHub Pages** on
every push to `main` (and on manual dispatch). Enable it once under
**Settings → Pages → Build and deployment → Source: GitHub Actions**. Published at
<https://fmi-basel.github.io/gfelsenb-anytrack/>.

## Release procedure

1. Land the release changes on `dev`; bump `version` in `pyproject.toml` and update
   `CHANGELOG.md`.
2. Bring `main` up to date **without** the dev-only notes:
   ```bash
   git checkout main
   git merge dev
   git rm -r --ignore-unmatch notes    # keep notes/ off main
   git commit -m "chore(main): drop dev-only notes/"   # only if the merge re-added it
   git push origin main
   ```
3. Tag the release (annotated) and push it, then cut a GitHub Release:
   ```bash
   git tag -a vX.Y.Z -m "anytrack vX.Y.Z — <summary>"
   git push origin vX.Y.Z
   gh release create vX.Y.Z --title "anytrack vX.Y.Z" --notes-file CHANGELOG.md
   git checkout dev
   ```

!!! warning "Published tags are immutable"
    Once a tag/release is pushed, treat it as frozen — ship fixes as the next patch
    (`vX.Y.Z+1`) rather than moving the tag.

## Conventions

- Match the surrounding code's style, naming, and comment density.
- Keep the classical pipeline dependency-light; heavy/optional backends (pose, SAM)
  stay behind extras and lazy imports.
