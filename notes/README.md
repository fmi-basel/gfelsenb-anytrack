# Development notes (dev-only)

These HTML files are **internal development artifacts** — analysis write-ups,
benchmarks, and the live pipeline-status tracker used while building `anytrack`.
They are intentionally kept on the `dev` branch only and are **removed from
`main`** so they don't ship with releases or bloat clones.

- `pipeline.html` — living status of every pipeline stage + test count.
- `background_methods.html`, `background_modelling.html`, `streaming_bg_parity.html`
  — background-model comparisons, de-ghosting, and streaming-vs-cv2 parity studies.
- `centroid_tracking.html` — detection/tracking analysis.
- `optimization_planning.html` — performance planning (speed × error matrix).
- `training-guide.html`, `tutorial.html` — early guides; user-facing content now
  lives in the published docs (`docs/`, served at
  https://fmi-basel.github.io/gfelsenb-anytrack/).

User-facing documentation is the MkDocs site under `docs/`, not these files.
