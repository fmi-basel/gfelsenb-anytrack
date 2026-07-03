"""sleap-nn pose engine (Milestone B4).

Heavy backends (``torch`` + ``sleap-nn``) are imported **lazily inside the
constructor**, so importing :mod:`anytrack.pose` stays dependency-free and the
classical tracking install is unaffected.

Loading and inference go through sleap-nn's supported high-level API
(:func:`sleap_nn.load_models` -> a ``Predictor``; ``Predictor.predict`` on a
``sleap_io.Labels``). This was chosen after inspecting the installed package:
sleap-nn's internal inference layers are wrapper objects (not plain
``nn.Module``\\s), so feeding raw crops to them is unsupported and brittle. The
robust path is to hand the ``Predictor`` a ``Labels`` whose instances are
anchored at our tracker centroids and let it crop + predict + refine.

The tracks-dataframe -> ``Labels`` -> predict -> long-format pose-table wiring
(the pose *stage*) lands in Milestone B5; this module owns loading the trained
model and running the predictor.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def _torch_device(device: str) -> str:
    """Resolve this package's device string to a concrete torch device."""
    d = (device or "auto").lower()
    if d not in ("auto", ""):
        return "gpu" if d == "cuda" else d
    try:
        import torch
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class SleapNNEngine:
    """Trained centered-instance model loaded via sleap-nn's ``Predictor``.

    ``predict_labels`` is the real inference entry point (used by the B5 pose
    stage). ``infer_batch`` (the crop-based :class:`~anytrack.pose.engine.PoseEngine`
    protocol) is intentionally unsupported here — see the module docstring.
    """

    def __init__(self, model_path: str, skeleton=None, device: str = "auto",
                 peak_threshold: float = 0.2):
        self.model_path = str(model_path)
        self.keypoint_names: Optional[List[str]] = list(skeleton.nodes) if skeleton else None
        self.peak_threshold = float(peak_threshold)
        self._device = _torch_device(device)
        self.predictor = None
        self._load()

    def _load(self) -> None:
        try:
            import sleap_nn
        except ImportError as e:  # pragma: no cover - only without the optional dep
            raise ImportError(
                "SleapNNEngine requires sleap-nn (the pose backend):\n"
                "    uv pip install --prerelease=allow 'sleap-nn[torch]'"
            ) from e
        # load_models(model_paths, **kwargs) -> Predictor. Pass device if the
        # installed version accepts it; fall back gracefully otherwise.
        try:
            self.predictor = sleap_nn.load_models([self.model_path], device=self._device)
        except TypeError:
            self.predictor = sleap_nn.load_models([self.model_path])
        if self.keypoint_names is None:
            skel = getattr(self.predictor, "skeleton", None)
            if skel is not None:
                self.keypoint_names = [n.name for n in skel.nodes]

    def predict_labels(self, labels, **kwargs):
        """Run the predictor on a ``sleap_io.Labels`` and return predicted labels.

        ``labels`` instances should be anchored at the tracker centroids; the
        centered-instance predictor crops around each anchor, runs the model, and
        applies sub-pixel (integral) refinement.
        """
        kwargs.setdefault("make_labels", True)
        kwargs.setdefault("peak_threshold", self.peak_threshold)
        return self.predictor.predict(labels, **kwargs)

    def infer_batch(self, crops: np.ndarray):
        raise NotImplementedError(
            "sleap-nn inference runs through predict_labels() on a sleap_io.Labels "
            "built from tracker centroids, not crop batches. The pose-stage wiring "
            "(tracks -> Labels -> predict -> pose table) lands in Milestone B5."
        )
