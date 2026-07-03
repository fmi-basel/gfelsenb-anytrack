"""Pose-model training via sleap-nn (Milestone B3). Entry point: ``anytrack-train``.

Trains a **centered-instance** keypoint model from hand-labeled data — the model
type that matches this package's pipeline, where the classical tracker already
provides a centroid and we run pose on a fixed-size crop around it.

Input is either a native label-store JSON (from ``anytrack-label``, exported to
``.slp`` on the fly) or an existing ``.slp``. Heavy deps (torch + sleap-nn) are
imported lazily, so importing this module stays cheap and the classical install
is unaffected. Install the backend with::

    uv pip install --prerelease=allow 'sleap-nn[torch]'

The trained model lands in ``<out_dir>/<run_name>``; point ``cfg.sleap_model_path``
at it (and set ``pose_enabled=True``) to run pose inference (Milestone B4/B5).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from ..config import load_config


def _accelerator(device: str) -> str:
    """Map this package's device string to a sleap-nn/Lightning accelerator."""
    return {"cuda": "gpu", "gpu": "gpu", "mps": "mps", "cpu": "cpu"}.get(
        (device or "auto").lower(), "auto")


def resolve_slp(labels_path: Path) -> "tuple[Path, Optional[int]]":
    """Return ``(slp_path, crop_size_hint)``, exporting a label store if needed."""
    labels_path = Path(labels_path)
    suffix = labels_path.suffix.lower()
    if suffix == ".slp":
        return labels_path, None
    if suffix == ".json":
        from .labeling import LabelStore
        store = LabelStore.load(labels_path)
        if store.n_labeled() == 0:
            raise ValueError(f"{labels_path} has no confirmed labeled frames to train on.")
        slp = labels_path.with_suffix(".slp")
        store.export_slp(slp)
        return slp, store.crop_size
    raise ValueError(f"labels must be a .json label store or a .slp file, got {labels_path.suffix!r}")


def train_pose(
    labels_path,
    cfg=None,
    *,
    out_dir=None,
    run_name: str = "fly_centered_instance",
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    crop_size: Optional[int] = None,
    validation_fraction: float = 0.1,
    steps_per_epoch: Optional[int] = None,
    backbone: str = "unet",
    intensity_aug: Optional[List[str]] = None,
    geometry_aug: Optional[List[str]] = None,
    device: Optional[str] = None,
    show_progress: bool = True,
) -> Path:
    """Train a centered-instance pose model and return the run directory.

    ``labels_path`` is a ``.json`` label store (auto-exported to ``.slp``) or an
    existing ``.slp``. Augmentation defaults suit a small hand-labeled set:
    brightness/contrast (illumination) + rotation/translate (pose variety and
    tolerance to the tracker centroid not sitting exactly on the anchor).
    """
    cfg = cfg or load_config()
    slp, crop_hint = resolve_slp(Path(labels_path))
    crop_size = int(crop_size or crop_hint or getattr(cfg, "crop_size", 96))
    device = device or getattr(cfg, "pose_device", "auto")

    out_dir = Path(out_dir) if out_dir is not None else slp.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    from sleap_nn.train import train as sleap_train

    sleap_train(
        train_labels_path=[str(slp)],
        validation_fraction=validation_fraction,
        crop_size=crop_size,
        head_configs="centered_instance",
        backbone_config=backbone,
        ensure_grayscale=True,                       # match B4 (1-channel crops)
        use_augmentations_train=True,
        intensity_aug=intensity_aug if intensity_aug is not None else ["brightness", "contrast"],
        geometry_aug=geometry_aug if geometry_aug is not None else ["rotation", "translate"],
        batch_size=batch_size,
        max_epochs=epochs,
        learning_rate=learning_rate,
        train_steps_per_epoch=steps_per_epoch,
        trainer_accelerator=_accelerator(device),
        enable_progress_bar=show_progress,
        save_ckpt=True,
        ckpt_dir=str(out_dir),
        run_name=run_name,
        seed=42,
    )
    return out_dir / run_name


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="anytrack-train",
        description="Train a centered-instance pose model from labels (Milestone B3).")
    ap.add_argument("--labels", required=True, type=Path,
                    help="Label store .json (from anytrack-label) or an existing .slp.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory to hold the <run-name> model folder (default: <labels dir>/models).")
    ap.add_argument("--run-name", default="fly_centered_instance", help="Model run/folder name.")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--crop-size", type=int, default=None, help="Override crop size (default: from store/cfg).")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--steps-per-epoch", type=int, default=None,
                    help="Cap training steps per epoch (small = quick smoke run).")
    ap.add_argument("--backbone", default="unet",
                    help="sleap-nn backbone (unet, unet_medium_rf, convnext_tiny, ...).")
    ap.add_argument("--device", default=None, help="auto|mps|cuda|cpu (default: cfg.pose_device).")
    args = ap.parse_args(argv)

    if not args.labels.exists():
        ap.error(f"labels not found: {args.labels}")

    cfg = load_config()
    model_dir = train_pose(
        args.labels, cfg, out_dir=args.out_dir, run_name=args.run_name,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        crop_size=args.crop_size, validation_fraction=args.val_fraction,
        steps_per_epoch=args.steps_per_epoch, backbone=args.backbone, device=args.device,
    )
    print(f"\nmodel saved to: {model_dir}")
    print("to use it for pose inference, set in your config:")
    print(f'    sleap_model_path = "{model_dir}"')
    print('    pose_backend = "sleap-nn"')
    print("    pose_enabled = true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
