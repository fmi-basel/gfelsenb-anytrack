from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict
import tomllib
from platformdirs import user_config_dir
from pathlib import Path

APP_NAME = "anytrack"

def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\"')

def _dump_toml(d: Dict[str, Any]) -> str:
    # Minimal TOML writer for flat key/value config.
    lines = []
    for k, v in d.items():
        if isinstance(v, str):
            lines.append(f'{k} = "{_toml_escape(v)}"')
        elif isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        else:
            # fallback to string repr
            lines.append(f'{k} = "{_toml_escape(str(v))}"')
    return "\n".join(lines) + "\n"

@dataclass
class AnyTrackConfig:
    # GMM Background modeling
    gmm_n_samples: int = 100                # Number of frames to sample for background
    gmm_bic_improvement: float = 10.0       # BIC improvement threshold for 2-component GMM
    gmm_min_std: float = 10.0               # Minimum std deviation to fit GMM (skip uniform pixels)
    gmm_reg_covar: float = 1e-3             # Covariance regularization for GMM fitting
    gmm_lowp: float = 120.0                 # Minimum brightness value for background

    # Arena detection
    arena_detection_enabled: bool = True    # Automatically detect arenas during background build
    arena_min_area_frac: float = 0.01       # Minimum arena area as fraction of image size
    arena_blur_sigma: float = 3.0           # Gaussian blur sigma within detected arenas

    # ROI detection (Hough)
    roi_hough_dp: float = 1.2
    roi_hough_min_dist_ratio: int = 3
    roi_hough_param1: int = 120
    roi_hough_param2: int = 35
    min_radius_ratio: float = 0.18
    max_radius_ratio: float = 0.3

    # Tracking
    bgdiff_type: str = "dark"  # dark (default), bright, or absolute
    thr_method: str = "fixed"  # otsu or fixed
    thr_fixed: int = 15
    morph_open: int = 3
    morph_close: int = 5
    max_centroids_per_roi: int = 1
    # How many blobs the detector surfaces to the tracker per frame. Must be > 1
    # so the tracker can disambiguate the fly from static artifacts (debris /
    # reflections) by proximity; the tracker still emits a single object.
    detect_max_candidates: int = 5
    expected_fly_area_min: int = 10
    expected_fly_area_max: int = 1500

    # Kalman / linking
    max_jump_px: float = 40.0
    miss_tolerance: int = 15  # frames
    use_kalman: bool = True

    # Kinematics / scaling
    arena_diameter_mm: float = 75.0

    # GUI
    preview_downscale: float = 1.0  # set <1 for speed on large frames
    trajectory_alpha: float = 0.5  # alpha transparency for trajectory overlay (0.0-1.0)

    # Fast mode (parallel ROI tracking with FFmpeg preprocessing)
    fast_mode: bool = True
    roi_downscale: int = 2  # Downscale factor for ROI videos (1, 2, or 4)
    n_tracking_workers: int = 4  # Number of parallel tracking workers
    use_hw_encode: bool = True  # Try hardware encoding (VideoToolbox on macOS)
    cleanup_roi_videos: bool = True  # Delete temp ROI videos after tracking

    # Background adaptation (opt-in). Off by default so tracking output is
    # unchanged vs. the static GMM; enable for slowly drifting illumination.
    bg_drift_correction: bool = False   # Correct global brightness drift from safe pixels
    bg_protect_radius_px: float = 25.0  # Full-res radius of the centroid exclusion zone
    bg_fg_dilate_px: int = 7            # Dilation (px) of the foreground exclusion mask
    bg_asym_update: bool = False        # Asymmetric per-pixel update on safe pixels
    bg_step_up: float = 1.0             # Update step toward brighter observations
    bg_step_down: float = 0.02          # Update step toward darker observations (<< step_up)

    # Dynamic centroid crops (for pose / labeling)
    crop_size: int = 128                # NxN crop centered on the tracked centroid
    crop_pad_mode: str = "background"   # "background" or "edge"
    crop_export_dir: str = ""           # Where exported crops + manifest are written

    # Output
    output_format: str = "parquet"      # "parquet" or "csv"
    output_dir: str = ""                # Default directory for saved results

    # Quality control
    qc_min_contrast: float = 10.0       # flag_low_contrast when detection contrast < this
    qc_montage_max: int = 25            # max flagged-frame thumbnails in the QC montage
    qc_montage_tile: int = 96           # thumbnail size (px) in the QC montage
    qc_overlay_downscale: int = 2       # downscale factor for the overlay video (1 = full res)
    qc_overlay_crf: int = 23            # libx264 CRF for the overlay video (lower = better/bigger)
    qc_overlay_stride: int = 5          # render every Nth frame in the overlay (1 = every frame)
    # Hardware H.264 (VideoToolbox) is ~1.8x faster than libx264 only at stride=1;
    # at the default stride=5 libx264 is faster AND ~12x smaller (adaptive CRF vs
    # fixed bitrate), so HW defaults OFF. Turn on for full-framerate overlays.
    qc_overlay_hw: bool = False         # use hardware H.264 encode (VideoToolbox) for the overlay
    qc_overlay_hw_bitrate: str = "6M"   # target bitrate for the hardware encoder

    # Local staging (copy source video off slow/network storage before processing).
    # Default off: benchmarks showed the network wasn't the bottleneck (preprocessing
    # is decode-bound). Set "auto"/"always" for genuinely slow/high-latency storage.
    stage_video_locally: str = "never"  # "never" | "auto" (copy off non-local storage) | "always"
    stage_dir: str = ""                 # Local scratch dir (default: platformdirs cache)
    cleanup_staged: bool = False        # Delete the staged copy after each run (else cache it)

def config_path() -> Path:
    cfg_dir = Path(user_config_dir(APP_NAME))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "config.toml"

def load_config() -> AnyTrackConfig:
    path = config_path()
    if not path.exists():
        cfg = AnyTrackConfig()
        save_config(cfg)
        return cfg
    data = tomllib.loads(path.read_text(encoding="utf-8"))

    # Config migration: detect old background parameters and remove them
    old_bg_params = [
        'bg_method', 'bg_brightness_threshold', 'bg_converge_eps',
        'bg_fg_thresh', 'bg_inpaint_ghosts', 'bg_ghost_area_min',
        'bg_ghost_area_max', 'bg_min_frames', 'bg_max_frames'
    ]
    needs_migration = any(k in data for k in old_bg_params)
    if needs_migration:
        print("Migrating config from old background system to GMM background system...")
        for k in old_bg_params:
            data.pop(k, None)

    # tolerate unknown keys
    kwargs: Dict[str, Any] = asdict(AnyTrackConfig())
    for k, v in data.items():
        if k in kwargs:
            kwargs[k] = v

    cfg = AnyTrackConfig(**kwargs)

    # Save migrated config
    if needs_migration:
        save_config(cfg)
        print("Config migration complete. New GMM parameters applied.")

    return cfg

def save_config(cfg: AnyTrackConfig) -> None:
    path = config_path()
    path.write_text(_dump_toml(asdict(cfg)), encoding="utf-8")
