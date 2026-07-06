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
    cv_threads_per_worker: int = 1  # OpenCV threads inside each tracking worker (0 = leave default)
    roi_stream_decode: bool = True  # Decode once and pipe raw gray crops to trackers (FIFO); else encode ROI mp4s
    bg_reuse_for_roi: bool = True  # Reuse the (unblurred) full-frame uniform bg per ROI instead of per-worker/first-N GMM (fixes streaming first-100-frame bake-in)
    bg_deghost: bool = True          # Lift a baked-in fly (very-long-dwell blind spot) to arena brightness via a fixture-safe near-max pass
    bg_deghost_percentile: float = 98  # Per-pixel near-max used to recover the arena where a fly baked in (robust to a fly that leaves >~2% of frames)
    bg_deghost_margin: int = 15      # Only lift pixels this many gray levels darker than their near-max (≈ thr_fixed; ignores clean arena + true dark fixtures)
    use_hw_decode: bool = True      # Try hardware-accelerated FFmpeg decode (probed; silent SW fallback if unsupported)
    hw_decode_backend: str = "auto" # auto|videotoolbox|none
    batch_concurrency: int = 0      # Videos processed at once in batch mode (0 = auto from cores, 1 = sequential)
    batch_core_budget: int = 0      # Cores to target for batch concurrency (0 = auto: performance cores)
    track_stride: int = 1           # Track every Nth frame; interpolate the rest (1 = every frame). Saves decode+detection
    detect_params_ref_downscale: int = 2  # Reference downscale the area/morph thresholds are tuned for (rescaled at other downscales)

    # Background adaptation (opt-in). Off by default so tracking output is
    # unchanged vs. the static GMM; enable for slowly drifting illumination.
    bg_drift_correction: bool = False   # Correct global brightness drift from safe pixels
    bg_protect_radius_px: float = 25.0  # Full-res radius of the centroid exclusion zone
    bg_fg_dilate_px: int = 7            # Dilation (px) of the foreground exclusion mask
    bg_asym_update: bool = False        # Asymmetric per-pixel update on safe pixels
    bg_step_up: float = 1.0             # Update step toward brighter observations
    bg_step_down: float = 0.02          # Update step toward darker observations (<< step_up)

    # Dynamic centroid crops (for pose / labeling)
    crop_size: int = 96                 # NxN crop centered on the tracked centroid
                                        # (fly ~40-50px + wing/jitter margin; override via --crop-size)
    crop_pad_mode: str = "background"   # "background" or "edge"
    crop_export_dir: str = ""           # Where exported crops + manifest are written

    # Output
    output_format: str = "parquet"      # "parquet" or "csv"
    output_dir: str = ""                # Default directory for saved results

    # Pose (Milestone B). The keypoint *schema* is a graph, so it lives in a JSON
    # skeleton sidecar (pose_skeleton), not here — config.toml is scalar-only.
    pose_enabled: bool = False          # run the pose stage after tracking
    pose_backend: str = "sleap-nn"      # concrete PoseEngine backend
    sleap_model_path: str = ""          # trained model path (empty -> mock engine)
    pose_skeleton: str = ""             # path to a skeleton JSON (empty -> built-in fly5)
    pose_every_n: int = 1               # infer pose every Nth tracked frame (1 = every frame)
    pose_batch_size: int = 64           # crops per inference batch
    pose_device: str = "auto"           # torch device for the engine: auto|mps|cuda|cpu
    # Per-node label colors for the labeling GUI, as "name:#hex,name:#hex,..."
    # (config.toml is scalar-only, so this is a string). Nodes not listed fall
    # back to a built-in default, then a distinct-color cycle.
    pose_node_colors: str = ("head:#ff5252,thorax:#ffd740,abdomen_tip:#40c4ff,"
                             "wingL:#69f0ae,wingR:#e040fb")
    # Temporal context loaded around each labeled frame in the GUI: +/- this many
    # neighbor frames (default 10 -> ~20 frames around it). Scrub through them to
    # read the fly's dynamics (motion direction disambiguates head/tail). Neighbor
    # crops use the anchor centroid (fixed box), so labels overlay consistently.
    pose_label_context: int = 10
    # Pose QC (B5): confidence gate for overlay/flags, and the frame-to-frame
    # body-axis reversal that flags a head/tail flip.
    pose_conf_min: float = 0.2          # keypoints below this are flagged / not drawn
    pose_headtail_flip_deg: float = 120.0
    pose_wing_extend_frac: float = 0.15  # wing lateral offset (frac of body len) to count as "extended"

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
    qc_crop_output_size: int = 800      # output px (square) for a cropped overlay (--crop-roi/--follow)

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
