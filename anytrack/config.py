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
    # Background modelling
    bg_method: str = "mean_excluding_fg"  # mean, median, mean_excluding_fg, mog2
    bg_max_frames: int = 300              # upper bound for sampling
    bg_min_frames: int = 30               # lower bound for sampling
    bg_converge_eps: float = 2.5          # stop when mean abs diff between successive bg < eps
    bg_fg_thresh: int = 25                # threshold for excluding moving pixels in mean_excluding_fg
    bg_inpaint_ghosts: bool = True
    bg_ghost_area_min: int = 20
    bg_ghost_area_max: int = 600

    # ROI detection (Hough)
    roi_hough_dp: float = 1.2
    roi_hough_min_dist_ratio: int = 3
    roi_hough_param1: int = 120
    roi_hough_param2: int = 35
    min_radius_ratio: float = 0.18
    max_radius_ratio: float = 0.3

    # Tracking
    thr_method: str = "otsu"  # otsu or fixed
    thr_fixed: int = 35
    morph_open: int = 3
    morph_close: int = 5
    max_centroids_per_roi: int = 1
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

    # Fast mode (parallel ROI tracking with FFmpeg preprocessing)
    fast_mode: bool = True
    roi_downscale: int = 2  # Downscale factor for ROI videos (1, 2, or 4)
    n_tracking_workers: int = 4  # Number of parallel tracking workers
    use_hw_encode: bool = True  # Try hardware encoding (VideoToolbox on macOS)
    cleanup_roi_videos: bool = True  # Delete temp ROI videos after tracking

def config_path() -> Path:
    cfg_dir = Path(user_config_dir(APP_NAME))
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "config.toml"

def load_config() -> AnyTrackConfig:
    path = config_path()
    print(path)
    if not path.exists():
        cfg = AnyTrackConfig()
        save_config(cfg)
        return cfg
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    # tolerate unknown keys
    kwargs: Dict[str, Any] = asdict(AnyTrackConfig())
    for k, v in data.items():
        if k in kwargs:
            kwargs[k] = v
    return AnyTrackConfig(**kwargs)

def save_config(cfg: AnyTrackConfig) -> None:
    path = config_path()
    path.write_text(_dump_toml(asdict(cfg)), encoding="utf-8")
