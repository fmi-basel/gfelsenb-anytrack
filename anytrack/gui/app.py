from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import tkinter as tk
import time

import cv2
import numpy as np
import queue
from PIL import Image, ImageTk
from tksheet import Sheet

from ..config import load_config, save_config, AnyTrackConfig
from ..io import load_video_asset
from ..background import build_background
from ..roi import detect_circular_rois
from ..session import TrackingSession
from ..models import VideoAsset, TrackingResult
from .progress import TrackingProgressDialog, make_progress_hook, ETAEstimator

class AnyTrackApp(tb.Window):
    def __init__(self):
        super().__init__(themename="flatly")
        self.title("anytrack")
        self.geometry("1200x750")
        # Centralized font size for small UI labels (match Labelframe label size)
        self.element_fontsize = 11
        self._apply_ui_fonts()


        self.cfg: AnyTrackConfig = load_config()

        self.video: Optional[VideoAsset] = None
        self.session: Optional[TrackingSession] = None
        self.result: Optional[TrackingResult] = None

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_cache: Dict[int, np.ndarray] = {}
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._current_frame_idx: int = 0

        self._last_disp_bgr = None
        self._resize_job = None   # <-- add this
        self._preview_wh = (0, 0)  # store preview labelframe size from Configure events
        self._last_render_t = 0.0  # monotonic time of last preview redraw
        self._final_redraw_job = None  # scheduled final HQ redraw after resizing stops
        self._final_redraw_delay_ms = 180  # debounce delay for HQ redraw

        self._background_gray: Optional[np.ndarray] = None  # cached background (grayscale)
        self.preview_mode: str = "video"  # "video" or "background"
        self._background_building: bool = False  # guard against concurrent background builds

        # Play/Pause functionality
        self._playing: bool = False
        self._play_job: Optional[str] = None
        self._play_fps: int = 30

        # Pan and Zoom functionality
        self._zoom_level: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._drag_start_x: Optional[int] = None
        self._drag_start_y: Optional[int] = None
        self._drag_start_pan_x: float = 0.0
        self._drag_start_pan_y: float = 0.0

        # Background debug window (optional)
        self.debug_bg_var = tk.BooleanVar(value=False)
        self._bg_debug_win = None
        self._bg_debug_q: Optional[queue.Queue] = None
        self._bg_debug_job = None
        self._bg_dbg_frame_img = None
        self._bg_dbg_model_img = None
        self._bg_dbg_indices_list = []          # list[int] sampled indices for current trial
        self._bg_dbg_cache = {}                # dict[int, dict] frame_idx -> {"frame": np.ndarray, "bg": np.ndarray}

        self._build_ui()

    def _apply_ui_fonts(self):
        """Apply centralized font sizing to key ttk widget styles."""
        # tb.Window provides a ttkbootstrap Style instance as `self.style`.
        # Labelframe titles use the ttk style `TLabelframe.Label`.
        try:
            self.style.configure(
                "TLabelframe.Label",
                font=("TkDefaultFont", self.element_fontsize),
            )
        except Exception:
            # If style isn't ready for some reason, fail silently.
            pass

    def _build_ui(self):
        # Top menu
        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open video…", command=self.open_asset)
        filemenu.add_command(label="Save config", command=self.save_cfg)

        filemenu.add_separator()

        filemenu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        filemenu_debug = tk.Menu(menubar, tearoff=0)
        filemenu_debug.add_checkbutton(label="Background debug window", variable=self.debug_bg_var)
        menubar.add_cascade(label="Debug", menu=filemenu_debug)
        self.config(menu=menubar)

        # Layout
        outer = tb.Frame(self, padding=6)
        outer.pack(fill=BOTH, expand=True)

        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=3)
        outer.rowconfigure(0, weight=1)

        left = tb.Labelframe(outer, text="Organizer", padding=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        right = tb.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)

        # Organizer tree
        self.tree = tb.Treeview(left, columns=("type",), show="tree headings")
        self.tree.heading("#0", text="Item")
        self.tree.heading("type", text="Type")
        self.tree.column("type", width=80, anchor="w")
        self.tree.pack(fill=BOTH, expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree.bind("<Double-1>", self.on_tree_double_click)

        # Buttons
        btns = tb.Frame(left)
        btns.pack(fill=X, pady=(6,0))
        tb.Button(btns, text="Auto-detect ROIs", command=self.auto_rois, bootstyle=PRIMARY).pack(side=LEFT, padx=(0,6))
        tb.Button(btns, text="Run tracking", command=self.run_tracking, bootstyle=SUCCESS).pack(side=LEFT, padx=(0,6))
        tb.Button(btns, text="Export CSV…", command=self.export_csv, bootstyle=SECONDARY).pack(side=LEFT)

        # Preview canvas + slider
        preview = tb.Labelframe(right, text="Preview", padding=6)
        preview.grid(row=0, column=0, sticky="nsew")
        preview.rowconfigure(0, weight=1)
        preview.rowconfigure(1, weight=0)
        preview.columnconfigure(0, weight=1)
        self.preview_frame = preview
        self.preview_lf = preview
        self.preview_frame.bind("<Configure>", self.on_preview_resize, add="+")

        self.canvas = tk.Canvas(preview, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Mouse bindings for pan and zoom
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows/macOS
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Double-Button-1>", self.on_canvas_double_click)
        # Trackpad pinch gesture (macOS) - only available in Tk 8.6.9+
        try:
            self.canvas.bind("<Magnify>", self.on_magnify)
        except Exception:
            pass  # Magnify event not supported in this Tk version

        slider_row = tb.Frame(preview)
        slider_row.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        slider_row.columnconfigure(0, weight=1)

        self.slider = tb.Scale(slider_row, from_=0, to=0, orient=HORIZONTAL, command=self.on_slider)
        self.slider.grid(row=0, column=0, sticky="ew")
        self.slider.configure(takefocus=True)

        # Play/Pause button
        self.play_pause_btn = tb.Button(
            slider_row,
            text="▶",
            width=3,
            command=self.toggle_play,
            bootstyle=SUCCESS
        )
        self.play_pause_btn.grid(row=0, column=1, sticky="w", padx=(8, 0))

        # Use a fixed-width font so digits don't shift visually.
        self.frame_label = tb.Label(
            slider_row,
            text="Frame: ---- / ----",
            font=("TkFixedFont", self.element_fontsize),
        )
        self.frame_label.grid(row=0, column=2, sticky="e", padx=(8, 0))

        # Click-to-jump on the trough
        self.slider.bind("<Button-1>", self.on_slider_click, add="+")

        # Arrow keys: step by one frame
        self.bind_all("<Left>", self.on_left_key)
        self.bind_all("<Right>", self.on_right_key)

        # Results table (TkSheet)
        table = tb.Labelframe(right, text="Results (preview)", padding=6)
        table.grid(row=1, column=0, sticky="nsew")
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)

        self.sheet = Sheet(table)
        self.sheet.enable_bindings(("single_select","row_select","column_select","drag_select","arrowkeys"))
        self.sheet.grid(row=0, column=0, sticky="nsew")

    def save_cfg(self):
        save_config(self.cfg)
        messagebox.showinfo("anytrack", "Config saved.")

    def _set_preview_mode(self, mode: str):
        # mode: "video" or "background"
        self.preview_mode = mode

        if mode == "background":
            if hasattr(self, "preview_lf"):
                self.preview_lf.configure(text="Background Preview")
            try:
                self.slider.configure(state="disabled")
            except Exception:
                pass
            try:
                self.frame_label.configure(text="Background")
            except Exception:
                pass
        else:
            if hasattr(self, "preview_lf"):
                self.preview_lf.configure(text="Preview")
            try:
                self.slider.configure(state="normal")
            except Exception:
                pass
            self._update_frame_label()

    def _ensure_background_async(self):
        if self.video is None:
            return
        if self._background_gray is not None:
            self.show_background()
            return

        # Don’t start multiple builds if user double-clicks repeatedly
        if self._background_building:
            return
        self._background_building = True

        dbg_enabled = bool(self.debug_bg_var.get())
        if dbg_enabled:
            self._open_bg_debug_window()

        def dbg_hook(event: str, payload: dict):
            if dbg_enabled:
                self._bg_debug_enqueue(event, payload)

        def worker():
            try:
                bg = build_background(
                    str(self.video.video_path),
                    method=self.cfg.bg_method,
                    k_min=max(10, self.cfg.bg_min_frames // 3),
                    k_max=max(60, self.cfg.bg_min_frames),
                    converge_eps=self.cfg.bg_converge_eps,
                    fg_thresh=self.cfg.bg_fg_thresh,
                    inpaint_ghosts=self.cfg.bg_inpaint_ghosts,
                    ghost_area_min=self.cfg.bg_ghost_area_min,
                    ghost_area_max=self.cfg.bg_ghost_area_max,
                    debug=dbg_enabled,
                    debug_hook=dbg_hook,
                    debug_stride=3,
                    debug_max_side=480,
                ).image
                self._background_gray = bg
                self.after(0, self.show_background)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("anytrack", f"Background build failed:\n{e}"))
            finally:
                self._background_building = False

        threading.Thread(target=worker, daemon=True).start()

    def show_background(self):
        if self._background_gray is None:
            return
        self._set_preview_mode("background")
        bgr = cv2.cvtColor(self._background_gray, cv2.COLOR_GRAY2BGR)
        self._last_disp_bgr = bgr
        self._render_bgr_to_canvas(bgr, resample="hq")

    def on_preview_resize(self, event):
        # Track latest preview size.
        self._preview_wh = (event.width, event.height)

        # FAST path while the user is actively resizing (mousemove):
        # Use a time-based throttle and a cheaper resampling filter.
        now = time.monotonic()
        if (now - self._last_render_t) >= 0.03:  # ~33 fps
            self._last_render_t = now
            if self._last_disp_bgr is not None:
                self._render_bgr_to_canvas(self._last_disp_bgr, resample="fast")

        # FINAL HQ redraw when resizing stops: debounce a high-quality redraw.
        if self._final_redraw_job is not None:
            self.after_cancel(self._final_redraw_job)
        self._final_redraw_job = self.after(self._final_redraw_delay_ms, self._final_hq_redraw)

    def _redraw_current(self):
        self._resize_job = None
        if self._last_disp_bgr is not None:
            self._render_bgr_to_canvas(self._last_disp_bgr, resample="hq")

    def _final_hq_redraw(self):
        self._final_redraw_job = None
        if self._last_disp_bgr is not None:
            self._render_bgr_to_canvas(self._last_disp_bgr, resample="hq")

    def _render_bgr_to_canvas(self, bgr: np.ndarray, resample: str = "hq"):
        # Render based on the *current canvas size* (canvas tracks the Labelframe via grid weights).
        cw = int(self.canvas.winfo_width())
        ch = int(self.canvas.winfo_height())
        if cw <= 1 or ch <= 1:
            # Layout may not be finalized yet (first draw right after opening).
            self.update_idletasks()
            cw = int(self.canvas.winfo_width())
            ch = int(self.canvas.winfo_height())
        cw = max(1, cw)
        ch = max(1, ch)

        disp = bgr
        if getattr(self.cfg, "preview_downscale", 1.0) != 1.0:
            disp = cv2.resize(disp, None, fx=self.cfg.preview_downscale, fy=self.cfg.preview_downscale)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        iw, ih = im.size

        # Aspect-fit into canvas
        scale = min(cw / iw, ch / ih)

        # Apply zoom
        scale *= self._zoom_level

        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))

        # Choose resampling mode
        Resampling = getattr(Image, "Resampling", Image)  # Pillow compat
        if resample == "fast":
            filt = Resampling.BILINEAR
        else:
            filt = Resampling.LANCZOS

        im_resized = im.resize((nw, nh), filt)

        # Letterbox to canvas size with pan offset
        frame_box = Image.new("RGB", (cw, ch), (0, 0, 0))

        # Calculate base position (centered)
        x0 = (cw - nw) // 2
        y0 = (ch - nh) // 2

        # Apply pan offset
        x0 += int(self._pan_x)
        y0 += int(self._pan_y)

        # Crop the image if it extends beyond canvas bounds
        # This handles the case where zoomed/panned image is larger than canvas
        paste_x = max(0, x0)
        paste_y = max(0, y0)

        crop_left = max(0, -x0)
        crop_top = max(0, -y0)
        crop_right = min(nw, cw - x0)
        crop_bottom = min(nh, ch - y0)

        if crop_right > crop_left and crop_bottom > crop_top:
            im_cropped = im_resized.crop((crop_left, crop_top, crop_right, crop_bottom))
            frame_box.paste(im_cropped, (paste_x, paste_y))

        self._photo = ImageTk.PhotoImage(frame_box)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw")

    def open_asset(self):
        video_path_str = filedialog.askopenfilename(
            title="Select AVI video",
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")],
        )
        if not video_path_str:
            return

        video_path = Path(video_path_str)
        csv_path = video_path.with_suffix(".csv")
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Timing CSV not found for video. Expected: {csv_path}"
            )

        self.video = load_video_asset(video_path, csv_path)
        self._background_gray = None
        self._set_preview_mode("video")
        self._background_building = False
        self._open_capture()
        self._populate_tree()
        self._set_slider_range()
        self._reset_zoom_pan()
        self.update_idletasks()
        self.show_frame(0)

    def _open_capture(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(self.video.video_path)) if self.video else None
        self._frame_cache.clear()

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        if not self.video:
            return
        vid_id = self.tree.insert("", "end", text=self.video.label(), values=("video",), open=True)
        # timing
        self.tree.insert(vid_id, "end", text=self.video.timing_csv_path.name, values=("timing",))
        self.tree.insert(vid_id, "end", text="Background", values=("background",))
        # ROIs
        rois_id = self.tree.insert(vid_id, "end", text="ROIs", values=("group",), open=True)
        for roi in self.video.rois:
            self.tree.insert(rois_id, "end", text=roi.name, values=("roi",))

        # Tracks
        trk_id = self.tree.insert(vid_id, "end", text="Tracks", values=("group",), open=True)
        if self.session and self.session.dataframe is not None and not self.session.dataframe.empty:
            for roi_name in sorted(self.session.dataframe["roi"].unique()):
                self.tree.insert(trk_id, "end", text=f"{roi_name}/track_1", values=("track",))

    def _set_slider_range(self):
        if not self.video or self.video.timing is None:
            self.slider.configure(from_=0, to=0)
            self._update_frame_label()
            return
        max_frame = int(self.video.timing["frame"].max())
        self.slider.configure(from_=0, to=max_frame)
        self._update_frame_label()

    def on_slider(self, val):
        if getattr(self, "preview_mode", "video") != "video":
            return
        try:
            idx = int(float(val))
        except Exception:
            return
        self.show_frame(idx)

    def _update_frame_label(self):
        # Fixed-width numeric fields to avoid the label "jumping" as digits change.
        try:
            max_frame = int(float(self.slider.cget("to")))
        except Exception:
            max_frame = 0

        total = max_frame + 1
        cur = int(self._current_frame_idx) + 1

        # Width equals number of digits in total (at least 4 for a stable look).
        width = max(4, len(str(total)))

        # Right-justify within the fixed field width.
        self.frame_label.configure(text=f"Frame: {cur:>{width}d} / {total:>{width}d}")

    def on_slider_click(self, event):
        """Jump the slider to the click position (anywhere on the bar)."""
        if getattr(self, "preview_mode", "video") != "video":
            return
        if self.video is None:
            return
        # Ensure geometry is up-to-date
        w = int(self.slider.winfo_width())
        if w <= 1:
            self.update_idletasks()
            w = int(self.slider.winfo_width())
        if w <= 1:
            return

        vmin = float(self.slider.cget("from"))
        vmax = float(self.slider.cget("to"))
        if vmax <= vmin:
            return

        # Map x coordinate to value
        x = max(0, min(event.x, w))
        frac = x / float(w)
        val = vmin + frac * (vmax - vmin)
        idx = int(round(val))

        self.slider.set(idx)
        self.show_frame(idx)

    def step_frame(self, delta: int):
        if getattr(self, "preview_mode", "video") != "video":
            return
        if self.video is None:
            return
        try:
            vmin = int(float(self.slider.cget("from")))
            vmax = int(float(self.slider.cget("to")))
        except Exception:
            return
        idx = int(self._current_frame_idx) + int(delta)
        idx = max(vmin, min(vmax, idx))
        self.slider.set(idx)
        self.show_frame(idx)

    def on_left_key(self, event=None):
        self.step_frame(-1)

    def on_right_key(self, event=None):
        self.step_frame(1)

    def toggle_play(self):
        """Toggle play/pause state for video playback."""
        if self.preview_mode != "video" or self.video is None:
            return

        self._playing = not self._playing

        if self._playing:
            # Start playing
            self.play_pause_btn.configure(text="⏸")
            self._play_next_frame()
        else:
            # Pause
            self.play_pause_btn.configure(text="▶")
            if self._play_job is not None:
                self.after_cancel(self._play_job)
                self._play_job = None

    def _play_next_frame(self):
        """Advance to the next frame during playback."""
        if not self._playing or self.preview_mode != "video" or self.video is None:
            self._playing = False
            self.play_pause_btn.configure(text="▶")
            return

        try:
            vmin = int(float(self.slider.cget("from")))
            vmax = int(float(self.slider.cget("to")))
        except Exception:
            return

        idx = int(self._current_frame_idx) + 1

        # Loop back to start when reaching the end
        if idx > vmax:
            idx = vmin

        self.slider.set(idx)
        self.show_frame(idx)

        # Schedule next frame
        delay_ms = int(1000 / self._play_fps)
        self._play_job = self.after(delay_ms, self._play_next_frame)

    def on_mouse_wheel(self, event):
        """Handle mouse wheel zoom."""
        if self._last_disp_bgr is None:
            return

        # Determine zoom direction
        if event.num == 4 or event.delta > 0:
            # Zoom in
            zoom_factor = 1.1
        elif event.num == 5 or event.delta < 0:
            # Zoom out
            zoom_factor = 0.9
        else:
            return

        # Update zoom level (clamp between 0.1 and 10.0)
        new_zoom = self._zoom_level * zoom_factor
        new_zoom = max(0.1, min(10.0, new_zoom))

        # Get mouse position relative to canvas
        canvas_x = event.x
        canvas_y = event.y

        # Adjust pan to zoom toward mouse position
        # The idea: keep the point under the mouse cursor fixed during zoom
        zoom_ratio = new_zoom / self._zoom_level

        self._pan_x = canvas_x - (canvas_x - self._pan_x) * zoom_ratio
        self._pan_y = canvas_y - (canvas_y - self._pan_y) * zoom_ratio

        self._zoom_level = new_zoom

        # Re-render with new zoom and pan
        self._render_bgr_to_canvas(self._last_disp_bgr, resample="hq")

    def on_magnify(self, event):
        """Handle trackpad pinch gesture for zooming (macOS)."""
        if self._last_disp_bgr is None:
            return

        # event.delta is the magnification factor
        # Positive values = zoom in (pinch out), negative = zoom out (pinch in)
        # Scale the delta to make gestures feel natural
        zoom_factor = 1.0 + (event.delta * 2.0)

        # Update zoom level (clamp between 0.1 and 10.0)
        new_zoom = self._zoom_level * zoom_factor
        new_zoom = max(0.1, min(10.0, new_zoom))

        # Get gesture center position (use canvas center if position not available)
        try:
            canvas_x = event.x
            canvas_y = event.y
        except AttributeError:
            # If position is not available, zoom toward canvas center
            canvas_x = self.canvas.winfo_width() // 2
            canvas_y = self.canvas.winfo_height() // 2

        # Adjust pan to zoom toward gesture center
        zoom_ratio = new_zoom / self._zoom_level

        self._pan_x = canvas_x - (canvas_x - self._pan_x) * zoom_ratio
        self._pan_y = canvas_y - (canvas_y - self._pan_y) * zoom_ratio

        self._zoom_level = new_zoom

        # Use fast rendering during gesture for smoothness
        self._render_bgr_to_canvas(self._last_disp_bgr, resample="fast")

    def on_canvas_press(self, event):
        """Handle mouse button press for panning."""
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._drag_start_pan_x = self._pan_x
        self._drag_start_pan_y = self._pan_y
        self.canvas.configure(cursor="fleur")

    def on_canvas_drag(self, event):
        """Handle mouse drag for panning."""
        if self._drag_start_x is None or self._drag_start_y is None:
            return

        if self._last_disp_bgr is None:
            return

        # Calculate drag delta
        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y

        # Update pan position
        self._pan_x = self._drag_start_pan_x + dx
        self._pan_y = self._drag_start_pan_y + dy

        # Re-render with new pan
        self._render_bgr_to_canvas(self._last_disp_bgr, resample="fast")

    def on_canvas_release(self, event):
        """Handle mouse button release after panning."""
        self._drag_start_x = None
        self._drag_start_y = None
        self.canvas.configure(cursor="")

        # Final high-quality render
        if self._last_disp_bgr is not None:
            self._render_bgr_to_canvas(self._last_disp_bgr, resample="hq")

    def on_canvas_double_click(self, event):
        """Reset zoom and pan on double-click."""
        self._reset_zoom_pan()

    def _reset_zoom_pan(self):
        """Reset zoom and pan to default values."""
        self._zoom_level = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

        if self._last_disp_bgr is not None:
            self._render_bgr_to_canvas(self._last_disp_bgr, resample="hq")

    def _read_frame(self, idx: int) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        if idx in self._frame_cache:
            return self._frame_cache[idx]
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._frame_cache[idx] = frame
        # keep cache small
        if len(self._frame_cache) > 40:
            for k in list(self._frame_cache.keys())[:10]:
                self._frame_cache.pop(k, None)
        return frame

    def show_frame(self, idx: int):
        self._current_frame_idx = idx
        frame = self._read_frame(idx)
        if frame is None:
            return

        # Overlay ROIs + tracked centroids if available
        disp = frame.copy()

        if self.video:
            for roi in self.video.rois:
                cv2.circle(disp, (int(roi.cx), int(roi.cy)), int(roi.r), (0, 255, 0), 2)

        if self.session and self.session.dataframe is not None and not self.session.dataframe.empty:
            df = self.session.dataframe
            fdf = df[df["frame"] == idx]
            for _, r in fdf.iterrows():
                x, y = int(r["x"]), int(r["y"])
                cv2.circle(disp, (x, y), 4, (0, 0, 255), -1)
                # angle arrow
                ang = np.deg2rad(float(r["angle_deg"]))
                x2 = int(x + 20 * np.cos(ang))
                y2 = int(y + 20 * np.sin(ang))
                cv2.line(disp, (x, y), (x2, y2), (255, 0, 0), 2)
            for each_roi in df.roi.unique():
                roidf = df[df.roi == each_roi]
                x,y = roidf.x.values, roidf.y.values
                points = np.vstack([x[:idx],y[:idx]]).T
                pts = points.astype(np.int32).reshape(-1,1,2) # now shape [3,1,2]
                cv2.polylines(disp, [pts], isClosed=False, color=(255, 0, 255), thickness = 3)

        self._last_disp_bgr = disp
        self._render_bgr_to_canvas(disp, resample="hq")
        if getattr(self, "preview_mode", "video") == "video":
            self._update_frame_label()

    def on_tree_double_click(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid:
            return

        values = self.tree.item(iid, "values")
        item_type = values[0] if values else ""

        if item_type == "background":
            self._ensure_background_async()
            return

        if item_type == "video":
            # If no video is loaded yet, open one. Otherwise switch to video preview.
            if self.video is None:
                self.open_asset()
                return
            self._set_preview_mode("video")
            try:
                idx = int(float(self.slider.get()))
            except Exception:
                idx = 0
            self.show_frame(idx)
            return

    def on_tree_select(self, event):
        # Placeholder: in future, selecting ROI could enable ROI editing, etc.
        pass

    def auto_rois(self):
        if not self.video:
            messagebox.showwarning("anytrack", "Open a video first.")
            return
        try:
            # build background quickly for ROI detection
            bg = build_background(
                str(self.video.video_path),
                method=self.cfg.bg_method,
                k_min=max(10, self.cfg.bg_min_frames//3),
                k_max=max(60, self.cfg.bg_min_frames),
                converge_eps=self.cfg.bg_converge_eps,
                fg_thresh=self.cfg.bg_fg_thresh,
                inpaint_ghosts=self.cfg.bg_inpaint_ghosts,
                ghost_area_min=self.cfg.bg_ghost_area_min,
                ghost_area_max=self.cfg.bg_ghost_area_max,
            ).image
            rois = detect_circular_rois(
                bg,
                dp=self.cfg.roi_hough_dp,
                min_dist_ratio=self.cfg.roi_hough_min_dist_ratio,
                param1=self.cfg.roi_hough_param1,
                param2=self.cfg.roi_hough_param2,
                min_radius_ratio=self.cfg.min_radius_ratio,
                max_radius_ratio=self.cfg.max_radius_ratio,
            )
            if not rois:
                messagebox.showinfo("anytrack", "No circles detected. Consider adjusting Hough params or manual ROIs.")
                return
            self.video.rois = rois
            self._populate_tree()
            self.show_frame(self._current_frame_idx)
        except Exception as e:
            messagebox.showerror("anytrack", f"ROI detection failed:\n{e}")

    def run_tracking(self):
        if not self.video:
            messagebox.showwarning("anytrack", "Open a video first.")
            return
        if not self.video.rois:
            messagebox.showwarning("anytrack", "Define ROIs first (auto-detect or manual).")
            return

        # Create event queue and cancel event for inter-thread communication
        q = queue.Queue(maxsize=200)
        cancel_event = threading.Event()
        hook = make_progress_hook(q)

        # Create and show modal progress dialog
        dialog = TrackingProgressDialog(self, q, cancel_event)

        def worker():
            try:
                session = TrackingSession(cfg=self.cfg, video=self.video)
                # progress_every can be tuned per config; use 1 (every frame) for now
                session.run(progress_hook=hook, cancel_event=cancel_event, progress_every=1)
            except Exception as e:
                # Ensure GUI sees the error
                try:
                    if q.full():
                        try:
                            q.get_nowait()
                        except Exception:
                            pass
                    q.put_nowait(("error", {"exc": repr(e)}))
                except Exception:
                    pass

        threading.Thread(target=worker, daemon=True).start()

        # Start polling the queue for progress events
        self._track_q = q
        self._track_dialog = dialog
        self._track_cancel_event = cancel_event
        self._track_job = self.after(50, self._track_poll)

    def _update_table(self, df):
        if df is None or df.empty:
            self.sheet.set_sheet_data([[]])
            return
        # show first N rows to stay responsive
        view = df.head(200).copy()
        headers = list(view.columns)
        data = view.values.tolist()
        self.sheet.headers(headers)
        self.sheet.set_sheet_data(data)

    def export_csv(self):
        if not self.session or self.session.dataframe is None or self.session.dataframe.empty:
            messagebox.showwarning("anytrack", "No tracking data to export yet.")
            return
        out = filedialog.asksaveasfilename(
            title="Save tracking CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not out:
            return
        self.session.dataframe.to_csv(out, index=False)
        messagebox.showinfo("anytrack", f"Saved: {out}")

    def _open_bg_debug_window(self):
        if self._bg_debug_win is not None and self._bg_debug_win.winfo_exists():
            self._bg_debug_win.lift()
            return

        win = tk.Toplevel(self)
        win.title("anytrack – Background Debug")
        win.geometry("900x520")

        self._bg_debug_win = win
        self._bg_debug_q = queue.Queue(maxsize=200)

        top = tb.Frame(win, padding=8)
        top.pack(fill="both", expand=True)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)
        top.rowconfigure(3, weight=1)

        self._bg_dbg_status = tb.Label(top, text="Waiting for background debug events…")
        self._bg_dbg_status.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        self._bg_dbg_indices = tk.Text(top, height=4, wrap="word")
        self._bg_dbg_indices.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self._bg_dbg_indices.insert("1.0", "(indices will appear here)\n")
        self._bg_dbg_indices.configure(state="disabled")

        # Slider to pick one of the sampled indices for the current trial
        slider_row = tb.Frame(top)
        slider_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        slider_row.columnconfigure(0, weight=1)

        self._bg_dbg_slider = tb.Scale(
            slider_row,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            command=self._bg_dbg_on_slider,
        )
        self._bg_dbg_slider.grid(row=0, column=0, sticky="ew")
        self._bg_dbg_slider.configure(state="disabled")

        self._bg_dbg_slider_label = tb.Label(slider_row, text="Sample: – / –")
        self._bg_dbg_slider_label.grid(row=0, column=1, sticky="e", padx=(8, 0))

        lf1 = tb.Labelframe(top, text="Sampled frame", padding=6)
        lf1.grid(row=3, column=0, sticky="nsew", padx=(0, 6))
        lf1.rowconfigure(0, weight=1)
        lf1.columnconfigure(0, weight=1)
        self._bg_dbg_frame_label = tb.Label(lf1)
        self._bg_dbg_frame_label.grid(row=0, column=0, sticky="nsew")

        lf2 = tb.Labelframe(top, text="Current model (bg)", padding=6)
        lf2.grid(row=3, column=1, sticky="nsew", padx=(6, 0))
        lf2.rowconfigure(0, weight=1)
        lf2.columnconfigure(0, weight=1)
        self._bg_dbg_model_label = tb.Label(lf2)
        self._bg_dbg_model_label.grid(row=0, column=0, sticky="nsew")

        def on_close():
            try:
                if self._bg_debug_job is not None:
                    self.after_cancel(self._bg_debug_job)
            except Exception:
                pass
            self._bg_debug_job = None
            self._bg_debug_win = None
            self._bg_debug_q = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)
        self._bg_debug_poll()

    def _bg_debug_enqueue(self, event: str, payload: dict):
        q = self._bg_debug_q
        if q is None:
            return
        try:
            if q.full():
                try:
                    q.get_nowait()
                except Exception:
                    pass
            q.put_nowait((event, payload))
        except Exception:
            return

    def _bg_debug_poll(self):
        win = self._bg_debug_win
        q = self._bg_debug_q
        if win is None or not win.winfo_exists() or q is None:
            self._bg_debug_job = None
            return

        try:
            while True:
                event, payload = q.get_nowait()

                if event == "trial":
                    k = payload.get("k")
                    idxs = payload.get("indices", [])
                    self._bg_dbg_indices_list = [int(x) for x in idxs]
                    self._bg_dbg_cache = {}

                    if self._bg_dbg_indices_list:
                        self._bg_dbg_slider.configure(from_=0, to=len(self._bg_dbg_indices_list) - 1, state="normal")
                        self._bg_dbg_slider.set(0)
                        self._bg_dbg_show_pos(0)
                    else:
                        self._bg_dbg_slider.configure(from_=0, to=0, state="disabled")
                        self._bg_dbg_slider_label.configure(text="Sample: – / –")
                    txt = f"Trial k={k}: {idxs}\n"
                    self._bg_dbg_indices.configure(state="normal")
                    self._bg_dbg_indices.insert("end", txt)
                    self._bg_dbg_indices.see("end")
                    self._bg_dbg_indices.configure(state="disabled")
                    self._bg_dbg_status.configure(text=f"Building background (k={k})…")

                elif event == "frame":
                    method = payload.get("method", "")
                    step = payload.get("step", "?")
                    steps = payload.get("steps", "?")
                    frame_idx = payload.get("frame_idx", "?")
                    self._bg_dbg_status.configure(text=f"{method}  step {step}/{steps}  frame {frame_idx}")

                    fr = payload.get("frame")
                    bg = payload.get("bg")

                    try:
                        fi = int(payload.get("frame_idx"))
                        self._bg_dbg_cache[fi] = {"frame": payload.get("frame"), "bg": payload.get("bg")}
                    except Exception:
                        fi = None
                    
                    if fi is not None and self._bg_dbg_indices_list:
                        pos = int(round(float(self._bg_dbg_slider.get())))
                        if 0 <= pos < len(self._bg_dbg_indices_list) and self._bg_dbg_indices_list[pos] == fi:
                            self._bg_dbg_show_pos(pos)

                    if fr is not None:
                        im = Image.fromarray(fr)
                        if im.mode != "RGB":
                            im = im.convert("RGB")
                        self._bg_dbg_frame_img = ImageTk.PhotoImage(im)
                        self._bg_dbg_frame_label.configure(image=self._bg_dbg_frame_img)

                    if bg is not None:
                        im2 = Image.fromarray(bg)
                        if im2.mode != "RGB":
                            im2 = im2.convert("RGB")
                        self._bg_dbg_model_img = ImageTk.PhotoImage(im2)
                        self._bg_dbg_model_label.configure(image=self._bg_dbg_model_img)

        except queue.Empty:
            pass

        self._bg_debug_job = self.after(50, self._bg_debug_poll)
    
    def _bg_dbg_on_slider(self, val):
        try:
            pos = int(round(float(val)))
        except Exception:
            pos = 0
        self._bg_dbg_show_pos(pos)

    def _bg_dbg_show_pos(self, pos: int):
        idxs = self._bg_dbg_indices_list
        if not idxs:
            self._bg_dbg_slider_label.configure(text="Sample: – / –")
            return

        pos = max(0, min(len(idxs) - 1, int(pos)))
        frame_idx = int(idxs[pos])

        # fixed-field label so it doesn’t jump
        width = max(3, len(str(len(idxs))))
        self._bg_dbg_slider_label.configure(
            text=f"Sample: {pos+1:>{width}d} / {len(idxs):>{width}d}   (frame {frame_idx})"
        )

        payload = self._bg_dbg_cache.get(frame_idx)
        if not payload:
            self._bg_dbg_status.configure(text=f"Waiting for debug data for frame {frame_idx}…")
            return

    def _track_poll(self):
        """Poll the tracking event queue and update the modal progress dialog."""
        q = getattr(self, "_track_q", None)
        dialog = getattr(self, "_track_dialog", None)
        if dialog is None or q is None:
            return
        try:
            while True:
                event, payload = q.get_nowait()

                if event == "started":
                    # optionally use total_frames from payload
                    total = payload.get("total_frames")
                    if total:
                        try:
                            # progressbar uses percent 0..100 internally
                            dialog._bar.configure(maximum=100)
                        except Exception:
                            pass
                    # initialize ETA estimator
                    try:
                        self._track_total_frames = int(total) if total is not None else None
                        # Use default alpha (0.05) for smoother estimates
                        self._eta_est = ETAEstimator()
                        self._eta_est.start()
                    except Exception:
                        self._eta_est = None

                elif event == "progress":
                    pct = payload.get("percent", 0.0)
                    frame_idx = payload.get("frame_idx")
                    t_s = payload.get("t_s")
                    try:
                        # prefer explicit frame_count from worker if provided
                        frames_done = payload.get("frame_count")
                        total = getattr(self, "_track_total_frames", None)
                        if frames_done is None and total is not None:
                            try:
                                frames_done = int(round(float(pct) * float(total)))
                            except Exception:
                                frames_done = None
                        # update ETA estimator
                        try:
                            if getattr(self, "_eta_est", None) is not None and frames_done is not None:
                                self._eta_est.update(frames_done)
                                eta_s = self._eta_est.eta_seconds(total, frames_done)
                                eta_text = self._eta_est.format_seconds(eta_s)
                                fps_val = self._eta_est.fps()
                                # Fixed-width: "---.- fps" or "123.4 fps"
                                fps_text = f"{fps_val:5.1f} fps" if fps_val is not None else " --.- fps"
                            else:
                                eta_text = "--:--"
                                fps_text = " --.- fps"
                        except Exception:
                            eta_text = "--:--"
                            fps_text = " --.- fps"
                        # Fixed-width formatting for stable display
                        # Frame field width based on total frames
                        frame_width = len(str(total)) if total else 5
                        pct_str = f"{pct*100:5.1f}%"
                        frame_str = f"{frame_idx:>{frame_width}}" if frame_idx is not None else "-" * frame_width
                        total_str = f"{total:>{frame_width}}" if total else "-" * frame_width
                        status_text = f"Frame {frame_str}/{total_str}  {pct_str}  {fps_text}  ETA {eta_text}"
                        dialog.update_progress(pct, frame=frame_idx, t_s=t_s, text=status_text)
                    except Exception:
                        pass

                elif event == "cancel_request":
                    # GUI requested cancel: ensure cancel_event is set (dialog already sets it)
                    try:
                        if getattr(self, "_track_cancel_event", None) is not None:
                            self._track_cancel_event.set()
                    except Exception:
                        pass

                elif event == "done":
                    # payload contains 'session' and 'dataframe'
                    try:
                        dialog.close()
                    except Exception:
                        pass
                    try:
                        sess = payload.get("session")
                        df = payload.get("dataframe")
                        if sess is not None:
                            self.session = sess
                        if df is not None:
                            self._update_table(df)
                        self._populate_tree()
                        self.show_frame(self._current_frame_idx)
                    except Exception as e:
                        messagebox.showerror("anytrack", f"Error applying tracking results:\n{e}")
                    return

                elif event == "error":
                    try:
                        dialog.set_error("Error during tracking")
                    except Exception:
                        pass
                    try:
                        exc = payload.get("exc")
                        messagebox.showerror("anytrack", f"Tracking failed:\n{exc}")
                    except Exception:
                        pass
                    try:
                        dialog.close()
                    except Exception:
                        pass
                    return

                elif event == "cancelled":
                    try:
                        dialog.close()
                    except Exception:
                        pass
                    messagebox.showinfo("anytrack", "Tracking cancelled.")
                    return

                elif event == "frame":
                    # Only update debug images when we explicitly receive a 'frame' event.
                    fr = payload.get("frame")
                    bg = payload.get("bg")

                    if fr is not None:
                        try:
                            im = Image.fromarray(fr).convert("RGB")
                            self._bg_dbg_frame_img = ImageTk.PhotoImage(im)
                            self._bg_dbg_frame_label.configure(image=self._bg_dbg_frame_img)
                        except Exception:
                            pass

                    if bg is not None:
                        try:
                            im2 = Image.fromarray(bg).convert("RGB")
                            self._bg_dbg_model_img = ImageTk.PhotoImage(im2)
                            self._bg_dbg_model_label.configure(image=self._bg_dbg_model_img)
                        except Exception:
                            pass

        except queue.Empty:
            pass

        # Reschedule
        try:
            self._track_job = self.after(50, self._track_poll)
        except Exception:
            pass

def run(
    fast_mode: bool = False,
    roi_downscale: int = 2,
    n_tracking_workers: Optional[int] = None,
):
    """
    Launch the anytrack GUI.

    Args:
        fast_mode: Enable fast mode (FFmpeg preprocessing + parallel tracking)
        roi_downscale: Downscale factor for ROI videos (1, 2, or 4)
        n_tracking_workers: Number of parallel tracking workers
    """
    app = AnyTrackApp()

    # Override config with CLI settings if provided
    if fast_mode:
        app.cfg.fast_mode = True
    if roi_downscale != 2:
        app.cfg.roi_downscale = roi_downscale
    if n_tracking_workers is not None:
        app.cfg.n_tracking_workers = n_tracking_workers

    print('FAST MODE:', app.cfg.fast_mode)
    app.mainloop()
