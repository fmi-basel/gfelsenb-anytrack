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
from ..background import build_background, BackgroundModel
from ..roi import detect_circular_rois, roi_mask
from ..session import TrackingSession
from ..models import VideoAsset, TrackingResult
from ..detector import debug_frame
from .progress import TrackingProgressDialog, make_progress_hook, ETAEstimator


class ZoomableImageViewer:
    """Reusable image viewer with pan and zoom capabilities."""

    def __init__(self, parent, bg="black", on_view_change=None, on_crosshair_move=None, **canvas_kwargs):
        """
        Create a zoomable/pannable image viewer.

        Args:
            parent: Parent widget
            bg: Background color for canvas
            on_view_change: Optional callback(zoom, pan_x, pan_y) called when view changes
            on_crosshair_move: Optional callback(canvas_x, canvas_y) called when mouse moves
            **canvas_kwargs: Additional arguments for canvas creation
        """
        self.canvas = tk.Canvas(parent, bg=bg, highlightthickness=0, **canvas_kwargs)

        # Pan and zoom state
        self._zoom_level = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start_x = None
        self._drag_start_y = None
        self._drag_start_pan_x = 0.0
        self._drag_start_pan_y = 0.0

        # Current image
        self._current_image = None  # np.ndarray in BGR format
        self._photo = None  # ImageTk.PhotoImage

        # Callback for view changes (for syncing viewers)
        self._on_view_change = on_view_change
        self._syncing = False  # Prevent recursive sync calls

        # Crosshair state
        self._on_crosshair_move = on_crosshair_move
        self._crosshair_x = None  # Canvas x position
        self._crosshair_y = None  # Canvas y position
        self._crosshair_lines = []  # Canvas line IDs

        # Bind mouse events
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)    # Linux scroll down
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.canvas.bind("<Motion>", self._on_mouse_motion)
        self.canvas.bind("<Leave>", self._on_mouse_leave)

        # Try to bind magnify gesture (macOS)
        try:
            self.canvas.bind("<Magnify>", self._on_magnify)
        except Exception:
            pass

    def set_image(self, image_bgr: np.ndarray):
        """Set the image to display (BGR format from OpenCV)."""
        self._current_image = image_bgr
        self.render()

    def reset_view(self):
        """Reset zoom and pan to defaults."""
        self._zoom_level = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.render()
        self._notify_view_change()

    def render(self, resample="hq"):
        """Render the current image with zoom and pan applied."""
        if self._current_image is None:
            return

        cw = max(1, int(self.canvas.winfo_width()))
        ch = max(1, int(self.canvas.winfo_height()))

        if cw <= 1 or ch <= 1:
            self.canvas.update_idletasks()
            cw = max(1, int(self.canvas.winfo_width()))
            ch = max(1, int(self.canvas.winfo_height()))

        # Convert to RGB
        rgb = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        iw, ih = im.size

        # Aspect-fit into canvas
        scale = min(cw / iw, ch / ih)
        scale *= self._zoom_level

        nw = max(1, int(iw * scale))
        nh = max(1, int(ih * scale))

        # Choose resampling mode (Pillow compatibility)
        try:
            if resample == "fast":
                filt = Image.Resampling.BILINEAR
            else:
                filt = Image.Resampling.LANCZOS
        except AttributeError:
            # Older Pillow versions
            if resample == "fast":
                filt = Image.BILINEAR  # type: ignore
            else:
                filt = Image.LANCZOS  # type: ignore

        im_resized = im.resize((nw, nh), filt)

        # Letterbox to canvas size with pan offset
        frame_box = Image.new("RGB", (cw, ch), (0, 0, 0))

        # Calculate base position (centered)
        x0 = (cw - nw) // 2
        y0 = (ch - nh) // 2

        # Apply pan offset
        x0 += int(self._pan_x)
        y0 += int(self._pan_y)

        # Crop if extending beyond canvas
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

    def set_view_from_sync(self, zoom: float, pan_x: float, pan_y: float):
        """Set zoom and pan from external sync without triggering callbacks."""
        self._syncing = True
        self._zoom_level = zoom
        self._pan_x = pan_x
        self._pan_y = pan_y
        self.render(resample="hq")
        self._syncing = False

    def _notify_view_change(self):
        """Notify callback of view change if not syncing."""
        if not self._syncing and self._on_view_change is not None:
            self._on_view_change(self._zoom_level, self._pan_x, self._pan_y)

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel zoom."""
        if self._current_image is None:
            return

        # Determine zoom direction
        if event.num == 4 or event.delta > 0:
            zoom_factor = 1.1
        elif event.num == 5 or event.delta < 0:
            zoom_factor = 0.9
        else:
            return

        # Update zoom level
        new_zoom = self._zoom_level * zoom_factor
        new_zoom = max(0.1, min(10.0, new_zoom))

        # Zoom toward mouse position
        canvas_x = event.x
        canvas_y = event.y
        zoom_ratio = new_zoom / self._zoom_level

        self._pan_x = canvas_x - (canvas_x - self._pan_x) * zoom_ratio
        self._pan_y = canvas_y - (canvas_y - self._pan_y) * zoom_ratio
        self._zoom_level = new_zoom

        self.render(resample="hq")
        self._notify_view_change()

    def _on_magnify(self, event):
        """Handle trackpad pinch gesture."""
        if self._current_image is None:
            return

        zoom_factor = 1.0 + (event.delta * 2.0)
        new_zoom = self._zoom_level * zoom_factor
        new_zoom = max(0.1, min(10.0, new_zoom))

        try:
            canvas_x = event.x
            canvas_y = event.y
        except AttributeError:
            canvas_x = self.canvas.winfo_width() // 2
            canvas_y = self.canvas.winfo_height() // 2

        zoom_ratio = new_zoom / self._zoom_level
        self._pan_x = canvas_x - (canvas_x - self._pan_x) * zoom_ratio
        self._pan_y = canvas_y - (canvas_y - self._pan_y) * zoom_ratio
        self._zoom_level = new_zoom

        self.render(resample="fast")
        self._notify_view_change()

    def _on_canvas_press(self, event):
        """Handle mouse button press for panning."""
        self._drag_start_x = event.x
        self._drag_start_y = event.y
        self._drag_start_pan_x = self._pan_x
        self._drag_start_pan_y = self._pan_y
        self.canvas.configure(cursor="fleur")

    def _on_canvas_drag(self, event):
        """Handle mouse drag for panning."""
        if self._drag_start_x is None or self._current_image is None:
            return

        dx = event.x - self._drag_start_x
        dy = event.y - self._drag_start_y

        self._pan_x = self._drag_start_pan_x + dx
        self._pan_y = self._drag_start_pan_y + dy

        self.render(resample="fast")

    def _on_canvas_release(self, event):
        """Handle mouse button release."""
        self._drag_start_x = None
        self._drag_start_y = None
        self.canvas.configure(cursor="")

        if self._current_image is not None:
            self.render(resample="hq")
            self._notify_view_change()

    def _on_canvas_double_click(self, event):
        """Reset zoom and pan on double-click."""
        self.reset_view()

    def _on_mouse_motion(self, event):
        """Handle mouse motion for crosshair display."""
        self._crosshair_x = event.x
        self._crosshair_y = event.y
        self._draw_crosshair()

        # Notify callback
        if self._on_crosshair_move is not None:
            self._on_crosshair_move(event.x, event.y)

    def _on_mouse_leave(self, event):
        """Clear crosshair when mouse leaves canvas."""
        self._crosshair_x = None
        self._crosshair_y = None
        self._clear_crosshair()

        # Notify callback with None to clear crosshairs in other viewers
        if self._on_crosshair_move is not None:
            self._on_crosshair_move(None, None)

    def set_crosshair(self, x, y):
        """Set crosshair position from external source (None to clear)."""
        if x is None or y is None:
            self.clear_crosshair()
        else:
            self._crosshair_x = x
            self._crosshair_y = y
            self._draw_crosshair()

    def clear_crosshair(self):
        """Clear the crosshair."""
        self._crosshair_x = None
        self._crosshair_y = None
        self._clear_crosshair()

    def _draw_crosshair(self):
        """Draw crosshair at current position."""
        self._clear_crosshair()

        if self._crosshair_x is None or self._crosshair_y is None:
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        if cw <= 1 or ch <= 1:
            return

        x, y = self._crosshair_x, self._crosshair_y

        # Draw horizontal line
        h_line = self.canvas.create_line(
            0, y, cw, y,
            fill="#FFFFFF", width=1, dash=(4, 4), tags="crosshair"
        )
        # Draw vertical line
        v_line = self.canvas.create_line(
            x, 0, x, ch,
            fill="#FFFFFF", width=1, dash=(4, 4), tags="crosshair"
        )

        self._crosshair_lines = [h_line, v_line]

    def _clear_crosshair(self):
        """Clear existing crosshair lines."""
        for line_id in self._crosshair_lines:
            try:
                self.canvas.delete(line_id)
            except Exception:
                pass
        self._crosshair_lines = []


class AnyTrackApp(tb.Window):
    # ROI color palette (up to 12 differentiable colors)
    ROI_COLORS = [
        "#FF8C00",  # orange (darkorange)
        "#1E90FF",  # dodgerblue
        "#32CD32",  # limegreen
        "#FF1493",  # deeppink
        "#FFD700",  # gold
        "#9370DB",  # mediumpurple
        "#00CED1",  # darkturquoise
        "#FF6347",  # tomato
        "#7FFF00",  # chartreuse
        "#FF69B4",  # hotpink
        "#00BFFF",  # deepskyblue
        "#FFA500",  # orange
    ]

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

        self._background_gray: Optional[np.ndarray] = None  # cached background (grayscale) - points to model.gmm
        self._background_model: Optional[BackgroundModel] = None  # Full background model (average, gmm, thresholds, arenas)
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

        # Tracking debug window
        self.debug_tracking_var = tk.BooleanVar(value=False)
        self._tracking_debug_win = None
        self._tracking_debug_widgets = {}  # roi_name -> dict of widgets
        self._open_tracking_debug_after_bg = False  # flag to open debug window after BG builds
        self._syncing_sliders = False  # flag to prevent circular slider updates

        # ROI colors (shared between main view and debug window)
        self._roi_colors = {}  # roi_name -> hex color

        self._build_ui()

        # Register close handler
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # Load previous session
        self.after(100, self._load_session)

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

    def _assign_roi_colors(self):
        """Assign colors to ROIs from the color palette."""
        self._roi_colors = {}
        if self.video and self.video.rois:
            for i, roi in enumerate(self.video.rois):
                self._roi_colors[roi.name] = self.ROI_COLORS[i % len(self.ROI_COLORS)]

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
        filemenu_debug.add_checkbutton(label="Tracking debug window", variable=self.debug_tracking_var, command=self._toggle_tracking_debug)
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

        # Always open debug window for background building
        dbg_enabled = True
        self._open_bg_debug_window()

        def dbg_hook(event: str, payload: dict):
            self._bg_debug_enqueue(event, payload)

        def worker():
            try:
                bg_model = build_background(
                    str(self.video.video_path),
                    n_samples=self.cfg.gmm_n_samples,
                    bic_improvement=self.cfg.gmm_bic_improvement,
                    min_std=self.cfg.gmm_min_std,
                    reg_covar=self.cfg.gmm_reg_covar,
                    lowp=self.cfg.gmm_lowp,
                    arena_detection=self.cfg.arena_detection_enabled,
                    arena_min_area_frac=self.cfg.arena_min_area_frac,
                    arena_blur_sigma=self.cfg.arena_blur_sigma,
                    debug=dbg_enabled,
                    debug_hook=dbg_hook,
                )
                self._background_model = bg_model
                self._background_gray = bg_model.gmm
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

        # Open tracking debug window if it was requested
        if self._open_tracking_debug_after_bg:
            self._open_tracking_debug_after_bg = False
            # Switch back to video mode for tracking debug
            self._set_preview_mode("video")
            self.show_frame(self._current_frame_idx)
            # Now open the tracking debug window
            self._open_tracking_debug_window()

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
        self._assign_roi_colors()  # Assign colors to ROIs
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

        # Draw ROI circles with their assigned colors
        if self.video:
            for roi in self.video.rois:
                roi_color_hex = self._roi_colors.get(roi.name, "#00FF00")
                # Convert hex to BGR
                roi_color_rgb = tuple(int(roi_color_hex[i:i+2], 16) for i in (1, 3, 5))
                roi_color_bgr = (roi_color_rgb[2], roi_color_rgb[1], roi_color_rgb[0])
                cv2.circle(disp, (int(roi.cx), int(roi.cy)), int(roi.r), roi_color_bgr, 2)

        # Draw trajectories with alpha blending
        if self.session and self.session.dataframe is not None and not self.session.dataframe.empty:
            df = self.session.dataframe
            fdf = df[df["frame"] == idx]

            # Create overlay for alpha-blended trajectories
            trajectory_alpha = getattr(self.cfg, 'trajectory_alpha', 0.5)
            overlay = disp.copy()

            for each_roi in df.roi.unique():
                roidf = df[df.roi == each_roi]
                x, y = roidf.x.values, roidf.y.values
                points = np.vstack([x[:idx], y[:idx]]).T
                pts = points.astype(np.int32).reshape(-1, 1, 2)

                # Get ROI color
                roi_color_hex = self._roi_colors.get(each_roi, "#FF00FF")
                roi_color_rgb = tuple(int(roi_color_hex[i:i+2], 16) for i in (1, 3, 5))
                roi_color_bgr = (roi_color_rgb[2], roi_color_rgb[1], roi_color_rgb[0])

                # Draw trajectory on overlay
                cv2.polylines(overlay, [pts], isClosed=False, color=roi_color_bgr, thickness=3)

            # Blend overlay with original using alpha
            cv2.addWeighted(overlay, trajectory_alpha, disp, 1 - trajectory_alpha, 0, disp)

            # Draw current positions (not alpha-blended)
            for _, r in fdf.iterrows():
                x, y = int(r["x"]), int(r["y"])
                roi_name = r["roi"]
                roi_color_hex = self._roi_colors.get(roi_name, "#FFFF00")
                roi_color_rgb = tuple(int(roi_color_hex[i:i+2], 16) for i in (1, 3, 5))
                roi_color_bgr = (roi_color_rgb[2], roi_color_rgb[1], roi_color_rgb[0])

                cv2.circle(disp, (x, y), 8, roi_color_bgr, 2)
                # angle arrow
                ang = np.deg2rad(float(r["angle_deg"]))
                x2 = int(x + 20 * np.cos(ang))
                y2 = int(y + 20 * np.sin(ang))
                cv2.line(disp, (x, y), (x2, y2), roi_color_bgr, 2)

        # Pose skeleton overlay (Milestone B5), when a pose table is present.
        self._draw_pose_overlay(disp, idx)

        self._last_disp_bgr = disp
        self._render_bgr_to_canvas(disp, resample="hq")
        if getattr(self, "preview_mode", "video") == "video":
            self._update_frame_label()

        # Update tracking debug window if open
        if self._tracking_debug_widgets:
            self._update_tracking_debug(frame, idx)

    def _draw_pose_overlay(self, disp, idx: int):
        """Draw predicted keypoints + skeleton for frame ``idx`` (full-res preview)."""
        sess = self.session
        pdf = getattr(sess, "pose_dataframe", None) if sess is not None else None
        if pdf is None or pdf.empty:
            return
        # Cache skeleton/colors, keyed by the pose df identity (auto-invalidates
        # when a new run replaces it).
        if getattr(self, "_pose_prep_id", None) != id(pdf):
            try:
                from ..pose.skeleton import get_skeleton
                from ..pose.labeling import resolve_node_colors
                from ..pose.qc_pose import _hex_to_bgr
                sk = get_skeleton(self.cfg)
                nodes = list(sk.nodes)
                colors = resolve_node_colors(nodes, getattr(self.cfg, "pose_node_colors", ""))
                self._pose_prep = (nodes, sk.edge_indices(),
                                   {n: _hex_to_bgr(colors[n]) for n in nodes},
                                   float(getattr(self.cfg, "pose_conf_min", 0.2)))
                self._pose_prep_id = id(pdf)
            except Exception:
                self._pose_prep = None
                self._pose_prep_id = id(pdf)
        prep = getattr(self, "_pose_prep", None)
        if not prep:
            return
        nodes, edges, node_bgr, cmin = prep
        fdf = pdf[pdf["frame"] == idx]
        if fdf.empty:
            return
        for (_roi, _tid), g in fdf.groupby(["roi", "track_id"]):
            inst = {r.keypoint: (r.x_full, r.y_full, r.score) for r in g.itertuples(index=False)}
            for ai, bi in edges:
                pa, pb = inst.get(nodes[ai]), inst.get(nodes[bi])
                if (pa and pb and pa[2] >= cmin and pb[2] >= cmin
                        and np.isfinite(pa[0]) and np.isfinite(pb[0])):
                    cv2.line(disp, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                             (210, 210, 210), 1, cv2.LINE_AA)
            for name, (x, y, s) in inst.items():
                if s >= cmin and np.isfinite(x) and np.isfinite(y):
                    cv2.circle(disp, (int(x), int(y)), 4, node_bgr.get(name, (255, 255, 255)),
                               -1, cv2.LINE_AA)

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
        """Auto-detect ROIs with arena detection and progress dialog."""
        if not self.video:
            messagebox.showwarning("anytrack", "Open a video first.")
            return

        # Create event queue and cancel event for inter-thread communication
        q = queue.Queue(maxsize=200)
        cancel_event = threading.Event()
        hook = make_progress_hook(q)

        # Create and show modal progress dialog
        dialog = TrackingProgressDialog(self, q, cancel_event, title="Detecting ROIs")

        def worker():
            try:
                # Build background with arena detection
                bg_model = build_background(
                    str(self.video.video_path),
                    n_samples=self.cfg.gmm_n_samples,
                    bic_improvement=self.cfg.gmm_bic_improvement,
                    min_std=self.cfg.gmm_min_std,
                    reg_covar=self.cfg.gmm_reg_covar,
                    lowp=self.cfg.gmm_lowp,
                    arena_detection=True,  # Force arena detection for auto ROI
                    arena_min_area_frac=self.cfg.arena_min_area_frac,
                    arena_blur_sigma=self.cfg.arena_blur_sigma,
                    debug=True,  # Enable debug events
                    debug_hook=hook,
                )

                # Send completion event with background model
                hook("done", {"bg_model": bg_model})

            except Exception as e:
                # Send error event
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
        self._auto_rois_q = q
        self._auto_rois_dialog = dialog
        self._auto_rois_cancel_event = cancel_event
        self._auto_rois_job = self.after(50, self._auto_rois_poll)

    def _auto_rois_poll(self):
        """Poll the ROI detection event queue and update the modal progress dialog."""
        q = getattr(self, "_auto_rois_q", None)
        dialog = getattr(self, "_auto_rois_dialog", None)
        if dialog is None or q is None:
            return

        try:
            while True:
                event, payload = q.get_nowait()

                if event == "sampling_progress":
                    # Sampling frames: 0-20% of total progress
                    step = payload.get("step", 0)
                    total = payload.get("total", 1)
                    pct = (step / total) * 0.20  # 0-20% range

                    # Initialize ETA estimator on first event
                    if not hasattr(self, "_auto_rois_eta"):
                        self._auto_rois_eta = ETAEstimator()
                        self._auto_rois_eta.start()
                        # Calculate total work based on video dimensions if available
                        if self.video and self.video.width and self.video.height:
                            total_pixels = self.video.width * self.video.height
                        else:
                            # Conservative estimate for 1920x1080
                            total_pixels = 1920 * 1080
                        # Total work = sampling steps + GMM pixels (GMM dominates)
                        self._auto_rois_total_work = total + total_pixels
                        self._auto_rois_n_samples = total

                    self._auto_rois_eta.update(step)
                    eta_s = self._auto_rois_eta.eta_seconds(self._auto_rois_total_work, step)
                    eta_text = self._auto_rois_eta.format_seconds(eta_s)

                    status_text = f"Sampling frames {step}/{total}  {pct*100:.1f}%  ETA {eta_text}"
                    dialog.update_progress(pct, text=status_text)

                elif event == "average_complete":
                    # Average complete: 20% progress
                    pct = 0.20
                    status_text = f"Computing GMM background...  {pct*100:.1f}%"
                    dialog.update_progress(pct, text=status_text)

                elif event == "gmm_progress":
                    # GMM fitting: 20-90% of total progress (main workload)
                    percent = payload.get("percent", 0.0)  # 0-100
                    pct = 0.20 + (percent / 100.0) * 0.70  # Map to 20-90% range
                    pixel = payload.get("pixel", 0)
                    total = payload.get("total", 1)

                    # Update ETA with pixel progress
                    if hasattr(self, "_auto_rois_eta") and hasattr(self, "_auto_rois_n_samples"):
                        # Add sampling offset to pixel count
                        work_done = self._auto_rois_n_samples + pixel
                        self._auto_rois_eta.update(work_done)
                        eta_s = self._auto_rois_eta.eta_seconds(self._auto_rois_total_work, work_done)
                        eta_text = self._auto_rois_eta.format_seconds(eta_s)
                    else:
                        eta_text = "--:--"

                    status_text = f"Fitting GMM models {pixel}/{total}  {pct*100:.1f}%  ETA {eta_text}"
                    dialog.update_progress(pct, text=status_text)

                elif event == "gmm_complete":
                    # GMM complete: 90% progress
                    pct = 0.90
                    status_text = f"Detecting arenas...  {pct*100:.1f}%"
                    dialog.update_progress(pct, text=status_text)

                elif event == "arenas_detected":
                    # Arenas detected: 95% progress
                    pct = 0.95
                    circles = payload.get("circles", [])
                    status_text = f"Found {len(circles)} arenas, applying blur...  {pct*100:.1f}%"
                    dialog.update_progress(pct, text=status_text)

                elif event == "blur_complete":
                    # Blur complete: 100% progress
                    pct = 1.0
                    status_text = f"Complete!  {pct*100:.1f}%"
                    dialog.update_progress(pct, text=status_text)

                elif event == "done":
                    # Background building complete, process results
                    try:
                        dialog.close()
                    except Exception:
                        pass

                    try:
                        bg_model = payload.get("bg_model")
                        if bg_model is None:
                            raise ValueError("No background model returned")

                        # Use detected arenas as ROIs if available
                        if bg_model.arena_circles and len(bg_model.arena_circles) > 0:
                            from ..models import CircleROI
                            self.video.rois = [
                                CircleROI(
                                    name=f"arena_{i+1:02d}",
                                    cx=cx, cy=cy, r=r
                                )
                                for i, (cx, cy, r) in enumerate(bg_model.arena_circles)
                            ]
                            self._background_model = bg_model
                            self._background_gray = bg_model.gmm
                            self._assign_roi_colors()
                            self._populate_tree()
                            self.show_background()
                            return

                        # Fallback to HoughCircles if arena detection didn't find enough arenas
                        bg = bg_model.gmm
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
                        self._background_model = bg_model
                        self._background_gray = bg_model.gmm
                        self._assign_roi_colors()
                        self._populate_tree()
                        self.show_frame(self._current_frame_idx)

                    except Exception as e:
                        messagebox.showerror("anytrack", f"Error processing ROI results:\n{e}")
                    finally:
                        # Clean up ETA estimator
                        if hasattr(self, "_auto_rois_eta"):
                            delattr(self, "_auto_rois_eta")
                        if hasattr(self, "_auto_rois_total_work"):
                            delattr(self, "_auto_rois_total_work")
                        if hasattr(self, "_auto_rois_n_samples"):
                            delattr(self, "_auto_rois_n_samples")
                    return

                elif event == "error":
                    try:
                        dialog.set_error("Error during ROI detection")
                    except Exception:
                        pass
                    try:
                        exc = payload.get("exc")
                        messagebox.showerror("anytrack", f"ROI detection failed:\n{exc}")
                    except Exception:
                        pass
                    try:
                        dialog.close()
                    except Exception:
                        pass
                    finally:
                        # Clean up ETA estimator
                        if hasattr(self, "_auto_rois_eta"):
                            delattr(self, "_auto_rois_eta")
                        if hasattr(self, "_auto_rois_total_work"):
                            delattr(self, "_auto_rois_total_work")
                        if hasattr(self, "_auto_rois_n_samples"):
                            delattr(self, "_auto_rois_n_samples")
                    return

                elif event == "cancel_request":
                    # User requested cancellation
                    try:
                        if getattr(self, "_auto_rois_cancel_event", None) is not None:
                            self._auto_rois_cancel_event.set()
                    except Exception:
                        pass

        except queue.Empty:
            pass

        # Reschedule
        try:
            self._auto_rois_job = self.after(50, self._auto_rois_poll)
        except Exception:
            pass

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
        """Open simplified debug window showing average and GMM backgrounds side-by-side."""
        if self._bg_debug_win is not None and self._bg_debug_win.winfo_exists():
            self._bg_debug_win.lift()
            return

        try:
            win = tk.Toplevel(self)
            win.title("anytrack – Background Build Progress")
            win.geometry("1400x700")

            self._bg_debug_win = win
            self._bg_debug_q = queue.Queue(maxsize=200)

            # State for storing background images
            self._bg_average_image = None
            self._bg_gmm_image = None

            # Main container
            main_frame = tb.Frame(win, padding=12)
            main_frame.pack(fill="both", expand=True)

            # Configure grid weights for side-by-side layout
            main_frame.rowconfigure(0, weight=0)  # Labels row
            main_frame.rowconfigure(1, weight=1)  # Images row
            main_frame.rowconfigure(2, weight=0)  # Progress row
            main_frame.columnconfigure(0, weight=1)  # Left panel (Average)
            main_frame.columnconfigure(1, weight=1)  # Right panel (GMM)

            # Column labels
            avg_label = tb.Label(main_frame, text="Average Background", font=("TkDefaultFont", 12, "bold"))
            avg_label.grid(row=0, column=0, pady=(0, 8))

            gmm_label = tb.Label(main_frame, text="GMM Background", font=("TkDefaultFont", 12, "bold"))
            gmm_label.grid(row=0, column=1, pady=(0, 8))

            # Left panel: Average background viewer
            avg_frame = tb.Frame(main_frame, relief="solid", borderwidth=1)
            avg_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
            avg_frame.rowconfigure(0, weight=1)
            avg_frame.columnconfigure(0, weight=1)

            self._bg_avg_viewer = ZoomableImageViewer(avg_frame, bg="#2a2a2a")
            self._bg_avg_viewer.canvas.grid(row=0, column=0, sticky="nsew")

            # Right panel: GMM background viewer
            gmm_frame = tb.Frame(main_frame, relief="solid", borderwidth=1)
            gmm_frame.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
            gmm_frame.rowconfigure(0, weight=1)
            gmm_frame.columnconfigure(0, weight=1)

            self._bg_gmm_viewer = ZoomableImageViewer(gmm_frame, bg="#2a2a2a")
            self._bg_gmm_viewer.canvas.grid(row=0, column=0, sticky="nsew")

            # Progress section (spans both columns)
            progress_frame = tb.Frame(main_frame)
            progress_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
            progress_frame.columnconfigure(0, weight=1)

            self._bg_status_label = tb.Label(
                progress_frame,
                text="Starting background build...",
                font=("TkFixedFont", 10)
            )
            self._bg_status_label.grid(row=0, column=0, sticky="w", pady=(0, 4))

            self._bg_progress_bar = ttk.Progressbar(
                progress_frame,
                orient="horizontal",
                length=400,
                mode="determinate"
            )
            self._bg_progress_bar.grid(row=1, column=0, sticky="ew")
            self._bg_progress_bar['value'] = 0

            def on_close():
                try:
                    if self._bg_debug_job is not None:
                        self.after_cancel(self._bg_debug_job)
                except Exception:
                    pass
                self._bg_debug_job = None
                self._bg_debug_win = None
                self._bg_debug_q = None
                self._bg_average_image = None
                self._bg_gmm_image = None
                win.destroy()

            win.protocol("WM_DELETE_WINDOW", on_close)
            win.lift()
            win.focus_force()
            self._bg_debug_poll()
        except Exception as e:
            print(f"ERROR: Failed to create background debug window: {e}")
            import traceback
            traceback.print_exc()
            self._bg_debug_win = None
            self._bg_debug_q = None

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
        """Poll the background debug event queue and update the side-by-side viewers."""
        win = self._bg_debug_win
        q = self._bg_debug_q
        if win is None or not win.winfo_exists() or q is None:
            self._bg_debug_job = None
            return

        try:
            while True:
                event, payload = q.get_nowait()

                if event == "sampling_progress":
                    # Sampling frames: update progress
                    step = payload.get("step", 0)
                    total = payload.get("total", 1)
                    pct = (step / total) * 20  # 0-20% range
                    status_text = f"Sampling frames {step}/{total}  ({pct:.1f}%)"
                    self._bg_status_label.configure(text=status_text)
                    self._bg_progress_bar['value'] = pct

                elif event == "average_complete":
                    # Average background complete: show in left panel
                    avg = payload.get("average")
                    if avg is not None:
                        self._bg_average_image = avg
                        # Convert grayscale to BGR for display
                        import cv2
                        avg_bgr = cv2.cvtColor(avg, cv2.COLOR_GRAY2BGR)
                        self._bg_avg_viewer.set_image(avg_bgr)
                    status_text = "Computing GMM background...  (20%)"
                    self._bg_status_label.configure(text=status_text)
                    self._bg_progress_bar['value'] = 20

                elif event == "gmm_progress":
                    # GMM fitting: update progress
                    percent = payload.get("percent", 0.0)  # 0-100
                    pct = 20 + (percent / 100.0) * 70  # Map to 20-90% range
                    pixel = payload.get("pixel", 0)
                    total = payload.get("total", 1)
                    status_text = f"Fitting GMM models {pixel}/{total}  ({pct:.1f}%)"
                    self._bg_status_label.configure(text=status_text)
                    self._bg_progress_bar['value'] = pct

                elif event == "gmm_complete":
                    # GMM complete: show in right panel
                    gmm = payload.get("gmm")
                    if gmm is not None:
                        self._bg_gmm_image = gmm
                        # Convert grayscale to BGR for display
                        import cv2
                        gmm_bgr = cv2.cvtColor(gmm, cv2.COLOR_GRAY2BGR)
                        self._bg_gmm_viewer.set_image(gmm_bgr)
                    status_text = "Detecting arenas...  (90%)"
                    self._bg_status_label.configure(text=status_text)
                    self._bg_progress_bar['value'] = 90

                elif event == "arenas_detected":
                    # Arenas detected
                    circles = payload.get("circles", [])
                    status_text = f"Found {len(circles)} arenas, applying blur...  (95%)"
                    self._bg_status_label.configure(text=status_text)
                    self._bg_progress_bar['value'] = 95

                elif event == "blur_complete":
                    # Blur complete: update right panel
                    blurred = payload.get("blurred")
                    if blurred is not None:
                        self._bg_gmm_image = blurred
                        # Convert grayscale to BGR for display
                        import cv2
                        blurred_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
                        self._bg_gmm_viewer.set_image(blurred_bgr)
                    status_text = "Complete!  (100%)"
                    self._bg_status_label.configure(text=status_text)
                    self._bg_progress_bar['value'] = 100

        except queue.Empty:
            pass

        self._bg_debug_job = self.after(50, self._bg_debug_poll)

    def _toggle_tracking_debug(self):
        """Toggle tracking debug window on/off."""
        if self.debug_tracking_var.get():
            self._open_tracking_debug_window()
        else:
            self._close_tracking_debug_window()

    def _open_tracking_debug_window(self):
        """Open the tracking debug window showing ROI crops, diffs, masks, and metrics."""
        if self._tracking_debug_win is not None and self._tracking_debug_win.winfo_exists():
            self._tracking_debug_win.lift()
            return

        if not self.video or not self.video.rois:
            messagebox.showinfo("anytrack", "Load a video with ROIs first to enable tracking debug.")
            self.debug_tracking_var.set(False)
            return

        # Build background if not already available (needed for tracking debug)
        if self._background_gray is None:
            messagebox.showinfo(
                "anytrack",
                "Building background model for tracking debug.\nThis may take a moment..."
            )
            self._open_tracking_debug_after_bg = True
            self._ensure_background_async()
            # The window will be created after background is built
            return

        win = tk.Toplevel(self)
        win.title("anytrack – Tracking Debug")
        win.geometry("1400x900")
        self._tracking_debug_win = win

        # Main container
        main_container = tb.Frame(win)
        main_container.pack(fill="both", expand=True, padx=6, pady=6)
        main_container.rowconfigure(0, weight=0)  # Frame slider
        main_container.rowconfigure(1, weight=0)  # Background viewer
        main_container.rowconfigure(2, weight=1)  # ROI scroll container
        main_container.columnconfigure(0, weight=1)

        # Frame slider at the top
        slider_frame = tb.Frame(main_container)
        slider_frame.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        slider_frame.columnconfigure(0, weight=1)

        tb.Label(slider_frame, text="Frame:", font=("TkDefaultFont", 10)).grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )

        self._debug_slider = tb.Scale(
            slider_frame,
            from_=0,
            to=int(self.slider.cget("to")) if self.video else 0,
            orient=HORIZONTAL,
            command=self._on_debug_slider
        )
        self._debug_slider.grid(row=0, column=1, sticky="ew")
        self._debug_slider.set(self._current_frame_idx)

        self._debug_frame_label = tb.Label(
            slider_frame,
            text=f"Frame: {self._current_frame_idx}",
            font=("TkFixedFont", 10),
            width=20
        )
        self._debug_frame_label.grid(row=0, column=2, sticky="e", padx=(8, 0))

        # Background viewer section
        bg_viewer_frame = tb.Labelframe(main_container, text="Background Model", padding=6)
        bg_viewer_frame.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        bg_viewer_frame.rowconfigure(0, weight=0, minsize=200)
        bg_viewer_frame.columnconfigure(0, weight=1)

        self._debug_bg_viewer = ZoomableImageViewer(bg_viewer_frame, bg="#2a2a2a")
        self._debug_bg_viewer.canvas.grid(row=0, column=0, sticky="nsew")

        # Will set background image with ROIs later (after ROI colors are assigned)
        self._needs_bg_update = True

        # Create a scrollable frame for multiple ROIs
        scroll_container = tb.Frame(main_container)
        scroll_container.grid(row=2, column=0, sticky="nsew")
        scroll_container.rowconfigure(0, weight=1)
        scroll_container.columnconfigure(0, weight=1)

        canvas = tk.Canvas(scroll_container, bg="#f0f0f0")
        scrollbar = tb.Scrollbar(scroll_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tb.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create debug panels for each ROI with color-coded labelframes
        self._tracking_debug_widgets = {}
        # ROI colors are already assigned in main app, just use them
        all_roi_viewers = []  # Collect all viewers for background crosshair sync
        for i, roi in enumerate(self.video.rois):
            # Use existing ROI color (already assigned in main app)
            roi_color = self._roi_colors.get(roi.name, self.ROI_COLORS[i % len(self.ROI_COLORS)])

            roi_frame = tb.Labelframe(
                scrollable_frame,
                text=f"ROI: {roi.name}",
                padding=10,
                bootstyle="primary"
            )
            roi_frame.grid(row=i, column=0, sticky="ew", padx=10, pady=10)

            # Apply color styling to the labelframe
            try:
                roi_frame.configure(borderwidth=3, relief="solid")
                # Create a custom style for this ROI
                style_name = f"ROI{i}.TLabelframe"
                self.style.configure(style_name, bordercolor=roi_color, borderwidth=3)
                self.style.configure(f"{style_name}.Label", foreground=roi_color, font=("TkDefaultFont", 11, "bold"))
                roi_frame.configure(style=style_name)
            except Exception:
                pass  # Fallback if styling fails

            # Create grid for images: ROI Crop | Diff | Binary Mask
            img_frame = tb.Frame(roi_frame, height=250)
            img_frame.grid(row=0, column=0, sticky="ew")
            img_frame.columnconfigure(0, weight=1)
            img_frame.columnconfigure(1, weight=1)
            img_frame.columnconfigure(2, weight=1)
            img_frame.rowconfigure(0, weight=1, minsize=250)

            # Create callbacks for syncing viewers
            def make_sync_callback(viewers_list):
                def on_view_change(zoom, pan_x, pan_y):
                    # Sync all viewers in this ROI
                    for v in viewers_list:
                        v.set_view_from_sync(zoom, pan_x, pan_y)
                return on_view_change

            def make_crosshair_callback(viewers_list, source_viewer, bg_viewer=None, roi_obj=None):
                def on_crosshair_move(canvas_x, canvas_y):
                    # Sync crosshair across all viewers in this ROI (same canvas space)
                    for v in viewers_list:
                        if v is not source_viewer:
                            v.set_crosshair(canvas_x, canvas_y)

                    # Sync to background viewer using video pixel coordinates
                    if bg_viewer is not None and bg_viewer is not source_viewer and roi_obj is not None:
                        if canvas_x is not None and canvas_y is not None:
                            # Convert from ROI-local canvas position to video pixel position
                            # ROI offset in video coordinates
                            x0 = int(max(0, roi_obj.cx - roi_obj.r))
                            y0 = int(max(0, roi_obj.cy - roi_obj.r))

                            # TODO: This is a simplified mapping assuming 1:1 canvas to ROI pixels
                            # In reality, we'd need to account for zoom/pan in the viewer
                            # For now, just pass the canvas coordinates to background viewer
                            bg_viewer.set_crosshair(canvas_x, canvas_y)
                        else:
                            bg_viewer.set_crosshair(None, None)
                return on_crosshair_move

            # Temporary list to hold viewers for sync callback
            roi_viewers = []

            # ROI Crop
            crop_lf = tb.Labelframe(img_frame, text="ROI Crop", padding=4)
            crop_lf.grid(row=0, column=0, padx=4, pady=4, sticky="nsew")
            crop_lf.rowconfigure(0, weight=1)
            crop_lf.columnconfigure(0, weight=1)
            crop_viewer = ZoomableImageViewer(crop_lf, bg="#2a2a2a")
            crop_viewer.canvas.grid(row=0, column=0, sticky="nsew")
            roi_viewers.append(crop_viewer)

            # Background-subtracted diff
            diff_lf = tb.Labelframe(img_frame, text="BG Subtracted", padding=4)
            diff_lf.grid(row=0, column=1, padx=4, pady=4, sticky="nsew")
            diff_lf.rowconfigure(0, weight=1)
            diff_lf.columnconfigure(0, weight=1)
            diff_viewer = ZoomableImageViewer(diff_lf, bg="#2a2a2a")
            diff_viewer.canvas.grid(row=0, column=0, sticky="nsew")
            roi_viewers.append(diff_viewer)

            # Binary mask
            mask_lf = tb.Labelframe(img_frame, text="Binary Mask", padding=4)
            mask_lf.grid(row=0, column=2, padx=4, pady=4, sticky="nsew")
            mask_lf.rowconfigure(0, weight=1)
            mask_lf.columnconfigure(0, weight=1)
            mask_viewer = ZoomableImageViewer(mask_lf, bg="#2a2a2a")
            mask_viewer.canvas.grid(row=0, column=0, sticky="nsew")
            roi_viewers.append(mask_viewer)

            # Set up sync callbacks for all viewers in this ROI
            sync_callback = make_sync_callback(roi_viewers)
            for v in roi_viewers:
                v._on_view_change = sync_callback
                # Set up crosshair callback for each viewer (include background viewer)
                v._on_crosshair_move = make_crosshair_callback(roi_viewers, v, self._debug_bg_viewer, roi)

            # Collect all viewers for background viewer crosshair sync
            all_roi_viewers.extend(roi_viewers)

            # Metrics display
            metrics_frame = tb.Frame(roi_frame)
            metrics_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))

            metrics_text = tk.Text(metrics_frame, height=6, width=80, font=("TkFixedFont", 10))
            metrics_text.pack(fill="both", expand=True)
            metrics_text.insert("1.0", "No tracking data yet...\n")
            metrics_text.configure(state="disabled")

            self._tracking_debug_widgets[roi.name] = {
                "crop_viewer": crop_viewer,
                "diff_viewer": diff_viewer,
                "mask_viewer": mask_viewer,
                "metrics_text": metrics_text,
            }

        # Set up background viewer's crosshair callback to sync with all ROI viewers
        if all_roi_viewers:
            def bg_crosshair_callback(x, y):
                # Sync crosshair to all ROI viewers
                for v in all_roi_viewers:
                    v.set_crosshair(x, y)
            self._debug_bg_viewer._on_crosshair_move = bg_crosshair_callback

        def on_close():
            self.debug_tracking_var.set(False)
            self._tracking_debug_win = None
            self._tracking_debug_widgets = {}
            self._debug_slider = None
            self._debug_frame_label = None
            self._debug_bg_viewer = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

        # Update background viewer with ROI circles now that colors are assigned
        if self._background_gray is not None and self._needs_bg_update:
            self._update_debug_background()
            self._needs_bg_update = False

        # Trigger initial update if video is loaded
        if self.video:
            self.show_frame(self._current_frame_idx)

    def _update_debug_background(self):
        """Update background viewer with ROI circles drawn in their assigned colors."""
        if self._background_gray is None or self._debug_bg_viewer is None:
            return

        # Convert background to BGR for drawing
        bg_bgr = cv2.cvtColor(self._background_gray, cv2.COLOR_GRAY2BGR)

        # Draw ROI circles with their assigned colors
        if hasattr(self, '_roi_colors') and self.video and self.video.rois:
            for roi in self.video.rois:
                color_hex = self._roi_colors.get(roi.name, "#FFFFFF")
                # Convert hex to BGR
                color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # RGB to BGR

                # Draw ROI circle
                cv2.circle(bg_bgr, (int(roi.cx), int(roi.cy)), int(roi.r), color_bgr, 3)
                # Draw center point
                cv2.circle(bg_bgr, (int(roi.cx), int(roi.cy)), 5, color_bgr, -1)

                # Optionally draw ROI name
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_pos = (int(roi.cx) - 20, int(roi.cy) - int(roi.r) - 10)
                cv2.putText(bg_bgr, roi.name, text_pos, font, 0.6, color_bgr, 2)

        self._debug_bg_viewer.set_image(bg_bgr)

    def _close_tracking_debug_window(self):
        """Close the tracking debug window."""
        if self._tracking_debug_win is not None and self._tracking_debug_win.winfo_exists():
            self._tracking_debug_win.destroy()
        self._tracking_debug_win = None
        self._tracking_debug_widgets = {}
        self._debug_slider = None
        self._debug_frame_label = None
        self._debug_bg_viewer = None

    def _on_debug_slider(self, val):
        """Handle debug window frame slider changes."""
        if self._syncing_sliders:
            return
        try:
            idx = int(float(val))
        except Exception:
            return
        # Update main app to show this frame
        self._syncing_sliders = True
        try:
            self.slider.set(idx)
            self.show_frame(idx)
        finally:
            self._syncing_sliders = False

    def _update_tracking_debug(self, frame: np.ndarray, frame_idx: int):
        """Update tracking debug window with current frame analysis."""
        if not self._tracking_debug_widgets or not self.video or not self.video.rois:
            return

        if self._background_gray is None:
            return  # Need background for debug

        # Update debug slider and label if they exist (with guard to prevent circular updates)
        if not self._syncing_sliders:
            if hasattr(self, '_debug_slider') and self._debug_slider is not None:
                try:
                    self._syncing_sliders = True
                    self._debug_slider.set(frame_idx)
                finally:
                    self._syncing_sliders = False
            if hasattr(self, '_debug_frame_label') and self._debug_frame_label is not None:
                try:
                    self._debug_frame_label.configure(text=f"Frame: {frame_idx}")
                except Exception:
                    pass

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for roi in self.video.rois:
            if roi.name not in self._tracking_debug_widgets:
                continue

            widgets = self._tracking_debug_widgets[roi.name]

            # Extract ROI region
            x0 = int(max(0, roi.cx - roi.r))
            y0 = int(max(0, roi.cy - roi.r))
            x1 = int(min(gray.shape[1], roi.cx + roi.r))
            y1 = int(min(gray.shape[0], roi.cy + roi.r))

            roi_gray = gray[y0:y1, x0:x1]
            bg_roi = self._background_gray[y0:y1, x0:x1]

            # Create mask for circular ROI
            roi_shape = (y1 - y0, x1 - x0)
            mask = roi_mask(roi_shape, roi, (x0, y0))

            # Render from the SAME detection path the tracker uses, so this
            # window can never drift from the real pipeline (diff, binary mask,
            # and candidates all come from detector.debug_frame).
            dbg = debug_frame(roi_gray, bg_roi, self.cfg, mask)
            diff = dbg.diff
            bw = dbg.mask
            candidates = dbg.candidates

            # Get ROI color for drawing
            roi_color_hex = self._roi_colors.get(roi.name, "#FFFFFF")
            # Convert hex to BGR
            roi_color_rgb = tuple(int(roi_color_hex[i:i+2], 16) for i in (1, 3, 5))
            roi_color_bgr = (roi_color_rgb[2], roi_color_rgb[1], roi_color_rgb[0])

            # Create visualizations
            # 1. ROI crop with detected centroid
            roi_vis = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
            if candidates:
                best = candidates[0]
                cv2.circle(roi_vis, (int(best.x), int(best.y)), 4, roi_color_bgr, -1)
                theta = np.radians(best.angle_deg)
                x_end = int(best.x + best.major/2 * np.cos(theta))
                y_end = int(best.y + best.major/2 * np.sin(theta))
                cv2.line(roi_vis, (int(best.x), int(best.y)), (x_end, y_end), roi_color_bgr, 1)

            # 2. Diff image (colorized for visibility)
            diff_vis = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)

            # 3. Binary mask (as BGR for consistency) with contours and ellipses
            bw_vis = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

            # Draw all detected contours and fitted ellipses in ROI color
            if candidates:
                for cand in candidates:
                    # Draw contour if available
                    if hasattr(cand, 'contour') and cand.contour is not None:
                        #cv2.drawContours(bw_vis, [cand.contour], -1, roi_color_bgr, 2)
                        ellipse = cv2.fitEllipse(cand.contour)
                        # ellipse is a RotatedRect: ((center_x, center_y), (major_axis, minor_axis), angle)
                        cv2.ellipse(bw_vis, ellipse, roi_color_bgr, 1) # Draw green ellipse on original image
                        (xc, yc), (d1, d2), angle = ellipse

                        # Calculate major axis endpoint for visualization
                        # Convert OpenCV angle to radians and adjust for Y-down coordinate system
                        theta = np.radians(angle - 90) 
                        length = d2 / 2  # Half of the major axis
                        x_end = int(xc + length * np.cos(theta))
                        y_end = int(yc + length * np.sin(theta))
                        cv2.line(bw_vis, (int(xc), int(yc)), (x_end, y_end), (255, 0, 0), 1)


                    # Draw fitted ellipse
                    """
                    cv2.ellipse(bw_vis, (int(cand.x), int(cand.y)),
                               (int(cand.major/2), int(cand.minor/2)),
                               cand.angle_deg, 0, 360, roi_color_bgr, 2)
                    """

            # Update viewers with new images (they handle scaling automatically)
            widgets["crop_viewer"].set_image(roi_vis)
            widgets["diff_viewer"].set_image(diff_vis)
            widgets["mask_viewer"].set_image(bw_vis)

            # Update metrics text
            metrics_text = widgets["metrics_text"]
            metrics_text.configure(state="normal")
            metrics_text.delete("1.0", "end")

            metrics_text.insert("end", f"Frame: {frame_idx}\n")
            metrics_text.insert("end", f"ROI Position: ({int(roi.cx)}, {int(roi.cy)}) r={int(roi.r)}\n")
            metrics_text.insert("end", f"Candidates detected: {len(candidates)}\n")

            if candidates:
                best = candidates[0]
                metrics_text.insert("end", f"\nBest Match:\n")
                metrics_text.insert("end", f"  Position: ({best.x + x0:.1f}, {best.y + y0:.1f})\n")
                metrics_text.insert("end", f"  Area: {best.area:.1f} px²\n")
                metrics_text.insert("end", f"  Angle: {best.angle_deg:.1f}°\n")
                metrics_text.insert("end", f"  Axes: {best.major:.1f} x {best.minor:.1f}\n")
            else:
                metrics_text.insert("end", f"\n⚠ No valid candidates found!\n")
                metrics_text.insert("end", f"  Check threshold and area limits.\n")

            metrics_text.configure(state="disabled")

    def _track_poll(self):
        """Poll the tracking event queue and update the modal progress dialog."""
        q = getattr(self, "_track_q", None)
        dialog = getattr(self, "_track_dialog", None)
        if dialog is None or q is None:
            return
        try:
            while True:
                event, payload = q.get_nowait()

                if event == "preprocessing":
                    # Handle preprocessing progress
                    roi = payload.get("roi", "")
                    index = payload.get("index", 0)
                    total = payload.get("total", 1)
                    percent = payload.get("percent", 0.0)

                    try:
                        if roi == "complete":
                            status_text = f"Preprocessing complete ({total} ROIs)"
                        else:
                            status_text = f"Preprocessing ROI {index + 1}/{total}: {roi}"

                        dialog.update_progress(percent, text=status_text)
                    except Exception:
                        pass

                elif event == "tracking":
                    # Handle parallel tracking progress
                    status = payload.get("status", "")

                    if status == "starting":
                        n_rois = payload.get("n_rois", 0)
                        n_workers = payload.get("n_workers", 1)
                        try:
                            dialog.update_progress(0.0, text=f"Starting tracking ({n_rois} ROIs, {n_workers} workers)...")
                        except Exception:
                            pass

                    elif status == "progress":
                        completed = payload.get("completed", 0)
                        total_rois = payload.get("total", 1)
                        percent = payload.get("percent", 0.0)
                        try:
                            status_text = f"Tracking ROI {completed}/{total_rois}  {percent*100:.1f}%"
                            dialog.update_progress(percent, text=status_text)
                        except Exception:
                            pass

                    elif status == "complete":
                        n_tracks = payload.get("n_tracks", 0)
                        try:
                            dialog.update_progress(1.0, text=f"Tracking complete ({n_tracks} tracks)")
                        except Exception:
                            pass

                elif event == "started":
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

                elif event == "pose":
                    try:
                        dialog.update_progress(0.98, text="Running pose estimation…")
                    except Exception:
                        pass

                elif event in ("pose_done", "pose_error"):
                    # Non-fatal: tracking still completes; the overlay reads the
                    # session's pose_dataframe on the "done" event.
                    if event == "pose_error":
                        try:
                            print("pose stage error:", payload.get("exc"))
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
                        # Save session after tracking completes
                        self._save_session()
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
                    # Frame event - could be used for future debug visualization
                    pass

        except queue.Empty:
            pass

        # Reschedule
        try:
            self._track_job = self.after(50, self._track_poll)
        except Exception:
            pass

    def _get_session_path(self) -> Path:
        """Get path to session file."""
        from platformdirs import user_data_dir
        data_dir = Path(user_data_dir("anytrack"))
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / "last_session.json"

    def _save_session(self):
        """Save current session state to file."""
        try:
            import json
            import pickle

            session_data = {}

            # Save video path
            if self.video is not None:
                session_data["video_path"] = str(self.video.video_path)
                session_data["current_frame"] = self._current_frame_idx

                # Save ROIs
                if self.video.rois:
                    session_data["rois"] = [
                        {
                            "name": roi.name,
                            "cx": roi.cx,
                            "cy": roi.cy,
                            "r": roi.r,
                            "n_targets": roi.n_targets,
                        }
                        for roi in self.video.rois
                    ]

            # Save background (to separate pickle file)
            # Save full BackgroundModel if available, otherwise fall back to _background_gray
            if self._background_model is not None:
                bg_path = self._get_session_path().parent / "last_background.pkl"
                with open(bg_path, "wb") as f:
                    pickle.dump(self._background_model, f)
                session_data["background_path"] = str(bg_path)
            elif self._background_gray is not None:
                # Backward compatibility: save plain grayscale if no model
                bg_path = self._get_session_path().parent / "last_background.pkl"
                with open(bg_path, "wb") as f:
                    pickle.dump(self._background_gray, f)
                session_data["background_path"] = str(bg_path)

            # Save tracking session (to separate pickle file)
            if self.session is not None:
                session_path = self._get_session_path().parent / "last_tracking.pkl"
                with open(session_path, "wb") as f:
                    pickle.dump(self.session, f)
                session_data["session_path"] = str(session_path)

            # Save UI state
            session_data["zoom_level"] = self._zoom_level
            session_data["pan_x"] = self._pan_x
            session_data["pan_y"] = self._pan_y
            session_data["preview_mode"] = self.preview_mode

            # Write session file
            with open(self._get_session_path(), "w") as f:
                json.dump(session_data, f, indent=2)

            print(f"Session saved to {self._get_session_path()}")

        except Exception as e:
            print(f"Error saving session: {e}")
            import traceback
            traceback.print_exc()

    def _load_session(self):
        """Load previous session state from file."""
        try:
            import json
            import pickle

            session_path = self._get_session_path()
            if not session_path.exists():
                return

            with open(session_path, "r") as f:
                session_data = json.load(f)

            # Load video
            video_path_str = session_data.get("video_path")
            if video_path_str and Path(video_path_str).exists():
                video_path = Path(video_path_str)
                csv_path = video_path.with_suffix(".csv")

                if csv_path.exists():
                    # Load video with timing data
                    self.video = load_video_asset(video_path, csv_path)
                    self._background_gray = None
                    self._set_preview_mode("video")
                    self._background_building = False
                    self._assign_roi_colors()
                    self._open_capture()
                    self._set_slider_range()
                    self._reset_zoom_pan()

                    # Load ROIs
                    rois_data = session_data.get("rois")
                    if rois_data:
                        from ..models import CircleROI
                        self.video.rois = [
                            CircleROI(**roi_data) for roi_data in rois_data
                        ]

                    self._populate_tree()

                    # Load current frame
                    current_frame = session_data.get("current_frame", 0)
                    if current_frame > 0:
                        self._current_frame_idx = current_frame
                else:
                    print(f"Timing CSV not found: {csv_path}")

            # Load background (only if video was loaded)
            if self.video is not None:
                bg_path = session_data.get("background_path")
                if bg_path and Path(bg_path).exists():
                    with open(bg_path, "rb") as f:
                        loaded = pickle.load(f)

                        # Check format: BackgroundModel or plain numpy array
                        if isinstance(loaded, np.ndarray):
                            # Old format: plain grayscale image
                            self._background_gray = loaded
                            self._background_model = None
                        else:
                            # New format: BackgroundModel object
                            self._background_model = loaded
                            self._background_gray = loaded.gmm

                # Load tracking session
                session_path_pkl = session_data.get("session_path")
                if session_path_pkl and Path(session_path_pkl).exists():
                    with open(session_path_pkl, "rb") as f:
                        self.session = pickle.load(f)
                        if self.session is not None:
                            self._update_table(self.session.dataframe)
                            self._populate_tree()

                # Load UI state
                self._zoom_level = session_data.get("zoom_level", 1.0)
                self._pan_x = session_data.get("pan_x", 0.0)
                self._pan_y = session_data.get("pan_y", 0.0)
                preview_mode = session_data.get("preview_mode", "video")

                # Update preview
                if preview_mode == "background" and self._background_gray is not None:
                    self._set_preview_mode("background")
                    self.show_background()
                elif self._current_frame_idx > 0:
                    self.show_frame(self._current_frame_idx)
                else:
                    self.show_frame(0)

            print(f"Session loaded from {session_path}")

        except Exception as e:
            print(f"Error loading session: {e}")
            import traceback
            traceback.print_exc()

    def _on_closing(self):
        """Handle window close event."""
        self._save_session()
        self.destroy()

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
