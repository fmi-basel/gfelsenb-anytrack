from __future__ import annotations

import argparse
import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from .gui import run as run_gui
from .background import build_background


def gui_main():
    """Entry point for `anytrack-gui` (and `anytrack gui`): launch the Tk GUI.

    This was formerly the bare `anytrack` command; `anytrack` is now a
    subcommand dispatcher (see anytrack.dispatch) and no longer starts the GUI
    by default.
    """
    parser = argparse.ArgumentParser(prog="anytrack-gui")
    parser.add_argument("--nogui", action="store_true", help="Do not start GUI (placeholder).")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode (FFmpeg preprocessing + parallel ROI tracking).",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=2,
        choices=[1, 2, 4],
        help="ROI downscale factor for fast mode (default: 2).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel tracking workers (default: CPU count).",
    )
    args = parser.parse_args()
    if args.nogui:
        print("CLI-only mode is not implemented in this skeleton. Run without --nogui.")
        return
    # Pass fast mode settings to GUI
    run_gui(
        fast_mode=args.fast,
        roi_downscale=args.downscale,
        n_tracking_workers=args.workers,
    )


class _BgDebugApp:
    """Standalone Tk app for debugging background modelling."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = tk.Tk()
        self.root.title("anytrack – Background Debug")
        self.root.geometry("1100x650")

        self.q: queue.Queue = queue.Queue(maxsize=500)
        self.indices: list[int] = []
        self.cache: dict[int, dict[str, object]] = {}  # frame_idx -> {"frame": np.ndarray, "bg": np.ndarray}
        self._bg_result = None
        self._closed = False

        self._frame_img = None
        self._bg_img = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start background worker
        self.worker = threading.Thread(target=self._run_background, daemon=True)
        self.worker.start()

        # Poll debug events
        self.root.after(30, self._poll)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(3, weight=1)

        self.status = ttk.Label(outer, text="Starting…")
        self.status.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        # indices / trials
        self.trials = tk.Text(outer, height=5, wrap="word")
        self.trials.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.trials.insert("1.0", "Trials (k and sampled indices) will appear here.\n")
        self.trials.configure(state="disabled")

        # slider row
        slider_row = ttk.Frame(outer)
        slider_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        slider_row.columnconfigure(0, weight=1)

        self.slider = ttk.Scale(
            slider_row, from_=0, to=0, orient=tk.HORIZONTAL, command=self._on_slider
        )
        self.slider.grid(row=0, column=0, sticky="ew")
        try:
            self.slider.configure(state="disabled")
        except Exception:
            pass

        self.slider_label = ttk.Label(slider_row, text="Sample: – / –")
        self.slider_label.grid(row=0, column=1, sticky="e", padx=(10, 0))

        # Click trough to jump
        self.slider.bind("<Button-1>", self._on_slider_click, add=True)

        lf1 = ttk.Labelframe(outer, text="Sampled frame")
        lf1.grid(row=3, column=0, sticky="nsew", padx=(0, 8))
        lf1.rowconfigure(0, weight=1)
        lf1.columnconfigure(0, weight=1)
        self.frame_label = ttk.Label(lf1)
        self.frame_label.grid(row=0, column=0, sticky="nsew")

        lf2 = ttk.Labelframe(outer, text="Current model (bg)")
        lf2.grid(row=3, column=1, sticky="nsew", padx=(8, 0))
        lf2.rowconfigure(0, weight=1)
        lf2.columnconfigure(0, weight=1)
        self.bg_label = ttk.Label(lf2)
        self.bg_label.grid(row=0, column=0, sticky="nsew")

        btn_row = ttk.Frame(outer)
        btn_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        btn_row.columnconfigure(0, weight=1)

        self.save_btn = ttk.Button(
            btn_row, text="Save background…", command=self._save_bg, state="disabled"
        )
        self.save_btn.grid(row=0, column=1, sticky="e")

    def _on_close(self):
        self._closed = True
        try:
            self.root.destroy()
        except Exception:
            pass

    def _enqueue(self, event: str, payload: dict):
        if self._closed:
            return
        try:
            if self.q.full():
                try:
                    self.q.get_nowait()
                except Exception:
                    pass
            self.q.put_nowait((event, payload))
        except Exception:
            return

    def _run_background(self):
        def hook(event: str, payload: dict):
            self._enqueue(event, payload)

        try:
            model = build_background(
                self.args.video,
                method=self.args.method,
                k_min=self.args.k_min,
                k_max=self.args.k_max,
                converge_eps=self.args.converge_eps,
                fg_thresh=self.args.fg_thresh,
                inpaint_ghosts=(not self.args.no_inpaint),
                ghost_area_min=self.args.ghost_area_min,
                ghost_area_max=self.args.ghost_area_max,
                sampling=self.args.sampling,
                pca_max_candidates=self.args.pca_max_candidates,
                pca_max_side=self.args.pca_max_side,
                kmeans_iters=self.args.kmeans_iters,
                kmeans_restarts=self.args.kmeans_restarts,
                kmeans_seed=self.args.kmeans_seed,
                debug=True,
                debug_hook=hook,
                debug_stride=self.args.debug_stride,
                debug_max_side=self.args.debug_max_side,
            )
            self._bg_result = model.image
            self._enqueue("done", {"ok": True})
        except Exception as e:
            self._enqueue("done", {"ok": False, "error": str(e)})

    def _poll(self):
        if self._closed:
            return

        try:
            while True:
                event, payload = self.q.get_nowait()

                if event == "trial":
                    k = payload.get("k")
                    idxs = payload.get("indices", [])
                    self.indices = [int(x) for x in idxs]
                    self.cache = {}

                    self.trials.configure(state="normal")
                    self.trials.insert("end", f"k={k}: {self.indices}\n")
                    self.trials.see("end")
                    self.trials.configure(state="disabled")

                    if self.indices:
                        try:
                            self.slider.configure(from_=0, to=len(self.indices) - 1, state="normal")
                        except Exception:
                            self.slider.configure(from_=0, to=len(self.indices) - 1)
                        self.slider.set(0)
                        self._show_pos(0)
                    else:
                        try:
                            self.slider.configure(state="disabled")
                        except Exception:
                            pass
                        self.slider_label.configure(text="Sample: – / –")

                elif event == "frame":
                    try:
                        fi = int(payload.get("frame_idx"))
                        self.cache[fi] = {
                            "frame": payload.get("frame"),
                            "bg": payload.get("bg"),
                        }
                    except Exception:
                        fi = None

                    method = payload.get("method", "")
                    step = payload.get("step", "?")
                    steps = payload.get("steps", "?")
                    frame_idx = payload.get("frame_idx", "?")
                    self.status.configure(text=f"{method}  step {step}/{steps}  frame {frame_idx}")

                    if fi is not None and self.indices:
                        try:
                            pos = int(round(float(self.slider.get())))
                        except Exception:
                            pos = 0
                        if 0 <= pos < len(self.indices) and self.indices[pos] == fi:
                            self._show_pos(pos)

                elif event == "done":
                    if payload.get("ok"):
                        self.status.configure(text="Done. Background built successfully.")
                        self.save_btn.configure(state="normal")
                        if self._bg_result is not None:
                            self._render_bg(self._bg_result)
                    else:
                        err = payload.get("error", "Unknown error")
                        self.status.configure(text=f"Failed: {err}")
                        messagebox.showerror("anytrack-bg", f"Background build failed:\n{err}")

        except queue.Empty:
            pass

        self.root.after(30, self._poll)

    def _on_slider(self, val):
        try:
            pos = int(round(float(val)))
        except Exception:
            pos = 0
        self._show_pos(pos)

    def _on_slider_click(self, event):
        try:
            w = self.slider.winfo_width()
            if w <= 1 or not self.indices:
                return
            frac = max(0.0, min(1.0, event.x / float(w)))
            pos = int(round(frac * (len(self.indices) - 1)))
            self.slider.set(pos)
            self._show_pos(pos)
        except Exception:
            return

    def _show_pos(self, pos: int):
        if not self.indices:
            self.slider_label.configure(text="Sample: – / –")
            return
        pos = max(0, min(len(self.indices) - 1, int(pos)))
        frame_idx = int(self.indices[pos])

        width = max(3, len(str(len(self.indices))))
        self.slider_label.configure(
            text=f"Sample: {pos+1:>{width}d} / {len(self.indices):>{width}d}   (frame {frame_idx})"
        )

        payload = self.cache.get(frame_idx)
        if not payload:
            return

        fr = payload.get("frame")
        bg = payload.get("bg")
        if fr is not None:
            self._render_frame(fr)
        if bg is not None:
            self._render_bg(bg)

    def _render_frame(self, gray):
        im = Image.fromarray(gray)
        self._frame_img = ImageTk.PhotoImage(im)
        self.frame_label.configure(image=self._frame_img)

    def _render_bg(self, gray):
        im = Image.fromarray(gray)
        self._bg_img = ImageTk.PhotoImage(im)
        self.bg_label.configure(image=self._bg_img)

    def _save_bg(self):
        if self._bg_result is None:
            return
        out = filedialog.asksaveasfilename(
            title="Save background",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*")],
        )
        if not out:
            return
        try:
            cv2.imwrite(out, self._bg_result)
            messagebox.showinfo("anytrack-bg", f"Saved background to:\n{out}")
        except Exception as e:
            messagebox.showerror("anytrack-bg", f"Failed to save background:\n{e}")


def bg_main():
    """Entry point for `anytrack-bg` (background debug UI)."""
    parser = argparse.ArgumentParser(
        prog="anytrack-bg", description="Debug anytrack background modelling."
    )
    parser.add_argument("video", nargs="?", help="Path to AVI video")

    parser.add_argument(
        "--method",
        default="mean_excluding_fg",
        choices=["mean_excluding_fg", "mean", "median", "mog2"],
        help="Background method",
    )
    parser.add_argument(
        "--sampling",
        default="uniform",
        choices=["uniform", "pca_kmeans"],
        help="Frame sampling strategy",
    )

    parser.add_argument("--k-min", dest="k_min", type=int, default=30)
    parser.add_argument("--k-max", dest="k_max", type=int, default=300)
    parser.add_argument("--converge-eps", dest="converge_eps", type=float, default=2.5)
    parser.add_argument("--fg-thresh", dest="fg_thresh", type=int, default=25)

    parser.add_argument("--no-inpaint", action="store_true", help="Disable ghost inpainting")
    parser.add_argument("--ghost-area-min", dest="ghost_area_min", type=int, default=20)
    parser.add_argument("--ghost-area-max", dest="ghost_area_max", type=int, default=600)

    parser.add_argument("--pca-max-candidates", dest="pca_max_candidates", type=int, default=400)
    parser.add_argument("--pca-max-side", dest="pca_max_side", type=int, default=96)
    parser.add_argument("--kmeans-iters", dest="kmeans_iters", type=int, default=50)
    parser.add_argument("--kmeans-restarts", dest="kmeans_restarts", type=int, default=2)
    parser.add_argument("--kmeans-seed", dest="kmeans_seed", type=int, default=0)

    parser.add_argument("--debug-stride", dest="debug_stride", type=int, default=3)
    parser.add_argument("--debug-max-side", dest="debug_max_side", type=int, default=480)

    args = parser.parse_args()

    if not args.video:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select video",
            filetypes=[("AVI", "*.avi"), ("All files", "*")],
        )
        root.destroy()
        if not path:
            return
        args.video = path

    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    app = _BgDebugApp(args)
    app.root.mainloop()

def roi_main():
    """Entry point for `anytrack-roi` (ROI debug UI)."""
    import argparse
    import os
    import threading
    import queue
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox

    import cv2
    import numpy as np
    from PIL import Image, ImageTk

    from .background import build_background
    from .roi import detect_circular_rois

    parser = argparse.ArgumentParser(prog="anytrack-roi", description="Debug anytrack ROI (arena) detection.")
    parser.add_argument("video", nargs="?", help="Path to AVI video (used to compute background)")
    # GMM Background parameters
    parser.add_argument("--gmm-n-samples", type=int, default=100,
                        help="Number of frames to sample for GMM background")
    parser.add_argument("--gmm-bic-improvement", type=float, default=10.0,
                        help="BIC improvement threshold for 2-component GMM")
    parser.add_argument("--gmm-min-std", type=float, default=10.0,
                        help="Minimum std deviation to fit GMM")
    parser.add_argument("--gmm-lowp", type=float, default=120.0,
                        help="Minimum brightness value for background")
    parser.add_argument("--arena-blur-sigma", type=float, default=3.0,
                        help="Gaussian blur sigma within detected arenas")

    args = parser.parse_args()

    if not args.video:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title="Select video", filetypes=[("AVI", "*.avi"), ("All files", "*")])
        root.destroy()
        if not path:
            return
        args.video = path

    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    class App:
        def __init__(self):
            self.root = tk.Tk()
            self.root.title("anytrack – ROI Debug")
            self.root.geometry("1250x750")

            self.bg_gray = None
            self.proc_gray = None
            self.th_mask = None
            self.rois = []

            self._overlay_pil = None
            self._mask_pil = None
            self._render_job = None

            self.q = queue.Queue(maxsize=200)
            self._job = None

            self._img_left = None
            self._img_right = None

            self._build_ui()
            self._start_background_build()

        def _build_ui(self):
            outer = ttk.Frame(self.root, padding=10)
            outer.pack(fill="both", expand=True)
            outer.columnconfigure(0, weight=1)
            outer.columnconfigure(1, weight=1)
            outer.rowconfigure(1, weight=1)

            self.status = ttk.Label(outer, text="Building background…")
            self.status.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

            # image panes
            lf1 = ttk.Labelframe(outer, text="Background + detected ROIs")
            lf1.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
            lf1.rowconfigure(0, weight=1)
            lf1.columnconfigure(0, weight=1)
            self.left = ttk.Label(lf1)
            self.left.grid(row=0, column=0, sticky="nsew")

            lf2 = ttk.Labelframe(outer, text="Adaptive threshold mask")
            lf2.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
            lf2.rowconfigure(0, weight=1)
            lf2.columnconfigure(0, weight=1)
            self.right = ttk.Label(lf2)
            self.right.grid(row=0, column=0, sticky="nsew")

            self.left.bind("<Configure>", lambda e: self._schedule_render())
            self.right.bind("<Configure>", lambda e: self._schedule_render())

            # controls + results
            bottom = ttk.Frame(outer)
            bottom.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
            bottom.columnconfigure(0, weight=1)
            bottom.columnconfigure(1, weight=1)

            ctrl = ttk.Labelframe(bottom, text="ROI parameters")
            ctrl.grid(row=0, column=0, sticky="ew", padx=(0, 6))
            ctrl.columnconfigure(1, weight=1)

            def add_spin(row, label, var, frm, to, inc=1):
                ttk.Label(ctrl, text=label).grid(row=row, column=0, sticky="w", pady=2)
                sp = ttk.Spinbox(ctrl, textvariable=var, from_=frm, to=to, increment=inc, width=8, command=self._schedule_detect)
                sp.grid(row=row, column=1, sticky="w", pady=2)
                sp.bind("<KeyRelease>", lambda e: self._schedule_detect())
                return sp

            def add_scale(row, label, var, frm, to, res=1.0):
                ttk.Label(ctrl, text=label).grid(row=row, column=2, sticky="w", padx=(16, 0), pady=2)
                sc = ttk.Scale(ctrl, variable=var, from_=frm, to=to, command=lambda v: self._schedule_detect())
                sc.grid(row=row, column=3, sticky="ew", pady=2)
                ttk.Label(ctrl, textvariable=var, width=6).grid(row=row, column=4, sticky="w", padx=(6, 0))
                ctrl.columnconfigure(3, weight=1)
                return sc

            # vars (match roi.detect_circular_rois defaults)
            self.v_block = tk.IntVar(value=342)
            self.v_csub = tk.IntVar(value=5)
            self.v_minr = tk.IntVar(value=300)
            self.v_maxr = tk.IntVar(value=700)
            self.v_mind = tk.IntVar(value=80)
            self.v_circ = tk.DoubleVar(value=0.65)
            self.v_fill = tk.DoubleVar(value=0.55)

            add_spin(0, "block_size (odd)", self.v_block, 3, 401, 2)
            add_spin(1, "c_sub", self.v_csub, -50, 50, 1)
            add_spin(2, "min_radius", self.v_minr, 1, 2000, 1)
            add_spin(3, "max_radius", self.v_maxr, 1, 4000, 1)
            add_spin(4, "min_dist", self.v_mind, 1, 4000, 1)
            add_scale(0, "circ_min", self.v_circ, 0.1, 0.95)
            add_scale(1, "fill_min", self.v_fill, 0.1, 0.95)

            ttk.Button(ctrl, text="Detect now", command=self._detect).grid(row=5, column=0, sticky="w", pady=(6, 0))
            ttk.Button(ctrl, text="Save mask…", command=self._save_mask).grid(row=5, column=1, sticky="w", pady=(6, 0))
            ttk.Button(ctrl, text="Save overlay…", command=self._save_overlay).grid(row=5, column=2, sticky="w", pady=(6, 0))

            res = ttk.Labelframe(bottom, text="Detected ROIs")
            res.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
            res.rowconfigure(0, weight=1)
            res.columnconfigure(0, weight=1)

            self.tree = ttk.Treeview(res, columns=("cx", "cy", "r"), show="headings", height=7)
            self.tree.heading("cx", text="cx")
            self.tree.heading("cy", text="cy")
            self.tree.heading("r", text="r")
            self.tree.grid(row=0, column=0, sticky="nsew")

            scr = ttk.Scrollbar(res, orient="vertical", command=self.tree.yview)
            scr.grid(row=0, column=1, sticky="ns")
            self.tree.configure(yscrollcommand=scr.set)

        def _start_background_build(self):
            def worker():
                try:
                    model = build_background(
                        args.video,
                        n_samples=args.gmm_n_samples,
                        bic_improvement=args.gmm_bic_improvement,
                        min_std=args.gmm_min_std,
                        lowp=args.gmm_lowp,
                        arena_detection=True,
                        arena_blur_sigma=args.arena_blur_sigma,
                        debug=False,
                    )
                    self.q.put(("bg", model.gmm))
                except Exception as e:
                    self.q.put(("err", str(e)))

            threading.Thread(target=worker, daemon=True).start()
            self.root.after(50, self._poll)

        def _poll(self):
            try:
                while True:
                    typ, payload = self.q.get_nowait()
                    if typ == "bg":
                        self.bg_gray = payload
                        self.status.configure(text="Background ready. Adjust parameters to detect ROIs.")
                        self._detect()
                    elif typ == "err":
                        messagebox.showerror("anytrack-roi", f"Background build failed:\n{payload}")
                        self.status.configure(text="Failed.")
                        return
            except queue.Empty:
                pass
            self.root.after(50, self._poll)

        def _schedule_detect(self):
            if self.bg_gray is None:
                return
            if self._job is not None:
                try:
                    self.root.after_cancel(self._job)
                except Exception:
                    pass
            self._job = self.root.after(150, self._detect)  # throttle updates

        def _compute_mask(self, bg):
            g = bg
            if g.ndim == 3:
                g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g, (0, 0), 2)
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                g = clahe.apply(g)
            except Exception:
                g = cv2.equalizeHist(g)

            bs = int(self.v_block.get())
            if bs < 3:
                bs = 3
            if bs % 2 == 0:
                bs += 1
            cs = int(self.v_csub.get())

            th = cv2.adaptiveThreshold(
                g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, cs
            )
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
            return g, th

        def _detect(self):
            if self.bg_gray is None:
                return

            self.proc_gray, self.th_mask = self._compute_mask(self.bg_gray)

            self.rois = detect_circular_rois(
                self.bg_gray,
                min_dist=int(self.v_mind.get()),
                min_radius=int(self.v_minr.get()),
                max_radius=int(self.v_maxr.get()),
                block_size=int(self.v_block.get()),
                c_sub=int(self.v_csub.get()),
                circ_min=float(self.v_circ.get()),
                fill_min=float(self.v_fill.get()),
            )

            self.status.configure(text=f"Detected {len(self.rois)} ROI(s).")
            self._render()

        def _render(self):
            # Build overlay PIL and mask PIL once, then scale-to-widget for display.
            bg = self.bg_gray
            bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR) if bg.ndim == 2 else bg.copy()

            for r in self.rois:
                cv2.circle(
                    bgr,
                    (int(round(r.cx)), int(round(r.cy))),
                    int(round(r.r)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    bgr,
                    r.name,
                    (int(r.cx + 5), int(r.cy - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            self._overlay_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

            if self.th_mask is not None:
                self._mask_pil = Image.fromarray(self.th_mask)
            else:
                self._mask_pil = None

            # Render scaled versions into the labels
            self._render_scaled()

            # Update table
            for i in self.tree.get_children():
                self.tree.delete(i)
            for r in self.rois:
                self.tree.insert("", "end", values=(f"{r.cx:.1f}", f"{r.cy:.1f}", f"{r.r:.1f}"))

        def _schedule_render(self):
            if self._render_job is not None:
                try:
                    self.root.after_cancel(self._render_job)
                except Exception:
                    pass
            self._render_job = self.root.after(30, self._render_scaled)

        def _fit_to_widget(self, pil_img: Image.Image, widget) -> Image.Image:
            # Fit image to widget while preserving aspect ratio (allows upscaling).
            w = max(1, widget.winfo_width())
            h = max(1, widget.winfo_height())

            iw, ih = pil_img.size
            if iw <= 0 or ih <= 0:
                return pil_img

            scale = min(w / iw, h / ih)
            new_w = max(1, int(round(iw * scale)))
            new_h = max(1, int(round(ih * scale)))

            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = Image.LANCZOS  # older Pillow

            if (new_w, new_h) == (iw, ih):
                return pil_img
            return pil_img.resize((new_w, new_h), resample=resample)

        def _render_scaled(self):
            self._render_job = None

            if self._overlay_pil is not None:
                im = self._fit_to_widget(self._overlay_pil, self.left)
                self._img_left = ImageTk.PhotoImage(im)
                self.left.configure(image=self._img_left)

            if self._mask_pil is not None:
                im2 = self._fit_to_widget(self._mask_pil, self.right)
                self._img_right = ImageTk.PhotoImage(im2)
                self.right.configure(image=self._img_right)


        def _schedule_render(self):
            if self._render_job is not None:
                try:
                    self.root.after_cancel(self._render_job)
                except Exception:
                    pass
            self._render_job = self.root.after(30, self._render_scaled)

        def _fit_to_widget(self, pil_img: Image.Image, widget) -> Image.Image:
            # Fit image to widget while preserving aspect ratio (allows upscaling).
            w = max(1, widget.winfo_width())
            h = max(1, widget.winfo_height())

            iw, ih = pil_img.size
            if iw <= 0 or ih <= 0:
                return pil_img

            scale = min(w / iw, h / ih)
            new_w = max(1, int(round(iw * scale)))
            new_h = max(1, int(round(ih * scale)))

            try:
                resample = Image.Resampling.LANCZOS
            except Exception:
                resample = Image.LANCZOS  # older Pillow

            if (new_w, new_h) == (iw, ih):
                return pil_img
            return pil_img.resize((new_w, new_h), resample=resample)

        def _render_scaled(self):
            self._render_job = None

            if self._overlay_pil is not None:
                im = self._fit_to_widget(self._overlay_pil, self.left)
                self._img_left = ImageTk.PhotoImage(im)
                self.left.configure(image=self._img_left)

            if self._mask_pil is not None:
                im2 = self._fit_to_widget(self._mask_pil, self.right)
                self._img_right = ImageTk.PhotoImage(im2)
                self.right.configure(image=self._img_right)

        def _save_mask(self):
            if self.th_mask is None:
                return
            out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
            if not out:
                return
            cv2.imwrite(out, self.th_mask)

        def _save_overlay(self):
            if self.bg_gray is None:
                return
            out = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
            if not out:
                return
            bg = self.bg_gray
            bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR) if bg.ndim == 2 else bg.copy()
            for r in self.rois:
                cv2.circle(bgr, (int(round(r.cx)), int(round(r.cy))), int(round(r.r)), (0, 255, 0), 2)
            cv2.imwrite(out, bgr)

    app = App()
    app.root.mainloop()