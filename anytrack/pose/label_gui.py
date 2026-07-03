"""Quick-label GUI (Milestone B2). Entry point: ``anytrack-label``.

A thin ttkbootstrap/Tk shell over :mod:`anytrack.pose.labeling`. Loads a video +
tracks table, samples frames, shows a zoomed centroid crop pre-seeded from the
tracked ellipse, and lets you click/adjust the 5 skeleton keypoints. Saves to a
native JSON store (resume-able) and can export ``.slp`` for SLEAP training.

Keyboard-first for speed:
    1-9        select keypoint            Left/Right   prev / next frame
    space      confirm frame + next       v            toggle active visibility
    b          toggle "bad frame"         w            swap wingL/wingR
    Backspace  clear active keypoint      Ctrl-S       save store

Click the canvas to place the active keypoint (auto-advances to the next);
drag an existing point to nudge it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

from ..config import load_config
from ..cropper import _load_tracks
from .labeling import (
    Keypoint, LabelStore, contrast_fg, crop_to_display, display_to_crop,
    extract_label_crops, resolve_node_colors,
)
from .skeleton import get_skeleton


class LabelApp:
    def __init__(self, store: LabelStore, crops: Dict[int, np.ndarray], save_path: Path,
                 zoom: float = 6.0, node_colors: Optional[Dict[str, str]] = None):
        import tkinter as tk
        from tkinter import ttk
        import ttkbootstrap as tb
        from ttkbootstrap.constants import PRIMARY, SUCCESS, DANGER, INFO, SECONDARY

        self.tk = tk
        self.store = store
        self.crops = crops
        self.save_path = Path(save_path)
        self.zoom = float(zoom)
        self.size = store.crop_size
        self.nodes: List[str] = list(store.skeleton.nodes)
        self.edges = store.skeleton.edge_indices()
        self.colors: Dict[str, str] = node_colors or resolve_node_colors(self.nodes)

        self.idx = 0            # current frame index into store.frames
        self.active = 0         # active node index
        self.offset = 0         # temporal-context offset from the anchor frame
        self.alpha = 1.0        # contrast
        self.beta = 0.0         # brightness offset
        self._drag: Optional[str] = None    # node locked for the current press-drag
        self._placing = False               # True when the press placed a new point
        self._photo = None      # keep a ref so Tk doesn't GC the image

        self.root = tb.Window(themename="darkly")
        self.root.title("anytrack — quick label")
        self.root.minsize(760, 560)
        self._ttk = ttk
        self._style = self.root.style              # ttkbootstrap Style (themed, non-native)
        self._node_style_names = [self._make_node_style(i, n) for i, n in enumerate(self.nodes)]

        disp = int(round(self.size * self.zoom))

        # --- layout ----------------------------------------------------------
        main = tb.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        left = tb.Frame(main)
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, width=disp, height=disp, background="#111",
                                highlightthickness=1, highlightbackground="#444",
                                cursor="tcross")
        self.canvas.pack(anchor="n")
        self.status = tb.Label(left, text="", anchor="w", bootstyle="inverse-dark")
        self.status.pack(fill="x", pady=(6, 0))

        side = tb.Frame(main, padding=(10, 0))
        side.pack(side="left", fill="y")

        self.progress = tb.Label(side, text="", font=("", 11, "bold"))
        self.progress.pack(anchor="w", pady=(0, 6))

        tb.Label(side, text="Keypoints (click or 1-9):").pack(anchor="w")
        self.node_buttons: List = []
        for i, node in enumerate(self.nodes):
            b = ttk.Button(side, text=f"   {i + 1}. {node}", width=18,
                           style=self._node_style_names[i],
                           command=lambda i=i: self._set_active(i))
            b.pack(anchor="w", pady=1)
            self.node_buttons.append(b)

        tb.Separator(side).pack(fill="x", pady=8)
        tb.Button(side, text="Toggle visible (v)", command=self._toggle_visible,
                  bootstyle=INFO).pack(fill="x", pady=1)
        tb.Button(side, text="Swap wings (w)", command=self._swap_wings,
                  bootstyle=INFO).pack(fill="x", pady=1)
        tb.Button(side, text="Clear active (Bksp)", command=self._clear_active,
                  bootstyle=SECONDARY).pack(fill="x", pady=1)
        self.bad_btn = tb.Button(side, text="Mark bad (b)", command=self._toggle_bad,
                                 bootstyle=DANGER)
        self.bad_btn.pack(fill="x", pady=1)

        tb.Separator(side).pack(fill="x", pady=8)
        tb.Label(side, text="Brightness").pack(anchor="w")
        self.bri = tb.Scale(side, from_=-100, to=100, value=0, command=self._on_bri)
        self.bri.pack(fill="x")
        tb.Label(side, text="Contrast").pack(anchor="w")
        self.con = tb.Scale(side, from_=0.3, to=3.0, value=1.0, command=self._on_con)
        self.con.pack(fill="x")

        tb.Separator(side).pack(fill="x", pady=8)
        nav = tb.Frame(side)
        nav.pack(fill="x")
        tb.Button(nav, text="◀ Prev", command=self.prev, bootstyle=SECONDARY).pack(
            side="left", expand=True, fill="x", padx=(0, 2))
        tb.Button(nav, text="Next ▶", command=self.next, bootstyle=PRIMARY).pack(
            side="left", expand=True, fill="x", padx=(2, 0))

        tb.Label(side, text="Dynamics (scrub context):").pack(anchor="w", pady=(8, 0))
        ctx = tb.Frame(side)
        ctx.pack(fill="x")
        tb.Button(ctx, text="◀ , ", command=lambda: self._context_step(-1),
                  bootstyle=SECONDARY).pack(side="left", expand=True, fill="x", padx=(0, 2))
        tb.Button(ctx, text="⦿ 0", command=self._context_reset,
                  bootstyle=INFO).pack(side="left", padx=2)
        tb.Button(ctx, text=" . ▶", command=lambda: self._context_step(1),
                  bootstyle=SECONDARY).pack(side="left", expand=True, fill="x", padx=(2, 0))
        self.ctx_label = tb.Label(side, text="", anchor="center")
        self.ctx_label.pack(fill="x")

        tb.Button(side, text="Confirm + Next (space)", command=self.confirm_next,
                  bootstyle=SUCCESS).pack(fill="x", pady=(6, 1))
        tb.Button(side, text="Save (Ctrl-S)", command=self.save,
                  bootstyle=SUCCESS).pack(fill="x", pady=1)
        tb.Button(side, text="Export .slp…", command=self.export_slp,
                  bootstyle=INFO).pack(fill="x", pady=1)

        # --- bindings --------------------------------------------------------
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.root.bind("<Left>", lambda e: self.prev())
        self.root.bind("<Right>", lambda e: self.next())
        # context scrub: , / . (also < / >) step through neighbor frames; 0 = anchor
        self.root.bind("<comma>", lambda e: self._context_step(-1))
        self.root.bind("<period>", lambda e: self._context_step(1))
        self.root.bind("<less>", lambda e: self._context_step(-1))
        self.root.bind("<greater>", lambda e: self._context_step(1))
        self.root.bind("0", lambda e: self._context_reset())
        self.root.bind("<space>", lambda e: self.confirm_next())
        self.root.bind("v", lambda e: self._toggle_visible())
        self.root.bind("w", lambda e: self._swap_wings())
        self.root.bind("b", lambda e: self._toggle_bad())
        self.root.bind("<BackSpace>", lambda e: self._clear_active())
        self.root.bind("<Delete>", lambda e: self._clear_active())
        self.root.bind("<Control-s>", lambda e: self.save())
        for i in range(len(self.nodes)):
            self.root.bind(str(i + 1), lambda e, i=i: self._set_active(i))

        self._render()

    # --- helpers -------------------------------------------------------------

    @property
    def frame(self):
        return self.store.frames[self.idx]

    def _make_node_style(self, i: int, node: str) -> str:
        """Register a ttk button style filled with this node's color.

        ttkbootstrap draws non-native themed widgets, so ``background`` is
        honored on macOS (unlike classic ``tk.Button``). Colors stay put across
        hover/press/focus via ``style.map``.
        """
        color = self.colors[node]
        fg = contrast_fg(color)
        name = f"anytrackNode{i}.TButton"
        self._style.configure(name, background=color, foreground=fg, bordercolor=color,
                              focuscolor=color, lightcolor=color, darkcolor=color,
                              relief="flat", anchor="w", font=("", 10))
        self._style.map(name,
                        background=[("active", color), ("pressed", color),
                                    ("focus", color), ("disabled", color)],
                        foreground=[("active", fg), ("pressed", fg)])
        return name

    def _avail_offsets(self) -> List[int]:
        """Sorted context offsets available for the current anchor (0 = anchor)."""
        af = int(self.frame.frame)
        return sorted(k - af for k in self.crops.get(self.idx, {}))

    def _crop_img(self) -> np.ndarray:
        af = int(self.frame.frame)
        win = self.crops.get(self.idx, {})
        crop = win.get(af + self.offset)
        if crop is None:
            crop = win.get(af)                       # fall back to the anchor frame
        if crop is None:
            return np.full((self.size, self.size), 40, np.uint8)
        adj = np.clip(self.alpha * (crop.astype(np.float32) - 128) + 128 + self.beta, 0, 255)
        return adj.astype(np.uint8)

    def _context_step(self, delta: int):
        offs = self._avail_offsets()
        if not offs:
            return
        self.offset = int(min(offs[-1], max(offs[0], self.offset + delta)))
        self._render()

    def _context_reset(self):
        self.offset = 0
        self._render()

    def _render(self):
        from PIL import Image, ImageTk

        disp = int(round(self.size * self.zoom))
        img = Image.fromarray(self._crop_img(), mode="L").convert("RGB")
        img = img.resize((disp, disp), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(img)
        c = self.canvas
        c.delete("all")
        c.create_image(0, 0, anchor="nw", image=self._photo)

        pts = self.frame.points
        # edges first (behind points)
        for a, b in self.edges:
            na, nb = self.nodes[a], self.nodes[b]
            pa, pb = pts.get(na), pts.get(nb)
            if pa and pb and pa.visible and pb.visible:
                ax, ay = crop_to_display(pa.x, pa.y, self.zoom)
                bx, by = crop_to_display(pb.x, pb.y, self.zoom)
                c.create_line(ax, ay, bx, by, fill="#888", width=2)
        # points + text annotations
        for i, node in enumerate(self.nodes):
            kp = pts.get(node)
            if kp is None:
                continue
            x, y = crop_to_display(kp.x, kp.y, self.zoom)
            color = self.colors[node]
            active = i == self.active
            r = 7 if active else 5
            outline = "#ffffff" if active else "#000000"
            width = 3 if active else 1
            if kp.visible:
                c.create_oval(x - r, y - r, x + r, y + r, fill=color,
                              outline=outline, width=width)
            else:
                c.create_line(x - r, y - r, x + r, y + r, fill=color, width=2)
                c.create_line(x - r, y + r, x + r, y - r, fill=color, width=2)
            # node name label, placed away from the canvas edge, with a shadow
            anchor = "e" if x > disp * 0.62 else "w"
            tx = x - (r + 4) if anchor == "e" else x + (r + 4)
            font = ("", 10, "bold") if active else ("", 8)
            c.create_text(tx + 1, y + 1, text=node, anchor=anchor, fill="#000000", font=font)
            c.create_text(tx, y, text=node, anchor=anchor, fill=color, font=font)

        for i, b in enumerate(self.node_buttons):
            mark = "▶" if i == self.active else "   "
            b.configure(text=f"{mark} {i + 1}. {self.nodes[i]}")
        self._update_labels()

    def _update_labels(self):
        fl = self.frame
        self.progress.configure(
            text=f"{self.idx + 1}/{len(self.store.frames)}   labeled {self.store.n_labeled()}")
        state = "BAD" if fl.bad_frame else ("✓ labeled" if fl.labeled else "unlabeled")
        self.bad_btn.configure(text="Unmark bad (b)" if fl.bad_frame else "Mark bad (b)")

        offs = self._avail_offsets()
        lo, hi = (offs[0], offs[-1]) if offs else (0, 0)
        at_anchor = self.offset == 0
        if at_anchor:
            self.ctx_label.configure(text=f"anchor · frame {fl.frame} · window [{lo:+d},{hi:+d}]")
        else:
            self.ctx_label.configure(
                text=f"t{self.offset:+d} · frame {int(fl.frame) + self.offset} · press 0 to edit")
        self.status.configure(
            text=f"roi={fl.roi}  frame={fl.frame}  active={self.nodes[self.active]}  [{state}]"
                 + ("" if at_anchor else "  — scrubbing (view only)"))

    # --- editing -------------------------------------------------------------

    def _set_active(self, i: int):
        self.active = i % len(self.nodes)
        self._render()

    def _place_active(self, x_disp: float, y_disp: float):
        xc, yc = display_to_crop(x_disp, y_disp, self.zoom)
        xc = float(min(max(0.0, xc), self.size))
        yc = float(min(max(0.0, yc), self.size))
        node = self.nodes[self.active]
        cur = self.frame.points.get(node)
        vis = cur.visible if cur else True
        self.frame.points[node] = Keypoint(xc, yc, visible=vis, score=1.0)
        self.frame.labeled = True

    def _nearest_node(self, x_disp: float, y_disp: float, thresh_px: float = 12.0):
        best, bd = None, thresh_px
        for node, kp in self.frame.points.items():
            dx, dy = crop_to_display(kp.x, kp.y, self.zoom)
            d = ((dx - x_disp) ** 2 + (dy - y_disp) ** 2) ** 0.5
            if d < bd:
                best, bd = node, d
        return best

    def _on_click(self, e):
        if self.offset != 0:                         # editing is anchor-only
            self._context_reset()
            self.status.configure(text="returned to anchor frame — click again to place")
            return
        hit = self._nearest_node(e.x, e.y)
        if hit is not None:                          # grab an existing point to reposition
            self._drag = hit
            self._placing = False
            self.active = self.nodes.index(hit)
            self._render()
            return
        # place the active node; lock the drag onto it and defer advancing to
        # release, so moving the mouse fine-tunes THIS point (not the next one).
        self._place_active(e.x, e.y)
        self._drag = self.nodes[self.active]
        self._placing = True
        self._render()

    def _on_drag(self, e):
        if self.offset != 0 or self._drag is None:   # only the locked node moves
            return
        node = self._drag
        xc, yc = display_to_crop(e.x, e.y, self.zoom)
        kp = self.frame.points.get(node)
        vis = kp.visible if kp else True
        self.frame.points[node] = Keypoint(
            float(min(max(0.0, xc), self.size)),
            float(min(max(0.0, yc), self.size)), visible=vis, score=1.0)
        self.frame.labeled = True
        self._render()

    def _on_release(self, e):
        # advance to the next node only after finishing a place gesture (not
        # after repositioning an existing point).
        if self._placing:
            self.active = (self.active + 1) % len(self.nodes)
        self._drag = None
        self._placing = False
        self._render()

    def _toggle_visible(self):
        node = self.nodes[self.active]
        kp = self.frame.points.get(node)
        if kp is None:
            kp = Keypoint(self.size / 2, self.size / 2)
        kp.visible = not kp.visible
        self.frame.points[node] = kp
        self.frame.labeled = True
        self._render()

    def _swap_wings(self):
        pts = self.frame.points
        if "wingL" in pts and "wingR" in pts:
            pts["wingL"], pts["wingR"] = pts["wingR"], pts["wingL"]
            self.frame.labeled = True
            self._render()

    def _clear_active(self):
        self.frame.points.pop(self.nodes[self.active], None)
        self._render()

    def _toggle_bad(self):
        self.frame.bad_frame = not self.frame.bad_frame
        self._render()

    # --- sliders -------------------------------------------------------------

    def _on_bri(self, v):
        self.beta = float(v)
        self._render()

    def _on_con(self, v):
        self.alpha = float(v)
        self._render()

    # --- navigation + IO -----------------------------------------------------

    def prev(self):
        self.idx = (self.idx - 1) % len(self.store.frames)
        self.active = 0
        self.offset = 0
        self._render()

    def next(self):
        self.idx = (self.idx + 1) % len(self.store.frames)
        self.active = 0
        self.offset = 0
        self._render()

    def confirm_next(self):
        self.frame.labeled = True
        self.save(quiet=True)
        self.next()

    def save(self, quiet: bool = False):
        self.store.save(self.save_path)
        if not quiet:
            self.status.configure(text=f"saved → {self.save_path}")

    def export_slp(self):
        from tkinter import messagebox
        out = self.save_path.with_suffix(".slp")
        try:
            self.store.export_slp(out)
        except ImportError as e:
            messagebox.showerror("sleap-io missing", str(e))
            return
        except Exception as e:  # noqa: BLE001 - surface any sleap-io error to the user
            messagebox.showerror("Export failed", f"{type(e).__name__}: {e}")
            return
        messagebox.showinfo("Exported", f"{self.store.n_labeled()} labeled frames → {out}")

    def run(self):
        self.root.mainloop()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="anytrack-label",
        description="Quick keypoint-labeling GUI for pose training (Milestone B2).")
    ap.add_argument("--video", required=True, type=Path, help="Source full-resolution video.")
    ap.add_argument("--tracks", type=Path, default=None,
                    help="Tracks table (.parquet/.csv) with roi/frame/x/y[/angle_deg/major/minor].")
    ap.add_argument("--labels", type=Path, default=None,
                    help="Label store JSON (resume if it exists; else where new labels are saved).")
    ap.add_argument("--n", type=int, default=100, help="Frames to sample when starting fresh.")
    ap.add_argument("--strategy", choices=["diversity", "uniform", "random"], default="diversity")
    ap.add_argument("--seed", type=int, default=0, help="Sampling RNG seed (deterministic).")
    ap.add_argument("--crop-size", type=int, default=None, help="Override cfg.crop_size.")
    ap.add_argument("--zoom", type=float, default=6.0, help="Canvas magnification.")
    ap.add_argument("--context", type=int, default=None,
                    help="+/- neighbor frames loaded per sample for the dynamics scrub "
                         "(default from config: pose_label_context).")
    ap.add_argument("--no-seed-ellipse", action="store_true",
                    help="Start with empty keypoints instead of ellipse-seeded.")
    args = ap.parse_args(argv)

    if not args.video.exists():
        ap.error(f"video not found: {args.video}")

    cfg = load_config()
    if args.crop_size is not None:
        cfg.crop_size = args.crop_size

    save_path = args.labels or args.video.with_name(args.video.stem + "_labels.json")

    if save_path.exists():
        store = LabelStore.load(save_path)
        print(f"resuming {store.n_labeled()}/{len(store)} labeled from {save_path}")
    else:
        if args.tracks is None or not args.tracks.exists():
            ap.error("a new session needs --tracks (existing tracks table). "
                     f"(no resumable store at {save_path})")
        df = _load_tracks(args.tracks)
        store = LabelStore.from_tracks(
            str(args.video), df, skeleton=get_skeleton(cfg), crop_size=cfg.crop_size,
            n=args.n, strategy=args.strategy, seed=args.seed,
            seed_ellipse=not args.no_seed_ellipse)
        print(f"sampled {len(store)} frames ({args.strategy}) from {args.tracks}")

    context = args.context if args.context is not None else getattr(cfg, "pose_label_context", 10)
    context = max(0, int(context))
    print(f"extracting crops (+/-{context} context frames per sample)…")
    video = SimpleNamespace(video_path=args.video)
    crops = extract_label_crops(video.video_path, store.frames, store.crop_size,
                                pad=cfg.crop_pad_mode, context=context, show_progress=True)

    colors = resolve_node_colors(store.skeleton.nodes, getattr(cfg, "pose_node_colors", ""))
    LabelApp(store, crops, save_path, zoom=args.zoom, node_colors=colors).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
