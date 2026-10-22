"""RSSILab GUI (Tkinter + Matplotlib).

    python -m rssilab [--plugin my_algorithms.py ...]
"""
from __future__ import annotations

import argparse
import copy
import queue
import threading
import time
import traceback
import warnings
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import matplotlib
matplotlib.use("TkAgg")
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Ellipse, Polygon, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from . import __version__, engine, metrics, presets  # noqa: E402
from .channel import effective_sigma, fading_stats  # noqa: E402
from .fingerprinting import survey_points  # noqa: E402
from .geometry import import_grid_layout, inside_obstacles, point_seg_dist  # noqa: E402
from .grid_layouts import LAYOUT_NAMES  # noqa: E402
from .localization import ALGORITHMS, DISTANCE, FINGERPRINTING, canonical, family_of, methods  # noqa: E402
from .model import (FADING_MODELS, LEGACY_PRESET, MATERIAL_COLORS, MATERIAL_TABLE, PRESETS,  # noqa: E402
                    Anchor, Obstacle, Technology, load_project, save_project)
from .trajectory import POINT_SOURCES, TRAJ_TYPES, TrajParams, build_route, static_points  # noqa: E402

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf",
           "#bcbd22", "#7f7f7f", "#393b79", "#ad494a", "#637939", "#8c6d31", "#7b4173", "#3182bd"]
BOTH = "Both"                                   # positioning approach: both families on the same RSS
APPROACHES = (DISTANCE, FINGERPRINTING, BOTH)
TYPE_LABEL = {DISTANCE: "Distance", FINGERPRINTING: "Fingerprint", engine.TRACKING: "Tracking"}
APP_NAME = "RSSILab Indoor Positioning Simulator"
REPO_URL = "https://github.com/GiovanniPettorru/RSSILab-Indoor-Positioning-Simulator"
GUIDE_URL = REPO_URL + "/blob/HEAD/docs/GUIDE_gui.md"
DOCS_URL = REPO_URL + "/tree/HEAD/docs"
QUICK_GUIDE = (
    ("1. Scenario",
     "Start from a preset (Load scenario) or a grid-world layout (Import). Set the room, the target area and "
     "the heights, choose 2D or 3D estimation and press Apply geometry. Add anchors with the Anchor tool, the "
     "anchor table or Number on perimeter + Place. Draw walls (press, drag, release) and blocks with the tools "
     "above the map, after choosing Material, Wall thickness and Block top z; the obstacle table sets the "
     "material, loss override and height range of the selected obstacle."),
    ("2. Radio",
     "Load a technology preset and adjust the link budget and the path loss (P0, n), the shadowing "
     "(correlation 0: a new value at every reading; > 0: a static field, the same at every visit of a place), "
     "the fading, the receiver sensitivity, the RSSI quantisation, the sample rate and the samples averaged in "
     "each fix. The text under the panel shows P0, the fading bias, the per-fix RSS std and the range."),
    ("3. Experiment",
     "Positioning: a static target on test points (Test point tool, Random points or Grid) with a number of "
     "fixes per point. Tracking: a target moving along waypoints (Waypoint tool), a random walk, a circle or a "
     "lawnmower path, routed around obstacles with A*. Trials repeat the experiment with new noise and, "
     "optionally, a new shadowing realisation."),
    ("4. Algorithms",
     "Choose the positioning approach (Distance-based, Fingerprinting or Both) and the methods. The "
     "fingerprinting models are trained on a survey of the scenario: set its grid spacing and the "
     "fingerprints per point (Show the survey grid on the map previews it). In Tracking, add the Kalman filter "
     "and the particle filter. The estimator calibration (Oracle or Manual) sets the path-loss model used by "
     "the distance-based methods. Press Run simulation; Stop interrupts it."),
    ("5. Results",
     "Summary: one row per method; click a row to show or hide that method in the plots. Per point "
     "(Positioning): bias, spread and errors on each test point. Per trial (Tracking): statistics of each "
     "trial. Plots: Map, Error CDF, Error detail and 3D view."),
    ("6. Analysis",
     "Export the RSS dataset, the statistics and the results per fix as CSV files (also from the File menu). "
     "The log reports the notes of each run."),
    ("Map editing",
     "Right-click removes the nearest item, Delete or Backspace removes the selection, Esc cancels a drawing; "
     "walls within 7 degrees of the axes are straightened. File > Save project stores the scenario and all "
     "the settings in a JSON file."),
    ("Your own methods",
     "Add them to plugins/user_algorithms.py (a template with one method per family) or to another .py file "
     "of the plugins folder, then restart RSSILab; files elsewhere can be loaded with: rssilab --plugin FILE. "
     "GUIDE_custom_algorithms explains how to write and evaluate them."),
    ("Documentation",
     "The docs folder of RSSILab has one guide per topic: GUIDE_gui (this interface, field by field), "
     "GUIDE_propagation (the channel model), GUIDE_algorithms (the positioning methods), GUIDE_tracking "
     "(routes, Kalman and particle filters), GUIDE_custom_algorithms (testing your own algorithm) and "
     "GUIDE_python_api (scripts and datasets). "
     "Help > User guide and Help > All guides open them online."),
)
TOOLS = [("Select", "select"), ("Anchor", "anchor"), ("Wall", "wall"), ("Block", "rect"),
         ("Test point", "testpoint"), ("Waypoint", "waypoint"), ("Move", "move"), ("Erase", "erase")]
TOOL_HINTS = {
    "select": "Click an obstacle or an anchor to select it (Delete removes it).",
    "anchor": "Click to add an anchor.",
    "wall": "Press, drag and release to draw a wall with the chosen material.",
    "rect": "Press, drag and release to draw a block (furniture, rack, pillar).",
    "testpoint": "Click to add a static test point (Positioning mode).",
    "waypoint": "Click to append a waypoint to the trajectory (Tracking mode).",
    "move": "Drag anchors, test points, waypoints or obstacles.",
    "erase": "Click an anchor, test point, waypoint or obstacle to remove it.",
}


def summary_cols(dim: int):
    hv = [("rmse_xy", "RMSE xy", 74), ("rmse_z", "RMSE z", 66)] if dim == 3 else []
    return ((("show", "Show", 46), ("algo", "Method (m; MSE m²)", 182), ("type", "Type", 76),
             ("mean", "Mean", 54), ("rmse", "RMSE", 56)) + tuple(hv) +
            (("mse", "MSE", 58), ("std", "Std", 50), ("min", "Min", 50), ("median", "Median", 62),
             ("p90", "P90", 52), ("p95", "P95", 52), ("max", "Max", 54), ("valid", "Valid %", 64),
             ("ms", "ms/fix", 56), ("train", "Train s", 60)))


def point_cols(dim: int):
    z = [("z", "z", 50)] if dim == 3 else []
    ez = [("est_z", "Est. z", 60)] if dim == 3 else []
    hv = [("rmse_xy", "RMSE xy", 64), ("rmse_z", "RMSE z", 60)] if dim == 3 else []
    return ((("id", "Point", 52), ("x", "x", 58), ("y", "y", 58)) + tuple(z) +
            (("est_x", "Est. x", 64), ("est_y", "Est. y", 64)) + tuple(ez) +
            (("bias", "Bias", 58), ("spread", "Spread", 60), ("mean", "Mean", 58), ("rmse", "RMSE", 58)) +
            tuple(hv) +
            (("mse", "MSE", 64), ("std", "Std", 56), ("min", "Min", 56), ("median", "Median", 60),
             ("p90", "P90", 56), ("max", "Max", 56), ("n", "Fixes", 56), ("valid", "Valid %", 60)))


def trial_cols(dim: int):
    hv = [("rmse_xy", "RMSE xy", 66), ("rmse_z", "RMSE z", 62)] if dim == 3 else []
    return ((("id", "Trial", 52), ("mean", "Mean", 62), ("rmse", "RMSE", 62)) + tuple(hv) +
            (("mse", "MSE", 66), ("std", "Std", 58), ("min", "Min", 58), ("median", "Median", 62),
             ("p90", "P90", 58), ("p95", "P95", 58), ("max", "Max", 58), ("n", "Fixes", 58),
             ("valid", "Valid %", 62)))


POINT_COLS = point_cols(2)


class ScrollFrame(ttk.Frame):
    """Vertically scrollable container for long parameter panels."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, padding=(6, 6))
        self.inner.bind("<Configure>", self._on_inner)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.canvas.configure(yscrollcommand=sb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _on_inner(self, _):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        w = self.inner.winfo_reqwidth()
        # Tk 9 returns cget("width") as a screen distance such as '10c': track it ourselves
        if getattr(self, "_width", None) != w:
            self._width = w
            self.canvas.configure(width=w)

    def scroll(self, steps: int) -> None:
        if self.inner.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(steps, "units")

    def scroll_pixels(self, px: float) -> None:
        total = self.inner.winfo_reqheight()
        if total > self.canvas.winfo_height():
            self.canvas.yview_moveto(min(max(self.canvas.yview()[0] + px / total, 0.0), 1.0))


def install_wheel_scrolling(root: tk.Misc) -> None:
    """One application-wide handler for the mouse wheel (and, with Tk 9, the touchpad)
    that scrolls the ScrollFrame under the pointer. Per-widget Enter/Leave bindings are
    unreliable (Tk 9 sends Leave to a frame when the pointer moves onto its children),
    and the wheel must not change the value of the comboboxes and spinboxes it passes
    over while a parameter panel is scrolled."""
    for cls in ("TCombobox", "TSpinbox"):
        for seq in ("<MouseWheel>", "<Shift-MouseWheel>", "<Alt-MouseWheel>", "<Button-4>", "<Button-5>",
                    "<TouchpadScroll>"):
            try:
                root.unbind_class(cls, seq)
            except tk.TclError:                   # event unknown to this Tk version
                pass

    def panel_under(e):
        try:
            w = root.winfo_containing(e.x_root, e.y_root)
        except (KeyError, tk.TclError):          # pointer on a Tk-internal widget (e.g. a popdown)
            return None
        while w is not None and not isinstance(w, ScrollFrame):
            if isinstance(w, (ttk.Treeview, tk.Text, tk.Listbox)):
                return None                       # widgets that scroll themselves
            w = w.master
        return w

    def wheel(e, steps=None):
        w = panel_under(e)
        if w is not None:
            if steps is None:
                steps = int(-e.delta / 120) or (-1 if e.delta > 0 else 1)
            w.scroll(2 * steps)

    def touchpad(e):                              # Tk 9: x and y deltas packed in %D
        dy = e.delta & 0xFFFF
        dy = dy - 0x10000 if dy & 0x8000 else dy
        w = panel_under(e) if dy else None
        if w is not None:
            w.scroll_pixels(-dy)

    root.bind_all("<MouseWheel>", wheel, add="+")
    root.bind_all("<Button-4>", lambda e: wheel(e, -1), add="+")      # X11 with Tk 8.6
    root.bind_all("<Button-5>", lambda e: wheel(e, 1), add="+")
    try:
        root.bind_all("<TouchpadScroll>", touchpad, add="+")
    except tk.TclError:                           # Tk 8.6 has no touchpad events
        pass


class RSSILabApp:
    def __init__(self, root: tk.Tk, plugins=None):
        self.root = root
        root.title(f"{APP_NAME} {__version__}")
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1500, sw - 40), min(930, sh - 90)        # fit small screens
        root.geometry(f"{w}x{h}")
        root.minsize(min(1100, w), min(700, h))
        self.scn = presets.default_scenario()
        self.v, self._labels = {}, {}
        self.results = None
        self.results_visible = False
        self.route_preview = None
        self.points_preview = None
        self.survey_preview = None
        self.hidden = set()
        self.sel = None                   # ("anchor"|"obstacle", index)
        self._drag = None
        self._draw_start = None
        self._preview_artist = None
        self._last_redraw = 0.0
        self._worker = None
        self._cancel = False
        self._playing = False
        self._time_artists = {}
        self.plugin_msgs = engine.load_plugins(plugins)

        self._style()
        self._menu()
        self._layout()
        install_wheel_scrolling(root)
        self._load_tech(presets.technology(LEGACY_PRESET))
        self._scn_to_fields()
        self._refresh_lists()
        self.draw_map()
        self._log(f"RSSILab {__version__}: reference scenario (90 x 60 m, 12 anchors), "
                  f"'{LEGACY_PRESET}' radio preset.")
        for m in self.plugin_msgs:
            self._log(m)
        root.bind("<Delete>", lambda e: self._delete_selected())
        root.bind("<BackSpace>", lambda e: self._delete_selected())      # the Delete key on macOS
        root.bind("<Escape>", lambda e: self._cancel_drawing())

    # ------------------------------------------------------------------ style
    def _style(self):
        st = ttk.Style(self.root)
        if "clam" in st.theme_names():
            st.theme_use("clam")
        # bold fonts derived from the platform default font; the macOS system fonts (families named
        # '.AppleSystemUIFont' and the like) cannot be requested by name, so there plain descriptions are used
        base = tkfont.nametofont("TkDefaultFont").actual()
        size = int(base["size"])                                    # negative: pixels
        if base["family"].startswith("."):
            self.font_bold, self.font_big = ("TkDefaultFont", 10, "bold"), ("TkDefaultFont", 11, "bold")
        else:
            self.font_bold = tkfont.Font(self.root, family=base["family"], size=size, weight="bold")
            self.font_big = tkfont.Font(self.root, family=base["family"], size=size + (1 if size > 0 else -1),
                                        weight="bold")
        fixed = tkfont.nametofont("TkFixedFont").actual()["family"]
        self.font_log = ("TkFixedFont", 9) if fixed.startswith(".") else tkfont.Font(self.root, family=fixed, size=9)
        st.configure("Run.TButton", font=self.font_big, padding=(10, 6), foreground="white", background="#1f6f5c")
        st.map("Run.TButton", background=[("active", "#25856e"), ("disabled", "#9db8b0")])
        st.configure("Hint.TLabel", foreground="#555555")
        st.configure("Treeview", rowheight=22)
        st.configure("TLabelframe.Label", font=self.font_bold)

    def _menu(self):
        mb = tk.Menu(self.root)
        fm = tk.Menu(mb, tearoff=0)
        fm.add_command(label="Open project...", command=self.open_project)
        fm.add_command(label="Save project...", command=self.save_project)
        fm.add_separator()
        fm.add_command(label="Export RSS dataset (CSV)...", command=self.export_dataset)
        fm.add_command(label="Export results per fix (CSV)...", command=self.export_results)
        fm.add_command(label="Export statistics (CSV)...", command=self.export_stats)
        fm.add_command(label="Save current figure...", command=self.save_figure)
        fm.add_separator()
        fm.add_command(label="Quit", command=self.root.destroy)
        mb.add_cascade(label="File", menu=fm)
        hm = tk.Menu(mb, tearoff=0)
        hm.add_command(label="Quick guide", command=self.show_help)
        hm.add_command(label="User guide (online)", command=self.open_guide)
        hm.add_command(label="All guides (online)", command=lambda: self.open_guide(DOCS_URL))
        hm.add_separator()
        hm.add_command(label="About RSSILab", command=self.show_about)
        mb.add_cascade(label="Help", menu=hm)
        self.root.config(menu=mb)

    # ---------------------------------------------------------------- widgets
    def _var(self, key, value, kind="str"):
        var = {"str": tk.StringVar, "bool": tk.BooleanVar}[kind](value=value)
        self.v[key] = var
        return var

    def _entry(self, parent, r, label, key, value, unit="", width=9, col=0):
        ttk.Label(parent, text=label).grid(row=r, column=col, sticky="w", padx=(2, 4), pady=2)
        e = ttk.Entry(parent, textvariable=self._var(key, str(value)), width=width)
        e.grid(row=r, column=col + 1, sticky="ew", pady=2)
        if unit:
            ttk.Label(parent, text=unit, style="Hint.TLabel").grid(row=r, column=col + 2, sticky="w", padx=3)
        self._labels[key] = label
        return e

    def _check(self, parent, r, label, key, value, col=0, span=3):
        c = ttk.Checkbutton(parent, text=label, variable=self._var(key, bool(value), "bool"))
        c.grid(row=r, column=col, columnspan=span, sticky="w", pady=2)
        return c

    def _combo(self, parent, r, label, key, values, value, width=24, col=0, span=2):
        if label:
            ttk.Label(parent, text=label).grid(row=r, column=col, sticky="w", padx=(2, 4), pady=2)
            col += 1
        cb = ttk.Combobox(parent, textvariable=self._var(key, value), values=list(values),
                          state="readonly", width=width)
        cb.grid(row=r, column=col, columnspan=span, sticky="ew", pady=2)
        return cb

    def F(self, key) -> float:
        try:
            return float(str(self.v[key].get()).replace(",", "."))
        except ValueError:
            raise ValueError(f"'{self._labels.get(key, key)}' must be a number.")

    def I(self, key) -> int:
        x = self.F(key)
        if x != int(x):
            raise ValueError(f"'{self._labels.get(key, key)}' must be an integer.")
        return int(x)

    def B(self, key) -> bool:
        return bool(self.v[key].get())

    def S(self, key) -> str:
        return str(self.v[key].get())

    def _set(self, key, value):
        if key in self.v:
            self.v[key].set(value if isinstance(self.v[key], tk.BooleanVar) else str(value))

    # ----------------------------------------------------------------- layout
    def _layout(self):
        # bars with a fixed height are packed first, so a small window shrinks the plots, not them
        sb = ttk.Frame(self.root, padding=(6, 2))
        sb.pack(fill="x", side="bottom")
        self.status = ttk.Label(sb, text=TOOL_HINTS["select"])
        self.status.pack(side="left")
        self.progress = ttk.Progressbar(sb, length=220, mode="determinate", maximum=1.0)
        self.progress.pack(side="right")
        self.lbl_xy = ttk.Label(sb, text="", width=24)
        self.lbl_xy.pack(side="right", padx=8)

        pw = ttk.PanedWindow(self.root, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, width=390)
        right = ttk.Frame(pw)
        pw.add(left, weight=0)
        pw.add(right, weight=1)

        run_bar = ttk.Frame(left, padding=6)
        run_bar.pack(side="bottom", fill="x")
        self.btn_run = ttk.Button(run_bar, text="Run simulation", style="Run.TButton", command=self.run_simulation)
        self.btn_run.pack(side="left", fill="x", expand=True)
        self.btn_stop = ttk.Button(run_bar, text="Stop", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(6, 0))

        nb = ttk.Notebook(left)
        nb.pack(fill="both", expand=True)
        for title, builder in (("Scenario", self._tab_scenario), ("Radio", self._tab_radio),
                               ("Experiment", self._tab_traj), ("Algorithms", self._tab_loc),
                               ("Analysis", self._tab_analysis)):
            sf = ScrollFrame(nb)
            nb.add(sf, text=title)
            builder(sf.inner)
            sf.inner.columnconfigure(0, weight=1)

        # tools
        tb = ttk.Frame(right, padding=(6, 4))
        tb.pack(fill="x")
        self._var("tool", "select")
        ttk.Label(tb, text="Tool").pack(side="left", padx=(0, 4))
        for text, val in TOOLS:
            ttk.Radiobutton(tb, text=text, value=val, variable=self.v["tool"],
                            command=self._tool_changed).pack(side="left", padx=2)
        tb = ttk.Frame(right, padding=(6, 0, 6, 4))      # second row: drawing properties
        tb.pack(fill="x")
        ttk.Label(tb, text="Draw with").pack(side="left", padx=(0, 4))
        ttk.Label(tb, text="Material").pack(side="left")
        ttk.Combobox(tb, textvariable=self._var("draw_mat", "Brick"), values=list(MATERIAL_TABLE),
                     state="readonly", width=10).pack(side="left", padx=3)
        ttk.Label(tb, text="Wall thickness").pack(side="left", padx=(6, 0))
        ttk.Entry(tb, textvariable=self._var("draw_thick", "0.2"), width=5).pack(side="left", padx=3)
        self._labels["draw_thick"] = "Wall thickness"
        ttk.Label(tb, text="Block top z (empty = ceiling)").pack(side="left", padx=(6, 0))
        ttk.Entry(tb, textvariable=self._var("draw_bz", ""), width=5).pack(side="left", padx=3)
        self._labels["draw_bz"] = "Block top z"
        ttk.Checkbutton(tb, text="Snap", variable=self._var("snap", True, "bool")).pack(side="left", padx=(8, 0))
        ttk.Entry(tb, textvariable=self._var("snap_step", "0.5"), width=4).pack(side="left", padx=3)
        self._labels["snap_step"] = "Snap step"
        ttk.Label(tb, text="m", style="Hint.TLabel").pack(side="left")

        vp = ttk.PanedWindow(right, orient="vertical")
        vp.pack(fill="both", expand=True)
        self.plots = ttk.Notebook(vp)
        vp.add(self.plots, weight=4)

        # map tab
        fmap = ttk.Frame(self.plots)
        self.plots.add(fmap, text="Map")
        self.fig_map = Figure(figsize=(9, 6), dpi=100)
        self.ax_map = self.fig_map.add_axes([0.07, 0.09, 0.88, 0.82])
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=fmap)
        self.nav = NavigationToolbar2Tk(self.canvas_map, fmap, pack_toolbar=False)
        self.nav.update()
        tl = ttk.Frame(fmap, padding=(4, 2))
        self.btn_play = ttk.Button(tl, text="Play", width=7, command=self._toggle_play, state="disabled")
        self.btn_play.pack(side="left")
        self.time_scale = ttk.Scale(tl, from_=0, to=1, orient="horizontal", command=self._on_time)
        self.time_scale.pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_time = ttk.Label(tl, text="No results yet", width=26)
        self.lbl_time.pack(side="left")
        tl.pack(side="bottom", fill="x")
        self.nav.pack(side="bottom", fill="x")
        self.canvas_map.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_map.mpl_connect("button_press_event", self._on_press)
        self.canvas_map.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas_map.mpl_connect("button_release_event", self._on_release)

        self.fig_cdf, self.canvas_cdf = self._plot_tab("Error CDF")
        self.fig_time, self.canvas_time = self._plot_tab("Error detail")
        self.fig_3d, self.canvas_3d = self._plot_tab("3D view")
        self.plots.bind("<<NotebookTabChanged>>", self._on_plot_tab)

        # results tables: overall summary, per point (positioning) / per trial (tracking)
        self.res_nb = ttk.Notebook(vp)
        vp.add(self.res_nb, weight=1)
        rf = ttk.Frame(self.res_nb, padding=(2, 2))
        self.res_nb.add(rf, text="Summary")
        self.tree = self._table(rf, summary_cols(2), height=7)
        self.tree.bind("<ButtonRelease-1>", self._toggle_algo)
        df = ttk.Frame(self.res_nb, padding=(2, 2))
        self.res_nb.add(df, text="Per point")
        top = ttk.Frame(df)
        top.pack(fill="x", pady=(0, 2))
        ttk.Label(top, text="Method").pack(side="left")
        self.cb_detail = ttk.Combobox(top, textvariable=self._var("d_algo", ""), state="readonly", width=30)
        self.cb_detail.pack(side="left", padx=4)
        self.cb_detail.bind("<<ComboboxSelected>>", lambda e: (self.fill_detail(), self.draw_detail()))
        self.lbl_detail = ttk.Label(top, text="", style="Hint.TLabel")
        self.lbl_detail.pack(side="left", padx=8)
        self.tree_detail = self._table(df, POINT_COLS, height=6)
        self.tree_detail.tag_configure("total", foreground="#1f4e9c")

    def _table(self, parent, cols, height=6):
        fr = ttk.Frame(parent)
        fr.pack(fill="both", expand=True)
        tree = ttk.Treeview(fr, show="headings", height=height)
        xs = ttk.Scrollbar(fr, orient="horizontal", command=tree.xview)
        ys = ttk.Scrollbar(fr, orient="vertical", command=tree.yview)
        tree.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        fr.rowconfigure(0, weight=1)
        fr.columnconfigure(0, weight=1)
        self._set_columns(tree, cols)
        return tree

    @staticmethod
    def _set_columns(tree, cols):
        tree.delete(*tree.get_children())
        tree.configure(columns=[c for c, _, _ in cols])
        for c, h, w in cols:
            tree.heading(c, text=h)
            tree.column(c, width=w, minwidth=w if c == "algo" else 40, stretch=c == "algo",
                        anchor="w" if c == "algo" else "center")

    def _plot_tab(self, title):
        """Figures re-compute their margins at every draw (tab switch, resize, HiDPI)."""
        fr = ttk.Frame(self.plots)
        self.plots.add(fr, text=title)
        fig = Figure(figsize=(9, 6), dpi=100, layout="constrained")
        cv = FigureCanvasTkAgg(fig, master=fr)
        nav = NavigationToolbar2Tk(cv, fr, pack_toolbar=False)
        nav.update()
        nav.pack(side="bottom", fill="x")
        cv.get_tk_widget().pack(fill="both", expand=True)
        return fig, cv

    # --------------------------------------------------------------- tab: scn
    def _tab_scenario(self, p):
        f = ttk.LabelFrame(p, text="Start from", padding=6)
        f.grid(row=0, column=0, sticky="ew", pady=4)
        self._combo(f, 0, None, "s_preset", presets.SCENARIO_PRESETS, list(presets.SCENARIO_PRESETS)[0],
                    width=30, span=2)
        ttk.Button(f, text="Load scenario", command=self.load_preset).grid(row=0, column=2, padx=4)
        ttk.Label(f, text="Grid-world layout").grid(row=1, column=0, sticky="w", pady=(6, 2))
        self._var("s_level", "4")
        self._labels["s_level"] = "Grid-world layout"
        sp = ttk.Spinbox(f, from_=1, to=len(LAYOUT_NAMES), textvariable=self.v["s_level"], width=5,
                         command=self._level_name)
        sp.grid(row=1, column=1, sticky="w", pady=(6, 2))
        sp.bind("<KeyRelease>", lambda e: self._level_name())
        ttk.Button(f, text="Import", command=self.import_level).grid(row=1, column=2, padx=4, pady=(6, 2))
        self.lbl_level = ttk.Label(f, text="", style="Hint.TLabel")
        self.lbl_level.grid(row=2, column=0, columnspan=3, sticky="w")
        self._check(f, 3, "Fit the grid to the target area (else whole room)", "s_onto", True)
        ttk.Label(f, text="Imports a 20 x 20 obstacle map (blocks of the drawing material) and its "
                          "start / goal as tracking waypoints.", style="Hint.TLabel",
                  wraplength=330).grid(row=4, column=0, columnspan=3, sticky="w")
        self._level_name()

        g = ttk.LabelFrame(p, text="Geometry (room centred at 0, 0)", padding=6)
        g.grid(row=1, column=0, sticky="ew", pady=4)
        self._entry(g, 0, "Room width", "g_rw", 90, "m")
        self._entry(g, 1, "Room height", "g_rh", 60, "m")
        self._entry(g, 2, "Target area width", "g_aw", 60, "m")
        self._entry(g, 3, "Target area height", "g_ah", 30, "m")
        self._entry(g, 4, "Target area centre x", "g_acx", 0, "m")
        self._entry(g, 5, "Target area centre y", "g_acy", 0, "m")
        self._entry(g, 6, "Ceiling height", "g_rz", 3.0, "m")
        self._entry(g, 7, "Default anchor height", "g_hanc", 0, "m")
        self._entry(g, 8, "Tag height", "g_htag", 0, "m")
        ttk.Label(g, text="2D: known height of the target. 3D: height of new test points and waypoints.",
                  style="Hint.TLabel", wraplength=330).grid(row=9, column=0, columnspan=3, sticky="w")
        dfr = ttk.Frame(g)
        dfr.grid(row=10, column=0, columnspan=3, sticky="w", pady=(6, 2))
        ttk.Label(dfr, text="Estimate").pack(side="left")
        self._var("g_dim", "2")
        ttk.Radiobutton(dfr, text="2D (x, y)", value="2", variable=self.v["g_dim"],
                        command=self._dim_changed).pack(side="left", padx=6)
        ttk.Radiobutton(dfr, text="3D (x, y, z)", value="3", variable=self.v["g_dim"],
                        command=self._dim_changed).pack(side="left")
        self.ent_z0 = self._entry(g, 11, "3D: target area lowest z", "g_az0", 0.0, "m")
        self.ent_z1 = self._entry(g, 12, "3D: target area highest z", "g_az1", 2.5, "m")
        ttk.Button(g, text="Apply geometry", command=lambda: self._guard(self.apply_geometry)).grid(
            row=13, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        a = ttk.LabelFrame(p, text="Anchors", padding=6)
        a.grid(row=2, column=0, sticky="ew", pady=4)
        self._entry(a, 0, "Number on perimeter", "g_nanc", 12, width=6)
        ttk.Button(a, text="Place", command=self.place_perimeter).grid(row=0, column=2, padx=4)
        self.tree_anc = ttk.Treeview(a, columns=("name", "x", "y", "z", "on"), show="headings", height=6)
        for c, w in (("name", 52), ("x", 66), ("y", 66), ("z", 52), ("on", 42)):
            self.tree_anc.heading(c, text=c.capitalize() if c != "on" else "On")
            self.tree_anc.column(c, width=w, anchor="center")
        self.tree_anc.grid(row=1, column=0, columnspan=3, sticky="ew", pady=4)
        self.tree_anc.bind("<<TreeviewSelect>>", self._anchor_selected)
        self._entry(a, 2, "x", "a_x", "", "m", width=8)
        self._entry(a, 3, "y", "a_y", "", "m", width=8)
        self._entry(a, 4, "z", "a_z", "", "m", width=8)
        bf = ttk.Frame(a)
        bf.grid(row=5, column=0, columnspan=3, sticky="ew")
        for t, cmd in (("Update", self.anchor_update), ("On/off", self.anchor_toggle),
                       ("Delete", self.anchor_delete), ("Clear all", self.anchor_clear)):
            ttk.Button(bf, text=t, command=cmd, width=8).pack(side="left", padx=1)
        self._entry(a, 6, "Alternate height", "g_halt", 0.5, "m", width=8)
        bh = ttk.Frame(a)
        bh.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        ttk.Button(bh, text="All at default height", command=self.anchors_same_height).pack(side="left", padx=1)
        ttk.Button(bh, text="Alternate heights", command=self.anchors_alternate).pack(side="left", padx=1)
        ttk.Label(a, text="3D needs anchors at different heights, otherwise z is barely observable.",
                  style="Hint.TLabel", wraplength=330).grid(row=8, column=0, columnspan=3, sticky="w")

        o = ttk.LabelFrame(p, text="Obstacles", padding=6)
        o.grid(row=3, column=0, sticky="ew", pady=4)
        self.tree_obs = ttk.Treeview(o, columns=("i", "kind", "mat", "db", "h"), show="headings", height=6)
        for c, h, w in (("i", "#", 30), ("kind", "Type", 46), ("mat", "Material", 78), ("db", "dB @ f", 58),
                        ("h", "z range", 72)):
            self.tree_obs.heading(c, text=h)
            self.tree_obs.column(c, width=w, anchor="center")
        self.tree_obs.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.tree_obs.bind("<<TreeviewSelect>>", self._obstacle_selected)
        self._combo(o, 1, "Material", "o_mat", MATERIAL_TABLE, "Brick", width=12, span=1)
        self._entry(o, 2, "Loss override", "o_loss", "", "dB (empty = table)", width=8)
        self._entry(o, 3, "Bottom z", "o_z0", "0", "m", width=8)
        self._entry(o, 4, "Top z", "o_z1", "", "m (empty = ceiling)", width=8)
        bo = ttk.Frame(o)
        bo.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        for t, cmd in (("Apply to selected", self.obstacle_apply), ("Delete", self.obstacle_delete),
                       ("Clear all", self.obstacle_clear)):
            ttk.Button(bo, text=t, command=cmd).pack(side="left", padx=1)
        ttk.Label(p, text="Material losses per crossing come from MATERIAL_TABLE in rssilab/model.py "
                          "(interpolated at the radio frequency).", style="Hint.TLabel",
                  wraplength=340).grid(row=4, column=0, sticky="w", pady=4)

    # ------------------------------------------------------------- tab: radio
    def _tab_radio(self, p):
        f = ttk.LabelFrame(p, text="Technology preset", padding=6)
        f.grid(row=0, column=0, sticky="ew", pady=4)
        self._combo(f, 0, None, "t_preset", PRESETS, LEGACY_PRESET, width=30, span=2)
        ttk.Button(f, text="Load preset", command=lambda: self._load_tech(presets.technology(self.S("t_preset")))
                   ).grid(row=0, column=2, padx=4)
        ttk.Label(f, text="Presets are indicative starting points: calibrate them with measurements.",
                  style="Hint.TLabel", wraplength=330).grid(row=1, column=0, columnspan=3, sticky="w")

        g = ttk.LabelFrame(p, text="Link budget and path loss", padding=6)
        g.grid(row=1, column=0, sticky="ew", pady=4)
        self._entry(g, 0, "Frequency", "t_freq", 2.4, "GHz")
        self._entry(g, 1, "TX power", "t_ptx", 0, "dBm")
        self._entry(g, 2, "Antenna gains - losses", "t_gains", 0, "dB")
        self._check(g, 3, "P0 from free-space loss at d0", "t_p0auto", False)
        self.ent_p0 = self._entry(g, 4, "P0 (RSS at d0)", "t_p0", -40, "dBm")
        self._entry(g, 5, "Reference distance d0", "t_d0", 1, "m")
        self._entry(g, 6, "Path-loss exponent n", "t_n", 4)

        s = ttk.LabelFrame(p, text="Random components", padding=6)
        s.grid(row=2, column=0, sticky="ew", pady=4)
        self._entry(s, 0, "Shadowing std", "t_sigma", 3, "dB")
        self._entry(s, 1, "Shadowing correlation", "t_corr", 0, "m (0 = i.i.d.)")
        self._combo(s, 2, "Fading", "t_fading", FADING_MODELS, "Rayleigh", width=18, span=2)
        self._entry(s, 3, "Rician K", "t_k", 6, "dB")
        self._entry(s, 4, "Legacy fading sigma", "t_legsig", 2)

        r = ttk.LabelFrame(p, text="Receiver", padding=6)
        r.grid(row=3, column=0, sticky="ew", pady=4)
        self._entry(r, 0, "Sensitivity", "t_sens", -200, "dBm (\u2264 -150 = off)")
        self._entry(r, 1, "RSSI quantisation", "t_quant", 0, "dB (0 = off)")
        self._entry(r, 2, "Sample rate", "t_rate", 10, "Hz")
        self._entry(r, 3, "Samples per fix", "t_ns", 1)

        self.lbl_tech = ttk.Label(p, text="", style="Hint.TLabel", wraplength=340, justify="left")
        self.lbl_tech.grid(row=4, column=0, sticky="w", pady=6)
        for k in ("t_freq", "t_ptx", "t_gains", "t_p0auto", "t_p0", "t_d0", "t_n", "t_sigma", "t_corr",
                  "t_fading", "t_k", "t_legsig", "t_sens", "t_ns", "t_rate"):
            self.v[k].trace_add("write", lambda *a: self._tech_info())

    # -------------------------------------------------------------- tab: traj
    def _tab_traj(self, p):
        f = ttk.LabelFrame(p, text="Experiment type", padding=6)
        f.grid(row=0, column=0, sticky="ew", pady=4)
        self._var("tr_mode", "Positioning")
        ttk.Radiobutton(f, text="Positioning: static target on test points", value="Positioning",
                        variable=self.v["tr_mode"], command=self._mode_changed).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(f, text="Tracking: target moving along a route", value="Tracking",
                        variable=self.v["tr_mode"], command=self._mode_changed).grid(row=1, column=0, sticky="w")

        self.frm_pos = ttk.LabelFrame(p, text="Positioning", padding=6)
        self.frm_pos.grid(row=1, column=0, sticky="ew", pady=4)
        g = self.frm_pos
        self._combo(g, 0, "Test points", "tr_src", POINT_SOURCES, POINT_SOURCES[0], width=16)
        self._entry(g, 1, "Random points: how many", "tr_nrand", 10)
        self._entry(g, 2, "Grid spacing", "tr_grid", 2, "m")
        self._entry(g, 3, "Fixes per point and trial", "tr_K", 10)
        self.ent_gridz = self._entry(g, 4, "3D: grid height", "tr_gridz", "", "m (empty = tag height)")
        bp = ttk.Frame(g)
        bp.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(bp, text="Preview points", command=self.preview_route).pack(side="left", padx=1)
        ttk.Button(bp, text="Random points to map", command=self.random_to_map).pack(side="left", padx=1)
        ttk.Button(bp, text="Clear points", command=self.clear_test_points).pack(side="left", padx=1)
        ttk.Label(g, text="Points on map: Test point tool, click on the map. Random points are drawn "
                          "once from the seed. Each fix averages 'Samples per fix' readings.",
                  style="Hint.TLabel", wraplength=330, justify="left").grid(row=6, column=0, columnspan=3,
                                                                            sticky="w", pady=(4, 0))

        self.frm_trk = ttk.LabelFrame(p, text="Tracking", padding=6)
        self.frm_trk.grid(row=2, column=0, sticky="ew", pady=4)
        t = self.frm_trk
        self._combo(t, 0, "Route", "tr_kind", TRAJ_TYPES, TRAJ_TYPES[0], width=22)
        self._entry(t, 1, "Speed", "tr_speed", 1.0, "m/s")
        self._check(t, 2, "Route around obstacles (A* planner)", "tr_astar", True)
        self._entry(t, 3, "A* grid resolution", "tr_res", 0.5, "m")
        self._entry(t, 4, "Clearance from obstacles", "tr_clear", 0.3, "m")
        self._entry(t, 5, "Random walk: waypoints", "tr_nwp", 6)
        self._entry(t, 6, "Circle: radius fraction", "tr_cfrac", 0.8)
        self._entry(t, 7, "Circle: laps", "tr_laps", 1)
        self._entry(t, 8, "Lawnmower: row spacing", "tr_lawn", 5, "m")
        bt = ttk.Frame(t)
        bt.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(bt, text="Preview route", command=self.preview_route).pack(side="left", padx=1)
        ttk.Button(bt, text="Clear waypoints", command=self.clear_waypoints).pack(side="left", padx=1)
        ttk.Button(bt, text="Reverse", command=self.reverse_waypoints).pack(side="left", padx=1)

        m = ttk.LabelFrame(p, text="Trials (Monte Carlo)", padding=6)
        m.grid(row=3, column=0, sticky="ew", pady=4)
        self._entry(m, 0, "Trials", "l_runs", 1)
        self._entry(m, 1, "Seed", "l_seed", 1)
        self._check(m, 2, "New shadowing realisation at every trial", "l_redraw", True)
        ttk.Label(m, text="Positioning: every trial repeats all fixes on all points. Tracking: every "
                          "trial repeats the whole route with new noise.",
                  style="Hint.TLabel", wraplength=330, justify="left").grid(row=3, column=0, columnspan=3, sticky="w")

        q = ttk.LabelFrame(p, text="Points of this experiment (x, y, z)", padding=6)
        q.grid(row=4, column=0, sticky="ew", pady=4)
        self.tree_pts = ttk.Treeview(q, columns=("id", "x", "y", "z"), show="headings", height=5)
        for c, w in (("id", 50), ("x", 72), ("y", 72), ("z", 60)):
            self.tree_pts.heading(c, text=c if c != "id" else "#")
            self.tree_pts.column(c, width=w, anchor="center")
        self.tree_pts.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.tree_pts.bind("<<TreeviewSelect>>", self._point_selected)
        self._entry(q, 1, "x", "p_x", "", "m", width=8)
        self._entry(q, 2, "y", "p_y", "", "m", width=8)
        self._entry(q, 3, "z", "p_z", "", "m (used in 3D)", width=8)
        bq = ttk.Frame(q)
        bq.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(bq, text="Update", command=self.point_update).pack(side="left", padx=1)
        ttk.Button(bq, text="Delete", command=self.point_delete).pack(side="left", padx=1)
        ttk.Button(bq, text="All at tag height", command=self.points_at_tag_height).pack(side="left", padx=1)
        ttk.Label(q, text="Test points in Positioning, waypoints in Tracking. New points get the tag height.",
                  style="Hint.TLabel", wraplength=330).grid(row=5, column=0, columnspan=3, sticky="w")
        self._mode_changed(redraw=False)

    def _mode_changed(self, redraw=True):
        pos = self.S("tr_mode") == "Positioning"
        for frame, on in ((self.frm_pos, pos), (self.frm_trk, not pos)):
            for w in frame.winfo_children():
                for ww in [w] + list(w.winfo_children()):
                    try:
                        ww.state(["!disabled"] if on else ["disabled"])
                    except (AttributeError, tk.TclError):
                        pass
        if hasattr(self, "res_nb"):
            self.res_nb.tab(1, text="Per point" if pos else "Per trial")
        if hasattr(self, "ent_z0"):
            self._dim_changed(redraw=False)
        if hasattr(self, "tree_pts"):
            self._refresh_points()
        if redraw:
            self.route_preview = None
            self.points_preview = None
            if self.results is not None and self.results.mode != self.S("tr_mode").lower():
                self.results_visible = False
            self.draw_map(keep_view=True)

    # --------------------------------------------------------------- tab: loc
    def _tab_loc(self, p):
        f = ttk.LabelFrame(p, text="Positioning approach", padding=6)
        f.grid(row=0, column=0, sticky="ew", pady=4)
        self._var("l_family", BOTH)
        for i, (val, txt) in enumerate(((DISTANCE, "Distance-based: path-loss model (ranges, likelihood)"),
                                        (FINGERPRINTING, "Fingerprinting: models trained on a survey"),
                                        (BOTH, "Both, compared on the same RSS"))):
            ttk.Radiobutton(f, text=txt, value=val, variable=self.v["l_family"],
                            command=self._family_changed).grid(row=i, column=0, sticky="w")
        default_on = set(engine.DEFAULT_METHODS)

        d = self.frm_dist = ttk.LabelFrame(p, text="Distance-based methods", padding=6)
        d.grid(row=1, column=0, sticky="ew", pady=4)
        r = 0
        for name in methods(DISTANCE):
            self._check(d, r, name, f"alg::{name}", name in default_on)
            r += 1
        self._entry(d, r, "Weighted centroid: anchors k", "l_wclk", 4)
        self._entry(d, r + 1, "Weighted centroid: exponent g", "l_wclg", 1.0)
        self._entry(d, r + 2, "MLE grid resolution", "l_mlres", 0.5, "m")
        self._check(d, r + 3, "Map-aware (walls in the MLE grid and PF)", "l_map", True)
        ttk.Label(d, text="Distances and likelihoods come from the estimator model (calibration below). "
                          "Trilateration uses the 3 strongest non-aligned anchors (4 in 3D), LLS and WLS "
                          "the linearised equations of all received anchors, NLS fits the ranges, MLE the RSS.",
                  style="Hint.TLabel", wraplength=330, justify="left").grid(row=r + 4, column=0, columnspan=3,
                                                                            sticky="w", pady=(4, 0))

        fp = self.frm_fp = ttk.LabelFrame(p, text="Fingerprinting methods", padding=6)
        fp.grid(row=2, column=0, sticky="ew", pady=4)
        r = 0
        for name in methods(FINGERPRINTING):
            self._check(fp, r, name, f"alg::{name}", name in default_on)
            r += 1
        ttk.Label(fp, text="Training survey", style="TLabelframe.Label").grid(row=r, column=0, columnspan=3,
                                                                              sticky="w", pady=(6, 0))
        self._entry(fp, r + 1, "Survey grid spacing", "l_rps", 2.0, "m")
        self._entry(fp, r + 2, "Fingerprints per point", "l_rpn", 20)
        ttk.Button(fp, text="Show the survey grid on the map", command=self.preview_survey).grid(
            row=r + 3, column=0, columnspan=3, sticky="ew", pady=(2, 0))
        ttk.Label(fp, text="Models", style="TLabelframe.Label").grid(row=r + 4, column=0, columnspan=3,
                                                                     sticky="w", pady=(6, 0))
        self._entry(fp, r + 5, "k-NN: neighbours K", "l_knnk", 10)
        self._check(fp, r + 6, "k-NN: weights 1/distance (else uniform)", "l_knnw", True)
        self._entry(fp, r + 7, "WKNN: neighbours K", "l_wknnk", 4)
        self._entry(fp, r + 8, "Decision tree: min. per leaf", "l_dtleaf", 5)
        self._entry(fp, r + 9, "Random forest: trees", "l_rftrees", 100)
        self._entry(fp, r + 10, "Neural network: hidden layers", "l_mlp", "64, 64")
        ttk.Label(fp, text="The survey is simulated in this scenario with these radio settings: reference "
                           "points on a grid over the free target area (offset by half a spacing from its "
                           "corner), each fingerprint averaged like a fix. The models are trained "
                           "again for every new shadowing realisation. WKNN uses the mean fingerprint of each "
                           "reference point (radio map), the other models every fingerprint.",
                  style="Hint.TLabel", wraplength=330, justify="left").grid(row=r + 11, column=0, columnspan=3,
                                                                             sticky="w", pady=(4, 0))

        g = ttk.LabelFrame(p, text="All methods", padding=6)
        g.grid(row=3, column=0, sticky="ew", pady=4)
        self.ent_zres = self._entry(g, 0, "3D: vertical grid spacing", "l_zres", 0.5, "m")
        self._check(g, 1, "Clip estimates to the target area", "l_clip", True)
        ttk.Label(g, text="3D: height layers of the MLE grid, of the particle filter and of the survey.",
                  style="Hint.TLabel", wraplength=330).grid(row=2, column=0, columnspan=3, sticky="w")

        t = ttk.LabelFrame(p, text="Tracking", padding=6)
        t.grid(row=4, column=0, sticky="ew", pady=4)
        self._check(t, 0, "Kalman filter (constant velocity) on", "l_kf", True, span=1)
        self.cb_kfbase = self._combo(t, 1, "KF input", "l_kfbase", ALGORITHMS, engine.LocParams().kf_base, width=26)
        self._entry(t, 2, "KF acceleration std", "l_kfq", 0.5, "m/s²")
        self._entry(t, 3, "KF measurement std", "l_kfr", 0, "m (0 = auto)")
        self._pf_widgets = [self._check(t, 4, "Particle filter (path-loss likelihood)", "l_pf", False)]
        self._pf_widgets += [self._entry(t, 5, "PF particles", "l_pfn", 1000),
                             self._entry(t, 6, "PF acceleration std", "l_pfacc", 0.5, "m/s²"),
                             self._entry(t, 7, "PF position jitter", "l_pfjit", 0, "m (0 = auto)"),
                             self._entry(t, 8, "PF likelihood std", "l_pfsig", 0, "dB (0 = auto)"),
                             self._entry(t, 9, "PF random injection", "l_pfinj", 0.02, "fraction")]
        ttk.Label(t, text="The particle filter weighs particles with the path-loss model: it is a "
                          "distance-based tracker.", style="Hint.TLabel", wraplength=330,
                  justify="left").grid(row=10, column=0, columnspan=3, sticky="w")

        c = self.frm_cal = ttk.LabelFrame(p, text="Estimator calibration (distance-based)", padding=6)
        c.grid(row=5, column=0, sticky="ew", pady=4)
        self._var("l_calib", "Oracle")
        ttk.Radiobutton(c, text="Oracle (true channel, fading bias included)", value="Oracle",
                        variable=self.v["l_calib"]).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Radiobutton(c, text="Manual (model mismatch)", value="Manual",
                        variable=self.v["l_calib"]).grid(row=1, column=0, columnspan=3, sticky="w")
        self._entry(c, 2, "P0 estimate", "l_p0", -40, "dBm")
        self._entry(c, 3, "n estimate", "l_n", 4)
        self._entry(c, 4, "RSS std estimate", "l_sig", 3, "dB")
        ttk.Button(c, text="Copy oracle values", command=self.copy_oracle).grid(row=5, column=0, columnspan=3,
                                                                                sticky="ew", pady=(4, 0))
        self._family_changed()

    @staticmethod
    def _set_state(widget, on: bool) -> None:
        """Enables or disables a widget and all its children."""
        for w in widget.winfo_children():
            try:
                w.state(["!disabled"] if on else ["disabled"])
            except (AttributeError, tk.TclError):
                pass
            RSSILabApp._set_state(w, on)

    def _allowed_methods(self):
        """Methods of the chosen positioning approach."""
        fam = self.S("l_family")
        return [n for n in ALGORITHMS if fam == BOTH or family_of(n) == fam]

    def _family_changed(self):
        fam = self.S("l_family")
        use_d, use_f = fam in (DISTANCE, BOTH), fam in (FINGERPRINTING, BOTH)
        self._set_state(self.frm_dist, use_d)
        self._set_state(self.frm_fp, use_f)
        self._set_state(self.frm_cal, use_d)
        for w in self._pf_widgets:
            w.state(["!disabled"] if use_d else ["disabled"])
        allowed = self._allowed_methods()
        self.cb_kfbase.configure(values=allowed)
        if allowed and self.S("l_kfbase") not in allowed:
            default = engine.LocParams().kf_base
            self._set("l_kfbase", default if default in allowed else allowed[0])

    # ---------------------------------------------------------- tab: analysis
    def _tab_analysis(self, p):
        x = ttk.LabelFrame(p, text="Export RSS dataset (radio map, CSV)", padding=6)
        x.grid(row=0, column=0, sticky="ew", pady=4)
        self._entry(x, 0, "Grid spacing", "x_spacing", 1.0, "m")
        self._entry(x, 1, "Samples per point", "x_samples", 100)
        self._entry(x, 2, "Seed", "x_seed", 1)
        self._check(x, 3, "Skip grid points inside obstacles", "x_skip", False)
        ttk.Button(x, text="Export dataset CSV...", command=self.export_dataset).grid(
            row=4, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Label(x, text="Columns Coord_X, Coord_Y (Coord_Z in 3D), Anc1_RSS... on a grid over the target "
                          "area at the tag height, with walls, materials and the radio settings above. "
                          "Missing readings are written as nan.",
                  style="Hint.TLabel", wraplength=330, justify="left").grid(row=5, column=0, columnspan=3, sticky="w")

        e = ttk.LabelFrame(p, text="Results", padding=6)
        e.grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(e, text="Export statistics CSV...", command=self.export_stats).pack(fill="x", pady=1)
        ttk.Button(e, text="Export results per fix CSV...", command=self.export_results).pack(fill="x", pady=1)
        ttk.Button(e, text="Save current figure...", command=self.save_figure).pack(fill="x", pady=1)

        lg = ttk.LabelFrame(p, text="Log", padding=6)
        lg.grid(row=2, column=0, sticky="nsew", pady=4)
        self.txt_log = tk.Text(lg, height=14, width=44, wrap="word", font=self.font_log,
                               state="disabled", relief="flat", background="#f6f6f4")
        self.txt_log.pack(fill="both", expand=True)

    # ------------------------------------------------------------ small utils
    def _log(self, msg):
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", time.strftime("%H:%M:%S ") + msg + "\n")
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")

    def _hint(self, msg):
        self.status.configure(text=msg)

    def _level_name(self):
        try:
            lv = int(self.v["s_level"].get())
            self.lbl_level.configure(text=LAYOUT_NAMES.get(lv, f"use 1-{len(LAYOUT_NAMES)}"))
        except (ValueError, tk.TclError):
            self.lbl_level.configure(text=f"use 1-{len(LAYOUT_NAMES)}")

    def _snap(self, x, y):
        if self.B("snap"):
            try:
                s = max(self.F("snap_step"), 1e-3)
                return round(x / s) * s, round(y / s) * s
            except ValueError:
                pass
        return x, y

    def _invalidate(self):
        self.results_visible = False
        self.route_preview = None
        self.points_preview = None
        self.survey_preview = None
        self._stop_play()

    # ------------------------------------------------------------- scenario I/O
    def _scn_to_fields(self):
        s = self.scn
        for k, val in (("g_rw", s.room_w), ("g_rh", s.room_h), ("g_aw", s.area_w), ("g_ah", s.area_h),
                       ("g_acx", s.area_cx), ("g_acy", s.area_cy), ("g_hanc", s.anchor_h),
                       ("g_htag", s.tag_h), ("g_nanc", len(s.anchors) or 12), ("g_rz", s.room_z),
                       ("g_az0", s.area_z0), ("g_az1", s.area_z1)):
            self._set(k, f"{val:g}")
        self._set("g_dim", str(s.dim))
        self._dim_changed(redraw=False)

    def apply_geometry(self, silent=False):
        """Copies the geometry fields into the scenario (all checked first: on error the
        scenario is left unchanged and ValueError explains why)."""
        s = self.scn
        rw, rh, aw, ah, acx, acy, hanc, htag, rz, az0, az1 = [self.F(k) for k in (
            "g_rw", "g_rh", "g_aw", "g_ah", "g_acx", "g_acy", "g_hanc", "g_htag", "g_rz", "g_az0", "g_az1")]
        if min(rw, rh, aw, ah, rz) <= 0:
            raise ValueError("Room, target area and ceiling sizes must be positive.")
        if not (0 <= az0 <= az1 <= rz):
            raise ValueError("3D target area: 0 <= lowest z <= highest z <= ceiling height.")
        if not (0 <= htag <= rz and 0 <= hanc <= rz):
            raise ValueError("Tag and anchor heights must be between the floor (0) and the ceiling.")
        if (abs(acx) + aw / 2 > rw / 2 + 1e-9) or (abs(acy) + ah / 2 > rh / 2 + 1e-9):
            raise ValueError("The target area must lie inside the room.")
        (s.room_w, s.room_h, s.area_w, s.area_h, s.area_cx, s.area_cy, s.anchor_h, s.tag_h,
         s.room_z, s.area_z0, s.area_z1) = rw, rh, aw, ah, acx, acy, hanc, htag, rz, az0, az1
        s.dim = int(self.S("g_dim"))
        if not silent:
            self._invalidate()
            self.draw_map()

    def _guard(self, fn, *a):
        try:
            return fn(*a)
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))

    def load_preset(self):
        self.scn = presets.SCENARIO_PRESETS[self.S("s_preset")]()
        self.sel = None
        self._scn_to_fields()
        self._invalidate()
        self._refresh_lists()
        self.draw_map()
        self._log(f"Loaded scenario '{self.S('s_preset')}'.")

    def import_level(self):
        try:
            self.apply_geometry(silent=True)
            msg = import_grid_layout(self.scn, self.I("s_level"), self.S("draw_mat"),
                                     onto_area=self.B("s_onto"))
        except Exception as e:
            messagebox.showerror("Import failed", str(e))
            return
        self._set("tr_kind", TRAJ_TYPES[0])
        self._set("tr_mode", "Tracking")
        self._mode_changed(redraw=False)
        self._invalidate()
        self._refresh_lists()
        self.draw_map()
        self._log(msg)

    def place_perimeter(self):
        try:
            self.apply_geometry(silent=True)
            n = self.I("g_nanc")
            if n < 3:
                raise ValueError("Use at least 3 anchors.")
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        self.scn.place_anchors_on_perimeter(n)
        self._invalidate()
        self._refresh_lists()
        self.draw_map()

    # -------------------------------------------------------- anchors / obstacles
    def _refresh_lists(self):
        self.tree_anc.delete(*self.tree_anc.get_children())
        for i, a in enumerate(self.scn.anchors):
            self.tree_anc.insert("", "end", iid=str(i), values=(
                a.name, f"{a.x:.2f}", f"{a.y:.2f}", f"{a.z:.2f}", "yes" if a.enabled else "no"))
        self.tree_obs.delete(*self.tree_obs.get_children())
        try:
            f = self.F("t_freq")
        except (ValueError, KeyError):
            f = 2.4
        for i, o in enumerate(self.scn.obstacles):
            zr = f"{o.z0:g}-" + ("ceil" if o.z1 is None else f"{o.z1:g}")
            loss = f"{o.loss(f):.1f}" + ("*" if o.loss_db is not None else "")
            self.tree_obs.insert("", "end", iid=str(i), values=(i + 1, o.kind, o.material, loss, zr))
        if hasattr(self, "tree_pts"):
            self._refresh_points()

    def _anchor_selected(self, _=None):
        s = self.tree_anc.selection()
        if s:
            i = int(s[0])
            a = self.scn.anchors[i]
            self._set("a_x", f"{a.x:g}")
            self._set("a_y", f"{a.y:g}")
            self._set("a_z", f"{a.z:g}")
            if self.sel != ("anchor", i):
                self.sel = ("anchor", i)
                self.draw_map(keep_view=True)

    def _obstacle_selected(self, _=None):
        s = self.tree_obs.selection()
        if s:
            i = int(s[0])
            o = self.scn.obstacles[i]
            self._set("o_mat", o.material)
            self._set("o_loss", "" if o.loss_db is None else f"{o.loss_db:g}")
            self._set("o_z0", f"{o.z0:g}")
            self._set("o_z1", "" if o.z1 is None else f"{o.z1:g}")
            if self.sel != ("obstacle", i):
                self.sel = ("obstacle", i)
                self.draw_map(keep_view=True)

    def _sel_index(self, tree):
        s = tree.selection()
        return int(s[0]) if s else None

    def anchor_update(self):
        i = self._sel_index(self.tree_anc)
        if i is None:
            return
        try:
            self.scn.anchors[i].x, self.scn.anchors[i].y = self.F("a_x"), self.F("a_y")
            self.scn.anchors[i].z = self.F("a_z")
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        self._after_edit(select=("anchor", i))

    def anchor_toggle(self):
        i = self._sel_index(self.tree_anc)
        if i is not None:
            self.scn.anchors[i].enabled = not self.scn.anchors[i].enabled
            self._after_edit(select=("anchor", i))

    def anchor_delete(self):
        i = self._sel_index(self.tree_anc)
        if i is not None:
            del self.scn.anchors[i]
            self.scn.rename_anchors()
            self._after_edit()

    def anchor_clear(self):
        self.scn.anchors = []
        self._after_edit()

    def anchors_same_height(self):
        try:
            h = self.F("g_hanc")
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        for a in self.scn.anchors:
            a.z = h
        self._after_edit()

    def anchors_alternate(self):
        """Odd anchors at the default height, even ones at the alternate height."""
        try:
            h1, h2 = self.F("g_hanc"), self.F("g_halt")
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        for i, a in enumerate(self.scn.anchors):
            a.z = h1 if i % 2 == 0 else h2
        self._after_edit()
        self._log(f"Anchor heights alternate between {h1:g} m and {h2:g} m.")

    def _dim_changed(self, redraw=True):
        three = self.S("g_dim") == "3"
        for e in (self.ent_z0, self.ent_z1):
            e.state(["!disabled"] if three else ["disabled"])
        for k in ("ent_gridz", "ent_zres"):
            if hasattr(self, k):
                getattr(self, k).state(["!disabled"] if three else ["disabled"])
        if redraw:
            self.scn.dim = 3 if three else 2
            self._invalidate()
            self.draw_map(keep_view=True)
            self._log(f"Estimation in {'3D (x, y, z)' if three else '2D (x, y), known tag height'}.")

    def obstacle_apply(self):
        i = self._sel_index(self.tree_obs)
        if i is None:
            return
        o = self.scn.obstacles[i]
        o.material = self.S("o_mat")
        txt = self.S("o_loss").strip()
        top = self.S("o_z1").strip()
        try:
            loss = float(txt.replace(",", ".")) if txt else None
            z0 = self.F("o_z0")
            z1 = float(top.replace(",", ".")) if top else None
        except ValueError:
            messagebox.showerror("Check the parameters", "Loss override and top z must be numbers or empty, "
                                                         "bottom z a number.")
            return
        if z0 < 0 or (z1 is not None and z1 <= z0) or (loss is not None and loss < 0):
            messagebox.showerror("Check the parameters", "Use 0 <= bottom z < top z, and a loss override >= 0 dB.")
            return
        o.loss_db, o.z0, o.z1 = loss, z0, z1
        self._after_edit(select=("obstacle", i))

    def obstacle_delete(self):
        i = self._sel_index(self.tree_obs)
        if i is not None:
            del self.scn.obstacles[i]
            self._after_edit()

    def obstacle_clear(self):
        self.scn.obstacles = []
        self._after_edit()

    def _delete_selected(self):
        focus = self.root.focus_get()
        if isinstance(focus, (ttk.Entry, tk.Entry, ttk.Spinbox, tk.Text)):
            return                              # the key is editing a text field
        for tree, delete in ((self.tree_pts, self.point_delete), (self.tree_anc, self.anchor_delete),
                             (self.tree_obs, self.obstacle_delete)):
            if focus is tree:                   # a table has the focus: delete its selected row
                delete()
                return
        if self.sel and self.sel[0] == "anchor" and self.sel[1] < len(self.scn.anchors):
            del self.scn.anchors[self.sel[1]]
            self.scn.rename_anchors()
        elif self.sel and self.sel[0] == "obstacle" and self.sel[1] < len(self.scn.obstacles):
            del self.scn.obstacles[self.sel[1]]
        else:
            return
        self._after_edit()

    def _field_or(self, key, default):
        """Value of a numeric field, or `default` when the field is not a valid number."""
        try:
            return self.F(key)
        except ValueError:
            return default

    def _after_edit(self, select=None):
        self.sel = select
        self._invalidate()
        self._refresh_lists()
        if select:
            tree = self.tree_anc if select[0] == "anchor" else self.tree_obs
            if tree.exists(str(select[1])):
                tree.selection_set(str(select[1]))
        self.draw_map(keep_view=True)

    # -------------------------------------------------------------- radio I/O
    def _load_tech(self, t: Technology):
        if t.name in PRESETS:
            self._set("t_preset", t.name)
        for k, val in (("t_freq", t.freq_ghz), ("t_ptx", t.tx_power_dbm), ("t_gains", t.gains_db),
                       ("t_p0", t.p0_dbm), ("t_d0", t.d0_m), ("t_n", t.path_loss_exp),
                       ("t_sigma", t.shadow_sigma_db), ("t_corr", t.shadow_corr_m), ("t_k", t.rician_k_db),
                       ("t_legsig", t.legacy_fading_sigma), ("t_sens", t.sensitivity_dbm),
                       ("t_quant", t.quantization_db), ("t_rate", t.sample_rate_hz), ("t_ns", t.samples_per_fix)):
            self._set(k, f"{val:g}")
        self._set("t_fading", t.fading)
        self._set("t_p0auto", t.p0_auto)
        self._tech_name = t.name
        self._tech_info()
        if hasattr(self, "tree_obs"):
            self._refresh_lists()

    def read_tech(self) -> Technology:
        t = Technology(
            name=getattr(self, "_tech_name", "Custom"), freq_ghz=self.F("t_freq"),
            tx_power_dbm=self.F("t_ptx"), gains_db=self.F("t_gains"), p0_auto=self.B("t_p0auto"),
            p0_dbm=self.F("t_p0"), d0_m=self.F("t_d0"), path_loss_exp=self.F("t_n"),
            shadow_sigma_db=self.F("t_sigma"), shadow_corr_m=self.F("t_corr"), fading=self.S("t_fading"),
            rician_k_db=self.F("t_k"), legacy_fading_sigma=self.F("t_legsig"),
            sensitivity_dbm=self.F("t_sens"), quantization_db=self.F("t_quant"),
            sample_rate_hz=self.F("t_rate"), samples_per_fix=self.I("t_ns"))
        if t.freq_ghz <= 0 or t.d0_m <= 0 or t.path_loss_exp <= 0 or t.sample_rate_hz <= 0:
            raise ValueError("Frequency, d0, path-loss exponent and sample rate must be positive.")
        if t.samples_per_fix < 1 or t.shadow_sigma_db < 0 or t.shadow_corr_m < 0 or t.quantization_db < 0:
            raise ValueError("Samples per fix >= 1; shadowing, correlation and quantisation >= 0.")
        if t.legacy_fading_sigma < 0:
            raise ValueError("The legacy fading sigma must be >= 0.")
        return t

    def _tech_info(self):
        if getattr(self, "_in_info", False):
            return
        self._in_info = True
        try:
            self._tech_info_inner()
        finally:
            self._in_info = False

    def _tech_info_inner(self):
        try:
            t = self.read_tech()
        except (ValueError, KeyError, tk.TclError):
            self.lbl_tech.configure(text="")
            return
        self.ent_p0.configure(state="disabled" if t.p0_auto else "normal")
        if t.p0_auto and self.v["t_p0"].get() != f"{t.p0():.2f}":
            self.v["t_p0"].set(f"{t.p0():.2f}")
        fm, fs = fading_stats(t)
        rng = t.nominal_range_m()
        txt = (f"P0 = {t.p0():.1f} dBm at d0 = {t.d0_m:g} m\n"
               f"Fading per reading: mean {fm:+.2f} dB, std {fs:.2f} dB\n"
               f"Per-fix RSS std (oracle calibration): {effective_sigma(t):.2f} dB\n"
               f"Range at sensitivity (no walls): {'unlimited' if not np.isfinite(rng) else f'{rng:.1f} m'}\n"
               f"Fix rate: {t.sample_rate_hz / t.samples_per_fix:.2f} Hz")
        if t.fading.startswith("Legacy"):
            txt += ("\nLegacy fading is always positive (biased): the oracle estimator adds its "
                    "mean to P0, a naive inversion with P0 alone underestimates distances.")
        self.lbl_tech.configure(text=txt)

    def read_traj(self) -> TrajParams:
        tp = TrajParams(mode=self.S("tr_mode"), point_source=self.S("tr_src"),
                        n_random_points=self.I("tr_nrand"), fixes_per_point=self.I("tr_K"),
                        grid_spacing=self.F("tr_grid"),
                        kind=self.S("tr_kind"), speed=self.F("tr_speed"), use_astar=self.B("tr_astar"),
                        astar_res=self.F("tr_res"), clearance=self.F("tr_clear"), n_random_wp=self.I("tr_nwp"),
                        circle_frac=self.F("tr_cfrac"), circle_laps=self.F("tr_laps"),
                        lawn_spacing=self.F("tr_lawn"),
                        grid_z=self.F("tr_gridz") if self.S("tr_gridz").strip() else None)
        if tp.speed <= 0 or tp.astar_res <= 0 or tp.grid_spacing <= 0:
            raise ValueError("Speed, A* resolution and grid spacing must be positive.")
        if tp.fixes_per_point < 1 or tp.n_random_points < 1:
            raise ValueError("Fixes per point and number of random points must be at least 1.")
        if tp.clearance < 0 or not (0 < tp.circle_frac <= 1) or tp.circle_laps <= 0 or tp.lawn_spacing <= 0:
            raise ValueError("Clearance >= 0, circle radius fraction in (0, 1], laps and lawnmower spacing > 0.")
        return tp

    def _mlp_layers(self):
        txt = self.S("l_mlp").replace(";", ",").replace(" ", ",")
        try:
            layers = tuple(int(x) for x in txt.split(",") if x.strip())
        except ValueError:
            layers = ()
        if not layers or min(layers) < 1:
            raise ValueError("'Neural network: hidden layers' must be positive integers separated by commas, "
                             "e.g. 64, 64.")
        return layers

    def read_loc(self) -> engine.LocParams:
        """Parameters of the chosen approach; the fields of the other approach (disabled)
        are not read, their defaults are used."""
        fam = self.S("l_family")
        use_d, use_f = fam in (DISTANCE, BOTH), fam in (FINGERPRINTING, BOTH)
        allowed = self._allowed_methods()
        kf_base = self.S("l_kfbase")
        if self.B("l_kf") and self.S("tr_mode") == "Tracking" and kf_base not in allowed:
            raise ValueError(f"The Kalman filter input '{kf_base}' does not belong to the chosen approach.")
        lp = engine.LocParams(
            algorithms=[n for n in allowed if self.B(f"alg::{n}")], z_res=self.F("l_zres"),
            clip_to_area=self.B("l_clip"), kf_enabled=self.B("l_kf"), kf_base=kf_base, kf_q=self.F("l_kfq"),
            kf_r=self.F("l_kfr"), mc_runs=self.I("l_runs"), seed=self.I("l_seed"), redraw_env=self.B("l_redraw"))
        if use_d:
            lp.pf_enabled, lp.pf_n, lp.pf_acc = self.B("l_pf"), self.I("l_pfn"), self.F("l_pfacc")
            lp.pf_jitter, lp.pf_sigma, lp.pf_inject = self.F("l_pfjit"), self.F("l_pfsig"), self.F("l_pfinj")
            lp.wcl_k, lp.wcl_g, lp.ml_res, lp.map_aware = (self.I("l_wclk"), self.F("l_wclg"), self.F("l_mlres"),
                                                           self.B("l_map"))
            lp.calib, lp.p0_est, lp.n_est, lp.sigma_est = (self.S("l_calib"), self.F("l_p0"), self.F("l_n"),
                                                           self.F("l_sig"))
            if min(lp.ml_res, lp.n_est, lp.sigma_est) <= 0:
                raise ValueError("The MLE grid resolution, the n estimate and the RSS std estimate must be positive.")
        if use_f:
            lp.rp_spacing, lp.rp_samples = self.F("l_rps"), self.I("l_rpn")
            lp.knn_k, lp.knn_weighted, lp.wknn_k = self.I("l_knnk"), self.B("l_knnw"), self.I("l_wknnk")
            lp.dt_min_leaf, lp.rf_trees, lp.mlp_hidden = self.I("l_dtleaf"), self.I("l_rftrees"), self._mlp_layers()
            if lp.rp_spacing <= 0 or min(lp.rp_samples, lp.knn_k, lp.wknn_k, lp.dt_min_leaf, lp.rf_trees) < 1:
                raise ValueError("The survey grid spacing must be positive; fingerprints per point, neighbours, "
                                 "leaf size and trees at least 1.")
        if lp.z_res <= 0 or lp.mc_runs < 1:
            raise ValueError("The vertical grid spacing must be positive and the trials at least 1.")
        return lp

    def copy_oracle(self):
        try:
            t = self.read_tech()
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        m = engine.estimator_model(self.scn, t, engine.LocParams())
        self._set("l_p0", f"{m.p0:.2f}")
        self._set("l_n", f"{m.n:g}")
        self._set("l_sig", f"{m.sigma:.2f}")

    # --------------------------------------------------------------- drawing
    def _algo_color(self, name):
        """Colour of a method: its position in the results, so that the methods of a run
        (trackers included) always get distinct colours."""
        names = self.results.names if self.results is not None else list(ALGORITHMS)
        i = names.index(name) if name in names else len(names)
        return PALETTE[i % len(PALETTE)]

    def _draw_scene(self, ax, light=False):
        s = self.scn
        rx0, ry0, rx1, ry1 = s.room_bounds()
        ax.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False, ec="#1f4e9c", lw=1.6, zorder=2))
        x0, y0, x1, y1 = s.area_bounds()
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#d64545", alpha=0.06 if light else 0.12,
                               ec="#d64545", ls="--", lw=1.0, zorder=1))
        for i, o in enumerate(s.obstacles):
            selected = self.sel == ("obstacle", i)
            ax.add_patch(Polygon(o.polygon(), closed=True, fc=MATERIAL_COLORS.get(o.material, "#888"),
                                 ec="#f2b705" if selected else "#222222", lw=2.2 if selected else 0.6,
                                 alpha=0.6 if light else 0.95, zorder=4))
        A = s.anchors
        for i, a in enumerate(A):
            selected = self.sel == ("anchor", i)
            ax.plot(a.x, a.y, "o", ms=11, mfc="#2e9e44" if a.enabled else "white",
                    mec="#f2b705" if selected else ("#1b5e2a" if a.enabled else "#999999"),
                    mew=2.5 if selected else 1.2, zorder=6)
            ax.annotate(a.name, (a.x, a.y), xytext=(6, 5), textcoords="offset points", fontsize=9,
                        color="#1b5e2a" if a.enabled else "#999999", zorder=7)

    def draw_map(self, keep_view=False):
        ax = self.ax_map
        xl, yl = (ax.get_xlim(), ax.get_ylim()) if keep_view else (None, None)
        ax.clear()
        s = self.scn
        self._draw_scene(ax)
        if self.survey_preview is not None and len(self.survey_preview):
            sp = self.survey_preview
            ax.plot(sp[:, 0], sp[:, 1], "+", color="#5f5f5f", ms=5, mew=0.9, alpha=0.85, zorder=3)
        self._time_artists = {}
        r = self.results if self.results_visible else None
        pos_mode = self.S("tr_mode") == "Positioning"
        if pos_mode:
            self._draw_positioning(ax, r if (r is not None and r.mode == "positioning") else None)
        else:
            self._draw_tracking(ax, r if (r is not None and r.mode == "tracking") else None)
        rx0, ry0, rx1, ry1 = s.room_bounds()
        m = 0.04 * max(rx1 - rx0, ry1 - ry0)
        ax.set_xlim(*(xl or (rx0 - m, rx1 + m)))
        ax.set_ylim(*(yl or (ry0 - m, ry1 + m)))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.15)
        if self.survey_preview is not None:
            nl = len(np.unique(np.round(self.survey_preview[:, 2], 6))) if len(self.survey_preview) else 0
            ax.set_title(f"Fingerprint survey: {len(self.survey_preview)} reference points (+)"
                         + (f" on {nl} height layers" if nl > 1 else ""), fontsize=10)
        elif not ax.get_legend():
            n_on = sum(a.enabled for a in s.anchors)
            what = (f"{len(s.test_points)} test points" if pos_mode else f"{len(s.waypoints)} waypoints")
            ax.set_title(f"{'Positioning' if pos_mode else 'Tracking'}: {len(s.anchors)} anchors ({n_on} on), "
                         f"{len(s.obstacles)} obstacles, {what}", fontsize=10)
        if self._time_artists:
            self._on_time(self.time_scale.get())
        self.canvas_map.draw_idle()
        if hasattr(self, "fig_3d"):
            self._on_plot_tab()

    # ---------------------------------------------------------------- 3D view
    def _on_plot_tab(self, _=None):
        try:
            if self.plots.tab(self.plots.select(), "text") == "3D view":
                self.draw_3d()
        except tk.TclError:
            pass

    def draw_3d(self):
        """Perspective view of room, obstacles (with heights), anchors and experiment.
        The vertical scale is exaggerated so that heights stay readable."""
        fig = self.fig_3d
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")
        s = self.scn
        rx0, ry0, rx1, ry1 = s.room_bounds()
        H = max(s.room_z, 0.1)
        ring = np.array([[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1], [rx0, ry0]])
        for z in (0.0, H):
            ax.plot(ring[:, 0], ring[:, 1], z, color="#1f4e9c", lw=0.8)
        for x, y in ring[:4]:
            ax.plot([x, x], [y, y], [0, H], color="#1f4e9c", lw=0.6)
        x0, y0, z0, x1, y1, z1 = s.area_bounds3()
        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
        for z in ((z0, z1) if z1 > z0 else (z0,)):
            ax.plot(box[:, 0], box[:, 1], z, "--", color="#d64545", lw=0.9)
        polys, cols = [], []
        for o in s.obstacles:
            zb, zt = o.z0, (H if o.z1 is None else o.z1)
            c = MATERIAL_COLORS.get(o.material, "#888888")
            if o.kind == "wall":
                polys.append([(o.x0, o.y0, zb), (o.x1, o.y1, zb), (o.x1, o.y1, zt), (o.x0, o.y0, zt)])
                cols.append(c)
            else:
                xa, xb, ya, yb = o.x0, o.x1, o.y0, o.y1
                v = lambda x, y, z: (x, y, z)  # noqa: E731
                faces = [[v(xa, ya, zb), v(xb, ya, zb), v(xb, yb, zb), v(xa, yb, zb)],
                         [v(xa, ya, zt), v(xb, ya, zt), v(xb, yb, zt), v(xa, yb, zt)],
                         [v(xa, ya, zb), v(xb, ya, zb), v(xb, ya, zt), v(xa, ya, zt)],
                         [v(xa, yb, zb), v(xb, yb, zb), v(xb, yb, zt), v(xa, yb, zt)],
                         [v(xa, ya, zb), v(xa, yb, zb), v(xa, yb, zt), v(xa, ya, zt)],
                         [v(xb, ya, zb), v(xb, yb, zb), v(xb, yb, zt), v(xb, ya, zt)]]
                polys += faces
                cols += [c] * 6
        if polys:
            ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="#333333",
                                                 linewidths=0.3, alpha=0.35))
        for a in s.anchors:
            ax.scatter([a.x], [a.y], [a.z], s=40, c="#2e9e44" if a.enabled else "white",
                       edgecolors="#1b5e2a", depthshade=False)
            ax.text(a.x, a.y, a.z + 0.04 * H, a.name, fontsize=7, color="#1b5e2a")
        r = self.results if self.results_visible else None
        if self.S("tr_mode") == "Positioning":
            rp = r if (r is not None and r.mode == "positioning") else None
            if rp is not None:
                pts = rp.points
            elif self.points_preview is not None:
                pts = self.points_preview
            else:
                pts = s.points3(s.test_points)
            if rp is not None:
                R, K, P = rp.trials, rp.fixes_per_point, len(pts)
                rng = np.random.default_rng(0)
                for n in rp.names:
                    if n in self.hidden:
                        continue
                    c = self._algo_color(n)
                    E = rp.est[n].reshape(R, P, K, 3).transpose(1, 0, 2, 3).reshape(P, R * K, 3)
                    cloud = E.reshape(-1, 3)
                    cloud = cloud[np.isfinite(cloud).all(1)]
                    if len(cloud) > 1500:
                        cloud = cloud[rng.choice(len(cloud), 1500, replace=False)]
                    ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=2, c=c, alpha=0.25, depthshade=False)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        m = np.nanmean(E, axis=1)
                    ax.scatter(m[:, 0], m[:, 1], m[:, 2], marker="X", s=50, c=c, depthshade=False, label=n)
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker="D", s=30, c="white", edgecolors="black",
                           depthshade=False, label="True position")
                for p in pts[:60]:
                    ax.plot([p[0], p[0]], [p[1], p[1]], [0, p[2]], color="#999999", lw=0.5)
        else:
            rt = r if (r is not None and r.mode == "tracking") else None
            if rt is not None:
                ax.plot(rt.gt[:, 0], rt.gt[:, 1], rt.gt[:, 2], color="black", lw=1.8, label="Ground truth")
                for n in rt.names:
                    if n not in self.hidden:
                        e = rt.est[n][0]
                        ax.plot(e[:, 0], e[:, 1], e[:, 2], color=self._algo_color(n), lw=0.9, alpha=0.8, label=n)
            else:
                route = self.route_preview if self.route_preview is not None else (
                    s.points3(s.waypoints) if s.waypoints else None)
                if route is not None and len(route):
                    ax.plot(route[:, 0], route[:, 1], route[:, 2], color="#555555", lw=1.2, label="Route")
        ax.set_xlim(rx0, rx1)
        ax.set_ylim(ry0, ry1)
        ax.set_zlim(0, H)
        W, Dp = rx1 - rx0, ry1 - ry0
        hz = max(H, 0.15 * max(W, Dp))
        ax.set_box_aspect((W, Dp, hz))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        mode = "3D (x, y, z)" if s.dim == 3 else "2D (x, y), known height"
        ax.set_title(f"Estimation: {mode}. Vertical scale x{hz / H:.1f}. Drag to rotate.", fontsize=9)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=7, ncol=2)
        self.canvas_3d.draw_idle()

    def _draw_tracking(self, ax, r):
        s = self.scn
        if r is None and self.route_preview is not None:
            rt = self.route_preview
            ax.plot(rt[:, 0], rt[:, 1], "-", color="#555555", lw=1.2, alpha=0.8, zorder=3)
        if s.waypoints:
            w = s.points3(s.waypoints)
            ax.plot(w[:, 0], w[:, 1], ":", color="#444444", lw=0.9, zorder=5)
            ax.plot(w[:, 0], w[:, 1], "s", mfc="white", mec="#222222", mew=1.3, ms=7, zorder=10)
            for i, (x, y) in enumerate(w[:, :2]):
                ax.annotate(str(i + 1), (x, y), xytext=(5, -11), textcoords="offset points", fontsize=8,
                            color="#222222", zorder=10)
        if r is None:
            return
        ax.plot(r.gt[:, 0], r.gt[:, 1], "-", color="black", lw=2.0, zorder=8, label="Ground truth")
        for n in r.names:
            if n in self.hidden:
                continue
            e = r.est[n][0]
            ax.plot(e[:, 0], e[:, 1], "-", color=self._algo_color(n), lw=1.1, alpha=0.8, zorder=9, label=n)
            self._time_artists[n] = ax.plot([], [], "o", color=self._algo_color(n), ms=8, mec="white",
                                            zorder=11)[0]
        self._time_artists["__gt__"] = ax.plot([], [], "*", color="black", ms=14, mec="white", zorder=12)[0]
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=4, fontsize=8, frameon=False,
                  borderaxespad=0.0)

    def _draw_positioning(self, ax, r):
        """True points (diamonds), estimate clouds, mean estimates (x) and 95% ellipses."""
        if r is not None:
            pts = r.points
        elif self.points_preview is not None:
            pts = self.points_preview
        else:
            pts = self.scn.points3(self.scn.test_points)
        P = len(pts)
        few = P <= 60
        if r is not None:
            R, K = r.trials, r.fixes_per_point
            for n in r.names:
                if n in self.hidden:
                    continue
                c = self._algo_color(n)
                E = r.est[n][..., :2].reshape(R, P, K, 2).transpose(1, 0, 2, 3).reshape(P, R * K, 2)
                cloud = E.reshape(-1, 2)
                cloud = cloud[np.isfinite(cloud).all(1)]
                if len(cloud) > 6000:
                    cloud = cloud[np.random.default_rng(0).choice(len(cloud), 6000, replace=False)]
                ax.plot(cloud[:, 0], cloud[:, 1], ".", color=c, ms=2.5, alpha=0.3, zorder=7)
                if few:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        m = np.nanmean(E, axis=1)
                    for i in range(P):
                        ax.plot([pts[i, 0], m[i, 0]], [pts[i, 1], m[i, 1]], "-", color=c, lw=1.0, zorder=8)
                        Ei = E[i][np.isfinite(E[i]).all(1)]
                        if len(Ei) > 3:
                            self._ellipse(ax, Ei, c)
                    ax.plot(m[:, 0], m[:, 1], "X", color=c, mec="white", mew=0.8, ms=9, zorder=9, label=n)
                else:
                    ax.plot([], [], ".", color=c, ms=8, label=n)
        if P:
            ax.plot(pts[:, 0], pts[:, 1], "D", mfc="white", mec="black", mew=1.4, ms=7 if few else 3,
                    zorder=10, label="True position" if r is not None else None)
            if few:
                for i, (x, y) in enumerate(pts[:, :2]):
                    ax.annotate(f"P{i + 1}", (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8,
                                color="black", zorder=11)
        if r is not None:
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=4, fontsize=8, frameon=False,
                      borderaxespad=0.0)

    @staticmethod
    def _ellipse(ax, pts, color):
        """95% confidence ellipse of a 2D point cloud."""
        cov = np.cov(pts.T)
        if not np.all(np.isfinite(cov)):
            return
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-12)
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        k = 2.4477                                   # sqrt(chi2_2(0.95))
        ax.add_patch(Ellipse(pts.mean(0), 2 * k * np.sqrt(vals[1]), 2 * k * np.sqrt(vals[0]), angle=ang,
                             fill=False, ec=color, lw=1.0, alpha=0.9, zorder=8))

    # ----------------------------------------------------------- mouse edits
    def _tool_changed(self):
        self._cancel_drawing()
        tool = self.S("tool")
        want = {"testpoint": "Positioning", "waypoint": "Tracking"}.get(tool)
        if want and self.S("tr_mode") != want:
            self._set("tr_mode", want)
            self._mode_changed()
        self._hint(TOOL_HINTS[tool])

    def _tol(self):
        xl = self.ax_map.get_xlim()
        return 0.02 * abs(xl[1] - xl[0])

    def _pick(self, x, y, kinds=("anchor", "testpoint", "waypoint", "obstacle")):
        """Nearest editable item; test points / waypoints only in their own mode."""
        tol = self._tol()
        best = None
        pos_mode = self.S("tr_mode") == "Positioning"
        if "testpoint" in kinds and pos_mode:
            for i, p in enumerate(self.scn.test_points):
                d = np.hypot(p[0] - x, p[1] - y)
                if d <= tol and (best is None or d < best[2]):
                    best = ("testpoint", i, d)
        if "anchor" in kinds:
            for i, a in enumerate(self.scn.anchors):
                d = np.hypot(a.x - x, a.y - y)
                if d <= tol and (best is None or d < best[2]):
                    best = ("anchor", i, d)
        if "waypoint" in kinds and not pos_mode:
            for i, p in enumerate(self.scn.waypoints):
                d = np.hypot(p[0] - x, p[1] - y)
                if d <= tol and (best is None or d < best[2]):
                    best = ("waypoint", i, d)
        if best:
            return best[:2]
        if "obstacle" in kinds:
            pt = np.array([[x, y]])
            for i in range(len(self.scn.obstacles) - 1, -1, -1):
                o = self.scn.obstacles[i]
                if o.kind == "wall":
                    hit = point_seg_dist(x, y, o.x0, o.y0, o.x1, o.y1) <= o.thickness / 2 + tol / 2
                else:
                    hit = inside_obstacles(pt, [o], margin=tol / 4, motion_only=False)[0]
                if hit:
                    return ("obstacle", i)
        return None

    def _on_press(self, ev):
        if ev.inaxes is not self.ax_map or self.nav.mode or ev.xdata is None:
            return
        self.canvas_map.get_tk_widget().focus_set()   # Delete now acts on the map selection
        x, y = self._snap(ev.xdata, ev.ydata)
        tool = self.S("tool")
        if ev.button == 3:
            hit = self._pick(ev.xdata, ev.ydata)
            if hit:
                self._remove(hit)
            return
        if tool == "anchor":
            self.scn.anchors.append(Anchor(f"a{len(self.scn.anchors) + 1}", x, y,
                                           z=self._field_or("g_hanc", self.scn.anchor_h)))
            self._after_edit(select=("anchor", len(self.scn.anchors) - 1))
        elif tool == "waypoint":
            self.scn.waypoints.append((x, y, self._field_or("g_htag", self.scn.tag_h)))
            self._after_edit()
        elif tool == "testpoint":
            self.scn.test_points.append((x, y, self._field_or("g_htag", self.scn.tag_h)))
            self._after_edit()
        elif tool in ("wall", "rect"):
            self._draw_start = (x, y)
            if tool == "wall":
                self._preview_artist = self.ax_map.plot([x, x], [y, y], "-", color="#f2b705", lw=3, zorder=20)[0]
            else:
                self._preview_artist = Rectangle((x, y), 0, 0, fc="#f2b705", alpha=0.4, ec="#b58900", zorder=20)
                self.ax_map.add_patch(self._preview_artist)
            self.canvas_map.draw_idle()
        elif tool == "move":
            hit = self._pick(ev.xdata, ev.ydata)
            if hit:
                self._drag = (hit[0], hit[1], x, y)
        elif tool == "erase":
            hit = self._pick(ev.xdata, ev.ydata)
            if hit:
                self._remove(hit)
        elif tool == "select":
            hit = self._pick(ev.xdata, ev.ydata, kinds=("anchor", "obstacle"))
            self.sel = hit
            for tree, kind in ((self.tree_anc, "anchor"), (self.tree_obs, "obstacle")):
                tree.selection_remove(tree.selection())
                if hit and hit[0] == kind and tree.exists(str(hit[1])):
                    tree.selection_set(str(hit[1]))
                    tree.see(str(hit[1]))
            self.draw_map(keep_view=True)

    def _on_motion(self, ev):
        if ev.inaxes is self.ax_map and ev.xdata is not None:
            self.lbl_xy.configure(text=f"x = {ev.xdata:7.2f}  y = {ev.ydata:7.2f}")
        else:
            return
        x, y = self._snap(ev.xdata, ev.ydata)
        if self._draw_start and self._preview_artist is not None:
            x0, y0 = self._draw_start
            if self.S("tool") == "wall":
                x, y = self._axis_snap(x0, y0, x, y)
                self._preview_artist.set_data([x0, x], [y0, y])
            else:
                self._preview_artist.set_bounds(min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            self.canvas_map.draw_idle()
        elif self._drag:
            kind, i, px, py = self._drag
            dx, dy = x - px, y - py
            if kind == "anchor":
                self.scn.anchors[i].x += dx
                self.scn.anchors[i].y += dy
            elif kind in ("waypoint", "testpoint"):
                lst = self.scn.waypoints if kind == "waypoint" else self.scn.test_points
                p = lst[i]
                lst[i] = (p[0] + dx, p[1] + dy) + tuple(p[2:])
            else:
                o = self.scn.obstacles[i]
                o.x0 += dx; o.x1 += dx; o.y0 += dy; o.y1 += dy
            self._drag = (kind, i, x, y)
            now = time.time()
            if now - self._last_redraw > 0.04:
                self._last_redraw = now
                self._invalidate()
                self.draw_map(keep_view=True)

    def _axis_snap(self, x0, y0, x, y):
        """Walls within 7 degrees of horizontal/vertical are straightened."""
        ang = np.degrees(np.arctan2(abs(y - y0), abs(x - x0)))
        if ang < 7:
            return x, y0
        if ang > 83:
            return x0, y
        return x, y

    def _on_release(self, ev):
        if self._drag:
            kind, i, _, _ = self._drag
            self._drag = None
            self._after_edit(select=(kind, i) if kind in ("anchor", "obstacle") else None)
            return
        if not self._draw_start:
            return
        x0, y0 = self._draw_start
        tool = self.S("tool")
        if ev.xdata is not None and ev.inaxes is self.ax_map:
            x, y = self._snap(ev.xdata, ev.ydata)
        else:
            x, y = x0, y0
        self._cancel_drawing()
        try:
            th = max(self.F("draw_thick"), 0.01)
        except ValueError:
            th = 0.2
        mat = self.S("draw_mat")
        if tool == "wall":
            x, y = self._axis_snap(x0, y0, x, y)
            if np.hypot(x - x0, y - y0) < 0.2:
                return
            self.scn.obstacles.append(Obstacle("wall", x0, y0, x, y, material=mat, thickness=th))
        else:
            if abs(x - x0) < 0.1 or abs(y - y0) < 0.1:
                return
            try:
                bz = self.S("draw_bz").strip()
                top = float(bz.replace(",", ".")) if bz else None
            except ValueError:
                top = None
            if top is not None and top <= 0:
                self._hint("Block top z must be positive: this block reaches the ceiling.")
                top = None
            self.scn.obstacles.append(Obstacle("rect", x0, y0, x, y, material=mat, z1=top).normalized())
        self._after_edit(select=("obstacle", len(self.scn.obstacles) - 1))

    def _cancel_drawing(self):
        self._draw_start = None
        if self._preview_artist is not None:
            try:
                self._preview_artist.remove()
            except Exception:
                pass
            self._preview_artist = None
            self.canvas_map.draw_idle()

    def _remove(self, hit):
        kind, i = hit
        if kind == "anchor":
            del self.scn.anchors[i]
            self.scn.rename_anchors()
        elif kind == "waypoint":
            del self.scn.waypoints[i]
        elif kind == "testpoint":
            del self.scn.test_points[i]
        else:
            del self.scn.obstacles[i]
        self._after_edit()

    # ------------------------------------------------------------ trajectory
    def preview_route(self):
        """Shows the test points (Positioning) or the A* route (Tracking) that a run would use."""
        try:
            self.apply_geometry(silent=True)
            tp = self.read_traj()
            rng = np.random.default_rng(self.I("l_seed"))       # same draws as engine.run
            warn = []
            if tp.positioning:
                pts, _ = static_points(self.scn, tp, rng, warn)
            else:
                route = build_route(self.scn, tp, rng, warn)
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        self._invalidate()
        if tp.positioning:
            self.points_preview = pts
            self._log(f"{len(pts)} test points ({tp.point_source}), {tp.fixes_per_point} fixes each per trial.")
        else:
            self.route_preview = route
            L = float(np.sum(np.hypot(*np.diff(route, axis=0).T))) if len(route) > 1 else 0.0
            self._log(f"Route: {L:.1f} m, {L / max(tp.speed, 1e-9):.1f} s at {tp.speed:g} m/s.")
        for w in warn:
            self._log(w)
        self.draw_map(keep_view=True)

    def random_to_map(self):
        """Copies the random points of the current seed into the editable test points."""
        try:
            self.apply_geometry(silent=True)
            tp = self.read_traj()
            pts = static_points(self.scn, TrajParams(point_source="Random points",
                                                     n_random_points=tp.n_random_points,
                                                     clearance=tp.clearance),
                                np.random.default_rng(self.I("l_seed")))[0]
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        self.scn.test_points = [tuple(round(float(c), 2) for c in p) for p in pts]
        self._set("tr_src", "Points on map")
        self._after_edit()
        self._log(f"{len(pts)} random points copied to the map: move or delete them as needed.")

    def preview_survey(self):
        """Shows the reference points of the fingerprint survey on the map."""
        try:
            self.apply_geometry(silent=True)
            sp, S, zres = self.F("l_rps"), self.I("l_rpn"), self.F("l_zres")
            if sp <= 0 or S < 1 or zres <= 0:
                raise ValueError("The survey grid spacing must be positive and the fingerprints per point "
                                 "at least 1.")
            pts = survey_points(self.scn, sp, zres)
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        self._invalidate()
        self.survey_preview = pts
        self._log(f"Fingerprint survey: {len(pts)} reference points every {sp:g} m x {S} fingerprints = "
                  f"{len(pts) * S} training fingerprints.")
        self.draw_map(keep_view=True)

    # ------------------------------------------------------------- point table
    def _cur_points(self):
        return self.scn.test_points if self.S("tr_mode") == "Positioning" else self.scn.waypoints

    def _refresh_points(self):
        self.tree_pts.delete(*self.tree_pts.get_children())
        pre = "P" if self.S("tr_mode") == "Positioning" else "W"
        for i, p in enumerate(self._cur_points()):
            z = p[2] if len(p) > 2 else self.scn.tag_h
            self.tree_pts.insert("", "end", iid=str(i),
                                 values=(f"{pre}{i + 1}", f"{p[0]:.2f}", f"{p[1]:.2f}", f"{z:.2f}"))

    def _point_selected(self, _=None):
        i = self._sel_index(self.tree_pts)
        if i is not None and i < len(self._cur_points()):
            p = self._cur_points()[i]
            self._set("p_x", f"{p[0]:g}")
            self._set("p_y", f"{p[1]:g}")
            self._set("p_z", f"{(p[2] if len(p) > 2 else self.scn.tag_h):g}")

    def point_update(self):
        i = self._sel_index(self.tree_pts)
        if i is None:
            return
        try:
            self._cur_points()[i] = (self.F("p_x"), self.F("p_y"), self.F("p_z"))
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        self._after_edit()
        if self.tree_pts.exists(str(i)):
            self.tree_pts.selection_set(str(i))

    def point_delete(self):
        i = self._sel_index(self.tree_pts)
        if i is not None:
            del self._cur_points()[i]
            self._after_edit()

    def points_at_tag_height(self):
        try:
            self.apply_geometry(silent=True)
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        lst = self._cur_points()
        lst[:] = [(p[0], p[1], self.scn.tag_h) for p in lst]
        self._after_edit()

    def clear_test_points(self):
        self.scn.test_points = []
        self._after_edit()

    def clear_waypoints(self):
        self.scn.waypoints = []
        self._after_edit()

    def reverse_waypoints(self):
        self.scn.waypoints = self.scn.waypoints[::-1]
        self._after_edit()

    # ------------------------------------------------------------- simulation
    def run_simulation(self):
        if self._worker is not None:
            return
        try:
            self.apply_geometry(silent=True)
            tech, tp, lp = self.read_tech(), self.read_traj(), self.read_loc()
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        scn = copy.deepcopy(self.scn)
        self._cancel = False
        self.btn_run.state(["disabled"])
        self.btn_stop.state(["!disabled"])
        self.progress["value"] = 0
        self._stop_play()
        q = queue.Queue()

        def work():
            try:
                res = engine.run(scn, tech, tp, lp, progress=lambda f, m: q.put(("p", f, m)),
                                 cancel=lambda: self._cancel)
                q.put(("done", res, time.time()))
            except InterruptedError:
                q.put(("stopped",))
            except (ValueError, ImportError) as e:
                q.put(("user", str(e)))
            except Exception as e:
                q.put(("err", str(e), traceback.format_exc()))

        self._t_start = time.time()
        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self._poll(q, tech, lp)

    def _poll(self, q, tech, lp):
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == "p":
                    self.progress["value"] = msg[1]
                    self._hint(msg[2])
                    continue
                self._worker = None
                self.btn_run.state(["!disabled"])
                self.btn_stop.state(["disabled"])
                self.progress["value"] = 1.0 if msg[0] == "done" else 0.0
                if msg[0] == "done":
                    self._show_results(msg[1], tech, lp, msg[2] - self._t_start)
                elif msg[0] == "stopped":
                    self._hint("Simulation stopped.")
                elif msg[0] == "user":
                    messagebox.showerror("Cannot run the simulation", msg[1])
                else:
                    self._log(msg[2])
                    messagebox.showerror("Simulation error", msg[1] + "\n\nDetails in the Analysis > Log panel.")
                return
        except queue.Empty:
            pass
        self.root.after(80, self._poll, q, tech, lp)

    def _stop(self):
        self._cancel = True

    @staticmethod
    def _fmt(v, nd=2):
        try:
            return "" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"
        except TypeError:
            return str(v)

    def _show_results(self, res: engine.Results, tech, lp, elapsed):
        self.results = res
        self.results_visible = True
        self.route_preview = None
        self.points_preview = None
        self.survey_preview = None
        if res.mode != self.S("tr_mode").lower():
            self._set("tr_mode", res.mode.capitalize())
            self._mode_changed(redraw=False)
        summ = res.summary()
        rank = sorted(res.names, key=lambda n: summ[n]["rmse"] if np.isfinite(summ[n]["rmse"]) else 1e18)
        self.hidden = set(rank[3:])
        f = self._fmt
        cols = summary_cols(res.dim)
        self._set_columns(self.tree, cols)
        for n in res.names:
            s = summ[n]
            tr = res.train_s.get(n, 0.0)
            fixed = {"show": "off" if n in self.hidden else "on", "algo": n,
                     "type": TYPE_LABEL.get(res.families.get(n, ""), ""), "valid": f(s["valid"], 1),
                     "ms": f(res.timing_ms[n], 3), "train": f(tr, 2) if tr > 0 else ""}
            self.tree.insert("", "end", iid=n, values=[fixed[c] if c in fixed else f(s[c]) for c, _, _ in cols])
        F, R = len(res.gt), res.trials
        tracking = res.mode == "tracking"
        self.time_scale.configure(to=max(F - 1, 1))
        self.time_scale.set(0)
        self.btn_play.state(["!disabled"] if tracking else ["disabled"])
        self.lbl_time.configure(text="" if tracking else "Static target: no timeline")
        self.cb_detail.configure(values=res.names)
        if self.S("d_algo") not in res.names:
            self._set("d_algo", rank[0])
        self.res_nb.tab(1, text="Per trial" if tracking else "Per point")
        self.draw_map(keep_view=True)
        self.draw_cdf()
        self.fill_detail()
        self.draw_detail()
        if tracking:
            what = f"{R} trial(s) x {F} fixes along the route"
        else:
            what = (f"{R} trial(s) x {len(res.points)} points x {res.fixes_per_point} fixes "
                    f"= {R * F} estimates per method")
        uses_model = DISTANCE in res.families.values() or any(n.startswith("PF") for n in res.names)
        model = (f" Path-loss estimator: P0={res.model.p0:.1f} dBm, n={res.model.n:g}, "
                 f"sigma={res.model.sigma:.2f} dB." if uses_model else "")
        self._log(f"Run done in {elapsed:.1f} s: {res.tech_name}, {what}.{model} Lowest RMSE: {rank[0]} "
                  f"({summ[rank[0]]['rmse']:.2f} m).")
        for n in res.notes:
            self._log(n)
        self._hint("Map shows the 3 lowest-RMSE methods: click a row of the Summary table to show or hide one.")

    def fill_detail(self):
        r = self.results
        if r is None:
            return
        n = self.S("d_algo") if self.S("d_algo") in r.names else r.names[0]
        f = self._fmt
        if r.mode == "positioning":
            pcols = point_cols(r.dim)
            self._set_columns(self.tree_detail, pcols)
            ps = r.point_stats(n)
            for i in range(len(r.points)):
                self.tree_detail.insert("", "end", values=[f"P{i + 1}"] + [
                    f(ps[k][i], 1 if k == "valid" else 2) if k != "n" else int(ps[k][i])
                    for k, _, _ in pcols[1:]])
            s = r.summary()[n]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                tot = dict(s, x=None, y=None, z=None, est_x=None, est_y=None, est_z=None,
                           bias=float(np.sqrt(np.nanmean(ps["bias"] ** 2))),
                           spread=float(np.sqrt(np.nanmean(ps["spread"] ** 2))))
            self.tree_detail.insert("", "end", tags=("total",), values=["All"] + [
                f(tot[k], 1 if k == "valid" else 2) if k != "n" else int(tot[k]) for k, _, _ in pcols[1:]])
            self.lbl_detail.configure(text="Bias = |mean estimate - true|, Spread = RMS distance from the mean "
                                           "estimate, MSE = Bias² + Spread². 'All' row: RMS over points.")
        else:
            tcols = trial_cols(r.dim)
            self._set_columns(self.tree_detail, tcols)
            ts = r.trial_stats(n)
            keys = [k for k, _, _ in tcols[1:]]
            for t in range(r.trials):
                self.tree_detail.insert("", "end", values=[t + 1] + [
                    f(ts[k][t], 1 if k == "valid" else 2) if k != "n" else int(ts[k][t]) for k in keys])
            if r.trials > 1:
                for lab, fn in (("Mean", np.nanmean), ("Std", np.nanstd)):
                    self.tree_detail.insert("", "end", tags=("total",), values=[lab] + [
                        f(float(fn(ts[k])), 1 if k == "valid" else 2) if k != "n" else "" for k in keys])
            self.lbl_detail.configure(text="One row per repetition of the route; last rows: mean and std "
                                           "of each statistic across trials.")

    def _toggle_algo(self, ev):
        item = self.tree.identify_row(ev.y)
        if not item or self.results is None:
            return
        if item in self.hidden:
            self.hidden.discard(item)
        else:
            self.hidden.add(item)
        self.tree.set(item, "show", "off" if item in self.hidden else "on")
        self.draw_map(keep_view=True)
        self.draw_cdf()
        self.draw_detail()

    def draw_cdf(self):
        r = self.results
        self.fig_cdf.clf()
        ax = self.fig_cdf.add_subplot(111)
        if r is not None:
            for n in r.names:
                if n in self.hidden:
                    continue
                x, y = metrics.ecdf(r.err[n])
                if x.size:
                    ax.step(x, y, where="post", color=self._algo_color(n), lw=1.6, label=n)
            ax.set_xlim(left=0)
            ax.set_ylim(0, 1.01)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="lower right", fontsize=9)
            else:
                ax.text(0.5, 0.5, "No method produced an estimate", ha="center", va="center",
                        transform=ax.transAxes, color="#666666")
            if r.mode == "positioning":
                sub = f"{r.trials} trial(s) x {len(r.points)} points x {r.fixes_per_point} fixes"
            else:
                sub = f"{r.trials} trial(s) x {len(r.gt)} fixes"
            ax.set_title(f"{r.tech_name}: {sub}", fontsize=10)
        ax.set_xlabel("Positioning error [m]")
        ax.set_ylabel("Empirical CDF")
        ax.grid(True, alpha=0.3)
        self.canvas_cdf.draw_idle()

    def draw_detail(self):
        r = self.results
        fig = self.fig_time
        fig.clf()
        if r is None:
            self.canvas_time.draw_idle()
            return
        shown = [n for n in r.names if n not in self.hidden]
        sel = self.S("d_algo") if self.S("d_algo") in r.names else r.names[0]
        if r.mode == "positioning":
            P = len(r.points)
            x = np.arange(1, P + 1)
            ax1, ax2 = fig.subplots(2, 1, sharex=True)
            for n in shown:
                ax1.plot(x, r.point_stats(n)["rmse"], "o-", ms=4, lw=1.0, color=self._algo_color(n), label=n)
            ps = r.point_stats(sel)
            ax1.set_ylabel("RMSE per point [m]")
            ax1.legend(fontsize=8, ncol=3)
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"{r.trials} trial(s) x {r.fixes_per_point} fixes per point", fontsize=10)
            if P <= 40:
                R, K = r.trials, r.fixes_per_point
                E = r.err[sel].reshape(R, P, K).transpose(1, 0, 2).reshape(P, R * K)
                data = [e[np.isfinite(e)] for e in E]
                ax2.boxplot(data, positions=x, widths=0.6, showfliers=True,
                            flierprops=dict(markersize=2, alpha=0.4),
                            medianprops=dict(color=self._algo_color(sel)))
                ax2.set_ylabel(f"Error [m]\n{sel}")
            else:
                ax2.plot(x, ps["bias"], "-", lw=1.0, label="Bias")
                ax2.plot(x, ps["spread"], "-", lw=1.0, label="Spread")
                ax2.set_ylabel(f"[m]  {sel}")
                ax2.legend(fontsize=8)
            ax2.set_xlabel("Test point")
            if P <= 40:
                ax2.set_xticks(x)
                ax2.set_xticklabels([f"P{i}" for i in x], fontsize=8)
            ax2.grid(True, alpha=0.3)
        else:
            rows = 2 if r.trials > 1 else 1
            axs = np.atleast_1d(fig.subplots(rows, 1))
            ax = axs[0]
            for n in shown:
                ax.plot(r.t, r.err[n][0], color=self._algo_color(n), lw=1.1, label=n)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Error, trial 1 [m]")
            ax.legend(fontsize=8, ncol=3)
            ax.grid(True, alpha=0.3)
            if rows == 2:
                tr = np.arange(1, r.trials + 1)
                for n in shown:
                    axs[1].plot(tr, r.trial_stats(n)["rmse"], "o-", ms=4, lw=1.0, color=self._algo_color(n), label=n)
                axs[1].set_xlabel("Trial")
                axs[1].set_ylabel("RMSE per trial [m]")
                axs[1].grid(True, alpha=0.3)
                if r.trials <= 30:
                    axs[1].set_xticks(tr)
        self.canvas_time.draw_idle()

    # ------------------------------------------------------------- time slider
    def _on_time(self, val):
        r = self.results
        if r is None or not self._time_artists or r.mode != "tracking":
            return
        k = int(round(float(val)))
        k = max(0, min(k, len(r.gt) - 1))
        for n, art in self._time_artists.items():
            p = r.gt[k] if n == "__gt__" else r.est[n][0, k]
            art.set_data([p[0]], [p[1]])
        self.lbl_time.configure(text=f"t = {r.t[k]:.1f} s   fix {k + 1}/{len(r.gt)}")
        self.canvas_map.draw_idle()

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
        else:
            self._playing = True
            self.btn_play.configure(text="Pause")
            if self.time_scale.get() >= float(self.time_scale.cget("to")):
                self.time_scale.set(0)
            self._tick()

    def _stop_play(self):
        self._playing = False
        if hasattr(self, "btn_play"):
            self.btn_play.configure(text="Play")

    def _tick(self):
        if not self._playing or self.results is None:
            return
        F = len(self.results.gt)
        k = self.time_scale.get() + max(1, F // 300)
        if k >= F - 1:
            self.time_scale.set(F - 1)
            self._stop_play()
            return
        self.time_scale.set(k)
        self.root.after(40, self._tick)

    # ------------------------------------------------------------------ files
    def _settings(self):
        return {k: v.get() for k, v in self.v.items()}

    def save_project(self):
        try:
            self.apply_geometry(silent=True)              # store the geometry shown in the fields
            tech = self.read_tech()
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("RSSILab project", "*.json")])
        if not path:
            return
        try:
            save_project(path, self.scn, tech, self._settings())
            self._log(f"Project saved: {path}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def open_project(self):
        path = filedialog.askopenfilename(filetypes=[("RSSILab project", "*.json"), ("All files", "*.*")])
        if not path:
            return
        try:
            scn, tech, settings = load_project(path)
        except Exception as e:
            messagebox.showerror("Open failed", f"{path} is not a valid RSSILab project.\n{e}")
            return
        self.scn = scn
        self.sel = None
        settings = self._upgrade_settings(settings)
        choices = {"s_preset": presets.SCENARIO_PRESETS, "t_preset": PRESETS, "t_fading": FADING_MODELS,
                   "tr_mode": ("Positioning", "Tracking"), "tr_src": POINT_SOURCES, "tr_kind": TRAJ_TYPES,
                   "l_calib": ("Oracle", "Manual"), "l_kfbase": ALGORITHMS, "l_family": APPROACHES,
                   "o_mat": MATERIAL_TABLE, "draw_mat": MATERIAL_TABLE,
                   "tool": [t for _, t in TOOLS]}
        renamed = dict(presets.OLD_SCENARIO_PRESET_NAMES, **{"Legacy (receive_RSS.py)": LEGACY_PRESET})
        for k, val in settings.items():
            if k not in self.v or k == "t_fading":        # the fading model comes with the technology
                continue
            if k in ("s_preset", "t_preset"):
                val = renamed.get(val, val)
            if k in choices and val not in choices[k]:
                continue                                  # value from another version: keep the default
            try:
                self.v[k].set(val)
            except tk.TclError:
                pass
        if tech is not None:
            self._load_tech(tech)
        self._scn_to_fields()
        self._mode_changed(redraw=False)
        self._family_changed()
        self._hint(TOOL_HINTS.get(self.S("tool"), ""))
        self.results = None
        self._invalidate()
        self._refresh_lists()
        self.draw_map()
        self._log(f"Project opened: {path}")

    @staticmethod
    def _upgrade_settings(settings: dict) -> dict:
        """Settings of projects saved before version 1.0: method names and the WKNN field."""
        out = {}
        for k, val in settings.items():
            if k.startswith("alg::"):
                k = "alg::" + canonical(k[5:])
            elif k == "l_kfbase":
                val = canonical(val)
            out.setdefault(k, val)
        if "l_family" not in settings:
            out["l_family"] = BOTH                                    # all the selected methods run
            if "l_knnk" in settings:                                  # it was the WKNN neighbours
                out["l_wknnk"] = out.pop("l_knnk")
        return out

    def export_dataset(self):
        if self._worker is not None:
            return
        try:
            self.apply_geometry(silent=True)
            t = self.read_tech()
            sp, ns, seed = self.F("x_spacing"), self.I("x_samples"), self.I("x_seed")
            if sp <= 0 or ns < 1:
                raise ValueError("Spacing must be positive and samples per point >= 1.")
            if len(self.scn.anchor_xy()) == 0:
                raise ValueError("Add at least one enabled anchor.")
        except ValueError as e:
            messagebox.showerror("Check the parameters", str(e))
            return
        x0, y0, x1, y1 = self.scn.area_bounds()
        n_pts = (int((x1 - x0) / sp + 1e-9) + 1) * (int((y1 - y0) / sp + 1e-9) + 1)
        rows = n_pts * ns
        n_anc = len(self.scn.anchor_xy())
        if rows > 5_000_000 and not messagebox.askyesno(
                "Large dataset", f"The file will have {rows:,} rows (about {rows * 9 * (n_anc + 2) / 1e9:.1f} GB). "
                                 f"It is written in the background; Stop interrupts it. Continue?"):
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="Dataset_simulated.csv",
                                            filetypes=[("CSV", "*.csv")])
        if not path:
            return
        scn, skip = copy.deepcopy(self.scn), self.B("x_skip")
        self._cancel = False
        self.btn_run.state(["disabled"])
        self.btn_stop.state(["!disabled"])
        self.progress["value"] = 0
        q = queue.Queue()

        def work():
            try:
                n = engine.export_dataset_csv(path, scn, t, sp, ns, seed, lambda f, m: q.put(("p", f, m)),
                                              skip_obstacles=skip, cancel=lambda: self._cancel)
                q.put(("done", n))
            except InterruptedError:
                q.put(("stopped",))
            except Exception as e:  # disk full, permissions...
                q.put(("err", str(e)))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()
        self._poll_export(q, path, n_anc)

    def _poll_export(self, q, path, n_anc):
        try:
            while True:
                msg = q.get_nowait()
                if msg[0] == "p":
                    self.progress["value"] = msg[1]
                    self._hint("Writing the dataset: " + msg[2])
                    continue
                self._worker = None
                self.btn_run.state(["!disabled"])
                self.btn_stop.state(["disabled"])
                self.progress["value"] = 1.0 if msg[0] == "done" else 0.0
                if msg[0] == "done":
                    self._hint("Dataset written.")
                    self._log(f"Dataset written: {msg[1]:,} rows, {n_anc} anchors -> {path}")
                elif msg[0] == "stopped":
                    self._hint("Dataset export stopped: the incomplete file was removed.")
                else:
                    messagebox.showerror("Export failed", msg[1])
                return
        except queue.Empty:
            pass
        self.root.after(80, self._poll_export, q, path, n_anc)

    def export_results(self):
        if self.results is None:
            messagebox.showinfo("Export results", "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="rssilab_results.csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            try:
                engine.export_results_csv(path, self.results)
                self._log(f"Results written: {path}")
            except Exception as e:
                messagebox.showerror("Export failed", str(e))

    def export_stats(self):
        if self.results is None:
            messagebox.showinfo("Export statistics", "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="rssilab_statistics.csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            try:
                engine.export_stats_csv(path, self.results)
                self._log(f"Statistics written (overall, per point / per trial): {path}")
            except Exception as e:
                messagebox.showerror("Export failed", str(e))

    def save_figure(self):
        figs = {"Map": self.fig_map, "Error CDF": self.fig_cdf, "Error detail": self.fig_time,
                "3D view": self.fig_3d}
        fig = figs[self.plots.tab(self.plots.select(), "text")]
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if path:
            fig.savefig(path, dpi=200, bbox_inches="tight")
            self._log(f"Figure saved: {path}")

    def show_help(self):
        """The quick guide, in a scrollable window."""
        win = tk.Toplevel(self.root)
        win.title("RSSILab quick guide")
        win.transient(self.root)
        frm = ttk.Frame(win, padding=8)
        frm.pack(fill="both", expand=True)
        txt = tk.Text(frm, wrap="word", width=84, height=30, padx=10, pady=8, relief="flat",
                      highlightthickness=0, font="TkDefaultFont")
        sb = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)
        txt.tag_configure("head", font=self.font_big, spacing1=8, spacing3=2)
        for title, body in QUICK_GUIDE:
            txt.insert("end", title + "\n", "head")
            txt.insert("end", body + "\n")
        txt.configure(state="disabled")
        bar = ttk.Frame(frm)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Button(bar, text="Open the user guide", command=self.open_guide).pack(side="left")
        ttk.Button(bar, text="Close", command=win.destroy).pack(side="right")
        win.bind("<Escape>", lambda e: win.destroy())
        return win

    def open_guide(self, url=GUIDE_URL):
        """Opens a guide of the docs folder (default: docs/GUIDE_gui.md) on GitHub in the web browser."""
        try:
            opened = webbrowser.open(url)
        except Exception:  # no usable browser
            opened = False
        if not opened:
            messagebox.showinfo("Documentation", f"The guides are in the docs folder of RSSILab "
                                                 f"and online at:\n{url}")

    def show_about(self):
        messagebox.showinfo("About RSSILab", (
            f"{APP_NAME} {__version__}\n\nGPS-free, RSSI-based indoor positioning and tracking research.\n\n"
            f"Giovanni Pettorru\nMIT License\n{REPO_URL}"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rssilab", description=f"{APP_NAME}: GPS-free, RSSI-based indoor "
                                 "positioning and tracking (graphical interface).")
    ap.add_argument("--plugin", action="append", default=[], metavar="FILE",
                    help="Python file with positioning methods registered with @register "
                         "(repeatable; the files of the plugins folder of the repository and "
                         "user_algorithms.py in the current folder are always loaded)")
    ap.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    args = ap.parse_args(argv)
    plugins = engine.plugin_candidates() + [p for p in args.plugin]
    root = tk.Tk()
    RSSILabApp(root, plugins=plugins)
    root.mainloop()


if __name__ == "__main__":
    main()
