"""Simulation pipeline (GUI-independent: usable from scripts and notebooks).

    from rssilab import engine, presets
    from rssilab.trajectory import TrajParams
    scn, tech = presets.default_scenario(), presets.technology("BLE 2.4 GHz")
    res = engine.run(scn, tech, TrajParams(mode="Positioning"), engine.LocParams(mc_runs=50))
    res.summary(); res.point_stats("WLS multilateration"); res.trial_stats("Random forest")
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import sys
import time
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import metrics
from .channel import Environment, average_fixes, effective_sigma, fading_stats
from .fingerprinting import build_survey, preload
from .geometry import inside_obstacles
from .localization import (ALGORITHMS, Context, EstimatorModel, canonical, family_of, kalman_cv,
                           particle_filter)
from .model import Scenario, Technology
from .trajectory import MODES, POINT_SOURCES, TrajParams, build_route, sample_route, static_points

PF_NAME = "PF (RSS, map-aware)"       # name of the particle filter in the results
PF_NAME_PLAIN = "PF (RSS)"            # ... when the estimator is not map-aware
TRACKING = "Tracking"                 # family of the Kalman and particle filters in the results
PLUGIN_FILE = "user_algorithms.py"   # loaded from the current folder
PLUGIN_DIR = "plugins"                # every .py file in it is loaded, in a clone of the repository
CALIBRATIONS = ("Oracle", "Manual")   # estimator calibration: true channel or typed values
DEFAULT_METHODS =("WLS multilateration", "NLS multilateration", "MLE (Levenberg-Marquardt)",
                   "k-NN", "Random forest")


@dataclass
class LocParams:
    algorithms: List[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    # distance-based methods
    wcl_k: int = 4                # weighted centroid: strongest anchors used
    wcl_g: float = 1.0            # weighted centroid: weight exponent
    ml_res: float = 0.5           # MLE grid search (and PF) grid resolution (m)
    map_aware: bool = True        # the estimator knows the floor plan (MLE grid search, PF)
    # fingerprinting: survey and models
    rp_spacing: float = 2.0       # survey grid spacing = resolution of the radio map (m)
    rp_samples: int = 20          # fingerprints recorded at each reference point
    knn_k: int = 10               # k-NN: neighbours among all the fingerprints
    knn_weighted: bool = True     # k-NN: weights 1/distance (else uniform)
    wknn_k: int = 4               # WKNN: neighbours among the reference points (radio map)
    dt_min_leaf: int = 5          # decision tree: minimum fingerprints per leaf
    rf_trees: int = 100           # random forest: number of trees
    mlp_hidden: Tuple[int, ...] = (64, 64)   # neural network: units of the hidden layers
    # all methods
    z_res: float = 0.5            # 3D: vertical spacing of the MLE grid and of the survey layers (m)
    clip_to_area: bool = True
    # trackers
    kf_enabled: bool = True
    kf_base: str = "MLE (Levenberg-Marquardt)"
    kf_q: float = 0.5             # acceleration std (m/s^2)
    kf_r: float = 0.0             # measurement std (m), 0 = estimated from data
    pf_enabled: bool = False
    pf_n: int = 1000
    pf_acc: float = 0.5           # acceleration std of the motion model (m/s^2)
    pf_jitter: float = 0.0        # extra position noise per fix (m), 0 = 0.1 * speed * dt
    pf_sigma: float = 0.0         # likelihood std (dB), 0 = model sigma
    pf_inject: float = 0.02       # fraction of particles redrawn at random each fix
    calib: str = "Oracle"         # "Oracle" or "Manual"
    p0_est: float = -40.0
    n_est: float = 4.0
    sigma_est: float = 3.0
    mc_runs: int = 1              # trials (Monte Carlo repetitions)
    seed: int = 1
    redraw_env: bool = True       # new shadowing realisation at every run


@dataclass
class Results:
    """mode is "positioning" (static test points) or "tracking" (moving target).
    Positions are always 3D: gt (F, 3), est[name] (R trials, F fixes, 3); in 2D
    the height is the known tag height. err[name] (R, F) is the error in the
    estimated dimensions (horizontal in 2D, 3D in 3D); err_h / err_v are its
    horizontal and vertical parts. In positioning the fixes are ordered point by
    point: fix f belongs to point point_id[f]. families[name] is "Distance-based",
    "Fingerprinting" or "Tracking"; timing_ms[name] is the time per fix and
    train_s[name] the training time per survey (fingerprinting)."""
    mode: str
    names: List[str]
    t: np.ndarray
    gt: np.ndarray
    est: Dict[str, np.ndarray]
    err: Dict[str, np.ndarray]
    rss: np.ndarray
    route: Optional[np.ndarray]
    grid_info: Optional[dict]
    timing_ms: Dict[str, float]
    anchors: np.ndarray
    anchor_names: List[str]
    model: EstimatorModel
    tech_name: str
    dt: float
    notes: List[str]
    points: Optional[np.ndarray] = None      # (P, 3) test points (positioning)
    fixes_per_point: int = 1
    point_id: Optional[np.ndarray] = None    # (F,)
    dim: int = 2
    err_h: Optional[Dict[str, np.ndarray]] = None
    err_v: Optional[Dict[str, np.ndarray]] = None
    families: Dict[str, str] = field(default_factory=dict)
    train_s: Dict[str, float] = field(default_factory=dict)
    survey_points: Optional[np.ndarray] = None   # (R, 3) reference points of the (last) survey

    @property
    def trials(self) -> int:
        return self.rss.shape[0]

    def summary(self) -> Dict[str, dict]:
        """Overall statistics; in 3D also rmse_xy (horizontal) and rmse_z (vertical)."""
        out = {}
        for n in self.names:
            s = metrics.summarize(self.err[n])
            s["rmse_xy"] = metrics.summarize(self.err_h[n])["rmse"]
            s["rmse_z"] = metrics.summarize(self.err_v[n])["rmse"] if self.dim == 3 else float("nan")
            out[n] = s
        return out

    def point_stats(self, name: str) -> dict:
        """Per test point statistics over trials x fixes (positioning only)."""
        return metrics.point_stats(self.est[name], self.points, self.fixes_per_point, self.dim)

    def trial_stats(self, name: str) -> dict:
        out = metrics.trial_stats(self.err[name])
        out["rmse_xy"] = metrics.trial_stats(self.err_h[name])["rmse"]
        out["rmse_z"] = (metrics.trial_stats(self.err_v[name])["rmse"] if self.dim == 3
                         else np.full(self.trials, np.nan))
        return out


# ---------------------------------------------------------------------------
def plugin_candidates() -> List[str]:
    """Plugin files loaded by default, in this order: every .py file of the plugins/
    folder of the RSSILab folder (names starting with '_' are skipped), found next
    to the rssilab package (run_rssilab.py, editable install) or in the current
    folder when it is the RSSILab folder (it contains run_rssilab.py);
    user_algorithms.py in the current folder; the files listed in the
    RSSILAB_PLUGINS environment variable (separated by os.pathsep; a listed file
    that does not exist is reported by load_plugins)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out: List[str] = []

    def add(path: str) -> None:
        path = os.path.realpath(os.path.expanduser(path))
        if path not in out:
            out.append(path)

    for base in (root, os.getcwd()):
        folder = os.path.join(base, PLUGIN_DIR)
        if os.path.isfile(os.path.join(base, "run_rssilab.py")) and os.path.isdir(folder):   # the RSSILab folder
            for name in sorted(os.listdir(folder)):
                if name.endswith(".py") and not name.startswith("_") and os.path.isfile(os.path.join(folder, name)):
                    add(os.path.join(folder, name))
    if os.path.isfile(os.path.join(os.getcwd(), PLUGIN_FILE)):
        add(os.path.join(os.getcwd(), PLUGIN_FILE))
    for c in os.environ.get("RSSILAB_PLUGINS", "").split(os.pathsep):
        if c.strip():
            add(c.strip())
    return out


def load_plugins(paths: Optional[Sequence[str]] = None) -> List[str]:
    """Runs plugin files: every function decorated with @register in them becomes
    a positioning method (GUI included). paths=None loads plugin_candidates().
    Returns log messages; a broken plugin is reported, not raised."""
    msgs, seen = [], set()
    for path in (plugin_candidates() if paths is None else list(paths)):
        path = os.path.realpath(os.path.expanduser(path))
        if path in seen:
            continue
        seen.add(path)
        before = {k: id(v) for k, v in ALGORITHMS.items()}
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError("file not found")
            modname = "rssilab_plugin_" + hashlib.md5(path.encode()).hexdigest()[:10]
            spec = importlib.util.spec_from_file_location(modname, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[modname] = mod
            spec.loader.exec_module(mod)
        except Exception as e:  # report plugin errors without breaking the app
            msgs.append(f"Plugin {path} not loaded: {type(e).__name__}: {e}")
            continue
        new = [k for k, v in ALGORITHMS.items() if before.get(k) != id(v)]
        msgs.append(f"Plugin {os.path.basename(path)} loaded: " + (", ".join(new) if new else "no new method"))
    return msgs


def estimator_model(scn: Scenario, tech: Technology, lp: LocParams) -> EstimatorModel:
    if lp.calib == "Manual":
        return EstimatorModel(lp.p0_est, lp.n_est, tech.d0_m, lp.sigma_est)
    fm, _ = fading_stats(tech)
    return EstimatorModel(tech.p0() + fm, tech.path_loss_exp, tech.d0_m, effective_sigma(tech))


def _robust_std_from_steps(z: np.ndarray) -> float:
    dz = np.diff(z, axis=0)
    dz = dz[np.isfinite(dz).all(1)]
    if len(dz) < 3:
        return 3.0
    return float(max(0.3, 1.4826 * np.median(np.abs(dz)) / np.sqrt(2)))


def _stream(seed: int, trial: int, *tags: int) -> np.random.Generator:
    """Random stream of one trial; tags separate independent uses (survey, method, filter)."""
    return np.random.default_rng([seed, trial + 1, *tags])


def run(scn: Scenario, tech: Technology, tp: TrajParams, lp: LocParams,
        progress: Optional[Callable[[float, str], None]] = None,
        cancel: Optional[Callable[[], bool]] = None) -> Results:
    """Runs a Monte Carlo experiment and returns its Results.

    Every trial draws new RSS readings (and, with lp.redraw_env, a new shadowing
    field and fingerprint survey, on which the fingerprinting models are trained
    again) from the seed sequence (lp.seed, trial), so a run is reproducible. All
    the methods see the same RSS; each has its own random stream, so adding or
    removing a method does not change the results of the others.
    progress(fraction, message) is called along the way; cancel() returning True
    stops the run with InterruptedError. Invalid settings raise ValueError with a
    message meant for the user."""
    prog = progress or (lambda f, m: None)
    if tp.mode not in MODES:
        raise ValueError(f"Unknown experiment mode {tp.mode!r}: use 'Positioning' or 'Tracking'.")
    if tp.positioning and tp.point_source not in POINT_SOURCES:
        raise ValueError(f"Unknown test point source {tp.point_source!r}: use "
                         + ", ".join(map(repr, POINT_SOURCES)) + ".")
    if lp.calib not in CALIBRATIONS:
        raise ValueError(f"Unknown estimator calibration {lp.calib!r}: use 'Oracle' or 'Manual'.")
    anchors = scn.anchor_xyz()
    D = 3 if scn.dim == 3 else 2
    if len(anchors) < D + 1:
        raise ValueError(f"At least {D + 1} enabled anchors are needed in {D}D.")
    notes: List[str] = []
    if D == 3 and np.ptp(anchors[:, 2]) < 0.05:
        notes.append("3D with all anchors at the same height: z is weakly observable "
                     "(see the RMSE z column); put some anchors at other heights.")
    rng0 = np.random.default_rng(lp.seed)
    Ns = max(1, int(tech.samples_per_fix))
    dt = Ns / max(tech.sample_rate_hz, 1e-9)

    prog(0.0, "Building the experiment")
    route, ginfo, points, point_id, K = None, None, None, None, 1
    if tp.positioning:
        mode = "positioning"
        points, ginfo = static_points(scn, tp, rng0, notes)
        K = max(1, int(tp.fixes_per_point))
        gt = np.repeat(points, K, axis=0)
        point_id = np.repeat(np.arange(len(points)), K)
        F = len(gt)
        sample_pts = np.repeat(gt, Ns, axis=0)
        t_fix = np.arange(F, dtype=float)
    else:
        mode = "tracking"
        route = build_route(scn, tp, rng0, notes)
        pos, t = sample_route(route, tp.speed, tech.sample_rate_hz)
        F = len(pos) // Ns
        if F < 2:
            raise ValueError("The route is too short for the chosen sample rate and samples per fix.")
        pos = pos[:F * Ns].reshape(F, Ns, 3)
        gt = pos.mean(axis=1)                          # truth = centroid of the readings of the fix
        t_fix = t[:F * Ns].reshape(F, Ns).mean(axis=1)  # ... i.e. the middle of the averaging window
        sample_pts = pos.reshape(-1, 3)

    model = estimator_model(scn, tech, lp)
    params = dict(wcl_k=lp.wcl_k, wcl_g=lp.wcl_g, ml_res=lp.ml_res, z_res=lp.z_res, map_aware=lp.map_aware,
                  knn_k=lp.knn_k, knn_weighted=lp.knn_weighted, wknn_k=lp.wknn_k, dt_min_leaf=lp.dt_min_leaf,
                  rf_trees=lp.rf_trees, mlp_hidden=tuple(lp.mlp_hidden))
    ctx = Context(scn, tech, model, params, rng0)

    def check_cancel():
        if cancel is not None and cancel():
            raise InterruptedError("Simulation stopped.")
    ctx.cancel = check_cancel

    algos = list(dict.fromkeys(canonical(a) for a in lp.algorithms))
    unknown = [a for a in algos if a not in ALGORITHMS]
    if unknown:
        raise ValueError(f"Unknown positioning method(s): {', '.join(map(repr, unknown))}. "
                         f"Available: {', '.join(ALGORITHMS)}.")
    kf_name, kf_base = None, canonical(lp.kf_base)
    if mode == "tracking" and lp.kf_enabled:
        if kf_base not in ALGORITHMS:
            raise ValueError(f"Unknown input method for the Kalman filter: {lp.kf_base!r}.")
        kf_name = f"KF + {kf_base}"
    pf_on = mode == "tracking" and lp.pf_enabled
    pf_name = PF_NAME if lp.map_aware else PF_NAME_PLAIN
    if mode == "positioning" and lp.pf_enabled:
        notes.append("The Kalman and particle filters are trackers: they run only in Tracking mode.")
    to_compute = list(dict.fromkeys(algos + ([kf_base] if kf_name else [])))
    names = algos + ([kf_name] if kf_name else []) + ([pf_name] if pf_on else [])
    if not names:
        raise ValueError("Select at least one positioning method.")
    families = {a: family_of(a) for a in algos}
    families.update({n: TRACKING for n in names if n not in families})
    need_fp = any("fingerprints" in getattr(ALGORITHMS[a], "needs", ()) for a in to_compute)
    if need_fp and (lp.rp_spacing <= 0 or lp.rp_samples < 1):
        raise ValueError("The survey grid spacing must be positive and the fingerprints per point >= 1.")
    preload(to_compute)

    R = max(1, int(lp.mc_runs))
    A = len(anchors)
    est = {n: np.full((R, F, 3), np.nan) for n in names}
    rss_all = np.full((R, F, A), np.nan, dtype=np.float32)
    timing = {n: 0.0 for n in names}
    train = {n: 0.0 for n in names}

    prog(0.02, "Computing mean RSS")
    mu = None
    env, survey, n_surveys = None, None, 0
    lo3, hi3 = np.array(scn.area_bounds3()[:3]), np.array(scn.area_bounds3()[3:])
    ax0, ay0, ax1, ay1 = scn.area_bounds()          # region covered by the shadowing field
    field_bounds = (min(ax0, sample_pts[:, 0].min()), min(ay0, sample_pts[:, 1].min()),
                    max(ax1, sample_pts[:, 0].max()), max(ay1, sample_pts[:, 1].max()))
    n_missing = 0
    for run_i in range(R):
        check_cancel()
        rng = _stream(lp.seed, run_i)
        if env is None or lp.redraw_env:
            env = Environment(scn, tech, rng, bounds=field_bounds)
            survey = None
            if env.note and env.note not in notes:
                notes.append(env.note)
        if mu is None:
            mu = env.mean(sample_pts)
        raw = env.measure(sample_pts, rng, mean=mu).reshape(F, Ns, A)
        n_missing += int(np.isnan(raw).sum())
        rss = average_fixes(raw)
        rss_all[run_i] = rss
        if need_fp and survey is None:
            prog((run_i + 0.1) / R, "Fingerprint survey")
            survey = build_survey(scn, env, lp.rp_spacing, lp.rp_samples, _stream(lp.seed, run_i, 1),
                                  Ns, lp.z_res, check=check_cancel)
            ctx.set_survey(survey)
            n_surveys += 1

        local, spent, trained = {}, {}, {}
        for i, a in enumerate(to_compute):
            check_cancel()
            prog((run_i + 0.2 + 0.7 * i / len(to_compute)) / R, f"Trial {run_i + 1}/{R}: {a}")
            ctx.rng = _stream(lp.seed, run_i, 2, zlib.crc32(a.encode()))
            ctx.train_time = 0.0
            t0 = time.perf_counter()
            e = np.asarray(ALGORITHMS[a](rss, ctx), dtype=float)
            elapsed = time.perf_counter() - t0
            if e.shape[0] != F:
                raise RuntimeError(f"'{a}' returned {e.shape[0]} estimates for {F} fixes.")
            e = ctx.lift(e.reshape(F, -1))
            if D == 2:
                e[:, 2] = scn.tag_h
            if lp.clip_to_area:
                e = np.clip(e, lo3, hi3)
            local[a] = e
            trained[a] = ctx.train_time
            spent[a] = (elapsed - ctx.train_time) * 1e3
            if a in est:
                est[a][run_i] = e
                timing[a] += spent[a]
                train[a] += trained[a]
        if kf_name:
            t0 = time.perf_counter()
            z = local[kf_base][:, :D]
            r_std = lp.kf_r if lp.kf_r > 0 else _robust_std_from_steps(z)
            kf = ctx.lift(kalman_cv(z, dt, lp.kf_q, r_std))
            if D == 2:
                kf[:, 2] = scn.tag_h
            est[kf_name][run_i] = np.clip(kf, lo3, hi3) if lp.clip_to_area else kf
            # the KF time includes its input method when that one is not among the selected methods
            timing[kf_name] += (time.perf_counter() - t0) * 1e3 + (0.0 if kf_base in est else spent[kf_base])
            train[kf_name] += 0.0 if kf_base in est else trained[kf_base]
            if run_i == 0 and lp.kf_r <= 0:
                notes.append(f"KF measurement std estimated from data: {r_std:.2f} m")
        if pf_on:
            check_cancel()
            prog((run_i + 0.9) / R, f"Trial {run_i + 1}/{R}: particle filter")
            t0 = time.perf_counter()
            jit = lp.pf_jitter if lp.pf_jitter > 0 else max(0.02, 0.1 * tp.speed * dt)
            sig = lp.pf_sigma if lp.pf_sigma > 0 else max(model.sigma, 1.0)
            est[pf_name][run_i] = ctx.lift(particle_filter(
                rss, ctx, max(50, lp.pf_n), dt, lp.pf_acc, jit, max(tp.speed, 0.1), sig, _stream(lp.seed, run_i, 3),
                inject=max(0.0, min(lp.pf_inject, 0.5))))
            timing[pf_name] += (time.perf_counter() - t0) * 1e3
        prog((run_i + 1) / R, f"Trial {run_i + 1}/{R} done")

    diff = {n: est[n] - gt[None] for n in names}
    err = {n: np.sqrt((diff[n][..., :D] ** 2).sum(-1)) for n in names}
    err_h = {n: np.hypot(diff[n][..., 0], diff[n][..., 1]) for n in names}
    err_v = {n: np.abs(diff[n][..., 2]) for n in names}
    timing = {n: v / (R * F) for n, v in timing.items()}
    train_s = {n: v / max(n_surveys, 1) for n, v in train.items()}
    if survey is not None:
        nrp = len(survey.points)
        notes.append(f"Fingerprint survey: {nrp} reference points every {lp.rp_spacing:g} m x "
                     f"{max(1, int(lp.rp_samples))} fingerprints = {len(survey.fingerprints)} training "
                     f"fingerprints" + (f", repeated for each of the {n_surveys} shadowing realisations."
                                        if n_surveys > 1 else "."))
        slow = [f"{n} {v:.2f} s" for n, v in train_s.items() if v >= 0.005]
        if slow:
            notes.append("Training time per survey: " + ", ".join(slow) + ".")
        if mode == "positioning":
            rps = {tuple(q) for q in np.round(survey.points[:, :D], 6)}
            same = sum(tuple(q) in rps for q in np.round(points[:, :D], 6))
            if same:
                notes.append(f"{same} of the {len(points)} test points coincide with survey reference points: "
                             f"fingerprinting errors there are optimistic (use test points off the survey grid, "
                             f"e.g. a grid with the survey spacing, or random points).")
    if n_missing:
        notes.append(f"{100.0 * n_missing / (R * F * Ns * A):.1f}% of the RSS readings were below the "
                     f"sensitivity; {100.0 * float(np.mean(np.isnan(rss_all))):.1f}% of the per-anchor "
                     f"averages of the fixes had no reading (missing RSS).")
    return Results(mode, names, t_fix, gt, est, err, rss_all, route, ginfo, timing,
                   anchors, scn.anchor_names(), model, tech.name, dt, notes,
                   points=points, fixes_per_point=K, point_id=point_id, dim=D,
                   err_h=err_h, err_v=err_v, families=families, train_s=train_s,
                   survey_points=None if survey is None else survey.points)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
def export_dataset_csv(path: str, scn: Scenario, tech: Technology, spacing: float,
                       samples: int, seed: int = 1, progress=None, skip_obstacles: bool = False,
                       cancel: Optional[Callable[[], bool]] = None) -> int:
    """Radio-map dataset with the columns Coord_X, Coord_Y, Anc1_RSS..AncN_RSS (the
    format of the original RSSILab dataset), on a grid over the target area at the
    tag height, `samples` readings per point; in 3D a Coord_Z column is added.
    Missing readings (below sensitivity) are written as nan. skip_obstacles drops
    the grid points that fall inside motion-blocking obstacles. Rows are written
    as they are generated, so the file size is not limited by memory. cancel()
    returning True stops the export (InterruptedError) and removes the partial file.
    Returns the number of data rows."""
    if spacing <= 0 or samples < 1:
        raise ValueError("Spacing must be positive and samples per point >= 1.")
    rng = np.random.default_rng(seed)
    env = Environment(scn, tech, rng)
    A = len(env.anchors)
    if A == 0:
        raise ValueError("Add at least one enabled anchor.")
    x0, y0, x1, y1 = scn.area_bounds()
    xs = np.arange(x0, x1 + 1e-9, spacing)
    ys = np.arange(y0, y1 + 1e-9, spacing)
    three = scn.dim == 3
    header = ["Coord_X", "Coord_Y"] + (["Coord_Z"] if three else []) + [f"Anc{i + 1}_RSS" for i in range(A)]
    n_rows = 0
    try:
        with open(path, "w", newline="") as f:
            f.write(",".join(header) + "\n")
            for k, y in enumerate(ys):
                if cancel is not None and cancel():
                    raise InterruptedError("Dataset export stopped.")
                pts = np.column_stack([xs, np.full_like(xs, y), np.full_like(xs, scn.tag_h)])
                if skip_obstacles and scn.obstacles:
                    pts = pts[~inside_obstacles(pts, scn.obstacles, room_z=scn.room_z)]
                if len(pts):
                    rep = np.repeat(pts, samples, axis=0)
                    rss = env.measure(rep, rng)
                    coords = rep if three else rep[:, :2]
                    np.savetxt(f, np.column_stack([coords, rss]), delimiter=",", fmt="%.6g")
                    n_rows += len(rep)
                if progress:
                    progress((k + 1) / len(ys), f"row {k + 1}/{len(ys)} of the grid")
    except InterruptedError:
        os.remove(path)
        raise
    return n_rows


def export_results_csv(path: str, res: Results) -> None:
    """One row per trial and fix: truth, RSS per anchor, estimate and error per method
    (in 3D also z, horizontal and vertical error)."""
    R, F, A = res.rss.shape
    three = res.dim == 3
    cols = ["trial", "fix", "point" if res.mode == "positioning" else "t", "x_true", "y_true", "z_true"] + \
        [f"RSS_{n}" for n in res.anchor_names]
    for n in res.names:
        cols += [f"{n}_x", f"{n}_y"] + ([f"{n}_z", f"{n}_err", f"{n}_err_xy", f"{n}_err_z"] if three
                                        else [f"{n}_err"])
    blocks = []
    for r in range(R):
        third = res.point_id + 1 if res.mode == "positioning" else res.t
        b = [np.full(F, r + 1), np.arange(F) + 1, third, res.gt[:, 0], res.gt[:, 1], res.gt[:, 2]]
        b += [res.rss[r, :, a] for a in range(A)]
        for n in res.names:
            b += [res.est[n][r, :, 0], res.est[n][r, :, 1]]
            b += ([res.est[n][r, :, 2], res.err[n][r], res.err_h[n][r], res.err_v[n][r]] if three
                  else [res.err[n][r]])
        blocks.append(np.column_stack(b))
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(cols)
        np.savetxt(f, np.vstack(blocks), delimiter=",", fmt="%.6g")


def export_stats_csv(path: str, res: Results) -> None:
    """Statistics in long format: one row per method and level (overall, per test
    point in Positioning, per trial in both modes); the overall rows also give the
    time per fix and the training time per survey."""
    keys = list(metrics.STAT_KEYS) + ["rmse_xy", "rmse_z"]
    pkeys = ["x", "y", "z", "est_x", "est_y", "est_z", "bias", "spread"]
    tkeys = ["ms_per_fix", "train_s"]
    nan = float("nan")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "method", "family", "id"] + pkeys + keys + tkeys)

        def row(level, name, idx, d, extra=None, times=None):
            ex = [extra.get(k, nan) if extra else nan for k in pkeys]
            tm = [times.get(k, nan) if times else nan for k in tkeys]
            w.writerow([level, name, res.families.get(name, ""), idx] + [f"{v:.6g}" for v in ex] +
                       [f"{d.get(k, nan):.6g}" for k in keys] + [f"{v:.6g}" for v in tm])

        summ = res.summary()
        for n in res.names:
            row("overall", n, "all", summ[n],
                times=dict(ms_per_fix=res.timing_ms.get(n, nan), train_s=res.train_s.get(n, nan)))
            if res.mode == "positioning":
                ps = res.point_stats(n)
                for i in range(len(res.points)):
                    row("point", n, f"P{i + 1}", {k: ps[k][i] for k in keys if k in ps},
                        {k: ps[k][i] for k in pkeys})
            ts = res.trial_stats(n)
            for r in range(res.trials):
                row("trial", n, r + 1, {k: ts[k][r] for k in keys if k in ts})
