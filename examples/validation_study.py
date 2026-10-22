"""Validation study: checks RSSILab against known results of RSS-based positioning.

    python examples/validation_study.py      (from the repository folder, about a minute)

Writes the figures of docs/VALIDATION.md to docs/images/ and prints the tables.
All experiments use fixed seeds: the numbers are reproducible.
"""
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                                 # run from a clone

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.ticker import NullFormatter              # noqa: E402

from rssilab import channel, engine, presets            # noqa: E402
from rssilab.localization import DISTANCE, methods      # noqa: E402
from rssilab.model import LEGACY_FADING, Anchor, Scenario, Technology   # noqa: E402
from rssilab.trajectory import TrajParams                # noqa: E402

IMG = os.path.join(ROOT, "docs", "images")
os.makedirs(IMG, exist_ok=True)

# chart style: light surface, recessive axes, fixed categorical order
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1", "#d1477a"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 1.8, "lines.markersize": 6,
})
T0 = time.time()


def noiseless(**kw):
    d = dict(name="validation", p0_auto=False, p0_dbm=-40.0, path_loss_exp=3.0, shadow_sigma_db=0.0,
             shadow_corr_m=0.0, fading="None", quantization_db=0.0, sensitivity_dbm=-200.0, samples_per_fix=1)
    d.update(kw)
    return Technology(**d)


def open_room(n_anchors=12, dim=2):
    s = Scenario(room_w=20, room_h=20, area_w=12, area_h=12)
    s.place_anchors_on_perimeter(n_anchors)
    s.test_points = [(0.0, 0.0), (3.0, 2.0), (-4.0, 4.0), (5.0, -5.0), (-2.0, -3.0)]
    if dim == 3:                                         # anchors alternating near the ceiling and the floor
        s.dim, s.room_z, s.area_z0, s.area_z1 = 3, 3.0, 0.2, 2.5
        for i, a in enumerate(s.anchors):
            a.z = 2.8 if i % 2 == 0 else 0.3
        s.test_points = [(x, y, z) for (x, y), z in zip(s.test_points, (1.0, 2.0, 0.5, 1.5, 2.2))]
    return s


def table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print()


def run_open(tech, algos, K, scn=None, **lp):
    d = dict(algorithms=algos, clip_to_area=False, seed=3, ml_res=0.1)
    d.update(lp)
    return engine.run(scn or open_room(), tech, TrajParams(mode="Positioning", fixes_per_point=K),
                      engine.LocParams(**d))


def log_axes(ax, xt, yt):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(xt); ax.set_xticklabels([f"{v:g}" for v in xt]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt]); ax.yaxis.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------------------
# 1. Distance-based methods without noise: exact solutions
# ---------------------------------------------------------------------------
print("## 1. Distance-based methods without noise (reference room, 12 anchors on the walls, test grid every "
      "2 m over the whole target area; max error in m)\n")
dist = methods(DISTANCE)
exact = {}
for dim in (2, 3):
    scn = presets.default_scenario()
    if dim == 3:                                          # anchors alternating near the ceiling and the floor
        scn.dim, scn.room_z, scn.area_z0, scn.area_z1 = 3, 3.0, 0.2, 2.5
        for i, a in enumerate(scn.anchors):
            a.z = 2.8 if i % 2 == 0 else 0.3
    tp = TrajParams(mode="Positioning", point_source="Grid", grid_spacing=2.0, grid_z=1.2 if dim == 3 else None)
    res = engine.run(scn, noiseless(), tp, engine.LocParams(algorithms=dist, clip_to_area=False, ml_res=0.25,
                                                             z_res=0.25))
    exact[dim] = {a: float(np.nanmax(res.err[a])) for a in dist}
    n_points = len(res.points)
table(["method", "2D", "3D"], [[a, f"{exact[2][a]:.1e}", f"{exact[3][a]:.1e}"] for a in dist])
print(f"{n_points} test points; MLE grid cell 0.25 m\n")

# ---------------------------------------------------------------------------
# 2. Error vs shadowing std
# ---------------------------------------------------------------------------
print("## 2. RMSE (m) vs shadowing std (12 anchors, 20 x 20 m, 5 points x 1000 fixes)\n")
sigmas = [0.5, 1, 2, 3, 4, 6, 8]
algos = ["Trilateration", "LLS multilateration", "WLS multilateration", "NLS multilateration",
         "MLE (Levenberg-Marquardt)", "MLE (grid search)"]
curves = {a: [] for a in algos}
for s in sigmas:
    res = run_open(noiseless(shadow_sigma_db=s), algos, K=1000)
    for a in algos:
        curves[a].append(res.summary()[a]["rmse"])
table(["σ (dB)"] + algos, [[f"{s:g}"] + [f"{curves[a][i]:.2f}" for a in algos] for i, s in enumerate(sigmas)])
print("RMSE / σ (m/dB) at σ = 0.5, 1, 2 dB: " + "; ".join(
    f"{a}: " + ", ".join(f"{curves[a][i] / sigmas[i]:.3f}" for i in range(3)) for a in algos) + "\n")

fig, ax = plt.subplots(figsize=(6.6, 4.2))
for c, a in zip(SERIES, algos):
    ax.plot(sigmas, curves[a], "o-", color=c, label=a, mec=SURFACE, mew=1.2, ms=5)
log_axes(ax, sigmas, [0.2, 0.5, 1, 2, 5, 10, 20, 50])
ax.set_xlabel("Shadowing std σ [dB]"); ax.set_ylabel("Position RMSE [m]")
ax.set_title("Distance-based methods: error proportional to σ")
ax.legend(loc="upper left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "validation_distance.png"), dpi=160); plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Number of anchors
# ---------------------------------------------------------------------------
print("## 3. RMSE (m) vs number of anchors (σ = 4 dB, 5 points x 500 fixes)\n")
ns = [4, 6, 8, 12, 16, 24]
algos3 = ["LLS multilateration", "WLS multilateration", "NLS multilateration", "MLE (Levenberg-Marquardt)"]
by_n = {a: [] for a in algos3}
for n in ns:
    res = run_open(noiseless(shadow_sigma_db=4.0), algos3, K=500, scn=open_room(n))
    for a in algos3:
        by_n[a].append(res.summary()[a]["rmse"])
table(["anchors"] + algos3, [[str(n)] + [f"{by_n[a][i]:.2f}" for a in algos3] for i, n in enumerate(ns)])

fig, ax = plt.subplots(figsize=(6.4, 3.8))
for c, a in zip(SERIES[1:], algos3):
    ax.plot(ns, by_n[a], "o-", color=c, label=a, mec=SURFACE, mew=1.2)
ax.set_xticks(ns); ax.set_ylim(bottom=0)
ax.set_xlabel("Anchors on the walls"); ax.set_ylabel("Position RMSE [m]")
ax.set_title("More anchors, lower error (σ = 4 dB)")
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "validation_anchors.png"), dpi=160); plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Fingerprinting: resolution of the survey
# ---------------------------------------------------------------------------
print("## 4. Fingerprinting: RMSE (m) vs survey grid spacing (small room, 4 anchors, static shadowing "
      "σ = 5 dB, d_c = 3 m, Rayleigh fading, 5 readings per fix, 20 fingerprints per point)\n")
fp_algos = methods("Fingerprinting")
spacings = [0.5, 1.0, 2.0, 3.0, 4.0]
tech4 = noiseless(shadow_sigma_db=5.0, shadow_corr_m=3.0, fading="Rayleigh", samples_per_fix=5)
tp4 = TrajParams(mode="Positioning", point_source="Random points", n_random_points=40, fixes_per_point=10)
by_sp, train, n_fp = {a: [] for a in fp_algos}, {}, {}
for sp in spacings:
    res = engine.run(presets.small_room(), tech4, tp4,
                     engine.LocParams(algorithms=fp_algos, rp_spacing=sp, rp_samples=20, mc_runs=3, seed=2))
    for a in fp_algos:
        by_sp[a].append(res.summary()[a]["rmse"])
    train[sp] = res.train_s
    n_fp[sp] = len(res.survey_points) * 20
table(["spacing (m)", "training fingerprints"] + fp_algos,
      [[f"{sp:g}", f"{n_fp[sp]:,}"] + [f"{by_sp[a][i]:.2f}" for a in fp_algos] for i, sp in enumerate(spacings)])
table(["training time per survey (s)"] + [f"{sp:g} m" for sp in spacings],
      [[a] + [f"{train[sp][a]:.2f}" for sp in spacings] for a in fp_algos])

fig, ax = plt.subplots(figsize=(6.4, 3.9))
for c, a in zip(SERIES, fp_algos):
    ax.plot(spacings, by_sp[a], "o-", color=c, label=a, mec=SURFACE, mew=1.2)
ax.set_xticks(spacings); ax.set_ylim(bottom=0)
ax.set_xlabel("Survey grid spacing [m]"); ax.set_ylabel("Position RMSE [m]")
ax.set_title("A finer survey trains better fingerprinting models")
ax.legend(loc="upper left", fontsize=8.5)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "validation_survey.png"), dpi=160); plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Fingerprinting vs distance-based: i.i.d. vs static shadowing
# ---------------------------------------------------------------------------
print("## 5. Fingerprinting vs distance-based, RMSE (m) (small room, 4 anchors, σ = 5 dB, 5 readings per fix, "
      "survey every 1 m)\n")
algos5 = ["WLS multilateration", "MLE (Levenberg-Marquardt)", "MLE (grid search)", "k-NN", "Random forest"]
tp5 = TrajParams(mode="Positioning", point_source="Random points", n_random_points=40, fixes_per_point=20)
cmp = {}
for label, corr in (("i.i.d. shadowing", 0.0), ("static shadowing, d_c = 3 m", 3.0)):
    res = engine.run(presets.small_room(), noiseless(shadow_sigma_db=5.0, shadow_corr_m=corr, samples_per_fix=5),
                     tp5, engine.LocParams(algorithms=algos5, rp_spacing=1.0, rp_samples=20, mc_runs=5, seed=2))
    cmp[label] = {a: res.summary()[a]["rmse"] for a in algos5}
table(["shadowing"] + algos5, [[k] + [f"{v[a]:.2f}" for a in algos5] for k, v in cmp.items()])

fig, ax = plt.subplots(figsize=(6.8, 3.8))
x = np.arange(len(cmp)); w = 0.16
for j, (c, a) in enumerate(zip(SERIES, algos5)):
    vals = [cmp[k][a] for k in cmp]
    bars = ax.bar(x + (j - 2) * w, vals, w - 0.02, color=c, label=a)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5,
                color=INK2)
ax.set_xticks(x); ax.set_xticklabels(list(cmp)); ax.grid(axis="x", visible=False)
ax.set_ylabel("Position RMSE [m]"); ax.set_ylim(0, max(max(v.values()) for v in cmp.values()) * 1.3)
ax.set_title("A survey learns static shadowing, a path-loss model cannot")
ax.legend(loc="upper left", fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(IMG, "validation_fingerprinting.png"), dpi=160); plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Spatially correlated shadowing: empirical correlation vs the model
# ---------------------------------------------------------------------------
print("## 6. Shadowing field: correlation vs distance (20 realisations, σ = 4 dB)\n")
r = np.linspace(0, 12, 25)
emp = {}
for dc in (2.0, 5.0):
    acc = np.zeros(len(r)); sd = []
    for k in range(20):
        f = channel.ShadowingField((-60, -60, 60, 60), 1, 4.0, dc, np.random.default_rng(k))
        g = np.random.default_rng(1000 + k)
        a = g.uniform(-45, 45, (3000, 2)); th = g.uniform(0, 2 * np.pi, 3000)
        va = f.sample(a)[:, 0]; sd.append(va.std())
        for i, d in enumerate(r):
            vb = f.sample(a + d * np.column_stack([np.cos(th), np.sin(th)]))[:, 0]
            acc[i] += np.corrcoef(va, vb)[0, 1]
    emp[dc] = acc / 20
    i_dc = int(np.argmin(np.abs(r - dc)))
    print(f"d_c = {dc:g} m: marginal std {np.mean(sd):.2f} dB, "
          f"correlation at d_c {emp[dc][i_dc]:.3f} (model 1/e = 0.368)")
print()
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for c, dc in zip(SERIES, (2.0, 5.0)):
    rr = np.linspace(0, 12, 200)
    ax.plot(rr, np.exp(-(rr / dc) ** 2), "-", color=c, lw=1.2, alpha=0.7)
    ax.plot(r, emp[dc], "o", color=c, mec=SURFACE, mew=1.0, label=f"correlation distance {dc:g} m")
ax.axhline(1 / math.e, color=AXIS, lw=1.0)
ax.text(11.9, 1 / math.e + 0.02, "1/e", ha="right", va="bottom", color=INK2, fontsize=9)
ax.set_xlabel("Distance between the two positions [m]"); ax.set_ylabel("Correlation of the shadowing")
ax.set_title("Correlated shadowing: simulated (dots) vs model exp(−r²/d_c²)")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(IMG, "validation_shadowing.png"), dpi=160); plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Channel statistics: fading closed forms and the legacy generator
# ---------------------------------------------------------------------------
print("## 7. Fading per reading: Monte Carlo (4 x 10^6 samples) vs exact\n")
rows = []
for name, t in (("Rayleigh", Technology(fading="Rayleigh")),
                ("Rician K = 3 dB", Technology(fading="Rician", rician_k_db=3)),
                ("Rician K = 10 dB", Technology(fading="Rician", rician_k_db=10)),
                ("Legacy, σ = 2", Technology(fading=LEGACY_FADING, legacy_fading_sigma=2.0))):
    xs = channel.fading_db(t, np.random.default_rng(1), 4_000_000)
    m, s = channel.fading_stats(t)
    rows.append([name, f"{xs.mean():+.3f} / {m:+.3f}", f"{xs.std():.3f} / {s:.3f}"])
table(["model", "mean dB (MC / exact)", "std dB (MC / exact)"], rows)

print("## 8. Legacy preset vs the original generator Pr = -40 - 40 log10(d) + 3 N(0,1) + 2 sqrt(-2 ln U)\n")
s1 = Scenario(); s1.anchors = [Anchor("a1", 0, 0)]
env = channel.Environment(s1, presets.technology("Legacy log-normal model"), np.random.default_rng(5))
g = np.random.default_rng(7)
rows = []
for d in (2.0, 13.7, 40.0):
    ref = -40 - 40 * np.log10(d) + 3 * g.standard_normal(200_000) + 2 * np.sqrt(-2 * np.log(1 - g.random(200_000)))
    sim = env.measure(np.tile([[d, 0.0, 0.0]], (200_000, 1)), np.random.default_rng(int(d * 10)))[:, 0]
    a, b = np.sort(ref), np.sort(sim); grid = np.concatenate([a, b])
    ks = np.max(np.abs(np.searchsorted(a, grid, "right") - np.searchsorted(b, grid, "right"))) / len(a)
    rows.append([f"{d:g}", f"{ref.mean():.3f} / {sim.mean():.3f}", f"{ref.std():.3f} / {sim.std():.3f}", f"{ks:.4f}"])
table(["d (m)", "mean dBm (reference / RSSILab)", "std dB (reference / RSSILab)", "KS distance"], rows)

# ---------------------------------------------------------------------------
# 9. Tracking: filters vs snapshot estimates
# ---------------------------------------------------------------------------
print("## 9. Tracking in the demo office (BLE preset, random walk, 10 trials)\n")
res = engine.run(presets.demo_office(), presets.technology("BLE 2.4 GHz"),
                 TrajParams(mode="Tracking", kind="Random walk", n_random_wp=8),
                 engine.LocParams(algorithms=["MLE (Levenberg-Marquardt)", "MLE (grid search)", "k-NN",
                                              "Random forest"],
                                  kf_enabled=True, pf_enabled=True, mc_runs=10, seed=5))
table(["method", "RMSE (m)", "median (m)", "P90 (m)"],
      [[n, f"{s['rmse']:.2f}", f"{s['median']:.2f}", f"{s['p90']:.2f}"] for n, s in res.summary().items()])
print(f"Figures written to {IMG} ({time.time() - T0:.0f} s)")
