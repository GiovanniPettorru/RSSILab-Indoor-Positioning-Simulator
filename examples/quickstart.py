"""RSSILab from a script: distance-based and fingerprinting methods in positioning,
tracking and 3D experiments, with CSV exports.

    python examples/quickstart.py            (from the repository folder)

Results are written to examples/output/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # run from a clone

from rssilab import engine, presets                      # noqa: E402
from rssilab.trajectory import TrajParams                # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

scn = presets.demo_office()                              # 90 x 60 m, 12 anchors, walls, doors, racks
tech = presets.technology("Wi-Fi HaLow 868 MHz (802.11ah)")
tech.path_loss_exp = 3.2                                 # any field can be changed
# two distance-based methods (path-loss model) and two fingerprinting models (trained on a survey
# of the office every 2 m, 20 fingerprints per reference point), compared on the same RSS
algos = ["WLS multilateration", "MLE (grid search)", "k-NN", "Random forest"]
survey = dict(rp_spacing=2.0, rp_samples=20)

# 1) Positioning: 20 random points, 50 fixes per point and trial, 10 trials
res = engine.run(scn, tech,
                 TrajParams(mode="Positioning", point_source="Random points",
                            n_random_points=20, fixes_per_point=50),
                 engine.LocParams(algorithms=algos, mc_runs=10, seed=7, **survey))
print("Positioning (2D)")
for name, s in res.summary().items():
    print(f"  {name:20s} {res.families[name]:15s} RMSE {s['rmse']:5.2f} m   median {s['median']:5.2f} m   "
          f"P90 {s['p90']:5.2f} m   training {res.train_s[name]:.2f} s")
ps = res.point_stats("Random forest")                     # arrays per test point: rmse, bias, spread...
print(f"  worst point for the random forest: P{ps['rmse'].argmax() + 1} (RMSE {ps['rmse'].max():.2f} m)")
engine.export_stats_csv(os.path.join(OUT, "positioning_stats.csv"), res)

# 2) Tracking: random walk, 10 trials, Kalman filter on the random forest and particle filter
res = engine.run(scn, tech, TrajParams(mode="Tracking", kind="Random walk", n_random_wp=8),
                 engine.LocParams(algorithms=algos, kf_enabled=True, kf_base="Random forest", pf_enabled=True,
                                  mc_runs=10, seed=7, **survey))
print("Tracking (2D)")
for name, s in res.summary().items():
    print(f"  {name:20s} RMSE {s['rmse']:5.2f} m   MSE {s['mse']:6.2f} m^2   max {s['max']:5.2f} m")
trials = res.trial_stats(engine.PF_NAME)                  # arrays per trial
print(f"  PF RMSE across trials: {trials['rmse'].mean():.2f} +- {trials['rmse'].std():.2f} m")
engine.export_results_csv(os.path.join(OUT, "tracking_fixes.csv"), res)

# 3) 3D: anchors alternating between ceiling and floor level, estimate (x, y, z); the survey is
#    repeated on height layers 0.5 m apart
scn.dim, scn.room_z, scn.area_z0, scn.area_z1 = 3, 3.0, 0.3, 2.5
for i, a in enumerate(scn.anchors):
    a.z = 2.8 if i % 2 == 0 else 0.4
res = engine.run(scn, tech, TrajParams(mode="Positioning", point_source="Random points"),
                 engine.LocParams(algorithms=algos, mc_runs=3, z_res=0.5, rp_spacing=2.0, rp_samples=10))
print("Positioning (3D)")
for name, s in res.summary().items():
    print(f"  {name:20s} RMSE xy {s['rmse_xy']:5.2f} m   RMSE z {s['rmse_z']:5.2f} m")
for note in res.notes:
    print("  note:", note)
print(f"CSV files written to {OUT}")
