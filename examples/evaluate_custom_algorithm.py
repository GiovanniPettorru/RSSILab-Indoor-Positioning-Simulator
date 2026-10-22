"""Evaluates the example methods of custom_algorithms.py against built-in methods:
RMSE vs shadowing, statistics per test point, CSV exports and a tracking run.

    python examples/evaluate_custom_algorithm.py      (from the repository folder)

See docs/GUIDE_custom_algorithms.md. Results are written to examples/output/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # run from a clone

from rssilab import engine, presets                      # noqa: E402
from rssilab.trajectory import TrajParams                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

for msg in engine.load_plugins([os.path.join(HERE, "custom_algorithms.py")]):
    print(msg)

scn = presets.default_scenario()                         # 90 x 60 m room, 12 anchors on the walls
tp = TrajParams(mode="Positioning", point_source="Random points", n_random_points=30, fixes_per_point=20)
baselines = ["WLS multilateration", "MLE (Levenberg-Marquardt)", "k-NN"]
mine = ["WLS (5 strongest anchors)", "Probabilistic radio map"]

# 1) RMSE vs the standard deviation of the (static) shadowing, 5 trials each
print("\nRMSE (m) vs shadowing std")
for sigma in (2, 4, 6, 8):
    tech = presets.technology("BLE 2.4 GHz")
    tech.shadow_sigma_db = sigma
    res = engine.run(scn, tech, tp, engine.LocParams(algorithms=baselines + mine, mc_runs=5, seed=1))
    print(f"  sigma {sigma} dB: " + ", ".join(f"{n} {s['rmse']:.2f}" for n, s in res.summary().items()))

# 2) the last configuration in detail: bias and spread per test point, CSV files
for name in mine:
    ps = res.point_stats(name)
    print(f"\n{name}: worst test point P{ps['rmse'].argmax() + 1}, RMSE {ps['rmse'].max():.2f} m "
          f"(bias {ps['bias'][ps['rmse'].argmax()]:.2f} m, spread {ps['spread'][ps['rmse'].argmax()]:.2f} m)")
engine.export_stats_csv(os.path.join(OUT, "custom_stats.csv"), res)
engine.export_results_csv(os.path.join(OUT, "custom_fixes.csv"), res)

# 3) Tracking: the moving-average tracker against the Kalman filter on the same input
res = engine.run(presets.demo_office(), presets.technology("BLE 2.4 GHz"), TrajParams(mode="Tracking"),
                 engine.LocParams(algorithms=["MLE (grid search)", "Grid search + moving average"],
                                  kf_base="MLE (grid search)", mc_runs=3, seed=1))
print("\nTracking in the demo office")
for name, s in res.summary().items():
    print(f"  {name:30s} RMSE {s['rmse']:5.2f} m   P90 {s['p90']:5.2f} m")
print(f"\nCSV files written to {OUT}")
