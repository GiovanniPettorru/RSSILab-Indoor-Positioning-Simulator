# Python API and plugins

Everything the interface does is available from Python, without Tkinter: build
scenarios in code, run Monte Carlo experiments in scripts or notebooks, generate RSS
datasets, and add your own positioning methods, which then appear in the
interface as well. The models are described in
[GUIDE_propagation.md](GUIDE_propagation.md),
[GUIDE_algorithms.md](GUIDE_algorithms.md) and [GUIDE_tracking.md](GUIDE_tracking.md);
the formats of the CSV files in [GUIDE_gui.md](GUIDE_gui.md#8-analysis-tab-and-files).

## Contents

1. [Setup](#1-setup)
2. [A first experiment](#2-a-first-experiment)
3. [Scenarios and technologies](#3-scenarios-and-technologies)
4. [Experiment parameters](#4-experiment-parameters)
5. [Positioning parameters](#5-positioning-parameters)
6. [Running and reading the results](#6-running-and-reading-the-results)
7. [RSS datasets and raw readings](#7-rss-datasets-and-raw-readings)
8. [Reproducibility](#8-reproducibility)
9. [Your own positioning methods](#9-your-own-positioning-methods)
10. [Examples and package layout](#10-examples-and-package-layout)

## 1. Setup

Install the package from the RSSILab folder with `python -m pip install .`, or with
`python -m pip install -e .` to keep using the files of the folder (editable
install, recommended if you change the code or add plugins). Without installing
(after `python -m pip install -r requirements.txt`), the `rssilab` package can be
imported by scripts placed in the RSSILab folder and by notebooks or interactive
sessions started there; the scripts of `examples/` add the folder to the import
path themselves. To import it from anywhere, install it.

## 2. A first experiment

```python
from rssilab import engine, presets
from rssilab.trajectory import TrajParams

scn = presets.small_room()                         # or default_scenario(), demo_office()
tech = presets.technology("BLE 2.4 GHz")           # any preset name of the Radio tab
tp = TrajParams(mode="Positioning", point_source="Random points", n_random_points=20, fixes_per_point=10)
lp = engine.LocParams(algorithms=["WLS multilateration", "k-NN", "Random forest"], mc_runs=5, seed=1)
res = engine.run(scn, tech, tp, lp)
print(res.summary()["Random forest"]["rmse"])
```

`engine.run` takes the scenario, the technology, the experiment parameters
(`TrajParams`) and the positioning parameters (`LocParams`), and returns a `Results`
object with every estimate and error.

## 3. Scenarios and technologies

`rssilab.model` defines the data model (distances in metres, room centred at (0, 0),
heights from the floor):

* `Scenario` with the fields `room_w, room_h` (room size), `area_cx, area_cy,
  area_w, area_h` (target area), `anchor_h` (default anchor height), `tag_h` (tag
  height), `dim` (2 or 3), `room_z` (ceiling height), `area_z0, area_z1` (vertical
  extent of the target volume in 3D), `anchors`, `obstacles`, `waypoints` and
  `test_points` (lists of (x, y) or (x, y, z) tuples);
  `place_anchors_on_perimeter(n)` places n anchors on the walls of the room.
* `Anchor(name, x, y, enabled=True, z=0.0)`.
* `Obstacle(kind, x0, y0, x1, y1, material="Brick", thickness=0.2, loss_db=None,
  blocks_motion=True, z0=0.0, z1=None)`: `kind` is `"wall"` (segment from (x0, y0)
  to (x1, y1)) or `"rect"` (axis-aligned block with opposite corners (x0, y0) and
  (x1, y1), in any order); `loss_db` overrides the material loss; `z0` and `z1` are
  the bottom and the top (None = ceiling); `blocks_motion=False` makes it
  transparent to motion.
* `Technology`, with the fields of the Radio tab: `freq_ghz, tx_power_dbm,
  gains_db, p0_auto, p0_dbm, d0_m, path_loss_exp, shadow_sigma_db, shadow_corr_m,
  fading` (`"None"`, `"Rayleigh"`, `"Rician"` or `"Legacy (Rayleigh amplitude in
  dB)"`), `rician_k_db, legacy_fading_sigma, sensitivity_dbm, quantization_db,
  sample_rate_hz, samples_per_fix`.

`rssilab.presets` provides `default_scenario()` (reference room),
`demo_office()`, `small_room()` and `technology(name)` (a copy of a radio preset;
the names are in `model.PRESETS`). `model.save_project(path, scn, tech)` and
`model.load_project(path)` write and read the JSON projects of the interface, and
`geometry.import_grid_layout(scn, level, material)` imports one of the 14 grid-world
layouts.

A scenario built in code:

```python
from rssilab import engine, presets
from rssilab.model import Obstacle, Scenario
from rssilab.trajectory import TrajParams

scn = Scenario(room_w=20, room_h=15, area_w=18, area_h=13, tag_h=1.0)
scn.place_anchors_on_perimeter(6)
for a in scn.anchors:
    a.z = 2.5                                            # anchors near the ceiling
scn.obstacles = [
    Obstacle("wall", 0, -7.5, 0, 3, material="Brick"),   # a wall with a 4.5 m opening
    Obstacle("rect", 4, 2, 6, 3, material="Furniture", z1=0.8),   # a desk 0.8 m high
]
tech = presets.technology("Wi-Fi 2.4 GHz")
tp = TrajParams(mode="Positioning", point_source="Grid", grid_spacing=1.0, fixes_per_point=20)
lp = engine.LocParams(algorithms=["WLS multilateration", "MLE (grid search)", "k-NN"],
                      mc_runs=3, rp_spacing=1.0)
res = engine.run(scn, tech, tp, lp)
for name, s in res.summary().items():
    print(f"{name:22s} RMSE {s['rmse']:.2f} m, valid {s['valid']:.0f}%")
```

## 4. Experiment parameters

`rssilab.trajectory.TrajParams`:

| Field | Default | Meaning |
|---|---|---|
| mode | "Positioning" | "Positioning" or "Tracking" |
| point_source | "Points on map" | "Points on map" (the scenario's `test_points`), "Random points" or "Grid" |
| n_random_points | 10 | random test points |
| fixes_per_point | 10 | fixes per point and trial |
| grid_spacing | 2.0 | test grid spacing (m) |
| grid_z | None | 3D test grid height (None = tag height) |
| kind | "Waypoints (click on map)" | route: also "Random walk", "Circle", "Lawnmower" |
| speed | 1.0 | m/s |
| use_astar, astar_res, clearance | True, 0.5, 0.3 | A* routing around obstacles; the clearance also applies to random test points |
| n_random_wp | 6 | random-walk waypoints |
| circle_frac, circle_laps | 0.8, 1.0 | circle route |
| lawn_spacing | 5.0 | lawnmower row spacing (m) |

## 5. Positioning parameters

`rssilab.engine.LocParams`:

| Field | Default | Meaning |
|---|---|---|
| algorithms | WLS, NLS, MLE (Levenberg-Marquardt), k-NN, Random forest | method names (`localization.methods()` lists them, `methods(DISTANCE)` or `methods(FINGERPRINTING)` by family) |
| wcl_k, wcl_g | 4, 1.0 | weighted centroid |
| ml_res | 0.5 | MLE grid resolution (m) |
| map_aware | True | walls in the MLE grid and the particle filter |
| rp_spacing, rp_samples | 2.0, 20 | survey grid spacing (m) and fingerprints per point |
| knn_k, knn_weighted | 10, True | k-NN |
| wknn_k | 4 | WKNN |
| dt_min_leaf | 5 | decision tree |
| rf_trees | 100 | random forest |
| mlp_hidden | (64, 64) | neural network |
| z_res | 0.5 | 3D vertical spacing of the MLE grid and of the survey (m) |
| clip_to_area | True | clip estimates to the target area |
| kf_enabled, kf_base, kf_q, kf_r | True, "MLE (Levenberg-Marquardt)", 0.5, 0.0 | Kalman filter (Tracking) |
| pf_enabled, pf_n, pf_acc, pf_jitter, pf_sigma, pf_inject | False, 1000, 0.5, 0.0, 0.0, 0.02 | particle filter (Tracking) |
| calib, p0_est, n_est, sigma_est | "Oracle", −40.0, 4.0, 3.0 | estimator calibration ("Oracle" or "Manual") |
| mc_runs | 1 | trials |
| seed | 1 | seed |
| redraw_env | True | new shadowing realisation (and survey) at every trial |

Method names used by earlier versions of RSSILab (for example "NLS (ML-LM)") are
accepted in `algorithms` and `kf_base`; the results use the current names.

## 6. Running and reading the results

`engine.run(scn, tech, tp, lp, progress=None, cancel=None)` returns a `Results`
object; `progress(fraction, message)` is called during the run and `cancel()`
returning True stops it with `InterruptedError`. Invalid settings (an unknown mode,
test point source, route, calibration or method name, too few anchors, no test
point, ...) raise `ValueError` with a readable message; counts such as trials and
fixes per point below 1 are taken as 1. A missing scikit-learn raises `ImportError`.

| Attribute | Content |
|---|---|
| mode, dim, names | "positioning" or "tracking", 2 or 3, method names in display order |
| gt | (F, 3) ground truth of the F fixes |
| est[name] | (trials, F, 3) estimates (NaN = no estimate; in 2D z is the tag height) |
| err, err_h, err_v [name] | (trials, F) error, horizontal error, vertical error |
| rss | (trials, F, anchors) averaged RSS of every fix |
| t, dt | Tracking: fix times (s); Positioning: fix index (0 … F − 1). dt: time per fix, samples per fix / sample rate (s) |
| points, fixes_per_point, point_id | Positioning: test points (P, 3), fixes per point, point of each fix |
| route, grid_info | Tracking route (K, 3); test grid information |
| families, timing_ms, train_s | family, time per fix (ms) and training time per survey (s) of each method |
| survey_points | reference points of the (last) survey, or None |
| anchors, anchor_names, model, tech_name, notes | enabled anchors, estimator model, technology name, notes of the run |

`res.summary()` returns the overall statistics per method (`mean, rmse, mse, std,
min, median, p90, p95, max, valid, n, rmse_xy, rmse_z`), `res.point_stats(name)`
the statistics per test point (with `bias`, `spread` and the mean estimate) and
`res.trial_stats(name)` the statistics per trial; `metrics.ecdf(err)` gives the
empirical CDF of an error array. `engine.export_results_csv(path, res)` and
`engine.export_stats_csv(path, res)` write the CSV files of the Analysis tab.

## 7. RSS datasets and raw readings

`engine.export_dataset_csv(path, scn, tech, spacing, samples, seed=1, progress=None,
skip_obstacles=False, cancel=None)` writes the radio-map dataset of *Export RSS
dataset*: one row per reading on a grid over the target area at the tag height. It
returns the number of data rows, takes `progress` and `cancel` as `engine.run`
does, and removes the partial file when it is cancelled.

For other layouts of data, generate the readings directly with the channel model:

```python
import numpy as np
from rssilab import presets
from rssilab.channel import Environment, average_fixes

scn, tech = presets.demo_office(), presets.technology("BLE 2.4 GHz")
rng = np.random.default_rng(1)
env = Environment(scn, tech, rng)                  # one realisation of the shadowing
points = np.array([[0.0, 0.0, scn.tag_h], [10.0, 5.0, scn.tag_h]])
readings = env.measure(np.repeat(points, 10, axis=0), rng)          # (20, A) dBm, NaN = not received
fixes = average_fixes(readings.reshape(len(points), 10, -1))        # (2, A): 10 readings per fix
```

`Environment(scn, tech, rng, bounds=None)` draws the shadowing field over the
target area (pass `bounds=(x0, y0, x1, y1)` to cover positions outside it);
`env.mean(points)` gives the mean RSS (path loss and obstacles) and
`env.measure(points, rng)` one reading per point and enabled anchor, with
shadowing, fading, quantisation and sensitivity.

## 8. Reproducibility

A run is a deterministic function of its inputs and seed. Trial r uses the random
stream (seed, r); the survey and each method have their own streams derived from it,
so adding or removing a method does not change the results of the others.

## 9. Your own positioning methods

A positioning method is a function `f(rss, ctx)` that receives the averaged RSS of all
the fixes of a trial and returns one position per fix, registered with
`rssilab.localization.register(name, needs=(), family=None)`:

```python
import numpy as np
from rssilab.localization import register

@register("My method")                  # family=FINGERPRINTING for a model trained on the survey
def my_method(rss, ctx):                # rss: (F, A) dBm, NaN = not received
    ...
    return estimates                    # (F, 2) in 2D, (F, 3) in 3D, NaN = no estimate
```

The family is `DISTANCE` (default) or `FINGERPRINTING` (also implied by
`needs=("fingerprints",)`); registering an existing name replaces that method.
`localization.methods(family=None)` lists the registered methods and
`engine.load_plugins(paths)` loads plugin files, returning one message per file.
The method then runs in `engine.run` and in the interface like the built-in ones.

[GUIDE_custom_algorithms.md](GUIDE_custom_algorithms.md) explains step by step how to
write, load and evaluate your own methods, with the full description of `ctx` and
complete examples.

## 10. Examples and package layout

The `examples/` folder contains complete scripts, to run from the RSSILab folder:

| Script | What it does |
|---|---|
| `quickstart.py` | positioning, tracking and 3D experiments with distance-based and fingerprinting methods; writes CSV files to `examples/output/` |
| `generate_dataset.py` | two RSS datasets in the format of the original RSSILab dataset: the legacy channel in the reference room and a BLE channel in the demo office; writes to `examples/output/` |
| `validation_study.py` | the validation study of [VALIDATION.md](VALIDATION.md) (about a minute) |
| `custom_algorithms.py` | example plugin with five methods of your own (see [GUIDE_custom_algorithms.md](GUIDE_custom_algorithms.md)) |
| `evaluate_custom_algorithm.py` | evaluates those methods against built-in ones (RMSE vs shadowing, per-point statistics, tracking); writes to `examples/output/` |

The `rssilab` package:

| Module | Content |
|---|---|
| `model.py` | materials, anchors, obstacles, technologies, scenario, JSON projects |
| `channel.py` | path loss, shadowing field, fading, receiver |
| `geometry.py` | intersections, multi-wall losses, occupancy grid, A* routing, layout import |
| `grid_layouts.py` | the 14 grid-world obstacle layouts |
| `trajectory.py` | test points (positioning) and routes (tracking) |
| `localization.py` | plugin registry, distance-based methods, Kalman and particle filters |
| `fingerprinting.py` | survey of the scenario and fingerprinting models |
| `metrics.py` | statistics (overall, per point, per trial), ECDF |
| `engine.py` | Monte Carlo pipeline, plugins, CSV exports |
| `presets.py` | ready-made scenarios and technologies |
| `gui.py` | graphical interface |
