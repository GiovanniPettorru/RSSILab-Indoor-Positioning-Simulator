# Testing your own algorithm

RSSILab is meant to be a test bench for positioning algorithms. This guide shows,
step by step, how to add your own method and evaluate it against the built-in ones:
on exactly the same RSS, over Monte Carlo trials, in the interface and from scripts.
It also shows how to use RSSILab only as a data generator when your algorithm lives
in another tool. The examples of this guide are in
[`examples/custom_algorithms.py`](../examples/custom_algorithms.py) and
[`examples/evaluate_custom_algorithm.py`](../examples/evaluate_custom_algorithm.py).

## Contents

1. [Three ways to test an algorithm](#1-three-ways-to-test-an-algorithm)
2. [Your first method, step by step](#2-your-first-method-step-by-step)
3. [The interface of a method](#3-the-interface-of-a-method)
4. [What a method can use: the context](#4-what-a-method-can-use-the-context)
5. [Example: a distance-based method](#5-example-a-distance-based-method)
6. [Example: a fingerprinting model from scikit-learn](#6-example-a-fingerprinting-model-from-scikit-learn)
7. [Example: a fingerprinting model of your own](#7-example-a-fingerprinting-model-of-your-own)
8. [Parameters and variants](#8-parameters-and-variants)
9. [Example: a tracker](#9-example-a-tracker)
10. [Evaluating your method from a script](#10-evaluating-your-method-from-a-script)
11. [Using RSSILab as a data generator](#11-using-rssilab-as-a-data-generator)
12. [Where plugin files are loaded from](#12-where-plugin-files-are-loaded-from)
13. [Checklist and troubleshooting](#13-checklist-and-troubleshooting)

## 1. Three ways to test an algorithm

| Way | When | What you get |
|---|---|---|
| **Plugin in the interface** | your algorithm can be written in Python | your method in the Algorithms tab, run on the same RSS as the built-in methods, with every table, plot and export of RSSILab |
| **Plugin from a script** | batch studies: parameter sweeps, many scenarios, figures for a paper | the same method driven by `engine.run`, results as NumPy arrays |
| **Data generator** | your algorithm lives in MATLAB, R or another framework | CSV files of simulated RSS (training and test sets) and RSSILab's metrics on your estimates |

The first two use the same code: a Python function registered as a positioning
method (a *plugin*). Sections 2 to 10 describe it; section 11 the third way.

## 2. Your first method, step by step

1. **Create a file** in the `plugins/` folder of RSSILab, for example
   `plugins/my_algorithms.py`. Every `.py` file of that folder is loaded when the
   simulator starts (section 12); `plugins/user_algorithms.py` is a template.
2. **Write a function** that takes the RSS of the fixes and returns one position per
   fix, and **register** it with a name:

   ```python
   import numpy as np
   from rssilab.localization import register

   @register("Strongest anchor")
   def strongest_anchor(rss, ctx):
       """Proximity: the position of the anchor received with the highest RSS."""
       out = np.full((len(rss), ctx.dim), np.nan)            # NaN = no estimate
       ok = ~np.all(np.isnan(rss), axis=1)                   # fixes with at least one reading
       best = np.argmax(np.where(np.isnan(rss[ok]), -np.inf, rss[ok]), axis=1)
       out[ok] = ctx.anchors[best]
       return out
   ```

3. **Start RSSILab** (`python run_rssilab.py` from the RSSILab folder). The log in the
   Analysis tab reports `Plugin my_algorithms.py loaded: Strongest anchor`.
4. In the **Algorithms** tab your method appears in the list of its family
   (*Distance-based methods* here). Tick it together with the methods you want to
   compare it with.
5. Set up the scenario, the radio and the experiment as usual and press **Run
   simulation**. Your method gets a row in the **Summary** (type, RMSE, percentiles,
   valid %, time per fix), its own **Per point** or **Per trial** statistics, its
   estimates on the map and its curve in the error CDF, and it is included in every
   CSV export.

After editing the file, restart RSSILab: plugins are loaded at start.

## 3. The interface of a method

```python
from rssilab.localization import DISTANCE, FINGERPRINTING, register

@register(name, family=DISTANCE)      # or family=FINGERPRINTING
def method(rss, ctx):
    ...
    return estimates
```

| Element | Meaning |
|---|---|
| `rss` | NumPy array (F, A): the averaged RSS (dBm) of the F fixes of one trial (rows) from the A enabled anchors (columns, in the order of the anchor table). NaN = the anchor was not received in that fix |
| `ctx` | the context: everything else the method may use (section 4) |
| return | array (F, 2) in 2D or (F, 3) in 3D with one position per fix, in the same order as the rows of `rss`. NaN (a whole row) = no estimate for that fix. In 3D a method that returns (F, 2) gets the tag height as z |
| `name` | the name shown in the interface and in the results; registering an existing name replaces that method |
| `family` | `DISTANCE` (default): listed among the distance-based methods, run with the *Distance-based* and *Both* approaches. `FINGERPRINTING`: listed among the fingerprinting methods; RSSILab simulates the survey and gives it to the method through `ctx` |

How RSSILab calls a method:

* **Once per trial**, with all the fixes of the trial at once, so that the method can
  be vectorised. In Positioning the rows are ordered test point by test point (the
  fixes of point 1, then those of point 2, ...); in Tracking they are consecutive
  fixes in time order.
* **On the same RSS as every other method.** The RSS are generated once per trial
  and shared, so differences in the results come from the methods only.
* **With its own random stream** `ctx.rng`, derived from the seed of the experiment,
  the trial and the name of the method: results are reproducible, and adding or
  removing other methods does not change them.
* Its output is clipped to the target area when *Clip estimates to the target area*
  is on, and the fixes without estimate lower its *Valid %*.
* Its computation time is reported per fix (*ms/fix*); the time spent in
  `ctx.trained()` is reported separately as training time (*Train s*).

A method must return exactly one row per fix: otherwise the run stops with an error
that names the method. An exception raised inside the method stops the run: the
interface shows its message in a dialog and the traceback in the log of the
Analysis tab; in scripts, `engine.run` raises it.

## 4. What a method can use: the context

| Member | Content |
|---|---|
| `ctx.dim` | 2 or 3: number of estimated coordinates |
| `ctx.anchors`, `ctx.anchors3` | enabled anchors, (A, dim) and (A, 3) |
| `ctx.z_fixed` | the known target height (tag height), used in 2D |
| `ctx.area`, `ctx.area3`, `ctx.lo`, `ctx.hi` | target area (x0, y0, x1, y1), target volume (x0, y0, z0, x1, y1, z1) and their bounds in the estimation space |
| `ctx.model` | estimator model: `p0`, `n`, `d0`, `sigma` (see [GUIDE_propagation.md](GUIDE_propagation.md#9-the-estimator-model)) |
| `ctx.rss_to_dist(rss)` | 3D distances from the RSS, inverting the estimator model |
| `ctx.ranges(rss)` | distances in the estimation space: 3D in 3D, horizontal in 2D |
| `ctx.ml_grid()` | a grid of candidate positions over the target area with the model RSS of every anchor at each candidate: `cand` (candidates), `H` (model RSS, with the obstacle losses when map-aware), `free` (candidates outside obstacles, when map-aware), cell sizes |
| `ctx.cell(P)` | index of the grid cell of positions P (F, dim) |
| `ctx.obstacles` | the obstacles of the scenario (see [GUIDE_python_api.md](GUIDE_python_api.md#3-scenarios-and-technologies)) |
| `ctx.params` | the parameters of the Algorithms tab (`knn_k`, `ml_res`, `map_aware`, ...) |
| `ctx.rng` | the random stream of the method (a NumPy `Generator`) |
| `ctx.check_cancel()` | call it now and then in long loops: it stops the method when the user presses Stop |
| `ctx.tech` | the true radio technology. Use it with care: an estimator that reads the true channel parameters has information a real system would not have; prefer `ctx.model`. Receiver settings such as `samples_per_fix` and `sample_rate_hz` are fair to use |
| `ctx.training()` | fingerprinting: (X, y), the features (N, A) and positions (N, dim) of every survey fingerprint |
| `ctx.features(rss)` | fingerprinting: the RSS with missing readings replaced as in the survey (sensitivity − 5 dB) |
| `ctx.trained(key, fit)` | fingerprinting: the result of `fit(X, y)` on the current survey, computed once per survey and cached under `key`; its time is reported as training time |
| `ctx.fingerprints` | fingerprinting: (points (R, 3), radio map (R, A)), the mean fingerprint of each reference point |
| `ctx.survey` | fingerprinting: the whole survey: `points`, `radio_map`, `positions`, `fingerprints` |

A distance-based method usually works with `ranges()` or with the RSS and
`ctx.model`; a map-aware one with `ml_grid()` and `obstacles`; a fingerprinting
model with `training()`, `features()` and `trained()`. How the survey is built is
described in [GUIDE_algorithms.md](GUIDE_algorithms.md#survey).

## 5. Example: a distance-based method

Weighted linear least squares on the *k* strongest anchors of each fix: the idea
being tested is that far anchors, whose ranges are the noisiest, may be better left
out. The equations are those of WLS multilateration (see
[GUIDE_algorithms.md](GUIDE_algorithms.md#trilateration-lls-and-wls-multilateration)),
written with a loop over the fixes for readability:

```python
import numpy as np
from rssilab.localization import register


def wls_k_strongest(rss, ctx, k):
    """Weighted linear least squares on the k strongest received anchors of each fix."""
    A = ctx.anchors                          # (A, dim) coordinates of the enabled anchors
    r = ctx.ranges(rss)                      # (F, A) ranges from the estimator model, NaN = not received
    d = ctx.rss_to_dist(rss)                 # (F, A) 3D distance estimates, used for the weights
    out = np.full((len(rss), ctx.dim), np.nan)
    for f in range(len(rss)):
        heard = np.flatnonzero(~np.isnan(rss[f]))
        use = heard[np.argsort(-rss[f, heard])][:k]          # the k strongest received anchors
        if len(use) < ctx.dim + 1:
            continue                                         # too few anchors: no estimate (NaN)
        # |p - a|^2 = r^2  ->  -2 a.p + |p|^2 = r^2 - |a|^2, unknowns (p, |p|^2)
        G = np.column_stack([-2 * A[use], np.ones(len(use))])
        b = r[f, use] ** 2 - (A[use] ** 2).sum(axis=1)
        sw = 1.0 / np.maximum(d[f, use], 0.1) ** 2           # square root of the weights 1/d^4
        out[f] = np.linalg.lstsq(G * sw[:, None], b * sw, rcond=None)[0][:ctx.dim]
    return out


@register("WLS (5 strongest anchors)")
def wls_5(rss, ctx):
    """Weighted linear least squares on the 5 strongest anchors of each fix."""
    return wls_k_strongest(rss, ctx, 5)
```

In 3D the system needs anchors at different heights (with all the anchors at one
height the z column is constant and z is undetermined). A loop over fixes is fast
enough for thousands of fixes; for very large experiments, vectorise it over the
fixes as the built-in methods do.

## 6. Example: a fingerprinting model from scikit-learn

Any regressor that maps a fingerprint to a position can be plugged in. RSSILab
simulates the survey; `ctx.trained` trains the model once per survey and caches it:

```python
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from rssilab.localization import FINGERPRINTING, register


@register("Extra trees", family=FINGERPRINTING)
def extra_trees(rss, ctx):
    """Extremely randomised trees, trained on the survey of the scenario."""
    def fit(X, y):                           # X (N, A): survey fingerprints, y (N, dim): their positions
        seed = int(ctx.rng.integers(2 ** 31 - 1))
        return ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1,
                                   random_state=seed).fit(X, y)

    model = ctx.trained("Extra trees", fit)  # trained once per survey, timed as training
    out = np.full((len(rss), ctx.dim), np.nan)
    ok = ~np.all(np.isnan(rss), axis=1)      # fixes with at least one reading
    if ok.any():
        out[ok] = model.predict(ctx.features(rss[ok])).reshape(-1, ctx.dim)
    return out
```

Notes:

* The survey uses the resolution set in the Algorithms tab (*Survey grid spacing*,
  *Fingerprints per point*) and the same environment realisation as the experiment;
  with *New shadowing realisation at every trial* it is repeated, and the model
  trained again, in every trial.
* Use a cache key unique to the method (here its name).
* The seed of the model comes from `ctx.rng`, so the experiment stays reproducible.
* `ctx.features()` fills missing readings exactly as in the training data.

## 7. Example: a fingerprinting model of your own

A probabilistic radio map in the spirit of Horus: at every reference point, each
anchor's RSS is modelled as a Gaussian with the mean and standard deviation of the
survey fingerprints of that point; the estimate is the mean of the reference points
weighted by their likelihood. The survey stores the fingerprints point by point
(all the fingerprints of the first reference point, then those of the second, ...),
so they can be reshaped per reference point:

```python
@register("Probabilistic radio map", family=FINGERPRINTING)
def probabilistic(rss, ctx):
    """Gaussian likelihood of the fix at every reference point (mean and std of each
    anchor over the fingerprints of the point); the estimate is the posterior mean."""
    def fit(X, y):
        pts = ctx.survey.points[:, :ctx.dim]               # (R, dim) reference points
        Xr = X.reshape(len(pts), -1, X.shape[1])           # (R, fingerprints per point, A)
        return pts, Xr.mean(axis=1), np.maximum(Xr.std(axis=1), 1.0)

    pts, mu, sd = ctx.trained("Probabilistic radio map", fit)
    q = ctx.features(rss)                                  # missing readings filled as in the survey
    out = np.full((len(rss), ctx.dim), np.nan)
    B = max(1, 2_000_000 // mu.size)                       # fixes per block: bounded memory
    for s in range(0, len(q), B):
        ctx.check_cancel()                                 # Stop stays responsive
        z = (q[s:s + B, None, :] - mu[None]) / sd[None]    # (block, R, A)
        ll = -0.5 * (z ** 2).sum(-1) - np.log(sd).sum(-1)  # log-likelihood of each reference point
        w = np.exp(ll - ll.max(axis=1, keepdims=True))
        out[s:s + B] = (w @ pts) / w.sum(axis=1, keepdims=True)
    out[np.all(np.isnan(rss), axis=1)] = np.nan
    return out
```

Processing the fixes in blocks keeps the memory bounded whatever the number of
fixes and of reference points, and `ctx.check_cancel()` lets the user stop a long
run.

## 8. Parameters and variants

The interface does not show parameters of plugin methods. Three ways to vary them:

* **Variants as separate methods.** Register the same function several times with
  different parameters; each variant gets its own row in the results, so a parameter
  sweep is a single run:

  ```python
  for k in (3, 4, 8):
      register(f"WLS ({k} strongest anchors)")(lambda rss, ctx, k=k: wls_k_strongest(rss, ctx, k))
  ```

  The `k=k` default argument binds the current value of k to each variant (without
  it every variant would use the last value).
* **Parameters of the Algorithms tab.** `ctx.params` holds them (for example
  `ctx.params["knn_k"]`), so a plugin can reuse one of the existing fields.
* **Module constants or environment variables**, read when the file is loaded.

## 9. Example: a tracker

In Tracking the rows of `rss` are consecutive fixes in time order, one every
`ctx.tech.samples_per_fix / ctx.tech.sample_rate_hz` seconds, so a method can
filter them. This one averages the estimates of the map-aware grid search, a
built-in method that a plugin can call directly, over the last five fixes:

```python
from rssilab.localization import mle_grid, register


@register("Grid search + moving average")
def smoothed_grid(rss, ctx):
    """Tracking: average of the map-aware grid search over the last 5 fixes."""
    est = mle_grid(rss, ctx)                 # a built-in method on the same fixes
    out = np.full_like(est, np.nan)
    for f in range(len(est)):
        window = est[max(0, f - 4):f + 1]
        if np.isfinite(window).all(axis=1).any():
            out[f] = np.nanmean(window, axis=0)
    return out
```

Use such methods only in Tracking: in Positioning the rows are grouped by test point
and a temporal filter would mix the fixes of different points. Your methods can
also be the input of the built-in Kalman filter (*KF input* in the Algorithms tab),
which lists the methods of the chosen approach, plugins included.

## 10. Evaluating your method from a script

The same plugin file can be loaded from a script, to run batch studies with
`engine.run` (the API is described in [GUIDE_python_api.md](GUIDE_python_api.md)).
For example, the RMSE of your method and of three baselines as the shadowing grows:

```python
from rssilab import engine, presets
from rssilab.trajectory import TrajParams

print(engine.load_plugins(["plugins/my_algorithms.py"]))      # run from the RSSILab folder

scn = presets.default_scenario()                               # 90 x 60 m room, 12 anchors
tp = TrajParams(mode="Positioning", point_source="Random points", n_random_points=30, fixes_per_point=20)
methods = ["WLS multilateration", "MLE (Levenberg-Marquardt)", "k-NN", "WLS (5 strongest anchors)"]

for sigma in (2, 4, 6, 8):
    tech = presets.technology("BLE 2.4 GHz")
    tech.shadow_sigma_db = sigma
    res = engine.run(scn, tech, tp, engine.LocParams(algorithms=methods, mc_runs=5, seed=1))
    print(f"sigma {sigma} dB: " + ", ".join(f"{n} {s['rmse']:.2f} m" for n, s in res.summary().items()))

ps = res.point_stats("WLS (5 strongest anchors)")             # per test point: rmse, bias, spread, ...
engine.export_stats_csv("my_method_stats.csv", res)
engine.export_results_csv("my_method_fixes.csv", res)
```

* `res.summary()` gives the overall statistics, `res.point_stats(name)` and
  `res.trial_stats(name)` the statistics per test point and per trial,
  `res.est[name]` and `res.err[name]` every estimate and error (trials × fixes), and
  `metrics.ecdf(res.err[name])` the empirical CDF for your own figures.
* Use the same seed and settings for every method you compare (one run does this
  automatically); increase the trials (`mc_runs`) until the differences you report
  are larger than their variation across trials (`res.trial_stats`).
* A complete script, which also runs the tracker of section 9, is
  [`examples/evaluate_custom_algorithm.py`](../examples/evaluate_custom_algorithm.py).

## 11. Using RSSILab as a data generator

When your algorithm runs in another tool, generate the data with RSSILab and compute
the metrics on the estimates you bring back.

**Ready-made files.** *Export RSS dataset* (Analysis tab, or
`engine.export_dataset_csv`) writes a radio map: readings on a grid over the target
area, one row per reading. *Export results per fix* writes, for every trial and fix
of an experiment, the ground truth and the averaged RSS of every anchor, together
with the estimates of the methods that were run (useful as test sets and baselines).
The formats are described in [GUIDE_gui.md](GUIDE_gui.md#8-analysis-tab-and-files).

**Training and test sets from the same environment.** With static shadowing, the
training data and the test data must come from the same realisation of the
environment, as in a real building. The channel model gives full control:

```python
import numpy as np
from rssilab import presets
from rssilab.channel import Environment, average_fixes
from rssilab.fingerprinting import survey_points
from rssilab.trajectory import random_points

scn, tech = presets.demo_office(), presets.technology("BLE 2.4 GHz")
rng = np.random.default_rng(1)
env = Environment(scn, tech, rng)                 # one shadowing realisation for training and test
Ns = tech.samples_per_fix

def fixes(points, k):
    """k position fixes at every point (each the average of Ns readings): (len(points) * k, A)."""
    raw = env.measure(np.repeat(points, k * Ns, axis=0), rng)
    return average_fixes(raw.reshape(len(points) * k, Ns, -1))

train_pts = survey_points(scn, spacing=1.0)       # reference points every metre, inside free space
test_pts = random_points(scn, 50, rng)            # 50 random test points
X_train, y_train = fixes(train_pts, 10), np.repeat(train_pts, 10, axis=0)
X_test, y_test = fixes(test_pts, 20), np.repeat(test_pts, 20, axis=0)
header = "x,y,z," + ",".join(f"RSS_a{i + 1}" for i in range(X_train.shape[1]))
np.savetxt("train.csv", np.column_stack([y_train, X_train]), delimiter=",", header=header, comments="", fmt="%.6g")
np.savetxt("test.csv", np.column_stack([y_test, X_test]), delimiter=",", header=header, comments="", fmt="%.6g")
```

Missing readings are written as `nan`. Then, with the estimates of your tool in
`my_estimates.csv` (one row x, y per test fix, in the order of `test.csv`):

```python
import numpy as np
from rssilab import metrics

test = np.loadtxt("test.csv", delimiter=",", skiprows=1)
est = np.loadtxt("my_estimates.csv", delimiter=",")          # (N, 2): x, y estimated by your tool
err = np.hypot(est[:, 0] - test[:, 0], est[:, 1] - test[:, 1])
print(metrics.summarize(err))                                  # mean, rmse, median, p90, p95, ...
```

The statistics are the same as in RSSILab's tables (see
[GUIDE_algorithms.md](GUIDE_algorithms.md#6-error-statistics)). For several trials,
create a new `Environment` (with a new seed) for each one.

## 12. Where plugin files are loaded from

* Every `.py` file of the `plugins/` folder of RSSILab (in alphabetical order; files
  whose name starts with `_` are skipped), when RSSILab is started from its folder
  (`python run_rssilab.py`, or `rssilab` / `python -m rssilab` launched in that
  folder) or runs from an editable install (`pip install -e .`).
* `user_algorithms.py` in the current folder.
* Files passed on the command line: `python run_rssilab.py --plugin path/to/file.py`
  (or `rssilab --plugin ...`; the option can be repeated).
* Files listed in the `RSSILAB_PLUGINS` environment variable (separated by `:` on
  Linux and macOS, `;` on Windows).
* From a script: `engine.load_plugins(["path/to/file.py"])`, or simply import the
  module that registers the methods.

The examples of this guide are not loaded by default: to try them in the interface,
start RSSILab with `--plugin examples/custom_algorithms.py` or copy the file into
`plugins/`.

## 13. Checklist and troubleshooting

**Before trusting the numbers**

* **Check the method without noise.** With shadowing 0, fading *None*, quantisation
  0, no sensitivity cut-off, no obstacles and the oracle calibration, a consistent
  distance-based method should find the position almost exactly (section 1 of
  [VALIDATION.md](VALIDATION.md) does this for the built-in methods). Large errors
  there point to a bug, not to the channel.
* **Look at Valid %.** A low value means that the method often returns no estimate
  (too few anchors, or a bug that produces NaN).
* **Look at bias and spread** in the Per point table: a large bias on every point
  usually means an inconsistency (wrong coordinates, units, or model).
* **Use enough trials** and the same seed for all the methods; for fingerprinting,
  avoid test points on the survey grid (the log warns about it).
* **Report the settings**: *File > Save project* stores every parameter of an
  experiment in a JSON file that reproduces it.

**Common problems**

| Symptom | Cause and fix |
|---|---|
| The method does not appear | The file was not loaded: check that it is in `plugins/` (or passed with `--plugin`) and restart RSSILab; the log lists the loaded plugins |
| "Plugin ... not loaded: ..." in the log | The file raised an error while loading (syntax error, missing import): the message shows the exception |
| "'name' returned N estimates for F fixes" | The method must return exactly one row per fix, also for the fixes without estimate (NaN rows) |
| The run stops with an error | An exception inside the method: the traceback is in the log of the Analysis tab (in a script, `engine.run` raises it) |
| Fingerprinting method listed among the distance-based ones | Register it with `family=FINGERPRINTING` |
| "No fingerprint survey" | `ctx.training()` was called by a method registered as distance-based |
| Results change when other methods are added | Use `ctx.rng` for every random number (not `np.random` directly) |
| The interface does not react to Stop | Call `ctx.check_cancel()` inside long loops |
