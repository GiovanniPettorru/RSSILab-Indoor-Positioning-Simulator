"""Example positioning methods written as RSSILab plugins (see docs/GUIDE_custom_algorithms.md).

    python run_rssilab.py --plugin examples/custom_algorithms.py   (or copy this file into plugins/)
    python examples/evaluate_custom_algorithm.py                   (compares them with built-in methods)

1. WLS (5 strongest anchors): a distance-based method.
2. Extra trees: a fingerprinting model from scikit-learn.
3. Probabilistic radio map: a fingerprinting model written from scratch.
4. WLS (3 / 4 / 8 strongest anchors): variants of one method with different parameters.
5. Grid search + moving average: a simple tracker built on a built-in method.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from rssilab.localization import FINGERPRINTING, mle_grid, register


# ---------------------------------------------------------------- 1. a distance-based method
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


# ---------------------------------------------------------------- 2. a fingerprinting model from scikit-learn
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


# ---------------------------------------------------------------- 3. a fingerprinting model of your own
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


# ---------------------------------------------------------------- 4. variants of one method
for k in (3, 4, 8):
    register(f"WLS ({k} strongest anchors)")(lambda rss, ctx, k=k: wls_k_strongest(rss, ctx, k))


# ---------------------------------------------------------------- 5. a tracker
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
