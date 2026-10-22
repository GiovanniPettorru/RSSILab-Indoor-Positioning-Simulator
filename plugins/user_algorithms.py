"""Template for your own positioning algorithms.

Every function decorated with @register appears in the Algorithms tab of the GUI
and is evaluated on exactly the same RSS realisations as the built-in methods.
Every .py file of this plugins/ folder is loaded automatically when RSSILab is
started from the RSSILab folder (python run_rssilab.py, or rssilab launched in that
folder) or runs from an editable install (pip install -e .): add your methods here
or in a new file next to this one. A user_algorithms.py in the current folder is
loaded too; other files can be passed with `rssilab --plugin my_file.py` or listed
in the RSSILAB_PLUGINS environment variable. Restart the GUI after editing. See
docs/GUIDE_custom_algorithms.md for a step-by-step guide and more examples.

    rss : ndarray (F, A)   averaged RSS (dBm) per fix and enabled anchor, NaN = not received
    ctx : Context          ctx.dim (2 or 3), ctx.anchors (A, dim), ctx.anchors3 (A, 3),
                           ctx.z_fixed (known target height in 2D),
                           ctx.area (x0, y0, x1, y1), ctx.area3 (x0, y0, z0, x1, y1, z1),
                           ctx.model (EstimatorModel: p0, n, d0, sigma),
                           ctx.rss_to_dist(rss), ctx.ranges(rss), ctx.ml_grid(),
                           ctx.obstacles, ctx.params, ctx.rng
    return ndarray (F, dim) estimated positions (NaN where no estimate); in 3D an
                           (F, 2) result gets the tag height as z.

The family decides where the method appears in the GUI: DISTANCE (default) or
FINGERPRINTING. A fingerprinting method gets the simulated survey of the scenario:
ctx.training() returns (X (N, A), y (N, dim)), the features of every survey
fingerprint (missing readings = sensitivity - 5 dB) and their positions;
ctx.features(rss) prepares the RSS of the fixes the same way;
ctx.trained("key", fit) calls fit(X, y) once per survey and caches the result
(its time is reported as training time); ctx.fingerprints = (points (R, 3),
radio map (R, A)) holds the mean fingerprint of each reference point.
"""
import numpy as np

from rssilab.localization import FINGERPRINTING, register


@register("Example: strongest anchor")
def strongest_anchor(rss, ctx):
    """Proximity baseline: the position of the anchor with the highest RSS."""
    out = np.full((len(rss), ctx.dim), np.nan)
    ok = ~np.all(np.isnan(rss), axis=1)
    idx = np.nanargmax(np.where(np.isnan(rss[ok]), -np.inf, rss[ok]), axis=1)
    out[ok] = ctx.anchors[idx]
    return out


@register("Example: nearest fingerprint", family=FINGERPRINTING)
def nearest_fingerprint(rss, ctx):
    """1-NN fingerprinting: the position of the closest survey fingerprint (in dB)."""
    X, y = ctx.training()
    x2 = ctx.trained("Example: nearest fingerprint", lambda X, y: (X ** 2).sum(1))
    q = ctx.features(rss)
    out = np.empty((len(rss), y.shape[1]))
    for s in range(0, len(q), 500):                       # blocks of fixes: bounded memory
        qs = q[s:s + 500]
        d2 = (qs ** 2).sum(1)[:, None] - 2 * qs @ X.T + x2[None]
        out[s:s + 500] = y[np.argmin(d2, axis=1)]
    out[np.all(np.isnan(rss), axis=1)] = np.nan          # nothing received: no estimate
    return out
