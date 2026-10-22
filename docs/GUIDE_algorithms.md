# Positioning algorithms

This guide describes the positioning methods of RSSILab: what each one computes,
with its formulas, its parameters and the details that matter when comparing
them, and the error statistics used to evaluate them. The channel that produces the
RSS is described in [GUIDE_propagation.md](GUIDE_propagation.md); the Kalman and
particle filters, which work on sequences of fixes, in
[GUIDE_tracking.md](GUIDE_tracking.md).

## Contents

1. [Two families of methods](#1-two-families-of-methods)
2. [Common conventions](#2-common-conventions)
3. [Distance-based methods](#3-distance-based-methods)
4. [Fingerprinting methods](#4-fingerprinting-methods)
5. [Comparing the methods](#5-comparing-the-methods)
6. [Error statistics](#6-error-statistics)
7. [Adding your own method](#7-adding-your-own-method)
8. [References](#8-references)

## 1. Two families of methods

| | Distance-based | Fingerprinting |
|---|---|---|
| Knowledge used | a path-loss model of the channel (the estimator model) and the anchor positions | a survey of the scenario: RSS recorded at known reference points |
| Principle | the RSS is turned into distances (ranges) or into a likelihood of the position | the mapping from RSS to position is learnt from the survey |
| Methods | Min-Max, weighted centroid, trilateration, LLS, WLS and NLS multilateration, MLE (Levenberg-Marquardt), MLE (grid search) | k-NN, WKNN (radio map), decision tree, random forest, neural network (MLP) |
| Sensitive to | model mismatch, walls, static shadowing, anchor geometry | survey resolution and size, changes between survey and use |
| Cost | per fix | training once per survey, then per fix |

In the Algorithms tab, the *Positioning approach* selects the distance-based
methods, the fingerprinting methods or both; with both, the two families are
evaluated on exactly the same RSS.

## 2. Common conventions

* **Input.** Every method receives the RSS of all the fixes of a trial, a matrix
  (F fixes × A enabled anchors) in dBm, where a missing value (NaN) means that the
  anchor was not received in that fix. It returns one position per fix, or no
  estimate (NaN) when it cannot compute one.
* **Estimation space.** In 2D the methods estimate (x, y) and the target height is
  the known *Tag height*; in 3D they estimate (x, y, z).
* **Ranges.** The distance-based methods invert the estimator model
  (P0_est, n_est, d0; see [GUIDE_propagation.md](GUIDE_propagation.md#9-the-estimator-model)):

  ```math
  \hat d_i = d_0\,10^{(P_{0,\mathrm{est}} - \mathrm{RSS}_i)/(10\,n_\mathrm{est})},
  \qquad
  r_i = \begin{cases}\sqrt{\max(\hat d_i^{\,2} - \Delta z_i^2,\ 10^{-6})} & \text{2D}\\ \hat d_i & \text{3D}\end{cases}
  ```

  d̂_i is the estimated 3D distance to anchor *i*; in 2D the known height difference
  Δz_i between anchor and target is removed, so that r_i is a horizontal range.
* **Clipping.** With *Clip estimates to the target area* (default) every estimate is
  clipped to the target area (the target volume in 3D).
* **Random streams.** Every method gets its own random stream for each trial, so
  methods that use randomness (random forest, neural network, plugins) are
  reproducible and do not change the results of the other methods.

## 3. Distance-based methods

### Min-Max

Each received anchor *i* defines a box [**a**_i − r_i, **a**_i + r_i] (a square in
2D, a cube in 3D) that contains the target if r_i is exact. Min-Max intersects the
boxes and returns the centre of the intersection:

```math
\mathbf{lo} = \max_i(\mathbf a_i - r_i),\qquad \mathbf{hi} = \min_i(\mathbf a_i + r_i),\qquad
\hat{\mathbf p} = (\mathbf{lo} + \mathbf{hi})/2
```

(maximum and minimum per coordinate, over the received anchors). It needs no
iteration and no matrix inversion, but it is biased even without noise: the centre
of the intersection is the target only in symmetric configurations. When the boxes
do not overlap, the same formula returns the midpoint between them.

### Weighted centroid

The mean of the anchor positions, weighted by the inverse of their estimated
distance:

```math
\hat{\mathbf p} = \frac{\sum_i w_i\,\mathbf a_i}{\sum_i w_i},\qquad w_i = \hat d_i^{\,-g}
```

over the k strongest received anchors (*Weighted centroid: anchors k*, default 4;
0 = all) with the exponent g (*Weighted centroid: exponent g*, default 1). The
estimate always lies inside the convex hull of the anchors used, so the method is
strongly biased (towards the anchors and the centre of the anchor layout); it is
robust and it never fails when at least one anchor is received. It is also the
starting point of the iterative methods.

### Trilateration, LLS and WLS multilateration

These methods solve the range equations ‖**p** − **a**_i‖² = r_i² after making them
linear. Coordinates are first centred on the mean position of the anchors, which
improves the conditioning of the systems.

**LLS multilateration** (linear least squares). The equation of a reference anchor
*r*, the strongest received one, is subtracted from the others, which removes
‖**p**‖²:

```math
2\,(\mathbf a_i - \mathbf a_r)^{\mathsf T}\mathbf p = \lVert\mathbf a_i\rVert^2 - \lVert\mathbf a_r\rVert^2 - r_i^2 + r_r^2,
\qquad i \neq r
```

The system (one equation per other received anchor) is solved in the
least-squares sense. All equations have the same weight, although the error of
r_i² grows quickly with the distance: far anchors, whose ranges are the noisiest,
weigh as much as near ones, and the error grows fast at large σ.

**WLS multilateration** (weighted least squares). No reference equation is
subtracted: ‖**p**‖² is kept as an extra unknown and every received anchor gives one
equation

```math
-2\,\mathbf a_i^{\mathsf T}\mathbf p + \lVert\mathbf p\rVert^2 = r_i^2 - \lVert\mathbf a_i\rVert^2
```

in the D + 1 unknowns (**p**, ‖**p**‖²). Each equation is weighted by the inverse
variance of its right-hand side. With log-normal noise, d̂_i = d_i·10^(−X/(10n)), so
the error of d̂_i² is approximately proportional to d_i² and its variance to d_i⁴:
the weights are w_i ∝ 1/d̂_i⁴ (computed from the 3D distance estimates, at least
0.1·d0). Near anchors dominate, as they should.

**Trilateration**. The classic minimal solution: for each fix, the D + 1 strongest
received anchors (3 in 2D, 4 in 3D) are taken in order of RSS, skipping any anchor
that is aligned with those already chosen (on their line in 2D, on their line or
plane in 3D, within 1 mm); the linearised equations of those anchors (as in LLS,
with the strongest as reference) are then solved exactly. Skipping aligned anchors
matters: anchors mounted on the same wall are collinear, and three collinear
anchors cannot fix a position.

**Common rules.**

* At least D + 1 received anchors are needed (3 in 2D, 4 in 3D).
* A fix whose anchors cannot determine the position gets no estimate rather than
  an arbitrary one: the system is declared singular when the smallest eigenvalue of
  its normal matrix is below 10⁻¹² times the largest one (for example all the
  received anchors on one line in 2D). This shows in the *Valid %* column.
* **3D with the anchors at one height.** If the anchors used in a fix span less
  than 5 cm in height, z cannot be obtained from the linear system. The methods then
  solve (x, y) from 3 anchors (the linear system in the horizontal plane) and take
  the height from the residual range: Δz² is the mean of r_i² − ‖(**a**_i −
  **p**)_xy‖² over the anchors used, and z = z_a ± Δz, where z_a is the anchor
  height; of the two solutions the one closer to the middle of the target volume is
  kept. For trilateration this applies when all the enabled anchors are at one
  height.

### NLS multilateration

Nonlinear least squares on the ranges:

```math
\hat{\mathbf p} = \arg\min_{\mathbf p}\ \sum_{i\ \mathrm{received}} \big(\lVert\mathbf p - \mathbf a_i\rVert - r_i\big)^2
```

solved with Levenberg-Marquardt (below), starting from the weighted centroid. It
does not suffer from the linearisation of LLS, but all ranges still have the same
weight.

### MLE (Levenberg-Marquardt)

The maximum-likelihood estimator for log-normal shadowing. If the per-fix RSS of
each anchor is Gaussian in dB around the model, with the same standard deviation,
the log-likelihood of a position is, up to constants, minus the sum of the squared
RSS residuals:

```math
\hat{\mathbf p} = \arg\min_{\mathbf p}\ \sum_{i\ \mathrm{received}}
\Big(\mathrm{RSS}_i - P_{0,\mathrm{est}} + 10\,n_\mathrm{est}\log_{10}\frac{d_i(\mathbf p)}{d_0}\Big)^2
```

where d_i(**p**) is the 3D distance between the candidate position (with the tag
height in 2D) and anchor *i*, at least 0.1·d0. Working on the RSS rather than on
the ranges, it weighs every anchor correctly for this noise model. It is solved with
Levenberg-Marquardt from the weighted centroid; like every local optimisation it can
stop in a local minimum when the noise is large or the geometry poor.

**Levenberg-Marquardt, as used by NLS and MLE.** For every fix the step **s**
solves (JᵀJ + λ·diag(JᵀJ))·**s** = Jᵀ**res**, where J is the Jacobian of the
residuals; a step that lowers the cost is accepted and λ is multiplied by 0.3,
otherwise it is rejected and λ is multiplied by 5 (λ starts at 10⁻² and stays
within [10⁻⁷, 10⁷]). Iterations stop when the step is shorter than 0.1 mm, when λ
reaches 10⁷ or after 200 iterations. In 3D the starting height is the middle of the
target volume. Fixes with fewer than D + 1 received anchors get no estimate.

### MLE (grid search)

The same likelihood as the MLE (Levenberg-Marquardt), maximised by exhaustive
search over a grid of candidate positions:

* candidates at the centres of the cells of a grid over the target area, with cells
  of about *MLE grid resolution* (default 0.5 m; the size is adjusted to divide the
  area exactly); in 3D on height layers about *3D: vertical grid spacing* apart
  (default 0.5 m);
* for each candidate **c** the model RSS H_i(**c**) of every anchor, computed once;
* the estimate is the candidate that minimises Σ_i (RSS_i − H_i(**c**))² over the
  received anchors.

With *Map-aware (walls in the MLE grid and PF)* (default on) the model includes the
losses of the obstacles of the scenario (multi-wall model, with the true materials
and heights) and the candidates are restricted to free space. The map-aware
estimator therefore knows the floor plan exactly; this is the best case of a
model-based method and typically far more accurate than the plain path-loss
inversion in buildings with walls. The search finds the global optimum (no local
minima) but its result is quantised to the grid, its cost grows with the number of
candidates times the number of anchors, and it needs at least one received anchor.

## 4. Fingerprinting methods

### Survey

The fingerprinting methods learn from a **survey**, the offline phase of
fingerprinting, simulated in the same scenario and with the same radio settings as
the experiment:

* **Reference points** on a grid over the target area with spacing *Survey grid
  spacing* (default 2 m), starting half a spacing from the corner of the area; points
  inside obstacles are removed. In 3D the grid is repeated on height layers: the
  target volume is divided into N_z = round((z1 − z0)/Δz) slices (Δz = *3D: vertical
  grid spacing*) and the layers are at the centres of the slices; a flat volume has
  one layer. Because of the half-spacing offset, a test grid with the same spacing
  never falls on the reference points; the log notes when test points coincide with
  reference points, since errors there are optimistic.
* **Fingerprints.** At every reference point the survey records *Fingerprints per
  point* fingerprints (default 20). Each fingerprint is obtained like a position fix:
  the average of *Samples per fix* readings through the same channel as the
  experiment (the same walls, the same shadowing realisation, independent fading and
  receiver). The survey of N reference points therefore holds N × fingerprints per
  point training examples.
* **Radio map.** The mean fingerprint of each reference point (the average of its
  fingerprints, over those in which each anchor was received) forms the radio map
  used by WKNN.
* **Missing readings.** In the survey and in the fixes, an anchor that is not
  received enters the models as sensitivity − 5 dB (−155 dBm when the sensitivity
  cut-off is off). These are the *features* of the models.
* **New realisations.** With *New shadowing realisation at every trial* the survey
  is repeated and the models are trained again in every trial; otherwise they are
  trained once. Training time is measured separately from the time per fix (*Train
  s* column).

The survey costs N_ref × fingerprints per point × samples per fix readings per
anchor; it is generated in blocks, so large surveys do not need large memory.

### k-NN

The k nearest neighbours among all the survey fingerprints, in signal space:

```math
\hat{\mathbf p} = \frac{\sum_{j \in \mathcal N_k} w_j\,\mathbf y_j}{\sum_{j \in \mathcal N_k} w_j},
\qquad w_j = \frac{1}{\lVert \mathbf x - \mathbf x_j\rVert + 10^{-6}}\ \text{(or } w_j = 1\text{)}
```

where **x** is the feature vector of the fix, **x**_j and **y**_j are the features
and the position of fingerprint *j*, the distance is Euclidean in dB, and 𝒩_k are
the k fingerprints closest to **x** (*k-NN: neighbours K*, default 10). *k-NN:
weights 1/distance (else uniform)* chooses the weights (default 1/distance). Using
every fingerprint rather than their mean keeps the spread of the RSS at each
reference point in the model. There is no training phase: the fingerprints are
stored.

### WKNN (radio map)

The classic radio-map variant (as in RADAR): the same weighted average, with weights
1/distance, over the K nearest reference points, comparing the fix with the mean
fingerprint of each reference point (*WKNN: neighbours K*, default 4).

### Decision tree

A regression tree (CART, squared-error splits) that predicts the position
(x, y, and z in 3D) from the feature vector, with at least *Decision tree: min. per
leaf* fingerprints in each leaf (default 5; larger values give smoother, less
overfitted trees). A single tree is fast but its estimates are piecewise constant.

### Random forest

An average of *Random forest: trees* regression trees (default 100), each grown on a
bootstrap sample of the fingerprints, trying √A randomly chosen anchors at each
split. Averaging many decorrelated trees reduces the variance of a single tree. The
trees are grown 10 at a time (the same trees as when they are grown at once), so
that *Stop* stays responsive, using all the processor cores.

### Neural network (MLP)

A multilayer perceptron with ReLU hidden layers (*Neural network: hidden layers*,
default 64, 64) that regresses the position from the feature vector:

* inputs standardised anchor by anchor (zero mean, unit standard deviation);
  positions centred and divided by a single scale, the largest standard deviation
  of their coordinates;
* trained with Adam (learning rate 0.003, mini-batches of 512, the scikit-learn
  default L2 penalty), one epoch at a time;
* early stopping on 10% of the fingerprints held out at random: training stops
  after 15 epochs without improvement of the validation error (at most 300 epochs)
  and the weights of the best epoch are kept.

### Implementation

k-NN and WKNN are implemented with NumPy; the decision tree, the random forest and
the neural network use scikit-learn (`DecisionTreeRegressor`,
`RandomForestRegressor`, `MLPRegressor`), with a random state drawn from the
stream of the method.

## 5. Comparing the methods

| Method | Uses | Received anchors needed | Bias without noise | Notes |
|---|---|---|---|---|
| Min-Max | ranges | 1 | yes | very cheap; biased |
| Weighted centroid | distances, k strongest anchors | 1 | yes, strong | robust; inside the anchor hull |
| Trilateration | ranges, D + 1 strongest anchors | D + 1 | no | exact without noise; uses only near anchors |
| LLS multilateration | ranges of all received anchors | D + 1 | no | unweighted: far anchors hurt at large σ |
| WLS multilateration | ranges, weights 1/d⁴ | D + 1 | no | linear and well weighted |
| NLS multilateration | ranges | D + 1 | no | iterative; equal weights |
| MLE (Levenberg-Marquardt) | RSS residuals | D + 1 | no | with the grid search, the most accurate model-based method at high SNR; local minima at large σ |
| MLE (grid search) | RSS residuals, walls if map-aware | 1 | grid quantisation | global optimum; knows the floor plan when map-aware |
| k-NN, WKNN | survey | 1 | depends on the survey | no model; learns walls and static shadowing |
| Decision tree, random forest, MLP | survey | 1 | depends on the survey | trained models; training time per survey |

D is the number of estimated coordinates (2 or 3). What to expect, from
[VALIDATION.md](VALIDATION.md):

* Without noise, trilateration, LLS, WLS, NLS and the MLE recover the position to
  numerical precision; Min-Max and the weighted centroid do not.
* With i.i.d. noise the error grows in proportion to σ at high SNR; the MLE is the
  most accurate, WLS and NLS come close, LLS degrades at large σ.
* With static (spatially correlated) shadowing, fingerprinting is much more accurate
  than the distance-based methods, because the survey contains the deviations from
  the path-loss law that the model ignores; with i.i.d. shadowing the two families
  are equivalent.
* A finer survey trains better models until its spacing resolves the shadowing
  field (below its correlation distance).

## 6. Error statistics

For every fix with an estimate the error is

```math
e = \lVert \hat{\mathbf p} - \mathbf p \rVert
```

computed in the estimation space: horizontal in 2D, 3D in 3D. RSSILab reports, over
the fixes with an estimate:

| Statistic | Definition |
|---|---|
| Mean | mean of e (m) |
| RMSE | sqrt(mean of e²) (m) |
| MSE | mean of e² (m²) |
| Std | standard deviation of e (population) |
| Min, Median, P90, P95, Max | order statistics of e |
| Valid % | share of fixes with an estimate |
| RMSE xy, RMSE z | 3D only: RMSE of the horizontal and of the vertical error |
| ms/fix | computation time per fix, training excluded |
| Train s | training time per survey (fingerprinting) |

**Per test point** (Positioning), over all the trials and fixes on that point, with
**m** the mean estimate:

```math
\mathrm{bias} = \lVert \mathbf m - \mathbf p \rVert,\qquad
\mathrm{spread} = \sqrt{\operatorname{mean}\lVert \hat{\mathbf p} - \mathbf m \rVert^2},\qquad
\mathrm{MSE} = \mathrm{bias}^2 + \mathrm{spread}^2
```

The bias measures systematic errors (model mismatch, walls, static shadowing, a
coarse survey, the bias of Min-Max and of the weighted centroid), the spread the
random ones (fading, i.i.d. shadowing). Static shadowing is the same in all the fixes
on a point: within one trial it appears as bias, across trials (with a new
realisation each time) as spread. The *All* row of the table gives the statistics
over all points, with bias and spread as RMS over the points.

**Per trial** (Tracking): the statistics of each trial and, with more than one
trial, their mean and standard deviation across trials.

The error CDF, the box plots and the CSV exports are described in
[GUIDE_gui.md](GUIDE_gui.md).

## 7. Adding your own method

A method is a Python function `f(rss, ctx)` registered with `@register`, in the
distance-based or in the fingerprinting family; it then appears in the Algorithms
tab and runs on exactly the same RSS as the built-in methods. `ctx` gives the
anchors, the estimator model, the ranges, the MLE grid and, for fingerprinting
methods, the survey with a cache for trained models.
[GUIDE_custom_algorithms.md](GUIDE_custom_algorithms.md) explains how to write, load
and evaluate your own methods, with complete examples.

## 8. References

* P. Bahl and V. N. Padmanabhan, "RADAR: an in-building RF-based user location and
  tracking system", Proc. IEEE INFOCOM, 2000 (fingerprinting with nearest
  neighbours in signal space).
* K. Langendoen and N. Reijers, "Distributed localization in wireless sensor
  networks: a quantitative comparison", Computer Networks, 43(4), 2003 (Min-Max).
* J. Blumenthal, R. Grossmann, F. Golatowski and D. Timmermann, "Weighted centroid
  localization in Zigbee-based sensor networks", IEEE International Symposium on
  Intelligent Signal Processing, 2007.
* K. W. Cheung, H. C. So, W.-K. Ma and Y. T. Chan, "Least squares algorithms for
  time-of-arrival-based mobile location", IEEE Transactions on Signal Processing,
  52(4), 2004 (linearised least squares).
* N. Patwari, J. N. Ash, S. Kyperountas, A. O. Hero, R. L. Moses and N. S. Correal,
  "Locating the nodes: cooperative localization in wireless sensor networks", IEEE
  Signal Processing Magazine, 22(4), 2005 (RSS-based maximum likelihood).
* L. Breiman, "Random forests", Machine Learning, 45(1), 2001.
