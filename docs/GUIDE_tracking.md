# Tracking: routes, Kalman filter and particle filter

In *Tracking* mode the target moves along a route and RSSILab estimates its
position fix after fix. This guide describes how routes are built and sampled and
the two trackers: a constant-velocity Kalman filter that smooths the fixes of any
positioning method, and a particle filter that works directly on the RSS. The
positioning methods are described in [GUIDE_algorithms.md](GUIDE_algorithms.md),
the channel in [GUIDE_propagation.md](GUIDE_propagation.md).

## Contents

1. [Routes](#1-routes)
2. [The A* planner](#2-the-a-planner)
3. [From the route to the fixes](#3-from-the-route-to-the-fixes)
4. [Kalman filter](#4-kalman-filter)
5. [Particle filter](#5-particle-filter)
6. [Results and practical notes](#6-results-and-practical-notes)
7. [References](#7-references)

## 1. Routes

The route is a polyline (a sequence of 3D points) chosen in the Experiment tab:

| Route | How it is built | Obstacles |
|---|---|---|
| Waypoints (click on map) | the waypoints in their order (at least 2), each with its own height in 3D; the height changes linearly along each leg | each leg routed with A* when *Route around obstacles* is on |
| Random walk | *Random walk: waypoints* points drawn at random among the free cells of the planning grid (from the seed), with random heights inside the target volume in 3D, joined in that order | always routed with A* |
| Circle | an ellipse centred on the target area, with semi-axes *Circle: radius fraction* times the half sizes of the area, run *Circle: laps* times, at the tag height (180 points per lap) | ignored |
| Lawnmower | back-and-forth rows parallel to the x axis, *Lawnmower: row spacing* apart (at least 0.5 m), kept max(clearance, 0.5 m) from the border of the target area, at the tag height | rows routed with A* when *Route around obstacles* is on |

Waypoints can also be placed outside the target area; the other routes stay inside
it. When the final route still crosses a motion-blocking obstacle (circle, A* off,
or no route found), the log warns that the target walks through it. **Preview
route** draws the route and logs its length and duration.

## 2. The A* planner

Routes avoid obstacles with the A* algorithm on an occupancy grid of the target
area:

* **Grid.** Square-ish cells of about *A\* grid resolution* (default 0.5 m), in plan
  view. A cell is blocked when its centre lies inside a motion-blocking obstacle
  grown by *Clearance from obstacles* (default 0.3 m) plus half the cell diagonal,
  so that thin walls cannot leak between cells. The planner works in plan view:
  the target walks on the floor, so desks and other low blocks are obstacles to
  walking even though links pass above them.
* **Search.** A* on the 4-connected grid (moves to the four neighbouring cells, unit
  cost) with the Euclidean distance to the goal as heuristic. Start and goal are
  moved to the nearest free cell when they fall in a blocked one.
* **Smoothing.** The staircase path of the grid is shortened by line-of-sight
  shortcuts: from each point the route jumps to the farthest following point that is
  visible without crossing blocked cells.
* **No route.** When the goal cannot be reached (for example a closed room), the leg
  becomes a straight segment and the log reports it.

## 3. From the route to the fixes

* The target moves at constant speed v (*Speed*) along the route, measured in 3D.
* Readings are taken every 1/f_s seconds (*Sample rate* in the Radio tab), that is
  every v/f_s metres along the route, from all the anchors at the same instants. A
  run is limited to 400,000 readings per anchor: lower the sample rate, raise the
  speed or shorten the route when the limit is reached.
* Every N_s consecutive readings (*Samples per fix*) form one fix; readings left at
  the end that do not fill a fix are dropped, and at least two fixes are needed.
* The **ground truth** of a fix is the mean position of its N_s readings and its
  **time** is the middle of its averaging window. Fixes are Δt = N_s/f_s seconds
  apart.

The RSS of each reading is generated at the position of that reading (see
[GUIDE_propagation.md](GUIDE_propagation.md#8-from-readings-to-a-position-fix)), so
a fix averages readings taken along a stretch of v·Δt metres of the route.

## 4. Kalman filter

*Kalman filter (constant velocity)* adds the method **KF + *input***: a linear Kalman
filter applied to the fixes of the *KF input* method (any positioning method of the
chosen approach, computed even when it is not selected).

**Model.** The state is the position and the velocity, **x** = (**p**, **v**), with
D = 2 or 3 coordinates each:

```math
\mathbf x_{k+1} = \begin{bmatrix} I & \Delta t\,I\\ 0 & I\end{bmatrix}\mathbf x_k + \mathbf w_k,
\qquad
\mathbf z_k = \begin{bmatrix} I & 0\end{bmatrix}\mathbf x_k + \boldsymbol\nu_k
```

* Process noise: a white random acceleration with standard deviation q (*KF
  acceleration std*, default 0.5 m/s²), integrated over one fix interval:
  Q = q²·G·Gᵀ with G = [Δt²/2·I; Δt·I] (the discrete white-noise acceleration model).
* Measurement noise: the fixes of the input method, with covariance R = r²·I, where r
  is *KF measurement std*.

**Automatic measurement std** (r = 0, the default). In each trial r is estimated
from the input fixes: with Δ**z** the differences between consecutive fixes (all
coordinates together),

```math
r = \max\!\Big(0.3\ \mathrm{m},\ \frac{1.4826\cdot\operatorname{median}|\Delta \mathbf z|}{\sqrt 2}\Big)
```

1.4826·median|·| is a robust estimate of a standard deviation, and the division by
√2 accounts for each difference containing the errors of two fixes; with fewer than 3
valid differences r = 3 m. The estimate assumes that the target moves little between
two fixes compared with the error of a fix; otherwise it overestimates r and the
filter smooths more. The value of the first trial is reported in the log.

**Recursion.** The filter starts at the first fix with an estimate, with position
equal to that fix, zero velocity and covariance diag(r²·I, 4·I) (velocity standard
deviation 2 m/s). At every following fix it predicts with the model and, when the
input fix has an estimate, corrects with the standard Kalman update; a fix without
an estimate gives the prediction only. The output is the filtered position at each
fix (a causal filter: no smoothing with future fixes), clipped to the target area
when *Clip estimates to the target area* is on.

**What it can and cannot do.** The filter reduces the random part of the error of its
input, the more the slower the target and the higher the fix rate. It cannot remove
a bias of the input (model mismatch, walls, static shadowing): a biased method stays
biased after filtering. Its time per fix includes the time of the input method when
that method is not selected.

## 5. Particle filter

*Particle filter (path-loss likelihood)* adds **PF (RSS, map-aware)**, or **PF (RSS)**
when the estimator is not map-aware. It is a sequential importance resampling (SIR)
filter that estimates the position directly from the RSS of each fix with the
likelihood of the estimator model. Since it uses the path-loss model, it belongs to
the distance-based family and is not available with the Fingerprinting approach.

**State and initialisation.** N particles (*PF particles*, default 1000, at least
50), each with a position and a velocity. Positions are drawn uniformly in the free
cells of the MLE grid (see
[GUIDE_algorithms.md](GUIDE_algorithms.md#mle-grid-search)); velocities are
Gaussian with standard deviation v/√D per coordinate, where v is the speed of the
target (at least 0.1 m/s), and with a smaller vertical component (×0.3) in 3D.

**Prediction**, for every fix, with Δt the fix interval:

```math
\mathbf v \leftarrow \mathbf v + \mathcal N\big(0,\ (\sigma_a\,\Delta t)^2 I\big),\qquad
\mathbf p' = \mathbf p + \mathbf v\,\Delta t + \mathcal N\big(0,\ j^2 I\big)
```

with σ_a the *PF acceleration std* (default 0.5 m/s²) and j the *PF position
jitter* (0 = automatic: 0.1·v·Δt, at least 0.02 m). In 3D the height is kept
inside the target volume.

**Constraints.** A move is rejected when the new position is outside the target
area (volume in 3D) and, with map-aware estimation, when it falls in a cell occupied
by an obstacle or when the step crosses a motion-blocking wall (with its height
range). A rejected particle stays where it was and draws a new random velocity, as if
it had bounced.

**Random injection.** At every fix a fraction *PF random injection* of the particles
(default 0.02, at most 0.5) is redrawn uniformly in free space with a new velocity
and weight 1/N. This lets the filter recover when the cloud is trapped in the
wrong room behind a wall.

**Update.** Each particle is weighted by the likelihood of the RSS of the fix:

```math
w \leftarrow w\cdot\exp\!\Big(-\frac12\sum_{i\ \mathrm{received}}\Big(\frac{\mathrm{RSS}_i - H_i(\mathbf p)}{\sigma_L}\Big)^2\Big)
```

where H_i(**p**) is the model RSS of anchor *i* in the cell of the MLE grid that
contains the particle (with the losses of the obstacles when map-aware) and σ_L is
the *PF likelihood std* (0 = automatic: the σ of the estimator model, at least 1 dB).
A fix without readings leaves the weights unchanged.

**Estimate and resampling.** The estimate is the weighted mean of the particles.
When the effective number of particles 1/Σw² falls below N/2, the particles are
resampled with systematic resampling and the weights reset to 1/N.

**Cost.** Each fix costs a likelihood evaluation per particle and anchor, plus a
crossing test per particle and wall when map-aware: the time per fix grows with the
number of particles and of walls. The resolution of the likelihood is that of the
MLE grid.

## 6. Results and practical notes

* In Tracking, the **Per trial** table gives the statistics of each trial (and their
  mean and standard deviation across trials); the **Map** shows the ground truth and
  the tracks of the first trial, with a time slider and **Play**; **Error detail**
  shows the error against time. See [GUIDE_gui.md](GUIDE_gui.md).
* The filters run only in Tracking mode; in Positioning they are ignored.
* Choose as KF input a method with little bias: the filter removes noise, not bias.
  In buildings with walls and static shadowing, the map-aware grid search and the
  fingerprinting methods are usually better inputs than the plain path-loss
  inversion (section 6 of [VALIDATION.md](VALIDATION.md)).
* For the particle filter, a larger likelihood std makes the filter more tolerant of
  model mismatch (walls not in the model, static shadowing) at the price of a wider
  cloud; more particles are needed for larger areas; a larger injection recovers
  faster from wrong rooms but adds noise.
* A higher sample rate or fewer samples per fix give more fixes per metre, which
  helps the filters, but each fix is noisier.

## 7. References

* P. E. Hart, N. J. Nilsson and B. Raphael, "A formal basis for the heuristic
  determination of minimum cost paths", IEEE Transactions on Systems Science and
  Cybernetics, 4(2), 1968 (A*).
* Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to
  Tracking and Navigation*, Wiley, 2001 (Kalman filter, white-noise acceleration
  model).
* N. J. Gordon, D. J. Salmond and A. F. M. Smith, "Novel approach to
  nonlinear/non-Gaussian Bayesian state estimation", IEE Proceedings F, 140(2), 1993
  (SIR particle filter).
* M. S. Arulampalam, S. Maskell, N. Gordon and T. Clapp, "A tutorial on particle
  filters for online nonlinear/non-Gaussian Bayesian tracking", IEEE Transactions on
  Signal Processing, 50(2), 2002.
