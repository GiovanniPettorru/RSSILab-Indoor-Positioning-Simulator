# Propagation model

This guide describes how RSSILab generates the received signal strength (RSS): the
path-loss law, the obstacles, the shadowing, the fading, the receiver and the
averaging that turns readings into a position fix, with the formulas and the
parameter values used by the code. It also describes the *estimator model*, which is
what the distance-based positioning methods believe about the channel.

The fields of the interface are listed in [GUIDE_gui.md](GUIDE_gui.md) (Radio
tab); the positioning methods are described in
[GUIDE_algorithms.md](GUIDE_algorithms.md).

## Contents

1. [Overview](#1-overview)
2. [Distance](#2-distance)
3. [Path loss](#3-path-loss)
4. [Obstacles: the multi-wall model](#4-obstacles-the-multi-wall-model)
5. [Shadowing](#5-shadowing)
6. [Fading](#6-fading)
7. [Receiver: quantisation and sensitivity](#7-receiver-quantisation-and-sensitivity)
8. [From readings to a position fix](#8-from-readings-to-a-position-fix)
9. [The estimator model](#9-the-estimator-model)
10. [The legacy model](#10-the-legacy-model)
11. [Technology presets](#11-technology-presets)
12. [Assumptions and limitations](#12-assumptions-and-limitations)
13. [Where it is implemented](#13-where-it-is-implemented)

## 1. Overview

For every enabled anchor *a* and every reading *k*, RSSILab computes the RSS in dBm
received at the target position **p**:

```math
\mathrm{RSS}_{a,k}(\mathbf p) \;=\; P_0 \;-\; 10\,n\,\log_{10}\frac{d_a(\mathbf p)}{d_0}
\;-\; \sum_{o\,\in\,\mathcal O_a(\mathbf p)} L_o(f) \;+\; S_a(\mathbf p) \;+\; F_{a,k}
```

| Symbol | Meaning | Section |
|---|---|---|
| P0 | mean RSS at the reference distance d0 (dBm) | [3](#3-path-loss) |
| n | path-loss exponent | [3](#3-path-loss) |
| d_a(**p**) | 3D distance between anchor *a* and the target (m) | [2](#2-distance) |
| 𝒪_a(**p**) | obstacles crossed by the straight link between anchor *a* and the target | [4](#4-obstacles-the-multi-wall-model) |
| L_o(f) | loss of obstacle *o* at the carrier frequency f (dB per crossing) | [4](#4-obstacles-the-multi-wall-model) |
| S_a(**p**) | shadowing (dB), i.i.d. per reading or a static field per anchor | [5](#5-shadowing) |
| F_a,k | fading (dB), independent for every reading | [6](#6-fading) |

The first three terms are the *mean RSS*: they depend only on the geometry and are
computed once for every position of an experiment. The random terms are then added,
the result is quantised and compared with the receiver sensitivity (a reading below
it is missing), and `samples per fix` consecutive readings are averaged into one
position fix:

1. mean RSS: path loss and obstacles;
2. + shadowing;
3. + fading;
4. quantisation (rounding to the RSSI step);
5. sensitivity cut-off: missing reading (NaN);
6. average of the received readings, in dB: one RSS value per anchor and fix.

The channel is reciprocal: the RSS is the same whether the target or the anchors
transmit. All the anchors share one technology (frequency, power, path loss and
random components).

## 2. Distance

Anchors and target have three coordinates (metres, room centred at (0, 0), heights
measured from the floor), and the path loss uses their 3D distance

```math
d_a(\mathbf p) = \max\big(\lVert \mathbf p - \mathbf a \rVert,\ 0.1\,d_0\big).
```

The lower limit 0.1·d0 avoids the singularity of the logarithm when the target is
on top of an anchor. In 2D estimation the target is at the known *Tag height*; in
3D it moves inside the target volume. Anchors at a different height from the target
are therefore never at distance zero, even when the target passes below them.

## 3. Path loss

RSSILab uses the log-distance model:

```math
\overline{\mathrm{RSS}}(d) = P_0 - 10\,n\,\log_{10}\frac{d}{d_0}\qquad [\mathrm{dBm}]
```

**P0** is the mean RSS at the reference distance d0. It is typed by hand or, with
*P0 from free-space loss at d0*, computed from the link budget:

```math
P_0 = P_\mathrm{TX} + G - \mathrm{FSPL}(d_0, f),\qquad
\mathrm{FSPL}(d_0,f) = 20\log_{10} d_0 + 20\log_{10} f - 147.55\ \mathrm{dB}
```

with P_TX the transmit power (dBm), G the antenna gains minus the system losses
(dB), d0 in metres and f in hertz; this is 20·log10(4π·d0·f/c). At 2.4 GHz and
d0 = 1 m the free-space loss is 40.05 dB, so a 20 dBm transmitter gives
P0 = −20.05 dBm.

**n** sets how fast the mean RSS decreases with distance: 2 in free space,
typically between about 2 and 4 indoors (higher with clutter and without line of
sight). Walls are not part of n: they are modelled separately (section 4).

**Range at sensitivity.** The Radio panel shows the distance at which the mean RSS,
without walls, reaches the sensitivity S:

```math
d_\mathrm{max} = d_0\,10^{(P_0 - S)/(10\,n)}
```

It is a nominal value: shadowing and fading make readings drop below the
sensitivity well before d_max, and some readings are still received beyond it.

## 4. Obstacles: the multi-wall model

Every obstacle that the straight line between anchor and target crosses subtracts
its loss once (multi-wall, or Motley-Keenan, model):

```math
L_\mathrm{walls} = \sum_{o\,\in\,\mathcal O_a(\mathbf p)} L_o(f)
```

There are no reflections, no diffraction and no ray tracing; the loss of an obstacle
does not depend on the angle of incidence, on the wall thickness or on the length
of the path inside it.

### Obstacle types and the crossing test

| Type | Geometry | When a link crosses it |
|---|---|---|
| Wall | segment between two points of the floor plan, any orientation, with a thickness | the link crosses the centre line of the wall in plan view, and the height of the link at the crossing point lies within the height range of the wall |
| Block | axis-aligned box: rectangle in plan view times a height range | the link (a 3D segment) intersects the box, faces included (slab test) |

Every obstacle has a height range [z0, z1]: bottom at z0 (0 = floor) and top at z1
(empty = up to the ceiling). Links are always traced in 3D, also in 2D estimation,
where the target is at the tag height. For a wall crossed at the fraction t of the
link from anchor **a** to target **p**, the height of the link there is
z_a + t·(z_p − z_a), and the wall attenuates the link only if that height is
between z0 and z1. As a consequence a link that passes above a desk or a half wall,
or below a hanging obstacle, is not attenuated, while a link from an anchor near
the ceiling to a target behind a full-height wall is.

The thickness of a wall is used only for drawing it and for motion (section
[Obstacles and motion](#obstacles-and-motion)); a door is a gap between two wall
segments.

### Losses per material

The loss of an obstacle is the value of its material in `MATERIAL_TABLE`
(`rssilab/model.py`), in dB per crossing, interpolated linearly in log10(f) between
the frequencies of the table and held constant outside them:

| Material | 0.868 GHz | 2.4 GHz | 5 GHz | 6.5 GHz |
|---|---|---|---|---|
| Drywall | 2.0 | 3.0 | 4.0 | 5.0 |
| Glass | 1.5 | 2.5 | 4.0 | 5.0 |
| Wood | 2.5 | 4.0 | 6.0 | 7.0 |
| Brick | 5.0 | 8.0 | 12.0 | 14.0 |
| Concrete | 9.0 | 13.0 | 20.0 | 24.0 |
| Metal | 20.0 | 26.0 | 30.0 | 32.0 |
| Furniture | 1.0 | 2.0 | 3.0 | 3.5 |

For example, a brick wall costs 8.08 dB at 2.437 GHz and 12.73 dB at 5.5 GHz. The
values are indicative, in line with the usual indoor references (ITU-R P.1238 and
P.2040, COST 231); calibrate them with your measurements. The table is the only
source of these values in the simulator, so editing it changes them everywhere. A
*Loss override* (dB per crossing, the same at every frequency) replaces the
material value for one obstacle.

### Obstacles and motion

Obstacles also define where the target can be. A wall occupies the band of half its
thickness on each side of its centre line, a block its box; points inside an
obstacle (within its height range) are not free. Free space is used for random test
points, test grids, the fingerprint survey, the routes of the A* planner, and the
candidate positions of the map-aware grid search and of the particle filter.
Through the Python API an obstacle can be made transparent to motion
(`Obstacle.blocks_motion = False`): it still attenuates links but no longer limits
where the target can go.

## 5. Shadowing

Shadowing is the slow variation of the RSS around the path-loss law caused by the
environment along the link. RSSILab models it as log-normal, a zero-mean Gaussian
in dB with standard deviation σ (*Shadowing std*), and offers two kinds, chosen by
the *Shadowing correlation* distance d_c.

### i.i.d. shadowing (d_c = 0)

A new, independent value S ~ N(0, σ²) is drawn for every reading, anchor and
position. This is the model of the original RSSILab dataset generator. Shadowing of
this kind behaves like an additional noise term: averaging N_s readings reduces its
standard deviation to σ/√N_s, and the same place gives different values at every
visit.

### Spatially correlated, static shadowing (d_c > 0)

Each anchor has its own random field S_a(x, y): a zero-mean Gaussian field with
standard deviation σ and correlation

```math
\rho(r) = \mathbb E\big[S_a(\mathbf p)\,S_a(\mathbf q)\big]/\sigma^2 = \exp\!\left(-\frac{r^2}{d_c^2}\right),
\qquad r = \lVert \mathbf p - \mathbf q \rVert_{xy}
```

so that the correlation is 1/e = 0.37 at a distance d_c and about 0.02 at 2·d_c. The
field is *static*: the same place always has the same shadowing, at every reading,
fix and visit. It therefore does not average out, it biases every distance-based
estimate at that place, and it is what a fingerprint survey can learn. This makes
the correlated model the realistic choice for comparing fingerprinting with
distance-based methods.

How a field is generated:

1. A regular grid covers the target area and every position measured in the
   experiment, enlarged by 3·d_c on each side (so that the periodic FFT does not
   correlate opposite edges). The grid spacing is Δ = max(d_c/8, 0.1 m), multiplied
   by 1.25 as many times as needed to keep the grid below 250,000 nodes.
2. White Gaussian noise on the grid is filtered in the frequency domain with a
   Gaussian kernel of standard deviation d_c/2. A Gaussian kernel of standard
   deviation s gives the correlation exp(−r²/(4s²)), that is exp(−r²/d_c²).
3. The filtered field is normalised to zero mean and unit variance and scaled by σ.
4. The field at a position is the bilinear interpolation of the four surrounding
   nodes, divided by the standard deviation of that interpolation (which depends on
   the position within the cell and on the correlation of neighbouring nodes), so
   that the standard deviation is σ everywhere, at the nodes and between them.

When the grid had to be coarsened beyond d_c/3, correlations at distances shorter
than about Δ are smoothed (the standard deviation is still σ) and the log of the
run says so. The fields of different anchors are independent. A field depends on
(x, y) only: in 3D every height above the same point of the floor plan has the same
shadowing. With *New shadowing realisation at every trial* every trial draws new
fields (and, for the fingerprinting methods, a new survey); otherwise all trials
share the fields of the first. Section 5 of [VALIDATION.md](VALIDATION.md) checks
the standard deviation and the correlation of the generated fields.

### What averages out

| Component | Changes between readings | Averaged by `samples per fix` | Same at every visit of a place |
|---|---|---|---|
| i.i.d. shadowing | yes | yes | no |
| static shadowing | no | no | yes |
| fading | yes | yes (around its mean) | no |

## 6. Fading

Fading is the fast variation of the RSS caused by multipath. RSSILab draws it
independently for every reading and anchor (no temporal correlation, no Doppler)
and adds it in dB. The physical models (Rayleigh, Rician) give the power of a
channel h with unit mean power, E|h|² = 1, so that fading does not change the mean
received *power*; its mean in *dB* is negative.

| Fading | Value added to the RSS (dB) | Mean (dB) | Std (dB) |
|---|---|---|---|
| None | 0 | 0 | 0 |
| Rayleigh | 10·log10\|h\|², \|h\|² exponential with mean 1 | −2.51 | 5.57 |
| Rician, K = 0 dB | 10·log10\|h\|² (see below) | −2.06 | 5.09 |
| Rician, K = 3 dB | | −1.55 | 4.38 |
| Rician, K = 6 dB | | −0.96 | 3.28 |
| Rician, K = 10 dB | | −0.41 | 2.00 |
| Rician, K = 15 dB | | −0.14 | 1.10 |
| Legacy, σ_L = 2 | σ_L·sqrt(−2 ln U), U uniform in (0, 1] | +2.51 | 1.31 |

**Rayleigh** (no line of sight): |h|² follows an exponential distribution with
mean 1. In dB, the mean is −10γ/ln 10 = −2.507 dB (γ is the Euler-Mascheroni
constant) and the standard deviation is (π/√6)·(10/ln 10) = 5.570 dB.

**Rician** with factor K (*Rician K*, in dB; K is the ratio of the line-of-sight
power to the scattered power):

```math
h = \sqrt{\frac{K}{K+1}} + \sqrt{\frac{1}{2(K+1)}}\,\big(X + jY\big),\qquad X, Y \sim \mathcal N(0, 1)
```

with K in linear units. A large K approaches no fading; K → −∞ dB is Rayleigh. The
mean and the standard deviation of 10·log10|h|² have no simple closed form: RSSILab
computes them by numerical integration of the density of ln|h|² (they are used by
the oracle calibration and shown in the Radio panel).

**Legacy** (*Legacy (Rayleigh amplitude in dB)*): the term σ_L·sqrt(−2 ln U) of the
original RSSILab dataset generator. It is a Rayleigh-distributed amplitude with
scale σ_L (*Legacy fading sigma*) added in dB: always positive, with mean
σ_L·sqrt(π/2) and standard deviation σ_L·sqrt(2 − π/2). It is not the power of a
fading channel and is kept only to reproduce that dataset (section 10).

To avoid log(0), |h|² is limited to 10⁻¹² (−120 dB).

## 7. Receiver: quantisation and sensitivity

After shadowing and fading, each reading goes through the receiver:

1. **RSSI quantisation**: the reading is rounded to the nearest multiple of the step
   q (*RSSI quantisation*, dB); 0 switches it off. Real radios report the RSSI in
   discrete steps, typically of 1 dB.
2. **Sensitivity**: a reading below the sensitivity S (*Sensitivity*, dBm) is not
   received and becomes missing (NaN). Values of −150 dBm or less switch the
   cut-off off.

Missing readings are what limits the range of real anchors: far anchors, and anchors
behind several walls, are heard only part of the time. The log of a run reports
the share of readings below the sensitivity.

## 8. From readings to a position fix

The receiver takes one reading per anchor every 1/f_s seconds (*Sample rate* f_s),
from all the anchors at the same instants. A **position fix** averages
N_s = *Samples per fix* consecutive readings of each anchor:

```math
\mathrm{RSS}_a^\mathrm{fix} = \frac{1}{|K_a|}\sum_{k \in K_a} \mathrm{RSS}_{a,k}
```

where K_a is the set of readings of anchor *a* that were received; the average is
taken in dB. An anchor with no received reading in the fix has a missing value, and
a fix in which no anchor is received gets no estimate from the positioning methods
(the Kalman and particle filters predict through it).

* In **Positioning** the target stands still: the N_s readings of a fix are taken
  at the same test point, and every test point gets `fixes per point` independent
  fixes in each trial.
* In **Tracking** the target moves at constant speed v: consecutive readings are
  v/f_s metres apart along the route, the ground truth of a fix is the mean position
  of its readings, its time is the middle of its averaging window, and the interval
  between fixes is N_s/f_s seconds. Averaging more readings reduces the noise but
  also smears the fix along the route.

**Standard deviation of a fix.** If σ_F is the standard deviation of the fading in
dB, the standard deviation of the per-fix RSS around its mean is

```math
\sigma_\mathrm{fix} =
\begin{cases}
\sqrt{\sigma^2 + \sigma_F^2/N_s} & \text{static shadowing } (d_c > 0)\\[4pt]
\sqrt{(\sigma^2 + \sigma_F^2)/N_s} & \text{i.i.d. shadowing } (d_c = 0)
\end{cases}
```

This is the *Per-fix RSS std* shown in the Radio panel and used by the oracle
calibration; it does not include the effect of quantisation and of missing
readings. Averaging in dB does not remove the mean of the fading: a fix carries the
fading bias of the table of section 6 (−2.51 dB with Rayleigh fading).

The same channel produces the RSS datasets of *Export RSS dataset*, with one row per
reading (no averaging), and the fingerprints of the survey, each averaged like a
fix (see [GUIDE_algorithms.md](GUIDE_algorithms.md#survey)).

## 9. The estimator model

The distance-based methods and the particle filter do not know the true channel:
they use an **estimator model** (P0_est, n_est, d0, σ_est), set in *Estimator
calibration*:

* **Oracle**: n_est = n, P0_est = P0 + E[F] (the true P0 plus the mean of the fading
  in dB, which is what a calibration at d0 would measure), σ_est = σ_fix of
  section 8.
* **Manual**: P0_est, n_est and σ_est as typed, to study model mismatch (for example
  a wrong path-loss exponent, or a P0 measured with a different device).

Even the oracle model knows only the mean path loss: it does not know the shadowing
realisation, and only the map-aware grid search and the map-aware particle filter
include the obstacles (with their true losses; see
[GUIDE_algorithms.md](GUIDE_algorithms.md)). The fingerprinting methods do not use
the estimator model at all: they learn from the survey.

**Inverting the model.** A measured RSS gives the distance estimate

```math
\hat d = d_0\,10^{(P_{0,\mathrm{est}} - \mathrm{RSS})/(10\,n_\mathrm{est})}
```

and, in 2D estimation, the horizontal range sqrt(d̂² − Δz²), where Δz is the known
height difference between anchor and target. With a correct model and Gaussian noise
of standard deviation σ_fix in dB, d̂ = d·10^(−X/(10n)), X ~ N(0, σ_fix²): its median
is the true distance but its mean is larger, by the factor
exp(½·(σ_fix·ln 10/(10n))²). The error grows with the distance (it is multiplicative),
which is why the weighted and the maximum-likelihood methods trust near anchors
more.

## 10. The legacy model

The *Legacy log-normal model* preset reproduces the channel of the original RSSILab
dataset generator:

```math
\mathrm{RSS} = -40 - 40\,\log_{10} d + 3\,N(0,1) + 2\sqrt{-2\ln U}
```

that is P0 = −40 dBm at d0 = 1 m, n = 4, i.i.d. shadowing with σ = 3 dB, legacy
fading with σ_L = 2, no quantisation, no sensitivity and one reading per fix. The
fading term has mean +2.51 dB, so a distance inversion that uses P0 = −40 dBm alone
underestimates distances by a factor 10^(−2.51/40) = 0.87, that is by about 13%; the
oracle calibration adds the 2.51 dB to P0. With the reference scenario, a 1 m grid
and 10,000 samples per point, *Export RSS dataset* reproduces the layout and the
channel of the original dataset; section 5 of [VALIDATION.md](VALIDATION.md)
compares the two generators statistically.

## 11. Technology presets

*Load preset* in the Radio tab fills every radio field. The presets are indicative
starting points, not certified radio parameters. P0 is computed from the transmit
power with d0 = 1 m, except for the legacy model; the last three columns are
derived from the others (sections 3, 6 and 8).

| Preset | f (GHz) | TX power, gains | P0 (dBm) | n | σ (dB) | d_c (m) | Fading | Sensitivity (dBm) | Quantisation (dB) | Rate (Hz) | Samples per fix | Fading mean / std (dB) | σ_fix (dB) | Range at sensitivity (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Legacy log-normal model | 2.4 | P0 typed | −40.00 | 4.0 | 3 | 0 | Legacy, σ_L = 2 | off | off | 10 | 1 | +2.51 / 1.31 | 3.27 | unlimited |
| Wi-Fi 2.4 GHz | 2.437 | 20 dBm, 0 dB | −20.19 | 3.0 | 4 | 3 | Rician K = 3 dB | −90 | 1 | 10 | 5 | −1.55 / 4.38 | 4.45 | 212 |
| Wi-Fi 5 GHz | 5.5 | 20 dBm, 0 dB | −27.26 | 3.3 | 5 | 2 | Rician K = 3 dB | −88 | 1 | 10 | 5 | −1.55 / 4.38 | 5.37 | 69 |
| Wi-Fi HaLow 868 MHz (802.11ah) | 0.866 | 14 dBm, 0 dB | −17.20 | 2.8 | 5 | 5 | Rician K = 3 dB | −100 | 1 | 5 | 5 | −1.55 / 4.38 | 5.37 | 906 |
| BLE 2.4 GHz | 2.426 | 0 dBm, −5 dB | −45.15 | 2.5 | 4 | 2 | Rayleigh | −95 | 1 | 10 | 10 | −2.51 / 5.57 | 4.37 | 99 |
| Zigbee / 802.15.4 | 2.405 | 3 dBm, 0 dB | −37.07 | 2.8 | 4 | 2 | Rayleigh | −97 | 1 | 4 | 4 | −2.51 / 5.57 | 4.87 | 138 |
| LoRa 868 MHz | 0.868 | 14 dBm, 0 dB | −17.22 | 2.7 | 6 | 6 | Rician K = 3 dB | −137 | 1 | 0.5 | 1 | −1.55 / 4.38 | 7.43 | 27,300 |
| UWB CH5 (RSS only) | 6.5 | −14.3 dBm, 0 dB | −63.01 | 2.2 | 3 | 1.5 | Rician K = 6 dB | −95 | 0.25 | 20 | 5 | −0.96 / 3.28 | 3.34 | 28.5 |

The ranges are those of the path-loss law alone: walls shorten them, by
10^(L/(10n)) for a total wall loss L.

## 12. Assumptions and limitations

* Single floor. Obstacles are vertical walls (any orientation in plan) and
  axis-aligned blocks, each with a height range; each crossed obstacle adds its loss
  once. There are no reflections, no diffraction and no ray tracing.
* All the anchors share one technology: the same transmit power, path-loss exponent,
  shadowing and fading statistics. Per-anchor differences can be emulated with
  obstacles or with plugins.
* Fading is independent between readings (no Doppler, no temporal correlation);
  shadowing is either i.i.d. per reading or static in space, independent between
  anchors, and does not vary with height.
* The receiver reads every anchor at every sampling instant; device heterogeneity,
  antenna orientation and body shadowing are not modelled.
* The fingerprint survey is taken in the same realisation of the environment as the
  experiment and with the same receiver: changes between survey and use (moved
  furniture, people, a different device) are not modelled.
* Material losses and technology presets are indicative: calibrate them with your
  measurements.

## 13. Where it is implemented

| Part of the model | Code |
|---|---|
| distances, mean RSS (path loss + obstacles) | `channel.distances`, `channel.mean_rss` |
| P0 from the link budget, range at sensitivity | `model.Technology.p0`, `model.Technology.nominal_range_m` |
| crossing tests, total obstacle loss | `geometry.obstacle_hits`, `geometry.wall_loss_matrix` |
| material losses | `model.MATERIAL_TABLE`, `model.material_loss` |
| correlated shadowing field | `channel.ShadowingField` |
| fading samples and their statistics | `channel.fading_db`, `channel.fading_stats` |
| one realisation of the environment, readings with receiver | `channel.Environment.measure` |
| average of the readings of a fix | `channel.average_fixes` |
| per-fix standard deviation | `channel.effective_sigma` |
| estimator model | `engine.estimator_model`, `localization.EstimatorModel` |
| technology presets | `model.PRESETS` |
