# dPCA: the maths, and how it links to a *sequence* (not a *direction*) of attractors

Written 2026-07-15. Companion to
[`where_we_are_HMM_rSLDS_dimred.md`](where_we_are_HMM_rSLDS_dimred.md) (§3.2 states the result)
and [`HMM_maths_and_readouts.md`](HMM_maths_and_readouts.md) (the directedness statistic this
doc contrasts dPCA with). This one is the **mathematics of demixed PCA** and the **conceptual
bridge** from it to the project's central claim — *decision-making is a temporal **sequence**
of metastable/attractor states, shared across areas, realised on area-specific manifolds*.

The one-line version: **dPCA gives you the geometry and existence of the shared sequence, in a
coordinate system that says nothing about its direction. Direction and attractor-vs-stopwatch
are the dynamics' job (HMM `forward_fraction`, rSLDS `order_consistency` / `delta_recurrent`).
dPCA and the dynamics are complementary halves of the same claim, and it is a category error to
ask either to do the other's work.**

Code: `src/dynamics/dpca.py` · run in `notebooks/02_dimred_msr.ipynb` §5–6.

---

## 1. The problem dPCA solves — PCA cannot say *why* a direction varies

Trial-averaged population activity is a tensor

$$X_{n}(t, s, d)$$

— firing rate of neuron $n$ at peri-stimulus time $t$, for stimulus side $s \in \{L, R\}$ and
the animal's decision $d \in \{L, R\}$. Ordinary PCA finds the orthonormal directions
$\mathbf{w}_k$ capturing the most variance of $X$ (flattened over $t,s,d$). Its components are a
*mixture*: PC1 is typically dominated by the huge condition-independent onset transient, with a
little stimulus and a little decision tuning smeared into it. You cannot read "how much of this
region's geometry is decision-related" off a PCA component, because the component does not
correspond to any single cause.

dPCA (Kobak, Brendel et al., *eLife* 2016;5:e10989) fixes this by **decomposing the variance by
cause first, then reducing dimension** — while keeping every component a *linear, decodable*
readout of the full population (so it is still a real projection of the data, not a fit to the
means only).

---

## 2. Step one — the marginalisation decomposition (an ANOVA on the population)

Split $X$ into additive parts, each depending on a subset of the task parameters. Using
averaging operators (denote by $\langle\cdot\rangle_{p}$ the mean over parameter $p$), and
writing $\phi$ for a subset of $\{s, d, t\}$:

$$X \;=\; \bar{X} \;+\; X_{t} \;+\; X_{s} \;+\; X_{d} \;+\; X_{sd} \;+\; X_{st} \;+\; X_{dt} \;+\; X_{sdt}$$

Each marginalisation is the part of the activity that depends on *exactly* the parameters in
$\phi$, obtained by averaging over the parameters **not** in $\phi$ and subtracting all
lower-order terms (inclusion–exclusion, exactly as in a factorial ANOVA). The three that carry
the project's argument:

- **Condition-independent (time):** $\;X_{t} = \langle X \rangle_{s,d} - \bar{X}\;$ — the part
  every trial shares regardless of stimulus or choice. The onset transient, arousal, movement
  ramp. **Usually the largest by far.**
- **Stimulus:** $\;X_{s} = \langle X \rangle_{d,t} - \bar X\;$ (a per-stimulus offset), plus the
  time-varying $X_{st} = \langle X\rangle_{d} - X_t - X_s - \bar X$. We report **"stimulus %"** =
  variance in $X_s + X_{st}$ (the package also emits an $X_{ts}$ bookkeeping term; we add it in).
- **Decision:** the analogous $X_d + X_{dt}$, reported as **"decision %"**.

The key algebraic fact — **why this is a clean variance split** — is that these marginalisations
are mutually **orthogonal** under the Frobenius inner product (the averaging makes cross-terms
vanish, the ANOVA orthogonality). Hence the total variance partitions **exactly**:

$$\lVert X - \bar X\rVert_F^2 \;=\; \lVert X_t\rVert_F^2 + \lVert X_s\rVert_F^2 + \lVert X_d\rVert_F^2 + \lVert X_{sd}\rVert_F^2 + \dots$$

So "time = 74 %, stimulus = 12 %, decision = 5 %" is a genuine, non-overlapping accounting of
where a region's population variance comes from. This split alone is already informative — but
its components can be high-dimensional, so dPCA adds a second step.

---

## 3. Step two — demix *and* reduce, with an encoder/decoder pair

For each marginalisation $\phi$, dPCA seeks a small number $q$ of components that (i) **capture**
$X_\phi$ and (ii) are a **linear readout of the whole data** $X$. It minimises

$$\mathcal{L} \;=\; \sum_{\phi}\; \big\lVert\, X_\phi \;-\; F_\phi\, D_\phi\, X \,\big\rVert_F^2 ,
\qquad D_\phi \in \mathbb{R}^{q\times N},\; F_\phi \in \mathbb{R}^{N\times q}.$$

- $D_\phi$ is the **decoder**: $\,\mathbf{z}_\phi = D_\phi X\,$ are the demixed components you plot
  (component 1 of $\phi = t$ is the "time axis", etc.).
- $F_\phi$ is the **encoder**, mapping the components back to reconstruct the marginalisation.

The one structural difference from PCA is the whole point: **PCA forces $F = D^\top$ with
orthonormal rows** (encode and decode along the same axes), whereas **dPCA lets $F_\phi \neq
D_\phi^\top$**. Relaxing orthonormality is exactly what lets a decision component be *decoded*
from the full activity while being *reconstructed* only onto the decision marginalisation — i.e.
demixed. Each $\phi$'s subproblem is a **reduced-rank regression** of target $X_\phi$ on regressor
$X$, with the closed form (ridge-regularised in practice, `regularizer=None` here since the design
is balanced):

$$F_\phi D_\phi \;=\; \big(X_\phi X^\top\big)\big(X X^\top + \lambda I\big)^{-1}\ \text{truncated to rank } q \text{ via the SVD of } X_\phi X^\top (XX^\top)^{-1/2}.$$

**Explained variance per marginalisation** — the number we actually report — is the fraction of
$\lVert X - \bar X\rVert_F^2$ captured by $\phi$'s components. That is what
`variance_by_marginalization` returns and what the §6 bar chart shows.

---

## 4. The result on our data (40 ms, session 5c0c560e)

| region | stimulus % | decision % | time % | **decision − stimulus** |
|---|---:|---:|---:|---:|
| VISp | 11.8 | 5.0 | 74.1 | **−6.8** |
| SCm  | 4.5  | 6.3  | 63.5 | **+1.8** |
| MRN  | 8.3  | 14.3 | 49.2 | **+6.0** |

*(the 40 ms re-run in nb 02 reproduces the ordering the 50 ms nb-04 run first found — so the
split is not a bin-size artefact.)* Two honest caveats travel with it, always:

1. **Time dominates everywhere** (74 → 64 → 49 %). Most of *any* raw trajectory plot is this one
   component. See §5 — that is not a nuisance here, it is the sequence backbone.
2. **The decision cells hold only K = 25 balanced trials** (the off-diagonal stimulus×decision
   cells are error trials; mice are ~85 % correct). The decision components are noisy. **Pool the
   5 replicate sessions before the ordering is a claim.**

---

## 5. The bridge: dPCA and a *sequence* of attractors

This is the part the project needs written down. There are four distinct statements, and keeping
them apart is what keeps the argument honest.

### 5.1 The condition-independent component *is* the sequence backbone

The claim under test is that the decision unfolds as a **temporal sequence of metastable
states**. If a population really hops through states $A \to B \to C$ on (almost) every trial,
then averaging over trials and conditions does **not** wash the sequence out — it traces the
**centroids of those states, in order**, as a smooth curve in firing-rate space. That curve *is*
$X_t$, the condition-independent marginalisation. So:

> **dPCA's largest component, the one it exists to quarantine as "just time", is precisely the
> geometric shadow of the metastable sequence.** Its dominance (49–74 %) is not a distraction
> from the project's claim — it is the claim's trial-averaged signature.

The stimulus and decision components are then **modulations riding on that backbone**: *where
along the sequence*, and *how strongly*, the population's position is bent by what the animal saw
and chose. "Decision % rises VISp → MRN" therefore reads as *the shared sequence carries more
choice-aligned deflection as you move into motor structures* — shared alphabet, area-specific
inflection. That is the project's framing, measured.

### 5.2 dPCA is direction-*agnostic* — it gives existence and shape, never arrow-of-time

A dPCA axis, like a PCA axis, is defined only **up to sign**: $D_\phi$ and $-D_\phi$ fit equally
well. The demixed trajectory has an intrinsic **order** (it is parameterised by $t$), but dPCA
attaches no *preferred direction* to the state space it lives in, and it makes no statement that
the population is *driven* from one end to the other. It establishes:

- **that a reproducible temporal trajectory exists** (the time component is large and structured),
- **its geometry** (how stimulus/decision bend it, how much variance each carries),

and nothing about **directedness**. This is the exact sense in which dPCA speaks to *sequence,
not direction*. It is the geometric complement of the dynamics statistics:

| method | object | what it establishes | blind to |
|---|---|---|---|
| **dPCA** ($X_t$ share, geometry) | trial-averaged trajectory | a shared temporal sequence **exists**, and its shape | direction; whether it is *driven* |
| **rSLDS `order_consistency`** | regime entry order | there is a **reliable order** (either direction) | direction (by design) |
| **HMM `forward_fraction`** | Viterbi transitions | given an ordering, traffic runs **forward** | whether the order is attractor-driven |
| **rSLDS `delta_recurrent`** | recurrence term $R$ | switching depends on **latent position**, not elapsed time (attractor vs **stopwatch**) | — |

Reading across the row is the whole argument: dPCA + `order_consistency` say *there is a
sequence*; `forward_fraction` says *it runs forward*; `delta_recurrent` says *it is pulled, not
timed*. **No single one of them is the claim.**

### 5.3 What dPCA can and cannot say about *attractors*

dPCA is **linear and trial-averaged**, so it cannot see an attractor as a dynamical object — no
fixed point, no contraction rate, no basin. It sees the **centroids** the attractors would
produce and the order they appear in. Concretely:

- A genuine attractor sequence and a smooth condition-independent ramp with no metastability
  produce **the same $X_t$**. dPCA cannot tell them apart. (This is the same identifiability wall
  that makes the *time* component so large.)
- The evidence that the centroids are **attractors** — regions the trajectory is *drawn into and
  held* — comes only from the rSLDS eigenvalues ($|\lambda|<1$ contracting) and, crucially, from
  `delta_recurrent` distinguishing *pulled to a place* from *timed by a clock*.

So dPCA sets the **geometry and the ordering that a dynamical model must then explain**; it is a
map of where to look, not proof of what is there. Used this way it is powerful and honest; used
as "we found the decision manifold" it would be overclaiming a linear trial-average.

### 5.4 "Shared across areas, area-specific manifolds" — dPCA's two-sided contribution

The project's geometry half has two clauses, and dPCA touches both:

- **Shared sequence:** the condition-independent $X_t$ is the same *kind* of object in every area
  (a large, time-structured backbone). The joint-PCA manifold (nb 02 §9) makes the "shared" part
  literal — the missing ~0.42 of variance is cross-area covariance, the common signal.
- **Area-specific realisation:** the *split* differs by area (decision − stimulus: −6.8 → +6.0).
  Same sequence, area-specific inflection — which is exactly *"shared latent dynamics on
  geometrically distinct, area-specific manifolds"*, stated as a measured quantity rather than a
  slogan.

---

## 6. Limitations — stated plainly

1. **Trial-averaged.** dPCA sees centroids, not single-trial dynamics; it cannot establish
   metastability, attractors, or direction. Those are §2 of the rSLDS.
2. **Linear.** A curved sequence is approximated by linear axes; a strongly nonlinear manifold
   would be distorted (GPFA/PHATE are the checks).
3. **K = 25 trials/cell.** The decision components are noisy; the ordering needs the 5-session
   pool before it is a claim.
4. **The time component is unidentifiable between "sequence of states" and "smooth ramp."** dPCA
   quantifies the backbone; only the dynamics can say what kind of object it is.
5. **Sign/direction is arbitrary.** Never read an arrow of time off a dPCA axis — that is
   `forward_fraction`'s job, and it needs an *externally* imposed ordering to mean anything.

---

## ✅ Tell me which to expand

- **P1** — the inclusion–exclusion (ANOVA) algebra of the marginalisations, worked through for
  $\{s, d, t\}$, and why the terms are orthogonal.
- **P2** — reduced-rank regression: the closed form, the role of the regulariser, and why
  $F \neq D^\top$ is what demixes.
- **P3** — the full mapping table (dPCA / `order_consistency` / `forward_fraction` /
  `delta_recurrent`) expanded into a single worked argument on our numbers.
- **P4** — the identifiability wall in §5.3: what experiment or model *could* separate an
  attractor sequence from a smooth ramp.
- **P5** — how a *joint* dPCA (pseudo-population over the 5 sessions) would be built, and why a
  pseudo-population is valid for dPCA but not for the HMM/rSLDS.
