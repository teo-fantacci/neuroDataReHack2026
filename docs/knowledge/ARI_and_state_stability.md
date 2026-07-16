# The state-stability analysis, and what the Adjusted Rand Index actually measures

Methods note, 2026-07-13. Companion to
[`choosing_K_for_the_HMM.md`](choosing_K_for_the_HMM.md), which explains *why* held-out
likelihood fails to select K. This one explains the criterion we use instead, in detail.

Code: `src/dynamics/hmm.py::state_stability` / `stability_scan`.
Runner: `scripts/run_state_stability.py` (all trials, EM restarts).

---

## 1. The idea in one sentence

> A state that appears when you fit trials 1–100 but vanishes when you fit trials 101–200 is
> not an attractor. It is a fitting artifact.

Our scientific claim is that decision-making passes through **metastable states** — that
identifiable, **reproducible** regimes exist. Held-out likelihood never tests that; it asks
"does this model predict spikes well", which rewards flexibility and is why the CV curve
never turns over. **Reproducibility is the claim, so measure reproducibility.**

## 2. The protocol

For each candidate K, and for each of `n_repeats` random splits:

1. **Partition the trials into three disjoint thirds: A, B, C.**
2. **Fit one HMM on A.** **Fit a completely separate HMM on B.** They share no trials and no
   parameters. Each is fitted `n_restarts` times, keeping the best-likelihood solution
   (see §6).
3. **Decode both models on C**, a set *neither* has ever seen. Viterbi gives two state
   sequences over the same time bins.
4. **Score the agreement between those two sequences.**

Average over repeats; sweep K; the **peak** is the answer.

### Why the third set C?

If we decoded on A ∪ B, each model would be scoring partly on its own training trials —
artificially good, and asymmetrically so. Judging both on a common, unseen C makes the
comparison symmetric and guarantees the agreement cannot come from either model having
memorised those trials.

### Why compare decodings, not parameters?

The obvious alternative is to match states between the two fits by their emission-rate
vectors λ_k and score the similarity. We deliberately do **not**, because it describes the
wrong thing. The claim is not "the two fits learned similar firing rates" — it is that
**trials pass through the same regimes at the same times**. Parameter similarity is

* **not sufficient** — two models can learn near-identical λ's and still deploy them
  differently across the trial; and
* **not necessary** — two models can parameterise the same temporal segmentation slightly
  differently and still agree perfectly about *when* the population is in which regime.

The decoded sequence **is** the scientific object. Score that.

---

## 3. Why the metric must ignore labels

HMM state labels are arbitrary. EM has no canonical ordering, so model A's "state 3" bears
no relation to model B's "state 3". (`fit_poisson_hmm` sorts states by occupancy
centre-of-mass, which gives *an* ordering — but that ordering is itself unstable: when two
states have similar timing the sort flips, and we would score a mere permutation as a
disagreement.)

So the metric must be **permutation-invariant**: it must compare the *partition* of time,
not the names attached to the parts.

## 4. The Rand Index

The Rand Index sidesteps labels by looking at **pairs of time bins**.

Take any two time bins *i* and *j*. Each decoding makes a **label-free** statement about that
pair: *same state* or *different states*. Two decodings agree about the pair if they both
say "same" or both say "different". Across all pairs:

| | decoding 2: same | decoding 2: different |
|---|---|---|
| **decoding 1: same** | a  (agree) | c |
| **decoding 1: different** | d | b  (agree) |

$$\mathrm{RI} = \frac{a + b}{a + b + c + d} = \frac{a+b}{\binom{n}{2}}$$

= the fraction of pairs the two decodings treat the same way. Notice that no state name ever
enters — relabelling either decoding leaves *a, b, c, d* untouched. That is exactly the
permutation-invariance we need.

### Why raw RI is not usable here

RI is badly inflated by chance, **and inflated differently at different K** — which is fatal,
because comparing across K is the whole point of the sweep.

The reason: as K grows, two *random* decodings put most pairs in "different states" simply
because there are more states to fall into. Those accidental "different/different" pairs
count as agreement in *b*. So RI drifts toward 1 with K for purely combinatorial reasons.
A K-selection curve built on raw RI would slope upward and tell you nothing — the same
failure mode as the likelihood.

## 5. The Adjusted Rand Index

ARI subtracts the agreement you would get **by chance** and rescales:

$$\mathrm{ARI} = \frac{\mathrm{RI} - \mathbb{E}[\mathrm{RI}]}{\max(\mathrm{RI}) - \mathbb{E}[\mathrm{RI}]}$$

where the expectation is taken over random labelings **with the same cluster sizes** (the
permutation / hypergeometric null). In contingency-table form, with n_ij bins in state *i*
of decoding 1 and state *j* of decoding 2, a_i and b_j the row/column sums:

$$\mathrm{ARI}=\frac{\sum_{ij}\binom{n_{ij}}{2}-\Big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\Big]\big/\binom{n}{2}}
{\tfrac12\Big[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\Big]-\Big[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\Big]\big/\binom{n}{2}}$$

Properties, and why each one matters to us:

| property | value | why we need it |
|---|---|---|
| permutation-invariant | — | state labels are arbitrary |
| chance level | **0** | so "no reproducible structure" has a *name*, and the criterion can say **no** |
| identical partitions | **1** | upper bound is meaningful |
| **comparable across K** | — | the chance correction removes the K-dependent inflation, which is the ONLY reason the ARI-vs-K curve can be read |
| can go **negative** | yes | agreement *worse* than chance — worth flagging if it ever happens |

`sklearn.metrics.adjusted_rand_score` implements this; we use it directly rather than
reimplementing.

## 6. The second metric, and why it is not the criterion

`matched_acc` — Hungarian-match the two label sets to maximise the confusion-matrix diagonal
(`scipy.optimize.linear_sum_assignment`), then report the fraction of bins that agree.

It is reported because it is **interpretable** ("the two models agree on 91% of time bins")
in a way ARI is not. But it is **not chance-corrected**, and under the null it *falls* with K
for free. Read alone it would always argue for the smallest K. **It is a readout, never the
criterion.** Always read it next to the ARI.

---

## 7. How to read the ARI-vs-K curve

* **A clear peak at K\*** → K\* reproducible states. This is the result.
* **Monotone decline from K=2** → the only structure two strangers agree on is the coarsest
  split (baseline vs evoked). The extra states do not replicate, and a "sequence of many
  metastable states" reading is not supported.
* **Near zero throughout** → no reproducible states at this bin size / trial count. A real
  finding, and one likelihood is constitutionally unable to report.
* **Rising to the edge of the K range** → suspicious. Check it is not a degenerate solution;
  do not take a boundary as an answer (that is the mistake the likelihood made twice).

## 8. Three ways this can mislead you

**(a) Stability is not correctness — a model can be *stably wrong*.**
If the data contain one strong, simple event (a rate step at stimulus onset), *both* fits
will find it every time and ARI at K=2 will be high. That validates nothing about metastable
attractors. So **"is the ARI high?" is the wrong question**; a high ARI at K=2 is nearly
guaranteed and scientifically empty. The right question is **"how far up in K does the ARI
stay high?"** — i.e. *how many* states reproduce.

**(b) It confounds "no such state" with "not enough trials to find it twice."**
The serious one. Each fit sees only ~1/3 of the trials. A real but subtle state may simply
not be recoverable *twice* from that many trials, and would present as instability — a false
negative. **Diagnostic: run the scan at several trial counts.** If the peak K marches right
as trials grow, you are data-limited, not state-limited. This is the same lesson the rSLDS
work reached independently (*"limited by TRIALS, not neurons"*), so expect it to bite.
Mitigation, applied in `scripts/run_state_stability.py`: use **every** usable trial in the
session, not a 400-trial subsample.

**(c) EM local minima depress the ARI.**
EM is non-convex. Two fits can disagree because they fell into different local optima from
different initialisations — bad *luck*, not different *data* — and a naive implementation
charges that to "the states are not real". This biases the ARI, and hence the selected K,
**downward**. Mitigation: `n_restarts > 1` refits each model several times and keeps the
best-likelihood solution, so a disagreement means different data. With `n_restarts=1` the
reported ARI is an **under-estimate** of true stability.

## 9. What we actually run

`scripts/run_state_stability.py`, on session `5c0c560e`:

* **all usable trials** (`MAX_TRIALS = None` → 843, not the 400-trial subsample) — mitigates (b)
* `n_restarts = 2` — mitigates (c)
* `n_repeats = 5` independent A/B/C splits per K
* units filtered at `MIN_RATE_HZ = 1.0` (see `select_active_units` for why)
* `K ∈ {2 … 10}`

Results are written to `results/hmm/state_stability_maxtrials.json` and recorded in
[`best_K_per_region.md`](best_K_per_region.md).
