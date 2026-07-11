# Recorded brain areas — IBL Brain-Wide Map vs SHIELD

Reference sheet for the two candidate datasets: which areas each records, their full
names (Allen CCF / Beryl), and what the literature says they do — with a bias toward the
**visual decision task** both datasets probe.

**Scope + honesty note.** Anatomy (what connects to what) is well established and cited.
*Task-functional* roles are less settled; for those I lean on the two large surveys that
used these exact paradigms — IBL et al. 2025 (*Nature* 645:177–189) and Steinmetz et al.
2019 (*Nature* 576:266–273) — and flag speculation as such. Full names are verified
against `iblatlas.BrainRegions`. Acronyms follow the Allen Mouse Brain Atlas
(Wang et al. 2020, *Cell* 181:936–953).

---

## Part 1 — IBL Brain-Wide Map (DANDI:000409)

**Task.** Head-fixed 2AFC: a grating of variable contrast appears left or right; the
mouse turns a wheel to bring it to centre. Includes a block-wise prior on stimulus side.
Decision variable is a signed choice (left/right).

**Our candidate visual→motor axis** (8 regions), and the **triplet** we fit first,
`{MRN, SCm, VISp}` — the only one that recurs across ≥5 sessions (see
`docs/worklog/2026-07-08_phase1-and-shield.md`).

### Cortex

| Acr | Full name | Function (anatomy = established; task = surveyed) |
|---|---|---|
| **VISp** | Primary visual area (V1) | First cortical stage of vision; retinotopic; receives dLGN input. In the task, shows the **earliest, shortest-latency stimulus divergence** of the cortical areas (IBL 2025). Steinmetz 2019 found visual cortex leads frontal cortex by ~40 ms only in a narrow pre-movement window — evidence for *distributed*, not strictly serial, flow. |
| **MOs** | Secondary motor area (premotor / M2; contains ALM rostrally) | Motor planning and preparation; **choice is decodable here** and MOs carries early choice signals (IBL 2025). The M2→superior-colliculus pathway is causally implicated in rodent visual decisions (Duan et al. 2021, *Nat Commun* 12:2727). |

### Thalamus

| Acr | Full name | Function |
|---|---|---|
| **LGd** | Dorsal lateral geniculate complex (dLGN) | First-order visual thalamus; the retina→dLGN→V1 relay is one of the best-characterised pathways in the brain. Short-latency stimulus responses. |
| **LP** | Lateral posterior nucleus (rodent pulvinar) | Higher-order ("associative") visual thalamus. Mediates **transthalamic** cortico-cortical routes and carries ascending superior-colliculus and retinal input (Oh 2014; Harris 2019). Candidate hub for indirect inter-areal signalling. |

### Basal ganglia

| Acr | Full name | Function |
|---|---|---|
| **CP** | Caudoputamen (dorsal striatum) | Main input nucleus of the basal ganglia; integrates cortical + thalamic + dopaminergic input for **action selection** and reinforcement. Large, reliably sampled unit yields in IBL (2,878 good units — the most of our axis). |

### Midbrain — *the two anchor regions of our triplet*

| Acr | Full name | Function |
|---|---|---|
| **SCm** | Superior colliculus, motor-related | Deep/intermediate SC layers driving **orienting movements**. Receives direct retinal input and is a major output target of visual cortex; **causally involved in visual decisions** in rodents. A natural sensorimotor bottleneck between vision and action. |
| **MRN** | Midbrain reticular nucleus | Premotor / arousal / movement-initiation. IBL 2025 places some of the **strongest subcortical choice signals** in midbrain reticular formation. Less classically "visual" than SCm, so a good test of whether decision dynamics are shared across areas with different tuning. |

### Hindbrain

| Acr | Full name | Function |
|---|---|---|
| **GRN** | Gigantocellular reticular nucleus | Medullary reticular formation near the motor output. Among the strongest encoders of **wheel movement / choice** brain-wide (IBL 2025). Late in the cascade — a candidate "sink" for the decision. Sparsely sampled (555 units, 14 sessions), so hard to co-record. |

**Why `{MRN, SCm, VISp}`.** It spans the axis end-to-end — cortical sensory input (VISp),
sensorimotor midbrain (SCm), premotor midbrain (MRN) — while being the only triplet IBL
records together often enough to treat sessions as replicates. It deliberately mixes a
"visual" area with two "motor/choice" areas, which is exactly the shared-dynamics /
distinct-geometry contrast the project is built on.

---

## Part 2 — SHIELD (DANDI:001051)

**Task.** Head-fixed **go/no-go visual change detection**: a stream of images plays; the
mouse licks when the image identity changes. Includes engagement/disengagement over the
session. Decision variable is go/no-go (detect vs withhold) — **not** a signed L/R choice,
so the IBL/NeuralFlow "decision-variable" framing must be re-justified before reuse.

**Coverage.** Up to 6 probes/session → far broader simultaneous sampling than IBL. The
list below is the union of areas seen in the sessions we streamed (a 6-probe session gave
**40 areas, 17 with ≥50 good units**); exact coverage varies by subject and session.

### Cortex — sensory & visual

| Acr | Full name | Function |
|---|---|---|
| **VISp** | Primary visual area | As above — the sensory driver of the change-detection task. |
| **VISrl** | Rostrolateral visual area | Higher-order visual cortex; multimodal (visual + somatosensory/motor) association. |
| **SSp** | Primary somatosensory area | Body/whisker representation; largest single-area yield in the example session (304). |
| **MOp** | Primary motor area (M1) | Motor execution (licking/movement). |
| **MOs** | Secondary motor area | Motor planning / premotor (see IBL). |

### Cortex — frontal / limbic (rich in SHIELD, sparse in IBL)

| Acr | Full name | Function |
|---|---|---|
| **PL** | Prelimbic area | Medial prefrontal; executive control, rule/value, decision. |
| **ILA** | Infralimbic area | Medial prefrontal; complementary to PL (often behavioural inhibition / extinction). |
| **ACAd / ACAv** | Anterior cingulate area, dorsal / ventral | Attention, action–outcome monitoring, task engagement. Directly relevant to SHIELD's engagement gating. |
| **ORBvl / ORBm** | Orbital area, ventrolateral / medial | Orbitofrontal; outcome value and expectation. |
| **RSPd / RSPv** | Retrosplenial area, dorsal / ventral | Spatial context and memory; visual–hippocampal interface. |

### Hippocampal formation (essentially absent from our IBL axis)

| Acr | Full name | Function |
|---|---|---|
| **CA1 / CA3** | Field CA1 / CA3 | Hippocampus proper; spatial and episodic/contextual coding. |
| **DG** | Dentate gyrus | Hippocampal input stage; pattern separation. |
| **SUB / ProS** | Subiculum / Prosubiculum | Principal hippocampal output; also appears in the IBL triplet session. |

### Thalamus

| Acr | Full name | Function |
|---|---|---|
| **RT** | Reticular nucleus of the thalamus | Inhibitory shell gating thalamocortical traffic; **attentional selection**. |
| **PO** | Posterior complex | Higher-order somatosensory thalamus (transthalamic routes). |
| **POL** | Posterior limiting nucleus | Multisensory posterior thalamus. |
| **VPM** | Ventral posteromedial nucleus | First-order somatosensory (whisker) relay. |
| **LD** | Lateral dorsal nucleus | Limbic thalamus; links to retrosplenial/hippocampal spatial system. |
| **AV** | Anteroventral nucleus | Anterior thalamus; head-direction / limbic. |
| **TH** | Thalamus (unresolved parent label) | Coarse label where finer assignment failed — treat with caution. |

### Striatum / septum / midbrain / olfactory

| Acr | Full name | Function |
|---|---|---|
| **CP** | Caudoputamen | Dorsal striatum (see IBL). |
| **ACB** | Nucleus accumbens | Ventral striatum; reward and motivation — relevant to lick/reward. |
| **LSc / LSr** | Lateral septal nucleus, caudal / rostral | Limbic; motivation, reward, anxiety. |
| **PAG** | Periaqueductal gray | Defensive/autonomic; behavioural state. |
| **SCs** | Superior colliculus, sensory-related | Superficial (visual) SC layers — the sensory counterpart of IBL's SCm. |
| **OP** | Olivary pretectal nucleus | Luminance / pupillary light reflex. |
| **AON / TTd / DP** | Anterior olfactory nucleus / Taenia tecta dorsal / Dorsal peduncular area | Olfactory–limbic; incidental to the visual task. |

### ⚠️ Non-tissue labels to exclude

`VL` (lateral ventricle), `V3` (third ventricle), `SEZ` (subependymal zone), plus
`No Area` / `root` / empty. These are **channel-localisation artifacts, not neural
populations** — drop any unit assigned to them before analysis. (`iblatlas` returns real
full names for the ventricles, which is why they look legitimate; they are not.)

---

## Cross-dataset summary

| | IBL 000409 | SHIELD 001051 |
|---|---|---|
| Task | 2AFC contrast, signed L/R choice | go/no-go change detection |
| Simultaneous areas/session (≥20 units) | ~3 on our axis (up to 9 overall) | ~17–29 (≥20 units), 40 total |
| Best for | task-matched, low-friction, replicate sessions | breadth: cortex + hippocampus + thalamus + midbrain at once |
| Frontal / hippocampal coverage | sparse | rich (PL, ILA, ACA, CA1/3, DG, SUB) |
| Decision variable | signed choice → matches NeuralFlow framing | detect/withhold → framing must be re-derived |

**Takeaway for area selection.** IBL is the substrate for the *core* claim on a compact,
task-matched visual→motor axis (`{MRN, SCm, VISp}`). SHIELD is the substrate for the
*ambitious* version — many systems recorded at once — at the cost of a different task and
hand-built preprocessing. Both remain open (per the human's decision, 2026-07-08).

---

## References

- Wang et al. 2020, *Cell* 181:936–953 — Allen Mouse Brain Common Coordinate Framework.
- Oh et al. 2014, *Nature* 508:207–214 — mesoscale connectome.
- Harris et al. 2019, *Nature* 575:195–202 — cortical connectivity by layer / cell class.
- IBL et al. 2025, *Nature* 645:177–189 — Brain-Wide Map (task-functional roles).
- Steinmetz et al. 2019, *Nature* 576:266–273 — distributed coding; visual-lead timing.
- Duan et al. 2021, *Nat Commun* 12:2727 — causal M2→SC pathway.
- Bennett et al. 2024, *Neuron* — SHIELD implant / dataset (DANDI:001051).

*Functional descriptions are deliberately conservative. Where a role is task-specific and
uncertain, prefer the IBL/Steinmetz surveys over textbook generalisation, and verify
before making a claim in the write-up.*
