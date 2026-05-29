# Thesis project context

## 1. The project

- **Title** — *The Explanatory Pluralism Test: Comparing Circuit-Level Explanations at Multiple Abstraction Levels*
- **Author** — Gabriele Volzone (matr. 1917002), `gabriele.volzone@gmail.com`
- **Institution** — Sapienza Università di Roma, Facoltà di Ingegneria dell'informazione, informatica e statistica, MSc Data Science
- **Advisor / Co-advisor** — Simone Scardapane / Marco Fasoli
- **Academic year** — 2025/2026

## 2. Research question

Is any level of abstraction in mechanistic interpretability (MI) uniformly best,
or does each serve different explanatory purposes? The thesis is an **empirical
test of explanatory pluralism**: the philosophical claim that
multiple levels of mechanism can be simultaneously warranted and that the right
level depends on the pragmatic goal of the explanation.

**Central conjecture under test.** There is a three-way trade-off among
*faithfulness*, *compressibility*, and *interventional precision*, and different
levels of description occupy different positions on this trade-off curve, with
no level dominating across all three. Prior MI work evaluates single levels;
this thesis is, to my knowledge, the first to compare levels directly against
common criteria on shared tasks.

## 3. Experimental design (one-screen summary)

**Model**   | GPT-2 small (124M) primarily; second model for robustness if feasible  
**Tasks**   | `ioi` (Wang et al., 2022) · `grokking` (Nanda et al., 2023)            
**Levels**  | `L1` neurons / params · `L2` attention heads + MLP sublayers · `L3` SAE features 
**Metrics** | `faithfulness` (recovery score) · `compressibility` (component count, bit-cost) · `intervention_precision` (predicted-vs-observed intervention effects) 
**Cross-level check** | Causal-abstraction / IIT (Geiger et al., 2021, 2022, 2025): does L3 abstract L2? does L2 abstract L1? (still not sure about this)

## 4. Repository structure

```
explanatory_pluralism_thesis/
│
├── data/                               # Pre-trained model checkpoints
│   ├── grokking_model.pt               # 1-layer transformer trained on modular addition (Nanda et al., 2023)
│   ├── grokking_sae.pt                 # Sparse autoencoder trained on the grokking model's MLP activations
│   ├── grokking_sae_l1_2e-1.pt         # Grokking SAE variant (L1 penalty 2e-1)
│   ├── gpt2_sae_layer3.pt              # SAE trained on GPT-2 Small layer 3 residual stream (IOI task)
│   └── gpt2_sae_layer6.pt              # SAE trained on GPT-2 Small layer 6 residual stream (IOI task)
│
├── notebooks/                          # Numbered analysis notebooks (run in order)
│   ├── 00_setup_and_data.ipynb         # Downloads and caches GPT-2 Small; environment setup
│   ├── 01_tasks.ipynb                  # Defines and validates the two tasks: IOI and grokking
│   ├── 02_level1_neurons.ipynb         # L1 — neuron / parameter level circuit extraction (grokking)
│   ├── 02bis_level1_neurons(IOI).ipynb # L1 — neuron / parameter level circuit extraction (IOI task)
│   ├── 03_level2_heads.ipynb           # L2 — attention heads + MLP sublayers (grokking task)
│   ├── 03bis_level2_heads(IOI).ipynb   # L2 — head-level activation patching (IOI task)
│   ├── 04_level3_sae.ipynb             # L3 — SAE on grokking MLP activations; SAE-feature circuits
│   ├── 04bis_level3_sae(IOI).ipynb     # L3 — SAE-feature circuits on GPT-2 Small (IOI task)
│   ├── 05_evaluation_criteria.ipynb    # Computes faithfulness, compressibility, interventional precision
│   ├── 06_causal_abstraction.ipynb     # Cross-level causal abstraction / IIT alignment checks (L3→L2→L1)
│   └── 07_results_and_plots.ipynb      # Aggregates all results and produces thesis figures
│
├── results/                            # Auto-generated outputs (committed for reproducibility)
│   ├── level1/                         # L1 scores (JSON) + recovery and cluster plots (grokking)
│   ├── level1IOI/                      # L1 scores (JSON) + neuron recovery plot (IOI task)
│   ├── level2/                         # L2 scores (JSON) + recovery plots (grokking)
│   ├── level2IOI/                      # L2 scores (JSON) + recovery plot (IOI task)
│   ├── level3/                         # L3 scores (JSON) + feature analysis and FFT plots (grokking)
│   ├── level3IOI/                      # L3 scores (JSON) + recovery and top-features plots (IOI task)
│   ├── criteria/                       # Three-criteria summary plot, Criterion 3 JSON, random-baseline plot
│   └── causal/                         # Causal abstraction alignment outputs (empty)
│
├── src/                                # Reusable Python modules imported by the notebooks
│   ├── __init__.py
│   ├── tasks.py                        # Dataset builders for IOI and grokking
│   ├── patching.py                     # Activation patching utilities
│   ├── metrics.py                      # Faithfulness, compressibility, and interventional precision
│   └── iit.py                          # Causal abstraction / IIT alignment helpers
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 5. Results

Results are organized by task and level. Each level is evaluated on three criteria: **faithfulness** (how well the circuit accounts for model behavior), **compressibility** (what fraction of the total components the circuit requires: lower is more compressed), and **interventional precision** (whether activating the circuit causally steers the model toward the predicted behavior).

---

### 5.1 Grokking task: modular arithmetic (mod 113), 1-layer transformer

The model solves `(a + b) mod 113` using Fourier features in the MLP. Three dominant frequencies are identified: **k = 14, 49, 17**.

#### Level 1: Neurons (Fourier frequency clusters)

The MLP has 512 neurons total. Clustering by their Fourier response assigns them to four groups:

| Cluster    | Neurons | Individual recovery |
|------------|---------|---------------------|
| freq\_14   | 250     | 0.456               |
| freq\_17   | 65      | 0.258               |
| freq\_49   | 72      | 0.144               |
| noise      | 125     | 0.142               |

Because all four groups are needed to recover the full behavior, the circuit is the entire MLP:

- **Faithfulness (C1):** 1.00 the neuron-level circuit perfectly accounts for model performance.
- **Compressibility (C2):** 1.00 no compression; all 512/512 neurons are included. This is the least compressed description.
- **Interventional precision (C3):** 0.995 patching only the three frequency clusters into a corrupted run steers the model to the correct answer on 99.5 % of examples. 

**Interpretation:** The neuron level tells the complete causal story but at the cost of no compression, you are essentially describing the whole MLP.

---

#### Level 2 Logit Fourier components

At this level the circuit is described as the set of trigonometric basis components `{cos(kx), sin(kx)}` that contribute to the logit of the correct answer. Six components exist; the greedy circuit selects four:

| Circuit component | Recovery score |
|-------------------|----------------|
| cos\_14           | 0.246          |
| sin\_14           | 0.318          |
| cos\_17           | 0.159          |
| sin\_17           | 0.131          |

The excluded pair `{cos_49, sin_49}` contributes only 0.15 combined and is dropped.

- **Faithfulness (C1):** 0.854 the four-component circuit recovers 85 % of the clean-vs-corrupted logit gap. Good but not perfect; the discarded frequency 49 carries genuine signal.
- **Compressibility (C2):** 0.667 the circuit needs 4 of the 6 available components. More compressed than L1, but not dramatically so.
- **Interventional precision (C3):** 0.655 patching the four circuit components correctly predicts the model's output on 65 % of examples. The Fourier-component level loses precision because it abstracts away *how* the frequencies interact through individual neurons.

**Interpretation:** The Fourier-component level gives the most recognizable mechanistic story (the model is doing clock arithmetic via Fourier features), but it is only a partial causal description.

---

#### Level 3 SAE features (Sparse Autoencoder, 2048 features)

The SAE reconstructs the MLP post-activations with 99.98 % variance explained, activating ~198 features per example on average. The greedy circuit selects features until 90 % of the logit gap is recovered:

- **Circuit size:** 158 features out of 2048 (7.7 % of all features).
- **Max single-feature RS:** 0.0415 no feature dominates alone; the circuit is highly distributed.
- **Faithfulness (C1):** 0.901 the 158-feature circuit recovers 90 % of the logit gap, matching the target.
- **Compressibility (C2):** 0.077 the circuit uses only 7.7 % of the dictionary; the SAE provides a substantially more compressed description than the raw neuron level.
- **Interventional precision (C3):** 1.000 patching the 158-circuit features into a corrupted run yields perfect prediction of the model output. The random baseline for 158 randomly chosen features gives a mean precision of 0.0007 (essentially zero), so the lift is essentially 1.0, confirming the circuit is not a reconstruction artifact.

**Interpretation:** The SAE-feature level achieves the best trade-off for grokking: near-perfect interventional precision (better than L2) at much higher compression than L1. However, it requires 158 components, making it harder to interpret intuitively than the 4-component L2 description.

---

### 5.2 IOI task Indirect Object Identification, GPT-2 Small

The task is identifying the indirect object in sentences like *"When Mary and John went to the store, John gave a book to \_\_\_"* (answer: *Mary*). The metric is the logit difference between the indirect-object name and the subject name.

#### Level 1 MLP Neurons (gradient attribution circuit)

GPT-2 Small has 12 layers × 3,072 MLP neurons = 36,864 neurons total. Gradient-attribution-based greedy selection identifies a circuit of 27 neurons:

- **Faithfulness (C1):** 0.911  27 neurons recover 91 % of the logit gap.
- **Compressibility (C2):** 0.00073 the circuit is extremely compact: 27 out of 36,864 neurons (0.073 %). This is the most compressed single-number description across all levels and tasks.
- **Interventional precision (C3):** 0.000 despite high faithfulness, patching only these 27 neurons into a corrupted run does not causally reproduce the correct answer. The neurons are necessary but not sufficient; they are embedded in a broader computational flow that the small circuit cannot replicate in isolation.

**Interpretation:** The neuron level for IOI is paradoxical: it produces a highly compressed, faithful circuit, yet the circuit has zero causal power when used for intervention. This suggests the 27-neuron description captures correlational structure rather than the core computational mechanism.

---

#### Level 2 Attention heads (activation patching)

The model has 12 layers × 12 heads = 144 heads. The greedy circuit selects just 2 heads:

| Head   | Recovery score | Role (known from literature)       |
|--------|----------------|------------------------------------|
| L9H9   | 0.832          | Name-mover head (Wang et al. 2022) |
| L9H6   | 0.448          | Backup name-mover head             |

- **Faithfulness (C1):** 1.272 the two-head circuit *overshoots* the logit gap by 27 %. This occurs because the heads also suppress competing tokens; their combined contribution exceeds the net gap. The circuit is over-complete rather than under-complete.
- **Compressibility (C2):** 0.014 only 2 of 144 heads; extremely compact (1.4 %).
- **Interventional precision (C3):** 0.155 patching L9H9 + L9H6 into a corrupted run correctly steers the model on only 15.5 % of examples, well above the 2.5 % chance level but far from perfect. The two heads alone are necessary but not sufficient to reproduce the full circuit behavior.

**Interpretation:** The head level gives a compact, intuitive description (the name-mover heads are the core of IOI), but the circuit is incomplete: additional heads (e.g. induction heads, S-inhibition heads) are needed for full causal coverage.

---

#### Level 3 SAE features (GPT-2 layer 3, 12,288 features)

The SAE reconstructs the residual stream at layer 3 with 99.93 % variance explained, activating ~71 features per example on average. The greedy circuit selects 96 features:

- **Circuit size:** 96 features out of 12,288 (0.78 % of the dictionary).
- **Max single-feature RS:** 0.030 even weaker individual contributions than in grokking.
- **Faithfulness (C1):** 0.901 the 96-feature circuit recovers 90 % of the logit gap.
- **Compressibility (C2):** 0.0078 highly compressed at 0.78 % of the dictionary.
- **Interventional precision (C3):** 0.000 patching the 96 circuit features produces zero causal steering. The random baseline for this task is also 0.000 (mean precision across 20 trials of 96 random features). Verdict: **artifact**. The circuit precision is not above the random baseline, meaning the greedy selection is picking features whose patching reconstructs the residual-stream bulk rather than the task-relevant computation. This is a known failure mode of SAE-based circuit extraction when the corrupted-run residual stream is close to the clean-run baseline (gap = 0.61 LD, versus 3.58 for L1).

**Interpretation:** The L3 circuit for IOI is not causally valid. The SAE at layer 3 does not sit at the right point in the computational graph to capture the name-mover mechanism, which is concentrated in later layers (layer 9). The result is a cautionary finding: high faithfulness scores alone do not imply a circuit is mechanistically meaningful. (we could try redo the experiment for layer 9)

---

### 5.3 Cross-level summary

| Task     | Level | Faithfulness (C1) | Compressibility (C2, lower = more compact) | Interventional Precision (C3) |
|----------|-------|-------------------|--------------------------------------------|-------------------------------|
| Grokking | L1    | **1.000**         | 1.000 (no compression)                     | 0.995                         |
| Grokking | L2    | 0.854             | 0.667                                      | 0.655                         |
| Grokking | L3    | 0.901             | **0.077**                                  | **1.000**                     |
| IOI      | L1    | 0.911             | **0.00073**                                | 0.000                         |
| IOI      | L2    | **1.272**         | 0.014                                      | **0.155**                     |
| IOI      | L3    | 0.901             | 0.0078                                     | 0.000 (artifact)              |

**Key takeaways supporting explanatory pluralism:**

1. **No level dominates on all three criteria.** On grokking, L1 wins on faithfulness and precision, L3 wins on compression + precision, and L2 gives the most human-interpretable story. On IOI, L2 wins on faithfulness and precision, L1 wins on compression.

2. **Faithfulness does not imply causal validity.** The IOI L1 and L3 circuits both achieve ~90 % faithfulness yet have zero interventional precision. A high recovery score is necessary but not sufficient for a mechanistically genuine circuit.

3. **Task structure matters.** Grokking's single-layer, Fourier-algorithmic structure makes all three levels causally coherent. IOI's multi-layer, multi-head computation concentrates the causal mechanism in a small set of late-layer attention heads (L2), making L2 the privileged level for intervention.

4. **SAE circuits require careful placement.** L3 works well for grokking (where the SAE covers the entire relevant computation) but fails for IOI (where the SAE sits at a layer upstream of the key mechanism). The right level of abstraction depends on where in the model the task-relevant computation occurs. (again, might need to try on later layer)