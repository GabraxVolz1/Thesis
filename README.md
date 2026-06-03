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
**Levels**  | `L1` neurons / params · `L2` attention heads · `L3` SAE features 
**Metrics** | `faithfulness` (recovery score) · `compressibility` (description lenght, component count) · `intervention_precision` (predicted-vs-observed intervention effects) 


## 4. Repository structure

```
explanatory_pluralism_thesis/
│
├── data/                               # Pre-trained model checkpoints
│   ├── grokking_model.pt               # 1-layer transformer trained on modular addition (Nanda et al., 2023)
│   ├── grokking_sae_l1_2e-1.pt         # Grokking SAE variant (L1 penalty 2e-1)
│   ├── gpt2_sae_layer3.pt              # SAE trained on GPT-2 Small MLP layer 3 post-GELU activations (IOI task)
│   └── gpt2_sae_layer9.pt              # SAE trained on GPT-2 Small MLP layer 9 post-GELU activations (IOI task)
│
├── notebooks/                          # Numbered analysis notebooks (run in order)
│   ├── 00_setup_and_data.ipynb             # Downloads and caches GPT-2 Small; environment setup
│   ├── 01_tasks.ipynb                      # Defines and validates the two tasks: IOI and grokking
│   ├── 02_level1_neurons.ipynb             # L1 — Fourier frequency neuron clusters, greedy circuit (grokking)
│   ├── 02bis_level1_neurons(IOI).ipynb     # L1 — gradient attribution + greedy neuron circuit (IOI task)
│   ├── 03_level2_heads.ipynb               # L2 — attention-head activation patching, greedy circuit (grokking)
│   ├── 03bis_level2_heads(IOI).ipynb       # L2 — head-level activation patching, greedy circuit (IOI task)
│   ├── 04_level3_sae.ipynb                 # L3 — SAE on grokking MLP activations; SAE-feature circuits
│   ├── 04bis_level3_saeMLP3(IOI).ipynb     # L3 — SAE-feature circuits on GPT-2 Small MLP layer 3 (IOI task)
│   ├── 04bis_level3_saeMLP9(IOI).ipynb     # L3 — SAE-feature circuits on GPT-2 Small MLP layer 9 (IOI task)
│   ├── 05_evaluation_criteria.ipynb        # Criterion 3 (interventional precision) for all levels and tasks
│   └── 07_results_and_plots.ipynb          # Aggregates all results and produces thesis figures
│
├── results/                            # Auto-generated outputs (committed for reproducibility)
│   ├── level1/                         # L1 scores (JSON) + recovery and cluster plots (grokking)
│   ├── level1IOI/                      # L1 scores (JSON) + neuron recovery plot (IOI task)
│   ├── level2/                         # L2 scores (JSON) + head recovery plot (grokking)
│   ├── level2IOI/                      # L2 scores (JSON) + recovery plot (IOI task)
│   ├── level3/                         # L3 scores (JSON) + feature analysis and FFT plots (grokking)
│   ├── level3IOI/                      # L3 scores (JSON) + recovery and top-features plots (IOI task, both MLP3 and MLP9)
│   ├── criteria/                       # Three-criteria summary plot, Criterion 3 JSON, random-baseline plots
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

The circuit selection strategy is **greedy cumulative**: components are sorted by individual recovery score (RS), added one at a time to the circuit, and the loop stops as soon as the joint RS ≥ 0.90. The same strategy is used at every level and for both tasks.

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

- **Faithfulness (C1):** 1.000 — the neuron-level circuit perfectly accounts for model performance.
- **Compressibility (C2):** 1.000 — no compression; all 512/512 neurons are included. This is the least compressed description.
- **Interventional precision (C3):** 0.995 — patching all frequency clusters from source into a corrupted run steers the model to the correct answer on 99.5 % of examples.

**Interpretation:** The neuron level tells the complete causal story but at the cost of no compression — you are essentially describing the whole MLP.

---

#### Level 2: Attention heads (activation patching)

The grokking model has 4 attention heads at layer 0. Each head is mean-ablated at the readout (`=`) position; the greedy circuit adds heads one at a time until RS ≥ 0.90.

| Head   | Individual RS |
|--------|---------------|
| head_0 | 0.173         |
| head_2 | 0.169         |
| head_3 | 0.131         |
| head_1 | 0.126         |

All four heads must be included to reach the RS target. Individual contributions are small (~0.13–0.17) but highly complementary — their joint recovery score is 1.000.

- **Faithfulness (C1):** 1.000 — all four heads together perfectly recover the model's behavior.
- **Compressibility (C2):** 1.000 — the circuit requires all 4 of 4 heads; no compression at the head level.
- **Interventional precision (C3):** 1.000

**Interpretation:** The attention-head level confirms that the 1-layer grokking model distributes its computation evenly across all four heads, with each carrying a roughly equal share (~15–17 % RS individually). Unlike IOI, where two heads dominate, grokking shows no head specialization at this granularity. The head-level description provides no compression advantage over counting individual heads, but it confirms that the full attention sublayer is causally responsible for the computation.

---

#### Level 3: SAE features (Sparse Autoencoder, 2048 features)

The SAE reconstructs the MLP post-activations with 99.98 % variance explained, activating ~198 features per example on average. The greedy circuit selects features until 90 % of the logit gap is recovered:

- **Circuit size:** 158 features out of 2048 (7.7 % of all features).
- **Max single-feature RS:** 0.042 — no feature dominates alone; the circuit is highly distributed.
- **Faithfulness (C1):** 0.901 — the 158-feature circuit recovers 90 % of the logit gap, matching the target.
- **Compressibility (C2):** 0.077 — the circuit uses only 7.7 % of the dictionary; the SAE provides a substantially more compressed description than the raw neuron level.
- **Interventional precision (C3):** 1.000 — patching the 158 circuit features into a corrupted run yields perfect prediction. The random baseline for 158 randomly chosen features gives a mean precision of 0.0007 (essentially zero), confirming the circuit is not a reconstruction artifact.

**Interpretation:** The SAE-feature level achieves the best trade-off for grokking: near-perfect interventional precision at much higher compression than L1. However, it requires 158 components, making it harder to interpret intuitively than an ideal compact description.

---

### 5.2 IOI task: Indirect Object Identification, GPT-2 Small

The task is identifying the indirect object in sentences like *"When Mary and John went to the store, John gave a book to \_\_\_"* (answer: *Mary*). The metric is the logit difference between the indirect-object name and the subject name.

#### Level 1: MLP Neurons (gradient attribution circuit)

GPT-2 Small has 12 layers × 3,072 MLP neurons = 36,864 neurons total. Gradient-attribution-based greedy selection identifies a circuit of 27 neurons:

- **Faithfulness (C1):** 0.911 — 27 neurons recover 91 % of the logit gap.
- **Compressibility (C2):** 0.00073 — the circuit is extremely compact: 27 out of 36,864 neurons (0.073 %). The most compressed description across all levels and tasks.
- **Interventional precision (C3):** 0.000 — despite high faithfulness, patching only these 27 neurons into a corrupted run does not causally reproduce the correct answer. The neurons are necessary but not sufficient; they are embedded in a broader computational flow that the small circuit cannot replicate in isolation.

**Interpretation:** The neuron level for IOI is paradoxical: it produces a highly compressed, faithful circuit, yet the circuit has zero causal power when used for intervention. This suggests the 27-neuron description captures correlational structure rather than the core computational mechanism.

---

#### Level 2: Attention heads (activation patching)

The model has 12 layers × 12 heads = 144 heads. The greedy circuit selects just 2 heads:

| Head   | Individual RS | Role (known from literature)       |
|--------|---------------|------------------------------------|
| L9H9   | 0.832         | Name-mover head (Wang et al. 2022) |
| L9H6   | 0.448         | Backup name-mover head             |

- **Faithfulness (C1):** 1.272 — the two-head circuit *overshoots* the logit gap by 27 %. This occurs because the heads also suppress competing tokens; their combined contribution exceeds the net gap. The circuit is over-complete rather than under-complete.
- **Compressibility (C2):** 0.014 — only 2 of 144 heads; extremely compact (1.4 %).
- **Interventional precision (C3):** 0.155 — patching L9H9 + L9H6 from a source run correctly steers the model on 15.5 % of examples, well above the 2.5 % chance level but far from perfect.

**Interpretation:** The head level gives a compact, intuitive description (the name-mover heads are the core of IOI), but the circuit is incomplete: additional heads (e.g. induction heads, S-inhibition heads) are needed for full causal coverage.

---

#### Level 3a: SAE features on MLP layer 3 (12,288 features)

MLP layer 3 was selected because it has the highest individual neuron RS among all 12 layers (max ≈ 0.080 from the L1 analysis). The SAE reconstructs MLP3 post-GELU activations with 99.93 % variance explained, activating ~71 features per example.

- **Circuit size:** 96 features out of 12,288 (0.78 %).
- **Max single-feature RS:** 0.030.
- **Faithfulness (C1):** 0.901 — the 96-feature circuit recovers 90 % of the logit gap at MLP3.
- **Compressibility (C2):** 0.0078 — highly compressed at 0.78 % of the dictionary.
- **Interventional precision (C3):** 0.000 (artifact) — patching the 96 circuit features produces zero causal steering, matching the random baseline (0.000 mean over 20 trials). High faithfulness does not imply causal validity here: the SAE at layer 3 sits upstream of the key name-mover mechanism, which is concentrated in later layers.

---

#### Level 3b: SAE features on MLP layer 9 (12,288 features)

Layer 9 hosts the name-mover heads (L9H9, L9H6 identified at Level 2) and is therefore a more plausible location for the task-relevant MLP computation. The SAE reconstructs MLP9 post-GELU activations with 99.99 % variance explained, activating ~135 features per example.

- **Circuit size:** 338 features out of 12,288 (2.75 %).
- **Max single-feature RS:** 0.030 — individual contributions remain weak, consistent with MLP3.
- **Gap to recover:** 0.507 LD (smaller than MLP3's 0.610, reflecting a weaker MLP contribution at this layer relative to the attention mechanism).
- **Full-restore RS:** 0.968 — all SAE features together recover 96.8 % of the MLP9 contribution.
- **Faithfulness (C1):** 0.901 — the 338-feature circuit matches the RS target.
- **Compressibility (C2):** 0.027 — less compressed than MLP3 (2.75 % vs 0.78 %), reflecting more distributed MLP activity at layer 9.
- **Interventional precision (C3):** 0.000

**Interpretation:** Both MLP3 and MLP9 SAE circuits achieve 90 % faithfulness, but MLP9 requires 3.5× more features (338 vs 96) for a smaller gap, suggesting MLP9's contribution to IOI is more diffuse. The causal status of MLP9 features is the same as MLP3

