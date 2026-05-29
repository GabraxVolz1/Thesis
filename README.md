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