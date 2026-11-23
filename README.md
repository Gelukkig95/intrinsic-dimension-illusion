# The Intrinsic Dimension Illusion in Contextual Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)](https://www.python.org/)

Code and reproducibility archive for **"The Intrinsic Dimension Illusion in Contextual Language Models: How Zipfian Twin Tokens and Positional Short-Circuits Distort Text Geometry"**.

## Key Findings

Applying homogeneous nearest-neighbor and persistent-homology intrinsic dimension (ID) estimators (`TwoNN`, `MLE`, `GRIDE`, `MST-PHD`) directly to single-passage LLM hidden states produces systematic measurement artifacts, even after exact duplicate-vector removal:

1. **Exact Beta–Digamma Doubleton Collapse (`Theorem 1`):** Inflected same-stem tokens and adjacent positions (`|i - j| <= 1`) hijack **77.1%–82.5%** of 1-NN pairs and **71.0%–73.8%** of Prim MST edges. Doubletons (stems occurring twice) anchor 1-NN inside tight intra-lexeme cones (`r_ctx`) while forcing 2-NN across the lexical manifold (`R_lex`), collapsing unmasked ID strictly below `min(d_lex, d_ctx)` (0.22%–2.22% error vs. Monte Carlo) and explaining the abrupt Layer `0 -> 1` plunge in RoPE transformers.
2. **Genre Complexity Reversal Across 12 Backbones (124M–8.0B):** Non-adjacent stem masking (`M4`) restores cross-lexical ID by **1.7x–4.1x** (to `15.4–26.9`) with zero sparsification bias on isotropic null spheres (`p >= 0.27`), eliminating or reversing the reported `Story > Science` gap on 10 of 12 backbones and every model `>= 1B` parameters, while tracing prior Gemma-Scope SAE claims to terminal pre-unembedding compression (`l = 26`).
3. **Resolving Llama-3-8B Shuffling & TunedLens Paradoxes:** On `Meta-Llama-3-8B`, `M4` removes late-layer token-shuffling crossovers (`-1.10 -> +10.45` ID at `l = 20`) and flips negative TunedLens entropy correlations positive (`r = -0.65 -> +0.21` at `l = 20`; `+0.62` at `l = 16`).
4. **AI-Text Detection Mechanism & Double Dissociation of Index-0 Sinks:** Causal-decoder AI detection exploits tighter intra-lexeme repetition radii (`r_ctx / R_lex`); stripping sequence adjacency alone (`|i - j| <= 1`) boosts zero-shot HC3 AUROC up to **84.08%**. Separately, ungated BOS-free decoders accumulate massive radial index-0 sinks (`37.9x–92.6x` norm) that censor Euclidean `MST-PHD`, whereas gated full-attention blocks (`Qwen3.5-0.8B`) cancel spikes via antiparallel updates (`cos in [-0.98, -0.91]`) while leaving angular twin-token collapse intact.

---

## 1. Quick CPU Verification (< 10 seconds, No GPU / Internet Required)

Regenerates `Theorem 1` (exact Beta–Digamma quadrature vs. Monte Carlo), planted `d_lex` calibration (`r = 0.998`), and Figures 1–3 into `reproduced_figures/`:

```bash
git clone https://github.com/gelukkig95/intrinsic-dimension-illusion.git
cd intrinsic-dimension-illusion
pip install numpy scipy matplotlib
python reproduce_tables_and_theorem1.py
```

---

## 2. Reusable Masking & Estimator API (`src/`)

Match the masking protocol to your measurement target:
- **Cross-lexical concept dimensionality:** `M4_stem_adj` + `drop_pos0=True` (on ungated BOS-free decoders).
- **Intra-lexeme repetition cohesion (e.g., zero-shot decoder AI detection):** `adj_only` (`|i - j| <= 1`) + `drop_pos0=True`.

```python
from src.masks import tokenize_fixed_length_with_metadata, build_mask_hierarchy
from src.estimators import masked_id_from_hidden_states

# Extract non-special token metadata for a passage of exact length L=150
meta = tokenize_fixed_length_with_metadata(tokenizer, text, target_len=150)
masks = build_mask_hierarchy(meta, window=1)

# H: [L, D] hidden states at your chosen layer (matching meta["keep_positions"])
res_m4 = masked_id_from_hidden_states(H, mask=masks["M4_stem_adj"], drop_pos0=True)
res_adj = masked_id_from_hidden_states(H, mask=masks["adj_only"], drop_pos0=True)
print(res_m4, res_adj)
```

---

## 3. Repository Layout

```text
intrinsic-dimension-illusion/
├── reproduce_tables_and_theorem1.py     # <10s standalone CPU verification & Figures 1-3
├── src/
│   ├── masks.py                         # Symbolic short-circuit mask hierarchy (M0..M4)
│   └── estimators.py                    # Masked TwoNN, MLE-k, GRIDE, and MST-PHD
├── experiments/                         # Full 12-backbone (124M-8.0B), SAE, TunedLens & audit scripts
│   ├── verify_idea1_all_4.py            # Mask hierarchy, MST-PHD, and core verifications
│   ├── run_modern_scaleup.py            # Multi-backbone extraction, TTR caliper matching, length sweep
│   ├── dissect_and_upgrade_10models.py  # Qwen3.5-0.8B 24-layer antiparallel sink cancellation & SmolLM3-3B
│   ├── audit_gemma2_full26_and_sae.py   # Gemma-2-2B (L13/L16/L26) + Gemma-Scope 16k SAE audit
│   ├── run_llama3_8b_tunedlens_audit.py # Zero-mmap Meta-Llama-3-8B streaming + official TunedLens
│   ├── run_critique_and_evasion_audits.py # Estimator-free r_ctx/R_lex, anti-repetition evasion, M5 synonyms
│   ├── final_killswitch_master_audit.py # Zero-LaTeX, pure-alpha subclouds, k/w sweeps, unit-sphere checks
│   ├── run_positive_semantic_control.py # Planted d_lex calibration & K-topic invariance
│   ├── refine_id_paper_v2.py            # Signed TTR controls & zero-plug-in Digamma-Beta verification
│   └── finish_7b_nommap.py              # Zero-mmap Qwen2.5-7B streaming & multilingual DE/KO audits
├── results/                             # Aggregate JSON summaries (zero third-party raw text)
└── paper/                               # Place your compiled preprint PDF here
```

All full-backbone experiments were executed without 4-bit/8-bit quantization on a single 6GB consumer GPU (`Python 3.11`, `PyTorch`). Raw third-party text corpora are not redistributed in this repository.

---

## Citation

```bibtex
@misc{gelukkig952026intrinsic,
  title  = {The Intrinsic Dimension Illusion in Contextual Language Models: How Zipfian Twin Tokens and Positional Short-Circuits Distort Text Geometry},
  author = {gelukkig95},
  year   = {2026},
  url    = {https://github.com/gelukkig95/intrinsic-dimension-illusion}
}
```
