[![DOI](https://zenodo.org/badge/1469347801.svg)](https://doi.org/10.5281/zenodo.24486039)

# The Intrinsic Dimension Illusion in Contextual Language Models

Single-passage intrinsic dimension (ID) estimates of LLM hidden states (`TwoNN`, `MLE`, `GRIDE`, `MST-PHD`) are widely reported in a narrow band (`ID ≈ 5–15`). We show this band—and several downstream conclusions built on it—is largely a measurement artifact caused by **Zipfian twin stems** and **adjacent sequence positions** ($`\lvert i - j \rvert \le 1`$) hijacking local neighborhoods.

---

## Core Mechanism & Exact Theory

### 1. Non-Adjacent Cross-Lexical Mask (M4)

Even beyond Layer 0 where every token vector is numerically distinct ($`d_{ij} > 11.4`$), same-stem tokens and adjacent positions hijack **77.1%–82.5%** of 1-NN pairs and **71.0%–73.8%** of Prim MST edges. Restricting distances via a representation-blind symbolic mask sets $`d_{ij}^{(\mathcal{M}_4)} = +\infty`$ whenever:

```math
\mathcal{M}_4(i, j) = \mathbb{I}\!\left[i = j \;\lor\; |i - j| \le 1 \;\lor\; w_i = w_j \;\lor\; u_i = u_j \;\lor\; s_i = s_j \;\lor\; \mathcal{M}_2(i, j) = 1\right]
```

where $`w_i`$ is the whitespace word-span index, $`u_i`$ is the normalized surface word, $`s_i`$ is the morphological stem, and $`\mathcal{M}_2`$ matches case/punctuation-normalized tokens.

### 2. Theorem 1 (Finite-Sample Beta–Digamma Doubleton Collapse)

Model token vectors as orthogonal sums of lexical prototypes and micro-contextual offsets on a product of hyperspheres:

```math
h_i = (u_{v(i)}, e_i) \in \left(R_{\text{lex}}\mathbb{S}^{d_{\text{lex}}}\right) \times \left(r_{\text{ctx}}\mathbb{S}^{d_{\text{ctx}}}\right), \qquad d_{\text{lex}} > 2,\; d_{\text{ctx}} \ge 2,\; r_{\text{ctx}} < R_{\text{lex}}
```

On $`R\mathbb{S}^d`$, squared normalized chordal distance follows $`B = \|x - y\|_2^2 / (4R^2) \sim \text{Beta}(d/2, d/2)`$. Defining the finite-sample log-spacing functional $`\Delta(d, m) := \frac{1}{2}\mathbb{E}[\log B_{(2:m)} - \log B_{(1:m)}]`$, the inverse Pareto TwoNN estimator $`\hat{d}_{\text{Pareto}}^{-1} = \frac{1}{L}\sum_{i=1}^L \log(r_{i,2}/r_{i,1})`$ satisfies:

```math
\mathbb{E}\!\left[\hat{d}_{\text{Pareto}}^{-1}\right] = p_1 \alpha_1 \Delta(d_{\text{lex}}, V-1) + \sum_{c \ge 3} p_c \Delta(d_{\text{ctx}}, c-1) + p_2 \Delta_{\text{jump}} + \mathcal{R}(V, R_{\text{lex}}, r_{\text{ctx}})
```

with $`\alpha_1 = \frac{V_1 - 1}{V - 1}`$, remainder $`\lvert\mathcal{R}\rvert = \mathcal{O}\!\left(V^{2/d_{\text{lex}}}\frac{r_{\text{ctx}}^2}{R_{\text{lex}}^2} + V\left(\frac{r_{\text{ctx}}}{R_{\text{lex}}}\right)^{d_{\text{lex}}}\log\frac{R_{\text{lex}}}{r_{\text{ctx}}}\right)`$, and closed-form doubleton ($`c_i = 2`$) cross-manifold jump:

```math
\Delta_{\text{jump}} = \log\!\left(\frac{R_{\text{lex}}}{r_{\text{ctx}}}\right) + \frac{1}{2}\mathbb{E}\!\left[\log B_{(1:V-1)}^{(d_{\text{lex}})}\right] - \frac{1}{2}\left[\psi\!\left(\frac{d_{\text{ctx}}}{2}\right) - \psi(d_{\text{ctx}})\right]
```

where $`\psi(z) = \frac{d}{dz}\log\Gamma(z)`$ is the digamma function.

**Implication:** Any stem occurring twice ($`p_2 > 0`$) anchors $`r_{i,1}`$ at its twin ($`r_{\text{ctx}}`$) while forcing $`r_{i,2}`$ to leap across the lexical manifold ($`R_{\text{lex}}`$). Because $`\mathbb{E}[\hat{d}_{\text{Pareto}}^{-1}]`$ grows at rate $`p_2 \log(R_{\text{lex}}/r_{\text{ctx}})`$, unmasked ID collapses strictly below $`\min(d_{\text{lex}}, d_{\text{ctx}})`$ (verified within **0.22%–2.22%** error against Monte Carlo), matching the abrupt Layer $`0 \to 1`$ ID plunge in RoPE transformers.

---

## Findings (12 Backbones, 124M–8.0B)

| Domain / Phenomenon | Prior Unmasked Observation ($`\mathcal{M}_0`$) | Finding After Short-Circuit Separation |
| :--- | :--- | :--- |
| **Internal-Layer ID Scale** | Confined to `ID ≈ 5–11` across hidden widths | $`\mathcal{M}_4`$ restores cross-lexical ID by **1.7x–4.1x** (`15.4–26.9`) with zero sparsification bias on isotropic null spheres ($`p \ge 0.27`$). |
| **Genre Complexity (Science vs. Story)** | Creative stories > scientific abstracts | $`\mathcal{M}_4`$ eliminates or reverses `Story > Science` on **10/12** backbones and every model $`\ge 1\text{B}`$ (Gemma-2-2B $`d=-0.65`$, SmolLM3-3B $`d=-2.03`$, Qwen2.5-7B $`d=-0.96`$, Llama-3-8B $`d=-1.68`$, $`p < 10^{-7}`$). Prior Gemma-Scope SAE claims trace to terminal pre-unembedding compression ($`l = 26`$) and repetitive boilerplate. |
| **Llama-3-8B Shuffling & TunedLens** | Shuffled prompts drop below intact text at $`l \approx 18\text{--}28`$; negative TunedLens entropy correlation ($`r \approx -0.65`$) | $`\mathcal{M}_4`$ eliminates late-layer shuffling crossovers (`-1.10 → +10.45` ID at $`l = 20`$, $`p < 10^{-12}`$) and flips TunedLens entropy correlations positive ($`r = -0.65 \to +0.21`$ at $`l = 20`$; $`+0.62`$ at $`l = 16`$). |
| **Zero-Shot AI Text Detection (HC3)** | Attributed to lower-dimensional AI semantic manifolds | Causal decoders actually exploit tighter intra-lexeme repetition radii ($`\hat{r}_{\text{ctx}} / \hat{R}_{\text{lex}}`$). Masking stems collapses decoder AUROC, while masking **adjacency alone** ($`\lvert i - j \rvert \le 1`$) boosts zero-shot AUROC up to **84.08%** (Qwen3-1.7B). |
| **Index-0 Attention Sinks (Double Dissociation)** | Ungated BOS-free decoders produce `51.7%–100%` MST-PHD NaNs | Ungated BOS-free decoders accumulate massive radial index-0 sinks (**37.9x–92.6x** norm). Gated full-attention hybrids (`Qwen3.5-0.8B`) cancel sinks via antiparallel residual updates ($`\cos \in [-0.98, -0.91]`$, `0.0%` NaNs) yet retain **80.8%** angular twin-token collapse. Excising index 0 restores $`\mathcal{M}_4`$ MST-PHD to `22.5–26.9`. |

---

## Verify Theorem 1 & Figures (< 10s on CPU)

```bash
pip install numpy scipy matplotlib
python reproduce_tables_and_theorem1.py
```

Even recent studies on LLM intrinsic dimension (e.g., Viswanathan et al., NeurIPS 2025; Pedashenko et al., EACL 2026; Yusupov et al., 2026) still rely on unmasked estimators without accounting for these short-circuit artifacts. Feel free to use or build upon these findings and code in your own papers as long as cite this GitHub repository or Zenodo DOI (10.5281/zenodo.24486039). Broader-scale validation is always welcome, though the collapse mechanism shown here is mathematically grounded and reproducible.
