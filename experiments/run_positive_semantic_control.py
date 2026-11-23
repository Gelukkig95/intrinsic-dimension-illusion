import gc
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr, ttest_ind, ttest_rel
from transformers import AutoModel, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from verify_idea1_all_4 import (
    build_masks,
    cohens_d_ind,
    estimate_phd_mst,
    estimate_twonn_mle,
    tokenize_fixed_length_with_metadata,
)
from run_modern_scaleup import (
    hf_cache_folder_for,
    load_arxiv_pool,
    load_stories_pool,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_stem_ttr(mt: dict, masks: dict) -> float:
    for k in ("stems", "tok_stems", "word_stems", "norm_stems"):
        if k in mt and len(mt[k]) > 0:
            vals = list(mt[k])
            return float(len(set(vals))) / float(len(vals))
    # Fallback: connected components of non-adjacent stem graph
    stem_graph = masks["M4_stem_adj"] & (~masks["adj_only"])
    n_comp, _ = connected_components(stem_graph.astype(np.int8), directed=False)
    return float(n_comp) / float(stem_graph.shape[0])


def greedy_caliper_match_1to1(
    ttr_low_k: np.ndarray,
    ttr_high_k: np.ndarray,
    vals_low_k: np.ndarray,
    vals_high_k: np.ndarray,
    caliper: float = 0.015,
) -> dict:
    used_high = np.zeros(len(ttr_high_k), dtype=bool)
    order_low = np.argsort(ttr_low_k)
    pairs_l, pairs_h = [], []
    for i in order_low:
        diffs = np.abs(ttr_high_k - ttr_low_k[i])
        diffs[used_high] = np.inf
        j = int(np.argmin(diffs))
        if diffs[j] <= caliper:
            used_high[j] = True
            pairs_l.append(vals_low_k[i])
            pairs_h.append(vals_high_k[j])
    if len(pairs_l) < 8:
        return {"n_pairs": len(pairs_l), "mean_k1": float("nan"), "mean_k5": float("nan"), "d": float("nan"), "p_paired": float("nan")}
    arr_l = np.asarray(pairs_l, dtype=np.float64)
    arr_h = np.asarray(pairs_h, dtype=np.float64)
    d_pool = cohens_d_ind(arr_h, arr_l)  # >0 means multi-topic (K>1) has higher ID than single-topic (K=1)
    _, p_rel = ttest_rel(arr_h, arr_l)
    return {
        "n_pairs": int(len(arr_l)),
        "mean_k1": float(np.mean(arr_l)),
        "mean_k_multi": float(np.mean(arr_h)),
        "d": float(d_pool),
        "p_paired": float(p_rel),
    }


def split_clean_sentences(doc: str) -> list[str]:
    raw_sents = re.split(r"(?<=[.!?])\s+", " ".join(doc.strip().split()))
    good = [s.strip() for s in raw_sents if 7 <= len(s.split()) <= 48]
    if not good:
        words = doc.strip().split()
        good = [" ".join(words[i : i + 25]) for i in range(0, len(words), 25) if len(words[i : i + 25]) >= 10]
    return good


def build_k_topic_pool(docs: list[str], k_topics: int, n_samples: int = 90) -> list[str]:
    """
    Constructs passages of ~210 words where the first ~130 words (~155-165 tokens)
    contain equal-sized complete-sentence chunks from k_topics unrelated documents,
    ensuring all K topics sit inside the exact L=150 token window.
    """
    if k_topics == 1:
        return docs[:n_samples]

    doc_sents = [split_clean_sentences(d) for d in docs]
    n_docs = len(doc_sents)
    target_words_per_topic = max(22, int(round(126.0 / k_topics)))
    strides = [0, 19, 41, 67, 97, 113]
    out_passages = []

    for idx in range(n_samples):
        chunks = []
        tail_buffer = []
        for t_i in range(k_topics):
            d_idx = (idx + strides[t_i]) % n_docs
            sents = doc_sents[d_idx]
            picked, w_cnt = [], 0
            for s in sents:
                picked.append(s)
                w_cnt += len(s.split())
                if w_cnt >= target_words_per_topic:
                    break
            # Trim overly long single chunk softly at word level only if > target + 14
            chunk_txt = " ".join(picked)
            c_words = chunk_txt.split()
            if len(c_words) > target_words_per_topic + 12:
                chunk_txt = " ".join(c_words[: target_words_per_topic + 8]) + "."
            chunks.append(chunk_txt)
            if len(sents) > len(picked):
                tail_buffer.append(sents[len(picked)])
            else:
                tail_buffer.append(sents[0])

        # Append extra tail buffer beyond word ~130 so total length always exceeds 150 BPE tokens
        full_mosaic = " ".join(chunks + tail_buffer)
        if len(full_mosaic.split()) < 195:
            extra_doc = doc_sents[(idx + 131) % n_docs]
            full_mosaic = full_mosaic + " " + " ".join(extra_doc)
        out_passages.append(full_mosaic)

    return out_passages


def run_part_a_planted_d_lex_calibration() -> dict:
    print("=" * 98)
    print(" [PART A] Ground-Truth Semantic Manifold Calibration: Sweeping True d_lex in {6,10,14,18,22}")
    print("          (L=150, d_ctx=8, R_lex/r_ctx=12x, Empirical Zipfian Multiplicities + Local Pos Drift)")
    print("=" * 98)
    d_lex_grid = [6, 10, 14, 18, 22]
    d_ctx, seq_l, scale_r = 8, 150, 12.0

    # Empirical Zipfian stem pattern (V=97 distinct stems in L=150)
    zipf_types = (
        list(range(62))
        + [62 + i for i in range(22) for _ in range(2)]
        + [84 + i for i in range(8) for _ in range(3)]
        + [92 + i for i in range(5) for _ in range(4)]
    )
    tok_types = np.asarray(zipf_types[:seq_l], dtype=np.int64)
    v_distinct = len(set(tok_types.tolist()))
    idx = np.arange(seq_l)
    m4_sim = (tok_types[:, None] == tok_types[None, :]) | (np.abs(idx[:, None] - idx[None, :]) <= 1)

    rng = np.random.RandomState(2026)
    res_a = {}
    raw_means, m4_means, oracle_means = [], [], []

    for d_lex in d_lex_grid:
        tw_raw, tw_m4, tw_oracle, ml_raw, ml_m4, ml_oracle = [], [], [], [], [], []
        for _ in range(45):
            u_raw = rng.randn(v_distinct, d_lex + 1)
            u_sph = (u_raw / np.linalg.norm(u_raw, axis=1, keepdims=True)) * scale_r

            # Oracle pure cross-lexical manifold (V=97 distinct prototypes, zero repetition/context confound)
            dm_orc = cdist(u_sph, u_sph)
            np.fill_diagonal(dm_orc, np.inf)
            t_o, _, m_o = estimate_twonn_mle(dm_orc)
            tw_oracle.append(t_o)
            ml_oracle.append(m_o)

            # Contextualized sequence with Zipfian twins + smooth AR(1) local positional correlation
            eps_ind = rng.randn(seq_l, d_ctx + 1)
            eps_ar = eps_ind.copy()
            for t in range(1, seq_l):
                eps_ar[t] = 0.65 * eps_ar[t - 1] + math.sqrt(1.0 - 0.65**2) * eps_ind[t]
            e_sph = eps_ar / np.linalg.norm(eps_ar, axis=1, keepdims=True)

            h = np.concatenate([u_sph[tok_types], e_sph], axis=1)
            dm = cdist(h, h)
            np.fill_diagonal(dm, np.inf)
            t_r, _, m_r = estimate_twonn_mle(dm)
            tw_raw.append(t_r)
            ml_raw.append(m_r)

            dm_m = dm.copy()
            dm_m[m4_sim] = np.inf
            t_m, _, m_m = estimate_twonn_mle(dm_m)
            tw_m4.append(t_m)
            ml_m4.append(m_m)

        raw_means.append(float(np.mean(ml_raw)))
        m4_means.append(float(np.mean(ml_m4)))
        oracle_means.append(float(np.mean(ml_oracle)))
        res_a[str(d_lex)] = {
            "twonn_raw": float(np.mean(tw_raw)),
            "twonn_m4": float(np.mean(tw_m4)),
            "twonn_oracle": float(np.mean(tw_oracle)),
            "mle5_raw": float(np.mean(ml_raw)),
            "mle5_m4": float(np.mean(ml_m4)),
            "mle5_oracle": float(np.mean(ml_oracle)),
        }
        recovery_pct = (np.mean(ml_m4) / np.mean(ml_oracle)) * 100.0
        print(
            f" * True d_lex={d_lex:2d} | Unmasked MLE-5={np.mean(ml_raw):5.2f} (TwoNN={np.mean(tw_raw):5.2f}) | "
            f"Masked M4 MLE-5={np.mean(ml_m4):5.2f} (TwoNN={np.mean(tw_m4):5.2f}) | "
            f"Pure-Prototype Oracle={np.mean(ml_oracle):5.2f} ({recovery_pct:5.1f}% of finite-V oracle)"
        )

    r_m4, _ = pearsonr(d_lex_grid, m4_means)
    slope_raw = (raw_means[-1] - raw_means[0]) / (d_lex_grid[-1] - d_lex_grid[0])
    slope_m4 = (m4_means[-1] - m4_means[0]) / (d_lex_grid[-1] - d_lex_grid[0])
    print(f" -> Sensitivity Slope d(MLE-5)/d(d_lex): Unmasked = {slope_raw:.3f} (blinded!) vs. Masked M4 = {slope_m4:.3f} (Pearson r={r_m4:.4f})")
    res_a["summary"] = {"pearson_r_m4": float(r_m4), "slope_raw": float(slope_raw), "slope_m4": float(slope_m4)}
    return res_a


def load_model_fp16_or_fp32(mid: str):
    use_dt = torch.float16 if DEVICE == "cuda" else torch.float32
    try:
        return AutoModel.from_pretrained(mid, dtype=use_dt).to(DEVICE).eval()
    except TypeError:
        return AutoModel.from_pretrained(mid, torch_dtype=use_dt).to(DEVICE).eval()


def run_part_b_real_llm_topic_sweep() -> dict:
    print("\n" + "=" * 98)
    print(" [PART B] Real-LLM Positive Semantic Control: K=1 vs. K=3 vs. K=5 Independent Topics (L=150)")
    print("          (Evaluated Within-Genre on Both arXiv & Hippocorpus + 1:1 Stem-TTR Caliper Matching)")
    print("=" * 98)
    stories_150, _ = load_stories_pool(min_words=175, target_n=150, long_min_words=265)
    arxiv_150, _ = load_arxiv_pool(target_n=150)

    candidates = [
        ("roberta-base", "RoBERTa-base (Encoder)"),
        ("answerdotai/ModernBERT-base", "ModernBERT-base (Encoder)"),
        ("gpt2", "GPT-2 (Decoder)"),
        ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B (Decoder)"),
        ("Qwen/Qwen3-1.7B", "Qwen3-1.7B (Decoder)"),
    ]
    selected_models = [c for c in candidates if hf_cache_folder_for(c[0]).exists()]
    if len(selected_models) < 3:
        # Include lightweight backbones if cache folder naming differs
        for c in candidates[:4]:
            if c not in selected_models:
                selected_models.append(c)

    n_per_k = 80
    corpora = {
        "Sci_arXiv": {k: build_k_topic_pool(arxiv_150, k, n_samples=n_per_k) for k in (1, 3, 5)},
        "Story_Hippo": {k: build_k_topic_pool(stories_150, k, n_samples=n_per_k) for k in (1, 3, 5)},
    }

    res_b = {}
    for mid, disp in selected_models:
        tok = AutoTokenizer.from_pretrained(mid)
        model = load_model_fp16_or_fp32(mid)
        mid_l = model.config.num_hidden_layers // 2
        res_b[mid] = {}
        print(f"\n [{disp}] (Layer {mid_l}/{model.config.num_hidden_layers})")

        for dom_name, k_dict in corpora.items():
            stats_k = {}
            for k in (1, 3, 5):
                ttrs, tw_u, tw_m, ml_u, ml_m, phd_ns = [], [], [], [], [], []
                for txt in k_dict[k]:
                    mt = tokenize_fixed_length_with_metadata(tok, txt, target_len=150)
                    if mt is None:
                        continue
                    ids = mt["full_ids"].to(DEVICE)
                    kp = mt["keep_positions"]
                    mks = build_masks(mt)
                    ttrs.append(extract_stem_ttr(mt, mks))

                    with torch.inference_mode():
                        h = model(ids, output_hidden_states=True).hidden_states[mid_l][0, kp].double().cpu().numpy()

                    dm = cdist(h, h)
                    np.fill_diagonal(dm, np.inf)
                    dm[dm <= 1e-6] = np.inf
                    tu, _, mu = estimate_twonn_mle(dm)

                    dm_m4 = dm.copy()
                    dm_m4[mks["M4_stem_adj"]] = np.inf
                    tm, _, mm = estimate_twonn_mle(dm_m4)

                    tw_u.append(tu)
                    ml_u.append(mu)
                    tw_m.append(tm)
                    ml_m.append(mm)

                stats_k[k] = {
                    "ttr": np.asarray(ttrs),
                    "tw_u": np.asarray(tw_u),
                    "ml_u": np.asarray(ml_u),
                    "tw_m": np.asarray(tw_m),
                    "ml_m": np.asarray(ml_m),
                }

            d_full_13 = cohens_d_ind(stats_k[3]["ml_m"], stats_k[1]["ml_m"])
            d_full_15 = cohens_d_ind(stats_k[5]["ml_m"], stats_k[1]["ml_m"])
            _, p_full_15 = ttest_ind(stats_k[5]["ml_m"], stats_k[1]["ml_m"])

            mat_13 = greedy_caliper_match_1to1(stats_k[1]["ttr"], stats_k[3]["ttr"], stats_k[1]["ml_m"], stats_k[3]["ml_m"], caliper=0.015)
            mat_15 = greedy_caliper_match_1to1(stats_k[1]["ttr"], stats_k[5]["ttr"], stats_k[1]["ml_m"], stats_k[5]["ml_m"], caliper=0.015)
            mat_15_tw = greedy_caliper_match_1to1(stats_k[1]["ttr"], stats_k[5]["ttr"], stats_k[1]["tw_m"], stats_k[5]["tw_m"], caliper=0.015)

            # OLS partial slope of topic count K in {1,3,5} controlling linearly for Stem-TTR
            all_k = np.concatenate([np.full(len(stats_k[k]["ml_m"]), float(k)) for k in (1, 3, 5)])
            all_ttr = np.concatenate([stats_k[k]["ttr"] for k in (1, 3, 5)])
            all_ml_m = np.concatenate([stats_k[k]["ml_m"] for k in (1, 3, 5)])
            X = np.column_stack([np.ones_like(all_k), all_k, all_ttr])
            beta = np.linalg.lstsq(X, all_ml_m, rcond=None)[0]
            # Partial correlation between K and M4 MLE-5 after residualizing Stem-TTR
            res_k = all_k - np.polyval(np.polyfit(all_ttr, all_k, 1), all_ttr)
            res_y = all_ml_m - np.polyval(np.polyfit(all_ttr, all_ml_m, 1), all_ttr)
            r_part, p_part = pearsonr(res_k, res_y)

            print(
                f"  * {dom_name:11s} | M4 MLE-5 (K=1->3->5): {np.mean(stats_k[1]['ml_m']):5.2f} -> {np.mean(stats_k[3]['ml_m']):5.2f} -> {np.mean(stats_k[5]['ml_m']):5.2f} "
                f"(Full K5-K1 d={d_full_15:+.2f}, p={p_full_15:.4f})"
            )
            print(
                f"                | TTR-Matched K1 vs K3 ({mat_13['n_pairs']:2d}p): d={mat_13['d']:+.2f} (p={mat_13['p_paired']:.4f}) | "
                f"K1 vs K5 ({mat_15['n_pairs']:2d}p): MLE d={mat_15['d']:+.2f} (p={mat_15['p_paired']:.4f}), TwoNN d={mat_15_tw['d']:+.2f}"
            )
            print(
                f"                | TTR-Controlled Partial r(K, M4_MLE5 | Stem-TTR) = {r_part:+.3f} (p={p_part:.4e}, slope={beta[1]:+.3f}/topic)"
            )

            res_b[mid][dom_name] = {
                "k1_mle5_m4": float(np.mean(stats_k[1]["ml_m"])),
                "k3_mle5_m4": float(np.mean(stats_k[3]["ml_m"])),
                "k5_mle5_m4": float(np.mean(stats_k[5]["ml_m"])),
                "k1_twonn_m4": float(np.mean(stats_k[1]["tw_m"])),
                "k5_twonn_m4": float(np.mean(stats_k[5]["tw_m"])),
                "d_full_k5_k1": float(d_full_15),
                "p_full_k5_k1": float(p_full_15),
                "matched_k3_k1": mat_13,
                "matched_k5_k1_mle": mat_15,
                "matched_k5_k1_twonn": mat_15_tw,
                "partial_r_given_ttr": float(r_part),
                "partial_p_given_ttr": float(p_part),
            }

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return res_b


def main() -> None:
    t0 = time.time()
    part_a = run_part_a_planted_d_lex_calibration()
    part_b = run_part_b_real_llm_topic_sweep()

    out_path = THIS_DIR / "positive_semantic_control.json"
    out_path.write_text(json.dumps({"part_a_calibration": part_a, "part_b_topic_sweep": part_b}, indent=2), encoding="utf-8")
    print("\n" + "=" * 98)
    print(f" [DONE in {time.time() - t0:.1f}s] Saved exact positive-control readout -> {out_path.name}")
    print("=" * 98)


if __name__ == "__main__":
    main()
