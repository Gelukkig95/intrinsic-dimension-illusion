"""
Clean, low-VRAM (<2.2 GB) verification script for:
  1. Direct estimator-free radius ratio (r_ctx / R_lex) & doubleton p_2 on HC3 (Section 6)
  2. ModernBERT-base & RoBERTa-base vs. Qwen3-1.7B HC3 2x2 factorial detection (Table 3a)
  3. Controlled anti-repetition prompt evasion on Qwen2.5-0.5B-Instruct (Section 6 & Table 4 Row 7)
  4. Exact hook-based static-embedding synonym mask M5 (Layer-0 cos >= 0.45) + Null S^20 control (Table 4 Row 3)
"""
import gc
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, ttest_ind
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from run_modern_scaleup import load_arxiv_pool, load_hc3_chatgpt_pool, load_stories_pool
from verify_idea1_all_4 import build_masks, cohens_d_ind, estimate_twonn_mle, tokenize_fixed_length_with_metadata

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_L = 150


class _StopAtMid(Exception):
    pass


def free_vram() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def auroc_hum_gt_ai(h: np.ndarray, a: np.ndarray) -> float:
    h, a = h[np.isfinite(h)], a[np.isfinite(a)]
    ranks = rankdata(np.concatenate([h, a]))
    n1, n0 = len(h), len(a)
    return float((np.sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0) * 100.0)


def match_caliper(ttr_h: np.ndarray, ttr_a: np.ndarray, cal: float = 0.015):
    used = np.zeros(len(ttr_a), dtype=bool)
    pairs = []
    for i in np.argsort(ttr_h):
        d = np.abs(ttr_a - ttr_h[i])
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= cal:
            used[j] = True
            pairs.append((int(i), int(j)))
    return pairs


def extract_exact_mid_states(mid: str, texts: list[str], tok):
    dt = torch.float16 if DEVICE == "cuda" else torch.float32
    try:
        mod = AutoModel.from_pretrained(mid, dtype=dt, low_cpu_mem_usage=True)
    except TypeError:
        mod = AutoModel.from_pretrained(mid, torch_dtype=dt, low_cpu_mem_usage=True)
    mod.eval()
    mid_l = mod.config.num_hidden_layers // 2

    # Attach forward hook strictly before any final LayerNorm / RMSNorm
    cap = {}
    hook_handle = None
    if hasattr(mod, "layers"):
        mod.layers = mod.layers[:mid_l]
        hook_handle = mod.layers[mid_l - 1].register_forward_hook(
            lambda m, i, o: (cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().double().cpu().numpy()), (_ for _ in ()).throw(_StopAtMid()))
        )
    elif hasattr(mod, "h"):
        mod.h = mod.h[:mid_l]
        hook_handle = mod.h[mid_l - 1].register_forward_hook(
            lambda m, i, o: (cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().double().cpu().numpy()), (_ for _ in ()).throw(_StopAtMid()))
        )
    mod.to(DEVICE)

    emb_w = mod.get_input_embeddings().weight.detach().float().cpu()
    items = []
    for t in texts:
        mt = tokenize_fixed_length_with_metadata(tok, t, SEQ_L)
        if mt is None:
            continue
        cap.clear()
        with torch.inference_mode():
            if hook_handle is not None:
                try:
                    mod(mt["full_ids"].to(DEVICE))
                except _StopAtMid:
                    pass
                h = cap["h"][0, mt["keep_positions"]]
            else:
                out = mod(mt["full_ids"].to(DEVICE), output_hidden_states=True)
                h = out.hidden_states[mid_l][0, mt["keep_positions"]].double().cpu().numpy()
        items.append((mt, h))

    if hook_handle is not None:
        hook_handle.remove()
    del mod
    free_vram()
    return items, emb_w


def run_all_audits() -> None:
    adj = np.abs(np.arange(SEQ_L)[:, None] - np.arange(SEQ_L)[None, :]) <= 1
    hc3_hum, hc3_ai = load_hc3_chatgpt_pool(target_pairs=80)

    print("[1/3] HC3 Encoder/Decoder Split + Estimator-Free r_ctx/R_lex Audit")
    for mid in ("roberta-base", "answerdotai/ModernBERT-base", "Qwen/Qwen3-1.7B"):
        tok = AutoTokenizer.from_pretrained(mid)
        h_items, _ = extract_exact_mid_states(mid, hc3_hum, tok)
        a_items, _ = extract_exact_mid_states(mid, hc3_ai, tok)

        def summarize(items):
            m0, ad, m4, ttr, rc_rl = [], [], [], [], []
            for mt, h in items:
                mk4 = build_masks(mt)["M4_stem_adj"]
                same_na = mk4 & (~adj)
                np.fill_diagonal(same_na, False)
                reach = ((np.eye(SEQ_L, dtype=bool) | same_na).astype(np.int32) @ (np.eye(SEQ_L, dtype=bool) | same_na).astype(np.int32)) > 0
                ttr.append(float(np.sum(1.0 / np.maximum(np.sum(reach, axis=1), 1)) / SEQ_L))
                dm = cdist(h, h); np.fill_diagonal(dm, np.inf); dm[dm <= 1e-6] = np.inf
                m0.append(estimate_twonn_mle(dm)[0])
                dma = dm.copy(); dma[adj] = np.inf
                ad.append(estimate_twonn_mle(dma)[0])
                dm4 = dm.copy(); dm4[mk4] = np.inf
                m4.append(estimate_twonn_mle(dm4)[0])
                rc_rl.append(float(np.median(dm[same_na]) / np.median(dm[~mk4])) if np.any(same_na) else np.nan)
            return np.array(m0), np.array(ad), np.array(m4), np.array(ttr), np.array(rc_rl)

        hm0, had, hm4, httr, hrr = summarize(h_items)
        am0, aad, am4, attr, arr = summarize(a_items)
        mp = match_caliper(httr, attr)
        ih, ia = [p[0] for p in mp], [p[1] for p in mp]
        print(
            f"  * {mid:28s} | Matched M0={auroc_hum_gt_ai(hm0[ih], am0[ia]):5.2f}% "
            f"Adj={auroc_hum_gt_ai(had[ih], aad[ia]):5.2f}% M4={auroc_hum_gt_ai(hm4[ih], am4[ia]):5.2f}% "
            f"| Pure r_ctx/R_lex AUC={auroc_hum_gt_ai(hrr[ih], arr[ia]):5.1f}%"
        )

    print("[2/3] Controlled Anti-Repetition Evasion Prompt (Qwen2.5-0.5B-Instruct -> Qwen3-1.7B)")
    gen_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tok_g = AutoTokenizer.from_pretrained(gen_id)
    if tok_g.pad_token is None:
        tok_g.pad_token = tok_g.eos_token
    tok_g.padding_side = "left"
    dt = torch.float16 if DEVICE == "cuda" else torch.float32
    try:
        gen_m = AutoModelForCausalLM.from_pretrained(gen_id, dtype=dt, low_cpu_mem_usage=True).to(DEVICE).eval()
    except TypeError:
        gen_m = AutoModelForCausalLM.from_pretrained(gen_id, torch_dtype=dt, low_cpu_mem_usage=True).to(DEVICE).eval()

    seeds = [" ".join(t.strip().split()[:22]) for t in hc3_ai[:36]]
    outs = {}
    for key, inst in (
        ("std", "Provide a clear, detailed 220-word explanation expanding on the topic below."),
        ("evade", "Write an informal, vivid 220-word human forum response on the topic below. Strictly avoid repeating the same nouns, verbs, or adjectives; substitute diverse synonyms throughout."),
    ):
        outs[key] = []
        prompts = [tok_g.apply_chat_template([{"role": "system", "content": inst}, {"role": "user", "content": f"Topic opening: {s}"}], tokenize=False, add_generation_prompt=True) for s in seeds]
        for b in range(0, len(prompts), 4):
            enc = tok_g(prompts[b : b + 4], return_tensors="pt", padding=True).to(DEVICE)
            with torch.inference_mode():
                g = gen_m.generate(**enc, max_new_tokens=225, min_new_tokens=185, do_sample=True, temperature=0.85, top_p=0.92, pad_token_id=tok_g.pad_token_id)
            outs[key].extend(tok_g.batch_decode(g[:, enc["input_ids"].shape[1] :].cpu(), skip_special_tokens=True))
            del enc, g
    del gen_m
    free_vram()

    tok_q = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    for key in ("std", "evade"):
        h_it, _ = extract_exact_mid_states("Qwen/Qwen3-1.7B", hc3_hum[:36], tok_q)
        g_it, _ = extract_exact_mid_states("Qwen/Qwen3-1.7B", outs[key], tok_q)
        h_adj = [estimate_twonn_mle(np.where(adj | np.eye(SEQ_L, dtype=bool), np.inf, cdist(h, h)))[0] for _, h in h_it]
        g_adj = [estimate_twonn_mle(np.where(adj | np.eye(SEQ_L, dtype=bool), np.inf, cdist(h, h)))[0] for _, h in g_it]
        print(f"  * Mode={key:5s} | Adj-Only AUROC = {auroc_hum_gt_ai(np.array(h_adj), np.array(g_adj)):5.1f}%")

    print("[3/3] Static-Embedding Synonym Mask M5 (Layer-0 Cosine >= 0.45) + Null S^20 Control")
    stories_150, _ = load_stories_pool(min_words=175, target_n=150, long_min_words=265)
    arxiv_150, _ = load_arxiv_pool(target_n=150)
    tok_r = AutoTokenizer.from_pretrained("roberta-base")
    sci_it, emb_w = extract_exact_mid_states("roberta-base", arxiv_150[:65], tok_r)
    sty_it, _ = extract_exact_mid_states("roberta-base", stories_150[:65], tok_r)
    emb_c = emb_w - emb_w.mean(dim=0, keepdim=True)
    emb_n = emb_c / torch.clamp(emb_c.norm(dim=1, keepdim=True), min=1e-8)

    rng = np.random.RandomState(2026)
    res_m4, res_m5, null_m5, hits = {0: [], 1: []}, {0: [], 1: []}, {0: [], 1: []}, []
    for flag, grp in ((0, sci_it), (1, sty_it)):
        for mt, h in grp:
            kp = mt["keep_positions"]
            tids = mt["full_ids"][0, kp].cpu()
            tstrs = [tok_r.decode([int(x)]).strip().lower() for x in tids.tolist()]
            is_c = np.array([len(s) >= 4 and s.isalpha() for s in tstrs], dtype=bool)
            cos_m = (emb_n[tids] @ emb_n[tids].T).numpy()
            m4 = build_masks(mt)["M4_stem_adj"]
            syn = (~m4) & (cos_m >= 0.45) & (is_c[:, None] & is_c[None, :])
            m5 = m4 | syn
            dm = cdist(h, h); np.fill_diagonal(dm, np.inf)
            dm4 = dm.copy(); dm4[m4] = np.inf
            nn1 = np.argmin(dm4, axis=1)
            hits.append(float(np.mean([syn[r, nn1[r]] for r in range(SEQ_L)])))
            dm5 = dm.copy(); dm5[m5] = np.inf
            res_m4[flag].append(estimate_twonn_mle(dm4)[2])
            res_m5[flag].append(estimate_twonn_mle(dm5)[2])
            z = rng.randn(SEQ_L, 21); z /= np.linalg.norm(z, axis=1, keepdims=True)
            dn = cdist(z, z); dn[m5] = np.inf
            null_m5[flag].append(estimate_twonn_mle(dn)[2])
    print(
        f"  * RoBERTa M5 | Synonym 1-NN={np.mean(hits)*100:.2f}% | d(M4)={cohens_d_ind(np.array(res_m4[1]), np.array(res_m4[0])):+.2f} "
        f"-> d(M5)={cohens_d_ind(np.array(res_m5[1]), np.array(res_m5[0])):+.2f} (Null p={ttest_ind(null_m5[1], null_m5[0])[1]:.2f})"
    )


if __name__ == "__main__":
    t0 = time.time()
    run_all_audits()
    print(f"[DONE in {time.time() - t0:.1f}s]")
