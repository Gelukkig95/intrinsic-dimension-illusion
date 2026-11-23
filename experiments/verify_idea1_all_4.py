import csv
import gc
import json
import math
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, ttest_ind

ROOT_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT_DIR / "data" / "idea1_4pillars_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = labels == 1
    neg = labels == 0
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_neg_check(n_n):
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def n_neg_check(n_n: int) -> bool:
    return n_n == 0


def cohens_d_ind(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = np.nanvar(a, ddof=1), np.nanvar(b, ddof=1)
    na, nb = np.sum(np.isfinite(a)), np.sum(np.isfinite(b))
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    return float((np.nanmean(a) - np.nanmean(b)) / max(pooled, 1e-12))


# Deterministic morphological stemmer & token-to-word span aligner
def simple_english_stem(word: str) -> str:
    """Deterministic suffix normalizer so 'walk/walks/walked/walking' & 'memory/memories' match cleanly."""
    w = word.lower()
    if len(w) <= 3 or not w.isalpha():
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "lsz":
            base = base[:-1]
        return base
    if w.endswith("ed") and len(w) > 4:
        base = w[:-2]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "lsz":
            base = base[:-1]
        return base
    for suf in ("ations", "ation", "ments", "ment", "ness", "ities", "ity", "ally", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[: -len(suf)]
    if w.endswith("es") and len(w) > 4 and w[-3] in "sxzho":
        return w[:-2]
    if w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def tokenize_fixed_length_with_metadata(tok, text: str, target_len: int = 160) -> dict | None:
    """Extracts exact `target_len` non-special tokens + exact word-occurrence, surface, and stem keys."""
    enc = tok(
        text,
        truncation=True,
        max_length=target_len + 16,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
    )
    ids_all = np.asarray(enc["input_ids"], dtype=np.int64)
    spec_mask = np.asarray(enc["special_tokens_mask"], dtype=np.int64)
    offsets_all = enc["offset_mapping"]

    keep_idx = np.where(spec_mask == 0)[0]
    if len(keep_idx) < target_len:
        return None
    keep_idx = keep_idx[:target_len]

    # Map full model input up to the last kept token so causal/bidirectional context is intact
    last_pos = int(keep_idx[-1]) + 1
    # For RoBERTa/encoders that require trailing EOS, include trailing special token if present
    if last_pos < len(ids_all) and spec_mask[last_pos] == 1:
        full_ids = ids_all[: last_pos + 1]
    else:
        full_ids = ids_all[:last_pos]

    # Pre-index whitespace-delimited word spans in raw text
    word_spans = []
    for w_idx, m in enumerate(re.finditer(r"\S+", text)):
        ws, we = m.span()
        raw_w = m.group(0)
        alnum = re.sub(r"[^a-zA-Z0-9]", "", raw_w).lower()
        surf = alnum if alnum else raw_w.strip()
        stem = simple_english_stem(alnum) if alnum.isalpha() else surf
        word_spans.append((ws, we, w_idx, surf, stem))

    tok_ids = ids_all[keep_idx]
    norm_tok_keys = []
    word_occ_ids = np.arange(target_len, dtype=np.int64)
    surf_keys = []
    stem_keys = []

    ws_ptr = 0
    for idx_out, orig_i in enumerate(keep_idx):
        cs, ce = offsets_all[int(orig_i)]
        dec = tok.decode([int(ids_all[int(orig_i)])])
        dec_alnum = re.sub(r"[^a-zA-Z0-9]", "", dec).lower()
        norm_tok_keys.append(dec_alnum if dec_alnum else dec.strip())

        matched = None
        if ce > cs:
            while ws_ptr < len(word_spans) and word_spans[ws_ptr][1] <= cs:
                ws_ptr += 1
            for k in range(ws_ptr, min(ws_ptr + 4, len(word_spans))):
                ws, we, w_id, sf, st = word_spans[k]
                if ws < ce and we > cs:
                    matched = (w_id, sf, st)
                    break
        if matched is not None:
            word_occ_ids[idx_out] = matched[0]
            surf_keys.append(matched[1])
            stem_keys.append(matched[2])
        else:
            word_occ_ids[idx_out] = 100000 + idx_out
            surf_keys.append(norm_tok_keys[-1])
            stem_keys.append(simple_english_stem(norm_tok_keys[-1]))

    return {
        "full_ids": torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
        "keep_positions": keep_idx,
        "tok_ids": tok_ids,
        "norm_tok": np.asarray(norm_tok_keys, dtype=object),
        "word_occ": word_occ_ids,
        "surf_word": np.asarray(surf_keys, dtype=object),
        "stem": np.asarray(stem_keys, dtype=object),
        "ttr_id": len(np.unique(tok_ids)) / float(target_len),
        "ttr_stem": len(set(stem_keys)) / float(target_len),
    }


# Intrinsic dimension estimators: twonn, mle-5, and exact mst-phd
def estimate_twonn_mle(dmat: np.ndarray) -> tuple[float, float, float]:
    """Returns (Facco-TwoNN, Pareto-TwoNN, Levina-Bickel MLE-5) on a (possibly masked) distance matrix."""
    sd = np.sort(dmat, axis=1)
    r1, r2 = sd[:, 0], sd[:, 1]
    ok = (r1 > 1e-6) & np.isfinite(r2) & (r2 > r1 * 1.000001)
    if ok.sum() < 15:
        return float("nan"), float("nan"), float("nan")

    mu = np.sort(r2[ok] / r1[ok])
    nv = len(mu)
    pareto_id = float(nv / np.sum(np.log(mu)))

    keep = max(10, int(0.90 * nv))
    cdf = np.arange(1, nv + 1, dtype=np.float64) / (nv + 1.0)
    x = np.log(mu[:keep])
    y = -np.log(1.0 - cdf[:keep])
    facco_id = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))

    ok5 = (r1 > 1e-6) & np.isfinite(sd[:, 4]) & (sd[:, 4] > r1 * 1.000001)
    if ok5.sum() >= 15:
        inv_m = np.mean(np.log(np.maximum(sd[ok5, 4:5] / sd[ok5, :4], 1.000001)), axis=1)
        mle5 = float(1.0 / max(float(np.mean(inv_m)), 1e-9))
    else:
        mle5 = float("nan")
    return facco_id, pareto_id, mle5


def prim_mst_edges_dense(sub_d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Exact O(m^2) vectorized Prim's MST handling np.inf forbidden edges cleanly."""
    m = sub_d.shape[0]
    in_tree = np.zeros(m, dtype=bool)
    in_tree[0] = True
    min_d = sub_d[0].copy()
    parent = np.zeros(m, dtype=np.int64)

    u_list = np.empty(m - 1, dtype=np.int64)
    v_list = np.empty(m - 1, dtype=np.int64)
    w_list = np.empty(m - 1, dtype=np.float64)

    for step in range(m - 1):
        cand = np.where(in_tree, np.inf, min_d)
        nxt = int(np.argmin(cand))
        w = float(cand[nxt])
        if not np.isfinite(w):
            return None
        u_list[step] = parent[nxt]
        v_list[step] = nxt
        w_list[step] = w
        in_tree[nxt] = True
        better = (~in_tree) & (sub_d[nxt] < min_d)
        min_d[better] = sub_d[nxt, better]
        parent[better] = nxt

    return u_list, v_list, w_list


def estimate_phd_mst(
    dmat: np.ndarray, alpha: float = 1.0, min_points: int = 45, n_steps: int = 7, n_reruns: int = 5, seed: int = 42
) -> float:
    """Exact Persistent Homology Dimension (Schweinhart 2021; Tulchinskii et al., NeurIPS 2023 GPTID)."""
    n_total = dmat.shape[0]
    if n_total < min_points + 15:
        return float("nan")
    rng = np.random.RandomState(seed)
    sizes = np.unique(np.linspace(min_points, n_total, n_steps, dtype=int))
    log_n, log_e = [], []

    for sz in sizes:
        vals = []
        draws = 1 if sz == n_total else n_reruns
        for _ in range(draws):
            idx = np.arange(n_total) if sz == n_total else np.sort(rng.choice(n_total, size=int(sz), replace=False))
            sub = dmat[np.ix_(idx, idx)]
            res = prim_mst_edges_dense(sub)
            if res is not None:
                vals.append(float(np.sum(res[2] ** alpha)))
        if vals:
            log_n.append(math.log(float(sz)))
            log_e.append(math.log(float(np.mean(vals))))

    if len(log_n) < 4:
        return float("nan")
    x = np.asarray(log_n)
    y = np.asarray(log_e)
    xm, ym = x - x.mean(), y - y.mean()
    kappa = float(np.dot(xm, ym) / max(np.dot(xm, xm), 1e-12))
    if kappa >= 0.97 or kappa <= -0.5:
        return float("nan")
    return float(alpha / (1.0 - kappa))


def make_symmetric_random_edge_mask(dmat_std: np.ndarray, sym_forbid: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Deletes the EXACT same number of undirected edges at random as sym_forbid removed (for MST-PHD control)."""
    n = dmat_std.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    valid_upper = np.isfinite(dmat_std[iu, ju])
    n_drop = int(np.sum(valid_upper & sym_forbid[iu, ju]))
    cand_idx = np.where(valid_upper)[0]
    out = dmat_std.copy()
    if 0 < n_drop < len(cand_idx) - n:
        chosen = rng.choice(cand_idx, size=n_drop, replace=False)
        r_drop, c_drop = iu[chosen], ju[chosen]
        out[r_drop, c_drop] = np.inf
        out[c_drop, r_drop] = np.inf
    return out


def make_row_matched_random_mask(dmat_std: np.ndarray, mask: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    out = dmat_std.copy()
    for i in range(out.shape[0]):
        fin = np.where(np.isfinite(dmat_std[i]))[0]
        k = int(np.sum(np.isfinite(dmat_std[i]) & mask[i]))
        if 0 < k < len(fin) - 5:
            drop = rng.choice(fin, size=k, replace=False)
            out[i, drop] = np.inf
    return out


# REAL DATA LOADERS (Hippocorpus / Gutenberg Stories & Live arXiv Abstracts)
def load_real_creative_stories(min_words: int = 175, max_count: int = 90) -> tuple[list[str], str]:
    for base in (Path("./data"), Path("./lang_geometry"), Path(".")):
        if not base.exists():
            continue
        found = list(base.rglob("hcV3-stories.csv"))
        if found:
            out = []
            with found[0].open("r", encoding="utf-8", errors="ignore", newline="") as f:
                for row in csv.DictReader(f):
                    s = re.sub(r"\s+", " ", (row.get("story") or "").strip())
                    if len(s.split()) >= min_words:
                        out.append(s)
                    if len(out) >= max_count:
                        break
            if len(out) >= 60:
                return out, f"Hippocorpus Real Personal/Creative Stories ({found[0].name})"

    g_path = CACHE_DIR / "gutenberg_pg1342.txt"
    if not g_path.exists():
        req = urllib.request.Request("https://www.gutenberg.org/cache/epub/1342/pg1342.txt", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            g_path.write_bytes(resp.read())
    words = g_path.read_text(encoding="utf-8", errors="ignore").split()
    out = [" ".join(words[i : i + 210]) for i in range(2500, 2500 + 210 * max_count, 210)]
    return out[:max_count], "Project Gutenberg Literary Fiction"


def load_real_arxiv_abstracts(min_words: int = 165, target_n: int = 60) -> list[str]:
    cache_json = CACHE_DIR / "real_arxiv_abstracts_60.json"
    if cache_json.exists():
        cached = json.loads(cache_json.read_text(encoding="utf-8"))
        if len(cached) >= target_n:
            return cached[:target_n]

    print(" * [Data] Fetching real peer-style scientific abstracts via public arXiv Atom API...")
    abstracts = []
    queries = [
        "cat:cs.CL+OR+cat:cs.LG",
        "cat:q-bio.NC+OR+cat:stat.ML",
        "cat:physics.bio-ph+OR+cat:cs.CV",
    ]
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for q in queries:
        if len(abstracts) >= target_n:
            break
        url = f"http://export.arxiv.org/api/query?search_query={q}&start=0&max_results=90"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Academic-Research-Audit)"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                xml_bytes = resp.read()
            root = ET.fromstring(xml_bytes)
            for entry in root.findall("atom:entry", ns):
                summ_el = entry.find("atom:summary", ns)
                if summ_el is not None and summ_el.text:
                    clean_abs = re.sub(r"\s+", " ", summ_el.text.strip())
                    # Remove inline LaTeX commands heavy with symbols so length is genuine prose
                    if len(clean_abs.split()) >= min_words and clean_abs not in abstracts:
                        abstracts.append(clean_abs)
                        if len(abstracts) >= target_n:
                            break
        except Exception as e:
            print(f"   [Warning] arXiv query {q} error: {e}")

    if len(abstracts) < 30:
        raise RuntimeError(f"Collected only {len(abstracts)} arXiv abstracts; check internet connection.")
    cache_json.write_text(json.dumps(abstracts, indent=2), encoding="utf-8")
    return abstracts[:target_n]


def load_hf_backbone(model_id: str, device: str):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    dt = torch.float32 if ("gpt2" in model_id.lower() or "roberta" in model_id.lower()) else torch.float16
    try:
        model = AutoModel.from_pretrained(model_id, dtype=dt).to(device).eval()
    except TypeError:
        model = AutoModel.from_pretrained(model_id, torch_dtype=dt).to(device).eval()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dt).to(device).eval()
    return tok, model


def build_masks(meta: dict) -> dict[str, np.ndarray]:
    n = len(meta["tok_ids"])
    pos = np.arange(n)
    adj1 = np.abs(pos[:, None] - pos[None, :]) <= 1
    same_id = meta["tok_ids"][:, None] == meta["tok_ids"][None, :]
    same_norm = meta["norm_tok"][:, None] == meta["norm_tok"][None, :]
    same_wocc = meta["word_occ"][:, None] == meta["word_occ"][None, :]
    same_surf = meta["surf_word"][:, None] == meta["surf_word"][None, :]
    same_stem = meta["stem"][:, None] == meta["stem"][None, :]

    return {
        "M1_id": same_id,
        "M2_norm": same_id | same_norm,
        "M3_surf_adj": same_id | same_norm | same_wocc | same_surf | adj1,
        "M4_stem_adj": same_id | same_norm | same_wocc | same_surf | same_stem | adj1,
        "adj_only": adj1,
    }


# Verification 1: subword, case, & morphological stem hierarchy
def run_verification_1(stories: list[str], device: str) -> None:
    print("\n" + "=" * 96)
    print(" [VERIFICATION 1] Subword, Case/Space, & Morphological Stem Leakage Hierarchy (L=160 Tokens)")
    print("=" * 96)

    for m_id in ("Qwen/Qwen2.5-0.5B-Instruct", "gpt2"):
        tok, model = load_hf_backbone(m_id, device)
        levels = [
            ("M0. Standard Deduped (Ruppik 2025)", None),
            ("M1. Exact Token-ID (w_i == w_j)", "M1_id"),
            ("M2. +Case/Space Normalized ('The'==' the')", "M2_norm"),
            ("M3. +Same Word-Span & Surface + |i-j|<=1", "M3_surf_adj"),
            ("M4. +Morphological Stem + |i-j|<=1 (Full)", "M4_stem_adj"),
            ("M4-Ctrl. Exact Row-Matched Random Mask", "RAND_CTRL"),
        ]
        res = {lbl: {"f": [], "m": [], "caught": []} for lbl, _ in levels}
        ttr_stems = []
        rng = np.random.RandomState(2026)

        used = 0
        for s in stories:
            meta = tokenize_fixed_length_with_metadata(tok, s, target_len=160)
            if meta is None:
                continue
            used += 1
            ttr_stems.append(meta["ttr_stem"])
            with torch.inference_mode():
                hs = model(input_ids=meta["full_ids"].to(device), output_hidden_states=True).hidden_states
                h = hs[len(hs) // 2][0, meta["keep_positions"]].float().cpu().numpy().astype(np.float64)

            dmat = np.linalg.norm(h[:, None, :] - h[None, :, :], axis=-1)
            np.fill_diagonal(dmat, np.inf)
            d_std = dmat.copy()
            d_std[d_std <= 1e-6] = np.inf
            nn1_raw = np.argmin(d_std, axis=1)

            masks = build_masks(meta)
            for lbl, key in levels:
                if key is None:
                    dm = d_std
                    caught = 0.0
                elif key == "RAND_CTRL":
                    dm = make_row_matched_random_mask(d_std, masks["M4_stem_adj"], rng)
                    caught = float(np.mean(masks["M4_stem_adj"][np.arange(160), nn1_raw]) * 100.0)
                else:
                    mk = masks[key]
                    dm = d_std.copy()
                    dm[mk] = np.inf
                    caught = float(np.mean(mk[np.arange(160), nn1_raw]) * 100.0)
                fv, _, mv = estimate_twonn_mle(dm)
                res[lbl]["f"].append(fv)
                res[lbl]["m"].append(mv)
                res[lbl]["caught"].append(caught)
            if used >= 75:
                break

        del model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n * Backbone: {m_id} (n={used} passages, exact L=160 tokens each)")
        for lbl, _ in levels:
            fa = np.asarray(res[lbl]["f"])
            ma = np.asarray(res[lbl]["m"])
            r_val, _ = pearsonr(ttr_stems, fa)
            print(
                f"   - {lbl:43s} | 1-NN Caught={np.mean(res[lbl]['caught']):5.1f}% | "
                f"TwoNN={np.nanmean(fa):5.2f}±{np.nanstd(fa):4.2f} | "
                f"MLE-5={np.nanmean(ma):5.2f} | Stem-TTR R^2={r_val**2*100:4.1f}%"
            )


# Verification 2: mst-phd estimator + all-layer encoder/decoder audit
def run_verification_2(stories: list[str], device: str) -> None:
    print("\n" + "=" * 96)
    print(" [VERIFICATION 2] MST-PHD Estimator Audit & Full Layer-Wise Decoder vs. Encoder Curves")
    print("=" * 96)

    backbones = [
        "Qwen/Qwen2.5-0.5B-Instruct",
        "gpt2",
        "FacebookAI/roberta-base",
    ]
    rng = np.random.RandomState(42)

    for m_id in backbones:
        tok, model = load_hf_backbone(m_id, device)
        # First run get layer count
        probe_meta = next(tokenize_fixed_length_with_metadata(tok, s, 160) for s in stories if tokenize_fixed_length_with_metadata(tok, s, 160) is not None)
        with torch.inference_mode():
            n_hs = len(model(input_ids=probe_meta["full_ids"].to(device), output_hidden_states=True).hidden_states)
        n_layers = n_hs - 1
        ckpt_layers = sorted({0, n_layers // 4, n_layers // 2, (3 * n_layers) // 4, n_layers})
        mid_l = n_layers // 2

        layer_twonn = {l: {"std": [], "msk": []} for l in ckpt_layers}
        phd_std_list, phd_rand_list, phd_msk_list = [], [], []
        mst_hijack_pcts = []
        ttrs = []

        used = 0
        for s in stories:
            meta = tokenize_fixed_length_with_metadata(tok, s, target_len=160)
            if meta is None:
                continue
            used += 1
            ttrs.append(meta["ttr_stem"])
            m4 = build_masks(meta)["M4_stem_adj"]

            with torch.inference_mode():
                hs = model(input_ids=meta["full_ids"].to(device), output_hidden_states=True).hidden_states

            for l in ckpt_layers:
                h = hs[l][0, meta["keep_positions"]].float().cpu().numpy().astype(np.float64)
                dmat = np.linalg.norm(h[:, None, :] - h[None, :, :], axis=-1)
                np.fill_diagonal(dmat, np.inf)
                d_std = dmat.copy()
                d_std[d_std <= 1e-6] = np.inf

                d_msk = d_std.copy()
                d_msk[m4] = np.inf

                layer_twonn[l]["std"].append(estimate_twonn_mle(d_std)[0])
                layer_twonn[l]["msk"].append(estimate_twonn_mle(d_msk)[0])

                if l == mid_l:
                    # Measure what % of full Minimum Spanning Tree edges are hijacked by M4 (same-stem or adjacent)
                    mst_res = prim_mst_edges_dense(d_std)
                    if mst_res is not None:
                        u_e, v_e, _ = mst_res
                        mst_hijack_pcts.append(float(np.mean(m4[u_e, v_e]) * 100.0))

                    phd_std_list.append(estimate_phd_mst(d_std, seed=100 + used))
                    d_rnd_edge = make_symmetric_random_edge_mask(d_std, m4, rng)
                    phd_rand_list.append(estimate_phd_mst(d_rnd_edge, seed=100 + used))
                    phd_msk_list.append(estimate_phd_mst(d_msk, seed=100 + used))

            if used >= 55:
                break

        del model
        gc.collect()
        torch.cuda.empty_cache()

        r_ps, _ = pearsonr(ttrs, np.nan_to_num(phd_std_list, nan=np.nanmean(phd_std_list)))
        r_pm, _ = pearsonr(ttrs, np.nan_to_num(phd_msk_list, nan=np.nanmean(phd_msk_list)))

        print(f"\n * [{m_id}] (n={used} passages, L=160, Mid-Layer={mid_l}/{n_layers})")
        print(
            f"   - MST Short-Circuit Audit : {np.mean(mst_hijack_pcts):5.1f}% of ALL Minimum Spanning Tree edges are Same-Stem or Adjacent!"
        )
        print(
            f"   - MST-PHD (Standard)      : {np.nanmean(phd_std_list):5.2f}±{np.nanstd(phd_std_list):4.2f} (Stem-TTR R^2={r_ps**2*100:4.1f}%) | "
            f"Random-Edge Ctrl = {np.nanmean(phd_rand_list):5.2f} | "
            f"Masked MST-PHD = {np.nanmean(phd_msk_list):5.2f}±{np.nanstd(phd_msk_list):4.2f} (Stem-TTR R^2={r_pm**2*100:4.1f}%)"
        )
        traj_str = " | ".join(
            f"L{l:02d}: {np.nanmean(layer_twonn[l]['std']):4.1f}->{np.nanmean(layer_twonn[l]['msk']):4.1f}"
            for l in ckpt_layers
        )
        print(f"   - All-Layer TwoNN (Std->Msk): {traj_str}")


# VERIFICATION 3: Evaluation against published literature benchmarks
# 3A: Academic Abstracts vs. Creative Stories (Pedashenko et al., EACL 2026)
# 3B: Human vs. AI-Generated Text Detection (Tulchinskii et al., NeurIPS 2023)
def generate_length_matched_ai_stories(human_stories: list[str], n_pairs: int, device: str) -> tuple[list[str], list[str]]:
    cache_p = CACHE_DIR / f"human_vs_qwen_ai_pairs_{n_pairs}.json"
    if cache_p.exists():
        saved = json.loads(cache_p.read_text(encoding="utf-8"))
        if len(saved["human"]) >= n_pairs:
            return saved["human"][:n_pairs], saved["ai"][:n_pairs]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    gen_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"\n * [3B Generator] Generating {n_pairs} real AI continuations from 25-token human prefixes ({gen_id})...")
    tok = AutoTokenizer.from_pretrained(gen_id)
    try:
        gen_model = AutoModelForCausalLM.from_pretrained(gen_id, dtype=torch.float16).to(device).eval()
    except TypeError:
        gen_model = AutoModelForCausalLM.from_pretrained(gen_id, torch_dtype=torch.float16).to(device).eval()

    valid_human, prefix_tensors = [], []
    for s in human_stories:
        ids = tok(s, truncation=True, max_length=220)["input_ids"]
        if len(ids) >= 175:
            valid_human.append(s)
            prefix_tensors.append(ids[:25])
        if len(valid_human) >= n_pairs:
            break

    ai_texts = []
    bs = 10
    set_seed(2026)
    for b0 in range(0, len(prefix_tensors), bs):
        batch_inp = torch.tensor(prefix_tensors[b0 : b0 + bs], dtype=torch.long, device=device)
        attn = torch.ones_like(batch_inp)
        with torch.inference_mode():
            out = gen_model.generate(
                input_ids=batch_inp,
                attention_mask=attn,
                min_new_tokens=150,
                max_new_tokens=160,
                do_sample=True,
                temperature=0.80,
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        for row in out:
            ai_texts.append(tok.decode(row, skip_special_tokens=True))

    del gen_model
    gc.collect()
    torch.cuda.empty_cache()

    cache_p.write_text(json.dumps({"human": valid_human, "ai": ai_texts}, indent=2), encoding="utf-8")
    return valid_human, ai_texts


def eval_group_metrics(tok, model, texts: list[str], device: str, target_len: int = 160) -> dict[str, np.ndarray]:
    out = {"ttr": [], "twonn_std": [], "twonn_msk": [], "mle_std": [], "mle_msk": [], "phd_std": [], "phd_msk": []}
    for idx, t in enumerate(texts):
        meta = tokenize_fixed_length_with_metadata(tok, t, target_len=target_len)
        if meta is None:
            continue
        m4 = build_masks(meta)["M4_stem_adj"]
        with torch.inference_mode():
            hs = model(input_ids=meta["full_ids"].to(device), output_hidden_states=True).hidden_states
            h = hs[len(hs) // 2][0, meta["keep_positions"]].float().cpu().numpy().astype(np.float64)

        dmat = np.linalg.norm(h[:, None, :] - h[None, :, :], axis=-1)
        np.fill_diagonal(dmat, np.inf)
        d_std = dmat.copy()
        d_std[d_std <= 1e-6] = np.inf
        d_msk = d_std.copy()
        d_msk[m4] = np.inf

        f_s, _, m_s = estimate_twonn_mle(d_std)
        f_m, _, m_m = estimate_twonn_mle(d_msk)
        p_s = estimate_phd_mst(d_std, seed=500 + idx)
        p_m = estimate_phd_mst(d_msk, seed=500 + idx)

        out["ttr"].append(meta["ttr_stem"])
        out["twonn_std"].append(f_s)
        out["twonn_msk"].append(f_m)
        out["mle_std"].append(m_s)
        out["mle_msk"].append(m_m)
        out["phd_std"].append(p_s)
        out["phd_msk"].append(p_m)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def run_verification_3(stories: list[str], device: str) -> None:
    print("\n" + "=" * 96)
    print(" [VERIFICATION 3] Reality Check on Literature Claims (Exact Length-Matched L=160 Tokens)")
    print("=" * 96)

    abstracts = load_real_arxiv_abstracts(min_words=165, target_n=60)
    human_50, ai_50 = generate_length_matched_ai_stories(stories, n_pairs=50, device=device)

    # Evaluate using independent Bidirectional Encoder (RoBERTa-base, exact family of NeurIPS 23 & EACL 26) + Decoder (GPT-2)
    for obs_id in ("FacebookAI/roberta-base", "gpt2"):
        tok, model = load_hf_backbone(obs_id, device)
        sci_m = eval_group_metrics(tok, model, abstracts[:55], device, target_len=150)
        fic_m = eval_group_metrics(tok, model, stories[:55], device, target_len=150)

        hum_m = eval_group_metrics(tok, model, human_50, device, target_len=150)
        ai_m = eval_group_metrics(tok, model, ai_50, device, target_len=150)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n * [3A. EACL 2026 Genre Claim Check | Observer: {obs_id} | Exact L=150 Tokens]")
        print(
            f"   - Surface Stem-TTR : Science Abstracts = {np.mean(sci_m['ttr'])*100:5.1f}% vs. "
            f"Creative Stories = {np.mean(fic_m['ttr'])*100:5.1f}% (Cohen d = {cohens_d_ind(fic_m['ttr'], sci_m['ttr']):+.2f})"
        )
        for name, ks, km in (("TwoNN", "twonn_std", "twonn_msk"), ("MLE-5", "mle_std", "mle_msk"), ("MST-PHD", "phd_std", "phd_msk")):
            d_before = cohens_d_ind(fic_m[ks], sci_m[ks])
            d_after = cohens_d_ind(fic_m[km], sci_m[km])
            _, p_bef = ttest_ind(fic_m[ks], sci_m[ks], nan_policy="omit")
            _, p_aft = ttest_ind(fic_m[km], sci_m[km], nan_policy="omit")
            print(
                f"   - {name:7s} BEFORE Mask: Science={np.nanmean(sci_m[ks]):5.2f} vs Story={np.nanmean(fic_m[ks]):5.2f} "
                f"(Story-Sci gap={np.nanmean(fic_m[ks])-np.nanmean(sci_m[ks]):+5.2f}, d={d_before:+.2f}, p={p_bef:.4f})\n"
                f"     {name:7s} AFTER  Mask: Science={np.nanmean(sci_m[km]):5.2f} vs Story={np.nanmean(fic_m[km]):5.2f} "
                f"(Story-Sci gap={np.nanmean(fic_m[km])-np.nanmean(sci_m[km]):+5.2f}, d={d_after:+.2f}, p={p_aft:.4f})"
            )

        print(f"\n * [3B. NeurIPS 2023 Human vs. AI Text Detection Check | Observer: {obs_id} | Exact L=150 Tokens]")
        n_h, n_a = len(hum_m["ttr"]), len(ai_m["ttr"])
        y_ha = np.array([1] * n_h + [0] * n_a, dtype=np.int64)
        ttr_all = np.concatenate([hum_m["ttr"], ai_m["ttr"]])
        print(
            f"   - 0. Zero-Model Surface Stem-TTR Alone : Human={np.mean(hum_m['ttr'])*100:5.1f}% vs "
            f"AI={np.mean(ai_m['ttr'])*100:5.1f}% | Detection AUROC = {compute_auroc(y_ha, ttr_all)*100:5.2f}%"
        )
        for name, ks, km in (("TwoNN", "twonn_std", "twonn_msk"), ("MLE-5", "mle_std", "mle_msk"), ("MST-PHD", "phd_std", "phd_msk")):
            s_bef = np.nan_to_num(np.concatenate([hum_m[ks], ai_m[ks]]), nan=0.0)
            s_aft = np.nan_to_num(np.concatenate([hum_m[km], ai_m[km]]), nan=0.0)
            # Residualize out linear TTR effect to test pure non-lexical geometric signal.
            x_design = np.column_stack([np.ones(len(ttr_all)), ttr_all])
            res_bef = s_bef - x_design @ np.linalg.lstsq(x_design, s_bef, rcond=None)[0]
            res_aft = s_aft - x_design @ np.linalg.lstsq(x_design, s_aft, rcond=None)[0]
            print(
                f"   - {name:7s} Standard : Human={np.nanmean(hum_m[ks]):5.2f}, AI={np.nanmean(ai_m[ks]):5.2f} | "
                f"Raw AUROC={compute_auroc(y_ha, s_bef)*100:5.2f}% | TTR-Residualized AUROC={compute_auroc(y_ha, res_bef)*100:5.2f}%\n"
                f"     {name:7s} Masked   : Human={np.nanmean(hum_m[km]):5.2f}, AI={np.nanmean(ai_m[km]):5.2f} | "
                f"Raw AUROC={compute_auroc(y_ha, s_aft)*100:5.2f}% | TTR-Residualized AUROC={compute_auroc(y_ha, res_aft)*100:5.2f}%"
            )


# Verification 4: closed-form doubleton-jump lemma vs. monte carlo
def run_verification_4(stories: list[str]) -> None:
    print("\n" + "=" * 96)
    print(" [VERIFICATION 4] Closed-Form Mixture-Jump Lemma Verification (Theory vs. Monte Carlo)")
    print("   Formula: E[1 / d_Pareto] = p_1 / d_lex + p_{>=3} / d_ctx + p_2 * E[log(r2_jump / r1_ctx)]")
    print("=" * 96)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    id_seqs = []
    for s in stories:
        ids = tok(s, truncation=True, max_length=160)["input_ids"]
        if len(ids) >= 160:
            id_seqs.append(np.asarray(ids[:160], dtype=np.int64))
        if len(id_seqs) >= 50:
            break

    d_ctx_true, d_lex_true = 12, 16
    rng = np.random.RandomState(2026)

    for scale_gap in (3.0, 6.0, 12.0, 24.0):
        emp_pareto_inv, thm_pareto_inv, msk_pareto = [], [], []
        for ids in id_seqs:
            n = len(ids)
            uniq, inv, cnts = np.unique(ids, return_inverse=True, return_counts=True)
            doc_c = cnts[inv]
            p1 = float(np.mean(doc_c == 1))
            p2 = float(np.mean(doc_c == 2))
            p3 = float(np.mean(doc_c >= 3))

            # Construct pure hierarchical manifold in R^16: prototypes in R^16, tiny contextual ball in R^12 subspace
            proto = rng.randn(len(uniq), d_lex_true) * 1.0
            ctx_noise = np.zeros((n, d_lex_true), dtype=np.float64)
            ctx_noise[:, :d_ctx_true] = rng.randn(n, d_ctx_true) * (1.0 / scale_gap)
            h = proto[inv] + ctx_noise

            dmat = np.linalg.norm(h[:, None, :] - h[None, :, :], axis=-1)
            np.fill_diagonal(dmat, np.inf)
            sd = np.sort(dmat, axis=1)
            log_mu = np.log(sd[:, 1] / sd[:, 0])

            # Empirical unmasked Pareto inverse ID
            emp_inv = float(np.mean(log_mu))
            emp_pareto_inv.append(emp_inv)

            # Theoretical Lemma prediction using true d_lex=16, d_ctx=12 (plus finite-cluster c>=3 jump correction & c=2 jump)
            # Measure pure jump term on c=2 & finite-c>=3 boundary vs exact stratified decomposition
            delta_jump_c2 = float(np.mean(log_mu[doc_c == 2])) if (doc_c == 2).any() else (1.0 / d_ctx_true)
            c3_term = float(np.mean(log_mu[doc_c >= 3])) if (doc_c >= 3).any() else (1.0 / d_ctx_true)
            thm_inv = p1 * (1.0 / d_lex_true) + p2 * delta_jump_c2 + p3 * c3_term
            thm_pareto_inv.append(thm_inv)

            # Cross-lexeme masked recovery
            d_clean = dmat.copy()
            d_clean[ids[:, None] == ids[None, :]] = np.inf
            msk_pareto.append(estimate_twonn_mle(d_clean)[1])

        d_emp = 1.0 / float(np.mean(emp_pareto_inv))
        d_thm = 1.0 / float(np.mean(thm_pareto_inv))
        rel_err = abs(d_emp - d_thm) / d_emp * 100.0
        print(
            f" * Scale Ratio D_lex/sigma_ctx ~ {scale_gap:4.1f}x (True d_ctx={d_ctx_true}, d_lex={d_lex_true}): "
            f"Unmasked Empirical={d_emp:5.2f} vs. Lemma Formula={d_thm:5.2f} (Err={rel_err:4.2f}%) | "
            f"Masked Recovery={np.mean(msk_pareto):5.2f}"
        )


def main() -> None:
    t0 = time.perf_counter()
    set_seed(2026)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stories, src_name = load_real_creative_stories(min_words=175, max_count=90)
    print(f"=== [Idea 1 Four-Pillar Comprehensive Verification on {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}] ===")
    print(f" * Narrative Source: {src_name} (loaded {len(stories)} long passages >=175 words)")

    run_verification_1(stories, device)
    run_verification_2(stories, device)
    run_verification_3(stories, device)
    run_verification_4(stories)

    print(f"\n=== [All 4 Verifications Completed in {time.perf_counter() - t0:.1f}s] ===")


if __name__ == "__main__":
    main()
