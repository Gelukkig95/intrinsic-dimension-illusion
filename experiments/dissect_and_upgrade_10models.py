import gc
import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import quad
from scipy.spatial.distance import cdist
from scipy.special import beta as beta_fn, betainc, digamma
from scipy.stats import rankdata, ttest_ind, ttest_rel
from transformers import AutoConfig, AutoModel, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

THIS_DIR = Path(__file__).resolve().parent
FIG_DIR = THIS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(THIS_DIR))

from run_modern_scaleup import load_arxiv_pool, load_hc3_chatgpt_pool, load_stories_pool
from verify_idea1_all_4 import (
    build_masks,
    cohens_d_ind,
    estimate_phd_mst,
    estimate_twonn_mle,
    tokenize_fixed_length_with_metadata,
)

SEQ_L = 150
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _StopForward(Exception):
    pass


def free_mem() -> float:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_b, _ = torch.cuda.mem_get_info()
        return free_b / (1024 ** 3)
    return 0.0


def resolve_text_backbone(mod):
    if hasattr(mod, "visual"):
        mod.visual = None
    if hasattr(mod, "language_model") and hasattr(mod.language_model, "layers"):
        return mod.language_model, mod.language_model.layers
    if hasattr(mod, "layers"):
        return mod, mod.layers
    if hasattr(mod, "model") and hasattr(mod.model, "layers"):
        if hasattr(mod.model, "visual"):
            mod.model.visual = None
        if hasattr(mod.model, "language_model") and hasattr(mod.model.language_model, "layers"):
            return mod.model.language_model, mod.model.language_model.layers
        return mod.model, mod.model.layers
    raise RuntimeError(f"Cannot locate .layers on {type(mod).__name__}")


def exact_stem_ttr(mt: dict, m4: np.ndarray, adj: np.ndarray) -> float:
    for k in ("stems", "token_stems", "stem_ids", "norm_stems"):
        if k in mt and len(mt[k]) == SEQ_L:
            return float(len(set(mt[k])) / SEQ_L)
    same_s = np.eye(SEQ_L, dtype=bool) | (m4 & (~adj))
    reach = (same_s.astype(np.int32) @ same_s.astype(np.int32)) > 0
    return float(np.sum(1.0 / np.maximum(np.sum(reach, axis=1), 1)) / SEQ_L)


def match_pairs_1to1(ttr_a: np.ndarray, ttr_b: np.ndarray, caliper: float = 0.015) -> list[tuple[int, int]]:
    used_b = np.zeros(len(ttr_b), dtype=bool)
    pairs = []
    for i in np.argsort(ttr_a):
        diffs = np.abs(ttr_b - ttr_a[i])
        diffs[used_b] = np.inf
        j = int(np.argmin(diffs))
        if diffs[j] <= caliper:
            used_b[j] = True
            pairs.append((int(i), int(j)))
    return pairs


def auroc_pos_gt_neg(pos: np.ndarray, neg: np.ndarray) -> float:
    p = pos[np.isfinite(pos)]
    n = neg[np.isfinite(neg)]
    ranks = rankdata(np.concatenate([p, n]))
    n1, n0 = len(p), len(n)
    return float((np.sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0) * 100.0)


def dissect_qwen35_all_layers(arxiv_150: list[str], stories_150: list[str]) -> list[float]:
    print("=" * 104)
    print(" [PART 1: MECHANISTIC SINK DISSECTION] Qwen/Qwen3.5-0.8B Across All 24 Layers (Linear vs. Gated Full-Attn)")
    print("=" * 104)
    mid = "Qwen/Qwen3.5-0.8B"
    tok = AutoTokenizer.from_pretrained(mid)
    dt = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if DEVICE == "cuda" else torch.float32)
    try:
        mod = AutoModel.from_pretrained(mid, dtype=dt, low_cpu_mem_usage=True)
    except TypeError:
        mod = AutoModel.from_pretrained(mid, torch_dtype=dt, low_cpu_mem_usage=True)
    mod.eval()
    text_mod, layers = resolve_text_backbone(mod)
    mod.to(DEVICE)

    n_layers = len(layers)
    layer_types = []
    for idx, blk in enumerate(layers):
        lt = getattr(blk, "layer_type", None)
        if lt is None:
            lt = "linear_attention" if hasattr(blk, "linear_attn") else "full_attention"
        layer_types.append(lt)

    # Check if first token is BOS or real content token
    sample_mt = tokenize_fixed_length_with_metadata(tok, arxiv_150[0], SEQ_L)
    first_kp = int(sample_mt["keep_positions"][0])
    first_tok_str = repr(tok.decode([int(sample_mt["full_ids"][0, first_kp])]))
    print(f" * Tokenizer BOS check: first content index = {first_kp} (token={first_tok_str}, bos_token={tok.bos_token})")

    # Hook embed_tokens (l=0), every decoder block output (l=1.24 unnormalized), plus final norm (post-LN)
    states = {}
    hooks = []
    hooks.append(text_mod.embed_tokens.register_forward_hook(lambda m, i, o: states.__setitem__(0, o.detach().float().cpu().numpy())))
    for idx, blk in enumerate(layers):
        def _mk(l_idx):
            def _fn(m, i, o):
                t = o[0] if isinstance(o, tuple) else o
                states[l_idx] = t.detach().float().cpu().numpy()
            return _fn
        hooks.append(blk.register_forward_hook(_mk(idx + 1)))
    if hasattr(text_mod, "norm") and text_mod.norm is not None:
        hooks.append(text_mod.norm.register_forward_hook(lambda m, i, o: states.__setitem__("final_norm", o.detach().float().cpu().numpy())))

    # Also separate attn/deltanet delta vs MLP delta specifically at Layer 11 (idx 10) and Layer 12 (idx 11)
    sub_deltas = {}
    for b_idx in (10, 11):
        blk = layers[b_idx]
        attn_mod = getattr(blk, "linear_attn", None) or getattr(blk, "self_attn", None)
        if attn_mod is not None:
            hooks.append(attn_mod.register_forward_hook(
                lambda m, i, o, bi=b_idx: sub_deltas.__setitem__((bi, "attn"), (o[0] if isinstance(o, tuple) else o).detach().float().cpu().numpy())
            ))
        if hasattr(blk, "mlp"):
            hooks.append(blk.mlp.register_forward_hook(
                lambda m, i, o, bi=b_idx: sub_deltas.__setitem__((bi, "mlp"), (o[0] if isinstance(o, tuple) else o).detach().float().cpu().numpy())
            ))

    norm0_by_l = [[] for _ in range(n_layers + 1)]
    med_rest_by_l = [[] for _ in range(n_layers + 1)]
    ratio_by_l = [[] for _ in range(n_layers + 1)]
    cos_step_0 = [[] for _ in range(n_layers + 1)]
    final_norm_ratios = []
    sub_norms = {(10, "attn"): [], (10, "mlp"): [], (11, "attn"): [], (11, "mlp"): []}

    pool = arxiv_150[:40] + stories_150[:40]
    for txt in pool:
        mt = tokenize_fixed_length_with_metadata(tok, txt, SEQ_L)
        if mt is None:
            continue
        kp = mt["keep_positions"]
        states.clear()
        sub_deltas.clear()
        with torch.inference_mode():
            mod(input_ids=mt["full_ids"].to(DEVICE), use_cache=False)

        for l_i in range(n_layers + 1):
            h = states[l_i][0, kp]  # (150, D)
            norms = np.linalg.norm(h, axis=1)
            n0 = float(norms[0])
            m_rest = float(np.median(norms[1:]))
            norm0_by_l[l_i].append(n0)
            med_rest_by_l[l_i].append(m_rest)
            ratio_by_l[l_i].append(n0 / max(m_rest, 1e-8))
            if l_i >= 1:
                h_prev0 = states[l_i - 1][0, kp[0]]
                dh0 = h[0] - h_prev0
                c_val = float(np.dot(h_prev0, dh0) / (np.linalg.norm(h_prev0) * np.linalg.norm(dh0) + 1e-8))
                cos_step_0[l_i].append(c_val)

        if "final_norm" in states:
            hf = states["final_norm"][0, kp]
            nf = np.linalg.norm(hf, axis=1)
            final_norm_ratios.append(float(nf[0] / np.median(nf[1:])))

        for k_sub in sub_norms:
            if k_sub in sub_deltas:
                ds = sub_deltas[k_sub][0, kp]
                dn = np.linalg.norm(ds, axis=1)
                sub_norms[k_sub].append((float(dn[0]), float(np.median(dn[1:]))))

    print("\n Layer | Block Type       | ||h_0||_2 | med_{i>0}||h_i|| | Ratio (pos=0/med) | cos(h_0^{l-1}, dh_0^l)")
    print("-" * 98)
    for l_i in range(n_layers + 1):
        btype = "embed_tokens" if l_i == 0 else layer_types[l_i - 1]
        c_str = "   ---" if l_i == 0 else f"{np.mean(cos_step_0[l_i]):+6.3f}"
        mark = " <-- Full Gated-Attn!" if btype == "full_attention" else ""
        print(
            f"  {l_i:2d}   | {btype:16s} | {np.mean(norm0_by_l[l_i]):9.2f} |    {np.mean(med_rest_by_l[l_i]):9.2f}     | "
            f"     {np.mean(ratio_by_l[l_i]):6.2f}x      |  {c_str}{mark}"
        )
    if final_norm_ratios:
        print(f"  24*  | post_final_norm  |    ---    |       ---        |      {np.mean(final_norm_ratios):6.2f}x      |     ---")

    for bi, bname in ((10, "Layer 11 (Linear DeltaNet)"), (11, "Layer 12 (Full Gated-Attn)")):
        a0, am = np.mean([x[0] for x in sub_norms[(bi, "attn")]]), np.mean([x[1] for x in sub_norms[(bi, "attn")]])
        m0, mm = np.mean([x[0] for x in sub_norms[(bi, "mlp")]]), np.mean([x[1] for x in sub_norms[(bi, "mlp")]])
        print(f" * Submodule L2 update at {bname}: Attn/GDN (pos0={a0:.2f}, med_i>0={am:.2f}) | MLP (pos0={m0:.2f}, med_i>0={mm:.2f})")

    # Build 25-point trajectory matching HuggingFace output_hidden_states convention (index 0.23 unnorm, index 24 post-final-norm)
    curve_for_fig3c = [float(np.mean(ratio_by_l[i])) for i in range(n_layers)]
    curve_for_fig3c.append(float(np.mean(final_norm_ratios)) if final_norm_ratios else float(np.mean(ratio_by_l[n_layers])))

    for h in hooks:
        h.remove()
    del mod, text_mod, layers
    free_mem()
    return curve_for_fig3c


def run_full_n300_audit(
    mid: str,
    arxiv_150: list[str],
    stories_150: list[str],
    hc3_hum: list[str],
    hc3_ai: list[str],
    extra_nope_layer: int | None = None,
) -> dict:
    print("\n" + "=" * 104)
    print(f" [PART 2: FULL N=300 + TTR-MATCHED + HC3 + MST-PHD SINK AUDIT] {mid}")
    print("=" * 104)
    cfg = AutoConfig.from_pretrained(mid)
    tok = AutoTokenizer.from_pretrained(mid)
    dt = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if DEVICE == "cuda" else torch.float32)
    try:
        mod = AutoModel.from_pretrained(mid, dtype=dt, low_cpu_mem_usage=True)
    except TypeError:
        mod = AutoModel.from_pretrained(mid, torch_dtype=dt, low_cpu_mem_usage=True)
    mod.eval()

    text_mod, layers = resolve_text_backbone(mod)
    total_l = len(layers)
    mid_l = total_l // 2

    if hasattr(cfg, "no_rope_layers") and cfg.no_rope_layers is not None:
        print(f" * SmolLM3 no_rope_layers check (1=RoPE, 0=NoPE): L16(blk15)={cfg.no_rope_layers[15]}, L18(blk17)={cfg.no_rope_layers[17]}")

    while len(layers) > mid_l:
        del layers[-1]
    free_gb = free_mem()
    param_gb = sum(p.numel() * p.element_size() for p in mod.parameters()) / (1024 ** 3)
    stream_layers = (DEVICE == "cuda") and (free_gb < param_gb + 0.5)

    hooks = []
    cap = {}
    if DEVICE == "cuda" and not stream_layers:
        mod.to(DEVICE)
    elif DEVICE == "cuda" and stream_layers:
        for name, child in text_mod.named_children():
            if name != "layers" and name != "norm":
                child.to(DEVICE)
        for blk in layers:
            blk.to("cpu")
            hooks.append(blk.register_forward_pre_hook(lambda m, inp: m.to(DEVICE, non_blocking=True)))
            hooks.append(blk.register_forward_hook(lambda m, inp, out: m.to("cpu")))

    def attach(slot: str, stop: bool):
        def _fn(m, i, o):
            h_t = o[0] if isinstance(o, tuple) else o
            cap[slot] = h_t.detach().double().cpu().numpy()
            if stream_layers:
                m.to("cpu")
            if stop:
                raise _StopForward()
        return _fn

    if extra_nope_layer is not None:
        hooks.append(layers[extra_nope_layer - 1].register_forward_hook(attach("nope", stop=False)))
    hooks.append(layers[mid_l - 1].register_forward_hook(attach("mid", stop=True)))

    adj150 = np.abs(np.arange(SEQ_L)[:, None] - np.arange(SEQ_L)[None, :]) <= 1

    def extract_split(texts: list[str], compute_phd_n: int = 0):
        out = {
            "tw0": [], "tw4": [], "ml0": [], "ml4": [], "ttr": [], "nn1": [], "sink": [],
            "tw_adj": [], "phd_all0": [], "phd_all4": [], "phd_pos0": [], "phd_pos4": [],
            "nope_ml0": [], "nope_ml4": [],
        }
        for idx, txt in enumerate(texts):
            mt = tokenize_fixed_length_with_metadata(tok, txt, SEQ_L)
            if mt is None:
                continue
            kp = mt["keep_positions"]
            m4 = build_masks(mt)["M4_stem_adj"]
            ttr_v = exact_stem_ttr(mt, m4, adj150)

            cap.clear()
            with torch.inference_mode():
                try:
                    mod(input_ids=mt["full_ids"].to(DEVICE), use_cache=False)
                except _StopForward:
                    pass

            h = cap["mid"][0, kp]
            norms = np.linalg.norm(h, axis=1)
            out["sink"].append(float(norms[0] / np.median(norms[1:])))

            dm = cdist(h, h)
            np.fill_diagonal(dm, np.inf)
            dm[dm <= 1e-6] = np.inf
            nn1 = np.argmin(dm, axis=1)
            out["nn1"].append(float(np.mean([m4[r_i, nn1[r_i]] for r_i in range(SEQ_L)])))

            tw0, _, ml0 = estimate_twonn_mle(dm)
            dma = dm.copy(); dma[adj150] = np.inf
            twa, _, _ = estimate_twonn_mle(dma)
            dm4 = dm.copy(); dm4[m4] = np.inf
            tw4, _, ml4 = estimate_twonn_mle(dm4)

            out["tw0"].append(tw0); out["tw4"].append(tw4)
            out["ml0"].append(ml0); out["ml4"].append(ml4)
            out["tw_adj"].append(twa); out["ttr"].append(ttr_v)

            if idx < compute_phd_n:
                out["phd_all0"].append(estimate_phd_mst(dm.copy()))
                out["phd_all4"].append(estimate_phd_mst(dm4.copy()))
                dm_p0 = dm[1:, 1:].copy()
                dm_p4 = dm_p0.copy(); dm_p4[m4[1:, 1:]] = np.inf
                out["phd_pos0"].append(estimate_phd_mst(dm_p0))
                out["phd_pos4"].append(estimate_phd_mst(dm_p4))

            if "nope" in cap:
                hn = cap["nope"][0, kp]
                dmn = cdist(hn, hn); np.fill_diagonal(dmn, np.inf); dmn[dmn <= 1e-6] = np.inf
                _, _, n_ml0 = estimate_twonn_mle(dmn)
                dmn4 = dmn.copy(); dmn4[m4] = np.inf
                _, _, n_ml4 = estimate_twonn_mle(dmn4)
                out["nope_ml0"].append(n_ml0); out["nope_ml4"].append(n_ml4)

        return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}

    t0 = time.time()
    sci = extract_split(arxiv_150, compute_phd_n=60)
    sty = extract_split(stories_150, compute_phd_n=60)
    hum = extract_split(hc3_hum[:80], compute_phd_n=0)
    ai = extract_split(hc3_ai[:80], compute_phd_n=0)
    dt_sec = time.time() - t0

    d_tw_bef = float(cohens_d_ind(sty["tw0"], sci["tw0"]))
    d_tw_aft = float(cohens_d_ind(sty["tw4"], sci["tw4"]))
    _, p_tw_aft = ttest_ind(sty["tw4"], sci["tw4"])
    d_ml_bef = float(cohens_d_ind(sty["ml0"], sci["ml0"]))
    d_ml_aft = float(cohens_d_ind(sty["ml4"], sci["ml4"]))
    _, p_ml_aft = ttest_ind(sty["ml4"], sci["ml4"])

    g_pairs = match_pairs_1to1(sci["ttr"], sty["ttr"], caliper=0.015)
    is_i, it_i = [p[0] for p in g_pairs], [p[1] for p in g_pairs]
    d_mat = float(cohens_d_ind(sty["ml4"][it_i], sci["ml4"][is_i]))
    _, p_mat = ttest_rel(sty["ml4"][it_i], sci["ml4"][is_i])

    h_pairs = match_pairs_1to1(hum["ttr"], ai["ttr"], caliper=0.015)
    ih, ia = [p[0] for p in h_pairs], [p[1] for p in h_pairs]
    hc3_res = {
        "full_m0": auroc_pos_gt_neg(hum["tw0"], ai["tw0"]),
        "full_adj": auroc_pos_gt_neg(hum["tw_adj"], ai["tw_adj"]),
        "full_m4": auroc_pos_gt_neg(hum["tw4"], ai["tw4"]),
        "mat_m0": auroc_pos_gt_neg(hum["tw0"][ih], ai["tw0"][ia]),
        "mat_adj": auroc_pos_gt_neg(hum["tw_adj"][ih], ai["tw_adj"][ia]),
        "mat_m4": auroc_pos_gt_neg(hum["tw4"][ih], ai["tw4"][ia]),
        "n_mat": len(h_pairs),
    }

    all0 = np.concatenate([sci["phd_all0"], sty["phd_all0"]])
    all4 = np.concatenate([sci["phd_all4"], sty["phd_all4"]])
    pos0 = np.concatenate([sci["phd_pos0"], sty["phd_pos0"]])
    pos4 = np.concatenate([sci["phd_pos4"], sty["phd_pos4"]])
    sink_res = {
        "norm_ratio": float(np.mean(np.concatenate([sci["sink"], sty["sink"]]))),
        "nn1_hijack": float(np.mean(np.concatenate([sci["nn1"], sty["nn1"]])) * 100.0),
        "nan_all0": float(np.mean(~np.isfinite(all0)) * 100.0),
        "nan_all4": float(np.mean(~np.isfinite(all4)) * 100.0),
        "nan_pos0": float(np.mean(~np.isfinite(pos0)) * 100.0),
        "nan_pos4": float(np.mean(~np.isfinite(pos4)) * 100.0),
        "all0_sci_sty": [float(np.nanmean(sci["phd_all0"])), float(np.nanmean(sty["phd_all0"]))],
        "all4_sci_sty": [float(np.nanmean(sci["phd_all4"])), float(np.nanmean(sty["phd_all4"]))],
        "pos0_sci_sty": [float(np.nanmean(sci["phd_pos0"])), float(np.nanmean(sty["phd_pos0"]))],
        "pos4_sci_sty": [float(np.nanmean(sci["phd_pos4"])), float(np.nanmean(sty["phd_pos4"]))],
    }

    res = {
        "layer": f"{mid_l}/{total_l}",
        "N_genre": int(len(sci["tw0"]) + len(sty["tw0"])),
        "n_matched_pairs": len(g_pairs),
        "twonn": [float(np.mean(sci["tw0"])), float(np.mean(sty["tw0"])), float(np.mean(sci["tw4"])), float(np.mean(sty["tw4"])), d_tw_bef, d_tw_aft, float(p_tw_aft)],
        "mle5": [float(np.mean(sci["ml0"])), float(np.mean(sty["ml0"])), float(np.mean(sci["ml4"])), float(np.mean(sty["ml4"])), d_ml_bef, d_ml_aft, float(p_ml_aft)],
        "ttr_matched_mle5": [d_mat, float(p_mat)],
        "hc3": hc3_res,
        "sink_phd": sink_res,
    }
    if len(sci["nope_ml0"]) > 0:
        res["nope_layer16_mle5"] = [
            float(np.mean(sci["nope_ml0"])), float(np.mean(sty["nope_ml0"])),
            float(np.mean(sci["nope_ml4"])), float(np.mean(sty["nope_ml4"])),
            float(cohens_d_ind(sty["nope_ml0"], sci["nope_ml0"])),
            float(cohens_d_ind(sty["nope_ml4"], sci["nope_ml4"])),
        ]

    print(f" * Done in {dt_sec:.1f}s | N={res['N_genre']} (TTR-matched={len(g_pairs)} pairs) | 1-NN Hijack={sink_res['nn1_hijack']:.1f}% | pos=0 Sink={sink_res['norm_ratio']:.2f}x")
    print(f" * TABLE 1 -> TwoNN: {res['twonn'][0]:.2f}/{res['twonn'][1]:.2f} -> {res['twonn'][2]:.2f}/{res['twonn'][3]:.2f} (d: {d_tw_bef:+.2f} -> {d_tw_aft:+.2f}, p={p_tw_aft:.4g})")
    print(f" * TABLE 1 -> MLE-5: {res['mle5'][0]:.2f}/{res['mle5'][1]:.2f} -> {res['mle5'][2]:.2f}/{res['mle5'][3]:.2f} (d: {d_ml_bef:+.2f} -> {d_ml_aft:+.2f}, p={p_ml_aft:.4g}) | Matched d={d_mat:+.2f} (p={p_mat:.4g})")
    if "nope_layer16_mle5" in res:
        nl = res["nope_layer16_mle5"]
        print(f" * Exact NoPE Layer 16/36 MLE-5: Raw {nl[0]:.2f}/{nl[1]:.2f} (d={nl[4]:+.2f}) -> M4 {nl[2]:.2f}/{nl[3]:.2f} (d={nl[5]:+.2f})")
    print(f" * HC3 TwoNN AUROC -> Full M0/Adj/M4 = {hc3_res['full_m0']:.2f}% / {hc3_res['full_adj']:.2f}% / {hc3_res['full_m4']:.2f}% | Matched ({hc3_res['n_mat']}p) = {hc3_res['mat_m0']:.2f}% / {hc3_res['mat_adj']:.2f}% / {hc3_res['mat_m4']:.2f}%")
    print(
        f" * MST-PHD (N=120) -> AllTok M0 NaN={sink_res['nan_all0']:.1f}% ({sink_res['all0_sci_sty'][0]:.2f}/{sink_res['all0_sci_sty'][1]:.2f}), "
        f"AllTok M4 NaN={sink_res['nan_all4']:.1f}% ({sink_res['all4_sci_sty'][0]:.2f}/{sink_res['all4_sci_sty'][1]:.2f}) | "
        f"DropSink(pos>0) M0 NaN={sink_res['nan_pos0']:.1f}%, M4 NaN={sink_res['nan_pos4']:.1f}% ({sink_res['pos4_sci_sty'][0]:.2f}/{sink_res['pos4_sci_sty'][1]:.2f})"
    )

    for h in hooks:
        h.remove()
    del mod, text_mod, layers
    free_mem()
    return res


def regenerate_figures_with_qwen35_curve(q35_res: dict, sm3_res: dict, q35_curve: list[float]) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9.3,
        "axes.titlesize": 10.3,
        "axes.labelsize": 9.3,
        "legend.fontsize": 7.8,
        "xtick.labelsize": 8.4,
        "ytick.labelsize": 8.4,
    })

    models = [
        "RoBERTa-base (125M)",
        "ModernBERT-base (149M)",
        "GPT-2 (124M)",
        "Qwen2.5-0.5B (0.5B)",
        "Qwen3.5-0.8B (0.8B, Hyb)",
        "OLMo-2-1B (1.0B)",
        "SmolLM2-1.7B (1.7B)",
        "Qwen3-1.7B (1.7B)",
        "SmolLM3-3B (3.1B, RNoPE)",
        "Qwen2.5-7B (7.6B, L14)",
    ]
    d_unm = np.array([-0.25, +0.92, +0.54, +1.35, q35_res["mle5"][4], +0.91, +0.52, +1.04, sm3_res["mle5"][4], +0.56])
    d_m4 = np.array([-1.81, -0.43, -0.53, +0.42, q35_res["mle5"][5], -0.53, -0.86, -0.03, sm3_res["mle5"][5], -0.96])
    d_mat = np.array([-1.54, -0.55, -0.56, +0.37, q35_res["ttr_matched_mle5"][0], -0.62, -0.80, +0.02, sm3_res["ttr_matched_mle5"][0], -0.80])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.65), gridspec_kw={"width_ratios": [1.15, 1.0]})
    y = np.arange(len(models))
    for i in range(len(models)):
        ax1.plot([min(d_unm[i], d_m4[i], d_mat[i]), max(d_unm[i], d_m4[i], d_mat[i])], [y[i], y[i]], color="#555555", lw=1.4, zorder=2)
    ax1.axvline(0.0, color="black", ls="--", lw=1.0, zorder=1)
    ax1.scatter(d_unm, y, marker="^", s=54, color="#d55e00", zorder=4, label=r"Unmasked ($\mathcal{M}_0$)")
    ax1.scatter(d_m4, y, marker="o", s=48, color="#0072b2", zorder=4, label=r"Masked ($\mathcal{M}_4$ Full)")
    ax1.scatter(d_mat, y, marker="D", s=50, facecolors="none", edgecolors="#009e73", linewidths=1.7, zorder=5, label=r"Masked ($\mathcal{M}_4$ TTR-Matched)")
    ax1.set_yticks(y); ax1.set_yticklabels(models); ax1.invert_yaxis()
    ax1.set_xlim(min(-2.65, float(np.min([d_unm, d_m4, d_mat])) - 0.18), 1.55)
    ax1.set_xlabel(r"Genre Effect Size: MLE-5 Cohen's $d$ ($\mu_{\mathrm{Story}} - \mu_{\mathrm{Sci}}$)")
    ax1.set_title("(a) Genre Illusion Reversal Across 10 Backbones", fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=2, frameon=True, columnspacing=0.9)

    lengths = [80, 150, 220, 300]
    for lbl, vals, col, mk, ls in [
        ("ModernBERT Unmasked", [0.64, 1.16, 1.19, 1.43], "#0072b2", "o", "--"),
        (r"ModernBERT Masked $\mathcal{M}_4$", [0.02, -0.29, -0.36, -0.35], "#0072b2", "o", "-"),
        ("GPT-2 Unmasked", [0.54, 1.08, 1.37, 1.62], "#009e73", "s", "--"),
        (r"GPT-2 Masked $\mathcal{M}_4$", [-0.16, -0.19, -0.28, -0.33], "#009e73", "s", "-"),
        ("Qwen3-1.7B Unmasked", [0.56, 1.19, 1.27, 1.40], "#cc79a7", "D", "--"),
        (r"Qwen3-1.7B Masked $\mathcal{M}_4$", [-0.23, 0.01, -0.10, -0.16], "#cc79a7", "D", "-"),
    ]:
        ax2.plot(lengths, vals, color=col, marker=mk, ls=ls, lw=1.6, ms=5.3, label=lbl, zorder=3)
    ax2.axhline(0.0, color="black", ls="--", lw=1.0, zorder=1)
    ax2.set_xticks(lengths); ax2.set_ylim(-0.48, 1.74)
    ax2.set_xlabel(r"Truncated Passage Length $L$ (Tokens)")
    ax2.set_ylabel(r"TwoNN Genre Cohen's $d$")
    ax2.set_title(r"(b) Monotonic Illusion Growth vs. Length $L$", fontweight="bold")
    ax2.grid(True, ls=":", alpha=0.4)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.50, -0.17), ncol=2, frameon=True, columnspacing=0.9)

    fig.subplots_adjust(wspace=0.27)
    fig.savefig(FIG_DIR / "fig2_genre_and_length.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure 3 with full 24-layer Qwen3.5-0.8B sink-free trajectory in panel (c)
    def exp_log_b12(d: int, m: int):
        a = d / 2.0
        bn = beta_fn(a, a)
        p1 = lambda x: m * ((1.0 - betainc(a, a, x)) ** (m - 1)) * (x ** (a - 1.0)) * ((1.0 - x) ** (a - 1.0)) / bn
        e1, _ = quad(lambda x: math.log(x) * p1(x), 1e-12, 1.0 - 1e-12, limit=150)
        if m == 1:
            return e1, float("nan")
        def p2(x):
            c = betainc(a, a, x)
            return m * (m - 1) * c * ((1.0 - c) ** (m - 2)) * (x ** (a - 1.0)) * ((1.0 - x) ** (a - 1.0)) / bn
        e2, _ = quad(lambda x: math.log(x) * p2(x), 1e-12, 1.0 - 1e-12, limit=150)
        return e1, e2

    d_lex, d_ctx, seq_l = 16, 12, 160
    mults = [1] * 66 + [2] * 24 + [3] * 10 + [4] * 4
    tok_v = np.array([v_id for v_id, c in enumerate(mults) for _ in range(c)], dtype=np.int64)
    V, V1 = len(mults), sum(1 for c in mults if c == 1)
    p_c = {c: (c * sum(1 for m in mults if m == c)) / float(seq_l) for c in set(mults)}
    alpha1 = (V1 - 1.0) / (V - 1.0)
    e1_l, e2_l = exp_log_b12(d_lex, V - 1)
    d_l = 0.5 * (e2_l - e1_l)
    d_c = {c: 0.5 * (exp_log_b12(d_ctx, c - 1)[1] - exp_log_b12(d_ctx, c - 1)[0]) for c in set(mults) if c >= 3}
    psi_c = float(digamma(d_ctx / 2.0) - digamma(d_ctx))

    dense_r = np.geomspace(2.0, 32.0, 60)
    thm_c = [1.0 / (p_c[1] * alpha1 * d_l + sum(p_c[c] * d_c[c] for c in p_c if c >= 3) + p_c[2] * (math.log(r) + 0.5 * e1_l - 0.5 * psi_c)) for r in dense_r]
    grid_r = [2.0, 4.0, 8.0, 16.0, 32.0]
    rng = np.random.RandomState(777)
    same_stem = tok_v[:, None] == tok_v[None, :]
    mc_raw, mc_m4 = [], []
    for r in grid_r:
        ri, mv = [], []
        for _ in range(50):
            u = rng.randn(V, d_lex + 1); u = (u / np.linalg.norm(u, axis=1, keepdims=True)) * r
            e = rng.randn(seq_l, d_ctx + 1); e /= np.linalg.norm(e, axis=1, keepdims=True)
            dm = cdist(np.concatenate([u[tok_v], e], axis=1), np.concatenate([u[tok_v], e], axis=1))
            np.fill_diagonal(dm, np.inf)
            ds = np.sort(dm, axis=1)
            ri.append(float(np.mean(np.log(ds[:, 1] / ds[:, 0]))))
            dm[same_stem] = np.inf
            dsm = np.sort(dm, axis=1)
            mv.append(1.0 / float(np.mean(np.log(dsm[:, 1] / dsm[:, 0]))))
        mc_raw.append(1.0 / float(np.mean(ri)))
        mc_m4.append(float(np.mean(mv)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.3, 4.15))
    ax1.axhline(16.0, color="#666666", ls=":", lw=1.2, label=r"True $d_{\mathrm{lex}} = 16$")
    ax1.axhline(12.0, color="#666666", ls="-.", lw=1.2, label=r"True $d_{\mathrm{ctx}} = 12$")
    ax1.plot(np.log2(dense_r), thm_c, color="#c56c19", lw=2.2, label="Theorem 1 (Zero Plug-in)")
    ax1.scatter(np.log2(grid_r), mc_raw, color="black", s=36, zorder=5, label="Monte Carlo Unmasked")
    ax1.plot(np.log2(grid_r), mc_m4, color="#0072b2", ls="--", marker="s", ms=5.5, lw=1.6, label=r"Monte Carlo Masked $\mathcal{M}_4$")
    ax1.set_xticks(np.log2(grid_r)); ax1.set_xticklabels(["2x", "4x", "8x", "16x", "32x"])
    ax1.set_ylim(0.5, 22.8)
    ax1.set_xlabel(r"Scale Separation $R_{\mathrm{lex}}/r_{\mathrm{ctx}}$")
    ax1.set_ylabel("Estimated Intrinsic Dimension")
    ax1.set_title("(a) Theorem 1 Doubleton Collapse", fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.50, -0.21), ncol=2, frameon=True, fontsize=7.5)

    conds = ["Unmasked\n" + r"($\mathcal{M}_0$)", "Adj-Only\n" + r"($|i-j|\le 1$)", "Stem-Only\n" + r"($s_i=s_j$)", "Full " + r"$\mathcal{M}_4$" + "\n(Stem+Adj)"]
    aucs = [78.28, 81.28, 56.26, 58.18]
    bars = ax2.bar(np.arange(4), aucs, width=0.58, color=["#d55e00", "#cc79a7", "#0072b2", "#009e73"], edgecolor="#222222", lw=1.0, zorder=3)
    ax2.axhline(50.0, color="black", ls="--", lw=1.1, zorder=4)
    for b, val in zip(bars, aucs):
        ax2.text(b.get_x() + b.get_width() / 2.0, val + 1.4, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=8.4)
    ax2.set_xticks(np.arange(4)); ax2.set_xticklabels(conds, fontsize=7.9)
    ax2.set_ylim(30.0, 92.5)
    ax2.set_ylabel("TTR-Matched TwoNN AUROC (%)")
    ax2.set_title(r"(b) HC3 Detection $2 \times 2$ Ablation", fontweight="bold")
    ax2.grid(True, axis="y", ls=":", alpha=0.4)

    lp_data = json.loads((THIS_DIR / "long_paper_all_layers.json").read_text(encoding="utf-8-sig"))
    max_y = 1.0
    for m_key, lbl, col, mk in [
        ("Qwen/Qwen3-1.7B", "Qwen3-1.7B (Ungated Sink)", "#c56c19", "o"),
        ("gpt2", "GPT-2 (Ungated Sink)", "#009e73", "s"),
        ("answerdotai/ModernBERT-base", "ModernBERT ([CLS] Stripped)", "#0072b2", "^"),
    ]:
        yv = np.asarray(lp_data[m_key]["sink_cloud0_ratio"]["mean"], dtype=np.float64)
        xv = np.linspace(0.0, 1.0, len(yv))
        max_y = max(max_y, float(np.max(yv)))
        ax3.plot(xv, yv, color=col, marker=mk, ms=3.9, lw=1.5, label=lbl, zorder=3)

    xv_q35 = np.linspace(0.0, 1.0, len(q35_curve))
    ax3.plot(xv_q35, q35_curve, color="#cc79a7", marker="D", ms=3.6, lw=1.7, ls="-", label="Qwen3.5-0.8B (Gated/Linear)", zorder=5)
    ax3.set_yscale("log")
    ax3.set_ylim(0.28, max_y * 1.85)
    ax3.set_xlabel(r"Normalized Depth ($l / L_{\max}$)")
    ax3.set_ylabel("Token-0 / Median L2-Norm Ratio (log)")
    ax3.set_title("(c) Emergence vs. Suppression of Sinks", fontweight="bold")
    ax3.grid(True, which="both", ls=":", alpha=0.35)
    ax3.legend(loc="upper center", bbox_to_anchor=(0.50, -0.21), ncol=2, frameon=True, fontsize=7.3)

    fig.subplots_adjust(wspace=0.27)
    fig.savefig(FIG_DIR / "fig3_theory_hc3_sink.pdf", bbox_inches="tight")
    plt.close(fig)
    print("\n * [SAVED FIGURES] figures/fig2_genre_and_length.pdf & figures/fig3_theory_hc3_sink.pdf (with 24-layer Qwen3.5 trajectory!)")


if __name__ == "__main__":
    stories_150, _ = load_stories_pool(min_words=175, target_n=150, long_min_words=265)
    arxiv_150, _ = load_arxiv_pool(target_n=150)
    hc3_hum, hc3_ai = load_hc3_chatgpt_pool(target_pairs=80)

    q35_curve = dissect_qwen35_all_layers(arxiv_150, stories_150)
    q35 = run_full_n300_audit("Qwen/Qwen3.5-0.8B", arxiv_150, stories_150, hc3_hum, hc3_ai)
    sm3 = run_full_n300_audit("HuggingFaceTB/SmolLM3-3B", arxiv_150, stories_150, hc3_hum, hc3_ai, extra_nope_layer=16)

    out_json = THIS_DIR.parent / "results" / "nextgen_2026_upgrade.json"
    out_json.write_text(json.dumps({"qwen35_24layer_sink_curve": q35_curve, "Qwen3.5-0.8B": q35, "SmolLM3-3B": sm3}, indent=2), encoding="utf-8")
    regenerate_figures_with_qwen35_curve(q35, sm3, q35_curve)
