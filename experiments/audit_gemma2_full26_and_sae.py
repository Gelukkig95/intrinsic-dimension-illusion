import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, rankdata, ttest_ind, ttest_rel
from transformers import AutoModel, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

THIS_DIR = Path(__file__).resolve().parent
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
hf_tok = os.environ["HF_TOKEN"].strip()

arxiv_150, _ = load_arxiv_pool(target_n=150)
stories_150, _ = load_stories_pool(min_words=175, target_n=150, long_min_words=265)
hc3_hum, hc3_ai = load_hc3_chatgpt_pool(target_pairs=80)

sae_p = hf_hub_download("google/gemma-scope-2b-pt-res", "layer_16/width_16k/average_l0_78/params.npz", token=hf_tok)
npz = np.load(sae_p)
W_enc = torch.from_numpy(npz["W_enc"]).float().cuda()
b_enc = torch.from_numpy(np.squeeze(npz["b_enc"])).float().cuda()
thr = torch.from_numpy(np.squeeze(npz["threshold"])).float().cuda()
W_dec = torch.from_numpy(npz["W_dec"]).float().cuda()
b_dec = torch.from_numpy(np.squeeze(npz["b_dec"])).float().cuda()

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=hf_tok)
try:
    mod = AutoModel.from_pretrained("google/gemma-2-2b", dtype=torch.bfloat16, low_cpu_mem_usage=True, token=hf_tok)
except TypeError:
    mod = AutoModel.from_pretrained("google/gemma-2-2b", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, token=hf_tok)
mod.eval()

# Explicit hook functions returning None so PyTorch never overwrites hidden_states
def _emb_pre(m, inp):
    return (inp[0].cpu(),)

def _emb_post(m, inp, out):
    return out.to("cuda", non_blocking=True)

def _blk_pre(m, inp):
    m.to("cuda", non_blocking=True)
    return None

def _blk_post(m, inp, out):
    m.to("cpu")
    return None

mod.embed_tokens.to("cpu")
mod.embed_tokens.register_forward_pre_hook(_emb_pre)
mod.embed_tokens.register_forward_hook(_emb_post)
if hasattr(mod, "rotary_emb"):
    mod.rotary_emb.to("cuda")
mod.norm.to("cuda")

for idx, blk in enumerate(mod.layers):
    if idx < 16:
        blk.to("cuda")
    else:
        blk.to("cpu")
        blk.register_forward_pre_hook(_blk_pre)
        blk.register_forward_hook(_blk_post)

cap = {}
def _make_np_hook(slot_name: str):
    def _fn(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        cap[slot_name] = t.detach().double().cpu().numpy()
        return None
    return _fn

def _l16_gpu_hook(m, inp, out):
    t = out[0] if isinstance(out, tuple) else out
    cap["L16_gpu"] = t.detach().float()
    return None

mod.layers[12].register_forward_hook(_make_np_hook("L13"))
mod.layers[16].register_forward_hook(_l16_gpu_hook)
mod.norm.register_forward_hook(_make_np_hook("L26"))

adj150 = np.abs(np.arange(SEQ_L)[:, None] - np.arange(SEQ_L)[None, :]) <= 1
stats = {
    k: {
        "sci_tw0": [], "sty_tw0": [], "sci_tw4": [], "sty_tw4": [],
        "sci_ml0": [], "sty_ml0": [], "sci_ml4": [], "sty_ml4": [],
        "sci_phd0": [], "sty_phd0": [], "sci_phd4": [], "sty_phd4": [],
    }
    for k in ("L13", "L16", "L26")
}
ttrs = {0: [], 1: []}
feats, bos_sink_ratios, content0_ratios, nn1_hijacks = [], [], [], []
sae_r2_list, sae_l0_list = [], []
use_bdec_center = None

t0 = time.time()
for is_sty, pool in ((0, arxiv_150), (1, stories_150)):
    pfx = "sty" if is_sty else "sci"
    for idx, txt in enumerate(pool):
        mt = tokenize_fixed_length_with_metadata(tok, txt, SEQ_L)
        if mt is None:
            continue
        kp = mt["keep_positions"]
        m4 = build_masks(mt)["M4_stem_adj"]
        same_s = np.eye(SEQ_L, dtype=bool) | (m4 & (~adj150))
        reach = (same_s.astype(np.int32) @ same_s.astype(np.int32)) > 0
        ttrs[is_sty].append(float(np.sum(1.0 / np.maximum(np.sum(reach, axis=1), 1)) / SEQ_L))

        cap.clear()
        with torch.inference_mode():
            mod(mt["full_ids"].cuda(), use_cache=False)
            x16 = cap["L16_gpu"][0, kp]  # (150, 2304) on CUDA

            if use_bdec_center is None:
                # Auto-verify official JumpReLU centering convention on passage 0
                pre_a = x16 @ W_enc + b_enc
                act_a = torch.where(pre_a > thr, pre_a, torch.zeros_like(pre_a))
                rec_a = act_a @ W_dec + b_dec
                r2_a = float(1.0 - ((x16 - rec_a) ** 2).sum() / ((x16 - x16.mean(0, keepdim=True)) ** 2).sum())

                pre_b = (x16 - b_dec) @ W_enc + b_enc
                act_b = torch.where(pre_b > thr, pre_b, torch.zeros_like(pre_b))
                rec_b = act_b @ W_dec + b_dec
                r2_b = float(1.0 - ((x16 - rec_b) ** 2).sum() / ((x16 - x16.mean(0, keepdim=True)) ** 2).sum())
                use_bdec_center = r2_b > r2_a
                print(f" * [Gemma-Scope Convention Verified] Uncentered R^2={r2_a:.4f} vs Centered R^2={r2_b:.4f} -> using center={use_bdec_center}")

            pre = ((x16 - b_dec) if use_bdec_center else x16) @ W_enc + b_enc
            acts = torch.where(pre > thr, pre, torch.zeros_like(pre))
            if idx < 25:
                rec = acts @ W_dec + b_dec
                sae_r2_list.append(float(1.0 - ((x16 - rec) ** 2).sum() / ((x16 - x16.mean(0, keepdim=True)) ** 2).sum()))
                sae_l0_list.append(float((acts > 0).float().sum(dim=-1).mean().item()))

            seq_act = acts.sum(dim=0).cpu().numpy()
            cap["L16"] = x16.double().cpu().numpy()[None, :, :]

        # Check <bos> norm (index 0 of full sequence) vs content norms at L13
        full_l13 = cap["L13"][0]
        c_norms = np.linalg.norm(full_l13[kp], axis=1)
        bos_norm = float(np.linalg.norm(full_l13[0]))
        bos_sink_ratios.append(bos_norm / float(np.median(c_norms)))
        content0_ratios.append(float(c_norms[0] / np.median(c_norms[1:])))

        feats.append(seq_act)
        for lk in ("L13", "L16", "L26"):
            h = cap[lk][0] if lk == "L16" else cap[lk][0, kp]
            dm = cdist(h, h); np.fill_diagonal(dm, np.inf); dm[dm <= 1e-6] = np.inf
            if lk == "L13":
                nn1 = np.argmin(dm, axis=1)
                nn1_hijacks.append(float(np.mean([m4[r_i, nn1[r_i]] for r_i in range(SEQ_L)])))
            tw0, _, ml0 = estimate_twonn_mle(dm)
            dm4 = dm.copy(); dm4[m4] = np.inf
            tw4, _, ml4 = estimate_twonn_mle(dm4)
            stats[lk][f"{pfx}_tw0"].append(tw0); stats[lk][f"{pfx}_tw4"].append(tw4)
            stats[lk][f"{pfx}_ml0"].append(ml0); stats[lk][f"{pfx}_ml4"].append(ml4)
            if idx < 60:
                stats[lk][f"{pfx}_phd0"].append(estimate_phd_mst(dm.copy()))
                stats[lk][f"{pfx}_phd4"].append(estimate_phd_mst(dm4.copy()))

# Also run HC3 Human vs ChatGPT (80 pairs) on Gemma-2-2B L13 & L26
def auroc_pos_neg(p: np.ndarray, n: np.ndarray) -> float:
    p, n = p[np.isfinite(p)], n[np.isfinite(n)]
    rk = rankdata(np.concatenate([p, n]))
    n1, n0 = len(p), len(n)
    return float((np.sum(rk[:n1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0) * 100.0)

hc3_store = {0: {"ttr": [], "L13": [], "L26": []}, 1: {"ttr": [], "L13": [], "L26": []}}
for is_hum, pool in ((1, hc3_hum[:80]), (0, hc3_ai[:80])):
    for txt in pool:
        mt = tokenize_fixed_length_with_metadata(tok, txt, SEQ_L)
        if mt is None:
            continue
        kp = mt["keep_positions"]
        m4 = build_masks(mt)["M4_stem_adj"]
        same_s = np.eye(SEQ_L, dtype=bool) | (m4 & (~adj150))
        reach = (same_s.astype(np.int32) @ same_s.astype(np.int32)) > 0
        hc3_store[is_hum]["ttr"].append(float(np.sum(1.0 / np.maximum(np.sum(reach, axis=1), 1)) / SEQ_L))
        cap.clear()
        with torch.inference_mode():
            mod(mt["full_ids"].cuda(), use_cache=False)
        for lk in ("L13", "L26"):
            h = cap[lk][0, kp]
            dm = cdist(h, h); np.fill_diagonal(dm, np.inf); dm[dm <= 1e-6] = np.inf
            tw0 = estimate_twonn_mle(dm)[0]
            dma = dm.copy(); dma[adj150] = np.inf
            twa = estimate_twonn_mle(dma)[0]
            dm4 = dm.copy(); dm4[m4] = np.inf
            tw4 = estimate_twonn_mle(dm4)[0]
            hc3_store[is_hum][lk].append((tw0, twa, tw4))

print(
    f"\n[Gemma-2-2B Full 26L + Gemma-Scope + HC3 Completed in {time.time()-t0:.1f}s | N_genre={len(feats)}]\n"
    f" * Gemma-Scope JumpReLU Fidelity: R^2={np.mean(sae_r2_list):.4f}, Mean L0={np.mean(sae_l0_list):.1f}\n"
    f" * 1-NN Short-Circuit Hijack (L13)={np.mean(nn1_hijacks)*100:.1f}% | <bos> Sink Ratio={np.mean(bos_sink_ratios):.2f}x -> Stripped Content-pos0 Ratio={np.mean(content0_ratios):.2f}x"
)

def match_1to1(ta: np.ndarray, tb: np.ndarray, cal: float = 0.015):
    used = np.zeros(len(tb), dtype=bool)
    pairs = []
    for i in np.argsort(ta):
        d = np.abs(tb - ta[i]); d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= cal:
            used[j] = True; pairs.append((int(i), int(j)))
    return pairs

t_sci, t_sty = np.array(ttrs[0]), np.array(ttrs[1])
mp = match_1to1(t_sci, t_sty, 0.015)
is_i, it_i = [p[0] for p in mp], [p[1] for p in mp]

hp = match_1to1(np.array(hc3_store[1]["ttr"]), np.array(hc3_store[0]["ttr"]), 0.015)
ih, ia = [p[0] for p in hp], [p[1] for p in hp]

out_summary = {
    "sae_r2": float(np.mean(sae_r2_list)),
    "sae_l0": float(np.mean(sae_l0_list)),
    "nn1_hijack_L13": float(np.mean(nn1_hijacks) * 100.0),
    "bos_sink_ratio": float(np.mean(bos_sink_ratios)),
    "content_pos0_ratio": float(np.mean(content0_ratios)),
    "layers": {},
    "hc3": {},
    "sae": {},
}

for lk in ("L13", "L26"):
    h_arr = np.array(hc3_store[1][lk]); a_arr = np.array(hc3_store[0][lk])
    full_auc = [auroc_pos_neg(h_arr[:, c], a_arr[:, c]) for c in range(3)]
    mat_auc = [auroc_pos_neg(h_arr[ih, c], a_arr[ia, c]) for c in range(3)]
    out_summary["hc3"][lk] = {"full_m0_adj_m4": full_auc, "mat_m0_adj_m4": mat_auc, "n_mat": len(hp)}
    print(f" * HC3 ({lk}) -> Full M0/Adj/M4 = {full_auc[0]:.2f}% / {full_auc[1]:.2f}% / {full_auc[2]:.2f}% | Matched ({len(hp)}p) = {mat_auc[0]:.2f}% / {mat_auc[1]:.2f}% / {mat_auc[2]:.2f}%")

for lk, lname in (("L13", "Mid Layer 13/26"), ("L16", "SAE Layer 16/26"), ("L26", "Final Layer 26/26 (Pedashenko)")):
    st = {k: np.asarray(v, dtype=np.float64) for k, v in stats[lk].items()}
    d_tw0, d_tw4 = cohens_d_ind(st["sty_tw0"], st["sci_tw0"]), cohens_d_ind(st["sty_tw4"], st["sci_tw4"])
    _, p_tw4 = ttest_ind(st["sty_tw4"], st["sci_tw4"])
    d_ml0, d_ml4 = cohens_d_ind(st["sty_ml0"], st["sci_ml0"]), cohens_d_ind(st["sty_ml4"], st["sci_ml4"])
    _, p_ml4 = ttest_ind(st["sty_ml4"], st["sci_ml4"])
    d_mat = cohens_d_ind(st["sty_ml4"][it_i], st["sci_ml4"][is_i])
    _, p_mat = ttest_rel(st["sty_ml4"][it_i], st["sci_ml4"][is_i])

    all_p0 = np.concatenate([st["sci_phd0"], st["sty_phd0"]])
    all_p4 = np.concatenate([st["sci_phd4"], st["sty_phd4"]])
    nan0 = float(np.mean(~np.isfinite(all_p0)) * 100.0)
    nan4 = float(np.mean(~np.isfinite(all_p4)) * 100.0)
    d_phd0 = cohens_d_ind(st["sty_phd0"][np.isfinite(st["sty_phd0"])], st["sci_phd0"][np.isfinite(st["sci_phd0"])])
    d_phd4 = cohens_d_ind(st["sty_phd4"][np.isfinite(st["sty_phd4"])], st["sci_phd4"][np.isfinite(st["sci_phd4"])])
    _, p_phd4 = ttest_ind(st["sty_phd4"][np.isfinite(st["sty_phd4"])], st["sci_phd4"][np.isfinite(st["sci_phd4"])])

    out_summary["layers"][lk] = {
        "twonn": [float(np.mean(st["sci_tw0"])), float(np.mean(st["sty_tw0"])), float(np.mean(st["sci_tw4"])), float(np.mean(st["sty_tw4"])), float(d_tw0), float(d_tw4), float(p_tw4)],
        "mle5": [float(np.mean(st["sci_ml0"])), float(np.mean(st["sty_ml0"])), float(np.mean(st["sci_ml4"])), float(np.mean(st["sty_ml4"])), float(d_ml0), float(d_ml4), float(p_ml4)],
        "matched_mle5": [len(mp), float(d_mat), float(p_mat)],
        "phd_n120": [float(np.nanmean(st["sci_phd0"])), float(np.nanmean(st["sty_phd0"])), float(d_phd0), nan0, float(np.nanmean(st["sci_phd4"])), float(np.nanmean(st["sty_phd4"])), float(d_phd4), nan4, float(p_phd4)],
    }
    print(
        f"\n -> [{lname}]\n"
        f"    TwoNN : {np.mean(st['sci_tw0']):5.2f}/{np.mean(st['sty_tw0']):5.2f} (d={d_tw0:+.2f}) -> M4 {np.mean(st['sci_tw4']):5.2f}/{np.mean(st['sty_tw4']):5.2f} (d={d_tw4:+.2f}, p={p_tw4:.3g})\n"
        f"    MLE-5 : {np.mean(st['sci_ml0']):5.2f}/{np.mean(st['sty_ml0']):5.2f} (d={d_ml0:+.2f}) -> M4 {np.mean(st['sci_ml4']):5.2f}/{np.mean(st['sty_ml4']):5.2f} (d={d_ml4:+.2f}, p={p_ml4:.3g} | Mat {len(mp)}p d={d_mat:+.2f}, p={p_mat:.3g})\n"
        f"    PHD   : {np.nanmean(st['sci_phd0']):5.2f}/{np.nanmean(st['sty_phd0']):5.2f} (d={d_phd0:+.2f}, NaN={nan0:.1f}%) -> M4 {np.nanmean(st['sci_phd4']):5.2f}/{np.nanmean(st['sty_phd4']):5.2f} (d={d_phd4:+.2f}, NaN={nan4:.1f}%, p={p_phd4:.3g})"
    )

F_mat = np.asarray(feats)
ttr_all = np.concatenate([t_sci, t_sty])
y16_0 = np.concatenate([stats["L16"]["sci_ml0"], stats["L16"]["sty_ml0"]])
y16_4 = np.concatenate([stats["L16"]["sci_ml4"], stats["L16"]["sty_ml4"]])
y26_0 = np.concatenate([stats["L26"]["sci_ml0"], stats["L26"]["sty_ml0"]])
y26_4 = np.concatenate([stats["L26"]["sci_ml4"], stats["L26"]["sty_ml4"]])

print("\n [Individual & Composite Gemma-Scope Feature Correlations Across N=300]")
for grp, fids in (("Neg_Academic", [5984, 14485, 2433, 8104, 5159]), ("Pos_Narrative", [15275, 5228, 6978, 9868, 1693])):
    z_sum = np.zeros(len(F_mat))
    r16_0_l, r16_4_l, r26_0_l, r26_4_l = [], [], [], []
    feat_rows = {}
    for fid in fids:
        fv = F_mat[:, fid]
        fire_pct = float(np.mean(fv > 0) * 100.0)
        if np.std(fv) > 1e-9:
            z_sum += (fv - np.mean(fv)) / np.std(fv)
            r16_0, p16_0 = pearsonr(fv, y16_0); r16_4, p16_4 = pearsonr(fv, y16_4)
            r26_0, p26_0 = pearsonr(fv, y26_0); r26_4, p26_4 = pearsonr(fv, y26_4)
            rt, _ = pearsonr(fv, ttr_all)
        else:
            r16_0 = p16_0 = r16_4 = p16_4 = r26_0 = p26_0 = r26_4 = p26_4 = rt = 0.0
        r16_0_l.append(r16_0); r16_4_l.append(r16_4); r26_0_l.append(r26_0); r26_4_l.append(r26_4)
        feat_rows[f"16-{fid}"] = [fire_pct, float(rt), float(r16_0), float(p16_0), float(r16_4), float(p16_4), float(r26_0), float(p26_0), float(r26_4), float(p26_4)]
        print(f"  16-{fid:<5d} (fires={fire_pct:4.1f}%, r_TTR={rt:+.2f}) | L16: M0={r16_0:+.3f}(p={p16_0:.2g})->M4={r16_4:+.3f}(p={p16_4:.2g}) | L26: M0={r26_0:+.3f}(p={p26_0:.2g})->M4={r26_4:+.3f}(p={p26_4:.2g})")
    rc16_0, pc16_0 = pearsonr(z_sum, y16_0); rc16_4, pc16_4 = pearsonr(z_sum, y16_4)
    rc26_0, pc26_0 = pearsonr(z_sum, y26_0); rc26_4, pc26_4 = pearsonr(z_sum, y26_4)
    out_summary["sae"][grp] = {
        "features": feat_rows,
        "mean_r_L16_m0_m4": [float(np.mean(r16_0_l)), float(np.mean(r16_4_l))],
        "mean_r_L26_m0_m4": [float(np.mean(r26_0_l)), float(np.mean(r26_4_l))],
        "z_composite_L16": [float(rc16_0), float(pc16_0), float(rc16_4), float(pc16_4)],
        "z_composite_L26": [float(rc26_0), float(pc26_0), float(rc26_4), float(pc26_4)],
    }
    print(
        f"  ==> [{grp} Summary] Mean r L16: {np.mean(r16_0_l):+.3f} -> {np.mean(r16_4_l):+.3f} (Z-comp: {rc16_0:+.3f} [p={pc16_0:.3g}] -> {rc16_4:+.3f} [p={pc16_4:.3g}])\n"
        f"      {'':19s} Mean r L26: {np.mean(r26_0_l):+.3f} -> {np.mean(r26_4_l):+.3f} (Z-comp: {rc26_0:+.3f} [p={pc26_0:.3g}] -> {rc26_4:+.3f} [p={pc26_4:.3g}])"
    )

(THIS_DIR.parent / "results" / "gemma2_full26_sae_stats.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")
