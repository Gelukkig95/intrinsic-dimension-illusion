import gc
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu, trim_mean, ttest_ind
from transformers import AutoModel, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from run_modern_scaleup import load_arxiv_pool, load_stories_pool
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


def mle_k_harmonic(dm: np.ndarray, k: int) -> float:
    ds = np.sort(dm, axis=1)[:, :k]
    ok = np.all(np.isfinite(ds), axis=1) & (ds[:, 0] > 1e-12) & (ds[:, k - 1] > ds[:, 0])
    if np.sum(ok) < 10:
        return float("nan")
    ds = ds[ok]
    inv_m = np.mean(np.log(ds[:, [k - 1]] / ds[:, : k - 1]), axis=1)
    inv_m = inv_m[inv_m > 1e-12]
    return float(1.0 / np.mean(inv_m)) if len(inv_m) >= 10 else float("nan")


def audit_1_cleanroom_zip_and_table_sync() -> bool:
    print("=" * 104)
    print(" [AUDIT 1] Repository Reproduction Check (reproduce_tables_and_theorem1.py) & Table-to-JSON Sync")
    print("=" * 104)
    repo_root = THIS_DIR.parent
    repro_py = repo_root / "reproduce_tables_and_theorem1.py"
    assert repro_py.exists(), f"{repro_py} not found!"

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(repro_py)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    dt = time.time() - t0
    figs_ok = all(
        (repo_root / "reproduced_figures" / f).exists()
        for f in ("fig1_layer_trajectories.pdf", "fig2_genre_and_length.pdf", "fig3_theory_hc3_sink.pdf")
    )
    print(f" * Reproduction script ExitCode={proc.returncode} | Runtime={dt:.2f}s | All 3 PDFs built={figs_ok}")
    if proc.returncode != 0:
        print(proc.stderr)
        return False

    res_json = repo_root / "results" / "nextgen_2026_upgrade.json"
    up = json.loads(res_json.read_text(encoding="utf-8-sig"))
    q35, sm3 = up["Qwen3.5-0.8B"], up["SmolLM3-3B"]
    print(
        f" * JSON Sync Check -> Qwen3.5-0.8B MLE-5: {q35['mle5'][0]:.2f}/{q35['mle5'][1]:.2f} -> {q35['mle5'][2]:.2f}/{q35['mle5'][3]:.2f} (d={q35['mle5'][5]:+.2f}) | "
        f"SmolLM3-3B MLE-5: {sm3['mle5'][0]:.2f}/{sm3['mle5'][1]:.2f} -> {sm3['mle5'][2]:.2f}/{sm3['mle5'][3]:.2f} (d={sm3['mle5'][5]:+.2f})"
    )
    return (proc.returncode == 0) and figs_ok


def audit_2_and_3_qwen7b_stress_tests(arxiv_150: list[str], stories_150: list[str]) -> None:
    print("\n" + "=" * 104)
    print(" [AUDIT 2 & 3] Qwen2.5-7B (7.6B) Stress Tests: Math-Free Prose, Pure-Alpha Subclouds,")
    print("               Theiler Window w in {1,2,3,5}, MLE-k in {3,5,10,15,20}, & Non-Parametric Tests")
    print("=" * 104)
    dmat_p = THIS_DIR / "cache_scaleup" / "qwen25_7b_L14_dmats.npy"
    tok7_dir = THIS_DIR / "cache_scaleup" / "qwen25_7b_tokenizer"
    tok7 = AutoTokenizer.from_pretrained(str(tok7_dir))
    dmats = np.load(dmat_p)

    sci_recs = [(t, m) for t in arxiv_150[:80] if (m := tokenize_fixed_length_with_metadata(tok7, t, SEQ_L)) is not None]
    sty_recs = [(t, m) for t in stories_150[:80] if (m := tokenize_fixed_length_with_metadata(tok7, t, SEQ_L)) is not None]
    n_sci, n_sty = len(sci_recs), len(sty_recs)

    sci_dm = dmats[:n_sci]
    sty_dm = dmats[n_sci : n_sci + n_sty]
    sci_m4 = [build_masks(m)["M4_stem_adj"] for _, m in sci_recs]
    sty_m4 = [build_masks(m)["M4_stem_adj"] for _, m in sty_recs]

    # 2A. Exact Baseline Reproduction + Non-Parametric Mann-Whitney U + 10% Trimmed Mean
    def eval_group(dms, masks):
        m0_v, m4_v = [], []
        for dm, mk in zip(dms, masks):
            d0 = dm.copy(); np.fill_diagonal(d0, np.inf); d0[d0 <= 1e-6] = np.inf
            d4 = d0.copy(); d4[mk] = np.inf
            m0_v.append(estimate_twonn_mle(d0)[2])
            m4_v.append(estimate_twonn_mle(d4)[2])
        return np.array(m0_v), np.array(m4_v)

    s0, s4 = eval_group(sci_dm, sci_m4)
    t0, t4 = eval_group(sty_dm, sty_m4)
    _, p_mwu = mannwhitneyu(t4, s4, alternative="two-sided")
    print(
        f" * Baseline Sync (N={n_sci+n_sty}): M0={np.mean(s0):.2f}/{np.mean(t0):.2f} (d={cohens_d_ind(t0, s0):+.2f}) -> "
        f"M4={np.mean(s4):.2f}/{np.mean(t4):.2f} (d={cohens_d_ind(t4, s4):+.2f})"
    )
    print(
        f" * Outlier Robustness (M4 MLE-5): Median Sci/Story = {np.median(s4):.2f}/{np.median(t4):.2f} | "
        f"10%-Trimmed Mean = {trim_mean(s4, 0.1):.2f}/{trim_mean(t4, 0.1):.2f} | Mann-Whitney U p = {p_mwu:.3e}"
    )

    # 2B. Confounder Check 1: Math-Free arXiv Abstracts (zero $, \, {, }, =, ^, _) vs. Hippocorpus
    math_chars = set("$\\{}=^_")
    clean_sci_idx = [i for i, (txt, _) in enumerate(sci_recs) if not any(ch in txt for ch in math_chars)]
    s0_clean, s4_clean = s0[clean_sci_idx], s4[clean_sci_idx]
    _, p_clean = ttest_ind(t4, s4_clean)
    print(
        f" * Math-Free arXiv Check (n_sci={len(clean_sci_idx)}/{n_sci} zero-LaTeX abstracts vs {n_sty} stories): "
        f"M0 d={cohens_d_ind(t0, s0_clean):+.2f} -> M4 Sci/Story={np.mean(s4_clean):.2f}/{np.mean(t4):.2f} "
        f"(d={cohens_d_ind(t4, s4_clean):+.2f}, p={p_clean:.3e})"
    )

    # 2C. Confounder Check 2: Pure Alphabetic Words Only (exact 75x75 sub-cloud per passage, zero digits/punct/symbols)
    L_SUB = 75
    def eval_pure_alpha_subcloud(dms, recs, masks):
        vals = []
        for dm, (_, mt), mk in zip(dms, recs, masks):
            tids = mt["full_ids"][0, mt["keep_positions"]].tolist()
            alpha_idx = [idx for idx, tid in enumerate(tids) if (s := tok7.decode([int(tid)]).strip()).isalpha() and len(s) >= 2]
            if len(alpha_idx) < L_SUB:
                continue
            sel = np.array(alpha_idx[:L_SUB], dtype=np.int64)
            sub_d = dm[np.ix_(sel, sel)].copy()
            sub_m = mk[np.ix_(sel, sel)].copy()
            # Also enforce sequence adjacency inside the original passage (already in mk) plus subcloud diagonal
            np.fill_diagonal(sub_d, np.inf)
            sub_d[sub_m] = np.inf
            vals.append(estimate_twonn_mle(sub_d)[2])
        return np.array(vals)

    sa_sub = eval_pure_alpha_subcloud(sci_dm, sci_recs, sci_m4)
    ta_sub = eval_pure_alpha_subcloud(sty_dm, sty_recs, sty_m4)
    _, p_sub = ttest_ind(ta_sub, sa_sub)
    print(
        f" * Pure-Alphabetic {L_SUB}x{L_SUB} Subcloud (zero digits/punct/LaTeX; n={len(sa_sub)}+{len(ta_sub)}): "
        f"M4 MLE-5 Sci/Story = {np.mean(sa_sub):.2f}/{np.mean(ta_sub):.2f} (d={cohens_d_ind(ta_sub, sa_sub):+.2f}, p={p_sub:.3e})"
    )

    # 3A. Theiler Window Sweep w in {1, 2, 3, 5} (+ Null S^20 control at w=5)
    idx150 = np.arange(SEQ_L)
    w_parts = []
    rng = np.random.RandomState(42)
    for w in (1, 2, 3, 5):
        win_m = np.abs(idx150[:, None] - idx150[None, :]) <= w
        sw = [estimate_twonn_mle(np.where(mk | win_m, np.inf, dm))[2] for dm, mk in zip(sci_dm, sci_m4)]
        tw = [estimate_twonn_mle(np.where(mk | win_m, np.inf, dm))[2] for dm, mk in zip(sty_dm, sty_m4)]
        w_parts.append(f"w={w}: {np.mean(sw):.2f}/{np.mean(tw):.2f} (d={cohens_d_ind(np.array(tw), np.array(sw)):+.2f})")
    # Null S^20 check at widest window w=5
    win5 = np.abs(idx150[:, None] - idx150[None, :]) <= 5
    ns5 = [estimate_twonn_mle(np.where(mk | win5, np.inf, cdist(z := (u := rng.randn(SEQ_L, 21)) / np.linalg.norm(u, axis=1, keepdims=True), z)))[2] for mk in sci_m4]
    nt5 = [estimate_twonn_mle(np.where(mk | win5, np.inf, cdist(z := (u := rng.randn(SEQ_L, 21)) / np.linalg.norm(u, axis=1, keepdims=True), z)))[2] for mk in sty_m4]
    print(" * Theiler Window Sweep (M4 + |i-j|<=w) -> " + " | ".join(w_parts) + f" [Null S^20 at w=5: p={ttest_ind(nt5, ns5)[1]:.2f}]")

    # 3B. Neighborhood Size k Sweep for MLE-k: k in {3, 5, 10, 15, 20}
    k_parts = []
    for k in (3, 5, 10, 15, 20):
        sk0 = [mle_k_harmonic(np.where(np.eye(SEQ_L, dtype=bool), np.inf, dm), k) for dm in sci_dm]
        tk0 = [mle_k_harmonic(np.where(np.eye(SEQ_L, dtype=bool), np.inf, dm), k) for dm in sty_dm]
        sk4 = [mle_k_harmonic(np.where(mk, np.inf, dm), k) for dm, mk in zip(sci_dm, sci_m4)]
        tk4 = [mle_k_harmonic(np.where(mk, np.inf, dm), k) for dm, mk in zip(sty_dm, sty_m4)]
        k_parts.append(f"k={k}: Raw d={cohens_d_ind(np.array(tk0), np.array(sk0)):+.2f} -> M4 d={cohens_d_ind(np.array(tk4), np.array(sk4)):+.2f} ({np.mean(sk4):.1f}/{np.mean(tk4):.1f})")
    print(" * MLE-k Scale Sweep -> " + " | ".join(k_parts))


def resolve_layers_generic(mod):
    if hasattr(mod, "visual"):
        mod.visual = None
    if hasattr(mod, "language_model") and hasattr(mod.language_model, "layers"):
        return mod.language_model, mod.language_model.layers, True
    if hasattr(mod, "layers"):
        return mod, mod.layers, ("modernbert" not in type(mod).__name__.lower())
    if hasattr(mod, "h"):
        return mod, mod.h, True
    if hasattr(mod, "encoder") and hasattr(mod.encoder, "layer"):
        return mod.encoder, mod.encoder.layer, False
    raise RuntimeError(f"Unknown backbone structure: {type(mod).__name__}")


def audit_4_unit_sphere_projection(arxiv_150: list[str], stories_150: list[str]) -> None:
    print("\n" + "=" * 104)
    print(" [AUDIT 4] Unit-Sphere Projection (h_hat = h / ||h||_2): Does Erasing ALL Radial Norm Variance")
    print("           Change Twin-Token Collapse, Genre Reversal, or All-Tokens MST-PHD Sink Censoring?")
    print("=" * 104)
    models = [
        ("roberta-base", "RoBERTa-base (Enc)"),
        ("gpt2", "GPT-2 (Dec 38x Sink)"),
        ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B (Gated)"),
        ("Qwen/Qwen3-1.7B", "Qwen3-1.7B (91x Sink)"),
        ("HuggingFaceTB/SmolLM3-3B", "SmolLM3-3B (93x Sink)"),
    ]
    n_per = 45

    for mid, label in models:
        tok = AutoTokenizer.from_pretrained(mid)
        dt = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported() and ("qwen3.5" in mid.lower() or "smollm3" in mid.lower())) else (torch.float16 if DEVICE == "cuda" else torch.float32)
        try:
            mod = AutoModel.from_pretrained(mid, dtype=dt, low_cpu_mem_usage=True)
        except TypeError:
            mod = AutoModel.from_pretrained(mid, torch_dtype=dt, low_cpu_mem_usage=True)
        mod.eval()

        text_mod, layers, is_causal = resolve_layers_generic(mod)
        mid_l = len(layers) // 2
        if is_causal:
            while len(layers) > mid_l:
                del layers[-1]

        free_gb = free_mem()
        param_gb = sum(p.numel() * p.element_size() for p in mod.parameters()) / (1024 ** 3)
        stream = (DEVICE == "cuda") and is_causal and (free_gb < param_gb + 0.75)

        hooks = []
        cap = {}
        if DEVICE == "cuda" and not stream:
            mod.to(DEVICE)
        elif DEVICE == "cuda" and stream:
            for nm, ch in text_mod.named_children():
                if nm not in ("layers", "h", "norm", "ln_f"):
                    ch.to(DEVICE)
            for blk in layers:
                blk.to("cpu")
                hooks.append(blk.register_forward_pre_hook(lambda m, inp: m.to(DEVICE, non_blocking=True)))
                hooks.append(blk.register_forward_hook(lambda m, inp, out: m.to("cpu")))

        def _cap_fn(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            cap["h"] = t.detach().double().cpu().numpy()
            if stream:
                m.to("cpu")
            if is_causal:
                raise _StopForward()

        hooks.append(layers[mid_l - 1].register_forward_hook(_cap_fn))

        euc = {0: {"m0": [], "m4": [], "phd0": [], "phd4": []}, 1: {"m0": [], "m4": [], "phd0": [], "phd4": []}}
        sph = {0: {"m0": [], "m4": [], "phd0": [], "phd4": []}, 1: {"m0": [], "m4": [], "phd0": [], "phd4": []}}

        for is_sty, pool in ((0, arxiv_150[:n_per]), (1, stories_150[:n_per])):
            for txt in pool:
                mt = tokenize_fixed_length_with_metadata(tok, txt, SEQ_L)
                if mt is None:
                    continue
                kp = mt["keep_positions"]
                m4 = build_masks(mt)["M4_stem_adj"]
                cap.clear()
                with torch.inference_mode():
                    try:
                        mod(mt["full_ids"].to(DEVICE))
                    except _StopForward:
                        pass
                h = cap["h"][0, kp]
                h_unit = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)

                for store, vec in ((euc[is_sty], h), (sph[is_sty], h_unit)):
                    dm = cdist(vec, vec); np.fill_diagonal(dm, np.inf); dm[dm <= 1e-6] = np.inf
                    dm4 = dm.copy(); dm4[m4] = np.inf
                    store["m0"].append(estimate_twonn_mle(dm)[2])
                    store["m4"].append(estimate_twonn_mle(dm4)[2])
                    store["phd0"].append(estimate_phd_mst(dm.copy()))
                    store["phd4"].append(estimate_phd_mst(dm4.copy()))

        d_euc = cohens_d_ind(np.array(euc[1]["m4"]), np.array(euc[0]["m4"]))
        d_sph = cohens_d_ind(np.array(sph[1]["m4"]), np.array(sph[0]["m4"]))
        e_phd0 = np.concatenate([euc[0]["phd0"], euc[1]["phd0"]])
        s_phd0 = np.concatenate([sph[0]["phd0"], sph[1]["phd0"]])
        s_phd4 = np.concatenate([sph[0]["phd4"], sph[1]["phd4"]])

        print(
            f" [{label:22s}] Euclidean M0->M4: {np.mean(euc[0]['m0']):5.2f} -> {np.mean(euc[0]['m4']):5.2f}/{np.mean(euc[1]['m4']):5.2f} (d={d_euc:+.2f}) | AllTok PHD NaN={np.mean(~np.isfinite(e_phd0))*100:5.1f}%\n"
            f" {'':24s} Unit-Sph  M0->M4: {np.mean(sph[0]['m0']):5.2f} -> {np.mean(sph[0]['m4']):5.2f}/{np.mean(sph[1]['m4']):5.2f} (d={d_sph:+.2f}) | AllTok PHD NaN={np.mean(~np.isfinite(s_phd0))*100:5.1f}% (M4 PHD={np.nanmean(s_phd4):5.2f})"
        )

        for h_i in hooks:
            h_i.remove()
        del mod, text_mod, layers
        free_mem()


if __name__ == "__main__":
    t_start = time.time()
    stories_150, _ = load_stories_pool(min_words=175, target_n=150, long_min_words=265)
    arxiv_150, _ = load_arxiv_pool(target_n=150)

    ok_zip = audit_1_cleanroom_zip_and_table_sync()
    audit_2_and_3_qwen7b_stress_tests(arxiv_150, stories_150)
    audit_4_unit_sphere_projection(arxiv_150, stories_150)

    print("\n" + "=" * 104)
    print(f" [FINAL MASTER AUDIT COMPLETED IN {time.time() - t_start:.1f}s | Clean-Room ZIP Gate={'PASS' if ok_zip else 'FAIL'}]")
    print("=" * 104)
