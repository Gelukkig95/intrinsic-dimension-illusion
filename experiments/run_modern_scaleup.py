import argparse
import csv
import gc
import importlib.util
import json
import math
import random
import re
import shutil
import socket
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind, ttest_rel

# Force IPv4 to prevent Windows IPv6 socket timeouts
_orig_getaddrinfo = socket.getaddrinfo
def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    res = _orig_getaddrinfo(host, port, family, type, proto, flags)
    v4 = [r for r in res if r[0] == socket.AF_INET]
    return v4 if v4 else res
socket.getaddrinfo = _ipv4_only

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
DATA_CACHE = THIS_DIR / "cache_scaleup"
DATA_CACHE.mkdir(parents=True, exist_ok=True)

from verify_idea1_all_4 import (
    build_masks,
    cohens_d_ind,
    estimate_phd_mst,
    estimate_twonn_mle,
    set_seed,
    tokenize_fixed_length_with_metadata,
)


def prewarm_cuda_cublas() -> None:
    """Initializes cuBLAS handle & reserves caching allocator workspace BEFORE large safetensors mmap."""
    if not torch.cuda.is_available():
        return
    a = torch.ones((256, 256), device="cuda", dtype=torch.float16)
    _ = a @ a
    reserve = torch.empty((48 * 1024 * 1024,), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    del a, reserve


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos, neg = (labels == 1), (labels == 0)
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
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


def bootstrap_diff_ci(story_vals: np.ndarray, sci_vals: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    a = story_vals[np.isfinite(story_vals)]
    b = sci_vals[np.isfinite(sci_vals)]
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    ia = rng.randint(0, len(a), size=(n_boot, len(a)))
    ib = rng.randint(0, len(b), size=(n_boot, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def greedy_caliper_match(ttr_a: np.ndarray, ttr_b: np.ndarray, caliper: float = 0.015) -> tuple[np.ndarray, np.ndarray]:
    used_b = np.zeros(len(ttr_b), dtype=bool)
    ia_list, ib_list = [], []
    order_a = np.argsort(np.abs(ttr_a - np.median(ttr_b)))
    for ia in order_a:
        diffs = np.where(used_b, np.inf, np.abs(ttr_b - ttr_a[ia]))
        jb = int(np.argmin(diffs))
        if diffs[jb] <= caliper:
            used_b[jb] = True
            ia_list.append(ia)
            ib_list.append(jb)
    return np.asarray(ia_list, dtype=np.int64), np.asarray(ib_list, dtype=np.int64)


def load_stories_pool(min_words: int = 175, target_n: int = 160, long_min_words: int = 265) -> tuple[list[str], list[str]]:
    stories_std, stories_long = [], []
    for base in (Path("./data"), Path("./lang_geometry"), Path(".")):
        if not base.exists():
            continue
        found = list(base.rglob("hcV3-stories.csv"))
        if found:
            with found[0].open("r", encoding="utf-8", errors="ignore", newline="") as f:
                for row in csv.DictReader(f):
                    s = re.sub(r"\s+", " ", (row.get("story") or "").strip())
                    wc = len(s.split())
                    if wc >= min_words and len(stories_std) < target_n:
                        stories_std.append(s)
                    if wc >= long_min_words and len(stories_long) < 70:
                        stories_long.append(s)
                    if len(stories_std) >= target_n and len(stories_long) >= 70:
                        break
            break
    if len(stories_std) < 100:
        raise RuntimeError(f"Expected >=100 Hippocorpus stories, found {len(stories_std)}.")
    return stories_std, stories_long


def load_arxiv_pool(target_n: int = 150) -> tuple[list[str], list[str]]:
    cache_p = DATA_CACHE / "arxiv_natural_pool_200.json"
    saved = json.loads(cache_p.read_text(encoding="utf-8"))
    return saved["std"][:target_n], saved["long"][:70]


def load_hc3_chatgpt_pool(target_pairs: int = 80) -> tuple[list[str], list[str]]:
    cache_p = DATA_CACHE / "hc3_natural_pairs_80.json"
    saved = json.loads(cache_p.read_text(encoding="utf-8"))
    return saved["human"][:target_pairs], saved["ai"][:target_pairs]


def hf_cache_folder_for(model_id: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model_id.replace("/", "--"))


def load_modern_backbone(candidates: list[str], size_class: str, device: str, probe_text: str):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

    # Always prefer float16 (CUDA_R_16F) over bfloat16 (CUDA_R_16BF) on Windows WDDM cuBLAS
    use_dt = torch.float32 if (device == "cpu" or size_class == "small_encoder") else torch.float16
    has_accel = importlib.util.find_spec("accelerate") is not None

    last_err = None
    for mid in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            cfg = AutoConfig.from_pretrained(mid)
            if hasattr(cfg, "reference_compile"):
                cfg.reference_compile = False
            if hasattr(cfg, "use_cache"):
                cfg.use_cache = False

            load_kw = {"config": cfg}
            if has_accel:
                load_kw["low_cpu_mem_usage"] = True
            if size_class == "large_3b" and device == "cuda" and has_accel:
                load_kw["device_map"] = "auto"
                load_kw["max_memory"] = {0: "4200MiB", "cpu": "10GiB"}

            model = None
            for attn_kw in ({"attn_implementation": "eager"}, {}):
                for dt_kw in ({"dtype": use_dt}, {"torch_dtype": use_dt}):
                    try:
                        model = AutoModel.from_pretrained(mid, **load_kw, **attn_kw, **dt_kw)
                        break
                    except TypeError:
                        continue
                    except Exception as e_inner:
                        try:
                            model = AutoModelForCausalLM.from_pretrained(mid, **load_kw, **attn_kw, **dt_kw)
                            break
                        except Exception:
                            last_err = e_inner
                if model is not None:
                    break

            if model is None:
                continue

            gc.collect()
            run_dev = device
            if "device_map" not in load_kw:
                try:
                    model = model.to(run_dev)
                except RuntimeError:
                    print("   [Fallback] Windows commit tight for GPU move; running cleanly in CPU float32.")
                    run_dev = "cpu"
                    model = model.float().to("cpu")

            model.eval()

            # Verify 150-token forward pass & finite hidden states right now (auto-fallback if fp16 overflows or WDDM denies cuBLAS)
            p_meta = tokenize_fixed_length_with_metadata(tok, probe_text, target_len=150)
            if p_meta is not None and run_dev == "cuda":
                try:
                    inp_d = next(model.parameters()).device
                    ids = p_meta["full_ids"].to(inp_d)
                    with torch.inference_mode():
                        hs = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False).hidden_states
                        h_chk = hs[len(hs) // 2][0, p_meta["keep_positions"]].float()
                        finite_ok = bool(torch.isfinite(h_chk).all().item())
                        del hs, h_chk
                    if not finite_ok:
                        raise RuntimeError("FP16 activation overflow detected; switching to exact FP32.")
                except RuntimeError as cuda_err:
                    print(f"   [Safe Fallback] {type(cuda_err).__name__} on CUDA ({str(cuda_err)[:60]}...) -> switching to exact CPU float32.")
                    model = model.to("cpu").float()
                    gc.collect()
                    torch.cuda.empty_cache()
                    run_dev = "cpu"

            return tok, model, mid, run_dev
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load {candidates}: {last_err}")


def eval_texts_at_length(tok, model, texts: list[str], run_device: str, target_len: int = 150, with_phd: bool = True) -> dict[str, np.ndarray]:
    out = {"ttr": [], "twonn_std": [], "twonn_msk": [], "mle_std": [], "mle_msk": [], "phd_std": [], "phd_msk": []}
    inp_dev = next(model.parameters()).device if run_device != "cpu" else torch.device("cpu")

    for idx, t in enumerate(texts):
        meta = tokenize_fixed_length_with_metadata(tok, t, target_len=target_len)
        if meta is None:
            continue
        m4 = build_masks(meta)["M4_stem_adj"]
        input_ids = meta["full_ids"].to(inp_dev)
        attn_mask = torch.ones_like(input_ids)

        try:
            with torch.inference_mode():
                hs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    use_cache=False,
                ).hidden_states
                mid_l = len(hs) // 2
                h = np.ascontiguousarray(hs[mid_l][0, meta["keep_positions"]].float().cpu().numpy(), dtype=np.float64)
                del hs
        except RuntimeError:
            # Mid-run WDDM commit spike protection: move model cleanly to CPU float32 without losing progress
            model.to("cpu").float()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            inp_dev = torch.device("cpu")
            with torch.inference_mode():
                hs = model(
                    input_ids=meta["full_ids"],
                    attention_mask=torch.ones_like(meta["full_ids"]),
                    output_hidden_states=True,
                    use_cache=False,
                ).hidden_states
                mid_l = len(hs) // 2
                h = np.ascontiguousarray(hs[mid_l][0, meta["keep_positions"]].float().cpu().numpy(), dtype=np.float64)
                del hs

        if not np.all(np.isfinite(h)):
            continue

        dmat = cdist(h, h, metric="euclidean")
        np.fill_diagonal(dmat, np.inf)
        d_std = dmat.copy()
        d_std[d_std <= 1e-6] = np.inf
        d_msk = d_std.copy()
        d_msk[m4] = np.inf

        f_s, _, m_s = estimate_twonn_mle(d_std)
        f_m, _, m_m = estimate_twonn_mle(d_msk)
        out["ttr"].append(meta["ttr_stem"])
        out["twonn_std"].append(f_s)
        out["twonn_msk"].append(f_m)
        out["mle_std"].append(m_s)
        out["mle_msk"].append(m_m)

        if with_phd:
            min_pts = max(35, min(45, target_len // 2))
            out["phd_std"].append(estimate_phd_mst(d_std, min_points=min_pts, n_reruns=4, seed=700 + idx))
            out["phd_msk"].append(estimate_phd_mst(d_msk, min_points=min_pts, n_reruns=4, seed=700 + idx))

    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-3b", action="store_true")
    parser.add_argument("--ephemeral-cache", action="store_true")
    args = parser.parse_args()

    t_start = time.perf_counter()
    set_seed(2026)
    prewarm_cuda_cublas()
    base_device = "cuda" if torch.cuda.is_available() else "cpu"
    free_gb = shutil.disk_usage(THIS_DIR).free / (1024**3)
    print(f"=== [Resuming Scale-Up Suite (FP16 + WDDM-Safe) | GPU: {torch.cuda.get_device_name(0) if base_device=='cuda' else 'CPU'} | Disk Free={free_gb:.1f} GB] ===")

    stories_150, stories_long = load_stories_pool(min_words=175, target_n=150, long_min_words=265)
    arxiv_150, arxiv_long = load_arxiv_pool(target_n=150)
    hc3_hum, hc3_ai = load_hc3_chatgpt_pool(target_pairs=80)

    model_specs = [
        ("Modern RoPE Encoder (ModernBERT-base, 149M)", ["answerdotai/ModernBERT-base"], "small_encoder"),
        ("AI2 Fully-Open Decoder (OLMo-2-1B)", ["allenai/OLMo-2-0425-1B"], "mid_decoder"),
        ("Llama-Arch 11T Decoder (SmolLM2-1.7B)", ["HuggingFaceTB/SmolLM2-1.7B-Instruct"], "mid_decoder"),
        ("Latest Qwen3 Decoder (Qwen3-1.7B / Qwen2.5-1.5B)", ["Qwen/Qwen3-1.7B", "Qwen/Qwen2.5-1.5B-Instruct"], "mid_decoder"),
    ]
    if args.include_3b:
        model_specs.append(("3B Scale-Up Decoder (Qwen2.5-3B)", ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct"], "large_3b"))

    saved_json_path = THIS_DIR / "modern_scaleup_results.json"
    all_saved = json.loads(saved_json_path.read_text(encoding="utf-8")) if saved_json_path.exists() else {}

    for label, cands, sz_cls in model_specs:
        done_id = next((c for c in cands if c in all_saved), None)
        if done_id is not None:
            print(f" * [Skip Completed] {done_id} already saved in {saved_json_path.name}")
            continue

        print("\n" + "=" * 98)
        print(f" [MODEL AUDIT] {label}")
        print("=" * 98)
        tok, model, used_id, used_dev = load_modern_backbone(cands, sz_cls, base_device, arxiv_150[0])

        try:
            sci = eval_texts_at_length(tok, model, arxiv_150, used_dev, target_len=150, with_phd=True)
            fic = eval_texts_at_length(tok, model, stories_150, used_dev, target_len=150, with_phd=True)
            hum = eval_texts_at_length(tok, model, hc3_hum, used_dev, target_len=150, with_phd=False)
            ai = eval_texts_at_length(tok, model, hc3_ai, used_dev, target_len=150, with_phd=False)

            ia, ib = greedy_caliper_match(sci["ttr"], fic["ttr"], caliper=0.015)
            print(
                f" * Active ID: {used_id} ({used_dev.upper()}) | Evaluated n_sci={len(sci['ttr'])}, n_story={len(fic['ttr'])} | "
                f"TTR-Matched Pairs (|dTTR|<=1.5%p) = {len(ia)}"
            )

            model_rec = {"model_id": used_id, "n_sci": int(len(sci["ttr"])), "n_story": int(len(fic["ttr"])), "n_matched": int(len(ia)), "metrics": {}}
            for m_lbl, ks, km in (("TwoNN", "twonn_std", "twonn_msk"), ("MLE-5", "mle_std", "mle_msk"), ("MST-PHD", "phd_std", "phd_msk")):
                d_bef = cohens_d_ind(fic[ks], sci[ks])
                d_aft = cohens_d_ind(fic[km], sci[km])
                ci_lo, ci_hi = bootstrap_diff_ci(fic[km], sci[km])
                _, p_aft = ttest_ind(fic[km], sci[km], nan_policy="omit")
                d_mat = cohens_d_ind(fic[km][ib], sci[km][ia]) if len(ia) >= 8 else float("nan")
                _, p_mat = ttest_rel(fic[km][ib], sci[km][ia], nan_policy="omit") if len(ia) >= 8 else (0.0, float("nan"))

                print(
                    f"   - {m_lbl:7s} Full (N={len(sci['ttr'])+len(fic['ttr'])}) : "
                    f"BEFORE Sci={np.nanmean(sci[ks]):5.2f}, Story={np.nanmean(fic[ks]):5.2f} (d={d_bef:+.2f}) -> "
                    f"AFTER Sci={np.nanmean(sci[km]):5.2f}, Story={np.nanmean(fic[km]):5.2f} "
                    f"(Gap={np.nanmean(fic[km])-np.nanmean(sci[km]):+5.2f} [{ci_lo:+.2f},{ci_hi:+.2f}], d={d_aft:+.2f}, p={p_aft:.4f})\n"
                    f"     {m_lbl:7s} TTR-Matched (n={len(ia):2d} pairs): "
                    f"AFTER Sci={np.nanmean(sci[km][ia]):5.2f} vs Story={np.nanmean(fic[km][ib]):5.2f} (d={d_mat:+.2f}, paired p={p_mat:.4f})"
                )
                model_rec["metrics"][m_lbl] = {
                    "sci_before": float(np.nanmean(sci[ks])),
                    "story_before": float(np.nanmean(fic[ks])),
                    "d_before": float(d_bef),
                    "sci_after": float(np.nanmean(sci[km])),
                    "story_after": float(np.nanmean(fic[km])),
                    "d_after": float(d_aft),
                    "p_after": float(p_aft),
                    "d_matched": float(d_mat),
                    "p_matched": float(p_mat),
                }

            y_hc = np.array([1] * len(hum["ttr"]) + [0] * len(ai["ttr"]), dtype=np.int64)
            ih, ic = greedy_caliper_match(hum["ttr"], ai["ttr"], caliper=0.015)
            y_hm = np.array([1] * len(ih) + [0] * len(ic), dtype=np.int64)
            au_tw_s = compute_auroc(y_hc, np.nan_to_num(np.concatenate([hum["twonn_std"], ai["twonn_std"]]))) * 100.0
            au_tw_m = compute_auroc(y_hc, np.nan_to_num(np.concatenate([hum["twonn_msk"], ai["twonn_msk"]]))) * 100.0
            aum_tw_s = compute_auroc(y_hm, np.nan_to_num(np.concatenate([hum["twonn_std"][ih], ai["twonn_std"][ic]]))) * 100.0 if len(ih) >= 8 else float("nan")
            aum_tw_m = compute_auroc(y_hm, np.nan_to_num(np.concatenate([hum["twonn_msk"][ih], ai["twonn_msk"][ic]]))) * 100.0 if len(ih) >= 8 else float("nan")
            print(
                f"   - [HC3 Real ChatGPT Check] TwoNN Full AUROC: Std={au_tw_s:5.2f}% -> Msk={au_tw_m:5.2f}% | "
                f"TTR-Matched ({len(ih)} pairs) AUROC: Std={aum_tw_s:5.2f}% -> Msk={aum_tw_m:5.2f}%"
            )

            if "Qwen" in used_id:
                print(f"   - [Sequence-Length Sweep on Identical Long Documents (n_long_sci={len(arxiv_long)}, n_long_story={len(stories_long)})]")
                sweep_rec = {}
                for seq_l in (80, 150, 220, 300):
                    sl_sci = eval_texts_at_length(tok, model, arxiv_long, used_dev, target_len=seq_l, with_phd=False)
                    sl_fic = eval_texts_at_length(tok, model, stories_long, used_dev, target_len=seq_l, with_phd=False)
                    if len(sl_sci["ttr"]) >= 20 and len(sl_fic["ttr"]) >= 20:
                        db = cohens_d_ind(sl_fic["twonn_std"], sl_sci["twonn_std"])
                        da = cohens_d_ind(sl_fic["twonn_msk"], sl_sci["twonn_msk"])
                        _, pa = ttest_ind(sl_fic["twonn_msk"], sl_sci["twonn_msk"], nan_policy="omit")
                        print(
                            f"     L={seq_l:3d} tokens (n={len(sl_sci['ttr']):2d}/{len(sl_fic['ttr']):2d}) | "
                            f"Mean TTR Sci={np.mean(sl_sci['ttr'])*100:4.1f}%/Story={np.mean(sl_fic['ttr'])*100:4.1f}% | "
                            f"TwoNN Std: {np.nanmean(sl_sci['twonn_std']):5.2f} vs {np.nanmean(sl_fic['twonn_std']):5.2f} (d={db:+.2f}) -> "
                            f"Masked: {np.nanmean(sl_sci['twonn_msk']):5.2f} vs {np.nanmean(sl_fic['twonn_msk']):5.2f} (d={da:+.2f}, p={pa:.4f})"
                        )
                        sweep_rec[str(seq_l)] = {"d_before": float(db), "d_after": float(da), "p_after": float(pa)}
                model_rec["length_sweep"] = sweep_rec

            all_saved[used_id] = model_rec
            saved_json_path.write_text(json.dumps(all_saved, indent=2), encoding="utf-8")
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if args.ephemeral_cache and (used_id in all_saved):
                cdir = hf_cache_folder_for(used_id)
                if cdir.exists():
                    shutil.rmtree(cdir, ignore_errors=True)
                    print(f"   * [Ephemeral Cache] Freed disk cache for {used_id}")

    print(f"\n=== [Scale-Up Suite Completed in {time.perf_counter() - t_start:.1f}s | Saved to {saved_json_path.name}] ===")


if __name__ == "__main__":
    main()
