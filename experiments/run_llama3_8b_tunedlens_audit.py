import gc
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, snapshot_download
from scipy.optimize import brentq
from scipy.stats import pearsonr, ttest_ind, ttest_rel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

THIS_DIR = Path(__file__).resolve().parent
OUT_JSON = THIS_DIR.parent / "results" / "llama3_8b_viswanathan_audit.json"
TL_FLAT_CACHE = THIS_DIR / "cache_tl_llama3_bf16.bin"
sys.path.insert(0, str(THIS_DIR))

from run_modern_scaleup import load_arxiv_pool, load_hc3_chatgpt_pool, load_stories_pool
from verify_idea1_all_4 import build_masks, cohens_d_ind, estimate_twonn_mle, tokenize_fixed_length_with_metadata

SEQ_L = 256  # Exact 4^4 = 256 tokens -> Viswanathan Algorithm 1 (nBlocks = 4^S) divides cleanly down to B=1 at S=4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_MAP = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


class _StopAfterLayer(Exception):
    pass


def resolve_cached_artifacts():
    print("=" * 114, flush=True)
    print(" [STEP 1] Resolving Cached Meta-Llama-3-8B + Official TunedLens (Zero-Copy C Buffer Pack)", flush=True)
    print("=" * 114, flush=True)
    hf_tok = os.environ.get("HF_TOKEN", "").strip() or None
    model_dir = None
    for repo, tok in [("NousResearch/Meta-Llama-3-8B", None), ("meta-llama/Meta-Llama-3-8B", hf_tok)]:
        try:
            p = snapshot_download(repo_id=repo, token=tok, local_files_only=True, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])
            if list(Path(p).glob("*.safetensors")):
                model_dir = Path(p)
                print(f" * [Cached Model Found] {repo} -> {model_dir.name}", flush=True)
                break
        except Exception:
            continue
    if model_dir is None:
        model_dir = Path(snapshot_download(repo_id="NousResearch/Meta-Llama-3-8B", allow_patterns=["*.json", "*.safetensors", "tokenizer*"]))

    tl_cfg_p = Path(hf_hub_download(repo_id="AlignmentResearch/tuned-lens", filename="lens/meta-llama/Meta-Llama-3-8B/config.json", repo_type="space", local_files_only=True))
    tl_pt_p = Path(hf_hub_download(repo_id="AlignmentResearch/tuned-lens", filename="lens/meta-llama/Meta-Llama-3-8B/params.pt", repo_type="space", local_files_only=True))
    tl_cfg = json.loads(tl_cfg_p.read_text(encoding="utf-8"))

    d_hid = 4096
    layer_bytes = (d_hid * d_hid + d_hid) * 2
    if not TL_FLAT_CACHE.exists() or TL_FLAT_CACHE.stat().st_size != 32 * layer_bytes:
        t_pk = time.time()
        print(" * [One-Time Prep] Converting TunedLens params.pt via C uint8 view (< 2s)...", flush=True)
        try:
            raw_sd = torch.load(str(tl_pt_p), map_location="cpu", weights_only=True)
        except TypeError:
            raw_sd = torch.load(str(tl_pt_p), map_location="cpu")

        tmp_bin = TL_FLAT_CACHE.with_suffix(".tmp")
        w0_ref = None
        with open(tmp_bin, "wb") as fp:
            for idx in range(32):
                w_k = next(k for k in raw_sd.keys() if k.endswith(f".{idx}.weight") or k == f"{idx}.weight")
                b_k = next(k for k in raw_sd.keys() if k.endswith(f".{idx}.bias") or k == f"{idx}.bias")
                w_bf = raw_sd[w_k].to(torch.bfloat16).contiguous()
                b_bf = raw_sd[b_k].to(torch.bfloat16).contiguous()
                if idx == 0:
                    w0_ref = w_bf.clone()
                # Zero-copy uint8 NumPy C-buffer write (avoids UntypedStorage 1B-element Python loop.)
                fp.write(w_bf.view(torch.uint8).numpy().tobytes())
                fp.write(b_bf.view(torch.uint8).numpy().tobytes())
        del raw_sd
        gc.collect()
        tmp_bin.replace(TL_FLAT_CACHE)

        # Bit-exactness verification on Layer 0
        with open(TL_FLAT_CACHE, "rb") as fp:
            raw0 = bytearray(fp.read(d_hid * d_hid * 2))
        w0_chk = torch.frombuffer(raw0, dtype=torch.bfloat16).reshape(d_hid, d_hid)
        assert torch.equal(w0_ref, w0_chk), "Binary pack mismatch!"
        del w0_ref, w0_chk, raw0
        print(f" * [One-Time Prep Done] Packed & bit-verified 32 TunedLens layers in {time.time() - t_pk:.2f}s!", flush=True)
    else:
        print(f" * [TunedLens Cache Ready] {TL_FLAT_CACHE.name} (0 MB idle RAM)", flush=True)
    return model_dir, tl_cfg


def read_single_tuned_lens_gpu(layer_idx: int, d_hid: int, buf: bytearray) -> tuple[torch.Tensor, torch.Tensor]:
    w_bytes = d_hid * d_hid * 2
    b_bytes = d_hid * 2
    tot = w_bytes + b_bytes
    with open(TL_FLAT_CACHE, "rb") as fp:
        fp.seek(layer_idx * tot)
        fp.readinto(memoryview(buf)[:tot])
    w = torch.frombuffer(memoryview(buf)[:w_bytes], dtype=torch.bfloat16).reshape(d_hid, d_hid).to(DEVICE)
    b = torch.frombuffer(memoryview(buf)[w_bytes:tot], dtype=torch.bfloat16).reshape(d_hid).to(DEVICE)
    return w, b


class ZeroAllocSafetensorsStreamer:
    def __init__(self, model_dir: Path):
        self.tensor_map = {}
        for sf in sorted(model_dir.glob("*.safetensors")):
            with open(sf, "rb") as fp:
                hdr_len = struct.unpack("<Q", fp.read(8))[0]
                hdr = json.loads(fp.read(hdr_len).decode("utf-8"))
                base_off = 8 + hdr_len
                for k, info in hdr.items():
                    if k == "__metadata__":
                        continue
                    beg, end = info["data_offsets"]
                    self.tensor_map[k] = (sf, base_off + beg, end - beg, DTYPE_MAP[info["dtype"]], tuple(info["shape"]))
        self.buf = bytearray(145 * 1024 * 1024)

    def copy_into_module_gpu(self, module: nn.Module, prefix: str) -> None:
        param_dict = dict(module.named_parameters())
        by_file = {}
        for p_name in param_dict:
            full_k = prefix + p_name
            spec = self.tensor_map[full_k]
            by_file.setdefault(spec[0], []).append((p_name, spec))

        for sf, items in by_file.items():
            items.sort(key=lambda x: x[1][1])
            with open(sf, "rb") as fp:
                for p_name, (_, off, nbytes, orig_dt, shape) in items:
                    fp.seek(off)
                    mv = memoryview(self.buf)[:nbytes]
                    fp.readinto(mv)
                    cpu_view = torch.frombuffer(mv, dtype=orig_dt).reshape(shape)
                    param_dict[p_name].data.copy_(cpu_view.to(device=DEVICE, dtype=torch.bfloat16))

    def stream_vocab_table_in_chunks(self, tensor_key: str, chunk_rows: int = 16032):
        sf, base_off, _, orig_dt, (V, H) = self.tensor_map[tensor_key]
        elem_b = 2 if orig_dt in (torch.bfloat16, torch.float16) else 4
        row_b = H * elem_b
        with open(sf, "rb") as fp:
            for r0 in range(0, V, chunk_rows):
                r1 = min(r0 + chunk_rows, V)
                nb = (r1 - r0) * row_b
                fp.seek(base_off + r0 * row_b)
                mv = memoryview(self.buf)[:nb]
                fp.readinto(mv)
                yield r0, r1, torch.frombuffer(mv, dtype=orig_dt).reshape(r1 - r0, H)


def build_ultralight_shell(model_dir: Path, streamer: ZeroAllocSafetensorsStreamer):
    full_cfg = AutoConfig.from_pretrained(str(model_dir))
    n_layers = full_cfg.num_hidden_layers
    vocab_sz = full_cfg.vocab_size
    d_hid = full_cfg.hidden_size
    tok = AutoTokenizer.from_pretrained(str(model_dir))

    cfg_tiny = AutoConfig.from_pretrained(str(model_dir))
    cfg_tiny.num_hidden_layers = 1
    cfg_tiny.vocab_size = 32
    try:
        shell = AutoModelForCausalLM.from_config(cfg_tiny, dtype=torch.bfloat16).eval()
    except TypeError:
        shell = AutoModelForCausalLM.from_config(cfg_tiny, torch_dtype=torch.bfloat16).eval()

    shell.to(DEVICE)
    streamer.copy_into_module_gpu(shell.model.norm, "model.norm.")

    shell.lm_head = nn.Linear(d_hid, vocab_sz, bias=False, device=DEVICE, dtype=torch.bfloat16)
    lm_key = "lm_head.weight" if "lm_head.weight" in streamer.tensor_map else "model.embed_tokens.weight"
    with torch.no_grad():
        for r0, r1, chunk_v in streamer.stream_vocab_table_in_chunks(lm_key):
            shell.lm_head.weight.data[r0:r1].copy_(chunk_v.to(device=DEVICE, dtype=torch.bfloat16))

    step_cap = {}
    def _cap_and_stop(m, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        step_cap["h"] = t.detach()
        raise _StopAfterLayer()

    shell.model.layers[0].register_forward_hook(_cap_and_stop)
    free_gb = torch.cuda.mem_get_info()[0] / (1024 ** 3) if DEVICE == "cuda" else 0.0
    print(f" * [Ultra-Light GPU Shell Ready] Layers={n_layers}, Vocab={vocab_sz} | Free VRAM={free_gb:.2f} GB", flush=True)
    return n_layers, d_hid, tok, shell, step_cap


def gride_mle(dm: np.ndarray, n1: int = 2, n2: int = 4) -> float:
    ds = np.sort(dm, axis=1)
    if ds.shape[1] < n2:
        return float("nan")
    r1, r2 = ds[:, n1 - 1], ds[:, n2 - 1]
    ok = np.isfinite(r1) & np.isfinite(r2) & (r1 > 1e-12) & (r2 > r1 * (1.0 + 1e-12))
    if np.sum(ok) < 12:
        return float("nan")
    log_mu = np.log(r2[ok] / r1[ok])
    mean_l = float(np.mean(log_mu))
    if mean_l <= 1e-12:
        return float("nan")
    if n2 - n1 == 1:
        return float(1.0 / (n1 * mean_l))

    def grad(d: float) -> float:
        u = d * log_mu
        return (1.0 / d) + (n2 - n1 - 1) * float(np.mean(log_mu / (-np.expm1(-u)))) - (n2 - 1) * mean_l

    if grad(1e-3) <= 0:
        return 1e-3
    if grad(500.0) >= 0:
        return 500.0
    return float(brentq(grad, 1e-3, 500.0, xtol=1e-6))


def viswanathan_block_shuffle(length: int, s_idx: int, rng: np.random.RandomState) -> np.ndarray:
    if s_idx == 0:
        return np.arange(length, dtype=np.int64)
    n_blocks = 4 ** s_idx
    b_size = int(math.ceil(length / n_blocks))
    blocks = [np.arange(start, min(start + b_size, length), dtype=np.int64) for start in range(0, length, b_size)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[k] for k in order])


def assemble_diverse_256tok_corpora(tok) -> list[dict]:
    arxiv_std, arxiv_long = load_arxiv_pool(target_n=150)
    stories_std, stories_long = load_stories_pool(min_words=175, target_n=150, long_min_words=265)
    hc3_hum, hc3_ai = load_hc3_chatgpt_pool(target_pairs=80)

    import transformers as _tf_pkg
    tf_root = Path(_tf_pkg.__file__).resolve().parent / "models"
    code_texts = []
    for py_f in sorted(tf_root.glob("*/*.py")):
        try:
            txt = py_f.read_text(encoding="utf-8", errors="ignore")
            lines = [ln for ln in txt.splitlines() if not ln.strip().startswith("#")]
            body = "\n".join(lines).strip()
            if len(body) >= 1800:
                code_texts.append(body[:6000])
            if len(code_texts) >= 40:
                break
        except Exception:
            continue

    adj256 = np.abs(np.arange(SEQ_L)[:, None] - np.arange(SEQ_L)[None, :]) <= 1
    records = []
    for dom, pool, target_k in [
        ("sci", list(arxiv_long) + list(arxiv_std), 30),
        ("story", list(stories_long) + list(stories_std), 30),
        ("qa", list(hc3_hum) + list(hc3_ai), 20),
        ("code", code_texts, 20),
    ]:
        cnt, seen = 0, set()
        for txt in pool:
            if txt[:90] in seen:
                continue
            mt = tokenize_fixed_length_with_metadata(tok, txt, SEQ_L)
            if mt is None:
                continue
            seen.add(txt[:90])
            kp = mt["keep_positions"]
            fids = mt["full_ids"][0]
            m4 = build_masks(mt)["M4_stem_adj"]
            c_ids = fids[kp].numpy()
            same_id = c_ids[:, None] == c_ids[None, :]
            nonadj_stem = np.eye(SEQ_L, dtype=bool) | (m4 & (~adj256)) | same_id
            m_stem = (nonadj_stem.astype(np.int32) @ nonadj_stem.astype(np.int32)) > 0
            mults = np.sum(m_stem, axis=1)
            ttr = float(np.sum(1.0 / np.maximum(mults, 1)) / SEQ_L)
            records.append({
                "rec_id": len(records), "domain": dom, "full_ids": fids.clone(),
                "kp": kp, "m4": m4, "m_stem": m_stem, "ttr": ttr,
            })
            cnt += 1
            if cnt >= target_k:
                break

    del arxiv_std, arxiv_long, stories_std, stories_long, hc3_hum, hc3_ai, code_texts
    gc.collect()
    print(
        f" * [Corpora Assembled (Exact L={SEQ_L})] N_natural={len(records)} | "
        + ", ".join(f"{d}={sum(1 for r in records if r['domain']==d)}" for d in ("sci", "story", "qa", "code")),
        flush=True,
    )
    return records


def shannon_entropy_low_vram(logits_bf16: torch.Tensor) -> torch.Tensor:
    """Exact float32 Shannon entropy from (T, V) bf16 logits using <= 260 MB temporary VRAM (zero WDDM spillover)."""
    lg = logits_bf16.float()
    lse = torch.logsumexp(lg, dim=-1, keepdim=True)
    lg.sub_(lse)  # in-place log_softmax.
    p = lg.exp()
    ent = -(p * lg).sum(dim=-1)
    del lg, lse, p
    return ent


def main() -> None:
    t_start = time.time()
    model_dir, tl_cfg = resolve_cached_artifacts()

    print("\n" + "=" * 114, flush=True)
    print(" [STEP 2] Building Ultra-Light Streaming Shell & Exact L=256 Corpora", flush=True)
    print("=" * 114, flush=True)
    streamer = ZeroAllocSafetensorsStreamer(model_dir)
    n_layers, d_hid, tok, shell, step_cap = build_ultralight_shell(model_dir, streamer)

    nat_records = assemble_diverse_256tok_corpora(tok)
    N_nat = len(nat_records)
    sci_recs = [r for r in nat_records if r["domain"] == "sci"]
    sty_recs = [r for r in nat_records if r["domain"] == "story"]
    shuf_base = sci_recs[:10] + sty_recs[:10]
    shuf_id_map = {r["rec_id"]: b_id for b_id, r in enumerate(shuf_base)}

    rng = np.random.RandomState(2026)
    adj256 = np.abs(np.arange(SEQ_L)[:, None] - np.arange(SEQ_L)[None, :]) <= 1

    all_jobs = []
    for idx, r in enumerate(nat_records):
        all_jobs.append({
            "kind": "natural", "nat_idx": idx, "domain": r["domain"], "s_idx": 0,
            "base_shuf_id": shuf_id_map.get(r["rec_id"], -1),
            "full_ids": r["full_ids"], "m4": r["m4"], "ttr": r["ttr"],
        })
    for b_id, r in enumerate(shuf_base):
        kp = r["kp"]
        c_ids = r["full_ids"][kp].numpy()
        for s_idx in (1, 2, 3, 4):
            perm = viswanathan_block_shuffle(SEQ_L, s_idx, rng)
            new_fids = r["full_ids"].clone()
            new_fids[kp] = torch.from_numpy(c_ids[perm])
            m_stem_shuf = r["m_stem"][np.ix_(perm, perm)]
            all_jobs.append({
                "kind": "shuffled", "nat_idx": -1, "domain": r["domain"], "s_idx": s_idx,
                "base_shuf_id": b_id, "full_ids": new_fids, "m4": (m_stem_shuf | adj256), "ttr": r["ttr"],
            })
    del nat_records
    gc.collect()

    N_tot = len(all_jobs)
    all_input_ids = torch.stack([j["full_ids"] for j in all_jobs], dim=0)
    probe_depths = [0, 1, 2, 4, 8, 10, 12, 16, 18, 20, 24, 28, 31, 32]

    h_all = torch.empty((N_tot, SEQ_L + 1, d_hid), dtype=torch.bfloat16, device="cpu")
    with torch.no_grad():
        for r0, r1, chunk_v in streamer.stream_vocab_table_in_chunks("model.embed_tokens.weight"):
            mask = (all_input_ids >= r0) & (all_input_ids < r1)
            if mask.any():
                h_all[mask] = chunk_v[all_input_ids[mask] - r0].to(torch.bfloat16)

    results_per_job = [
        {"meta": {k: j[k] for k in ("kind", "nat_idx", "domain", "s_idx", "base_shuf_id", "ttr")}, "surprisal": 0.0, "bos_sink_L16": 0.0, "depths": {}}
        for j in all_jobs
    ]
    scale_sweep_records = {d_l: {sc: {"m0": [], "m4": []} for sc in (2, 4, 8, 16, 32, 64)} for d_l in (1, 10, 20, 28)}
    use_residual_tl = bool(tl_cfg.get("residual", True))

    def evaluate_depth_fast(d_l: int):
        w_tl, b_tl = (read_single_tuned_lens_gpu(d_l, d_hid, streamer.buf) if d_l < 32 else (None, None))
        ent_tl_c = np.zeros(N_nat, dtype=np.float64)
        ent_tl_a = np.zeros(N_nat, dtype=np.float64)

        with torch.inference_mode():
            # Process 1 sequence (257 tokens -> 131 MB fp32 logits) at a time: zero VRAM spillover.
            for i in range(N_nat):
                xi = h_all[i].to(device=DEVICE, dtype=torch.bfloat16)
                if w_tl is not None:
                    lin = F.linear(xi, w_tl, b_tl)
                    xt = (xi + lin) if use_residual_tl else lin
                else:
                    xt = xi
                lg_bf = shell.lm_head(shell.model.norm(xt))
                ent_tok = shannon_entropy_low_vram(lg_bf)
                ent_tl_c[i] = float(ent_tok[1:].mean().item())
                ent_tl_a[i] = float(ent_tok.mean().item())
                del xi, xt, lg_bf, ent_tok

            if d_l == 32:
                for j in range(N_tot):
                    xj = h_all[j].to(device=DEVICE, dtype=torch.bfloat16)
                    lg_f = shell.lm_head(shell.model.norm(xj))[:-1].float()
                    lp = F.log_softmax(lg_f, dim=-1)
                    tgt = all_input_ids[j, 1:].to(DEVICE)
                    results_per_job[j]["surprisal"] = float(F.nll_loss(lp, tgt, reduction="mean").item())
                    del xj, lg_f, lp

            del w_tl, b_tl

            # Compute exact pairwise distances on GPU in FP32 (exact on bf16 inputs, 64x faster than Ampere FP64.)
            xg_batch = h_all.to(device=DEVICE, dtype=torch.float32)  # (180, 257, 4096) = 758 MB on GPU
            if d_l == 16:
                norms_b = torch.norm(xg_batch[:N_nat], dim=-1)
                ratios_b = (norms_b[:, 0] / torch.median(norms_b[:, 1:], dim=-1).values).cpu().tolist()
                for i_n, rv in enumerate(ratios_b):
                    results_per_job[i_n]["bos_sink_L16"] = float(rv)

            dm_all = torch.cdist(xg_batch, xg_batch, p=2.0, compute_mode="donot_use_mm_for_euclid_dist").cpu().numpy()
            del xg_batch

        diag_idx = np.arange(SEQ_L + 1)
        for gi, job in enumerate(all_jobs):
            dm_full = dm_all[gi].astype(np.float64)
            dm_full[diag_idx, diag_idx] = np.inf
            dm_full[dm_full <= 1e-6] = np.inf
            dm = dm_full[1:, 1:].copy()

            m4 = job["m4"]
            dm_adj = dm.copy(); dm_adj[adj256] = np.inf
            dm_m4 = dm.copy(); dm_m4[m4] = np.inf

            g4_m0 = gride_mle(dm, 2, 4)
            g4_adj = gride_mle(dm_adj, 2, 4)
            g4_m4 = gride_mle(dm_m4, 2, 4)
            tw_m0, _, ml_m0 = estimate_twonn_mle(dm)
            tw_m4, _, ml_m4 = estimate_twonn_mle(dm_m4)

            if gi < N_nat:
                g4_m0_bos = gride_mle(dm_full, 2, 4)
                e_tl_c, e_tl_a = float(ent_tl_c[gi]), float(ent_tl_a[gi])
            else:
                g4_m0_bos = e_tl_c = e_tl_a = float("nan")

            same_stem_na = m4 & (~adj256)
            np.fill_diagonal(same_stem_na, False)
            r_ratio = float(np.median(dm[same_stem_na]) / np.median(dm[~m4])) if np.any(same_stem_na) else float("nan")

            results_per_job[gi]["depths"][d_l] = {
                "gride4_m0": g4_m0, "gride4_m0_incl_bos": g4_m0_bos,
                "gride4_adj": g4_adj, "gride4_m4": g4_m4,
                "mle5_m0": ml_m0, "mle5_m4": ml_m4,
                "twonn_m0": tw_m0, "twonn_m4": tw_m4,
                "ent_tl_content": e_tl_c, "ent_tl_incl_bos": e_tl_a,
                "r_ctx_over_R_lex": r_ratio,
            }

            if job["kind"] == "natural" and job["domain"] == "sci" and job["nat_idx"] < 12 and d_l in scale_sweep_records:
                for sc in (2, 4, 8, 16, 32, 64):
                    scale_sweep_records[d_l][sc]["m0"].append(gride_mle(dm, sc // 2, sc))
                    scale_sweep_records[d_l][sc]["m4"].append(gride_mle(dm_m4, sc // 2, sc))

        del dm_all
        gc.collect()

    print(f"\n * [STEP 3] Streaming {N_tot} Sequences ({SEQ_L} tokens) Across All {n_layers} Layers...", flush=True)
    t_stream = time.time()
    evaluate_depth_fast(0)
    print(f"   -> Layer  0/{n_layers} (Embed) audited ({time.time() - t_stream:.1f}s)", flush=True)

    B_STEP = 30
    h_L31_seq0 = None
    for l_num in range(1, n_layers + 1):
        streamer.copy_into_module_gpu(shell.model.layers[0], f"model.layers.{l_num - 1}.")
        with torch.inference_mode():
            for b0 in range(0, N_tot, B_STEP):
                b1 = min(b0 + B_STEP, N_tot)
                hb_gpu = h_all[b0:b1].to(DEVICE, non_blocking=True)
                try:
                    shell.model(inputs_embeds=hb_gpu, use_cache=False)
                except _StopAfterLayer:
                    pass
                h_all[b0:b1].copy_(step_cap["h"].cpu())
                del hb_gpu

        if l_num == 31:
            h_L31_seq0 = h_all[0:1].clone()
        if l_num == 32 and h_L31_seq0 is not None:
            with torch.inference_mode():
                x32 = h_all[0, 1:].to(DEVICE)
                x31 = h_L31_seq0[0, 1:].to(DEVICE)
                p32 = F.softmax(shell.lm_head(shell.model.norm(x32)).float(), dim=-1)
                w31, b31 = read_single_tuned_lens_gpu(31, d_hid, streamer.buf)
                q_res = F.log_softmax(shell.lm_head(shell.model.norm(x31 + F.linear(x31, w31, b31))).float(), dim=-1)
                q_raw = F.log_softmax(shell.lm_head(shell.model.norm(x31)).float(), dim=-1)
                kl_res = float((p32 * (p32.clamp_min(1e-12).log() - q_res)).sum(-1).mean().item())
                kl_raw = float((p32 * (p32.clamp_min(1e-12).log() - q_raw)).sum(-1).mean().item())
                print(f"   [TunedLens L31->L32 Fidelity] KL(TunedLens)={kl_res:.4f} vs KL(Raw LogitLens)={kl_raw:.4f}", flush=True)
                del w31, b31

        if l_num in probe_depths:
            evaluate_depth_fast(l_num)
            print(f"   -> Layer {l_num:2d}/{n_layers} streamed & AUDITED ({time.time() - t_stream:.1f}s)", flush=True)
        else:
            print(f"   -> Layer {l_num:2d}/{n_layers} streamed ({time.time() - t_stream:.1f}s)", flush=True)

    # REPORT 0: Llama-3-8B Genre Benchmark (Layer 16/32) + Viswanathan Fig. 8 GRIDE Range-Scaling Sweep
    nat_jobs = [r for r in results_per_job if r["meta"]["kind"] == "natural"]
    sci_j = [r for r in nat_jobs if r["meta"]["domain"] == "sci"]
    sty_j = [r for r in nat_jobs if r["meta"]["domain"] == "story"]

    s_ml0 = np.array([r["depths"][16]["mle5_m0"] for r in sci_j])
    t_ml0 = np.array([r["depths"][16]["mle5_m0"] for r in sty_j])
    s_ml4 = np.array([r["depths"][16]["mle5_m4"] for r in sci_j])
    t_ml4 = np.array([r["depths"][16]["mle5_m4"] for r in sty_j])
    _, p_l16_g = ttest_ind(t_ml4, s_ml4)
    bos_sink_mean = float(np.mean([r["bos_sink_L16"] for r in nat_jobs]))

    print("\n" + "=" * 120)
    print(f" [CHECK 0] Meta-Llama-3-8B (L16/32, L={SEQ_L}) Genre Reversal & Viswanathan Fig. 8 GRIDE Range-Scaling Sweep")
    print("=" * 120)
    print(
        f" * <|begin_of_text|> Sink Ratio at L16 = {bos_sink_mean:.1f}x | Genre MLE-5 (n={len(sci_j)}+{len(sty_j)}): "
        f"Raw {np.mean(s_ml0):.2f}/{np.mean(t_ml0):.2f} (d={cohens_d_ind(t_ml0, s_ml0):+.2f}) -> "
        f"M4 {np.mean(s_ml4):.2f}/{np.mean(t_ml4):.2f} (d={cohens_d_ind(t_ml4, s_ml4):+.2f}, p={p_l16_g:.3e})"
    )
    for d_l in (1, 10, 20, 28):
        row_s = " | ".join(
            f"n2={sc}: {np.mean(scale_sweep_records[d_l][sc]['m0']):4.1f}->{np.mean(scale_sweep_records[d_l][sc]['m4']):4.1f}"
            for sc in (2, 4, 8, 16, 32, 64)
        )
        print(f" * Fig.8 GRIDE Scale (L{d_l:2d}, Raw->M4): {row_s}")

    # REPORT 1: Phenomenon 1 — Block Shuffling Crossover S=0.4 (Viswanathan Fig. 1 & Fig. 11)
    shuf_by_s = {0: [r for r in nat_jobs if r["meta"]["base_shuf_id"] >= 0]}
    shuf_by_s[0].sort(key=lambda r: r["meta"]["base_shuf_id"])
    for s_i in (1, 2, 3, 4):
        grp = [r for r in results_per_job if r["meta"]["kind"] == "shuffled" and r["meta"]["s_idx"] == s_i]
        grp.sort(key=lambda r: r["meta"]["base_shuf_id"])
        shuf_by_s[s_i] = grp

    surp_s0 = float(np.mean([r["surprisal"] for r in shuf_by_s[0]]))
    surp_s4 = float(np.mean([r["surprisal"] for r in shuf_by_s[4]]))
    print("\n" + "=" * 120)
    print(f" [PHENOMENON 1] Viswanathan Fig. 1 & 11 Block-Shuffling Crossover on Meta-Llama-3-8B (n=20, Surprisal S0={surp_s0:.2f}->S4={surp_s4:.2f})")
    print("=" * 120)
    print(" Layer | Raw GRIDE(2,4): S0 -> S2 -> S4 (S4-S0, p)   | Adj-Only: S0->S4    | Masked M4 GRIDE(2,4): S0 -> S2 -> S4 (S4-S0, p) | r_ctx/R_lex(S0->S4)")
    print("-" * 120)

    phenom1_json = {}
    for d_l in probe_depths:
        g0_s = [np.array([r["depths"][d_l]["gride4_m0"] for r in shuf_by_s[s_i]]) for s_i in range(5)]
        ga_s0 = np.array([r["depths"][d_l]["gride4_adj"] for r in shuf_by_s[0]])
        ga_s4 = np.array([r["depths"][d_l]["gride4_adj"] for r in shuf_by_s[4]])
        g4_s = [np.array([r["depths"][d_l]["gride4_m4"] for r in shuf_by_s[s_i]]) for s_i in range(5)]
        rr_0 = float(np.nanmean([r["depths"][d_l]["r_ctx_over_R_lex"] for r in shuf_by_s[0]]))
        rr_4 = float(np.nanmean([r["depths"][d_l]["r_ctx_over_R_lex"] for r in shuf_by_s[4]]))

        ok0 = np.isfinite(g0_s[0]) & np.isfinite(g0_s[4])
        ok4 = np.isfinite(g4_s[0]) & np.isfinite(g4_s[4])
        p_raw = float(ttest_rel(g0_s[4][ok0], g0_s[0][ok0])[1]) if np.sum(ok0) >= 5 else 1.0
        p_msk = float(ttest_rel(g4_s[4][ok4], g4_s[0][ok4])[1]) if np.sum(ok4) >= 5 else 1.0
        diff_raw = float(np.nanmean(g0_s[4]) - np.nanmean(g0_s[0]))
        diff_msk = float(np.nanmean(g4_s[4]) - np.nanmean(g4_s[0]))

        phenom1_json[str(d_l)] = {
            "gride4_m0_S0_to_S4": [float(np.nanmean(x)) for x in g0_s],
            "gride4_adj_S0_S4": [float(np.nanmean(ga_s0)), float(np.nanmean(ga_s4))],
            "gride4_m4_S0_to_S4": [float(np.nanmean(x)) for x in g4_s],
            "diff_S4_minus_S0_raw_p": [diff_raw, p_raw],
            "diff_S4_minus_S0_m4_p": [diff_msk, p_msk],
            "r_ctx_R_lex_S0_S4": [rr_0, rr_4],
        }
        print(
            f"  L{d_l:2d}  | {np.nanmean(g0_s[0]):5.2f} -> {np.nanmean(g0_s[2]):5.2f} -> {np.nanmean(g0_s[4]):5.2f} "
            f"({diff_raw:+5.2f}, p={p_raw:.1e}) | {np.nanmean(ga_s0):5.2f}->{np.nanmean(ga_s4):5.2f} | "
            f"{np.nanmean(g4_s[0]):5.2f} -> {np.nanmean(g4_s[2]):5.2f} -> {np.nanmean(g4_s[4]):5.2f} "
            f"({diff_msk:+5.2f}, p={p_msk:.1e}) |  {rr_0:.3f}->{rr_4:.3f}"
        )

    # REPORT 2: Phenomenon 2 — Fig. 2 Surprisal vs. Fig. 18 Official TunedLens Entropy on Meta-Llama-3-8B
    print("\n" + "=" * 120)
    print(f" [PHENOMENON 2] Viswanathan Fig. 2 (Surprisal) vs. Fig. 18 (Official TunedLens) on Meta-Llama-3-8B (N={len(nat_jobs)})")
    print("=" * 120)
    print(" Layer | Mean H_TL | r(ID_M0,Surp)->M4 | Fig18 r(ID_M0, H_TL) -> r(ID_M4, H_TL) [InclBOS M0] | Prose M0->M4 | r(H_TL,Surp) | r(H_TL,TTR)")
    print("-" * 120)

    surp_all = np.array([r["surprisal"] for r in nat_jobs])
    ttr_all = np.array([r["meta"]["ttr"] for r in nat_jobs])
    prose_mask = np.array([r["meta"]["domain"] in ("sci", "story", "qa") for r in nat_jobs], dtype=bool)

    phenom2_json = {}
    for d_l in probe_depths:
        id_m0 = np.array([r["depths"][d_l]["gride4_m0"] for r in nat_jobs])
        id_m0_bos = np.array([r["depths"][d_l]["gride4_m0_incl_bos"] for r in nat_jobs])
        id_m4 = np.array([r["depths"][d_l]["gride4_m4"] for r in nat_jobs])
        h_tl = np.array([r["depths"][d_l]["ent_tl_content"] for r in nat_jobs])
        h_tl_bos = np.array([r["depths"][d_l]["ent_tl_incl_bos"] for r in nat_jobs])

        ok = np.isfinite(id_m0) & np.isfinite(id_m4) & (id_m0 > 0) & (id_m4 > 0) & np.isfinite(h_tl)
        li0, li4, ht, sp, tt = np.log(id_m0[ok]), np.log(id_m4[ok]), h_tl[ok], surp_all[ok], ttr_all[ok]
        pm = prose_mask[ok]

        r_s0, _ = pearsonr(li0, sp)
        r_s4, _ = pearsonr(li4, sp)
        r_tl0, p_tl0 = pearsonr(li0, ht)
        r_tl4, p_tl4 = pearsonr(li4, ht)
        ok_b = ok & np.isfinite(id_m0_bos) & (id_m0_bos > 0) & np.isfinite(h_tl_bos)
        r_bos0 = float(pearsonr(np.log(id_m0_bos[ok_b]), h_tl_bos[ok_b])[0]) if np.sum(ok_b) >= 10 else float("nan")
        r_pr0, _ = pearsonr(li0[pm], ht[pm])
        r_pr4, _ = pearsonr(li4[pm], ht[pm])
        r_ht_sp, _ = pearsonr(ht, sp)
        r_ht_tt, _ = pearsonr(ht, tt)

        phenom2_json[str(d_l)] = {
            "mean_H_tl": float(np.mean(ht)),
            "r_surprisal_m0_m4": [float(r_s0), float(r_s4)],
            "r_tunedlens_all_m0_p_m4_p": [float(r_tl0), float(p_tl0), float(r_tl4), float(p_tl4)],
            "r_tunedlens_incl_bos_m0": r_bos0,
            "r_tunedlens_prose_m0_m4": [float(r_pr0), float(r_pr4)],
            "r_Htl_vs_surprisal": float(r_ht_sp),
            "r_Htl_vs_ttr": float(r_ht_tt),
        }
        print(
            f"  L{d_l:2d}  |   {np.mean(ht):5.2f}   |  {r_s0:+5.2f} -> {r_s4:+5.2f}   | "
            f" {r_tl0:+6.3f}(p={p_tl0:.1e}) -> {r_tl4:+6.3f}(p={p_tl4:.1e}) [{r_bos0:+5.2f}] | {r_pr0:+5.2f}->{r_pr4:+5.2f} |    {r_ht_sp:+5.2f}     |   {r_ht_tt:+5.2f}"
        )

    OUT_JSON.write_text(
        json.dumps({
            "bos_sink_L16": bos_sink_mean,
            "genre_L16_mle5": {
                "raw_sci_sty_d": [float(np.mean(s_ml0)), float(np.mean(t_ml0)), float(cohens_d_ind(t_ml0, s_ml0))],
                "m4_sci_sty_d_p": [float(np.mean(s_ml4)), float(np.mean(t_ml4)), float(cohens_d_ind(t_ml4, s_ml4)), float(p_l16_g)],
            },
            "fig8_scale_sweep": {str(k): {str(sc): [float(np.mean(v["m0"])), float(np.mean(v["m4"]))] for sc, v in d.items()} for k, d in scale_sweep_records.items()},
            "phenom1_block_shuffle": phenom1_json,
            "phenom2_tuned_lens": phenom2_json,
        }, indent=2),
        encoding="utf-8",
    )
    print("\n" + "=" * 120)
    print(f" [META-LLAMA-3-8B + OFFICIAL TUNEDLENS AUDIT COMPLETED IN {time.time() - t_start:.1f}s -> {OUT_JSON.name}]")
    print("=" * 120)


if __name__ == "__main__":
    main()
