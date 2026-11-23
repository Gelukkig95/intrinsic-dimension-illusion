import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

import gc
import json
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind, ttest_rel

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
DATA_CACHE = THIS_DIR / "cache_scaleup"
TMP_7B_DIR = DATA_CACHE / "tmp_qwen25_7b_stream"
DMAT_BACKUP = DATA_CACHE / "qwen25_7b_L14_dmats.npy"

from verify_idea1_all_4 import (
    build_masks,
    cohens_d_ind,
    estimate_twonn_mle,
    set_seed,
    tokenize_fixed_length_with_metadata,
)
from run_modern_scaleup import (
    compute_auroc,
    greedy_caliper_match,
    load_arxiv_pool,
    load_hc3_chatgpt_pool,
    load_stories_pool,
    prewarm_cuda_cublas,
)
try:
    from finish_defense_4a_4b import tokenize_multilingual_clean
except ImportError:
    def _simple_de_ko_stem(word: str, lang: str) -> str:
        w = word.lower()
        if lang == "de":
            if len(w) <= 4:
                return w
            for suf in ("ungen", "lichkeiten", "keiten", "heiten", "ische", "ischen", "ischer", "endes", "enden", "ern", "ers", "en", "er", "es", "em", "st", "te", "n", "e", "s"):
                if w.endswith(suf) and len(w) - len(suf) >= 4:
                    return w[:-len(suf)]
            return w
        if lang == "ko":
            if len(w) <= 2:
                return w
            for p in ("에서는", "으로는", "에게는", "이라는", "에서도", "으로", "에서", "에게", "까지", "부터", "처럼", "보다", "하고", "이며", "이다", "은", "는", "이", "가", "을", "를", "의", "에", "도", "로", "와", "과"):
                if w.endswith(p) and len(w) - len(p) >= 2:
                    return w[:-len(p)]
            return w
        return w

    def tokenize_multilingual_clean(tok, text: str, lang: str, target_len: int = 150) -> dict | None:
        """Byte-BPE-safe multilingual tokenizer & M4 mask builder (standalone fallback)."""
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
        last_pos = int(keep_idx[-1]) + 1
        full_ids = ids_all[: last_pos + 1] if (last_pos < len(ids_all) and spec_mask[last_pos] == 1) else ids_all[:last_pos]

        word_spans = []
        for wi, m in enumerate(re.finditer(r"\S+", text)):
            ws, we = m.span()
            raw_w = m.group(0)
            cleaned = re.sub(r"[^\w]", "", raw_w, flags=re.UNICODE).lower()
            surf = cleaned if cleaned else raw_w.strip()
            stem = _simple_de_ko_stem(surf, lang)
            word_spans.append((ws, we, wi, surf, stem))

        tok_ids = ids_all[keep_idx]
        word_occ = np.arange(target_len, dtype=np.int64) + 100000
        surf_keys, stem_keys, norm_keys = [], [], []
        ptr = 0
        for out_i, orig_i in enumerate(keep_idx):
            cs, ce = offsets_all[int(orig_i)]
            dec = tok.decode([int(ids_all[int(orig_i)])], errors="ignore").strip().lower()
            norm_keys.append(re.sub(r"[^\w]", "", dec, flags=re.UNICODE) or dec)
            matched = None
            if ce > cs:
                while ptr < len(word_spans) and word_spans[ptr][1] <= cs:
                    ptr += 1
                for k in range(ptr, min(ptr + 4, len(word_spans))):
                    ws, we, wi, sf, st = word_spans[k]
                    if ws < ce and we > cs:
                        matched = (wi, sf, st)
                        break
            if matched is not None:
                word_occ[out_i] = matched[0]
                surf_keys.append(matched[1])
                stem_keys.append(matched[2])
            else:
                surf_keys.append(norm_keys[-1])
                stem_keys.append(_simple_de_ko_stem(norm_keys[-1], lang))

        pos = np.arange(target_len)
        adj1 = np.abs(pos[:, None] - pos[None, :]) <= 1
        same_id = tok_ids[:, None] == tok_ids[None, :]
        same_norm = np.asarray(norm_keys, dtype=object)[:, None] == np.asarray(norm_keys, dtype=object)[None, :]
        same_wocc = word_occ[:, None] == word_occ[None, :]
        same_surf = np.asarray(surf_keys, dtype=object)[:, None] == np.asarray(surf_keys, dtype=object)[None, :]
        same_stem = np.asarray(stem_keys, dtype=object)[:, None] == np.asarray(stem_keys, dtype=object)[None, :]
        mask_m4 = same_id | same_norm | same_wocc | same_surf | same_stem | adj1
        return {
            "full_ids": torch.tensor(full_ids, dtype=torch.long).unsqueeze(0),
            "keep_positions": keep_idx,
            "tok_ids": tok_ids,
            "ttr_stem": len(set(stem_keys)) / float(target_len),
            "mask_m4": mask_m4,
        }

HF_BASE = "https://huggingface.co/Qwen/Qwen2.5-7B/resolve/main"
UA_STR = "IdIllusionGeometryAudit/1.0"


class SeekSafetensorReader:
    DTYPE_MAP = {
        "BF16": (torch.bfloat16, 2),
        "F16": (torch.float16, 2),
        "F32": (torch.float32, 4),
        "F64": (torch.float64, 8),
        "I64": (torch.int64, 8),
        "I32": (torch.int32, 4),
    }

    def __init__(self, path: Path) -> None:
        self.path = path
        self.file_size = path.stat().st_size
        self.f = open(path, "rb")
        try:
            raw_len = self.f.read(8)
            if len(raw_len) < 8:
                raise EOFError(f"Empty header in {path.name}")
            hdr_len = struct.unpack("<Q", raw_len)[0]
            if hdr_len <= 0 or hdr_len > 50_000_000:
                raise ValueError(f"Invalid header length {hdr_len} in {path.name}")
            hdr_obj = json.loads(self.f.read(hdr_len).decode("utf-8"))
            self.data_base = 8 + hdr_len
            self.tensors = {k: v for k, v in hdr_obj.items() if k != "__metadata__"}
            max_end = max((v["data_offsets"][1] for v in self.tensors.values()), default=0)
            if self.data_base + max_end > self.file_size:
                raise EOFError(f"Truncated shard {path.name}")
        except Exception:
            self.f.close()
            raise

    def __enter__(self) -> "SeekSafetensorReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.f.close()

    def keys(self) -> set[str]:
        return set(self.tensors.keys())

    def gather_embedding_rows(self, key: str, id_sequences: list[torch.Tensor]) -> list[torch.Tensor]:
        info = self.tensors[key]
        dt, elem_bytes = self.DTYPE_MAP[info["dtype"]]
        hidden_sz = int(info["shape"][1])
        row_bytes = hidden_sz * elem_bytes
        base_off = self.data_base + int(info["data_offsets"][0])

        uniq_ids = sorted({int(tid) for seq in id_sequences for tid in seq.view(-1).tolist()})
        id_to_vec: dict[int, torch.Tensor] = {}
        for tid in uniq_ids:
            self.f.seek(base_off + tid * row_bytes)
            buf = bytearray(self.f.read(row_bytes))
            id_to_vec[tid] = torch.frombuffer(buf, dtype=dt).to(dtype=torch.float16).clone()

        out_bank = []
        for seq in id_sequences:
            rows = [id_to_vec[int(tid)] for tid in seq.view(-1).tolist()]
            out_bank.append(torch.stack(rows, dim=0).contiguous())
        del id_to_vec
        return out_bank

    def read_tensor(self, key: str, target_dtype: torch.dtype, target_device: torch.device) -> torch.Tensor:
        info = self.tensors[key]
        dt, _ = self.DTYPE_MAP[info["dtype"]]
        start_rel, end_rel = int(info["data_offsets"][0]), int(info["data_offsets"][1])
        n_bytes = end_rel - start_rel
        self.f.seek(self.data_base + start_rel)

        buf = bytearray(n_bytes)
        mv = memoryview(buf)
        pos = 0
        while pos < n_bytes:
            n_read = self.f.readinto(mv[pos:])
            if not n_read:
                raise EOFError(f"Unexpected EOF while reading {key}")
            pos += n_read

        out = torch.frombuffer(buf, dtype=dt).reshape(info["shape"]).to(device=target_device, dtype=target_dtype).clone()
        del mv, buf
        return out


def curl_download(filename: str, dest: Path, min_bytes: int = 100) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"   * Reusing cached {filename} ({dest.stat().st_size / 1e9:.2f} GB)")
        return dest
    url = f"{HF_BASE}/{filename}"
    cmd = [
        (shutil.which("curl") or "curl"), "-L", "--fail", "--retry", "4", "--retry-delay", "2",
        "-C", "-", "-A", UA_STR, "-o", str(dest), url,
    ]
    proc = subprocess.run(cmd)
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size < min_bytes:
        raise RuntimeError(f"curl failed on {filename} (exit={proc.returncode})")
    return dest


def open_verified_shard(shard_fn: str) -> SeekSafetensorReader:
    sp = curl_download(shard_fn, TMP_7B_DIR / shard_fn, min_bytes=100_000_000)
    try:
        return SeekSafetensorReader(sp)
    except (EOFError, ValueError) as err:
        print(f"   [Shard Repair] {err} -> re-downloading {shard_fn} cleanly...")
        sp.unlink(missing_ok=True)
        sp = curl_download(shard_fn, TMP_7B_DIR / shard_fn, min_bytes=100_000_000)
        return SeekSafetensorReader(sp)


def tokenize_multilingual_with_word_adj(tok, text: str, lang: str, target_len: int = 150) -> dict | None:
    base = tokenize_multilingual_clean(tok, text, lang, target_len=target_len)
    if base is None:
        return None
    enc = tok(text, truncation=True, max_length=target_len + 16, return_offsets_mapping=True, return_special_tokens_mask=True)
    offsets_all = enc["offset_mapping"]
    keep_idx = base["keep_positions"]

    spans = [(m.start(), m.end(), wi) for wi, m in enumerate(re.finditer(r"\S+", text))]
    w_occ = np.arange(target_len, dtype=np.int64) + 100000
    ptr = 0
    for out_i, orig_i in enumerate(keep_idx):
        cs, ce = offsets_all[int(orig_i)]
        if ce > cs:
            while ptr < len(spans) and spans[ptr][1] <= cs:
                ptr += 1
            for k in range(ptr, min(ptr + 4, len(spans))):
                if spans[k][0] < ce and spans[k][1] > cs:
                    w_occ[out_i] = spans[k][2]
                    break
    base["mask_m4_word_adj"] = base["mask_m4"] | (np.abs(w_occ[:, None] - w_occ[None, :]) <= 1)
    return base


def stream_exact_7b_dmats(flat_metas: list[dict]) -> np.ndarray:
    """
    Streams Layers 00..13 in FP16, then reduces each passage ONE AT A TIME to its (150, 150)
    distance matrix while freeing h_bank[i] immediately (85 MB total instead of 2,034 MB!).
    """
    if DMAT_BACKUP.exists():
        cached = np.load(DMAT_BACKUP)
        if cached.shape == (len(flat_metas), 150, 150):
            print(f" * Loaded backed-up 7B distance matrices from {DMAT_BACKUP.name} {cached.shape}!")
            return cached

    from transformers import AutoConfig, AutoModel

    all_id_tensors = [m["full_ids"] for m in flat_metas]
    curl_download("config.json", TMP_7B_DIR / "config.json", min_bytes=200)
    idx_path = curl_download("model.safetensors.index.json", TMP_7B_DIR / "model.safetensors.index.json", min_bytes=1000)

    cfg = AutoConfig.from_pretrained(str(TMP_7B_DIR))
    orig_l = int(cfg.num_hidden_layers)
    mid_l = orig_l // 2

    cfg_1 = AutoConfig.from_pretrained(str(TMP_7B_DIR))
    cfg_1.num_hidden_layers = 1
    cfg_1.vocab_size = 16
    cfg_1.pad_token_id = None
    cfg_1.bos_token_id = 0
    cfg_1.eos_token_id = 1
    cfg_1.use_cache = False
    cfg_1._attn_implementation = "eager"
    if getattr(cfg_1, "layer_types", None) is not None and isinstance(cfg_1.layer_types, list):
        cfg_1.layer_types = cfg_1.layer_types[:1]

    run_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dt = torch.float16 if run_dev.type == "cuda" else torch.float32

    prev_dt = torch.get_default_dtype()
    try:
        torch.set_default_dtype(run_dt)
        runner_1l = AutoModel.from_config(cfg_1)
    finally:
        torch.set_default_dtype(prev_dt)

    if hasattr(runner_1l, "norm"):
        runner_1l.norm = nn.Identity()
    runner_1l = runner_1l.to(device=run_dev, dtype=run_dt).eval()

    wmap: dict[str, str] = json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]
    layer0_param_map = dict(runner_1l.layers[0].named_parameters())
    layer_subkeys = list(layer0_param_map.keys())

    needed_shards = []
    for k_ckpt, shard_fn in wmap.items():
        if k_ckpt == "model.embed_tokens.weight" or (
            k_ckpt.startswith("model.layers.") and int(k_ckpt.split(".")[2]) < mid_l
        ):
            if shard_fn not in needed_shards:
                needed_shards.append(shard_fn)
    needed_shards.sort()

    by_len: dict[int, list[int]] = {}
    for idx, t_ids in enumerate(all_id_tensors):
        by_len.setdefault(int(t_ids.shape[1]), []).append(idx)

    h_bank: list[torch.Tensor | None] | None = None
    pending_cpu: dict[str, torch.Tensor] = {}
    next_l = 0

    for s_idx, shard_fn in enumerate(needed_shards, start=1):
        print(f"\n * [Shard {s_idx}/{len(needed_shards)}] Opening {shard_fn} via Zero-mmap SeekSafetensorReader...")
        sp = TMP_7B_DIR / shard_fn

        with open_verified_shard(shard_fn) as reader:
            sf_keys = reader.keys()

            if h_bank is None and "model.embed_tokens.weight" in sf_keys:
                h_bank = reader.gather_embedding_rows("model.embed_tokens.weight", all_id_tensors)  # type: ignore
                gc.collect()
                print(f"   -> Embedded {len(h_bank)} passages via row-seek (508.6 MB FP16).")

            while next_l < mid_l:
                req_keys = [f"model.layers.{next_l}.{sk}" for sk in layer_subkeys]
                if all((rk in pending_cpu or rk in sf_keys) for rk in req_keys):
                    for sk in layer_subkeys:
                        rk = f"model.layers.{next_l}.{sk}"
                        if rk in pending_cpu:
                            t_gpu = pending_cpu.pop(rk).to(device=run_dev, dtype=run_dt)
                        else:
                            t_gpu = reader.read_tensor(rk, target_dtype=run_dt, target_device=run_dev)
                        layer0_param_map[sk].data.copy_(t_gpu)
                        del t_gpu

                    with torch.inference_mode():
                        for seq_len, idxs in by_len.items():
                            for b0 in range(0, len(idxs), 12):
                                chunk = idxs[b0 : b0 + 12]
                                bh = torch.stack([h_bank[i] for i in chunk], dim=0).to(device=run_dev, dtype=run_dt)  # type: ignore
                                bm = torch.ones((len(chunk), seq_len), device=run_dev, dtype=torch.long)
                                bout = runner_1l(inputs_embeds=bh, attention_mask=bm, use_cache=False).last_hidden_state
                                bout_cpu = bout.to(device="cpu", dtype=torch.float16).contiguous()
                                for loc_i, orig_i in enumerate(chunk):
                                    h_bank[orig_i] = bout_cpu[loc_i]  # type: ignore
                                del bh, bm, bout, bout_cpu
                    if run_dev.type == "cuda":
                        torch.cuda.synchronize()
                    print(f"   -> Completed exact Layer {next_l:02d}/{orig_l - 1}")
                    next_l += 1
                else:
                    for l_rem in range(next_l, mid_l):
                        for sk in layer_subkeys:
                            rk = f"model.layers.{l_rem}.{sk}"
                            if rk in sf_keys and rk not in pending_cpu:
                                pending_cpu[rk] = reader.read_tensor(rk, target_dtype=torch.float16, target_device=torch.device("cpu"))
                    break

        sp.unlink(missing_ok=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"   * Closed & deleted {shard_fn} cleanly.")

    assert next_l == mid_l and h_bank is not None, f"Expected {mid_l} layers, ran {next_l}"

    # Free GPU layer & temp shard dir BEFORE computing pairwise distances
    del runner_1l, pending_cpu
    shutil.rmtree(TMP_7B_DIR, ignore_errors=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # In-place 1-by-1 reduction: converts ONE passage (4.1 MB) at a time and frees h_bank[i] immediately.
    n_tot = len(h_bank)
    dmats = np.empty((n_tot, 150, 150), dtype=np.float64)
    for i in range(n_tot):
        t_fp16 = h_bank[i]
        h_bank[i] = None  # Immediately drop reference to 1.07 MB tensor.
        kp = flat_metas[i]["keep_positions"]
        h64 = np.ascontiguousarray(t_fp16[kp].float().numpy(), dtype=np.float64)  # type: ignore
        dmats[i] = cdist(h64, h64, metric="euclidean")
        del t_fp16, h64

    del h_bank
    gc.collect()
    np.save(DMAT_BACKUP, dmats)
    print(f" * Saved all {n_tot} exact 7B pairwise distance matrices ({dmats.nbytes / 1e6:.1f} MB) to {DMAT_BACKUP.name}!")
    return dmats


def main() -> None:
    t0 = time.perf_counter()
    set_seed(2026)
    prewarm_cuda_cublas()

    from transformers import AutoTokenizer

    tok_dir = DATA_CACHE / "qwen25_7b_tokenizer"
    for fn in ("tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.json"):
        curl_download(fn, tok_dir / fn, min_bytes=100)
    tok7 = AutoTokenizer.from_pretrained(str(tok_dir))

    stories_150, _ = load_stories_pool(min_words=175, target_n=150, long_min_words=265)
    arxiv_150, _ = load_arxiv_pool(target_n=150)
    hc3_hum, hc3_ai = load_hc3_chatgpt_pool(target_pairs=80)
    ml_saved = json.loads((DATA_CACHE / "multilingual_de_ko_verified_55.json").read_text(encoding="utf-8"))

    groups = {
        "en_sci": [m for t in arxiv_150[:80] if (m := tokenize_fixed_length_with_metadata(tok7, t, 150)) is not None],
        "en_fic": [m for t in stories_150[:80] if (m := tokenize_fixed_length_with_metadata(tok7, t, 150)) is not None],
        "hc3_hum": [m for t in hc3_hum[:50] if (m := tokenize_fixed_length_with_metadata(tok7, t, 150)) is not None],
        "hc3_ai": [m for t in hc3_ai[:50] if (m := tokenize_fixed_length_with_metadata(tok7, t, 150)) is not None],
        "de_sci": [m for t in ml_saved["de_sci"][:55] if (m := tokenize_multilingual_with_word_adj(tok7, t, "de", 150)) is not None],
        "de_fic": [m for t in ml_saved["de_fic"][:55] if (m := tokenize_multilingual_with_word_adj(tok7, t, "de", 150)) is not None],
        "ko_sci": [m for t in ml_saved["ko_sci"][:55] if (m := tokenize_multilingual_with_word_adj(tok7, t, "ko", 150)) is not None],
        "ko_fic": [m for t in ml_saved["ko_fic"][:55] if (m := tokenize_multilingual_with_word_adj(tok7, t, "ko", 150)) is not None],
    }

    flat_metas = []
    offsets = {}
    for g_name, m_list in groups.items():
        s_i = len(flat_metas)
        flat_metas.extend(m_list)
        offsets[g_name] = (s_i, len(flat_metas))

    print(f"=== [Qwen/Qwen2.5-7B Zero-mmap + Streaming Distance Reducer | Passages={len(flat_metas)}] ===")
    dmats = stream_exact_7b_dmats(flat_metas)

    def compute_metrics_group(g_name: str, mask_mode: str = "en") -> dict[str, np.ndarray]:
        s_i, e_i = offsets[g_name]
        out = {"ttr": [], "twonn_std": [], "twonn_msk": [], "mle_std": [], "mle_msk": []}
        for idx in range(s_i, e_i):
            meta = flat_metas[idx]
            dmat = dmats[idx].copy()
            np.fill_diagonal(dmat, np.inf)
            d_std = dmat.copy()
            d_std[d_std <= 1e-6] = np.inf
            d_msk = d_std.copy()
            mk = build_masks(meta)["M4_stem_adj"] if mask_mode == "en" else meta[mask_mode]
            d_msk[mk] = np.inf
            fs, _, ms = estimate_twonn_mle(d_std)
            fm, _, mm = estimate_twonn_mle(d_msk)
            out["ttr"].append(meta["ttr_stem"])
            out["twonn_std"].append(fs)
            out["twonn_msk"].append(fm)
            out["mle_std"].append(ms)
            out["mle_msk"].append(mm)
        return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}

    sci7 = compute_metrics_group("en_sci", "en")
    fic7 = compute_metrics_group("en_fic", "en")
    hum7 = compute_metrics_group("hc3_hum", "en")
    ai7 = compute_metrics_group("hc3_ai", "en")

    print("\n" + "=" * 98)
    print(" [DEFENSE 4B RESULTS] Qwen/Qwen2.5-7B Flagship (Exact Layer 14/28, Zero Quantization)")
    print("=" * 98)
    ia, ib = greedy_caliper_match(sci7["ttr"], fic7["ttr"], caliper=0.015)
    rec7 = {"model_id": "Qwen/Qwen2.5-7B", "layer": "14/28"}
    for name, ks, km in (("TwoNN", "twonn_std", "twonn_msk"), ("MLE-5", "mle_std", "mle_msk")):
        db = cohens_d_ind(fic7[ks], sci7[ks])
        da = cohens_d_ind(fic7[km], sci7[km])
        _, pa = ttest_ind(fic7[km], sci7[km], nan_policy="omit")
        dm = cohens_d_ind(fic7[km][ib], sci7[km][ia])
        _, pm = ttest_rel(fic7[km][ib], sci7[km][ia], nan_policy="omit")
        print(
            f" * {name:6s} EN Full (N={len(sci7['ttr'])+len(fic7['ttr'])}): "
            f"BEFORE Sci={np.nanmean(sci7[ks]):5.2f}, Story={np.nanmean(fic7[ks]):5.2f} (d={db:+.2f}) -> "
            f"AFTER Sci={np.nanmean(sci7[km]):5.2f}, Story={np.nanmean(fic7[km]):5.2f} (d={da:+.2f}, p={pa:.4f}) | "
            f"TTR-Matched({len(ia)}p) d={dm:+.2f} (p={pm:.4f})"
        )
        rec7[name] = {
            "sci_before": float(np.nanmean(sci7[ks])),
            "story_before": float(np.nanmean(fic7[ks])),
            "before_d": float(db),
            "sci_after": float(np.nanmean(sci7[km])),
            "story_after": float(np.nanmean(fic7[km])),
            "after_d": float(da),
            "after_p": float(pa),
            "matched_d": float(dm),
            "matched_p": float(pm),
        }

    yh = np.array([1] * len(hum7["ttr"]) + [0] * len(ai7["ttr"]), dtype=np.int64)
    ih7, ic7 = greedy_caliper_match(hum7["ttr"], ai7["ttr"], caliper=0.015)
    ym7 = np.array([1] * len(ih7) + [0] * len(ic7), dtype=np.int64)
    au_s = compute_auroc(yh, np.nan_to_num(np.concatenate([hum7["twonn_std"], ai7["twonn_std"]]))) * 100.0
    au_m = compute_auroc(yh, np.nan_to_num(np.concatenate([hum7["twonn_msk"], ai7["twonn_msk"]]))) * 100.0
    aum_s = compute_auroc(ym7, np.nan_to_num(np.concatenate([hum7["twonn_std"][ih7], ai7["twonn_std"][ic7]]))) * 100.0
    aum_m = compute_auroc(ym7, np.nan_to_num(np.concatenate([hum7["twonn_msk"][ih7], ai7["twonn_msk"][ic7]]))) * 100.0
    print(
        f" * [7B HC3 Real ChatGPT] TwoNN Full AUROC: Std={au_s:5.2f}% -> Msk={au_m:5.2f}% | "
        f"TTR-Matched({len(ih7)}p): Std={aum_s:5.2f}% -> Msk={aum_m:5.2f}%"
    )
    rec7["hc3_auroc"] = {"full": [float(au_s), float(au_m)], "matched": [float(aum_s), float(aum_m)]}

    print("\n * === [DEFENSE 4A+4B] Multilingual DE & KO on Qwen/Qwen2.5-7B (Token-Adj vs. Word-Span-Adj) ===")
    rec7["multilingual"] = {}
    for lang, title in (("de", "German (DE)"), ("ko", "Korean (KO)")):
        sm_tok = compute_metrics_group(f"{lang}_sci", "mask_m4")
        fm_tok = compute_metrics_group(f"{lang}_fic", "mask_m4")
        sm_wrd = compute_metrics_group(f"{lang}_sci", "mask_m4_word_adj")
        fm_wrd = compute_metrics_group(f"{lang}_fic", "mask_m4_word_adj")
        ia_l, ib_l = greedy_caliper_match(sm_wrd["ttr"], fm_wrd["ttr"], caliper=0.020)

        db_ml = cohens_d_ind(fm_tok["mle_std"], sm_tok["mle_std"])
        da_tok = cohens_d_ind(fm_tok["mle_msk"], sm_tok["mle_msk"])
        da_wrd = cohens_d_ind(fm_wrd["mle_msk"], sm_wrd["mle_msk"])
        dm_wrd = cohens_d_ind(fm_wrd["mle_msk"][ib_l], sm_wrd["mle_msk"][ia_l]) if len(ia_l) >= 8 else float("nan")

        print(
            f"   [{title}] (n={len(sm_wrd['ttr'])}/{len(fm_wrd['ttr'])}, Matched={len(ia_l)}p)\n"
            f"     - TwoNN : Unmasked Sci={np.nanmean(sm_wrd['twonn_std']):5.2f}/Fic={np.nanmean(fm_wrd['twonn_std']):5.2f} -> "
            f"Word-Adj Masked Sci={np.nanmean(sm_wrd['twonn_msk']):5.2f}/Fic={np.nanmean(fm_wrd['twonn_msk']):5.2f}\n"
            f"     - MLE-5 : BEFORE d={db_ml:+.2f} -> Token-Adj(|i-j|<=1) d={da_tok:+.2f} -> "
            f"Word-Adj(|w_a-w_b|<=1) d={da_wrd:+.2f} (TTR-Matched d={dm_wrd:+.2f})"
        )
        rec7["multilingual"][lang] = {
            "twonn_unmasked": [float(np.nanmean(sm_wrd["twonn_std"])), float(np.nanmean(fm_wrd["twonn_std"]))],
            "twonn_word_masked": [float(np.nanmean(sm_wrd["twonn_msk"])), float(np.nanmean(fm_wrd["twonn_msk"]))],
            "mle_before_d": float(db_ml),
            "mle_tok_adj_d": float(da_tok),
            "mle_word_adj_d": float(da_wrd),
            "mle_word_adj_matched_d": float(dm_wrd),
        }

    final_json = {
        "defense1_null_sphere_p": {"S12": 0.3572, "S20": 0.2722},
        "defense2_qwen3_hc3_matched_auroc": {"unmasked": 78.28, "adj_only": 81.28, "stem_only": 56.26, "full_m4": 58.18},
        "defense3_qwen3_pos0_sink": {"norm_ratio": 91.3, "phd_unmasked": [11.68, 13.34, 1.18], "phd_masked": [23.19, 23.70, 0.12]},
        "defense4a_qwen25_05b": {
            "de_mle": {"before": [8.27, 10.11, 1.29], "after": [18.20, 19.86, 1.04], "matched_d": 0.38},
            "ko_mle": {"before": [6.84, 9.44, 1.89], "after": [18.25, 21.34, 1.51], "matched_d": 1.71},
        },
        "defense4b_qwen25_7b": rec7,
    }
    out_p = THIS_DIR / "four_defenses_results.json"
    out_p.write_text(json.dumps(final_json, indent=2), encoding="utf-8")
    print(f"\n=== [7B Audit Completed in {time.perf_counter() - t0:.1f}s | Saved {out_p.name}] ===")


if __name__ == "__main__":
    main()
