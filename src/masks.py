"""Representation-blind symbolic short-circuit masking hierarchy (M0..M4)."""

from __future__ import annotations

import re
import numpy as np


def simple_english_stem(word: str) -> str:
    """Deterministic suffix normalizer matching inflected English stems."""
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


def tokenize_fixed_length_with_metadata(tok, text: str, target_len: int = 150) -> dict | None:
    """Extracts exact `target_len` non-special tokens with word-span, surface, and stem metadata."""
    import torch

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
    if last_pos < len(ids_all) and spec_mask[last_pos] == 1:
        full_ids = ids_all[: last_pos + 1]
    else:
        full_ids = ids_all[:last_pos]

    word_spans = []
    for w_idx, m in enumerate(re.finditer(r"\S+", text)):
        ws, we = m.span()
        raw_w = m.group(0)
        alnum = re.sub(r"[^a-zA-Z0-9]", "", raw_w).lower()
        surf = alnum if alnum else raw_w.strip()
        stem = simple_english_stem(alnum) if alnum.isalpha() else surf
        word_spans.append((ws, we, w_idx, surf, stem))

    tok_ids = ids_all[keep_idx]
    norm_tok_keys, surf_keys, stem_keys = [], [], []
    word_occ_ids = np.arange(target_len, dtype=np.int64)

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


def build_adjacency_mask(seq_len: int, window: int = 1) -> np.ndarray:
    """Theiler temporal-autocorrelation mask forbidding pairs with |i - j| <= window."""
    pos = np.arange(seq_len)
    return np.abs(pos[:, None] - pos[None, :]) <= int(window)


def build_mask_hierarchy(meta: dict, window: int = 1) -> dict[str, np.ndarray]:
    """Builds the nested symbolic masks M1..M4 and adjacency-only mask from token metadata."""
    n = len(meta["tok_ids"])
    adj = build_adjacency_mask(n, window=window)
    same_id = meta["tok_ids"][:, None] == meta["tok_ids"][None, :]
    same_norm = meta["norm_tok"][:, None] == meta["norm_tok"][None, :]
    same_wocc = meta["word_occ"][:, None] == meta["word_occ"][None, :]
    same_surf = meta["surf_word"][:, None] == meta["surf_word"][None, :]
    same_stem = meta["stem"][:, None] == meta["stem"][None, :]

    return {
        "M1_id": same_id,
        "M2_norm": same_id | same_norm,
        "M3_surf_adj": same_id | same_norm | same_wocc | same_surf | adj,
        "M4_stem_adj": same_id | same_norm | same_wocc | same_surf | same_stem | adj,
        "adj_only": adj,
    }
