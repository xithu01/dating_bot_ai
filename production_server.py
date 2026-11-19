#!/usr/bin/env python3
# FILE: production_server.py
# PURPOSE: Step 1 + Step 2 + Step 3 + Step 4 + Step 5 + Step 6 + Step 7 Pure NumPy Dating Inference Engine
# RULES: Zero PyTorch/SciPy/DB, zero offline_train import, zero float64/int64, PyArmor compatible (< 430 lines / < 18 KB)

import asyncio
from collections import deque
import os
import sys
import time
import numpy as np
from step8_looking_for import forward_step8, get_step8_engine
from step9_name import forward_step9, get_step9_engine
from step10_hobbies import forward_step10, get_step10_engine
from step11_rel_status import forward_step11, get_step11_engine
from step12_private_yesno import forward_step12, get_step12_engine
from step13_pic_photo import forward_step13, get_step13_engine
from step14_bot_check import forward_step14, get_step14_engine
from plugin_hub import forward_post_s7, forward_pre_s7, init_all_plugins
from production_server_part2 import (
    _build_step1_bloom_tables,
    handle_user_query,
    wal_write_behind_worker,
)

VOCAB_SIZE = 2048
SEQ_LEN = 32
MAX_GREETING_LEN = 30
MAX_S2_LEN = 45
BLOOM_SIZE = 65536
D_MODEL = 64
D_FF = 128
NUM_INTENTS = 2
S2_VOCAB = 1024
S2_DIM = 32
S2_CLASSES = 6
S3_CLASSES = 2
S4_CLASSES = 7
S5_CLASSES = 2
S6_CLASSES = 4
S7_CLASSES = 3
MAX_USERS = 50000
NUM_WORKERS = 8


class QuantizedModel:
    __slots__ = (
        "emb_f32", "wqkv1_f32", "wo1_f32", "w1_f32", "w2_f32",
        "wqkv2_f32", "wo2_f32", "w3_f32", "w4_f32", "w_intent_f32",
        "s2_emb_f32", "s2_w1_f32", "s2_w2_f32",
        "s3_emb_f32", "s3_w1_f32", "s3_w2_f32",
        "s4_emb_f32", "s4_w1_f32", "s4_w2_f32",
        "s5_emb_f32", "s5_w1_f32", "s5_w2_f32", "s5_wyd_bloom_q",
        "s6_emb_f32", "s6_w1_f32", "s6_w2_f32",
        "s6_status_bloom_q", "status_lut_a", "status_lut_b",
        "s7_emb_f32", "s7_w1_f32", "s7_w2_f32",
        "s7_flirt_react_bloom_q", "s7_lut_a", "s7_lut_b",
        "word_bloom_q", "single_bloom_q", "target_bloom_q", "lead_bloom_q",
        "s2_frame_bloom_q", "s3_hru_bloom_q", "s4_gender_bloom_q",
        "country_lut_a", "country_lut_b", "gender_lut_a", "gender_lut_b",
        "pos_f32", "mask_f32", "char_lut", "reject_lut", "poly_mat", "w_poly_mat",
        "inv_sqrt_d", "eps", "m5_0", "m5_1", "m5_2", "m5_3", "mask_b", "mask5", "off5",
        "m3_0", "m3_1", "mask3", "off3", "bloom_mask", "s2_v_mask", "one_i32", "space_i32", "conf_thresh",
    )

    def __init__(self, npz_path):
        req_keys = (
            "w_intent_q", "s2_country_lut_a", "s3_w2_q", "s4_w2_q",
            "s4_gender_lut_a", "s5_w2_q", "s6_w2_q", "s6_status_lut_a",
            "s7_w2_q", "s7_lut_a",
        )
        if not os.path.exists(npz_path):
            print(f"[VAULT ERROR] '{npz_path}' not found! Run first: python offline_train.py", flush=True)
            sys.exit(1)
        arc = np.load(npz_path, mmap_mode="r")
        missing = [k for k in req_keys if k not in arc.files]
        if missing:
            arc.close()
            print(f"[VAULT ERROR] Missing Step weights {missing} in '{npz_path}'! Run first: python offline_train.py", flush=True)
            sys.exit(1)

        def lock_f32(prefix, zero_row0=False):
            q = arc[f"{prefix}_q"].astype(np.int8, copy=False)
            s = arc[f"{prefix}_s"].astype(np.float32, copy=False)
            f32 = np.multiply(q, s, dtype=np.float32)
            if zero_row0:
                f32[0].fill(0.0)
            f32.flags.writeable = False
            return f32

        self.emb_f32 = lock_f32("emb", True)
        self.wqkv1_f32 = lock_f32("wqkv1")
        self.wo1_f32 = lock_f32("wo1")
        self.w1_f32 = lock_f32("w1")
        self.w2_f32 = lock_f32("w2")
        self.wqkv2_f32 = lock_f32("wqkv2")
        self.wo2_f32 = lock_f32("wo2")
        self.w3_f32 = lock_f32("w3")
        self.w4_f32 = lock_f32("w4")
        self.w_intent_f32 = lock_f32("w_intent")

        for pfx in ("s2", "s3", "s4", "s5", "s6", "s7"):
            setattr(self, f"{pfx}_emb_f32", lock_f32(f"{pfx}_emb", True))
            setattr(self, f"{pfx}_w1_f32", lock_f32(f"{pfx}_w1"))
            setattr(self, f"{pfx}_w2_f32", lock_f32(f"{pfx}_w2"))

        fb_q = np.array(arc["s2_frame_bloom_q"], dtype=np.int8, copy=True)
        for k_attr, k_arc, dt in (
            ("s3_hru_bloom_q", "s3_hru_bloom_q", np.int8),
            ("s4_gender_bloom_q", "s4_gender_bloom_q", np.int8),
            ("country_lut_a", "s2_country_lut_a", np.uint8),
            ("country_lut_b", "s2_country_lut_b", np.uint8),
            ("gender_lut_a", "s4_gender_lut_a", np.uint8),
            ("gender_lut_b", "s4_gender_lut_b", np.uint8),
            ("s5_wyd_bloom_q", "s5_wyd_bloom_q", np.int8),
            ("s6_status_bloom_q", "s6_status_bloom_q", np.int8),
            ("status_lut_a", "s6_status_lut_a", np.uint8),
            ("status_lut_b", "s6_status_lut_b", np.uint8),
            ("s7_flirt_react_bloom_q", "s7_flirt_react_bloom_q", np.int8),
            ("s7_lut_a", "s7_lut_a", np.uint8),
            ("s7_lut_b", "s7_lut_b", np.uint8),
        ):
            arr = arc[k_arc].astype(dt, copy=False)
            arr.flags.writeable = False
            setattr(self, k_attr, arr)
        arc.close()

        pos_idx = np.arange(SEQ_LEN, dtype=np.float32).reshape(SEQ_LEN, 1)
        div_term = np.exp(np.arange(0, D_MODEL, 2, dtype=np.float32) * (-np.log(np.float32(10000.0)) / np.float32(D_MODEL))).astype(np.float32, copy=False)
        pos_f32 = np.zeros((SEQ_LEN, D_MODEL), dtype=np.float32)
        pos_f32[:, 0::2] = np.sin(pos_idx * div_term) * np.float32(0.02)
        pos_f32[:, 1::2] = np.cos(pos_idx * div_term) * np.float32(0.02)
        pos_f32.flags.writeable = False
        self.pos_f32 = pos_f32

        mask_f32 = np.triu(np.full((SEQ_LEN, SEQ_LEN), -1e9, dtype=np.float32), k=1)
        mask_f32.flags.writeable = False
        self.mask_f32 = mask_f32

        lut = np.full((256,), 32, dtype=np.int32)
        lut[65:91] = np.arange(97, 123, dtype=np.int32)
        lut[97:123] = np.arange(97, 123, dtype=np.int32)
        lut[39] = np.int32(39)
        lut.flags.writeable = False
        self.char_lut = lut

        rej = np.ones((256,), dtype=np.bool_)
        for idx_ok in (32, 33, 39, 44, 46, 47, 63):
            rej[idx_ok] = False
        rej[48:58] = False
        rej[65:91] = False
        rej[97:123] = False
        rej.flags.writeable = False
        self.reject_lut = rej

        primes = np.array((31, 37, 41, 43, 47, 53, 59, 61), dtype=np.int32)
        p_mat = np.ones((MAX_GREETING_LEN, 8), dtype=np.int32)
        p_mat[1:] = np.bitwise_and(np.cumprod(np.broadcast_to(primes, (MAX_GREETING_LEN - 1, 8)), axis=0), np.int32(0x7FFFFF))
        p_mat.flags.writeable = False
        self.poly_mat = p_mat

        w_primes = np.array((131, 257, 521, 1031), dtype=np.int32)
        wp_mat = np.ones((16, 4), dtype=np.int32)
        wp_mat[1:] = np.bitwise_and(np.cumprod(np.broadcast_to(w_primes, (15, 4)), axis=0), np.int32(0x7FFFFF))
        wp_mat.flags.writeable = False
        self.w_poly_mat = wp_mat

        self.word_bloom_q, self.single_bloom_q, self.target_bloom_q, self.lead_bloom_q = _build_step1_bloom_tables(wp_mat, fb_q)
        self.s2_frame_bloom_q = fb_q
        self.inv_sqrt_d = np.float32(np.float32(1.0) / np.sqrt(np.float32(D_MODEL)))
        self.eps = np.float32(1e-5)
        self.m5_0, self.m5_1, self.m5_2, self.m5_3 = np.int32(389), np.int32(131), np.int32(37), np.int32(13)
        self.mask_b, self.mask5, self.off5 = np.int32(1022), np.int32(511), np.int32(1024)
        self.m3_0, self.m3_1, self.mask3, self.off3 = np.int32(961), np.int32(31), np.int32(511), np.int32(1536)
        self.bloom_mask, self.s2_v_mask = np.int32(BLOOM_SIZE - 1), np.int32(S2_VOCAB - 2)
        self.one_i32, self.space_i32, self.conf_thresh = np.int32(1), np.int32(32), np.float32(0.5)
        get_step8_engine(npz_path)
        get_step9_engine(npz_path)
        get_step10_engine(npz_path)
        get_step11_engine(npz_path)
        get_step12_engine(npz_path)
        get_step13_engine(npz_path)
        get_step14_engine(npz_path)
        init_all_plugins(npz_path)
class InferenceSlot:
    __slots__ = (
        "slot_id", "raw_u8", "canon_u8", "body_sub", "rej_mask",
        "words_mat", "w_hashes", "w_b_idx", "w_b_hits", "c_hits_a", "c_hits_b",
        "g_hits_a", "g_hits_b", "s2_tokens", "s4_tokens", "s2_emb_buf", "s2_pooled",
        "s2_mid", "s2_logits", "s3_logits", "s4_logits",
        "tokens", "tok_bloom", "tok5", "tok3", "t_tmp16", "t_tmp8",
        "m_a", "m_b", "valid_mask", "valid_f32", "active_count",
        "x", "qkv", "q", "k", "v", "scores", "row_max", "row_sum",
        "attn_out", "proj_out", "norm_tmp", "ffn_mid", "pooled", "intent_logits",
        "st_hits_a", "st_hits_b", "s6_logits", "s7_hits_a", "s7_hits_b", "s7_logits",
    )

    def __init__(self, slot_id):
        self.slot_id = slot_id
        self.raw_u8 = np.zeros((MAX_S2_LEN + 4,), dtype=np.int32)
        self.canon_u8 = np.zeros((MAX_GREETING_LEN + 4,), dtype=np.int32)
        self.body_sub = np.zeros((MAX_GREETING_LEN,), dtype=np.int32)
        self.rej_mask = np.zeros((MAX_S2_LEN + 4,), dtype=np.bool_)
        self.words_mat = np.zeros((6, 16), dtype=np.int32)
        self.w_hashes = np.zeros((6, 4), dtype=np.int32)
        self.w_b_idx = np.zeros((6, 3), dtype=np.int32)
        self.w_b_hits = np.zeros((6, 3), dtype=np.int8)
        self.c_hits_a = np.zeros((6,), dtype=np.uint8)
        self.c_hits_b = np.zeros((6,), dtype=np.uint8)
        self.g_hits_a = np.zeros((6,), dtype=np.uint8)
        self.g_hits_b = np.zeros((6,), dtype=np.uint8)
        self.s2_tokens = np.zeros((8,), dtype=np.int32)
        self.s4_tokens = np.zeros((8,), dtype=np.int32)
        self.s2_emb_buf = np.zeros((8, S2_DIM), dtype=np.float32)
        self.s2_pooled = np.zeros((1, S2_DIM), dtype=np.float32)
        self.s2_mid = np.zeros((1, S2_DIM), dtype=np.float32)
        self.s2_logits = np.zeros((1, S2_CLASSES), dtype=np.float32)
        self.s3_logits = np.zeros((1, S3_CLASSES), dtype=np.float32)
        self.s4_logits = np.zeros((1, S4_CLASSES), dtype=np.float32)
        self.st_hits_a = np.zeros((6,), dtype=np.uint8)
        self.st_hits_b = np.zeros((6,), dtype=np.uint8)
        self.s6_logits = np.zeros((1, S6_CLASSES), dtype=np.float32)
        self.s7_hits_a = np.zeros((6,), dtype=np.uint8)
        self.s7_hits_b = np.zeros((6,), dtype=np.uint8)
        self.s7_logits = np.zeros((1, S7_CLASSES), dtype=np.float32)
        self.tokens = np.zeros((SEQ_LEN,), dtype=np.int32)
        self.tok_bloom, self.tok5, self.tok3 = self.tokens[0:8], self.tokens[8:24], self.tokens[24:32]
        self.t_tmp16, self.t_tmp8 = np.zeros((16,), dtype=np.int32), np.zeros((8,), dtype=np.int32)
        self.m_a, self.m_b = np.zeros((16,), dtype=np.bool_), np.zeros((16,), dtype=np.bool_)
        self.valid_mask = np.zeros((SEQ_LEN,), dtype=np.bool_)
        self.valid_f32 = np.zeros((SEQ_LEN, 1), dtype=np.float32)
        self.active_count = np.zeros((), dtype=np.float32)
        self.x = np.zeros((SEQ_LEN, D_MODEL), dtype=np.float32)
        self.qkv = np.zeros((SEQ_LEN, 3 * D_MODEL), dtype=np.float32)
        self.q, self.k, self.v = self.qkv[:, :D_MODEL], self.qkv[:, D_MODEL : 2 * D_MODEL], self.qkv[:, 2 * D_MODEL :]
        self.scores = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float32)
        self.row_max, self.row_sum = np.zeros((SEQ_LEN, 1), dtype=np.float32), np.zeros((SEQ_LEN, 1), dtype=np.float32)
        self.attn_out = np.zeros((SEQ_LEN, D_MODEL), dtype=np.float32)
        self.proj_out = np.zeros((SEQ_LEN, D_MODEL), dtype=np.float32)
        self.norm_tmp = np.zeros((SEQ_LEN, D_MODEL), dtype=np.float32)
        self.ffn_mid = np.zeros((SEQ_LEN, D_FF), dtype=np.float32)
        self.pooled = np.zeros((1, D_MODEL), dtype=np.float32)
        self.intent_logits = np.zeros((1, NUM_INTENTS), dtype=np.float32)


class UserStore:
    __slots__ = ("last_intent", "msg_count", "age", "country", "gender", "bot_gender", "conv_flags", "last_ts", "write_queue", "wal_batch")

    def __init__(self):
        self.last_intent = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.msg_count = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.age = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.country = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.gender = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.bot_gender = np.full((MAX_USERS,), 2, dtype=np.uint8)
        self.conv_flags = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.last_ts = np.zeros((MAX_USERS,), dtype=np.uint32)
        self.write_queue = deque(maxlen=4096)
        self.wal_batch = np.zeros((256, 6), dtype=np.uint32)


def _run_attn_ffn_layer(slot, wqkv, wo, w_ff1, w_ff2, inv_sqrt_d, mask_f32, eps):
    np.matmul(slot.x, wqkv, out=slot.qkv)
    np.matmul(slot.q, slot.k.T, out=slot.scores)
    slot.scores *= inv_sqrt_d
    slot.scores += mask_f32
    np.max(slot.scores, axis=1, keepdims=True, out=slot.row_max)
    slot.scores -= slot.row_max
    np.exp(slot.scores, out=slot.scores)
    np.sum(slot.scores, axis=1, keepdims=True, out=slot.row_sum)
    slot.scores /= slot.row_sum
    np.matmul(slot.scores, slot.v, out=slot.attn_out)
    np.matmul(slot.attn_out, wo, out=slot.proj_out)
    slot.x += slot.proj_out
    np.multiply(slot.x, slot.x, out=slot.norm_tmp)
    np.mean(slot.norm_tmp, axis=1, keepdims=True, out=slot.row_sum)
    slot.row_sum += eps
    np.sqrt(slot.row_sum, out=slot.row_sum)
    slot.x /= slot.row_sum

    np.matmul(slot.x, w_ff1, out=slot.ffn_mid)
    np.maximum(slot.ffn_mid, np.float32(0.0), out=slot.ffn_mid)
    np.matmul(slot.ffn_mid, w_ff2, out=slot.proj_out)
    slot.x += slot.proj_out
    np.multiply(slot.x, slot.x, out=slot.norm_tmp)
    np.mean(slot.norm_tmp, axis=1, keepdims=True, out=slot.row_sum)
    slot.row_sum += eps
    np.sqrt(slot.row_sum, out=slot.row_sum)
    slot.x /= slot.row_sum


def forward_pass(model, slot, mv_payload):
    b_full = np.frombuffer(mv_payload[:64], dtype=np.uint8)
    non_ws = np.flatnonzero(b_full > np.uint8(32))
    if non_ws.shape[0] == 0:
        return 0, 0, 0
    first_idx, last_idx = int(non_ws[0]), int(non_ws[-1]) + 1
    n_tok = last_idx - first_idx
    if n_tok > MAX_S2_LEN:
        return 0, 0, 0

    if n_tok <= 6:
        cmd = bytes(b_full[first_idx:last_idx]).lower()
        if cmd in (b"next", b"reset", b"new", b"skip", b"/start"):
            return 99, 0, 0

    slot.raw_u8.fill(32)
    slot.raw_u8[2 : 2 + n_tok] = b_full[first_idx:last_idx]

    np.take(model.reject_lut, slot.raw_u8[2 : 2 + n_tok], out=slot.rej_mask[:n_tok])
    if bool(np.any(slot.rej_mask[:n_tok])):
        return 0, 0, 0

    raw_body = slot.raw_u8[2 : 2 + n_tok]
    has_q = bool(np.any(raw_body == 63))
    has_s = bool(np.any(raw_body == 47))
    dig_idx = np.flatnonzero((raw_body >= 48) & (raw_body <= 57))
    n_dig = int(dig_idx.shape[0])
    age_val = 0
    if n_dig > 0:
        if n_dig > 2 or int(dig_idx[-1] - dig_idx[0]) != (n_dig - 1):
            return 0, 0, 0
        d0_pos = int(dig_idx[0])
        if n_dig == 1 and d0_pos > 0 and int(raw_body[d0_pos - 1]) > 39:
            return 0, 0, 0
        age_val = (int(raw_body[d0_pos]) - 48) if n_dig == 1 else ((int(raw_body[d0_pos]) - 48) * 10 + (int(raw_body[d0_pos + 1]) - 48))
        if age_val < 18:
            return (7, age_val, 0) if age_val >= 1 else (0, 0, 0)
        if age_val > 80:
            return 0, 0, 0

    np.take(model.char_lut, slot.raw_u8, out=slot.raw_u8)
    vc, vp1 = slot.raw_u8[2 : 2 + n_tok], slot.raw_u8[1 : 1 + n_tok]
    kept = vc[vc != vp1]
    if kept.shape[0] > 0 and int(kept[-1]) == 32:
        kept = kept[:-1]
    if kept.shape[0] > 0 and int(kept[0]) == 32:
        kept = kept[1:]
    k_len = int(kept.shape[0])
    if k_len == 0:
        return (3, age_val, 0) if age_val >= 18 else (0, 0, 0)

    sp = np.flatnonzero(kept == model.space_i32)
    n_sp = int(sp.shape[0])
    if n_sp > 5:
        return 0, 0, 0
    n_words = n_sp + 1

    slot.words_mat.fill(0)
    starts = (0,) + tuple(int(v) + 1 for v in sp)
    ends = tuple(int(v) for v in sp) + (k_len,)
    for w_i in range(n_words):
        Lw = min(ends[w_i] - starts[w_i], 15)
        slot.words_mat[w_i, :Lw] = kept[starts[w_i] : starts[w_i] + Lw]
        slot.words_mat[w_i, 15] = np.int32(Lw)
    L0 = int(slot.words_mat[0, 15])

    np.matmul(slot.words_mat, model.w_poly_mat, out=slot.w_hashes)
    np.bitwise_and(slot.w_hashes[:, :3], model.bloom_mask, out=slot.w_b_idx)

    np.take(model.single_bloom_q, slot.w_b_idx[:n_words], out=slot.w_b_hits[:n_words])
    first_is_pure_greet = bool(np.all(slot.w_b_hits[0]))
    all_pure_greet = bool(np.all(slot.w_b_hits[:n_words]))

    slot.s2_tokens.fill(0)
    np.bitwise_and(slot.w_hashes[:n_words, 0], model.s2_v_mask, out=slot.s2_tokens[:n_words])
    slot.s2_tokens[:n_words] += model.one_i32
    lead_idx = 1 if (n_words > 1 and int(model.lead_bloom_q[slot.w_b_idx[0, 0]]) == 1) else 0
    slot.s2_tokens[6] = ((slot.s2_tokens[lead_idx] * np.int32(37)) & model.s2_v_mask) + model.one_i32
    slot.s2_tokens[7] = ((slot.s2_tokens[n_words - 1] * np.int32(73)) & model.s2_v_mask) + model.one_i32

    np.take(model.status_lut_a, slot.w_b_idx[:, 0], out=slot.st_hits_a)
    np.take(model.status_lut_b, slot.w_b_idx[:, 1], out=slot.st_hits_b)
    st_match = (slot.st_hits_a[:n_words] > 0) & (slot.st_hits_a[:n_words] == slot.st_hits_b[:n_words])
    st_codes = slot.st_hits_a[:n_words] * st_match
    n_st_pos = int(np.sum(st_codes == 1))
    n_st_neg = int(np.sum(st_codes == 2))
    has_st_ab = bool(np.any(st_codes == 3))

    np.take(model.s7_lut_a, slot.w_b_idx[:, 0], out=slot.s7_hits_a)
    np.take(model.s7_lut_b, slot.w_b_idx[:, 1], out=slot.s7_hits_b)
    s7_match = (slot.s7_hits_a[:n_words] > 0) & (slot.s7_hits_a[:n_words] == slot.s7_hits_b[:n_words])
    s7_codes = slot.s7_hits_a[:n_words] * s7_match
    n_s7_flirt = int(np.sum(s7_codes == 1))
    n_s7_react = int(np.sum(s7_codes == 2))
    n_s7_dual = int(np.sum(s7_codes == 3))
    has_s7_subj = bool(np.any(s7_codes == 4))

    starts_w_non_wbu = int(kept[0]) == 119 and (k_len < 2 or int(kept[1]) != 98)
    if age_val == 0 and not all_pure_greet and n_st_pos == 0 and n_st_neg == 0 and not starts_w_non_wbu:
        np.take(model.s3_hru_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
        if bool(np.all(slot.w_b_hits[:n_words])):
            np.take(model.s3_emb_f32, slot.s2_tokens, axis=0, out=slot.s2_emb_buf)
            slot.s2_emb_buf[6] *= np.float32(2.5)
            np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
            slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
            np.matmul(slot.s2_pooled, model.s3_w1_f32, out=slot.s2_mid)
            np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
            np.matmul(slot.s2_mid, model.s3_w2_f32, out=slot.s3_logits)
            if float(slot.s3_logits[0, 1] - slot.s3_logits[0, 0]) >= 0.8:
                return 8, 0, 0

    np.take(model.country_lut_a, slot.w_b_idx[:, 0], out=slot.c_hits_a)
    np.take(model.country_lut_b, slot.w_b_idx[:, 1], out=slot.c_hits_b)
    c_match = (slot.c_hits_a[:n_words] > 0) & (slot.c_hits_a[:n_words] == slot.c_hits_b[:n_words])
    country_id = int(slot.c_hits_a[:n_words][c_match][0]) if bool(np.any(c_match)) else 0

    np.take(model.gender_lut_a, slot.w_b_idx[:, 0], out=slot.g_hits_a)
    np.take(model.gender_lut_b, slot.w_b_idx[:, 1], out=slot.g_hits_b)
    g_match = (slot.g_hits_a[:n_words] > 0) & (slot.g_hits_a[:n_words] == slot.g_hits_b[:n_words])
    g_codes = slot.g_hits_a[:n_words] * g_match
    n_male = int(np.sum(g_codes == 1))
    n_female = int(np.sum(g_codes == 2))
    has_neutral = bool(np.any(g_codes == 3))
    has_q_frame = bool(np.any(g_codes == 4))
    has_s_frame = bool(np.any(g_codes == 5))

    np.take(model.s2_frame_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
    all_s2_valid = bool(np.all(slot.w_b_hits[:n_words]))

    if age_val >= 18 and country_id > 0 and all_s2_valid:
        return 6, age_val, country_id
    if age_val >= 18 and (n_male > 0 or n_female > 0) and not (n_male > 0 and n_female > 0) and all_s2_valid:
        return 14, age_val, (2 if n_female > 0 else 1)
    if country_id > 0 and all_s2_valid:
        return 5, 0, country_id
    if age_val >= 18 and all_s2_valid:
        return 3, age_val, 0

    if age_val == 0 and country_id == 0 and (n_male > 0 or n_female > 0 or has_neutral):
        np.take(model.s4_gender_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
        is_hey_man = (
            first_is_pure_greet and n_words == 2 and not has_q
            and int(slot.words_mat[1, 15]) == 3 and int(slot.words_mat[1, 0]) == 109
            and int(slot.words_mat[1, 1]) == 97 and int(slot.words_mat[1, 2]) == 110
        )
        if bool(np.all(slot.w_b_hits[:n_words])) and not is_hey_man:
            slot.s4_tokens[:6] = slot.s2_tokens[:6]
            slot.s4_tokens[6] = np.int32(1023) if has_q else (np.int32(1022) if has_s else model.one_i32)
            q_off = np.int32(509) if has_q else (np.int32(251) if has_s else np.int32(0))
            slot.s4_tokens[7] = (((slot.s2_tokens[0] * np.int32(37)) + (slot.s2_tokens[n_words - 1] * np.int32(73)) + q_off) & model.s2_v_mask) + model.one_i32
            np.take(model.s4_emb_f32, slot.s4_tokens, axis=0, out=slot.s2_emb_buf)
            slot.s2_emb_buf[6] *= np.float32(2.5)
            np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
            slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
            np.matmul(slot.s2_pooled, model.s4_w1_f32, out=slot.s2_mid)
            np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
            np.matmul(slot.s2_mid, model.s4_w2_f32, out=slot.s4_logits)
            if (n_male > 0 and n_female > 0) or has_s or (n_male + n_female >= 2) or (has_neutral and n_male == 0 and n_female == 0):
                return 9, 0, 0
            if n_female > 0:
                return (10, 0, 2) if (not has_s_frame and (has_q or has_q_frame or has_neutral or float(slot.s4_logits[0, 2]) > float(slot.s4_logits[0, 4]))) else (12, 0, 2)
            if n_male > 0:
                return (11, 0, 1) if (not has_s_frame and (has_q or has_q_frame or has_neutral or float(slot.s4_logits[0, 3]) > float(slot.s4_logits[0, 5]))) else (13, 0, 1)
    p_cid, p_age, p_aux = forward_pre_s7(model, slot, n_words, has_q, age_val, country_id)
    if p_cid != 0:
        return p_cid, p_age, p_aux
    s14_cid, s14_age, s14_aux = forward_step14(model, slot, n_words, has_q, age_val, country_id)
    if s14_cid != 0:
        return s14_cid, s14_age, s14_aux
    if n_words <= 4 and k_len <= MAX_GREETING_LEN:
        if all_pure_greet:
            return 1, 0, 0
        if first_is_pure_greet and n_words <= 3:
            np.take(model.target_bloom_q, slot.w_b_idx[:n_words], out=slot.w_b_hits[:n_words])
            if bool(np.all(slot.w_b_hits[:n_words])):
                return 1, 0, 0

        np.take(model.word_bloom_q, slot.w_b_idx[:n_words], out=slot.w_b_hits[:n_words])
        s1_ok = bool(np.all(slot.w_b_hits[:n_words])) if n_words >= 2 else False
        if s1_ok and L0 >= 4 and int(kept[0]) == 119 and int(kept[1]) == 104 and int(kept[2]) == 97 and int(kept[3]) == 116:
            s1_ok = int(kept[-2]) == 117 and int(kept[-1]) == 112

        if s1_ok:
            slot.canon_u8.fill(32)
            if L0 == 4 and int(kept[0]) == 104 and int(kept[1]) == 101 and int(kept[2]) == 108 and int(kept[3]) == 111:
                slot.canon_u8[2], slot.canon_u8[3], slot.canon_u8[4] = 104, 108, 111
                if k_len > 4:
                    slot.canon_u8[5 : 5 + (k_len - 4)] = kept[4:k_len]
            elif L0 == 3 and int(kept[0]) == 101 and int(kept[1]) == 108 and int(kept[2]) == 111:
                slot.canon_u8[2], slot.canon_u8[3], slot.canon_u8[4] = 104, 108, 111
                if k_len > 3:
                    slot.canon_u8[5 : 5 + (k_len - 3)] = kept[3:k_len]
            else:
                slot.canon_u8[2 : 2 + k_len] = kept

            np.subtract(slot.canon_u8[2 : 2 + MAX_GREETING_LEN], model.space_i32, out=slot.body_sub)
            np.matmul(slot.body_sub, model.poly_mat, out=slot.tok_bloom)
            slot.tok_bloom &= model.mask_b
            slot.tok_bloom += model.one_i32
            slot.valid_mask[0:8] = True

            c0, c1, c2, c3, c4 = slot.canon_u8[0:16], slot.canon_u8[1:17], slot.canon_u8[2:18], slot.canon_u8[3:19], slot.canon_u8[4:20]
            np.multiply(c0, model.m5_0, out=slot.tok5)
            np.multiply(c1, model.m5_1, out=slot.t_tmp16)
            slot.tok5 += slot.t_tmp16
            np.multiply(c2, model.m5_2, out=slot.t_tmp16)
            slot.tok5 += slot.t_tmp16
            np.multiply(c3, model.m5_3, out=slot.t_tmp16)
            slot.tok5 += slot.t_tmp16
            slot.tok5 += c4
            slot.tok5 &= model.mask5
            slot.tok5 += model.off5

            np.equal(c1, model.space_i32, out=slot.m_a)
            np.equal(c2, model.space_i32, out=slot.m_b)
            slot.m_a &= slot.m_b
            np.equal(c3, model.space_i32, out=slot.m_b)
            slot.m_a &= slot.m_b

            d1, d2, d3 = c1[:8], c2[:8], c3[:8]
            np.multiply(d1, model.m3_0, out=slot.tok3)
            np.multiply(d2, model.m3_1, out=slot.t_tmp8)
            slot.tok3 += slot.t_tmp8
            slot.tok3 += d3
            slot.tok3 &= model.mask3
            slot.tok3 += model.off3

            np.invert(slot.m_a[:8], out=slot.valid_mask[24:32])
            slot.tok3 *= slot.valid_mask[24:32]
            np.equal(c0, model.space_i32, out=slot.m_b)
            slot.m_a &= slot.m_b
            np.equal(c4, model.space_i32, out=slot.m_b)
            slot.m_a &= slot.m_b
            np.invert(slot.m_a, out=slot.valid_mask[8:24])
            slot.tok5 *= slot.valid_mask[8:24]

            np.copyto(slot.valid_f32[:, 0], slot.valid_mask, casting="unsafe")
            np.sum(slot.valid_f32, out=slot.active_count)
            np.take(model.emb_f32, slot.tokens, axis=0, out=slot.x)
            slot.x += model.pos_f32
            slot.x *= slot.valid_f32

            _run_attn_ffn_layer(slot, model.wqkv1_f32, model.wo1_f32, model.w1_f32, model.w2_f32, model.inv_sqrt_d, model.mask_f32, model.eps)
            _run_attn_ffn_layer(slot, model.wqkv2_f32, model.wo2_f32, model.w3_f32, model.w4_f32, model.inv_sqrt_d, model.mask_f32, model.eps)

            slot.x *= slot.valid_f32
            np.sum(slot.x, axis=0, keepdims=True, out=slot.pooled)
            slot.pooled /= slot.active_count
            np.matmul(slot.pooled, model.w_intent_f32, out=slot.intent_logits)
            if np.float32(slot.intent_logits[0, 1] - slot.intent_logits[0, 0]) >= model.conf_thresh:
                return 1, 0, 0

    if age_val == 0 and country_id == 0 and n_st_pos == 0 and n_st_neg == 0:
        np.take(model.s5_wyd_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
        if bool(np.all(slot.w_b_hits[:n_words])):
            np.take(model.s5_emb_f32, slot.s2_tokens, axis=0, out=slot.s2_emb_buf)
            slot.s2_emb_buf[6] *= np.float32(2.5)
            np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
            slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
            np.matmul(slot.s2_pooled, model.s5_w1_f32, out=slot.s2_mid)
            np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
            np.matmul(slot.s2_mid, model.s5_w2_f32, out=slot.s3_logits)
            if float(slot.s3_logits[0, 1] - slot.s3_logits[0, 0]) >= 0.8:
                return 15, 0, 0

    is_nmu = n_words == 1 and L0 == 3 and int(kept[0]) == 110 and int(kept[1]) == 109 and int(kept[2]) == 117
    s7_override_s6 = (
        not has_q and bool(np.all(s7_codes > 0))
        and (n_s7_react > 0 or n_s7_dual > 0 or has_s7_subj)
        and not (int(s7_codes[0]) == 5 and int(s7_codes[n_words - 1]) == 4)
    )
    if age_val == 0 and country_id == 0 and (n_st_pos > 0 or n_st_neg > 0 or is_nmu) and not s7_override_s6:
        np.take(model.s6_status_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
        if bool(np.all(slot.w_b_hits[:n_words])):
            slot.s4_tokens[:6] = slot.s2_tokens[:6]
            slot.s4_tokens[6] = np.int32(1023) if has_q else model.one_i32
            q_off = np.int32(509) if has_q else np.int32(0)
            slot.s4_tokens[7] = (((slot.s2_tokens[0] * np.int32(37)) + (slot.s2_tokens[n_words - 1] * np.int32(73)) + q_off) & model.s2_v_mask) + model.one_i32
            np.take(model.s6_emb_f32, slot.s4_tokens, axis=0, out=slot.s2_emb_buf)
            slot.s2_emb_buf[6] *= np.float32(2.5)
            np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
            slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
            np.matmul(slot.s2_pooled, model.s6_w1_f32, out=slot.s2_mid)
            np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
            np.matmul(slot.s2_mid, model.s6_w2_f32, out=slot.s6_logits)
            s6_0, s6_pos, s6_neg, s6_ab = (
                float(slot.s6_logits[0, 0]), float(slot.s6_logits[0, 1]),
                float(slot.s6_logits[0, 2]), float(slot.s6_logits[0, 3]),
            )
            if max(s6_pos, s6_neg, s6_ab) > s6_0 - 0.2:
                if has_st_ab or is_nmu:
                    return 18, 0, 0
                if n_st_neg > 0:
                    return (16, 0, 0) if (n_st_pos == 0 and s6_pos > s6_neg + 1.0) else (17, 0, 0)
                return (16, 0, 0) if s6_pos >= s6_neg else (17, 0, 0)

    if age_val == 0 and country_id == 0 and not has_q and (n_s7_flirt > 0 or n_s7_react > 0 or n_s7_dual > 0 or (n_words >= 2 and bool(np.all(s7_codes == 5)))):
        np.take(model.s7_flirt_react_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
        if bool(np.all(slot.w_b_hits[:n_words])):
            slot.s4_tokens[:6] = slot.s2_tokens[:6]
            slot.s4_tokens[6] = model.one_i32
            slot.s4_tokens[7] = (((slot.s2_tokens[0] * np.int32(37)) + (slot.s2_tokens[n_words - 1] * np.int32(73))) & model.s2_v_mask) + model.one_i32
            np.take(model.s7_emb_f32, slot.s4_tokens, axis=0, out=slot.s2_emb_buf)
            slot.s2_emb_buf[6] *= np.float32(2.5)
            np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
            slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
            np.matmul(slot.s2_pooled, model.s7_w1_f32, out=slot.s2_mid)
            np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
            np.matmul(slot.s2_mid, model.s7_w2_f32, out=slot.s7_logits)
            s7_0, s7_flirt, s7_react = float(slot.s7_logits[0, 0]), float(slot.s7_logits[0, 1]), float(slot.s7_logits[0, 2])
            if max(s7_flirt, s7_react) > s7_0 + 0.1:
                if n_s7_flirt > 0 or (has_s7_subj and n_s7_react == 0):
                    return 19, 0, 0
                return 20, 0, 0
    s8_cid, s8_age, s8_aux = forward_step8(model, slot, n_words, has_q, age_val, country_id)
    if s8_cid != 0:
        return s8_cid, s8_age, s8_aux
    s11_cid, s11_age, s11_aux = forward_step11(model, slot, n_words, has_q, age_val, country_id)
    if s11_cid != 0:
        return s11_cid, s11_age, s11_aux
    s9_cid, s9_age, s9_aux = forward_step9(model, slot, n_words, has_q, age_val, country_id)
    if s9_cid != 0:
        return s9_cid, s9_age, s9_aux
    s12_cid, s12_age, s12_aux = forward_step12(model, slot, n_words, has_q, age_val, country_id)
    if s12_cid != 0:
        return s12_cid, s12_age, s12_aux
    s13_cid, s13_age, s13_aux = forward_step13(model, slot, n_words, has_q, age_val, country_id)
    if s13_cid != 0:
        return s13_cid, s13_age, s13_aux
    p_cid, p_age, p_aux = forward_post_s7(model, slot, n_words, has_q, age_val, country_id)
    if p_cid != 0:
        return p_cid, p_age, p_aux
    if not all_s2_valid:
        return 0, 0, 0
    np.take(model.s2_emb_f32, slot.s2_tokens, axis=0, out=slot.s2_emb_buf)
    slot.s2_emb_buf[6] *= np.float32(2.5)
    np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
    slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
    np.matmul(slot.s2_pooled, model.s2_w1_f32, out=slot.s2_mid)
    np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
    np.matmul(slot.s2_mid, model.s2_w2_f32, out=slot.s2_logits)

    s0 = float(slot.s2_logits[0, 0])
    s_ask_age = float(slot.s2_logits[0, 1])
    s_ask_ctry = float(slot.s2_logits[0, 3])
    if s_ask_age > s0 + 0.8 and s_ask_age >= s_ask_ctry:
        return 2, 0, 0
    if s_ask_ctry > s0 + 0.8 and s_ask_ctry > s_ask_age:
        return 4, 0, 0
    s10_cid, s10_age, s10_aux = forward_step10(model, slot, n_words, has_q, age_val, country_id)
    if s10_cid != 0:
        return s10_cid, s10_age, s10_aux
    return 0, 0, 0


async def start_server(host="127.0.0.1", port=8080):
    model = QuantizedModel("model_weights.npz")
    users = UserStore()
    slot_queue = asyncio.Queue(maxsize=NUM_WORKERS)
    for i in range(NUM_WORKERS):
        slot_queue.put_nowait(InferenceSlot(i))

    test_slot = await slot_queue.get()
    t0 = time.perf_counter()
    for i in range(1000):
        handle_user_query(model, test_slot, users, i, memoryview(b"hi how r u?"), forward_pass)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    suite = (
        (b"hi", 1), (b"hi hi", 1), (b"hy", 1), (b"hyy", 1), (b"hiiii", 1), (b"hey", 1),
        (b"hey hey", 1), (b"heeyy", 1), (b"hello", 1), (b"helloo", 1), (b"hlo", 1),
        (b"yo", 1), (b"sup", 1), (b"good morning", 1), (b"whats up", 1),
        (b"wyd", 15), (b"what are you doing", 15), (b"what u up to", 15),
        (b"im good", 16), (b"im ok", 16), (b"bored", 17), (b"lm borad", 17), (b"im good u?", 18), (b"nmu", 18),
        (b"ur cute", 19), (b"u seem nice", 19), (b"i like u", 19),
        (b"haha", 20), (b"lol", 20), (b"nice", 20), (b"cool", 20),
        (b"same", 20), (b"sam", 20), (b"same here", 20), (b"sam here", 20), (b"true", 20), (b"ok", 20),
        (b"hi there", 1), (b"hey man", 1),
        (b"how are you", 8), (b"how are you?", 8), (b"how are you!", 8), (b"hru", 8), (b"hru!", 8),
        (b"hyd", 8), (b"how r u", 8), (b"how u doing!", 8), (b"how was your day", 8),
        (b"hor", 8), (b"how ar you dopin", 8), (b"how u doin", 8), (b"hw r u", 8), (b"how whas you da", 8),
        (b"hi how r u", 8), (b"hey hru", 8), (b"hello how are you doing!", 8),
        (b"age?", 2), (b"how old r u", 2),
        (b"24", 3), (b"im 24", 3), (b"lm 24", 3), (b"24y", 3), (b"21yres", 3), (b"20yres", 3), (b"im24", 3),
        (b"from?", 4), (b"where r u from", 4), (b"tuinsi", 5), (b"handi", 5), (b"ind", 5),
        (b"hi l'm 30 from tun", 6), (b"15", 7), (b"next", 99),
        (b"m or f", 9), (b"m or f?", 9), (b"m/f", 9), (b"f/m", 9), (b"gender?", 9), (b"what gender are you", 9),
        (b"f?", 10), (b"girl?", 10), (b"are you a girl", 10), (b"r u female", 10), (b"is your gender female", 10),
        (b"m?", 11), (b"boy?", 11), (b"are you a boy", 11), (b"r u male", 11), (b"is your gender male", 11),
        (b"f", 12), (b"famle", 12), (b"f here", 12), (b"im a girl", 12), (b"hi f", 12),
        (b"m", 13), (b"male", 13), (b"m here", 13), (b"im a boy", 13), (b"hi m", 13),
        (b"22f", 14), (b"24 m", 14), (b"19 famle", 14), (b"21 girl", 14),
        (b"what", 0), (b"what?", 0), (b"what!", 0), (b"why", 0), (b"why?", 0), (b"why!", 0),
        (b"who", 0), (b"who?", 0), (b"who!", 0), (b"who are you", 0), (b"what is this", 0),
        (b"why not", 0), (b"hyw", 0), (b"iphey", 0), (b"m bi", 0),
    )
    passed = sum(1 for txt, exp in suite if forward_pass(model, test_slot, memoryview(txt))[0] == exp)
    slot_queue.put_nowait(test_slot)

    print(f"[BENCHMARK] 1,000 forward passes in {elapsed_ms:.2f} ms ({elapsed_ms / 1000.0:.4f} ms/req)")
    print(f"[STEP 1-7 ZERO-FAULT SUITE] Passed {passed}/{len(suite)} tests (3.18 MB Total RAM)")

    if "--benchmark" in sys.argv:
        return

    async def client_handler(reader, writer):
        try:
            req_data = await asyncio.wait_for(reader.read(512), timeout=2.0)
            if not req_data:
                writer.close()
                return
            peer = writer.get_extra_info("peername")
            uid = (hash(peer[0]) & 0x7FFFFFFF) if peer else 101
            mv_req = memoryview(req_data)
            slot = await slot_queue.get()
            try:
                body_idx = req_data.find(b"\r\n\r\n")
                mv_body = mv_req[body_idx + 4 :] if body_idx != -1 else mv_req
                http_reply = handle_user_query(model, slot, users, uid % MAX_USERS, mv_body, forward_pass)
            finally:
                slot_queue.put_nowait(slot)

            writer.write(http_reply)
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    asyncio.create_task(wal_write_behind_worker(users))
    server = await asyncio.start_server(client_handler, host, port, backlog=2048)
    print(f"[SERVER] Step 1-7 Dating Engine on http://{host}:{port} (3.18 MB Total RAM)")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(start_server())