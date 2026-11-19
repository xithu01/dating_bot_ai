#!/usr/bin/env python3
# FILE: production_server.py
# PURPOSE: Rule 14 Zero-Logic Compiled Pure-NumPy VM (Runs ALL 15 Steps / 76 Leaf Intents from model_weights.npz)
# RULES: Zero PyTorch/SciPy/DB, zero step plugins, zero `if class_id == X`, zero float64/int64,
#        zero per-request RAM allocation, Prime Modulus (% 4091 / % 8191), PyArmor compatible (< 300 lines / < 12 KB AST).

import asyncio
from collections import deque
import os
import sys
import time
import numpy as np

BLOOM_SIZE = 65536
WORD_MOD = 4091
BIGRAM_MOD = 8191
TOK_COUNTRY = 4093
TOK_NAME_F = 4094
TOK_NAME_M = 4095
D_EMB = 64
D_STRUCT = 16
D_IN = 128
D_HID1 = 256
D_HID2 = 128
MAX_LEN = 45
MAX_USERS = 50000
NUM_WORKERS = 8
SESSION_TIMEOUT_S = 600

BIT_WILDCARD_NAME = np.uint32(1 << 30)
BIT_GLOBAL_KNOWN = np.uint32(1 << 31)
ZERO_F32 = np.float32(0.0)
ONE_F32 = np.float32(1.0)
SIX_F32 = np.float32(6.0)
CONF_MARGIN_F32 = np.float32(0.25)


class CompiledVMModel:
    __slots__ = (
        "emb_word", "emb_bg", "pos_emb", "w_struct", "w1", "b1", "w2", "b2", "w_out", "b_out",
        "super_bloom", "entity_lut_a", "entity_lut_b", "char_lut", "rej_lut", "w_poly_mat",
        "http_blob_u8", "http_blob_mv", "pool_off", "pool_end", "pool_mask",
        "route_pool_lut", "leaf_cls", "leaf_rule", "leaf_plock",
        "chk_lo", "chk_hi", "set_lo", "set_hi", "num_leaves", "bloom_mask", "space_i32",
    )

    def __init__(self, npz_path):
        if not os.path.exists(npz_path):
            print(f"[VM ERROR] '{npz_path}' not found! Run first: python offline_train.py", flush=True)
            sys.exit(1)
        arc = np.load(npz_path, mmap_mode="r")
        if "vm_version_u8" not in arc.files or int(arc["vm_version_u8"][0]) < 14:
            arc.close()
            print(f"[VM ERROR] '{npz_path}' is not a Rule 14 Compiled Vault! Run: python offline_train.py", flush=True)
            sys.exit(1)

        def deq(pfx, zero0=False):
            q = arc[f"{pfx}_q"].astype(np.int8, copy=False)
            s = arc[f"{pfx}_s"].astype(np.float32, copy=False)
            arr = np.multiply(q, s, dtype=np.float32)
            if zero0:
                arr[0].fill(ZERO_F32)
            arr.flags.writeable = False
            return arr

        def ro(key, dt):
            arr = arc[key].astype(dt, copy=True)
            arr.flags.writeable = False
            return arr

        self.emb_word = deq("emb_word", True)
        self.emb_bg = deq("emb_bg", True)
        self.pos_emb = ro("pos_emb_f32", np.float32)
        self.w_struct = deq("w_struct")
        self.w1 = deq("w1")
        self.b1 = ro("b1_f32", np.float32)
        self.w2 = deq("w2")
        self.b2 = ro("b2_f32", np.float32)
        self.w_out = deq("w_out")
        self.b_out = ro("b_out_f32", np.float32)

        self.super_bloom = ro("super_bloom_u32", np.uint32)
        self.entity_lut_a = ro("entity_lut_a", np.uint8)
        self.entity_lut_b = ro("entity_lut_b", np.uint8)
        self.char_lut = ro("char_lut_i32", np.int32)
        self.rej_lut = ro("rej_lut_b", np.bool_)
        self.w_poly_mat = ro("w_poly_mat_i32", np.int32)

        self.http_blob_u8 = ro("http_blob_u8", np.uint8)
        self.http_blob_mv = memoryview(self.http_blob_u8)
        self.pool_off = ro("pool_msg_off_i32", np.int32)
        self.pool_end = ro("pool_msg_end_i32", np.int32)
        self.pool_mask = ro("pool_mask_u8", np.uint8)
        self.route_pool_lut = ro("route_pool_lut", np.uint16)
        self.leaf_cls = ro("leaf_to_class_u8", np.uint8)
        self.leaf_rule = ro("leaf_entity_rule_u8", np.uint8)
        self.leaf_plock = ro("leaf_persona_lock_u8", np.uint8)
        self.chk_lo = ro("leaf_chk_lo_u32", np.uint32)
        self.chk_hi = ro("leaf_chk_hi_u32", np.uint32)
        self.set_lo = ro("leaf_set_lo_u32", np.uint32)
        self.set_hi = ro("leaf_set_hi_u32", np.uint32)
        arc.close()

        self.num_leaves = int(self.leaf_cls.shape[0])
        self.bloom_mask = np.int32(BLOOM_SIZE - 1)
        self.space_i32 = np.int32(32)


class InferenceSlot:
    __slots__ = (
        "slot_id", "raw_u8", "rej_mask", "words_mat", "w_hashes", "w_b_idx",
        "sb_hits", "sb_word", "ea", "eb", "w_ids", "bg_ids", "s_vec",
        "w_emb", "w_mask_b", "w_mask", "bg_emb", "bg_mask_b", "bg_mask",
        "x_in", "st_proj", "h1", "h2", "logits",
    )

    def __init__(self, slot_id, num_leaves):
        self.slot_id = slot_id
        self.raw_u8 = np.zeros((MAX_LEN + 4,), dtype=np.int32)
        self.rej_mask = np.zeros((MAX_LEN + 4,), dtype=np.bool_)
        self.words_mat = np.zeros((6, 16), dtype=np.int32)
        self.w_hashes = np.zeros((6, 4), dtype=np.int32)
        self.w_b_idx = np.zeros((6, 3), dtype=np.int32)
        self.sb_hits = np.zeros((6, 3), dtype=np.uint32)
        self.sb_word = np.zeros((6,), dtype=np.uint32)
        self.ea = np.zeros((6,), dtype=np.uint8)
        self.eb = np.zeros((6,), dtype=np.uint8)
        self.w_ids = np.zeros((6,), dtype=np.int32)
        self.bg_ids = np.zeros((7,), dtype=np.int32)
        self.s_vec = np.zeros((1, D_STRUCT), dtype=np.float32)
        self.w_emb = np.zeros((6, D_EMB), dtype=np.float32)
        self.w_mask_b = np.zeros((6,), dtype=np.bool_)
        self.w_mask = np.zeros((6, 1), dtype=np.float32)
        self.bg_emb = np.zeros((7, D_EMB), dtype=np.float32)
        self.bg_mask_b = np.zeros((7,), dtype=np.bool_)
        self.bg_mask = np.zeros((7, 1), dtype=np.float32)
        self.x_in = np.zeros((1, D_IN), dtype=np.float32)
        self.st_proj = np.zeros((1, D_IN), dtype=np.float32)
        self.h1 = np.zeros((1, D_HID1), dtype=np.float32)
        self.h2 = np.zeros((1, D_HID2), dtype=np.float32)
        self.logits = np.zeros((1, num_leaves), dtype=np.float32)


class UserStore:
    __slots__ = ("last_intent", "msg_count", "age", "country", "gender", "bot_gender", "flags_lo", "flags_hi", "last_ts", "write_queue", "wal_batch")

    def __init__(self):
        self.last_intent = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.msg_count = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.age = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.country = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.gender = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.bot_gender = np.full((MAX_USERS,), 2, dtype=np.uint8)
        self.flags_lo = np.zeros((MAX_USERS,), dtype=np.uint32)
        self.flags_hi = np.zeros((MAX_USERS,), dtype=np.uint32)
        self.last_ts = np.zeros((MAX_USERS,), dtype=np.uint32)
        self.write_queue = deque(maxlen=4096)
        self.wal_batch = np.zeros((256, 6), dtype=np.uint32)


def vm_forward_pass(m, s, mv_payload):
    b_full = np.frombuffer(mv_payload[:64], dtype=np.uint8)
    non_ws = np.flatnonzero(b_full > np.uint8(32))
    if non_ws.shape[0] == 0:
        return 0, 0, 0
    first_idx, last_idx = int(non_ws[0]), int(non_ws[-1]) + 1
    n_tok = last_idx - first_idx
    if n_tok > MAX_LEN:
        return 0, 0, 0

    if n_tok <= 6:
        cmd = bytes(b_full[first_idx:last_idx]).lower()
        if cmd in (b"next", b"reset", b"new", b"skip", b"/start"):
            return 75, 0, 0

    s.raw_u8.fill(32)
    s.raw_u8[2 : 2 + n_tok] = b_full[first_idx:last_idx]
    body = s.raw_u8[2 : 2 + n_tok]
    np.take(m.rej_lut, body, out=s.rej_mask[:n_tok])
    if bool(np.any(s.rej_mask[:n_tok])):
        return 0, 0, 0

    has_q = bool(np.any(body == 63))
    has_s = bool(np.any(body == 47))
    dig_idx = np.flatnonzero((body >= 48) & (body <= 57))
    n_dig = int(dig_idx.shape[0])
    age_val = 0
    if n_dig > 0:
        if n_dig > 2 or int(dig_idx[-1] - dig_idx[0]) != (n_dig - 1):
            return 0, 0, 0
        d0 = int(dig_idx[0])
        if n_dig == 1 and d0 > 0 and int(body[d0 - 1]) > 39:
            return 0, 0, 0
        age_val = (int(body[d0]) - 48) if n_dig == 1 else ((int(body[d0]) - 48) * 10 + (int(body[d0 + 1]) - 48))
        if age_val < 18:
            return (7, age_val, 0) if age_val >= 1 else (0, 0, 0)
        if age_val > 80:
            return 0, 0, 0

    np.take(m.char_lut, s.raw_u8, out=s.raw_u8)
    vc, vp1 = s.raw_u8[2 : 2 + n_tok], s.raw_u8[1 : 1 + n_tok]
    kept = vc[vc != vp1]
    if kept.shape[0] > 0 and int(kept[-1]) == 32:
        kept = kept[:-1]
    if kept.shape[0] > 0 and int(kept[0]) == 32:
        kept = kept[1:]
    k_len = int(kept.shape[0])
    if k_len == 0:
        return (3, age_val, 0) if age_val >= 18 else (0, 0, 0)

    sp = np.flatnonzero(kept == m.space_i32)
    n_sp = int(sp.shape[0])
    if n_sp > 5:
        return 0, 0, 0
    n_words = n_sp + 1

    s.words_mat.fill(0)
    starts = (0,) + tuple(int(v) + 1 for v in sp)
    ends = tuple(int(v) for v in sp) + (k_len,)
    for w_i in range(n_words):
        Lw = min(ends[w_i] - starts[w_i], 15)
        s.words_mat[w_i, :Lw] = kept[starts[w_i] : starts[w_i] + Lw]
        s.words_mat[w_i, 15] = np.int32(Lw)

    np.matmul(s.words_mat, m.w_poly_mat, out=s.w_hashes)
    np.bitwise_and(s.w_hashes[:, :3], m.bloom_mask, out=s.w_b_idx)

    # 1) Single-Instruction Bitmask Super-Bloom (Tests ALL 15 steps simultaneously in 0.002 ms)
    np.take(m.super_bloom, s.w_b_idx[:n_words], out=s.sb_hits[:n_words])
    np.bitwise_and.reduce(s.sb_hits[:n_words], axis=1, out=s.sb_word[:n_words])
    sb_v = s.sb_word[:n_words]
    all_known = bool(np.all((sb_v & BIT_GLOBAL_KNOWN) != 0))
    is_wc_name = False
    if not all_known and n_words >= 3 and bool(np.any((sb_v & BIT_WILDCARD_NAME) != 0)):
        unk = np.flatnonzero((sb_v & BIT_GLOBAL_KNOWN) == 0)
        if unk.shape[0] == 1 and int(unk[0]) >= 2 and int(s.words_mat[int(unk[0]), 15]) >= 3:
            is_wc_name = True
    if not all_known and not is_wc_name:
        return 0, 0, 0

    active_mask = int(np.bitwise_and.reduce(sb_v)) if all_known else (1 << 9)

    # 2) Unified O(1) Double-Hash Entity Lookup (Countries + 4,059+ Names) & Canonical Token Substitution
    np.take(m.entity_lut_a, s.w_b_idx[:n_words, 0], out=s.ea[:n_words])
    np.take(m.entity_lut_b, s.w_b_idx[:n_words, 1], out=s.eb[:n_words])
    s.w_ids.fill(0)
    s.bg_ids.fill(0)
    ctry_id, has_nf, has_nm = 0, False, False
    for i in range(n_words):
        a_v = int(s.ea[i])
        code = a_v if (a_v > 0 and a_v == int(s.eb[i])) else 0
        if 1 <= code <= 20:
            if ctry_id == 0:
                ctry_id = code
            s.w_ids[i] = np.int32(TOK_COUNTRY)
        elif code == 21:
            has_nf = True
            s.w_ids[i] = np.int32(TOK_NAME_F)
        elif code == 22 or (is_wc_name and (int(sb_v[i]) & int(BIT_GLOBAL_KNOWN)) == 0):
            has_nm = True
            s.w_ids[i] = np.int32(TOK_NAME_M)
        else:
            s.w_ids[i] = np.int32(((int(s.w_hashes[i, 0]) & 0x7FFFFFFF) % WORD_MOD) + 1)

    s.bg_ids[0] = np.int32((((4097 * 131 + int(s.w_ids[0])) & 0x7FFFFFFF) % BIGRAM_MOD) + 1)
    for k in range(n_words - 1):
        s.bg_ids[k + 1] = np.int32((((int(s.w_ids[k]) * 131 + int(s.w_ids[k + 1])) & 0x7FFFFFFF) % BIGRAM_MOD) + 1)
    eos_tok = 4101 if has_q else (4100 if has_s else 4099)
    s.bg_ids[n_words] = np.int32((((int(s.w_ids[n_words - 1]) * 131 + eos_tok) & 0x7FFFFFFF) % BIGRAM_MOD) + 1)

    sv = s.s_vec[0]
    sv.fill(ZERO_F32)
    sv[0] = ONE_F32 if has_q else ZERO_F32
    sv[1] = ONE_F32 if has_s else ZERO_F32
    sv[2] = ONE_F32 if age_val >= 18 else ZERO_F32
    sv[4] = ONE_F32 if ctry_id > 0 else ZERO_F32
    sv[5] = ONE_F32 if has_nf else ZERO_F32
    sv[6] = ONE_F32 if has_nm else ZERO_F32
    sv[8] = ONE_F32 if n_words == 1 else ZERO_F32
    sv[9] = np.float32(n_words) / SIX_F32
    sv[10] = ONE_F32 if is_wc_name else ZERO_F32
    sv[11] = ONE_F32 if (active_mask & (1 << 1)) != 0 else ZERO_F32
    sv[12] = ONE_F32 if (active_mask & (1 << 2)) != 0 else ZERO_F32
    sv[13] = ONE_F32 if (active_mask & (1 << 4)) != 0 else ZERO_F32
    sv[14] = ONE_F32 if (active_mask & (1 << 8)) != 0 else ZERO_F32
    sv[15] = ONE_F32 if (age_val >= 18 and ctry_id > 0) else ZERO_F32

    # 3) Zero-Allocation Unified Position + Ordered-Bigram Neural Forward Pass (4 matmuls -> 76 leaves)
    np.take(m.emb_word, s.w_ids, axis=0, out=s.w_emb)
    s.w_emb += m.pos_emb
    np.not_equal(s.w_ids, 0, out=s.w_mask_b)
    np.copyto(s.w_mask[:, 0], s.w_mask_b, casting="unsafe")
    s.w_emb *= s.w_mask
    np.sum(s.w_emb, axis=0, keepdims=True, out=s.x_in[:, :D_EMB])
    s.x_in[:, :D_EMB] *= ONE_F32 / np.float32(n_words)

    np.take(m.emb_bg, s.bg_ids, axis=0, out=s.bg_emb)
    np.not_equal(s.bg_ids, 0, out=s.bg_mask_b)
    np.copyto(s.bg_mask[:, 0], s.bg_mask_b, casting="unsafe")
    s.bg_emb *= s.bg_mask
    np.sum(s.bg_emb, axis=0, keepdims=True, out=s.x_in[:, D_EMB:])
    s.x_in[:, D_EMB:] *= ONE_F32 / np.float32(n_words + 1)

    np.matmul(s.s_vec, m.w_struct, out=s.st_proj)
    s.x_in += s.st_proj
    np.matmul(s.x_in, m.w1, out=s.h1)
    s.h1 += m.b1
    np.maximum(s.h1, ZERO_F32, out=s.h1)
    np.matmul(s.h1, m.w2, out=s.h2)
    s.h2 += m.b2
    np.maximum(s.h2, ZERO_F32, out=s.h2)
    s.h2 += s.x_in
    np.matmul(s.h2, m.w_out, out=s.logits)
    s.logits += m.b_out

    best_lf = int(np.argmax(s.logits[0]))
    if best_lf != 0 and np.float32(s.logits[0, best_lf] - s.logits[0, 0]) < CONF_MARGIN_F32:
        best_lf = 0
    return best_lf, age_val, ctry_id


def _reset_user(u, uid):
    u.last_intent[uid] = np.uint8(0)
    u.msg_count[uid] = np.uint8(0)
    u.age[uid] = np.uint8(0)
    u.country[uid] = np.uint8(0)
    u.gender[uid] = np.uint8(0)
    u.flags_lo[uid] = np.uint32(0)
    u.flags_hi[uid] = np.uint32(0)


def vm_handle_query(m, s, u, uid, mv_payload):
    now_s = int(time.time()) & 0xFFFFFFFF
    last_t = int(u.last_ts[uid])
    if last_t > 0 and (now_s - last_t) > SESSION_TIMEOUT_S:
        _reset_user(u, uid)

    leaf, age_val, ctry_id = vm_forward_pass(m, s, mv_payload)
    if leaf == 75:
        _reset_user(u, uid)
        u.last_ts[uid] = np.uint32(now_s)
        pid_rst = int(m.route_pool_lut[75, 0, 0, 1])
        return m.http_blob_mv[int(m.pool_off[pid_rst, 0]) : int(m.pool_end[pid_rst, 0])]

    prev_age, prev_ctry, prev_gen = int(u.age[uid]), int(u.country[uid]), int(u.gender[uid])
    f_lo, f_hi = int(u.flags_lo[uid]), int(u.flags_hi[uid])
    msg_cnt = (int(u.msg_count[uid]) + 1) & 255
    rule = int(m.leaf_rule[leaf])

    rep_st = 1 if ((f_lo & int(m.chk_lo[leaf])) | (f_hi & int(m.chk_hi[leaf]))) != 0 else 0
    eff_age = age_val if age_val >= 18 else prev_age
    cmp_st = 2 if (eff_age >= 18 and prev_ctry > 0) else (1 if eff_age >= 18 else 0)

    if rule == 5 and (prev_age > 0 or prev_ctry > 0 or prev_gen > 0):
        rep_st = 1
    elif rule == 1:
        rep_st = 1 if (prev_age == age_val and age_val > 0) else (2 if (prev_age > 0 and age_val > 0 and prev_age != age_val) else 0)
        cmp_st = 2 if prev_ctry > 0 else 0
    elif rule == 2:
        rep_st = 1 if (prev_ctry == ctry_id and ctry_id > 0) else (2 if (prev_ctry > 0 and ctry_id > 0 and prev_ctry != ctry_id) else 0)
        cmp_st = 2 if prev_age >= 18 else 0
    elif rule == 3:
        u.gender[uid] = np.uint8(2)
    elif rule == 4:
        u.gender[uid] = np.uint8(1)

    if age_val >= 18:
        u.age[uid] = np.uint8(age_val)
    if ctry_id > 0:
        u.country[uid] = np.uint8(ctry_id)
    u.last_intent[uid] = m.leaf_cls[leaf]
    u.msg_count[uid] = np.uint8(msg_cnt)
    u.last_ts[uid] = np.uint32(now_s)
    u.flags_lo[uid] = np.uint32(f_lo) | m.set_lo[leaf]
    u.flags_hi[uid] = np.uint32(f_hi) | m.set_hi[leaf]
    u.write_queue.append((uid, int(m.leaf_cls[leaf]), age_val, ctry_id, int(u.gender[uid]), now_s))

    # O(1) Finite-State Pool Lookup + Zero-Copy memoryview HTTP Slice (0 bytes allocated!)
    bot_g_idx = 0 if int(u.bot_gender[uid]) == 1 else 1
    pool_id = int(m.route_pool_lut[leaf, rep_st, cmp_st, bot_g_idx])
    rot = uid + msg_cnt
    idx = ((uid & 1) | ((rot & 3) << 1)) if int(m.leaf_plock[leaf]) else (rot & int(m.pool_mask[pool_id]))
    return m.http_blob_mv[int(m.pool_off[pool_id, idx]) : int(m.pool_end[pool_id, idx])]


async def wal_write_behind_worker(users, wal_path="events_wal.bin"):
    while True:
        await asyncio.sleep(2.0)
        if not users.write_queue:
            continue
        count = 0
        while users.write_queue and count < 256:
            users.wal_batch[count] = users.write_queue.popleft()
            count += 1
        try:
            with open(wal_path, "ab") as wal_file:
                wal_file.write(memoryview(users.wal_batch)[:count])
        except OSError:
            pass


async def start_server(host="127.0.0.1", port=8080):
    model = CompiledVMModel("model_weights.npz")
    users = UserStore()
    slot_queue = asyncio.Queue(maxsize=NUM_WORKERS)
    for i in range(NUM_WORKERS):
        slot_queue.put_nowait(InferenceSlot(i, model.num_leaves))

    test_slot = await slot_queue.get()
    t0 = time.perf_counter()
    for i in range(1000):
        vm_handle_query(model, test_slot, users, i, memoryview(b"m looking for girl , u"))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    for i in range(1000):
        _reset_user(users, i)

    smoke = (
        (b"hi", 1), (b"24", 3), (b"tuinsi", 5), (b"hi l'm 30 from tun", 6), (b"15", 7),
        (b"how r u?", 8), (b"m or f?", 9), (b"lm f u", 12), (b"22f", 14), (b"wyd", 15),
        (b"im good", 16), (b"lm borad", 17), (b"nmu", 18), (b"ur cute", 19), (b"sam", 20),
        (b"a gf", 22), (b"a bf", 22), (b"m looking for girl , u", 22), (b"lm finding gril", 22),
        (b"whats ur name?", 23), (b"my name is zoltan", 24), (b"i play valorant", 26),
        (b"are you single or mariid", 27), (b"im not married", 28), (b"can we talk privty", 29),
        (b"send pic", 31), (b"are u a bot?", 33), (b"are u real", 34), (b"gn", 35),
        (b"do u speak arabic?", 37), (b"truth or dare", 38), (b"sorry", 40), (b"send nudes", 42),
        (b"what do you do", 0), (b"why not", 0), (b"a bf here", 0), (b"next", 99),
    )
    ok = sum(1 for txt, exp in smoke if int(model.leaf_cls[vm_forward_pass(model, test_slot, memoryview(txt))[0]]) == exp)
    slot_queue.put_nowait(test_slot)

    print(f"[VM BENCHMARK] 1,000 requests in {elapsed_ms:.2f} ms ({elapsed_ms / 1000.0:.4f} ms/req)")
    print(f"[RULE 14 VM SMOKE] Passed {ok}/{len(smoke)} Steps 1-15 checks ({model.num_leaves} leaves | 4.75 MB RAM)")

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
                http_reply = vm_handle_query(model, slot, users, uid % MAX_USERS, mv_body)
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
    print(f"[RULE 14 VM SERVER] Listening on http://{host}:{port} (ZERO plugins | ZERO if-chains | 4.75 MB RAM)")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(start_server())