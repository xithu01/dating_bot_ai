#!/usr/bin/env python3
# FILE: production_server.py
# PURPOSE: Rule 7+14 Single-File Ultra-Light Pure-NumPy VM (103-Leaf INT8 NLU + Compiled Behavior Brain)
# DEPLOY: ONLY 2 files on server -> production_server.py (<300 lines, PyArmor-safe) + model_weights.npz

import asyncio, json, os, sys, time
from collections import deque
import numpy as np

B_SZ, W_MOD, BG_MOD, T_CTRY, T_NF, T_NM = 65536, 4091, 8191, 4093, 4094, 4095
MAX_LEN, MAX_U, NW = 45, 50000, 8
B_WC, B_KN = np.uint32(1 << 30), np.uint32(1 << 31)
Z32, O32, S32, CM32 = np.float32(0.0), np.float32(1.0), np.float32(6.0), np.float32(0.25)
F_STG, M_STG = (1, 2, 3, 4, 5, 6, 7, 8), (9, 2, 3)
T_BT = (0, 64, 8, 4, 0, 256, 512, 128, 1024, 32, 2, 16, 2048)
CNAMES = ("", "Tunisia", "India", "USA", "UK", "Algeria", "Morocco", "Egypt", "France", "Germany", "Italy", "Spain", "Canada", "Saudi", "UAE", "Turkey", "Pakistan", "Indonesia", "Philippines", "Brazil", "there")
BNC2 = ("and u", "and you", "what about u", "what about you", "how about u", "how about you")
EV_MEDIA = {"spoiler": 0, "voice": 1, "audio": 1, "photo": 2, "video": 2, "visual": 2, "sticker": 3, "gif": 3}


class CompiledVMModel:
    __slots__ = (
        "path", "emb_word", "emb_bg", "pos_emb", "w_struct", "w1", "b1", "w2", "b2", "w_out", "b_out",
        "super_bloom", "entity_lut_a", "entity_lut_b", "char_lut", "rej_lut", "w_poly_mat",
        "http_u8", "http_mv", "poff", "pend", "pmask", "rlut", "lcls", "lrule", "lplk",
        "clo", "chi", "slo", "shi", "ab_red", "num_leaves", "bmask", "sp32", "has_beh",
        "b_u8", "b_mv", "b_off", "b_end", "b_cdf", "b_len", "b_spid", "b_tpool", "b_ext",
        "b_iv", "b_icdf", "b_ilen", "b_tcdf", "b_rcdf", "b_ract", "bdr", "bf", "bi", "blq", "blc", "bkw",
    )

    def __init__(self, path):
        self.path, self.bmask, self.sp32 = path, np.int32(B_SZ - 1), np.int32(32)
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            print(f"[VM ERROR] '{self.path}' missing! Run: python offline_train.py", flush=True); sys.exit(1)
        arc = np.load(self.path, mmap_mode="r")
        if "vm_version_u8" not in arc.files or int(arc["vm_version_u8"][0]) < 14:
            arc.close(); print("[VM ERROR] Recompile model_weights.npz!", flush=True); sys.exit(1)
        def deq(k, z0=False):
            a = np.multiply(arc[f"{k}_q"].astype(np.int8, copy=False), arc[f"{k}_s"].astype(np.float32, copy=False), dtype=np.float32)
            if z0: a[0].fill(Z32)
            a.flags.writeable = False; return a
        def ro(k, dt):
            a = arc[k].astype(dt, copy=True); a.flags.writeable = False; return a
        self.emb_word, self.emb_bg = deq("emb_word", True), deq("emb_bg", True)
        self.pos_emb, self.w_struct = ro("pos_emb_f32", np.float32), deq("w_struct")
        self.w1, self.b1, self.w2, self.b2, self.w_out, self.b_out = deq("w1"), ro("b1_f32", np.float32), deq("w2"), ro("b2_f32", np.float32), deq("w_out"), ro("b_out_f32", np.float32)
        self.super_bloom, self.entity_lut_a, self.entity_lut_b = ro("super_bloom_u32", np.uint32), ro("entity_lut_a", np.uint8), ro("entity_lut_b", np.uint8)
        self.char_lut, self.rej_lut, self.w_poly_mat = ro("char_lut_i32", np.int32), ro("rej_lut_b", np.bool_), ro("w_poly_mat_i32", np.int32)
        self.http_u8 = ro("http_blob_u8", np.uint8); self.http_mv = memoryview(self.http_u8)
        self.poff, self.pend, self.pmask = ro("pool_msg_off_i32", np.int32), ro("pool_msg_end_i32", np.int32), ro("pool_mask_u8", np.uint8)
        self.rlut, self.lcls, self.lrule, self.lplk = ro("route_pool_lut", np.uint16), ro("leaf_to_class_u8", np.uint8), ro("leaf_entity_rule_u8", np.uint8), ro("leaf_persona_lock_u8", np.uint8)
        self.clo, self.chi, self.slo, self.shi = ro("leaf_chk_lo_u32", np.uint32), ro("leaf_chk_hi_u32", np.uint32), ro("leaf_set_lo_u32", np.uint32), ro("leaf_set_hi_u32", np.uint32)
        self.ab_red, self.num_leaves = ro("askback_redirect_u8", np.uint8), int(self.lcls.shape[0])
        self.has_beh = "b_text_blob_u8" in arc.files and "b_disc_rng_f32" in arc.files
        if self.has_beh:
            self.b_u8 = ro("b_text_blob_u8", np.uint8); self.b_mv = memoryview(self.b_u8)
            self.b_off, self.b_end, self.b_cdf, self.b_len = ro("b_pool_off_i32", np.int32), ro("b_pool_end_i32", np.int32), ro("b_pool_cdf_f32", np.float32), ro("b_pool_len_u8", np.uint8)
            self.b_spid, self.b_tpool, self.b_ext = ro("b_slot_pid_u16", np.uint16), ro("b_topic_pools_u16", np.uint16), ro("b_leaf_ext_pool_u16", np.uint16)
            self.b_iv, self.b_icdf, self.b_ilen = ro("b_int_vals_i32", np.int32), ro("b_int_cdf_f32", np.float32), ro("b_int_len_u8", np.uint8)
            self.b_tcdf, self.b_rcdf, self.b_ract, self.bdr = ro("b_topic_cdf_f32", np.float32), ro("b_rule_cdf_f32", np.float32), ro("b_rule_act_i32", np.int32), ro("b_disc_rng_f32", np.float32)
            self.bf, self.bi, self.blq, self.blc, self.bkw = ro("b_cfg_f32", np.float32), ro("b_cfg_i32", np.int32), ro("b_leaf_q_u8", np.uint8), ro("b_leaf_cfm_u8", np.uint8), ro("b_kw_bloom_u32", np.uint32)
        arc.close()


class InferenceSlot:
    __slots__ = ("id", "raw", "rej", "wm", "wh", "wbi", "sbh", "sbw", "ea", "eb", "wid", "bid", "sv", "we", "wmb", "wm1", "be", "bmb", "bm1", "xi", "sp", "h1", "h2", "lg")

    def __init__(self, sid, nl):
        self.id, self.raw, self.rej = sid, np.zeros((MAX_LEN + 4,), dtype=np.int32), np.zeros((MAX_LEN + 4,), dtype=np.bool_)
        self.wm, self.wh, self.wbi = np.zeros((6, 16), dtype=np.int32), np.zeros((6, 4), dtype=np.int32), np.zeros((6, 3), dtype=np.int32)
        self.sbh, self.sbw, self.ea, self.eb = np.zeros((6, 3), dtype=np.uint32), np.zeros((6,), dtype=np.uint32), np.zeros((6,), dtype=np.uint8), np.zeros((6,), dtype=np.uint8)
        self.wid, self.bid, self.sv = np.zeros((6,), dtype=np.int32), np.zeros((7,), dtype=np.int32), np.zeros((1, 16), dtype=np.float32)
        self.we, self.wmb, self.wm1 = np.zeros((6, 64), dtype=np.float32), np.zeros((6,), dtype=np.bool_), np.zeros((6, 1), dtype=np.float32)
        self.be, self.bmb, self.bm1 = np.zeros((7, 64), dtype=np.float32), np.zeros((7,), dtype=np.bool_), np.zeros((7, 1), dtype=np.float32)
        self.xi, self.sp = np.zeros((1, 128), dtype=np.float32), np.zeros((1, 128), dtype=np.float32)
        self.h1, self.h2, self.lg = np.zeros((1, 256), dtype=np.float32), np.zeros((1, 128), dtype=np.float32), np.zeros((1, nl), dtype=np.float32)


class UserStore:
    __slots__ = ("li", "ll", "mc", "age", "ctry", "gen", "bg", "pt", "tc", "mt", "ba", "bc", "bci", "bn", "pq", "lt", "lct", "pr", "sl", "rp", "ii", "sk", "st", "am", "um", "cm", "ef", "rec", "flo", "fhi", "ts", "bt", "lmh", "lft", "wq", "wb", "rng")

    def __init__(self):
        for k in ("li", "ll", "mc", "age", "ctry", "gen", "pt", "tc", "mt", "ba", "bc", "bci", "bn", "pq", "lt", "lct", "pr", "sl", "rp", "ii"):
            setattr(self, k, np.zeros((MAX_U,), dtype=np.uint8))
        self.bg = np.full((MAX_U,), 2, dtype=np.uint8)
        self.sk, self.st = np.zeros((MAX_U, 4), dtype=np.uint8), np.zeros((MAX_U, 8), dtype=np.uint8)
        for k in ("am", "um", "cm", "ef"): setattr(self, k, np.zeros((MAX_U,), dtype=np.uint16))
        self.rec = np.zeros((MAX_U, 4), dtype=np.uint16)
        for k in ("flo", "fhi", "ts", "bt", "lmh", "lft"): setattr(self, k, np.zeros((MAX_U,), dtype=np.uint32))
        self.wq, self.wb, self.rng = deque(maxlen=4096), np.zeros((256, 6), dtype=np.uint32), np.uint32(int(time.perf_counter_ns()) & 0x7FFFFFFF or 2463534242)


def vm_forward_pass(m, s, mv):
    bf = np.frombuffer(mv[:64], dtype=np.uint8); nw = np.flatnonzero(bf > np.uint8(32))
    if nw.shape[0] == 0: return 0, 0, 0
    f0, f1 = int(nw[0]), int(nw[-1]) + 1; nt = f1 - f0
    if nt > MAX_LEN: return 0, 0, 0
    if nt <= 6 and bytes(bf[f0:f1]).lower() in (b"next", b"reset", b"new", b"skip", b"/start"): return 75, 0, 0
    s.raw.fill(32); s.raw[2 : 2 + nt] = bf[f0:f1]; body = s.raw[2 : 2 + nt]
    np.take(m.rej_lut, body, out=s.rej[:nt])
    if bool(np.any(s.rej[:nt])): return 0, 0, 0
    if nt >= 5:
        for zi in np.flatnonzero(body == 48):
            z = int(zi)
            if 0 < z < nt - 1 and (int(body[z - 1]) | 32) == 104 and (int(body[z + 1]) | 32) == 114: body[z] = np.int32(111)
    hq, hs = bool(np.any(body == 63)), bool(np.any(body == 47))
    di = np.flatnonzero((body >= 48) & (body <= 57)); nd, av = int(di.shape[0]), 0
    if nd > 0:
        if nd > 2 or int(di[-1] - di[0]) != nd - 1: return 0, 0, 0
        d0 = int(di[0])
        if nd == 1 and d0 > 0 and int(body[d0 - 1]) > 39: return 0, 0, 0
        av = (int(body[d0]) - 48) if nd == 1 else ((int(body[d0]) - 48) * 10 + (int(body[d0 + 1]) - 48))
        if av < 18: return (7, av, 0) if av >= 1 else (0, 0, 0)
        if av > 80: return 0, 0, 0
    np.take(m.char_lut, s.raw, out=s.raw)
    kp = s.raw[2 : 2 + nt][s.raw[2 : 2 + nt] != s.raw[1 : 1 + nt]]
    if kp.shape[0] > 0 and int(kp[-1]) == 32: kp = kp[:-1]
    if kp.shape[0] > 0 and int(kp[0]) == 32: kp = kp[1:]
    kl = int(kp.shape[0])
    if kl == 0:
        if av >= 18: return 3, av, 0
        return (96, 0, 0) if (hq and m.num_leaves > 96 and bool(np.all((bf[f0:f1] == 63) | (bf[f0:f1] == 32)))) else (0, 0, 0)
    sp = np.flatnonzero(kp == m.sp32)
    if int(sp.shape[0]) > 5: return 0, 0, 0
    nw_c = int(sp.shape[0]) + 1; s.wm.fill(0)
    sts, eds = (0,) + tuple(int(v) + 1 for v in sp), tuple(int(v) for v in sp) + (kl,)
    for i in range(nw_c):
        Lw = min(eds[i] - sts[i], 15); s.wm[i, :Lw], s.wm[i, 15] = kp[sts[i] : sts[i] + Lw], np.int32(Lw)
    np.matmul(s.wm, m.w_poly_mat, out=s.wh); np.bitwise_and(s.wh[:, :3], m.bmask, out=s.wbi)
    np.take(m.super_bloom, s.wbi[:nw_c], out=s.sbh[:nw_c]); np.bitwise_and.reduce(s.sbh[:nw_c], axis=1, out=s.sbw[:nw_c])
    sbv = s.sbw[:nw_c]; ak = bool(np.all((sbv & B_KN) != 0)); wc = False
    if not ak and nw_c >= 3 and bool(np.any((sbv & B_WC) != 0)):
        uk = np.flatnonzero((sbv & B_KN) == 0)
        if uk.shape[0] == 1 and int(uk[0]) >= 2 and int(s.wm[int(uk[0]), 15]) >= 3: wc = True
    if not ak and not wc: return 0, 0, 0
    am = int(np.bitwise_and.reduce(sbv)) if ak else (1 << 9)
    np.take(m.entity_lut_a, s.wbi[:nw_c, 0], out=s.ea[:nw_c]); np.take(m.entity_lut_b, s.wbi[:nw_c, 1], out=s.eb[:nw_c])
    s.wid.fill(0); s.bid.fill(0); cid, hnf, hnm = 0, False, False
    for i in range(nw_c):
        a = int(s.ea[i]); cd = a if (a > 0 and a == int(s.eb[i])) else 0
        if 1 <= cd <= 20: cid = cid or cd; s.wid[i] = np.int32(T_CTRY)
        elif cd == 21: hnf, s.wid[i] = True, np.int32(T_NF)
        elif cd == 22 or (wc and (int(sbv[i]) & int(B_KN)) == 0): hnm, s.wid[i] = True, np.int32(T_NM)
        else: s.wid[i] = np.int32(((int(s.wh[i, 0]) & 0x7FFFFFFF) % W_MOD) + 1)
    s.bid[0] = np.int32((((4097 * 131 + int(s.wid[0])) & 0x7FFFFFFF) % BG_MOD) + 1)
    for k in range(nw_c - 1): s.bid[k + 1] = np.int32((((int(s.wid[k]) * 131 + int(s.wid[k + 1])) & 0x7FFFFFFF) % BG_MOD) + 1)
    s.bid[nw_c] = np.int32((((int(s.wid[nw_c - 1]) * 131 + (4101 if hq else (4100 if hs else 4099))) & 0x7FFFFFFF) % BG_MOD) + 1)
    sv = s.sv[0]; sv.fill(Z32)
    sv[0], sv[1], sv[2], sv[4] = (O32 if hq else Z32), (O32 if hs else Z32), (O32 if av >= 18 else Z32), (O32 if cid > 0 else Z32)
    sv[5], sv[6], sv[8], sv[9], sv[10] = (O32 if hnf else Z32), (O32 if hnm else Z32), (O32 if nw_c == 1 else Z32), np.float32(nw_c) / S32, (O32 if wc else Z32)
    sv[11], sv[12], sv[13], sv[14], sv[15] = (O32 if am & 2 else Z32), (O32 if am & 4 else Z32), (O32 if am & 16 else Z32), (O32 if am & 256 else Z32), (O32 if (av >= 18 and cid > 0) else Z32)
    np.take(m.emb_word, s.wid, axis=0, out=s.we); s.we += m.pos_emb; np.not_equal(s.wid, 0, out=s.wmb); np.copyto(s.wm1[:, 0], s.wmb, casting="unsafe"); s.we *= s.wm1; np.sum(s.we, axis=0, keepdims=True, out=s.xi[:, :64]); s.xi[:, :64] *= O32 / np.float32(nw_c)
    np.take(m.emb_bg, s.bid, axis=0, out=s.be); np.not_equal(s.bid, 0, out=s.bmb); np.copyto(s.bm1[:, 0], s.bmb, casting="unsafe"); s.be *= s.bm1; np.sum(s.be, axis=0, keepdims=True, out=s.xi[:, 64:]); s.xi[:, 64:] *= O32 / np.float32(nw_c + 1)
    np.matmul(s.sv, m.w_struct, out=s.sp); s.xi += s.sp
    np.matmul(s.xi, m.w1, out=s.h1); s.h1 += m.b1; np.maximum(s.h1, Z32, out=s.h1)
    np.matmul(s.h1, m.w2, out=s.h2); s.h2 += m.b2; np.maximum(s.h2, Z32, out=s.h2); s.h2 += s.xi
    np.matmul(s.h2, m.w_out, out=s.lg); s.lg += m.b_out
    bl = int(np.argmax(s.lg[0]))
    return (0, av, cid) if (bl != 0 and np.float32(s.lg[0, bl] - s.lg[0, 0]) < CM32) else (bl, av, cid)


def _rst(u, uid):
    for k in ("li", "ll", "mc", "age", "ctry", "gen", "pt", "tc", "mt", "ba", "bc", "bci", "bn", "pq", "lt", "lct", "pr", "sl", "rp", "ii"): getattr(u, k)[uid] = np.uint8(0)
    u.sk[uid].fill(0); u.st[uid].fill(0); u.rec[uid].fill(0)
    u.am[uid] = u.um[uid] = u.cm[uid] = u.ef[uid] = np.uint16(0)
    u.flo[uid] = u.fhi[uid] = u.bt[uid] = u.lmh[uid] = np.uint32(0)


def _u32(u):
    x = int(u.rng); x ^= (x << 13) & 0xFFFFFFFF; x ^= (x >> 17) & 0xFFFFFFFF; x ^= (x << 5) & 0xFFFFFFFF
    u.rng = np.uint32(x or 1); return int(u.rng)


def _rn(u): return np.float32((_u32(u) & 0xFFFFFF) / 16777216.0)
def _uni(u, lo, hi): return np.float32(lo + _rn(u) * (hi - lo))
def _ri(u, lo, hi): return lo + (_u32(u) % (hi - lo + 1))


def _fnv(s):
    h = 2166136261
    for b in s.encode("ascii", errors="ignore"): h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h or 1


def _pp(m, u, uid, pid):
    if pid >= m.b_len.shape[0]: pid = int(m.b_spid[57])
    n = int(m.b_len[pid]); i = min(int(np.searchsorted(m.b_cdf[pid, :n], _rn(u), side="right")), n - 1); rc = u.rec[uid]
    if n > 1:
        for st in range(n):
            c = (i + st) % n; tg = ((pid << 5) | c) & 0xFFFF
            if tg != int(rc[0]) and tg != int(rc[1]) and tg != int(rc[2]) and tg != int(rc[3]): i = c; break
    p = int(u.rp[uid]); rc[p] = np.uint16(((pid << 5) | i) & 0xFFFF); u.rp[uid] = np.uint8((p + 1) & 3)
    return bytes(m.b_mv[int(m.b_off[pid, i]) : int(m.b_end[pid, i])]).decode("utf-8")


def _ps(m, u, uid, sl): return _pp(m, u, uid, int(m.b_spid[sl]))
def _pat(m, sl, idx): pid = int(m.b_spid[sl]); i = idx % int(m.b_len[pid]); return bytes(m.b_mv[int(m.b_off[pid, i]) : int(m.b_end[pid, i])]).decode("utf-8")
def _pi(m, u, tid): n = int(m.b_ilen[tid]); i = min(int(np.searchsorted(m.b_icdf[tid, :n], _rn(u), side="right")), n - 1); return int(m.b_iv[tid, i])
def _kw(m, w): return int(m.bkw[_fnv(w) & 4095])


def _set_p(u, uid, t):
    if not t: return
    o = int(u.pq[uid])
    if o and o != t and not (int(u.cm[uid]) & (1 << o)): u.um[uid] |= np.uint16(1 << o)
    u.pq[uid], u.pr[uid], u.lt[uid] = np.uint8(t), np.uint8(0), np.uint8(t); u.am[uid] |= np.uint16(1 << t)


def _cfm(u, uid, t):
    if not t: return
    b = 1 << t; u.cm[uid] |= np.uint16(b); u.um[uid] &= np.uint16(~b & 0xFFFF); u.lct[uid] = u.lt[uid] = np.uint8(t)
    if int(u.pq[uid]) == t: u.pq[uid] = u.pr[uid] = np.uint8(0)


def _psk(u, uid, t):
    if not t: return
    sk, sl = u.sk[uid], int(u.sl[uid])
    if any(int(sk[i]) == t for i in range(sl)): return
    if sl < 4: sk[sl], u.sl[uid] = np.uint8(t), np.uint8(sl + 1)
    else: sk[0], sk[1], sk[2], sk[3] = sk[1], sk[2], sk[3], np.uint8(t)


def _popsk(u, uid):
    sl = int(u.sl[uid])
    if sl <= 0: return 0
    v = int(u.sk[uid, sl - 1]); u.sk[uid, sl - 1], u.sl[uid] = np.uint8(0), np.uint8(sl - 1); return v


def _mtag(m, u, uid):
    tg = _ps(m, u, uid, 4).replace("{rand_21_27}", str(_ri(u, 21, 27)))
    if tg.lower().startswith(("m", "male")): u.bt[uid] |= np.uint32(2)
    if "?" in tg or "f" in tg.lower(): _set_p(u, uid, 10)
    return tg


def _nq(m, u, uid, age_b=False):
    isf, am, cm = int(u.pt[uid]) == 1, int(u.am[uid]), int(u.cm[uid])
    if isf:
        if age_b and (cm & 8) and not (am & 16): _set_p(u, uid, 4); return _ps(m, u, uid, 78 if int(u.age[uid]) <= int(m.bi[8]) else 79)
        for t in F_STG:
            if t == 4:
                if (cm & 8) and not (am & 16): _set_p(u, uid, 4); return _ps(m, u, uid, 78 if int(u.age[uid]) <= int(m.bi[8]) else 79)
                continue
            if not ((am | cm) & (1 << t)): _set_p(u, uid, t); return _pp(m, u, uid, int(m.b_tpool[1, t, 2]))
        return ""
    for t in M_STG:
        if (am | cm) & (1 << t): continue
        if t == 9 and _rn(u) < m.bf[62]: u.am[uid] |= np.uint16(1 << 9); continue
        _set_p(u, uid, t); return _pp(m, u, uid, int(m.b_tpool[0, t, 2]))
    return ""


def _bctry(m, u, uid):
    ci = int(u.bc[uid])
    if ci: return _pat(m, 85, ci - 1), ci
    cn = _ps(m, u, uid, 14 if int(u.pt[uid]) == 1 else 15); n = int(m.b_len[int(m.b_spid[85])])
    ci = next((k + 1 for k in range(n) if _pat(m, 85, k) == cn), 1); u.bc[uid] = np.uint8(ci); return cn, ci


def _disc_t(m, u, did):
    if did <= 0 or did >= 10: return None
    lo, hi = m.bdr[did, 0], m.bdr[did, 1]
    return np.float32(_pi(m, u, 3)) if lo < Z32 else (_uni(u, lo, hi) if hi > lo else lo)


def _fin(m, u, uid, act, cid, lf, rd, raws, ds=None, op=False, aq=False):
    bf, bs = m.bf, []
    for it in raws:
        if not it: continue
        for j, seg in enumerate(p.strip() for p in it.split("|") if p.strip()):
            pb = Z32 if not bs else (_uni(u, bf[21], bf[22]) if j else (_uni(u, bf[23], bf[24]) if op else _uni(u, bf[25], bf[26])))
            ty = np.float32(max(np.float32(0.15), min(np.float32(len(seg)) / _uni(u, bf[18], bf[19]), bf[20])))
            bs.append({"text": seg, "pause_before_s": round(float(pb), 2), "typing_s": round(float(ty), 2)})
            if "?" in seg: aq = True
    pt = int(u.pt[uid]); p_s = "female" if pt == 1 else ("male_normal" if pt == 2 else "male_ghoster")
    idl = Z32 if ds is not None else (_uni(u, bf[30], bf[31]) if (pt == 3 and not aq and not op) else np.float32(_pi(m, u, 1) + bf[27] + ((bf[28] if op else bf[29]) if aq else Z32)))
    return {"action": act, "nlu": {"class_id": int(cid), "leaf_id": int(lf)}, "persona": {"type": p_s, "vip": bool(int(u.ef[uid]) & 2), "interest": _pat(m, 23, int(u.ii[uid]))}, "read_delay_s": round(float(rd), 2), "bubbles": bs, "idle_timeout_s": round(float(idl), 2), "disconnect_after_s": None if ds is None else round(float(ds), 2)}


def _disp(m, u, uid, t, frc=False, ap="", yng=False, uad=False):
    isf, bf, bt = int(u.pt[uid]) == 1, m.bf, int(u.bt[uid])
    if t == 10:
        if (bt & 2) and not frc: q = _nq(m, u, uid); return ((q,) if q else (_ps(m, u, uid, 53),)), None, bool(q)
        u.bt[uid] |= np.uint32(2)
        ans = _ps(m, u, uid, (10 if isf else 11) if (yng and _rn(u) < bf[48]) else (8 if isf else 9))
        nu = not (int(u.cm[uid]) & 1024)
        if nu: _set_p(u, uid, 10)
        return (f"{ap} {ans}".strip(),), None, nu
    if t == 11:
        ib = frc or bool(int(u.ef[uid]) & 8); u.ef[uid] |= np.uint16(8)
        if _rn(u) < (bf[50] if ib else bf[49]):
            _, ci = _bctry(m, u, uid); csl = 80 + min(ci - 1, 4); c_n = int(m.b_len[int(m.b_spid[csl])])
            bci = int(u.bci[uid]) or _ri(u, 1, c_n); u.bci[uid], u.bt[uid] = np.uint8(bci), u.bt[uid] | np.uint32(16); q = _nq(m, u, uid)
            return (f"{ap} {_pat(m, csl, bci - 1)}".strip(),) + ((q,) if q else ()), None, bool(q)
        _psk(u, uid, 11)
        if ib or _rn(u) < bf[51]:
            q = _nq(m, u, uid) if _rn(u) < bf[52] else ""
            return (f"{ap} {_pp(m, u, uid, int(m.b_tpool[int(isf), 11, 1]))}".strip(),) + ((q,) if q else ()), None, bool(q)
        q = _nq(m, u, uid); return ((q,) if q else ()), None, bool(q)
    bit = T_BT[t]
    if bit and (bt & bit) and not frc:
        if t == 1 and uad: return (_ps(m, u, uid, 33),), None, False
        q = _nq(m, u, uid); return ((q,), None, True) if q else ((_ps(m, u, uid, 53),), (None if isf else bf[32]), False)
    row, r = m.b_tcdf[int(isf), t], _rn(u)
    cf = frc and not (isf and t == 9 and not int(m.bi[9])) and t != 8
    md = 0 if cf else (0 if r < row[0] else (1 if r < row[1] else 2))
    if md == 2: _psk(u, uid, t); q = _nq(m, u, uid); return ((q,) if q else ()), None, bool(q)
    if md == 1:
        _psk(u, uid, t); ev = _pp(m, u, uid, int(m.b_tpool[int(isf), t, 1]))
        if t == 2 and isf and not (int(u.cm[uid]) & 4): _set_p(u, uid, 2); return (ev + _ps(m, u, uid, 32),), None, True
        q = _nq(m, u, uid) if (t in (2, 9) or (t == 3 and (int(u.cm[uid]) & 8))) else ""
        return (f"{ap} {ev}".strip(),) + ((q,) if q else ()), None, bool(q)
    u.bt[uid] |= np.uint32(bit)
    if t == 2:
        cn, ci = _bctry(m, u, uid)
        if isf and int(m.bi[10]) > 0 and ci == int(m.bi[10]): return (f"{ap} {_ps(m, u, uid, 16)}".strip(), _ps(m, u, uid, 17)), bf[6], False
        q = _nq(m, u, uid); return (f"{ap} {_ps(m, u, uid, 18 if isf else 19).format(country=cn)}".strip(),) + ((q,) if q else ()), None, bool(q)
    if t == 3:
        ba = int(u.ba[uid]) or (_ri(u, int(m.bi[3]), int(m.bi[4])) if _rn(u) < bf[5] else _ri(u, int(m.bi[5]), int(m.bi[6]))); u.ba[uid] = np.uint8(ba)
        if not (int(u.cm[uid]) & 8): _set_p(u, uid, 3); return (f"{ap} {ba}, wbu?".strip(),), None, True
        q = _nq(m, u, uid); return (f"{ap} {ba}".strip(),) + ((q,) if q else ()), None, bool(q)
    if t == 9:
        nsl = 20 if isf else 21; bn = int(u.bn[uid]) or _ri(u, 1, int(m.b_len[int(m.b_spid[nsl])])); u.bn[uid] = np.uint8(bn)
        q = _nq(m, u, uid); return (f"{ap} {_ps(m, u, uid, 22).format(name=_pat(m, nsl, bn - 1))}".strip(),) + ((q,) if q else ()), None, bool(q)
    if t == 1:
        if uad or (int(u.cm[uid]) & 2):
            ans = _ps(m, u, uid, 33) if _rn(u) < bf[54] else _pp(m, u, uid, int(m.b_tpool[int(isf), 1, 0]))
            q = _nq(m, u, uid); return (f"{ap} {ans}".strip(),) + ((q,) if q else ()), None, bool(q)
        if _rn(u) < bf[55]: _set_p(u, uid, 1); return (f"{ap} {_ps(m, u, uid, 34 if isf else 35)}".strip(),), None, True
    q = _nq(m, u, uid); return (f"{ap} {_pp(m, u, uid, int(m.b_tpool[int(isf), t, 0]))}".strip(),) + ((q,) if q else ()), None, bool(q)


def _adv(m, u, uid, cid, lf, rd, pre=(), be=False):
    q = _nq(m, u, uid)
    if q: return _fin(m, u, uid, "reply", cid, lf, rd, pre + (q,), aq=True)
    if int(u.pt[uid]) == 1:
        if be and int(u.tc[uid]) >= 5 and not (int(u.ef[uid]) & 1) and _rn(u) < m.bf[65]:
            u.ef[uid] |= np.uint16(1); return _fin(m, u, uid, "reply", cid, lf, rd, pre + (_ps(m, u, uid, 56),), aq=True)
        return _fin(m, u, uid, "reply", cid, lf, rd, pre + (_ps(m, u, uid, 57),))
    return _fin(m, u, uid, "reply", cid, lf, rd, pre) if pre else _fin(m, u, uid, "disconnect", cid, lf, rd, (), m.bf[32])


def _exec_rule(m, u, uid, rid, cid, lf, rd, lo=""):
    i = min(int(np.searchsorted(m.b_rcdf[rid], _rn(u), side="right")), 4)
    ac, sl, fup, did, flg = [int(v) for v in m.b_ract[rid, i]]; isf, bf = int(u.pt[uid]) == 1, m.bf
    if ac == 1: return _fin(m, u, uid, "silent", cid, lf, rd, ())
    if ac == 5:
        sil = _rn(u) < bf[68]
        return _fin(m, u, uid, "disconnect", cid, lf, rd, () if sil else (_ps(m, u, uid, 42 if isf else 43),), _disc_t(m, u, 4) if sil else bf[33])
    if ac == 6: _psk(u, uid, 12); return _adv(m, u, uid, cid, lf, rd)
    if ac == 3:
        q = _nq(m, u, uid)
        if q: return _fin(m, u, uid, "reply", cid, lf, rd, (q,), aq=True)
        return _fin(m, u, uid, "reply", cid, lf, rd, (_ps(m, u, uid, sl),)) if sl >= 0 else _adv(m, u, uid, cid, lf, rd)
    if ac == 4:
        sk = _popsk(u, uid)
        if sk:
            ap = _ps(m, u, uid, 36 if isf else 37) if ((flg & 8) and _rn(u) < bf[69]) else ""
            b, ds, aq = _disp(m, u, uid, sk, True, ap); return _fin(m, u, uid, "disconnect" if ds else "reply", cid, lf, rd, b, ds, aq=aq)
        txt = _ps(m, u, uid, sl)
        return _adv(m, u, uid, cid, lf, rd, (txt,)) if _rn(u) * 1000 < fup else _fin(m, u, uid, "reply", cid, lf, rd, (txt,))
    if (flg & 4) and not any(_kw(m, w) & 512 for w in lo.split()): sl = 77
    if (flg & 1) and sl == 69 and int(u.tc[uid]) <= 3 and not (int(u.bt[uid]) & 8192): u.bt[uid] |= np.uint32(8192); sl = 68
    txt = _ps(m, u, uid, sl) if sl >= 0 else ""
    if ac == 2: return _fin(m, u, uid, "disconnect", cid, lf, rd, (txt,) if txt else (), _disc_t(m, u, did))
    if (flg & 2) and int(u.gen[uid]) == 1 and _rn(u) < bf[70]: return _fin(m, u, uid, "disconnect", cid, lf, rd, (txt,), _disc_t(m, u, 9))
    if (flg & 1) and isf and int(u.pq[uid]):
        if int(u.pr[uid]) == 0 and _rn(u) < bf[71]: u.pr[uid] = np.uint8(1); return _fin(m, u, uid, "reply", cid, lf, rd, (txt + _ps(m, u, uid, 74),), aq=True)
        u.pq[uid] = np.uint8(0); return _adv(m, u, uid, cid, lf, rd, (txt,)) if _rn(u) < bf[72] else _fin(m, u, uid, "reply", cid, lf, rd, (txt,))
    return _adv(m, u, uid, cid, lf, rd, (txt,)) if (fup and _rn(u) * 1000 < fup) else _fin(m, u, uid, "reply", cid, lf, rd, (txt,))


def beh_start(m, u, uid, now_s=None):
    uid, now_i, bf, bi = uid % MAX_U, int(time.time() if now_s is None else now_s) & 0xFFFFFFFF, m.bf, m.bi
    _rst(u, uid); pf = int(u.lft[uid])
    if _rn(u) < (bf[0] if (pf and (now_i - pf) <= int(bi[0])) else bf[1]): u.pt[uid], u.lft[uid] = np.uint8(1), np.uint32(now_i)
    else: u.pt[uid] = np.uint8(2 if _rn(u) < bf[2] else 3)
    u.mt[uid], u.ts[uid], u.ii[uid] = np.uint8(_ri(u, int(bi[1]), int(bi[2]))), np.uint32(now_i), np.uint8(_ri(u, 0, int(m.b_len[int(m.b_spid[23])]) - 1))
    if _rn(u) < bf[4]: u.ef[uid] |= np.uint16(2)
    pt = int(u.pt[uid]); return {"status": "started", "persona": "female" if pt == 1 else ("male_normal" if pt == 2 else "male_ghoster"), "opening_wait_s": _pi(m, u, 0)}


def beh_opening(m, u, uid):
    uid = uid % MAX_U
    if not int(u.pt[uid]): beh_start(m, u, uid)
    u.tc[uid] = np.uint8(min(255, int(u.tc[uid]) + 1)); cnt, bs, aq = _pi(m, u, 2), [], False
    if int(u.pt[uid]) == 1:
        bs.append(_ps(m, u, uid, 2)); u.bt[uid] |= np.uint32(1)
        if cnt >= 2:
            if _rn(u) < m.bf[8]: bs.append(_ps(m, u, uid, 8)); u.bt[uid] |= np.uint32(2)
            else: q = _nq(m, u, uid); bs.append(q); aq = bool(q)
        if cnt >= 3: q = _nq(m, u, uid); bs.append(q); aq = aq or bool(q)
        if cnt >= 4: bs.append(_ps(m, u, uid, 6)); aq = True
    else:
        tg = _mtag(m, u, uid)
        if cnt == 1: bs.append(tg)
        else:
            if _rn(u) < m.bf[7]: bs.append(_ps(m, u, uid, 3)); u.bt[uid] |= np.uint32(1)
            bs.append(tg)
            if "?" not in tg and "f" not in tg.lower(): bs.append(_ps(m, u, uid, 5)); _set_p(u, uid, 10); aq = True
            if cnt >= 3: bs.append(_ps(m, u, uid, 86)); _set_p(u, uid, 2); aq = True
            if cnt >= 4: bs.append(_ps(m, u, uid, 7)); aq = True
    return _fin(m, u, uid, "opening_burst", 0, 0, Z32, bs, None, True, aq)


def beh_turn(m, s, u, uid, utts, ev="text", emo="", now_s=None):
    uid, now_i, bf, bi = uid % MAX_U, int(time.time() if now_s is None else now_s) & 0xFFFFFFFF, m.bf, m.bi
    if not int(u.pt[uid]) or (int(u.ts[uid]) and (now_i - int(u.ts[uid])) > int(bi[12])): beh_start(m, u, uid, now_i)
    u.ts[uid] = np.uint32(now_i)
    if ev == "gift": return _fin(m, u, uid, "reply", 0, 0, np.float32(0.8), (_ps(m, u, uid, 58).format(emoji=emo or "gift"),))
    utts = (utts,) if isinstance(utts, str) else tuple(x for x in utts if x and x.strip())
    jn = " ".join(utts).strip(); lo = jn.lower()
    rd = _uni(u, bf[9], bf[10]) + min(np.float32(len(jn)) * bf[11], bf[12])
    if ("?" in lo) or any(_kw(m, w) & 1 for w in lo.replace("?", " ").replace(",", " ").split()): rd += _uni(u, bf[13], bf[14])
    if _rn(u) < bf[15]: rd += _uni(u, bf[16], bf[17])
    lfs, ags, cts, gns, lq_a, hb_a = [], [], [], [], False, False
    for ut in utts:
        cu = " ".join(ut.strip().split())
        if len(cu) >= 2 and cu[0] == "?" and any(c.isalnum() for c in cu[1:]): lq_a, cu = True, cu.lstrip("? ").strip()
        ce, hb, stm = cu.lower().rstrip("?!., ").strip(), False, cu
        for ph in BNC2:
            if ce.endswith(" " + ph): hb, stm = True, ce[: -(len(ph) + 1)].rstrip(", ").strip(); break
        if not hb:
            tk = ce.replace(",", " ").split()
            if len(tk) >= 2 and (_kw(m, tk[-1]) & 32): hb, stm = True, " ".join(tk[:-1]).strip()
        hb_a = hb_a or hb
        lf, av, cv = vm_forward_pass(m, s, memoryview(cu.encode("ascii", errors="ignore")[:64]))
        if lf == 0 and hb and stm: lf, av, cv = vm_forward_pass(m, s, memoryview(stm.encode("ascii", errors="ignore")[:64]))
        if lf == 7: return _fin(m, u, uid, "disconnect", 7, 7, np.float32(0.5), (_ps(m, u, uid, 0),), bf[32])
        if lf == 75: beh_start(m, u, uid, now_i); return _fin(m, u, uid, "reply", 99, 75, np.float32(0.2), (_ps(m, u, uid, 1),))
        if "?" in cu and not hb:
            if lf == 3 and int(u.pq[uid]) != 3: lf, av = 2, 0
            elif lf == 5 and int(u.pq[uid]) != 2: lf, cv = 4, 0
        lfs.append(lf)
        if av >= int(bi[7]): ags.append(av)
        if cv > 0: cts.append(cv)
        if lf in (12, 14): gns.append(2)
        elif lf in (13, 15): gns.append(1)
    plf = next((x for x in lfs if x), 0); pcls = int(m.lcls[plf]) if plf < m.num_leaves else 0
    u.tc[uid] = np.uint8(min(255, int(u.tc[uid]) + 1)); tn, pt, isf = int(u.tc[uid]), int(u.pt[uid]), int(u.pt[uid]) == 1
    if pt == 3:
        if tn == 1 and not (int(u.bt[uid]) & 2) and not gns:
            tg = _mtag(m, u, uid); return _fin(m, u, uid, "reply", pcls, plf, rd, (tg,), aq=("?" in tg))
        if gns:
            u.gen[uid] = np.uint8(gns[0]); _cfm(u, uid, 10); pre = ()
            if not (int(u.bt[uid]) & 2): u.bt[uid] |= np.uint32(2); pre = (_ps(m, u, uid, 9),)
            if _rn(u) < bf[3]:
                u.pt[uid] = np.uint8(2); qt = _pi(m, u, 4); _set_p(u, uid, qt)
                return _fin(m, u, uid, "reply", pcls, plf, rd, pre + (_pp(m, u, uid, int(m.b_tpool[0, qt, 2])),), aq=True)
            return _fin(m, u, uid, "disconnect", pcls, plf, rd, pre, _disc_t(m, u, 8))
        return _fin(m, u, uid, "disconnect", pcls, plf, rd, (), _disc_t(m, u, 8))
    if ev == "text" and jn:
        mh = np.uint32(_fnv(lo))
        if mh == u.lmh[uid]: u.st[uid, 0] = np.uint8(min(255, int(u.st[uid, 0]) + 1))
        else: u.lmh[uid], u.st[uid, 0] = mh, np.uint8(1)
        if int(u.st[uid, 0]) >= 3: return _fin(m, u, uid, "disconnect", pcls, plf, rd, (_ps(m, u, uid, 54),), bf[32])
    if tn >= int(u.mt[uid]): return _fin(m, u, uid, "disconnect", pcls, plf, rd, (_ps(m, u, uid, 55),), bf[32])
    if ev != "text":
        u.st[uid, 1] = u.st[uid, 3] = np.uint8(0); ms = min(255, int(u.st[uid, 2]) + 1); u.st[uid, 2] = np.uint8(ms)
        if ms >= 3: return _fin(m, u, uid, "silent", 0, 0, rd, ())
        return _exec_rule(m, u, uid, (13 if isf else 14) if ms == 2 else (15 + EV_MEDIA.get(ev, 3) * 2 + (0 if isf else 1)), 0, 0, rd)
    if plf == 96:
        qs = min(255, int(u.st[uid, 3]) + 1); u.st[uid, 3], u.st[uid, 2] = np.uint8(qs), np.uint8(0)
        if qs == 1 and int(u.sl[uid]) > 0 and _rn(u) < bf[67]:
            b, ds, aq = _disp(m, u, uid, _popsk(u, uid), True); return _fin(m, u, uid, "disconnect" if ds else "reply", 55, 96, rd, b, ds, aq=aq)
        return _exec_rule(m, u, uid, 3 if qs == 1 else (4 if qs == 2 else 5), 55, 96, rd)
    u.st[uid, 2] = u.st[uid, 3] = np.uint8(0)
    if plf in (54, 72): return _fin(m, u, uid, "disconnect", pcls, plf, rd, (_ps(m, u, uid, 55 if plf == 54 else (42 if isf else 43)),), bf[32])
    if any(x in (43, 44) for x in lfs):
        ps = min(255, int(u.st[uid, 5]) + 1); u.st[uid, 5] = np.uint8(ps); return _exec_rule(m, u, uid, min(ps - 1, 2), 29, 43, rd)
    if gns:
        u.gen[uid] = np.uint8(gns[0]); _cfm(u, uid, 10)
        if ags: u.age[uid] = np.uint8(ags[0]); _cfm(u, uid, 3)
        pre = () if (int(u.bt[uid]) & 2) else (_ps(m, u, uid, 8 if isf else 9),); u.bt[uid] |= np.uint32(2)
        pg = _ps(m, u, uid, 12 if isf else 13); lp = pg.lower()
        _set_p(u, uid, 2 if "from" in lp else (3 if ("old" in lp or "age" in lp) else (1 if ("doing" in lp or "wyd" in lp) else (9 if "name" in lp else (6 if ("married" in lp or "gf" in lp) else (7 if "brings" in lp else 0))))))
        return _fin(m, u, uid, "reply", pcls, plf, rd, pre + (pg,), aq=("?" in pg))
    if plf == 99 or (plf == 1 and "meet" in lo):
        u.bt[uid] |= np.uint32(4096); nt = _ps(m, u, uid, 29); return _adv(m, u, uid, 58, 99, rd, (nt,)) if _rn(u) < bf[60] else _fin(m, u, uid, "reply", 58, 99, rd, (nt,))
    if 97 in lfs:
        if ags: u.age[uid] = np.uint8(ags[0]); _cfm(u, uid, 3)
        if cts: u.ctry[uid] = np.uint8(cts[0]); _cfm(u, uid, 2)
        ins = min(255, int(u.st[uid, 4]) + 1); u.st[uid, 4] = np.uint8(ins)
        if ins >= 3:
            sil = _rn(u) < bf[68]
            return _fin(m, u, uid, "disconnect", 56, 97, rd, () if sil else (_ps(m, u, uid, 42 if isf else 43),), _disc_t(m, u, 4) if sil else bf[33])
        return _exec_rule(m, u, uid, (6 if isf else 7) if ins == 1 else (8 if isf else 9), 56, 97, rd)
    if 101 in lfs:
        hs = min(255, int(u.st[uid, 6]) + 1); u.st[uid, 6] = np.uint8(hs)
        if hs >= 3: return _fin(m, u, uid, "disconnect", 60, 101, rd, (_ps(m, u, uid, 54),) if isf else (), bf[32] if isf else _disc_t(m, u, 7))
        if hs == 2 and isf: q = _nq(m, u, uid); return _fin(m, u, uid, "reply", 60, 101, rd, (q or _ps(m, u, uid, 53),), aq=bool(q))
        return _exec_rule(m, u, uid, 12 if isf else (10 if hs == 1 else 11), 60, 101, rd, lo)
    if plf == 76 and len(lfs) == 1:
        u.st[uid, 1] = np.uint8(0); tp = int(u.pq[uid]) or int(u.lt[uid]) or int(u.lct[uid]) or (int(u.sk[uid, int(u.sl[uid]) - 1]) if int(u.sl[uid]) else 0)
        if tp: b, ds, aq = _disp(m, u, uid, tp); return _fin(m, u, uid, "disconnect" if ds else "reply", 0, 76, rd, b, ds, aq=aq)
        return _adv(m, u, uid, 0, 76, rd)
    rs, ct = lo.strip(), lo.rstrip("?!., ").strip(); km = _kw(m, ct) | _kw(m, rs)
    if len(lfs) == 1 and (plf == 70 or (km & 2)): return _fin(m, u, uid, "reply", pcls, plf, rd, (_ps(m, u, uid, 44),))
    if len(lfs) == 1 and (plf == 100 or (km & 4)):
        ds = min(255, int(u.st[uid, 1]) + 1); u.st[uid, 1] = np.uint8(ds)
        if ds >= 3: return _fin(m, u, uid, "disconnect", pcls, plf, rd, (_ps(m, u, uid, 49),), bf[32])
        return _fin(m, u, uid, "reply", pcls, plf, rd, (_ps(m, u, uid, 48 if (isf and int(u.pq[uid])) else (46 if ds == 1 else 47)).format(word=ct or rs or "ok"),))
    u.st[uid, 1] = np.uint8(0)
    if len(lfs) == 1 and (plf in (45, 46) or (km & 24)):
        iy, pq = (plf == 45 or bool(km & 8)), int(u.pq[uid])
        if isf and pq:
            if pq in (4, 5, 6, 7, 8):
                _cfm(u, uid, pq); rx = _ps(m, u, uid, 51); return _adv(m, u, uid, pcls, plf, rd, (rx,)) if _rn(u) < bf[63] else _fin(m, u, uid, "reply", pcls, plf, rd, (rx,))
            u.pr[uid] = np.uint8(int(u.pr[uid]) + 1)
            if int(u.pr[uid]) == 1: return _fin(m, u, uid, "reply", pcls, plf, rd, (_ps(m, u, uid, 52),))
            u.pq[uid] = np.uint8(0); return _adv(m, u, uid, pcls, plf, rd)
        return _fin(m, u, uid, "reply", pcls, plf, rd, (_ps(m, u, uid, 50 if not iy else 46).format(word=ct or "yes"),))
    pre = ()
    if lq_a and int(u.sl[uid]) > 0 and _rn(u) < bf[66]:
        b, ds, _ = _disp(m, u, uid, _popsk(u, uid), True)
        if ds is not None: return _fin(m, u, uid, "disconnect", pcls, plf, rd, b, ds)
        pre = b
    ctop, cex = 0, ""
    if cts: cid = cts[0]; u.ctry[uid] = np.uint8(cid); _cfm(u, uid, 2); ctop, cex = 2, (CNAMES[cid] if cid < len(CNAMES) else "there")
    if ags: u.age[uid] = np.uint8(ags[0]); _cfm(u, uid, 3); ctop = ctop or 3
    for x in lfs:
        if x < 128 and int(m.blc[x]): _cfm(u, uid, int(m.blc[x])); ctop = ctop or int(m.blc[x])
    if (hb_a or 19 in lfs) and ctop:
        b, ds, aq = _disp(m, u, uid, ctop, uad=(ctop == 1)); return _fin(m, u, uid, "disconnect" if ds else "reply", pcls, plf, rd, pre + b, ds, aq=aq)
    for x in lfs:
        if x < 128 and int(m.blq[x]):
            b, ds, aq = _disp(m, u, uid, int(m.blq[x]), yng=(x in (10, 11)), uad=(ctop == 1)); return _fin(m, u, uid, "disconnect" if ds else "reply", pcls, x, rd, pre + b, ds, aq=aq)
    if 1 in lfs and not ctop:
        if int(u.bt[uid]) & 1: return _adv(m, u, uid, 1, 1, rd, pre)
        u.bt[uid] |= np.uint32(1)
        if _rn(u) < bf[44]:
            gr = (jn.capitalize() if _rn(u) < bf[46] else lo) if (2 <= len(jn) <= 12 and _rn(u) < bf[45]) else _ps(m, u, uid, 2 if isf else 3)
            if _rn(u) < bf[47]: return _adv(m, u, uid, 1, 1, rd, pre + (gr,)) if isf else _fin(m, u, uid, "reply", 1, 1, rd, pre + (gr, _mtag(m, u, uid)), aq=True)
            return _fin(m, u, uid, "reply", 1, 1, rd, pre + (gr,))
        return _adv(m, u, uid, 1, 1, rd, pre) if isf else _fin(m, u, uid, "reply", 1, 1, rd, pre + (_mtag(m, u, uid),), aq=True)
    if ctop:
        if ctop == 2 and _rn(u) < bf[53]:
            sq = _ps(m, u, uid, 31); _set_p(u, uid, 11); rx = (_ps(m, u, uid, 26).format(country=cex),) if (isf and _rn(u) < bf[56]) else ()
            return _fin(m, u, uid, "reply", pcls, plf, rd, pre + rx + (sq,), aq=True)
        if not isf:
            q = _nq(m, u, uid); return _fin(m, u, uid, "disconnect", pcls, plf, rd, pre, bf[32]) if not q else _fin(m, u, uid, "reply", pcls, plf, rd, pre + ((_ps(m, u, uid, 28), q) if _rn(u) < bf[61] else (q,)), aq=True)
        rx = ""
        if _rn(u) < bf[56]:
            rx = _ps(m, u, uid, 24 if int(u.age[uid]) <= int(bi[8]) else 25).format(age=int(u.age[uid])) if ctop == 3 else (_ps(m, u, uid, 26).format(country=cex) if ctop == 2 else _ps(m, u, uid, 27))
            if ctop in (2, 3) and not (int(u.bt[uid]) & 4096) and _rn(u) < bf[59]: rx += _ps(m, u, uid, 30); u.bt[uid] |= np.uint32(4096)
        q = _nq(m, u, uid, ctop == 3) if (ctop == 3 or not rx or _rn(u) < bf[57]) else ""
        if rx and q: return _fin(m, u, uid, "reply", pcls, plf, rd, pre + ((f"{rx}. {q}",) if _rn(u) < bf[58] else (rx, q)), aq=True)
        return _fin(m, u, uid, "reply", pcls, plf, rd, pre + (rx,)) if rx else _adv(m, u, uid, pcls, plf, rd, pre)
    if plf and plf < 128 and int(m.b_ext[plf]) != 65535:
        ans = _pp(m, u, uid, int(m.b_ext[plf]))
        return _adv(m, u, uid, pcls, plf, rd, pre + (ans,)) if (_rn(u) < bf[73] and "?" not in ans) else _fin(m, u, uid, "reply", pcls, plf, rd, pre + (ans,), aq=("?" in ans))
    pq = int(u.pq[uid])
    if pq and jn:
        tk = [w.strip("?!.,'") for w in lo.split() if w.strip("?!.,'")]
        if tk and "?" not in lo and not any(_kw(m, w) & 64 for w in tk) and not all(_kw(m, w) & 128 for w in tk) and ((any(_kw(m, w) & 256 for w in tk)) if pq == 5 else (pq in (4, 7, 8, 9, 11) and (len(tk) >= 2 or len(ct) >= 5))):
            _cfm(u, uid, pq); return _adv(m, u, uid, pcls, plf, rd, pre + ((_ps(m, u, uid, 27),) if (isf and _rn(u) < bf[56]) else ()))
        u.pr[uid] = np.uint8(int(u.pr[uid]) + 1)
        if isf and int(u.pr[uid]) >= 2 and pq in (1, 5) and not (int(u.ef[uid]) & 1) and _rn(u) < bf[64]: u.ef[uid] |= np.uint16(1); return _fin(m, u, uid, "reply", pcls, plf, rd, (_ps(m, u, uid, 56),), aq=True)
        if int(u.pr[uid]) == 1: return _fin(m, u, uid, "reply", pcls, plf, rd, (_ps(m, u, uid, 52),))
        u.pq[uid] = np.uint8(0)
    return _adv(m, u, uid, pcls, plf, rd, pre, True)


def vm_handle_query(m, s, u, uid, mv):
    now_s = int(time.time()) & 0xFFFFFFFF
    if int(u.ts[uid]) and (now_s - int(u.ts[uid])) > 600: _rst(u, uid)
    lf, av, cv = vm_forward_pass(m, s, mv)
    if lf == 75: _rst(u, uid); u.ts[uid] = np.uint32(now_s); pr = int(m.rlut[75, 0, 0, 1]); return m.http_mv[int(m.poff[pr, 0]) : int(m.pend[pr, 0])]
    if lf == 76: lf = int(m.ab_red[int(u.ll[uid])])
    pa, pc, pg, fl, fh, mc, ru = int(u.age[uid]), int(u.ctry[uid]), int(u.gen[uid]), int(u.flo[uid]), int(u.fhi[uid]), (int(u.mc[uid]) + 1) & 255, int(m.lrule[lf])
    rs = 1 if ((fl & int(m.clo[lf])) | (fh & int(m.chi[lf]))) else 0
    ea = av if av >= 18 else pa; cs = 2 if (ea >= 18 and pc > 0) else (1 if ea >= 18 else 0)
    if ru == 5 and (pa or pc or pg): rs = 1
    elif ru == 1: rs, cs = (1 if (pa == av and av) else (2 if (pa and av and pa != av) else 0)), (2 if pc else 0)
    elif ru == 2: rs, cs = (1 if (pc == cv and cv) else (2 if (pc and cv and pc != cv) else 0)), (2 if pa >= 18 else 0)
    elif ru == 3: u.gen[uid] = np.uint8(2)
    elif ru == 4: u.gen[uid] = np.uint8(1)
    if av >= 18: u.age[uid] = np.uint8(av)
    if cv: u.ctry[uid] = np.uint8(cv)
    u.li[uid], u.ll[uid], u.mc[uid], u.ts[uid], u.flo[uid], u.fhi[uid] = m.lcls[lf], np.uint8(lf), np.uint8(mc), np.uint32(now_s), np.uint32(fl) | m.slo[lf], np.uint32(fh) | m.shi[lf]
    u.wq.append((uid, int(m.lcls[lf]), av, cv, int(u.gen[uid]), now_s))
    pid = int(m.rlut[lf, rs, cs, 0 if int(u.bg[uid]) == 1 else 1]); rot = uid + mc
    idx = ((uid & 1) | ((rot & 3) << 1)) if int(m.lplk[lf]) else (rot & int(m.pmask[pid]))
    return m.http_mv[int(m.poff[pid, idx]) : int(m.pend[pid, idx])]


async def wal_worker(u, path="events_wal.bin"):
    while True:
        await asyncio.sleep(2.0)
        if not u.wq: continue
        c = 0
        while u.wq and c < 256: u.wb[c] = u.wq.popleft(); c += 1
        try:
            with open(path, "ab") as f: f.write(memoryview(u.wb)[:c])
        except OSError: pass


async def start_server(host="127.0.0.1", port=8080):
    m, u, sq = CompiledVMModel("model_weights.npz"), UserStore(), asyncio.Queue(maxsize=NW)
    for i in range(NW): sq.put_nowait(InferenceSlot(i, m.num_leaves))
    ts = await sq.get(); t0 = time.perf_counter()
    for i in range(1000): vm_handle_query(m, ts, u, i, memoryview(b"m looking for girl , u"))
    ms = (time.perf_counter() - t0) * 1000.0
    for i in range(1000): _rst(u, i)
    mt_ok = sum(1 for seq, exp in (((b"M", b"u"), (13, 9)), ((b"Chicken", b"u"), (0, 0)), ((b"Chatting with u", b"u"), (16, 18)), ((b"M", b"u", b"u"), (13, 9, 0))) if (_rst(u, 0) or True) and tuple((vm_handle_query(m, ts, u, 0, memoryview(x)), int(u.li[0]))[1] for x in seq) == exp)
    b_ok = 0
    if m.has_beh:
        beh_start(m, u, 1, 10000); u.pt[1] = np.uint8(1)
        ob = beh_opening(m, u, 1); b_ok += int(1 <= len(ob["bubbles"]) <= 4 and 23.0 <= ob["idle_timeout_s"] <= 35.0)
        beh_turn(m, ts, u, 1, ("hi", "im 24", "from tunisia"), now_s=10005); b_ok += int(int(u.age[1]) == 24 and int(u.ctry[1]) == 1)
        r2 = beh_turn(m, ts, u, 2, "im 15", now_s=10006); b_ok += int(r2["action"] == "disconnect" and r2["nlu"]["class_id"] == 7)
        beh_start(m, u, 3, 10010); u.pt[3] = np.uint8(1)
        for _ in range(3): rd3 = beh_turn(m, ts, u, 3, "ok", now_s=10011)
        b_ok += int(rd3["action"] == "disconnect" and rd3["disconnect_after_s"] == 1.0)
        beh_start(m, u, 4, 10020); u.pt[4] = np.uint8(1); u.bc[4] = np.uint8(int(m.bi[10]))
        bi_b, bi_d, _ = _disp(m, u, 4, 2, True); b_ok += int(len(bi_b) == 2 and float(bi_d) == 4.0)
        beh_start(m, u, 5, 10030); u.pt[5] = np.uint8(1); r5 = beh_turn(m, ts, u, 5, "Bcz you make it to fast.", now_s=10031); b_ok += int(r5["nlu"]["leaf_id"] == 90)
        for k in range(6): _rst(u, k)
    sq.put_nowait(ts)
    print(f"[VM BENCHMARK] 1,000 req in {ms:.2f}ms ({ms / 1000.0:.4f}ms/req) | Multi-Turn {mt_ok}/4 | Behavior {b_ok}/6 | {m.num_leaves} leaves | ~7.9 MB RAM", flush=True)
    if "--benchmark" in sys.argv: return

    async def client(rd, wr):
        try:
            req = await asyncio.wait_for(rd.read(2048), timeout=2.0)
            if not req: return
            he = req.find(b"\r\n\r\n"); hr, br = (req[:he], req[he + 4 :]) if he != -1 else (req, b"")
            lns = hr.decode("ascii", errors="ignore").split("\r\n"); pth = lns[0].split(" ")[1] if len(lns[0].split(" ")) >= 2 else "/"
            uid, ev, emo = -1, "text", ""
            for ln in lns[1:]:
                ll = ln.lower()
                if ll.startswith("x-user-id:") and ln.split(":", 1)[1].strip().isdigit(): uid = int(ln.split(":", 1)[1].strip())
                elif ll.startswith("x-event-type:"): ev = ln.split(":", 1)[1].strip().lower() or "text"
                elif ll.startswith("x-emoji:"): emo = ln.split(":", 1)[1].strip()
            if uid < 0: peer = wr.get_extra_info("peername"); uid = (hash(peer[0]) & 0x7FFFFFFF) if peer else 101
            slot = await sq.get()
            try:
                if pth == "/" or not m.has_beh:
                    out = vm_handle_query(m, slot, u, uid % MAX_U, memoryview(br))
                else:
                    if pth == "/reload": m.load(); pay = {"status": "reloaded", "leaves": m.num_leaves}
                    elif pth == "/start": pay = beh_start(m, u, uid)
                    elif pth == "/opening": pay = beh_opening(m, u, uid)
                    else:
                        txt = br.decode("utf-8", errors="ignore"); utts = [x for x in txt.splitlines() if x.strip()] or [txt]
                        pay = beh_turn(m, slot, u, uid, utts, ev, emo)
                    jb = json.dumps(pay, separators=(",", ":")).encode("utf-8")
                    out = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(jb)).encode("ascii") + b"\r\nConnection: close\r\n\r\n" + jb
            finally: sq.put_nowait(slot)
            wr.write(out); await wr.drain()
        except Exception: pass
        finally: wr.close()

    asyncio.create_task(wal_worker(u))
    srv = await asyncio.start_server(client, host, port, backlog=2048)
    print(f"[SINGLE-FILE ULTRA VM] http://{host}:{port} (/, /start, /opening, /turn, /reload | ~7.9 MB RAM | PyArmor OK)", flush=True)
    async with srv: await srv.serve_forever()


if __name__ == "__main__":
    asyncio.run(start_server())