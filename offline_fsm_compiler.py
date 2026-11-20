#!/usr/bin/env python3
# FILE: offline_fsm_compiler.py
# PURPOSE: Rule 15 Generic Offline Spec Auto-Discovery, HTTP Blob Packer & 4D O(1) FSM Table Compiler
# RULES: Enforces Rules 1-15 (<= 220 lines, zero float64/int64, auto-loads all offline_spec_*.py modules)

import glob
import importlib
import os
import numpy as np

# --- 64-BIT CONVERSATION FLAGS SPLIT INTO TWO np.uint32 BITMASKS (ZERO int64/uint64!) ---
F_ZERO = np.uint32(0)
F_GREETED = np.uint32(1 << 0)
F_ASKED_AGE = np.uint32(1 << 1)
F_ASKED_COUNTRY = np.uint32(1 << 2)
F_ASKED_HRU = np.uint32(1 << 3)
F_ASKED_GENDER = np.uint32(1 << 4)
F_ASKED_WYD = np.uint32(1 << 5)
F_TOLD_STATUS = np.uint32(1 << 6)
F_FLIRTED = np.uint32(1 << 7)
F_ASKED_LOOKING = np.uint32(1 << 8)
F_STATED_LOOKING = np.uint32(1 << 9)
F_BOT_NAMED = np.uint32(1 << 10)
F_USER_NAMED = np.uint32(1 << 11)
F_ASKED_REL = np.uint32(1 << 12)
F_ASKED_PRIV = np.uint32(1 << 13)
F_ASKED_PIC = np.uint32(1 << 14)
F_BOT_CHECKED = np.uint32(1 << 15)

# Universal Leaf Entity Rules (leaf_entity_rule_u8)
RULE_STD = 0
RULE_AGE = 1
RULE_COUNTRY = 2
RULE_SET_G_F = 3
RULE_SET_G_M = 4
RULE_GREET = 5


class CompiledSpecBundle:
    __slots__ = (
        "num_leaves", "fsm_tables", "leaf_step_map", "core_seeds",
        "zero_fault_suite", "multi_turn_suite", "txt_loaders", "spec_names",
    )

    def __init__(self, num_leaves, fsm_tables, leaf_step_map, core_seeds, zf_suite, mt_suite, txt_loaders, spec_names):
        self.num_leaves = num_leaves
        self.fsm_tables = fsm_tables
        self.leaf_step_map = leaf_step_map
        self.core_seeds = core_seeds
        self.zero_fault_suite = zf_suite
        self.multi_turn_suite = mt_suite
        self.txt_loaders = txt_loaders
        self.spec_names = spec_names


def _pack_http(class_id, is_greeting_str, intent_str, reply_str, kbd_str):
    body = (
        f'{{"class_id":{class_id},"is_greeting":{is_greeting_str},'
        f'"intent":"{intent_str}","reply":"{reply_str}","kbd":{kbd_str}}}'
    ).encode("utf-8")
    hdr = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
        + str(len(body)).encode("ascii")
        + b"\r\nConnection: close\r\n\r\n"
    )
    return hdr + body


def discover_and_compile_all_specs(spec_pattern="offline_spec_*.py"):
    """Auto-discovers every offline_spec_*.py module and compiles ALL pools, FSM tables, seeds & tests."""
    spec_files = sorted(glob.glob(spec_pattern))
    if not spec_files:
        raise FileNotFoundError(f"[FSM COMPILER ERROR] No '{spec_pattern}' modules found in '{os.getcwd()}'!")

    all_pools = {}
    all_bindings = []
    all_redirects = []
    leaf_step_map = {}
    core_seeds = {}
    zf_list = []
    mt_list = []
    txt_loaders = []
    spec_names = []

    for sf in spec_files:
        mod_name = os.path.splitext(os.path.basename(sf))[0]
        mod = importlib.import_module(mod_name)
        spec_names.append(mod_name)
        all_pools.update(getattr(mod, "POOLS", {}))
        all_bindings.extend(getattr(mod, "LEAF_BINDINGS", ()))
        all_redirects.extend(getattr(mod, "ASKBACK_REDIRECTS", ()))
        leaf_step_map.update(getattr(mod, "LEAF_STEP_BITS", {}))
        for lf, phrases in getattr(mod, "CORE_LEAF_SEEDS", {}).items():
            core_seeds[lf] = core_seeds.get(lf, ()) + tuple(phrases)
        zf_list.extend(getattr(mod, "ZERO_FAULT_SUITE", ()))
        mt_list.extend(getattr(mod, "MULTI_TURN_SUITE", ()))
        loader_fn = getattr(mod, "load_txt_samples", None)
        if callable(loader_fn):
            txt_loaders.append(loader_fn)

    num_leaves = max(b[0] for b in all_bindings) + 1
    pool_names = list(all_pools.keys())
    pid = {name: i for i, name in enumerate(pool_names)}
    n_pools = len(pool_names)

    # 1. Pack Contiguous HTTP Blob + Offset/End LUTs (Zero-Copy memoryview in production!)
    blob_chunks = []
    cur_off = 0
    pool_off = np.zeros((n_pools, 32), dtype=np.int32)
    pool_end = np.zeros((n_pools, 32), dtype=np.int32)
    pool_mask = np.zeros((n_pools,), dtype=np.uint8)

    for p_idx, name in enumerate(pool_names):
        cid, is_g, int_s, msgs, kbd = all_pools[name]
        pool_mask[p_idx] = np.uint8(len(msgs) - 1)
        for j, m_txt in enumerate(msgs):
            raw_b = _pack_http(cid, is_g, int_s, m_txt, kbd)
            pool_off[p_idx, j] = np.int32(cur_off)
            cur_off += len(raw_b)
            pool_end[p_idx, j] = np.int32(cur_off)
            blob_chunks.append(raw_b)

    http_blob_u8 = np.frombuffer(b"".join(blob_chunks), dtype=np.uint8).copy()

    # 2. Compile 4D O(1) Finite-State Transition Table [leaf, rep_st(0..2), cmp_st(0..2), bot_g(0=M, 1=F)]
    route_lut = np.zeros((num_leaves, 3, 3, 2), dtype=np.uint16)
    leaf_cls = np.zeros((num_leaves,), dtype=np.uint8)
    leaf_rule = np.zeros((num_leaves,), dtype=np.uint8)
    leaf_plock = np.zeros((num_leaves,), dtype=np.uint8)
    chk_lo = np.zeros((num_leaves,), dtype=np.uint32)
    chk_hi = np.zeros((num_leaves,), dtype=np.uint32)
    set_lo = np.zeros((num_leaves,), dtype=np.uint32)
    set_hi = np.zeros((num_leaves,), dtype=np.uint32)

    for leaf, cid, p_norm, kw in all_bindings:
        leaf_cls[leaf] = np.uint8(cid)
        leaf_rule[leaf] = np.uint8(kw.get("rule", RULE_STD))
        leaf_plock[leaf] = np.uint8(kw.get("plock", 0))
        chk_lo[leaf] = np.uint32(kw.get("chk", F_ZERO))
        chk_hi[leaf] = np.uint32(kw.get("chk_hi", F_ZERO))
        set_lo[leaf] = np.uint32(kw.get("set_f", F_GREETED))
        set_hi[leaf] = np.uint32(kw.get("set_hi", F_ZERO))

        p_cmp, p_rep, p_chg = kw.get("p_cmp"), kw.get("p_rep"), kw.get("p_chg")
        p_part, p_m = kw.get("p_part"), kw.get("p_m")
        p_m_cmp, p_m_rep, p_m_part = kw.get("p_m_cmp"), kw.get("p_m_rep"), kw.get("p_m_part")

        pf_0 = pid[p_norm]
        pf_p = pid[p_part] if p_part else pf_0
        pf_c = pid[p_cmp] if p_cmp else pf_0
        pf_r = pid[p_rep] if p_rep else pf_0
        pf_x = pid[p_chg] if p_chg else pf_r

        pm_0 = pid[p_m] if p_m else pf_0
        pm_p = pid[p_m_part] if p_m_part else (pid[p_part] if (p_part and not p_m) else pm_0)
        pm_c = pid[p_m_cmp] if p_m_cmp else (pid[p_cmp] if (p_cmp and not p_m) else pm_0)
        pm_r = pid[p_m_rep] if p_m_rep else (pid[p_rep] if (p_rep and not p_m) else pm_0)
        pm_x = pid[p_chg] if p_chg else pm_r

        route_lut[leaf, 0, 0, 1], route_lut[leaf, 0, 1, 1], route_lut[leaf, 0, 2, 1] = pf_0, pf_p, pf_c
        route_lut[leaf, 0, 0, 0], route_lut[leaf, 0, 1, 0], route_lut[leaf, 0, 2, 0] = pm_0, pm_p, pm_c
        route_lut[leaf, 1, :, 1], route_lut[leaf, 1, :, 0] = pf_r, pm_r
        route_lut[leaf, 2, :, 1], route_lut[leaf, 2, :, 0] = pf_x, pm_x

    # 3. Compile O(1) Context Ask-Back Redirect Table
    askback_redirect = np.zeros((num_leaves,), dtype=np.uint8)
    for prev_lf, target_lf in all_redirects:
        if 0 <= prev_lf < num_leaves and 0 <= target_lf < num_leaves:
            askback_redirect[prev_lf] = np.uint8(target_lf)

    fsm_tables = {
        "http_blob_u8": http_blob_u8,
        "pool_msg_off_i32": pool_off,
        "pool_msg_end_i32": pool_end,
        "pool_mask_u8": pool_mask,
        "route_pool_lut": route_lut,
        "leaf_to_class_u8": leaf_cls,
        "leaf_entity_rule_u8": leaf_rule,
        "leaf_persona_lock_u8": leaf_plock,
        "leaf_chk_lo_u32": chk_lo,
        "leaf_chk_hi_u32": chk_hi,
        "leaf_set_lo_u32": set_lo,
        "leaf_set_hi_u32": set_hi,
        "askback_redirect_u8": askback_redirect,
    }
    return CompiledSpecBundle(
        num_leaves, fsm_tables, leaf_step_map, core_seeds,
        tuple(zf_list), tuple(mt_list), tuple(txt_loaders), tuple(spec_names),
    )


if __name__ == "__main__":
    bundle = discover_and_compile_all_specs()
    kb = sum(a.nbytes for a in bundle.fsm_tables.values()) / 1024.0
    print(f"[FSM COMPILER OK] Auto-loaded {len(bundle.spec_names)} specs ({', '.join(bundle.spec_names)}) -> {bundle.num_leaves} leaves, {len(bundle.zero_fault_suite)} tests ({kb:.2f} KB)", flush=True)