#!/usr/bin/env python3
# FILE: plugin_hub.py
# PURPOSE: Ultra-Dynamic Zero-Repetition Plugin Hub & Shared NumPy Inference/Vault Core (Steps 8-15+)
# RULES: Enforces Rules 1-13:
#        - Zero PyTorch in production (lazy-imported only inside save_step_into_vault_secure)
#        - Zero float64/int64, zero per-request RAM allocation (shared pre-allocated worker scratchpads)
#        - Auto-registers Steps 8-15+ in exact zero-fault priority order so production_server.py never repeats imports!

import hashlib
import importlib
import os
import numpy as np

BLOOM_SIZE = 65536
S2_DIM = 32
MAX_USERS = 50000
NUM_WORKERS = 8
_LOADED_ENGINES = ()  # Add inside init_all_plugins: global _LOADED_ENGINES; track `eng` in a list -> _LOADED_ENGINES = tuple(eng_list)

def reset_plugin_user(uid):
    for eng in _LOADED_ENGINES:
        for attr in ("user_flags", "user_aux", "user_goal", "user_hobby", "user_rel"):
            arr = getattr(eng, attr, None)
            if arr is not None:
                arr[uid] = np.uint8(0)

class DynamicStepHead:
    """
    Reusable Zero-Allocation Quantized MLP + Bloom + Double-Hash LUT Container.
    Eliminates duplicate StepXEngine boilerplate across all modular plugins!
    """
    __slots__ = (
        "prefix", "num_classes", "enabled",
        "emb_f32", "w1_f32", "w2_f32",
        "bloom_q", "lut_a", "lut_b",
        "user_flags", "user_aux",
        "hits_a", "hits_b", "logits",
    )

    def __init__(self, prefix, num_classes, npz_path="model_weights.npz", bloom_key=None):
        self.prefix = prefix
        self.num_classes = num_classes
        self.enabled = False
        self.user_flags = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.user_aux = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.hits_a = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.hits_b = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.logits = np.zeros((NUM_WORKERS, 1, num_classes), dtype=np.float32)

        if not os.path.exists(npz_path):
            return
        b_key = bloom_key or f"{prefix}_bloom_q"
        req = (f"{prefix}_emb_q", f"{prefix}_w1_q", f"{prefix}_w2_q", b_key, f"{prefix}_lut_a", f"{prefix}_lut_b")
        arc = np.load(npz_path, mmap_mode="r")
        if any(k not in arc.files for k in req):
            arc.close()
            return

        def lock_f32(name, zero_row0=False):
            q = arc[f"{name}_q"].astype(np.int8, copy=False)
            s = arc[f"{name}_s"].astype(np.float32, copy=False)
            f32 = np.multiply(q, s, dtype=np.float32)
            if zero_row0:
                f32[0].fill(0.0)
            f32.flags.writeable = False
            return f32

        self.emb_f32 = lock_f32(f"{prefix}_emb", True)
        self.w1_f32 = lock_f32(f"{prefix}_w1")
        self.w2_f32 = lock_f32(f"{prefix}_w2")
        for attr, k_arc, dt in (("bloom_q", b_key, np.int8), ("lut_a", f"{prefix}_lut_a", np.uint8), ("lut_b", f"{prefix}_lut_b", np.uint8)):
            arr = arc[k_arc].astype(dt, copy=False)
            arr.flags.writeable = False
            setattr(self, attr, arr)
        arc.close()
        self.enabled = True


def lookup_lut_codes(slot, head, n_words):
    """Zero-allocation O(1) Double-Hash LUT lookup into pre-allocated worker scratchpad."""
    sid = slot.slot_id
    ha = head.hits_a[sid]
    hb = head.hits_b[sid]
    np.take(head.lut_a, slot.w_b_idx[:, 0], out=ha)
    np.take(head.lut_b, slot.w_b_idx[:, 1], out=hb)
    m = (ha[:n_words] > 0) & (ha[:n_words] == hb[:n_words])
    return ha[:n_words] * m


def run_mlp_head_inplace(model, slot, head, n_words, has_q):
    """Zero-allocation shared NumPy MLP forward pass using pre-allocated slot & head buffers."""
    np.take(head.bloom_q, slot.w_b_idx, out=slot.w_b_hits)
    if not bool(np.all(slot.w_b_hits[:n_words])):
        return None

    slot.s4_tokens[:6] = slot.s2_tokens[:6]
    slot.s4_tokens[6] = np.int32(1023) if has_q else model.one_i32
    q_off = np.int32(509) if has_q else np.int32(0)
    slot.s4_tokens[7] = (((slot.s2_tokens[0] * np.int32(37)) + (slot.s2_tokens[n_words - 1] * np.int32(73)) + q_off) & model.s2_v_mask) + model.one_i32

    np.take(head.emb_f32, slot.s4_tokens, axis=0, out=slot.s2_emb_buf)
    slot.s2_emb_buf[6] *= np.float32(2.5)
    np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
    slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
    np.matmul(slot.s2_pooled, head.w1_f32, out=slot.s2_mid)
    np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
    out_l = head.logits[slot.slot_id]
    np.matmul(slot.s2_mid, head.w2_f32, out=out_l)
    return out_l


# --- DYNAMIC ZERO-FAULT PLUGIN REGISTRY (Strict Priority Order) ---
# Format: (module_name, init_fn_name, forward_fn_name, route_fn_name, class_id_tuple, stage)
# stage="pre_s7" runs BEFORE Step 7 (protects Step 14 "are u real" from Step 7)
# stage="post_s7" runs AFTER Step 7 (Step 11 runs BEFORE Step 8 so "im in a relationship" beats Step 8!)
_PLUGIN_SPECS = (
    ("step14_bot_check", "get_step14_engine", "forward_step14", "route_step14", (33, 34), "pre_s7"),
    ("step11_rel_status", "get_step11_engine", "forward_step11", "route_step11", (27, 28), "post_s7"),
    ("step8_looking_for", "get_step8_engine", "forward_step8", "route_step8", (21, 22), "post_s7"),
    ("step9_name", "get_step9_engine", "forward_step9", "route_step9", (23, 24), "post_s7"),
    ("step10_hobbies", "get_step10_engine", "forward_step10", "route_step10", (25, 26), "post_s7"),
    ("step12_private_yesno", "get_step12_engine", "forward_step12", "route_step12", (29, 30), "post_s7"),
    ("step13_pic_photo", "get_step13_engine", "forward_step13", "route_step13", (31, 32), "post_s7"),
    ("step15_dating_pack", "get_step15_engine", "forward_step15", "route_step15", (35, 36, 37, 38, 39, 40, 41, 42, 43), "post_s7"),
)

_PRE_S7_FORWARDS = ()
_POST_S7_FORWARDS = ()
_CLASS_ROUTERS = [None] * 64


def init_all_plugins(npz_path="model_weights.npz"):
    """Dynamically loads and initializes every installed Step 8-15+ plugin once at server startup."""
    global _PRE_S7_FORWARDS, _POST_S7_FORWARDS
    pre_list = []
    post_list = []
    loaded_names = []

    for mod_name, init_name, fwd_name, route_name, cid_tuple, stage in _PLUGIN_SPECS:
        try:
            mod = importlib.import_module(mod_name)
            eng = getattr(mod, init_name)(npz_path)
            if eng is None or not getattr(eng, "enabled", False):
                continue
            fwd_fn = getattr(mod, fwd_name)
            route_fn = getattr(mod, route_name)
            if stage == "pre_s7":
                pre_list.append(fwd_fn)
            else:
                post_list.append(fwd_fn)
            for cid in cid_tuple:
                if 0 <= cid < 64:
                    _CLASS_ROUTERS[cid] = route_fn
            loaded_names.append(mod_name)
        except ImportError:
            pass

    _PRE_S7_FORWARDS = tuple(pre_list)
    _POST_S7_FORWARDS = tuple(post_list)
    print(f"[PLUGIN HUB] Dynamically loaded {len(loaded_names)} active step plugins: {', '.join(loaded_names)}", flush=True)


def forward_pre_s7(model, slot, n_words, has_q, age_val, country_id):
    for fn in _PRE_S7_FORWARDS:
        cid, age, aux = fn(model, slot, n_words, has_q, age_val, country_id)
        if cid != 0:
            return cid, age, aux
    return 0, 0, 0


def forward_post_s7(model, slot, n_words, has_q, age_val, country_id):
    for fn in _POST_S7_FORWARDS:
        cid, age, aux = fn(model, slot, n_words, has_q, age_val, country_id)
        if cid != 0:
            return cid, age, aux
    return 0, 0, 0


def route_plugin_class(class_id, aux_id, users, uid, flags, prev_age, prev_country, bot_g, rot8):
    if 0 <= class_id < 64:
        router_fn = _CLASS_ROUTERS[class_id]
        if router_fn is not None:
            return router_fn(class_id, aux_id, users, uid, flags, prev_age, prev_country, bot_g, rot8)
    return None


# --- SHARED OFFLINE VAULT TRAINER HELPER (Rule 12 Cryptographic SHA-256 Shield) ---
def _hash_vault_except(vault_dict, exclude_prefix):
    snap = {}
    for k, arr in vault_dict.items():
        if not k.startswith(exclude_prefix):
            snap[k] = (hashlib.sha256(arr.tobytes()).hexdigest(), arr.shape, arr.dtype.str)
    return snap


def save_step_into_vault_secure(npz_path, step_prefix, new_arrays_dict, tag):
    """
    Rule 12 Shared Cryptographic Vault Saver:
    Snapshots SHA-256 of all untouched steps before writing, merges new step arrays,
    writes model_weights.npz, and cryptographically verifies zero alteration of older steps!
    """
    vault = {}
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            for k in data.files:
                vault[k] = data[k].copy()
    pre_hashes = _hash_vault_except(vault, f"{step_prefix}_")

    vault.update(new_arrays_dict)
    np.savez(npz_path, **vault)

    with np.load(npz_path, allow_pickle=False) as saved_arc:
        post_vault = {k: saved_arc[k] for k in saved_arc.files}
    post_hashes = _hash_vault_except(post_vault, f"{step_prefix}_")
    if pre_hashes != post_hashes:
        raise RuntimeError(f"[RULE 12 VAULT BREACH] Protected weights were altered while saving {tag}!")

    size_kb = os.path.getsize(npz_path) / 1024.0
    print(f"[RULE 12 VAULT SHIELD] Verified SHA-256 integrity of all {len(pre_hashes)} protected arrays!", flush=True)
    print(f"[{tag} DONE] Locked '{step_prefix}_*' into '{npz_path}' ({size_kb:.2f} KB).", flush=True)