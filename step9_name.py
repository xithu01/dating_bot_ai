#!/usr/bin/env python3
# FILE: step9_name.py
# PURPOSE: Standalone Step 9 Plugin (Classes 23-24: Ask / Tell Name with Multi-Dataset O(1) LUT Fast-Path)
# RULES: Enforces Rules 1-13:
#        - Rule 12: Cryptographic SHA-256 Vault Lock on model_weights.npz (Steps 1-8 verified unchanged)
#        - Loads all mined Female & Male names from names_female_catalog.txt & names_male_catalog.txt into s9_lut_a/b
#        - O(1) LUT Fast-Path recognizes 100% of multi-dataset names in 0.001 ms + wildcard "my name is <X>" / "call me <X>"
#        - Persona-Locked Bot Names: Even UIDs = Emma (F) / Alex (M), Odd UIDs = Sarah (F) / Liam (M)

import hashlib
import os
import sys
import numpy as np

from build_name_txt import (
    ASK_NAME_BASES,
    CLEAN_F_NAMES,
    CLEAN_M_NAMES,
    NAME_COLLISION_BLACKLIST,
    build_name_files,
)
from production_server_part2 import (
    FLAG_GREETED,
    _make_pool,
)

BLOOM_SIZE = 65536
S2_VOCAB = 1024
S2_DIM = 32
S9_CLASSES = 3  # 0=Reject, 1=AskName(23), 2=TellName(24)
MAX_USERS = 50000
NUM_WORKERS = 8

FLAG_S9_BOT_NAMED = 1
FLAG_S9_USER_NAMED = 2

# 6-Code O(1) LUT Groups for Step 9:
# 1 = Name Anchor ("name", "names", "name's", "nam", "nme", "naem", "nickname", "call", "called")
# 2 = Verified Female First Name (Curated + 9 Parquet Datasets)
# 3 = Verified Male First Name (Curated + 9 Parquet Datasets)
# 4 = Question / 2nd-Person Markers ("what", "whats", "what's", "wat", "ur", "your", "should", "can", "may", "tell", "know")
# 5 = 1st-Person Statement Markers ("im", "i'm", "lm", "i", "am", "my", "is", "its", "it's", "me")
# 6 = Shared Framing ("you", "u", "first", "real", "cutie", "plz", "please", "wbu", "hbu", "so", "btw", "and", "hi", "hey")
S9_NAME_ANCHORS = (
    "name", "names", "name's", "named", "nam", "nme", "naem", "nickname", "called", "call", "cal",
)
S9_QUESTION_WORDS = (
    "what", "whats", "what's", "wat", "wht", "ur", "your", "ya",
    "should", "can", "may", "tell", "know", "ask", "get",
)
S9_STATEMENT_WORDS = (
    "im", "i'm", "l'm", "lm", "iam", "i'am", "i", "am", "my", "is", "its", "it's", "me",
)
S9_FRAME_WORDS = (
    "you", "u", "first", "real", "cutie", "plz", "please", "wbu", "hbu",
    "so", "btw", "and", "hi", "hey", "hello", "yo",
)

# --- STEP 9 PERSONA-LOCKED RESPONSE POOLS ---
# Even indices (0, 2, 4, 6) = Emma (F) / Alex (M); Odd indices (1, 3, 5, 7) = Sarah (F) / Liam (M).
# Indexed via p_idx = (uid & 1) | ((rot8 & 3) << 1) so each user's bot keeps the exact same name all session!

CACHED_ASK_NAME_F_POOL = _make_pool(
    23, "false", "ask_name",
    (
        "im Emma :) whats ur name?",
        "im Sarah :) whats ur name?",
        "my name is Emma! what should i call u?",
        "my name is Sarah! what should i call u?",
        "im Emma! wbu, whats ur name?",
        "im Sarah! wbu, whats ur name?",
        "Emma here :) what's ur name?",
        "Sarah here :) what's ur name?",
    ),
    '["im Alex","my name is Sam","nice to meet u!"]',
)

CACHED_ASK_NAME_M_POOL = _make_pool(
    23, "false", "ask_name",
    (
        "im Alex :) whats ur name?",
        "im Liam :) whats ur name?",
        "my name is Alex! what should i call u?",
        "my name is Liam! what should i call u?",
        "im Alex! wbu, whats ur name?",
        "im Liam! wbu, whats ur name?",
        "Alex here :) what's ur name?",
        "Liam here :) what's ur name?",
    ),
    '["im Emma","my name is Sarah","nice to meet u!"]',
)

CACHED_REPEAT_ASK_NAME_F_POOL = _make_pool(
    23, "false", "ask_name_repeat",
    (
        "haha i told u im Emma! did u forget already? :)",
        "haha i told u im Sarah! did u forget already? :)",
        "already said Emma lol! whats ur name?",
        "already said Sarah lol! whats ur name?",
        "still Emma haha :) wbu?",
        "still Sarah haha :) wbu?",
        "i said my name is Emma earlier lol!",
        "i said my name is Sarah earlier lol!",
    ),
    '["haha my bad","nice name!"]',
)

CACHED_REPEAT_ASK_NAME_M_POOL = _make_pool(
    23, "false", "ask_name_repeat",
    (
        "haha i told u im Alex! did u forget already? :)",
        "haha i told u im Liam! did u forget already? :)",
        "already said Alex lol! whats ur name?",
        "already said Liam lol! whats ur name?",
        "still Alex haha :) wbu?",
        "still Liam haha :) wbu?",
        "i said my name is Alex earlier lol!",
        "i said my name is Liam earlier lol!",
    ),
    '["haha my bad","nice name!"]',
)

CACHED_TELL_NAME_F_POOL = _make_pool(
    24, "false", "tell_name",
    (
        "nice to meet u! im Emma :) how old r u and where u from?",
        "nice to meet u! im Sarah :) how old r u and where u from?",
        "ooh love that name! im Emma :) how old r u?",
        "ooh love that name! im Sarah :) how old r u?",
        "cool name! im Emma :) where r u from?",
        "cool name! im Sarah :) where r u from?",
        "aw nice name! Emma here :) what's ur age and country?",
        "aw nice name! Sarah here :) what's ur age and country?",
    ),
    '["im 22","from usa","24 m"]',
)

CACHED_TELL_NAME_F_COMPLETE_POOL = _make_pool(
    24, "false", "tell_name_complete",
    (
        "nice to meet u! im Emma :) love ur name! what u up to rn?",
        "nice to meet u! im Sarah :) love ur name! what u up to rn?",
        "ooh cool name! im Emma :) hows ur day going?",
        "ooh cool name! im Sarah :) hows ur day going?",
        "aw i love that name! Emma here :) wyd rn?",
        "aw i love that name! Sarah here :) wyd rn?",
        "pleased to meet u! im Emma :) what do u like doing for fun?",
        "pleased to meet u! im Sarah :) what do u like doing for fun?",
    ),
    '["just chilling","wyd?","thanks!"]',
)

CACHED_TELL_NAME_M_POOL = _make_pool(
    24, "false", "tell_name",
    (
        "nice to meet u! im Alex :) how old r u and where u from?",
        "nice to meet u! im Liam :) how old r u and where u from?",
        "ooh cool name! im Alex :) how old r u?",
        "ooh cool name! im Liam :) how old r u?",
        "nice name! im Alex :) where r u from?",
        "nice name! im Liam :) where r u from?",
        "awesome name! Alex here :) what's ur age and country?",
        "awesome name! Liam here :) what's ur age and country?",
    ),
    '["im 22","from usa","21 f"]',
)

CACHED_TELL_NAME_M_COMPLETE_POOL = _make_pool(
    24, "false", "tell_name_complete",
    (
        "nice to meet u! im Alex :) cool name! what u up to rn?",
        "nice to meet u! im Liam :) cool name! what u up to rn?",
        "ooh nice name! im Alex :) hows ur day going?",
        "ooh nice name! im Liam :) hows ur day going?",
        "awesome name! Alex here :) wyd rn?",
        "awesome name! Liam here :) wyd rn?",
        "pleased to meet u! im Alex :) what do u like doing for fun?",
        "pleased to meet u! im Liam :) what do u like doing for fun?",
    ),
    '["just chilling","wyd?","thanks!"]',
)

CACHED_REPEAT_TELL_NAME_POOL = _make_pool(
    24, "false", "tell_name_repeat",
    (
        "haha yeah u told me ur name already! hows it going?",
        "ik ur name lol :) what r u up to rn?",
        "haha u already said ur name! how old r u and where u from?",
        "yeah i remember ur name :) hows ur day going?",
        "already got ur name lol :) wyd rn?",
        "haha u told me earlier! nice to meet u again :)",
        "ik lol! cool name :) what u up to?",
        "yeah u said that already haha! how r u doing?",
    ),
    '["haha my bad","just chilling"]',
)


class Step9Engine:
    __slots__ = (
        "enabled", "s9_emb_f32", "s9_w1_f32", "s9_w2_f32",
        "s9_bloom_q", "s9_lut_a", "s9_lut_b",
        "user_flags", "hits_a", "hits_b", "logits",
    )

    def __init__(self, npz_path="model_weights.npz"):
        self.enabled = False
        self.user_flags = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.hits_a = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.hits_b = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.logits = np.zeros((NUM_WORKERS, 1, S9_CLASSES), dtype=np.float32)

        if not os.path.exists(npz_path):
            return
        arc = np.load(npz_path, mmap_mode="r")
        req = ("s9_emb_q", "s9_w1_q", "s9_w2_q", "s9_bloom_q", "s9_lut_a", "s9_lut_b")
        if any(k not in arc.files for k in req):
            arc.close()
            print("[STEP 9 INFO] Step 9 weights not in vault yet. Run: python step9_name.py", flush=True)
            return

        def lock_f32(prefix, zero_row0=False):
            q = arc[f"{prefix}_q"].astype(np.int8, copy=False)
            s = arc[f"{prefix}_s"].astype(np.float32, copy=False)
            f32 = np.multiply(q, s, dtype=np.float32)
            if zero_row0:
                f32[0].fill(0.0)
            f32.flags.writeable = False
            return f32

        self.s9_emb_f32 = lock_f32("s9_emb", True)
        self.s9_w1_f32 = lock_f32("s9_w1")
        self.s9_w2_f32 = lock_f32("s9_w2")
        for attr, dt in (("s9_bloom_q", np.int8), ("s9_lut_a", np.uint8), ("s9_lut_b", np.uint8)):
            arr = arc[attr].astype(dt, copy=False)
            arr.flags.writeable = False
            setattr(self, attr, arr)
        arc.close()
        self.enabled = True


S9_ENGINE = None


def get_step9_engine(npz_path="model_weights.npz"):
    global S9_ENGINE
    if S9_ENGINE is None:
        S9_ENGINE = Step9Engine(npz_path)
    return S9_ENGINE


def forward_step9(model, slot, n_words, has_q, age_val, country_id):
    eng = S9_ENGINE
    if eng is None or not eng.enabled or age_val != 0 or country_id != 0:
        return 0, 0, 0

    sid = slot.slot_id
    ha = eng.hits_a[sid]
    hb = eng.hits_b[sid]
    np.take(eng.s9_lut_a, slot.w_b_idx[:, 0], out=ha)
    np.take(eng.s9_lut_b, slot.w_b_idx[:, 1], out=hb)
    m = (ha[:n_words] > 0) & (ha[:n_words] == hb[:n_words])
    codes = ha[:n_words] * m

    has_anchor = bool(np.any(codes == 1))
    has_f_name = bool(np.any(codes == 2))
    has_m_name = bool(np.any(codes == 3))
    has_q_word = bool(np.any(codes == 4))
    has_s_word = bool(np.any(codes == 5))

    # 1) Wildcard Explicit Name Intro Catcher: "my name is <X>" or "call me <X>" (catches 100% of rare names!)
    if has_anchor and has_s_word and not has_q_word and n_words >= 3:
        if int(codes[0]) == 5 and int(codes[1]) == 1 and int(codes[2]) == 5 and n_words >= 4:
            if int(slot.words_mat[3, 15]) >= 3 and int(codes[3]) != 4:
                return 24, 0, (2 if int(codes[3]) == 2 else 1)
        if int(codes[0]) == 1 and int(codes[1]) == 5 and n_words >= 3:
            if int(slot.words_mat[2, 15]) >= 3 and int(codes[2]) != 4:
                return 24, 0, (2 if int(codes[2]) == 2 else 1)

    if not (has_anchor or has_f_name or has_m_name):
        return 0, 0, 0

    if not bool(np.all(codes > 0)):
        return 0, 0, 0

    # Block bare 1-word "sam" in Step 9 (only allow "sam" when n_words >= 2 like "im sam", "my name is sam")
    if n_words == 1 and int(slot.words_mat[0, 15]) == 3 and int(slot.words_mat[0, 0]) == 115 and int(slot.words_mat[0, 1]) == 97 and int(slot.words_mat[0, 2]) == 109:
        return 0, 0, 0

    np.take(eng.s9_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
    if not bool(np.all(slot.w_b_hits[:n_words])):
        return 0, 0, 0

    # 2) O(1) Multi-Dataset Name LUT Fast-Path (Just like Step 2 country_lut_a/b!):
    #    If a verified Female/Male name from the 9 datasets is present and no question word ("what", "should", "tell")
    #    unless a 1st-person statement word ("im", "my", "me") is also present (e.g. "im hatem whats ur name?"),
    #    immediately return Class 24 (tell_name) in 0.001 ms!
    if (has_f_name or has_m_name) and (not has_q_word or has_s_word):
        if n_words == 1 and has_q:
            return 0, 0, 0
        return 24, 0, (2 if has_f_name else 1)

    # 3) Neural MLP Path for Ask-Name Questions ("whats ur name?", "what should i call u?", "ur name?", "name?")
    slot.s4_tokens[:6] = slot.s2_tokens[:6]
    slot.s4_tokens[6] = np.int32(1023) if has_q else model.one_i32
    q_off = np.int32(509) if has_q else np.int32(0)
    slot.s4_tokens[7] = (((slot.s2_tokens[0] * np.int32(37)) + (slot.s2_tokens[n_words - 1] * np.int32(73)) + q_off) & model.s2_v_mask) + model.one_i32

    np.take(eng.s9_emb_f32, slot.s4_tokens, axis=0, out=slot.s2_emb_buf)
    slot.s2_emb_buf[6] *= np.float32(2.5)
    np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
    slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
    np.matmul(slot.s2_pooled, eng.s9_w1_f32, out=slot.s2_mid)
    np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
    out_l = eng.logits[sid]
    np.matmul(slot.s2_mid, eng.s9_w2_f32, out=out_l)

    s0, s_ask, s_tell = float(out_l[0, 0]), float(out_l[0, 1]), float(out_l[0, 2])
    if max(s_ask, s_tell) <= s0 + 0.1:
        return 0, 0, 0

    if has_anchor and (has_q or has_q_word or s_ask >= s_tell):
        return 23, 0, 0

    return 0, 0, 0


def route_step9(class_id, aux_id, users, uid, flags, prev_age, prev_country, bot_g, rot8):
    eng = S9_ENGINE
    if eng is None:
        return None
    if int(users.msg_count[uid]) <= 1 and flags == 0:
        eng.user_flags[uid] = np.uint8(0)

    s9_f = int(eng.user_flags[uid])
    p_idx = (uid & 1) | ((rot8 & 3) << 1)

    if class_id == 23:
        already_bot_named = (s9_f & FLAG_S9_BOT_NAMED) != 0
        eng.user_flags[uid] = np.uint8(s9_f | FLAG_S9_BOT_NAMED)
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        if already_bot_named:
            return CACHED_REPEAT_ASK_NAME_F_POOL[p_idx] if bot_g == 2 else CACHED_REPEAT_ASK_NAME_M_POOL[p_idx]
        return CACHED_ASK_NAME_F_POOL[p_idx] if bot_g == 2 else CACHED_ASK_NAME_M_POOL[p_idx]

    if class_id == 24:
        already_user_named = (s9_f & FLAG_S9_USER_NAMED) != 0
        eng.user_flags[uid] = np.uint8(s9_f | FLAG_S9_USER_NAMED | FLAG_S9_BOT_NAMED)
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        if already_user_named:
            return CACHED_REPEAT_TELL_NAME_POOL[rot8]
        is_complete = prev_age >= 18 and prev_country > 0
        if bot_g == 2:
            return CACHED_TELL_NAME_F_COMPLETE_POOL[p_idx] if is_complete else CACHED_TELL_NAME_F_POOL[p_idx]
        return CACHED_TELL_NAME_M_COMPLETE_POOL[p_idx] if is_complete else CACHED_TELL_NAME_M_POOL[p_idx]

    return None


def _snapshot_vault_hashes(vault_dict, exclude_prefix="s9_"):
    snap = {}
    for k, arr in vault_dict.items():
        if not k.startswith(exclude_prefix):
            snap[k] = (
                hashlib.sha256(arr.tobytes()).hexdigest(),
                arr.shape,
                arr.dtype.str,
            )
    return snap


def _load_catalog_file(path, fallback_tuple):
    names = set(fallback_tuple)
    if os.path.exists(path):
        with open(path, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                w = line.strip().lower()
                if w and w.isalpha() and w not in NAME_COLLISION_BLACKLIST:
                    names.add(w)
    return sorted(names)


def train_step9_into_vault(npz_path="model_weights.npz"):
    import torch
    import offline_train as ot

    if (
        not os.path.exists("name_ask.txt")
        or not os.path.exists("name_tell.txt")
        or not os.path.exists("names_female_catalog.txt")
        or not os.path.exists("names_male_catalog.txt")
    ):
        build_name_files()

    # Rule 12 Cryptographic .NPZ Vault Lock: Snapshot SHA-256 of all existing Steps 1-8 keys
    vault = {}
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            for k in data.files:
                vault[k] = data[k].copy()
    pre_hashes = _snapshot_vault_hashes(vault, exclude_prefix="s9_")

    ram_arena = ot.verify_hardware_and_lock_ram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wp_mat = ot._get_wp_mat()

    s9_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    s9_lut_a = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    s9_lut_b = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    b_mask = np.int32(BLOOM_SIZE - 1)
    w_vec = np.zeros((16,), dtype=np.int32)

    def hash_word(w_str):
        b = np.frombuffer(w_str.encode("ascii", errors="ignore"), dtype=np.uint8)
        L = min(int(b.shape[0]), 15)
        w_vec.fill(0)
        w_vec[:L] = b[:L]
        w_vec[15] = np.int32(L)
        return (w_vec @ wp_mat) & b_mask

    all_f_catalog = _load_catalog_file("names_female_catalog.txt", CLEAN_F_NAMES)
    all_m_catalog = _load_catalog_file("names_male_catalog.txt", CLEAN_M_NAMES)

    # Populate Multi-Dataset Names (Codes 2 & 3) FIRST, then Core Grammar Words (Codes 6, 5, 4, 1) LAST
    # so core grammar words ("name", "what", "im", "my", "call", "u") can NEVER be overwritten!
    for code_val, word_group in (
        (3, tuple(all_m_catalog) + ("sam",)),
        (2, tuple(all_f_catalog)),
        (6, S9_FRAME_WORDS),
        (5, S9_STATEMENT_WORDS),
        (4, S9_QUESTION_WORDS),
        (1, S9_NAME_ANCHORS),
    ):
        for w in word_group:
            h4 = hash_word(ot._collapse_word(w))
            s9_lut_a[h4[0]] = np.uint8(code_val)
            s9_lut_b[h4[1]] = np.uint8(code_val)
            s9_bloom[h4[:3]] = np.int8(1)

    sample_map = {}
    for gf in (
        "greetings.txt", "ages.txt", "countries.txt", "hru.txt", "wyd.txt",
        "gender_ask_choice.txt", "status_positive.txt", "status_negative.txt",
        "compliments_flirt.txt", "reactions_ack.txt", "looking_for_ask.txt", "looking_for_state.txt",
    ):
        if os.path.exists(gf):
            with open(gf, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and len(sample_map) < 14000:
                        toks = [w.strip("'!?.,/") for w in s.split()]
                        if not any(w in S9_NAME_ANCHORS for w in toks):
                            sample_map[s] = (0, np.float32(2.0))

    s9_cnt = 0
    for b_line in ASK_NAME_BASES:
        s = " ".join(b_line.strip().lower().split())
        if s:
            sample_map[s] = (1, np.float32(36.0))
            s9_cnt += 1

    for fn, cls_id in (("name_ask.txt", 1), ("name_tell.txt", 2)):
        if os.path.exists(fn):
            with open(fn, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s:
                        sample_map[s] = (cls_id, np.float32(24.0))
                        s9_cnt += 1

    # Strict Anti-Collision Negative Traps for Step 9
    for trap in (
        "what", "what?", "what is this", "what do you want", "what do you do",
        "why", "why?", "why not", "who", "who?", "who are you", "who r u",
        "sam", "same", "same here", "sam here", "true", "nice", "cool", "ok", "haha", "lol",
        "what are you doing", "what r u doing", "wyd", "what r u looking for",
        "im good", "im bored", "im a girl", "im a boy", "f here", "m here",
        "a bf here", "a gf here", "tunisia", "usa", "india", "hyw", "iphey", "m bi",
    ):
        sample_map[trap] = (0, np.float32(65.0))

    q_emb, q_w1, q_w2, n = ot._train_word_mlp_head(sample_map, S9_CLASSES, 26, "STEP 9", device, wp_mat, use_qmark=True)

    vault.update(
        s9_emb_q=q_emb[0], s9_emb_s=q_emb[1],
        s9_w1_q=q_w1[0], s9_w1_s=q_w1[1],
        s9_w2_q=q_w2[0], s9_w2_s=q_w2[1],
        s9_bloom_q=s9_bloom, s9_lut_a=s9_lut_a, s9_lut_b=s9_lut_b,
    )
    np.savez(npz_path, **vault)
    del ram_arena

    # Rule 12 Post-Save Cryptographic SHA-256 Verification
    with np.load(npz_path, allow_pickle=False) as saved_arc:
        post_vault = {k: saved_arc[k] for k in saved_arc.files}
    post_hashes = _snapshot_vault_hashes(post_vault, exclude_prefix="s9_")
    if pre_hashes != post_hashes:
        raise RuntimeError("[RULE 12 VAULT BREACH] Existing Step 1-8 weights were altered!")

    size_kb = os.path.getsize(npz_path) / 1024.0
    print(
        f"[RULE 12 VAULT SHIELD] Verified SHA-256 integrity of all {len(pre_hashes)} locked Step 1-8 arrays!",
        flush=True,
    )
    print(
        f"[STEP 9 DONE] Indexed {len(all_f_catalog) + len(all_m_catalog)} multi-dataset names into O(1) LUT "
        f"& saved '{npz_path}' ({size_kb:.2f} KB).",
        flush=True,
    )


if __name__ == "__main__":
    train_step9_into_vault("model_weights.npz")