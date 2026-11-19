#!/usr/bin/env python3
# FILE: step11_rel_status.py
# PURPOSE: Standalone Step 11 Plugin (Classes 27-28: Ask / Tell Relationship Status)
# RULES: Enforces Rules 1-13:
#        - Rule 12: Cryptographic SHA-256 Vault Lock on model_weights.npz (Steps 1-10 verified unchanged)
#        - Handles "are you single?", "are you single or mariid", "do u have a bf/gf?", "im single u?", "im married", "im taken"
#        - Smart Negation Handling ("im not married" & "i dont have a gf" -> Single, never misclassified as Married/Taken!)
#        - Zero collision with Step 8 ("looking for a relationship" / "a gf" stays Step 8; "im in a relationship" routes here!)

import hashlib
import os
import sys
import numpy as np

from build_rel_status_txt import (
    ASK_REL_BASES,
    CLEAN_TELL_REL_TEMPLATES,
    S11_FRAME_WORDS,
    S11_MARRIED_WORDS,
    S11_SINGLE_WORDS,
    S11_STATEMENT_WORDS,
    S11_STATUS_ANCHORS,
    S11_TAKEN_WORDS,
    STRICT_ALLOWED_S11_VOCAB,
    build_rel_status_files,
)
from production_server_part2 import (
    FLAG_GREETED,
    _make_pool,
)

BLOOM_SIZE = 65536
S2_VOCAB = 1024
S2_DIM = 32
S11_CLASSES = 3  # 0=Reject, 1=AskRelStatus(27), 2=TellRelStatus(28)
MAX_USERS = 50000
NUM_WORKERS = 8

FLAG_S11_ASKED = 1
FLAG_S11_TOLD = 2

# 8-Code O(1) LUT Groups for Step 11:
# 1 = Single / Available / Divorced ("single", "singel", "singl", "divorced", "available", "unmarried")
# 2 = Married / Engaged / Spouse ("married", "maried", "mariid", "marrid", "engaged", "wife", "husband")
# 3 = Taken / Relationship / BF / GF ("taken", "takn", "committed", "relationship", "relashionship", "bf", "gf", "boyfriend", "girlfriend")
# 4 = Status Anchor ("status")
# 5 = Ownership Verbs ("have", "hav", "got")
# 6 = Negation Words ("dont", "don't", "not", "no" — flips "not married" / "dont have a gf" into Single!)
# 7 = Question / Choice Markers ("are", "ar", "r", "is", "do", "what", "whats", "what's", "wat", "or")
# 8 = Statement & Shared Framing ("im", "lm", "i", "am", "you", "u", "ur", "your", "in", "a", "here", "wbu", "hbu", "still")
S11_HAVE_VERBS = ("have", "hav", "got")
S11_NEG_WORDS = ("dont", "don't", "not", "no")
S11_Q_MARKERS = ("are", "ar", "r", "is", "do", "what", "whats", "what's", "wat", "or")
S11_STMT_FRAME = S11_STATEMENT_WORDS + S11_FRAME_WORDS + ("currently", "still")

ALL_S11_CORE_SET = frozenset(S11_SINGLE_WORDS + S11_MARRIED_WORDS + ("taken", "takn", "committed", "status"))

# --- STEP 11 PERSONA-AWARE & STATUS-CONFIRMING RESPONSE POOLS (8 items each) ---

# Class 27 (User asks "are you single?", "are you single or mariid?", "do u have a bf/gf?") -> Bot F (bot_g == 2)
CACHED_ASK_REL_F_POOL = _make_pool(
    27, "false", "ask_relationship_status",
    (
        "im single :) no bf right now! wbu, are u single or married?",
        "yep im single! not married or taken :) what about u?",
        "single here :) no boyfriend lol! wbu, single or taken?",
        "im 100% single :) not married! how about u?",
        "single and free :) no bf! are u single or taken?",
        "yep im single! never been married :) wbu?",
        "im single :) what about u, single or in a relationship?",
        "single here! no bf right now :) are u single too?",
    ),
    '["im single","im single u?","im taken"]',
)

# Class 27 -> Bot M (bot_g == 1)
CACHED_ASK_REL_M_POOL = _make_pool(
    27, "false", "ask_relationship_status",
    (
        "im single :) no gf right now! wbu, are u single or married?",
        "yep im single! not married or taken :) what about u?",
        "single here :) no girlfriend lol! wbu, single or taken?",
        "im 100% single :) not married! how about u?",
        "single and free :) no gf! are u single or taken?",
        "yep im single! never been married :) wbu?",
        "im single :) what about u, single or in a relationship?",
        "single here! no gf right now :) are u single too?",
    ),
    '["im single","im single u?","im taken"]',
)

CACHED_REPEAT_ASK_REL_POOL = _make_pool(
    27, "false", "ask_relationship_status_repeat",
    (
        "haha i told u im single! not married or taken :) wbu?",
        "already said im single lol! are u single too?",
        "still single haha :) forgot already? what about u?",
        "haha yep still 100% single! wbu, single or married?",
        "i told u earlier im single lol :) how about u?",
        "still single here haha :) what r u up to?",
        "haha yes im single! u asked me that already :)",
        "still single and chilling lol :) wbu?",
    ),
    '["haha my bad","im single too"]',
)

# Class 28 (User is Single / Not Married / No BF or GF / Divorced — aux_id == 1)
CACHED_TELL_SINGLE_POOL = _make_pool(
    28, "false", "tell_rel_single",
    (
        "ooh nice! im single too :) how old r u and where u from?",
        "same here, im single too :) how old r u?",
        "nice! we are both single then :) where r u from?",
        "ooh cool, im single too :) what's ur age and country?",
        "haha nice! single here too :) how old are u and where u from?",
        "sweet, im single as well :) what country u in?",
        "ooh awesome! im single too :) how old r u?",
        "nice! both single :) ur age and where r u from?",
    ),
    '["im 22","from usa","24 m"]',
)

CACHED_TELL_SINGLE_COMPLETE_POOL = _make_pool(
    28, "false", "tell_rel_single_complete",
    (
        "ooh nice! im single too :) maybe we will vibe! what u up to rn?",
        "same here, single too :) hows ur day going?",
        "haha we are both single then ;) what r u looking for here?",
        "nice! im single too :) what do u like doing for fun?",
        "ooh sweet, single here too :) ur fun to talk to!",
        "awesome! both single :) just chilling here right now, wbu?",
        "nice! im single as well :) wyd rn?",
        "ooh cool, im single too :) glad we matched!",
    ),
    '["just chilling","ur cute","wyd?"]',
)

# Class 28 (User is Taken / In a Relationship — aux_id == 2)
CACHED_TELL_TAKEN_POOL = _make_pool(
    28, "false", "tell_rel_taken",
    (
        "aw nice, happy for u! im single myself, always down to chat as friends :) hows ur day?",
        "oh cool! im single, happy to just chill and talk as friends :) what u up to?",
        "nice, wish u guys the best! im single, down to make new friends :) wyd?",
        "aw thats sweet! im single myself, love making cool friends here :) how r u?",
        "oh nice! im single, always happy to chat as friends :) what r ur hobbies?",
        "cool, happy for u :) im single, just chilling and making friends! wbu?",
        "aw nice! im single myself, we can totally chat as friends :) hows it going?",
        "oh awesome! im single, always down for friendly chat :) what u doing rn?",
    ),
    '["just making friends","hows ur day?","wyd?"]',
)

# Class 28 (User is Married / Engaged — aux_id == 3)
CACHED_TELL_MARRIED_POOL = _make_pool(
    28, "false", "tell_rel_married",
    (
        "oh nice! im single myself, always happy to chat as friends :) hows ur day going?",
        "aw nice, happy for u! im single, down to just chill and talk as friends :) wyd?",
        "oh cool! im single myself, love making friendly conversation :) what u up to?",
        "nice, wish u all the best! im single, happy to chat as friends :) how r u?",
        "oh awesome! im single, always down for cool friendly chat :) what r ur hobbies?",
        "aw nice! im single myself, just chilling and making friends here :) wbu?",
        "oh cool! im single, we can totally just talk as friends :) hows it going?",
        "nice! im single myself, always down to pass time chatting :) what u doing rn?",
    ),
    '["just chatting","hows ur day?","wyd?"]',
)


class Step11Engine:
    __slots__ = (
        "enabled", "s11_emb_f32", "s11_w1_f32", "s11_w2_f32",
        "s11_bloom_q", "s11_lut_a", "s11_lut_b",
        "user_flags", "user_rel", "hits_a", "hits_b", "logits",
    )

    def __init__(self, npz_path="model_weights.npz"):
        self.enabled = False
        self.user_flags = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.user_rel = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.hits_a = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.hits_b = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.logits = np.zeros((NUM_WORKERS, 1, S11_CLASSES), dtype=np.float32)

        if not os.path.exists(npz_path):
            return
        arc = np.load(npz_path, mmap_mode="r")
        req = ("s11_emb_q", "s11_w1_q", "s11_w2_q", "s11_bloom_q", "s11_lut_a", "s11_lut_b")
        if any(k not in arc.files for k in req):
            arc.close()
            print("[STEP 11 INFO] Step 11 weights not in vault yet. Run: python step11_rel_status.py", flush=True)
            return

        def lock_f32(prefix, zero_row0=False):
            q = arc[f"{prefix}_q"].astype(np.int8, copy=False)
            s = arc[f"{prefix}_s"].astype(np.float32, copy=False)
            f32 = np.multiply(q, s, dtype=np.float32)
            if zero_row0:
                f32[0].fill(0.0)
            f32.flags.writeable = False
            return f32

        self.s11_emb_f32 = lock_f32("s11_emb", True)
        self.s11_w1_f32 = lock_f32("s11_w1")
        self.s11_w2_f32 = lock_f32("s11_w2")
        for attr, dt in (("s11_bloom_q", np.int8), ("s11_lut_a", np.uint8), ("s11_lut_b", np.uint8)):
            arr = arc[attr].astype(dt, copy=False)
            arr.flags.writeable = False
            setattr(self, attr, arr)
        arc.close()
        self.enabled = True


S11_ENGINE = None


def get_step11_engine(npz_path="model_weights.npz"):
    global S11_ENGINE
    if S11_ENGINE is None:
        S11_ENGINE = Step11Engine(npz_path)
    return S11_ENGINE


def forward_step11(model, slot, n_words, has_q, age_val, country_id):
    eng = S11_ENGINE
    if eng is None or not eng.enabled or age_val != 0 or country_id != 0:
        return 0, 0, 0

    sid = slot.slot_id
    ha = eng.hits_a[sid]
    hb = eng.hits_b[sid]
    np.take(eng.s11_lut_a, slot.w_b_idx[:, 0], out=ha)
    np.take(eng.s11_lut_b, slot.w_b_idx[:, 1], out=hb)
    m = (ha[:n_words] > 0) & (ha[:n_words] == hb[:n_words])
    codes = ha[:n_words] * m

    if not bool(np.all(codes > 0)):
        return 0, 0, 0

    has_single = bool(np.any(codes == 1))
    has_married = bool(np.any(codes == 2))
    has_taken = bool(np.any(codes == 3))
    has_status = bool(np.any(codes == 4))
    has_have = bool(np.any(codes == 5))
    has_neg = bool(np.any(codes == 6))
    has_q_word = bool(np.any(codes == 7))

    # Check if "in" ("in a relationship") or "taken"/"committed" is present
    # This strictly protects Step 8 direct answers ("a gf", "a bf", "a relationship", "dating")!
    has_in = False
    has_explicit_taken = False
    for w_i in range(n_words):
        c0 = int(slot.words_mat[w_i, 0])
        Lw = int(slot.words_mat[w_i, 15])
        if Lw == 2 and c0 == 105 and int(slot.words_mat[w_i, 1]) == 110:
            has_in = True
        elif int(codes[w_i]) == 3 and (c0 == 116 or c0 == 99):  # "taken" / "takn" / "committed"
            has_explicit_taken = True

    has_valid_s11_anchor = (
        has_single
        or has_married
        or has_status
        or has_explicit_taken
        or (has_taken and (has_have or has_neg or has_in))
    )
    if not has_valid_s11_anchor:
        return 0, 0, 0

    np.take(eng.s11_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
    if not bool(np.all(slot.w_b_hits[:n_words])):
        return 0, 0, 0

    slot.s4_tokens[:6] = slot.s2_tokens[:6]
    slot.s4_tokens[6] = np.int32(1023) if has_q else model.one_i32
    q_off = np.int32(509) if has_q else np.int32(0)
    slot.s4_tokens[7] = (((slot.s2_tokens[0] * np.int32(37)) + (slot.s2_tokens[n_words - 1] * np.int32(73)) + q_off) & model.s2_v_mask) + model.one_i32

    np.take(eng.s11_emb_f32, slot.s4_tokens, axis=0, out=slot.s2_emb_buf)
    slot.s2_emb_buf[6] *= np.float32(2.5)
    np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
    slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
    np.matmul(slot.s2_pooled, eng.s11_w1_f32, out=slot.s2_mid)
    np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
    out_l = eng.logits[sid]
    np.matmul(slot.s2_mid, eng.s11_w2_f32, out=out_l)

    s0, s_ask, s_tell = float(out_l[0, 0]), float(out_l[0, 1]), float(out_l[0, 2])
    if max(s_ask, s_tell) <= s0 + 0.1:
        return 0, 0, 0

    # Check if sentence starts with "u" / "you" / "ur" / "your" / "got" (e.g. "u single?", "got a bf?")
    w0_c0 = int(slot.words_mat[0, 0])
    w0_L = int(slot.words_mat[0, 15])
    starts_2nd_or_got = (
        (w0_c0 == 117 and w0_L <= 2)  # "u", "ur"
        or (w0_c0 == 121 and w0_L in (3, 4))  # "you", "your"
        or (w0_c0 == 103 and w0_L == 3 and int(codes[0]) == 5)  # "got"
    )

    if has_q_word or has_status or starts_2nd_or_got or (n_words == 1 and has_q):
        return 27, 0, 0

    # Statement overrides trailing "u?" / "wbu?"!
    # Smart Negation: "im not married", "im not taken", "i dont have a gf", "no bf" -> status_id = 1 (Single!)
    if has_neg:
        status_id = 1
    elif has_married:
        status_id = 3
    elif has_single:
        status_id = 1
    else:
        status_id = 2

    return 28, 0, status_id


def route_step11(class_id, aux_id, users, uid, flags, prev_age, prev_country, bot_g, rot8):
    eng = S11_ENGINE
    if eng is None:
        return None
    if int(users.msg_count[uid]) <= 1 and flags == 0:
        eng.user_flags[uid] = np.uint8(0)
        eng.user_rel[uid] = np.uint8(0)

    s11_f = int(eng.user_flags[uid])
    if class_id == 27:
        already_asked = (s11_f & FLAG_S11_ASKED) != 0
        eng.user_flags[uid] = np.uint8(s11_f | FLAG_S11_ASKED)
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        if already_asked:
            return CACHED_REPEAT_ASK_REL_POOL[rot8]
        return CACHED_ASK_REL_F_POOL[rot8] if bot_g == 2 else CACHED_ASK_REL_M_POOL[rot8]

    if class_id == 28:
        eng.user_flags[uid] = np.uint8(s11_f | FLAG_S11_TOLD)
        if aux_id > 0:
            eng.user_rel[uid] = np.uint8(aux_id)
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        if aux_id == 3:
            return CACHED_TELL_MARRIED_POOL[rot8]
        if aux_id == 2:
            return CACHED_TELL_TAKEN_POOL[rot8]
        is_complete = prev_age >= 18 and prev_country > 0
        return CACHED_TELL_SINGLE_COMPLETE_POOL[rot8] if is_complete else CACHED_TELL_SINGLE_POOL[rot8]

    return None


def _snapshot_vault_hashes(vault_dict, exclude_prefix="s11_"):
    snap = {}
    for k, arr in vault_dict.items():
        if not k.startswith(exclude_prefix):
            snap[k] = (
                hashlib.sha256(arr.tobytes()).hexdigest(),
                arr.shape,
                arr.dtype.str,
            )
    return snap


def train_step11_into_vault(npz_path="model_weights.npz"):
    import torch
    import offline_train as ot

    if not os.path.exists("rel_status_ask.txt") or not os.path.exists("rel_status_tell.txt"):
        build_rel_status_files()

    # Rule 12 Cryptographic .NPZ Vault Lock: Snapshot SHA-256 of all existing Steps 1-10 keys
    vault = {}
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            for k in data.files:
                vault[k] = data[k].copy()
    pre_hashes = _snapshot_vault_hashes(vault, exclude_prefix="s11_")

    ram_arena = ot.verify_hardware_and_lock_ram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wp_mat = ot._get_wp_mat()

    s11_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    s11_lut_a = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    s11_lut_b = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    b_mask = np.int32(BLOOM_SIZE - 1)
    w_vec = np.zeros((16,), dtype=np.int32)

    def hash_word(w_str):
        b = np.frombuffer(w_str.encode("ascii", errors="ignore"), dtype=np.uint8)
        L = min(int(b.shape[0]), 15)
        w_vec.fill(0)
        w_vec[:L] = b[:L]
        w_vec[15] = np.int32(L)
        return (w_vec @ wp_mat) & b_mask

    for code_val, word_group in (
        (8, S11_STMT_FRAME),
        (7, S11_Q_MARKERS),
        (6, S11_NEG_WORDS),
        (5, S11_HAVE_VERBS),
        (4, S11_STATUS_ANCHORS),
        (3, S11_TAKEN_WORDS),
        (2, S11_MARRIED_WORDS),
        (1, S11_SINGLE_WORDS),
    ):
        for w in word_group:
            h4 = hash_word(ot._collapse_word(w))
            s11_lut_a[h4[0]] = np.uint8(code_val)
            s11_lut_b[h4[1]] = np.uint8(code_val)
            s11_bloom[h4[:3]] = np.int8(1)

    sample_map = {}
    for gf in (
        "greetings.txt", "ages.txt", "countries.txt", "hru.txt", "wyd.txt",
        "gender_ask_choice.txt", "status_positive.txt", "status_negative.txt",
        "compliments_flirt.txt", "reactions_ack.txt",
        "looking_for_ask.txt", "looking_for_state.txt",
        "name_ask.txt", "hobbies_ask.txt", "hobbies_tell.txt",
    ):
        if os.path.exists(gf):
            with open(gf, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and len(sample_map) < 14000:
                        toks = [w.strip("'!?.,/") for w in s.split()]
                        if not any(w in ALL_S11_CORE_SET for w in toks):
                            sample_map[s] = (0, np.float32(2.0))

    s11_cnt = 0
    for fn, cls_id, builtins in (
        ("rel_status_ask.txt", 1, ASK_REL_BASES),
        ("rel_status_tell.txt", 2, CLEAN_TELL_REL_TEMPLATES),
    ):
        for b_line in builtins:
            s = " ".join(b_line.strip().lower().split())
            if s:
                sample_map[s] = (cls_id, np.float32(36.0))
                s11_cnt += 1
        if os.path.exists(fn):
            with open(fn, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and ot._is_whitelist_clean(s, STRICT_ALLOWED_S11_VOCAB):
                        sample_map[s] = (cls_id, np.float32(24.0))
                        s11_cnt += 1

    # Strict Anti-Collision Negative Traps (Protects Step 8 "a gf", "a bf", "a relationship" & Rule 13 "a bf here")
    for trap in (
        "what", "what?", "what is this", "what do you want", "what do you do",
        "why", "why?", "why not", "who", "who?", "who are you", "who r u",
        "a gf", "gf", "a bf", "bf", "a relationship", "relationship", "dating",
        "looking for a gf", "looking for a bf", "looking for a relationship",
        "m looking for girl , u", "lm finding gril", "a bf here", "a gf here",
        "f here", "m here", "im a girl", "im a boy", "m or f?",
        "im good", "im bored", "same", "sam", "ur cute", "whats ur name", "what r ur hobbies",
    ):
        sample_map[trap] = (0, np.float32(65.0))

    q_emb, q_w1, q_w2, n = ot._train_word_mlp_head(sample_map, S11_CLASSES, 28, "STEP 11", device, wp_mat, use_qmark=True)

    vault.update(
        s11_emb_q=q_emb[0], s11_emb_s=q_emb[1],
        s11_w1_q=q_w1[0], s11_w1_s=q_w1[1],
        s11_w2_q=q_w2[0], s11_w2_s=q_w2[1],
        s11_bloom_q=s11_bloom, s11_lut_a=s11_lut_a, s11_lut_b=s11_lut_b,
    )
    np.savez(npz_path, **vault)
    del ram_arena

    # Rule 12 Post-Save Cryptographic SHA-256 Verification
    with np.load(npz_path, allow_pickle=False) as saved_arc:
        post_vault = {k: saved_arc[k] for k in saved_arc.files}
    post_hashes = _snapshot_vault_hashes(post_vault, exclude_prefix="s11_")
    if pre_hashes != post_hashes:
        raise RuntimeError("[RULE 12 VAULT BREACH] Existing Step 1-10 weights were altered!")

    size_kb = os.path.getsize(npz_path) / 1024.0
    print(f"[RULE 12 VAULT SHIELD] Verified SHA-256 integrity of all {len(pre_hashes)} locked Step 1-10 arrays!", flush=True)
    print(f"[STEP 11 DONE] Locked Step 11 Relationship Status head & LUT into '{npz_path}' ({s11_cnt} lines, {n} total, {size_kb:.2f} KB).", flush=True)


if __name__ == "__main__":
    train_step11_into_vault("model_weights.npz")