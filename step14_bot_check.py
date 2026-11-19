#!/usr/bin/env python3
# FILE: step14_bot_check.py
# PURPOSE: Standalone Step 14 Plugin (Classes 33-34: Bot/Fake Denial & Real Person Confirmation)
# RULES: Enforces Rules 1-13:
#        - Rule 12: Cryptographic SHA-256 Vault Lock on model_weights.npz (Steps 1-13 verified unchanged)
#        - NEVER says "yes" to bot checks — ALWAYS clearly replies "no, im not a bot! im a real person :)"
#        - Protects Step 7 ("for real", bare "real"), Step 4 ("are you a girl?"), Step 9 ("real name"), Class 0 ("who are you")

import hashlib
import os
import sys
import numpy as np

from build_bot_check_txt import (
    ASK_IS_BOT_BASES,
    ASK_IS_REAL_BASES,
    S14_BOT_FAKE_WORDS,
    S14_FRAME_WORDS,
    S14_PROVE_WORDS,
    S14_QUESTION_WORDS,
    S14_REAL_HUMAN_WORDS,
    STRICT_ALLOWED_S14_VOCAB,
    build_bot_check_files,
)
from production_server_part2 import (
    FLAG_GREETED,
    _make_pool,
)

BLOOM_SIZE = 65536
S2_VOCAB = 1024
S2_DIM = 32
S14_CLASSES = 3  # 0=Reject, 1=AskIsBotFake(33), 2=AskIsRealHuman(34)
MAX_USERS = 50000
NUM_WORKERS = 8

FLAG_S14_CHECKED = 1

# 5-Code O(1) LUT Groups for Step 14:
# 1 = Bot / AI / Robot / Fake / Catfish / Scam ("bot", "ai", "robot", "fake", "catfish", "scam", "chatgpt", "gpt")
# 2 = Real / Human / Person / Alive / Legit ("real", "reel", "human", "person", "alive", "legit")
# 3 = Prove / Action ("prove", "proof", "verify", "sound", "talk", "reply", "answer", "act", "seem")
# 4 = Question / Choice Markers ("are", "ar", "r", "is", "am", "do", "why", "or")
# 5 = Shared Framing ("u", "you", "ur", "a", "an", "like", "not", "girl", "boy", "fast", "so", "too")
ALL_S14_ANCHOR_SET = frozenset(S14_BOT_FAKE_WORDS + S14_REAL_HUMAN_WORDS + ("prove", "proof", "verify"))

# --- STEP 14 RESPONSE POOLS (NEVER says "yes" — ALWAYS clearly states "no, im not a bot / im 100% real!") ---

# Class 33 (User asks "are u a bot?", "bot?", "u fake?", "are u ai?", "bot or real?") -> Explicit NO!
CACHED_DENY_BOT_POOL = _make_pool(
    33, "false", "deny_is_bot",
    (
        "haha no im not a bot! im a real person :) how old r u and where u from?",
        "no way lol, im not a bot! just bored and chatting here :) how old r u?",
        "nope, not a bot or ai! im 100% real :) where r u from?",
        "nah im not a bot haha! why would u think that? whats ur age and country?",
        "no lol im definitely not a bot! just chilling in bed texting u :) how old r u?",
        "nope im a real person, not a bot haha :) where u from?",
        "haha no! im 100% real, not a bot or fake :) how old are u?",
        "no im not a bot lol :) just a fast typer! where r u from?",
    ),
    '["haha ok sorry","im 22","from usa"]',
)

CACHED_DENY_BOT_COMPLETE_POOL = _make_pool(
    33, "false", "deny_is_bot_complete",
    (
        "haha no im not a bot! im 100% real :) just a fast typer lol, wyd rn?",
        "no way lol, im not a bot! just chilling in bed texting u :) hows ur day going?",
        "nope, definitely not a bot or ai! im a real person :) what music do u like?",
        "nah im not a bot haha! why would u think that? what r u up to?",
        "no lol im not fake or a bot :) just bored and chatting here! wbu?",
        "nope im a real human, not a bot haha :) what do u like doing for fun?",
        "haha no! im 100% real, promise :) wyd rn?",
        "no im not a bot lol! i just type fast on my phone :) hows it going?",
    ),
    '["haha ok sorry","just chilling","wyd?"]',
)

# Class 34 (User asks "are u real?", "are u human?", "prove ur real") -> Clear "im 100% real, not a bot!" (NO bare "yes")
CACHED_CONFIRM_REAL_POOL = _make_pool(
    34, "false", "confirm_is_real",
    (
        "im 100% real, not a bot haha! :) how old r u and where u from?",
        "im a real person lol, definitely not a bot :) how old r u?",
        "haha im real, not a bot or fake! :) where r u from?",
        "of course im real, not a bot :) just bored and chatting here! ur age and country?",
        "im a real human haha, not a bot :) how old are u?",
        "definitely real, not a bot lol :) just chilling in bed! where u from?",
        "im 100% real and not a bot :) why do u ask? how old r u?",
        "real person right here, not a bot haha :) what country u from?",
    ),
    '["haha good","im 22","from usa"]',
)

CACHED_CONFIRM_REAL_COMPLETE_POOL = _make_pool(
    34, "false", "confirm_is_real_complete",
    (
        "im 100% real, not a bot haha! :) just chilling and chatting here, wyd rn?",
        "im a real person lol, definitely not a bot :) hows ur day going?",
        "haha im real, not a bot or fake! :) what do u like doing for fun?",
        "of course im real, not a bot :) just a fast typer on my phone! wbu?",
        "im a real human haha, not a bot :) what music or movies r u into?",
        "definitely real, not a bot lol :) why would u think otherwise? wyd?",
        "im 100% real and not a bot :) just vibing here right now!",
        "real person right here, not a bot haha :) what r u up to today?",
    ),
    '["haha good","just chilling","wyd?"]',
)

CACHED_REPEAT_BOT_CHECK_POOL = _make_pool(
    33, "false", "deny_is_bot_repeat",
    (
        "haha no, still not a bot lol! i promise im 100% real :)",
        "nope still a real person haha! why don't u believe me? :)",
        "nah im really not a bot lol! just a real person chilling here :)",
        "haha no! u asked me that already, im 100% real not a bot :)",
        "nope still human haha :) i just type fast on my phone!",
        "no way lol, im still not a bot! let's talk about something fun :)",
        "haha no im real, not a bot! pinky promise :)",
        "still not a bot lol :) what r u up to rn?",
    ),
    '["haha ok i believe u","my bad lol","wyd?"]',
)


class Step14Engine:
    __slots__ = (
        "enabled", "s14_emb_f32", "s14_w1_f32", "s14_w2_f32",
        "s14_bloom_q", "s14_lut_a", "s14_lut_b",
        "user_flags", "hits_a", "hits_b", "logits",
    )

    def __init__(self, npz_path="model_weights.npz"):
        self.enabled = False
        self.user_flags = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.hits_a = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.hits_b = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.logits = np.zeros((NUM_WORKERS, 1, S14_CLASSES), dtype=np.float32)

        if not os.path.exists(npz_path):
            return
        arc = np.load(npz_path, mmap_mode="r")
        req = ("s14_emb_q", "s14_w1_q", "s14_w2_q", "s14_bloom_q", "s14_lut_a", "s14_lut_b")
        if any(k not in arc.files for k in req):
            arc.close()
            print("[STEP 14 INFO] Step 14 weights not in vault yet. Run: python step14_bot_check.py", flush=True)
            return

        def lock_f32(prefix, zero_row0=False):
            q = arc[f"{prefix}_q"].astype(np.int8, copy=False)
            s = arc[f"{prefix}_s"].astype(np.float32, copy=False)
            f32 = np.multiply(q, s, dtype=np.float32)
            if zero_row0:
                f32[0].fill(0.0)
            f32.flags.writeable = False
            return f32

        self.s14_emb_f32 = lock_f32("s14_emb", True)
        self.s14_w1_f32 = lock_f32("s14_w1")
        self.s14_w2_f32 = lock_f32("s14_w2")
        for attr, dt in (("s14_bloom_q", np.int8), ("s14_lut_a", np.uint8), ("s14_lut_b", np.uint8)):
            arr = arc[attr].astype(dt, copy=False)
            arr.flags.writeable = False
            setattr(self, attr, arr)
        arc.close()
        self.enabled = True


S14_ENGINE = None


def get_step14_engine(npz_path="model_weights.npz"):
    global S14_ENGINE
    if S14_ENGINE is None:
        S14_ENGINE = Step14Engine(npz_path)
    return S14_ENGINE


def forward_step14(model, slot, n_words, has_q, age_val, country_id):
    eng = S14_ENGINE
    if eng is None or not eng.enabled or age_val != 0 or country_id != 0:
        return 0, 0, 0

    sid = slot.slot_id
    ha = eng.hits_a[sid]
    hb = eng.hits_b[sid]
    np.take(eng.s14_lut_a, slot.w_b_idx[:, 0], out=ha)
    np.take(eng.s14_lut_b, slot.w_b_idx[:, 1], out=hb)
    m = (ha[:n_words] > 0) & (ha[:n_words] == hb[:n_words])
    codes = ha[:n_words] * m

    if not bool(np.all(codes > 0)):
        return 0, 0, 0

    has_bot = bool(np.any(codes == 1))
    has_real = bool(np.any(codes == 2))
    has_prove = bool(np.any(codes == 3))

    # Protect Step 7 bare "real" (without "?"): require "?" or >= 2 words when only has_real is set
    if has_real and not has_bot and n_words == 1 and not has_q:
        return 0, 0, 0

    # Check "fast" for "u reply so fast" / "u answer too fast"
    has_fast = False
    for w_i in range(n_words):
        if int(slot.words_mat[w_i, 15]) == 4 and int(slot.words_mat[w_i, 0]) == 102 and int(slot.words_mat[w_i, 1]) == 97:
            has_fast = True
            break

    if not (has_bot or has_real or (has_prove and has_fast)):
        return 0, 0, 0

    np.take(eng.s14_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
    if not bool(np.all(slot.w_b_hits[:n_words])):
        return 0, 0, 0

    slot.s4_tokens[:6] = slot.s2_tokens[:6]
    slot.s4_tokens[6] = np.int32(1023) if has_q else model.one_i32
    q_off = np.int32(509) if has_q else np.int32(0)
    slot.s4_tokens[7] = (((slot.s2_tokens[0] * np.int32(37)) + (slot.s2_tokens[n_words - 1] * np.int32(73)) + q_off) & model.s2_v_mask) + model.one_i32

    np.take(eng.s14_emb_f32, slot.s4_tokens, axis=0, out=slot.s2_emb_buf)
    slot.s2_emb_buf[6] *= np.float32(2.5)
    np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
    slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
    np.matmul(slot.s2_pooled, eng.s14_w1_f32, out=slot.s2_mid)
    np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
    out_l = eng.logits[sid]
    np.matmul(slot.s2_mid, eng.s14_w2_f32, out=out_l)

    s0, s_bot, s_real = float(out_l[0, 0]), float(out_l[0, 1]), float(out_l[0, 2])
    if max(s_bot, s_real) <= s0 + 0.1:
        return 0, 0, 0

    # Whenever ANY bot/ai/fake/scam/fast-reply word is present (including "bot or real?"), route to Class 33 (Explicit NO!)
    if has_bot or (has_prove and has_fast):
        return 33, 0, 0

    return 34, 0, 0


def route_step14(class_id, aux_id, users, uid, flags, prev_age, prev_country, bot_g, rot8):
    eng = S14_ENGINE
    if eng is None:
        return None
    if int(users.msg_count[uid]) <= 1 and flags == 0:
        eng.user_flags[uid] = np.uint8(0)

    s14_f = int(eng.user_flags[uid])
    already_checked = (s14_f & FLAG_S14_CHECKED) != 0
    eng.user_flags[uid] = np.uint8(s14_f | FLAG_S14_CHECKED)
    users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)

    if already_checked:
        return CACHED_REPEAT_BOT_CHECK_POOL[rot8]

    is_complete = prev_age >= 18 and prev_country > 0
    if class_id == 33:
        return CACHED_DENY_BOT_COMPLETE_POOL[rot8] if is_complete else CACHED_DENY_BOT_POOL[rot8]

    if class_id == 34:
        return CACHED_CONFIRM_REAL_COMPLETE_POOL[rot8] if is_complete else CACHED_CONFIRM_REAL_POOL[rot8]

    return None


def _snapshot_vault_hashes(vault_dict, exclude_prefix="s14_"):
    snap = {}
    for k, arr in vault_dict.items():
        if not k.startswith(exclude_prefix):
            snap[k] = (
                hashlib.sha256(arr.tobytes()).hexdigest(),
                arr.shape,
                arr.dtype.str,
            )
    return snap


def train_step14_into_vault(npz_path="model_weights.npz"):
    import torch
    import offline_train as ot

    if not os.path.exists("bot_check_is_bot.txt") or not os.path.exists("bot_check_is_real.txt"):
        build_bot_check_files()

    # Rule 12 Cryptographic .NPZ Vault Lock: Snapshot SHA-256 of all existing Steps 1-13 keys
    vault = {}
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            for k in data.files:
                vault[k] = data[k].copy()
    pre_hashes = _snapshot_vault_hashes(vault, exclude_prefix="s14_")

    ram_arena = ot.verify_hardware_and_lock_ram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wp_mat = ot._get_wp_mat()

    s14_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    s14_lut_a = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    s14_lut_b = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
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
        (5, S14_FRAME_WORDS),
        (4, S14_QUESTION_WORDS),
        (3, S14_PROVE_WORDS),
        (2, S14_REAL_HUMAN_WORDS),
        (1, S14_BOT_FAKE_WORDS),
    ):
        for w in word_group:
            h4 = hash_word(ot._collapse_word(w))
            s14_lut_a[h4[0]] = np.uint8(code_val)
            s14_lut_b[h4[1]] = np.uint8(code_val)
            s14_bloom[h4[:3]] = np.int8(1)

    sample_map = {}
    for gf in (
        "greetings.txt", "ages.txt", "countries.txt", "hru.txt", "wyd.txt",
        "gender_ask_choice.txt", "status_positive.txt", "status_negative.txt",
        "compliments_flirt.txt", "reactions_ack.txt",
        "looking_for_ask.txt", "looking_for_state.txt",
        "name_ask.txt", "hobbies_ask.txt", "hobbies_tell.txt",
        "rel_status_ask.txt", "rel_status_tell.txt",
        "private_social_ask.txt", "yes_no_maybe.txt",
        "pic_photo_ask.txt", "pic_photo_offer.txt",
    ):
        if os.path.exists(gf):
            with open(gf, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and len(sample_map) < 14000:
                        toks = [w.strip("'!?.,/") for w in s.split()]
                        if not any(w in ALL_S14_ANCHOR_SET for w in toks):
                            sample_map[s] = (0, np.float32(2.0))

    s14_cnt = 0
    for fn, cls_id, builtins in (
        ("bot_check_is_bot.txt", 1, ASK_IS_BOT_BASES),
        ("bot_check_is_real.txt", 2, ASK_IS_REAL_BASES),
    ):
        for b_line in builtins:
            s = " ".join(b_line.strip().lower().split())
            if s:
                sample_map[s] = (cls_id, np.float32(36.0))
                s14_cnt += 1
        if os.path.exists(fn):
            with open(fn, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and ot._is_whitelist_clean(s, STRICT_ALLOWED_S14_VOCAB):
                        sample_map[s] = (cls_id, np.float32(24.0))
                        s14_cnt += 1

    # Strict Anti-Collision Negative Traps (Protects Step 7 "for real"/"real", Step 4 "are you a girl?", Class 0 "who are you")
    for trap in (
        "what", "what?", "what is this", "what do you want", "what do you do",
        "why", "why?", "why not", "who", "who?", "who are you", "who r u",
        "real", "for real", "fr", "true", "same", "sam", "haha", "lol",
        "are you a girl", "are u a girl", "are you a boy", "are u a boy", "m or f?",
        "whats ur real name", "whats your real name", "im good", "im bored",
        "are u single", "can we talk private", "send pic", "u look good",
    ):
        sample_map[trap] = (0, np.float32(65.0))

    q_emb, q_w1, q_w2, n = ot._train_word_mlp_head(sample_map, S14_CLASSES, 28, "STEP 14", device, wp_mat, use_qmark=True)

    vault.update(
        s14_emb_q=q_emb[0], s14_emb_s=q_emb[1],
        s14_w1_q=q_w1[0], s14_w1_s=q_w1[1],
        s14_w2_q=q_w2[0], s14_w2_s=q_w2[1],
        s14_bloom_q=s14_bloom, s14_lut_a=s14_lut_a, s14_lut_b=s14_lut_b,
    )
    np.savez(npz_path, **vault)
    del ram_arena

    # Rule 12 Post-Save Cryptographic SHA-256 Verification
    with np.load(npz_path, allow_pickle=False) as saved_arc:
        post_vault = {k: saved_arc[k] for k in saved_arc.files}
    post_hashes = _snapshot_vault_hashes(post_vault, exclude_prefix="s14_")
    if pre_hashes != post_hashes:
        raise RuntimeError("[RULE 12 VAULT BREACH] Existing Step 1-13 weights were altered!")

    size_kb = os.path.getsize(npz_path) / 1024.0
    print(f"[RULE 12 VAULT SHIELD] Verified SHA-256 integrity of all {len(pre_hashes)} locked Step 1-13 arrays!", flush=True)
    print(f"[STEP 14 DONE] Locked Step 14 Bot-Check head & LUT into '{npz_path}' ({s14_cnt} lines, {n} total, {size_kb:.2f} KB).", flush=True)


if __name__ == "__main__":
    train_step14_into_vault("model_weights.npz")