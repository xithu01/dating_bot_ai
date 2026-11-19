#!/usr/bin/env python3
# FILE: step13_pic_photo.py
# PURPOSE: Standalone Step 13 Plugin (Classes 31-32: Pic / Photo / Selfie Requests & Appearance)
# RULES: Enforces Rules 1-13:
#        - Rule 12: Cryptographic SHA-256 Vault Lock on model_weights.npz (Steps 1-12 verified unchanged)
#        - Handles "send pic", "send selfie", "pic for pic", "can i see u?", "what do u look like?", "how tall r u?"
#        - Handles "wanna see me?", "can i send a pic?", "ill send first", "u first", "i dont send pics", "no pics"
#        - Persona-aware appearance replies (Bot F vs Bot M) + safe chat-first boundaries on pic requests

import hashlib
import os
import sys
import numpy as np

from build_pic_photo_txt import (
    ASK_PIC_BASES,
    OFFER_REFUSE_PIC_BASES,
    S13_APPEARANCE_WORDS,
    S13_FRAME_WORDS,
    S13_PIC_NOUNS,
    S13_PIC_VERBS,
    S13_QUESTION_WORDS,
    STRICT_ALLOWED_S13_VOCAB,
    build_pic_photo_files,
)
from production_server_part2 import (
    FLAG_GREETED,
    _make_pool,
)

BLOOM_SIZE = 65536
S2_VOCAB = 1024
S2_DIM = 32
S13_CLASSES = 3  # 0=Reject, 1=AskPicPhoto(31), 2=OfferRefusePic(32)
MAX_USERS = 50000
NUM_WORKERS = 8

FLAG_S13_ASKED_PIC = 1

# 7-Code O(1) LUT Groups for Step 13:
# 1 = Pic / Photo / Selfie / Face Nouns ("pic", "pics", "photo", "selfie", "face", "urself")
# 2 = Appearance Words ("look", "describe", "tall", "height", "hair", "eyes", "color", "build")
# 3 = Pic Action Verbs ("send", "sending", "show", "see", "trade", "swap", "exchange", "share")
# 4 = Offer / First Words ("first", "ill", "i'll", "wanna", "want", "mine")
# 5 = Negation / Refuse Words ("dont", "don't", "no", "not", "never")
# 6 = Question Words ("can", "could", "may", "will", "would", "what", "whats", "how", "do", "are", "r")
# 7 = Shared Framing ("u", "you", "ur", "your", "i", "im", "me", "my", "we", "a", "of", "for", "to", "like", "let", "lemme")
S13_OFFER_FIRST_WORDS = ("first", "ill", "i'll", "wanna", "want", "mine")
S13_REFUSE_NEG_WORDS = ("dont", "don't", "no", "not", "never")

ALL_S13_ANCHOR_SET = frozenset(S13_PIC_NOUNS + S13_APPEARANCE_WORDS + ("trade", "swap", "exchange", "first"))

# --- STEP 13 PERSONA-AWARE & BOUNDARY RESPONSE POOLS (8 items each) ---

# Class 31 (aux_id == 1): Direct Pic / Selfie / Trade Request (Age & Country NOT known yet)
CACHED_ASK_PIC_POOL = _make_pool(
    31, "false", "ask_pic_photo",
    (
        "haha slow down! i don't send pics right away, i don't even know how old u are or where u from :)",
        "maybe after we chat and vibe a bit first! how old r u and where u from?",
        "i don't send selfies right away haha :) tell me ur age and country first!",
        "haha not so fast! let's get to know each other first :) how old r u?",
        "i like to talk and vibe first before sharing any pics :) where r u from?",
        "slow down lol :) let's chat a bit first! whats ur age and country?",
        "haha maybe later if we click! how old are u and where u live?",
        "i only share pics after we chat for a bit :) how old r u?",
    ),
    '["im 22","from usa","what do u look like?"]',
)

# Class 31 (aux_id == 1): Direct Pic / Selfie Request (Age & Country ALREADY known)
CACHED_ASK_PIC_COMPLETE_POOL = _make_pool(
    31, "false", "ask_pic_photo_complete",
    (
        "haha maybe after we get to know each other a bit more :) i don't send selfies right away!",
        "i like to vibe and talk first before sharing pics :) what r ur hobbies?",
        "slow down haha :) let's keep chatting a little longer first! what music do u like?",
        "maybe in a bit if we keep clicking ;) tell me more about urself first!",
        "haha i'm shy about sending pics right away! let's talk more first :)",
        "not right away lol :) i prefer vibing here first! what do u do for fun?",
        "haha patience! let's chat a bit more before trading pics :) wyd rn?",
        "i keep my pics private until we vibe more :) u can ask what i look like tho!",
    ),
    '["what do u look like?","haha fair","wyd?"]',
)

CACHED_REPEAT_ASK_PIC_POOL = _make_pool(
    31, "false", "ask_pic_photo_repeat",
    (
        "haha patience lol! i don't send pics right away, let's chat a bit more first :)",
        "still too soon for pics haha :) let's just keep vibing here!",
        "haha u really want a selfie huh? let's talk a little longer first :)",
        "slow down lol :) i told u i don't share pics right away!",
        "maybe later haha :) let's get to know each other better first!",
        "no pics yet haha :) tell me what u like doing for fun!",
        "haha i'm still shy about sending pics so fast! let's keep talking :)",
        "patience is key lol :) let's vibe here a bit more first!",
    ),
    '["haha ok","what do u look like?","wyd?"]',
)

# Class 31 (aux_id == 2): Appearance Question ("what do u look like?", "how tall r u?") -> Bot F (bot_g == 2)
CACHED_APPEARANCE_F_POOL = _make_pool(
    31, "false", "ask_appearance_f",
    (
        "im 5'5, brunette hair, brown eyes, and a cute smile :) what do u look like?",
        "im around 5'5 with long dark hair and brown eyes :) wbu, describe urself!",
        "5'5, dark brown hair, hazel eyes, and slim :) what do u look like?",
        "im about 5'4 with wavy brown hair and a big smile :) how about u?",
        "brunette hair, brown eyes, around 5'5 :) wbu, how tall r u?",
        "im 5'5, long dark hair, brown eyes, pretty chill style :) describe urself!",
        "around 5'5 with soft brown hair and dark eyes :) what do u look like?",
        "im 5'5, brunette, brown eyes, always smiling :) wbu?",
    ),
    '["u sound cute!","im 5ft 10","wanna see me?"]',
)

# Class 31 (aux_id == 2): Appearance Question -> Bot M (bot_g == 1)
CACHED_APPEARANCE_M_POOL = _make_pool(
    31, "false", "ask_appearance_m",
    (
        "im 5'11, dark hair, brown eyes, and athletic build :) what do u look like?",
        "im around 5'11 with short dark hair and go to the gym :) wbu, describe urself!",
        "5'11, black hair, brown eyes, pretty fit :) what do u look like?",
        "im about 6'0 with dark hair and a nice smile :) how about u?",
        "dark hair, brown eyes, around 5'11 :) wbu, how tall r u?",
        "im 5'11, short dark hair, athletic build, chill style :) describe urself!",
        "around 5'11 with dark wavy hair and brown eyes :) what do u look like?",
        "im 6'0, dark hair, gym build, always smiling :) wbu?",
    ),
    '["u sound cute!","im 5ft 5","wanna see me?"]',
)

# Class 32 (aux_id == 1): User Offering to Send Their Pic ("wanna see me?", "can i send a pic?", "ill send first")
CACHED_OFFER_PIC_POOL = _make_pool(
    32, "false", "offer_pic",
    (
        "aw sure if u want to! :) i'm shy about sending mine right away tho haha",
        "haha up to u! i like getting to know someone first too :)",
        "sure if u feel comfortable! :) i usually wait a bit before sending mine",
        "aw that's sweet, only if u want to :) no pressure at all!",
        "haha u can if u want! i'm just a bit shy about sending selfies so fast :)",
        "totally up to u :) i enjoy just chatting and vibing first too!",
        "sure if u want :) no pressure tho, we can just keep talking too!",
        "aw nice, only if u feel comfortable! :) i like getting to know u first",
    ),
    '["no worries","what do u look like?","wyd?"]',
)

# Class 32 (aux_id == 2): User Saying "u first" or Refusing Pics ("i dont send pics", "no pics")
CACHED_REFUSE_OR_FIRST_PIC_POOL = _make_pool(
    32, "false", "refuse_or_first_pic",
    (
        "haha totally fair! no pressure on pics at all, let's just chill and talk :)",
        "no worries at all :) we don't need to send pics, let's just vibe!",
        "haha i'm shy too, so let's just keep chatting without pics for now :)",
        "totally respect that! :) chatting is more fun anyway, what r ur hobbies?",
        "no problem at all :) let's just talk and get to know each other!",
        "haha fair enough! we can just keep it chill and chat :) wyd rn?",
        "all good :) no pics needed, tell me what u like doing for fun!",
        "totally fine! :) let's just keep vibing here, hows ur day going?",
    ),
    '["thanks!","what r ur hobbies?","wyd?"]',
)


class Step13Engine:
    __slots__ = (
        "enabled", "s13_emb_f32", "s13_w1_f32", "s13_w2_f32",
        "s13_bloom_q", "s13_lut_a", "s13_lut_b",
        "user_flags", "hits_a", "hits_b", "logits",
    )

    def __init__(self, npz_path="model_weights.npz"):
        self.enabled = False
        self.user_flags = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.hits_a = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.hits_b = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.logits = np.zeros((NUM_WORKERS, 1, S13_CLASSES), dtype=np.float32)

        if not os.path.exists(npz_path):
            return
        arc = np.load(npz_path, mmap_mode="r")
        req = ("s13_emb_q", "s13_w1_q", "s13_w2_q", "s13_bloom_q", "s13_lut_a", "s13_lut_b")
        if any(k not in arc.files for k in req):
            arc.close()
            print("[STEP 13 INFO] Step 13 weights not in vault yet. Run: python step13_pic_photo.py", flush=True)
            return

        def lock_f32(prefix, zero_row0=False):
            q = arc[f"{prefix}_q"].astype(np.int8, copy=False)
            s = arc[f"{prefix}_s"].astype(np.float32, copy=False)
            f32 = np.multiply(q, s, dtype=np.float32)
            if zero_row0:
                f32[0].fill(0.0)
            f32.flags.writeable = False
            return f32

        self.s13_emb_f32 = lock_f32("s13_emb", True)
        self.s13_w1_f32 = lock_f32("s13_w1")
        self.s13_w2_f32 = lock_f32("s13_w2")
        for attr, dt in (("s13_bloom_q", np.int8), ("s13_lut_a", np.uint8), ("s13_lut_b", np.uint8)):
            arr = arc[attr].astype(dt, copy=False)
            arr.flags.writeable = False
            setattr(self, attr, arr)
        arc.close()
        self.enabled = True


S13_ENGINE = None


def get_step13_engine(npz_path="model_weights.npz"):
    global S13_ENGINE
    if S13_ENGINE is None:
        S13_ENGINE = Step13Engine(npz_path)
    return S13_ENGINE


def forward_step13(model, slot, n_words, has_q, age_val, country_id):
    eng = S13_ENGINE
    if eng is None or not eng.enabled or age_val != 0 or country_id != 0:
        return 0, 0, 0

    sid = slot.slot_id
    ha = eng.hits_a[sid]
    hb = eng.hits_b[sid]
    np.take(eng.s13_lut_a, slot.w_b_idx[:, 0], out=ha)
    np.take(eng.s13_lut_b, slot.w_b_idx[:, 1], out=hb)
    m = (ha[:n_words] > 0) & (ha[:n_words] == hb[:n_words])
    codes = ha[:n_words] * m

    if not bool(np.all(codes > 0)):
        return 0, 0, 0

    has_pic = bool(np.any(codes == 1))
    has_appear = bool(np.any(codes == 2))
    has_verb = bool(np.any(codes == 3))
    has_offer_first = bool(np.any(codes == 4))
    has_neg = bool(np.any(codes == 5))

    # Must contain at least one Pic/Appearance/PicVerb/First signal (blocks "what do you want", "no", "how are you")
    if not (has_pic or has_appear or has_verb or (has_offer_first and n_words == 2)):
        return 0, 0, 0

    np.take(eng.s13_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
    if not bool(np.all(slot.w_b_hits[:n_words])):
        return 0, 0, 0

    slot.s4_tokens[:6] = slot.s2_tokens[:6]
    slot.s4_tokens[6] = np.int32(1023) if has_q else model.one_i32
    q_off = np.int32(509) if has_q else np.int32(0)
    slot.s4_tokens[7] = (((slot.s2_tokens[0] * np.int32(37)) + (slot.s2_tokens[n_words - 1] * np.int32(73)) + q_off) & model.s2_v_mask) + model.one_i32

    np.take(eng.s13_emb_f32, slot.s4_tokens, axis=0, out=slot.s2_emb_buf)
    slot.s2_emb_buf[6] *= np.float32(2.5)
    np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
    slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
    np.matmul(slot.s2_pooled, eng.s13_w1_f32, out=slot.s2_mid)
    np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
    out_l = eng.logits[sid]
    np.matmul(slot.s2_mid, eng.s13_w2_f32, out=out_l)

    s0, s_ask, s_offer = float(out_l[0, 0]), float(out_l[0, 1]), float(out_l[0, 2])
    if max(s_ask, s_offer) <= s0 + 0.1:
        return 0, 0, 0

    # Check if refusing pics ("i dont send pics", "no pics", "not sending pics")
    if has_neg and (has_pic or has_verb):
        return 32, 0, 2

    # Check for "first" ("u first" / "send first" vs "ill send first")
    has_word_first = False
    has_i_before_send = False
    for w_i in range(n_words):
        c0 = int(slot.words_mat[w_i, 0])
        Lw = int(slot.words_mat[w_i, 15])
        if Lw == 5 and c0 == 102 and int(slot.words_mat[w_i, 1]) == 105:  # "first"
            has_word_first = True
        if int(codes[w_i]) == 3 and w_i > 0:
            prev_c0 = int(slot.words_mat[w_i - 1, 0])
            # "i send", "ill send", "i'll send", "i can send"
            if prev_c0 == 105 or (w_i >= 2 and int(slot.words_mat[w_i - 2, 0]) == 105):
                # Check it's "send" (115, 101, 110) not "see" (115, 101)
                if int(slot.words_mat[w_i, 0]) == 115 and int(slot.words_mat[w_i, 1]) == 101 and int(slot.words_mat[w_i, 2]) == 110:
                    has_i_before_send = True

    if has_word_first:
        return (32, 0, 1) if has_i_before_send else (32, 0, 2)

    if has_i_before_send or s_offer > s_ask + 0.35:
        return 32, 0, 1

    # Class 31: Appearance ("what do u look like?", "how tall r u?") vs Direct Pic/Selfie Request
    if has_appear and not has_pic:
        return 31, 0, 2
    return 31, 0, 1


def route_step13(class_id, aux_id, users, uid, flags, prev_age, prev_country, bot_g, rot8):
    eng = S13_ENGINE
    if eng is None:
        return None
    if int(users.msg_count[uid]) <= 1 and flags == 0:
        eng.user_flags[uid] = np.uint8(0)

    s13_f = int(eng.user_flags[uid])
    is_complete = prev_age >= 18 and prev_country > 0
    users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)

    if class_id == 31:
        if aux_id == 2:
            return CACHED_APPEARANCE_F_POOL[rot8] if bot_g == 2 else CACHED_APPEARANCE_M_POOL[rot8]
        already_asked = (s13_f & FLAG_S13_ASKED_PIC) != 0
        eng.user_flags[uid] = np.uint8(s13_f | FLAG_S13_ASKED_PIC)
        if already_asked:
            return CACHED_REPEAT_ASK_PIC_POOL[rot8]
        return CACHED_ASK_PIC_COMPLETE_POOL[rot8] if is_complete else CACHED_ASK_PIC_POOL[rot8]

    if class_id == 32:
        if aux_id == 2:
            return CACHED_REFUSE_OR_FIRST_PIC_POOL[rot8]
        return CACHED_OFFER_PIC_POOL[rot8]

    return None


def _snapshot_vault_hashes(vault_dict, exclude_prefix="s13_"):
    snap = {}
    for k, arr in vault_dict.items():
        if not k.startswith(exclude_prefix):
            snap[k] = (
                hashlib.sha256(arr.tobytes()).hexdigest(),
                arr.shape,
                arr.dtype.str,
            )
    return snap


def train_step13_into_vault(npz_path="model_weights.npz"):
    import torch
    import offline_train as ot

    if not os.path.exists("pic_photo_ask.txt") or not os.path.exists("pic_photo_offer.txt"):
        build_pic_photo_files()

    # Rule 12 Cryptographic .NPZ Vault Lock: Snapshot SHA-256 of all existing Steps 1-12 keys
    vault = {}
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            for k in data.files:
                vault[k] = data[k].copy()
    pre_hashes = _snapshot_vault_hashes(vault, exclude_prefix="s13_")

    ram_arena = ot.verify_hardware_and_lock_ram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wp_mat = ot._get_wp_mat()

    s13_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    s13_lut_a = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    s13_lut_b = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
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
        (7, S13_FRAME_WORDS),
        (6, S13_QUESTION_WORDS),
        (5, S13_REFUSE_NEG_WORDS),
        (4, S13_OFFER_FIRST_WORDS),
        (3, S13_PIC_VERBS),
        (2, S13_APPEARANCE_WORDS),
        (1, S13_PIC_NOUNS),
    ):
        for w in word_group:
            h4 = hash_word(ot._collapse_word(w))
            s13_lut_a[h4[0]] = np.uint8(code_val)
            s13_lut_b[h4[1]] = np.uint8(code_val)
            s13_bloom[h4[:3]] = np.int8(1)

    sample_map = {}
    for gf in (
        "greetings.txt", "ages.txt", "countries.txt", "hru.txt", "wyd.txt",
        "gender_ask_choice.txt", "status_positive.txt", "status_negative.txt",
        "compliments_flirt.txt", "reactions_ack.txt",
        "looking_for_ask.txt", "looking_for_state.txt",
        "name_ask.txt", "hobbies_ask.txt", "hobbies_tell.txt",
        "rel_status_ask.txt", "rel_status_tell.txt",
        "private_social_ask.txt", "yes_no_maybe.txt",
    ):
        if os.path.exists(gf):
            with open(gf, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and len(sample_map) < 14000:
                        toks = [w.strip("'!?.,/") for w in s.split()]
                        if not any(w in ALL_S13_ANCHOR_SET for w in toks):
                            sample_map[s] = (0, np.float32(2.0))

    s13_cnt = 0
    for fn, cls_id, builtins in (
        ("pic_photo_ask.txt", 1, ASK_PIC_BASES),
        ("pic_photo_offer.txt", 2, OFFER_REFUSE_PIC_BASES),
    ):
        for b_line in builtins:
            s = " ".join(b_line.strip().lower().split())
            if s:
                sample_map[s] = (cls_id, np.float32(36.0))
                s13_cnt += 1
        if os.path.exists(fn):
            with open(fn, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and ot._is_whitelist_clean(s, STRICT_ALLOWED_S13_VOCAB):
                        sample_map[s] = (cls_id, np.float32(24.0))
                        s13_cnt += 1

    # Strict Anti-Collision Negative Traps
    for trap in (
        "what", "what?", "what is this", "what do you want", "what do you do",
        "why", "why?", "why not", "who", "who?", "who are you", "who r u",
        "good to see you", "nice to see you", "long time no see",
        "u look good", "you look good", "u look cute", "u look nice", "u look hot",
        "i like photography", "photography", "send ur username", "can we talk private",
        "no", "nope", "nah", "yes", "maybe", "idk", "im good", "im bored",
        "how are you", "how old are you", "are u single", "whats ur name",
    ):
        sample_map[trap] = (0, np.float32(65.0))

    q_emb, q_w1, q_w2, n = ot._train_word_mlp_head(sample_map, S13_CLASSES, 28, "STEP 13", device, wp_mat, use_qmark=True)

    vault.update(
        s13_emb_q=q_emb[0], s13_emb_s=q_emb[1],
        s13_w1_q=q_w1[0], s13_w1_s=q_w1[1],
        s13_w2_q=q_w2[0], s13_w2_s=q_w2[1],
        s13_bloom_q=s13_bloom, s13_lut_a=s13_lut_a, s13_lut_b=s13_lut_b,
    )
    np.savez(npz_path, **vault)
    del ram_arena

    # Rule 12 Post-Save Cryptographic SHA-256 Verification
    with np.load(npz_path, allow_pickle=False) as saved_arc:
        post_vault = {k: saved_arc[k] for k in saved_arc.files}
    post_hashes = _snapshot_vault_hashes(post_vault, exclude_prefix="s13_")
    if pre_hashes != post_hashes:
        raise RuntimeError("[RULE 12 VAULT BREACH] Existing Step 1-12 weights were altered!")

    size_kb = os.path.getsize(npz_path) / 1024.0
    print(f"[RULE 12 VAULT SHIELD] Verified SHA-256 integrity of all {len(pre_hashes)} locked Step 1-12 arrays!", flush=True)
    print(f"[STEP 13 DONE] Locked Step 13 Pic/Photo head & LUT into '{npz_path}' ({s13_cnt} lines, {n} total, {size_kb:.2f} KB).", flush=True)


if __name__ == "__main__":
    train_step13_into_vault("model_weights.npz")