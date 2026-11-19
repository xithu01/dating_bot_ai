#!/usr/bin/env python3
# FILE: step12_private_yesno.py
# PURPOSE: Standalone Step 12 Plugin (Classes 29-30: Telegram Private/Social Requests & Yes/No/Maybe Replies)
# RULES: Enforces Rules 1-13:
#        - Rule 12: Cryptographic SHA-256 Vault Lock on model_weights.npz (Steps 1-11 verified unchanged)
#        - Handles "can we talk privty", "can we talk private?", "telgrm?", "ur tg?", "dm me", "snap?", "insta?"
#        - Handles "yes", "no", "nope", "nah", "maybe", "idk", "not sure", "i dont have snap", "lets stay here"
#        - Zero collision with Step 7 ("yeah", "ok", "sure"), Step 11 ("no gf", "no bf"), or Class 0 ("why not")

import hashlib
import os
import sys
import numpy as np

from build_private_yesno_txt import (
    ASK_PRIVATE_BASES,
    S12_ACTION_VERBS,
    S12_FRAME_WORDS,
    S12_MAYBE_WORDS,
    S12_NEG_WORDS,
    S12_NO_WORDS,
    S12_PRIVATE_TG_WORDS,
    S12_SOCIAL_WORDS,
    S12_YES_WORDS,
    STRICT_ALLOWED_S12_VOCAB,
    YES_NO_MAYBE_BASES,
    build_private_yesno_files,
)
from production_server_part2 import (
    FLAG_GREETED,
    _make_pool,
)

BLOOM_SIZE = 65536
S2_VOCAB = 1024
S2_DIM = 32
S12_CLASSES = 3  # 0=Reject, 1=AskPrivateSocial(29), 2=ReplyYesNoMaybe(30)
MAX_USERS = 50000
NUM_WORKERS = 8

FLAG_S12_ASKED_PRIV = 1

# 8-Code O(1) LUT Groups for Step 12:
# 1 = Private / DM / Telegram / Username ("private", "privty", "pv", "dm", "telegram", "telgrm", "tg", "username")
# 2 = External Socials / Number / Call ("snap", "insta", "ig", "whatsapp", "discord", "number", "phone", "video", "voice")
# 3 = Yes ("yes", "yess", "ys")
# 4 = No ("no", "noo", "nop", "nope", "nah", "na")
# 5 = Maybe / Unsure / IDK ("maybe", "mayb", "mby", "idk", "dunno", "unsure", "depends", "possibly", "probably")
# 6 = Negation ("dont", "don't", "not", "never" — turns "i dont have snap" into Class 30 No!)
# 7 = Action Verbs ("talk", "chat", "text", "msg", "add", "send", "give", "move", "go", "come", "call", "have", "got", "stay")
# 8 = Shared Framing ("can", "we", "i", "u", "ur", "lets", "in", "on", "to", "here", "what", "sure", "plz", "thanks")
ALL_S12_ANCHOR_SET = frozenset(
    S12_PRIVATE_TG_WORDS + S12_SOCIAL_WORDS + S12_YES_WORDS + S12_NO_WORDS + S12_MAYBE_WORDS
)

# --- STEP 12 CONTEXT-AWARE RESPONSE POOLS (8 items each) ---

# Class 29 (aux_id == 1): Private Chat / PV / DM / Telegram / Username
CACHED_ASK_PRIVATE_POOL = _make_pool(
    29, "false", "ask_private_chat",
    (
        "haha let's chat here a bit first! i don't even know how old u are or where u from :)",
        "maybe later after we vibe a bit! how old r u and where u from?",
        "slow down haha :) let's talk here first! whats ur age and country?",
        "i only share my telegram after we get to know each other a bit :) how old r u?",
        "let's vibe here first before going private :) where r u from?",
        "haha maybe in a bit if we click! tell me ur age and country first :)",
        "i like chatting here first before giving my tg :) how old r u?",
        "not right away haha :) let's talk here a little first! where u from?",
    ),
    '["im 22","from usa","ok lets chat here"]',
)

CACHED_ASK_PRIVATE_COMPLETE_POOL = _make_pool(
    29, "false", "ask_private_chat_complete",
    (
        "haha maybe after we chat a little more here :) tell me about ur hobbies first!",
        "i like to vibe here a bit before moving to private :) what do u like doing for fun?",
        "maybe later if we keep clicking ;) what music or games r u into?",
        "let's stay here for a bit first! i don't give my telegram right away haha :)",
        "haha patience! let's get to know each other a little more first :) wyd rn?",
        "maybe in a little bit :) i like seeing if we vibe here first! whats ur favorite movie?",
        "slow down lol :) let's talk here a bit longer! are u single?",
        "i usually keep it here until we vibe more :) tell me something fun about u!",
    ),
    '["ok lets stay here","sure","what r ur hobbies?"]',
)

# Class 29 (aux_id == 2): External Socials (Snap / Insta / WhatsApp / Number / Video Call)
CACHED_ASK_SOCIAL_POOL = _make_pool(
    29, "false", "ask_social_media",
    (
        "i don't give out my socials right away haha :) let's chat here first! how old r u?",
        "maybe after we get to know each other a bit better :) where r u from?",
        "haha let's talk on here first before sharing snap or insta :) whats ur age?",
        "i keep my socials private until we vibe a bit :) how old r u and where u from?",
        "slow down lol :) let's chat here first! what country u in?",
        "maybe later if we click :) i don't share my number or socials right away! ur age?",
        "let's get to know each other here first :) how old r u?",
        "haha not so fast! let's vibe here first before sharing socials :) where u from?",
    ),
    '["im 22","from usa","no worries"]',
)

CACHED_ASK_SOCIAL_COMPLETE_POOL = _make_pool(
    29, "false", "ask_social_media_complete",
    (
        "i don't share my socials right away haha :) let's keep chatting here! what r ur hobbies?",
        "maybe later if we keep vibing :) let's talk here a bit more first! wyd rn?",
        "haha patience! i like getting to know someone here before adding them :) what music u like?",
        "let's just chill and talk here for now :) tell me what u like doing for fun!",
        "maybe in a bit :) i keep my snap and insta private at first! hows ur day going?",
        "haha slow down lol :) let's keep talking here first! do u watch anime or play games?",
        "not right away :) i prefer chatting here until we click more! wyd?",
        "let's stay here a little longer first :) u seem cool tho!",
    ),
    '["ok lets stay here","what music u like?","wyd?"]',
)

CACHED_REPEAT_ASK_PRIVATE_POOL = _make_pool(
    29, "false", "ask_private_repeat",
    (
        "haha patience lol! let's vibe here a little longer first :)",
        "still too soon haha :) let's just keep chatting here!",
        "haha u really want to go private huh? let's talk here a bit more first :)",
        "slow down lol :) we are having a good chat right here!",
        "maybe later haha :) let's get to know each other better first!",
        "patience is key lol :) tell me more about urself first!",
        "haha i told u, let's chat here a bit longer before sharing tg or socials :)",
        "not yet haha :) let's keep vibing here for now!",
    ),
    '["haha ok","fair enough","wyd?"]',
)

# Class 30 (aux_id == 1): Yes / Affirmative
CACHED_REPLY_YES_POOL = _make_pool(
    30, "false", "reply_yes",
    (
        "nice! :) btw how old r u and where u from?",
        "awesome :) how old r u?",
        "yay :) where r u from btw?",
        "cool! :) what's ur age and country?",
        "sweet :) how old are u and where r u from?",
        "love that :) where u from?",
        "great! :) btw what's ur age?",
        "nice :) tell me ur age and where u live!",
    ),
    '["im 22","from usa","24 m"]',
)

CACHED_REPLY_YES_COMPLETE_POOL = _make_pool(
    30, "false", "reply_yes_complete",
    (
        "awesome! :) so what u up to rn?",
        "nice! :) hows ur day going so far?",
        "sweet :) what do u like doing for fun?",
        "love that :) what music or movies r u into?",
        "cool! :) u seem really fun to talk to!",
        "yay :) just chilling here right now, wbu?",
        "great :) what r u looking for on here?",
        "nice! :) tell me something cool about u!",
    ),
    '["just chilling","wyd?","what music u like?"]',
)

# Class 30 (aux_id == 2): No / Negative / Don't have social
CACHED_REPLY_NO_POOL = _make_pool(
    30, "false", "reply_no",
    (
        "no worries at all :) we can just chat here! how old r u?",
        "that's totally fine :) where r u from btw?",
        "all good! :) let's just chill and talk here, whats ur age and country?",
        "no problem at all :) how old are u?",
        "haha that's fair :) where u from?",
        "totally fine! :) what's ur age and where u from?",
        "no worries :) we can just vibe here! how old r u?",
        "all good :) tell me where r u from!",
    ),
    '["im 22","from usa","ok cool"]',
)

CACHED_REPLY_NO_COMPLETE_POOL = _make_pool(
    30, "false", "reply_no_complete",
    (
        "no worries at all! :) we can just chill and talk here, wyd rn?",
        "that's totally fine :) so what do u like doing for fun?",
        "all good! :) hows ur day going so far?",
        "no problem :) what kind of music or games do u like?",
        "haha fair enough :) just chilling here, what u up to?",
        "totally fine :) let's just keep chatting here! what r ur hobbies?",
        "no worries :) u still seem really chill to talk to!",
        "all good :) what r u up to today?",
    ),
    '["just chilling","what r ur hobbies?","wyd?"]',
)

# Class 30 (aux_id == 3): Maybe / IDK / Not sure / Let's stay here
CACHED_REPLY_MAYBE_POOL = _make_pool(
    30, "false", "reply_maybe",
    (
        "haha no rush at all :) let's get to know each other! how old r u?",
        "fair enough :) we can just vibe here! where r u from?",
        "haha that's totally fine :) whats ur age and country?",
        "no pressure at all :) how old are u and where u from?",
        "haha i feel u :) let's just chill and talk! how old r u?",
        "all good, take ur time :) where r u from?",
        "haha fair enough :) btw what's ur age?",
        "totally cool :) let's just chat here! where u from?",
    ),
    '["im 22","from usa","24 m"]',
)

CACHED_REPLY_MAYBE_COMPLETE_POOL = _make_pool(
    30, "false", "reply_maybe_complete",
    (
        "haha fair enough, no rush :) tell me what u like doing for fun!",
        "totally fine :) let's just chill and talk here! wyd rn?",
        "haha no pressure at all :) hows ur day going?",
        "all good :) what kind of music or movies do u like?",
        "fair enough :) just vibing here right now, wbu?",
        "haha take ur time :) what r ur hobbies?",
        "no worries at all :) u seem really chill!",
        "totally cool :) what u up to today?",
    ),
    '["just chilling","wyd?","what music u like?"]',
)


class Step12Engine:
    __slots__ = (
        "enabled", "s12_emb_f32", "s12_w1_f32", "s12_w2_f32",
        "s12_bloom_q", "s12_lut_a", "s12_lut_b",
        "user_flags", "hits_a", "hits_b", "logits",
    )

    def __init__(self, npz_path="model_weights.npz"):
        self.enabled = False
        self.user_flags = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.hits_a = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.hits_b = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.logits = np.zeros((NUM_WORKERS, 1, S12_CLASSES), dtype=np.float32)

        if not os.path.exists(npz_path):
            return
        arc = np.load(npz_path, mmap_mode="r")
        req = ("s12_emb_q", "s12_w1_q", "s12_w2_q", "s12_bloom_q", "s12_lut_a", "s12_lut_b")
        if any(k not in arc.files for k in req):
            arc.close()
            print("[STEP 12 INFO] Step 12 weights not in vault yet. Run: python step12_private_yesno.py", flush=True)
            return

        def lock_f32(prefix, zero_row0=False):
            q = arc[f"{prefix}_q"].astype(np.int8, copy=False)
            s = arc[f"{prefix}_s"].astype(np.float32, copy=False)
            f32 = np.multiply(q, s, dtype=np.float32)
            if zero_row0:
                f32[0].fill(0.0)
            f32.flags.writeable = False
            return f32

        self.s12_emb_f32 = lock_f32("s12_emb", True)
        self.s12_w1_f32 = lock_f32("s12_w1")
        self.s12_w2_f32 = lock_f32("s12_w2")
        for attr, dt in (("s12_bloom_q", np.int8), ("s12_lut_a", np.uint8), ("s12_lut_b", np.uint8)):
            arr = arc[attr].astype(dt, copy=False)
            arr.flags.writeable = False
            setattr(self, attr, arr)
        arc.close()
        self.enabled = True


S12_ENGINE = None


def get_step12_engine(npz_path="model_weights.npz"):
    global S12_ENGINE
    if S12_ENGINE is None:
        S12_ENGINE = Step12Engine(npz_path)
    return S12_ENGINE


def forward_step12(model, slot, n_words, has_q, age_val, country_id):
    eng = S12_ENGINE
    if eng is None or not eng.enabled or age_val != 0 or country_id != 0:
        return 0, 0, 0

    sid = slot.slot_id
    ha = eng.hits_a[sid]
    hb = eng.hits_b[sid]
    np.take(eng.s12_lut_a, slot.w_b_idx[:, 0], out=ha)
    np.take(eng.s12_lut_b, slot.w_b_idx[:, 1], out=hb)
    m = (ha[:n_words] > 0) & (ha[:n_words] == hb[:n_words])
    codes = ha[:n_words] * m

    if not bool(np.all(codes > 0)):
        return 0, 0, 0

    has_priv = bool(np.any(codes == 1))
    has_soc = bool(np.any(codes == 2))
    has_yes = bool(np.any(codes == 3))
    has_no = bool(np.any(codes == 4))
    has_maybe = bool(np.any(codes == 5))
    has_neg = bool(np.any(codes == 6))
    has_act = bool(np.any(codes == 7))

    # Check special multi-word triggers: "not sure", "i dont know", "lets stay/talk here", "add/text me"
    has_sure_or_know = False
    has_lets = False
    has_here = False
    for w_i in range(n_words):
        c0 = int(slot.words_mat[w_i, 0])
        Lw = int(slot.words_mat[w_i, 15])
        if (Lw == 4 and c0 == 115 and int(slot.words_mat[w_i, 1]) == 117) or (Lw == 4 and c0 == 107 and int(slot.words_mat[w_i, 1]) == 110):
            has_sure_or_know = True
        elif Lw >= 4 and c0 == 108 and int(slot.words_mat[w_i, 1]) == 101 and int(slot.words_mat[w_i, 2]) == 116:
            has_lets = True
        elif Lw == 4 and c0 == 104 and int(slot.words_mat[w_i, 1]) == 101 and int(slot.words_mat[w_i, 2]) == 114:
            has_here = True

    has_direct_dm_act = False
    if has_act and n_words >= 2:
        w0_c0 = int(slot.words_mat[0, 0])
        # "add me", "text me", "msg me", "can i text/msg/add u"
        for w_i in range(n_words):
            if int(codes[w_i]) == 7 and int(slot.words_mat[w_i, 0]) in (97, 116, 109):  # add, text, msg/message
                if int(slot.words_mat[w_i, 1]) in (100, 101, 115):  # ad(d), te(xt), ms(g)/me(ssage)
                    has_direct_dm_act = True
                    break

    has_valid_s12 = (
        has_priv or has_soc or has_yes or has_no or has_maybe
        or (has_neg and has_sure_or_know)
        or (has_lets and has_here and has_act)
        or has_direct_dm_act
    )
    if not has_valid_s12:
        return 0, 0, 0

    np.take(eng.s12_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
    if not bool(np.all(slot.w_b_hits[:n_words])):
        return 0, 0, 0

    slot.s4_tokens[:6] = slot.s2_tokens[:6]
    slot.s4_tokens[6] = np.int32(1023) if has_q else model.one_i32
    q_off = np.int32(509) if has_q else np.int32(0)
    slot.s4_tokens[7] = (((slot.s2_tokens[0] * np.int32(37)) + (slot.s2_tokens[n_words - 1] * np.int32(73)) + q_off) & model.s2_v_mask) + model.one_i32

    np.take(eng.s12_emb_f32, slot.s4_tokens, axis=0, out=slot.s2_emb_buf)
    slot.s2_emb_buf[6] *= np.float32(2.5)
    np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
    slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
    np.matmul(slot.s2_pooled, eng.s12_w1_f32, out=slot.s2_mid)
    np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
    out_l = eng.logits[sid]
    np.matmul(slot.s2_mid, eng.s12_w2_f32, out=out_l)

    s0, s_priv, s_yn = float(out_l[0, 0]), float(out_l[0, 1]), float(out_l[0, 2])
    if max(s_priv, s_yn) <= s0 + 0.1:
        return 0, 0, 0

    # Smart Negation: "i dont have snap/insta/tg" -> Class 30 (No, aux_id=2), NOT Class 29!
    if has_neg and (has_priv or has_soc) and not has_q:
        return 30, 0, 2

    if has_priv or has_soc or has_direct_dm_act:
        return 29, 0, (2 if has_soc and not has_priv else 1)

    if has_q:
        return 0, 0, 0

    if has_yes:
        return 30, 0, 1
    if has_no:
        return 30, 0, 2
    return 30, 0, 3


def route_step12(class_id, aux_id, users, uid, flags, prev_age, prev_country, bot_g, rot8):
    eng = S12_ENGINE
    if eng is None:
        return None
    if int(users.msg_count[uid]) <= 1 and flags == 0:
        eng.user_flags[uid] = np.uint8(0)

    s12_f = int(eng.user_flags[uid])
    is_complete = prev_age >= 18 and prev_country > 0
    users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)

    if class_id == 29:
        already_asked = (s12_f & FLAG_S12_ASKED_PRIV) != 0
        eng.user_flags[uid] = np.uint8(s12_f | FLAG_S12_ASKED_PRIV)
        if already_asked:
            return CACHED_REPEAT_ASK_PRIVATE_POOL[rot8]
        if aux_id == 2:
            return CACHED_ASK_SOCIAL_COMPLETE_POOL[rot8] if is_complete else CACHED_ASK_SOCIAL_POOL[rot8]
        return CACHED_ASK_PRIVATE_COMPLETE_POOL[rot8] if is_complete else CACHED_ASK_PRIVATE_POOL[rot8]

    if class_id == 30:
        if aux_id == 1:
            return CACHED_REPLY_YES_COMPLETE_POOL[rot8] if is_complete else CACHED_REPLY_YES_POOL[rot8]
        if aux_id == 2:
            return CACHED_REPLY_NO_COMPLETE_POOL[rot8] if is_complete else CACHED_REPLY_NO_POOL[rot8]
        return CACHED_REPLY_MAYBE_COMPLETE_POOL[rot8] if is_complete else CACHED_REPLY_MAYBE_POOL[rot8]

    return None


def _snapshot_vault_hashes(vault_dict, exclude_prefix="s12_"):
    snap = {}
    for k, arr in vault_dict.items():
        if not k.startswith(exclude_prefix):
            snap[k] = (
                hashlib.sha256(arr.tobytes()).hexdigest(),
                arr.shape,
                arr.dtype.str,
            )
    return snap


def train_step12_into_vault(npz_path="model_weights.npz"):
    import torch
    import offline_train as ot

    if not os.path.exists("private_social_ask.txt") or not os.path.exists("yes_no_maybe.txt"):
        build_private_yesno_files()

    # Rule 12 Cryptographic .NPZ Vault Lock: Snapshot SHA-256 of all existing Steps 1-11 keys
    vault = {}
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            for k in data.files:
                vault[k] = data[k].copy()
    pre_hashes = _snapshot_vault_hashes(vault, exclude_prefix="s12_")

    ram_arena = ot.verify_hardware_and_lock_ram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wp_mat = ot._get_wp_mat()

    s12_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    s12_lut_a = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    s12_lut_b = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
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
        (8, S12_FRAME_WORDS),
        (7, S12_ACTION_VERBS),
        (6, S12_NEG_WORDS),
        (5, S12_MAYBE_WORDS),
        (4, S12_NO_WORDS),
        (3, S12_YES_WORDS),
        (2, S12_SOCIAL_WORDS),
        (1, S12_PRIVATE_TG_WORDS),
    ):
        for w in word_group:
            h4 = hash_word(ot._collapse_word(w))
            s12_lut_a[h4[0]] = np.uint8(code_val)
            s12_lut_b[h4[1]] = np.uint8(code_val)
            s12_bloom[h4[:3]] = np.int8(1)

    sample_map = {}
    for gf in (
        "greetings.txt", "ages.txt", "countries.txt", "hru.txt", "wyd.txt",
        "gender_ask_choice.txt", "status_positive.txt", "status_negative.txt",
        "compliments_flirt.txt", "reactions_ack.txt",
        "looking_for_ask.txt", "looking_for_state.txt",
        "name_ask.txt", "hobbies_ask.txt", "hobbies_tell.txt",
        "rel_status_ask.txt", "rel_status_tell.txt",
    ):
        if os.path.exists(gf):
            with open(gf, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and len(sample_map) < 14000:
                        toks = [w.strip("'!?.,/") for w in s.split()]
                        if not any(w in ALL_S12_ANCHOR_SET for w in toks):
                            sample_map[s] = (0, np.float32(2.0))

    s12_cnt = 0
    for fn, cls_id, builtins in (
        ("private_social_ask.txt", 1, ASK_PRIVATE_BASES),
        ("yes_no_maybe.txt", 2, YES_NO_MAYBE_BASES),
    ):
        for b_line in builtins:
            s = " ".join(b_line.strip().lower().split())
            if s:
                sample_map[s] = (cls_id, np.float32(36.0))
                s12_cnt += 1
        if os.path.exists(fn):
            with open(fn, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and ot._is_whitelist_clean(s, STRICT_ALLOWED_S12_VOCAB):
                        sample_map[s] = (cls_id, np.float32(24.0))
                        s12_cnt += 1

    # Strict Anti-Collision Negative Traps (Protects Class 0 "why not", Step 7 "yeah"/"ok", Step 8 "here to talk", Step 11 "no gf")
    for trap in (
        "what", "what?", "what is this", "what do you want", "what do you do",
        "why", "why?", "why not", "who", "who?", "who are you", "who r u",
        "yeah", "yea", "yep", "yup", "ok", "okay", "sure", "ofc", "haha", "lol", "same", "sam",
        "here to talk", "just here to talk", "here to chat", "just chatting",
        "no gf", "no bf", "i dont have a gf", "i dont have a bf", "not married",
        "a bf here", "a gf here", "f here", "m here", "call me alex", "what should i call u",
        "im good", "im bored", "ur cute", "are u single", "are u single or mariid",
    ):
        sample_map[trap] = (0, np.float32(65.0))

    q_emb, q_w1, q_w2, n = ot._train_word_mlp_head(sample_map, S12_CLASSES, 28, "STEP 12", device, wp_mat, use_qmark=True)

    vault.update(
        s12_emb_q=q_emb[0], s12_emb_s=q_emb[1],
        s12_w1_q=q_w1[0], s12_w1_s=q_w1[1],
        s12_w2_q=q_w2[0], s12_w2_s=q_w2[1],
        s12_bloom_q=s12_bloom, s12_lut_a=s12_lut_a, s12_lut_b=s12_lut_b,
    )
    np.savez(npz_path, **vault)
    del ram_arena

    # Rule 12 Post-Save Cryptographic SHA-256 Verification
    with np.load(npz_path, allow_pickle=False) as saved_arc:
        post_vault = {k: saved_arc[k] for k in saved_arc.files}
    post_hashes = _snapshot_vault_hashes(post_vault, exclude_prefix="s12_")
    if pre_hashes != post_hashes:
        raise RuntimeError("[RULE 12 VAULT BREACH] Existing Step 1-11 weights were altered!")

    size_kb = os.path.getsize(npz_path) / 1024.0
    print(f"[RULE 12 VAULT SHIELD] Verified SHA-256 integrity of all {len(pre_hashes)} locked Step 1-11 arrays!", flush=True)
    print(f"[STEP 12 DONE] Locked Step 12 Private/Social & Yes/No head into '{npz_path}' ({s12_cnt} lines, {n} total, {size_kb:.2f} KB).", flush=True)


if __name__ == "__main__":
    train_step12_into_vault("model_weights.npz")