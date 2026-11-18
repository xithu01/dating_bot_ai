#!/usr/bin/env python3
# FILE: step8_looking_for.py
# PURPOSE: Standalone Step 8 Plugin (Classes 21-22: Ask / State "What Are You Looking For?")
#          Handles:
#          - "m looking for girl , u", "lm finding gril", "f looking for boy wbu?"
#          - Questions ("what r u looking for?", "u looking for a girl?", "looking for friends or dating?")
#          - Direct responses ("gf", "a gf", "bf", "a bf", "friends", "new friends", "dating", "relationship")
#          - Persona-aware replies (Bot F vs Bot M) + automatic user gender extraction from "m looking..." / "f looking..."

import os
import sys
import numpy as np

from production_server_part2 import (
    FLAG_GREETED,
    _make_pool,
)

BLOOM_SIZE = 65536
S2_VOCAB = 1024
S2_DIM = 32
S8_CLASSES = 3  # 0=Reject, 1=AskLookingFor(21), 2=StateLookingFor(22)
MAX_USERS = 50000
NUM_WORKERS = 8

FLAG_S8_ASKED = 1
FLAG_S8_STATED = 2

# 10-Code Position-Aware O(1) LUT Groups:
# 1 = Core Looking/Finding Verbs ("looking", "finding", "seeking", "searching", "brings", "goal")
# 2 = Direct Female Targets ("gf", "girlfriend", "grilfriend", "wife", "wifey")
# 3 = Direct Male Targets ("bf", "boyfriend", "hubby", "husband")
# 4 = Direct Romance Targets ("relationship", "dating", "romance", "soulmate", "partner", "flirting")
# 5 = Direct Friend Targets ("friends", "freinds", "frends", "friendship", "company")
# 6 = Female Nouns ("girl", "gril", "gurl", "girls", "grils", "woman", "female", "famle", "lady", "f")
# 7 = Male Nouns ("boy", "boi", "boys", "guy", "guys", "man", "male", "maale", "dude", "m")
# 8 = Question / Choice Markers ("what", "wat", "wht", "why", "who", "or")
# 9 = Action / Desire Verbs ("making", "make", "want", "wanna", "need", "trying", "tryna", "meet", "meeting")
# 10 = Framing & Soft Targets ("im", "lm", "for", "to", "here", "u", "wbu", "chat", "chatting", "love", "fun")
S8_CORE_LOOK_VERBS = (
    "looking", "lookin", "loking", "lokin", "finding", "findin", "find", "fnding",
    "seeking", "seekin", "seek", "searching", "searchin", "search",
    "brings", "bring", "goal", "purpose", "reason",
)
S8_DIRECT_FEMALE_TARGETS = (
    "gf", "gfs", "girlfriend", "grilfriend", "girlfreind", "wifey", "wife",
)
S8_DIRECT_MALE_TARGETS = (
    "bf", "bfs", "boyfriend", "boifriend", "boyfreind", "hubby", "husband",
)
S8_DIRECT_ROMANCE_TARGETS = (
    "relationship", "relashionship", "relationships", "partner", "soulmate",
    "dating", "date", "dates", "romance", "romantic", "flirting", "flirt", "hookup",
)
S8_DIRECT_FRIEND_TARGETS = (
    "friends", "freind", "freinds", "frends", "frend", "friendship", "bestie", "company",
)
S8_FEMALE_NOUNS = (
    "girl", "gril", "gurl", "girls", "grils", "gurls", "woman", "women",
    "female", "famle", "famel", "females", "lady", "ladies", "f",
)
S8_MALE_NOUNS = (
    "boy", "boi", "boys", "bois", "guy", "guys", "man", "men",
    "male", "maale", "males", "dude", "dudes", "m",
)
S8_Q_CHOICE_WORDS = (
    "what", "whats", "what's", "wat", "wht", "why", "who", "or",
)
S8_ACTION_VERBS = (
    "making", "make", "trying", "tryna", "want", "wan", "wanna", "wanting",
    "need", "needing", "hoping", "meet", "meeting",
)
S8_FRAME_WORDS = (
    "im", "i'm", "l'm", "lm", "iam", "i'am", "i", "am", "just", "only", "mostly",
    "are", "ar", "r", "do", "you", "u", "ya", "your", "ur", "for", "to", "here",
    "on", "in", "a", "an", "new", "some", "good", "nice", "cute", "cool", "real",
    "serious", "true", "friend", "love", "fun", "someone", "somebody", "people",
    "ppl", "connection", "chat", "chatting", "chatin", "chating", "talk", "talking",
    "tbh", "ngl", "lol", "haha", "wbu", "hbu", "so", "btw", "and", "about", "with",
    "hi", "hey", "hello", "yo",
)

ALL_S8_ANCHOR_SET = frozenset(
    S8_CORE_LOOK_VERBS + S8_DIRECT_FEMALE_TARGETS + S8_DIRECT_MALE_TARGETS
    + S8_DIRECT_ROMANCE_TARGETS + S8_DIRECT_FRIEND_TARGETS + ("making",)
)

BUILTIN_ASK_LOOKING = (
    "what are you looking for", "what are you looking for?", "what r u looking for",
    "what r u looking for?", "what u looking for", "what u looking for?",
    "what are u looking for", "wat u looking for", "what r u lookin for",
    "what are you finding", "what r u finding", "what u finding",
    "what r u looking for here", "what are you here for", "what r u here for",
    "what do u want here", "what u want here", "who r u looking for", "who u looking for",
    "why are you here", "why r u here", "why r u on here", "why r u on here?",
    "why are you on here", "what brings u here", "what brings you here",
    "looking for?", "looking for what?", "finding what?", "wbu what r u looking for",
    "looking for friends or dating", "looking for friends or dating?",
    "looking for dating or friends", "looking for boy or girl", "looking for girl or boy",
    "are you looking for friends or dating", "are u looking for a bf", "r u looking for a bf",
    "are u looking for a gf", "r u looking for a gf", "u looking for a girl",
    "u looking for a girl?", "u looking for a boy", "u looking for a boy?",
    "u looking for a gf", "u looking for a bf", "u looking for friends",
    "are u looking for friends", "are u looking for a relationship",
    "friends?", "dating?", "relationship?", "gf?", "bf?",
)

BUILTIN_STATE_LOOKING = (
    # User exact test cases & variants
    "m looking for girl , u", "m looking for girl u", "m looking for girl u?",
    "m looking for a girl", "m looking for a girl u?", "m looking for gril",
    "m looking for gf", "m looking for a gf", "m looking for f", "m looking for female",
    "lm finding gril", "im finding gril", "lm finding girl", "im finding a girl",
    "lm looking for gril", "im looking for girl", "im looking for a girl",
    "f looking for boy", "f looking for boy u", "f looking for a boy",
    "f looking for guy", "f looking for bf", "f looking for m", "girl looking for boy",
    "boy looking for girl", "guy looking for girl",
    # Friends / Chat
    "looking for friends", "im looking for friends", "just looking for friends",
    "looking for new friends", "looking for some friends", "making friends",
    "making new friends", "just making friends", "im making friends",
    "here to make friends", "looking to make friends", "here to chat",
    "just here to chat", "looking to chat", "just looking to chat",
    "looking for friends u?", "im looking for friends u?", "making friends wbu?",
    # GF / BF / Dating / Relationship
    "looking for a gf", "im looking for a gf", "looking for gf", "looking for a girlfriend",
    "im looking for a girlfriend", "looking for a bf", "im looking for a bf",
    "looking for bf", "looking for a boyfriend", "im looking for a boyfriend",
    "looking for a relationship", "im looking for a relationship", "looking for dating",
    "im looking for dating", "looking for love", "im looking for love",
    "looking for fun", "just looking for fun", "here for dating", "here for friends",
    "i want a gf", "i want a girl", "want a gf", "need a gf", "tryna find a gf",
    "tryna find a girl", "trying to find a girl",
    # Direct Short Responses
    "gf", "a gf", "girlfriend", "a girlfriend", "grilfriend", "gf u?", "a gf wbu?",
    "bf", "a bf", "boyfriend", "a boyfriend", "bf u?", "a bf wbu?",
    "friends", "new friends", "some friends", "good friends", "just friends",
    "only friends", "freinds", "frends", "friendship", "friends u?", "friends wbu?",
    "dating", "just dating", "relationship", "a relationship", "serious relationship",
    "romance", "soulmate", "a soulmate", "dating u?", "relationship wbu?",
)

# --- STEP 8 PERSONA-AWARE & TARGET-CONFIRMING RESPONSE POOLS (8 items each) ---
CACHED_ASK_LOOKING_F_POOL = _make_pool(
    21, "false", "ask_looking_for",
    (
        "mostly looking to make new friends or maybe meet a cute guy :) wbu?",
        "im here to chat, make friends, or maybe date if we vibe! what r u looking for?",
        "looking for friends or maybe a bf if we click :) wbu?",
        "just looking for cool people to vibe with, maybe dating! what about u?",
        "here to make friends and see where things go :) what r u looking for?",
        "looking for a nice guy or new friends! wbu?",
        "mostly friends or dating if the vibe is right :) what brings u here?",
        "just looking for someone fun to talk to, maybe more :) wbu?",
    ),
    '["looking for a girl","looking for friends","looking for dating"]',
)

CACHED_ASK_LOOKING_M_POOL = _make_pool(
    21, "false", "ask_looking_for",
    (
        "mostly looking to make new friends or maybe meet a cute girl :) wbu?",
        "im here to chat, make friends, or maybe date if we vibe! what r u looking for?",
        "looking for friends or maybe a gf if we click :) wbu?",
        "just looking for cool people to vibe with, maybe dating! what about u?",
        "here to make friends and see where things go :) what r u looking for?",
        "looking for a nice girl or new friends! wbu?",
        "mostly friends or dating if the vibe is right :) what brings u here?",
        "just looking for someone fun to talk to, maybe more :) wbu?",
    ),
    '["looking for a boy","looking for friends","looking for dating"]',
)

CACHED_REPEAT_ASK_LOOKING_POOL = _make_pool(
    21, "false", "ask_looking_for_repeat",
    (
        "haha i told u! looking for friends or dating if we vibe :) wbu?",
        "already said lol! here to make friends or maybe more :) what about u?",
        "haha still looking for good vibes, friends or dating! wbu?",
        "told u earlier lol :) friends or maybe dating if we click!",
        "haha u asked me that already! just looking for cool people :) wbu?",
        "still here for friends or maybe dating lol! what r u looking for?",
        "haha i said friends or dating if we vibe! wbu?",
        "already told u lol :) looking for someone fun to talk to!",
    ),
    '["looking for friends","looking for dating"]',
)

# User is looking for a Girl / GF (aux_id == 2) -> Bot is Female (bot_g == 2)
CACHED_STATE_GIRL_BOT_F_POOL = _make_pool(
    22, "false", "state_looking_for_girl",
    (
        "ooh nice! well im a girl and im looking for a nice guy or friends :) how old r u?",
        "lucky u, im a girl :) im looking for friends or dating if we vibe! where r u from?",
        "aw nice! F here looking for a sweet guy or friends :) how old r u and where u from?",
        "haha well im a girl ;) im looking for good vibes or dating! how old r u?",
        "ooh cool! im F and open to dating if we click :) what's ur age and country?",
        "nice! girl here looking for friends or maybe a bf :) how old r u?",
        "haha maybe u found her ;) im F looking for friends or dating! where u from?",
        "aw sweet! im a girl looking to meet cool people too :) how old are u?",
    ),
    '["im 22","from usa","24 m"]',
)

CACHED_STATE_GIRL_BOT_F_COMPLETE_POOL = _make_pool(
    22, "false", "state_looking_for_girl_complete",
    (
        "haha lucky u, im a girl ;) im looking for a nice guy or friends too! wyd rn?",
        "ooh well im F and open to dating if we vibe :) hows ur day going?",
        "haha maybe u found what ur looking for right here ;) what u up to?",
        "aw nice! im a girl looking for good vibes or dating too :) wyd?",
        "well im a girl :) let's see if we click then! what do u like doing for fun?",
        "ooh i like that :) F here looking for someone chill too! hows ur day?",
        "haha well im right here ;) im F looking for friends or dating too!",
        "nice! im a girl and i think we have a good vibe already :) wyd rn?",
    ),
    '["ur cute","just chilling","wyd?"]',
)

# User is looking for a Girl / GF (aux_id == 2) -> Bot is Male (bot_g == 1)
CACHED_STATE_GIRL_BOT_M_POOL = _make_pool(
    22, "false", "state_looking_for_girl",
    (
        "ah im a guy actually, just looking for friends here! how old r u?",
        "im M here lol, looking for a girl too! where r u from?",
        "haha im a guy actually, same here looking for a girl or friends! ur age?",
        "ah im M lol, just making friends or looking for a gf too! where u from?",
        "im a guy actually :) down to be friends tho! how old r u?",
        "haha same bro, im M looking for a girl or friends too! where r u from?",
        "ah im a boy lol, but we can chill as friends! how old are u?",
        "im M here actually, looking for friends or a gf too! what country u from?",
    ),
    '["im 22","from usa","ok cool"]',
)

# User is looking for a Boy / BF (aux_id == 3) -> Bot is Female (bot_g == 2)
CACHED_STATE_BOY_BOT_F_POOL = _make_pool(
    22, "false", "state_looking_for_boy",
    (
        "ah im a girl actually, just looking for friends or a guy too! how old r u?",
        "im F here lol, looking for a guy too! we can be friends tho, where r u from?",
        "haha im a girl actually! always down to make new friends tho, ur age?",
        "ah im F lol, looking for a bf too! let's be friends :) where u from?",
        "im a girl actually :) down to chat as friends tho! how old r u?",
        "haha same girl, im F looking for a guy or friends too! where r u from?",
        "ah im female lol, but we can definitely be friends! how old are u?",
        "im F here actually! love making new friends tho, what country u from?",
    ),
    '["im 22","from usa","let''s be friends"]',
)

# User is looking for a Boy / BF (aux_id == 3) -> Bot is Male (bot_g == 1)
CACHED_STATE_BOY_BOT_M_POOL = _make_pool(
    22, "false", "state_looking_for_boy",
    (
        "ooh nice! well im a guy and im looking for a nice girl or friends :) how old r u?",
        "lucky u, im a guy :) im looking for friends or dating if we vibe! where r u from?",
        "aw nice! M here looking for a sweet girl or friends :) how old r u and where u from?",
        "haha well im a guy ;) im looking for good vibes or dating! how old r u?",
        "ooh cool! im M and open to dating if we click :) what's ur age and country?",
        "nice! guy here looking for friends or maybe a gf :) how old r u?",
        "haha maybe u found him ;) im M looking for friends or dating! where u from?",
        "aw sweet! im a guy looking to meet cool people too :) how old are u?",
    ),
    '["im 22","from usa","21 f"]',
)

# User is looking for General Dating / Relationship / Love (aux_id == 4)
CACHED_STATE_DATING_POOL = _make_pool(
    22, "false", "state_looking_for_dating",
    (
        "ooh nice! im open to dating or friends too if we vibe :) how old r u and where u from?",
        "aw i hope u find someone special! im open to dating too :) how old r u?",
        "nice! im looking for friends or a relationship if we click :) where r u from?",
        "ooh i like that :) im open to dating too! how old r u and what country u in?",
        "aw thats sweet! im looking for good vibes or dating too :) ur age and country?",
        "haha maybe u found what ur looking for right here ;) how old r u?",
        "nice! im open to dating too when the vibe is right :) where u from?",
        "ooh cool! im looking for friends or dating too :) how old are u and where r u from?",
    ),
    '["im 22","from usa","24 m"]',
)

CACHED_STATE_DATING_COMPLETE_POOL = _make_pool(
    22, "false", "state_looking_for_dating_complete",
    (
        "ooh i like that :) im open to dating too if we click! what u up to rn?",
        "haha maybe u found what ur looking for right here ;) hows ur day going?",
        "aw thats sweet! im open to dating too and i like ur vibe already :)",
        "ooh nice! im definitely open to dating if we vibe :) wyd rn?",
        "love that :) im looking for the same thing! what do u like doing for fun?",
        "haha well im right here ;) im open to dating too! what r u up to today?",
        "aw i hope we vibe then :) ur really fun to talk to!",
        "nice! im looking for friends or dating too :) chilling here right now, wbu?",
    ),
    '["ur cute","just chilling","wyd?"]',
)

# User is looking for Friends / Chat (aux_id == 1)
CACHED_STATE_FRIENDS_POOL = _make_pool(
    22, "false", "state_looking_for_friends",
    (
        "nice! im here to make new friends too :) how old r u and where u from?",
        "aw same, im looking for friends too! how old r u?",
        "cool! i love meeting new friends here :) where r u from?",
        "awesome! we can definitely be friends :) what's ur age and country?",
        "nice! im looking for friends too :) how old are u and where u from?",
        "sweet! always down for new friends :) what country u in?",
        "ooh cool! making friends is the best, same here :) how old r u?",
        "nice! let's be friends then :) ur age and where u from?",
    ),
    '["im 22","from usa","24 m"]',
)

CACHED_STATE_FRIENDS_COMPLETE_POOL = _make_pool(
    22, "false", "state_looking_for_friends_complete",
    (
        "nice! im looking for friends too, we can definitely be friends :) wyd rn?",
        "awesome! im always down for new friends :) hows ur day going?",
        "sweet! u seem really chill to be friends with :) what u up to?",
        "cool! glad we matched then, im here for friends too :) wyd?",
        "nice! making new friends is the best :) just chilling here, wbu?",
        "aw yay, let's be friends then :) how is ur day going?",
        "sweet! love making cool new friends :) what u up to today?",
        "awesome! u seem super nice :) what r u doing rn?",
    ),
    '["just chilling","ur nice too","wyd?"]',
)


class Step8Engine:
    __slots__ = (
        "enabled", "s8_emb_f32", "s8_w1_f32", "s8_w2_f32",
        "s8_bloom_q", "s8_lut_a", "s8_lut_b",
        "user_flags", "user_goal",
        "hits_a", "hits_b", "logits",
    )

    def __init__(self, npz_path="model_weights.npz"):
        self.enabled = False
        self.user_flags = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.user_goal = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.hits_a = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.hits_b = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.logits = np.zeros((NUM_WORKERS, 1, S8_CLASSES), dtype=np.float32)

        if not os.path.exists(npz_path):
            return
        arc = np.load(npz_path, mmap_mode="r")
        req = ("s8_emb_q", "s8_w1_q", "s8_w2_q", "s8_bloom_q", "s8_lut_a", "s8_lut_b")
        if any(k not in arc.files for k in req):
            arc.close()
            print("[STEP 8 INFO] Step 8 weights not in vault yet. Run: python step8_looking_for.py", flush=True)
            return

        def lock_f32(prefix, zero_row0=False):
            q = arc[f"{prefix}_q"].astype(np.int8, copy=False)
            s = arc[f"{prefix}_s"].astype(np.float32, copy=False)
            f32 = np.multiply(q, s, dtype=np.float32)
            if zero_row0:
                f32[0].fill(0.0)
            f32.flags.writeable = False
            return f32

        self.s8_emb_f32 = lock_f32("s8_emb", True)
        self.s8_w1_f32 = lock_f32("s8_w1")
        self.s8_w2_f32 = lock_f32("s8_w2")
        for attr, dt in (("s8_bloom_q", np.int8), ("s8_lut_a", np.uint8), ("s8_lut_b", np.uint8)):
            arr = arc[attr].astype(dt, copy=False)
            arr.flags.writeable = False
            setattr(self, attr, arr)
        arc.close()
        self.enabled = True


S8_ENGINE = None


def get_step8_engine(npz_path="model_weights.npz"):
    global S8_ENGINE
    if S8_ENGINE is None:
        S8_ENGINE = Step8Engine(npz_path)
    return S8_ENGINE


def forward_step8(model, slot, n_words, has_q, age_val, country_id):
    eng = S8_ENGINE
    if eng is None or not eng.enabled or age_val != 0 or country_id != 0:
        return 0, 0, 0

    sid = slot.slot_id
    ha = eng.hits_a[sid]
    hb = eng.hits_b[sid]
    np.take(eng.s8_lut_a, slot.w_b_idx[:, 0], out=ha)
    np.take(eng.s8_lut_b, slot.w_b_idx[:, 1], out=hb)
    m = (ha[:n_words] > 0) & (ha[:n_words] == hb[:n_words])
    codes = ha[:n_words] * m

    if not bool(np.all(codes > 0)):
        return 0, 0, 0

    has_core_v = bool(np.any(codes == 1))
    has_direct_f = bool(np.any(codes == 2))
    has_direct_m = bool(np.any(codes == 3))
    has_direct_rom = bool(np.any(codes == 4))
    has_direct_fr = bool(np.any(codes == 5))
    has_noun_f = bool(np.any(codes == 6))
    has_noun_m = bool(np.any(codes == 7))
    has_q_word = bool(np.any(codes == 8))
    has_act_v = bool(np.any(codes == 9))

    has_any_direct = has_direct_f or has_direct_m or has_direct_rom or has_direct_fr
    has_any_verb = has_core_v or has_act_v

    # Check if "here" + ("for"/"to"/"why"/"what") is present (e.g., "here to chat", "why r u here")
    has_here = False
    has_for_to = False
    for w_i in range(n_words):
        c0 = int(slot.words_mat[w_i, 0])
        Lw = int(slot.words_mat[w_i, 15])
        if Lw == 4 and c0 == 104 and int(slot.words_mat[w_i, 1]) == 101 and int(slot.words_mat[w_i, 2]) == 114:
            has_here = True
        elif (Lw == 3 and c0 == 102 and int(slot.words_mat[w_i, 1]) == 111) or (Lw == 2 and c0 == 116 and int(slot.words_mat[w_i, 1]) == 111):
            has_for_to = True

    if not (has_core_v or has_any_direct or (has_act_v and (has_noun_f or has_noun_m or has_for_to or n_words >= 2)) or (has_here and (has_q_word or has_for_to))):
        return 0, 0, 0

    np.take(eng.s8_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
    if not bool(np.all(slot.w_b_hits[:n_words])):
        return 0, 0, 0

    slot.s4_tokens[:6] = slot.s2_tokens[:6]
    slot.s4_tokens[6] = np.int32(1023) if has_q else model.one_i32
    q_off = np.int32(509) if has_q else np.int32(0)
    slot.s4_tokens[7] = (((slot.s2_tokens[0] * np.int32(37)) + (slot.s2_tokens[n_words - 1] * np.int32(73)) + q_off) & model.s2_v_mask) + model.one_i32

    np.take(eng.s8_emb_f32, slot.s4_tokens, axis=0, out=slot.s2_emb_buf)
    slot.s2_emb_buf[6] *= np.float32(2.5)
    np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
    slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
    np.matmul(slot.s2_pooled, eng.s8_w1_f32, out=slot.s2_mid)
    np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
    out_l = eng.logits[sid]
    np.matmul(slot.s2_mid, eng.s8_w2_f32, out=out_l)

    s0, s_ask, s_state = float(out_l[0, 0]), float(out_l[0, 1]), float(out_l[0, 2])
    if max(s_ask, s_state) <= s0 + 0.1:
        return 0, 0, 0

    # Find first verb or "here" position to separate subject prefix ("m", "f", "u") from target ("girl", "boy")
    v_idx = -1
    for w_i in range(n_words):
        if int(codes[w_i]) in (1, 9):
            v_idx = w_i
            break

    # Check if 2nd-person pronoun ("u", "you") appears BEFORE the verb (e.g. "u looking for a girl?" -> Class 21)
    u_before_verb = False
    if v_idx > 0:
        for w_i in range(v_idx):
            c0 = int(slot.words_mat[w_i, 0])
            Lw = int(slot.words_mat[w_i, 15])
            if (Lw == 1 and c0 == 117) or (Lw == 3 and c0 == 121 and int(slot.words_mat[w_i, 1]) == 111):
                u_before_verb = True
                break

    if has_q_word or u_before_verb or (n_words == 1 and has_q):
        return 21, 0, 0

    after_v = codes[v_idx + 1 :] if v_idx >= 0 else codes
    tgt_f = has_direct_f or bool(np.any(after_v == 6))
    tgt_m = has_direct_m or bool(np.any(after_v == 7))
    tgt_rom = has_direct_rom
    tgt_fr = has_direct_fr

    # If no explicit target after verb (e.g. "looking for?", "wbu looking for"), it's Class 21 unless "here to chat/make friends"
    if not (tgt_f or tgt_m or tgt_rom or tgt_fr or (has_here and has_for_to) or (has_act_v and n_words >= 2)):
        return 21, 0, 0

    # Determine self-gender prefix before the verb (e.g. "m looking for girl" -> self_g=10, "f looking for boy" -> self_g=20)
    self_g_off = 0
    if v_idx > 0:
        if int(codes[0]) == 7:
            self_g_off = 10
        elif int(codes[0]) == 6:
            self_g_off = 20

    if tgt_f:
        goal_id = 2
    elif tgt_m:
        goal_id = 3
    elif tgt_rom:
        goal_id = 4
    else:
        goal_id = 1
    return 22, 0, goal_id + self_g_off


def route_step8(class_id, aux_id, users, uid, flags, prev_age, prev_country, bot_g, rot8):
    eng = S8_ENGINE
    if eng is None:
        return None
    if int(users.msg_count[uid]) <= 1 and flags == 0:
        eng.user_flags[uid] = np.uint8(0)
        eng.user_goal[uid] = np.uint8(0)

    s8_f = int(eng.user_flags[uid])
    if class_id == 21:
        already_asked = (s8_f & FLAG_S8_ASKED) != 0
        eng.user_flags[uid] = np.uint8(s8_f | FLAG_S8_ASKED)
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        if already_asked:
            return CACHED_REPEAT_ASK_LOOKING_POOL[rot8]
        return CACHED_ASK_LOOKING_F_POOL[rot8] if bot_g == 2 else CACHED_ASK_LOOKING_M_POOL[rot8]

    if class_id == 22:
        self_g = aux_id // 10
        goal_id = aux_id % 10
        if self_g in (1, 2):
            users.gender[uid] = np.uint8(self_g)
        eng.user_flags[uid] = np.uint8(s8_f | FLAG_S8_STATED)
        if goal_id > 0:
            eng.user_goal[uid] = np.uint8(goal_id)
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        is_complete = prev_age >= 18 and prev_country > 0

        if goal_id == 2:
            if bot_g == 2:
                return CACHED_STATE_GIRL_BOT_F_COMPLETE_POOL[rot8] if is_complete else CACHED_STATE_GIRL_BOT_F_POOL[rot8]
            return CACHED_STATE_GIRL_BOT_M_POOL[rot8]
        if goal_id == 3:
            if bot_g == 1:
                return CACHED_STATE_BOY_BOT_M_POOL[rot8]
            return CACHED_STATE_BOY_BOT_F_POOL[rot8]
        if goal_id == 4:
            return CACHED_STATE_DATING_COMPLETE_POOL[rot8] if is_complete else CACHED_STATE_DATING_POOL[rot8]
        return CACHED_STATE_FRIENDS_COMPLETE_POOL[rot8] if is_complete else CACHED_STATE_FRIENDS_POOL[rot8]

    return None


def train_step8_into_vault(npz_path="model_weights.npz"):
    import torch
    import offline_train as ot
    import build_looking_for_txt

    build_looking_for_txt.build_looking_for_files()

    ram_arena = ot.verify_hardware_and_lock_ram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wp_mat = ot._get_wp_mat()

    s8_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    s8_lut_a = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    s8_lut_b = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
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
        (10, S8_FRAME_WORDS),
        (9, S8_ACTION_VERBS),
        (8, S8_Q_CHOICE_WORDS),
        (7, S8_MALE_NOUNS),
        (6, S8_FEMALE_NOUNS),
        (5, S8_DIRECT_FRIEND_TARGETS),
        (4, S8_DIRECT_ROMANCE_TARGETS),
        (3, S8_DIRECT_MALE_TARGETS),
        (2, S8_DIRECT_FEMALE_TARGETS),
        (1, S8_CORE_LOOK_VERBS),
    ):
        for w in word_group:
            h4 = hash_word(ot._collapse_word(w))
            s8_lut_a[h4[0]] = np.uint8(code_val)
            s8_lut_b[h4[1]] = np.uint8(code_val)
            s8_bloom[h4[:3]] = np.int8(1)

    sample_map = {}
    for gf in (
        "greetings.txt", "ages.txt", "countries.txt", "hru.txt", "wyd.txt",
        "gender_ask_choice.txt", "status_positive.txt", "status_negative.txt",
        "compliments_flirt.txt", "reactions_ack.txt",
    ):
        if os.path.exists(gf):
            with open(gf, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and len(sample_map) < 14000:
                        toks = [w.strip("'!?.,/") for w in s.split()]
                        if not any(w in ALL_S8_ANCHOR_SET for w in toks):
                            sample_map[s] = (0, np.float32(2.0))

    s8_cnt = 0
    for fn, cls_id, builtins in (
        ("looking_for_ask.txt", 1, BUILTIN_ASK_LOOKING),
        ("looking_for_state.txt", 2, BUILTIN_STATE_LOOKING),
    ):
        for b_line in builtins:
            s = " ".join(b_line.strip().lower().split())
            if s:
                sample_map[s] = (cls_id, np.float32(36.0))
                s8_cnt += 1
                clean_s = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in s)
                for w in ot._collapse_word(clean_s).split():
                    s8_bloom[hash_word(w)[:3]] = np.int8(1)
        if os.path.exists(fn):
            with open(fn, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s:
                        sample_map[s] = (cls_id, np.float32(24.0))
                        s8_cnt += 1
                        clean_s = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in s)
                        for w in ot._collapse_word(clean_s).split():
                            s8_bloom[hash_word(w)[:3]] = np.int8(1)

    for trap in ot.QUESTION_AND_REACTION_TRAPS + (
        "what are you doing", "what r u doing", "what u up to", "wyd",
        "just chatting", "chatting", "im chatting", "im chat", "chilling",
        "u look good", "you look good", "u look cute", "ur cute", "u seem nice",
        "same", "sam", "same here", "sam here", "m here", "f here", "im here",
        "im a girl", "im a boy", "are u a girl", "are u a boy", "f", "m",
        "where are you from", "how old are you", "how are you", "m or f?",
    ):
        sample_map[trap] = (0, np.float32(65.0))

    q_emb, q_w1, q_w2, n = ot._train_word_mlp_head(sample_map, S8_CLASSES, 28, "STEP 8", device, wp_mat, use_qmark=True)

    vault = {}
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            for k in data.files:
                vault[k] = data[k]

    vault.update(
        s8_emb_q=q_emb[0], s8_emb_s=q_emb[1],
        s8_w1_q=q_w1[0], s8_w1_s=q_w1[1],
        s8_w2_q=q_w2[0], s8_w2_s=q_w2[1],
        s8_bloom_q=s8_bloom, s8_lut_a=s8_lut_a, s8_lut_b=s8_lut_b,
    )
    np.savez(npz_path, **vault)
    del ram_arena
    size_kb = os.path.getsize(npz_path) / 1024.0
    print(f"[STEP 8 DONE] Locked Step 8 into '{npz_path}' ({s8_cnt} lines, {n} total, {size_kb:.2f} KB).", flush=True)

    # Instant Built-In Zero-Fault Verification on User's Exact Test Cases
    global S8_ENGINE
    S8_ENGINE = None
    get_step8_engine(npz_path)
    from production_server import InferenceSlot, QuantizedModel, forward_pass
    m_test = QuantizedModel(npz_path)
    s_test = InferenceSlot(0)
    s8_tests = (
        (b"m looking for girl , u", 22),
        (b"lm finding gril", 22),
        (b"f looking for boy wbu?", 22),
        (b"im looking for a gf", 22),
        (b"looking for friends", 22),
        (b"making friends", 22),
        (b"gf", 22),
        (b"a bf", 22),
        (b"friends", 22),
        (b"dating", 22),
        (b"what r u looking for?", 21),
        (b"why r u on here?", 21),
        (b"looking for friends or dating?", 21),
        (b"u looking for a girl?", 21),
        (b"friends?", 21),
    )
    passed = sum(1 for txt, exp in s8_tests if forward_pass(m_test, s_test, memoryview(txt))[0] == exp)
    print(f"[STEP 8 VERIFICATION] Passed {passed}/{len(s8_tests)} tests (including 'm looking for girl , u' & 'lm finding gril')!", flush=True)


if __name__ == "__main__":
    train_step8_into_vault("model_weights.npz")