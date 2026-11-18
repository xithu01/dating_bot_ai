# FILE: production_server_part2.py
# PURPOSE: Step 1-4 Routing Engine & Response Pools (Gender Yes/No, Double-Memory, CACHED HTTP)
# RULES: Zero PyTorch/SciPy/DB, zero float64/int64, PyArmor compatible (< 400 lines)

import asyncio
import time
import numpy as np
from collections import deque

BLOOM_SIZE = 65536
SESSION_TIMEOUT_S = 600
MAX_USERS = 50000

# Conversation Memory Flags
FLAG_GREETED = 1
FLAG_ASKED_AGE = 2
FLAG_ASKED_COUNTRY = 4
FLAG_ASKED_HRU = 8
FLAG_ASKED_GENDER = 16

_VALID_SINGLE_GREETING_WORDS = (
    "hi", "hey", "helo", "hlo", "hy", "hie", "hei", "hiya", "heya", "hulo", "elo",
    "yo", "sup", "wasup", "wazup", "whazup", "howdy", "gretings", "greeting",
    "salutations", "welcome", "aloha", "gday", "morning", "mornin", "afternon", "evening",
)

_VALID_GREETING_TARGETS = (
    "there", "u", "you", "friend", "stranger", "cutie", "gorgeous", "handsome",
    "bro", "man", "mate", "budy", "everyone", "al", "partner",
)

_VALID_MULTI_GREETING_WORDS = _VALID_SINGLE_GREETING_WORDS + _VALID_GREETING_TARGETS + (
    "god", "day", "top", "of", "the", "whats", "what's", "what", "is", "up",
    "nice", "pleased", "glad", "to", "met", "se", "long", "time", "no",
)

_EXTRA_S2_AGE_FRAMING_WORDS = _VALID_MULTI_GREETING_WORDS + (
    "im", "i'm", "l'm", "lm", "iam", "i'am", "i", "am", "me", "my", "from", "in",
    "live", "living", "age", "old", "olds", "y", "yo", "yr", "yrs", "yrz", "yres",
    "yers", "yer", "yea", "year", "years", "yeas", "yars", "yar", "yaers", "yaer",
    "yro", "yold", "o", "m", "f", "male", "female", "boy", "girl", "guy", "woman",
    "where", "country", "which", "and", "about", "how", "are", "r", "ya", "today",
    "it", "your", "ur", "do",
)

_GREETING_LEAD_WORDS = ("hi", "hey", "helo", "hlo", "elo", "yo", "sup", "hiya", "heya", "howdy", "hy")


def _pack_reply(class_id, is_greeting_str, intent_str, reply_str, kbd_str):
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


def _make_pool(class_id, is_g, intent_s, msgs, kbd_s):
    return tuple(_pack_reply(class_id, is_g, intent_s, m, kbd_s) for m in msgs)


# --- CLASS 0, 7, 99 (Rejects & Resets) ---
CACHED_REJECT_RESPONSE = _pack_reply(0, "false", "not_valid", "say hi, how r u, ur age (18+), or where u are from!", '["hey","how r u?","im 22"]')
CACHED_UNDERAGE_RESPONSE = _pack_reply(7, "false", "underage_reject", "sorry, 18+ adults only!", '["Exit"]')
CACHED_RESET_RESPONSE = _pack_reply(99, "false", "system_reset", "new chat started! say hi :)", '["hey","hi!"]')

# --- STEP 1: GREETINGS (CLASS 1) ---
CACHED_GREETING_POOL = _make_pool(1, "true", "english_greeting", ("hey", "hi", "heyy", "yo", "hey there", "hii", "sup", "hello!"), '["good u?","how r u?"]')
CACHED_REPEAT_GREETING_POOL = _make_pool(1, "true", "english_greeting_repeat", ("hey again lol", "haha hi again!", "we already said hi :)", "yo again! how r u?"), '["haha im good","how r u?"]')

# --- STEP 2: AGE & COUNTRY (CLASS 2..6) ---
CACHED_ASK_AGE_POOL = _make_pool(2, "false", "ask_age", ("22 u?", "im 23, how old r u?", "22! wbu?", "im 23 :)", "22 here, u?", "23! how old are u", "im 22, ur age?", "23 u?"), '["im 21","im 24"]')
CACHED_REPEAT_ASK_AGE_POOL = _make_pool(2, "false", "ask_age_repeat", ("haha i told u im 22! u?", "already said 22 lol! how old r u?", "im 22 haha, forgot already?"), '["haha my bad","nice!"]')
CACHED_TELL_AGE_POOL = _make_pool(3, "false", "tell_age", ("nice! im 22, where u from?", "cool im 23! u from?", "nice age! im 22, from?"), '["usa","uk"]')
CACHED_AGE_COMPLETE_POOL = _make_pool(3, "false", "tell_age_complete", ("nice! im 22 too :) hows ur day going?", "cool age! im 23, what u up to today?"), '["going great!","chilling u?"]')
CACHED_REPEAT_AGE_POOL = _make_pool(3, "false", "tell_age_repeat", ("haha u already said ur age! where r u from?", "ik ur age lol! what country u in?"), '["usa","uk"]')
CACHED_CHANGED_AGE_POOL = _make_pool(3, "false", "tell_age_changed", ("wait u said a different age before lol! where u from?", "haha changing ur age now? lol how r u"), '["haha jk","lol my bad"]')
CACHED_ASK_COUNTRY_POOL = _make_pool(4, "false", "ask_country", ("from usa, u?", "uk! where u from?", "im from usa :)", "canada, wbu?", "usa here! u from?", "from uk, where r u from?", "usa! what about u?", "im in usa, u?"), '["nice!","me too"]')
CACHED_REPEAT_ASK_COUNTRY_POOL = _make_pool(4, "false", "ask_country_repeat", ("haha i said usa! where u from?", "told u im from usa lol :)", "already said usa haha! wbu?", "still in usa lol! forgot already?"), '["haha my bad","cool!"]')
CACHED_TELL_COUNTRY_POOL = _make_pool(5, "false", "tell_country", ("ooh nice! im from usa, how old r u?", "cool place! im from uk, age?", "nice! im from usa :) ur age?"), '["im 21","im 24"]')
CACHED_COUNTRY_COMPLETE_POOL = _make_pool(5, "false", "tell_country_complete", ("ooh nice! im from usa :) hows ur day going?", "cool place! im from uk, what u up to?"), '["going great!","chilling u?"]')
CACHED_REPEAT_COUNTRY_POOL = _make_pool(5, "false", "tell_country_repeat", ("ik ur from there lol! how old r u?", "haha u already told me ur country! whats up?", "yeah u said that already lol! how r u?"), '["haha im good","chilling"]')
CACHED_CHANGED_COUNTRY_POOL = _make_pool(5, "false", "tell_country_changed", ("wait u said a different country before lol!", "haha moving countries fast lol! how r u?"), '["haha jk","traveling lol"]')
CACHED_COMBINED_AGE_COUNTRY_POOL = _make_pool(6, "true", "tell_age_and_country", ("hey! nice im 22 from usa, how r u?", "hi! cool im 23 from uk, hru?", "nice to meet u! 22 from usa here :)", "heyy! im 22 from canada, hows it going?"), '["im good u?","doing great!"]')

# --- STEP 3: ASK HOW ARE YOU (CLASS 8) ---
CACHED_ASK_HRU_POOL = _make_pool(8, "false", "ask_how_are_you", ("im doing great! how old r u and where u from?", "doing good thanks! wbu, what's ur age and country?", "im good! im 22 from usa, how old r u and where u from?", "all good here! how old are u and where r u from?"), '["im 22","from usa"]')

# --- STEP 4: GENDER (CLASS 9..14) ---
# Female Bot Persona (BOT_PERSONA_GENDER = 2) Responses
CACHED_GENDER_CHOICE_F_POOL = _make_pool(9, "false", "ask_gender_choice", ("I'm F 😊, you?", "im a girl! wbu?", "female here, you?", "F, u?"), '["M","F"]')
CACHED_GENDER_YES_F_POOL = _make_pool(10, "false", "ask_gender_is_female", ("yes im a girl!", "yep F here!", "yeah female :)", "yes! u?"), '["M","F"]')
CACHED_GENDER_NO_F_POOL = _make_pool(11, "false", "ask_gender_is_male", ("no im a girl actually!", "nope F here!", "im female lol", "no, im F! u?"), '["M","F"]')
# Male Bot Persona (BOT_PERSONA_GENDER = 1) Responses
CACHED_GENDER_CHOICE_M_POOL = _make_pool(9, "false", "ask_gender_choice", ("I'm M 😊, you?", "im a guy! wbu?", "male here, you?", "M, u?"), '["M","F"]')
CACHED_GENDER_YES_M_POOL = _make_pool(11, "false", "ask_gender_is_male", ("yes im a guy!", "yep M here!", "yeah male :)", "yes! u?"), '["M","F"]')
CACHED_GENDER_NO_M_POOL = _make_pool(10, "false", "ask_gender_is_female", ("no im a guy actually!", "nope M here!", "im male lol", "no, im M! u?"), '["M","F"]')

# Tell Gender Responses
CACHED_TELL_GENDER_POOL = _make_pool(12, "false", "state_gender", ("nice! how old r u?", "cool! ur age?", "nice :) how old?", "sweet, age?"), '["22","24"]')
CACHED_TELL_GENDER_ASK_COUNTRY_POOL = _make_pool(12, "false", "state_gender_ask_country", ("nice! what country?", "cool! where u from?", "awesome, country?", "sweet, where r u from?"), '["USA","UK"]')
CACHED_TELL_AGE_GENDER_POOL = _make_pool(14, "false", "state_age_gender", ("cool! where u from?", "nice! what country?", "awesome, where u from?", "sweet! u from?"), '["USA","UK"]')
CACHED_PROFILE_COMPLETE_POOL = _make_pool(12, "false", "profile_complete", ("awesome! hows ur day going?", "sweet! what u up to?", "nice! how are u doing?", "cool :) whats up?"), '["chilling","working"]')


def _build_step1_bloom_tables(w_poly_mat, s2_frame_bloom):
    word_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    single_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    target_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    lead_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    b_mask = np.int32(BLOOM_SIZE - 1)
    w_vec = np.zeros((16,), dtype=np.int32)

    def h_idx(w_str):
        b = np.frombuffer(w_str.encode("ascii"), dtype=np.uint8)
        L = min(int(b.shape[0]), 15)
        w_vec.fill(0)
        w_vec[:L] = b[:L]
        w_vec[15] = np.int32(L)
        return (w_vec @ w_poly_mat)[:3] & b_mask

    for w in _VALID_SINGLE_GREETING_WORDS:
        idx = h_idx(w)
        single_bloom[idx] = np.int8(1)
        word_bloom[idx] = np.int8(1)
        target_bloom[idx] = np.int8(1)
    for w in _VALID_GREETING_TARGETS:
        target_bloom[h_idx(w)] = np.int8(1)
    for w in _VALID_MULTI_GREETING_WORDS:
        word_bloom[h_idx(w)] = np.int8(1)
    for w in _GREETING_LEAD_WORDS:
        lead_bloom[h_idx(w)[0]] = np.int8(1)
    for w in _EXTRA_S2_AGE_FRAMING_WORDS:
        collapsed = "".join(w[i] for i in range(len(w)) if i == 0 or w[i] != w[i - 1])
        s2_frame_bloom[h_idx(collapsed)[:3]] = np.int8(1)

    word_bloom.flags.writeable = False
    single_bloom.flags.writeable = False
    target_bloom.flags.writeable = False
    lead_bloom.flags.writeable = False
    s2_frame_bloom.flags.writeable = False
    return word_bloom, single_bloom, target_bloom, lead_bloom


class UserStore:
    __slots__ = ("last_intent", "msg_count", "age", "country", "gender", "bot_gender", "conv_flags", "last_ts", "write_queue", "wal_batch")

    def __init__(self):
        self.last_intent = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.msg_count = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.age = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.country = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.gender = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.bot_gender = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.conv_flags = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.last_ts = np.zeros((MAX_USERS,), dtype=np.uint32)
        self.write_queue = deque(maxlen=4096)
        self.wal_batch = np.zeros((256, 6), dtype=np.uint32)


def reset_user_session(users, uid):
    users.last_intent[uid] = np.uint8(0)
    users.msg_count[uid] = np.uint8(0)
    users.age[uid] = np.uint8(0)
    users.country[uid] = np.uint8(0)
    users.gender[uid] = np.uint8(0)
    users.conv_flags[uid] = np.uint8(0)


def handle_user_query(model, slot, users, uid, mv_payload, forward_pass_fn):
    now_s = int(time.time()) & 0xFFFFFFFF
    last_t = int(users.last_ts[uid])
    if last_t > 0 and (now_s - last_t) > SESSION_TIMEOUT_S:
        reset_user_session(users, uid)

    class_id, age_val, country_id = forward_pass_fn(model, slot, mv_payload)
    if class_id == 99:
        reset_user_session(users, uid)
        users.last_ts[uid] = np.uint32(now_s)
        return CACHED_RESET_RESPONSE

    flags = int(users.conv_flags[uid])
    prev_age = int(users.age[uid])
    prev_country = int(users.country[uid])
    prev_gender = int(users.gender[uid])
    msg_cnt = (int(users.msg_count[uid]) + 1) & 255

    users.last_intent[uid] = np.uint8(class_id)
    users.msg_count[uid] = np.uint8(msg_cnt)
    users.last_ts[uid] = np.uint32(now_s)
    
    if age_val >= 18:
        users.age[uid] = np.uint8(age_val)
    if country_id > 0:
        users.country[uid] = np.uint8(country_id)

    # Initialize BOT_PERSONA_GENDER to Female (2) if blank for this session
    if int(users.bot_gender[uid]) == 0:
        users.bot_gender[uid] = np.uint8(2)

    bot_g = int(users.bot_gender[uid])

    if class_id in (12, 13, 14):
        if class_id == 12:
            users.gender[uid] = np.uint8(2)
        elif class_id == 13:
            users.gender[uid] = np.uint8(1)
        elif class_id == 14:
            has_f = bool(np.any(slot.raw_u8 == 102))  # 'f' is 102
            users.gender[uid] = np.uint8(2 if has_f else 1)

    users.write_queue.append((uid, class_id, age_val, country_id, int(users.gender[uid]), now_s))
    rot8 = (uid + msg_cnt) & 3  # Rotate modulo size of mini-pools

    # --- Step 1: Greeting ---
    if class_id == 1:
        already_in_conv = (flags & FLAG_GREETED) != 0 or prev_age > 0 or prev_country > 0 or prev_gender > 0
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        return CACHED_REPEAT_GREETING_POOL[rot8] if already_in_conv else CACHED_GREETING_POOL[(uid + msg_cnt) & 31]

    # --- Step 2: Age & Country ---
    if class_id == 2:
        already_asked_age = (flags & FLAG_ASKED_AGE) != 0
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED | FLAG_ASKED_AGE)
        return CACHED_REPEAT_ASK_AGE_POOL[rot8] if already_asked_age else CACHED_ASK_AGE_POOL[rot8]
    if class_id == 3:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        if prev_age == age_val:
            return CACHED_REPEAT_AGE_POOL[rot8]
        if prev_age > 0 and prev_age != age_val:
            return CACHED_CHANGED_AGE_POOL[rot8]
        return CACHED_AGE_COMPLETE_POOL[rot8] if prev_country > 0 else CACHED_TELL_AGE_POOL[rot8]
    if class_id == 4:
        already_asked_ctry = (flags & FLAG_ASKED_COUNTRY) != 0
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED | FLAG_ASKED_COUNTRY)
        return CACHED_REPEAT_ASK_COUNTRY_POOL[rot8] if already_asked_ctry else CACHED_ASK_COUNTRY_POOL[rot8]
    if class_id == 5:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        if prev_country == country_id:
            return CACHED_REPEAT_COUNTRY_POOL[rot8]
        if prev_country > 0 and prev_country != country_id:
            return CACHED_CHANGED_COUNTRY_POOL[rot8]
        return CACHED_COUNTRY_COMPLETE_POOL[rot8] if prev_age >= 18 else CACHED_TELL_COUNTRY_POOL[rot8]
    if class_id == 6:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        return CACHED_COMBINED_AGE_COUNTRY_POOL[rot8]

    # --- Step 3: HRU / Status ---
    if class_id == 8:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED | FLAG_ASKED_HRU)
        return CACHED_ASK_HRU_POOL[rot8]

    # --- Step 4: Gender Yes/No Router ---
    if class_id == 9:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED | FLAG_ASKED_GENDER)
        return CACHED_GENDER_CHOICE_F_POOL[rot8] if bot_g == 2 else CACHED_GENDER_CHOICE_M_POOL[rot8]
    if class_id == 10:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED | FLAG_ASKED_GENDER)
        return CACHED_GENDER_YES_F_POOL[rot8] if bot_g == 2 else CACHED_GENDER_NO_M_POOL[rot8]
    if class_id == 11:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED | FLAG_ASKED_GENDER)
        return CACHED_GENDER_NO_F_POOL[rot8] if bot_g == 2 else CACHED_GENDER_YES_M_POOL[rot8]
    if class_id in (12, 13):
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        if prev_age > 0 and prev_country > 0:
            return CACHED_PROFILE_COMPLETE_POOL[rot8]
        if prev_age > 0:
            return CACHED_TELL_GENDER_ASK_COUNTRY_POOL[rot8]
        return CACHED_TELL_GENDER_POOL[rot8]
    if class_id == 14:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        if prev_country > 0:
            return CACHED_PROFILE_COMPLETE_POOL[rot8]
        return CACHED_TELL_AGE_GENDER_POOL[rot8]

    if class_id == 7:
        return CACHED_UNDERAGE_RESPONSE
    return CACHED_REJECT_RESPONSE


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