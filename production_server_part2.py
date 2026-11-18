​​#!/usr/bin/env python3
# FILE: production_server_part2.py
# PURPOSE: Step 1-4 Routing Engine & Response Pools (Gender Yes/No, Double-Memory, CACHED HTTP)
# RULES: Zero PyTorch/SciPy/DB, zero float64/int64, PyArmor compatible (< 400 lines / < 18 KB)

import asyncio
import time
import numpy as np

BLOOM_SIZE = 65536
SESSION_TIMEOUT_S = 600

# Conversation Memory Bitflags (np.uint8)
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
    "yro", "yold", "o", "m", "f", "male", "famle", "famel", "femael", "femle",
    "female", "boy", "boi", "girl", "gril", "gurl", "guy", "woman", "womman", "w",
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
CACHED_REJECT_RESPONSE = _pack_reply(0, "false", "not_valid", "say hi, how r u, M or F, ur age (18+), or where u are from!", '["hey","how r u?","im 22"]')
CACHED_UNDERAGE_RESPONSE = _pack_reply(7, "false", "underage_reject", "sorry, 18+ adults only!", '["Exit"]')
CACHED_RESET_RESPONSE = _pack_reply(99, "false", "system_reset", "new chat started! say hi :)", '["hey","hi!"]')

# --- STEP 1: GREETINGS (CLASS 1 — 32 items for & 31, 8 items for repeat & 7) ---
CACHED_GREETING_POOL = _make_pool(
    1, "true", "english_greeting",
    (
        "hey", "hi", "heyy", "yo", "hey there", "hii", "sup", "hello!",
        "hey how r u", "hi! hru?", "yo whats up", "heyy :)", "hiya", "hey! hows it going",
        "hiii", "sup! hru", "hello there", "hey u", "hi there!", "yo! how are u",
        "heyyy", "whats up!", "hey! nice to meet u", "hi :) how r u", "howdy!", "hey whats up",
        "hii there", "hello :)", "yo! whats good", "hey! hru doing", "hi! hows u", "heyy! whats up",
    ),
    '["good u?","how r u?"]',
)

CACHED_REPEAT_GREETING_POOL = _make_pool(
    1, "true", "english_greeting_repeat",
    ("hey again lol", "haha hi again!", "we already said hi :)", "yo again! how r u?", "hii again haha", "already said hey lol", "haha hey x2! hru?", "hi again! whats up?"),
    '["haha im good","how r u?"]',
)

# --- STEP 2: AGE & COUNTRY (CLASS 2..6 — Strictly 8 items per pool for & 7) ---
CACHED_ASK_AGE_POOL = _make_pool(
    2, "false", "ask_age",
    ("22 u?", "im 23, how old r u?", "22! wbu?", "im 23 :)", "22 here, u?", "23! how old are u", "im 22, ur age?", "23 u?"),
    '["im 21","im 24","im 28"]',
)

CACHED_REPEAT_ASK_AGE_POOL = _make_pool(
    2, "false", "ask_age_repeat",
    ("haha i told u im 22! u?", "already said 22 lol! how old r u?", "im 22 haha, forgot already?", "told u 22 lol :) wbu?", "still 22 haha! how r u?", "i said 22 earlier lol!", "haha 22! u asked me that already :)", "22 lol! whats up?"),
    '["haha my bad","nice!"]',
)

CACHED_TELL_AGE_POOL = _make_pool(
    3, "false", "tell_age",
    ("nice! im 22, where u from?", "cool im 23! u from?", "nice age! im 22, from?", "22 here! where r u from?", "ooh nice, im 23! country?", "cool! 22 here, u from?", "nice! im 23, where u live?", "sweet, im 22! where u from?"),
    '["usa","uk","tunisia"]',
)

CACHED_AGE_COMPLETE_POOL = _make_pool(
    3, "false", "tell_age_complete",
    ("nice! im 22 too :) hows ur day going?", "cool age! im 23, what u up to today?", "sweet! 22 here :) how r u doing?", "ooh nice! im 23, hows it going?", "nice! 22 here, what u doing rn?", "cool! im 23 :) having a good day?", "awesome! im 22, whats up?", "nice age! 23 here, hows ur day?"),
    '["going great!","chilling u?"]',
)

CACHED_REPEAT_AGE_POOL = _make_pool(
    3, "false", "tell_age_repeat",
    ("haha u already said ur age! where r u from?", "ik ur age lol! what country u in?", "u told me ur age already haha! u from?", "haha yeah ik! where u from?", "already got ur age lol :) where u live?", "haha u said that! what country?", "ik lol! where r u from?", "yeah u told me haha! u from?"),
    '["usa","uk","tunisia"]',
)

CACHED_CHANGED_AGE_POOL = _make_pool(
    3, "false", "tell_age_changed",
    ("wait u said a different age before lol! where u from?", "haha changing ur age now? lol how r u", "wait didnt u say another age earlier lol?", "haha u changed ur age! which one is real?", "hold on u said a different age before lol!", "haha new age unlocked lol! whats up?", "wait u changed ur age haha!", "lol u said another age before! how r u?"),
    '["haha jk","lol my bad"]',
)

CACHED_ASK_COUNTRY_POOL = _make_pool(
    4, "false", "ask_country",
    ("from usa, u?", "uk! where u from?", "im from usa :)", "canada, wbu?", "usa here! u from?", "from uk, where r u from?", "usa! what about u?", "im in usa, u?"),
    '["nice!","me too","cool"]',
)

CACHED_REPEAT_ASK_COUNTRY_POOL = _make_pool(
    4, "false", "ask_country_repeat",
    ("haha i said usa! where u from?", "told u im from usa lol :)", "already said usa haha! wbu?", "still in usa lol! forgot already?", "i told u usa earlier haha!", "usa lol! u asked me that already :)", "haha im from usa! how r u?", "said usa before lol! whats up?"),
    '["haha my bad","cool!"]',
)

CACHED_TELL_COUNTRY_POOL = _make_pool(
    5, "false", "tell_country",
    ("ooh nice! im from usa, how old r u?", "cool place! im from uk, age?", "nice! im from usa :) ur age?", "love that! im from usa, how old r u?", "awesome! usa here, how old u?", "nice country! im from uk, age?", "cool! im from canada, how old r u?", "sweet! usa here, ur age?"),
    '["im 21","im 24","im 28"]',
)

CACHED_COUNTRY_COMPLETE_POOL = _make_pool(
    5, "false", "tell_country_complete",
    ("ooh nice! im from usa :) hows ur day going?", "cool place! im from uk, what u up to?", "love that! usa here :) how r u doing?", "awesome place! im from canada, whats up?", "nice! im from usa, what u doing rn?", "sweet! uk here :) having a good day?", "ooh cool! im from usa, hows it going?", "nice country! usa here, hows ur day?"),
    '["going great!","chilling u?"]',
)

CACHED_REPEAT_COUNTRY_POOL = _make_pool(
    5, "false", "tell_country_repeat",
    ("ik ur from there lol! how old r u?", "haha u already told me ur country! whats up?", "yeah u said that already lol! how r u?", "ik lol! nice place :) what u up to?", "haha already got ur country! how old r u?", "u told me already lol! hows it going?", "yeah ik haha! whats up?", "haha u said that! how r u doing?"),
    '["haha im good","chilling"]',
)

CACHED_CHANGED_COUNTRY_POOL = _make_pool(
    5, "false", "tell_country_changed",
    ("wait u said a different country before lol!", "haha moving countries fast lol! how r u?", "wait didnt u say another country earlier lol?", "haha traveling the world? u said another place before!", "hold on u changed ur country lol!", "haha new country unlocked! whats up?", "wait u said somewhere else before haha!", "lol changing countries now? how r u?"),
    '["haha jk","traveling lol"]',
)

CACHED_COMBINED_AGE_COUNTRY_POOL = _make_pool(
    6, "true", "tell_age_and_country",
    ("hey! nice im 22 from usa, how r u?", "hi! cool im 23 from uk, hru?", "nice to meet u! 22 from usa here :)", "heyy! im 22 from canada, hows it going?", "yo! 23 from usa here, whats up?", "hey there! im 22 from uk, how r u?", "hi! 22 from usa here, nice to meet u!", "heyy! cool, im 23 from usa, hru?"),
    '["im good u?","doing great!"]',
)

# --- STEP 3: ASK HOW ARE YOU (CLASS 8 — 8 items) ---
CACHED_ASK_HRU_POOL = _make_pool(
    8, "false", "ask_how_are_you",
    (
        "im doing great! how old r u and where u from?",
        "doing good thanks! wbu, what's ur age and country?",
        "im good! im 22 from usa, how old r u and where u from?",
        "all good here! how old are u and where r u from?",
        "pretty good! im 23 from uk, ur age and country?",
        "doing great :) how old r u and where u live?",
        "im great thanks! wbu, age and where u from?",
        "good! chilling here, how old r u and what country?",
    ),
    '["im 22","from usa","24 from uk"]',
)

# --- STEP 4: GENDER & YES/NO SYSTEM (CLASS 9..14 — Strictly 8 items per pool) ---
CACHED_GENDER_CHOICE_F_POOL = _make_pool(
    9, "false", "ask_gender_choice",
    ("im F :) u?", "im a girl! wbu?", "female here, u?", "F! what about u?", "im a girl :) u M or F?", "F here! wbu?", "im female, how about u?", "girl here :) u?"),
    '["im M","im F"]',
)

CACHED_GENDER_CHOICE_M_POOL = _make_pool(
    9, "false", "ask_gender_choice",
    ("im M :) u?", "im a guy! wbu?", "male here, u?", "M! what about u?", "im a boy :) u M or F?", "M here! wbu?", "im male, how about u?", "guy here :) u?"),
    '["im F","im M"]',
)

CACHED_GENDER_YES_F_POOL = _make_pool(
    10, "false", "ask_gender_is_female",
    ("yes im a girl! u?", "yep F here! wbu?", "yes! im female :) u?", "yeah im a girl, how old r u?", "yes F here! where u from?", "yep im a girl :) wbu?", "yes! female here, u M or F?", "yeah F! what about u?"),
    '["im M","im F","im 22"]',
)

CACHED_GENDER_NO_F_POOL = _make_pool(
    11, "false", "ask_gender_is_male",
    ("no im a girl actually! u?", "nope F here! wbu?", "no, im female :) u?", "nope im a girl lol, wbu?", "no im F! how old r u?", "nope, girl here :) u?", "no im female! where u from?", "nope F! what about u?"),
    '["im M","im F","im 22"]',
)

CACHED_GENDER_NO_M_POOL = _make_pool(
    10, "false", "ask_gender_is_female",
    ("no im a guy actually! u?", "nope M here! wbu?", "no, im male :) u?", "nope im a boy lol, wbu?", "no im M! how old r u?", "nope, guy here :) u?", "no im male! where u from?", "nope M! what about u?"),
    '["im F","im M","im 22"]',
)

CACHED_GENDER_YES_M_POOL = _make_pool(
    11, "false", "ask_gender_is_male",
    ("yes im a guy! u?", "yep M here! wbu?", "yes! im male :) u?", "yeah im a boy, how old r u?", "yes M here! where u from?", "yep im a guy :) wbu?", "yes! male here, u M or F?", "yeah M! what about u?"),
    '["im F","im M","im 22"]',
)

CACHED_TELL_GENDER_F_POOL = _make_pool(
    12, "false", "state_gender_female",
    ("nice! how old r u and where u from?", "cool! ur age and country?", "nice :) how old r u?", "sweet! age and where u from?", "ooh nice! how old r u?", "cool :) what country u from?", "nice! im 22, how old r u?", "awesome! how old are u?"),
    '["im 21","im 24","from usa"]',
)

CACHED_TELL_GENDER_M_POOL = _make_pool(
    13, "false", "state_gender_male",
    ("nice! how old r u and where u from?", "cool! ur age and country?", "nice :) how old r u?", "sweet! age and where u from?", "ooh nice! how old r u?", "cool :) what country u from?", "nice! im 22, how old r u?", "awesome! how old are u?"),
    '["im 21","im 24","from usa"]',
)

CACHED_TELL_AGE_GENDER_POOL = _make_pool(
    14, "false", "state_age_and_gender",
    ("nice! im 22, where r u from?", "cool! 22 here, what country u from?", "sweet! where u from?", "ooh nice! im 23, where u live?", "awesome! im 22, u from?", "nice age! where r u from?", "cool :) im 22, what country?", "sweet! im 23, where u from?"),
    '["usa","uk","tunisia"]',
)


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

    class_id, age_val, aux_id = forward_pass_fn(model, slot, mv_payload)
    if class_id == 99:
        reset_user_session(users, uid)
        users.last_ts[uid] = np.uint32(now_s)
        return CACHED_RESET_RESPONSE

    flags = int(users.conv_flags[uid])
    prev_age = int(users.age[uid])
    prev_country = int(users.country[uid])
    prev_gender = int(users.gender[uid])
    msg_cnt = (int(users.msg_count[uid]) + 1) & 255

    country_id = aux_id if class_id in (5, 6) else 0
    gender_id = aux_id if class_id in (12, 13, 14) else 0

    users.last_intent[uid] = np.uint8(class_id)
    users.msg_count[uid] = np.uint8(msg_cnt)
    users.last_ts[uid] = np.uint32(now_s)

    if age_val >= 18:
        users.age[uid] = np.uint8(age_val)
    if country_id > 0:
        users.country[uid] = np.uint8(country_id)
    if gender_id > 0:
        users.gender[uid] = np.uint8(gender_id)

    bot_g = int(users.bot_gender[uid])
    if bot_g == 0:
        bot_g = 2
        users.bot_gender[uid] = np.uint8(2)

    users.write_queue.append((uid, class_id, age_val, country_id, int(users.gender[uid]), now_s))
    rot8 = (uid + msg_cnt) & 7

    if class_id == 1:
        already_in_conv = (flags & FLAG_GREETED) != 0 or prev_age > 0 or prev_country > 0 or prev_gender > 0
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        return CACHED_REPEAT_GREETING_POOL[rot8] if already_in_conv else CACHED_GREETING_POOL[(uid + msg_cnt) & 31]

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

    if class_id == 7:
        return CACHED_UNDERAGE_RESPONSE

    if class_id == 8:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED | FLAG_ASKED_HRU)
        return CACHED_ASK_HRU_POOL[rot8]

    if class_id == 9:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED | FLAG_ASKED_GENDER)
        return CACHED_GENDER_CHOICE_F_POOL[rot8] if bot_g == 2 else CACHED_GENDER_CHOICE_M_POOL[rot8]

    if class_id == 10:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED | FLAG_ASKED_GENDER)
        return CACHED_GENDER_YES_F_POOL[rot8] if bot_g == 2 else CACHED_GENDER_NO_M_POOL[rot8]

    if class_id == 11:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED | FLAG_ASKED_GENDER)
        return CACHED_GENDER_NO_F_POOL[rot8] if bot_g == 2 else CACHED_GENDER_YES_M_POOL[rot8]

    if class_id == 12:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        return CACHED_AGE_COMPLETE_POOL[rot8] if prev_age >= 18 else CACHED_TELL_GENDER_F_POOL[rot8]

    if class_id == 13:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        return CACHED_AGE_COMPLETE_POOL[rot8] if prev_age >= 18 else CACHED_TELL_GENDER_M_POOL[rot8]

    if class_id == 14:
        users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
        return CACHED_AGE_COMPLETE_POOL[rot8] if prev_country > 0 else CACHED_TELL_AGE_GENDER_POOL[rot8]

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