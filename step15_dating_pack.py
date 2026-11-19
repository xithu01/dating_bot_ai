#!/usr/bin/env python3
# FILE: step15_dating_pack.py
# PURPOSE: Unified All-in-One Step 15 Plugin (Classes 35-43 — 9 Real-World Telegram Dating Intents)
#          Powered by plugin_hub.py (Zero Repeated Engine/MLP/Vault Boilerplate!)
# RULES: Enforces Rules 1-13 (100% Pure NumPy in production, zero float64/int64, SHA-256 Vault Shield).

import os
import numpy as np

from build_step15_pack_txt import (
    C35_GOODBYE_WORDS,
    C36_WORK_STUDY_WORDS,
    C37_LANGUAGE_WORDS,
    C38_AMA_GAME_WORDS,
    C39_FLIRT_MEET_WORDS,
    C40_THANKS_SORRY_WORDS,
    C41_PING_CLARIFY_WORDS,
    C42_NSFW_RUDE_WORDS,
    C43_TIMEZONE_WORDS,
    S15_CLASS_SPECS,
    S15_FRAME_WORDS,
    STRICT_ALLOWED_S15_VOCAB,
    build_step15_pack_files,
)
from plugin_hub import (
    BLOOM_SIZE,
    DynamicStepHead,
    lookup_lut_codes,
    run_mlp_head_inplace,
    save_step_into_vault_secure,
)
from production_server_part2 import (
    FLAG_GREETED,
    _make_pool,
)

S15_CLASSES = 10  # 0=Reject, 1..9 -> Classes 35..43

ALL_S15_ANCHOR_SET = frozenset(
    C35_GOODBYE_WORDS + C36_WORK_STUDY_WORDS + C37_LANGUAGE_WORDS + C38_AMA_GAME_WORDS
    + C39_FLIRT_MEET_WORDS + C40_THANKS_SORRY_WORDS + C41_PING_CLARIFY_WORDS
    + C42_NSFW_RUDE_WORDS + C43_TIMEZONE_WORDS
)

# --- 100% HUMAN-STYLE LIVE TELEGRAM RESPONSE POOLS (8 items each, Sub-Intent Aware) ---

# Class 35: Goodbyes, Goodnight & BRB
P35_BYE = _make_pool(
    35, "false", "goodbye",
    (
        "aw bye! it was really fun chatting with u, take care :)",
        "bye! hope u have a great rest of ur day :)",
        "see ya! nice talking to u, take care!",
        "aw ok gtg soon too! bye, have a good one :)",
        "bye bye! catch u later, stay safe :)",
        "take care! it was nice meeting u here :)",
        "ok see u later! have a good day :)",
        "bye! thanks for the fun chat, take care :)",
    ),
    '["bye!","take care!"]',
)

P35_GN = _make_pool(
    35, "false", "goodnight",
    (
        "aw goodnight! sweet dreams, sleep well :)",
        "gn! hope u get some good rest tonight :)",
        "goodnight! it was nice chatting with u, sleep tight :)",
        "sweet dreams! gn, take care :)",
        "aw ok sleep well! goodnight :)",
        "gn! talk again sometime, have good dreams :)",
        "goodnight! get some cozy sleep :)",
        "nighty! sweet dreams and take care :)",
    ),
    '["gn!","sweet dreams"]',
)

P35_BRB = _make_pool(
    35, "false", "brb",
    (
        "ok np, take ur time! ill be right here :)",
        "sure thing, no rush! ill wait here :)",
        "haha ok brb is fine, just text when ur back :)",
        "no worries, take a sec! im still here :)",
        "ok ill wait :) take ur time!",
        "all good! message me when ur back :)",
        "sure np! ill be chilling here :)",
        "ok hold on is fine :) no rush!",
    ),
    '["im back!","ok thanks"]',
)

# Class 36: Work / Study / Student
P36_ASK = _make_pool(
    36, "false", "ask_work_study",
    (
        "im in college right now and work part time :) wbu, do u work or study?",
        "im a student at uni :) what about u, work or study?",
        "mostly studying in college and chilling :) wbu, are u a student or working?",
        "i study at university right now :) do u work or go to school?",
        "college student here :) what about u, do u work or study?",
        "im studying and doing some part time work :) wbu?",
        "in uni right now :) how about u, work or study?",
        "im a student :) wbu, what do u study or do?",
    ),
    '["im a student","i work","in college"]',
)

P36_TELL = _make_pool(
    36, "false", "tell_work_study",
    (
        "oh nice! im a student in college myself :) do u like it?",
        "cool! keeping busy is good :) im in uni right now too!",
        "ooh awesome! sounds like u work hard :) what do u do to relax after?",
        "nice! im studying in college too :) hows it going for u?",
        "respect! :) im a uni student myself, always nice meeting driven people!",
        "ooh cool! i feel u, balancing life is tough :) what r ur hobbies?",
        "awesome :) im in college too! do u get much free time?",
        "nice! glad to hear that :) so what do u like doing for fun?",
    ),
    '["yeah its cool","what r ur hobbies?","wyd?"]',
)

# Class 37: Language Check
P37_LANG = _make_pool(
    37, "false", "language_check",
    (
        "i mostly speak english! :) what languages do u speak?",
        "just english for me haha :) what about u, where r u from?",
        "english mostly, still learning other languages :) wbu?",
        "i only speak english right now :) what language do u speak at home?",
        "english is my main language :) do u speak more than one?",
        "haha only english here! :) i love different accents tho, where u from?",
        "mostly english :) u can teach me some words in ur language haha!",
        "english only for me :) what languages do u know?",
    ),
    '["i speak arabic","i speak french","only english"]',
)

P37_BAD_ENG = _make_pool(
    37, "false", "language_reassure",
    (
        "haha don't worry at all, ur english is totally fine! i understand u 100% :)",
        "no worries! ur english is actually really good, i understand everything :)",
        "aw don't apologize, speaking more than one language is impressive! :)",
        "ur english is great honestly! :) no need to worry at all",
        "haha i understand u perfectly! :) don't stress about english",
        "all good! ur doing great in english :) where r u from btw?",
        "don't even worry about it, i understand u 100% :) ur doing awesome!",
        "no problem at all! ur english is totally clear to me :)",
    ),
    '["thanks!","im from tunisia","haha ok"]',
)

# Class 38: Playful AMA / Truth or Dare / Tell Me About Yourself
P38_AMA = _make_pool(
    38, "false", "playful_ama",
    (
        "ooh ok my turn to ask! what's one thing that always makes u smile? :)",
        "haha alright! if u could travel anywhere right now, where would u go?",
        "ooh fun! ok tell me, are u a morning person or a night owl? :)",
        "let's do it! what's ur biggest green flag in someone u date? :)",
        "alright my question: what's the best movie or song u never get tired of?",
        "ooh ok! what's something random about u that most people don't know? :)",
        "haha love this! dogs or cats, and beach or mountains? :)",
        "ok here's one: what's ur ideal chill weekend? :)",
    ),
    '["night owl","beach and music","truth or dare"]',
)

P38_TOD = _make_pool(
    38, "false", "truth_or_dare",
    (
        "ooh truth or dare! i pick truth first haha, ask me anything :) then it's ur turn!",
        "haha im down for truth or dare! keep it clean tho ;) truth or dare for u?",
        "ooh fun! ok u go first, truth or dare? :)",
        "im in! i'll pick truth, ask me a good question :) wbu, truth or dare?",
        "haha classic game! u want truth or dare first? :)",
        "let's play! keep it chill tho haha :) truth or dare?",
        "ooh love truth or dare! ask me a truth first, then im asking u :)",
        "down! truth or dare for u first? :)",
    ),
    '["truth","dare","ask me a truth"]',
)

P38_BIO_F = _make_pool(
    38, "false", "tell_about_self",
    (
        "im a 22yo girl, chill vibe, love music, movies, late night chats and meeting cool people :) wbu?",
        "well im F, 22, love pop/rnb, netflix, going to the gym and traveling :) tell me about urself!",
        "im a pretty chill girl, love music, coffee, anime and good conversations :) what about u?",
        "22 F here, friendly and a bit shy at first, love music and funny people :) wbu?",
        "im a college student, love listening to music, watching movies and vibing :) tell me about u!",
        "well im a night owl girl who loves music, traveling and meeting new friends :) wbu?",
        "im 22 F, super chill, love good humor, music and cozy days in bed :) ur turn!",
        "just a chill 22yo girl looking for good vibes, laughs and cool chats :) tell me about urself!",
    ),
    '["u sound cool!","im 24","what music u like?"]',
)

P38_BIO_M = _make_pool(
    38, "false", "tell_about_self",
    (
        "im a 23yo guy, chill vibe, love music, gaming, gym and meeting cool people :) wbu?",
        "well im M, 23, love hiphop, movies, working out and traveling :) tell me about urself!",
        "im a pretty chill guy, love music, sports, games and good conversations :) what about u?",
        "23 M here, friendly and easygoing, love music and funny people :) wbu?",
        "im a college student, love listening to music, gaming and vibing :) tell me about u!",
        "well im a night owl guy who loves music, traveling and meeting new friends :) wbu?",
        "im 23 M, super chill, love good humor, gym and cozy late night chats :) ur turn!",
        "just a chill 23yo guy looking for good vibes, laughs and cool chats :) tell me about urself!",
    ),
    '["u sound cool!","im 22","what games u play?"]',
)

# Class 39: Flirt Escalation / Meet / Date / Affection
P39_LIKE = _make_pool(
    39, "false", "flirt_like_me",
    (
        "haha u definitely have a cute vibe so far! ;) let's keep talking and see!",
        "aw i really like talking to u! :) ur super fun to vibe with",
        "haha maybe a little ;) ur pretty smooth!",
        "u seem really sweet and cool so far :) i enjoy chatting with u!",
        "haha well im still here talking to u, so that's a good sign ;) ",
        "aw u definitely passed the vibe check :) ur cute!",
        "haha slow down charmer ;) i like ur energy tho!",
        "u seem really nice! :) definitely enjoying our chat so far",
    ),
    '["haha thanks!","ur cute too","wyd rn?"]',
)

P39_MEET = _make_pool(
    39, "false", "flirt_meet_date",
    (
        "haha slow down! let's get to know each other better first :) where r u even from?",
        "maybe one day if we live close and really click ;) let's chat more first!",
        "haha u move fast! i like getting to know someone really well before meeting :)",
        "aw that's cute, let's vibe online a bit more first tho! :) what country u in?",
        "haha maybe after we talk more and see if we're even in the same country :)",
        "sounds fun in theory, let's keep chatting here first tho! :)",
        "haha patience charmer ;) tell me more about urself first!",
        "i never rush meeting strangers haha, let's keep vibing here first! :)",
    ),
    '["haha fair","im from usa","ur cute"]',
)

P39_AFFECTION = _make_pool(
    39, "false", "flirt_affection",
    (
        "aww that's so sweet! sending u a big virtual hug :) ",
        "haha aw ur making me smile! virtual hug right back :)",
        "aww stop ur too sweet! :) u know how to make someone blush",
        "haha missed chatting with u too! :) hows ur day been?",
        "aww sending u warm vibes and a virtual hug! :)",
        "haha ur such a sweetheart :) u always know what to say!",
        "aww that's adorable :) virtual hug for u!",
        "haha smooth! ur making me blush over here :)",
    ),
    '["haha thanks","ur sweet","wyd rn?"]',
)

# Class 40: Thanks & Sorry
P40_THANKS = _make_pool(
    40, "false", "polite_thanks",
    (
        "aw no problem at all! :) ",
        "anytime! :) hows ur day going so far?",
        "haha ur welcome! :) so what u up to rn?",
        "of course! :) glad i could make u smile",
        "no worries at all :) what else u doing today?",
        "anytime! :) u seem really polite, i like that",
        "ur welcome! :) so tell me, what r ur hobbies?",
        "haha np at all! :) what u up to?",
    ),
    '["just chilling","wyd?","what r ur hobbies?"]',
)

P40_SORRY = _make_pool(
    40, "false", "polite_sorry",
    (
        "haha no worries at all! u didn't do anything wrong :)",
        "aw don't even apologize, it's totally fine! :)",
        "all good! no need to say sorry at all :)",
        "haha it's ok, don't stress about it! :)",
        "no problem at all :) we're totally cool!",
        "aw it's completely fine! :) so what were we talking about?",
        "haha my bad too if anything! all good :)",
        "don't worry about it at all :) everything is chill!",
    ),
    '["thanks!","ok cool","wyd?"]',
)

# Class 41: Ping & Clarification
P41_PING = _make_pool(
    41, "false", "ping_here",
    (
        "yep im right here! haha sorry, what's up? :)",
        "still here! :) just chilling on my phone, wbu?",
        "yeah im here! :) didn't go anywhere lol",
        "right here! :) what's on ur mind?",
        "yep still here listening! :) go ahead",
        "haha yeah im here! :) thought i lost u for a sec",
        "im here! :) so what u up to rn?",
        "still right here :) tell me what's up!",
    ),
    '["haha ok good","wyd rn?","ask me anything"]',
)

P41_CLARIFY = _make_pool(
    41, "false", "clarify_wdym",
    (
        "haha sorry if that sounded confusing! i just meant let's chat and get to know each other :)",
        "oh my bad! i just meant tell me a bit about urself, like ur hobbies or how ur day is going :)",
        "haha nothing complicated! just making conversation with u :)",
        "oops sorry, i typed that weird! basically just asking what u up to :)",
        "haha no worries, i just meant let's chill and talk! hows ur day going?",
        "oh i just meant tell me more about u! :) what do u like doing for fun?",
        "haha ignore my messy texting! just vibing and chatting with u :)",
        "sorry if i wasn't clear! :) basically just asking how things are going with u",
    ),
    '["oh ok haha","got it!","im good"]',
)

# Class 42: NSFW & Rude Boundary Shield
P42_NSFW = _make_pool(
    42, "false", "nsfw_boundary",
    (
        "haha nope, im not into that! keep it clean and respectful or im skipping :)",
        "no thanks, not here for nsfw stuff! let's keep the chat respectful",
        "nah im not doing that lol, keep it chill or find someone else",
        "nope! normal respectful chat only please, not into sexting or nudes",
        "haha hard pass on that! keep it clean if u wanna keep talking",
        "not into that at all, sorry! let's keep things normal and respectful",
        "no way lol, im just here for normal fun chats, keep it clean!",
        "nah keep it respectful please, im not here for dirty talk or nudes",
    ),
    '["sorry my bad","ok lets chat normal"]',
)

P42_RUDE = _make_pool(
    42, "false", "rude_boundary",
    (
        "wow ok chill lol, no need to be rude!",
        "hey keep it respectful, if ur not feeling the chat u can just type next :)",
        "oof bad mood today? no need to be toxic lol",
        "chill out haha, let's keep the vibes friendly!",
        "no need to be rude, we're just here to have a chill chat :)",
        "yikes ok relax! type next if u don't wanna talk nicely",
        "haha ok tough guy, keep it respectful or skip :)",
        "let's keep it friendly please, no bad vibes here!",
    ),
    '["sorry my bad","next"]',
)

# Class 43: Time Zone & Weather
P43_TIME = _make_pool(
    43, "false", "timezone_late",
    (
        "haha yeah im a huge night owl lol! what time is it where u are?",
        "i always stay up way too late scrolling on my phone :) what time is it for u?",
        "time zones are crazy on here haha! is it late where u live?",
        "haha i can never sleep early! what time is it over there?",
        "just chilling and losing track of time :) is it night or day where u are?",
        "yeah late night chats are always the best vibe :) what time is it there?",
        "haha my sleep schedule is a mess lol! what time is it for u right now?",
        "always up late texting and listening to music :) is it late where u are?",
    ),
    '["its late here","its midnight","cant sleep"]',
)

P43_WEATHER = _make_pool(
    43, "false", "weather_chat",
    (
        "the weather is pretty chill here today! hows the weather where u are?",
        "it's kinda nice and cozy outside here :) is it hot or cold where u live?",
        "pretty mild weather here right now! what's it like where u are?",
        "cozy weather for staying in bed haha :) is it warm or cold there?",
        "not bad at all here today! :) do u like hot summer weather or cold cozy weather?",
        "kinda chill outside so im staying cozy inside :) hows the weather there?",
        "it's pretty decent weather here :) what's it like in ur city?",
        "nice weather today! :) is it raining or sunny where u are?",
    ),
    '["its cold here","its hot here","pretty nice"]',
)

S15_HEAD = None


def get_step15_engine(npz_path="model_weights.npz"):
    global S15_HEAD
    if S15_HEAD is None:
        S15_HEAD = DynamicStepHead("s15", S15_CLASSES, npz_path)
    return S15_HEAD


def _word_eq(words_mat, w_i, target_bytes):
    Lw = int(words_mat[w_i, 15])
    if Lw != len(target_bytes):
        return False
    for k, b_val in enumerate(target_bytes):
        if int(words_mat[w_i, k]) != b_val:
            return False
    return True


def forward_step15(model, slot, n_words, has_q, age_val, country_id):
    head = S15_HEAD
    if head is None or not head.enabled or age_val != 0 or country_id != 0:
        return 0, 0, 0

    codes = lookup_lut_codes(slot, head, n_words)
    if not bool(np.all(codes > 0)):
        return 0, 0, 0

    # Must contain at least one Step 15 category anchor (Codes 1..9)
    if not bool(np.any((codes >= 1) & (codes <= 9))):
        return 0, 0, 0

    out_l = run_mlp_head_inplace(model, slot, head, n_words, has_q)
    if out_l is None:
        return 0, 0, 0

    s0 = float(out_l[0, 0])
    best_k = int(np.argmax(out_l[0, 1:])) + 1
    if float(out_l[0, best_k]) <= s0 + 0.15:
        return 0, 0, 0

    # Priority Safety Override: Explicit NSFW/Rude anchor (Code 8) always routes to Class 42 Boundary Shield!
    if bool(np.any(codes == 8)):
        cid = 42
    else:
        cid = 34 + best_k  # Maps 1..9 -> Classes 35..43

    # Extract Smart Sub-Intent aux_id for 100% Human-Like Replies
    wm = slot.words_mat
    aux_id = 1
    if cid == 35:
        for i in range(n_words):
            if _word_eq(wm, i, b"gn") or _word_eq(wm, i, b"godnight") or _word_eq(wm, i, b"night") or _word_eq(wm, i, b"slep") or _word_eq(wm, i, b"dreams") or _word_eq(wm, i, b"nighty") or _word_eq(wm, i, b"gnite"):
                aux_id = 2
                break
            if _word_eq(wm, i, b"brb") or _word_eq(wm, i, b"sec") or _word_eq(wm, i, b"min") or _word_eq(wm, i, b"hold") or _word_eq(wm, i, b"back") or _word_eq(wm, i, b"moment"):
                aux_id = 3
                break
    elif cid == 36:
        is_q = has_q or _word_eq(wm, 0, b"do") or _word_eq(wm, 0, b"are") or _word_eq(wm, 0, b"r") or _word_eq(wm, 0, b"what") or _word_eq(wm, 0, b"whats") or _word_eq(wm, 0, b"where") or _word_eq(wm, 0, b"u")
        aux_id = 1 if is_q else 2
    elif cid == 37:
        for i in range(n_words):
            if _word_eq(wm, i, b"bad") or _word_eq(wm, i, b"fluent"):
                aux_id = 2
                break
    elif cid == 38:
        for i in range(n_words):
            if _word_eq(wm, i, b"truth") or _word_eq(wm, i, b"dare"):
                aux_id = 2
                break
            if _word_eq(wm, i, b"yourself") or _word_eq(wm, i, b"urself") or _word_eq(wm, i, b"secret"):
                aux_id = 3
                break
    elif cid == 39:
        for i in range(n_words):
            if _word_eq(wm, i, b"met") or _word_eq(wm, i, b"date") or _word_eq(wm, i, b"come") or _word_eq(wm, i, b"hangout"):
                aux_id = 2
                break
            if _word_eq(wm, i, b"hug") or _word_eq(wm, i, b"kis") or _word_eq(wm, i, b"cudle") or _word_eq(wm, i, b"mis") or _word_eq(wm, i, b"love"):
                aux_id = 3
                break
    elif cid == 40:
        for i in range(n_words):
            if _word_eq(wm, i, b"sory") or _word_eq(wm, i, b"sry") or _word_eq(wm, i, b"bad") or _word_eq(wm, i, b"ops") or _word_eq(wm, i, b"forgive") or _word_eq(wm, i, b"apologize") or _word_eq(wm, i, b"excuse"):
                aux_id = 2
                break
    elif cid == 41:
        for i in range(n_words):
            if _word_eq(wm, i, b"wdym") or _word_eq(wm, i, b"mean") or _word_eq(wm, i, b"huh") or _word_eq(wm, i, b"understand") or _word_eq(wm, i, b"confused") or _word_eq(wm, i, b"explain") or _word_eq(wm, i, b"repeat"):
                aux_id = 2
                break
    elif cid == 42:
        for i in range(n_words):
            if _word_eq(wm, i, b"fuck") or _word_eq(wm, i, b"fk") or _word_eq(wm, i, b"fck") or _word_eq(wm, i, b"bitch") or _word_eq(wm, i, b"stupid") or _word_eq(wm, i, b"idiot") or _word_eq(wm, i, b"dumb") or _word_eq(wm, i, b"ugly") or _word_eq(wm, i, b"shut") or _word_eq(wm, i, b"stfu") or _word_eq(wm, i, b"loser") or _word_eq(wm, i, b"boring"):
                aux_id = 2
                break
    elif cid == 43:
        for i in range(n_words):
            if _word_eq(wm, i, b"weather") or _word_eq(wm, i, b"cold") or _word_eq(wm, i, b"hot") or _word_eq(wm, i, b"raining") or _word_eq(wm, i, b"rain") or _word_eq(wm, i, b"snow"):
                aux_id = 2
                break

    return cid, 0, aux_id


def route_step15(class_id, aux_id, users, uid, flags, prev_age, prev_country, bot_g, rot8):
    users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
    if class_id == 35:
        return P35_GN[rot8] if aux_id == 2 else (P35_BRB[rot8] if aux_id == 3 else P35_BYE[rot8])
    if class_id == 36:
        return P36_ASK[rot8] if aux_id == 1 else P36_TELL[rot8]
    if class_id == 37:
        return P37_BAD_ENG[rot8] if aux_id == 2 else P37_LANG[rot8]
    if class_id == 38:
        if aux_id == 2:
            return P38_TOD[rot8]
        if aux_id == 3:
            return P38_BIO_F[rot8] if bot_g == 2 else P38_BIO_M[rot8]
        return P38_AMA[rot8]
    if class_id == 39:
        return P39_MEET[rot8] if aux_id == 2 else (P39_AFFECTION[rot8] if aux_id == 3 else P39_LIKE[rot8])
    if class_id == 40:
        return P40_SORRY[rot8] if aux_id == 2 else P40_THANKS[rot8]
    if class_id == 41:
        return P41_CLARIFY[rot8] if aux_id == 2 else P41_PING[rot8]
    if class_id == 42:
        return P42_RUDE[rot8] if aux_id == 2 else P42_NSFW[rot8]
    if class_id == 43:
        return P43_WEATHER[rot8] if aux_id == 2 else P43_TIME[rot8]
    return None


def train_step15_into_vault(npz_path="model_weights.npz"):
    import torch
    import offline_train as ot

    if any(not os.path.exists(fn) for _, fn, _ in S15_CLASS_SPECS):
        build_step15_pack_files()

    ram_arena = ot.verify_hardware_and_lock_ram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wp_mat = ot._get_wp_mat()

    s15_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    s15_lut_a = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    s15_lut_b = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    b_mask = np.int32(BLOOM_SIZE - 1)
    w_vec = np.zeros((16,), dtype=np.int32)

    def hash_word(w_str):
        b = np.frombuffer(w_str.encode("ascii", errors="ignore"), dtype=np.uint8)
        L = min(int(b.shape[0]), 15)
        w_vec.fill(0)
        w_vec[:L] = b[:L]
        w_vec[15] = np.int32(L)
        return (w_vec @ wp_mat) & b_mask

    # Populate 10-Code O(1) LUTs: Framing (10) first, Categories (1..9) next, NSFW Shield (8) LAST for max priority
    for code_val, word_group in (
        (10, S15_FRAME_WORDS),
        (9, C43_TIMEZONE_WORDS),
        (7, tuple(w for w in C41_PING_CLARIFY_WORDS if w != "hello")),
        (6, C40_THANKS_SORRY_WORDS),
        (5, C39_FLIRT_MEET_WORDS),
        (4, C38_AMA_GAME_WORDS),
        (3, C37_LANGUAGE_WORDS),
        (2, C36_WORK_STUDY_WORDS),
        (1, C35_GOODBYE_WORDS),
        (8, C42_NSFW_RUDE_WORDS),
    ):
        for w in word_group:
            h4 = hash_word(ot._collapse_word(w))
            s15_lut_a[h4[0]] = np.uint8(code_val)
            s15_lut_b[h4[1]] = np.uint8(code_val)
            s15_bloom[h4[:3]] = np.int8(1)

    sample_map = {}
    old_step_files = (
        "greetings.txt", "ages.txt", "countries.txt", "hru.txt", "wyd.txt",
        "gender_ask_choice.txt", "status_positive.txt", "status_negative.txt",
        "compliments_flirt.txt", "reactions_ack.txt",
        "looking_for_ask.txt", "looking_for_state.txt",
        "name_ask.txt", "hobbies_ask.txt", "hobbies_tell.txt",
        "rel_status_ask.txt", "rel_status_tell.txt",
        "private_social_ask.txt", "yes_no_maybe.txt",
        "pic_photo_ask.txt", "pic_photo_offer.txt",
        "bot_check_is_bot.txt", "bot_check_is_real.txt",
    )
    for gf in old_step_files:
        if os.path.exists(gf):
            with open(gf, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and len(sample_map) < 14000:
                        toks = [w.strip("'!?.,/") for w in s.split()]
                        if not any(w in ALL_S15_ANCHOR_SET for w in toks):
                            sample_map[s] = (0, np.float32(2.0))

    s15_cnt = 0
    for cid, fn, builtins in S15_CLASS_SPECS:
        head_cls = cid - 34  # 35..43 -> 1..9
        for b_line in builtins:
            s = " ".join(b_line.strip().lower().split())
            if s:
                sample_map[s] = (head_cls, np.float32(36.0))
                s15_cnt += 1
        if os.path.exists(fn):
            with open(fn, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and ot._is_whitelist_clean(s, STRICT_ALLOWED_S15_VOCAB):
                        sample_map[s] = (head_cls, np.float32(24.0))
                        s15_cnt += 1

    # Strict Anti-Collision Negative Traps (Protects Steps 1-14 & Class 0)
    for trap in (
        "what", "what?", "what is this", "what do you want", "what do you do",
        "why", "why?", "why not", "who", "who?", "who are you", "who r u",
        "hi", "hey", "hello", "welcome", "hi there", "hey there", "hello there",
        "nice to meet u", "good to see you", "good morning", "whats up",
        "working", "just working", "at work", "studying", "just studying",
        "im sleepy", "sleepy", "cant sleep", "in bed", "all good", "not bad", "im ok", "im fine",
        "i like u", "i like you", "ur cute", "u look good", "same", "sam", "haha", "lol",
        "french", "spanish", "german", "italian", "turkish", "hindi", "english",
        "what games do u play", "playing games", "your sex", "what sex are you",
        "looking for a gf", "a gf", "a bf", "dating", "are u single", "are u a bot",
        "are u real", "can we talk private", "send pic", "whats ur name", "hyw", "iphey", "m bi",
    ):
        sample_map[trap] = (0, np.float32(65.0))

    q_emb, q_w1, q_w2, n = ot._train_word_mlp_head(sample_map, S15_CLASSES, 32, "STEP 15 PACK", device, wp_mat, use_qmark=True)
    del ram_arena

    # Rule 12 Shared Cryptographic SHA-256 Vault Lock via plugin_hub.py!
    save_step_into_vault_secure(
        npz_path,
        "s15",
        {
            "s15_emb_q": q_emb[0], "s15_emb_s": q_emb[1],
            "s15_w1_q": q_w1[0], "s15_w1_s": q_w1[1],
            "s15_w2_q": q_w2[0], "s15_w2_s": q_w2[1],
            "s15_bloom_q": s15_bloom, "s15_lut_a": s15_lut_a, "s15_lut_b": s15_lut_b,
        },
        f"STEP 15 PACK ({s15_cnt} lines, {n} total)",
    )


if __name__ == "__main__":
    train_step15_into_vault("model_weights.npz")