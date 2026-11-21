#!/usr/bin/env python3
# FILE: offline_spec_06_step18_behavior.py
# PURPOSE: Rule 15 Offline Spec Module 6 — Step 18 Human Behavior NLU Pack (Classes 55-61, Leaves 96-102)
#          Adds Pure '?', Insistence, State/City Ask, NTMU variants, Dry Tokens, Horny & Secret Ask
#          Auto-discovered by offline_fsm_compiler.py & offline_train.py — ZERO regression on Steps 1-17!
# RULES: Enforces Rules 1-15 (<= 220 lines, zero float64/int64, power-of-2 pool lengths, zero old-spec collisions)

from offline_fsm_compiler import F_GREETED, F_ZERO

POOLS = {
    # Class 55 (Leaf 96): Pure Question Marks ("?", "??", "???")
    "s18_qmark": (55, "false", "pure_qmark", (
        "Huh?", "Wdym?", "Wait what lol", "What do u mean?",
        "I dont get it haha", "How so?", "What??", "Use words lol",
    ), '["wdym?","nothing","wyd?"]'),

    # Class 56 (Leaf 97): User Insistence ("you didnt answer me", "answer my question")
    "s18_insist": (56, "false", "user_insist", (
        "Oh sry haha, what did u ask?", "My bad lol, ask me again!",
        "What didn't I answer?", "Wait what did you ask?",
        "Wdym? I did answer haha", "Oh haha my bad, tell me again",
        "Chill lol, what question?", "Huh? What question did I miss?",
    ), '["where u from?","how old r u?","nvm"]'),

    # Class 57 (Leaf 98): Ask Bot's State or City ("state?", "which city?", "where in india")
    "s18_state": (57, "false", "ask_state_city", (
        "It's a secret haha", "Not sharing my city on here",
        "Guess lol", "Rather not say my city",
        "Take a guess!", "Secret for now haha, wbu?",
        "Not sharing my exact city here", "Guess first haha",
    ), '["haha fair","where u from?","ok"]'),

    # Class 58 (Leaf 99): Nice to Meet You ("glad to meet u", "nice meeting u", "nice to meet u too")
    "s18_ntmu": (58, "false", "nice_to_meet_you", (
        "Nice to meet u too!", "nice to meet u!",
        "Aww, nice to meet u!", "Nice to meet u too :)",
        "Nice to meet u haha!", "Glad to meet u too!",
        "Nice meeting u too :)", "Aww nice to meet u too!",
    ), '["how r u?","where u from?","im 22"]'),

    # Class 59 (Leaf 100): Pure Dry / Minimal Replies ("hmm", "umm", "idc", "meh", "ohh")
    "s18_dry": (59, "false", "short_dry_reply", (
        "Anything else? lol", "Ohh okay, and?",
        "Haha thats all?", "Tell me a bit more",
        "You're quiet haha", "Not much to say?",
        "I'm listening :)", "Okay then haha",
    ), '["just chilling","wyd?","ask me anything"]'),

    # Class 60 (Leaf 101): Horny / Sexual Check ("horny", "h0rny", "horny wbu")
    "s18_horny_f": (60, "false", "horny_prompt", (
        "Not really", "A little maybe", "Not rn", "Depends",
        "Not in the mood", "Not really haha", "Not rn lol", "Nope not in the mood",
    ), '["ok sorry","lets just chat","wyd?"]'),
    "s18_horny_m": (60, "false", "horny_prompt", (
        "Im a man", "Im into females", "Im into girls", "Im a guy bro",
        "Bro im M", "Not into guys", "Im a guy lol", "Nah im M bro",
    ), '["my bad","ok","next"]'),

    # Class 61 (Leaf 102): Ask Secret / Intimate ("tell me a secret", "whats ur secret")
    "s18_secret_f": (61, "false", "ask_secret_intimate", (
        "I asked you first", "It's a secret", "You tell me first", "You first",
        "Rather not say", "Let's leave that one", "Secret haha, u first", "Not telling yet lol",
    ), '["ok ill go first","haha fair","truth or dare"]'),
    "s18_secret_m": (61, "false", "ask_secret_intimate", (
        "You tell me first", "It's a secret", "Not telling", "You first",
        "Rather not say", "Let's leave that one", "Secret lol, u first", "Not telling yet",
    ), '["ok ill go first","haha fair","truth or dare"]'),
}

LEAF_BINDINGS = (
    (96, 55, "s18_qmark", {"set_f": F_ZERO}),
    (97, 56, "s18_insist", {"set_f": F_GREETED}),
    (98, 57, "s18_state", {"set_f": F_GREETED}),
    (99, 58, "s18_ntmu", {"set_f": F_GREETED}),
    (100, 59, "s18_dry", {"set_f": F_GREETED}),
    (101, 60, "s18_horny_f", {"p_m": "s18_horny_m", "set_f": F_GREETED}),
    (102, 61, "s18_secret_f", {"p_m": "s18_secret_m", "set_f": F_GREETED}),
)

LEAF_STEP_BITS = {96: 28, 97: 28, 98: 28, 99: 29, 100: 29, 101: 29, 102: 29}

ASKBACK_REDIRECTS = (
    (99, 99),
    (101, 101),
    (102, 102),
)

CORE_LEAF_SEEDS = {
    97: (
        "you didnt answer me", "you don't answer me", "u didnt answer me", "u didn't answer me",
        "u didnt reply", "u didn't reply", "you didnt reply", "why wont you tell me",
        "why won't you tell me", "why wont u tell me", "answer my question", "please answer my question",
        "answer me", "answer first", "reply to me", "you ignored me", "u ignored me",
        "you dodged my question", "u dodged my question", "i asked you first", "i asked u first",
        "tell me yours", "what about my question", "and mine?", "dont ignore me", "stop ignoring me",
    ),
    98: (
        "state", "state?", "city", "city?", "which state", "which state?", "what state", "what state?",
        "which city", "which city?", "what city", "what city?", "which part", "which part?",
        "what province", "what province?", "ur state", "ur state?", "your state?",
        "ur city", "ur city?", "your city?", "where in indonesia", "where in india",
        "where in canada", "where in usa", "where in uk",
    ),
    99: (
        "nice to meet ya", "glad to meet u", "glad to meet you", "pleasure meeting you",
        "pleasure meeting u", "nice meeting u", "nice meeting you", "good meeting u",
        "happy to meet u", "happy to meet you", "nice to meet u too", "nice to meet you too",
        "glad to meet u too",
    ),
    100: (
        "hm", "hmm", "hmmm", "um", "umm", "ummm", "idc", "oko", "gt", "mhm", "meh",
        "uh", "uhh", "oh", "ohh", "ah", "ahh", "oh ok", "oh okay", "ah ok", "ah okay",
    ),
    101: (
        "horny", "horny?", "im horny", "i'm horny", "lm horny", "so horny", "very horny",
        "feeling horny", "horny u?", "horny wbu", "horny wbu?", "horny hbu?",
        "are u horny", "are u horny?", "are you horny", "r u horny", "u horny?", "u horny",
    ),
    102: (
        "tell me a secret", "tell me your secret", "tell me ur secret", "whats ur secret",
        "what is your secret", "got any secrets", "got any secrets?", "any secrets?",
        "share a secret", "do u have a secret", "tell me something secret",
    ),
}

ZERO_FAULT_SUITE = (
    (b"?", 55, 96), (b"??", 55, 96), (b"???", 55, 96),
    (b"you didnt answer me", 56, 97), (b"answer my question", 56, 97),
    (b"you ignored me", 56, 97), (b"i asked you first", 56, 97),
    (b"state?", 57, 98), (b"which state?", 57, 98), (b"which city?", 57, 98), (b"where in indonesia", 57, 98),
    (b"glad to meet u", 58, 99), (b"nice meeting u", 58, 99), (b"nice to meet u too", 58, 99),
    (b"hmm", 59, 100), (b"umm", 59, 100), (b"idc", 59, 100), (b"meh", 59, 100),
    (b"horny", 60, 101), (b"h0rny", 60, 101), (b"im horny", 60, 101), (b"horny wbu", 60, 101),
    (b"tell me a secret", 61, 102), (b"whats ur secret", 61, 102), (b"got any secrets?", 61, 102),
)

MULTI_TURN_SUITE = (
    ((b"im horny", b"u?"), (101, 101)),
    ((b"tell me a secret", b"u?"), (102, 102)),
    ((b"glad to meet u", b"u?"), (99, 99)),
)