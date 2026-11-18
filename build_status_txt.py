#!/usr/bin/env python3
# FILE: build_status_txt.py
# PURPOSE: Build Step 6 Status & Activity Reply datasets (Classes 16, 17, 18) with 0 collisions against Steps 1-5
# RULES: Enforces Rules 1-11 (Zero float64/int64, strict 3-bucket disjoint isolation, zero placeholders)

import logging
import os
import re
import warnings
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CLEAN_STATUS_RE = re.compile(r"^[a-z\s'!?.,]+$")
SPACE_RE = re.compile(r"\s+")

# Strictly blocks Step 1 greetings ("good morning"), Step 2 age/country, Step 4 gender, and Class 0 traps
FORBIDDEN_STATUS_WORDS = frozenset((
    "morning", "mornin", "afternoon", "afternon", "evening", "day", "night", "gn", "luck", "job",
    "meet", "see", "welcome", "hello", "aloha", "howdy", "greetings", "salutations",
    "old", "olds", "age", "from", "country", "live", "living", "where", "why", "who",
    "m", "f", "w", "male", "maale", "mael", "female", "famle", "famel", "femael", "femle",
    "boy", "boi", "girl", "gril", "gurl", "man", "woman", "womman", "guy", "lady", "dude",
    "gender", "gander", "gnder", "sex", "bi", "single",
    "tun", "tn", "tunis", "tunisia", "tuinsi", "usa", "uk", "india", "ind", "handi",
    "france", "germany", "canada", "bye", "goodbye", "thanks", "thank",
))

# Tokens that strictly mark a status line as Class 18 (Status + Ask-Back)
ASK_BACK_TOKENS = frozenset(("u", "you", "wbu", "hbu", "nmu", "hru", "hyd", "wyd", "wud", "about"))

# 1. Class 16 (S6 head class 1): Positive Moods & Active/Chill Activities ("im good", "just chilling in bed", "im chat")
POSITIVE_SUBJECT_PREFIXES = (
    "", "im ", "i'm ", "i am ", "lm ", "l'm ", "iam ",
    "hi im ", "hey im ", "hello im ", "yo im ", "hi i'm ", "hey i'm ", "hi lm ", "hey lm ",
)

POSITIVE_MOOD_CORES = (
    "good", "doing good", "doin good", "diong good", "great", "doing great", "doin great",
    "fine", "doing fine", "all good", "pretty good", "very good", "so good",
    "alright", "all right", "doing alright", "well", "doing well",
    "ok", "okay", "doing ok", "doing okay", "feeling good", "feeling great", "not bad",
    "awesome", "amazing", "fantastic", "happy",
)

POSITIVE_ACTIVITY_CORES = (
    "chat", "chatting", "chatin", "chating", "just chatting", "just chating", "just chat",
    "chatting here", "talking", "just talking",
    "chilling", "chillin", "chiling", "just chilling", "just chillin", "just chiling",
    "chilling in bed", "chillin in bed", "just chilling in bed", "just chillin in bed",
    "chilling at home", "chillin at home", "just chilling at home", "chilling here",
    "relaxing", "just relaxing", "relaxing in bed", "relaxing at home", "resting", "just resting",
    "in bed", "just in bed", "laying in bed", "lying in bed", "just laying in bed", "laying down",
    "at home", "just at home", "staying home", "home",
    "working", "just working", "at work", "busy", "a bit busy",
    "studying", "just studying", "reading", "just reading",
    "listening to music", "listening music", "watching tv", "watching a movie", "watching movies",
    "watching youtube", "watching netflix", "gaming", "playing games", "playing a game",
    "scrolling", "just scrolling", "on my phone", "eating", "just eating", "cooking", "drinking coffee",
)

# 2. Class 17 (S6 head class 2): Negative, Bored, Tired, or "Nothing Much" States ("lm borad", "nothing much, just bored")
NEGATIVE_BORED_TIRED_CORES = (
    "bored", "borad", "boerd", "borde", "bord", "broed",
    "just bored", "just borad", "just boerd", "just borde",
    "so bored", "so borad", "very bored", "super bored", "hella bored", "feeling bored", "feeling borad",
    "bored in bed", "borad in bed", "bored at home", "borad at home", "bored here",
    "tired", "tierd", "tird", "just tired", "so tired", "very tired", "super tired", "feeling tired",
    "sleepy", "so sleepy", "just sleepy", "exhausted", "cant sleep", "can't sleep",
    "sad", "just sad", "so sad", "feeling sad", "lonely", "so lonely", "feeling lonely",
    "bad", "doing bad", "not good", "not doing good", "not so good", "sick", "stressed",
)

NEGATIVE_NOTHING_STANDALONE = (
    "nothing", "nothin", "nth", "nada",
    "nothing much", "nothin much", "nothing much really", "nothing really", "nothing special",
    "not much", "not much really", "nm", "not doing much", "not doin much", "not doing anything",
    "doing nothing", "doin nothing", "just nothing", "same old",
    "nothing much just bored", "nothing much just borad", "not much just bored", "not much just borad",
    "nothing just bored", "nothing just borad", "nothing much just chilling", "not much just chilling",
    "nothing much just in bed", "not much just in bed", "chilling and bored", "in bed and bored",
    "in bed bored", "just in bed bored", "laying in bed bored",
)

# 3. Class 18 (S6 head class 3): Status / Activity + Ask-Back ("im good u?", "lm borad u?", "nothing much wbu?")
ASK_BACK_BASES = (
    "im good", "i'm good", "lm good", "good", "doing good", "im doing good",
    "im great", "great", "doing great", "im fine", "fine", "im ok", "all good", "not bad",
    "im chat", "im chatting", "chatting", "just chatting",
    "chilling", "chillin", "just chilling", "im chilling", "in bed", "just in bed",
    "chilling in bed", "just chilling in bed", "relaxing", "just relaxing", "at home", "working",
    "bored", "borad", "boerd", "im bored", "i'm bored", "lm bored", "lm borad", "im borad", "just bored", "so bored",
    "nothing", "nothing much", "not much", "nm", "doing nothing", "not doing much",
    "nothing much just bored", "not much just bored",
    "tired", "im tired", "lm tired", "sleepy", "im sleepy", "lonely", "im lonely",
)

ASK_BACK_TAILS = (
    "u", "u?", "you", "you?", "wbu", "wbu?", "hbu", "hbu?",
    "and u", "and u?", "and you", "and you?",
    "how about u", "how about u?", "how about you", "how about you?",
    "what about u", "what about u?", "what about you", "what about you?",
    "hru", "hru?", "wyd", "wyd?",
)


def _clean_status_line(text, allow_ask_back=False):
    s = SPACE_RE.sub(" ", text.strip().lower())
    if not s or len(s) > 42 or not CLEAN_STATUS_RE.match(s):
        return ""
    words = [w.strip("'!?.,") for w in s.split() if w.strip("'!?.,")]
    if not words or len(words) > 6:
        return ""
    if any(w in FORBIDDEN_STATUS_WORDS for w in words):
        return ""
    has_ab = any(w in ASK_BACK_TOKENS for w in words)
    if allow_ask_back and not has_ab:
        return ""
    if not allow_ask_back and has_ab:
        return ""
    return s


def _write_u8_file(path, lines_set):
    sorted_lines = sorted(lines_set)
    payload = ("\n".join(sorted_lines) + "\n").encode("ascii", errors="ignore")
    arr_u8 = np.frombuffer(payload, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(arr_u8.tobytes())
    return np.int32(len(sorted_lines)), np.int32(arr_u8.nbytes)


def build_all_status_files(
    pos_path="status_positive.txt",
    neg_status_path="status_negative.txt",
    ask_back_path="status_ask_back.txt",
    step1_neg_path="negatives.txt",
):
    pos_set = set()
    neg_status_set = set()
    ask_back_set = set()

    # Build Class 16: Positive Moods & Active/Chill States
    for pfx in POSITIVE_SUBJECT_PREFIXES:
        for core in POSITIVE_MOOD_CORES + POSITIVE_ACTIVITY_CORES:
            if pfx and core in ("all good", "not bad"):
                continue
            base = f"{pfx}{core}".strip()
            for sfx in ("", " rn", " now", " today", " thanks", " thx", " ty", " lol", " haha"):
                for end in ("", "!"):
                    c = _clean_status_line(f"{base}{sfx}{end}", allow_ask_back=False)
                    if c:
                        pos_set.add(c)

    # Build Class 17: Negative, Bored, Tired, or "Nothing Much" States
    for pfx in POSITIVE_SUBJECT_PREFIXES:
        for core in NEGATIVE_BORED_TIRED_CORES:
            base = f"{pfx}{core}".strip()
            for sfx in ("", " rn", " now", " today", " lol", " haha"):
                for end in ("", "!"):
                    c = _clean_status_line(f"{base}{sfx}{end}", allow_ask_back=False)
                    if c:
                        neg_status_set.add(c)

    for core in NEGATIVE_NOTHING_STANDALONE:
        for pfx in ("", "hi ", "hey ", "yo "):
            base = f"{pfx}{core}".strip()
            for sfx in ("", " rn", " lol", " haha"):
                for end in ("", "!"):
                    c = _clean_status_line(f"{base}{sfx}{end}", allow_ask_back=False)
                    if c:
                        neg_status_set.add(c)

    # Build Class 18: Status + Ask-Back ("im good u?", "lm borad u?", "nmu")
    for ab_short in ("nmu", "nmu?", "nmu!", "nm u", "nm u?", "nm wbu", "nm wbu?"):
        c = _clean_status_line(ab_short, allow_ask_back=True)
        if c:
            ask_back_set.add(c)

    for base in ASK_BACK_BASES:
        for tail in ASK_BACK_TAILS:
            c = _clean_status_line(f"{base} {tail}", allow_ask_back=True)
            if c:
                ask_back_set.add(c)
            c_comma = _clean_status_line(f"{base}, {tail}", allow_ask_back=True)
            if c_comma:
                ask_back_set.add(c_comma)

    # Enforce strict disjoint hierarchy: Ask-Back (18) > Negative/Bored/Nothing (17) > Positive (16)
    pos_set.difference_update(neg_status_set | ask_back_set)
    neg_status_set.difference_update(ask_back_set)

    n_pos, b_pos = _write_u8_file(pos_path, pos_set)
    n_neg_s, b_neg_s = _write_u8_file(neg_status_path, neg_status_set)
    n_ab, b_ab = _write_u8_file(ask_back_path, ask_back_set)

    # Sync compact non-greeting-prefixed status phrases into negatives.txt so Step 1 rejects them
    existing_neg = set()
    if os.path.exists(step1_neg_path):
        with open(step1_neg_path, "r", encoding="ascii", errors="ignore") as fn:
            existing_neg = {line.strip().lower() for line in fn if line.strip()}
    for item in (pos_set | neg_status_set | ask_back_set):
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ", "sup ")):
            existing_neg.add(item)
    _, _ = _write_u8_file(step1_neg_path, existing_neg)

    tot_lines = np.int32(n_pos + n_neg_s + n_ab)
    tot_kb = np.float32(b_pos + b_neg_s + b_ab) / np.float32(1024.0)
    print(
        f"[CLEAN STATUS STEP 6] Wrote {int(tot_lines)} total lines ({float(tot_kb):.2f} KB): "
        f"pos={int(n_pos)} ({float(b_pos)/1024.0:.2f} KB), "
        f"neg={int(n_neg_s)} ({float(b_neg_s)/1024.0:.2f} KB), "
        f"ask_back={int(n_ab)} ({float(b_ab)/1024.0:.2f} KB)",
        flush=True,
    )
    return tot_lines, tot_kb


def generate_all_status_files():
    return build_all_status_files()


if __name__ == "__main__":
    build_all_status_files()