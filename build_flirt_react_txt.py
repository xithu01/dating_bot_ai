#!/usr/bin/env python3
# FILE: build_flirt_react_txt.py
# PURPOSE: Build 100% clean compliments_flirt.txt (Class 19) & reactions_ack.txt (Class 20)
#          Strictly isolated from Steps 1-6 (Greetings, Age/Country, HRU, Gender, WYD, Status)
#          Includes optional HuggingFace dataset import with 100% strict regex & token filtering.

import logging
import os
import re
import warnings
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CLEAN_LINE_RE = re.compile(r"^[a-z\s'!?.,]+$")
SPACE_RE = re.compile(r"\s+")

# Strictly matches Class 19 (compliment_flirt) while blocking "hey cutie" (Step 1) and "im good" (Step 6)
STRICT_FLIRT_RE = re.compile(
    r"^(?:(?:hi+|he+y+|he+l+o+|yo+|a+w+|o+h+|w+o+w+|h+a+h+a+|l+o+l+|t+b+h+|n+g+l+)\s+)?"
    r"(?:"
    r"(?:y+o+u+\s+a+r+e+|y+o+u+'?r+e+|u+\s+a+r+e+|u+\s+r+|u+r+|y+a+\s+r+|"
    r"(?:y+o+u+|u+)\s+(?:s+e+e+m+|l+o+o+k+|s+o+u+n+d+))"
    r"(?:\s+(?:s+o+|v+e+r+y+|r+e+a+l+y+|s+u+p+e+r+|p+r+e+t+y+|h+e+l+a+|t+o+o+))?\s+"
    r"(?:c+u+t+e+|c+u+t+i+e+|s+w+e+e+t+|s+w+e+a+t+|n+i+c+e+|c+o+o+l+|h+o+t+|p+r+e+t+y+|"
    r"g+o+r+g+e+o+u+s+|b+e+a+u+t+i+f+u+l+|h+a+n+d+s+o+m+e+|s+e+x+y+|a+t+r+a+c+t+i+v+e+|"
    r"a+d+o+r+a+b+l+e+|l+o+v+e+l+y+|c+h+a+r+m+i+n+g+|f+u+n+y+|f+u+n+|a+w+e+s+o+m+e+|"
    r"a+m+a+z+i+n+g+|s+m+a+r+t+|k+i+n+d+|c+h+i+l+)|"
    r"(?:(?:i+|l+)\s+)?(?:r+e+a+l+y+\s+)?(?:l+i+k+e+|l+u+v+|l+o+v+e+)\s+"
    r"(?:y+o+u+|u+|y+a+|(?:t+a+l+k+i+n+g+|s+p+e+a+k+i+n+g+)\s+(?:t+o+|w+i+t+h+)\s+(?:y+o+u+|u+)|"
    r"(?:y+o+u+r+|u+r+)\s+(?:v+i+b+e+|e+n+e+r+g+y+|s+m+i+l+e+|s+t+y+l+e+))|"
    r"(?:y+o+u+|u+)\s+(?:h+a+v+e+|g+o+t+)\s+a+\s+(?:n+i+c+e+|g+o+o+d+|g+r+e+a+t+|c+u+t+e+|c+o+o+l+)\s+"
    r"(?:v+i+b+e+|s+m+i+l+e+|e+n+e+r+g+y+)|"
    r"(?:y+o+u+\s+a+r+e+|y+o+u+'?r+e+|u+\s+r+|u+r+)\s+m+y+\s+t+y+p+e+|"
    r"(?:y+o+u+|u+)\s+m+a+k+e+\s+m+e+\s+(?:s+m+i+l+e+|b+l+u+s+h+)"
    r")"
    r"(?:\s+(?:t+o+o+|t+b+h+|n+g+l+|l+o+l+|h+a+h+a+|t+h+o+))?"
    r"[\s!.]*$",
    re.IGNORECASE,
)

# Strictly matches Class 20 (short_reaction_ack) including "same" and "sam" variants,
# while blocking questions ("?") and Step 6 statuses ("im good", "im ok", "bored")
STRICT_REACT_RE = re.compile(
    r"^(?:"
    r"(?:h+a+h+a+(?:h+a+)*|h+e+h+e+(?:h+e+)*|l+o+l+|l+m+a+o+|l+m+f+a+o+|r+o+f+l+)|"
    r"(?:(?:o+h+|o+o+h+|a+h+|a+w+|h+a+h+a+|l+o+l+|y+e+a+h*|y+e+p+|y+u+p+|o+k+|o+k+a+y+)\s+)?"
    r"(?:n+i+c+e+|c+o+o+l+|s+w+e+e+t+|a+w+e+s+o+m+e+|a+m+a+z+i+n+g+|d+o+p+e+|l+i+t+|f+i+r+e+|"
    r"b+e+t+|w+o+r+d+|f+a+i+r+(?:\s+e+n+o+u+g+h+)?|s+o+u+n+d+s+\s+(?:g+o+o+d+|n+i+c+e+|c+o+o+l+|f+u+n+)|"
    r"t+h+a+t+'?s\s+(?:n+i+c+e+|c+o+o+l+|t+r+u+e+|f+a+i+r+))|"
    r"(?:(?:h+a+h+a+|l+o+l+|y+e+a+h*|y+e+p+|o+h+|a+w+)\s+)?"
    r"(?:s+a+m+e+|s+a+m+)(?:\s+(?:h+e+r+e+|t+o+\s+(?:y+o+u+|u+)|t+b+h+|f+r+|t+h+o+|l+o+l+|h+a+h+a+))?|"
    r"(?:(?:h+a+h+a+|l+o+l+|y+e+a+h*|o+h+)\s+)?m+e+\s+t+o+(?:\s+(?:l+o+l+|h+a+h+a+))?|"
    r"(?:(?:h+a+h+a+|l+o+l+|y+e+a+h*|s+o+|t+o+o+|v+e+r+y+)\s+)?"
    r"(?:t+r+u+e+|t+r+u+|f+a+c+t+s+|f+a+x+|f+r+(?:\s+f+r+)?|f+o+r+\s+r+e+a+l+|"
    r"i+k+r+|i+\s+k+n+o+w+(?:\s+r+i+g+h+t+)?|r+i+g+h+t+|r+e+a+l+|a+g+r+e+e+d*|"
    r"t+o+t+a+l+y+|d+e+f+i+n+i+t+e+l+y+|e+x+a+c+t+l+y*)"
    r"(?:\s+(?:l+o+l+|h+a+h+a+|t+h+a+t+))?|"
    r"(?:(?:h+a+h+a+|l+o+l+|o+h+|a+h+|y+e+a+h*|y+e+p+)\s+)?"
    r"(?:o+k+|o+k+a+y+|o+k+e+y+|o+k+i+e*|k+k+|a+l+r+i+g+h+t+|a+l+r+|a+i+g+h+t+|"
    r"y+e+a+h+|y+e+a+|y+e+p+|y+u+p+|y+a+s+|s+u+r+e+|o+f+\s+c+o+u+r+s+e+|o+f+c+|"
    r"a+h+a+|w+o+w+|o+m+g+)"
    r"(?:\s+(?:n+i+c+e+|c+o+o+l+|s+w+e+e+t+|b+e+t+|f+a+i+r+|l+o+l+|h+a+h+a+|t+h+e+n+))?"
    r")[\s!.]*$",
    re.IGNORECASE,
)

# Forbidden tokens that belong to Steps 1-6 or Class 0 Traps
FLIRT_REACT_FORBIDDEN_WORDS = frozenset((
    # Step 1 Greeting Exclusives
    "meet", "met", "see", "se", "morning", "mornin", "afternoon", "afternon", "evening",
    "night", "gn", "day", "gday", "long", "time", "welcome", "salutations", "greetings",
    "gretings", "howdy", "aloha", "stranger", "friend", "partner", "everyone", " luck", "job",
    # Step 2 Age & Country Exclusives
    "old", "olds", "age", "from", "country", "where", "which", "live", "living",
    "year", "years", "yr", "yrs", "yres", "yers",
    # Step 3 Ask HRU Exclusives
    "how", "hows", "how's", "hru", "hyd", "hor", "wbu", "hbu", "going", "goin",
    # Step 4 Gender Exclusives
    "m", "f", "male", "maale", "mael", "female", "famle", "famel", "femael", "femle",
    "boy", "boi", "girl", "gril", "gurl", "man", "woman", "womman", "w", "guy", "lady",
    "dude", "gender", "gander", "gnder", "sex", "feminine", "masculine",
    # Step 5 Ask WYD Exclusives
    "wyd", "wud", "whatcha", "doing", "doin", "diong", "dopin", "up", "anything",
    # Step 6 Status & Activity Exclusives
    "im", "i'm", "l'm", "lm", "iam", "i'am", "am", "feeling", "fine", "well", "happy",
    "bored", "borad", "boerd", "borde", "bord", "broed", "tired", "tierd", "tird",
    "sleepy", "exhausted", "sad", "lonely", "bad", "sick", "stressed", "nothing",
    "nothin", "nth", "nada", "nm", "nmu", "chat", "chatting", "chatin", "chating",
    "chilling", "chillin", "chiling", "relaxing", "resting", "bed", "laying", "lying",
    "home", "working", "work", "busy", "studying", "reading", "listening", "music",
    "watching", "tv", "movie", "movies", "youtube", "netflix", "gaming", "playing",
    "games", "game", "scrolling", "phone", "eating", "cooking", "drinking", "coffee",
    # Class 0 Question & Trivia Traps
    "what", "whats", "what's", "wat", "wht", "why", "who", "when", "name", "help",
    "salam", "bonjour", "salut", "marhba",
))

FLIRT_SUBJECTS = (
    "ur", "u r", "you are", "you're", "youre", "u seem", "you seem",
    "u look", "you look", "u sound", "you sound",
)
FLIRT_INTENSIFIERS = ("", "so", "very", "really", "super", "pretty", "too")
FLIRT_ADJECTIVES = (
    "cute", "cuute", "sweet", "sweat", "nice", "cool", "hot", "hott", "pretty", "prety",
    "gorgeous", "beautiful", "handsome", "sexy", "attractive", "adorable", "lovely",
    "charming", "funny", "fun", "awesome", "amazing", "smart", "kind", "chill",
)
FLIRT_DIRECT_PHRASES = (
    "i like u", "i like you", "i really like u", "i really like you", "like u", "like you",
    "i like talking to u", "i like talking to you", "like talking to u", "i love talking to u",
    "i like ur vibe", "i love ur vibe", "love ur vibe", "love your vibe", "like ur vibe",
    "i like ur energy", "i like ur smile", "i love ur smile", "u have a nice vibe",
    "you have a nice vibe", "u got a good vibe", "u have a cute smile", "ur my type",
    "you are my type", "u r my type", "you're my type", "u make me smile", "you make me smile",
    "u make me blush", "ur cutie", "u r a cutie", "you are a cutie",
)
FLIRT_PREFIXES = ("", "hi", "hey", "aw", "aww", "ooh", "wow", "haha", "lol", "tbh", "ngl")
FLIRT_SUFFIXES = ("", "too", "lol", "haha", "tbh", "ngl", "tho")

BASE_REACTIONS = (
    # Laughter
    "haha", "hahaha", "hahahaha", "hehe", "hehehe", "lol", "loll", "lolll",
    "lmao", "lmaoo", "lmfao", "rofl",
    # Positive / Cool / Nice
    "nice", "niice", "cool", "coool", "sweet", "awesome", "amazing", "dope", "lit", "fire",
    "bet", "word", "fair", "fair enough", "oh nice", "oh cool", "ooh nice", "ooh cool",
    "aw nice", "haha nice", "haha cool", "lol nice", "lol cool", "yeah nice", "yeah cool",
    "sounds good", "sounds nice", "sounds cool", "sounds fun", "thats nice", "that's nice",
    "thats cool", "that's cool", "thats fair", "that's fair",
    # "same" & "sam" (English + realistic fast-typing variants)
    "same", "samee", "sameee", "saame", "sam", "saam", "samm",
    "same here", "sam here", "samee here", "saame here", "sam here lol", "same here lol",
    "same here haha", "sam here haha", "haha same", "haha sam", "lol same", "lol sam",
    "same lol", "sam lol", "same haha", "sam haha", "yeah same", "yea same", "yep same",
    "oh same", "aw same", "same tbh", "same fr", "same tho", "same to u", "same to you",
    "me too", "me to", "me too lol", "me too haha", "haha me too", "lol me too", "yeah me too",
    # Agreement / True / Facts
    "true", "tru", "truee", "so true", "too true", "very true", "true that", "true lol",
    "true haha", "haha true", "lol true", "yeah true", "thats true", "that's true",
    "facts", "fax", "fr", "fr fr", "for real", "ikr", "i know right", "i know", "right",
    "real", "agreed", "i agree", "totally", "definitely", "exact", "exactly",
    # OK / Affirmations (without "im", so Step 6 "im ok" stays strictly isolated)
    "ok", "okay", "okey", "oki", "okie", "kk", "alright", "alr", "aight",
    "ok nice", "ok cool", "okay nice", "okay cool", "ok sweet", "ok bet", "ok fair",
    "ok lol", "ok haha", "okay lol", "okay haha", "haha ok", "lol ok", "oh ok", "oh okay",
    "ah ok", "ah okay", "yeah ok", "yea ok", "ok then", "okay then", "alright cool",
    "yeah", "yea", "yep", "yup", "yas", "yass", "sure", "ofc", "of course", "aha", "wow", "omg",
)


def is_valid_flirt_line(text):
    s = SPACE_RE.sub(" ", text.strip().lower())
    if not s or len(s) > 40 or "?" in s or not CLEAN_LINE_RE.match(s):
        return ""
    words = s.split()
    if not words or len(words) < 2 or len(words) > 6:
        return ""
    for w in words:
        if w.strip("'!?.,") in FLIRT_REACT_FORBIDDEN_WORDS:
            return ""
    return s if STRICT_FLIRT_RE.match(s) else ""


def is_valid_react_line(text):
    s = SPACE_RE.sub(" ", text.strip().lower())
    if not s or len(s) > 32 or "?" in s or not CLEAN_LINE_RE.match(s):
        return ""
    words = s.split()
    if not words or len(words) > 4:
        return ""
    for w in words:
        if w.strip("'!?.,") in FLIRT_REACT_FORBIDDEN_WORDS:
            return ""
    return s if STRICT_REACT_RE.match(s) else ""


def _try_fetch_hf_dialog_lines(flirt_set, react_set):
    try:
        from datasets import load_dataset
        ds = load_dataset("daily_dialog", split="train", streaming=True)
        count = 0
        for row in ds:
            if count >= 1500:
                break
            count += 1
            for turn in row.get("dialog", ()):
                for sent in re.split(r"[.!?]+", turn):
                    c_f = is_valid_flirt_line(sent)
                    if c_f:
                        flirt_set.add(c_f)
                    else:
                        c_r = is_valid_react_line(sent)
                        if c_r:
                            react_set.add(c_r)
    except Exception:
        pass


def build_flirt_and_react_files(
    flirt_path="compliments_flirt.txt",
    react_path="reactions_ack.txt",
    neg_path="negatives.txt",
):
    flirt_set = set()
    react_set = set()

    for subj in FLIRT_SUBJECTS:
        for inten in FLIRT_INTENSIFIERS:
            for adj in FLIRT_ADJECTIVES:
                core = f"{subj} {inten} {adj}".strip() if inten else f"{subj} {adj}"
                core = SPACE_RE.sub(" ", core)
                for end in ("", "!"):
                    c = is_valid_flirt_line(f"{core}{end}")
                    if c:
                        flirt_set.add(c)
                for pfx in FLIRT_PREFIXES:
                    if pfx:
                        c = is_valid_flirt_line(f"{pfx} {core}")
                        if c:
                            flirt_set.add(c)
                for sfx in FLIRT_SUFFIXES:
                    if sfx:
                        c = is_valid_flirt_line(f"{core} {sfx}")
                        if c:
                            flirt_set.add(c)

    for phrase in FLIRT_DIRECT_PHRASES:
        for pfx in ("", "hey", "aw", "haha", "lol", "tbh", "ngl"):
            base = f"{pfx} {phrase}".strip() if pfx else phrase
            for sfx in ("", "too", "lol", "haha", "tbh"):
                full = f"{base} {sfx}".strip() if sfx else base
                for end in ("", "!"):
                    c = is_valid_flirt_line(f"{full}{end}")
                    if c:
                        flirt_set.add(c)

    for r_phrase in BASE_REACTIONS:
        for end in ("", "!"):
            c = is_valid_react_line(f"{r_phrase}{end}")
            if c and c not in flirt_set:
                react_set.add(c)

    _try_fetch_hf_dialog_lines(flirt_set, react_set)
    react_set.difference_update(flirt_set)

    sorted_flirt = sorted(flirt_set)
    sorted_react = sorted(react_set)

    flirt_u8 = np.frombuffer(("\n".join(sorted_flirt) + "\n").encode("ascii"), dtype=np.uint8)
    with open(flirt_path, "wb") as ff:
        ff.write(flirt_u8.tobytes())

    react_u8 = np.frombuffer(("\n".join(sorted_react) + "\n").encode("ascii"), dtype=np.uint8)
    with open(react_path, "wb") as fr:
        fr.write(react_u8.tobytes())

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn:
            existing_neg = {line.strip().lower() for line in fn if line.strip()}
    for item in sorted_flirt + sorted_react:
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ", "sup ")):
            existing_neg.add(item)
    neg_u8 = np.frombuffer(("\n".join(sorted(existing_neg)) + "\n").encode("ascii"), dtype=np.uint8)
    with open(neg_path, "wb") as fn:
        fn.write(neg_u8.tobytes())

    print(
        f"[CLEAN STEP 7] Saved {len(sorted_flirt)} lines to '{flirt_path}' ({flirt_u8.nbytes / 1024.0:.2f} KB) "
        f"& {len(sorted_react)} lines to '{react_path}' ({react_u8.nbytes / 1024.0:.2f} KB)",
        flush=True,
    )


if __name__ == "__main__":
    build_flirt_and_react_files()