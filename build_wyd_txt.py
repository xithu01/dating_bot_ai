#!/usr/bin/env python3
# FILE: build_wyd_txt.py
# PURPOSE: Build clean wyd.txt (Class 15: ask_wyd) isolated from Step 1 ("whats up"), Step 3 ("how u doing") & Class 0 ("what do u do")

import logging
import os
import re
import warnings
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CLEAN_WYD_RE = re.compile(r"^[a-z\s'!?.,]+$")
SPACE_RE = re.compile(r"\s+")

# Strictly matches WYD / activity questions while blocking "how are you doing", "whats up", and "what do you do"
STRICT_WYD_RE = re.compile(
    r"^(?:(?:hi+|he+y+|he+l+o+|hlo+|yo+|sup+)\s+)?"
    r"(?:"
    r"w+y+d+|w+u+d+|w+h+a+t+c+h+a+\s+(?:d+o+i+n+g*|u+p+\s+t+o+)|"
    r"(?:w+h+a+t+|w+a+t+|w+h+t+)\s+(?:(?:a+r+e+|a+r+|r+)\s+)?(?:y+o+u+|y+u+o+|u+|y+a+)\s+"
    r"(?:d+o+i+n+g*|d+o+p+i+n+|d+i+o+n+g+|u+p+\s+t+o+)|"
    r"(?:w+h+a+t+'?s|w+h+a+t+s)\s+(?:y+o+u+|u+)\s+(?:d+o+i+n+g*|u+p+\s+t+o+)|"
    r"(?:d+o+i+n+g*)\s+a+n+y+t+h+i+n+g+(?:\s+f+u+n+)?"
    r")"
    r"(?:\s+(?:r+n+|n+o+w+|t+o+d+a+y+|t+o+n+i+g+h+t+|t+h+e+r+e+))?"
    r"[\s!?.]*$",
    re.IGNORECASE,
)

# Forbids "how" (Step 3 HRU), "do" ("what do you do" Class 0 trap), and Age/Country/Gender tokens
WYD_FORBIDDEN_WORDS = frozenset((
    "how", "hows", "how's", "hru", "hyd", "hor", "wbu", "hbu", "do", "want", "this", "that",
    "name", "time", "why", "who", "where", "old", "age", "from", "country", "m", "f", "male",
    "female", "boy", "girl", "man", "woman", "gender", "sex", "lol", "haha",
))

BASE_WYD_PHRASES = (
    "wyd", "wud", "what are you doing", "what r u doing", "what are u doing",
    "what u doing", "what ya doing", "what are you doin", "what r u doin",
    "what are u doin", "what u doin", "whatcha doing", "whatcha doin",
    "what are you up to", "what r u up to", "what are u up to", "what u up to",
    "whatcha up to", "whats u doing", "what's u doing", "whats u up to",
    "doing anything", "doing anything fun", "doin anything", "doin anything fun",
    # Realistic keyboard typos
    "wat u doing", "wat r u doing", "wat are u doing", "wht u doing", "wht r u doing",
    "what ar u doing", "what ar you doing", "what u diong", "what r u diong",
    "what are you diong", "what u dopin", "what r u dopin", "wyd rn", "wud rn",
)

GREETING_PREFIXES = ("", "hi", "hey", "hello", "yo", "sup", "hii", "heyy")
TIME_SUFFIXES = ("", "rn", "now", "today", "tonight")
ENDINGS = ("", "?", "!", "?!")


def is_valid_wyd_line(text):
    s = SPACE_RE.sub(" ", text.strip().lower())
    if not s or len(s) > 38 or not CLEAN_WYD_RE.match(s):
        return ""
    words = s.split()
    if not words or len(words) > 6:
        return ""
    for w in words:
        if w.strip("'!?.,") in WYD_FORBIDDEN_WORDS:
            return ""
    return s if STRICT_WYD_RE.match(s) else ""


def build_wyd_file(output_path="wyd.txt", neg_path="negatives.txt"):
    wyd_set = set()
    for phrase in BASE_WYD_PHRASES:
        for sfx in TIME_SUFFIXES:
            base = f"{phrase} {sfx}".strip() if (sfx and sfx not in phrase) else phrase
            for end in ENDINGS:
                c = is_valid_wyd_line(f"{base}{end}")
                if c:
                    wyd_set.add(c)
        for pfx in GREETING_PREFIXES:
            if pfx:
                for end in ("", "?", "!"):
                    c = is_valid_wyd_line(f"{pfx} {phrase}{end}")
                    if c:
                        wyd_set.add(c)

    sorted_wyd = sorted(wyd_set)
    wyd_u8 = np.frombuffer(("\n".join(sorted_wyd) + "\n").encode("ascii"), dtype=np.uint8)
    with open(output_path, "wb") as f:
        f.write(wyd_u8.tobytes())

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn:
            existing_neg = {line.strip().lower() for line in fn if line.strip()}
    for item in sorted_wyd:
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ", "sup ")):
            existing_neg.add(item)
    neg_u8 = np.frombuffer(("\n".join(sorted(existing_neg)) + "\n").encode("ascii"), dtype=np.uint8)
    with open(neg_path, "wb") as fn:
        fn.write(neg_u8.tobytes())

    print(f"[CLEAN WYD] Overwrote {len(sorted_wyd)} pure ask_wyd lines to '{output_path}' ({wyd_u8.nbytes / 1024.0:.2f} KB)", flush=True)


if __name__ == "__main__":
    build_wyd_file()