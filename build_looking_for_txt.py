#!/usr/bin/env python3
# FILE: build_looking_for_txt.py
# PURPOSE: Zero-CVE Isolated Dataset Builder for Step 8 (Classes 21 & 22: Ask / State "Looking For")
# RULES: Enforces Rule 12 (Parquet-only revision="refs/convert/parquet", zero daily_dialog.py script download,
#        7-bit ASCII byte sanitizer, 100% closed-vocabulary whitelist, zero access to model_weights.npz).

import logging
import os
import re
import warnings
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "0"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CLEAN_LINE_RE = re.compile(r"^[a-z\s'!?.,/]+$")
SPACE_RE = re.compile(r"\s+")

# Strictly matches Class 21 (ask_looking_for)
STRICT_ASK_LOOKING_RE = re.compile(
    r"^(?:(?:hi+|he+y+|so+|b+t+w+|w+b+u+|h+b+u+|a+n+d+)\s+)?"
    r"(?:"
    r"(?:w+h+a+t+|w+a+t+|w+h+t+|w+h+o+)\s+(?:(?:a+r+e+|a+r+|r+|d+o+)\s+)?(?:y+o+u+|u+|y+a+)\s+"
    r"(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|s+e+e+k+i+n+g*|s+e+a+r+c+h+i+n+g*|w+a+n+t+|h+e+r+e+)"
    r"(?:\s+f+o+r+)?(?:\s+h+e+r+e+)?|"
    r"(?:w+h+a+t+'?s|w+h+a+t+s)\s+(?:y+o+u+r+|u+r+)\s+(?:g+o+a+l+|p+u+r+p+o+s+e+)\s+h+e+r+e+|"
    r"(?:w+h+a+t+|w+a+t+)\s+b+r+i+n+g+s+\s+(?:y+o+u+|u+)\s+h+e+r+e+|"
    r"w+h+y+\s+(?:(?:a+r+e+|a+r+|r+)\s+)?(?:y+o+u+|u+)\s+(?:o+n+\s+)?h+e+r+e+|"
    r"(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|s+e+e+k+i+n+g*|s+e+a+r+c+h+i+n+g*)\s+f+o+r+(?:\s+w+h+a+t+)?|"
    r"(?:(?:a+r+e+|r+)\s+(?:y+o+u+|u+)\s+|(?:y+o+u+|u+)\s+)?(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|h+e+r+e+)\s+"
    r"(?:f+o+r+\s+)?"
    r"(?:f+r+i+e+n+d+s*|d+a+t+i+n+g+|l+o+v+e+|f+u+n+|r+e+l+a+t+i+o+n+s+h+i+p+|g+f+|b+f+|"
    r"g+i+r+l+s*|g+r+i+l+s*|b+o+y+s*|g+u+y+s*)"
    r"\s+o+r+\s+"
    r"(?:f+r+i+e+n+d+s*|d+a+t+i+n+g+|l+o+v+e+|f+u+n+|r+e+l+a+t+i+o+n+s+h+i+p+|c+h+a+t+i+n+g*|g+f+|b+f+|"
    r"g+i+r+l+s*|g+r+i+l+s*|b+o+y+s*|g+u+y+s*)|"
    r"(?:(?:a+r+e+|r+|d+o+)\s+(?:y+o+u+|u+)|(?:y+o+u+|u+))\s+"
    r"(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|s+e+e+k+i+n+g*|s+e+a+r+c+h+i+n+g*|w+a+n+t+)\s+"
    r"(?:f+o+r+\s+)?(?:a+\s+)?"
    r"(?:f+r+i+e+n+d+s*|g+f+|b+f+|g+i+r+l+f+r+i+e+n+d+|b+o+y+f+r+i+e+n+d+|g+i+r+l+s*|g+r+i+l+s*|"
    r"b+o+y+s*|g+u+y+s*|r+e+l+a+t+i+o+n+s+h+i+p+|d+a+t+i+n+g+|l+o+v+e+|f+u+n+|s+o+m+e+o+n+e+)|"
    r"(?:f+r+i+e+n+d+s*|d+a+t+i+n+g+|r+e+l+a+t+i+o+n+s+h+i+p+|g+f+|b+f+)\s*\?+"
    r")[\s!?.]*$",
    re.IGNORECASE,
)

# Strictly matches Class 22 (state_looking_for)
STRICT_STATE_LOOKING_RE = re.compile(
    r"^(?:"
    r"(?:(?:i+m+|i+'?m+|l+'?m+|l+m+|i+\s+a+m+|i+|m+|f+|b+o+y+|g+i+r+l+|g+r+i+l+|g+u+y+|j+u+s+t+|o+n+l+y+)\s+)?"
    r"(?:"
    r"(?:(?:j+u+s+t+|o+n+l+y+)\s+)?"
    r"(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|f+i+n+d+|s+e+e+k+i+n+g*|s+e+a+r+c+h+i+n+g*|"
    r"w+a+n+t+i+n+g*|w+a+n+t+|w+a+n+a+|n+e+e+d+|t+r+y+n+a+\s+f+i+n+d+|t+r+y+i+n+g+\s+t+o+\s+f+i+n+d+)"
    r"(?:\s+(?:f+o+r+|t+o+\s+m+a+k+e+|t+o+\s+m+e+e+t+|t+o+\s+f+i+n+d+|t+o+))?\s+"
    r"(?:a+\s+|n+e+w+\s+|s+o+m+e+\s+|g+o+o+d+\s+|n+i+c+e+\s+|c+u+t+e+\s+)?"
    r"(?:f+r+i+e+n+d+s*|f+r+e+i+n+d+s*|f+r+e+n+d+s*|f+r+i+e+n+d+s+h+i+p+|"
    r"g+f+s*|g+i+r+l+f+r+i+e+n+d+|g+r+i+l+f+r+i+e+n+d+|b+f+s*|b+o+y+f+r+i+e+n+d+|"
    r"g+i+r+l+s*|g+r+i+l+s*|g+u+r+l+s*|w+o+m+a+n+|f+e+m+a+l+e+|f+a+m+l+e+|l+a+d+y+|f+|"
    r"b+o+y+s*|b+o+i+s*|g+u+y+s*|m+a+n+|m+a+l+e+|m+|"
    r"p+a+r+t+n+e+r+|s+o+u+l+m+a+t+e+|r+e+l+a+t+i+o+n+s+h+i+p+|d+a+t+i+n+g+|d+a+t+e+s*|"
    r"l+o+v+e+|r+o+m+a+n+c+e+|f+l+i+r+t+i+n+g*|s+o+m+e+o+n+e+|p+e+o+p+l+e+|"
    r"c+o+n+e+c+t+i+o+n+|f+u+n+|c+o+m+p+a+n+y+|c+h+a+t+i+n+g*|c+h+a+t+|t+a+l+k+)|"
    r"(?:(?:j+u+s+t+)\s+)?(?:m+a+k+i+n+g*|m+a+k+e+|m+e+e+t+i+n+g*|m+e+e+t+)\s+"
    r"(?:a+\s+|n+e+w+\s+|s+o+m+e+\s+)?"
    r"(?:f+r+i+e+n+d+s*|f+r+e+i+n+d+s*|f+r+e+n+d+s*|p+e+o+p+l+e+|g+i+r+l+s*|g+r+i+l+s*|b+o+y+s*|g+u+y+s*|g+f+|b+f+)|"
    r"(?:(?:j+u+s+t+)\s+)?h+e+r+e+\s+(?:f+o+r+|t+o+|t+o+\s+m+a+k+e+|t+o+\s+f+i+n+d+|t+o+\s+m+e+e+t+)\s+"
    r"(?:a+\s+|n+e+w+\s+)?"
    r"(?:f+r+i+e+n+d+s*|f+r+e+i+n+d+s*|d+a+t+i+n+g+|l+o+v+e+|f+u+n+|r+e+l+a+t+i+o+n+s+h+i+p+|"
    r"g+f+|b+f+|g+i+r+l+f+r+i+e+n+d+|b+o+y+f+r+i+e+n+d+|g+i+r+l+s*|g+r+i+l+s*|b+o+y+s*|g+u+y+s*|"
    r"c+h+a+t+i+n+g*|c+h+a+t+|t+a+l+k+)"
    r")|"
    r"(?:(?:a+|j+u+s+t+|o+n+l+y+|n+e+w+|s+o+m+e+|g+o+o+d+|s+e+r+i+o+u+s+|t+r+u+e+)\s+)?"
    r"(?:f+r+i+e+n+d+s+|f+r+e+i+n+d+s+|f+r+e+n+d+s+|f+r+i+e+n+d+s+h+i+p+|"
    r"g+f+|g+i+r+l+f+r+i+e+n+d+|g+r+i+l+f+r+i+e+n+d+|b+f+|b+o+y+f+r+i+e+n+d+|"
    r"r+e+l+a+t+i+o+n+s+h+i+p+|d+a+t+i+n+g+|r+o+m+a+n+c+e+|s+o+u+l+m+a+t+e+)"
    r")"
    r"(?:[\s,]+(?:h+e+r+e+|t+b+h+|l+o+l+|h+a+h+a+|a+n+d+\s+u+|a+n+d+\s+y+o+u+|u+|y+o+u+|w+b+u+|h+b+u+))?"
    r"[\s!?.]*$",
    re.IGNORECASE,
)

# Rule 12 Gate 2: 100% Closed Whitelist Vocabulary (Blocks unknown words from entering .txt or Bloom filters)
STRICT_ALLOWED_S8_VOCAB = frozenset((
    "looking", "lookin", "loking", "lokin", "finding", "findin", "find", "fnding",
    "seeking", "seekin", "seek", "searching", "searchin", "search",
    "brings", "bring", "goal", "purpose", "reason",
    "gf", "gfs", "girlfriend", "grilfriend", "girlfreind", "wifey", "wife",
    "bf", "bfs", "boyfriend", "boifriend", "boyfreind", "hubby", "husband",
    "relationship", "relashionship", "relationships", "partner", "soulmate",
    "dating", "date", "dates", "romance", "romantic", "flirting", "flirt", "hookup",
    "friends", "friend", "freind", "freinds", "frends", "frend", "friendship", "bestie", "company",
    "girl", "gril", "gurl", "girls", "grils", "gurls", "woman", "women",
    "female", "famle", "famel", "females", "lady", "ladies", "f",
    "boy", "boi", "boys", "bois", "guy", "guys", "man", "men",
    "male", "maale", "males", "dude", "dudes", "m",
    "what", "whats", "what's", "wat", "wht", "why", "who", "or",
    "making", "make", "trying", "tryna", "want", "wan", "wanna", "wanting",
    "need", "needing", "hoping", "meet", "meeting",
    "im", "i'm", "l'm", "lm", "iam", "i'am", "i", "am", "just", "only", "mostly",
    "are", "ar", "r", "do", "you", "u", "ya", "your", "ur", "for", "to", "here",
    "on", "in", "a", "an", "new", "some", "good", "nice", "cute", "cool", "real",
    "serious", "true", "love", "fun", "someone", "somebody", "people",
    "ppl", "connection", "chat", "chatting", "chatin", "chating", "talk", "talking",
    "tbh", "ngl", "lol", "haha", "wbu", "hbu", "so", "btw", "and", "about", "with",
    "hi", "hey", "hello", "yo",
))

ASK_LOOKING_BASES = (
    "what are you looking for", "what r u looking for", "what are u looking for",
    "what u looking for", "what ya looking for", "wat u looking for", "wht u looking for",
    "what r u loking for", "what u lookin for", "what r u lookin for",
    "what are you finding", "what r u finding", "what u finding",
    "what are you looking for here", "what r u looking for here", "what u looking for here",
    "what are you seeking", "what r u seeking", "what are you searching for",
    "what do u want here", "what u want here", "who r u looking for", "who u looking for",
    "what are you here for", "what r u here for", "what u here for",
    "why are you here", "why r u here", "why are u here", "why u here",
    "why are you on here", "why r u on here", "why are u on here", "why u on here",
    "what brings you here", "what brings u here", "whats your goal here", "whats ur goal here",
    "looking for", "looking for what", "lookin for", "finding what",
    "looking for friends or dating", "looking for dating or friends",
    "looking for friends or relationship", "looking for fun or dating",
    "looking for gf or friends", "looking for bf or friends",
    "looking for boy or girl", "looking for girl or boy", "u looking for girl or boy",
    "are you looking for friends or dating", "r u looking for friends or dating",
    "are u looking for a bf", "r u looking for a bf", "are u looking for a gf", "r u looking for a gf",
    "u looking for a girl", "u looking for a boy", "u looking for a gf", "u looking for a bf",
    "u looking for friends", "u looking for dating", "are u looking for a girl",
    "are u looking for a boy", "are u looking for friends", "r u looking for friends",
    "are u looking for a relationship", "r u looking for a relationship",
    "are u looking for dating", "r u looking for dating", "are u looking for love",
    "friends?", "dating?", "relationship?", "gf?", "bf?",
)

STATE_LOOKING_PREFIXES = ("", "im", "i'm", "lm", "l'm", "i am", "i", "m", "f", "boy", "girl", "guy", "just", "im just")
STATE_LOOKING_VERBS = (
    "looking for", "lookin for", "loking for", "lokin for",
    "finding", "findin", "find", "looking", "loking",
    "seeking", "searching for", "want", "wanting", "need", "here for", "here to find",
)
STATE_LOOKING_TARGETS = (
    "girl", "a girl", "gril", "a gril", "gurl", "girls", "grils",
    "gf", "a gf", "girlfriend", "a girlfriend", "grilfriend",
    "female", "a female", "famle", "woman", "a woman", "f",
    "boy", "a boy", "boi", "boys", "guy", "a guy", "guys",
    "bf", "a bf", "boyfriend", "a boyfriend", "man", "a man", "male", "m",
    "friends", "new friends", "some friends", "good friends", "freinds", "frends", "friendship",
    "a relationship", "relationship", "a partner", "partner", "a soulmate", "soulmate",
    "dating", "a date", "dates", "love", "romance", "flirting", "fun", "company",
    "someone", "new people", "people", "a connection", "chatting",
)
STATE_DIRECT_PHRASES = (
    "gf", "a gf", "girlfriend", "a girlfriend", "grilfriend",
    "bf", "a bf", "boyfriend", "a boyfriend",
    "friends", "new friends", "some friends", "good friends", "just friends", "only friends",
    "freinds", "frends", "friendship",
    "dating", "just dating", "relationship", "a relationship", "serious relationship",
    "romance", "soulmate", "a soulmate",
    "making friends", "making new friends", "just making friends", "im making friends",
    "make friends", "make new friends", "looking to make friends", "here to make friends",
    "just here to make friends", "meeting new friends", "meet new friends", "meet girls",
    "here to meet girls", "looking to chat", "just looking to chat",
    "here to chat", "just here to chat", "here to talk", "just here to talk",
    "looking to date", "here to date", "tryna find a gf", "tryna find a girl",
    "trying to find a gf", "trying to find a girl", "trying to find a bf",
)
STATE_SUFFIXES = ("", "here", "tbh", "lol", "haha", "u", ", u", "wbu", ", wbu", "hbu", "and u")


def _sanitize_ascii_bytes(raw_text):
    # Rule 12 Gate 1: Pure 7-bit printable ASCII byte strip (blocks any CVE payload or control chars)
    if not isinstance(raw_text, str):
        return ""
    b = np.frombuffer(raw_text[:64].encode("ascii", errors="ignore"), dtype=np.uint8)
    valid = b[(b >= np.uint8(32)) & (b <= np.uint8(122))]
    if valid.shape[0] == 0:
        return ""
    return valid.tobytes().decode("ascii", errors="ignore")


def is_valid_ask_looking(text):
    clean_ascii = _sanitize_ascii_bytes(text)
    s = SPACE_RE.sub(" ", clean_ascii.strip().lower())
    if not s or len(s) > 44 or not CLEAN_LINE_RE.match(s):
        return ""
    words = [w.strip("'!?.,/") for w in s.split() if w.strip("'!?.,/")]
    if not words or len(words) > 6:
        return ""
    # Rule 12 Gate 2: Closed-Vocabulary Whitelist Check
    for w in words:
        if w not in STRICT_ALLOWED_S8_VOCAB:
            return ""
    return s if STRICT_ASK_LOOKING_RE.match(s) else ""


def is_valid_state_looking(text):
    clean_ascii = _sanitize_ascii_bytes(text)
    s = SPACE_RE.sub(" ", clean_ascii.strip().lower())
    if not s or len(s) > 44 or not CLEAN_LINE_RE.match(s):
        return ""
    words = [w.strip("'!?.,/") for w in s.split() if w.strip("'!?.,/")]
    if not words or len(words) > 6:
        return ""
    if len(words) == 1 and "?" in s:
        return ""
    # Rule 12 Gate 2: Closed-Vocabulary Whitelist Check
    for w in words:
        if w not in STRICT_ALLOWED_S8_VOCAB or w == "or":
            return ""
    return s if STRICT_STATE_LOOKING_RE.match(s) else ""


def _safe_fetch_parquet_dialog_lines(ask_set, state_set):
    # Rule 12: Streams ONLY pure Parquet data (revision="refs/convert/parquet")
    # NEVER downloads or executes daily_dialog.py!
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "daily_dialog",
            split="train",
            revision="refs/convert/parquet",
            streaming=True,
        )
        count = 0
        for row in ds:
            if count >= 1500:
                break
            count += 1
            turns = row.get("dialog", ())
            if not isinstance(turns, (list, tuple)):
                continue
            for turn in turns:
                if not isinstance(turn, str):
                    continue
                for sent in re.split(r"[.!]+", turn[:120]):
                    ca = is_valid_ask_looking(sent)
                    if ca:
                        ask_set.add(ca)
                    else:
                        cs = is_valid_state_looking(sent)
                        if cs:
                            state_set.add(cs)
    except Exception:
        pass


def build_looking_for_files(
    ask_path="looking_for_ask.txt",
    state_path="looking_for_state.txt",
    neg_path="negatives.txt",
):
    ask_set = set()
    state_set = set()

    for base in ASK_LOOKING_BASES:
        if "?" in base:
            c = is_valid_ask_looking(base)
            if c:
                ask_set.add(c)
            continue
        for pfx in ("", "hi", "hey", "so", "btw", "wbu", "hbu"):
            full = f"{pfx} {base}".strip() if pfx else base
            for end in ("", "?", "!"):
                c = is_valid_ask_looking(f"{full}{end}")
                if c:
                    ask_set.add(c)

    for pfx in STATE_LOOKING_PREFIXES:
        for vb in STATE_LOOKING_VERBS:
            if pfx == "i" and vb not in ("want", "need", "find"):
                continue
            for tgt in STATE_LOOKING_TARGETS:
                core = f"{pfx} {vb} {tgt}".strip() if pfx else f"{vb} {tgt}"
                core = SPACE_RE.sub(" ", core)
                for sfx in STATE_SUFFIXES:
                    full = f"{core} {sfx}".strip() if sfx else core
                    is_ab = "u" in sfx
                    ends = ("", "?") if is_ab else ("", "!")
                    for end in ends:
                        c = is_valid_state_looking(f"{full}{end}")
                        if c:
                            state_set.add(c)

    for d_phrase in STATE_DIRECT_PHRASES:
        for sfx in STATE_SUFFIXES:
            full = f"{d_phrase} {sfx}".strip() if sfx else d_phrase
            is_ab = "u" in sfx
            ends = ("", "?") if is_ab else ("", "!")
            for end in ends:
                c = is_valid_state_looking(f"{full}{end}")
                if c:
                    state_set.add(c)

    _safe_fetch_parquet_dialog_lines(ask_set, state_set)
    state_set.difference_update(ask_set)

    sorted_ask = sorted(ask_set)
    sorted_state = sorted(state_set)

    ask_u8 = np.frombuffer(("\n".join(sorted_ask) + "\n").encode("ascii"), dtype=np.uint8)
    with open(ask_path, "wb") as fa:
        fa.write(ask_u8.tobytes())

    state_u8 = np.frombuffer(("\n".join(sorted_state) + "\n").encode("ascii"), dtype=np.uint8)
    with open(state_path, "wb") as fs:
        fs.write(state_u8.tobytes())

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn:
            existing_neg = {line.strip().lower() for line in fn if line.strip()}
    for item in sorted_ask + sorted_state:
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ", "sup ")):
            existing_neg.add(item)
    neg_u8 = np.frombuffer(("\n".join(sorted(existing_neg)) + "\n").encode("ascii"), dtype=np.uint8)
    with open(neg_path, "wb") as fn:
        fn.write(neg_u8.tobytes())

    print(
        f"[RULE 12 PARQUET-ISOLATED STEP 8] Saved {len(sorted_ask)} lines to '{ask_path}' ({ask_u8.nbytes / 1024.0:.2f} KB) "
        f"& {len(sorted_state)} lines to '{state_path}' ({state_u8.nbytes / 1024.0:.2f} KB) — ZERO .py scripts!",
        flush=True,
    )


if __name__ == "__main__":
    build_looking_for_files()