#!/usr/bin/env python3
# FILE: build_wyd_txt.py
# PURPOSE: Build 100% clean wyd.txt (Class 15: ask_wyd) isolated from Steps 1-4 & 6-8
# RULES: Enforces Rules 1-13:
#        - Rule 12: Parquet-only fetch (revision="refs/convert/parquet"), 7-bit ASCII byte strip, closed vocab whitelist
#        - Rule 13: Zero "!" / "?!" / "?" Cartesian punctuation clones, cross-step analyzer, canonical deduplication

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

CLEAN_WYD_RE = re.compile(r"^[a-z\s'?]+$")
SPACE_RE = re.compile(r"\s+")

# Rule 13 Cross-Step Collision Guard (Blocks Step 1 "whats up", Step 3 "how u doing", Step 6 "doing good", Step 8 "looking for", Class 0 "what do u do")
STEP_OTHER_COLLISION_RE = re.compile(
    r"^(?:"
    r"(?:w+h+a+t+'?s\s+u+p+|w+h+a+t+\s+i+s+\s+u+p+)|"
    r".*\b(?:h+o+w+|h+o+w+'?s|h+r+u+|h+y+d+|h+o+r+|w+b+u+|h+b+u+|d+o+|w+a+n+t+|n+a+m+e+|t+i+m+e+|w+h+y+|w+h+o+|w+h+e+r+e+)\b.*|"
    r".*\b(?:o+l+d+|a+g+e+|f+r+o+m+|c+o+u+n+t+r+y+|m+a+l+e+|f+e+m+a+l+e+|b+o+y+|g+i+r+l+|g+e+n+d+e+r+|s+e+x+)\b.*|"
    r".*\b(?:g+o+o+d+|g+r+e+a+t+|f+i+n+e+|o+k+|b+o+r+e+d+|n+o+t+h+i+n+g+|c+h+i+l+i+n+g*|c+h+a+t+i+n+g*)\b.*|"
    r".*\b(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|s+e+e+k+i+n+g*|f+r+i+e+n+d+s*|g+f+|b+f+|d+a+t+i+n+g+|r+e+l+a+t+i+o+n+s+h+i+p+)\b.*"
    r")$",
    re.IGNORECASE,
)

# Strictly matches WYD / activity questions (Zero "!" / "?!" clone bloat)
STRICT_WYD_RE = re.compile(
    r"^(?:(?:hi+|he+y+|he+l+o+|hlo+|yo+|sup+)\s+)?"
    r"(?:"
    r"w+y+d+|w+u+d+|w+h+a+t+c+h+a+\s+(?:d+o+i+n+g*|u+p+\s+t+o+)|"
    r"(?:w+h+a+t+|w+a+t+|w+h+t+)\s+(?:(?:a+r+e+|a+r+|r+)\s+)?(?:y+o+u+|y+u+o+|u+|y+a+)\s+"
    r"(?:d+o+i+n+g*|d+o+p+i+n+|d+i+o+n+g+|u+p+\s+t+o+)|"
    r"(?:w+h+a+t+'?s|w+h+a+t+s)\s+(?:y+o+u+|u+)\s+(?:d+o+i+n+g*|u+p+\s+t+o+)|"
    r"(?:d+o+i+n+g*)\s+a+n+y+t+h+i+n+g+(?:\s+f+u+n+)?"
    r")"
    r"(?:\s+(?:r+n+|n+o+w+|t+o+d+a+y+|t+o+n+i+g+h+t+|t+h+e+r+e+))?\??$",
    re.IGNORECASE,
)

# Rule 12 Gate 2: Closed Whitelist Vocabulary for Step 5 (Blocks unknown words from poisoning s5_wyd_bloom_q)
STRICT_ALLOWED_S5_VOCAB = frozenset((
    "hi", "hii", "hey", "heyy", "hello", "helo", "hlo", "yo", "sup",
    "wyd", "wud", "whatcha", "what", "whats", "what's", "wat", "wht",
    "are", "ar", "r", "you", "yuo", "u", "ya",
    "doing", "doin", "dopin", "diong", "up", "to", "anything", "fun",
    "rn", "now", "today", "tonight", "there",
))

BASE_WYD_PHRASES = (
    "wyd", "wud", "what are you doing", "what r u doing", "what are u doing",
    "what u doing", "what ya doing", "what are you doin", "what r u doin",
    "what are u doin", "what u doin", "whatcha doing", "whatcha doin",
    "what are you up to", "what r u up to", "what are u up to", "what u up to",
    "whatcha up to", "whats u doing", "what's u doing", "whats u up to",
    "doing anything", "doing anything fun", "doin anything", "doin anything fun",
    "wat u doing", "wat r u doing", "wat are u doing", "wht u doing", "wht r u doing",
    "what ar u doing", "what ar you doing", "what u diong", "what r u diong",
    "what are you diong", "what u dopin", "what r u dopin", "wyd rn", "wud rn",
)

GREETING_PREFIXES = ("", "hi", "hey", "hello", "yo", "sup")
TIME_SUFFIXES = ("", "rn", "now", "today", "tonight")


def _sanitize_ascii_bytes(raw_text):
    if not isinstance(raw_text, str):
        return ""
    b = np.frombuffer(raw_text[:64].encode("ascii", errors="ignore"), dtype=np.uint8)
    valid = b[(b >= np.uint8(32)) & (b <= np.uint8(122))]
    if valid.shape[0] == 0:
        return ""
    return valid.tobytes().decode("ascii", errors="ignore")


def _collapse_str(s):
    return "".join(s[i] for i in range(len(s)) if i == 0 or s[i] != s[i - 1])


def _canonical_token_sig(s):
    # Step 5 MLP uses use_qmark=False in offline_train.py, so deduplicate purely by collapsed word tuple
    clean_w = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in s.lower())
    words = tuple(w for w in _collapse_str(clean_w).split() if w)
    return (words, False)


def categorize_line_across_all_steps(raw_text):
    clean_ascii = _sanitize_ascii_bytes(raw_text)
    if "/" in clean_ascii:
        return 0, ""
    s = SPACE_RE.sub(" ", clean_ascii.strip().lower()).rstrip("!.,")
    if not s or len(s) > 38 or not CLEAN_WYD_RE.match(s):
        return 0, ""

    if STEP_OTHER_COLLISION_RE.match(s):
        return 1, s

    words = [w.strip("?'") for w in s.split() if w.strip("?'")]
    if not words or len(words) > 6:
        return 0, ""

    for w in words:
        if w not in STRICT_ALLOWED_S5_VOCAB:
            return 0, ""

    if STRICT_WYD_RE.match(s):
        return 15, s.rstrip("?")
    return 0, ""


def _safe_fetch_parquet_wyd_lines(wyd_map):
    # Rule 12: Streams ONLY pure Parquet data (revision="refs/convert/parquet", zero remote .py scripts!)
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
            if count >= 2000:
                break
            count += 1
            turns = row.get("dialog", ())
            if not isinstance(turns, (list, tuple)):
                continue
            for turn in turns:
                if not isinstance(turn, str):
                    continue
                for sent in re.split(r"[.!]+", turn[:90]):
                    cid, clean_s = categorize_line_across_all_steps(sent)
                    if cid == 15:
                        sig = _canonical_token_sig(clean_s)
                        if sig not in wyd_map:
                            wyd_map[sig] = clean_s
    except Exception:
        pass


def build_wyd_file(output_path="wyd.txt", neg_path="negatives.txt"):
    wyd_map = {}

    for phrase in BASE_WYD_PHRASES:
        for sfx in TIME_SUFFIXES:
            base = f"{phrase} {sfx}".strip() if (sfx and sfx not in phrase) else phrase
            cid, c = categorize_line_across_all_steps(base)
            if cid == 15:
                sig = _canonical_token_sig(c)
                if sig not in wyd_map:
                    wyd_map[sig] = c
        for pfx in GREETING_PREFIXES:
            if pfx:
                cid, c = categorize_line_across_all_steps(f"{pfx} {phrase}")
                if cid == 15:
                    sig = _canonical_token_sig(c)
                    if sig not in wyd_map:
                        wyd_map[sig] = c

    _safe_fetch_parquet_wyd_lines(wyd_map)

    sorted_wyd = sorted(set(wyd_map.values()))
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

    print(f"[CLEAN STEP 5 WYD RULES 12+13] Saved {len(sorted_wyd)} unique ask_wyd lines to '{output_path}' ({wyd_u8.nbytes / 1024.0:.2f} KB)", flush=True)


if __name__ == "__main__":
    build_wyd_file()