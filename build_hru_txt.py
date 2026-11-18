#!/usr/bin/env python3
# FILE: build_hru_txt.py
# PURPOSE: Build 100% clean hru.txt (Class 8: ask_how_are_you) isolated from Steps 1-2 & 4-8
# RULES: Enforces Rules 1-13:
#        - Rule 12: Parquet-only fetch (revision="refs/convert/parquet"), 7-bit ASCII byte strip, closed vocab whitelist
#        - Rule 13: Zero "!" / "?!" / "!!" / "." Cartesian punctuation clones, cross-step analyzer, canonical deduplication

import json
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

CLEAN_HRU_CHARS_RE = re.compile(r"^[a-z\s'?]+$")
SPACE_RE = re.compile(r"\s+")

# Rule 13 Cross-Step Collision Guard (Blocks Steps 1, 2, 4, 5, 6, 7, 8 & Class 0 traps)
STEP_OTHER_COLLISION_RE = re.compile(
    r"^(?:.*(?:"
    r"\b(?:w+h+a+t+|w+a+t+|w+h+t+|w+y+d+|w+u+d+|w+h+a+t+c+h+a+|w+h+y+|w+h+o+|w+h+e+r+e+|w+h+i+c+h+)\b|"
    r"\b(?:o+l+d+|a+g+e+|f+r+o+m+|c+o+u+n+t+r+y+|l+i+v+e+|m+u+c+h+|f+a+r+|t+a+l+l+|l+o+n+g+)\b|"
    r"\b(?:m+a+l+e+|f+e+m+a+l+e+|f+a+m+l+e+|b+o+y+|g+i+r+l+|g+r+i+l+|w+o+m+a+n+|g+e+n+d+e+r+|s+e+x+)\b|"
    r"\b(?:i+m+|i+'?m+|l+m+|b+o+r+e+d+|b+o+r+a+d+|t+i+r+e+d+|c+h+i+l+i+n+g*|c+h+a+t+i+n+g*|n+m+u+)\b|"
    r"\b(?:c+u+t+e+|s+e+e+m+|l+i+k+e+|v+i+b+e+|s+a+m+e+|s+a+m+|t+r+u+e+|l+o+l+|h+a+h+a+)\b|"
    r"\b(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|s+e+e+k+i+n+g*|f+r+i+e+n+d+s+|g+f+|b+f+|d+a+t+i+n+g+|r+e+l+a+t+i+o+n+s+h+i+p+)\b"
    r").*|"
    r"(?:d+o+i+n+g+\s+(?:g+o+o+d+|g+r+e+a+t+|f+i+n+e+|o+k+|w+e+l+l+)))$",
    re.IGNORECASE,
)

# Strictly matches genuine "how are you" / status questions (Zero "!" / "!!" / "." clone bloat)
STRICT_HRU_RE = re.compile(
    r"^(?:(?:hi+|he+y+|he+l+o+|hlo+|yo+|sup+|h+i+y+a+|h+o+w+d+y+)\s+)?"
    r"(?:"
    r"h+r+u+(?:\s+(?:d+o+i+n+g*|t+o+d+a+y+))?|"
    r"h+y+d+(?:\s+t+o+d+a+y+)?|"
    r"h+o+r+|w+b+u+|h+b+u+|"
    r"(?:h+o+w+|h+w+|h+o+i+w+)\s+(?:a+r+e+|a+r+|a+e+r+|r+e*|i+s+|w+a+s+|w+h+a+s+|h+a+v+e+|d+o+)\s+"
    r"(?:y+o+u+|y+u+o+|y+u+|u+|y+a+|i+t+|y+o+u+r+|u+r+|t+h+i+n+g+s+|e+v+e+r+y+t+h+i+n+g+)"
    r"(?:\s+(?:d+o+i+n+g*|d+o+p+i+n+|d+i+o+n+g+|d+o+|g+o+i+n+g*|b+e+e+n+|t+o+d+a+y+|d+a+y*|f+e+e+l+i+n+g+))?"
    r"(?:\s+t+o+d+a+y+)?|"
    r"(?:h+o+w+|h+w+)\s+(?:u+|y+a+|y+o+u+)\s+(?:d+o+i+n+g*|d+o+p+i+n+|d+i+o+n+g+|b+e+e+n+)(?:\s+t+o+d+a+y+)?|"
    r"h+o+w+'?s\s+(?:i+t\s+g+o+i+n+g*|u+|y+o+u+|e+v+e+r+y+t+h+i+n+g*|(?:y+o+u+r|u+r)\s+d+a+y*)(?:\s+t+o+d+a+y+)?|"
    r"(?:a+r+e+\s+)?(?:y+o+u+|u+)\s+(?:g+o+o+d+|o+k+a*y*|a+l+r+i+g+h+t+)"
    r")"
    r"(?:\s+(?:t+o+d+a+y+|r+n+|t+h+e+r+e+|b+r+o+|f+r+i+e+n+d+|s+t+r+a+n+g+e+r+|c+u+t+i+e+))?\??$",
    re.IGNORECASE,
)

# Rule 12 Gate 2: Closed Whitelist Vocabulary for Step 3 (Blocks unknown words from poisoning s3_hru_bloom_q)
STRICT_ALLOWED_S3_VOCAB = frozenset((
    "hi", "hii", "hey", "heyy", "hello", "helloo", "helo", "hlo", "yo", "sup", "hiya", "howdy",
    "hru", "hyd", "hor", "wbu", "hbu",
    "how", "hows", "how's", "hw", "hoiw",
    "are", "ar", "aer", "aare", "r", "re", "is", "was", "whas", "have", "do",
    "you", "yuo", "yu", "u", "ya", "it", "your", "ur", "things", "everything",
    "doing", "doin", "dopin", "diong", "going", "goin", "been", "today", "day", "da", "feeling",
    "good", "ok", "okay", "alright", "rn", "there", "bro", "friend", "stranger", "cutie",
))

BASE_HRU_PHRASES = (
    "how are you", "how are you doing", "how are you today", "how are you doing today",
    "how was your day", "how is your day", "how is it going", "how's it going",
    "hows it going", "how have you been", "how do you do", "how are things",
    "how is everything", "hows everything", "how are you feeling",
    "hru", "hyd", "hru doing", "hru doin", "hru today", "hyd today",
    "hows u", "how's u", "how r u", "how are u", "how are ya",
    "how u doing", "how u doin", "how r u doing", "how r u doin",
    "how are u doing", "how are u doin", "hows ur day", "how's ur day",
    "how is ur day", "how was ur day", "how have u been", "how u been",
    "wbu", "hbu", "u good", "are u good", "are you good", "u ok", "are u ok", "are you okay",
    "hor", "how ar you dopin", "how ar you doing", "how ar u doin", "how ar u",
    "hw r u", "hw are u", "hw are you", "how u diong", "how are you diong",
    "how whas you da", "how whas your day", "how was you day", "how is ur da",
    "how r yu", "how are yuo", "how r yuo", "hows it goin", "how's it goin",
    "how is it goin", "how re u", "hoiw are you", "how aer you", "how aare you",
)

GREETING_PREFIXES = ("", "hi", "hey", "hello", "yo", "sup", "hiya", "howdy")
HRU_TARGETS = ("", "today", "rn", "there", "bro", "friend", "stranger", "cutie")


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
    # Step 3 MLP uses use_qmark=False in offline_train.py, so deduplicate purely by collapsed word tuple
    clean_w = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in s.lower())
    words = tuple(w for w in _collapse_str(clean_w).split() if w)
    return (words, False)


def categorize_line_across_all_steps(raw_text):
    clean_ascii = _sanitize_ascii_bytes(raw_text)
    if "/" in clean_ascii:
        return 0, ""
    s = SPACE_RE.sub(" ", clean_ascii.strip().lower()).rstrip("!.,")
    if not s or len(s) > 38 or not CLEAN_HRU_CHARS_RE.match(s):
        return 0, ""

    if STEP_OTHER_COLLISION_RE.match(s):
        return 1, s

    words = [w.strip("?'") for w in s.split() if w.strip("?'")]
    if not words or len(words) > 6:
        return 0, ""

    for w in words:
        if w not in STRICT_ALLOWED_S3_VOCAB:
            return 0, ""

    if STRICT_HRU_RE.match(s):
        return 8, s.rstrip("?")
    return 0, ""


def _safe_fetch_parquet_and_jsonl_hru(hru_map):
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
            if count >= 2500:
                break
            count += 1
            turns = row.get("dialog", ())
            if not isinstance(turns, (list, tuple)):
                continue
            for turn in turns[:3]:
                if not isinstance(turn, str):
                    continue
                for sent in re.split(r"[.!]+", turn[:90]):
                    cid, clean_s = categorize_line_across_all_steps(sent)
                    if cid == 8:
                        sig = _canonical_token_sig(clean_s)
                        if sig not in hru_map:
                            hru_map[sig] = clean_s
    except Exception:
        pass

    if os.path.exists("bot_chat_history.jsonl"):
        try:
            with open("bot_chat_history.jsonl", "r", encoding="utf-8", errors="ignore") as jf:
                for line in jf:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("sender") == "system":
                        continue
                    cid, clean_s = categorize_line_across_all_steps(str(rec.get("text", "")))
                    if cid == 8:
                        sig = _canonical_token_sig(clean_s)
                        if sig not in hru_map:
                            hru_map[sig] = clean_s
        except Exception:
            pass


def build_hru_file(output_path="hru.txt", neg_path="negatives.txt"):
    hru_map = {}

    for phrase in BASE_HRU_PHRASES:
        cid, c = categorize_line_across_all_steps(phrase)
        if cid == 8:
            sig = _canonical_token_sig(c)
            if sig not in hru_map:
                hru_map[sig] = c
        for tgt in HRU_TARGETS:
            if tgt and tgt not in phrase:
                cid, c = categorize_line_across_all_steps(f"{phrase} {tgt}")
                if cid == 8:
                    sig = _canonical_token_sig(c)
                    if sig not in hru_map:
                        hru_map[sig] = c

    for pfx in GREETING_PREFIXES:
        if not pfx:
            continue
        for phrase in BASE_HRU_PHRASES:
            if phrase in ("wbu", "hbu", "hor"):
                continue
            cid, c = categorize_line_across_all_steps(f"{pfx} {phrase}")
            if cid == 8:
                sig = _canonical_token_sig(c)
                if sig not in hru_map:
                    hru_map[sig] = c

    _safe_fetch_parquet_and_jsonl_hru(hru_map)

    sorted_hru = sorted(set(hru_map.values()))
    out_u8 = np.frombuffer(("\n".join(sorted_hru) + "\n").encode("ascii", errors="ignore"), dtype=np.uint8)
    with open(output_path, "wb") as fh:
        fh.write(out_u8.tobytes())

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn:
            existing_neg = {line.strip().lower() for line in fn if line.strip()}
    for item in sorted_hru:
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ", "sup ")):
            existing_neg.add(item)
    neg_u8 = np.frombuffer(("\n".join(sorted(existing_neg)) + "\n").encode("ascii"), dtype=np.uint8)
    with open(neg_path, "wb") as fn:
        fn.write(neg_u8.tobytes())

    n_lines = np.int32(len(sorted_hru))
    n_kb = np.float32(out_u8.nbytes) / np.float32(1024.0)
    print(f"[CLEAN STEP 3 HRU RULES 12+13] Saved {int(n_lines)} unique ask_how_are_you lines to '{output_path}' ({float(n_kb):.2f} KB)", flush=True)


if __name__ == "__main__":
    build_hru_file()