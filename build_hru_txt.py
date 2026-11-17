#!/usr/bin/env python3
# FILE: build_hru_txt.py
# PURPOSE: Build clean hru.txt (Class 8: ask_how_are_you) with slang, targets, punctuation & keyboard typos
# RULES: Enforces zero float64/int64, strict isolation against Class 0 ("what"/"why"/"who"), Step 2, and Step 4

import json
import logging
import os
import re
import warnings
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CLEAN_HRU_CHARS_RE = re.compile(r"^[a-z\s'!?.,]+$")
SPACE_RE = re.compile(r"\s+")

# Strictly matches genuine "how are you" / status questions (including "how do you do" & trailing targets)
# while blocking "what", "why", "who", "how old", "where from", and all gender phrases.
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
    r"(?:\s+(?:t+o+d+a+y+|r+n+|t+h+e+r+e+|b+r+o+|f+r+i+e+n+d+|s+t+r+a+n+g+e+r+|c+u+t+i+e+))?"
    r"[\s!?.]*$",
    re.IGNORECASE,
)

# Words that must NEVER appear in hru.txt (prevents overlap with Class 0, Step 2 Age/Country, or Step 4 Gender)
HRU_FORBIDDEN_WORDS = frozenset((
    "old", "olds", "age", "from", "country", "live", "living", "where",
    "what", "whats", "what's", "why", "who", "which", "much", "far", "tall",
    "many", "long", "price", "cost", "name", "lol", "haha", "lmao", "rofl",
    "m", "f", "w", "male", "maale", "mael", "female", "famle", "famel", "femael", "femle",
    "boy", "boi", "girl", "gril", "gurl", "man", "woman", "womman", "guy", "lady",
    "gender", "gander", "gnder", "sex", "feminine", "masculine",
    "tun", "tn", "tunis", "tunisia", "tuinsi", "usa", "uk", "india", "ind", "handi",
))

BASE_HRU_PHRASES = (
    # Standard English
    "how are you", "how are you doing", "how are you today", "how are you doing today",
    "how was your day", "how is your day", "how is it going", "how's it going",
    "hows it going", "how have you been", "how do you do", "how are things",
    "how is everything", "hows everything", "how are you feeling",
    # Chat Slang
    "hru", "hyd", "hru doing", "hru doin", "hru today", "hyd today",
    "hows u", "how's u", "how r u", "how are u", "how are ya",
    "how u doing", "how u doin", "how r u doing", "how r u doin",
    "how are u doing", "how are u doin", "hows ur day", "how's ur day",
    "how is ur day", "how was ur day", "how have u been", "how u been",
    "wbu", "hbu", "u good", "are u good", "are you good", "u ok", "are u ok", "are you okay",
    # Realistic Keyboard Typos
    "hor", "how ar you dopin", "how ar you doing", "how ar u doin", "how ar u",
    "hw r u", "hw are u", "hw are you", "how u diong", "how are you diong",
    "how whas you da", "how whas your day", "how was you day", "how is ur da",
    "how r yu", "how are yuo", "how r yuo", "hows it goin", "how's it goin",
    "how is it goin", "how re u", "hoiw are you", "how aer you", "how aare you",
)

GREETING_PREFIXES = (
    "", "hi", "hey", "hello", "yo", "sup", "hii", "heyy", "helloo", "hiya", "howdy",
)

HRU_TARGETS = (
    "", "today", "rn", "there", "bro", "friend", "stranger", "cutie",
)

PUNCTUATION_ENDINGS = ("", "?", "!", "?!", "!!", ".")


def is_valid_hru_line(text):
    if not isinstance(text, str):
        return ""
    s = SPACE_RE.sub(" ", text.strip().lower())
    if not s or len(s) > 38:
        return ""
    if not CLEAN_HRU_CHARS_RE.match(s):
        return ""
    words = s.split()
    if len(words) == 0 or len(words) > 6:
        return ""
    for w in words:
        if w.strip("'!?.,") in HRU_FORBIDDEN_WORDS:
            return ""
    if not STRICT_HRU_RE.match(s):
        return ""
    return s


def pull_hf_and_jsonl_hru(hru_set):
    try:
        from datasets import load_dataset
        for repo, split, col in (
            ("bavard/personachat_truecased", "train", "history"),
            ("AlekseyKorshuk/persona-chat", "train", "utterances"),
            ("tanaos/synthetic-intent-classifier-dataset-v1", "train", "text"),
        ):
            try:
                ds = load_dataset(repo, split=split)
                if col not in ds.column_names:
                    continue
                for item in ds[col]:
                    candidates = item if isinstance(item, list) else (str(item),)
                    for raw_s in candidates:
                        c = is_valid_hru_line(raw_s)
                        if c:
                            hru_set.add(c)
            except Exception:
                pass
    except ImportError:
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
                    c = is_valid_hru_line(str(rec.get("text", "")))
                    if c:
                        hru_set.add(c)
        except Exception:
            pass


def build_hru_file(output_path="hru.txt"):
    hru_set = set()
    pull_hf_and_jsonl_hru(hru_set)

    # 1. Standalone status / hru phrases + punctuation (!, ?, ?!, !!, .) + targets
    for phrase in BASE_HRU_PHRASES:
        for end in PUNCTUATION_ENDINGS:
            c = is_valid_hru_line(f"{phrase}{end}")
            if c:
                hru_set.add(c)
        for tgt in HRU_TARGETS:
            if tgt and tgt not in phrase:
                for end in ("", "?", "!"):
                    c = is_valid_hru_line(f"{phrase} {tgt}{end}")
                    if c:
                        hru_set.add(c)

    # 2. Combined Greeting + Status Question (e.g. "hi how r u", "hey hru", "hello how are you doing!")
    for pfx in GREETING_PREFIXES:
        if not pfx:
            continue
        for phrase in BASE_HRU_PHRASES:
            if phrase in ("wbu", "hbu", "hor"):
                continue
            combined = f"{pfx} {phrase}".strip()
            for end in ("", "?", "!"):
                c = is_valid_hru_line(f"{combined}{end}")
                if c:
                    hru_set.add(c)

    sorted_hru = sorted(hru_set)
    payload = ("\n".join(sorted_hru) + "\n").encode("ascii", errors="ignore")
    out_u8 = np.frombuffer(payload, dtype=np.uint8)
    with open(output_path, "wb") as fh:
        fh.write(out_u8.tobytes())

    n_lines = np.int32(len(sorted_hru))
    n_kb = np.float32(out_u8.nbytes) / np.float32(1024.0)
    print(f"[CLEAN HRU] Overwrote {int(n_lines)} pure ask_how_are_you lines to '{output_path}' ({float(n_kb):.2f} KB)", flush=True)


if __name__ == "__main__":
    build_hru_file()
