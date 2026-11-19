#!/usr/bin/env python3
# FILE: build_rel_status_txt.py
# PURPOSE: Zero-CVE Multi-Dataset Builder for Step 11:
#          - rel_status_ask.txt (Class 27: ask_relationship_status)
#          - rel_status_tell.txt (Class 28: tell_relationship_status)
# RULES: Enforces Rules 1-13 (Parquet-only fetch, 7-bit ASCII byte strip, closed vocab whitelist,
#        zero "!" / filler bloat, blocks Step 8 "looking for a relationship" & Rule 13 "a bf here" / "a gf here").

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

CLEAN_LINE_RE = re.compile(r"^[a-z\s'?,]+$")
SPACE_RE = re.compile(r"\s+")

# Rule 13 Cross-Step Collision Guard:
# Blocks Step 8 ("looking for a gf/bf/relationship"), Step 4 bare gender ("f here", "m here"),
# Rule 13 false saves ("a bf here", "a gf here"), and Steps 1-7/9-10 words.
STEP1_TO_10_COLLISION_RE = re.compile(
    r"^(?:"
    r".*\b(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|s+e+e+k+i+n+g*|s+e+a+r+c+h+i+n+g*|w+a+n+t+|n+e+e+d+|m+a+k+i+n+g*)\b.*|"
    r"(?:(?:a+\s+)?(?:b+f+|g+f+|b+o+y+f+r+i+e+n+d+|g+i+r+l+f+r+i+e+n+d+|r+e+l+a+t+i+o+n+s+h+i+p+)(?:\s+h+e+r+e+)?)|"
    r".*\b(?:h+r+u+|h+y+d+|w+y+d+|w+u+d+|b+o+r+e+d+|c+h+i+l+i+n+g*|o+l+d+|a+g+e+|f+r+o+m+|c+o+u+n+t+r+y+|n+a+m+e+|h+o+b+b*y+|m+u+s+i+c+|g+a+m+e+s*|a+n+i+m+e+)\b.*"
    r")[\s?]*$",
    re.IGNORECASE,
)

# Closed Step 11 Vocabulary (Exact match with step11_rel_status.py LUTs)
S11_SINGLE_WORDS = (
    "single", "singel", "singl", "singe", "sngle", "available", "unmarried",
    "divorced", "divorcd", "widowed", "alone",
)
S11_MARRIED_WORDS = (
    "married", "maried", "mariid", "marrid", "marid", "marryed", "engaged", "engagd",
    "wife", "husband", "hubby", "wifey", "spouse",
)
S11_TAKEN_WORDS = (
    "taken", "takn", "dating", "relationship", "relashionship", "committed",
    "bf", "boyfriend", "boifriend", "gf", "girlfriend", "grilfriend", "partner",
)
S11_STATUS_ANCHORS = ("status",)
S11_HAVE_NEG_WORDS = ("have", "hav", "got", "dont", "don't", "do", "not", "no", "currently", "still")
S11_QUESTION_WORDS = ("are", "ar", "r", "is", "what", "whats", "what's", "wat", "or")
S11_STATEMENT_WORDS = ("im", "i'm", "l'm", "lm", "iam", "i'am", "i", "am", "my", "m", "f")
S11_FRAME_WORDS = (
    "you", "u", "ya", "your", "ur", "in", "a", "an", "any", "right", "now", "rn",
    "here", "wbu", "hbu", "and", "so", "btw", "hi", "hey", "actually", "just",
)

STRICT_ALLOWED_S11_VOCAB = frozenset(
    S11_SINGLE_WORDS + S11_MARRIED_WORDS + S11_TAKEN_WORDS + S11_STATUS_ANCHORS
    + S11_HAVE_NEG_WORDS + S11_QUESTION_WORDS + S11_STATEMENT_WORDS + S11_FRAME_WORDS
)

# Strictly matches Class 27 (ask_relationship_status)
STRICT_ASK_REL_RE = re.compile(
    r"^(?:(?:hi+|he+y+|so+|b+t+w+|w+b+u+|h+b+u+|a+n+d+)\s+)?"
    r"(?:"
    r"(?:(?:a+r+e+|a+r+|r+)\s+(?:y+o+u+|u+|y+a+)|(?:y+o+u+|u+))\s+"
    r"(?:s+t+i+l+l*\s+)?(?:"
    r"(?:s+i+n+g+e*l+e*|s+n+g+l+e+|m+a+r+i+e*d+|m+a+r+y+e+d+|t+a+k+e*n+|d+i+v+o+r+c+e*d+|e+n+g+a+g+e*d+|a+v+a+i+l+a+b+l+e+)"
    r"(?:\s+o+r+\s+(?:s+i+n+g+e*l+e*|m+a+r+i+e*d+|m+a+r+y+e+d+|t+a+k+e*n+|d+i+v+o+r+c+e*d+|i+n+\s+a+\s+r+e+l+a+t*s*h*i+o+n+s+h+i+p+))?|"
    r"i+n+\s+a+\s+r+e+l+a+t*s*h*i+o+n+s+h+i+p+"
    r")|"
    r"(?:s+i+n+g+e*l+e*|m+a+r+i+e*d+|t+a+k+e*n+)\s+o+r+\s+"
    r"(?:s+i+n+g+e*l+e*|m+a+r+i+e*d+|t+a+k+e*n+|i+n+\s+a+\s+r+e+l+a+t*s*h*i+o+n+s+h+i+p+)|"
    r"(?:d+o+\s+(?:y+o+u+|u+)|(?:y+o+u+|u+))\s+(?:h+a+v+e*|g+o+t+)\s+a+\s+"
    r"(?:b+f+|g+f+|b+o+y+f+r+i+e+n+d+|g+i+r+l+f+r+i+e+n+d+|w+i+f+e+|h+u+s+b+a+n+d+|p+a+r+t+n+e+r+)|"
    r"g+o+t+\s+a+\s+(?:b+f+|g+f+|b+o+y+f+r+i+e+n+d+|g+i+r+l+f+r+i+e+n+d+|w+i+f+e+|h+u+s+b+a+n+d+)|"
    r"(?:w+h+a+t+'?s|w+h+a+t+s|w+h+a+t+\s+i+s+)\s+(?:y+o+u+r+|u+r+)\s+r+e+l+a+t*s*h*i+o+n+s+h+i+p+\s+s+t+a+t+u+s+|"
    r"(?:y+o+u+r+|u+r+)?\s*r+e+l+a+t*s*h*i+o+n+s+h+i+p+\s+s+t+a+t+u+s+|"
    r"(?:s+i+n+g+e*l+e*|m+a+r+i+e*d+|t+a+k+e*n+|d+i+v+o+r+c+e*d+)\?"
    r")\??$",
    re.IGNORECASE,
)

# Strictly matches Class 28 (tell_relationship_status)
STRICT_TELL_REL_RE = re.compile(
    r"^(?:"
    r"(?:(?:i+m+|i+'?m+|l+'?m+|l+m+|i+\s+a+m+|m+|f+)\s+)?(?:s+t+i+l+l*\s+|a+c+t+u+a+l+y+\s+|n+o+t+\s+)?"
    r"(?:s+i+n+g+e*l+e*|s+n+g+l+e+|m+a+r+i+e*d+|m+a+r+y+e+d+|t+a+k+e*n+|d+i+v+o+r+c+e*d+|e+n+g+a+g+e*d+|a+v+a+i+l+a+b+l+e+|u+n+m+a+r+i+e+d+)"
    r"(?:\s+h+e+r+e+)?|"
    r"(?:(?:i+m+|i+'?m+|l+m+|i+\s+a+m+)\s+)?(?:n+o+t+\s+)?i+n+\s+a+\s+r+e+l+a+t*s*h*i+o+n+s+h+i+p+|"
    r"(?:i+\s+)?(?:d+o+n+'?t+\s+h+a+v+e*|h+a+v+e*|g+o+t+)\s+(?:a+\s+|n+o+\s+)?"
    r"(?:b+f+|g+f+|b+o+y+f+r+i+e+n+d+|g+i+r+l+f+r+i+e+n+d+|w+i+f+e+|h+u+s+b+a+n+d+|p+a+r+t+n+e+r+)|"
    r"n+o+\s+(?:b+f+|g+f+|b+o+y+f+r+i+e+n+d+|g+i+r+l+f+r+i+e+n+d+|w+i+f+e+|h+u+s+b+a+n+d+)"
    r")(?:(?:\s*,\s*|\s+)(?:u+|w+b+u+|h+b+u+|a+n+d+\s+u+)\??)?$",
    re.IGNORECASE,
)

ASK_REL_BASES = (
    # Single questions & typos
    "are you single", "are u single", "r u single", "u single", "you single",
    "are you singel", "are u singel", "r u singel", "u singel", "are u singl",
    "are you still single", "r u still single", "single?", "singel?",
    # Single or Married / Taken choice questions (including user's exact "mariid" typo!)
    "are you single or married", "are u single or married", "r u single or married",
    "u single or married", "are you single or mariid", "are u single or mariid",
    "r u single or mariid", "are you single or maried", "are u single or maried",
    "are u single or marrid", "single or married", "single or mariid", "single or maried",
    "married or single", "mariid or single",
    "are you single or taken", "are u single or taken", "r u single or taken",
    "u single or taken", "single or taken", "taken or single",
    "are you single or in a relationship", "are u single or in a relationship",
    "single or in a relationship",
    # Married / Taken / In a relationship questions
    "are you married", "are u married", "r u married", "u married",
    "are you mariid", "are u mariid", "r u mariid", "are u maried", "r u maried",
    "married?", "mariid?", "maried?",
    "are you taken", "are u taken", "r u taken", "u taken", "taken?",
    "are you engaged", "are u divorced", "divorced?",
    "are you in a relationship", "are u in a relationship", "r u in a relationship",
    "u in a relationship", "are u in a relashionship",
    # Have a BF / GF / Wife / Husband questions
    "do you have a bf", "do u have a bf", "u have a bf", "got a bf",
    "do you have a boyfriend", "do u have a boyfriend", "u have a boyfriend",
    "do you have a gf", "do u have a gf", "u have a gf", "got a gf",
    "do you have a girlfriend", "do u have a girlfriend", "u have a girlfriend",
    "do u have a wife", "do u have a husband", "do u have a partner",
    # Relationship Status questions
    "whats your relationship status", "whats ur relationship status",
    "what is your relationship status", "relationship status", "ur relationship status",
)

CLEAN_TELL_REL_TEMPLATES = (
    # Single / Available / Divorced (aux_id = 1)
    "single", "singel", "im single", "i'm single", "lm single", "i am single",
    "im singel", "lm singel", "im singl", "m single", "f single",
    "single here", "im still single", "im available", "im unmarried",
    "im not married", "not married", "im not taken", "not taken",
    "im not in a relationship", "not in a relationship",
    "i dont have a gf", "i don't have a gf", "dont have a gf", "no gf",
    "i dont have a girlfriend", "no girlfriend",
    "i dont have a bf", "i don't have a bf", "dont have a bf", "no bf",
    "i dont have a boyfriend", "no boyfriend",
    "divorced", "im divorced", "i'm divorced", "lm divorced",
    # Taken / In a Relationship (aux_id = 2)
    "taken", "im taken", "i'm taken", "lm taken", "i am taken",
    "in a relationship", "im in a relationship", "i'm in a relationship",
    "lm in a relationship", "i am in a relationship", "im in a relashionship",
    "i have a gf", "i got a gf", "i have a girlfriend",
    "i have a bf", "i got a bf", "i have a boyfriend", "i have a partner",
    # Married / Engaged (aux_id = 3)
    "married", "maried", "mariid", "im married", "i'm married", "lm married",
    "i am married", "im maried", "lm maried", "im mariid", "lm mariid",
    "im marrid", "im engaged", "i have a wife", "i have a husband",
)


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
    has_q = "?" in s
    clean_w = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in s.lower())
    words = tuple(w for w in _collapse_str(clean_w).split() if w)
    return (words, has_q)


def categorize_line_across_all_steps(raw_text):
    clean_ascii = _sanitize_ascii_bytes(raw_text)
    if "/" in clean_ascii:
        return 0, ""
    s = SPACE_RE.sub(" ", clean_ascii.strip().lower()).rstrip("!.")
    if not s or len(s) > 44 or not CLEAN_LINE_RE.match(s):
        return 0, ""

    if STEP1_TO_10_COLLISION_RE.match(s):
        return 1, s

    words = [w.strip("'?,") for w in s.split() if w.strip("'?,")]
    if not words or len(words) > 6:
        return 0, ""

    for w in words:
        if w not in STRICT_ALLOWED_S11_VOCAB:
            return 0, ""

    if STRICT_ASK_REL_RE.match(s):
        return 27, s

    if len(words) == 1 and "?" in s:
        return 0, ""
    if "or" in words:
        return 0, ""

    if STRICT_TELL_REL_RE.match(s):
        return 28, s

    return 0, ""


def _safe_fetch_parquet_rel_lines(ask_map, tell_map):
    # Rule 12: Streams ONLY pure Parquet data (revision="refs/convert/parquet", zero remote .py scripts!)
    stats = {"scanned": 0, "blocked_overlaps": 0, "ask_added": 0, "tell_added": 0}
    try:
        from datasets import load_dataset
    except ImportError:
        return stats

    dialog_repos = (
        ("daily_dialog", "refs/convert/parquet", "dialog", 2500),
        ("bavard/personachat_truecased", "refs/convert/parquet", "history", 1500),
        ("AlekseyKorshuk/persona-chat", "refs/convert/parquet", "utterances", 1500),
    )

    for repo, rev, col, max_rows in dialog_repos:
        try:
            ds = load_dataset(repo, split="train", revision=rev, streaming=True)
            c = 0
            for row in ds:
                if c >= max_rows:
                    break
                c += 1
                stats["scanned"] += 1
                val = row.get(col, ())
                turns = (val,) if isinstance(val, str) else (val if isinstance(val, (list, tuple)) else ())
                for turn in turns:
                    if not isinstance(turn, str):
                        continue
                    for sent in re.split(r"[.!]+", turn[:110]):
                        cid, clean_s = categorize_line_across_all_steps(sent)
                        if 1 <= cid <= 26:
                            stats["blocked_overlaps"] += 1
                        elif cid == 27:
                            sig = _canonical_token_sig(clean_s)
                            if sig not in ask_map:
                                ask_map[sig] = clean_s
                                stats["ask_added"] += 1
                        elif cid == 28:
                            sig = _canonical_token_sig(clean_s)
                            if sig not in tell_map:
                                tell_map[sig] = clean_s
                                stats["tell_added"] += 1
        except Exception:
            pass
    return stats


def _write_u8_file(path, lines_iterable):
    sorted_lines = sorted(set(lines_iterable))
    u8 = np.frombuffer(("\n".join(sorted_lines) + "\n").encode("ascii"), dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(u8.tobytes())
    return len(sorted_lines), u8.nbytes


def build_rel_status_files(
    ask_path="rel_status_ask.txt",
    tell_path="rel_status_tell.txt",
    neg_path="negatives.txt",
):
    ask_map = {}
    tell_map = {}

    for base in ASK_REL_BASES:
        if "?" in base:
            cid, c = categorize_line_across_all_steps(base)
            if cid == 27:
                ask_map[_canonical_token_sig(c)] = c
            continue
        for pfx in ("", "hey", "so", "btw", "wbu"):
            full = f"{pfx} {base}".strip() if pfx else base
            for end in ("", "?"):
                cid, c = categorize_line_across_all_steps(f"{full}{end}")
                if cid == 27:
                    sig = _canonical_token_sig(c)
                    if sig not in ask_map:
                        ask_map[sig] = c

    for tmpl in CLEAN_TELL_REL_TEMPLATES:
        for sfx in ("", " , u", " u?", " wbu?"):
            cand = f"{tmpl}{sfx}".strip()
            cid, c = categorize_line_across_all_steps(cand)
            if cid == 28:
                sig = _canonical_token_sig(c)
                if sig not in tell_map:
                    tell_map[sig] = c

    hf_stats = _safe_fetch_parquet_rel_lines(ask_map, tell_map)

    for k_sig in list(tell_map.keys()):
        if k_sig in ask_map:
            del tell_map[k_sig]

    n_ask, b_ask = _write_u8_file(ask_path, ask_map.values())
    n_tell, b_tell = _write_u8_file(tell_path, tell_map.values())

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn:
            existing_neg = {line.strip().lower() for line in fn if line.strip()}
    for item in list(ask_map.values()) + list(tell_map.values()):
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ")):
            existing_neg.add(item)
    _, _ = _write_u8_file(neg_path, existing_neg)

    print(
        f"[PARQUET REL-STATUS ANALYZER] Scanned {hf_stats['scanned']} rows | "
        f"Blocked {hf_stats['blocked_overlaps']} Step 1-10 overlaps | "
        f"Added {hf_stats['ask_added']} ask & {hf_stats['tell_added']} tell lines!",
        flush=True,
    )
    print(
        f"[CLEAN STEP 11 REL-STATUS RULES 12+13] Saved {n_ask} unique ask lines to '{ask_path}' ({b_ask / 1024.0:.2f} KB) "
        f"& {n_tell} unique tell lines to '{tell_path}' ({b_tell / 1024.0:.2f} KB)",
        flush=True,
    )


if __name__ == "__main__":
    build_rel_status_files()