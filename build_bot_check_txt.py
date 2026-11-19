#!/usr/bin/env python3
# FILE: build_bot_check_txt.py
# PURPOSE: Zero-CVE Multi-Dataset Builder for Step 14:
#          - bot_check_is_bot.txt (Class 33: ask_is_bot_fake -> Bot replies NO, I'm not a bot!)
#          - bot_check_is_real.txt (Class 34: ask_is_real_human -> Bot replies YES, I'm 100% real!)
# RULES: Enforces Rules 1-13 (Parquet-only fetch, 7-bit ASCII byte strip, closed vocab whitelist,
#        zero "!" / filler bloat, protects Step 7 "for real"/"real", Step 4 gender, Step 9 "real name", Class 0 "who are you").

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
# Protects Step 7 ("for real", bare "real"), Step 4 ("are you a girl?"), Step 9 ("whats ur real name"),
# Step 8 ("looking for..."), Step 11 ("single/married"), and Class 0 ("who are you", "what is this").
STEP1_TO_13_COLLISION_RE = re.compile(
    r"^(?:"
    r"(?:f+o+r+\s+r+e+a+l+|r+e+a+l+)|"
    r"(?:w+h+o+\s+(?:a+r+e+|r+)\s+(?:y+o+u+|u+)|w+h+a+t+\s+(?:a+r+e+|r+)\s+(?:y+o+u+|u+))|"
    r"(?:(?:a+r+e+|r+)\s+(?:y+o+u+|u+)\s+(?:a+\s+)?(?:g+i+r+l+|b+o+y+|m+a+l+e+|f+e+m+a+l+e+|w+o+m+a+n+|m+a+n+))|"
    r".*\b(?:n+a+m+e+|h+o+b+b*y+|m+u+s+i+c+|g+a+m+e+s*|a+n+i+m+e+|s+i+n+g+e*l+e*|m+a+r+i+e*d+|t+a+k+e*n+)\b.*|"
    r".*\b(?:p+i+c+s*|p+h+o+t+o+s*|s+e+l+f+i+e*s*|t+e+l+e*g+r+a*m+|s+n+a+p+|i+n+s+t+a+|p+r+i+v+a*t*e*y*)\b.*|"
    r".*\b(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|f+r+i+e+n+d+s*|g+f+|b+f+|d+a+t+i+n+g+|r+e+l+a+t+i+o+n+s+h+i+p+|o+l+d+|a+g+e+|f+r+o+m+|c+o+u+n+t+r+y+)\b.*"
    r")$",
    re.IGNORECASE,
)

# Closed Step 14 Vocabulary (Exact match with step14_bot_check.py LUTs)
S14_BOT_FAKE_WORDS = (
    "bot", "bott", "bots", "ai", "robot", "robt", "roboot", "fake", "fke", "fak",
    "catfish", "scam", "scammer", "scamer", "script", "program", "programmed",
    "automated", "auto", "chatgpt", "gpt", "machine", "computer", "npc",
)
S14_REAL_HUMAN_WORDS = (
    "real", "reel", "human", "humn", "humen", "person", "persn", "alive", "actual", "legit",
)
S14_PROVE_WORDS = ("prove", "proof", "verify", "sound", "talk", "reply", "answer", "act", "seem")
S14_QUESTION_WORDS = ("are", "ar", "r", "is", "am", "do", "why", "or")
S14_FRAME_WORDS = (
    "u", "you", "ur", "your", "youre", "you're", "i", "a", "an", "like", "not",
    "talking", "chatting", "to", "with", "girl", "boy", "guy", "woman", "man",
    "female", "male", "fast", "so", "too", "really", "actually", "even", "just",
    "wait", "hey", "hi", "bro", "me", "that", "it",
)

STRICT_ALLOWED_S14_VOCAB = frozenset(
    S14_BOT_FAKE_WORDS + S14_REAL_HUMAN_WORDS + S14_PROVE_WORDS + S14_QUESTION_WORDS + S14_FRAME_WORDS
)

# Strictly matches Class 33 (ask_is_bot_fake -> Bot replies NO!)
STRICT_IS_BOT_RE = re.compile(
    r"^(?:(?:w+a+i+t+|he+y+|hi+|s+o+)\s+)?"
    r"(?:"
    r"(?:(?:a+r+e+|a+r+|r+)\s+(?:y+o+u+|u+)|(?:y+o+u+|u+|u+r+|y+o+u+'?r+e+))\s+"
    r"(?:a+\s+|a+n+\s+|j+u+s+t+\s+|r+e+a+l+y+\s+)?(?:"
    r"b+o+t+s*|a+i+|r+o+b+o*t+|f+a+k+e*|c+a+t+f+i+s+h+|s+c+a+m+e*r*|s+c+r+i+p+t+|"
    r"p+r+o+g+r+a+m+e*d*|a+u+t+o+m+a+t+e+d+|c+h+a+t+g+p+t+|g+p+t+|m+a+c+h+i+n+e+|c+o+m+p+u+t+e+r+|n+p+c+"
    r")(?:\s+o+r+\s+(?:r+e+a+l+|h+u+m+a*n+|p+e+r+s+o+n+))?|"
    r"(?:(?:a+r+e+|a+r+|r+)\s+(?:y+o+u+|u+)|(?:y+o+u+|u+))\s+(?:r+e+a+l+|h+u+m+a*n+)\s+o+r+\s+"
    r"(?:a+\s+)?(?:b+o+t+|a+i+|r+o+b+o*t+|f+a+k+e+)|"
    r"(?:b+o+t+|a+i+|r+o+b+o*t+|f+a+k+e+)\s+o+r+\s+(?:r+e+a+l+|h+u+m+a*n+|p+e+r+s+o+n+)|"
    r"(?:r+e+a+l+|h+u+m+a*n+)\s+o+r+\s+(?:b+o+t+|a+i+|r+o+b+o*t+|f+a+k+e+)|"
    r"(?:a+m+\s+i+\s+t+a+l+k+i+n+g+\s+t+o+|i+s+\s+(?:t+h+i+s+|i+t+))\s+(?:a+\s+|a+n+\s+)?(?:b+o+t+|a+i+|r+o+b+o*t+|f+a+k+e+)|"
    r"(?:y+o+u+|u+)\s+(?:s+o+u+n+d+|t+a+l+k+|r+e+p+l+y+|a+c+t+|s+e+e+m+)\s+l+i+k+e+\s+(?:a+\s+|a+n+\s+)?(?:b+o+t+|a+i+|r+o+b+o*t+)|"
    r"(?:y+o+u+|u+)\s+(?:r+e+p+l+y+|a+n+s+w+e+r+)\s+(?:s+o+|t+o+o+)\s+f+a+s+t+|"
    r"(?:b+o+t+|a+i+|r+o+b+o*t+|f+a+k+e+|c+a+t+f+i+s+h+|s+c+a+m+e*r*|c+h+a+t+g+p+t+)\??"
    r")\??$",
    re.IGNORECASE,
)

# Strictly matches Class 34 (ask_is_real_human -> Bot replies YES!)
STRICT_IS_REAL_RE = re.compile(
    r"^(?:(?:w+a+i+t+|he+y+|hi+|s+o+)\s+)?"
    r"(?:"
    r"(?:(?:a+r+e+|a+r+|r+)\s+(?:y+o+u+|u+)|(?:y+o+u+|u+))\s+"
    r"(?:e+v+e+n+\s+|r+e+a+l+y+\s+|a+c+t+u+a+l+y+\s+)?(?:"
    r"(?:a+\s+)?(?:r+e+a+l+|r+e+e+l+)(?:\s+(?:p+e+r+s+o*n+|h+u+m+a*n+|g+i+r+l+|b+o+y+|g+u+y+|w+o+m+a+n+|m+a+n+|f+e+m+a+l+e+|m+a+l+e+))?|"
    r"(?:a+\s+)?(?:h+u+m+a*n+|h+u+m+e+n+|p+e+r+s+o*n+|a+l+i+v+e+|l+e+g+i+t+)"
    r")|"
    r"(?:p+r+o+v+e+|s+h+o+w+)\s+(?:(?:t+h+a+t+\s+)?(?:y+o+u+\s+a+r+e+|y+o+u+'?r+e+|u+\s+r+|u+r+)|i+t+)\s*(?:r+e+a+l+|h+u+m+a*n+|n+o+t+\s+a+\s+b+o+t+)?|"
    r"(?:r+e+a+l+|h+u+m+a*n+|l+e+g+i+t+)\?+"
    r")\??$",
    re.IGNORECASE,
)

ASK_IS_BOT_BASES = (
    # Direct Bot / AI / Robot / Fake Questions -> Bot replies NO!
    "are you a bot", "are u a bot", "r u a bot", "u a bot", "you a bot",
    "are you bot", "are u bot", "r u bot", "u bot", "ur a bot", "youre a bot", "you're a bot",
    "are you a bott", "are u a bott", "bot?", "bot", "bots?",
    "are you ai", "are u ai", "r u ai", "are you an ai", "are u an ai", "u ai", "ur ai", "ai?",
    "are you a robot", "are u a robot", "r u a robot", "u a robot", "are u robot", "robot?", "robt?",
    "are you fake", "are u fake", "r u fake", "u fake", "you fake", "ur fake", "fake?",
    "are you a catfish", "are u a catfish", "r u catfish", "catfish?",
    "are you a scam", "are u a scam", "are u a scammer", "scam?", "scammer?",
    "are you chatgpt", "are u chatgpt", "r u chatgpt", "chatgpt?", "gpt?",
    "are you a script", "are u automated", "are u a computer", "are u a machine", "are u an npc",
    "is this a bot", "is it a bot", "is this ai", "am i talking to a bot", "am i talking to ai",
    "u sound like a bot", "you sound like a bot", "u talk like a bot", "u act like a bot",
    "u seem like a bot", "u reply like a bot", "u reply so fast", "u reply too fast",
    "you answer so fast", "u answer too fast",
    # Choice Questions ("bot or real?", "real or fake?") -> Bot replies NO, I'm real!
    "bot or real", "real or bot", "bot or human", "human or bot",
    "real or fake", "fake or real", "ai or real", "ai or human",
    "are u real or a bot", "are you real or a bot", "are u a bot or real",
    "are you a bot or human", "are u real or fake", "are you real or fake",
)

ASK_IS_REAL_BASES = (
    # Direct Real / Human / Person Questions -> Bot replies YES!
    "are you real", "are u real", "r u real", "u real", "you real",
    "are you reel", "are u reel", "are you actually real", "are u really real", "are u even real",
    "real?", "human?", "legit?",
    "are you human", "are u human", "r u human", "u human", "you human",
    "are you a human", "are u a human", "r u a human", "are u humen",
    "are you a real person", "are u a real person", "r u a real person", "u a real person",
    "are you a person", "are u a person", "are u a real human",
    "are you a real girl", "are u a real girl", "r u a real girl",
    "are you a real boy", "are u a real boy", "are u a real guy",
    "are you a real woman", "are u a real woman", "are u a real female",
    "are you legit", "are u legit", "are you alive", "are u alive",
    "prove ur real", "prove u r real", "prove you are real", "prove youre real",
    "prove ur human", "prove ur not a bot",
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

    if STEP1_TO_13_COLLISION_RE.match(s):
        return 1, s

    words = [w.strip("'?,") for w in s.split() if w.strip("'?,")]
    if not words or len(words) > 6:
        return 0, ""

    for w in words:
        if w not in STRICT_ALLOWED_S14_VOCAB:
            return 0, ""

    if STRICT_IS_BOT_RE.match(s):
        return 33, s

    if STRICT_IS_REAL_RE.match(s):
        return 34, s

    return 0, ""


def _safe_fetch_parquet_bot_lines(bot_map, real_map):
    # Rule 12: Streams ONLY pure Parquet data (revision="refs/convert/parquet", zero remote .py scripts!)
    stats = {"scanned": 0, "blocked_overlaps": 0, "bot_added": 0, "real_added": 0}
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
                    for sent in re.split(r"[.!]+", turn[:100]):
                        cid, clean_s = categorize_line_across_all_steps(sent)
                        if 1 <= cid <= 32:
                            stats["blocked_overlaps"] += 1
                        elif cid == 33:
                            sig = _canonical_token_sig(clean_s)
                            if sig not in bot_map:
                                bot_map[sig] = clean_s
                                stats["bot_added"] += 1
                        elif cid == 34:
                            sig = _canonical_token_sig(clean_s)
                            if sig not in real_map:
                                real_map[sig] = clean_s
                                stats["real_added"] += 1
        except Exception:
            pass
    return stats


def _write_u8_file(path, lines_iterable):
    sorted_lines = sorted(set(lines_iterable))
    u8 = np.frombuffer(("\n".join(sorted_lines) + "\n").encode("ascii"), dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(u8.tobytes())
    return len(sorted_lines), u8.nbytes


def build_bot_check_files(
    bot_path="bot_check_is_bot.txt",
    real_path="bot_check_is_real.txt",
    neg_path="negatives.txt",
):
    bot_map = {}
    real_map = {}

    for base in ASK_IS_BOT_BASES:
        if "?" in base:
            cid, c = categorize_line_across_all_steps(base)
            if cid == 33:
                bot_map[_canonical_token_sig(c)] = c
            continue
        for pfx in ("", "wait", "hey"):
            full = f"{pfx} {base}".strip() if pfx else base
            for end in ("", "?"):
                cid, c = categorize_line_across_all_steps(f"{full}{end}")
                if cid == 33:
                    sig = _canonical_token_sig(c)
                    if sig not in bot_map:
                        bot_map[sig] = c

    for base in ASK_IS_REAL_BASES:
        if "?" in base:
            cid, c = categorize_line_across_all_steps(base)
            if cid == 34:
                real_map[_canonical_token_sig(c)] = c
            continue
        for pfx in ("", "wait", "hey"):
            full = f"{pfx} {base}".strip() if pfx else base
            for end in ("", "?"):
                cid, c = categorize_line_across_all_steps(f"{full}{end}")
                if cid == 34:
                    sig = _canonical_token_sig(c)
                    if sig not in real_map:
                        real_map[sig] = c

    hf_stats = _safe_fetch_parquet_bot_lines(bot_map, real_map)

    for k_sig in list(real_map.keys()):
        if k_sig in bot_map:
            del real_map[k_sig]

    n_bot, b_bot = _write_u8_file(bot_path, bot_map.values())
    n_real, b_real = _write_u8_file(real_path, real_map.values())

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn:
            existing_neg = {line.strip().lower() for line in fn if line.strip()}
    for item in list(bot_map.values()) + list(real_map.values()):
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ")):
            existing_neg.add(item)
    _, _ = _write_u8_file(neg_path, existing_neg)

    print(
        f"[PARQUET STEP 14 ANALYZER] Scanned {hf_stats['scanned']} rows | "
        f"Blocked {hf_stats['blocked_overlaps']} Step 1-13 overlaps | "
        f"Added {hf_stats['bot_added']} is_bot & {hf_stats['real_added']} is_real lines!",
        flush=True,
    )
    print(
        f"[CLEAN STEP 14 BOT-CHECK RULES 12+13] Saved {n_bot} unique is_bot lines to '{bot_path}' ({b_bot / 1024.0:.2f} KB) "
        f"& {n_real} unique is_real lines to '{real_path}' ({b_real / 1024.0:.2f} KB)",
        flush=True,
    )


if __name__ == "__main__":
    build_bot_check_files()