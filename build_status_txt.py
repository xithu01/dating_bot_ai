#!/usr/bin/env python3
# FILE: build_status_txt.py
# PURPOSE: Build Step 6 Status & Activity Reply datasets (Classes 16, 17, 18) with 0 collisions against Steps 1-5 & 7-8
# RULES: Enforces Rules 1-13:
#        - Rule 12: Parquet-only fetch (revision="refs/convert/parquet"), 7-bit ASCII byte strip, closed vocab whitelist
#        - Rule 13: Zero "!" / "lol" / "haha" Cartesian filler bloat, cross-step analyzer, canonical deduplication

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

CLEAN_STATUS_RE = re.compile(r"^[a-z\s'?,]+$")
SPACE_RE = re.compile(r"\s+")

# Rule 13 Cross-Step Collision Guard:
# Blocks Step 1 ("good morning"), Step 3 ("how are you"), Step 5 ("wyd"),
# Step 7 standalone reactions ("ok", "okay", "alright", "awesome", "amazing", "same", "lol", "haha"),
# and Step 8 ("here to chat", "looking to chat", "looking for friends")
STEP_OTHER_COLLISION_RE = re.compile(
    r"^(?:"
    r"(?:g+o+o+d+\s+(?:m+o+r+n+i+n+g+|a+f+t+e+r+n+o+o+n+|e+v+e+n+i+n+g+|d+a+y+|n+i+g+h+t+|l+u+c+k+|j+o+b+))|"
    r"(?:h+o+w+\s+(?:a+r+e+|r+)\s+(?:y+o+u+|u+).*|h+r+u+|h+y+d+|w+y+d+|w+u+d+|w+h+a+t+\s+(?:a+r+e+|r+)?\s*(?:y+o+u+|u+)\s+d+o+i+n+g*)|"
    r"(?:o+k+|o+k+a+y+|a+l+r+i+g+h+t+|a+w+e+s+o+m+e+|a+m+a+z+i+n+g+|s+a+m+e+|s+a+m+|h+a+h+a+|l+o+l+)|"
    r"(?:(?:j+u+s+t+\s+)?(?:h+e+r+e+|l+o+k+i+n+g*)\s+t+o+\s+(?:c+h+a+t+|t+a+l+k+))|"
    r".*\b(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|s+e+e+k+i+n+g*|f+r+i+e+n+d+s*|g+f+|b+f+|d+a+t+i+n+g+|r+e+l+a+t+i+o+n+s+h+i+p+)\b.*|"
    r".*\b(?:o+l+d+|a+g+e+|f+r+o+m+|c+o+u+n+t+r+y+|w+h+e+r+e+|w+h+y+|w+h+o+|m+a+l+e+|f+e+m+a+l+e+|b+o+y+|g+i+r+l+|g+e+n+d+e+r+|s+e+x+)\b.*"
    r")$",
    re.IGNORECASE,
)

# Rule 12 Gate 2: Closed Whitelist Vocabulary for Step 6 (Exact match with offline_train.py STRICT_S6_VOCAB)
STATUS_POS_ANCHOR_WORDS = (
    "good", "great", "fine", "alright", "well", "ok", "okay", "awesome", "amazing", "fantastic", "happy",
    "chat", "chatting", "chatin", "chating", "talking", "chilling", "chillin", "chiling",
    "relaxing", "resting", "bed", "laying", "lying", "home", "working", "work", "busy",
    "studying", "reading", "listening", "music", "watching", "tv", "movie", "movies",
    "youtube", "netflix", "gaming", "playing", "games", "game", "scrolling", "phone",
    "eating", "cooking", "drinking", "coffee",
)
STATUS_NEG_ANCHOR_WORDS = (
    "bored", "borad", "boerd", "borde", "bord", "broed", "tired", "tierd", "tird",
    "sleepy", "exhausted", "sad", "lonely", "bad", "sick", "stressed",
    "nothing", "nothin", "nth", "nada", "nm",
)
ASK_BACK_TOKENS = frozenset(("u", "you", "wbu", "hbu", "nmu", "hru", "hyd", "wyd", "wud", "about"))
STATUS_FRAME_WORDS = (
    "im", "i'm", "l'm", "lm", "iam", "i'am", "i", "am", "doing", "doin", "diong", "just",
    "so", "very", "super", "hella", "feeling", "all", "pretty", "not", "much", "really",
    "special", "anything", "same", "in", "at", "on", "my", "to", "a", "bit", "down",
    "here", "staying", "cant", "can't", "sleep", "rn", "now", "today", "thanks", "thx",
    "ty", "lol", "haha", "and", "how", "what", "about", "hi", "hey", "hello", "yo",
)

STRICT_ALLOWED_S6_VOCAB = frozenset(
    STATUS_POS_ANCHOR_WORDS + STATUS_NEG_ANCHOR_WORDS + tuple(ASK_BACK_TOKENS) + STATUS_FRAME_WORDS
)

# 1. Class 16: Positive Moods & Active/Chill Activities (Zero Cartesian filler bloat)
POSITIVE_SUBJECT_PREFIXES = (
    "", "im ", "i'm ", "i am ", "lm ", "l'm ", "iam ", "hi im ", "hey im ",
)

# Note: bare "ok", "okay", "alright", "awesome", "amazing" require a subject/verb prefix so bare "ok" stays in Step 7 Class 20
PREFIX_REQUIRED_MOODS = frozenset(("ok", "okay", "alright", "awesome", "amazing", "fantastic", "happy"))

POSITIVE_MOOD_CORES = (
    "good", "doing good", "doin good", "diong good", "great", "doing great", "doin great",
    "fine", "doing fine", "all good", "pretty good", "very good", "so good",
    "alright", "doing alright", "well", "doing well",
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

# 2. Class 17: Negative, Bored, Tired, or "Nothing Much" States
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
    "doing nothing", "doin nothing", "just nothing",
    "nothing much just bored", "nothing much just borad", "not much just bored", "not much just borad",
    "nothing just bored", "nothing just borad", "nothing much just chilling", "not much just chilling",
    "nothing much just in bed", "not much just in bed", "chilling and bored", "in bed and bored",
    "in bed bored", "just in bed bored", "laying in bed bored",
)

# 3. Class 18: Status / Activity + Ask-Back ("im good u?", "lm borad u?", "nothing much wbu?", "nmu")
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
    "u", "u?", "you?", "wbu", "wbu?", "hbu?",
    "and u?", "and you?", "how about u?", "what about u?", "hru?", "wyd?",
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


def _clean_status_line(raw_text, allow_ask_back=False):
    clean_ascii = _sanitize_ascii_bytes(raw_text)
    if "/" in clean_ascii:
        return ""
    s = SPACE_RE.sub(" ", clean_ascii.strip().lower()).rstrip("!.")
    if not s or len(s) > 42 or not CLEAN_STATUS_RE.match(s):
        return ""
    if STEP_OTHER_COLLISION_RE.match(s):
        return ""
    words = [w.strip("'?,") for w in s.split() if w.strip("'?,")]
    if not words or len(words) > 6:
        return ""
    if any(w not in STRICT_ALLOWED_S6_VOCAB for w in words):
        return ""
    has_ab = any(w in ASK_BACK_TOKENS for w in words)
    if allow_ask_back and not has_ab:
        return ""
    if not allow_ask_back and (has_ab or "?" in s):
        return ""
    return s


def categorize_line_across_all_steps(raw_text):
    """
    Rule 12 & 13 Multi-Class Analyzer for Step 6 Parquet Stream:
    Returns (16..18, clean_s) or (0, "").
    """
    s_ab = _clean_status_line(raw_text, allow_ask_back=True)
    if s_ab:
        words = [w.strip("'?,") for w in s_ab.split() if w.strip("'?,")]
        has_pos = any(w in STATUS_POS_ANCHOR_WORDS for w in words)
        has_neg = any(w in STATUS_NEG_ANCHOR_WORDS for w in words)
        if has_pos or has_neg or "nmu" in words:
            return 18, s_ab

    s_st = _clean_status_line(raw_text, allow_ask_back=False)
    if s_st:
        words = [w.strip("'?,") for w in s_st.split() if w.strip("'?,")]
        has_pos = any(w in STATUS_POS_ANCHOR_WORDS for w in words)
        has_neg = any(w in STATUS_NEG_ANCHOR_WORDS for w in words)
        if has_neg:
            if "not" in words and "bad" in words and len(words) <= 3:
                return 16, s_st
            return 17, s_st
        if has_pos:
            return 16, s_st
    return 0, ""


def _safe_fetch_parquet_status_lines(pos_map, neg_map, ab_map):
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "daily_dialog",
            split="train",
            revision="refs/convert/parquet",
            streaming=True,
        )
        count = 0
        target_maps = {16: pos_map, 17: neg_map, 18: ab_map}
        for row in ds:
            if count >= 2000:
                break
            count += 1
            for turn in row.get("dialog", ()):
                if not isinstance(turn, str):
                    continue
                for sent in re.split(r"[.!]+", turn[:90]):
                    cid, clean_s = categorize_line_across_all_steps(sent)
                    if cid in target_maps:
                        sig = _canonical_token_sig(clean_s)
                        if sig not in target_maps[cid]:
                            target_maps[cid][sig] = clean_s
    except Exception:
        pass


def _write_u8_file(path, lines_iterable):
    sorted_lines = sorted(set(lines_iterable))
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
    pos_map = {}
    neg_status_map = {}
    ask_back_map = {}

    # 1. Build Class 16: Positive Moods & Active/Chill States (Zero Cartesian "lol"/"haha"/"!" bloat)
    for pfx in POSITIVE_SUBJECT_PREFIXES:
        for core in POSITIVE_MOOD_CORES + POSITIVE_ACTIVITY_CORES:
            if not pfx and core in PREFIX_REQUIRED_MOODS:
                continue
            if pfx and core in ("all good", "not bad", "chat"):
                if not (pfx == "im " and core == "chat"):
                    continue
            base = f"{pfx}{core}".strip()
            for sfx in ("", " rn", " thanks"):
                if sfx == " thanks" and core in POSITIVE_ACTIVITY_CORES:
                    continue
                c = _clean_status_line(f"{base}{sfx}", allow_ask_back=False)
                if c:
                    sig = _canonical_token_sig(c)
                    if sig not in pos_map:
                        pos_map[sig] = c

    # 2. Build Class 17: Negative, Bored, Tired, or "Nothing Much" States
    for pfx in POSITIVE_SUBJECT_PREFIXES:
        for core in NEGATIVE_BORED_TIRED_CORES:
            base = f"{pfx}{core}".strip()
            for sfx in ("", " rn"):
                c = _clean_status_line(f"{base}{sfx}", allow_ask_back=False)
                if c:
                    sig = _canonical_token_sig(c)
                    if sig not in neg_status_map:
                        neg_status_map[sig] = c

    for core in NEGATIVE_NOTHING_STANDALONE:
        for sfx in ("", " rn"):
            c = _clean_status_line(f"{core}{sfx}", allow_ask_back=False)
            if c:
                sig = _canonical_token_sig(c)
                if sig not in neg_status_map:
                    neg_status_map[sig] = c

    # 3. Build Class 18: Status + Ask-Back ("im good u?", "lm borad u?", "nmu")
    for ab_short in ("nmu", "nmu?", "nm u", "nm u?", "nm wbu?"):
        c = _clean_status_line(ab_short, allow_ask_back=True)
        if c:
            ask_back_map[_canonical_token_sig(c)] = c

    for base in ASK_BACK_BASES:
        for tail in ASK_BACK_TAILS:
            c = _clean_status_line(f"{base} {tail}", allow_ask_back=True)
            if c:
                sig = _canonical_token_sig(c)
                if sig not in ask_back_map:
                    ask_back_map[sig] = c

    _safe_fetch_parquet_status_lines(pos_map, neg_status_map, ask_back_map)

    # Disjoint hierarchy: Ask-Back (18) > Negative/Bored/Nothing (17) > Positive (16)
    for k in list(pos_map.keys()):
        if k in neg_status_map or k in ask_back_map:
            del pos_map[k]
    for k in list(neg_status_map.keys()):
        if k in ask_back_map:
            del neg_status_map[k]

    n_pos, b_pos = _write_u8_file(pos_path, pos_map.values())
    n_neg_s, b_neg_s = _write_u8_file(neg_status_path, neg_status_map.values())
    n_ab, b_ab = _write_u8_file(ask_back_path, ask_back_map.values())

    existing_neg = set()
    if os.path.exists(step1_neg_path):
        with open(step1_neg_path, "r", encoding="ascii", errors="ignore") as fn:
            existing_neg = {line.strip().lower() for line in fn if line.strip()}
    all_st = set(pos_map.values()) | set(neg_status_map.values()) | set(ask_back_map.values())
    for item in all_st:
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ", "sup ")):
            existing_neg.add(item)
    _, _ = _write_u8_file(step1_neg_path, existing_neg)

    tot_lines = np.int32(n_pos + n_neg_s + n_ab)
    tot_kb = np.float32(b_pos + b_neg_s + b_ab) / np.float32(1024.0)
    print(
        f"[CLEAN STEP 6 STATUS RULES 12+13] Saved {int(tot_lines)} unique lines ({float(tot_kb):.2f} KB): "
        f"pos={int(n_pos)}, neg={int(n_neg_s)}, ask_back={int(n_ab)}",
        flush=True,
    )
    return tot_lines, tot_kb


def generate_all_status_files():
    return build_all_status_files()


if __name__ == "__main__":
    build_all_status_files()