#!/usr/bin/env python3
# FILE: build_step18_behavior_txt.py
# PURPOSE: Build Step 18 Human Behavior NLU datasets (Classes 56-61, Leaves 97-102):
#          s18_insist.txt, s18_state_city.txt, s18_ntmu.txt, s18_dry.txt, s18_horny.txt, s18_secret.txt
# RULES: Enforces Rules 1-13:
#        - Rule 12: HF_DATASETS_TRUST_REMOTE_CODE="0", Parquet-only fetch, 3-stage sanitization, zero .npz access
#        - Rule 13: Zero Cartesian filler bloat, cross-step collision shield, canonical deduplication

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

CLEAN_ASCII_RE = re.compile(r"^[a-z\s'?]+$")
SPACE_RE = re.compile(r"\s+")

# Protects existing Steps 1-17 Zero-Fault anchors from ever being duplicated into Step 18 .txt files
OLD_SPEC_PROTECTED = frozenset((
    "nice to meet u", "nice to meet you", "pleased to meet you", "good to see you",
    "ok", "okay", "kk", "k", "idk", "maybe", "u first", "you first", "send first",
    "tell me about urself", "tell me about yourself", "tell me a fun fact",
    "send nudes", "wanna sext", "why do u ask", "u reply too fast", "u reply so fast",
))

# Cross-Step Collision Guard (Blocks Steps 1-17 unrelated domains)
STEP1_TO_17_COLLISION_RE = re.compile(
    r"\b(?:hru|hyd|wyd|wud|age|old|gender|male|female|famle|boy|girl|gril|"
    r"snap|insta|telegram|telgrm|whatsapp|nudes|sext|selfie|photo|pic|bot|chatgpt|"
    r"valorant|minecraft|anime|pizza|sushi|zodiac|scorpio|joke|weather|fast|instant)\b",
    re.IGNORECASE,
)

RAW_INSIST = (
    "you didnt answer me", "you don't answer me", "u didnt answer me", "u didn't answer me",
    "u dont answer me", "u don't answer me", "u didnt answer", "you didnt answer",
    "u didnt reply", "u didn't reply", "you didnt reply", "you didn't reply",
    "u never answered", "u never replied", "why wont u answer", "why wont you answer",
    "why wont you tell me", "why won't you tell me", "why wont u tell me", "why didnt u answer",
    "answer my question", "answer the question", "please answer my question", "plz answer my question",
    "pls answer me", "answer me", "answer me first", "answer first", "reply to me",
    "you ignored me", "u ignored me", "u ignored my question", "you ignored my question",
    "dont ignore me", "don't ignore me", "stop ignoring me", "why u ignoring me",
    "you dodged my question", "u dodged my question", "u avoided my question",
    "i asked you first", "i asked u first", "i asked first", "tell me yours", "tell me urs",
    "what about my question", "what about mine", "and mine?", "u skipped my question",
    "waiting for ur answer", "still waiting for ur answer", "answer my q",
)

RAW_STATE_CITY = (
    "state", "state?", "stat?", "city", "city?", "cty?", "province?", "region?",
    "which state", "which state?", "what state", "what state?", "wich state?",
    "which city", "which city?", "what city", "what city?", "wich city?",
    "which part", "which part?", "what part", "what part?", "which province", "what province",
    "which region", "what region", "which town", "what town",
    "ur state", "ur state?", "your state", "your state?", "whats ur state", "what is your state",
    "ur city", "ur city?", "your city", "your city?", "whats ur city", "what is your city",
    "where in indonesia", "where in india", "where in canada", "where in usa",
    "where in uk", "where in turkey", "where in philippines", "where in tunisia",
    "where in france", "where in germany", "where in brazil", "where in egypt",
)

RAW_NTMU = (
    "nice to meet ya", "glad to meet u", "glad to meet you", "glad to meet ya",
    "pleasure meeting u", "pleasure meeting you", "a pleasure meeting u",
    "nice meeting u", "nice meeting you", "nice meeting ya", "good meeting u", "good meeting you",
    "happy to meet u", "happy to meet you", "lovely to meet u", "lovely to meet you",
    "nice to meet u too", "nice to meet you too", "glad to meet u too", "glad to meet you too",
    "nice meeting u too", "nice meeting you too", "pleasure to meet u", "pleasure to meet you",
    "so nice to meet u", "so nice to meet you", "great to meet u", "great to meet you",
    "great meeting u", "nice meet u", "nice meet you",
)

RAW_DRY = (
    "hm", "hmm", "hmmm", "hmmmm", "um", "umm", "ummm", "idc", "oko", "gt",
    "mhm", "mhmm", "meh", "uh", "uhh", "uhhh", "oh", "ohh", "ohhh", "ah", "ahh", "ahhh",
    "oh ok", "oh okay", "ah ok", "ah okay", "oh alright", "ohh ok", "ohh okay",
    "hmm ok", "hmm okay", "umm ok", "um ok", "alr", "aight", "ohk", "oke", "okee", "okey",
)

RAW_HORNY = (
    "horny", "horny?", "im horny", "i'm horny", "lm horny", "i am horny",
    "so horny", "im so horny", "very horny", "im very horny", "really horny",
    "feeling horny", "im feeling horny", "feel horny", "kinda horny", "a bit horny",
    "little horny", "getting horny", "horny rn", "im horny rn",
    "horny u?", "horny u", "horny wbu", "horny wbu?", "horny hbu", "horny hbu?",
    "im horny u?", "im horny wbu?", "im horny wbu", "are u horny", "are u horny?",
    "are you horny", "are you horny?", "r u horny", "r u horny?", "u horny", "u horny?",
    "you horny?", "u feeling horny?", "are u feeling horny?", "still horny",
)

RAW_SECRET = (
    "tell me a secret", "tell me ur secret", "tell me your secret", "tell me one secret",
    "tell me some secret", "whats ur secret", "what's ur secret", "whats your secret",
    "what is your secret", "what is ur secret", "got any secrets", "got any secrets?",
    "do u have a secret", "do u have a secret?", "do you have a secret",
    "do u have any secrets", "do u have any secrets?", "any secrets?", "any secret?",
    "share a secret", "share ur secret", "confess a secret", "confess something",
    "tell me something secret", "tell me a deep secret", "tell me a dark secret",
    "tell me a dirty secret", "tell me a little secret", "u got any secrets?",
    "have any secrets?", "secret?", "secrets?",
)

STRICT_ALLOWED_S18_VOCAB = frozenset(
    w.strip("?") for grp in (RAW_INSIST, RAW_STATE_CITY, RAW_NTMU, RAW_DRY, RAW_HORNY, RAW_SECRET)
    for line in grp for w in line.lower().split() if w.strip("?")
)


def _sanitize_ascii_bytes(raw_text):
    if not isinstance(raw_text, str):
        return ""
    b = np.frombuffer(raw_text[:64].encode("ascii", errors="ignore"), dtype=np.uint8)
    valid = b[(b >= np.uint8(32)) & (b <= np.uint8(122))]
    return "" if valid.shape[0] == 0 else valid.tobytes().decode("ascii", errors="ignore")


def _collapse_str(s):
    return "".join(s[i] for i in range(len(s)) if i == 0 or s[i] != s[i - 1])


def _canonical_token_sig(s):
    clean_w = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in s.lower())
    return (tuple(w for w in _collapse_str(clean_w).split() if w), "?" in s)


def _clean_s18(raw_text):
    s = SPACE_RE.sub(" ", _sanitize_ascii_bytes(raw_text).strip().lower()).rstrip("!.,")
    if not s or len(s) > 44 or not CLEAN_ASCII_RE.match(s):
        return ""
    if s in OLD_SPEC_PROTECTED or STEP1_TO_17_COLLISION_RE.search(s):
        return ""
    toks = [w.strip("?") for w in s.split() if w.strip("?")]
    if not toks or len(toks) > 6 or any(w not in STRICT_ALLOWED_S18_VOCAB for w in toks):
        return ""
    return s


def _mine_parquet_s18(buckets):
    try:
        from datasets import load_dataset
        ds = load_dataset("daily_dialog", split="train", revision="refs/convert/parquet", streaming=True)
        for idx, row in enumerate(ds):
            if idx >= 2000:
                break
            for turn in row.get("dialog", ()):
                if not isinstance(turn, str):
                    continue
                for sent in re.split(r"[.!]+", turn[:90]):
                    s = _clean_s18(sent)
                    if not s:
                        continue
                    if "secret" in s: key = "secret"
                    elif "horny" in s: key = "horny"
                    elif ("meet" in s or "meeting" in s) and any(k in s for k in ("glad", "pleasure", "lovely", "happy", "great", "too")): key = "ntmu"
                    elif any(k in s for k in ("ignored", "dodged", "answer my", "didnt answer", "didn't answer")): key = "insist"
                    elif any(k in s for k in ("which city", "what city", "which state", "what state", "which province")): key = "state"
                    else: continue
                    sig = _canonical_token_sig(s)
                    if sig not in buckets[key]:
                        buckets[key][sig] = s
    except Exception:
        pass


def build_step18_behavior_files():
    specs = (
        ("insist", "s18_insist.txt", RAW_INSIST),
        ("state", "s18_state_city.txt", RAW_STATE_CITY),
        ("ntmu", "s18_ntmu.txt", RAW_NTMU),
        ("dry", "s18_dry.txt", RAW_DRY),
        ("horny", "s18_horny.txt", RAW_HORNY),
        ("secret", "s18_secret.txt", RAW_SECRET),
    )
    buckets = {k: {} for k, _, _ in specs}
    for k, _, raw_lines in specs:
        for line in raw_lines:
            c = _clean_s18(line)
            if c:
                sig = _canonical_token_sig(c)
                if sig not in buckets[k]:
                    buckets[k][sig] = c

    _mine_parquet_s18(buckets)

    tot_lines, tot_bytes = np.int32(0), np.int32(0)
    for k, out_path, _ in specs:
        lines = sorted(buckets[k].values())
        arr_u8 = np.frombuffer(("\n".join(lines) + "\n").encode("ascii", errors="ignore"), dtype=np.uint8)
        with open(out_path, "wb") as f:
            f.write(arr_u8.tobytes())
        tot_lines += np.int32(len(lines))
        tot_bytes += np.int32(arr_u8.nbytes)
    print(f"[CLEAN STEP 18 BEHAVIOR RULES 12+13] Saved 6 pure ASCII files | {int(tot_lines)} unique lines ({float(np.float32(tot_bytes) / np.float32(1024.0)):.2f} KB)", flush=True)


if __name__ == "__main__":
    build_step18_behavior_files()