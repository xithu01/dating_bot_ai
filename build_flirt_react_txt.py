#!/usr/bin/env python3
# FILE: build_flirt_react_txt.py
# PURPOSE: Ultra-Clean Dataset Builder for Step 7: compliments_flirt.txt (Class 19) & reactions_ack.txt (Class 20)
# RULES: Enforces Rules 1-13:
#        - Rule 12: Parquet-only fetch (revision="refs/convert/parquet"), 7-bit ASCII byte strip, closed vocab whitelist
#        - Rule 13: Zero Cartesian filler ("tbh", "lol", "haha", "!") bloat, cross-step analyzer, canonical deduplication

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

CLEAN_LINE_RE = re.compile(r"^[a-z\s']+$")
SPACE_RE = re.compile(r"\s+")

# Rule 13 Cross-Step Collision Guard (Blocks Steps 1-6 & Step 8 phrases from ever entering Step 7)
STEP1_TO_6_AND_8_COLLISION_RE = re.compile(
    r"^(?:"
    r"(?:hi+|he+y+|he+l+o+|yo+|sup+)(?:\s+(?:t+h+e+r+e+|c+u+t+i+e+|g+o+r+g+e+o+u+s+|h+a+n+d+s+o+m+e+|b+r+o+|m+a+n+|f+r+i+e+n+d+))?|"
    r"(?:n+i+c+e+\s+t+o+\s+m+e+e+t+\s+(?:y+o+u+|u+)|g+o+o+d+\s+(?:m+o+r+n+i+n+g+|a+f+t+e+r+n+o+o+n+|e+v+e+n+i+n+g+|d+a+y+|n+i+g+h+t+|l+u+c+k+|j+o+b+))|"
    r"(?:(?:i+m+|i+'?m+|l+m+|d+o+i+n+g*|a+l+l+|p+r+e+t+y+)\s+(?:g+o+o+d+|g+r+e+a+t+|f+i+n+e+|o+k+|o+k+a+y+|a+l+r+i+g+h+t+)|g+o+o+d+|f+i+n+e+)|"
    r"(?:(?:g+o+o+d+|f+i+n+e+)\s+(?:u+|y+o+u+|w+b+u+|h+b+u+))|"
    r"(?:.*(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|s+e+e+k+i+n+g*|f+r+i+e+n+d+s*|g+f+|b+f+|d+a+t+i+n+g+|r+e+l+a+t+i+o+n+s+h+i+p+).*)"
    r")$",
    re.IGNORECASE,
)

# Strictly matches Class 19 (compliment_flirt) — zero "!" or Cartesian filler bloat
STRICT_FLIRT_RE = re.compile(
    r"^(?:(?:hi+|he+y+|a+w+|o+h+|w+o+w+)\s+)?"
    r"(?:"
    r"(?:y+o+u+\s+a+r+e+|y+o+u+'?r+e+|u+\s+a+r+e+|u+\s+r+|u+r+|"
    r"(?:y+o+u+|u+)\s+(?:s+e+e+m+|l+o+o+k+|s+o+u+n+d+))"
    r"(?:\s+(?:s+o+|v+e+r+y+|r+e+a+l+y+|s+u+p+e+r+|p+r+e+t+y+))?\s+"
    r"(?:c+u+t+e+|c+u+t+i+e+|s+w+e+e+t+|s+w+e+a+t+|n+i+c+e+|c+o+o+l+|h+o+t+|p+r+e+t+y+|"
    r"g+o+r+g+e+o+u+s+|b+e+a+u+t+i+f+u+l+|h+a+n+d+s+o+m+e+|s+e+x+y+|a+t+r+a+c+t+i+v+e+|"
    r"a+d+o+r+a+b+l+e+|l+o+v+e+l+y+|c+h+a+r+m+i+n+g+|f+u+n+y+|f+u+n+|a+w+e+s+o+m+e+|"
    r"a+m+a+z+i+n+g+|s+m+a+r+t+|k+i+n+d+|c+h+i+l+)|"
    r"(?:(?:i+|l+)\s+)?(?:r+e+a+l+y+\s+)?(?:l+i+k+e+|l+u+v+|l+o+v+e+)\s+"
    r"(?:y+o+u+|u+|(?:t+a+l+k+i+n+g+|s+p+e+a+k+i+n+g+)\s+(?:t+o+|w+i+t+h+)\s+(?:y+o+u+|u+)|"
    r"(?:y+o+u+r+|u+r+)\s+(?:v+i+b+e+|e+n+e+r+g+y+|s+m+i+l+e+|s+t+y+l+e+))|"
    r"(?:y+o+u+|u+)\s+(?:h+a+v+e+|g+o+t+)\s+a+\s+(?:n+i+c+e+|g+o+o+d+|g+r+e+a+t+|c+u+t+e+|c+o+o+l+)\s+"
    r"(?:v+i+b+e+|s+m+i+l+e+|e+n+e+r+g+y+)|"
    r"(?:y+o+u+\s+a+r+e+|y+o+u+'?r+e+|u+\s+r+|u+r+)\s+m+y+\s+t+y+p+e+|"
    r"(?:y+o+u+|u+)\s+m+a+k+e+\s+m+e+\s+(?:s+m+i+l+e+|b+l+u+s+h+)"
    r")(?:\s+t+o+o+)?$",
    re.IGNORECASE,
)

# Strictly matches Class 20 (short_reaction_ack) including "same" and "sam"
STRICT_REACT_RE = re.compile(
    r"^(?:"
    r"(?:h+a+h+a+(?:h+a+)*|h+e+h+e+(?:h+e+)*|l+o+l+|l+m+a+o+|l+m+f+a+o+|r+o+f+l+)|"
    r"(?:(?:o+h+|o+o+h+|a+h+|a+w+|h+a+h+a+|l+o+l+|y+e+a+h*|y+e+p+|y+u+p+|o+k+|o+k+a+y+)\s+)?"
    r"(?:n+i+c+e+|c+o+o+l+|s+w+e+e+t+|a+w+e+s+o+m+e+|a+m+a+z+i+n+g+|d+o+p+e+|l+i+t+|f+i+r+e+|"
    r"b+e+t+|w+o+r+d+|f+a+i+r+(?:\s+e+n+o+u+g+h+)?|s+o+u+n+d+s+\s+(?:g+o+o+d+|n+i+c+e+|c+o+o+l+|f+u+n+)|"
    r"t+h+a+t+'?s\s+(?:n+i+c+e+|c+o+o+l+|t+r+u+e+|f+a+i+r+))|"
    r"(?:(?:h+a+h+a+|l+o+l+|y+e+a+h*|y+e+p+|o+h+|a+w+)\s+)?"
    r"(?:s+a+m+e+|s+a+m+)(?:\s+(?:h+e+r+e+|t+o+\s+(?:y+o+u+|u+)|l+o+l+|h+a+h+a+))?|"
    r"(?:(?:h+a+h+a+|l+o+l+|y+e+a+h*|o+h+)\s+)?m+e+\s+t+o+(?:\s+(?:l+o+l+|h+a+h+a+))?|"
    r"(?:(?:h+a+h+a+|l+o+l+|y+e+a+h*|s+o+|t+o+o+|v+e+r+y+)\s+)?"
    r"(?:t+r+u+e+|t+r+u+|f+a+c+t+s+|f+a+x+|f+r+(?:\s+f+r+)?|f+o+r+\s+r+e+a+l+|"
    r"i+k+r+|i+\s+k+n+o+w+(?:\s+r+i+g+h+t+)?|r+i+g+h+t+|r+e+a+l+|a+g+r+e+e+d*|"
    r"t+o+t+a+l+y+|d+e+f+i+n+i+t+e+l+y+|e+x+a+c+t+l+y*)"
    r"(?:\s+(?:l+o+l+|h+a+h+a+|t+h+a+t+))?|"
    r"(?:(?:h+a+h+a+|l+o+l+|o+h+|a+h+|y+e+a+h*|y+e+p+)\s+)?"
    r"(?:o+k+|o+k+a+y+|o+k+e+y+|o+k+i+e*|k+k+|a+l+r+i+g+h+t+|a+l+r+|a+i+g+h+t+|"
    r"y+e+a+h+|y+e+a+|y+e+p+|y+u+p+|y+a+s+|s+u+r+e+|o+f+\s+c+o+u+r+s+e+|o+f+c+|a+h+a+|w+o+w+|o+m+g+)"
    r"(?:\s+(?:n+i+c+e+|c+o+o+l+|s+w+e+e+t+|b+e+t+|f+a+i+r+|l+o+l+|h+a+h+a+|t+h+e+n+))?"
    r")$",
    re.IGNORECASE,
)

# Rule 12 Gate 2: Closed Whitelist Vocabulary for Step 7 (Exact match with offline_train.py STRICT_S7_VOCAB)
STRICT_ALLOWED_S7_VOCAB = frozenset((
    "cute", "cuute", "cutie", "hot", "hott", "pretty", "prety", "gorgeous", "beautiful",
    "handsome", "sexy", "attractive", "adorable", "lovely", "charming", "funny", "smart",
    "kind", "chill", "like", "luv", "love", "vibe", "energy", "smile", "style", "type",
    "blush", "talking", "speaking",
    "haha", "hahaha", "hehe", "lol", "lmao", "lmfao", "rofl", "dope", "lit", "fire",
    "bet", "word", "fair", "same", "sam", "true", "tru", "facts", "fax", "fr", "ikr",
    "right", "real", "agreed", "agree", "totally", "definitely", "exact", "exactly",
    "kk", "okey", "oki", "okie", "alr", "aight", "yeah", "yea", "yep", "yup", "yas",
    "yass", "sure", "ofc", "aha", "wow", "omg",
    "nice", "niice", "cool", "coool", "sweet", "sweat", "awesome", "amazing",
    "fun", "ok", "okay", "alright",
    "u", "you", "ur", "your", "you're", "youre", "ya", "seem", "look", "sound", "make",
    "i", "l", "me", "my", "are", "r", "so", "very", "really", "realy", "super", "hella",
    "too", "to", "with", "have", "got", "a", "good", "great", "here", "oh", "ooh", "ah",
    "aw", "aww", "thats", "that's", "that", "sounds", "enough", "for", "know", "of",
    "course", "then", "hi", "hey", "helo", "yo",
))

FLIRT_SUBJECTS = (
    "ur", "u r", "you are", "you're", "u seem", "you seem", "u look", "you look", "u sound", "you sound",
)
FLIRT_INTENSIFIERS = ("", "so", "very", "really", "super")
FLIRT_ADJECTIVES = (
    "cute", "sweet", "nice", "cool", "hot", "pretty", "gorgeous", "beautiful",
    "handsome", "sexy", "attractive", "adorable", "lovely", "charming", "funny",
    "fun", "awesome", "amazing", "smart", "kind", "chill",
)
FLIRT_DIRECT_PHRASES = (
    "i like u", "i like you", "i really like u", "i really like you", "like u", "like you",
    "i like talking to u", "i like talking to you", "like talking to u", "i love talking to u",
    "i like ur vibe", "i love ur vibe", "love ur vibe", "love your vibe", "like ur vibe",
    "i like ur energy", "i like ur smile", "i love ur smile", "u have a nice vibe",
    "you have a nice vibe", "u got a good vibe", "u have a cute smile", "ur my type",
    "you are my type", "u r my type", "you're my type", "u make me smile", "you make me smile",
    "u make me blush", "ur cutie", "u r a cutie", "you are a cutie",
    "hey ur cute", "aw ur sweet", "u look good", "you look good",
)

BASE_REACTIONS = (
    "haha", "hahaha", "hehe", "lol", "lmao", "lmfao", "rofl",
    "nice", "cool", "sweet", "awesome", "amazing", "dope", "lit", "fire",
    "bet", "word", "fair", "fair enough", "oh nice", "oh cool", "ooh nice", "ooh cool",
    "aw nice", "haha nice", "haha cool", "lol nice", "lol cool", "yeah nice", "yeah cool",
    "sounds good", "sounds nice", "sounds cool", "sounds fun", "thats nice", "that's nice",
    "thats cool", "that's cool", "thats fair", "that's fair",
    "same", "samee", "saame", "sam", "saam", "samm",
    "same here", "sam here", "same here lol", "sam here lol", "same here haha", "sam here haha",
    "haha same", "haha sam", "lol same", "lol sam", "same lol", "sam lol",
    "same haha", "sam haha", "yeah same", "yea same", "yep same", "oh same", "aw same",
    "same to u", "same to you",
    "me too", "me to", "me too lol", "me too haha", "haha me too", "lol me too", "yeah me too",
    "true", "tru", "so true", "too true", "very true", "true that", "true lol",
    "true haha", "haha true", "lol true", "yeah true", "thats true", "that's true",
    "facts", "fax", "fr", "fr fr", "for real", "ikr", "i know right", "i know", "right",
    "real", "agreed", "i agree", "totally", "definitely", "exact", "exactly",
    "ok", "okay", "okey", "oki", "okie", "kk", "alright", "alr", "aight",
    "ok nice", "ok cool", "okay nice", "okay cool", "ok sweet", "ok bet", "ok fair",
    "ok lol", "ok haha", "okay lol", "okay haha", "haha ok", "lol ok", "oh ok", "oh okay",
    "ah ok", "ah okay", "yeah ok", "yea ok", "ok then", "okay then", "alright cool",
    "yeah", "yea", "yep", "yup", "yas", "sure", "ofc", "of course", "aha", "wow", "omg",
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
    clean_w = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in s.lower())
    words = tuple(w for w in _collapse_str(clean_w).split() if w)
    return (words, False)


def categorize_line_across_all_steps(raw_text):
    clean_ascii = _sanitize_ascii_bytes(raw_text)
    if "?" in clean_ascii or "/" in clean_ascii:
        return 0, ""
    s = SPACE_RE.sub(" ", clean_ascii.strip().lower()).rstrip("!.,")
    if not s or len(s) > 40 or not CLEAN_LINE_RE.match(s):
        return 0, ""

    if STEP1_TO_6_AND_8_COLLISION_RE.match(s):
        return 1, s

    words = [w.strip("'") for w in s.split() if w.strip("'")]
    if not words or len(words) > 6:
        return 0, ""

    for w in words:
        if w not in STRICT_ALLOWED_S7_VOCAB:
            return 0, ""

    if len(words) >= 2 and STRICT_FLIRT_RE.match(s):
        return 19, s
    if len(words) <= 4 and STRICT_REACT_RE.match(s):
        return 20, s
    return 0, ""


def _safe_fetch_parquet_dialog_lines(flirt_map, react_map):
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
            for turn in turns:
                if not isinstance(turn, str):
                    continue
                for sent in re.split(r"[.!]+", turn[:120]):
                    cid, clean_s = categorize_line_across_all_steps(sent)
                    if cid == 19:
                        sig = _canonical_token_sig(clean_s)
                        if sig not in flirt_map:
                            flirt_map[sig] = clean_s
                    elif cid == 20:
                        sig = _canonical_token_sig(clean_s)
                        if sig not in react_map:
                            react_map[sig] = clean_s
    except Exception:
        pass


def build_flirt_and_react_files(
    flirt_path="compliments_flirt.txt",
    react_path="reactions_ack.txt",
    neg_path="negatives.txt",
):
    flirt_map = {}
    react_map = {}

    for subj in FLIRT_SUBJECTS:
        for inten in FLIRT_INTENSIFIERS:
            for adj in FLIRT_ADJECTIVES:
                core = f"{subj} {inten} {adj}".strip() if inten else f"{subj} {adj}"
                core = SPACE_RE.sub(" ", core)
                cid, c = categorize_line_across_all_steps(core)
                if cid == 19:
                    sig = _canonical_token_sig(c)
                    if sig not in flirt_map:
                        flirt_map[sig] = c

    for phrase in FLIRT_DIRECT_PHRASES:
        cid, c = categorize_line_across_all_steps(phrase)
        if cid == 19:
            sig = _canonical_token_sig(c)
            if sig not in flirt_map:
                flirt_map[sig] = c

    for r_phrase in BASE_REACTIONS:
        cid, c = categorize_line_across_all_steps(r_phrase)
        if cid == 20:
            sig = _canonical_token_sig(c)
            if sig not in react_map or r_phrase in ("same", "sam", "same here", "sam here"):
                # Keep both "same" and "sam" explicitly since their collapsed signatures differ ("same" vs "sam")
                react_map[sig] = c

    _safe_fetch_parquet_dialog_lines(flirt_map, react_map)
    for k_sig in list(react_map.keys()):
        if k_sig in flirt_map:
            del react_map[k_sig]

    sorted_flirt = sorted(set(flirt_map.values()))
    sorted_react = sorted(set(react_map.values()))

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
        f"[CLEAN STEP 7 RULES 12+13] Saved {len(sorted_flirt)} unique lines to '{flirt_path}' ({flirt_u8.nbytes / 1024.0:.2f} KB) "
        f"& {len(sorted_react)} unique lines to '{react_path}' ({react_u8.nbytes / 1024.0:.2f} KB)",
        flush=True,
    )


if __name__ == "__main__":
    build_flirt_and_react_files()