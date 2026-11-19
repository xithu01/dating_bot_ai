#!/usr/bin/env python3
# FILE: build_pic_photo_txt.py
# PURPOSE: Zero-CVE Multi-Dataset Builder for Step 13:
#          - pic_photo_ask.txt (Class 31: ask_pic_photo — send pic/selfie/photo/face/what do u look like)
#          - pic_photo_offer.txt (Class 32: offer_or_refuse_pic — wanna see me/can i send a pic/u first/no pics)
# RULES: Enforces Rules 1-13 (Parquet-only fetch, 7-bit ASCII byte strip, closed vocab whitelist,
#        zero "!" / filler bloat, protects Step 1 "good to see u", Step 7 "u look good", Step 10 "photography").

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
# Protects Step 1 ("good/nice to see you"), Step 3 ("how are you"), Step 7 ("u look good/cute/nice"),
# Step 8 ("looking for..."), Step 10 ("photography"), Step 12 ("telegram/snap/insta/private").
STEP1_TO_12_COLLISION_RE = re.compile(
    r"^(?:"
    r"(?:(?:g+o+o+d+|n+i+c+e+|g+l+a+d+|p+l+e+a+s+e+d+)\s+t+o+\s+s+e+e+\s+(?:y+o+u+|u+)|l+o+n+g+\s+t+i+m+e+\s+n+o+\s+s+e+e+)|"
    r"(?:(?:y+o+u+|u+)\s+l+o+o+k+\s+(?:g+o+o+d+|c+u+t+e+|n+i+c+e+|h+o+t+|p+r+e+t+y+|s+w+e+e+t+|a+m+a+z+i+n+g+))|"
    r".*\b(?:p+h+o+t+o+g+r+a+p+h+y+|t+e+l+e*g+r+a*m+|t+g+|s+n+a+p+(?:c+h+a+t+)?|i+n+s+t+a+(?:g+r+a+m+)?|i+g+|w+h*a+t+s+a+p+p+|p+r+i+v+a*t*e*y*|p+v+|d+m+|u+s+e+r+n+a+m+e*)\b.*|"
    r".*\b(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|f+r+i+e+n+d+s*|g+f+|b+f+|d+a+t+i+n+g+|r+e+l+a+t+i+o+n+s+h+i+p+|s+i+n+g+e*l+e*|m+a+r+i+e*d+|t+a+k+e*n+)\b.*|"
    r".*\b(?:h+r+u+|h+y+d+|w+y+d+|o+l+d+|a+g+e+|f+r+o+m+|c+o+u+n+t+r+y+|n+a+m+e+|h+o+b+b*y+|m+u+s+i+c+|g+a+m+e+s*|a+n+i+m+e+)\b.*"
    r")[\s?]*$",
    re.IGNORECASE,
)

# Closed Step 13 Vocabulary (Exact match with step13_pic_photo.py LUTs)
S13_PIC_NOUNS = (
    "pic", "pics", "pik", "piks", "picture", "pictures", "pictur",
    "photo", "photos", "foto", "fotos", "selfie", "selfies", "selfy",
    "face", "urself", "yourself", "img", "image",
)
S13_APPEARANCE_WORDS = (
    "look", "looks", "describe", "tall", "height", "hair", "eyes", "color", "colour", "build", "body",
)
S13_PIC_VERBS = (
    "send", "sending", "snd", "show", "see", "trade", "swap", "exchange", "share", "shareing",
)
S13_OFFER_REFUSE_WORDS = (
    "first", "ill", "i'll", "wanna", "want", "dont", "don't", "no", "not", "never", "mine",
)
S13_QUESTION_WORDS = (
    "can", "could", "may", "will", "would", "what", "whats", "what's", "wat", "how", "do", "are", "r",
)
S13_FRAME_WORDS = (
    "u", "you", "ur", "your", "i", "im", "i'm", "me", "my", "we", "a", "an", "of", "for", "to",
    "is", "like", "let", "lemme", "plz", "pls", "please", "one", "real", "now", "rn",
    "hi", "hey", "so", "btw", "if",
)

STRICT_ALLOWED_S13_VOCAB = frozenset(
    S13_PIC_NOUNS + S13_APPEARANCE_WORDS + S13_PIC_VERBS
    + S13_OFFER_REFUSE_WORDS + S13_QUESTION_WORDS + S13_FRAME_WORDS
)

# Strictly matches Class 31 (ask_pic_photo)
STRICT_ASK_PIC_RE = re.compile(
    r"^(?:(?:hi+|he+y+|so+|b+t+w+)\s+)?"
    r"(?:"
    r"(?:(?:c+a+n+|c+o+u+l+d+|w+i+l+l*|w+o+u+l+d+)\s+(?:y+o+u+|u+)\s+)?"
    r"(?:s+e+n+d+|s+n+d+|s+h+o+w+|s+h+a+r+e+)\s+(?:m+e+\s+)?(?:a+\s+|y+o+u+r+\s+|u+r+\s+|o+n+e+\s+)?"
    r"(?:p+i+c+s*|p+i+k+s*|p+i+c+t+u+r+e*s*|p+h+o+t+o+s*|f+o+t+o+s*|s+e+l+f+i+e*s*|s+e+l+f+y+|f+a+c+e+|u+r+s+e+l+f+|y+o+u+r+s+e+l+f+)|"
    r"(?:(?:c+a+n+|c+o+u+l+d+|m+a+y+)\s+i+\s+|l+e+t+\s+m+e+\s+|l+e+m+m*e+\s+)s+e+e+\s+"
    r"(?:y+o+u+|u+|(?:a+\s+|y+o+u+r+\s+|u+r+\s+)?(?:p+i+c+s*|p+h+o+t+o+s*|s+e+l+f+i+e*s*|f+a+c+e+))|"
    r"(?:w+a+n+n*a+\s+|w+a+n+t+\s+t+o+\s+|c+a+n+\s+w+e+\s+)?(?:t+r+a+d+e+|s+w+a+p+|e+x+c+h+a+n+g+e+)\s+(?:p+i+c+s*|p+h+o+t+o+s*|s+e+l+f+i+e*s*)|"
    r"(?:p+i+c+|p+h+o+t+o+|s+e+l+f+i+e+)\s+(?:f+o+r+\s+(?:p+i+c+|p+h+o+t+o+|s+e+l+f+i+e+)|t+r+a+d+e+|s+w+a+p+)|"
    r"(?:y+o+u+r+|u+r+)?\s*(?:p+i+c+s*|p+i+k+s*|p+i+c+t+u+r+e*s*|p+h+o+t+o+s*|f+o+t+o+s*|s+e+l+f+i+e*s*|f+a+c+e+)\??|"
    r"(?:w+h+a+t+|w+a+t+)\s+(?:d+o+\s+)?(?:y+o+u+|u+)\s+l+o+o+k+\s+l+i+k+e+|"
    r"h+o+w+\s+(?:d+o+\s+)?(?:y+o+u+|u+)\s+l+o+o+k+|"
    r"d+e+s+c+r+i+b+e+\s+(?:y+o+u+r+s+e+l+f+|u+r+s+e+l+f+)|"
    r"h+o+w+\s+t+a+l+l*\s+(?:a+r+e+|r+)\s+(?:y+o+u+|u+)|"
    r"(?:w+h+a+t+'?s|w+h+a+t+s|w+h+a+t+\s+i+s+)\s+(?:y+o+u+r+|u+r+)\s+(?:h+e+i+g+h+t+|h+a+i+r+\s+c+o+l+o+u*r+|e+y+e+\s+c+o+l+o+u*r+)|"
    r"w+h+a+t+\s+c+o+l+o+u*r+\s+(?:i+s+|a+r+e+)\s+(?:y+o+u+r+|u+r+)\s+(?:h+a+i+r+|e+y+e+s+)"
    r")(?:\s+(?:p+l+z+|p+l+s+|p+l+e+a+s+e+|r+n+|n+o+w+))?\??$",
    re.IGNORECASE,
)

# Strictly matches Class 32 (offer_or_refuse_pic)
STRICT_OFFER_REFUSE_PIC_RE = re.compile(
    r"^(?:"
    r"(?:(?:d+o+\s+(?:y+o+u+|u+)\s+)?(?:w+a+n+n*a+|w+a+n+t+(?:\s+t+o+)?))\s+s+e+e+\s+"
    r"(?:m+e+|m+y+\s+(?:p+i+c+s*|p+h+o+t+o+s*|s+e+l+f+i+e*s*|f+a+c+e+)|a+\s+(?:p+i+c+|p+h+o+t+o+|s+e+l+f+i+e+))|"
    r"(?:d+o+\s+(?:y+o+u+|u+)\s+)?w+a+n+t+\s+(?:m+y+|a+)\s+(?:p+i+c+|p+h+o+t+o+|s+e+l+f+i+e+)(?:\s+o+f+\s+m+e+)?|"
    r"(?:c+a+n+|m+a+y+|s+h+o+u+l+d+)\s+i+\s+s+e+n+d+\s+(?:y+o+u+\s+|u+\s+)?(?:a+\s+|m+y+\s+)?(?:p+i+c+s*|p+h+o+t+o+s*|s+e+l+f+i+e*s*)|"
    r"(?:i+'?l+l*|i+\s+c+a+n+|i+\s+w+i+l+l*)\s+s+e+n+d+\s+(?:(?:y+o+u+|u+)\s+)?(?:a+\s+|m+y+\s+)?(?:p+i+c+s*|p+h+o+t+o+s*|s+e+l+f+i+e*s*|f+i+r+s+t+)|"
    r"(?:y+o+u+|u+|s+e+n+d+)\s+f+i+r+s+t+|"
    r"(?:i+\s+)?(?:d+o+n+'?t+|n+e+v+e+r+)\s+s+e+n+d+\s+(?:p+i+c+s*|p+h+o+t+o+s*|s+e+l+f+i+e*s*)|"
    r"(?:n+o+|n+o+t+\s+s+e+n+d+i+n+g+)\s+(?:p+i+c+s*|p+h+o+t+o+s*|s+e+l+f+i+e*s*)"
    r")\??$",
    re.IGNORECASE,
)

ASK_PIC_BASES = (
    # Direct Pic / Photo / Selfie Requests
    "send pic", "send a pic", "send pics", "send ur pic", "send your pic", "send me a pic",
    "can u send a pic", "can you send a pic", "can u send pic", "can u send pics",
    "can u send me a pic", "can u send a selfie", "can you send a selfie", "send selfie",
    "send a selfie", "send me a selfie", "send ur selfie", "send photo", "send a photo",
    "send photos", "send me a photo", "send ur photo", "send foto", "send pik",
    "show ur face", "show your face", "show me ur face", "show me your face", "show urself",
    "can i see u", "can i see you", "can i see a pic", "can i see ur pic", "can i see ur face",
    "can i see a selfie", "let me see u", "let me see you", "lemme see u", "lemme see ur face",
    "pic for pic", "photo for photo", "selfie for selfie", "pic trade", "trade pics",
    "wanna trade pics", "want to trade pics", "can we trade pics", "swap pics",
    "pic?", "pics?", "selfie?", "selfies?", "photo?", "photos?", "foto?", "ur pic?", "your pic?",
    # Appearance / What Do You Look Like
    "what do you look like", "what do u look like", "what u look like", "wat u look like",
    "how do you look", "how do u look", "how u look",
    "describe yourself", "describe urself",
    "how tall are you", "how tall r u", "how tall are u", "whats ur height", "whats your height",
    "what color is ur hair", "what color is your hair", "what color are ur eyes", "whats ur hair color",
)

OFFER_REFUSE_PIC_BASES = (
    # User Offering Their Pic (aux_id = 1)
    "wanna see me", "wanna see me?", "want to see me", "want to see me?",
    "do u wanna see me", "do u want to see me", "wanna see my pic", "wanna see my pic?",
    "wanna see my face", "wanna see a pic", "want my pic", "want my pic?",
    "want a pic of me", "want a selfie", "can i send a pic", "can i send a pic?",
    "can i send u a pic", "can i send you a pic", "can i send a selfie", "can i send u a selfie",
    "ill send a pic", "i'll send a pic", "i can send a pic", "i can send a selfie",
    "ill send first", "i'll send first", "i can send first", "ill send my pic",
    # User Saying "u first" or Refusing Pics (aux_id = 2)
    "u first", "you first", "u first?", "send first",
    "i dont send pics", "i don't send pics", "dont send pics", "i never send pics",
    "i dont send photos", "i dont send selfies", "no pics", "no photos", "no selfies",
    "not sending pics", "not sending photos", "not sending selfies",
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

    if STEP1_TO_12_COLLISION_RE.match(s):
        return 1, s

    words = [w.strip("'?,") for w in s.split() if w.strip("'?,")]
    if not words or len(words) > 6:
        return 0, ""

    for w in words:
        if w not in STRICT_ALLOWED_S13_VOCAB:
            return 0, ""

    if STRICT_OFFER_REFUSE_PIC_RE.match(s):
        return 32, s

    if STRICT_ASK_PIC_RE.match(s):
        return 31, s

    return 0, ""


def _safe_fetch_parquet_pic_lines(ask_map, offer_map):
    # Rule 12: Streams ONLY pure Parquet data (revision="refs/convert/parquet", zero remote .py scripts!)
    stats = {"scanned": 0, "blocked_overlaps": 0, "ask_added": 0, "offer_added": 0}
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
                        if 1 <= cid <= 30:
                            stats["blocked_overlaps"] += 1
                        elif cid == 31:
                            sig = _canonical_token_sig(clean_s)
                            if sig not in ask_map:
                                ask_map[sig] = clean_s
                                stats["ask_added"] += 1
                        elif cid == 32:
                            sig = _canonical_token_sig(clean_s)
                            if sig not in offer_map:
                                offer_map[sig] = clean_s
                                stats["offer_added"] += 1
        except Exception:
            pass
    return stats


def _write_u8_file(path, lines_iterable):
    sorted_lines = sorted(set(lines_iterable))
    u8 = np.frombuffer(("\n".join(sorted_lines) + "\n").encode("ascii"), dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(u8.tobytes())
    return len(sorted_lines), u8.nbytes


def build_pic_photo_files(
    ask_path="pic_photo_ask.txt",
    offer_path="pic_photo_offer.txt",
    neg_path="negatives.txt",
):
    ask_map = {}
    offer_map = {}

    for base in ASK_PIC_BASES:
        if "?" in base:
            cid, c = categorize_line_across_all_steps(base)
            if cid == 31:
                ask_map[_canonical_token_sig(c)] = c
            continue
        for pfx in ("", "hey", "so"):
            full = f"{pfx} {base}".strip() if pfx else base
            for end in ("", "?"):
                cid, c = categorize_line_across_all_steps(f"{full}{end}")
                if cid == 31:
                    sig = _canonical_token_sig(c)
                    if sig not in ask_map:
                        ask_map[sig] = c

    for base in OFFER_REFUSE_PIC_BASES:
        cid, c = categorize_line_across_all_steps(base)
        if cid == 32:
            sig = _canonical_token_sig(c)
            if sig not in offer_map:
                offer_map[sig] = c

    hf_stats = _safe_fetch_parquet_pic_lines(ask_map, offer_map)

    for k_sig in list(offer_map.keys()):
        if k_sig in ask_map:
            del offer_map[k_sig]

    n_ask, b_ask = _write_u8_file(ask_path, ask_map.values())
    n_offer, b_offer = _write_u8_file(offer_path, offer_map.values())

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn:
            existing_neg = {line.strip().lower() for line in fn if line.strip()}
    for item in list(ask_map.values()) + list(offer_map.values()):
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ")):
            existing_neg.add(item)
    _, _ = _write_u8_file(neg_path, existing_neg)

    print(
        f"[PARQUET STEP 13 ANALYZER] Scanned {hf_stats['scanned']} rows | "
        f"Blocked {hf_stats['blocked_overlaps']} Step 1-12 overlaps | "
        f"Added {hf_stats['ask_added']} ask & {hf_stats['offer_added']} offer/refuse lines!",
        flush=True,
    )
    print(
        f"[CLEAN STEP 13 PIC/PHOTO RULES 12+13] Saved {n_ask} unique ask lines to '{ask_path}' ({b_ask / 1024.0:.2f} KB) "
        f"& {n_offer} unique offer/refuse lines to '{offer_path}' ({b_offer / 1024.0:.2f} KB)",
        flush=True,
    )


if __name__ == "__main__":
    build_pic_photo_files()