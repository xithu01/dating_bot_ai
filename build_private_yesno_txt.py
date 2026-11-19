#!/usr/bin/env python3
# FILE: build_private_yesno_txt.py
# PURPOSE: Zero-CVE Multi-Dataset Builder for Step 12:
#          - private_social_ask.txt (Class 29: ask_private_chat_social — Telegram, Private/PV/DM, Snap, Insta, WhatsApp)
#          - yes_no_maybe.txt (Class 30: reply_yes_no_maybe — Yes, No, Maybe, IDK, Don't have social)
# RULES: Enforces Rules 1-13 (Parquet-only fetch, 7-bit ASCII byte strip, closed vocab whitelist,
#        zero "!" / filler bloat, protects Step 7 "yeah"/"ok"/"sure" & Class 0 "why not").

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
# Protects Class 0 ("why not", "what is this", "who are you"), Step 7 ("yeah", "yep", "ok", "sure"),
# Step 8 ("looking for...", "here to talk"), Step 9 ("whats ur name"), Step 11 ("single", "married", "bf", "gf").
STEP1_TO_11_COLLISION_RE = re.compile(
    r"^(?:"
    r"(?:w+h+y+\s+n+o+t+|w+h+a+t+\s+i+s+\s+t+h+i+s+|w+h+o+\s+a+r+e+\s+y+o+u+)|"
    r"(?:y+e+a+h*|y+e+p+|y+u+p+|o+k+|o+k+a+y+|s+u+r+e+|o+f+c+|h+a+h+a+|l+o+l+|s+a+m+e+|s+a+m+)|"
    r"(?:(?:j+u+s+t+\s+)?h+e+r+e+\s+t+o+\s+(?:t+a+l+k+|c+h+a+t+))|"
    r".*\b(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|f+r+i+e+n+d+s*|g+f+|b+f+|g+i+r+l+f+r+i+e+n+d+|b+o+y+f+r+i+e+n+d+|d+a+t+i+n+g+|r+e+l+a+t+i+o+n+s+h+i+p+)\b.*|"
    r".*\b(?:s+i+n+g+e*l+e*|m+a+r+i+e*d+|t+a+k+e*n+|h+r+u+|h+y+d+|w+y+d+|o+l+d+|a+g+e+|f+r+o+m+|c+o+u+n+t+r+y+|n+a+m+e+|h+o+b+b*y+|m+u+s+i+c+|g+a+m+e+s*|a+n+i+m+e+)\b.*"
    r")[\s?]*$",
    re.IGNORECASE,
)

# Closed Step 12 Vocabulary (Exact match with step12_private_yesno.py LUTs)
S12_PRIVATE_TG_WORDS = (
    "private", "privty", "privte", "privat", "privet", "priv", "prv", "pv", "pvt",
    "dm", "dms", "pm", "pms", "inbox", "direct",
    "telegram", "telgrm", "telgram", "tele", "tg", "tlg",
    "username", "usernam", "user", "usr", "handle",
)
S12_SOCIAL_WORDS = (
    "snap", "snapchat", "snp", "sc", "insta", "instagram", "inst", "ig",
    "whatsapp", "watsapp", "whatsap", "wa", "tiktok", "discord",
    "number", "num", "phone", "vc", "video", "voice", "socials", "social",
)
S12_YES_WORDS = ("yes", "yess", "ys")
S12_NO_WORDS = ("no", "noo", "nop", "nope", "nah", "na")
S12_MAYBE_WORDS = ("maybe", "mayb", "mby", "idk", "dunno", "unsure", "depends", "possibly", "probably")
S12_ACTION_VERBS = (
    "talk", "chat", "text", "msg", "message", "add", "send", "give", "share",
    "move", "go", "come", "switch", "call", "have", "hav", "got", "use", "know", "stay",
)
S12_NEG_WORDS = ("dont", "don't", "not", "never")
S12_FRAME_WORDS = (
    "can", "could", "may", "shall", "should", "do", "will", "want", "wanna",
    "we", "i", "im", "i'm", "u", "you", "ur", "your", "me", "my", "lets", "let's",
    "in", "on", "to", "at", "here", "there", "what", "whats", "what's", "wat",
    "is", "it", "a", "an", "any", "of", "course", "sure", "please", "plz", "pls",
    "thanks", "thx", "ty", "ok", "okay", "sorry", "really", "actually", "just",
    "hi", "hey", "so", "btw", "and",
)

STRICT_ALLOWED_S12_VOCAB = frozenset(
    S12_PRIVATE_TG_WORDS + S12_SOCIAL_WORDS + S12_YES_WORDS + S12_NO_WORDS
    + S12_MAYBE_WORDS + S12_ACTION_VERBS + S12_NEG_WORDS + S12_FRAME_WORDS
)

# Strictly matches Class 29 (ask_private_chat_social)
STRICT_ASK_PRIVATE_RE = re.compile(
    r"^(?:(?:hi+|he+y+|so+|b+t+w+|y+e+s+)\s+)?"
    r"(?:"
    r"(?:c+a+n+|c+o+u+l+d+|w+a+n+a+|w+a+n+t+\s+t+o+|d+o+\s+y+o+u+\s+w+a+n+t+\s+t+o+|l+e+t+'?s)\s+"
    r"(?:(?:w+e+|i+|u+)\s+)?(?:t+a+l+k+|c+h+a+t+|t+e+x+t+|m+s+g+|m+e+s+a+g+e+|m+o+v+e+|g+o+|c+o+m+e+|s+w+i+t+c+h+)\s+"
    r"(?:i+n+\s+|t+o+\s+|o+n+\s+)?"
    r"(?:p+r+i+v+a*t*e*y*|p+r+i+v+|p+r+v+|p+v+|p+v+t+|d+m+s*|p+m+s*|i+n+b+o+x+|"
    r"t+e+l+e*g+r+a*m+|t+g+|t+e+l+e+|s+n+a+p+(?:c+h+a+t+)?|i+n+s+t+a+(?:g+r+a+m+)?|i+g+|w+h*a+t+s+a+p+p*|d+i+s+c+o+r+d+)|"
    r"(?:(?:c+a+n+|c+o+u+l+d+|m+a+y+)\s+i+\s+)?(?:d+m+|p+m+|t+e+x+t+|m+s+g+|m+e+s+a+g+e+|a+d+d+)\s+(?:y+o+u+|u+|m+e+)(?:\s+o+n+\s+(?:t+g+|t+e+l+e*g+r+a*m+|s+n+a+p+|i+n+s+t+a+|i+g+))?|"
    r"(?:d+m+|p+m+|t+e+x+t+|a+d+d+)\s+m+e+(?:\s+(?:o+n+\s+)?(?:t+g+|t+e+l+e*g+r+a*m+|s+n+a+p+|i+n+s+t+a+|i+g+|p+v+))?|"
    r"(?:w+h+a+t+'?s|w+h+a+t+s|w+h+a+t+\s+i+s+|g+i+v+e+\s+m+e+|s+e+n+d+|s+h+a+r+e+|c+a+n+\s+i+\s+(?:h+a+v+e+|g+e+t+))\s+"
    r"(?:y+o+u+r+|u+r+)\s+"
    r"(?:t+e+l+e*g+r+a*m+|t+g+|t+e+l+e+|u+s+e+r+n+a+m+e*|u+s+e+r+|h+a+n+d+l+e+|"
    r"s+n+a+p+(?:c+h+a+t+)?|s+c+|i+n+s+t+a+(?:g+r+a+m+)?|i+g+|w+h*a+t+s+a+p+p*|w+a+|t+i+k+t+o+k+|d+i+s+c+o+r+d+|n+u+m+b+e+r+|n+u+m+|p+h+o+n+e+|s+o+c+i+a+l+s*)|"
    r"(?:d+o+\s+(?:y+o+u+|u+)|(?:y+o+u+|u+)|g+o+t+)\s+(?:h+a+v+e*|g+o+t+|u+s+e+)\s*(?:a+\s+)?"
    r"(?:t+e+l+e*g+r+a*m+|t+g+|u+s+e+r+n+a+m+e*|s+n+a+p+(?:c+h+a+t+)?|s+c+|i+n+s+t+a+(?:g+r+a+m+)?|i+g+|w+h*a+t+s+a+p+p*|t+i+k+t+o+k+|d+i+s+c+o+r+d+|s+o+c+i+a+l+s*)|"
    r"(?:y+o+u+r+|u+r+)?\s*(?:t+e+l+e*g+r+a*m+|t+g+|u+s+e+r+n+a+m+e*|s+n+a+p+(?:c+h+a+t+)?|i+n+s+t+a+(?:g+r+a+m+)?|i+g+|w+h*a+t+s+a+p+p+|d+i+s+c+o+r+d+|n+u+m+b+e+r+|p+r+i+v+a*t*e*y*|p+r+i+v+|p+v+|d+m+|p+m+|i+n+b+o+x+)\??|"
    r"(?:c+a+n+\s+w+e+\s+)?(?:v+i+d+e+o+|v+o+i+c+e+)\s+c+a+l+l+|"
    r"(?:t+a+l+k+|c+h+a+t+|c+o+m+e+|g+o+)\s+(?:i+n+\s+|t+o+\s+)?(?:p+r+i+v+a*t*e*y*|p+r+i+v+|p+v+|d+m+)"
    r")(?:\s+(?:p+l+z+|p+l+s+|p+l+e+a+s+e+))?\??$",
    re.IGNORECASE,
)

# Strictly matches Class 30 (reply_yes_no_maybe)
STRICT_YES_NO_MAYBE_RE = re.compile(
    r"^(?:"
    r"(?:y+e+s+|y+s+)(?:\s+(?:p+l+z+|p+l+s+|p+l+e+a+s+e+|s+u+r+e+|o+f+\s+c+o+u+r+s+e+))?|"
    r"(?:n+o+|n+o+p+e*|n+a+h+)(?:\s+(?:t+h+a+n+k+s+|t+h+x+|t+y+|s+o+r+y+|n+o+t+\s+r+e+a+l+y+))?|"
    r"(?:m+a+y+b+e*|m+b+y+|i+d+k+|i+\s+d+o+n+'?t+\s+k+n+o+w+|d+u+n+n+o+|n+o+t+\s+s+u+r+e+|i+m+\s+n+o+t+\s+s+u+r+e+|u+n+s+u+r+e+|d+e+p+e+n+d+s+)|"
    r"(?:n+o+\s+)?(?:i+\s+)?d+o+n+'?t+\s+(?:h+a+v+e*|u+s+e+)\s+(?:t+g+|t+e+l+e*g+r+a*m+|s+n+a+p+(?:c+h+a+t+)?|i+n+s+t+a+(?:g+r+a+m+)?|i+g+|w+h*a+t+s+a+p+p*|s+o+c+i+a+l+s*)|"
    r"(?:o+k+\s+|y+e+s+\s+)?l+e+t+'?s\s+(?:s+t+a+y+|t+a+l+k+|c+h+a+t+)\s+h+e+r+e+"
    r")$",
    re.IGNORECASE,
)

ASK_PRIVATE_BASES = (
    # Private / PV / DM / Telegram Requests (including user's exact "privty" & "telgrm" typos!)
    "can we talk private", "can we talk in private", "can we talk privty", "can we talk in privty",
    "can we talk privte", "can we talk privat", "can we chat private", "can we chat in private",
    "can we talk pv", "can we talk in pv", "can we talk in dm", "can we dm", "can i dm you",
    "can i dm u", "can i pm u", "can i text u", "can i message u", "can i msg u",
    "lets talk private", "let's talk private", "lets talk in private", "lets talk in pv",
    "lets move to private", "lets go private", "come private", "come pv", "talk private",
    "talk in private", "chat in private", "dm me", "pm me", "text me", "add me",
    "private?", "privty?", "priv?", "pv?", "dm?", "pm?", "inbox?",
    # Telegram & Username
    "can we talk on telegram", "can we talk on telgrm", "can we talk on tg", "lets talk on telegram",
    "whats your telegram", "whats ur telegram", "whats ur telgrm", "whats ur tg",
    "what is your telegram", "do you have telegram", "do u have telegram", "do u have tg",
    "u have telegram", "u have tg", "got telegram", "ur telegram", "ur tg", "telegram?", "telgrm?", "tg?",
    "whats your username", "whats ur username", "what is your username", "can i have ur username",
    "can i get ur username", "send ur username", "give me ur username", "ur username", "ur user",
    "username?", "add me on tg", "add me on telegram", "dm me on telegram",
    # External Socials (Snap, Insta, WhatsApp, Discord, Number, Video Call)
    "do you have snap", "do u have snap", "do u have snapchat", "u have snap", "got snap",
    "whats your snap", "whats ur snap", "whats ur snapchat", "ur snap", "snap?", "snapchat?",
    "do you have insta", "do u have insta", "do u have instagram", "do u have ig", "u have insta",
    "whats your insta", "whats ur insta", "whats ur instagram", "whats ur ig", "ur insta", "ur ig",
    "insta?", "instagram?", "ig?",
    "do you have whatsapp", "do u have whatsapp", "whats ur whatsapp", "whatsapp?",
    "do u have discord", "whats ur discord", "discord?",
    "whats your number", "whats ur number", "can i have ur number", "can i get ur number", "ur number?",
    "do u have socials", "whats ur socials", "can we video call", "video call?", "voice call?",
)

YES_NO_MAYBE_BASES = (
    # Yes (aux_id = 1)
    "yes", "yess", "yes please", "yes plz", "yes pls", "yes sure", "yes of course",
    # No / Don't have social (aux_id = 2)
    "no", "noo", "nope", "nah", "nop", "no thanks", "no thx", "no ty", "no sorry", "no not really",
    "i dont have snap", "i don't have snap", "i dont have snapchat", "dont have snap",
    "i dont have insta", "i don't have insta", "i dont have instagram", "dont have insta",
    "i dont have ig", "i dont have tg", "i dont have telegram", "i dont have whatsapp",
    "i dont have socials", "no i dont have snap", "no i dont have insta",
    # Maybe / Unsure / Stay Here (aux_id = 3)
    "maybe", "mayb", "mby", "idk", "i dont know", "i don't know", "dunno",
    "not sure", "im not sure", "i'm not sure", "unsure", "depends",
    "lets stay here", "let's stay here", "ok lets stay here", "lets talk here", "lets chat here",
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

    if STEP1_TO_11_COLLISION_RE.match(s):
        return 1, s

    words = [w.strip("'?,") for w in s.split() if w.strip("'?,")]
    if not words or len(words) > 6:
        return 0, ""

    for w in words:
        if w not in STRICT_ALLOWED_S12_VOCAB:
            return 0, ""

    if "?" not in s and STRICT_YES_NO_MAYBE_RE.match(s):
        return 30, s

    if STRICT_ASK_PRIVATE_RE.match(s):
        return 29, s

    return 0, ""


def _safe_fetch_parquet_lines(priv_map, yn_map):
    # Rule 12: Streams ONLY pure Parquet data (revision="refs/convert/parquet", zero remote .py scripts!)
    stats = {"scanned": 0, "blocked_overlaps": 0, "priv_added": 0, "yn_added": 0}
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
                        if 1 <= cid <= 28:
                            stats["blocked_overlaps"] += 1
                        elif cid == 29:
                            sig = _canonical_token_sig(clean_s)
                            if sig not in priv_map:
                                priv_map[sig] = clean_s
                                stats["priv_added"] += 1
                        elif cid == 30:
                            sig = _canonical_token_sig(clean_s)
                            if sig not in yn_map:
                                yn_map[sig] = clean_s
                                stats["yn_added"] += 1
        except Exception:
            pass
    return stats


def _write_u8_file(path, lines_iterable):
    sorted_lines = sorted(set(lines_iterable))
    u8 = np.frombuffer(("\n".join(sorted_lines) + "\n").encode("ascii"), dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(u8.tobytes())
    return len(sorted_lines), u8.nbytes


def build_private_yesno_files(
    priv_path="private_social_ask.txt",
    yn_path="yes_no_maybe.txt",
    neg_path="negatives.txt",
):
    priv_map = {}
    yn_map = {}

    for base in ASK_PRIVATE_BASES:
        if "?" in base:
            cid, c = categorize_line_across_all_steps(base)
            if cid == 29:
                priv_map[_canonical_token_sig(c)] = c
            continue
        for pfx in ("", "hey", "so", "btw"):
            full = f"{pfx} {base}".strip() if pfx else base
            for end in ("", "?"):
                cid, c = categorize_line_across_all_steps(f"{full}{end}")
                if cid == 29:
                    sig = _canonical_token_sig(c)
                    if sig not in priv_map:
                        priv_map[sig] = c

    for base in YES_NO_MAYBE_BASES:
        cid, c = categorize_line_across_all_steps(base)
        if cid == 30:
            sig = _canonical_token_sig(c)
            if sig not in yn_map:
                yn_map[sig] = c

    hf_stats = _safe_fetch_parquet_lines(priv_map, yn_map)

    for k_sig in list(yn_map.keys()):
        if k_sig in priv_map:
            del yn_map[k_sig]

    n_priv, b_priv = _write_u8_file(priv_path, priv_map.values())
    n_yn, b_yn = _write_u8_file(yn_path, yn_map.values())

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn:
            existing_neg = {line.strip().lower() for line in fn if line.strip()}
    for item in list(priv_map.values()) + list(yn_map.values()):
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ")):
            existing_neg.add(item)
    _, _ = _write_u8_file(neg_path, existing_neg)

    print(
        f"[PARQUET STEP 12 ANALYZER] Scanned {hf_stats['scanned']} rows | "
        f"Blocked {hf_stats['blocked_overlaps']} Step 1-11 overlaps | "
        f"Added {hf_stats['priv_added']} private/social & {hf_stats['yn_added']} yes/no lines!",
        flush=True,
    )
    print(
        f"[CLEAN STEP 12 RULES 12+13] Saved {n_priv} unique private/social lines to '{priv_path}' ({b_priv / 1024.0:.2f} KB) "
        f"& {n_yn} unique yes/no/maybe lines to '{yn_path}' ({b_yn / 1024.0:.2f} KB)",
        flush=True,
    )


if __name__ == "__main__":
    build_private_yesno_files()