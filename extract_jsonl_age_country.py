# FILE: extract_jsonl_age_country.py
# PURPOSE: Mine real age & country replies (after "from?" / "age?") from bot_chat_history.jsonl
# PYARMOR COMPATIBLE: < 140 lines, < 5.5 KB

import json
import os
import re

SPACE_RE = re.compile(r"\s+")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")

# Triggers when bot/user asks for location or age in bot_chat_history.jsonl
ASK_FROM_RE = re.compile(r"^(?:from\??|where\s+(?:are\s+you|r\s+u|u)\s+from\??|u\s+from\??|country\??)$", re.IGNORECASE)
ASK_AGE_RE = re.compile(r"^(?:age\??|ur\s+age\??|your\s+age\??|how\s+old(?:\s+are\s+you|\s+r\s+u)?\??)$", re.IGNORECASE)

INLINE_AGE_COUNTRY_RE = re.compile(
    r"\b(?:im|i'm|l'm|i\s+am|from|in)\s+(?:[1-7][0-9]|[a-z]{2,12})\b",
    re.IGNORECASE,
)

NON_LOCATION_SKIP = frozenset((
    "hi", "hey", "hello", "yo", "sup", "yes", "no", "ok", "okay", "why", "what",
    "u", "you", "u first", "secret", "none", "idk", "bye", "lol", "haha", "same",
))


def extract_age_country_from_jsonl(
    jsonl_path="bot_chat_history.jsonl",
    staging_path="age_country_staging.txt",
    neg_path="negatives.txt",
):
    if not os.path.exists(jsonl_path):
        print(f"[SKIP] '{jsonl_path}' not found (using synthetic age/country dataset).")
        return

    staged = set()
    if os.path.exists(staging_path):
        with open(staging_path, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                s = line.strip().lower()
                if s:
                    staged.add(s)

    neg_set = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                s = line.strip().lower()
                if s:
                    neg_set.add(s)

    prev_uid = None
    prev_was_question = False
    mined_count = 0
    total_rows = 0

    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as jf:
        for line in jf:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            sender = rec.get("sender", "")
            uid = rec.get("uid")
            if sender == "system":
                prev_was_question = False
                continue

            raw_t = str(rec.get("text", "")).strip()
            if not raw_t or raw_t.startswith("[") or not raw_t.isascii():
                prev_was_question = False
                continue

            norm = SPACE_RE.sub(" ", REPEAT_CHAR_RE.sub(r"\1\1", raw_t.lower())).strip()
            if not norm or len(norm) > 30:
                prev_was_question = False
                continue

            # 1. Check if this message is a direct reply to "from?" or "age?" in the same chat session
            if prev_was_question and uid == prev_uid and norm not in NON_LOCATION_SKIP:
                if len(norm.split()) <= 4 and norm not in staged:
                    staged.add(norm)
                    neg_set.add(norm)
                    mined_count += 1

            # 2. Check if this message contains inline age/country ("im 24", "from tun", "hi l'm 30 from tun")
            if INLINE_AGE_COUNTRY_RE.search(norm) or ASK_FROM_RE.match(norm) or ASK_AGE_RE.match(norm):
                if norm not in staged:
                    staged.add(norm)
                    neg_set.add(norm)
                    mined_count += 1

            prev_uid = uid
            prev_was_question = bool(ASK_FROM_RE.match(norm) or ASK_AGE_RE.match(norm))

    with open(staging_path, "w", encoding="ascii") as fs:
        for item in sorted(staged):
            fs.write(item + "\n")

    with open(neg_path, "w", encoding="ascii") as fn:
        for item in sorted(neg_set):
            fn.write(item + "\n")

    print(f"[JSONL MINER] Scanned {total_rows} rows -> Mined {mined_count} real age/country lines into '{staging_path}' & '{neg_path}'")


if __name__ == "__main__":
    extract_age_country_from_jsonl()