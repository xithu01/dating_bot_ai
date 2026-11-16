# FILE: extract_jsonl_greetings.py
# PURPOSE: Extract 100% clean Class 1 greetings & Class 0 non-greetings from bot_chat_history.jsonl
# PYARMOR COMPATIBLE: < 120 lines, < 5 KB

import json
import os
import re
from build_greetings_txt import is_valid_clean_greeting

REPEAT_CHAR_RE = re.compile(r"(.)\1{3,}")
SPACE_RE = re.compile(r"\s+")


def process_jsonl_logs(
    jsonl_path="bot_chat_history.jsonl",
    pos_path="greetings.txt",
    neg_path="negatives.txt",
):
    if not os.path.exists(jsonl_path):
        print(f"[SKIP] '{jsonl_path}' not found.")
        return

    existing_pos = set()
    if os.path.exists(pos_path):
        with open(pos_path, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                s = line.strip().lower()
                if s:
                    existing_pos.add(s)

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                s = line.strip().lower()
                if s:
                    existing_neg.add(s)

    new_pos = set()
    new_neg = set()
    total_rows = 0

    with open(jsonl_path, "r", encoding="utf-8", errors="ignore") as jf:
        for line in jf:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("sender") == "system":
                continue

            raw_text = record.get("text", "")
            if not isinstance(raw_text, str):
                continue
            text = raw_text.strip()
            if not text or text.startswith("[") or len(text) > 60 or not text.isascii():
                continue

            norm = SPACE_RE.sub(" ", REPEAT_CHAR_RE.sub(r"\1\1\1", text.lower())).strip()
            if not norm:
                continue

            clean_pos = is_valid_clean_greeting(norm)
            if clean_pos:
                if clean_pos not in existing_pos:
                    new_pos.add(clean_pos)
            else:
                if norm not in existing_pos and norm not in existing_neg:
                    if len(existing_neg) + len(new_neg) < 15000:
                        new_neg.add(norm)

    if new_pos:
        with open(pos_path, "a", encoding="ascii") as out_p:
            for item in sorted(new_pos):
                out_p.write(item + "\n")

    if new_neg:
        with open(neg_path, "a", encoding="ascii") as out_n:
            for item in sorted(new_neg):
                out_n.write(item + "\n")

    print(f"[SCAN] Processed {total_rows} JSONL rows from '{jsonl_path}'")
    print(f"[CLASS 1] Added {len(new_pos)} pure greetings -> '{pos_path}' ({len(existing_pos) + len(new_pos)} total)")
    print(f"[CLASS 0] Added {len(new_neg)} non-greetings -> '{neg_path}' ({len(existing_neg) + len(new_neg)} total)")


if __name__ == "__main__":
    process_jsonl_logs()