# FILE: extract_jsonl_greetings.py
# PURPOSE: Stream bot_chat_history.jsonl, extract clean English greetings, and append unique ones to greetings.txt

import json
import os
import re

# Matches pure English chat greetings including real chat typos seen in your logs (hlo, hy, hie, hei, hiyaa, wassuppp)
GREETING_PATTERN = re.compile(
    r"^(?:"
    r"h+i+|h+e+y+|h+e+l+o+|h+l+o+|h+y+|h+i+e+|h+e+i+|h+i+y+a+|h+e+y+a+|h+u+l+o+|e+l+o+|"
    r"y+o+|s+u+p+|w+a+s+u+p+|w+h+a+t+s\s+u+p+|w+h+a+t\s+is\s+u+p+|h+o+w+d+y+|"
    r"g+r+e+e+t+i+n+g+s*|s+a+l+u+t+a+t+i+o+n+s*|w+e+l+c+o+m+e+|a+l+o+h+a+|g\s*d+a+y+|"
    r"g+o+o+d\s+(?:m+o+r+n+i+n+g*|a+f+t+e+r+n+o+o+n+|e+v+e+n+i+n+g*|d+a+y+)|"
    r"m+o+r+n+i+n+g*|a+f+t+e+r+n+o+o+n+|e+v+e+n+i+n+g+|"
    r"h+o+w\s+(?:a+r+e|r)\s+(?:y+o+u|u|y+a)(?:\s+d+o+i+n+g*)?|h+r+u+|h+y+d+|"
    r"h+o+w+s\s+(?:i+t\s+g+o+i+n+g*|e+v+e+r+y+t+h+i+n+g*|u+r\s+d+a+y+)|"
    r"n+i+c+e\s+t+o\s+m+e+e+t\s+(?:y+o+u|u)|l+o+n+g\s+t+i+m+e\s+n+o\s+s+e+e+"
    r")(?:[\s!?.:,;)(\-]+(?:t+h+e+r+e+|y+o+u|u|b+r+o+|m+a+n+|m+a+t+e+|f+r+i+e+n+d+|s+t+r+a+n+g+e+r+|h+a+h+a+|l+o+l+))?[\s!?.:;)(]*$",
    re.IGNORECASE,
)

REPEAT_CHAR_RE = re.compile(r"(.)\1{3,}")
SPACE_RE = re.compile(r"\s+")


def clean_greeting(raw_text):
    if not isinstance(raw_text, str):
        return ""
    text = raw_text.strip()
    if not text or text.startswith("[") or len(text) > 30:
        return ""
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return ""

    # Normalize extreme letter spam (e.g., "hiiiiiii" -> "hiii")
    normalized = REPEAT_CHAR_RE.sub(r"\1\1\1", text.lower())
    normalized = SPACE_RE.sub(" ", normalized).strip()

    if len(normalized) > 25 or len(normalized.split()) > 4:
        return ""
    if GREETING_PATTERN.match(normalized):
        return normalized
    return ""


def append_from_jsonl(jsonl_path="bot_chat_history.jsonl", txt_path="greetings.txt"):
    existing = set()
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                s = line.strip().lower()
                if s:
                    existing.add(s)

    new_greetings = set()
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

            cleaned = clean_greeting(record.get("text", ""))
            if cleaned and cleaned not in existing:
                new_greetings.add(cleaned)

    if new_greetings:
        needs_newline = False
        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
            with open(txt_path, "rb") as check_f:
                check_f.seek(-1, os.SEEK_END)
                needs_newline = check_f.read(1) != b"\n"

        with open(txt_path, "a", encoding="ascii") as out_f:
            if needs_newline:
                out_f.write("\n")
            for item in sorted(new_greetings):
                out_f.write(item + "\n")

    print(f"[SCAN] Read {total_rows} lines from '{jsonl_path}'")
    print(f"[ADDED] Appended {len(new_greetings)} new unique greetings to '{txt_path}'")
    print(f"[TOTAL] '{txt_path}' now has {len(existing) + len(new_greetings)} total unique greetings")


if __name__ == "__main__":
    append_from_jsonl()