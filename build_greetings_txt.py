# FILE: build_greetings_txt.py
# PURPOSE: Build 100% pure greetings.txt (Class 1) & anti-poisoned negatives.txt (Class 0)
# PYARMOR COMPATIBLE: < 245 lines, < 10 KB

import json
import logging
import os
import re
import warnings

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CLEAN_CHARS_RE = re.compile(r"^[a-z\s'!?.,]+$")
SPACE_RE = re.compile(r"\s+")
REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")

# Prevents JSONL/HF lines starting with greeting words from accidentally entering negatives.txt
GREETING_PREFIX_GUARD_RE = re.compile(
    r"^(?:h+i+|h+e+y+|h+e+l+o+|h+l+o+|h+y+|h+i+e+|h+e+i+|h+i+y+a+|h+e+y+a+|e+l+o+|"
    r"y+o+|s+u+p+|w+a+s+u+p+|w+h+a+t+'?s\s+u+p+|h+o+w+d+y+|g+o+o+d\s+m+o+r+n+i+n+g*|"
    r"m+o+r+n+i+n+g*|g+o+o+d\s+e+v+e+n+i+n+g*|g+r+e+e+t+i+n+g+s*|w+e+l+c+o+m+e+|h+r+u+|h+y+d+)\b",
    re.IGNORECASE,
)

STRICT_GREETING_RE = re.compile(
    r"^(?:"
    r"h+i+|h+e+y+|h+e+l+o+|h+l+o+|h+y+|h+i+e+|h+e+i+|h+i+y+a+|h+e+y+a+|h+u+l+o+|e+l+o+|"
    r"y+o+|s+u+p+|w+a+s+u+p+|w+a+z+u+p+|w+h+a+t+'?s\s+u+p+|w+h+a+t\s+i+s\s+u+p+|h+o+w+d+y+|"
    r"g+r+e+e+t+i+n+g+s*|s+a+l+u+t+a+t+i+o+n+s*|w+e+l+c+o+m+e+|a+l+o+h+a+|g\s*d+a+y+|"
    r"(?:g+o+o+d\s+)?(?:m+o+r+n+i+n+g*|m+o+r+n+i+n+|a+f+t+e+r+n+o+o+n+|e+v+e+n+i+n+g*|d+a+y+)|"
    r"h+o+w\s+(?:a+r+e|r)\s+(?:y+o+u|u|y+a)(?:\s+d+o+i+n+g*|\s+t+o+d+a+y+)?|h+r+u+|h+y+d+|"
    r"h+o+w+'?s\s+(?:i+t\s+g+o+i+n+g*|e+v+e+r+y+t+h+i+n+g*|(?:y+o+u+r|u+r)\s+d+a+y+)|"
    r"h+o+w\s+i+s\s+(?:i+t\s+g+o+i+n+g*|(?:y+o+u+r|u+r)\s+d+a+y+)|h+o+w\s+h+a+v+e\s+(?:y+o+u|u)\s+b+e+e+n+|"
    r"(?:n+i+c+e|p+l+e+a+s+e+d|g+l+a+d|g+o+o+d)\s+t+o\s+(?:m+e+e+t|s+e+e)\s+(?:y+o+u|u)|"
    r"l+o+n+g\s+t+i+m+e\s+n+o\s+s+e+e+"
    r")"
    r"(?:[\s,!?.]+(?:"
    r"t+h+e+r+e+|y+o+u|u|f+r+i+e+n+d+|s+t+r+a+n+g+e+r+|c+u+t+i+e+|g+o+r+g+e+o+u+s+|h+a+n+d+s+o+m+e+|"
    r"b+r+o+|b+u+d+d+y+|a+l+l+|e+v+e+r+y+o+n+e+|"
    r"h+o+w\s+(?:a+r+e|r)\s+(?:y+o+u|u)|h+r+u+|h+y+d+|w+h+a+t+'?s\s+u+p+|w+a+s+u+p+|"
    r"h+o+w+'?s\s+i+t\s+g+o+i+n+g*|n+i+c+e\s+t+o\s+m+e+e+t\s+(?:y+o+u|u)|"
    r"g+o+o+d\s+m+o+r+n+i+n+g+|g+o+o+d\s+e+v+e+n+i+n+g+"
    r"))?[\s!?.]*$",
    re.IGNORECASE,
)

BLACKLIST_WORDS = frozenset((
    "siri", "alexa", "cortana", "google", "olly", "computer", "bot", "robot", "assistant",
    "help", "order", "refund", "price", "cost", "weather", "music", "timer", "alarm",
    "account", "ticket", "support", "problem", "issue", "cancel", "stop", "bye", "night",
    "from", "age", "old", "tun", "tunisia", "usa", "uk", "india", "france", "germany",
    "im", "i'm", "l'm", "male", "female", "boy", "girl", "m", "f",
))

HELLO_SALUTATIONS = (
    "hi", "hii", "hiii", "hey", "heyy", "heyyy", "hello", "helloo", "hellooo",
    "yo", "yoo", "yooo", "hiya", "hiyaa", "heya", "heyaa", "howdy", "ello",
    "hlo", "hy", "hyy", "hie", "hei", "hullo",
)

TIME_AND_FORMAL = (
    "good morning", "morning", "mornin", "good afternoon", "afternoon",
    "good evening", "evening", "good day", "greetings", "salutations", "welcome", "aloha",
)

SLANG_AND_QUESTIONS = (
    "sup", "supp", "wassup", "wassupp", "wazzup", "whats up", "what's up",
    "what is up", "how are you", "how r u", "how are u", "how are ya", "how u doing",
    "how are you doing", "how are you today", "hows it going", "how's it going",
    "how is it going", "hows everything", "how have you been", "how is your day",
    "hows ur day", "hru", "hyd", "nice to meet you", "nice to meet u",
    "pleased to meet you", "glad to meet you", "good to see you", "long time no see",
)

TARGETS = ("", "there", "u", "you", "friend", "stranger", "cutie", "gorgeous", "handsome", "bro", "buddy")

FOLLOW_QUESTIONS = (
    "", "hru", "hyd", "whats up", "what's up", "wassup", "how r u", "how are u",
    "how are you", "hows it going", "how's it going", "nice to meet u",
    "nice to meet you", "good morning", "good evening",
)

ENDINGS = ("", "!", "?")

# Explicitly includes "hi m", "hi f", "hey m", "hey f", "hello m", "hello f" as Class 0
CORE_HARD_NEGATIVES = (
    "hi m", "hi f", "hey m", "hey f", "hello m", "hello f", "yo m", "yo f", "sup m", "sup f",
    "hii m", "hii f", "heyy m", "heyy f", "m hi", "f hi", "m hey", "f hey", "hi m here", "hi f here",
    "hi im from tun", "hi im from tunisia", "hey im from usa", "hello from tunisia",
    "hi from tun", "hey from uk", "hi im twenty from tun", "hi m from tun", "hey f from usa",
    "hi how old are you", "hey where are you from", "f or m", "m or f", "f?", "m?",
    "from?", "so", "m", "f", "u", "u d?", "u f?", "m here", "f here", "same",
    "you in bed?", "your name?", "ur name?", "charlotte", "from", "age?", "m bi",
    "do you get my text!", "how old are you", "where are you from", "asl",
    "are you single", "are you a boy or girl", "what is your name",
    "tun", "tunisia", "india", "usa", "uk", "canada", "germany", "france", "italy", "spain",
    "help", "helping", "helper", "helmet", "helicopter", "hell", "shell", "high",
    "higher", "highway", "hill", "history", "historical", "him", "his", "hit",
    "hide", "hidden", "he", "her", "hers", "here", "hero", "hear", "heart",
    "heat", "heavy", "health", "hello kitty", "hellod", "hellox", "xhello",
    "good", "good job", "good luck", "good night", "gn", "morning coffee", "morning news",
    "you", "your", "yours", "young", "youtube", "yo mama", "supper", "super",
    "support", "supply", "how", "how old", "how much", "how far", "how tall", "how to",
    "what time is it", "what do you do", "what", "why", "who", "bye", "goodbye",
    "see you", "see ya", "take care", "later", "cya", "gtg", "thank you", "thanks",
    "thx", "ty", "ok", "okay", "yes", "no", "maybe", "lol", "haha", "bored",
    "bonjour", "salut", "coucou", "bonsoir", "salam", "marhba", "ahla", "kifech",
    "hola", "buenos dias", "ciao", "hallo", "namaste", "konnichiwa", "privet", "merhaba", "oi", "ola",
)


def is_valid_clean_greeting(text):
    if not isinstance(text, str):
        return ""
    s = SPACE_RE.sub(" ", text.strip().lower())
    if not s or len(s) > 30:
        return ""
    words = s.split()
    if len(words) == 0 or len(words) > 4:
        return ""
    if not CLEAN_CHARS_RE.match(s):
        return ""
    for w in words:
        if w.strip("'!?.,") in BLACKLIST_WORDS:
            return ""
    if not STRICT_GREETING_RE.match(s):
        return ""
    return s


def is_safe_negative_line(s):
    if not s or len(s) > 30 or not CLEAN_CHARS_RE.match(s):
        return False
    # Never allow a line starting with a greeting word into negatives.txt unless it has a blacklist word
    if GREETING_PREFIX_GUARD_RE.match(s):
        words = [w.strip("'!?.,") for w in s.split()]
        if not any(w in BLACKLIST_WORDS for w in words):
            return False
    return True


def pull_datasets_pos_and_neg(pos_set, neg_set):
    try:
        from datasets import load_dataset
        ds_tanaos = load_dataset("tanaos/synthetic-intent-classifier-dataset-v1", split="train")
        for txt_val, lbl_val in zip(ds_tanaos["text"], ds_tanaos["label"]):
            s = SPACE_RE.sub(" ", str(txt_val).strip().lower())
            if not s or not s.isascii():
                continue
            if int(lbl_val) == 0:
                c = is_valid_clean_greeting(s)
                if c:
                    pos_set.add(c)
            elif len(neg_set) < 8000 and is_safe_negative_line(s[:30]):
                neg_set.add(s[:30])
    except Exception:
        pass

    if os.path.exists("bot_chat_history.jsonl"):
        try:
            with open("bot_chat_history.jsonl", "r", encoding="utf-8", errors="ignore") as jf:
                for line in jf:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("sender") == "system":
                        continue
                    raw_t = str(rec.get("text", "")).strip()
                    if not raw_t or raw_t.startswith("[") or not raw_t.isascii():
                        continue
                    norm = SPACE_RE.sub(" ", REPEAT_CHAR_RE.sub(r"\1\1", raw_t.lower())).strip()
                    c = is_valid_clean_greeting(norm)
                    if c:
                        pos_set.add(c)
                    elif len(neg_set) < 12000 and is_safe_negative_line(norm[:30]):
                        neg_set.add(norm[:30])
        except Exception:
            pass


def generate_greetings_and_negatives(pos_path="greetings.txt", neg_path="negatives.txt"):
    pos_set = set()
    neg_set = set(CORE_HARD_NEGATIVES)

    pull_datasets_pos_and_neg(pos_set, neg_set)

    for sal in HELLO_SALUTATIONS:
        for tgt in TARGETS:
            base = f"{sal} {tgt}".strip()
            for end in ENDINGS:
                c = is_valid_clean_greeting(f"{base}{end}")
                if c:
                    pos_set.add(c)
        for fq in FOLLOW_QUESTIONS:
            base = f"{sal} {fq}".strip()
            for end in ENDINGS:
                c = is_valid_clean_greeting(f"{base}{end}")
                if c:
                    pos_set.add(c)

    for tf in TIME_AND_FORMAL:
        for tgt in TARGETS:
            base = f"{tf} {tgt}".strip()
            for end in ("", "!"):
                c = is_valid_clean_greeting(f"{base}{end}")
                if c:
                    pos_set.add(c)

    for sq in SLANG_AND_QUESTIONS:
        for tgt in ("", "there", "bro", "stranger", "friend"):
            base = f"{sq} {tgt}".strip()
            for end in ENDINGS:
                c = is_valid_clean_greeting(f"{base}{end}")
                if c:
                    pos_set.add(c)

    neg_set.difference_update(pos_set)

    sorted_pos = sorted(pos_set)
    with open(pos_path, "w", encoding="ascii") as fp:
        for line in sorted_pos:
            fp.write(line + "\n")

    sorted_neg = sorted(neg_set)
    with open(neg_path, "w", encoding="ascii") as fn:
        for line in sorted_neg:
            fn.write(line + "\n")

    print(f"[CLEAN BUILD] Wrote {len(sorted_pos)} pure greetings to '{pos_path}'", flush=True)
    print(f"[CLEAN BUILD] Wrote {len(sorted_neg)} anti-poisoned negatives to '{neg_path}'", flush=True)


def generate_greetings_file(output_path="greetings.txt"):
    generate_greetings_and_negatives(pos_path=output_path, neg_path="negatives.txt")


if __name__ == "__main__":
    generate_greetings_and_negatives()