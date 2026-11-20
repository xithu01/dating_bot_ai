#!/usr/bin/env python3
# FILE: build_step17_english_txt.py
# PURPOSE: Zero-CVE Rules 12+13 Dataset Builder for Step 17: Natural Conversational English (Classes 50-54)
#          Handles "Bcz you make it to fast.", "cuz nothing to do", "i believe u", "no way", "i guess", "why do u ask?"
# RULES: Enforces Rules 1-15 (Parquet-only revision="refs/convert/parquet", closed vocab whitelist,
#        strict cross-step collision guard protecting Steps 1-16 & Class 0 traps, canonical token deduplication).

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
# Strictly protects Steps 1-16 & Class 0 traps ("u reply so fast", "why not", "fair enough", "for real", "im bored", "Chicken")
STEP1_TO_16_COLLISION_RE = re.compile(
    r"^(?:"
    r"(?:w+h+a+t+|w+h+y+|w+h+o+|w+h+y+\s+n+o+t+|w+h+a+t+\s+i+s+\s+t+h+i+s+|w+h+a+t+\s+d+o+\s+(?:y+o+u+|u+)\s+d+o+|w+h+o+\s+(?:a+r+e+|r+)\s+(?:y+o+u+|u+))|"
    r"(?:(?:y+o+u+|u+)\s+(?:r+e+p+l+y+|a+n+s+w+e+r+)\s+(?:s+o+|t+o+o+)\s+f+a+s+t+)|"
    r"(?:f+a+i+r+\s+e+n+o+u+g+h+|f+o+r+\s+r+e+a+l+|r+e+a+l+|s+u+r+e+|d+e+f+i+n+i+t+e+l+y+|t+o+t+a+l+y+|o+f+c+|y+e+s+|n+o+|n+o+p+e+|n+a+h+|m+a+y+b+e+|i+d+k+|n+o+t+\s+s+u+r+e+)|"
    r"(?:(?:i+m+|i+'?m+|j+u+s+t+)?\s*(?:b+o+r+e+d+|l+o+n+e+l+y+|t+i+r+e+d+|c+h+i+l+i+n+g*|r+e+l+a+x+i+n+g+|c+a+n+'?t+\s+s+l+e+e+p+|n+o+t+h+i+n+g+\s+m+u+c+h+))|"
    r"(?:c+h+i+c+k+e+n+|p+i+z+z+a+|b+u+r+g+e+r+|t+r+u+t+h+|d+a+r+e+|n+i+g+h+t+\s+o+w+l+|c+o+f+f+e+e+|t+e+a+)|"
    r".*\b(?:h+r+u+|h+y+d+|w+y+d+|o+l+d+|a+g+e+|f+r+o+m+|c+o+u+n+t+r+y+|n+a+m+e+|s+i+n+g+e*l+e*|m+a+r+i+e*d+|t+a+k+e*n+|b+o+t+|a+i+|c+a+t+f+i+s+h+)\b.*|"
    r".*\b(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|p+r+i+v+a*t*e*y*|t+e+l+e*g+r+a*m+|s+n+a+p+|i+n+s+t+a+|s+e+l+f+i+e*|n+u+d+e+s*|s+e+x+t+|z+o+d+i+a+c+|j+o+k+e+)\b.*"
    r")$",
    re.IGNORECASE,
)

# Rule 12 Gate 2: Closed Whitelist Vocabulary for Step 17 Conversational English
STRICT_ALLOWED_S17_VOCAB = frozenset((
    # Casual "because" & fast-reply/typing vocabulary
    "bcz", "bc", "cuz", "cos", "cz", "because", "cause",
    "make", "makes", "made", "type", "typing", "text", "texting", "texted",
    "reply", "replies", "replied", "replying", "respond", "responded", "responding",
    "answer", "answered", "answering", "write", "writing", "wrote", "send", "sent",
    "fast", "quick", "quickly", "instant", "instantly", "speed", "second", "seconds", "sec",
    # Reasoning / Boredom / Chatting explanation
    "bored", "nothing", "else", "alone", "lonely", "home", "asleep", "sleep", "awake",
    "passing", "killing", "time", "felt", "chat", "chatting", "talk", "talking", "everyone", "nobody",
    # Trust / Believe / Doubt / Disbelief
    "believe", "trust", "makes", "sense", "got", "see", "glad", "hear", "good", "know",
    "way", "really", "seriously", "serious", "sure", "cap", "lying", "lie", "liar", "doubt",
    "honest", "honestly", "swear", "promise", "kidding", "joking", "jk", "crazy", "wild", "weird", "funny",
    # Natural English conversation bridges
    "guess", "think", "hope", "hopefully", "probably", "later", "knows", "kinda", "kind",
    "sorta", "sort", "little", "bit", "nevermind", "nvm", "forget", "matter", "doesnt", "doesn't",
    "whatever", "mind", "care", "special", "always", "depends", "maybe", "course", "obviously",
    # Curious / Asking Why
    "why", "ask", "asked", "asking", "curious", "say", "said", "come", "happened", "wrong", "matter",
    # Shared English grammar words
    "i", "im", "i'm", "lm", "am", "u", "you", "ur", "your", "youre", "you're", "we", "me", "my",
    "it", "its", "it's", "that", "thats", "that's", "this", "is", "are", "ar", "r", "was", "were",
    "do", "does", "did", "didnt", "didn't", "dont", "don't", "cant", "can't", "will", "would", "can",
    "have", "had", "want", "like", "to", "too", "so", "very", "back", "in", "a", "an", "the", "of",
    "for", "on", "at", "with", "up", "about", "just", "only", "not", "no", "ok", "okay", "alright",
    "oh", "ah", "haha", "yeah", "right", "now", "here", "stop", "lets", "let's", "for", "fr",
))

# 6 Disjoint Step 17 Natural English Sub-Classes (Leaves 90..95)
S17_SPECS = (
    # Leaf 90 (Class 50): Fast-Reply / "Bcz you make it to fast." explanations
    (90, "s17_fast_reply.txt", (
        "bcz you make it to fast", "bcz u make it to fast", "bcz you make it too fast",
        "bcz u make it too fast", "because you make it too fast", "because you make it to fast",
        "cuz you make it too fast", "cuz u make it to fast", "bc you make it too fast",
        "you make it too fast", "u make it too fast", "you make it to fast", "u make it to fast",
        "bcz u reply too fast", "bcz you reply too fast", "bcz u reply so fast", "bcz u reply to fast",
        "because you reply too fast", "because u reply so fast", "cuz u reply too fast", "bc u reply so fast",
        "bcz u type too fast", "bcz you type so fast", "because you type too fast", "cuz u type so fast",
        "u type too fast", "you type too fast", "u type so fast", "you type so fast", "u typing so fast",
        "bcz u answer too fast", "because u answer so fast", "cuz u answer too quick", "bc u answered so fast",
        "u answered so fast", "u answered too fast", "u answered in a second", "u replied in a second",
        "u respond too fast", "u respond so fast", "u respond too quick", "u text back so fast",
        "u text back too fast", "you text back so quick", "u write so fast", "u send it so fast",
        "ur replies are too fast", "your replies are so fast", "that was instant", "that was too fast",
        "that was so fast", "too fast", "so fast", "way too fast", "bcz ur too fast", "cuz ur so fast",
        "bcz u respond instantly", "u replied instantly", "instant reply",
    )),
    # Leaf 91 (Class 51): Natural English "Because / Why" Explanations (bored, alone, passing time)
    (91, "s17_reason_chat.txt", (
        "bcz im bored", "cuz im bored", "because im bored", "bc im bored", "cos im bored",
        "bcz nothing to do", "cuz nothing to do", "because nothing to do", "bc i have nothing to do",
        "bcz i have nothing to do", "because i have nothing to do", "i have nothing to do",
        "nothing else to do", "got nothing to do", "had nothing to do", "nothing to do",
        "bcz i cant sleep", "cuz i cant sleep", "because i cant sleep", "bc i can't sleep",
        "bcz im alone", "cuz im alone at home", "because im lonely", "bcz everyone is asleep",
        "because everyone is asleep", "bcz i want to chat", "cuz i want to talk", "because i felt like chatting",
        "just passing time", "passing time", "just killing time", "killing time", "just felt like talking",
        "just felt like chatting", "bcz im awake", "cuz im still awake",
    )),
    # Leaf 92 (Class 52): Trust, Relief & Understanding Reactions
    (92, "s17_trust_ok.txt", (
        "i believe u", "i believe you", "ok i believe u", "ok i believe you", "okay i believe u",
        "haha ok i believe u", "alright i believe u", "i trust u", "i trust you", "ok i trust u",
        "makes sense", "that makes sense", "oh that makes sense", "got it", "oh got it",
        "i see", "oh i see", "ah i see", "good to know", "nice to know", "glad to hear that",
        "thats good", "that's good", "thats good to know", "oh ok i see", "alright fair",
    )),
    # Leaf 93 (Class 52): Doubt, Teasing, Surprise & Disbelief
    (93, "s17_doubt_tease.txt", (
        "i dont believe u", "i don't believe u", "i dont believe you", "i don't believe you",
        "dont believe u", "don't believe u", "no way", "no way?", "really?", "for real?", "fr?",
        "seriously?", "are u serious", "are u serious?", "are you serious", "are you serious?",
        "u serious?", "r u serious?", "u sure?", "are u sure", "are u sure?", "are you sure?",
        "r u sure?", "cap", "thats cap", "stop lying", "ur lying", "youre lying", "you're lying",
        "dont lie", "don't lie", "dont lie to me", "liar", "i doubt it", "doubt it",
        "yeah right", "are u kidding", "are u kidding me", "u kidding?", "u joking?", "no joke?",
        "thats crazy", "that's crazy", "thats wild", "thats weird", "thats funny",
    )),
    # Leaf 94 (Class 53): Natural English Casual Conversation Bridges
    (94, "s17_casual_bridge.txt", (
        "i guess", "i guess so", "guess so", "i think so", "think so", "i hope so", "hope so",
        "hopefully", "probably", "maybe later", "we will see", "we'll see", "lets see", "let's see",
        "who knows", "not really", "not much really", "kind of", "kinda", "sort of", "sorta",
        "a little bit", "just a little", "a bit", "for sure", "obviously", "of course",
        "nevermind", "nvm", "forget it", "doesnt matter", "doesn't matter", "it doesnt matter",
        "up to u", "up to you", "whatever u want", "whatever you want", "i dont mind", "i don't mind",
        "i dont care", "i don't care", "nothing special", "same as always", "its whatever",
    )),
    # Leaf 95 (Class 54): Curious "Why Do You Ask?" / "How Come?" Follow-Ups (NEVER bare "why" or "why not"!)
    (95, "s17_ask_why.txt", (
        "why do u ask", "why do u ask?", "why do you ask", "why do you ask?", "why u ask", "why u ask?",
        "why did u ask", "why did u ask?", "why did you ask", "why did you ask?", "why asking", "why asking?",
        "just curious", "im just curious", "i'm just curious", "curious",
        "why would u say that", "why u say that", "why did u say that", "what makes u say that",
        "how come", "how come?", "what happened", "what happened?", "whats wrong", "what's wrong",
        "whats wrong?", "is something wrong", "anything wrong?",
    )),
)

PARQUET_MINING_PATTERNS = (
    (90, re.compile(r"^(?:(?:bcz|bc|cuz|because)\s+)?(?:you|u)\s+(?:reply|type|answer|respond|text|make\s+it)\s+(?:so|too|to)\s+(?:fast|quick)$", re.I)),
    (91, re.compile(r"^(?:(?:bcz|bc|cuz|because)\s+(?:i'?m\s+bored|i\s+have\s+nothing\s+to\s+do|i\s+can'?t\s+sleep)|just\s+(?:passing|killing)\s+time|nothing\s+else\s+to\s+do)$", re.I)),
    (92, re.compile(r"^(?:(?:ok\s+)?i\s+(?:believe|trust)\s+(?:you|u)|(?:that\s+)?makes\s+sense|(?:oh\s+)?i\s+see|got\s+it|good\s+to\s+know|glad\s+to\s+hear\s+that)$", re.I)),
    (93, re.compile(r"^(?:i\s+don'?t\s+believe\s+(?:you|u)|no\s+way|are\s+(?:you|u)\s+(?:serious|sure|kidding)|(?:really|seriously)\?|don'?t\s+lie|i\s+doubt\s+it|that'?s\s+crazy)$", re.I)),
    (94, re.compile(r"^(?:i\s+(?:guess|think|hope)\s+so|i\s+guess|maybe\s+later|who\s+knows|not\s+really|kind\s+of|nevermind|forget\s+it|up\s+to\s+(?:you|u)|i\s+don'?t\s+(?:mind|care)|doesn'?t\s+matter)$", re.I)),
    (95, re.compile(r"^(?:why\s+(?:do|did)\s+(?:you|u)\s+ask|how\s+come|just\s+curious|what\s+happened|what'?s\s+wrong)\??$", re.I)),
)


def _sanitize_ascii_bytes(raw_text):
    if not isinstance(raw_text, str):
        return ""
    b = np.frombuffer(raw_text[:64].encode("ascii", errors="ignore"), dtype=np.uint8)
    valid = b[(b >= np.uint8(32)) & (b <= np.uint8(122))]
    return valid.tobytes().decode("ascii", errors="ignore") if valid.shape[0] > 0 else ""


def _collapse_str(s):
    return "".join(s[i] for i in range(len(s)) if i == 0 or s[i] != s[i - 1])


def _canonical_token_sig(s):
    clean_w = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in s.lower())
    return (tuple(w for w in _collapse_str(clean_w).split() if w), "?" in s)


def _validate_clean_s17(raw_text):
    clean_ascii = _sanitize_ascii_bytes(raw_text)
    if "/" in clean_ascii:
        return ""
    s = SPACE_RE.sub(" ", clean_ascii.strip().lower()).rstrip("!.")
    if not s or len(s) > 44 or not CLEAN_LINE_RE.match(s):
        return ""
    if STEP1_TO_16_COLLISION_RE.match(s):
        return ""
    words = [w.strip("'?,") for w in s.split() if w.strip("'?,")]
    if not words or len(words) > 6:
        return ""
    if any(w not in STRICT_ALLOWED_S17_VOCAB for w in words):
        return ""
    return s


def _safe_fetch_parquet_english(leaf_maps):
    stats = {"scanned": 0, "mined": 0}
    try:
        from datasets import load_dataset
    except ImportError:
        return stats
    for repo, col, cap in (
        ("daily_dialog", "dialog", 3000),
        ("bavard/personachat_truecased", "history", 2000),
        ("AlekseyKorshuk/persona-chat", "utterances", 2000),
    ):
        try:
            ds = load_dataset(repo, split="train", revision="refs/convert/parquet", streaming=True)
            c = 0
            for row in ds:
                if c >= cap:
                    break
                c += 1
                stats["scanned"] += 1
                val = row.get(col, ())
                turns = (val,) if isinstance(val, str) else (val if isinstance(val, (list, tuple)) else ())
                for turn in turns:
                    if not isinstance(turn, str):
                        continue
                    for sent in re.split(r"[.!]+", turn[:100]):
                        cs = _validate_clean_s17(sent)
                        if not cs:
                            continue
                        for lf, pat in PARQUET_MINING_PATTERNS:
                            if pat.match(cs):
                                sig = _canonical_token_sig(cs)
                                if sig not in leaf_maps[lf]:
                                    leaf_maps[lf][sig] = cs
                                    stats["mined"] += 1
                                break
        except Exception:
            pass
    return stats


def build_step17_english_files():
    leaf_maps = {lf: {} for lf, _, _ in S17_SPECS}
    for lf, _, phrases in S17_SPECS:
        for p in phrases:
            c = _validate_clean_s17(p)
            if c:
                sig = _canonical_token_sig(c)
                if sig not in leaf_maps[lf]:
                    leaf_maps[lf][sig] = c

    hf_stats = _safe_fetch_parquet_english(leaf_maps)

    # Disjoint priority across leaves 90..95
    seen = set()
    tot_lines, tot_bytes = 0, 0
    for lf, fn, _ in S17_SPECS:
        for sig in list(leaf_maps[lf].keys()):
            if sig in seen:
                del leaf_maps[lf][sig]
            else:
                seen.add(sig)
        sorted_lines = sorted(set(leaf_maps[lf].values()))
        u8 = np.frombuffer(("\n".join(sorted_lines) + "\n").encode("ascii"), dtype=np.uint8)
        with open(fn, "wb") as f:
            f.write(u8.tobytes())
        tot_lines += len(sorted_lines)
        tot_bytes += u8.nbytes

    print(
        f"[PARQUET STEP 17 ENGLISH] Scanned {hf_stats['scanned']} rows | Mined {hf_stats['mined']} natural English lines!",
        flush=True,
    )
    print(
        f"[CLEAN STEP 17 RULES 12+13] Saved {tot_lines} unique lines across 6 s17_*.txt files ({tot_bytes / 1024.0:.2f} KB) — ZERO Step 1-16 overlap!",
        flush=True,
    )


if __name__ == "__main__":
    build_step17_english_files()