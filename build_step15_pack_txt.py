#!/usr/bin/env python3
# FILE: build_step15_pack_txt.py
# PURPOSE: Zero-CVE 5-Dataset Parquet Builder for Step 15 Unified Dating Pack (Classes 35-43)
# RULES: Enforces Rules 1-13:
#        - Rule 12: Streams 5 HF Parquet datasets (revision="refs/convert/parquet", 0 remote .py scripts),
#                   7-bit ASCII byte strip, closed vocabulary whitelist
#        - Rule 13: Zero "!" / Cartesian filler bloat, cross-step collision analyzer protecting Steps 1-14,
#                   canonical token-signature deduplication

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
# Strictly protects Steps 1-14 & Class 0 traps so ZERO lines from earlier steps leak into Step 15!
STEP1_TO_14_COLLISION_RE = re.compile(
    r"^(?:"
    r"(?:w+h+a+t+|w+h+y+|w+h+o+|w+h+y+\s+n+o+t+|w+h+a+t+\s+i+s+\s+t+h+i+s+|w+h+a+t+\s+d+o+\s+(?:y+o+u+|u+)\s+d+o+|w+h+o+\s+a+r+e+\s+(?:y+o+u+|u+))|"
    r"(?:w+e+l+c+o+m+e+|h+e+l+o+\s+t+h+e+r+e+|h+i+\s+t+h+e+r+e+|h+e+y+\s+t+h+e+r+e+|(?:g+o+o+d+|n+i+c+e+)\s+t+o+\s+(?:m+e+e+t+|s+e+e+)\s+(?:y+o+u+|u+))|"
    r"(?:(?:i+m+|i+'?m+|j+u+s+t+)?\s*(?:w+o+r+k+i+n+g+|s+t+u+d+y+i+n+g+|a+t+\s+w+o+r+k+|s+l+e+e+p+y+|i+n+\s+b+e+d+|a+l+l+\s+g+o+o+d+|n+o+t+\s+b+a+d+|o+k+|f+i+n+e+|c+a+n+'?t+\s+s+l+e+e+p+))|"
    r"(?:(?:i+\s+)?(?:r+e+a+l+y+\s+)?l+i+k+e+\s+(?:y+o+u+|u+)|u+r+\s+c+u+t+e+|u+\s+l+o+o+k+\s+g+o+o+d+)|"
    r"(?:f+r+e+n+c+h+|s+p+a+n+i+s+h+|g+e+r+m+a+n+|i+t+a+l+i+a+n+|t+u+r+k+i+s+h+|h+i+n+d+i+|e+n+g+l+i+s+h+)|"
    r"(?:(?:y+o+u+r+|u+r+)\s+s+e+x+|w+h+a+t+\s+s+e+x+\s+a+r+e+\s+(?:y+o+u+|u+))|"
    r"(?:(?:w+h+a+t+\s+)?g+a+m+e+s+\s+(?:d+o+\s+)?(?:y+o+u+|u+)\s+p+l+a+y+|p+l+a+y+i+n+g+\s+g+a+m+e+s+)|"
    r".*\b(?:h+r+u+|h+y+d+|w+y+d+|o+l+d+|a+g+e+|f+r+o+m+|c+o+u+n+t+r+y+|n+a+m+e+|s+i+n+g+e*l+e*|m+a+r+i+e*d+|t+a+k+e*n+|b+o+t+|c+a+t+f+i+s+h+)\b.*|"
    r".*\b(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|s+e+e+k+i+n+g*|p+r+i+v+a*t*e*y*|t+e+l+e*g+r+a*m+|s+n+a+p+|i+n+s+t+a+|s+e+l+f+i+e*)\b.*"
    r")[\s?]*$",
    re.IGNORECASE,
)

# 9 Disjoint Category Anchor Vocabularies (Exact match with step15_dating_pack.py LUTs)
C35_GOODBYE_WORDS = (
    "bye", "byee", "goodbye", "cya", "gtg", "gotta", "goodnight", "gn", "gnite",
    "night", "nighty", "sleep", "dreams", "leaving", "later", "ttyl", "peace",
    "brb", "sec", "min", "moment", "hold", "back", "care",
)
C36_WORK_STUDY_WORDS = (
    "student", "college", "university", "uni", "school", "study", "studies", "major",
    "degree", "class", "classes", "exam", "exams", "job", "career", "work", "office",
    "boss", "unemployed", "graduate", "graduated", "intern", "internship",
)
C37_LANGUAGE_WORDS = (
    "speak", "speaks", "speaking", "language", "languages", "lang", "arabic", "arabi",
    "french", "francais", "spanish", "espanol", "german", "deutsch", "italian",
    "russian", "turkish", "hindi", "urdu", "portuguese", "english", "parle", "hablas",
    "tatkalam", "fluent", "translate",
)
C38_AMA_GAME_WORDS = (
    "ama", "question", "questions", "truth", "dare", "secret", "secrets",
    "yourself", "urself", "rather", "game", "twenty", "never", "ever", "fact", "facts",
)
C39_FLIRT_MEET_WORDS = (
    "miss", "missed", "missing", "hug", "hugs", "kiss", "kisses", "cuddle", "cuddles",
    "marry", "date", "meet", "meetup", "hangout", "gf", "bf", "girlfriend", "boyfriend",
    "love", "cute", "type",
)
C40_THANKS_SORRY_WORDS = (
    "thanks", "thank", "thx", "ty", "tysm", "appreciate", "welcome", "np", "problem",
    "prob", "worries", "worry", "anytime", "sorry", "sry", "apologies", "apologize",
    "forgive", "oops", "excuse", "pardon", "bad",
)
C41_PING_CLARIFY_WORDS = (
    "there", "wdym", "mean", "means", "meaning", "huh", "understand", "confused",
    "explain", "repeat", "listening", "anyone", "anybody", "hello",
)
C42_NSFW_RUDE_WORDS = (
    "nude", "nudes", "naked", "sext", "sexting", "horny", "dick", "cock", "pussy",
    "boobs", "tits", "ass", "sex", "fwb", "nsfw", "onlyfans", "dirty",
    "fuck", "fucking", "fk", "fck", "bitch", "slut", "whore", "asshole",
    "stupid", "idiot", "dumb", "ugly", "loser", "boring", "shut", "stfu", "screw", "lost",
)
C43_TIMEZONE_WORDS = (
    "time", "timezone", "late", "midnight", "awake", "weather", "cold", "hot",
    "raining", "rain", "sunny", "snow", "snowing", "morning", "clock",
)

S15_FRAME_WORDS = (
    "i", "im", "i'm", "lm", "am", "u", "you", "ur", "your", "youre", "you're", "we",
    "me", "my", "it", "its", "it's", "that", "this", "is", "are", "ar", "r", "do",
    "does", "did", "didnt", "didn't", "dont", "don't", "cant", "can't", "can", "could",
    "would", "will", "ill", "i'll", "should", "have", "got", "going", "gonna", "go",
    "get", "give", "gimme", "send", "show", "ask", "tell", "talk", "talking", "chat",
    "chatting", "play", "say", "take", "wait", "come", "see", "be", "think", "like",
    "want", "wanna", "lets", "let's", "a", "an", "the", "to", "in", "on", "at", "of",
    "for", "with", "about", "up", "out", "over", "here", "now", "still", "only", "so",
    "very", "really", "too", "much", "lot", "some", "something", "anything", "one",
    "sweet", "nice", "good", "fine", "ok", "full", "not", "no", "or", "and", "what",
    "whats", "what's", "wat", "which", "where", "why", "how", "hows", "how's",
    "hey", "hi", "wbu", "hbu", "please", "plz", "body", "waste", "bed",
)

STRICT_ALLOWED_S15_VOCAB = frozenset(
    C35_GOODBYE_WORDS + C36_WORK_STUDY_WORDS + C37_LANGUAGE_WORDS + C38_AMA_GAME_WORDS
    + C39_FLIRT_MEET_WORDS + C40_THANKS_SORRY_WORDS + C41_PING_CLARIFY_WORDS
    + C42_NSFW_RUDE_WORDS + C43_TIMEZONE_WORDS + S15_FRAME_WORDS
)

# --- STRICT DISJOINT PATTERNS FOR EACH OF THE 9 CLASSES (35..43) ---

S15_CLASS_SPECS = (
    (35, "s15_goodbye_gn.txt", (
        # Goodbyes
        "bye", "byee", "goodbye", "bye bye", "cya", "see ya", "see u", "see you",
        "see u later", "see you later", "talk later", "talk to u later", "talk to you later",
        "ttyl", "gtg", "gotta go", "got to go", "i gotta go", "i have to go", "have to go",
        "im leaving", "leaving now", "take care", "later", "peace out",
        "nice talking to u", "nice talking to you", "nice chatting with u", "nice chatting with you",
        # Goodnight / Sleep
        "goodnight", "good night", "gn", "gnite", "nighty", "sweet dreams",
        "going to sleep", "im going to sleep", "going to bed", "im going to bed",
        "gonna sleep", "im gonna sleep", "time to sleep", "time for bed", "i need to sleep",
        # BRB / Pause
        "brb", "be right back", "ill be right back", "i'll be right back", "ill be back",
        "one sec", "wait a sec", "wait a min", "wait a moment", "hold on", "one min",
        "give me a sec", "gimme a sec", "give me a min", "wait please",
    )),
    (36, "s15_work_study.txt", (
        # Questions
        "do u work or study", "do you work or study", "work or study", "study or work",
        "are u a student", "are you a student", "r u a student", "u a student",
        "are u in college", "are you in college", "r u in college", "u in college",
        "are u in university", "are you in university", "u in uni", "are u in uni",
        "are u in school", "u in school", "what do u study", "what do you study",
        "what r u studying", "what are you studying", "whats ur major", "whats your major",
        "whats ur job", "whats your job", "what is your job", "do u have a job",
        "do you have a job", "do u work", "do you work", "where do u work", "where do u study",
        # Statements
        "im a student", "i'm a student", "lm a student", "i am a student", "student",
        "in college", "im in college", "i'm in college", "im a college student",
        "in university", "im in university", "in uni", "im in uni", "in school", "im in school",
        "i study", "i work", "i have a job", "i work full time", "i work at an office",
        "im unemployed", "i graduated", "i have exams", "i have class",
        "im a student u?", "im in college wbu?", "i work u?", "im in uni u?",
    )),
    (37, "s15_language.txt", (
        # Questions
        "do u speak arabic", "do you speak arabic", "u speak arabic", "speak arabic", "arabic?",
        "can u speak arabic", "can you speak arabic", "tatkalam arabi",
        "do u speak french", "do you speak french", "u speak french", "speak french",
        "can u speak french", "parle francais",
        "do u speak spanish", "do you speak spanish", "u speak spanish", "speak spanish", "hablas espanol",
        "do u speak german", "do you speak german", "speak german",
        "do u speak italian", "do u speak turkish", "do u speak russian", "do u speak hindi",
        "do u speak english", "do you speak english", "u speak english", "speak english",
        "only english?", "english only?", "u only speak english",
        "what languages do u speak", "what languages do you speak", "what language do u speak",
        "which languages do u speak", "whats ur language", "ur language?",
        # Statements
        "i speak arabic", "i speak french", "i speak spanish", "i speak german", "i speak english",
        "i speak arabic and english", "i speak french and english", "i only speak english",
        "my english is bad", "sorry my english is bad", "bad english", "my english is not good",
        "im not fluent in english", "english is not my first language",
    )),
    (38, "s15_playful_ama.txt", (
        "ask me anything", "ask me something", "ask me a question", "ask me questions", "ask me",
        "u can ask me anything", "you can ask me anything", "ama",
        "tell me about yourself", "tell me about urself", "tell me something about u",
        "tell me something about yourself", "tell me a secret", "tell me a fun fact",
        "truth or dare", "truth or dare?", "wanna play truth or dare", "lets play truth or dare",
        "play truth or dare", "lets play a game", "let's play a game", "wanna play a game",
        "want to play a game", "would you rather", "would u rather", "play would u rather",
        "question game", "lets play question game", "never have i ever",
    )),
    (39, "s15_flirt_meet.txt", (
        "do u like me", "do you like me", "u like me?", "do u think im cute", "am i cute",
        "am i ur type", "am i your type", "would u date me", "would you date me",
        "wanna date me", "date me", "be my gf", "be my bf", "be my girlfriend", "be my boyfriend",
        "marry me", "will u marry me", "wanna go on a date", "want to go on a date",
        "go on a date", "lets go on a date", "can we go on a date",
        "can we meet", "can we meet up", "wanna meet", "wanna meet up", "want to meet",
        "want to meet up", "lets meet", "let's meet", "lets meet up", "come over",
        "i miss u", "i miss you", "miss u", "miss you", "do u miss me", "u miss me",
        "send a hug", "hug me", "give me a hug", "i want a hug", "need a hug",
        "kiss me", "send a kiss", "give me a kiss", "wanna cuddle", "cuddle me",
        "i love u", "i love you", "love u", "love you", "do u love me",
    )),
    (40, "s15_thanks_sorry.txt", (
        # Thanks / Welcome / NP
        "thanks", "thank you", "thank u", "thanks a lot", "thank u so much", "thank you so much",
        "thx", "ty", "tysm", "appreciate it", "i appreciate it", "appreciate u",
        "ur welcome", "youre welcome", "you're welcome", "you are welcome",
        "no problem", "no prob", "np", "no worries", "dont worry", "don't worry", "anytime",
        "its ok", "it's ok", "its fine", "it's fine", "thats ok", "that's ok", "thats fine",
        # Sorry / Apologies
        "sorry", "im sorry", "i'm sorry", "i am sorry", "so sorry", "sry",
        "my bad", "sorry my bad", "oops", "oops sorry", "apologies", "i apologize",
        "forgive me", "didnt mean to", "didn't mean to", "i didnt mean to", "excuse me", "pardon",
    )),
    (41, "s15_ping_clarify.txt", (
        # Pings
        "u there", "u there?", "you there", "you there?", "are u there", "are you there",
        "r u there", "still there", "still there?", "u still there", "are u still there",
        "are you still there", "anyone there", "anybody there", "u here?", "are u here?",
        "r u here?", "hey u there?", "u listening", "are u listening",
        # Clarification
        "wdym", "wdym?", "what do u mean", "what do you mean", "what u mean", "wat u mean",
        "what does that mean", "huh", "huh?", "what did u say", "what did you say",
        "i didnt understand", "i didn't understand", "dont understand", "don't understand",
        "i dont understand", "im confused", "i'm confused", "confused", "say that again",
        "repeat that", "can u explain", "explain please",
    )),
    (42, "s15_nsfw_rude.txt", (
        # NSFW / Sexting Boundary
        "send nudes", "nudes", "nudes?", "got nudes", "send nude", "send me nudes",
        "wanna sext", "want to sext", "sexting", "sexting?", "lets sext", "let's sext",
        "im horny", "i'm horny", "horny", "are u horny", "u horny", "horny?",
        "send boobs", "show boobs", "show tits", "send dick", "wanna see my dick",
        "suck my dick", "lets have sex", "wanna have sex", "want sex", "sex?",
        "get naked", "are u naked", "naked", "show ur body", "show body",
        "talk dirty", "dirty talk", "wanna talk dirty", "fwb", "fwb?", "nsfw", "onlyfans",
        # Rude / Toxic Shield
        "fuck you", "fuck u", "fk u", "fck u", "go fuck yourself", "shut up", "stfu",
        "bitch", "ur a bitch", "slut", "whore", "asshole", "stupid", "ur stupid",
        "you are stupid", "idiot", "ur an idiot", "dumb", "ur dumb", "ugly", "ur ugly",
        "loser", "ur boring", "boring", "waste of time", "get lost", "screw you",
    )),
    (43, "s15_timezone.txt", (
        # Time Zone & Late Night
        "what time is it", "what time is it there", "what time is it for u",
        "whats the time", "whats the time there", "what's the time", "what timezone r u in",
        "whats ur timezone", "is it late there", "is it late", "is it night there",
        "is it morning there", "its late here", "it's late here", "its really late",
        "its so late", "its midnight here", "its midnight", "it's midnight",
        "why r u still awake", "why are u still awake", "why are you still awake",
        "why r u up so late", "why are u up so late", "still awake", "still awake?",
        "are u still awake", "u still awake", "why cant u sleep", "why can't u sleep",
        # Weather
        "whats the weather like", "what's the weather like", "hows the weather",
        "how's the weather", "how is the weather", "whats the weather there",
        "is it cold there", "is it hot there", "is it raining", "is it raining there",
        "its cold here", "its hot here", "its raining here",
    )),
)

# Fast regex patterns for mining Parquet dialogue lines across Classes 35-43
PARQUET_MINING_PATTERNS = (
    (35, re.compile(r"^(?:good\s*night|goodbye|bye\s*bye|bye|see\s+(?:you|u)\s+later|take\s+care|i\s+(?:have|gotta|got)\s+to\s+go|be\s+right\s+back|hold\s+on|wait\s+a\s+(?:sec|min|moment))$", re.I)),
    (36, re.compile(r"^(?:do\s+(?:you|u)\s+(?:work|study)(?:\s+or\s+(?:work|study))?|are\s+(?:you|u)\s+a\s+student|what\s+do\s+(?:you|u)\s+study|i(?:'?m|\s+am)\s+a\s+student|i(?:'?m|\s+am)\s+in\s+(?:college|university|school))\??$", re.I)),
    (37, re.compile(r"^(?:do\s+(?:you|u)\s+speak\s+(?:arabic|french|spanish|german|italian|turkish|hindi|english)|what\s+languages?\s+do\s+(?:you|u)\s+speak|i\s+speak\s+(?:arabic|french|spanish|german|english))\??$", re.I)),
    (38, re.compile(r"^(?:ask\s+me\s+(?:anything|a\s+question)|tell\s+me\s+about\s+(?:yourself|urself)|truth\s+or\s+dare|let'?s\s+play\s+a\s+game)\??$", re.I)),
    (39, re.compile(r"^(?:do\s+(?:you|u)\s+like\s+me|would\s+(?:you|u)\s+date\s+me|i\s+miss\s+(?:you|u)|miss\s+(?:you|u)|i\s+love\s+(?:you|u)|can\s+we\s+meet(?:\s+up)?|marry\s+me)\??$", re.I)),
    (40, re.compile(r"^(?:thanks?(?:\s+(?:you|u)(?:\s+so\s+much)?|\s+a\s+lot)?|you(?:'?re|\s+are)\s+welcome|no\s+(?:problem|worries)|don'?t\s+worry|(?:i(?:'?m|\s+am)\s+)?(?:so\s+)?sorry|my\s+bad|forgive\s+me|excuse\s+me)$", re.I)),
    (41, re.compile(r"^(?:are\s+(?:you|u)\s+(?:still\s+)?there|what\s+do\s+(?:you|u)\s+mean|i\s+don'?t\s+understand|i\s+didn'?t\s+understand|say\s+that\s+again|huh)\??$", re.I)),
    (42, re.compile(r"^(?:shut\s+up|fuck\s+(?:you|u)|you(?:'?re|\s+are)\s+(?:stupid|ugly|boring|an\s+idiot)|get\s+lost)$", re.I)),
    (43, re.compile(r"^(?:what\s+time\s+is\s+it(?:\s+there)?|how(?:'?s|\s+is)\s+the\s+weather|what(?:'?s|\s+is)\s+the\s+weather\s+like|is\s+it\s+(?:late|cold|hot|raining)(?:\s+there)?|are\s+(?:you|u)\s+still\s+awake)\??$", re.I)),
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


def _validate_clean_s15(raw_text):
    clean_ascii = _sanitize_ascii_bytes(raw_text)
    if "/" in clean_ascii:
        return ""
    s = SPACE_RE.sub(" ", clean_ascii.strip().lower()).rstrip("!.")
    if not s or len(s) > 44 or not CLEAN_LINE_RE.match(s):
        return ""
    if STEP1_TO_14_COLLISION_RE.match(s):
        return ""
    words = [w.strip("'?,") for w in s.split() if w.strip("'?,")]
    if not words or len(words) > 6:
        return ""
    if any(w not in STRICT_ALLOWED_S15_VOCAB for w in words):
        return ""
    return s


def _safe_fetch_5_parquet_datasets(class_maps):
    # Rule 12: Streams 5 HuggingFace Parquet datasets (revision="refs/convert/parquet", 0 remote .py scripts!)
    stats = {"datasets_hit": 0, "rows_scanned": 0, "lines_added": 0}
    try:
        from datasets import load_dataset
    except ImportError:
        return stats

    repos = (
        ("daily_dialog", "refs/convert/parquet", "dialog", 3000),
        ("bavard/personachat_truecased", "refs/convert/parquet", "history", 2000),
        ("AlekseyKorshuk/persona-chat", "refs/convert/parquet", "utterances", 2000),
        ("empathetic_dialogues", "refs/convert/parquet", "utterance", 2000),
        ("blended_skill_talk", "refs/convert/parquet", "previous_utterance", 2000),
    )

    for repo, rev, col, max_rows in repos:
        try:
            ds = load_dataset(repo, split="train", revision=rev, streaming=True)
            stats["datasets_hit"] += 1
            c = 0
            for row in ds:
                if c >= max_rows:
                    break
                c += 1
                stats["rows_scanned"] += 1
                val = row.get(col, ())
                turns = (val,) if isinstance(val, str) else (val if isinstance(val, (list, tuple)) else ())
                for turn in turns:
                    if not isinstance(turn, str):
                        continue
                    for sent in re.split(r"[.!]+", turn[:100]):
                        clean_s = _validate_clean_s15(sent)
                        if not clean_s:
                            continue
                        for cid, pat in PARQUET_MINING_PATTERNS:
                            if pat.match(clean_s):
                                sig = _canonical_token_sig(clean_s)
                                if sig not in class_maps[cid]:
                                    class_maps[cid][sig] = clean_s
                                    stats["lines_added"] += 1
                                break
        except Exception:
            pass
    return stats


def _write_u8_file(path, lines_iterable):
    sorted_lines = sorted(set(lines_iterable))
    u8 = np.frombuffer(("\n".join(sorted_lines) + "\n").encode("ascii"), dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(u8.tobytes())
    return len(sorted_lines), u8.nbytes


def build_step15_pack_files(neg_path="negatives.txt"):
    class_maps = {cid: {} for cid, _, _ in S15_CLASS_SPECS}

    for cid, _, bases in S15_CLASS_SPECS:
        for base in bases:
            ends = ("",) if "?" in base else (("", "?") if cid in (36, 37, 38, 39, 41, 43) else ("",))
            for end in ends:
                c = _validate_clean_s15(f"{base}{end}")
                if c:
                    sig = _canonical_token_sig(c)
                    if sig not in class_maps[cid]:
                        class_maps[cid][sig] = c

    hf_stats = _safe_fetch_5_parquet_datasets(class_maps)

    # Enforce 100% disjoint signatures across all 9 classes (priority: NSFW 42 > earlier classes)
    seen_sigs = set()
    for cid in (42, 35, 36, 37, 38, 39, 40, 41, 43):
        cmap = class_maps[cid]
        for sig in list(cmap.keys()):
            if sig in seen_sigs:
                del cmap[sig]
            else:
                seen_sigs.add(sig)

    tot_lines, tot_bytes = 0, 0
    all_s15_lines = []
    for cid, fn, _ in S15_CLASS_SPECS:
        cnt, nb = _write_u8_file(fn, class_maps[cid].values())
        tot_lines += cnt
        tot_bytes += nb
        all_s15_lines.extend(class_maps[cid].values())

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn_in:
            existing_neg = {line.strip().lower() for line in fn_in if line.strip()}
    for item in all_s15_lines:
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ", "yo ")):
            existing_neg.add(item)
    _, _ = _write_u8_file(neg_path, existing_neg)

    print(
        f"[5-DATASET PARQUET HARVESTER] Hit {hf_stats['datasets_hit']} HF Parquet datasets | "
        f"Scanned {hf_stats['rows_scanned']} rows | Mined {hf_stats['lines_added']} clean Step 15 lines!",
        flush=True,
    )
    print(
        f"[CLEAN STEP 15 PACK RULES 12+13] Saved {tot_lines} unique lines across 9 class files (Classes 35-43, {tot_bytes / 1024.0:.2f} KB total)!",
        flush=True,
    )


if __name__ == "__main__":
    build_step15_pack_files()