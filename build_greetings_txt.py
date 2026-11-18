#!/usr/bin/env python3
# FILE: build_greetings_txt.py
# PURPOSE: Build 100% pure greetings.txt (Class 1) & negatives.txt (Class 0)
# RULES: Enforces Rules 1-13:
#        - Rule 12: Parquet-only fetch (revision="refs/convert/parquet"), 7-bit ASCII byte strip, closed vocab whitelist
#        - Rule 13: Zero "!" / "?" Cartesian punctuation clones, cross-step collision analyzer, canonical deduplication

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

CLEAN_CHARS_RE = re.compile(r"^[a-z\s']+$")
SPACE_RE = re.compile(r"\s+")

# Rule 13 Cross-Step Collision Guard (Blocks Steps 2-8 & Class 0 traps from ever entering greetings.txt)
STEP2_TO_8_COLLISION_RE = re.compile(
    r"^(?:.*(?:"
    r"\b(?:h+o+w+|h+r+u+|h+y+d+|h+o+r+|w+b+u+|h+b+u+|d+o+i+n+g*|g+o+i+n+g*|w+y+d+|w+u+d+)\b|"
    r"\b(?:o+l+d+|a+g+e+|f+r+o+m+|c+o+u+n+t+r+y+|w+h+e+r+e+|l+i+v+e+)\b|"
    r"\b(?:m+a+l+e+|f+e+m+a+l+e+|f+a+m+l+e+|b+o+y+|g+i+r+l+|g+r+i+l+|w+o+m+a+n+|g+e+n+d+e+r+|s+e+x+)\b|"
    r"\b(?:b+o+r+e+d+|b+o+r+a+d+|t+i+r+e+d+|c+h+i+l+i+n+g*|c+h+a+t+i+n+g*|n+m+u+|n+o+t+h+i+n+g+)\b|"
    r"\b(?:c+u+t+e+|s+e+e+m+|l+i+k+e+|v+i+b+e+|s+a+m+e+|s+a+m+|t+r+u+e+|l+o+l+|h+a+h+a+)\b|"
    r"\b(?:l+o+k+i+n+g*|f+i+n+d+i+n+g*|s+e+e+k+i+n+g*|f+r+i+e+n+d+s+|g+f+|b+f+|d+a+t+i+n+g+|r+e+l+a+t+i+o+n+s+h+i+p+)\b"
    r").*)$",
    re.IGNORECASE,
)

# Pure English greetings only (Zero "!" / "?" clone bloat)
STRICT_GREETING_RE = re.compile(
    r"^(?:"
    r"h+i+|h+e+y+|h+e+l+o+|h+l+o+|h+y+|h+i+e+|h+e+i+|h+i+y+a+|h+e+y+a+|h+u+l+o+|e+l+o+|"
    r"y+o+|s+u+p+|w+a+s+u+p+|w+a+z+u+p+|w+h+a+t+'?s\s+u+p+|w+h+a+t\s+i+s\s+u+p+|h+o+w+d+y+|"
    r"g+r+e+e+t+i+n+g+s*|s+a+l+u+t+a+t+i+o+n+s*|w+e+l+c+o+m+e+|a+l+o+h+a+|g\s*d+a+y+|"
    r"(?:g+o+o+d\s+)?(?:m+o+r+n+i+n+g*|m+o+r+n+i+n+|a+f+t+e+r+n+o+o+n+|e+v+e+n+i+n+g*|d+a+y+)|"
    r"(?:n+i+c+e|p+l+e+a+s+e+d|g+l+a+d|g+o+o+d)\s+t+o\s+(?:m+e+e+t|s+e+e)\s+(?:y+o+u|u)|"
    r"l+o+n+g\s+t+i+m+e\s+n+o\s+s+e+e+"
    r")"
    r"(?:\s+(?:"
    r"t+h+e+r+e+|y+o+u|u|f+r+i+e+n+d+|s+t+r+a+n+g+e+r+|c+u+t+i+e+|g+o+r+g+e+o+u+s+|h+a+n+d+s+o+m+e+|"
    r"b+r+o+|m+a+n+|b+u+d+d+y+|a+l+l+|e+v+e+r+y+o+n+e+|"
    r"w+h+a+t+'?s\s+u+p+|w+a+s+u+p+|n+i+c+e\s+t+o\s+m+e+e+t\s+(?:y+o+u|u)|"
    r"g+o+o+d\s+m+o+r+n+i+n+g+|g+o+o+d\s+e+v+e+n+i+n+g+"
    r"))?$",
    re.IGNORECASE,
)

# Rule 12 Gate 2: Closed Whitelist Vocabulary for Step 1 Greetings
STRICT_ALLOWED_S1_VOCAB = frozenset((
    "hi", "hii", "hiii", "hiiii", "hiiiii", "hey", "heyy", "heyyy", "heyyyy", "heey", "heeyy",
    "hello", "helloo", "hellooo", "helo", "hlo", "ello", "hullo", "hulo", "hy", "hyy", "hyyy",
    "hie", "hei", "hiya", "hiyaa", "hiyaaa", "heya", "heyaa", "heyaaa", "howdy",
    "yo", "yoo", "yooo", "sup", "supp", "suuup", "wassup", "wassupp", "wasup", "wazzup", "wazup",
    "whats", "what's", "what", "is", "up", "good", "morning", "mornin", "afternoon", "afternon",
    "evening", "day", "gday", "greetings", "gretings", "greeting", "salutations", "welcome", "aloha",
    "nice", "pleased", "glad", "to", "meet", "met", "see", "se", "long", "time", "no",
    "there", "you", "u", "friend", "stranger", "cutie", "gorgeous", "handsome",
    "bro", "man", "mate", "buddy", "budy", "all", "al", "everyone", "partner",
))

COUNTRY_ALIASES_AND_TYPOS = (
    "tunisia", "tunis", "tuinsi", "tuinsia", "tunsia", "tunisa", "tunisi", "tunisie",
    "tunisian", "tounsi", "tounsia", "tun", "tn", "sfax", "sousse",
    "india", "ind", "indian", "hindi", "handi", "inida", "indai", "hindustan",
    "bharat", "desi", "mumbai", "delhi",
    "algeria", "alg", "dz", "dziri", "algerie", "algerian", "algiers",
    "morocco", "maroc", "ma", "morocan", "moroccan", "maghreb", "casablanca",
    "egypt", "egy", "eg", "egyptian", "masr", "misr", "cairo",
    "saudi", "ksa", "uae", "dubai", "qatar", "kuwait", "jordan", "lebanon",
    "usa", "america", "american", "states", "nyc", "cali", "california", "texas",
    "uk", "britain", "british", "england", "english", "london",
    "france", "fr", "french", "frnace", "paris",
    "germany", "de", "german", "geramny", "berlin",
    "italy", "italia", "italian", "rome", "spain", "espana", "spanish", "madrid",
    "canada", "ca", "canadian", "turkey", "turk", "turkiye", "istanbul",
    "brazil", "brasil", "pak", "pakistan", "paki", "bd", "bangla", "bangladesh",
    "indo", "indonesia", "ph", "philippines", "pinoy", "pinay", "filipino",
)

HELLO_SALUTATIONS = (
    "hi", "hii", "hiii", "hey", "heyy", "heyyy", "heeyy", "hello", "helloo", "hellooo",
    "yo", "yoo", "yooo", "hiya", "heya", "howdy", "ello", "hlo", "hy", "hyy", "hie", "hei", "hullo",
)

TIME_AND_FORMAL = (
    "good morning", "morning", "mornin", "good afternoon", "afternoon",
    "good evening", "evening", "good day", "greetings", "salutations", "welcome", "aloha",
)

SLANG_GREETINGS = (
    "sup", "supp", "wassup", "wassupp", "wazzup", "whats up", "what's up",
    "what is up", "nice to meet you", "nice to meet u",
    "pleased to meet you", "glad to meet you", "good to see you", "long time no see",
)

TARGETS = ("", "there", "u", "you", "friend", "stranger", "cutie", "gorgeous", "handsome", "bro", "buddy")

FOLLOW_GREETINGS = (
    "", "whats up", "what's up", "wassup", "nice to meet u",
    "nice to meet you", "good morning", "good evening",
)

# Step 1 Hard Negatives (Zero "!" clone bloat)
CORE_HARD_NEGATIVES = COUNTRY_ALIASES_AND_TYPOS + (
    "how are you", "how are you?", "hru", "hru?", "hyd", "hyd?",
    "how r u", "how r u?", "how are u", "how u doing", "how are you doing",
    "how's it going", "hows it going", "how is it going", "how was your day", "how is your day",
    "how have you been", "hru doing", "hows u", "wbu", "hor", "how ar you dopin", "how u doin",
    "hw r u", "how whas you da", "hi how r u", "hey hru", "hello how are you",
    "hi how are you", "hey how r u", "yo hru", "doing", "going",
    "what", "what?", "what is this", "what is this?", "what is that",
    "what do you want", "what time is it", "what do you do", "what is your name", "ur name?",
    "why", "why?", "why not", "why not?", "why me",
    "who", "who?", "who are you", "who are you?", "who is this",
    "who r u", "who is that", "lol", "lol?", "haha", "lmao", "rofl",
    "m", "f", "w", "m?", "f?", "w?", "m or f", "m or f?", "f or m", "f or m?", "m/f", "f/m",
    "m f", "f m", "m f?", "f m?", "male", "female", "famle", "famel", "femael", "femle",
    "boy", "boi", "girl", "gril", "gurl", "man", "woman", "womman", "girl?", "boy?", "male?",
    "female?", "woman?", "man?", "are you a girl", "are you a boy", "are u a girl", "are u a boy",
    "r u a girl", "r u a boy", "you a girl?", "you a boy?", "girl or boy", "boy or girl",
    "male or female", "female or male", "what gender are you", "whats your gender", "gender",
    "gender?", "your sex", "f here", "m here", "im a girl", "im a boy", "im female", "im male",
    "a bf here", "a gf here", "a boyfriend here", "a girlfriend here",
    "22f", "22 f", "24m", "24 m", "19 famle", "21 girl", "23 boy",
    "hi m", "hi f", "hey m", "hey f", "hello m", "hello f", "yo m", "yo f", "sup m", "sup f",
    "m hi", "f hi", "m hey", "f hey", "hi m here", "hi f here",
    "wyd", "wyd?", "wud", "what are you doing", "what r u doing", "what u up to",
    "im good", "im ok", "bored", "lm borad", "nothing much", "nmu", "im good u?",
    "ur cute", "u seem nice", "i like u", "same", "sam", "same here", "sam here", "true", "cool", "nice",
    "what r u looking for", "why r u on here", "m looking for girl , u", "lm finding gril", "looking for friends",
    "hyw", "hyk", "hyb", "hypx", "hya", "ih", "ie", "iphey", "xhey", "hix", "hellod", "hellox",
    "hi from tunis", "hi from tuinsi", "hi from handi", "hi from ind", "hey from tuinsi",
    "hi im from tun", "hi im from tuinsi", "hi im from handi", "hi im from ind",
    "hi how old are you", "hey where are you from", "from?", "so", "u", "u d?", "u f?",
    "you in bed?", "your name?", "charlotte", "from", "age?", "m bi",
    "how old are you", "where are you from", "asl", "are you single", "are you a boy or girl",
    "help", "helping", "helper", "helmet", "helicopter", "hell", "shell", "high",
    "higher", "highway", "hill", "history", "historical", "him", "his", "hit",
    "hide", "hidden", "he", "her", "hers", "here", "hero", "hear", "heart",
    "heat", "heavy", "health", "hello kitty", "xhello", "good", "good job",
    "good luck", "good night", "gn", "morning coffee", "morning news", "you",
    "your", "yours", "young", "youtube", "yo mama", "supper", "super", "support",
    "supply", "how old", "how much", "how far", "how tall", "how to",
    "bye", "goodbye", "see you", "see ya", "take care", "later", "cya", "gtg",
    "thank you", "thanks", "thx", "ty", "ok", "okay", "yes", "no", "maybe",
    "bonjour", "salut", "coucou", "bonsoir", "salam", "marhba", "ahla", "kifech",
    "hola", "buenos dias", "ciao", "hallo", "namaste", "konnichiwa", "privet", "merhaba", "oi", "ola",
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
    if not s or len(s) > 30 or not CLEAN_CHARS_RE.match(s):
        return 0, ""

    if STEP2_TO_8_COLLISION_RE.match(s):
        return 2, s

    words = [w.strip("'") for w in s.split() if w.strip("'")]
    if not words or len(words) > 4:
        return 0, ""

    for w in words:
        if w not in STRICT_ALLOWED_S1_VOCAB:
            return 0, ""

    if STRICT_GREETING_RE.match(s):
        return 1, s
    return 0, ""


def _safe_fetch_parquet_greetings(pos_map):
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
            if count >= 2000:
                break
            count += 1
            turns = row.get("dialog", ())
            if not isinstance(turns, (list, tuple)):
                continue
            for turn in turns[:2]:
                if not isinstance(turn, str):
                    continue
                for sent in re.split(r"[.!?]+", turn[:80]):
                    cid, clean_s = categorize_line_across_all_steps(sent)
                    if cid == 1:
                        sig = _canonical_token_sig(clean_s)
                        if sig not in pos_map:
                            pos_map[sig] = clean_s
    except Exception:
        pass


def generate_country_typos_and_slang(neg_set):
    long_countries = (
        "tunisia", "tunisi", "algeria", "morocco", "egyptian", "indian", "france",
        "germany", "canada", "turkey", "brazil", "pakistan", "indonesia", "philippines",
    )
    for w in long_countries:
        neg_set.add(w)
        for i in range(1, len(w) - 1):
            swapped = w[:i] + w[i + 1] + w[i] + w[i + 2 :]
            if len(swapped) >= 5 and swapped not in ("train", "chains"):
                neg_set.add(swapped)

    genders = ("m", "f", "male", "female", "famle", "boy", "girl")
    for c in COUNTRY_ALIASES_AND_TYPOS:
        neg_set.add(c)
        neg_set.add(f"from {c}")
        neg_set.add(f"im from {c}")
        neg_set.add(f"u from {c}?")
        for g in genders:
            neg_set.add(f"{g} from {c}"[:30])
            neg_set.add(f"im {g} from {c}"[:30])


def generate_greetings_and_negatives(pos_path="greetings.txt", neg_path="negatives.txt"):
    pos_map = {}
    pos_explicit = set()
    neg_set = {SPACE_RE.sub(" ", s.strip().lower()) for s in CORE_HARD_NEGATIVES if s.strip()}

    generate_country_typos_and_slang(neg_set)

    # Keep single-word letter-stretch greetings explicitly ("hi", "hii", "hiii", "hey", "heyy")
    for sal in HELLO_SALUTATIONS:
        cid, c = categorize_line_across_all_steps(sal)
        if cid == 1:
            pos_explicit.add(c)
            pos_map[_canonical_token_sig(c)] = c
        for tgt in TARGETS:
            if tgt:
                cid, c = categorize_line_across_all_steps(f"{sal} {tgt}")
                if cid == 1:
                    sig = _canonical_token_sig(c)
                    if sig not in pos_map:
                        pos_map[sig] = c
        for fg in FOLLOW_GREETINGS:
            if fg:
                cid, c = categorize_line_across_all_steps(f"{sal} {fg}")
                if cid == 1:
                    sig = _canonical_token_sig(c)
                    if sig not in pos_map:
                        pos_map[sig] = c

    for tf in TIME_AND_FORMAL:
        for tgt in TARGETS:
            base = f"{tf} {tgt}".strip()
            cid, c = categorize_line_across_all_steps(base)
            if cid == 1:
                sig = _canonical_token_sig(c)
                if sig not in pos_map:
                    pos_map[sig] = c

    for sg in SLANG_GREETINGS:
        for tgt in ("", "there", "bro", "stranger", "friend"):
            base = f"{sg} {tgt}".strip()
            cid, c = categorize_line_across_all_steps(base)
            if cid == 1:
                sig = _canonical_token_sig(c)
                if sig not in pos_map:
                    pos_map[sig] = c

    for extra_g in ("hey man", "hi man", "yo man"):
        pos_explicit.add(extra_g)

    _safe_fetch_parquet_greetings(pos_map)

    final_pos_set = set(pos_map.values()) | pos_explicit
    neg_set.difference_update(final_pos_set)

    sorted_pos = sorted(final_pos_set)
    pos_u8 = np.frombuffer(("\n".join(sorted_pos) + "\n").encode("ascii", errors="ignore"), dtype=np.uint8)
    with open(pos_path, "wb") as fp:
        fp.write(pos_u8.tobytes())

    sorted_neg = sorted(neg_set)
    neg_u8 = np.frombuffer(("\n".join(sorted_neg) + "\n").encode("ascii", errors="ignore"), dtype=np.uint8)
    with open(neg_path, "wb") as fn:
        fn.write(neg_u8.tobytes())

    n_pos, n_neg = np.int32(len(sorted_pos)), np.int32(len(sorted_neg))
    print(f"[CLEAN STEP 1 RULES 12+13] Saved {int(n_pos)} unique greetings to '{pos_path}' ({pos_u8.nbytes / 1024.0:.2f} KB)", flush=True)
    print(f"[CLEAN STEP 1 NEGATIVES] Saved {int(n_neg)} hard negatives to '{neg_path}' ({neg_u8.nbytes / 1024.0:.2f} KB)", flush=True)


def generate_greetings_file(output_path="greetings.txt"):
    generate_greetings_and_negatives(pos_path=output_path, neg_path="negatives.txt")


def generate_datasets(pos_path="greetings.txt", neg_path="negatives.txt"):
    generate_greetings_and_negatives(pos_path=pos_path, neg_path=neg_path)


if __name__ == "__main__":
    generate_greetings_and_negatives()