#!/usr/bin/env python3
# FILE: build_greetings_txt.py
# PURPOSE: Build pure greetings.txt (Class 1, zero hru/status/gender) & negatives.txt (Class 0, includes hru, gender & what/why/who traps)
# RULES: Enforces zero float64/int64, strict isolation against Step 2, Step 3, Step 4, and Class 0

import logging
import os
import re
import warnings
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CLEAN_CHARS_RE = re.compile(r"^[a-z\s'!?.,]+$")
SPACE_RE = re.compile(r"\s+")

# Pure English greetings only (all "how are you", "hru", "hyd", "doing", "going", and gender tokens removed)
STRICT_GREETING_RE = re.compile(
    r"^(?:"
    r"h+i+|h+e+y+|h+e+l+o+|h+l+o+|h+y+|h+i+e+|h+e+i+|h+i+y+a+|h+e+y+a+|h+u+l+o+|e+l+o+|"
    r"y+o+|s+u+p+|w+a+s+u+p+|w+a+z+u+p+|w+h+a+t+'?s\s+u+p+|w+h+a+t\s+i+s\s+u+p+|h+o+w+d+y+|"
    r"g+r+e+e+t+i+n+g+s*|s+a+l+u+t+a+t+i+o+n+s*|w+e+l+c+o+m+e+|a+l+o+h+a+|g\s*d+a+y+|"
    r"(?:g+o+o+d\s+)?(?:m+o+r+n+i+n+g*|m+o+r+n+i+n+|a+f+t+e+r+n+o+o+n+|e+v+e+n+i+n+g*|d+a+y+)|"
    r"(?:n+i+c+e|p+l+e+a+s+e+d|g+l+a+d|g+o+o+d)\s+t+o\s+(?:m+e+e+t|s+e+e)\s+(?:y+o+u|u)|"
    r"l+o+n+g\s+t+i+m+e\s+n+o\s+s+e+e+"
    r")"
    r"(?:[\s,!?.]+(?:"
    r"t+h+e+r+e+|y+o+u|u|f+r+i+e+n+d+|s+t+r+a+n+g+e+r+|c+u+t+i+e+|g+o+r+g+e+o+u+s+|h+a+n+d+s+o+m+e+|"
    r"b+r+o+|b+u+d+d+y+|a+l+l+|e+v+e+r+y+o+n+e+|"
    r"w+h+a+t+'?s\s+u+p+|w+a+s+u+p+|n+i+c+e\s+t+o\s+m+e+e+t\s+(?:y+o+u|u)|"
    r"g+o+o+d\s+m+o+r+n+i+n+g+|g+o+o+d\s+e+v+e+n+i+n+g+"
    r"))?[\s!?.]*$",
    re.IGNORECASE,
)

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

# Blocks status/hru words AND all Step 4 gender words/typos from greetings.txt
BLACKLIST_WORDS = frozenset(
    COUNTRY_ALIASES_AND_TYPOS
    + (
        "how", "hows", "how's", "are", "r", "doing", "doin", "going", "goin", "hru", "hyd",
        "was", "been", "wbu", "hbu", "hor", "why", "who", "this", "not", "lol", "haha",
        "siri", "alexa", "cortana", "google", "olly", "computer", "bot", "robot", "assistant",
        "help", "order", "refund", "price", "cost", "weather", "music", "timer", "alarm",
        "account", "ticket", "support", "problem", "issue", "cancel", "stop", "bye", "night",
        "from", "age", "old", "olds", "im", "i'm", "l'm", "lm", "iam",
        "m", "f", "w", "male", "maale", "mael", "female", "famle", "famel", "femael", "femle",
        "boy", "boi", "girl", "gril", "gurl", "man", "woman", "womman", "guy", "lady", "dude",
        "gender", "gander", "gnder", "sex", "feminine", "masculine",
    )
)

HELLO_SALUTATIONS = (
    "hi", "hii", "hiii", "hey", "heyy", "heyyy", "heeyy", "hello", "helloo", "hellooo",
    "yo", "yoo", "yooo", "hiya", "hiyaa", "heya", "heyaa", "howdy", "ello",
    "hlo", "hy", "hyy", "hie", "hei", "hullo",
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

ENDINGS = ("", "!", "?")

# Step 1 Hard Negatives: Includes all HRU variations, Step 4 Gender variations, AND "what"/"why"/"who" traps
CORE_HARD_NEGATIVES = COUNTRY_ALIASES_AND_TYPOS + (
    # Status / HRU variations (must be rejected by Step 1 so they route to Step 3 Class 8)
    "how are you", "how are you?", "how are you!", "hru", "hru?", "hru!", "hyd", "hyd?", "hyd!",
    "how r u", "how r u?", "how are u", "how u doing", "how u doing!", "how are you doing",
    "how's it going", "hows it going", "how is it going", "how was your day", "how is your day",
    "how have you been", "hru doing", "hows u", "wbu", "hor", "how ar you dopin", "how u doin",
    "hw r u", "how whas you da", "hi how r u", "hey hru", "hello how are you",
    "hello how are you doing!", "hi how are you", "hey how r u", "yo hru", "doing", "going",
    # Question & reaction traps (must strictly be Class 0, never Class 1, 8, or 9-14)
    "what", "what?", "what!", "what?!", "what!!", "what is this", "what is this?", "what is that",
    "what do you want", "what time is it", "what do you do", "what is your name", "ur name?",
    "why", "why?", "why!", "why?!", "why!!", "why not", "why not?", "why not!", "why me",
    "who", "who?", "who!", "who?!", "who!!", "who are you", "who are you?", "who is this",
    "who r u", "who is that", "lol", "lol!", "lol?", "haha", "haha!", "lmao", "rofl",
    # Step 4 Gender traps (must be rejected by Step 1 so they route to Step 4 Classes 9-14)
    "m", "f", "w", "m?", "f?", "w?", "m or f", "m or f?", "f or m", "f or m?", "m/f", "f/m",
    "m f", "f m", "m f?", "f m?", "male", "female", "famle", "famel", "femael", "femle",
    "boy", "boi", "girl", "gril", "gurl", "man", "woman", "womman", "girl?", "boy?", "male?",
    "female?", "woman?", "man?", "are you a girl", "are you a boy", "are u a girl", "are u a boy",
    "r u a girl", "r u a boy", "you a girl?", "you a boy?", "girl or boy", "boy or girl",
    "male or female", "female or male", "what gender are you", "whats your gender", "gender",
    "gender?", "your sex", "f here", "m here", "im a girl", "im a boy", "im female", "im male",
    "22f", "22 f", "24m", "24 m", "19 famle", "21 girl", "23 boy",
    "hi m", "hi f", "hey m", "hey f", "hello m", "hello f", "yo m", "yo f", "sup m", "sup f",
    "m hi", "f hi", "m hey", "f hey", "hi m here", "hi f here",
    # Gibberish & boundary traps
    "hyw", "hyk", "hyb", "hypx", "hya", "ih", "ie", "iphey", "xhey", "hix", "hellod", "hellox",
    "hi from tunis", "hi from tuinsi", "hi from handi", "hi from ind", "hey from tuinsi",
    "hi im from tun", "hi im from tuinsi", "hi im from handi", "hi im from ind",
    "hi how old are you", "hey where are you from", "from?", "so", "u", "u d?", "u f?", "same",
    "you in bed?", "your name?", "charlotte", "from", "age?", "m bi",
    "do you get my text!", "how old are you", "where are you from", "asl",
    "are you single", "are you a boy or girl",
    "help", "helping", "helper", "helmet", "helicopter", "hell", "shell", "high",
    "higher", "highway", "hill", "history", "historical", "him", "his", "hit",
    "hide", "hidden", "he", "her", "hers", "here", "hero", "hear", "heart",
    "heat", "heavy", "health", "hello kitty", "xhello", "good", "good job",
    "good luck", "good night", "gn", "morning coffee", "morning news", "you",
    "your", "yours", "young", "youtube", "yo mama", "supper", "super", "support",
    "supply", "how old", "how much", "how far", "how tall", "how to",
    "bye", "goodbye", "see you", "see ya", "take care", "later", "cya", "gtg",
    "thank you", "thanks", "thx", "ty", "ok", "okay", "yes", "no", "maybe", "bored",
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
    pos_set = set()
    neg_set = { SPACE_RE.sub(" ", s.strip().lower()) for s in CORE_HARD_NEGATIVES if s.strip() }

    generate_country_typos_and_slang(neg_set)

    for sal in HELLO_SALUTATIONS:
        for tgt in TARGETS:
            base = f"{sal} {tgt}".strip()
            for end in ENDINGS:
                c = is_valid_clean_greeting(f"{base}{end}")
                if c:
                    pos_set.add(c)
        for fg in FOLLOW_GREETINGS:
            base = f"{sal} {fg}".strip()
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

    for sg in SLANG_GREETINGS:
        for tgt in ("", "there", "bro", "stranger", "friend"):
            base = f"{sg} {tgt}".strip()
            for end in ENDINGS:
                c = is_valid_clean_greeting(f"{base}{end}")
                if c:
                    pos_set.add(c)

    neg_set.difference_update(pos_set)

    sorted_pos = sorted(pos_set)
    pos_bytes = ("\n".join(sorted_pos) + "\n").encode("ascii", errors="ignore")
    pos_u8 = np.frombuffer(pos_bytes, dtype=np.uint8)
    with open(pos_path, "wb") as fp:
        fp.write(pos_u8.tobytes())

    sorted_neg = sorted(neg_set)
    neg_bytes = ("\n".join(sorted_neg) + "\n").encode("ascii", errors="ignore")
    neg_u8 = np.frombuffer(neg_bytes, dtype=np.uint8)
    with open(neg_path, "wb") as fn:
        fn.write(neg_u8.tobytes())

    n_pos, n_neg = np.int32(len(sorted_pos)), np.int32(len(sorted_neg))
    print(f"[CLEAN BUILD] Overwrote {int(n_pos)} pure greetings to '{pos_path}' ({pos_u8.nbytes / 1024.0:.2f} KB, 0 hru/gender lines)", flush=True)
    print(f"[CLEAN BUILD] Overwrote {int(n_neg)} hard negatives (with hru, gender & what/why/who traps) to '{neg_path}' ({neg_u8.nbytes / 1024.0:.2f} KB)", flush=True)


def generate_greetings_file(output_path="greetings.txt"):
    generate_greetings_and_negatives(pos_path=output_path, neg_path="negatives.txt")


def generate_datasets(pos_path="greetings.txt", neg_path="negatives.txt"):
    generate_greetings_and_negatives(pos_path=pos_path, neg_path=neg_path)


if __name__ == "__main__":
    generate_greetings_and_negatives()
