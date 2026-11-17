#!/usr/bin/env python3
# FILE: build_age_country_txt.py
# PURPOSE: Build 100% pure ages.txt (with lm 24, 24y, 21yres, 20yres) & countries.txt (zero hru/gender contamination)
# RULES: Enforces zero float64/int64, strict isolation against Step 1, Step 3, and Step 4

import logging
import os
import re
import warnings
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CLEAN_ASCII_RE = re.compile(r"^[a-z0-9\s'!?.,]+$")
SPACE_RE = re.compile(r"\s+")

# Blocks any status, "how are you", or gender phrases from polluting ages.txt or countries.txt
CONTAMINATION_RE = re.compile(
    r"\b(?:hru|hyd|doing|doin|going|goin|wbu|hbu|hor|"
    r"how\s+(?:are|r|is|was|have)\s+(?:you|u|ya|it|your|ur)|"
    r"how\s+(?:u|ya)\s+doin|hows\s+(?:it|u|ur|your)|how's\s+(?:it|u|ur|your)|"
    r"m|f|male|female|famle|famel|femael|femle|boy|boi|girl|gril|gurl|man|woman|womman|guy|lady|"
    r"gender|gand|sex)\b",
    re.IGNORECASE,
)

# Curated Country Catalog
COUNTRY_CATALOG = (
    ("tunisia", ("tun", "tn", "tunis", "tounsi", "tounsia", "tunsia", "tuinsi", "tunisie", "sfax", "sousse"), ("tunisia", "tunisi", "tunisian")),
    ("india", ("ind", "hindi", "handi", "desi", "bharat", "mumbai", "delhi"), ("india", "indian", "hindustan")),
    ("usa", ("usa", "murica", "states", "nyc", "cali", "texas", "florida"), ("america", "american", "california")),
    ("uk", ("uk", "brit", "london"), ("britain", "british", "england", "english", "scotland")),
    ("algeria", ("alg", "dz", "dziri", "algiers", "oran"), ("algeria", "algerie", "algerian")),
    ("morocco", ("ma", "maroc", "casa", "rabat"), ("morocco", "moroccan", "maghreb", "casablanca")),
    ("egypt", ("egy", "eg", "masr", "misr", "cairo"), ("egypt", "egyptian", "alexandria")),
    ("france", ("fr", "paris", "lyon"), ("france", "french")),
    ("germany", ("de", "ger", "berlin"), ("germany", "german")),
    ("italy", ("ita", "rome", "milan"), ("italy", "italia", "italian")),
    ("spain", ("esp", "madrid"), ("spain", "espana", "spanish")),
    ("canada", ("ca", "can", "toronto"), ("canada", "canadian")),
    ("saudi", ("ksa", "riyadh"), ("saudi", "arabia")),
    ("uae", ("uae", "dubai", "abudhabi"), ("emirates", "emirati")),
    ("turkey", ("tr", "turk", "istanbul"), ("turkey", "turkiye", "turkish")),
    ("pakistan", ("pak", "paki", "lahore", "karachi"), ("pakistan", "pakistani")),
    ("indonesia", ("indo", "bali", "jakarta"), ("indonesia", "indonesian")),
    ("philippines", ("ph", "pinoy", "pinay", "manila"), ("philippines", "filipino", "filipina")),
    ("brazil", ("br", "rio"), ("brazil", "brasil", "brazilian")),
)

ENGLISH_COLLISION_BLACKLIST = frozenset((
    "in", "us", "it", "me", "so", "to", "hi", "no", "do", "am", "be", "or", "by", "my",
    "hru", "hyd", "hor", "how", "are", "doing", "doin", "going", "goin", "what", "why", "who", "lol",
    "and", "end", "mind", "find", "kind", "wind", "hand", "hind", "band", "land", "sand",
    "pain", "spin", "span", "stain", "train", "chain", "chains", "main", "rain", "gain",
    "man", "men", "woman", "women", "human", "roman", "male", "mail", "mall", "her",
    "here", "hear", "germs", "dance", "lance", "trance", "turkeys", "idea", "ideal",
    "nice", "split", "bath", "reading", "mobile", "normal", "lima", "victoria",
    "grace", "hope", "young", "best", "most", "ware", "deal", "sale", "tours", "angers",
))

STRICT_AGE_CHAT_RE = re.compile(
    r"^(?:"
    r"age\??|ur\s+age\??|your\s+age\??|what(?:'s|\s+is)\s+(?:your|ur)\s+age\??|"
    r"how\s+old\s+(?:are\s+you|r\s+u|are\s+u|u)\??|how\s+old\??|"
    r"(?:(?:hi|hey|hello)\s+)?(?:i\s+am|im|i'm|l'm|lm|iam)\s*(?:1[89]|[2-7][0-9]|80)(?:\s*(?:y|yo|yr|yrs|yres|yers|years|years\s+old))?|"
    r"(?:1[89]|[2-7][0-9]|80)(?:\s*(?:y|yo|yr|yrs|yres|yers|years|years\s+old))?"
    r")[\s!?.]*$",
    re.IGNORECASE,
)

STRICT_FROM_QUESTION_RE = re.compile(
    r"^(?:"
    r"from\??|u\s+from\??|where\s+from\??|country\??|ur\s+country\??|your\s+country\??|"
    r"where\s+(?:are\s+you|r\s+u|are\s+u|u)\s+from\??|"
    r"where\s+do\s+(?:you|u)\s+live\??|what\s+country\s+(?:are\s+you|r\s+u)\s+from\??"
    r")[\s!?.]*$",
    re.IGNORECASE,
)

AGE_QUESTIONS = (
    "age?", "ur age?", "your age?", "what is your age", "whats ur age",
    "how old are you", "how old r u", "how old are u", "how old u", "how old", "u how old?",
)

COUNTRY_QUESTIONS = (
    "from?", "u from?", "where from?", "where are you from", "where r u from",
    "where are u from", "where u from", "country?", "ur country?", "which country", "where do u live",
)


def generate_1edit_typos(word):
    out = {word}
    if len(word) < 5:
        return out
    for i in range(1, len(word) - 1):
        swapped = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        if swapped not in ENGLISH_COLLISION_BLACKLIST and not CONTAMINATION_RE.search(swapped):
            out.add(swapped)
    if len(word) >= 6:
        for i in range(1, len(word) - 1):
            dropped = word[:i] + word[i + 1 :]
            if len(dropped) >= 5 and dropped not in ENGLISH_COLLISION_BLACKLIST and not CONTAMINATION_RE.search(dropped):
                out.add(dropped)
    return out


def pull_huggingface_countries_and_cities(country_aliases):
    try:
        from datasets import load_dataset
    except ImportError:
        return

    try:
        ds_countries = load_dataset("bamwor/world-countries", split="train")
        for name_raw in ds_countries["name"]:
            w = SPACE_RE.sub(" ", str(name_raw).strip().lower())
            if w and len(w) <= 18 and w.isalpha() and w not in ENGLISH_COLLISION_BLACKLIST:
                country_aliases.update(generate_1edit_typos(w))
    except Exception:
        pass

    try:
        ds_cities = load_dataset("bamwor/world-cities", split="train")
        for name_raw, feat_code, pop_val in zip(
            ds_cities["name"], ds_cities["feature_code"], ds_cities["population"]
        ):
            is_capital = str(feat_code) == "PPLC"
            is_megacity = int(pop_val or 0) >= 1500000
            if is_capital or is_megacity:
                c_name = str(name_raw).strip().lower()
                if 4 <= len(c_name) <= 15 and c_name.isalpha() and c_name not in ENGLISH_COLLISION_BLACKLIST:
                    country_aliases.add(c_name)
    except Exception:
        pass


def pull_huggingface_age_and_from_chats(age_lines, country_lines, country_aliases):
    try:
        from datasets import load_dataset
    except ImportError:
        return

    inline_loc_re = re.compile(
        r"^(?:(?:hi|hey|hello)\s+)?(?:i\s+am|im|i'm|l'm|lm)?\s*(?:from|in|live\s+in)\s+([a-z]{2,15})[\s!?.]*$",
        re.IGNORECASE,
    )

    for repo, split, col in (
        ("bavard/personachat_truecased", "train", "history"),
        ("AlekseyKorshuk/persona-chat", "train", "utterances"),
        ("tanaos/synthetic-intent-classifier-dataset-v1", "train", "text"),
    ):
        try:
            ds = load_dataset(repo, split=split)
            if col not in ds.column_names:
                continue
            for item in ds[col]:
                candidates = item if isinstance(item, list) else (str(item),)
                for raw_s in candidates:
                    if not isinstance(raw_s, str):
                        continue
                    s = SPACE_RE.sub(" ", raw_s.strip().lower())
                    if not s or len(s) > 30 or not CLEAN_ASCII_RE.match(s) or CONTAMINATION_RE.search(s):
                        continue
                    if STRICT_AGE_CHAT_RE.match(s):
                        age_lines.add(s.rstrip("!.?"))
                    elif STRICT_FROM_QUESTION_RE.match(s):
                        country_lines.add(s)
                    else:
                        m = inline_loc_re.match(s)
                        if m and m.group(1).lower() in country_aliases:
                            country_lines.add(s.rstrip("!.?"))
        except Exception:
            pass


def build_age_and_country_files(
    ages_path="ages.txt",
    countries_path="countries.txt",
    staging_path="age_country_staging.txt",
    neg_path="negatives.txt",
):
    age_lines = set(AGE_QUESTIONS)
    for age in range(18, 81):
        age_lines.add(f"{age}")
        age_lines.add(f"im {age}")
        age_lines.add(f"i'm {age}")
        age_lines.add(f"l'm {age}")
        age_lines.add(f"lm {age}")
        age_lines.add(f"iam {age}")
        age_lines.add(f"i am {age}")
        age_lines.add(f"im{age}")
        age_lines.add(f"lm{age}")
        age_lines.add(f"{age}y")
        age_lines.add(f"{age} y")
        age_lines.add(f"{age}yo")
        age_lines.add(f"{age} yo")
        age_lines.add(f"{age}yr")
        age_lines.add(f"{age} yr")
        age_lines.add(f"{age}yrs")
        age_lines.add(f"{age} yrs")
        age_lines.add(f"{age}yres")
        age_lines.add(f"{age} yres")
        age_lines.add(f"{age}yers")
        age_lines.add(f"{age} yers")
        age_lines.add(f"{age}years")
        age_lines.add(f"{age} years")

    country_aliases = set()
    for canon, short_codes, long_roots in COUNTRY_CATALOG:
        country_aliases.add(canon)
        for sc in short_codes:
            if sc not in ENGLISH_COLLISION_BLACKLIST:
                country_aliases.add(sc)
        for lr in long_roots:
            country_aliases.update(generate_1edit_typos(lr))

    pull_huggingface_countries_and_cities(country_aliases)

    country_lines = set(COUNTRY_QUESTIONS)
    pull_huggingface_age_and_from_chats(age_lines, country_lines, country_aliases)

    for alias in sorted(country_aliases):
        if CONTAMINATION_RE.search(alias):
            continue
        country_lines.add(alias)
        country_lines.add(f"from {alias}"[:30])
        country_lines.add(f"im from {alias}"[:30])
        country_lines.add(f"i'm from {alias}"[:30])
        country_lines.add(f"lm from {alias}"[:30])
        country_lines.add(f"in {alias}"[:30])
        country_lines.add(f"hi from {alias}"[:30])
        country_lines.add(f"hey from {alias}"[:30])
        country_lines.add(f"hi im from {alias}"[:30])
        country_lines.add(f"hi l'm 30 from {alias}"[:30])
        country_lines.add(f"hi lm 24 from {alias}"[:30])

    sorted_ages = sorted(s for s in age_lines if not CONTAMINATION_RE.search(s))
    ages_bytes = ("\n".join(sorted_ages) + "\n").encode("ascii", errors="ignore")
    ages_u8 = np.frombuffer(ages_bytes, dtype=np.uint8)
    with open(ages_path, "wb") as fa:
        fa.write(ages_u8.tobytes())

    sorted_countries = sorted(s for s in country_lines if not CONTAMINATION_RE.search(s))
    country_bytes = ("\n".join(sorted_countries) + "\n").encode("ascii", errors="ignore")
    country_u8 = np.frombuffer(country_bytes, dtype=np.uint8)
    with open(countries_path, "wb") as fc:
        fc.write(country_u8.tobytes())

    all_staged = sorted(set(sorted_ages) | set(sorted_countries))
    staged_bytes = ("\n".join(all_staged) + "\n").encode("ascii", errors="ignore")
    staged_u8 = np.frombuffer(staged_bytes, dtype=np.uint8)
    with open(staging_path, "wb") as fs:
        fs.write(staged_u8.tobytes())

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn:
            for line in fn:
                s = line.strip().lower()
                if s:
                    existing_neg.add(s)

    for item in all_staged:
        if len(item) <= 30:
            existing_neg.add(item)

    neg_bytes = ("\n".join(sorted(existing_neg)) + "\n").encode("ascii", errors="ignore")
    neg_u8 = np.frombuffer(neg_bytes, dtype=np.uint8)
    with open(neg_path, "wb") as fn:
        fn.write(neg_u8.tobytes())

    n_ages, n_countries = np.int32(len(sorted_ages)), np.int32(len(sorted_countries))
    print(f"[CLEAN AGES] Overwrote {int(n_ages)} pure age lines to '{ages_path}' ({ages_u8.nbytes / 1024.0:.2f} KB)", flush=True)
    print(f"[CLEAN COUNTRIES] Overwrote {int(n_countries)} pure country lines to '{countries_path}' ({country_u8.nbytes / 1024.0:.2f} KB)", flush=True)


if __name__ == "__main__":
    build_age_and_country_files()
