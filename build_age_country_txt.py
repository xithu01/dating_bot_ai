#!/usr/bin/env python3
# FILE: build_age_country_txt.py
# PURPOSE: Build 100% pure ages.txt & countries.txt (Classes 2-6) with zero cross-step contamination
# RULES: Enforces Rules 1-13:
#        - Rule 12: Parquet-only fetch (revision="refs/convert/parquet"), 7-bit ASCII byte strip, closed vocab whitelist
#        - Rule 13: Zero "!" clone bloat, cross-step collision analyzer (Steps 1, 3-8), canonical deduplication

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

CLEAN_ASCII_RE = re.compile(r"^[a-z0-9\s'?]+$")
SPACE_RE = re.compile(r"\s+")

# Rule 13 Cross-Step Collision Guard: Blocks Steps 3-8 tokens from polluting ages.txt or countries.txt
CONTAMINATION_RE = re.compile(
    r"\b(?:hru|hyd|doing|doin|going|goin|wbu|hbu|hor|wyd|wud|whatcha|"
    r"how\s+(?:are|r|is|was|have)\s+(?:you|u|ya|it|your|ur)|"
    r"how\s+(?:u|ya)\s+doin|hows\s+(?:it|u|ur|your)|how's\s+(?:it|u|ur|your)|"
    r"m|f|male|female|famle|famel|femael|femle|boy|boi|girl|gril|gurl|man|woman|womman|guy|lady|"
    r"gender|gand|sex|bored|borad|tired|chilling|chatting|nmu|cute|sweet|same|sam|true|"
    r"looking|lookin|loking|finding|seeking|friends|gf|bf|girlfriend|boyfriend|dating|relationship)\b",
    re.IGNORECASE,
)

# Curated Country Catalog (Matches COUNTRY_GROUPS in offline_train.py)
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
    "same", "sam", "true", "cool", "cute", "good", "fine", "ok", "okay", "bored",
))

# Rule 12 Gate 2: Closed Framing Whitelist for Step 2 Age & Country
STRICT_ALLOWED_S2_FRAMING_VOCAB = frozenset((
    "hi", "hey", "hello", "yo", "i", "am", "im", "i'm", "l'm", "lm", "iam", "i'am",
    "age", "ur", "your", "what", "whats", "what's", "is", "how", "old", "are", "r",
    "you", "u", "y", "yo", "yr", "yrs", "yres", "yers", "years",
    "from", "where", "country", "which", "do", "live", "in",
))

STRICT_AGE_CHAT_RE = re.compile(
    r"^(?:"
    r"age\??|ur\s+age\??|your\s+age\??|what(?:'s|\s+is)\s+(?:your|ur)\s+age\??|"
    r"how\s+old\s+(?:are\s+you|r\s+u|are\s+u|u)\??|how\s+old\??|"
    r"(?:(?:hi|hey|hello)\s+)?(?:i\s+am|im|i'm|l'm|lm|iam)\s*(?:1[89]|[2-7][0-9]|80)(?:\s*(?:y|yo|yr|yrs|yres|yers|years|years\s+old))?|"
    r"(?:1[89]|[2-7][0-9]|80)(?:\s*(?:y|yo|yr|yrs|yres|yers|years|years\s+old))?"
    r")$",
    re.IGNORECASE,
)

STRICT_FROM_QUESTION_RE = re.compile(
    r"^(?:"
    r"from\??|u\s+from\??|where\s+from\??|country\??|ur\s+country\??|your\s+country\??|"
    r"where\s+(?:are\s+you|r\s+u|are\s+u|u)\s+from\??|"
    r"where\s+do\s+(?:you|u)\s+live\??|what\s+country\s+(?:are\s+you|r\s+u)\s+from\??"
    r")$",
    re.IGNORECASE,
)

INLINE_LOC_RE = re.compile(
    r"^(?:(?:hi|hey|hello)\s+)?(?:i\s+am|im|i'm|l'm|lm)?\s*(?:from|in|live\s+in)\s+([a-z]{2,15})$",
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


def _sanitize_ascii_bytes(raw_text):
    if not isinstance(raw_text, str):
        return ""
    b = np.frombuffer(raw_text[:64].encode("ascii", errors="ignore"), dtype=np.uint8)
    valid = b[(b >= np.uint8(32)) & (b <= np.uint8(122))]
    if valid.shape[0] == 0:
        return ""
    return valid.tobytes().decode("ascii", errors="ignore")


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


def categorize_line_across_all_steps(raw_text, country_aliases):
    """
    Rule 12 & 13 Multi-Class Analyzer for Step 2:
    Returns:
      - (0, "") if invalid or contaminated by Steps 3-8
      - (2, s) if valid Age question/statement
      - (4, s) if valid Country question/statement
    """
    clean_ascii = _sanitize_ascii_bytes(raw_text)
    s = SPACE_RE.sub(" ", clean_ascii.strip().lower()).rstrip("!.,")
    if not s or len(s) > 30 or not CLEAN_ASCII_RE.match(s) or CONTAMINATION_RE.search(s):
        return 0, ""

    if STRICT_AGE_CHAT_RE.match(s):
        return 2, s.rstrip("?") if not any(q in s for q in ("age", "old")) else s
    if STRICT_FROM_QUESTION_RE.match(s):
        return 4, s

    m = INLINE_LOC_RE.match(s)
    if m and m.group(1).lower() in country_aliases:
        words = [w.strip("'") for w in s.split() if w.strip("'")]
        if all((w in STRICT_ALLOWED_S2_FRAMING_VOCAB or w in country_aliases) for w in words):
            return 4, s
    return 0, ""


def pull_parquet_countries_and_chats(age_lines, country_lines, country_aliases):
    # Rule 12: Streams ONLY pure Parquet branches (revision="refs/convert/parquet", zero remote .py scripts!)
    try:
        from datasets import load_dataset
    except ImportError:
        return

    try:
        ds_countries = load_dataset(
            "bamwor/world-countries",
            split="train",
            revision="refs/convert/parquet",
            streaming=True,
        )
        for row in ds_countries:
            w = SPACE_RE.sub(" ", _sanitize_ascii_bytes(str(row.get("name", ""))).strip().lower())
            if w and 3 <= len(w) <= 18 and w.isalpha() and w not in ENGLISH_COLLISION_BLACKLIST and not CONTAMINATION_RE.search(w):
                country_aliases.update(generate_1edit_typos(w))
    except Exception:
        pass

    try:
        ds_cities = load_dataset(
            "bamwor/world-cities",
            split="train",
            revision="refs/convert/parquet",
            streaming=True,
        )
        for row in ds_cities:
            is_capital = str(row.get("feature_code", "")) == "PPLC"
            is_megacity = int(row.get("population") or 0) >= 1500000
            if is_capital or is_megacity:
                c_name = _sanitize_ascii_bytes(str(row.get("name", ""))).strip().lower()
                if 4 <= len(c_name) <= 15 and c_name.isalpha() and c_name not in ENGLISH_COLLISION_BLACKLIST and not CONTAMINATION_RE.search(c_name):
                    country_aliases.add(c_name)
    except Exception:
        pass

    try:
        ds_dialog = load_dataset(
            "daily_dialog",
            split="train",
            revision="refs/convert/parquet",
            streaming=True,
        )
        count = 0
        for row in ds_dialog:
            if count >= 2000:
                break
            count += 1
            for turn in row.get("dialog", ()):
                if not isinstance(turn, str):
                    continue
                for sent in re.split(r"[.!]+", turn[:90]):
                    cid, clean_s = categorize_line_across_all_steps(sent, country_aliases)
                    if cid == 2:
                        age_lines.add(clean_s)
                    elif cid == 4:
                        country_lines.add(clean_s)
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
        for tmpl in (
            f"{age}", f"im {age}", f"i'm {age}", f"l'm {age}", f"lm {age}",
            f"iam {age}", f"i am {age}", f"im{age}", f"lm{age}",
            f"{age}y", f"{age} y", f"{age}yo", f"{age} yo",
            f"{age}yr", f"{age} yr", f"{age}yrs", f"{age} yrs",
            f"{age}yres", f"{age} yres", f"{age}yers", f"{age} yers",
            f"{age}years", f"{age} years",
        ):
            age_lines.add(tmpl)

    country_aliases = set()
    for canon, short_codes, long_roots in COUNTRY_CATALOG:
        country_aliases.add(canon)
        for sc in short_codes:
            if sc not in ENGLISH_COLLISION_BLACKLIST:
                country_aliases.add(sc)
        for lr in long_roots:
            country_aliases.update(generate_1edit_typos(lr))

    country_lines = set(COUNTRY_QUESTIONS)
    pull_parquet_countries_and_chats(age_lines, country_lines, country_aliases)

    for alias in sorted(country_aliases):
        if CONTAMINATION_RE.search(alias):
            continue
        for tmpl in (
            alias,
            f"from {alias}"[:30],
            f"im from {alias}"[:30],
            f"i'm from {alias}"[:30],
            f"lm from {alias}"[:30],
            f"in {alias}"[:30],
            f"hi from {alias}"[:30],
            f"hey from {alias}"[:30],
            f"hi im from {alias}"[:30],
            f"hi l'm 30 from {alias}"[:30],
            f"hi lm 24 from {alias}"[:30],
        ):
            country_lines.add(tmpl)

    sorted_ages = sorted(s for s in age_lines if not CONTAMINATION_RE.search(s))
    ages_u8 = np.frombuffer(("\n".join(sorted_ages) + "\n").encode("ascii", errors="ignore"), dtype=np.uint8)
    with open(ages_path, "wb") as fa:
        fa.write(ages_u8.tobytes())

    sorted_countries = sorted(s for s in country_lines if not CONTAMINATION_RE.search(s))
    country_u8 = np.frombuffer(("\n".join(sorted_countries) + "\n").encode("ascii", errors="ignore"), dtype=np.uint8)
    with open(countries_path, "wb") as fc:
        fc.write(country_u8.tobytes())

    all_staged = sorted(set(sorted_ages) | set(sorted_countries))
    staged_u8 = np.frombuffer(("\n".join(all_staged) + "\n").encode("ascii", errors="ignore"), dtype=np.uint8)
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
        if len(item) <= 30 and not item.startswith(("hi ", "hey ", "hello ")):
            existing_neg.add(item)

    neg_u8 = np.frombuffer(("\n".join(sorted(existing_neg)) + "\n").encode("ascii", errors="ignore"), dtype=np.uint8)
    with open(neg_path, "wb") as fn:
        fn.write(neg_u8.tobytes())

    n_ages, n_countries = np.int32(len(sorted_ages)), np.int32(len(sorted_countries))
    print(f"[CLEAN STEP 2 AGES RULES 12+13] Saved {int(n_ages)} pure age lines to '{ages_path}' ({ages_u8.nbytes / 1024.0:.2f} KB)", flush=True)
    print(f"[CLEAN STEP 2 COUNTRIES RULES 12+13] Saved {int(n_countries)} pure country lines to '{countries_path}' ({country_u8.nbytes / 1024.0:.2f} KB)", flush=True)


if __name__ == "__main__":
    build_age_and_country_files()