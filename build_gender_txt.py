#!/usr/bin/env python3
# FILE: build_gender_txt.py
# PURPOSE: Build Step 4 Gender & Age+Gender datasets (Classes 9-14) + underage_traps.txt (Class 7)
# RULES: Enforces Rules 1-13:
#        - Rule 12: Parquet-only fetch (revision="refs/convert/parquet"), 7-bit ASCII byte strip, closed vocab whitelist
#        - Rule 13: Zero "!" clone bloat, cross-step collision analyzer (blocks Step 1 "hey man" & Step 8 "looking for girl")

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

CLEAN_GENDER_RE = re.compile(r"^[a-z0-9\s'?/]+$")
SPACE_RE = re.compile(r"\s+")

# Rule 13 Cross-Step Collision Guard: Blocks Step 1 ("hey man"), Step 2, Step 3, Step 5, Step 6, Step 7, Step 8
STEP_OTHER_COLLISION_RE = re.compile(
    r"^(?:"
    r"(?:he+y+|hi+|y+o+|he+l+o+)\s+m+a+n+|"
    r".*\b(?:hru|hyd|hor|wbu|hbu|doing|doin|going|goin|wyd|wud|whatcha)\b.*|"
    r".*\b(?:old|olds|from|country|live|living|where|why|who)\b.*|"
    r".*\b(?:bored|borad|tired|chilling|chatting|nmu|cute|sweet|same|sam|true|lol|haha)\b.*|"
    r".*\b(?:looking|lookin|loking|finding|seeking|searching|want|need|friends|gf|bf|girlfriend|boyfriend|dating|relationship)\b.*"
    r")$",
    re.IGNORECASE,
)

# Rule 12 Gate 2: Closed Whitelist Vocabulary for Step 4 Gender (Classes 9-13)
FEMALE_NOUNS = ("f", "female", "famle", "famel", "femael", "femle", "girl", "gril", "gurl", "woman", "womman", "w", "lady")
MALE_NOUNS = ("m", "male", "maale", "mael", "boy", "boi", "man", "masculine", "guy", "dude")
NEUTRAL_GENDER_WORDS = ("gender", "gander", "gnder", "sex")
QUESTION_FRAME_TOKENS = frozenset((
    "are", "r", "you", "u", "ur", "your", "you're", "youre", "is", "what", "whats", "what's", "which", "tell",
))
STATEMENT_FRAME_TOKENS = (
    "im", "i'm", "l'm", "lm", "iam", "i'am", "i", "am", "me", "here", "a", "an", "or",
    "hi", "hey", "helo", "hlo", "elo", "hello", "yo", "sup", "hiya", "heya", "howdy", "hy",
)

STRICT_ALLOWED_S4_VOCAB = frozenset(
    FEMALE_NOUNS + MALE_NOUNS + NEUTRAL_GENDER_WORDS
    + tuple(QUESTION_FRAME_TOKENS) + STATEMENT_FRAME_TOKENS + ("feminine",)
)

# 1. Class 9 (S4 head class 1): Open / Choice Gender Questions ("m or f?", "m/f", "what gender are you")
RAW_GENDER_CHOICE = (
    "m or f", "m or f?", "f or m", "f or m?", "m/f", "f/m", "m/f?", "f/m?",
    "m f", "f m", "m f?", "f m?", "girl or boy", "girl or boy?", "boy or girl", "boy or girl?",
    "girl or male", "girl or male?", "boy or female", "boy or female?",
    "male or female", "male or female?", "female or male", "female or male?",
    "male or girl", "male or girl?", "female or boy", "female or boy?",
    "m or female", "m or female?", "f or male", "f or male?",
    "male or f", "male or f?", "female or m", "female or m?",
    "m or girl", "m or girl?", "f or boy", "f or boy?",
    "boy or m", "boy or m?", "girl or f", "girl or f?",
    "woman or man", "woman or man?", "man or woman", "man or woman?",
    "girl or woman", "boy or man", "female or girl", "male or boy",
    "woman or girl", "man or boy", "are you a boy or girl", "are you a girl or boy",
    "are u a boy or girl", "are u a girl or boy", "r u a boy or girl", "r u a girl or boy",
    "are you male or female", "are you female or male",
    "are you a male or female", "are you a female or male",
    "are u male or female", "are u female or male", "r u male or female", "r u female or male",
    "r u m or f", "r u f or m", "are u m or f", "are u f or m", "you m or f", "u m or f",
    "what gender are you", "what gender r u", "what is your gender", "whats your gender",
    "what's your gender", "whats ur gender", "what's ur gender", "your gender", "ur gender",
    "u gender", "gender", "gender?", "gander", "gander?", "gnder", "gnder?",
    "your sex", "ur sex", "what sex are you", "which gender are you", "which sex are you",
    "what are you male or female", "tell me your gender", "tell me ur gender",
)

# 2. Class 10 (S4 head class 2): Binary Yes/No Questions Asking if Bot is Female ("f?", "are you a girl")
RAW_ASK_IS_FEMALE = (
    "f?", "girl?", "female?", "woman?", "w?", "famle?", "famel?", "femael?", "femle?",
    "gril?", "gurl?", "womman?", "lady?", "u f?", "you f?", "ur f?",
    "are you a girl", "are you a girl?", "are you girl", "are you girl?",
    "are u a girl", "are u a girl?", "are u girl", "are u girl?",
    "you a girl?", "you a girl", "r u a girl", "r u a girl?", "r u girl", "r u girl?",
    "u a girl", "u a girl?", "ur a girl", "ur a girl?", "your a girl", "your a girl?",
    "are you female", "are you female?", "are u female", "are u female?",
    "r u female", "r u female?", "you female", "you female?", "u female", "u female?",
    "are you a female", "are you a female?", "are u a female", "are u a female?",
    "r u a female", "r u a female?", "r you a girl", "r you a girl?",
    "r you girl", "r you girl?", "r you female", "r you female?",
    "you're a girl", "you're a girl?", "you are a girl", "you are a girl?",
    "you're female", "you're female?", "you are female", "you are female?",
    "is your gender female", "is your gender female?", "is ur gender female", "is ur gender female?",
    "are you feminine", "are you feminine?", "are u feminine", "r u feminine",
    "are you a woman", "are you a woman?", "are u a woman", "are u a woman?",
    "r u a woman", "r u a woman?", "you a woman", "you a woman?", "u a woman", "u a woman?",
    "are u a gril", "are u a gril?", "r u a gril", "r u a gril?",
    "are u famle", "are u famle?", "r u famle", "r u famle?", "u famle?",
    "ur a gurl", "ur a gurl?", "r u a gurl?",
)

# 3. Class 11 (S4 head class 3): Binary Yes/No Questions Asking if Bot is Male ("m?", "are you a boy")
RAW_ASK_IS_MALE = (
    "m?", "boy?", "male?", "man?", "maale?", "mael?", "boi?", "guy?", "dude?",
    "u m?", "you m?", "ur m?",
    "are you a boy", "are you a boy?", "are you boy", "are you boy?",
    "are u a boy", "are u a boy?", "are u boy", "are u boy?",
    "r u a boy", "r u a boy?", "r u boy", "r u boy?",
    "you a boy", "you a boy?", "u a boy", "u a boy?",
    "ur a boy", "ur a boy?", "your a boy", "your a boy?",
    "are you male", "are you male?", "are u male", "are u male?",
    "r u male", "r u male?", "you male", "you male?", "u male", "u male?",
    "are you a male", "are you a male?", "are u a male", "are u a male?",
    "r u a male", "r u a male?", "r you a boy", "r you a boy?",
    "r you boy", "r you boy?", "r you male", "r you male?",
    "you're a boy", "you're a boy?", "you are a boy", "you are a boy?",
    "you're male", "you're male?", "you are male", "you are male?",
    "is your gender male", "is your gender male?", "is ur gender male", "is ur gender male?",
    "are you masculine", "are you masculine?", "are u masculine", "r u masculine",
    "are you a man", "are you a man?", "are u a man", "are u a man?",
    "r u a man", "r u a man?", "you a man", "you a man?", "u a man", "u a man?",
    "are u a boi", "are u a boi?", "r u a boi", "r u a boi?",
    "are u maale", "are u maale?", "r u maale", "r u maale?", "r u mael?",
)

# Rule 13: Zero "!" clone suffix bloat
STATEMENT_PREFIXES = (
    "", "im ", "i'm ", "i am ", "lm ", "l'm ", "iam ",
    "im a ", "i'm a ", "i am a ", "lm a ",
    "hi ", "hey ", "hello ", "yo ", "hi im ", "hey im ",
)
STATEMENT_SUFFIXES = ("", " here")


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
    has_s = "/" in s
    clean_w = "".join(ch if ("a" <= ch <= "z" or "0" <= ch <= "9" or ch == " " or ch == "'") else " " for ch in s.lower())
    words = tuple(w for w in _collapse_str(clean_w).split() if w)
    return (words, has_q, has_s)


def _clean_line(raw_text, allow_digits=False):
    clean_ascii = _sanitize_ascii_bytes(raw_text)
    s = SPACE_RE.sub(" ", clean_ascii.strip().lower()).rstrip("!.,")
    if not s or len(s) > 42 or not CLEAN_GENDER_RE.match(s):
        return ""
    if STEP_OTHER_COLLISION_RE.match(s):
        return ""
    toks = [w.strip("'?/") for w in s.replace("/", " ").split() if w.strip("'?/")]
    if not toks or len(toks) > 6:
        return ""
    if not allow_digits:
        if any(w not in STRICT_ALLOWED_S4_VOCAB for w in toks):
            return ""
    return s


def categorize_line_across_all_steps(raw_text):
    """
    Rule 12 & 13 Multi-Class Analyzer for Step 4 Parquet Stream:
    Returns (9..13, clean_s) or (0, "").
    """
    s = _clean_line(raw_text, allow_digits=False)
    if not s:
        return 0, ""
    toks = [w.strip("'?/") for w in s.replace("/", " ").split() if w.strip("'?/")]
    has_f = any(w in FEMALE_NOUNS for w in toks)
    has_m = any(w in MALE_NOUNS for w in toks)
    has_n = any(w in NEUTRAL_GENDER_WORDS for w in toks)
    if not (has_f or has_m or has_n):
        return 0, ""
    has_q_tok = "?" in s or any(w in QUESTION_FRAME_TOKENS for w in toks)
    has_s_tok = any(w in ("im", "i'm", "l'm", "lm", "iam", "i", "am", "here") for w in toks)

    if (has_f and has_m) or "/" in s or "or" in toks or (has_n and not has_f and not has_m):
        return 9, s
    if has_f and not has_s_tok and has_q_tok:
        return 10, s
    if has_m and not has_s_tok and has_q_tok:
        return 11, s
    if has_f and has_s_tok and "?" not in s:
        return 12, s
    if has_m and has_s_tok and "?" not in s:
        return 13, s
    return 0, ""


def _safe_fetch_parquet_gender_lines(choice_map, ask_f_map, ask_m_map, state_f_map, state_m_map):
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "daily_dialog",
            split="train",
            revision="refs/convert/parquet",
            streaming=True,
        )
        count = 0
        target_maps = {9: choice_map, 10: ask_f_map, 11: ask_m_map, 12: state_f_map, 13: state_m_map}
        for row in ds:
            if count >= 2000:
                break
            count += 1
            for turn in row.get("dialog", ()):
                if not isinstance(turn, str):
                    continue
                for sent in re.split(r"[.!]+", turn[:90]):
                    cid, clean_s = categorize_line_across_all_steps(sent)
                    if cid in target_maps:
                        sig = _canonical_token_sig(clean_s)
                        if sig not in target_maps[cid]:
                            target_maps[cid][sig] = clean_s
    except Exception:
        pass


def _write_u8_file(path, lines_iterable):
    sorted_lines = sorted(set(lines_iterable))
    payload = ("\n".join(sorted_lines) + "\n").encode("ascii", errors="ignore")
    arr_u8 = np.frombuffer(payload, dtype=np.uint8)
    with open(path, "wb") as f:
        f.write(arr_u8.tobytes())
    return np.int32(len(sorted_lines)), np.int32(arr_u8.nbytes)


def generate_all_gender_files(
    choice_path="gender_ask_choice.txt",
    ask_f_path="gender_ask_is_female.txt",
    ask_m_path="gender_ask_is_male.txt",
    state_f_path="gender_state_female.txt",
    state_m_path="gender_state_male.txt",
    age_gender_path="age_gender.txt",
    underage_path="underage_traps.txt",
    neg_path="negatives.txt",
):
    choice_map, ask_f_map, ask_m_map = {}, {}, {}
    state_f_map, state_m_map = {}, {}
    age_gender_set, underage_set = set(), set()

    for s in RAW_GENDER_CHOICE:
        for sfx in ("", "?"):
            c = _clean_line(f"{s.rstrip('?!.')}{sfx}")
            if c:
                sig = _canonical_token_sig(c)
                if sig not in choice_map:
                    choice_map[sig] = c

    for s in RAW_ASK_IS_FEMALE:
        c = _clean_line(s)
        if c:
            ask_f_map[_canonical_token_sig(c)] = c
            if "?" not in c:
                c_q = _clean_line(f"{c}?")
                if c_q:
                    ask_f_map[_canonical_token_sig(c_q)] = c_q

    for s in RAW_ASK_IS_MALE:
        c = _clean_line(s)
        if c:
            ask_m_map[_canonical_token_sig(c)] = c
            if "?" not in c:
                c_q = _clean_line(f"{c}?")
                if c_q:
                    ask_m_map[_canonical_token_sig(c_q)] = c_q

    for fn in FEMALE_NOUNS:
        for pfx in STATEMENT_PREFIXES:
            if pfx.endswith("a ") and fn in ("f", "w", "female", "famle", "famel", "femael", "femle"):
                continue
            for sfx in STATEMENT_SUFFIXES:
                c = _clean_line(f"{pfx}{fn}{sfx}")
                if c and "?" not in c and "/" not in c:
                    toks = [w.strip("'") for w in c.split()]
                    if not any(w in QUESTION_FRAME_TOKENS for w in toks):
                        sig = _canonical_token_sig(c)
                        if sig not in state_f_map or fn in ("f", "female", "famle", "girl"):
                            state_f_map[sig] = c

    for mn in ("m", "male", "maale", "mael", "boy", "boi", "man", "guy", "dude"):
        for pfx in STATEMENT_PREFIXES:
            if pfx.endswith("a ") and mn in ("m", "male", "maale", "mael"):
                continue
            for sfx in STATEMENT_SUFFIXES:
                c = _clean_line(f"{pfx}{mn}{sfx}")
                if c and "?" not in c and "/" not in c:
                    toks = [w.strip("'") for w in c.split()]
                    if not any(w in QUESTION_FRAME_TOKENS for w in toks):
                        sig = _canonical_token_sig(c)
                        if sig not in state_m_map or mn in ("m", "male", "boy"):
                            state_m_map[sig] = c

    _safe_fetch_parquet_gender_lines(choice_map, ask_f_map, ask_m_map, state_f_map, state_m_map)

    for k in list(state_f_map.keys()):
        if k in ask_f_map or k in choice_map:
            del state_f_map[k]
    for k in list(state_m_map.keys()):
        if k in ask_m_map or k in choice_map:
            del state_m_map[k]

    ag_genders = ("f", "m", "female", "famle", "male", "girl", "gril", "boy", "boi", "woman", "man")
    for age in range(18, 81):
        for g in ag_genders:
            for cand in (
                f"{age}{g}", f"{age} {g}", f"{g}{age}", f"{g} {age}",
                f"im {age} {g}", f"i'm {age} {g}", f"lm {age} {g}", f"i am {age} {g}",
                f"im {age}{g}", f"hi im {age} {g}", f"hey {age} {g}", f"{age} yo {g}",
            ):
                c = _clean_line(cand, allow_digits=True)
                if c:
                    age_gender_set.add(c)

    for u_age in range(10, 18):
        underage_set.add(f"{u_age}")
        underage_set.add(f"im {u_age}")
        underage_set.add(f"i am {u_age}")
        for g in ("f", "m", "female", "famle", "male", "girl", "boy"):
            underage_set.add(f"{u_age}{g}")
            underage_set.add(f"{u_age} {g}")
            underage_set.add(f"{g} {u_age}")
            underage_set.add(f"im {u_age} {g}")

    n_c, b_c = _write_u8_file(choice_path, choice_map.values())
    n_af, b_af = _write_u8_file(ask_f_path, ask_f_map.values())
    n_am, b_am = _write_u8_file(ask_m_path, ask_m_map.values())
    n_sf, b_sf = _write_u8_file(state_f_path, state_f_map.values())
    n_sm, b_sm = _write_u8_file(state_m_path, state_m_map.values())
    n_ag, b_ag = _write_u8_file(age_gender_path, age_gender_set)
    n_un, b_un = _write_u8_file(underage_path, underage_set)

    existing_neg = set()
    if os.path.exists(neg_path):
        with open(neg_path, "r", encoding="ascii", errors="ignore") as fn:
            for line in fn:
                s = line.strip().lower()
                if s:
                    existing_neg.add(s)
    all_g = set(choice_map.values()) | set(ask_f_map.values()) | set(ask_m_map.values()) | set(state_f_map.values()) | set(state_m_map.values())
    for g_item in all_g:
        if len(g_item) <= 30 and not g_item.startswith(("hi ", "hey ", "hello ", "yo ")):
            existing_neg.add(g_item)
    _, _ = _write_u8_file(neg_path, existing_neg)

    tot_lines = np.int32(n_c + n_af + n_am + n_sf + n_sm + n_ag + n_un)
    tot_kb = np.float32(b_c + b_af + b_am + b_sf + b_sm + b_ag + b_un) / np.float32(1024.0)
    print(
        f"[CLEAN STEP 4 GENDER RULES 12+13] Saved {int(tot_lines)} total lines ({float(tot_kb):.2f} KB): "
        f"choice={int(n_c)}, ask_f={int(n_af)}, ask_m={int(n_am)}, "
        f"state_f={int(n_sf)}, state_m={int(n_sm)}, age_gender={int(n_ag)}, underage={int(n_un)}",
        flush=True,
    )
    return tot_lines, tot_kb


if __name__ == "__main__":
    generate_all_gender_files()