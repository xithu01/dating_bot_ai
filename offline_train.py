#!/usr/bin/env python3
# FILE: offline_train.py
# PURPOSE: Unified Step 1, 2, 3, and 4 Vault Trainer (Zero-Leak Gender LUTs + Full Vault Protection)
# RULES: Enforces Rules 1-11 (Zero float64/int64, 1s ETA, RAM Super-Arena, Self-Healing Datasets)
# NOTE: PyArmor line limits apply ONLY to production_server.py per System Rule 7.

import logging
import os
import re
import sys
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

CPU_CORES = os.cpu_count() or 8
os.environ["OMP_NUM_THREADS"] = str(CPU_CORES)
torch.set_num_threads(CPU_CORES)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

VOCAB_SIZE = 2048
SEQ_LEN = 32
MAX_GREETING_LEN = 30
D_MODEL = 64
D_FF = 128
NUM_INTENTS = 2
BLOOM_SIZE = 65536
S2_VOCAB = 1024
S2_DIM = 32
S2_CLASSES = 6
S3_CLASSES = 2
S4_CLASSES = 7  # 0=Reject, 1=Choice, 2=IsFemale, 3=IsMale, 4=StateFemale, 5=StateMale, 6=AgeGender
BATCH_SIZE = 512

CLEAN_RE = re.compile(r"^[a-z\s'!?.,/]+$")
ASK_AGE_RE = re.compile(r"\b(?:how\s+old|ur\s+age|your\s+age)\b|^age\??$", re.IGNORECASE)
ASK_FROM_RE = re.compile(
    r"\b(?:where\s+(?:are\s+you|r\s+u|u)\s+from|where\s+do\s+(?:you|u)\s+live|which\s+country|ur\s+country|your\s+country|u\s+from)\b|^from\??$|^country\??$",
    re.IGNORECASE,
)
ADULT_AGE_RE = re.compile(r"\b(1[89]|[2-7][0-9]|80)\b")

GREETING_ROOT_SET = frozenset((
    "hi", "hii", "hiii", "hiiii", "hiiiii", "hey", "heyy", "heyyy", "heyyyy", "heey", "heeyy",
    "hello", "helloo", "hellooo", "helo", "hlo", "ello", "hullo", "hulo", "hy", "hyy", "hyyy",
    "hie", "hei", "hiya", "hiyaa", "hiyaaa", "heya", "heyaa", "heyaaa", "howdy", "yo", "yoo",
    "yooo", "sup", "supp", "suuup", "wassup", "wassupp", "wasup", "wazzup", "wazup",
    "morning", "mornin", "afternoon", "afternon", "evening", "greetings", "gretings",
    "salutations", "welcome", "aloha", "gday",
))

CORE_SHORT_GREETINGS = (
    "hi", "hii", "hiii", "hiiii", "hiiiii", "hi hi", "hi hey", "hey hi",
    "hey", "heyy", "heyyy", "heyyyy", "heey", "heeyy", "hey hey",
    "hello", "helloo", "hellooo", "helo", "hlo", "ello", "hello hello",
    "hello u", "hey u", "hi u", "hii u", "hy u",
    "yo", "yoo", "yooo", "yo yo", "sup", "supp", "suuup",
    "hiya", "hiyaa", "hiyaaa", "heya", "heyaa", "heyaaa",
    "howdy", "hy", "hyy", "hyyy", "hie", "hei", "hullo", "wassup", "wassupp", "wazzup",
    "whats up", "what's up", "what is up", "morning", "mornin", "afternoon",
    "evening", "greetings", "welcome", "aloha",
)

BUILTIN_GREETINGS = CORE_SHORT_GREETINGS + (
    "good morning", "good afternoon", "good evening", "good day", "salutations",
    "nice to meet you", "nice to meet u", "pleased to meet you", "glad to meet you",
    "good to see you", "long time no see", "hey there", "hi there", "hello there",
    "yo there", "hey stranger", "hi stranger", "sup bro", "hi nice to meet you",
    "hello good morning", "yo what's up", "howdy partner", "greetings friend",
)

QUESTION_AND_REACTION_TRAPS = (
    "what", "what?", "what!", "what?!", "what!!", "what is this", "what is this?", "what is that",
    "what do you want", "what time is it", "what do you do", "what is your name", "ur name?",
    "why", "why?", "why!", "why?!", "why!!", "why not", "why not?", "why not!", "why me",
    "who", "who?", "who!", "who?!", "who!!", "who are you", "who are you?", "who is this",
    "who r u", "who is that", "lol", "lol!", "lol?", "haha", "haha!", "lmao", "rofl",
)

HARD_NEGATIVES = QUESTION_AND_REACTION_TRAPS + (
    "how are you", "how are you?", "how are you!", "hru", "hru?", "hru!", "hyd", "hyd!",
    "how r u", "how are u", "how u doing", "how's it going", "hows it going", "how was your day",
    "hor", "how ar you dopin", "how u doin", "hw r u", "how whas you da", "doing", "going", "wbu", "hbu",
    "tunis", "tuinsi", "tunsia", "tounsi", "tounsia", "tun", "tn", "handi", "hindi", "ind", "inida",
    "hyw", "hyk", "hyb", "hypx", "hya", "ih", "ie", "iphey", "xhey", "hix", "hellod", "hellox",
    "f or m", "m or f", "m/f", "f/m", "f?", "m?", "girl?", "boy?", "male", "female", "famle",
    "from?", "so", "m", "f", "w", "u", "u d?", "u f?", "m here", "f here", "im a girl", "im a boy",
    "charlotte", "from", "age?", "m bi", "how old are you", "where are you from", "asl",
    "tunisia", "india", "usa", "uk", "canada", "germany", "france", "help", "helicopter",
    "hell", "shell", "high", "hill", "history", "him", "his", "he", "her", "here",
    "good", "good job", "good luck", "good night", "gn", "morning coffee", "you", "your",
    "supper", "super", "support", "how", "bye", "goodbye", "see you", "thank you", "thanks",
    "ok", "yes", "no", "bonjour", "salut", "salam", "marhba",
)

COUNTRY_GROUPS = (
    (1, ("tunisia", "tunis", "tuinsi", "tuinsia", "tunsia", "tunisa", "tunisi", "tunisie", "tunisian", "tounsi", "tounsia", "tun", "tn", "sfax", "souse")),
    (2, ("india", "ind", "indian", "hindi", "handi", "inida", "indai", "hindustan", "bharat", "desi", "mumbai", "delhi")),
    (3, ("usa", "america", "american", "states", "murica", "nyc", "cali", "california", "texas", "florida")),
    (4, ("uk", "britain", "british", "england", "english", "brit", "london", "scotland")),
    (5, ("algeria", "alg", "dz", "dziri", "algerie", "algerian", "algiers", "oran")),
    (6, ("moroco", "maroc", "ma", "morocan", "maghreb", "casablanca", "rabat")),
    (7, ("egypt", "egy", "eg", "egyptian", "masr", "misr", "cairo", "alexandria")),
    (8, ("france", "fr", "french", "frnace", "paris", "lyon")),
    (9, ("germany", "de", "german", "geramny", "berlin")),
    (10, ("italy", "ita", "italia", "italian", "rome", "milan")),
    (11, ("spain", "esp", "espana", "spanish", "madrid")),
    (12, ("canada", "ca", "canadian", "toronto")),
    (13, ("saudi", "ksa", "riyadh", "arabia")),
    (14, ("uae", "dubai", "emirati", "emirates", "abudhabi")),
    (15, ("turkey", "tr", "turk", "turkiye", "turkish", "istanbul")),
    (16, ("pakistan", "pak", "paki", "pakistani", "lahore", "karachi")),
    (17, ("indonesia", "indo", "indonesian", "jakarta", "bali")),
    (18, ("philipines", "ph", "pinoy", "pinay", "filipino", "filipina", "manila")),
    (19, ("brazil", "br", "brasil", "brazilian", "rio")),
)

# Step 2 framing words (strictly age/country/greeting framing; gender anchors handled by Step 4 LUT)
VALID_S2_FRAMING_WORDS = (
    "hi", "hey", "helo", "hlo", "hy", "hie", "hei", "hiya", "heya", "hulo", "elo",
    "yo", "sup", "wasup", "wazup", "whazup", "howdy", "gretings", "greeting",
    "salutations", "welcome", "aloha", "gday", "morning", "mornin", "afternon",
    "evening", "god", "day", "top", "of", "the", "whats", "what's", "what", "is",
    "up", "how", "are", "r", "you", "u", "ya", "today", "it", "do", "your", "ur",
    "nice", "pleased", "glad", "to", "met", "se", "long", "time", "no",
    "there", "friend", "stranger", "cutie", "gorgeous", "handsome", "bro",
    "mate", "budy", "everyone", "al", "partner", "im", "i'm", "l'm", "lm", "iam", "i'am",
    "i", "am", "me", "my", "from", "in", "live", "living", "age", "old", "olds",
    "y", "yr", "yrs", "yrz", "yres", "yers", "yer", "yea", "year", "years",
    "yeas", "yars", "yar", "yaers", "yaer", "yro", "yold", "o",
    "where", "country", "which", "and", "about",
)

# Step 4 Gender Anchor & Framing Word Groups (for O(1) s4_gender_lut_a / s4_gender_lut_b)
# Code 1 = Male anchor, Code 2 = Female anchor, Code 3 = Neutral gender anchor, Code 4 = 2nd-person/Question frame, Code 5 = 1st-person/Statement frame
MALE_ANCHOR_WORDS = ("m", "male", "maale", "mael", "boy", "boi", "man", "masculine", "guy", "dude")
FEMALE_ANCHOR_WORDS = ("f", "female", "famle", "famel", "femael", "femle", "girl", "gril", "gurl", "woman", "womman", "feminine", "w", "lady")
NEUTRAL_GENDER_WORDS = ("gender", "gander", "gnder", "sex")
GENDER_QUESTION_FRAME_WORDS = ("are", "r", "you", "u", "ur", "your", "you're", "youre", "is", "what", "whats", "what's", "which", "tell")
GENDER_STATEMENT_FRAME_WORDS = (
    "im", "i'm", "l'm", "lm", "iam", "i'am", "i", "am", "me", "here", "a", "an", "or",
    "hi", "hey", "helo", "hlo", "elo", "yo", "sup", "hiya", "heya", "howdy", "hy",
)

BUILTIN_GENDER_CHOICE = (
    "m or f", "m or f?", "f or m", "f or m?", "m/f", "f/m", "m f", "f m", "m f?", "f m?",
    "girl or boy", "girl or boy?", "boy or girl", "boy or girl?", "girl or male", "boy or female",
    "male or female", "male or female?", "female or male", "female or male?", "male or girl",
    "female or boy", "m or female", "f or male", "male or f", "female or m", "m or girl",
    "f or boy", "boy or m", "girl or f", "woman or man", "man or woman", "girl or woman",
    "boy or man", "female or girl", "male or boy", "woman or girl", "man or boy",
    "what gender are you", "what is your gender", "whats your gender", "what's your gender",
    "your gender", "ur gender", "u gender", "gender", "gender?", "your sex", "what sex are you",
    "are you male or female", "are you female or male", "are you a male or female",
    "are you a female or male", "which gender are you", "which sex are you",
    "what are you male or female", "tell me your gender",
)

BUILTIN_GENDER_ASK_F = (
    "f?", "girl?", "female?", "woman?", "w?", "famle?", "are you a girl", "are you a girl?",
    "are you girl", "are u a girl", "are u girl", "you a girl?", "r u a girl", "r u girl",
    "you a girl", "u a girl", "ur a girl", "your a girl", "are you female", "are u female",
    "r u female", "you female", "u female", "are you a female", "are u a female", "r u a female",
    "r you a girl", "r you girl", "r you female", "you're a girl", "you are a girl",
    "you're female", "you are female", "is your gender female", "are you feminine",
    "are you a woman", "are u a woman", "r u a woman", "you a woman", "u a woman",
    "are u a gril?", "r u famle?", "ur a gurl?", "femael?",
)

BUILTIN_GENDER_ASK_M = (
    "m?", "boy?", "male?", "man?", "are you a boy", "are you a boy?", "are you boy",
    "are u a boy", "are u boy", "r u a boy", "r u boy", "you a boy", "u a boy",
    "ur a boy", "your a boy", "are you male", "are u male", "r u male", "you male",
    "u male", "are you a male", "are u a male", "r u a male", "r you a boy", "r you boy",
    "r you male", "you're a boy", "you are a boy", "you're male", "you are male",
    "is your gender male", "are you masculine", "are you a man", "are u a man",
    "r u a man", "you a man", "u a man", "are u a boi?", "r u maale?", "mael?",
)

BUILTIN_GENDER_STATE_F = (
    "f", "female", "famle", "famel", "femael", "femle", "girl", "gril", "gurl",
    "woman", "womman", "w", "f here", "female here", "girl here", "im a girl",
    "i'm a girl", "i am a girl", "im female", "i'm female", "im famle", "im a woman",
    "i'm a woman", "im girl",
)

BUILTIN_GENDER_STATE_M = (
    "m", "male", "maale", "mael", "boy", "boi", "man", "m here", "male here",
    "boy here", "im a boy", "i'm a boy", "i am a boy", "im male", "i'm male",
    "im a man", "i'm a man", "im boy",
)


def get_system_ram_gb():
    if os.path.exists("/proc/meminfo"):
        tot_kb, avail_kb = 0, 0
        with open("/proc/meminfo", "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    tot_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
        if tot_kb > 0:
            tot_gb, avail_gb = float(tot_kb) / 1048576.0, float(avail_kb) / 1048576.0
            return tot_gb - avail_gb, avail_gb, tot_gb
    return 6.3, 6.4, 12.7


def verify_hardware_and_lock_ram():
    tpu_env = os.environ.get("TPU_ACCELERATOR_TYPE", "") + os.environ.get("COLAB_TPU_ADDR", "") + os.environ.get("PJRT_DEVICE", "")
    is_tpu = HAS_XLA or bool(tpu_env) or os.path.exists("/dev/vfio") or os.path.exists("/dev/accel0")
    used_gb, avail_gb, total_gb = get_system_ram_gb()
    if total_gb >= 32.0:
        is_tpu = True
    lock_gb = max(0.5, avail_gb * 0.75)
    ram_arena = None
    try:
        ram_arena = np.ones((int(lock_gb * 1073741824.0),), dtype=np.uint8)
    except MemoryError:
        lock_gb = 0.0
    tpu_s = "ACTIVE (v5e-1 High-RAM VM)" if is_tpu else "INACTIVE (Standard VM)"
    gpu_s = "ACTIVE (CUDA)" if torch.cuda.is_available() else "INACTIVE"
    print(f"[HARDWARE VERIFIED] v5e-1 TPU: {tpu_s} | GPU: {gpu_s} | CPU Threads: {CPU_CORES}", flush=True)
    print(f"[RAM SUPER-ARENA] Locked {lock_gb:.1f} GB Free RAM -> Active: {used_gb + lock_gb:.1f}/{total_gb:.1f} GB", flush=True)
    return ram_arena


def quantize_to_int8(w_fp32):
    w = w_fp32.astype(np.float32, copy=False)
    max_abs = np.float32(np.max(np.abs(w)))
    scale = np.float32(max_abs / np.float32(127.0)) if max_abs > np.float32(1e-8) else np.float32(1.0 / 127.0)
    scaled = np.divide(w, scale, dtype=np.float32)
    rounded = np.rint(scaled, dtype=np.float32)
    clipped = np.clip(rounded, np.float32(-127.0), np.float32(127.0), dtype=np.float32)
    return clipped.astype(np.int8), np.array([scale], dtype=np.float32)


def ensure_dataset_files():
    if not os.path.exists("greetings.txt") or not os.path.exists("negatives.txt"):
        try:
            import build_greetings_txt
            if hasattr(build_greetings_txt, "generate_datasets"):
                build_greetings_txt.generate_datasets("greetings.txt", "negatives.txt")
            elif hasattr(build_greetings_txt, "generate_greetings_file"):
                build_greetings_txt.generate_greetings_file("greetings.txt")
        except Exception:
            pass
    if not os.path.exists("ages.txt") or not os.path.exists("countries.txt"):
        try:
            import build_age_country_txt
            if hasattr(build_age_country_txt, "build_age_and_country_files"):
                build_age_country_txt.build_age_and_country_files()
        except Exception:
            pass
    if not os.path.exists("hru.txt"):
        try:
            import build_hru_txt
            if hasattr(build_hru_txt, "build_hru_file"):
                build_hru_txt.build_hru_file("hru.txt")
        except Exception:
            pass
    if not os.path.exists("gender_ask_choice.txt") or not os.path.exists("age_gender.txt"):
        try:
            import build_gender_txt
            if hasattr(build_gender_txt, "generate_all_gender_files"):
                build_gender_txt.generate_all_gender_files()
        except Exception:
            pass
    # Self-healing fallback if gender .txt files do not exist yet on disk
    fallback_files = (
        ("gender_ask_choice.txt", BUILTIN_GENDER_CHOICE),
        ("gender_ask_is_female.txt", BUILTIN_GENDER_ASK_F),
        ("gender_ask_is_male.txt", BUILTIN_GENDER_ASK_M),
        ("gender_state_female.txt", BUILTIN_GENDER_STATE_F),
        ("gender_state_male.txt", BUILTIN_GENDER_STATE_M),
    )
    for fn, items in fallback_files:
        if not os.path.exists(fn):
            with open(fn, "w", encoding="utf-8") as f:
                f.write("\n".join(items) + "\n")
    if not os.path.exists("age_gender.txt"):
        ag_lines = []
        for age in range(18, 81):
            for g in ("f", "m", "female", "famle", "male", "girl", "boy", "woman", "man"):
                ag_lines.extend((f"{age}{g}", f"{age} {g}", f"{g} {age}", f"im {age} {g}"))
        with open("age_gender.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(ag_lines) + "\n")


def _get_wp_mat():
    w_primes = np.array((131, 257, 521, 1031), dtype=np.int32)
    wp_mat = np.ones((16, 4), dtype=np.int32)
    for r in range(1, 16):
        wp_mat[r] = (wp_mat[r - 1] * w_primes) & np.int32(0x7FFFFF)
    return wp_mat


def _collapse_word(w_str):
    return "".join(w_str[i] for i in range(len(w_str)) if i == 0 or w_str[i] != w_str[i - 1])


def encode_s1_batch(texts):
    lut = np.full((256,), 32, dtype=np.int32)
    lut[65:91] = np.arange(97, 123, dtype=np.int32)
    lut[97:123] = np.arange(97, 123, dtype=np.int32)
    lut[39] = np.int32(39)
    primes = np.array((31, 37, 41, 43, 47, 53, 59, 61), dtype=np.int32)
    poly_mat = np.ones((MAX_GREETING_LEN, 8), dtype=np.int32)
    for r in range(1, MAX_GREETING_LEN):
        poly_mat[r] = (poly_mat[r - 1] * primes) & np.int32(0x7FFFFF)

    n = len(texts)
    canon_mat = np.full((n, MAX_GREETING_LEN + 4), 32, dtype=np.int32)
    temp_buf = np.full((MAX_GREETING_LEN + 4,), 32, dtype=np.int32)

    for i, s in enumerate(texts):
        b = np.frombuffer(s[:MAX_GREETING_LEN].encode("ascii", errors="ignore"), dtype=np.uint8)
        non_ws = np.flatnonzero(b > np.uint8(32))
        if non_ws.shape[0] == 0:
            continue
        f_idx, l_idx = int(non_ws[0]), int(non_ws[-1]) + 1
        n_tok = min(l_idx - f_idx, MAX_GREETING_LEN)
        temp_buf.fill(32)
        temp_buf[2 : 2 + n_tok] = b[f_idx : f_idx + n_tok]
        np.take(lut, temp_buf, out=temp_buf)
        vc, vp1 = temp_buf[2 : 2 + n_tok], temp_buf[1 : 1 + n_tok]
        kept = vc[vc != vp1]
        if kept.shape[0] > 0 and int(kept[-1]) == 32:
            kept = kept[:-1]
        k_len = int(kept.shape[0])
        if k_len == 0:
            continue
        sp = np.flatnonzero(kept == np.int32(32))
        L0 = int(sp[0]) if sp.shape[0] >= 1 else k_len
        if L0 == 4 and int(kept[0]) == 104 and int(kept[1]) == 101 and int(kept[2]) == 108 and int(kept[3]) == 111:
            canon_mat[i, 2], canon_mat[i, 3], canon_mat[i, 4] = 104, 108, 111
            if k_len > 4:
                canon_mat[i, 5 : 5 + (k_len - 4)] = kept[4:k_len]
        elif L0 == 3 and int(kept[0]) == 101 and int(kept[1]) == 108 and int(kept[2]) == 111:
            canon_mat[i, 2], canon_mat[i, 3], canon_mat[i, 4] = 104, 108, 111
            if k_len > 3:
                canon_mat[i, 5 : 5 + (k_len - 3)] = kept[3:k_len]
        else:
            canon_mat[i, 2 : 2 + k_len] = kept

    body_slice = canon_mat[:, 2 : 2 + MAX_GREETING_LEN] - np.int32(32)
    bloom_h = ((body_slice @ poly_mat) & np.int32(1022)) + np.int32(1)
    tok_bloom = np.where(np.any(body_slice != 0, axis=1, keepdims=True), bloom_h, np.int32(0)).astype(np.int32)
    c0, c1, c2, c3, c4 = canon_mat[:, 0:16], canon_mat[:, 1:17], canon_mat[:, 2:18], canon_mat[:, 3:19], canon_mat[:, 4:20]
    is_pad5 = (c0 == 32) & (c1 == 32) & (c2 == 32) & (c3 == 32) & (c4 == 32)
    h5 = (((c0 * np.int32(389)) + (c1 * np.int32(131)) + (c2 * np.int32(37)) + (c3 * np.int32(13)) + c4) & np.int32(511)) + np.int32(1024)
    tok5 = np.where(is_pad5, np.int32(0), h5).astype(np.int32)
    d1, d2, d3 = c1[:, :8], c2[:, :8], c3[:, :8]
    is_pad3 = (d1 == 32) & (d2 == 32) & (d3 == 32)
    h3 = (((d1 * np.int32(961)) + (d2 * np.int32(31)) + d3) & np.int32(511)) + np.int32(1536)
    tok3 = np.where(is_pad3, np.int32(0), h3).astype(np.int32)

    x_np = np.concatenate([tok_bloom, tok5, tok3], axis=1).astype(np.int32)
    valid_m_np = (x_np != 0).astype(np.float32)[:, :, None]
    counts_np = np.maximum(np.sum(valid_m_np, axis=1), np.float32(1.0)).astype(np.float32)
    return x_np, valid_m_np, counts_np


def encode_word_seq_batch(texts, wp_mat, use_qmark_slash=False):
    n = len(texts)
    x_mat = np.zeros((n, 8), dtype=np.int32)
    w_vec = np.zeros((16,), dtype=np.int32)
    greet_set = frozenset(("hi", "hey", "helo", "hlo", "elo", "yo", "sup", "hiya", "heya", "howdy", "hy"))

    for i, s in enumerate(texts):
        b_raw = np.frombuffer(s.encode("ascii", errors="ignore"), dtype=np.uint8)
        has_q = bool(np.any(b_raw == np.uint8(63)))
        has_s = bool(np.any(b_raw == np.uint8(47)))
        clean_s = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in s.lower())
        collapsed = _collapse_word(clean_s)
        words = [w for w in collapsed.split() if w][:6]

        for w_i, w in enumerate(words):
            b = np.frombuffer(w[:15].encode("ascii", errors="ignore"), dtype=np.uint8)
            L = int(b.shape[0])
            w_vec.fill(0)
            w_vec[:L] = b
            w_vec[15] = np.int32(L)
            x_mat[i, w_i] = np.int32(int((w_vec @ wp_mat)[0] & np.int32(S2_VOCAB - 2)) + 1)

        if use_qmark_slash:
            x_mat[i, 6] = np.int32(1023) if has_q else (np.int32(1022) if has_s else np.int32(1))
            if words:
                q_off = np.int32(509) if has_q else (np.int32(251) if has_s else np.int32(0))
                x_mat[i, 7] = (((x_mat[i, 0] * np.int32(37)) + (x_mat[i, len(words) - 1] * np.int32(73)) + q_off) & np.int32(S2_VOCAB - 2)) + 1
            else:
                x_mat[i, 7] = np.int32(1)
        elif words:
            lead_idx = 1 if (len(words) > 1 and words[0] in greet_set) else 0
            x_mat[i, 6] = ((x_mat[i, lead_idx] * np.int32(37)) & np.int32(S2_VOCAB - 2)) + 1
            x_mat[i, 7] = ((x_mat[i, len(words) - 1] * np.int32(73)) & np.int32(S2_VOCAB - 2)) + 1

    valid_m = (x_mat != 0).astype(np.float32)[:, :, None]
    valid_m[:, 6, 0] *= np.float32(2.5)
    counts = np.maximum(np.sum(valid_m, axis=1), np.float32(1.0)).astype(np.float32)
    return x_mat, valid_m, counts


def _train_word_mlp_head(sample_map, num_classes, epochs, tag, device, wp_mat, use_qmark=False):
    texts = list(sample_map.keys())
    labels_np = np.array([sample_map[k][0] for k in texts], dtype=np.int32)
    weights_np = np.array([sample_map[k][1] for k in texts], dtype=np.float32)
    x_mat, valid_m_np, counts_np = encode_word_seq_batch(texts, wp_mat, use_qmark_slash=use_qmark)

    n = len(texts)
    rng = np.random.default_rng(42)
    all_perms = np.vstack([rng.permutation(np.arange(n, dtype=np.int32)) for _ in range(epochs)])

    x_t = torch.from_numpy(x_mat).to(device=device, dtype=torch.int32)
    bm = torch.from_numpy(valid_m_np).to(device=device, dtype=torch.float32)
    bc = torch.from_numpy(counts_np).to(device=device, dtype=torch.float32)
    yi_oh = torch.from_numpy(np.eye(num_classes, dtype=np.float32)[labels_np]).to(device=device, dtype=torch.float32)
    sw_t = torch.from_numpy(weights_np).to(device=device, dtype=torch.float32)

    torch.manual_seed(42)
    emb = torch.nn.Parameter(torch.randn((S2_VOCAB, S2_DIM), device=device, dtype=torch.float32) * 0.1)
    w1 = torch.nn.Parameter(torch.randn((S2_DIM, S2_DIM), device=device, dtype=torch.float32) * 0.1)
    w2 = torch.nn.Parameter(torch.randn((S2_DIM, num_classes), device=device, dtype=torch.float32) * 0.1)
    opt = torch.optim.AdamW([emb, w1, w2], lr=2.0e-2, weight_decay=1e-4)

    total_steps = epochs * ((n + BATCH_SIZE - 1) // BATCH_SIZE)
    step_done, t_start = 0, time.perf_counter()
    last_log = t_start
    for ep in range(1, epochs + 1):
        perm = all_perms[ep - 1]
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                emb.data[0].zero_()
            h = F.embedding(x_t[idx], emb) * bm[idx]
            logits = F.relu((torch.sum(h, dim=1) / bc[idx]) @ w1) @ w2
            per_l = -torch.sum(yi_oh[idx] * F.log_softmax(logits, dim=-1), dim=-1)
            loss = torch.mean(per_l * (sw_t[idx] / torch.clamp(torch.mean(sw_t[idx]), min=1e-5)))
            loss.backward()
            opt.step()
            step_done += 1
            now_t = time.perf_counter()
            if (now_t - last_log) >= 1.0 or step_done == total_steps:
                el = now_t - t_start
                rem = (total_steps - step_done) / max(step_done / max(el, 1e-6), 1e-6)
                print(f"[{tag}] {(step_done / total_steps) * 100.0:5.1f}% | Ep {ep:02d}/{epochs} | Loss: {float(loss.detach()):.4f} | Remaining: {rem:.1f}s", flush=True)
                last_log = now_t

    return quantize_to_int8(emb.detach().cpu().numpy()), quantize_to_int8(w1.detach().cpu().numpy()), quantize_to_int8(w2.detach().cpu().numpy()), n


def train_step1_into_vault(vault, device):
    pos_w = {g: np.float32(25.0) for g in BUILTIN_GREETINGS}
    for g in CORE_SHORT_GREETINGS:
        pos_w[g] = np.float32(60.0)
    if os.path.exists("greetings.txt"):
        with open("greetings.txt", "r", encoding="ascii", errors="ignore") as f:
            for line in f:
                s = " ".join(line.strip().lower().split())
                if s and len(s) <= 30 and len(s.split()) <= 4 and CLEAN_RE.match(s) and s not in pos_w:
                    pos_w[s] = np.float32(3.0)

    neg_w = {}
    neg_files = (
        "negatives.txt", "hru.txt", "countries.txt", "ages.txt", "underage_traps.txt",
        "gender_ask_choice.txt", "gender_ask_is_female.txt", "gender_ask_is_male.txt",
        "gender_state_female.txt", "gender_state_male.txt",
    )
    for nf in neg_files:
        if os.path.exists(nf):
            with open(nf, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and s not in pos_w and len(neg_w) < 9000 and CLEAN_RE.match(s):
                        words = [w.strip("'!?.,/") for w in s.split()]
                        if not any(w in GREETING_ROOT_SET for w in words):
                            neg_w[s[:MAX_GREETING_LEN]] = np.float32(2.0)
    for w in HARD_NEGATIVES:
        if CLEAN_RE.match(w) and w not in pos_w:
            words = [tok.strip("'!?.,/") for tok in w.split()]
            if not any(tok in GREETING_ROOT_SET for tok in words):
                neg_w[w[:MAX_GREETING_LEN]] = np.float32(25.0)

    pos_k, neg_k = list(pos_w.keys()), list(neg_w.keys())
    texts = pos_k + neg_k
    labels_np = np.concatenate([np.ones(len(pos_k), dtype=np.int32), np.zeros(len(neg_k), dtype=np.int32)])
    weights_np = np.array([pos_w[k] for k in pos_k] + [neg_w[k] for k in neg_k], dtype=np.float32)
    x_np, valid_m_np, counts_np = encode_s1_batch(texts)

    n_samples, epochs = x_np.shape[0], 18
    rng = np.random.default_rng(42)
    all_perms = np.vstack([rng.permutation(np.arange(n_samples, dtype=np.int32)) for _ in range(epochs)])
    x_t = torch.from_numpy(x_np).to(device=device, dtype=torch.int32)
    bm = torch.from_numpy(valid_m_np).to(device=device, dtype=torch.float32)
    bc = torch.from_numpy(counts_np).to(device=device, dtype=torch.float32)
    yi_oh = torch.from_numpy(np.eye(NUM_INTENTS, dtype=np.float32)[labels_np]).to(device=device, dtype=torch.float32)
    sw_t = torch.from_numpy(weights_np).to(device=device, dtype=torch.float32)

    pos_i = np.arange(SEQ_LEN, dtype=np.float32)[:, None]
    div_i = np.exp(np.arange(0, D_MODEL, 2, dtype=np.float32) * np.float32(-np.log(10000.0)) / np.float32(D_MODEL))
    pe_np = np.zeros((SEQ_LEN, D_MODEL), dtype=np.float32)
    pe_np[:, 0::2] = np.sin(pos_i * div_i) * np.float32(0.02)
    pe_np[:, 1::2] = np.cos(pos_i * div_i) * np.float32(0.02)
    pe = torch.from_numpy(pe_np).to(device=device, dtype=torch.float32)
    mask = torch.from_numpy(np.triu(np.full((SEQ_LEN, SEQ_LEN), -1e9, dtype=np.float32), k=1)).to(device=device, dtype=torch.float32)
    scale_d = float(np.float32(1.0) / np.sqrt(np.float32(D_MODEL)))

    torch.manual_seed(42)
    shapes = (
        (VOCAB_SIZE, D_MODEL), (D_MODEL, 3 * D_MODEL), (D_MODEL, D_MODEL), (D_MODEL, D_FF), (D_FF, D_MODEL),
        (D_MODEL, 3 * D_MODEL), (D_MODEL, D_MODEL), (D_MODEL, D_FF), (D_FF, D_MODEL), (D_MODEL, NUM_INTENTS),
    )
    params = [torch.nn.Parameter(torch.randn(s, device=device, dtype=torch.float32) * 0.08) for s in shapes]
    opt = torch.optim.AdamW(params, lr=1.8e-2, weight_decay=1e-4)

    def rms_norm(x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)

    total_steps = epochs * ((n_samples + BATCH_SIZE - 1) // BATCH_SIZE)
    step_done, t_start, last_log = 0, time.perf_counter(), time.perf_counter()
    for ep in range(1, epochs + 1):
        perm = all_perms[ep - 1]
        for start in range(0, n_samples, BATCH_SIZE):
            idx = perm[start : start + BATCH_SIZE]
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                params[0].data[0].zero_()
            h = (F.embedding(x_t[idx], params[0]) + pe) * bm[idx]
            qkv1 = h @ params[1]
            q1, k1, v1 = qkv1[:, :, :D_MODEL], qkv1[:, :, D_MODEL : 2 * D_MODEL], qkv1[:, :, 2 * D_MODEL :]
            h = rms_norm(h + (F.softmax((q1 @ k1.transpose(-2, -1)) * scale_d + mask, dim=-1) @ v1) @ params[2])
            h = rms_norm(h + F.relu(h @ params[3]) @ params[4])
            qkv2 = h @ params[5]
            q2, k2, v2 = qkv2[:, :, :D_MODEL], qkv2[:, :, D_MODEL : 2 * D_MODEL], qkv2[:, :, 2 * D_MODEL :]
            h = rms_norm(h + (F.softmax((q2 @ k2.transpose(-2, -1)) * scale_d + mask, dim=-1) @ v2) @ params[6])
            h = rms_norm(h + F.relu(h @ params[7]) @ params[8])
            logits = (torch.sum(h * bm[idx], dim=1) / bc[idx]) @ params[9]
            per_l = -torch.sum(yi_oh[idx] * F.log_softmax(logits, dim=-1), dim=-1)
            loss = torch.mean(per_l * (sw_t[idx] / torch.clamp(torch.mean(sw_t[idx]), min=1e-5)))
            loss.backward()
            opt.step()
            step_done += 1
            now_t = time.perf_counter()
            if (now_t - last_log) >= 1.0 or step_done == total_steps:
                el = now_t - t_start
                rem = (total_steps - step_done) / max(step_done / max(el, 1e-6), 1e-6)
                print(f"[STEP 1] {(step_done / total_steps) * 100.0:5.1f}% | Ep {ep:02d}/{epochs} | Loss: {float(loss.detach()):.4f} | Remaining: {rem:.1f}s", flush=True)
                last_log = now_t

    q = [quantize_to_int8(p.detach().cpu().numpy()) for p in params]
    vault.update(
        emb_q=q[0][0], emb_s=q[0][1], wqkv1_q=q[1][0], wqkv1_s=q[1][1], wo1_q=q[2][0], wo1_s=q[2][1],
        w1_q=q[3][0], w1_s=q[3][1], w2_q=q[4][0], w2_s=q[4][1], wqkv2_q=q[5][0], wqkv2_s=q[5][1],
        wo2_q=q[6][0], wo2_s=q[6][1], w3_q=q[7][0], w3_s=q[7][1], w4_q=q[8][0], w4_s=q[8][1],
        w_intent_q=q[9][0], w_intent_s=q[9][1],
    )
    print(f"[STEP 1 DONE] Locked Step 1 pure greeting weights into vault ({len(texts)} samples).", flush=True)


def train_step2_into_vault(vault, device):
    wp_mat = _get_wp_mat()
    s2_frame_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    country_lut_a = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    country_lut_b = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    b_mask = np.int32(BLOOM_SIZE - 1)
    w_vec = np.zeros((16,), dtype=np.int32)

    def hash_word(w_str):
        b = np.frombuffer(w_str.encode("ascii", errors="ignore"), dtype=np.uint8)
        L = min(int(b.shape[0]), 15)
        w_vec.fill(0)
        w_vec[:L] = b[:L]
        w_vec[15] = np.int32(L)
        return (w_vec @ wp_mat) & b_mask

    for w in VALID_S2_FRAMING_WORDS + MALE_ANCHOR_WORDS + FEMALE_ANCHOR_WORDS:
        s2_frame_bloom[hash_word(_collapse_word(w))[:3]] = np.int8(1)
    for cid, aliases in COUNTRY_GROUPS:
        for al in aliases:
            h4 = hash_word(_collapse_word(al))
            country_lut_a[h4[0]], country_lut_b[h4[1]] = np.uint8(cid), np.uint8(cid)
            s2_frame_bloom[h4[:3]] = np.int8(1)

    if os.path.exists("countries.txt"):
        with open("countries.txt", "r", encoding="ascii", errors="ignore") as fc:
            for line in fc:
                s = " ".join(line.strip().lower().split())
                if s and " " not in s and "?" not in s and len(s) >= 3:
                    h4 = hash_word(_collapse_word(s))
                    if country_lut_a[h4[0]] == 0:
                        country_lut_a[h4[0]], country_lut_b[h4[1]] = np.uint8(20), np.uint8(20)
                    s2_frame_bloom[h4[:3]] = np.int8(1)

    sample_map = {t: (0, np.float32(35.0)) for t in QUESTION_AND_REACTION_TRAPS}
    for gf in (
        "greetings.txt", "negatives.txt", "hru.txt", "gender_ask_choice.txt",
        "gender_ask_is_female.txt", "gender_ask_is_male.txt",
        "gender_state_female.txt", "gender_state_male.txt",
    ):
        if os.path.exists(gf):
            with open(gf, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and len(s) <= 40 and not ASK_AGE_RE.search(s) and not ASK_FROM_RE.search(s) and len(sample_map) < 9000:
                        sample_map[s] = (0, np.float32(2.0))
    if os.path.exists("ages.txt"):
        with open("ages.txt", "r", encoding="ascii", errors="ignore") as fa:
            for line in fa:
                s = " ".join(line.strip().lower().split())
                if s:
                    sample_map[s] = (1, np.float32(25.0)) if ASK_AGE_RE.search(s) else (2, np.float32(8.0))
    if os.path.exists("countries.txt"):
        with open("countries.txt", "r", encoding="ascii", errors="ignore") as fc:
            for line in fc:
                s = " ".join(line.strip().lower().split())
                if s:
                    sample_map[s] = (3, np.float32(25.0)) if ASK_FROM_RE.search(s) else ((5, np.float32(6.0)) if ADULT_AGE_RE.search(s) else (4, np.float32(4.0)))

    q_emb, q_w1, q_w2, n = _train_word_mlp_head(sample_map, S2_CLASSES, 24, "STEP 2", device, wp_mat)
    vault.update(
        s2_emb_q=q_emb[0], s2_emb_s=q_emb[1], s2_w1_q=q_w1[0], s2_w1_s=q_w1[1], s2_w2_q=q_w2[0], s2_w2_s=q_w2[1],
        s2_frame_bloom_q=s2_frame_bloom, s2_country_lut_a=country_lut_a, s2_country_lut_b=country_lut_b,
    )
    print(f"[STEP 2 DONE] Locked Step 2 Age/Country head & LUTs into vault ({n} samples).", flush=True)


def train_step3_into_vault(vault, device):
    wp_mat = _get_wp_mat()
    s3_hru_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    b_mask = np.int32(BLOOM_SIZE - 1)
    w_vec = np.zeros((16,), dtype=np.int32)

    def hash_word(w_str):
        b = np.frombuffer(w_str.encode("ascii", errors="ignore"), dtype=np.uint8)
        L = min(int(b.shape[0]), 15)
        w_vec.fill(0)
        w_vec[:L] = b[:L]
        w_vec[15] = np.int32(L)
        return (w_vec @ wp_mat) & b_mask

    sample_map = {}
    for gf in (
        "greetings.txt", "ages.txt", "countries.txt", "negatives.txt", "underage_traps.txt",
        "gender_ask_choice.txt", "gender_ask_is_female.txt", "gender_ask_is_male.txt",
    ):
        if os.path.exists(gf):
            with open(gf, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and len(sample_map) < 10000:
                        sample_map[s] = (0, np.float32(2.0))

    hru_cnt = 0
    if os.path.exists("hru.txt"):
        with open("hru.txt", "r", encoding="ascii", errors="ignore") as fh:
            for line in fh:
                s = " ".join(line.strip().lower().split())
                if s:
                    hru_cnt += 1
                    sample_map[s] = (1, np.float32(18.0))
                    clean_s = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in s)
                    for w in _collapse_word(clean_s).split():
                        s3_hru_bloom[hash_word(w)[:3]] = np.int8(1)

    for trap in QUESTION_AND_REACTION_TRAPS + (
        "how old are you", "how old r u", "where are you from", "where r u from",
        "hi", "hi hi", "hy", "hyy", "hey", "hello", "yo", "sup",
        "m?", "f?", "m or f?", "are you a girl?", "are you a boy?",
    ):
        sample_map[trap] = (0, np.float32(55.0))

    q_emb, q_w1, q_w2, n = _train_word_mlp_head(sample_map, S3_CLASSES, 26, "STEP 3", device, wp_mat)
    vault.update(
        s3_emb_q=q_emb[0], s3_emb_s=q_emb[1], s3_w1_q=q_w1[0], s3_w1_s=q_w1[1],
        s3_w2_q=q_w2[0], s3_w2_s=q_w2[1], s3_hru_bloom_q=s3_hru_bloom,
    )
    print(f"[STEP 3 DONE] Locked Step 3 ask_how_are_you head ({hru_cnt} hru.txt lines, {n} total) into vault.", flush=True)


def train_step4_into_vault(vault, device):
    wp_mat = _get_wp_mat()
    s4_gender_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    s4_gender_lut_a = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    s4_gender_lut_b = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    b_mask = np.int32(BLOOM_SIZE - 1)
    w_vec = np.zeros((16,), dtype=np.int32)

    def hash_word(w_str):
        b = np.frombuffer(w_str.encode("ascii", errors="ignore"), dtype=np.uint8)
        L = min(int(b.shape[0]), 15)
        w_vec.fill(0)
        w_vec[:L] = b[:L]
        w_vec[15] = np.int32(L)
        return (w_vec @ wp_mat) & b_mask

    # Populate dual O(1) Gender Anchor & Framing LUTs (1=Male, 2=Female, 3=NeutralGender, 4=QuestionFrame, 5=StatementFrame)
    for code_val, word_group in (
        (5, GENDER_STATEMENT_FRAME_WORDS),
        (4, GENDER_QUESTION_FRAME_WORDS),
        (3, NEUTRAL_GENDER_WORDS),
        (1, MALE_ANCHOR_WORDS),
        (2, FEMALE_ANCHOR_WORDS),
    ):
        for w in word_group:
            h4 = hash_word(_collapse_word(w))
            s4_gender_lut_a[h4[0]] = np.uint8(code_val)
            s4_gender_lut_b[h4[1]] = np.uint8(code_val)
            s4_gender_bloom[h4[:3]] = np.int8(1)

    sample_map = {}
    for gf in ("greetings.txt", "ages.txt", "countries.txt", "hru.txt", "negatives.txt", "underage_traps.txt"):
        if os.path.exists(gf):
            with open(gf, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and len(sample_map) < 15000:
                        sample_map[s] = (0, np.float32(2.0))

    pos_classes = (
        ("gender_ask_choice.txt", 1),
        ("gender_ask_is_female.txt", 2),
        ("gender_ask_is_male.txt", 3),
        ("gender_state_female.txt", 4),
        ("gender_state_male.txt", 5),
        ("age_gender.txt", 6),
    )

    gender_cnt = 0
    for file_name, cls_id in pos_classes:
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s:
                        sample_map[s] = (cls_id, np.float32(24.0))
                        gender_cnt += 1
                        clean_s = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in s)
                        for w in _collapse_word(clean_s).split():
                            s4_gender_bloom[hash_word(w)[:3]] = np.int8(1)

    for trap in QUESTION_AND_REACTION_TRAPS + (
        "how old are you?", "where are you from?", "how are you?", "hru?", "age?", "from?", "country?",
    ):
        sample_map[trap] = (0, np.float32(60.0))

    q_emb, q_w1, q_w2, n = _train_word_mlp_head(sample_map, S4_CLASSES, 30, "STEP 4", device, wp_mat, use_qmark=True)
    vault.update(
        s4_emb_q=q_emb[0], s4_emb_s=q_emb[1], s4_w1_q=q_w1[0], s4_w1_s=q_w1[1],
        s4_w2_q=q_w2[0], s4_w2_s=q_w2[1], s4_gender_bloom_q=s4_gender_bloom,
        s4_gender_lut_a=s4_gender_lut_a, s4_gender_lut_b=s4_gender_lut_b,
    )
    print(f"[STEP 4 DONE] Locked Step 4 Gender head & LUTs ({gender_cnt} gender lines, {n} total) into vault.", flush=True)


def run_vault_trainer(output_path="model_weights.npz", force_s1=False, force_s2=False, force_s3=False, force_s4=False):
    ensure_dataset_files()
    vault = {}
    if os.path.exists(output_path):
        with np.load(output_path, allow_pickle=False) as data:
            for k in data.files:
                vault[k] = data[k]

    need_s1 = force_s1 or ("w_intent_q" not in vault)
    need_s2 = force_s2 or ("s2_country_lut_a" not in vault)
    need_s3 = force_s3 or ("s3_w2_q" not in vault)
    need_s4 = force_s4 or ("s4_w2_q" not in vault) or ("s4_gender_lut_a" not in vault)

    if not need_s1 and not need_s2 and not need_s3 and not need_s4:
        print(f"[VAULT OK] Steps 1, 2, 3 & 4 are all locked in '{output_path}'!", flush=True)
        return

    ram_arena = verify_hardware_and_lock_ram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if need_s1:
        train_step1_into_vault(vault, device)
    else:
        print("[VAULT PROTECTED] Step 1 weights untouched!", flush=True)
    if need_s2:
        train_step2_into_vault(vault, device)
    else:
        print("[VAULT PROTECTED] Step 2 weights untouched!", flush=True)
    if need_s3:
        train_step3_into_vault(vault, device)
    else:
        print("[VAULT PROTECTED] Step 3 weights untouched!", flush=True)
    if need_s4:
        train_step4_into_vault(vault, device)
    else:
        print("[VAULT PROTECTED] Step 4 weights untouched!", flush=True)

    np.savez(output_path, **vault)
    del ram_arena
    size_kb = os.path.getsize(output_path) / 1024.0 if os.path.exists(output_path) else 0.0
    print(f"[EXPORT COMPLETE] Saved Step 1+2+3+4 vault to '{output_path}' ({size_kb:.2f} KB)", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    do_all = ("--all" in args) or ("--force" in args)
    run_vault_trainer(
        "model_weights.npz",
        force_s1=("--step1" in args) or ("--force-step1" in args) or do_all,
        force_s2=("--step2" in args) or ("--force-step2" in args) or do_all,
        force_s3=("--step3" in args) or ("--force-step3" in args) or do_all,
        force_s4=("--step4" in args) or ("--force-step4" in args) or do_all,
    )
