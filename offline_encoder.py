#!/usr/bin/env python3
# FILE: offline_encoder.py
# PURPOSE: Rule 15 Modular Pure-NumPy Encoder, Super-Bloom Builder, Entity LUT Builder & INT8 Verifier
# RULES: Enforces Rules 1-15 (<= 220 lines, zero float64/int64, Prime Modulus % 4091 / % 8191, bit-exact with production_server.py)

import os
import numpy as np

BLOOM_SIZE = 65536
WORD_VOCAB = 4096
BIGRAM_VOCAB = 8192
WORD_MOD = 4091
BIGRAM_MOD = 8191
TOK_COUNTRY = 4093
TOK_NAME_F = 4094
TOK_NAME_M = 4095
D_EMB = 64
D_STRUCT = 16
D_IN = 128
D_HID1 = 256
D_HID2 = 128
MAX_LEN = 45

BIT_WILDCARD_NAME = np.uint32(1 << 30)
BIT_GLOBAL_KNOWN = np.uint32(1 << 31)

COUNTRY_GROUPS = (
    (1, ("tunisia", "tunis", "tuinsi", "tuinsia", "tunsia", "tunisa", "tunisi", "tunisie", "tunisian", "tounsi", "tounsia", "tun", "tn", "sfax", "souse")),
    (2, ("india", "ind", "indian", "hindi", "handi", "inida", "indai", "hindustan", "bharat", "desi", "mumbai", "delhi", "bangalore", "kerala", "punjab")),
    (3, ("usa", "america", "american", "states", "murica", "nyc", "newyork", "cali", "california", "texas", "florida")),
    (4, ("uk", "britain", "british", "england", "english", "brit", "london", "scotland")),
    (5, ("algeria", "alg", "dz", "dziri", "algerie", "algerian", "algiers", "oran")),
    (6, ("moroco", "maroc", "ma", "morocan", "maghreb", "casablanca", "rabat")),
    (7, ("egypt", "egy", "eg", "egyptian", "masr", "misr", "cairo", "alexandria")),
    (8, ("france", "fr", "french", "frnace", "paris", "lyon")),
    (9, ("germany", "de", "german", "geramny", "berlin")),
    (10, ("italy", "ita", "italia", "italian", "rome", "milan")),
    (11, ("spain", "esp", "espana", "spanish", "madrid")),
    (12, ("canada", "ca", "canadian", "toronto", "ontario", "vancouver", "montreal")),
    (13, ("saudi", "ksa", "riyadh", "arabia")),
    (14, ("uae", "dubai", "emirati", "emirates", "abudhabi")),
    (15, ("turkey", "tr", "turk", "turkiye", "turkish", "istanbul", "ankara", "izmir", "antalya")),
    (16, ("pakistan", "pak", "paki", "pakistani", "lahore", "karachi")),
    (17, ("indonesia", "indo", "indonesian", "jakarta", "bali", "bandung", "surabaya")),
    (18, ("philipines", "ph", "pinoy", "pinay", "filipino", "filipina", "manila", "cebu", "davao", "quezon")),
    (19, ("brazil", "br", "brasil", "brazilian", "rio")),
)

DEFAULT_F_NAMES = (
    "emma", "sarah", "sara", "olivia", "ava", "sophia", "sofia", "isabella", "mia", "charlotte",
    "emily", "chloe", "jessica", "ashley", "amanda", "elena", "nina", "maria", "maya", "lina",
    "nour", "meriem", "mariem", "salma", "ines", "yasmine", "yasmin", "fatima", "amina", "leila",
    "layla", "hana", "hannah", "kenza", "zaynab", "zeineb", "priya", "ananya", "elif", "zeynep",
)
DEFAULT_M_NAMES = (
    "alex", "liam", "noah", "oliver", "william", "james", "lucas", "michael", "daniel", "jacob",
    "jack", "david", "john", "ryan", "kevin", "jason", "justin", "jake", "chris", "max",
    "hatem", "omar", "ahmed", "ahmad", "mohamed", "mohammed", "ali", "youssef", "yousef", "amine",
    "yassine", "mehdi", "karim", "rami", "walid", "bilel", "bilal", "hamza", "aymen", "anas",
    "zied", "rayen", "tarak", "tarek", "tariq", "rahul", "rohit", "arjun", "carlos", "mateo",
    "vihaan", "reyansh", "siddharth", "vikram", "karan",
)

NAME_BLACKLIST = frozenset((
    "sam", "same", "hi", "hey", "hello", "yo", "sup", "age", "old", "from", "country", "where",
    "how", "hru", "hyd", "hor", "wbu", "hbu", "wyd", "wud", "good", "great", "fine", "ok", "okay",
    "bored", "borad", "tired", "sad", "nothing", "nm", "nmu", "chat", "chatting", "chilling",
    "cute", "sweet", "nice", "cool", "hot", "pretty", "like", "love", "haha", "lol", "true",
    "looking", "finding", "friends", "gf", "bf", "dating", "m", "f", "male", "female", "famle",
    "boy", "girl", "gril", "man", "woman", "guy", "lady", "gender", "sex", "single", "married",
    "taken", "bot", "real", "human", "pic", "photo", "selfie", "bye", "gn", "brb", "thanks",
    "sorry", "yes", "no", "nope", "nah", "maybe", "idk", "what", "why", "who", "when", "here",
    "there", "will", "may", "can", "art", "bill", "bob", "mark", "grace", "hope", "faith", "joy",
    "summer", "autumn", "april", "june", "august", "angel", "rose", "ruby", "jade", "amber",
    "tunisia", "tunis", "india", "usa", "uk", "london", "paris", "france", "germany", "canada",
    "jordan", "chad", "georgia", "sydney", "brooklyn", "austin", "dallas", "victoria", "milan", "rome",
    "chicken", "pizza", "burger", "apple", "table", "chair",
))


def quantize_to_int8(w_fp32):
    w = w_fp32.astype(np.float32, copy=False)
    max_abs = np.float32(np.max(np.abs(w)))
    scale = np.float32(max_abs / np.float32(127.0)) if max_abs > np.float32(1e-8) else np.float32(1.0 / 127.0)
    clipped = np.clip(np.rint(np.divide(w, scale, dtype=np.float32), dtype=np.float32), np.float32(-127.0), np.float32(127.0), dtype=np.float32)
    return clipped.astype(np.int8), np.array([scale], dtype=np.float32)


def make_shared_luts():
    char_lut = np.full((256,), 32, dtype=np.int32)
    char_lut[65:91] = np.arange(97, 123, dtype=np.int32)
    char_lut[97:123] = np.arange(97, 123, dtype=np.int32)
    char_lut[39] = np.int32(39)
    rej_lut = np.ones((256,), dtype=np.bool_)
    for idx_ok in (32, 33, 39, 44, 46, 47, 63):
        rej_lut[idx_ok] = False
    rej_lut[48:58], rej_lut[65:91], rej_lut[97:123] = False, False, False
    w_primes = np.array((131, 257, 521, 1031), dtype=np.int32)
    wp_mat = np.ones((16, 4), dtype=np.int32)
    for r in range(1, 16):
        wp_mat[r] = (wp_mat[r - 1] * w_primes) & np.int32(0x7FFFFF)
    return char_lut, rej_lut, wp_mat


def collapse_str(s):
    return "".join(s[i] for i in range(len(s)) if i == 0 or s[i] != s[i - 1])


def canonical_sig(s):
    clean_w = "".join(ch if ("a" <= ch <= "z" or "0" <= ch <= "9" or ch == " " or ch == "'") else " " for ch in s.lower())
    return (tuple(w for w in collapse_str(clean_w).split() if w), "?" in s, "/" in s)


def _hash_single_word(w_str, wp_mat):
    w_vec = np.zeros((16,), dtype=np.int32)
    b = np.frombuffer(w_str[:15].encode("ascii", errors="ignore"), dtype=np.uint8)
    L = int(b.shape[0])
    w_vec[:L], w_vec[15] = b, np.int32(L)
    return (w_vec @ wp_mat) & np.int32(BLOOM_SIZE - 1)


def build_entity_and_super_bloom(sample_map, wp_mat, leaf_step_map):
    entity_lut_a = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    entity_lut_b = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    super_bloom = np.zeros((BLOOM_SIZE,), dtype=np.uint32)

    f_names, m_names = set(DEFAULT_F_NAMES), set(DEFAULT_M_NAMES)
    for fn, tgt, other in (("names_female_catalog.txt", f_names, set()), ("names_male_catalog.txt", m_names, f_names)):
        if os.path.exists(fn):
            with open(fn, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    w = line.strip().lower()
                    if w and w.isalpha() and w not in NAME_BLACKLIST and w not in other:
                        tgt.add(w)

    s9_bit = np.uint32(1 << 9) | BIT_GLOBAL_KNOWN
    for code, nset in ((21, f_names), (22, m_names)):
        for w in nset:
            h = _hash_single_word(collapse_str(w), wp_mat)
            entity_lut_a[h[0]], entity_lut_b[h[1]] = np.uint8(code), np.uint8(code)
            super_bloom[h[:3]] |= s9_bit

    s2_bit = np.uint32(1 << 2) | BIT_GLOBAL_KNOWN
    if os.path.exists("countries.txt"):
        with open("countries.txt", "r", encoding="ascii", errors="ignore") as fc:
            for line in fc:
                w = line.strip().lower()
                if w and " " not in w and "?" not in w and len(w) >= 3 and w.isalpha() and w not in NAME_BLACKLIST:
                    h = _hash_single_word(collapse_str(w), wp_mat)
                    if entity_lut_a[h[0]] == 0 or entity_lut_a[h[0]] >= 21:
                        entity_lut_a[h[0]], entity_lut_b[h[1]] = np.uint8(20), np.uint8(20)
                    super_bloom[h[:3]] |= s2_bit
    for cid, aliases in COUNTRY_GROUPS:
        for al in aliases:
            h = _hash_single_word(collapse_str(al), wp_mat)
            entity_lut_a[h[0]], entity_lut_b[h[1]] = np.uint8(cid), np.uint8(cid)
            super_bloom[h[:3]] |= s2_bit

    for text, (leaf, _w) in sample_map.items():
        if leaf == 0 or leaf == 75 or leaf == 96:
            continue
        step_b = int(leaf_step_map.get(leaf, 15))
        bit_val = np.uint32(1 << min(step_b, 29)) | BIT_GLOBAL_KNOWN
        clean_s = "".join(ch if ("a" <= ch <= "z" or ch == " " or ch == "'") else " " for ch in text.lower())
        for w in collapse_str(clean_s).split():
            if w:
                super_bloom[_hash_single_word(w, wp_mat)[:3]] |= bit_val

    for wn in ("name", "names", "name's", "call", "cal"):
        super_bloom[_hash_single_word(collapse_str(wn), wp_mat)[:3]] |= BIT_WILDCARD_NAME

    print(f"[SUPER-BLOOM & ENTITY LUT] Indexed {len(f_names) + len(m_names)} names + countries into unified O(1) tables!", flush=True)
    return super_bloom, entity_lut_a, entity_lut_b


def encode_message_exact(raw_bytes, char_lut, rej_lut, wp_mat, super_bloom, entity_lut_a, entity_lut_b):
    w_ids, bg_ids, s_vec = np.zeros((6,), dtype=np.int32), np.zeros((7,), dtype=np.int32), np.zeros((D_STRUCT,), dtype=np.float32)
    b_full = np.frombuffer(raw_bytes[:64], dtype=np.uint8)
    non_ws = np.flatnonzero(b_full > np.uint8(32))
    if non_ws.shape[0] == 0:
        return 0, 0, 0, w_ids, bg_ids, s_vec
    first_idx, last_idx = int(non_ws[0]), int(non_ws[-1]) + 1
    n_tok = last_idx - first_idx
    if n_tok > MAX_LEN:
        return 0, 0, 0, w_ids, bg_ids, s_vec
    if n_tok <= 6 and bytes(b_full[first_idx:last_idx]).lower() in (b"next", b"reset", b"new", b"skip", b"/start"):
        return 75, 0, 0, w_ids, bg_ids, s_vec

    raw_u8 = np.full((MAX_LEN + 4,), 32, dtype=np.int32)
    raw_u8[2 : 2 + n_tok] = b_full[first_idx:last_idx]
    body = raw_u8[2 : 2 + n_tok]
    if bool(np.any(rej_lut[body])):
        return 0, 0, 0, w_ids, bg_ids, s_vec
    if n_tok >= 5:
        for zi in np.flatnonzero(body == 48):
            z = int(zi)
            if z > 0 and z + 1 < n_tok and (int(body[z - 1]) | 32) == 104 and (int(body[z + 1]) | 32) == 114:
                body[z] = np.int32(111)

    has_q, has_s = bool(np.any(body == 63)), bool(np.any(body == 47))
    dig_idx = np.flatnonzero((body >= 48) & (body <= 57))
    n_dig, age_val = int(dig_idx.shape[0]), 0
    if n_dig > 0:
        if n_dig > 2 or int(dig_idx[-1] - dig_idx[0]) != (n_dig - 1):
            return 0, 0, 0, w_ids, bg_ids, s_vec
        d0 = int(dig_idx[0])
        if n_dig == 1 and d0 > 0 and int(body[d0 - 1]) > 39:
            return 0, 0, 0, w_ids, bg_ids, s_vec
        age_val = (int(body[d0]) - 48) if n_dig == 1 else ((int(body[d0]) - 48) * 10 + (int(body[d0 + 1]) - 48))
        if age_val < 18:
            return (7, age_val, 0, w_ids, bg_ids, s_vec) if age_val >= 1 else (0, 0, 0, w_ids, bg_ids, s_vec)
        if age_val > 80:
            return 0, 0, 0, w_ids, bg_ids, s_vec

    np.take(char_lut, raw_u8, out=raw_u8)
    kept = raw_u8[2 : 2 + n_tok][raw_u8[2 : 2 + n_tok] != raw_u8[1 : 1 + n_tok]]
    if kept.shape[0] > 0 and int(kept[-1]) == 32:
        kept = kept[:-1]
    if kept.shape[0] > 0 and int(kept[0]) == 32:
        kept = kept[1:]
    k_len = int(kept.shape[0])
    if k_len == 0:
        if age_val >= 18:
            return 3, age_val, 0, w_ids, bg_ids, s_vec
        if has_q and bool(np.all((b_full[first_idx:last_idx] == 63) | (b_full[first_idx:last_idx] == 32))):
            return 96, 0, 0, w_ids, bg_ids, s_vec
        return 0, 0, 0, w_ids, bg_ids, s_vec

    sp = np.flatnonzero(kept == np.int32(32))
    if int(sp.shape[0]) > 5:
        return 0, 0, 0, w_ids, bg_ids, s_vec
    n_words = int(sp.shape[0]) + 1

    words_mat = np.zeros((6, 16), dtype=np.int32)
    starts, ends = (0,) + tuple(int(v) + 1 for v in sp), tuple(int(v) for v in sp) + (k_len,)
    for w_i in range(n_words):
        Lw = min(ends[w_i] - starts[w_i], 15)
        words_mat[w_i, :Lw], words_mat[w_i, 15] = kept[starts[w_i] : starts[w_i] + Lw], np.int32(Lw)

    w_hashes = words_mat @ wp_mat
    w_b_idx = w_hashes[:, :3] & np.int32(BLOOM_SIZE - 1)
    sb_word = np.bitwise_and.reduce(super_bloom[w_b_idx[:n_words]], axis=1)
    all_known = bool(np.all((sb_word & BIT_GLOBAL_KNOWN) != 0))
    is_wc_name = False
    if not all_known and n_words >= 3 and bool(np.any((sb_word & BIT_WILDCARD_NAME) != 0)):
        unk = np.flatnonzero((sb_word & BIT_GLOBAL_KNOWN) == 0)
        if unk.shape[0] == 1 and int(unk[0]) >= 2 and int(words_mat[int(unk[0]), 15]) >= 3:
            is_wc_name = True
    if not all_known and not is_wc_name:
        return 0, 0, 0, w_ids, bg_ids, s_vec

    active_mask = int(np.bitwise_and.reduce(sb_word)) if all_known else (1 << 9)
    ea, eb = entity_lut_a[w_b_idx[:n_words, 0]], entity_lut_b[w_b_idx[:n_words, 1]]
    ec = np.where((ea > 0) & (ea == eb), ea, np.uint8(0))

    ctry_id, has_nf, has_nm = 0, False, False
    for i in range(n_words):
        code = int(ec[i])
        if 1 <= code <= 20:
            ctry_id = code if ctry_id == 0 else ctry_id
            w_ids[i] = np.int32(TOK_COUNTRY)
        elif code == 21:
            has_nf, w_ids[i] = True, np.int32(TOK_NAME_F)
        elif code == 22 or (is_wc_name and (int(sb_word[i]) & int(BIT_GLOBAL_KNOWN)) == 0):
            has_nm, w_ids[i] = True, np.int32(TOK_NAME_M)
        else:
            w_ids[i] = np.int32(((int(w_hashes[i, 0]) & 0x7FFFFFFF) % WORD_MOD) + 1)

    bg_ids[0] = np.int32((((4097 * 131 + int(w_ids[0])) & 0x7FFFFFFF) % BIGRAM_MOD) + 1)
    for k in range(n_words - 1):
        bg_ids[k + 1] = np.int32((((int(w_ids[k]) * 131 + int(w_ids[k + 1])) & 0x7FFFFFFF) % BIGRAM_MOD) + 1)
    eos_tok = 4101 if has_q else (4100 if has_s else 4099)
    bg_ids[n_words] = np.int32((((int(w_ids[n_words - 1]) * 131 + eos_tok) & 0x7FFFFFFF) % BIGRAM_MOD) + 1)

    s_vec[0], s_vec[1] = (np.float32(1.0) if has_q else np.float32(0.0)), (np.float32(1.0) if has_s else np.float32(0.0))
    s_vec[2], s_vec[4] = (np.float32(1.0) if age_val >= 18 else np.float32(0.0)), (np.float32(1.0) if ctry_id > 0 else np.float32(0.0))
    s_vec[5], s_vec[6] = (np.float32(1.0) if has_nf else np.float32(0.0)), (np.float32(1.0) if has_nm else np.float32(0.0))
    s_vec[8], s_vec[9] = (np.float32(1.0) if n_words == 1 else np.float32(0.0)), np.float32(n_words) / np.float32(6.0)
    s_vec[10] = np.float32(1.0) if is_wc_name else np.float32(0.0)
    s_vec[11], s_vec[12] = (np.float32(1.0) if (active_mask & (1 << 1)) != 0 else np.float32(0.0)), (np.float32(1.0) if (active_mask & (1 << 2)) != 0 else np.float32(0.0))
    s_vec[13], s_vec[14] = (np.float32(1.0) if (active_mask & (1 << 4)) != 0 else np.float32(0.0)), (np.float32(1.0) if (active_mask & (1 << 8)) != 0 else np.float32(0.0))
    s_vec[15] = np.float32(1.0) if (age_val >= 18 and ctry_id > 0) else np.float32(0.0)
    return -1, age_val, ctry_id, w_ids, bg_ids, s_vec


def numpy_forward_single(raw_bytes, char_lut, rej_lut, wp_mat, super_bloom, entity_lut_a, entity_lut_b, np_w):
    fast_lf, _, _, w_ids, bg_ids, s_vec = encode_message_exact(
        raw_bytes, char_lut, rej_lut, wp_mat, super_bloom, entity_lut_a, entity_lut_b
    )
    if fast_lf != -1:
        return fast_lf
    w_mask = (w_ids != 0).astype(np.float32)[:, None]
    h_word = np.sum((np_w["emb_word"][w_ids] + np_w["pos_emb"]) * w_mask, axis=0, keepdims=True) / np.float32(max(int(np.sum(w_ids != 0)), 1))
    bg_mask = (bg_ids != 0).astype(np.float32)[:, None]
    h_bg = np.sum(np_w["emb_bg"][bg_ids] * bg_mask, axis=0, keepdims=True) / np.float32(max(int(np.sum(bg_ids != 0)), 1))
    x_in = np.concatenate([h_word, h_bg], axis=1) + (s_vec[None, :] @ np_w["w_struct"])
    h1 = np.maximum(x_in @ np_w["w1"] + np_w["b1"], np.float32(0.0))
    h2 = np.maximum(h1 @ np_w["w2"] + np_w["b2"], np.float32(0.0)) + x_in
    logits = h2 @ np_w["w_out"] + np_w["b_out"]
    best_lf = int(np.argmax(logits[0]))
    return 0 if (best_lf != 0 and np.float32(logits[0, best_lf] - logits[0, 0]) < np.float32(0.25)) else best_lf