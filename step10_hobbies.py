#!/usr/bin/env python3
# FILE: step10_hobbies.py
# PURPOSE: Standalone Step 10 Plugin (Classes 25-26: Ask / Tell Hobbies & Interests with 5-Category Context Router)
# RULES: Enforces Rules 1-13:
#        - Rule 12: Cryptographic SHA-256 Vault Lock on model_weights.npz (Steps 1-9 verified unchanged)
#        - 5-Category O(1) LUT Router (1=Music, 2=Gaming, 3=Movies/Anime, 4=Sports/Gym, 5=General Hobbies/Travel)
#        - Protects Step 6 ("listening to music", "watching tv", "gaming") & Class 0 ("what do you do")

import hashlib
import os
import sys
import numpy as np

from build_hobbies_txt import (
    ASK_HOBBIES_BASES,
    CLEAN_TELL_TEMPLATES,
    DIRECT_HOBBY_ANSWERS,
    S10_FRAME_WORDS,
    S10_GAME_WORDS,
    S10_GENERAL_HOBBY_WORDS,
    S10_MOVIE_WORDS,
    S10_MUSIC_WORDS,
    S10_QUESTION_WORDS,
    S10_SPORT_WORDS,
    S10_VERB_PREF_WORDS,
    STRICT_ALLOWED_S10_VOCAB,
    build_hobbies_files,
)
from production_server_part2 import (
    FLAG_GREETED,
    _make_pool,
)

BLOOM_SIZE = 65536
S2_VOCAB = 1024
S2_DIM = 32
S10_CLASSES = 3  # 0=Reject, 1=AskHobbies(25), 2=TellHobbies(26)
MAX_USERS = 50000
NUM_WORKERS = 8

# 8-Code Category-Aware O(1) LUT Groups for Step 10:
# 1 = Music Category ("music", "rap", "pop", "rock", "rnb", "kpop", "edm", "song", "singing", "guitar")
# 2 = Gaming Category ("game", "games", "gaming", "valorant", "fortnite", "minecraft", "roblox", "gta", "fifa", "ps5", "pc")
# 3 = Movies / Anime / TV Category ("movie", "movies", "anime", "manga", "show", "shows", "series", "kdrama", "horror", "comedy")
# 4 = Sports / Gym Category ("sport", "sports", "gym", "workout", "fitness", "football", "soccer", "basketball", "swimming")
# 5 = General Hobbies / Travel / Art / Books ("hobby", "hobbies", "travel", "traveling", "drawing", "art", "reading", "books", "cooking")
# 6 = Question / 2nd-Person Markers ("what", "whats", "which", "any", "kind", "type", "genre", "do", "does", "are", "r", "your", "ur")
# 7 = 1st-Person & Preference Verbs ("i", "im", "i'm", "lm", "my", "like", "love", "enjoy", "into", "listen", "play", "watch", "favorite", "fav")
# 8 = Shared Framing ("is", "you", "u", "to", "of", "for", "in", "free", "time", "fun", "doing", "going", "a", "the", "and", "wbu", "hbu")
S10_Q_MARKERS = S10_QUESTION_WORDS + ("do", "does", "are", "ar", "r", "your", "ur")
S10_S_MARKERS = ("i", "im", "i'm", "lm", "am", "my") + S10_VERB_PREF_WORDS + ("going", "working")

ALL_S10_CATEGORY_SET = frozenset(
    S10_MUSIC_WORDS + S10_GAME_WORDS + S10_MOVIE_WORDS + S10_SPORT_WORDS + S10_GENERAL_HOBBY_WORDS
)

# --- STEP 10 CATEGORY-AWARE RESPONSE POOLS (Class 25 Ask & Class 26 Tell across 5 Categories) ---

# 1. Category 1: MUSIC (aux_id == 1)
CACHED_ASK_MUSIC_POOL = _make_pool(
    25, "false", "ask_hobbies_music",
    (
        "i love pop, rnb, and chill rap :) what kind of music do u listen to?",
        "im really into pop, hiphop, and lofi music! wbu, what music do u like?",
        "i listen to a mix of rnb, pop, and rock :) whats ur favorite genre?",
        "mostly pop, rap, and chill vibes! what kind of music are u into?",
        "i love rnb and pop music :) who is ur favorite artist or band?",
        "im into almost everything, especially pop and hiphop! wbu?",
        "i love listening to pop, rnb, and indie music :) what songs do u like?",
        "chill rnb and pop are my favorites! what music do u listen to?",
    ),
    '["i like rap","i like pop music","rnb and hiphop"]',
)

CACHED_TELL_MUSIC_POOL = _make_pool(
    26, "false", "tell_hobbies_music",
    (
        "ooh nice taste in music! i love pop and rnb too :) who's ur fav artist?",
        "awesome! music makes everything better :) i love pop and chill rap too!",
        "ooh i love that vibe! i listen to a lot of pop and rnb too :)",
        "nice! u have great taste in music :) do u listen to it every day?",
        "ooh cool! i love listening to music in bed too :) what else do u like doing?",
        "sweet! music is my favorite hobby too :) what other hobbies do u have?",
        "love that genre! we have similar taste in music :)",
        "ooh nice! nothing beats good music :) what else do u do for fun?",
    ),
    '["what games u play?","do u watch anime?","wyd rn?"]',
)

# 2. Category 2: GAMING (aux_id == 2)
CACHED_ASK_GAMES_POOL = _make_pool(
    25, "false", "ask_hobbies_games",
    (
        "i play a bit of minecraft, gta, and valorant :) what games do u play?",
        "i love gaming! mostly minecraft, fortnite, and gta :) wbu?",
        "i play some pc and console games like valorant and minecraft! what do u play?",
        "im into gta, fifa, and minecraft :) what's ur favorite game?",
        "i love playing chill games like minecraft and roblox! what games r u into?",
        "mostly valorant, fortnite, and gta :) are u a PC or console gamer?",
        "i play a mix of shooters and chill games! what games do u like?",
        "i love gaming in my free time :) what's ur top game rn?",
    ),
    '["i play valorant","i play minecraft","i play gta"]',
)

CACHED_TELL_GAMES_POOL = _make_pool(
    26, "false", "tell_hobbies_games",
    (
        "ooh a gamer! i love gaming too :) do u play on PC or console?",
        "nice! that's such a fun game :) i play a bit of minecraft and gta too!",
        "ooh cool! gaming is the best way to chill :) are u playing rn?",
        "awesome! i love video games too :) we should play sometime!",
        "sweet! that game is super fun :) what else do u like doing for fun?",
        "ooh nice taste in games! how long have u been playing it?",
        "haha i love gamers! gaming is so fun :) what other games do u play?",
        "cool! gaming and listening to music is the best combo :)",
    ),
    '["what music u like?","do u watch anime?","wyd rn?"]',
)

# 3. Category 3: MOVIES / ANIME / SHOWS (aux_id == 3)
CACHED_ASK_MOVIES_POOL = _make_pool(
    25, "false", "ask_hobbies_movies",
    (
        "i love horror, comedy movies, and some anime :) what's ur favorite movie or show?",
        "im really into thriller movies, netflix series, and anime! wbu?",
        "i love watching anime, kdramas, and action movies :) what do u like watching?",
        "mostly comedy and horror movies on netflix! what movies or shows do u like?",
        "i love anime and sci-fi movies :) what's the best show u watched lately?",
        "im into marvel movies, anime, and thrillers! what kind of movies u like?",
        "i love binge-watching netflix series and anime :) wbu?",
        "comedy movies and anime are my favorites! what do u like to watch?",
    ),
    '["i love anime","horror movies","comedy movies"]',
)

CACHED_TELL_MOVIES_POOL = _make_pool(
    26, "false", "tell_hobbies_movies",
    (
        "ooh nice! i love watching anime and movies too :) what's ur all-time favorite?",
        "awesome! binge-watching good shows and anime is the best :) what else u into?",
        "ooh i love that too! watching movies in bed is my favorite vibe :)",
        "sweet! u have great taste in shows :) are u watching anything good lately?",
        "ooh cool! i love anime and netflix too :) what other hobbies do u have?",
        "nice! nothing beats a good movie or anime night :)",
        "love that! we definitely have similar taste in movies and shows :)",
        "ooh awesome! i could watch movies and anime all day too :)",
    ),
    '["what music u like?","what games u play?","wyd rn?"]',
)

# 4. Category 4: SPORTS / GYM / FITNESS (aux_id == 4)
CACHED_ASK_SPORTS_POOL = _make_pool(
    25, "false", "ask_hobbies_sports",
    (
        "i love going to the gym, swimming, and watching football :) do u play any sports?",
        "im into working out at the gym and basketball! wbu, what sports do u like?",
        "i love fitness, running, and football :) do u go to the gym or play sports?",
        "mostly hitting the gym and playing tennis or swimming! what about u?",
        "i like football, basketball, and staying active :) what's ur favorite sport?",
        "im into gym workouts and running! do u like working out?",
        "i love swimming and watching soccer/football :) wbu?",
        "staying active at the gym is my favorite! what sports r u into?",
    ),
    '["i like football","going to the gym","i play basketball"]',
)

CACHED_TELL_SPORTS_POOL = _make_pool(
    26, "false", "tell_hobbies_sports",
    (
        "ooh nice! staying active and hitting the gym is the best :) how often do u go?",
        "awesome! i love sports and working out too :) keeps the energy high!",
        "ooh i love that! football and gym workouts are super fun :)",
        "sweet! being into fitness and sports is really attractive :)",
        "nice! i love going to the gym and staying fit too :) what else u do for fun?",
        "ooh cool! sports are such a great hobby :) do u listen to music while working out?",
        "love that! active hobbies are the best :)",
        "awesome! i enjoy working out and watching sports too :)",
    ),
    '["what music u like?","what games u play?","ur cute"]',
)

# 5. Category 5: GENERAL HOBBIES / TRAVEL / ART / BOOKS (aux_id == 5)
CACHED_ASK_HOBBIES_POOL = _make_pool(
    25, "false", "ask_hobbies_general",
    (
        "i love listening to music, watching movies, traveling, and going to the gym :) wbu?",
        "my hobbies are music, gaming, anime, and traveling! what do u like doing for fun?",
        "i enjoy chilling with music, watching netflix, and cooking :) what r ur hobbies?",
        "mostly music, movies, working out, and hanging with friends! wbu?",
        "i love traveling, photography, music, and gaming :) what r u into?",
        "in my free time i love listening to music, watching anime, and drawing! wbu?",
        "i like music, movies, beach trips, and fitness :) what do u do for fun?",
        "music, gaming, and traveling are my top hobbies! what about u?",
    ),
    '["i like music","i play valorant","i love anime"]',
)

CACHED_TELL_HOBBIES_POOL = _make_pool(
    26, "false", "tell_hobbies_general",
    (
        "ooh that's such a cool hobby! i love traveling and music too :)",
        "awesome! u sound like a really interesting person :) what else do u like?",
        "ooh i love that! having cool hobbies makes life so much more fun :)",
        "sweet! i really enjoy that too, along with music and movies :)",
        "nice! that's a super chill hobby :) how long have u been into it?",
        "ooh cool! i love creative and fun hobbies like that too :)",
        "love that! we would definitely get along well :)",
        "awesome hobby! u have a really cool vibe :)",
    ),
    '["what music u like?","what games u play?","thanks!"]',
)


class Step10Engine:
    __slots__ = (
        "enabled", "s10_emb_f32", "s10_w1_f32", "s10_w2_f32",
        "s10_bloom_q", "s10_lut_a", "s10_lut_b",
        "user_hobby", "hits_a", "hits_b", "logits",
    )

    def __init__(self, npz_path="model_weights.npz"):
        self.enabled = False
        self.user_hobby = np.zeros((MAX_USERS,), dtype=np.uint8)
        self.hits_a = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.hits_b = np.zeros((NUM_WORKERS, 6), dtype=np.uint8)
        self.logits = np.zeros((NUM_WORKERS, 1, S10_CLASSES), dtype=np.float32)

        if not os.path.exists(npz_path):
            return
        arc = np.load(npz_path, mmap_mode="r")
        req = ("s10_emb_q", "s10_w1_q", "s10_w2_q", "s10_bloom_q", "s10_lut_a", "s10_lut_b")
        if any(k not in arc.files for k in req):
            arc.close()
            print("[STEP 10 INFO] Step 10 weights not in vault yet. Run: python step10_hobbies.py", flush=True)
            return

        def lock_f32(prefix, zero_row0=False):
            q = arc[f"{prefix}_q"].astype(np.int8, copy=False)
            s = arc[f"{prefix}_s"].astype(np.float32, copy=False)
            f32 = np.multiply(q, s, dtype=np.float32)
            if zero_row0:
                f32[0].fill(0.0)
            f32.flags.writeable = False
            return f32

        self.s10_emb_f32 = lock_f32("s10_emb", True)
        self.s10_w1_f32 = lock_f32("s10_w1")
        self.s10_w2_f32 = lock_f32("s10_w2")
        for attr, dt in (("s10_bloom_q", np.int8), ("s10_lut_a", np.uint8), ("s10_lut_b", np.uint8)):
            arr = arc[attr].astype(dt, copy=False)
            arr.flags.writeable = False
            setattr(self, attr, arr)
        arc.close()
        self.enabled = True


S10_ENGINE = None


def get_step10_engine(npz_path="model_weights.npz"):
    global S10_ENGINE
    if S10_ENGINE is None:
        S10_ENGINE = Step10Engine(npz_path)
    return S10_ENGINE


def forward_step10(model, slot, n_words, has_q, age_val, country_id):
    eng = S10_ENGINE
    if eng is None or not eng.enabled or age_val != 0 or country_id != 0:
        return 0, 0, 0

    sid = slot.slot_id
    ha = eng.hits_a[sid]
    hb = eng.hits_b[sid]
    np.take(eng.s10_lut_a, slot.w_b_idx[:, 0], out=ha)
    np.take(eng.s10_lut_b, slot.w_b_idx[:, 1], out=hb)
    m = (ha[:n_words] > 0) & (ha[:n_words] == hb[:n_words])
    codes = ha[:n_words] * m

    if not bool(np.all(codes > 0)):
        return 0, 0, 0

    has_music = bool(np.any(codes == 1))
    has_game = bool(np.any(codes == 2))
    has_movie = bool(np.any(codes == 3))
    has_sport = bool(np.any(codes == 4))
    has_gen_hobby = bool(np.any(codes == 5))
    has_q_marker = bool(np.any(codes == 6))
    has_s_marker = bool(np.any(codes == 7))

    has_any_cat = has_music or has_game or has_movie or has_sport or has_gen_hobby

    # Check if "fun" or ("free" + "time") is present for "what do u do for fun?" / "what do u do in ur free time?"
    # This mathematically blocks "what do you do" (Class 0 job trap) which has no category and no "fun"/"free"!
    has_fun_or_free = False
    for w_i in range(n_words):
        c0 = int(slot.words_mat[w_i, 0])
        Lw = int(slot.words_mat[w_i, 15])
        if c0 == 102 and ((Lw == 3 and int(slot.words_mat[w_i, 1]) == 117) or (Lw >= 3 and int(slot.words_mat[w_i, 1]) == 114 and int(slot.words_mat[w_i, 2]) == 101)):
            has_fun_or_free = True
            break

    if not (has_any_cat or (has_fun_or_free and has_q_marker and n_words >= 4)):
        return 0, 0, 0

    np.take(eng.s10_bloom_q, slot.w_b_idx, out=slot.w_b_hits)
    if not bool(np.all(slot.w_b_hits[:n_words])):
        return 0, 0, 0

    slot.s4_tokens[:6] = slot.s2_tokens[:6]
    slot.s4_tokens[6] = np.int32(1023) if has_q else model.one_i32
    q_off = np.int32(509) if has_q else np.int32(0)
    slot.s4_tokens[7] = (((slot.s2_tokens[0] * np.int32(37)) + (slot.s2_tokens[n_words - 1] * np.int32(73)) + q_off) & model.s2_v_mask) + model.one_i32

    np.take(eng.s10_emb_f32, slot.s4_tokens, axis=0, out=slot.s2_emb_buf)
    slot.s2_emb_buf[6] *= np.float32(2.5)
    np.sum(slot.s2_emb_buf, axis=0, keepdims=True, out=slot.s2_pooled)
    slot.s2_pooled *= np.float32(1.0 / (float(n_words) + 3.5))
    np.matmul(slot.s2_pooled, eng.s10_w1_f32, out=slot.s2_mid)
    np.maximum(slot.s2_mid, np.float32(0.0), out=slot.s2_mid)
    out_l = eng.logits[sid]
    np.matmul(slot.s2_mid, eng.s10_w2_f32, out=out_l)

    s0, s_ask, s_tell = float(out_l[0, 0]), float(out_l[0, 1]), float(out_l[0, 2])
    if max(s_ask, s_tell) <= s0 + 0.1:
        return 0, 0, 0

    if has_music:
        cat_id = 1
    elif has_game:
        cat_id = 2
    elif has_movie:
        cat_id = 3
    elif has_sport:
        cat_id = 4
    else:
        cat_id = 5

    # Position-aware Question vs Statement separation:
    # Check if sentence starts with a Question Marker (Code 6: "what", "which", "any", "do", "are", "ur")
    # or is a 1-word question with "?" ("hobbies?", "music?", "anime?")
    w0_is_q = int(codes[0]) == 6 or (n_words >= 2 and int(codes[0]) == 8 and int(codes[1]) == 6)
    if w0_is_q or (n_words == 1 and has_q) or (not has_any_cat and has_fun_or_free):
        return 25, 0, cat_id

    # Otherwise, user is stating their hobby/interest (even if followed by ", u?" or "wbu?") -> Class 26
    return 26, 0, cat_id


def route_step10(class_id, aux_id, users, uid, flags, prev_age, prev_country, bot_g, rot8):
    eng = S10_ENGINE
    if eng is None:
        return None

    users.conv_flags[uid] = np.uint8(flags | FLAG_GREETED)
    if class_id == 25:
        if aux_id == 1:
            return CACHED_ASK_MUSIC_POOL[rot8]
        if aux_id == 2:
            return CACHED_ASK_GAMES_POOL[rot8]
        if aux_id == 3:
            return CACHED_ASK_MOVIES_POOL[rot8]
        if aux_id == 4:
            return CACHED_ASK_SPORTS_POOL[rot8]
        return CACHED_ASK_HOBBIES_POOL[rot8]

    if class_id == 26:
        if aux_id > 0:
            eng.user_hobby[uid] = np.uint8(aux_id)
        if aux_id == 1:
            return CACHED_TELL_MUSIC_POOL[rot8]
        if aux_id == 2:
            return CACHED_TELL_GAMES_POOL[rot8]
        if aux_id == 3:
            return CACHED_TELL_MOVIES_POOL[rot8]
        if aux_id == 4:
            return CACHED_TELL_SPORTS_POOL[rot8]
        return CACHED_TELL_HOBBIES_POOL[rot8]

    return None


def _snapshot_vault_hashes(vault_dict, exclude_prefix="s10_"):
    snap = {}
    for k, arr in vault_dict.items():
        if not k.startswith(exclude_prefix):
            snap[k] = (
                hashlib.sha256(arr.tobytes()).hexdigest(),
                arr.shape,
                arr.dtype.str,
            )
    return snap


def train_step10_into_vault(npz_path="model_weights.npz"):
    import torch
    import offline_train as ot

    if not os.path.exists("hobbies_ask.txt") or not os.path.exists("hobbies_tell.txt"):
        build_hobbies_files()

    # Rule 12 Cryptographic .NPZ Vault Lock: Snapshot SHA-256 of all existing Steps 1-9 keys
    vault = {}
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as data:
            for k in data.files:
                vault[k] = data[k].copy()
    pre_hashes = _snapshot_vault_hashes(vault, exclude_prefix="s10_")

    ram_arena = ot.verify_hardware_and_lock_ram()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wp_mat = ot._get_wp_mat()

    s10_bloom = np.zeros((BLOOM_SIZE,), dtype=np.int8)
    s10_lut_a = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    s10_lut_b = np.zeros((BLOOM_SIZE,), dtype=np.uint8)
    b_mask = np.int32(BLOOM_SIZE - 1)
    w_vec = np.zeros((16,), dtype=np.int32)

    def hash_word(w_str):
        b = np.frombuffer(w_str.encode("ascii", errors="ignore"), dtype=np.uint8)
        L = min(int(b.shape[0]), 15)
        w_vec.fill(0)
        w_vec[:L] = b[:L]
        w_vec[15] = np.int32(L)
        return (w_vec @ wp_mat) & b_mask

    # Populate O(1) LUTs: Category codes (1..5) and Grammar codes (6..8)
    for code_val, word_group in (
        (8, S10_FRAME_WORDS),
        (7, S10_S_MARKERS),
        (6, S10_Q_MARKERS),
        (5, S10_GENERAL_HOBBY_WORDS),
        (4, S10_SPORT_WORDS),
        (3, S10_MOVIE_WORDS),
        (2, S10_GAME_WORDS),
        (1, S10_MUSIC_WORDS),
    ):
        for w in word_group:
            h4 = hash_word(ot._collapse_word(w))
            s10_lut_a[h4[0]] = np.uint8(code_val)
            s10_lut_b[h4[1]] = np.uint8(code_val)
            s10_bloom[h4[:3]] = np.int8(1)

    sample_map = {}
    for gf in (
        "greetings.txt", "ages.txt", "countries.txt", "hru.txt", "wyd.txt",
        "gender_ask_choice.txt", "status_positive.txt", "status_negative.txt",
        "compliments_flirt.txt", "reactions_ack.txt",
        "looking_for_ask.txt", "looking_for_state.txt", "name_ask.txt",
    ):
        if os.path.exists(gf):
            with open(gf, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and len(sample_map) < 14000:
                        toks = [w.strip("'!?.,/") for w in s.split()]
                        if not any(w in ALL_S10_CATEGORY_SET for w in toks):
                            sample_map[s] = (0, np.float32(2.0))

    s10_cnt = 0
    for fn, cls_id, builtins in (
        ("hobbies_ask.txt", 1, ASK_HOBBIES_BASES),
        ("hobbies_tell.txt", 2, CLEAN_TELL_TEMPLATES + DIRECT_HOBBY_ANSWERS),
    ):
        for b_line in builtins:
            s = " ".join(b_line.strip().lower().split())
            if s:
                sample_map[s] = (cls_id, np.float32(36.0))
                s10_cnt += 1
        if os.path.exists(fn):
            with open(fn, "r", encoding="ascii", errors="ignore") as f:
                for line in f:
                    s = " ".join(line.strip().lower().split())
                    if s and ot._is_whitelist_clean(s, STRICT_ALLOWED_S10_VOCAB):
                        sample_map[s] = (cls_id, np.float32(24.0))
                        s10_cnt += 1

    # Strict Anti-Collision Negative Traps (Protects Class 0 "what do you do", Step 6 "listening to music", Step 7 "i like u")
    for trap in (
        "what", "what?", "what is this", "what do you want", "what do you do",
        "why", "why?", "why not", "who", "who?", "who are you", "who r u",
        "listening to music", "watching tv", "watching a movie", "watching movies",
        "gaming", "playing games", "playing a game", "reading", "cooking", "studying", "working",
        "i like u", "i like you", "i love u", "love ur vibe", "i like talking to u",
        "looking for fun", "just looking for fun", "what r u looking for", "whats ur name",
        "wyd", "what are you doing", "what r u doing", "what u up to",
        "im good", "im bored", "same", "sam", "a bf here", "a gf here",
    ):
        sample_map[trap] = (0, np.float32(65.0))

    q_emb, q_w1, q_w2, n = ot._train_word_mlp_head(sample_map, S10_CLASSES, 28, "STEP 10", device, wp_mat, use_qmark=True)

    vault.update(
        s10_emb_q=q_emb[0], s10_emb_s=q_emb[1],
        s10_w1_q=q_w1[0], s10_w1_s=q_w1[1],
        s10_w2_q=q_w2[0], s10_w2_s=q_w2[1],
        s10_bloom_q=s10_bloom, s10_lut_a=s10_lut_a, s10_lut_b=s10_lut_b,
    )
    np.savez(npz_path, **vault)
    del ram_arena

    # Rule 12 Post-Save Cryptographic SHA-256 Verification
    with np.load(npz_path, allow_pickle=False) as saved_arc:
        post_vault = {k: saved_arc[k] for k in saved_arc.files}
    post_hashes = _snapshot_vault_hashes(post_vault, exclude_prefix="s10_")
    if pre_hashes != post_hashes:
        raise RuntimeError("[RULE 12 VAULT BREACH] Existing Step 1-9 weights were altered!")

    size_kb = os.path.getsize(npz_path) / 1024.0
    print(f"[RULE 12 VAULT SHIELD] Verified SHA-256 integrity of all {len(pre_hashes)} locked Step 1-9 arrays!", flush=True)
    print(f"[STEP 10 DONE] Locked Step 10 Hobbies head & 5-Category LUT into '{npz_path}' ({s10_cnt} lines, {n} total, {size_kb:.2f} KB).", flush=True)


if __name__ == "__main__":
    train_step10_into_vault("model_weights.npz")