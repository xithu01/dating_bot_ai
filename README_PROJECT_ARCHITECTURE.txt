================================================================================
PROJECT: ULTRA-LOW-RAM 1-ON-1 RANDOM DATING CHAT AI (RULE 14+15 PURE NUMPY VM)
TARGET: 50,000 Concurrent Users | ~4.84 MB Total RAM | ~0.42 ms/req CPU Latency
CURRENT STATE: STEPS 1 TO 17 LOCKED (96 LEAVES / 56 INTENT CLASSES: 0-54 + 99)
================================================================================

1. CORE ARCHITECTURAL OVERVIEW (RULE 14 & RULE 15)
--------------------------------------------------------------------------------
We deploy ONLY TWO FILES to the production server:
  1) `production_server.py` (295 lines / ~11.9 KB AST — PyArmor-ready, 0 plugins, 0 `if class_id == X` chains)
  2) `model_weights.npz` (~1.55 MB — contains all quantized int8 neural weights, Super-Bloom, 4,059+ names & countries Entity LUT, 4D Finite-State Routing Table, Multi-Turn Ask-Back Redirect Table, and all pre-rendered HTTP/JSON byte responses)

All intelligence, word-order grammar, negation, statement-overrides-question priority, persona routing, and response pools are compiled OFFLINE by modular scripts (each <= 220 lines) into `model_weights.npz`.

2. THE 10-FAULT "ZERO-FAULT" HISTORICAL LEDGER (NEVER REGRESS!)
--------------------------------------------------------------------------------
* FAULT 1: Statement vs. Question Override ("lm f u", "im f u?", "m looking for girl , u")
  - Solved natively by Ordered Bigrams (`("im", "f")`, `("girl", "u")`) + Positional Embeddings (`pos_emb[0..5]`) in `UnifiedRoutingNet`. Statements with trailing `"u?"`/`"wbu?"` always preserve the user's statement leaf.
* FAULT 2: Persona & Name Consistency ("u?" Ignorance & Mid-Session Name Switching)
  - `route_pool_lut` splits every pool by `bot_g_idx` (`0=Male Bot`, `1=Female Bot`).
  - `leaf_persona_lock_u8[leaf] == 1` (Step 9 Names) locks index bit 0 to `(uid & 1) | ((rot & 3) << 1)` so Even UIDs always talk to Emma (F) / Alex (M) and Odd UIDs always talk to Sarah (F) / Liam (M) without ever switching mid-chat.
* FAULT 3: "What do you do" (Job Trap -> Class 0) vs. "What are you doing" (Step 5) vs. "What do u do for fun?" (Step 10)
  - Ordered bigrams (`("you", "do")` + EOS vs `("you", "doing")` vs `("for", "fun")`) isolate `"what do you do"` strictly to Leaf 0 (Class 0).
* FAULT 4 & 5: "How are you doing" (Step 3) vs. "What are you doing" (Step 5) vs. "Whats up" (Step 1)
  - Ordered bigrams (`BOS -> "how"` vs `BOS -> "what"` + `"up" -> EOS`) separate all three with 0 overlap.
* FAULT 6: Memory Leak / Type Pollution via Float64 or Int64
  - Every array in `model_weights.npz` is audited against `(np.int8, np.uint8, np.uint16, np.int32, np.uint32, np.float32, np.bool_)` before saving. 64-bit conversation flags use two `np.uint32` arrays (`flags_lo`, `flags_hi`).
* FAULT 7: Bitmask Hash Collision (`& 4090` dropped bits 0 & 2, making `'g'=103` collide with `'b'=98` on `"a gf"` vs `"a bf"`)
  - Fixed permanently by using Prime Modulus `% 4091` (`WORD_MOD = 4091`) for word hashes and Mersenne Prime `% 8191` (`BIGRAM_MOD = 8191`) for ordered bigrams in both `offline_encoder.py` and `production_server.py`.
* FAULT 8: Multi-Turn Split Ask-Back ("M" -> next msg "u" vs. "Chicken" -> next msg "u")
  - Standalone ask-backs (`"u"`, `"u?"`, `"and u?"`, `"wbu?"`, `"what about u?"`) classify as **Leaf 76 (`context_ask_back`)**.
  - In `production_server.py`, 2 lines of O(1) table lookup resolve context:
    `if leaf == 76: leaf = int(m.askback_redirect[int(u.last_leaf[uid])])`
  - If `last_leaf` was `"M"` (`Leaf 13`), `askback_redirect[13] -> Leaf 9` (which sees `F_ASKED_GENDER` already set and replies from `g_choice_f_known`: `"im F :) how old r u?"` without re-asking `"u?"`!).
  - If `last_leaf` was `"Chicken"` (`Leaf 0`) or a question leaf (`Leaf 9`), `askback_redirect[0] -> Leaf 0` (Class 0 Reject — zero hallucination and zero infinite `"u"` spam loops!).
* FAULT 9: "Chatting with u" / "talking with u" / "texting u"
  - Dedicated **Leaf 77 (`status_chatting_with_u`, Class 16)** so trailing `"with u"` is recognized as a warm activity statement (`"aww haha same, just chilling in bed chatting with u!"`) with `askback_redirect[77] = 19`.
* FAULT 10: `negatives.txt` Cross-Step Poisoning & Deleted `.txt` Rebuild Loop
  - `offline_train.py` only admits lines from `negatives.txt` whose `canonical_sig(s)` does not match any positive leaf `1..95`.
  - `offline_train.py` tracks `builder_manifest_u8` and `specs_digest_u8` inside `model_weights.npz` (never checks `.txt` file existence on disk to decide whether to rebuild) and carries forward `entity_lut_a/b` and `super_bloom_u32` so deleting `.txt` files after training is 100% safe.

3. COMPLETE MAP OF ALL 96 LEAVES & 56 INTENT CLASSES (STEPS 1 TO 17)
--------------------------------------------------------------------------------
[Defined in `offline_spec_01_core.py` — Steps 1-7 + Reset + Ask-Back + Chatting-with-u]
- Leaf 0  (Class 0):  not_valid (Gibberish, traps, "what do you do", "why not", "who are you", "a bf here", "Chicken")
- Leaf 1  (Class 1):  english_greeting ("hi", "hey", "whats up", "good morning", "nice to meet u") [RULE_GREET]
- Leaf 2  (Class 2):  ask_age ("age?", "how old r u")
- Leaf 3  (Class 3):  tell_age ("24", "im 24", "21yres") [RULE_AGE: normal / repeat / changed]
- Leaf 4  (Class 4):  ask_country ("from?", "where r u from")
- Leaf 5  (Class 5):  tell_country ("usa", "tuinsi", "handi", "im from tunisia") [RULE_COUNTRY: normal / repeat / changed]
- Leaf 6  (Class 6):  tell_age_and_country ("hi l'm 30 from tun", "24 from uk")
- Leaf 7  (Class 7):  underage_reject (Ages 1..17 fast-path)
- Leaf 8  (Class 8):  ask_how_are_you ("how are you", "hru", "hyd", "hor", "how u doing")
- Leaf 9  (Class 9):  ask_gender_choice ("m or f?", "m/f", "gender?") [rep_st=1 -> g_choice_f/m_known without "u?"]
- Leaf 10 (Class 10): ask_gender_is_female ("f?", "girl?", "are u a girl")
- Leaf 11 (Class 11): ask_gender_is_male ("m?", "boy?", "are u a boy")
- Leaf 12 (Class 12): state_gender_female ("f", "famle", "im a girl", "lm f u") [RULE_SET_G_F]
- Leaf 13 (Class 13): state_gender_male ("m", "male", "im a boy", "im m u?") [RULE_SET_G_M]
- Leaf 14 (Class 14): state_age_and_gender_f ("22f", "19 famle", "21 girl") [RULE_SET_G_F]
- Leaf 15 (Class 14): state_age_and_gender_m ("24 m", "22 male", "23 boy") [RULE_SET_G_M]
- Leaf 16 (Class 15): ask_wyd ("wyd", "what are you doing", "what u up to")
- Leaf 17 (Class 16): status_positive ("im good", "im ok", "just chilling in bed", "just chatting", "listening to music")
- Leaf 18 (Class 17): status_negative ("bored", "lm borad", "nothing much", "im tired", "cant sleep")
- Leaf 19 (Class 18): status_ask_back ("im good u?", "lm borad u?", "nmu", "nothing much wbu?")
- Leaf 20 (Class 19): compliment_flirt ("ur cute", "u seem nice", "i like u", "u look good", "love ur vibe")
- Leaf 21 (Class 20): short_reaction_ack ("haha", "lol", "nice", "cool", "true", "ok", "same", "sam", "same here", "for real")
- Leaf 75 (Class 99): system_reset ("next", "reset", "new", "skip", "/start")
- Leaf 76 (Class 0*): context_ask_back ("u", "u?", "and u?", "wbu?", "hbu?", "what about u?") -> redirected via `askback_redirect_u8[last_leaf]`!
- Leaf 77 (Class 16): status_chatting_with_u ("Chatting with u", "talking with u", "texting u")

[Defined in `offline_spec_02_dating.py` — Steps 8-11]
- Leaf 22 (Class 21): ask_looking_for ("what r u looking for?", "why r u on here?", "looking for friends or dating?")
- Leaf 23 (Class 22): state_looking_friends ("looking for friends", "making friends", "friends", "here to chat")
- Leaf 24 (Class 22): state_looking_girl ("m looking for girl , u", "lm finding gril", "im looking for a gf", "a gf")
- Leaf 25 (Class 22): state_looking_boy ("f looking for boy wbu?", "im looking for a bf", "a bf")
- Leaf 26 (Class 22): state_looking_dating ("looking for dating", "a relationship", "looking for love")
- Leaf 27 (Class 23): ask_name ("whats ur name?", "ur name?", "name?", "what should i call u?") [plock=1: Emma/Sarah & Alex/Liam]
- Leaf 28 (Class 24): tell_name ("hatem", "im hatem", "my name is sarah", "im sam", "my name is zoltan" wildcard) [plock=1]
- Leaf 29..33 (Class 25): ask_hobbies (29=Music, 30=Games, 31=Movies/Anime, 32=Sports/Gym, 33=General/Fun)
- Leaf 34..38 (Class 26): tell_hobbies (34=Music, 35=Games, 36=Movies/Anime, 37=Sports/Gym, 38=General/Travel)
- Leaf 39 (Class 27): ask_rel_status ("are you single?", "are you single or mariid", "do u have a bf?", "are u in a relationship?")
- Leaf 40 (Class 28): tell_rel_single ("im single", "im single u?", "im not married", "i dont have a gf", "divorced")
- Leaf 41 (Class 28): tell_rel_taken ("im taken", "im in a relationship", "i have a gf", "i have a bf")
- Leaf 42 (Class 28): tell_rel_married ("im married", "lm mariid", "im engaged", "i have a wife")

[Defined in `offline_spec_03_live.py` — Steps 12-15]
- Leaf 43 (Class 29): ask_private_tg ("can we talk privty", "can we talk private?", "ur telegram?", "telgrm?", "dm me")
- Leaf 44 (Class 29): ask_social_media ("do u have snap?", "ur insta?", "whatsapp?", "ur number?", "video call?")
- Leaf 45 (Class 30): reply_yes ("yes", "yes plz", "yes sure")
- Leaf 46 (Class 30): reply_no ("no", "nope", "nah", "no thanks", "i dont have snap")
- Leaf 47 (Class 30): reply_maybe ("maybe", "idk", "i dont know", "not sure", "lets stay here")
- Leaf 48 (Class 31): ask_pic_photo ("send pic", "send a selfie", "pic for pic", "can i see u?")
- Leaf 49 (Class 31): ask_appearance ("what do u look like?", "how tall r u?", "describe urself")
- Leaf 50 (Class 32): offer_pic ("wanna see me?", "can i send a pic?", "ill send first")
- Leaf 51 (Class 32): refuse_or_first_pic ("u first", "i dont send pics", "no pics")
- Leaf 52 (Class 33): ask_is_bot_fake ("are u a bot?", "bot?", "are u ai?", "bot or real?", "u reply so fast") -> Explicit NO!
- Leaf 53 (Class 34): ask_is_real_human ("are u real?", "are u real", "are u human?", "prove ur real") -> "im 100% real, not a bot!"
- Leaf 54..56 (Class 35): goodbye (54="bye"/"gtg"), goodnight (55="gn"/"goodnight"), brb (56="brb"/"wait a sec")
- Leaf 57..58 (Class 36): ask_work_study (57="do u work or study?"), tell_work_study (58="im a student"/"in college")
- Leaf 59..60 (Class 37): language_check (59="do u speak arabic?"), language_bad_eng (60="my english is bad")
- Leaf 61..63 (Class 38): playful_ama (61="ask me anything"), truth_or_dare (62="truth or dare"), tell_about_self (63="tell me about urself")
- Leaf 64..66 (Class 39): flirt_like_me (64="do u like me?"), flirt_meet_date (65="can we meet?"), flirt_affection (66="i miss u"/"send a hug")
- Leaf 67..68 (Class 40): polite_thanks (67="thanks"/"np"), polite_sorry (68="sorry"/"my bad")
- Leaf 69..70 (Class 41): ping_here (69="u there?"), clarify_wdym (70="wdym"/"what do u mean?")
- Leaf 71..72 (Class 42): nsfw_boundary (71="send nudes"/"wanna sext"), rude_boundary (72="shut up"/"fuck u")
- Leaf 73..74 (Class 43): timezone_late (73="what time is it there?"), weather_chat (74="hows the weather")

[Defined in `offline_spec_04_step16.py` — Step 16 Deep Dating & Interactive Pack]
- Leaf 78..79 (Class 44): tod_pick_truth (78="truth"/"i pick truth"), tod_pick_dare (79="dare"/"give me a dare")
- Leaf 80 (Class 45): vibe_this_or_that ("night owl", "morning person", "im an introvert", "beach and music", "coffee", "tea")
- Leaf 81..82 (Class 46): ask_dating_type (81="whats ur type?"/"whats ur green flag"), ask_ldr_age_pref (82="do u mind long distance?")
- Leaf 83..85 (Class 47): ask_lifestyle (83="do u have pets?"/"do u smoke?"), tell_lifestyle (84="i have a dog"), zodiac (85="whats ur zodiac sign?"/"im a scorpio")
- Leaf 86..87 (Class 48): ask_joke_laugh (86="tell me a joke"/"cheer me up"), vent_bad_day (87="i had a bad day"/"just broke up")
- Leaf 88..89 (Class 49): ask_fav_food (88="whats ur fav food?"/"do u like sushi?"), tell_food_hungry (89="im hungry"/"i love pizza")

[Defined in `offline_spec_05_step17_english.py` — Step 17 Natural Conversational English]
- Leaf 90 (Class 50): reason_fast_reply_bot ("Bcz you make it to fast.", "bcz u reply too fast", "u type too fast", "that was instant")
- Leaf 91 (Class 51): reason_explain_chat ("bcz im bored", "cuz nothing to do", "just killing time")
- Leaf 92 (Class 52): react_trust_believe ("i believe u", "haha ok i believe u", "that makes sense", "oh i see")
- Leaf 93 (Class 52): react_doubt_skeptic ("i dont believe u", "no way", "really?", "seriously?", "stop lying")
- Leaf 94 (Class 53): english_casual_bridge ("i guess", "kinda", "maybe later", "up to u", "nvm", "who knows")
- Leaf 95 (Class 54): ask_why_curious ("why do u ask?", "how come?", "just curious", "what happened?")

4. EXACT `model_weights.npz` VAULT SCHEMA
--------------------------------------------------------------------------------
- `vm_version_u8`: (1,) np.uint8 = [14]
- `specs_digest_u8`: (32,) np.uint8 (SHA-256 of offline_encoder.py, offline_fsm_compiler.py, and all offline_spec_*.py)
- `builder_manifest_u8`: (M,) np.uint8 (ASCII "build_X_txt.py:sha16\n..." manifest so deleted .txt files never trigger rebuilds)
- `emb_word_q` (4096, 64) np.int8 + `emb_word_s` (1,) np.float32
- `emb_bg_q` (8192, 64) np.int8 + `emb_bg_s` (1,) np.float32
- `pos_emb_f32` (6, 64) np.float32
- `w_struct_q` (16, 128) np.int8 + `w_struct_s` (1,) np.float32
- `w1_q` (128, 256) np.int8 + `w1_s` (1,) np.float32 + `b1_f32` (1, 256) np.float32
- `w2_q` (256, 128) np.int8 + `w2_s` (1,) np.float32 + `b2_f32` (1, 128) np.float32
- `w_out_q` (128, NUM_LEAVES) np.int8 + `w_out_s` (1,) np.float32 + `b_out_f32` (1, NUM_LEAVES) np.float32
- `super_bloom_u32` (65536,) np.uint32
- `entity_lut_a` & `entity_lut_b` (65536,) np.uint8
- `char_lut_i32` (256,) np.int32, `rej_lut_b` (256,) np.bool_, `w_poly_mat_i32` (16, 4) np.int32
- `http_blob_u8` (N_BYTES,) np.uint8, `pool_msg_off_i32` & `pool_msg_end_i32` (N_POOLS, 32) np.int32, `pool_mask_u8` (N_POOLS,) np.uint8
- `route_pool_lut` (NUM_LEAVES, 3, 3, 2) np.uint16
- `leaf_to_class_u8`, `leaf_entity_rule_u8`, `leaf_persona_lock_u8`, `askback_redirect_u8` (NUM_LEAVES,) np.uint8
- `leaf_chk_lo_u32`, `leaf_chk_hi_u32`, `leaf_set_lo_u32`, `leaf_set_hi_u32` (NUM_LEAVES,) np.uint32

5. HOW TO ADD STEP 18+ OR FIX AN EDGE CASE IN THE NEW CHAT (RULE 15)
--------------------------------------------------------------------------------
- NEVER re-output `production_server.py`, `offline_encoder.py`, `offline_fsm_compiler.py`, `offline_train.py`, or `offline_spec_01..05.py` unless specifically asked.
- To add a new Step 18: generate ONLY `build_step18_*_txt.py` (File 1) and `offline_spec_06_step18_*.py` (File 2, starting at Leaf 96 / Class 55). Running `python offline_train.py` will auto-detect the new builder and new spec, run only the new builder, compile all leaves into `model_weights.npz`, and verify 100% Zero Faults automatically!