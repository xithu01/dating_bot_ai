================================================================================
PROJECT: ULTRA-LOW-RAM 1-ON-1 RANDOM DATING CHAT AI (PURE NUMPY ENGINE)
TARGET: 50,000 Concurrent Users | < 50MB AI RAM | < 500MB Total Server RAM
================================================================================

1. CORE ARCHITECTURAL DIRECTIVES (STRICTLY ENFORCED)
--------------------------------------------------------------------------------
- ZERO-ALLOCATION INFERENCE: `production_server.py` must allocate 0 bytes of new RAM during inference. It uses a pre-allocated `InferenceSlot` with `np.zeros` scratchpads, updating them using `out=` (e.g., `np.matmul(a, b, out=slot.buf)`).
- 8-BIT QUANTIZATION: `offline_train.py` exports weights as `np.int8` with `np.float32` scales. ZERO `np.float64` or `np.int64` allowed globally.
- PYARMOR LIMITS: `production_server.py` and `production_server_part2.py` must each remain strictly under 430 lines and < 18 KB AST complexity. Response pools and routing logic live in `part2` to protect `production_server.py`.
- O(1) LOOKUPS: We bypass slow neural math for keywords by using pre-computed Bloom Filters (`s4_gender_bloom_q`, etc.) and flat Look-Up Tables (`s6_status_lut_a`).

2. THE "ZERO-FAULT" HISTORICAL LEDGER (PREVIOUS BUGS FIXED - DO NOT REGRESS!)
--------------------------------------------------------------------------------
To prevent the AI from making past mistakes when writing new steps, respect these fixed collisions:

* FAULT 1: The "Statement vs. Question" Override (Step 4 Gender Clash)
  - Bug: User typed "lm f u" (Statement: "im f", Question: "u?"). The AI saw the question mark/word and instantly forced Class 10 (asking the bot's gender), completely ignoring the user's gender statement.
  - Fix: Implemented `has_s_frame = bool(np.any(g_codes == 5))`. The router now uses `if (not has_s_frame and has_q_frame...)`. Statements ALWAYS override questions in a single sentence so data isn't lost.
* FAULT 2: The Persona "u?" Ignorance
  - Bug: When user typed "im f, u?", the bot saved the female gender but replied with generic text: "nice! how old r u?".
  - Fix: Split Class 12 and 13 response pools by `bot_gender`. The bot now dynamically says "im a guy too! how old r u?" based on its own configuration.
* FAULT 3: "What do you do" (Job) vs. "What are you doing" (Activity)
  - Bug: Both contain "what" and "do".
  - Fix: "what do you do" is hardcoded into `QUESTION_AND_REACTION_TRAPS` (Class 0 Reject). `build_wyd_txt.py` strictly forces words like "doing" or "up to" for Class 15.
* FAULT 4: "How" vs "What" Collision (Step 3 vs Step 5)
  - Bug: AI confused "how are you doing" (Step 3) with "what are you doing" (Step 5).
  - Fix: `WYD_FORBIDDEN_WORDS` strictly bans the word "how", "hru", "hyd". Step 5 is strictly "what/wyd", Step 3 is strictly "how/hru".
* FAULT 5: "Whats up" (Step 1) vs "What are you up to" (Step 5)
  - Bug: Moving "whats up" to Step 5 broke Step 1 greeting flows.
  - Fix: "whats up" stays in Step 1. Step 5 is limited to "what u up to" / "wyd".
* FAULT 6: Memory Leak via Float64
  - Bug: Python defaults to float64 for decimals, breaking RAM ceilings.
  - Fix: Enforced `np.float32(X.X)` wrapper on EVERY float and scalar.

3. THE 19 INTENT CLASSES (0-18 + 99)
--------------------------------------------------------------------------------
- Class 0:  not_valid (Gibberish, traps, jobs, math, trivia)
- Class 1:  english_greeting ("hi", "whats up")
- Class 2:  ask_age ("how old r u")
- Class 3:  tell_age ("im 24")
- Class 4:  ask_country ("where u from")
- Class 5:  tell_country ("usa")
- Class 6:  tell_age_and_country ("24 m usa")
- Class 7:  underage_reject (Ages 10-17)
- Class 8:  ask_how_are_you ("hru?")
- Class 9:  ask_gender_choice ("m or f?")
- Class 10: ask_gender_is_female ("are u a girl?")
- Class 11: ask_gender_is_male ("are u a guy?")
- Class 12: state_gender_female ("im a girl")
- Class 13: state_gender_male ("im a boy")
- Class 14: state_age_and_gender ("24m")
- Class 15: ask_wyd ("wyd?", "what are you doing")
- Class 16: status_positive ("im good", "chilling in bed")
- Class 17: status_negative ("bored", "nothing much")
- Class 18: status_ask_back ("im good u?", "nmu")
- Class 99: system_reset (Commands like next, skip)

4. FILE & FUNCTION DEEP-DIVE DIRECTORY
--------------------------------------------------------------------------------

[A] DATASET GENERATORS (Run these first)
1. build_greetings_txt.py: 
   - `generate_greetings_and_negatives()`: Generates `greetings.txt` and `negatives.txt`. Blocks all other Step keywords so Step 1 is isolated.
2. build_age_country_txt.py: 
   - `build_age_and_country_files()`: Generates `ages.txt` (18-80) and `countries.txt`.
3. build_hru_txt.py: 
   - `build_hru_file()`: Generates `hru.txt`. Forbids "what".
4. build_gender_txt.py:
   - `generate_all_gender_files()`: Creates 5 gender text files (Class 9-14).
5. build_wyd_txt.py:
   - `build_wyd_file()`: Generates `wyd.txt`. Strictly isolates Class 15 from Class 0 ("what do you do") and Class 3 ("how are you doing").
6. build_status_txt.py:
   - `build_all_status_files()`: Creates `status_positive.txt`, `status_negative.txt`, and `status_ask_back.txt`. Disjoint 3-bucket hierarchy (Class 16, 17, 18).

[B] THE NEURAL FORGE
7. offline_train.py:
   - `verify_hardware_and_lock_ram()`: Detects TPU/GPU and locks 75% of free RAM into a `np.uint8` Super-Arena so Colab doesn't OOM crash.
   - `_train_word_mlp_head()`: The core PyTorch MLP trainer. Uses `time.perf_counter()` to print exactly once per 1.0s to avoid console slowdowns.
   - `encode_s1_batch()` & `encode_word_seq_batch()`: Converts strings to O(1) hashed integer token arrays for fast MLP processing.
   - `train_stepX_into_vault(vault, device)`: Functions 1 through 6. Trains each intent head independently and saves to `model_weights.npz` without overwriting older steps.

[C] PRODUCTION ENGINE
8. production_server_part2.py:
   - `_make_pool()`: JSON stringifies pre-rendered response payloads for blazing fast 200 OK returns.
   - `FLAG_GREETED`, `FLAG_ASKED_WYD`, etc.: Bitflags to track conversation memory.
   - `handle_user_query()`: The massive Traffic Cop `if/elif` block. Checks `class_id` against bitflags and `UserStore` memory (Age, Country) to return the exact right JSON pool, bypassing PyArmor AST size limits.
9. production_server.py:
   - `QuantizedModel`: Uses `__slots__` and `lock_f32()` to load `model_weights.npz` exactly once at server startup via `mmap_mode='r'`. Dequantizes `int8` weights to `float32` in RAM once.
   - `InferenceSlot`: Pre-allocates NumPy arrays (`s2_emb_buf`, `s3_logits`, `s6_logits`) for each async worker to avoid allocating RAM dynamically per request.
   - `forward_pass()`: The O(1) NumPy math engine. Executes Bloom Filter checks first (`sX_bloom_q`); if triggered, runs `np.matmul` MLP math using `out=slot.buf`. Returns `class_id`, `age_val`, `country_id`.
   - `start_server()`: Asyncio API handler for incoming POST requests.

[D] AUTOMATED TESTING
10. test_studio.py:
    - `run_tests()`: Hits 127.0.0.1:8080 with byte payloads to verify all Intent Classes (0-18, 99) trigger correctly.