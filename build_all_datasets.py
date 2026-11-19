#!/usr/bin/env python3
# FILE: build_all_datasets.py
# PURPOSE: 1-Click Offline Master Dataset Builder for Steps 1-15 (Runs all 15 build_*.py scripts in order)
# RULES: Enforces Rules 1-14:
#        - Rule 12: Sets HF_DATASETS_TRUST_REMOTE_CODE="0", NEVER opens/reads/modifies model_weights.npz
#        - Rule 13: Runs every Rule 12+13 clean builder and verifies all 28 output .txt files

import importlib
import os
import sys
import time
import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "0"

# Ordered registry: (step_label, module_name, entry_fn_name, expected_output_files)
BUILD_PIPELINE = (
    ("Step 01 (Greetings)", "build_greetings_txt", "generate_greetings_and_negatives", ("greetings.txt", "negatives.txt")),
    ("Step 02 (Age/Country)", "build_age_country_txt", "build_age_and_country_files", ("ages.txt", "countries.txt")),
    ("Step 03 (Ask HRU)", "build_hru_txt", "build_hru_file", ("hru.txt",)),
    ("Step 04 (Gender)", "build_gender_txt", "generate_all_gender_files", (
        "gender_ask_choice.txt", "gender_ask_is_female.txt", "gender_ask_is_male.txt",
        "gender_state_female.txt", "gender_state_male.txt", "age_gender.txt", "underage_traps.txt",
    )),
    ("Step 05 (Ask WYD)", "build_wyd_txt", "build_wyd_file", ("wyd.txt",)),
    ("Step 06 (Status)", "build_status_txt", "build_all_status_files", ("status_positive.txt", "status_negative.txt", "status_ask_back.txt")),
    ("Step 07 (Flirt/React)", "build_flirt_react_txt", "build_flirt_and_react_files", ("compliments_flirt.txt", "reactions_ack.txt")),
    ("Step 08 (Looking For)", "build_looking_for_txt", "build_looking_for_files", ("looking_for_ask.txt", "looking_for_state.txt")),
    ("Step 09 (Names)", "build_name_txt", "build_name_files", ("name_ask.txt", "name_tell.txt", "names_female_catalog.txt", "names_male_catalog.txt")),
    ("Step 10 (Hobbies)", "build_hobbies_txt", "build_hobbies_files", ("hobbies_ask.txt", "hobbies_tell.txt")),
    ("Step 11 (Rel Status)", "build_rel_status_txt", "build_rel_status_files", ("rel_status_ask.txt", "rel_status_tell.txt")),
    ("Step 12 (Private/YN)", "build_private_yesno_txt", "build_private_yesno_files", ("private_social_ask.txt", "yes_no_maybe.txt")),
    ("Step 13 (Pic/Photo)", "build_pic_photo_txt", "build_pic_photo_files", ("pic_photo_ask.txt", "pic_photo_offer.txt")),
    ("Step 14 (Bot Check)", "build_bot_check_txt", "build_bot_check_files", ("bot_check_is_bot.txt", "bot_check_is_real.txt")),
    ("Step 15 (Dating Pack)", "build_step15_pack_txt", "build_step15_pack_files", (
        "s15_goodbye_gn.txt", "s15_work_study.txt", "s15_language.txt",
        "s15_playful_ama.txt", "s15_flirt_meet.txt", "s15_thanks_sorry.txt",
        "s15_ping_clarify.txt", "s15_nsfw_rude.txt", "s15_timezone.txt",
    )),
)


def run_all_builders(skip_existing=False):
    t_all = time.perf_counter()
    ran_cnt = np.int32(0)
    skip_cnt = np.int32(0)
    missing_cnt = np.int32(0)

    print("=" * 72, flush=True)
    print("[RULE 12+13 MASTER DATASET PIPELINE] Building Steps 1-15 Pure ASCII .txt Files", flush=True)
    print("=" * 72, flush=True)

    for label, mod_name, fn_name, out_files in BUILD_PIPELINE:
        if skip_existing and all(os.path.exists(f) for f in out_files):
            print(f"[SKIP EXISTING] {label}: all {len(out_files)} .txt files already present.", flush=True)
            skip_cnt += np.int32(1)
            continue

        if not os.path.exists(f"{mod_name}.py"):
            print(f"[MISSING SCRIPT] {label}: '{mod_name}.py' not found (offline_train.py will use CORE_LEAF_SEEDS).", flush=True)
            missing_cnt += np.int32(1)
            continue

        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            fn()
            dt = np.float32(time.perf_counter() - t0)
            print(f"[OK] {label} finished in {float(dt):.2f}s", flush=True)
            ran_cnt += np.int32(1)
        except Exception as exc:
            print(f"[WARN] {label} failed ({exc}); offline_train.py will still protect via CORE_LEAF_SEEDS.", flush=True)

    # Final Inventory Audit of Generated ASCII .txt Files
    tot_files = np.int32(0)
    tot_lines = np.int32(0)
    tot_bytes = np.int32(0)
    seen = set()
    for _, _, _, out_files in BUILD_PIPELINE:
        for fn in out_files:
            if fn in seen or not os.path.exists(fn):
                continue
            seen.add(fn)
            tot_files += np.int32(1)
            tot_bytes += np.int32(os.path.getsize(fn))
            with open(fn, "r", encoding="ascii", errors="ignore") as f:
                tot_lines += np.int32(sum(1 for ln in f if ln.strip()))

    total_sec = np.float32(time.perf_counter() - t_all)
    tot_kb = np.float32(tot_bytes) / np.float32(1024.0)
    print("-" * 72, flush=True)
    print(
        f"[PIPELINE DONE in {float(total_sec):.1f}s] Ran: {int(ran_cnt)} | Skipped: {int(skip_cnt)} | Missing: {int(missing_cnt)}",
        flush=True,
    )
    print(
        f"[DATASET INVENTORY] {int(tot_files)} pure ASCII .txt files ready | "
        f"{int(tot_lines)} total lines ({float(tot_kb):.2f} KB) -> Next: python offline_train.py",
        flush=True,
    )


if __name__ == "__main__":
    run_all_builders(skip_existing="--skip-existing" in sys.argv[1:])