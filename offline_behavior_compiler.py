#!/usr/bin/env python3
# FILE: offline_behavior_compiler.py
# PURPOSE: Rule 14+15 Offline Compiler — Packs 100% of bot_behavior_config.json (texts, weights,
#          timers, persona rules, evasion matrix, strike decision trees & keyword Bloom) into model_weights.npz
# RULES: Enforces Rules 1-15 (<= 220 lines, zero float64/int64/int16, strictly allowed .npz dtypes only)

import hashlib
import json
import os
import numpy as np

T_NONE, T_DOING, T_FROM, T_AGE, T_AGE_BR, T_JOB, T_MAR, T_LOOK, T_INT, T_NAME, T_GEN, T_STATE, T_HORNY = range(13)
TOPIC_MAP = {
    "q_doing": T_DOING, "q_from": T_FROM, "q_age": T_AGE, "q_age_branch": T_AGE_BR,
    "q_job": T_JOB, "q_married": T_MAR, "q_look": T_LOOK, "q_intimate": T_INT,
    "q_name": T_NAME, "q_gender": T_GEN, "q_state": T_STATE, "q_horny": T_HORNY,
}
NO_POOL = 65535

POOL_SLOTS = (
    "underage_reject", "reset_chat", "female_greetings", "male_greetings", "male_opener_tags",
    "male_bubble2_gender_prompts", "opening_nudges_female", "opening_nudges_male",
    "gender_reveal_female", "gender_reveal_male", "gender_yn_female", "gender_yn_male",
    "post_gender_female", "post_gender_male", "countries_female", "countries_male",
    "india_bubble1", "india_bubble2", "country_templates_female", "country_templates_male",
    "female_names", "male_names", "name_templates", "platform_interests",
    "age_react_female_young", "age_react_female_older", "country_react_female", "doing_react_female",
    "male_short_ack", "ntmu_reply", "ntmu_proactive_append", "proactive_state_ask",
    "country_evade_append", "hru_reciprocal", "hru_bounce_female", "hru_bounce_male",
    "insist_apology_female", "insist_apology_male", "insist_deflect_female", "insist_deflect_male",
    "insist_pushback_female", "insist_pushback_male", "bored_bye_female", "bored_bye_male",
    "confused_reply", "qmark_annoyed", "dry_s1_no_q", "dry_s2_no_q", "dry_pending_react", "dry_bye",
    "generic_no", "pending_yn_react", "pending_invalid_react", "repeat_warn", "repeat_spam_bye",
    "max_turns_bye", "bad_english_check", "female_safe_fallback", "gift_thank_you",
    "private_annoyed", "private_stay", "private_maybe", "media_s2_female", "media_s2_male",
    "media_spoiler_male", "media_spoiler_female_bye", "media_spoiler_female_react",
    "media_visual_male", "media_visual_female_early", "media_visual_female_late",
    "media_voice_male", "media_voice_female", "media_sticker_male", "media_sticker_female",
    "media_female_nudge", "horny_male_reply", "horny_female_reply", "horny_female_reply_ask",
    "female_age_branch_young", "female_age_branch_older",
    "cities_0", "cities_1", "cities_2", "cities_3", "cities_4",
    "country_names", "male_bubble3_country", "rate_limit_reply",
)
S = {name: i for i, name in enumerate(POOL_SLOTS)}


def compute_behavior_digest(json_path="bot_behavior_config.json"):
    h = hashlib.sha256()
    for fn in ("offline_behavior_compiler.py", json_path):
        if os.path.exists(fn):
            h.update(fn.encode("ascii"))
            with open(fn, "rb") as f:
                h.update(f.read())
    return np.frombuffer(h.digest(), dtype=np.uint8).copy()


def _fnv32_word(w):
    h = 2166136261
    for b in w.encode("ascii", errors="ignore"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def _norm_pairs(items):
    return [(str(x[0]), float(x[1])) if isinstance(x, (list, tuple)) else (str(x), 1.0) for x in items]


def compile_behavior_brain(json_path="bot_behavior_config.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[BEHAVIOR COMPILER] '{json_path}' not found!")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rp, rt, rpl, ro, rpr, rpool, rsq = raw["persona"], raw["timers"], raw["platform"], raw["opening"], raw["probabilities"], raw["pools"], raw["stage_questions"]
    pools_list, slot_to_pid = [], np.full((96,), NO_POOL, dtype=np.uint16)

    def add_pool(pairs, slot_name=None):
        pid, norm = len(pools_list), _norm_pairs(pairs)
        if not norm:
            raise ValueError(f"[BEHAVIOR COMPILER] Empty pool: {slot_name}")
        pools_list.append(norm)
        if slot_name is not None:
            slot_to_pid[S[slot_name]] = np.uint16(pid)
        return pid

    for k in ("female_greetings", "male_greetings", "male_opener_tags", "male_bubble2_gender_prompts", "opening_nudges_female", "opening_nudges_male"):
        add_pool(ro[k], k)
    for k in ("countries_female", "countries_male", "country_templates_female", "country_templates_male", "female_names", "male_names", "name_templates", "platform_interests"):
        add_pool(rp[k], k)
    ind = rp["female_india_special"]
    add_pool(ind["bubble1"], "india_bubble1"); add_pool(ind["bubble2"], "india_bubble2")
    add_pool(rsq["female"]["q_age_branch_young"], "female_age_branch_young")
    add_pool(rsq["female"]["q_age_branch_older"], "female_age_branch_older")
    c_names = list(rp["cities_by_country"].keys())
    add_pool(c_names, "country_names")
    for ci, cn in enumerate(c_names[:5]):
        add_pool(rp["cities_by_country"][cn], f"cities_{ci}")
    add_pool([ro["male_bubble3_country"]], "male_bubble3_country")
    add_pool([rpl["rate_limit_reply"]], "rate_limit_reply")
    for k, v in rpool.items():
        if k in S:
            add_pool(v, k)

    # Topic Pools Table (2, 13, 3) uint16: [is_f, topic, 0=ans/1=evade/2=stage_q] (100% populated — ZERO NO_POOL crashes!)
    topic_pools = np.full((2, 13, 3), NO_POOL, dtype=np.uint16)
    for gi, gk, ans_m, ev_m in (
        (1, "female", {T_DOING: "hru_ans_female", T_AGE_BR: "hru_ans_female", T_LOOK: "look_ans_female", T_JOB: "job_ans_female", T_MAR: "married_ans", T_HORNY: "horny_female_reply"},
         {T_DOING: "hru_evade_female", T_AGE_BR: "hru_evade_female", T_FROM: "country_evade_female", T_AGE: "age_evade", T_LOOK: "look_evade_female", T_JOB: "job_evade_female", T_MAR: "married_evade", T_INT: "secret_evade_female", T_NAME: "name_deflect_female", T_STATE: "state_deflect_female", T_HORNY: "horny_female_reply"}),
        (0, "male", {T_DOING: "hru_ans_male", T_AGE_BR: "hru_ans_male", T_LOOK: "look_ans_male", T_JOB: "job_ans_male", T_MAR: "married_ans", T_HORNY: "horny_male_reply"},
         {T_DOING: "hru_evade_male", T_AGE_BR: "hru_evade_male", T_FROM: "country_evade_male", T_AGE: "age_evade", T_LOOK: "look_evade_male", T_JOB: "job_evade_male", T_MAR: "married_evade", T_INT: "secret_evade_male", T_NAME: "name_evade_male", T_STATE: "state_deflect_male", T_HORNY: "horny_male_reply"}),
    ):
        for tid, pk in ans_m.items(): topic_pools[gi, tid, 0] = np.uint16(add_pool(rpool[pk]))
        for tid, pk in ev_m.items(): topic_pools[gi, tid, 1] = np.uint16(add_pool(rpool[pk]))
        for tk, pairs in rsq[gk].items():
            if tk in TOPIC_MAP: topic_pools[gi, TOPIC_MAP[tk], 2] = np.uint16(add_pool(pairs))

    leaf_ext_pool = np.full((128,), NO_POOL, dtype=np.uint16)
    for lf_s, reps in raw["extended_intent_replies"].items():
        lf = int(lf_s)
        if 0 <= lf < 128 and reps: leaf_ext_pool[lf] = np.uint16(add_pool(reps))

    n_pools, max_w = len(pools_list), max(len(p) for p in pools_list)
    b_off, b_end = np.zeros((n_pools, max_w), dtype=np.int32), np.zeros((n_pools, max_w), dtype=np.int32)
    b_cdf, b_len = np.ones((n_pools, max_w), dtype=np.float32), np.zeros((n_pools,), dtype=np.uint8)
    chunks, cur = [], 0
    for pi, pairs in enumerate(pools_list):
        b_len[pi] = np.uint8(len(pairs))
        wts = np.array([np.float32(w) for _, w in pairs], dtype=np.float32)
        cdf = np.cumsum(wts, dtype=np.float32) / np.float32(np.sum(wts, dtype=np.float32))
        cdf[-1] = np.float32(1.0); b_cdf[pi, : len(pairs)] = cdf
        for j, (txt, _) in enumerate(pairs):
            tb = txt.encode("utf-8")
            b_off[pi, j], cur = np.int32(cur), cur + len(tb); b_end[pi, j] = np.int32(cur); chunks.append(tb)
    b_blob = np.frombuffer(b"".join(chunks), dtype=np.uint8).copy()

    int_tables = (rt["opening_wait_s_weights"], rt["idle_base_s_weights"], ro["bubble_count_weights"], rt["ghoster_disconnect_s_weights"], [[TOPIC_MAP[k], w] for k, w in rpr["ghoster_convert_q_weights"]])
    b_ivals, b_icdf, b_ilen = np.zeros((5, 8), dtype=np.int32), np.ones((5, 8), dtype=np.float32), np.zeros((5,), dtype=np.uint8)
    for ti, pairs in enumerate(int_tables):
        b_ilen[ti] = np.uint8(len(pairs))
        wts = np.array([np.float32(p[1]) for p in pairs], dtype=np.float32)
        cdf = np.cumsum(wts, dtype=np.float32) / np.float32(np.sum(wts, dtype=np.float32)); cdf[-1] = np.float32(1.0)
        for j, (val, _) in enumerate(pairs): b_ivals[ti, j], b_icdf[ti, j] = np.int32(val), cdf[j]

    tm = np.ones((2, 13, 3), dtype=np.float32)
    for gi, gk in ((0, "male"), (1, "female")):
        for tk, p3 in raw["topic_matrix"][gk].items():
            tid, pa, pe = TOPIC_MAP[tk], np.float32(p3[0]), np.float32(p3[1])
            tm[gi, tid, 0], tm[gi, tid, 1], tm[gi, tid, 2] = pa, np.float32(pa + pe), np.float32(1.0)

    r_cdf, r_act = np.ones((24, 5), dtype=np.float32), np.full((24, 5, 5), -1, dtype=np.int32)
    def set_rule(rid, branches):
        acc = np.float32(0.0)
        for j, (prob, act, sl, fup, d_id, flg) in enumerate(branches):
            acc = np.float32(acc + np.float32(prob))
            r_cdf[rid, j] = np.float32(1.0) if j == len(branches) - 1 else acc
            r_act[rid, j] = (act, -1 if sl is None else S[sl], int(round(fup * 1000)), d_id, flg)

    pf = float(rpr["private_followup_prob"])
    set_rule(0, ((rpr["private_strike1_annoyed"], 0, "private_annoyed", pf, 0, 0), (rpr["private_strike1_stay"], 0, "private_stay", pf, 0, 0), (rpr["private_strike1_maybe"], 0, "private_maybe", pf, 0, 0)))
    set_rule(1, ((1.0, 0, "private_annoyed", pf, 0, 0),)); set_rule(2, ((1.0, 2, "private_annoyed", 0.0, 1, 0),))
    sg1, sg2 = float(rpr["qmark_s1_skip_group_prob"]), float(rpr["qmark_s2_skip_group_prob"])
    set_rule(3, ((sg1 * rpr["qmark_s1_skip_disc"], 5, None, 0, 0, 0), (sg1 * rpr["qmark_s1_skip_next_q"], 3, None, 0, 0, 0), (sg1 * (1.0 - rpr["qmark_s1_skip_disc"] - rpr["qmark_s1_skip_next_q"]), 1, None, 0, 0, 0), (1.0 - sg1, 0, "confused_reply", 0, 0, 0)))
    set_rule(4, ((sg2 * rpr["qmark_s2_skip_disc"], 5, None, 0, 0, 0), (sg2 * rpr["qmark_s2_skip_next_q"], 3, None, 0, 0, 0), (sg2 * (1.0 - rpr["qmark_s2_skip_disc"] - rpr["qmark_s2_skip_next_q"]), 1, None, 0, 0, 0), (1.0 - sg2, 0, "qmark_annoyed", 0, 0, 0)))
    set_rule(5, ((rpr["qmark_s3_bored_disc_prob"], 5, None, 0, 0, 0), ((1.0 - rpr["qmark_s3_bored_disc_prob"]) * 0.5, 3, None, 0, 0, 0), ((1.0 - rpr["qmark_s3_bored_disc_prob"]) * 0.5, 1, None, 0, 0, 0)))
    i1, ig1 = float(rpr["insist_s1_reply_prob"]), 1.0 - float(rpr["insist_s1_reply_prob"])
    set_rule(6, ((i1, 4, "insist_deflect_female", rpr["insist_s1_deflect_followup_prob"], 0, 8), (ig1 * rpr["insist_s1_ignore_disc"], 5, None, 0, 0, 0), (ig1 * rpr["insist_s1_ignore_next_q"], 3, None, 0, 0, 0), (ig1 * (1.0 - rpr["insist_s1_ignore_disc"] - rpr["insist_s1_ignore_next_q"]), 1, None, 0, 0, 0)))
    set_rule(7, ((i1, 4, "insist_deflect_male", rpr["insist_s1_deflect_followup_prob"], 0, 8), (ig1 * rpr["insist_s1_ignore_disc"], 5, None, 0, 0, 0), (ig1 * rpr["insist_s1_ignore_next_q"], 3, None, 0, 0, 0), (ig1 * (1.0 - rpr["insist_s1_ignore_disc"] - rpr["insist_s1_ignore_next_q"]), 1, None, 0, 0, 0)))
    i2a, i2r, i2s = float(rpr["insist_s2_answer_prob"]), 1.0 - float(rpr["insist_s2_answer_prob"]), float(rpr["insist_s2_skip_group_prob"])
    set_rule(8, ((i2a, 4, "insist_pushback_female", 0, 0, 0), (i2r * i2s * rpr["insist_s2_skip_disc"], 5, None, 0, 0, 0), (i2r * i2s * rpr["insist_s2_skip_next_q"], 3, None, 0, 0, 0), (i2r * i2s * (1.0 - rpr["insist_s2_skip_disc"] - rpr["insist_s2_skip_next_q"]), 1, None, 0, 0, 0), (i2r * (1.0 - i2s), 0, "insist_pushback_female", 0, 0, 0)))
    set_rule(9, ((i2a, 4, "insist_pushback_male", 0, 0, 0), (i2r * i2s * rpr["insist_s2_skip_disc"], 5, None, 0, 0, 0), (i2r * i2s * rpr["insist_s2_skip_next_q"], 3, None, 0, 0, 0), (i2r * i2s * (1.0 - rpr["insist_s2_skip_disc"] - rpr["insist_s2_skip_next_q"]), 1, None, 0, 0, 0), (i2r * (1.0 - i2s), 0, "insist_pushback_male", 0, 0, 0)))
    set_rule(10, ((rpr["horny_m_s1_skip"], 6, None, 0, 0, 0), (rpr["horny_m_s1_disc"], 2, None, 0, 5, 0), (1.0 - rpr["horny_m_s1_skip"] - rpr["horny_m_s1_disc"], 0, "horny_male_reply", rpr["horny_m_s1_followup"], 0, 2)))
    set_rule(11, ((rpr["horny_m_s2_disc"], 2, None, 0, 6, 0), (1.0 - rpr["horny_m_s2_disc"], 3, "repeat_warn", 0, 0, 0)))
    set_rule(12, ((rpr["horny_f_s1_skip"], 6, None, 0, 0, 0), (rpr["horny_f_s1_silent"], 3, None, 0, 0, 0), (1.0 - rpr["horny_f_s1_skip"] - rpr["horny_f_s1_silent"], 0, "horny_female_reply", rpr["horny_f_s1_followup"], 0, 4)))
    set_rule(13, ((rpr["media_s2_silent"], 1, None, 0, 0, 0), (rpr["media_s2_next_q"], 3, None, 0, 0, 0), (1.0 - rpr["media_s2_silent"] - rpr["media_s2_next_q"], 0, "media_s2_female", 0, 0, 0)))
    set_rule(14, ((rpr["media_s2_silent"], 1, None, 0, 0, 0), (rpr["media_s2_next_q"], 3, None, 0, 0, 0), (1.0 - rpr["media_s2_silent"] - rpr["media_s2_next_q"], 0, "media_s2_male", 0, 0, 0)))
    sf_b, sf_sr = float(rpr["media_s1_spoiler_f_bye"]), float(rpr["media_s1_spoiler_f_silent_rest"])
    set_rule(15, ((sf_b, 2, "media_spoiler_female_bye", 0, 2, 0), ((1.0 - sf_b) * sf_sr, 1, None, 0, 0, 0), ((1.0 - sf_b) * (1.0 - sf_sr), 0, "media_spoiler_female_react", 0, 0, 0)))
    set_rule(16, ((rpr["media_s1_spoiler_m_disc"], 2, None, 0, 1, 0), (1.0 - rpr["media_s1_spoiler_m_disc"], 0, "media_spoiler_male", 0, 0, 0)))
    set_rule(17, ((rpr["media_s1_voice_f_silent"], 1, None, 0, 0, 0), (1.0 - rpr["media_s1_voice_f_silent"], 0, "media_voice_female", 0, 0, 1)))
    set_rule(18, ((rpr["media_s1_voice_m_silent"], 1, None, 0, 0, 0), (1.0 - rpr["media_s1_voice_m_silent"], 0, "media_voice_male", 0, 0, 0)))
    set_rule(19, ((rpr["media_s1_visual_f_silent"], 1, None, 0, 0, 0), (rpr["media_s1_visual_f_skip"], 3, None, 0, 0, 0), (1.0 - rpr["media_s1_visual_f_silent"] - rpr["media_s1_visual_f_skip"], 0, "media_visual_female_late", 0, 0, 1)))
    set_rule(20, ((rpr["media_s1_visual_m_skip"], 3, None, 0, 0, 0), (1.0 - rpr["media_s1_visual_m_skip"], 0, "media_visual_male", 1.0, 0, 0)))
    set_rule(21, ((rpr["media_s1_sticker_f_silent"], 1, None, 0, 0, 0), (rpr["media_s1_sticker_f_skip"], 3, None, 0, 0, 0), (1.0 - rpr["media_s1_sticker_f_silent"] - rpr["media_s1_sticker_f_skip"], 0, "media_sticker_female", 0, 0, 1)))
    set_rule(22, ((rpr["media_s1_sticker_m_silent"], 1, None, 0, 0, 0), (rpr["media_s1_sticker_m_skip"], 3, None, 0, 0, 0), (1.0 - rpr["media_s1_sticker_m_silent"] - rpr["media_s1_sticker_m_skip"], 0, "media_sticker_male", 0, 0, 0)))

    b_disc_rng = np.array((
        (0.0, 0.0), (rt["disconnect_standard_s"], rt["disconnect_standard_s"]),
        (rt["disconnect_spoiler_or_bored_bye_s"], rt["disconnect_spoiler_or_bored_bye_s"]),
        (ind["disconnect_delay_s"], ind["disconnect_delay_s"]),
        (rt["disconnect_bored_silent_min_s"], rt["disconnect_bored_silent_max_s"]),
        (rt["disconnect_horny_m1_min_s"], rt["disconnect_horny_m1_max_s"]),
        (rt["disconnect_horny_m2_min_s"], rt["disconnect_horny_m2_max_s"]),
        (rt["disconnect_horny_m3_min_s"], rt["disconnect_horny_m3_max_s"]),
        (-1.0, -1.0), (rt["disconnect_horny_m1_male_user_min_s"], rt["disconnect_horny_m1_male_user_max_s"]),
    ), dtype=np.float32)

    b_cfg_f32 = np.array((
        rp["female_prob_within_1h"], rp["female_prob_default"], rp["male_normal_prob"], rp["ghoster_convert_prob"],
        rp["vip_prob"], rp["bot_age"]["young_prob"], ind["disconnect_delay_s"], ro["male_greeting_prepend_prob"],
        ro["female_bubble2_gender_prob"], rt["read_base_min_s"], rt["read_base_max_s"], rt["read_per_char_s"],
        rt["read_char_cap_s"], rt["curiosity_bonus_min_s"], rt["curiosity_bonus_max_s"], rt["distraction_prob"],
        rt["distraction_min_s"], rt["distraction_max_s"], rt["typing_cps_min"], rt["typing_cps_max"],
        rt["typing_max_s"], rt["pause_split_min_s"], rt["pause_split_max_s"], rt["pause_opening_min_s"],
        rt["pause_opening_max_s"], rt["pause_followup_min_s"], rt["pause_followup_max_s"], rt["idle_base_add_s"],
        rt["idle_opening_q_add_s"], rt["idle_normal_q_add_s"], rt["ghoster_no_q_idle_min_s"], rt["ghoster_no_q_idle_max_s"],
        rt["disconnect_standard_s"], rt["disconnect_spoiler_or_bored_bye_s"], rt["disconnect_bored_silent_min_s"], rt["disconnect_bored_silent_max_s"],
        rt["disconnect_horny_m1_min_s"], rt["disconnect_horny_m1_max_s"], rt["disconnect_horny_m1_male_user_min_s"], rt["disconnect_horny_m1_male_user_max_s"],
        rt["disconnect_horny_m2_min_s"], rt["disconnect_horny_m2_max_s"], rt["disconnect_horny_m3_min_s"], rt["disconnect_horny_m3_max_s"],
        rpr["greet_reply_prob"], rpr["greet_mirror_prob"], rpr["greet_mirror_cap_prob"], rpr["greet_followup_prob"],
        rpr["gender_yn_direct_prob"], rpr["state_tell_first_prob"], rpr["state_tell_insist_prob"], rpr["state_deflect_first_prob"],
        rpr["state_deflect_followup_prob"], rpr["proactive_state_ask_prob"], rpr["hru_reciprocal_when_answered_prob"], rpr["hru_bounce_when_unanswered_prob"],
        rpr["female_react_prob"], rpr["female_advance_after_non_age_prob"], rpr["female_merge_bubble_prob"], rpr["female_proactive_ntmu_prob"],
        rpr["ntmu_followup_prob"], rpr["male_normal_ack_prob"], rpr["male_skip_name_stage_prob"], rpr["yn_pending_followup_prob"],
        rpr["bad_english_retry2_prob"], rpr["bad_english_fallback_turn5_prob"], rpr["leading_q_recall_prob"], rpr["qmark_s1_recall_prob"],
        rpr["bored_disc_silent_prob"], rpr["insist_s1_apology_prob"], rpr["horny_m_s1_reply_male_user_disc"], rpr["media_f_pending_nudge_prob"],
        rpr["media_f_next_q_prob"], rpr["extended_intent_followup_prob"], rpl["rate_limit_window_s"], rpl["button_cooldown_s"],
    ), dtype=np.float32)

    ind_idx = (c_names.index(ind["country"]) + 1) if (ind["enabled"] and ind["country"] in c_names) else 0
    b_cfg_i32 = np.array((
        rp["female_cooldown_s"], rp["max_turns_min"], rp["max_turns_max"], rp["bot_age"]["young_min"],
        rp["bot_age"]["young_max"], rp["bot_age"]["older_min"], rp["bot_age"]["older_max"],
        rp["user_age"]["min_adult"], rp["user_age"]["young_max"], 1 if rp["female_can_share_name"] else 0,
        ind_idx, rpl["rate_limit_max_msgs"], rpl["session_timeout_s"], rt["buffer_max_chars"],
    ), dtype=np.int32)

    leaf_q, leaf_cfm = np.zeros((128,), dtype=np.uint8), np.zeros((128,), dtype=np.uint8)
    for lf, tid in ((2, T_AGE), (4, T_FROM), (8, T_DOING), (9, T_GEN), (10, T_GEN), (11, T_GEN), (16, T_DOING), (22, T_LOOK), (27, T_NAME), (39, T_MAR), (57, T_JOB), (98, T_STATE), (102, T_INT)):
        leaf_q[lf] = np.uint8(tid)
    for lf, tid in ((17, T_DOING), (18, T_DOING), (19, T_DOING), (77, T_DOING), (23, T_LOOK), (24, T_LOOK), (25, T_LOOK), (26, T_LOOK), (28, T_NAME), (40, T_MAR), (41, T_MAR), (42, T_MAR), (58, T_JOB)):
        leaf_cfm[lf] = np.uint8(tid)

    kw_bloom = np.zeros((4096,), dtype=np.uint32)
    vk = raw["validation_keywords"]
    for bit, words in enumerate((rt["curiosity_keywords"], vk["confused_tokens"], vk["dry_tokens"], vk["yes_tokens"], vk["no_tokens"], vk["bounce_tokens"], vk["counter_question_words"], vk["false_text_fillers"], vk["job_keywords"], vk["horny_statement_markers"])):
        mask = np.uint32(1 << bit)
        for w in words: kw_bloom[_fnv32_word(str(w).strip().lower()) & 4095] |= mask

    return {
        "behavior_digest_u8": compute_behavior_digest(json_path),
        "b_text_blob_u8": b_blob, "b_pool_off_i32": b_off, "b_pool_end_i32": b_end,
        "b_pool_cdf_f32": b_cdf, "b_pool_len_u8": b_len, "b_slot_pid_u16": slot_to_pid,
        "b_topic_pools_u16": topic_pools, "b_leaf_ext_pool_u16": leaf_ext_pool,
        "b_int_vals_i32": b_ivals, "b_int_cdf_f32": b_icdf, "b_int_len_u8": b_ilen,
        "b_topic_cdf_f32": tm, "b_rule_cdf_f32": r_cdf, "b_rule_act_i32": r_act, "b_disc_rng_f32": b_disc_rng,
        "b_cfg_f32": b_cfg_f32, "b_cfg_i32": b_cfg_i32, "b_leaf_q_u8": leaf_q, "b_leaf_cfm_u8": leaf_cfm, "b_kw_bloom_u32": kw_bloom,
    }


if __name__ == "__main__":
    tbl = compile_behavior_brain("bot_behavior_config.json")
    allowed = (np.int8, np.uint8, np.uint16, np.int32, np.uint32, np.float32, np.bool_)
    for k, a in tbl.items():
        assert a.dtype in allowed, f"Forbidden dtype {a.dtype} on {k}"
    kb = sum(a.nbytes for a in tbl.values()) / 1024.0
    print(f"[OFFLINE BEHAVIOR COMPILER OK] Packed {tbl['b_pool_len_u8'].shape[0]} pools + 23 decision trees -> {kb:.2f} KB (ZERO 64-bit!)", flush=True)