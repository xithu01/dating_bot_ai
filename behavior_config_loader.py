#!/usr/bin/env python3
# FILE: behavior_config_loader.py
# PURPOSE: Pure-NumPy Startup Compiler & Validator for bot_behavior_config.json
# RULES: Enforces Rules 1-15 (<= 220 lines, zero float64/int64, __slots__, read-only CDF arrays, O(1) sampling)

import json
import os
import numpy as np

# Canonical uint8 Topic IDs shared across Behavior Engine & Config
TOPIC_NONE = 0
TOPIC_DOING = 1
TOPIC_FROM = 2
TOPIC_AGE = 3
TOPIC_AGE_BRANCH = 4
TOPIC_JOB = 5
TOPIC_MARRIED = 6
TOPIC_LOOK = 7
TOPIC_INTIMATE = 8
TOPIC_NAME = 9
TOPIC_GENDER = 10
TOPIC_STATE = 11
TOPIC_HORNY = 12
NUM_TOPICS = 13

TOPIC_KEY_TO_ID = {
    "q_doing": TOPIC_DOING, "q_from": TOPIC_FROM, "q_age": TOPIC_AGE,
    "q_age_branch": TOPIC_AGE_BRANCH, "q_job": TOPIC_JOB, "q_married": TOPIC_MARRIED,
    "q_look": TOPIC_LOOK, "q_intimate": TOPIC_INTIMATE, "q_name": TOPIC_NAME,
    "q_gender": TOPIC_GENDER, "q_state": TOPIC_STATE, "q_horny": TOPIC_HORNY,
}


class WeightedPool:
    __slots__ = ("items", "cdf", "ids", "size")

    def __init__(self, pairs, id_start):
        if not pairs:
            raise ValueError("[CONFIG ERROR] WeightedPool cannot be empty")
        items = tuple(p[0] if isinstance(p, (list, tuple)) else p for p in pairs)
        wts = np.array(
            [np.float32(p[1]) if isinstance(p, (list, tuple)) else np.float32(1.0) for p in pairs],
            dtype=np.float32,
        )
        tot = np.float32(np.sum(wts, dtype=np.float32))
        if tot <= np.float32(0.0):
            raise ValueError("[CONFIG ERROR] WeightedPool sum of weights must be > 0")
        cdf = np.cumsum(wts, dtype=np.float32) / tot
        cdf[-1] = np.float32(1.0)
        cdf.flags.writeable = False
        ids = np.arange(id_start, id_start + len(items), dtype=np.uint16)
        ids.flags.writeable = False
        self.items, self.cdf, self.ids, self.size = items, cdf, ids, len(items)

    def pick(self, u_f32, avoid_ring=None):
        idx = int(np.searchsorted(self.cdf, np.float32(u_f32), side="right"))
        if idx >= self.size:
            idx = self.size - 1
        if avoid_ring is not None and self.size > 1:
            a0, a1, a2, a3 = int(avoid_ring[0]), int(avoid_ring[1]), int(avoid_ring[2]), int(avoid_ring[3])
            for step in range(self.size):
                cand = (idx + step) % self.size
                pid = int(self.ids[cand])
                if pid != a0 and pid != a1 and pid != a2 and pid != a3:
                    idx = cand
                    break
        return self.items[idx], int(self.ids[idx])


class WeightedIntPool:
    __slots__ = ("vals", "cdf", "size")

    def __init__(self, pairs):
        if not pairs:
            raise ValueError("[CONFIG ERROR] WeightedIntPool cannot be empty")
        vals = np.array([int(p[0]) for p in pairs], dtype=np.int32)
        wts = np.array([np.float32(p[1]) for p in pairs], dtype=np.float32)
        tot = np.float32(np.sum(wts, dtype=np.float32))
        if tot <= np.float32(0.0):
            raise ValueError("[CONFIG ERROR] WeightedIntPool sum of weights must be > 0")
        cdf = np.cumsum(wts, dtype=np.float32) / tot
        cdf[-1] = np.float32(1.0)
        vals.flags.writeable = False
        cdf.flags.writeable = False
        self.vals, self.cdf, self.size = vals, cdf, int(vals.shape[0])

    def pick(self, u_f32):
        idx = int(np.searchsorted(self.cdf, np.float32(u_f32), side="right"))
        return int(self.vals[idx if idx < self.size else self.size - 1])


class PersonaCfg:
    __slots__ = (
        "female_prob_within_1h", "female_prob_default", "female_cooldown_s",
        "male_normal_prob", "male_ghoster_prob", "ghoster_convert_prob",
        "max_turns_min", "max_turns_max", "vip_prob",
        "bot_age_young_prob", "bot_age_young_min", "bot_age_young_max",
        "bot_age_older_min", "bot_age_older_max",
        "user_min_adult", "user_max_adult", "user_young_max",
        "female_can_share_name", "female_names", "male_names", "name_templates",
        "countries_female", "countries_male", "country_templates_female", "country_templates_male",
        "india_enabled", "india_country", "india_bubble1", "india_bubble2", "india_disconnect_s",
        "country_names", "country_to_idx", "cities_by_cidx", "platform_interests",
    )


class TimersCfg:
    __slots__ = (
        "buffer_debounce_s", "buffer_max_chars", "opening_wait_pool",
        "read_base_min_s", "read_base_max_s", "read_per_char_s", "read_char_cap_s",
        "curiosity_bonus_min_s", "curiosity_bonus_max_s", "curiosity_words", "curiosity_phrases",
        "distraction_prob", "distraction_min_s", "distraction_max_s",
        "typing_cps_min", "typing_cps_max", "typing_max_s",
        "pause_split_min_s", "pause_split_max_s", "pause_opening_min_s", "pause_opening_max_s",
        "pause_followup_min_s", "pause_followup_max_s", "idle_base_pool", "idle_base_add_s",
        "idle_opening_q_add_s", "idle_normal_q_add_s", "idle_disconnect_after_s",
        "ghoster_no_q_idle_min_s", "ghoster_no_q_idle_max_s",
        "disconnect_standard_s", "disconnect_spoiler_or_bored_bye_s", "disconnect_india_s",
        "disconnect_bored_silent_min_s", "disconnect_bored_silent_max_s",
        "disconnect_horny_m1_min_s", "disconnect_horny_m1_max_s",
        "disconnect_horny_m1_male_user_min_s", "disconnect_horny_m1_male_user_max_s",
        "disconnect_horny_m2_min_s", "disconnect_horny_m2_max_s",
        "disconnect_horny_m3_min_s", "disconnect_horny_m3_max_s", "ghoster_disconnect_pool",
    )


class PlatformCfg:
    __slots__ = (
        "rate_limit_max_msgs", "rate_limit_window_s", "rate_limit_reply", "button_cooldown_s",
        "quota_window_s", "quota_max_bot_matches", "quota_max_human_partners",
        "afk_timeout_s", "purge_timeout_s", "session_timeout_s",
    )


class OpeningCfg:
    __slots__ = (
        "bubble_count_pool", "male_greeting_prepend_prob", "female_bubble2_gender_prob",
        "female_native_greeting_prob", "female_greetings", "male_greetings",
        "male_opener_tags", "male_bubble2_gender_prompts", "male_bubble3_country",
        "opening_nudges_male", "opening_nudges_female",
    )


class ProbsCfg:
    __slots__ = (
        "greet_reply_prob", "greet_mirror_prob", "greet_mirror_cap_prob", "greet_followup_prob",
        "gender_yn_direct_prob", "state_tell_first_prob", "state_tell_insist_prob",
        "state_deflect_first_prob", "state_deflect_followup_prob", "proactive_state_ask_prob",
        "hru_reciprocal_when_answered_prob", "hru_bounce_when_unanswered_prob",
        "female_react_prob", "female_advance_after_non_age_prob", "female_merge_bubble_prob",
        "female_proactive_ntmu_prob", "ntmu_followup_prob", "male_normal_ack_prob",
        "male_skip_name_stage_prob", "yn_pending_followup_prob", "bad_english_retry2_prob",
        "bad_english_fallback_turn5_prob", "leading_q_recall_prob",
        "private_strike1_annoyed", "private_strike1_stay", "private_strike1_maybe", "private_followup_prob",
        "qmark_s1_recall_prob", "qmark_s1_skip_group_prob", "qmark_s1_skip_disc", "qmark_s1_skip_next_q",
        "qmark_s2_skip_group_prob", "qmark_s2_skip_disc", "qmark_s2_skip_next_q", "qmark_s3_bored_disc_prob",
        "bored_disc_silent_prob", "insist_s1_reply_prob", "insist_s1_apology_prob",
        "insist_s1_deflect_followup_prob", "insist_s1_ignore_disc", "insist_s1_ignore_next_q",
        "insist_s2_answer_prob", "insist_s2_skip_group_prob", "insist_s2_skip_disc", "insist_s2_skip_next_q",
        "horny_m_s1_skip", "horny_m_s1_disc", "horny_m_s1_reply_male_user_disc", "horny_m_s1_followup",
        "horny_m_s2_disc", "horny_f_s1_skip", "horny_f_s1_silent", "horny_f_s1_followup",
        "media_s2_silent", "media_s2_next_q", "media_s1_spoiler_m_disc", "media_s1_spoiler_f_bye",
        "media_s1_spoiler_f_silent_rest", "media_s1_voice_m_silent", "media_s1_voice_f_silent",
        "media_s1_visual_m_skip", "media_s1_visual_f_silent", "media_s1_visual_f_skip",
        "media_s1_sticker_m_silent", "media_s1_sticker_m_skip", "media_s1_sticker_f_silent",
        "media_s1_sticker_f_skip", "media_f_pending_nudge_prob", "media_f_next_q_prob",
        "extended_intent_followup_prob", "ghoster_convert_q_pool",
    )


class ValKwCfg:
    __slots__ = (
        "confused_tokens", "dry_tokens", "yes_tokens", "no_tokens", "bounce_tokens",
        "counter_question_words", "false_text_fillers", "job_keywords", "horny_statement_markers",
    )


class CompiledBehaviorConfig:
    __slots__ = (
        "source_path", "mtime_i32", "persona", "timers", "platform", "opening", "probs",
        "val_kw", "topic_matrix_f32", "female_stage_pools", "female_age_branch_young",
        "female_age_branch_older", "male_stage_pools", "pools", "ext_replies", "total_phrases",
    )


def _mk_pool(pairs, next_id):
    p = WeightedPool(pairs, next_id[0])
    next_id[0] += p.size
    return p


def load_behavior_config(json_path="bot_behavior_config.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"[CONFIG ERROR] '{json_path}' not found!")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    next_id = [1]
    cfg = CompiledBehaviorConfig()
    cfg.source_path, cfg.mtime_i32 = json_path, np.int32(int(os.path.getmtime(json_path)))

    rp, p = raw["persona"], PersonaCfg()
    for k in ("female_prob_within_1h", "female_prob_default", "male_normal_prob", "male_ghoster_prob", "ghoster_convert_prob", "vip_prob"):
        setattr(p, k, np.float32(rp[k]))
    p.female_cooldown_s, p.max_turns_min, p.max_turns_max = int(rp["female_cooldown_s"]), int(rp["max_turns_min"]), int(rp["max_turns_max"])
    ba, ua = rp["bot_age"], rp["user_age"]
    p.bot_age_young_prob, p.bot_age_young_min, p.bot_age_young_max = np.float32(ba["young_prob"]), int(ba["young_min"]), int(ba["young_max"])
    p.bot_age_older_min, p.bot_age_older_max = int(ba["older_min"]), int(ba["older_max"])
    p.user_min_adult, p.user_max_adult, p.user_young_max = int(ua["min_adult"]), int(ua["max_adult"]), int(ua["young_max"])
    p.female_can_share_name = bool(rp["female_can_share_name"])
    p.female_names, p.male_names, p.name_templates = tuple(rp["female_names"]), tuple(rp["male_names"]), tuple(rp["name_templates"])
    p.countries_female, p.countries_male = _mk_pool(rp["countries_female"], next_id), _mk_pool(rp["countries_male"], next_id)
    p.country_templates_female, p.country_templates_male = tuple(rp["country_templates_female"]), tuple(rp["country_templates_male"])
    ind = rp["female_india_special"]
    p.india_enabled, p.india_country = bool(ind["enabled"]), str(ind["country"])
    p.india_bubble1, p.india_bubble2 = _mk_pool(ind["bubble1"], next_id), _mk_pool(ind["bubble2"], next_id)
    p.india_disconnect_s = np.float32(ind["disconnect_delay_s"])
    c_names = tuple(rp["cities_by_country"].keys())
    p.country_names = c_names
    p.country_to_idx = {name: i + 1 for i, name in enumerate(c_names)}
    p.cities_by_cidx = ((),) + tuple(tuple(rp["cities_by_country"][n]) for n in c_names)
    p.platform_interests = tuple(rp["platform_interests"])
    cfg.persona = p

    rt, t = raw["timers"], TimersCfg()
    t.buffer_max_chars = int(rt["buffer_max_chars"])
    t.opening_wait_pool = WeightedIntPool(rt["opening_wait_s_weights"])
    t.idle_base_pool = WeightedIntPool(rt["idle_base_s_weights"])
    t.ghoster_disconnect_pool = WeightedIntPool(rt["ghoster_disconnect_s_weights"])
    cw = [str(w).lower() for w in rt["curiosity_keywords"]]
    t.curiosity_words = frozenset(w for w in cw if " " not in w)
    t.curiosity_phrases = tuple(w for w in cw if " " in w)
    skip_t = {"buffer_max_chars", "opening_wait_s_weights", "idle_base_s_weights", "ghoster_disconnect_s_weights", "curiosity_keywords"}
    for k, v in rt.items():
        if k not in skip_t:
            setattr(t, k, np.float32(v))
    cfg.timers = t

    rpl, pl = raw["platform"], PlatformCfg()
    for k in ("rate_limit_max_msgs", "quota_window_s", "quota_max_bot_matches", "quota_max_human_partners", "afk_timeout_s", "purge_timeout_s", "session_timeout_s"):
        setattr(pl, k, int(rpl[k]))
    pl.rate_limit_window_s, pl.button_cooldown_s = np.float32(rpl["rate_limit_window_s"]), np.float32(rpl["button_cooldown_s"])
    pl.rate_limit_reply = str(rpl["rate_limit_reply"])
    cfg.platform = pl

    ro, o = raw["opening"], OpeningCfg()
    o.bubble_count_pool = WeightedIntPool(ro["bubble_count_weights"])
    for k in ("male_greeting_prepend_prob", "female_bubble2_gender_prob", "female_native_greeting_prob"):
        setattr(o, k, np.float32(ro[k]))
    for k in ("female_greetings", "male_greetings", "male_opener_tags", "male_bubble2_gender_prompts", "opening_nudges_male", "opening_nudges_female"):
        setattr(o, k, _mk_pool(ro[k], next_id))
    o.male_bubble3_country = str(ro["male_bubble3_country"])
    cfg.opening = o

    rpr, pr = raw["probabilities"], ProbsCfg()
    for k, v in rpr.items():
        if k == "ghoster_convert_q_weights":
            pr.ghoster_convert_q_pool = WeightedIntPool([[TOPIC_KEY_TO_ID[tk], wt] for tk, wt in v])
        else:
            setattr(pr, k, np.float32(v))
    cfg.probs = pr

    tm = np.ones((2, NUM_TOPICS, 3), dtype=np.float32)
    for g_idx, g_key in ((0, "male"), (1, "female")):
        for tk, probs3 in raw["topic_matrix"][g_key].items():
            tid = TOPIC_KEY_TO_ID[tk]
            p_ans, p_ev = np.float32(probs3[0]), np.float32(probs3[1])
            tm[g_idx, tid, 0], tm[g_idx, tid, 1], tm[g_idx, tid, 2] = p_ans, np.float32(p_ans + p_ev), np.float32(1.0)
    tm.flags.writeable = False
    cfg.topic_matrix_f32 = tm

    f_st, m_st = [None] * NUM_TOPICS, [None] * NUM_TOPICS
    for tk, pairs in raw["stage_questions"]["female"].items():
        if tk == "q_age_branch_young":
            cfg.female_age_branch_young = _mk_pool(pairs, next_id)
        elif tk == "q_age_branch_older":
            cfg.female_age_branch_older = _mk_pool(pairs, next_id)
        elif tk in TOPIC_KEY_TO_ID:
            f_st[TOPIC_KEY_TO_ID[tk]] = _mk_pool(pairs, next_id)
    for tk, pairs in raw["stage_questions"]["male"].items():
        if tk in TOPIC_KEY_TO_ID:
            m_st[TOPIC_KEY_TO_ID[tk]] = _mk_pool(pairs, next_id)
    cfg.female_stage_pools, cfg.male_stage_pools = tuple(f_st), tuple(m_st)
    cfg.pools = {k: _mk_pool(v, next_id) for k, v in raw["pools"].items()}

    vk = ValKwCfg()
    for k, v in raw["validation_keywords"].items():
        setattr(vk, k, frozenset(str(x).lower() for x in v))
    cfg.val_kw = vk

    ext = [None] * 256
    for lf_str, replies in raw["extended_intent_replies"].items():
        lf = int(lf_str)
        if 0 <= lf < 256 and replies:
            ext[lf] = _mk_pool(replies, next_id)
    cfg.ext_replies = tuple(ext)
    cfg.total_phrases = next_id[0] - 1
    return cfg


if __name__ == "__main__":
    c = load_behavior_config("bot_behavior_config.json")
    arr_b = c.topic_matrix_f32.nbytes + sum(p.cdf.nbytes + p.ids.nbytes for p in c.pools.values())
    assert c.topic_matrix_f32.dtype == np.float32 and c.timers.opening_wait_pool.vals.dtype == np.int32
    print(f"[CONFIG COMPILER OK] Loaded {len(c.pools)} pools + {sum(x is not None for x in c.ext_replies)} extended NLU pools ({c.total_phrases} unique phrase IDs, {arr_b / 1024.0:.2f} KB CDF arrays, ZERO float64/int64)", flush=True)