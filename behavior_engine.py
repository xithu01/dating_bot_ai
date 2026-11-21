#!/usr/bin/env python3
# FILE: behavior_engine.py
# PURPOSE: Pure-NumPy 50,000-User Human Behavior Director, 14-Priority Decision Engine & Timing Planner
# RULES: Enforces Rules 1-6 & 14 (Zero PyTorch/SciPy, zero float64/int64, __slots__, zero per-request dicts,
#        reads all replies/weights/timers from editable bot_behavior_config.json via behavior_config_loader.py)

import time
import numpy as np

from behavior_config_loader import (
    TOPIC_AGE, TOPIC_AGE_BRANCH, TOPIC_DOING, TOPIC_FROM, TOPIC_GENDER,
    TOPIC_HORNY, TOPIC_INTIMATE, TOPIC_JOB, TOPIC_LOOK, TOPIC_MARRIED, TOPIC_NAME,
    TOPIC_NONE, TOPIC_STATE, load_behavior_config,
)
from production_server import CompiledVMModel, InferenceSlot, MAX_USERS, vm_forward_pass

P_UNROLLED, P_FEMALE, P_MALE_NORMAL, P_MALE_GHOSTER = 0, 1, 2, 3

BT_GREETED = np.uint32(1 << 0)
BT_GENDER = np.uint32(1 << 1)
BT_AGE = np.uint32(1 << 2)
BT_COUNTRY = np.uint32(1 << 3)
BT_STATE = np.uint32(1 << 4)
BT_NAME = np.uint32(1 << 5)
BT_DOING = np.uint32(1 << 6)
BT_LOOK = np.uint32(1 << 7)
BT_JOB = np.uint32(1 << 8)
BT_MARRIED = np.uint32(1 << 9)
BT_SECRET = np.uint32(1 << 10)
BT_HORNY = np.uint32(1 << 11)
BT_NTMU = np.uint32(1 << 12)
BT_EARLY_PIC = np.uint32(1 << 13)

EF_BAD_ENG_ASKED = np.uint16(1 << 0)
EF_VIP_BADGE = np.uint16(1 << 1)
EF_LAST_ASKED_Q = np.uint16(1 << 2)
EF_STATE_ASKED_ONCE = np.uint16(1 << 3)

ST_IDENT, ST_DRY, ST_MEDIA, ST_QMARK, ST_INSIST, ST_PRIVATE, ST_HORNY, ST_INTENT_REP = 0, 1, 2, 3, 4, 5, 6, 7

COUNTRY_ID_NAMES = (
    "", "Tunisia", "India", "USA", "UK", "Algeria", "Morocco", "Egypt", "France", "Germany",
    "Italy", "Spain", "Canada", "Saudi", "UAE", "Turkey", "Pakistan", "Indonesia",
    "Philippines", "Brazil", "there",
)

TOPIC_TO_BT = (
    np.uint32(0), BT_DOING, BT_COUNTRY, BT_AGE, np.uint32(0), BT_JOB, BT_MARRIED,
    BT_LOOK, BT_SECRET, BT_NAME, BT_GENDER, BT_STATE, BT_HORNY,
)

# O(1) Module-Level Lookup Tuples (Zero per-request dict creation!)
EVADE_KEY_F = (
    "female_safe_fallback", "hru_evade_female", "country_evade_female", "age_evade",
    "female_safe_fallback", "job_evade_female", "married_evade", "look_evade_female",
    "secret_evade_female", "name_deflect_female", "female_safe_fallback", "state_deflect_female",
    "horny_female_reply",
)
EVADE_KEY_M = (
    "male_short_ack", "hru_evade_male", "country_evade_male", "age_evade",
    "male_short_ack", "job_evade_male", "married_evade", "look_evade_male",
    "secret_evade_male", "name_evade_male", "male_short_ack", "state_deflect_male",
    "horny_male_reply",
)
ANS_KEY_F = (
    "female_safe_fallback", "hru_ans_female", "female_safe_fallback", "female_safe_fallback",
    "female_safe_fallback", "job_ans_female", "married_ans", "look_ans_female",
    "secret_evade_female", "name_deflect_female", "gender_reveal_female", "state_deflect_female",
    "horny_female_reply",
)
ANS_KEY_M = (
    "male_short_ack", "hru_ans_male", "male_short_ack", "male_short_ack",
    "male_short_ack", "job_ans_male", "married_ans", "look_ans_male",
    "secret_evade_male", "name_evade_male", "gender_reveal_male", "state_deflect_male",
    "horny_male_reply",
)

_Q_LUT = [TOPIC_NONE] * 128
for _lf, _tid in (
    (2, TOPIC_AGE), (4, TOPIC_FROM), (8, TOPIC_DOING), (9, TOPIC_GENDER), (10, TOPIC_GENDER),
    (11, TOPIC_GENDER), (16, TOPIC_DOING), (22, TOPIC_LOOK), (27, TOPIC_NAME),
    (39, TOPIC_MARRIED), (57, TOPIC_JOB), (98, TOPIC_STATE), (102, TOPIC_INTIMATE),
):
    _Q_LUT[_lf] = _tid
LEAF_TO_Q_TOPIC = tuple(_Q_LUT)

FEMALE_STAGE_ORDER = (
    TOPIC_DOING, TOPIC_FROM, TOPIC_AGE, TOPIC_AGE_BRANCH,
    TOPIC_JOB, TOPIC_MARRIED, TOPIC_LOOK, TOPIC_INTIMATE,
)
MALE_STAGE_ORDER = (TOPIC_NAME, TOPIC_FROM, TOPIC_AGE)
YN_ACCEPT_TOPICS = frozenset((TOPIC_MARRIED, TOPIC_JOB, TOPIC_AGE_BRANCH, TOPIC_LOOK, TOPIC_INTIMATE))
OPEN_PENDING_TOPICS = frozenset((TOPIC_JOB, TOPIC_LOOK, TOPIC_INTIMATE, TOPIC_AGE_BRANCH, TOPIC_STATE, TOPIC_NAME))
TRAILING_BOUNCE_2W = ("and u", "and you", "what about u", "what about you", "how about u", "how about you")
TRAILING_BOUNCE_1W = frozenset(("u", "you", "wbu", "hbu", "yours", "urself", "yourself"))


class BubbleStep:
    __slots__ = ("text", "pause_before_s", "typing_s")

    def __init__(self, text, pause_before_s, typing_s):
        self.text, self.pause_before_s, self.typing_s = text, np.float32(pause_before_s), np.float32(typing_s)

    def to_dict(self):
        return {"text": self.text, "pause_before_s": round(float(self.pause_before_s), 2), "typing_s": round(float(self.typing_s), 2)}


class TurnAction:
    __slots__ = (
        "action", "nlu_class", "nlu_leaf", "persona", "vip", "interest",
        "read_delay_s", "bubbles", "idle_timeout_s", "disconnect_after_s",
    )

    def __init__(self, action, nlu_class, nlu_leaf, persona, vip, interest, read_delay_s, bubbles, idle_timeout_s, disconnect_after_s):
        self.action, self.nlu_class, self.nlu_leaf = action, int(nlu_class), int(nlu_leaf)
        self.persona, self.vip, self.interest = persona, bool(vip), interest
        self.read_delay_s, self.bubbles, self.idle_timeout_s = np.float32(read_delay_s), bubbles, np.float32(idle_timeout_s)
        self.disconnect_after_s = None if disconnect_after_s is None else np.float32(disconnect_after_s)

    def to_dict(self):
        return {
            "action": self.action,
            "nlu": {"class_id": self.nlu_class, "leaf_id": self.nlu_leaf},
            "persona": {"type": self.persona, "vip": self.vip, "interest": self.interest},
            "read_delay_s": round(float(self.read_delay_s), 2),
            "bubbles": [b.to_dict() for b in self.bubbles],
            "idle_timeout_s": round(float(self.idle_timeout_s), 2),
            "disconnect_after_s": None if self.disconnect_after_s is None else round(float(self.disconnect_after_s), 2),
        }


class BehaviorStore:
    __slots__ = (
        "persona_type", "turn_count", "max_turns", "bot_age", "bot_country", "bot_city", "bot_name_id",
        "user_gender", "user_age", "user_country", "user_mood", "user_job", "user_rel", "user_hobby",
        "pending_q", "last_topic", "last_confirmed_topic", "pending_retries", "skipped_len",
        "last_leaf", "recent_pos", "interest_idx", "skipped_stack", "streaks",
        "asked_mask", "unanswered_mask", "confirmed_mask", "extra_flags", "recent_phrases",
        "bot_told_mask", "last_msg_hash", "last_female_ts", "last_ts",
    )

    def __init__(self, max_users=MAX_USERS):
        for s in (
            "persona_type", "turn_count", "max_turns", "bot_age", "bot_country", "bot_city", "bot_name_id",
            "user_gender", "user_age", "user_country", "user_mood", "user_job", "user_rel", "user_hobby",
            "pending_q", "last_topic", "last_confirmed_topic", "pending_retries", "skipped_len",
            "last_leaf", "recent_pos", "interest_idx",
        ):
            setattr(self, s, np.zeros((max_users,), dtype=np.uint8))
        self.skipped_stack = np.zeros((max_users, 4), dtype=np.uint8)
        self.streaks = np.zeros((max_users, 8), dtype=np.uint8)
        for s in ("asked_mask", "unanswered_mask", "confirmed_mask", "extra_flags"):
            setattr(self, s, np.zeros((max_users,), dtype=np.uint16))
        self.recent_phrases = np.zeros((max_users, 4), dtype=np.uint16)
        for s in ("bot_told_mask", "last_msg_hash", "last_female_ts", "last_ts"):
            setattr(self, s, np.zeros((max_users,), dtype=np.uint32))

    def total_nbytes(self):
        return sum(getattr(self, s).nbytes for s in self.__slots__)

    def reset_session(self, uid):
        for s in (
            "persona_type", "turn_count", "max_turns", "bot_age", "bot_country", "bot_city", "bot_name_id",
            "user_gender", "user_age", "user_country", "user_mood", "user_job", "user_rel", "user_hobby",
            "pending_q", "last_topic", "last_confirmed_topic", "pending_retries", "skipped_len",
            "last_leaf", "recent_pos", "interest_idx",
        ):
            getattr(self, s)[uid] = np.uint8(0)
        self.skipped_stack[uid].fill(np.uint8(0))
        self.streaks[uid].fill(np.uint8(0))
        self.asked_mask[uid] = self.unanswered_mask[uid] = self.confirmed_mask[uid] = self.extra_flags[uid] = np.uint16(0)
        self.recent_phrases[uid].fill(np.uint16(0))
        self.bot_told_mask[uid] = self.last_msg_hash[uid] = np.uint32(0)


class BehaviorDirector:
    __slots__ = ("model", "cfg", "store", "slot", "_rng")

    def __init__(self, model, cfg, store=None, seed=0):
        self.model, self.cfg = model, cfg
        self.store = store if store is not None else BehaviorStore()
        self.slot = InferenceSlot(0, model.num_leaves)
        s0 = int(seed) if seed else (int(time.perf_counter_ns()) & 0x7FFFFFFF)
        self._rng = np.uint32(s0 if s0 != 0 else 2463534242)

    def reload_config(self, json_path=None):
        self.cfg = load_behavior_config(json_path or self.cfg.source_path)

    def _u32(self):
        x = int(self._rng)
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        self._rng = np.uint32(x if x != 0 else 1)
        return int(self._rng)

    def _rand(self):
        return np.float32((self._u32() & 0x00FFFFFF) / 16777216.0)

    def _uni(self, lo, hi):
        return np.float32(np.float32(lo) + self._rand() * (np.float32(hi) - np.float32(lo)))

    def _randint(self, lo, hi):
        return lo + (self._u32() % (hi - lo + 1))

    def _fnv32(self, s):
        h = 2166136261
        for b in s.encode("ascii", errors="ignore"):
            h = ((h ^ b) * 16777619) & 0xFFFFFFFF
        return np.uint32(h if h != 0 else 1)

    def _pick(self, uid, pool):
        txt, pid = pool.pick(self._rand(), self.store.recent_phrases[uid])
        pos = int(self.store.recent_pos[uid])
        self.store.recent_phrases[uid, pos] = np.uint16(pid)
        self.store.recent_pos[uid] = np.uint8((pos + 1) & 3)
        return txt

    def _p(self, uid, key):
        return self._pick(uid, self.cfg.pools[key])

    def _is_f(self, uid):
        return int(self.store.persona_type[uid]) == P_FEMALE

    def _persona_label(self, uid):
        pt = int(self.store.persona_type[uid])
        return "female" if pt == P_FEMALE else ("male_normal" if pt == P_MALE_NORMAL else "male_ghoster")

    def _push_skip(self, uid, topic):
        if topic == TOPIC_NONE: return
        st, sl = self.store.skipped_stack[uid], int(self.store.skipped_len[uid])
        for i in range(sl):
            if int(st[i]) == topic: return
        if sl < 4:
            st[sl], self.store.skipped_len[uid] = np.uint8(topic), np.uint8(sl + 1)
        else:
            st[0], st[1], st[2], st[3] = st[1], st[2], st[3], np.uint8(topic)

    def _pop_skip(self, uid):
        sl = int(self.store.skipped_len[uid])
        if sl <= 0: return TOPIC_NONE
        top = int(self.store.skipped_stack[uid, sl - 1])
        self.store.skipped_stack[uid, sl - 1], self.store.skipped_len[uid] = np.uint8(0), np.uint8(sl - 1)
        return top

    def _peek_skip(self, uid):
        sl = int(self.store.skipped_len[uid])
        return int(self.store.skipped_stack[uid, sl - 1]) if sl > 0 else TOPIC_NONE

    def _set_pending(self, uid, topic):
        if topic == TOPIC_NONE: return
        old = int(self.store.pending_q[uid])
        if old != TOPIC_NONE and old != topic and (int(self.store.confirmed_mask[uid]) & (1 << old)) == 0:
            self.store.unanswered_mask[uid] |= np.uint16(1 << old)
        self.store.pending_q[uid], self.store.pending_retries[uid] = np.uint8(topic), np.uint8(0)
        self.store.last_topic[uid] = np.uint8(topic)
        self.store.asked_mask[uid] |= np.uint16(1 << topic)

    def _confirm_topic(self, uid, topic):
        if topic == TOPIC_NONE: return
        bit = np.uint16(1 << topic)
        self.store.confirmed_mask[uid] |= bit
        self.store.unanswered_mask[uid] &= np.uint16(~int(bit) & 0xFFFF)
        self.store.last_confirmed_topic[uid] = self.store.last_topic[uid] = np.uint8(topic)
        if int(self.store.pending_q[uid]) == topic:
            self.store.pending_q[uid] = self.store.pending_retries[uid] = np.uint8(0)

    def _ensure_bot_age(self, uid):
        a = int(self.store.bot_age[uid])
        if a > 0: return a
        pc = self.cfg.persona
        a = self._randint(pc.bot_age_young_min, pc.bot_age_young_max) if self._rand() < pc.bot_age_young_prob else self._randint(pc.bot_age_older_min, pc.bot_age_older_max)
        self.store.bot_age[uid] = np.uint8(a)
        return a

    def _ensure_bot_country(self, uid):
        cidx = int(self.store.bot_country[uid])
        if cidx > 0: return self.cfg.persona.country_names[cidx - 1], cidx
        pc = self.cfg.persona
        cname = self._pick(uid, pc.countries_female if self._is_f(uid) else pc.countries_male)
        cidx = pc.country_to_idx.get(cname, 1)
        self.store.bot_country[uid] = np.uint8(cidx)
        return cname, cidx

    def _ensure_bot_city(self, uid):
        _, cidx = self._ensure_bot_country(uid)
        cities, ci = self.cfg.persona.cities_by_cidx[cidx], int(self.store.bot_city[uid])
        if ci == 0:
            ci = self._randint(1, len(cities))
            self.store.bot_city[uid] = np.uint8(ci)
        return cities[ci - 1]

    def _ensure_bot_name(self, uid):
        names, ni = (self.cfg.persona.female_names if self._is_f(uid) else self.cfg.persona.male_names), int(self.store.bot_name_id[uid])
        if ni == 0:
            ni = self._randint(1, len(names))
            self.store.bot_name_id[uid] = np.uint8(ni)
        return names[ni - 1]

    def _typing_s(self, text):
        t = self.cfg.timers
        dur = np.float32(len(text)) / self._uni(t.typing_cps_min, t.typing_cps_max)
        return np.float32(max(np.float32(0.15), min(dur, t.typing_max_s)))

    def _read_delay_s(self, raw_text):
        t = self.cfg.timers
        d = self._uni(t.read_base_min_s, t.read_base_max_s) + min(np.float32(len(raw_text)) * t.read_per_char_s, t.read_char_cap_s)
        lo = raw_text.lower()
        curious = ("?" in lo) or any(ph in lo for ph in t.curiosity_phrases) or any(w in t.curiosity_words for w in lo.replace("?", " ").replace("!", " ").replace(",", " ").split())
        if curious: d += self._uni(t.curiosity_bonus_min_s, t.curiosity_bonus_max_s)
        if self._rand() < t.distraction_prob: d += self._uni(t.distraction_min_s, t.distraction_max_s)
        return np.float32(d)

    def _idle_timeout_s(self, uid, is_opening, asked_q):
        t = self.cfg.timers
        if int(self.store.persona_type[uid]) == P_MALE_GHOSTER and not asked_q and not is_opening:
            return self._uni(t.ghoster_no_q_idle_min_s, t.ghoster_no_q_idle_max_s)
        base = np.float32(t.idle_base_pool.pick(self._rand())) + t.idle_base_add_s
        return np.float32(base + (t.idle_opening_q_add_s if is_opening else t.idle_normal_q_add_s) if asked_q else base)

    def _pack_bubbles(self, raw_items, is_opening=False):
        t, out = self.cfg.timers, []
        for item in raw_items:
            if not item: continue
            for j, seg in enumerate(p.strip() for p in item.split("|") if p.strip()):
                pb = np.float32(0.0) if not out else (self._uni(t.pause_split_min_s, t.pause_split_max_s) if j > 0 else (self._uni(t.pause_opening_min_s, t.pause_opening_max_s) if is_opening else self._uni(t.pause_followup_min_s, t.pause_followup_max_s)))
                out.append(BubbleStep(seg, pb, self._typing_s(seg)))
        return tuple(out)

    def _finish(self, uid, action, nlu_cls, nlu_lf, read_d, raw_bubbles, disc_s=None, is_opening=False, asked_q=False):
        bubbles = self._pack_bubbles(raw_bubbles, is_opening=is_opening)
        asked_q = asked_q or any("?" in b.text for b in bubbles)
        self.store.extra_flags[uid] = (self.store.extra_flags[uid] | EF_LAST_ASKED_Q) if asked_q else (self.store.extra_flags[uid] & np.uint16(~int(EF_LAST_ASKED_Q) & 0xFFFF))
        idle_s = np.float32(0.0) if disc_s is not None else self._idle_timeout_s(uid, is_opening, asked_q)
        vip = (int(self.store.extra_flags[uid]) & int(EF_VIP_BADGE)) != 0
        return TurnAction(action, nlu_cls, nlu_lf, self._persona_label(uid), vip, self.cfg.persona.platform_interests[int(self.store.interest_idx[uid])], read_d, bubbles, idle_s, disc_s)

    def start_chat(self, uid, now_s=None, user_interests=None):
        uid = int(uid) % MAX_USERS
        now_i = int(time.time() if now_s is None else now_s) & 0xFFFFFFFF
        self.store.reset_session(uid)
        pc, prev_f = self.cfg.persona, int(self.store.last_female_ts[uid])
        f_prob = pc.female_prob_within_1h if (prev_f > 0 and (now_i - prev_f) <= pc.female_cooldown_s) else pc.female_prob_default
        if self._rand() < f_prob:
            self.store.persona_type[uid], self.store.last_female_ts[uid] = np.uint8(P_FEMALE), np.uint32(now_i)
        else:
            self.store.persona_type[uid] = np.uint8(P_MALE_NORMAL if self._rand() < pc.male_normal_prob else P_MALE_GHOSTER)
        self.store.max_turns[uid], self.store.last_ts[uid] = np.uint8(self._randint(pc.max_turns_min, pc.max_turns_max)), np.uint32(now_i)
        if self._rand() < pc.vip_prob: self.store.extra_flags[uid] |= EF_VIP_BADGE
        i_idx = self._randint(0, len(pc.platform_interests) - 1)
        if user_interests:
            for ui in user_interests:
                for k, pi in enumerate(pc.platform_interests):
                    if pi.lower() == str(ui).lower(): i_idx = k; break
        self.store.interest_idx[uid] = np.uint8(i_idx)
        return self._persona_label(uid), self.cfg.timers.opening_wait_pool.pick(self._rand())

    def _ensure_session(self, uid, now_s):
        last_t = int(self.store.last_ts[uid])
        if int(self.store.persona_type[uid]) == P_UNROLLED or (last_t > 0 and (now_s - last_t) > self.cfg.platform.session_timeout_s):
            self.start_chat(uid, now_s=now_s)
        self.store.last_ts[uid] = np.uint32(now_s)

    def _next_stage_q(self, uid, prefer_age_branch=False):
        asked, conf = int(self.store.asked_mask[uid]), int(self.store.confirmed_mask[uid])
        if self._is_f(uid):
            if prefer_age_branch and (conf & (1 << TOPIC_AGE)) != 0 and (asked & (1 << TOPIC_AGE_BRANCH)) == 0:
                pool = self.cfg.female_age_branch_young if int(self.store.user_age[uid]) <= self.cfg.persona.user_young_max else self.cfg.female_age_branch_older
                q = self._pick(uid, pool); self._set_pending(uid, TOPIC_AGE_BRANCH); return q, TOPIC_AGE_BRANCH
            for tid in FEMALE_STAGE_ORDER:
                if tid == TOPIC_AGE_BRANCH:
                    if (conf & (1 << TOPIC_AGE)) == 0 or (asked & (1 << TOPIC_AGE_BRANCH)) != 0: continue
                    pool = self.cfg.female_age_branch_young if int(self.store.user_age[uid]) <= self.cfg.persona.user_young_max else self.cfg.female_age_branch_older
                    q = self._pick(uid, pool); self._set_pending(uid, TOPIC_AGE_BRANCH); return q, TOPIC_AGE_BRANCH
                if (asked & (1 << tid)) != 0 or (conf & (1 << tid)) != 0: continue
                q = self._pick(uid, self.cfg.female_stage_pools[tid]); self._set_pending(uid, tid); return q, tid
            return None, TOPIC_NONE
        for tid in MALE_STAGE_ORDER:
            if (asked & (1 << tid)) != 0 or (conf & (1 << tid)) != 0: continue
            if tid == TOPIC_NAME and self._rand() < self.cfg.probs.male_skip_name_stage_prob:
                self.store.asked_mask[uid] |= np.uint16(1 << TOPIC_NAME); continue
            q = self._pick(uid, self.cfg.male_stage_pools[tid]); self._set_pending(uid, tid); return q, tid
        return None, TOPIC_NONE

    def _male_opener_tag(self, uid):
        tag = self._pick(uid, self.cfg.opening.male_opener_tags)
        if "{rand_21_27}" in tag: tag = tag.replace("{rand_21_27}", str(self._randint(21, 27)))
        if tag.lower().startswith(("m", "male")): self.store.bot_told_mask[uid] |= BT_GENDER
        if "?" in tag or "f" in tag.lower(): self._set_pending(uid, TOPIC_GENDER)
        return tag

    def build_opening_burst(self, uid, now_s=None):
        uid = int(uid) % MAX_USERS
        self._ensure_session(uid, int(time.time() if now_s is None else now_s) & 0xFFFFFFFF)
        self.store.turn_count[uid] = np.uint8(min(255, int(self.store.turn_count[uid]) + 1))
        oc, cnt = self.cfg.opening, self.cfg.opening.bubble_count_pool.pick(self._rand())
        bubbles, asked_q = [], False
        if self._is_f(uid):
            bubbles.append(self._pick(uid, oc.female_greetings)); self.store.bot_told_mask[uid] |= BT_GREETED
            if cnt >= 2:
                if self._rand() < oc.female_bubble2_gender_prob:
                    bubbles.append(self._p(uid, "gender_reveal_female")); self.store.bot_told_mask[uid] |= BT_GENDER
                else:
                    q, _ = self._next_stage_q(uid)
                    if q: bubbles.append(q); asked_q = True
            if cnt >= 3:
                q, _ = self._next_stage_q(uid)
                if q: bubbles.append(q); asked_q = True
            if cnt >= 4: bubbles.append(self._pick(uid, oc.opening_nudges_female)); asked_q = True
        else:
            tag = self._male_opener_tag(uid)
            if cnt == 1: bubbles.append(tag)
            else:
                if self._rand() < oc.male_greeting_prepend_prob:
                    bubbles.append(self._pick(uid, oc.male_greetings)); self.store.bot_told_mask[uid] |= BT_GREETED
                bubbles.append(tag)
                if "?" not in tag and "f" not in tag.lower():
                    bubbles.append(self._pick(uid, oc.male_bubble2_gender_prompts)); self._set_pending(uid, TOPIC_GENDER); asked_q = True
                if cnt >= 3: bubbles.append(oc.male_bubble3_country); self._set_pending(uid, TOPIC_FROM); asked_q = True
                if cnt >= 4: bubbles.append(self._pick(uid, oc.opening_nudges_male)); asked_q = True
        return self._finish(uid, "opening_burst", 0, 0, np.float32(0.0), bubbles, None, True, asked_q)

    def _bored_disconnect(self, uid, read_d, nlu_cls, nlu_lf):
        t = self.cfg.timers
        if self._rand() < self.cfg.probs.bored_disc_silent_prob:
            return self._finish(uid, "disconnect", nlu_cls, nlu_lf, read_d, (), self._uni(t.disconnect_bored_silent_min_s, t.disconnect_bored_silent_max_s))
        return self._finish(uid, "disconnect", nlu_cls, nlu_lf, read_d, (self._p(uid, "bored_bye_female" if self._is_f(uid) else "bored_bye_male"),), t.disconnect_spoiler_or_bored_bye_s)

    def _advance_or_fallback(self, uid, read_d, nlu_cls, nlu_lf, lead_bubbles=(), allow_bad_eng=False):
        q, _ = self._next_stage_q(uid)
        if q: return self._finish(uid, "reply", nlu_cls, nlu_lf, read_d, tuple(lead_bubbles) + (q,), asked_q=True)
        if self._is_f(uid):
            if allow_bad_eng and int(self.store.turn_count[uid]) >= 5 and (int(self.store.extra_flags[uid]) & int(EF_BAD_ENG_ASKED)) == 0 and self._rand() < self.cfg.probs.bad_english_fallback_turn5_prob:
                self.store.extra_flags[uid] |= EF_BAD_ENG_ASKED
                return self._finish(uid, "reply", nlu_cls, nlu_lf, read_d, tuple(lead_bubbles) + (self._p(uid, "bad_english_check"),), asked_q=True)
            return self._finish(uid, "reply", nlu_cls, nlu_lf, read_d, tuple(lead_bubbles) + (self._p(uid, "female_safe_fallback"),))
        if lead_bubbles: return self._finish(uid, "reply", nlu_cls, nlu_lf, read_d, lead_bubbles)
        return self._finish(uid, "disconnect", nlu_cls, nlu_lf, read_d, (), self.cfg.timers.disconnect_standard_s)

    def _dispatch_topic(self, uid, topic, force_answer=False, apol="", user_yn_gender=False, user_answered_doing=False):
        is_f, pr, told = self._is_f(uid), self.cfg.probs, int(self.store.bot_told_mask[uid])
        if topic == TOPIC_GENDER:
            if (told & int(BT_GENDER)) != 0 and not force_answer:
                q, _ = self._next_stage_q(uid)
                return ((q,) if q else (self._p(uid, "repeat_warn"),)), None, bool(q)
            self.store.bot_told_mask[uid] |= BT_GENDER
            ans = self._p(uid, ("gender_yn_female" if is_f else "gender_yn_male") if (user_yn_gender and self._rand() < pr.gender_yn_direct_prob) else ("gender_reveal_female" if is_f else "gender_reveal_male"))
            need_u = (int(self.store.confirmed_mask[uid]) & (1 << TOPIC_GENDER)) == 0
            if need_u: self._set_pending(uid, TOPIC_GENDER)
            return (f"{apol} {ans}".strip(),), None, need_u

        if topic == TOPIC_STATE:
            insist_b = force_answer or ((int(self.store.extra_flags[uid]) & int(EF_STATE_ASKED_ONCE)) != 0)
            self.store.extra_flags[uid] |= EF_STATE_ASKED_ONCE
            if self._rand() < (pr.state_tell_insist_prob if insist_b else pr.state_tell_first_prob):
                self.store.bot_told_mask[uid] |= BT_STATE
                q, _ = self._next_stage_q(uid)
                return (f"{apol} {self._ensure_bot_city(uid)}".strip(),) + ((q,) if q else ()), None, bool(q)
            self._push_skip(uid, TOPIC_STATE)
            if insist_b or self._rand() < pr.state_deflect_first_prob:
                d = self._p(uid, "state_deflect_female" if is_f else "state_deflect_male")
                q, _ = self._next_stage_q(uid) if self._rand() < pr.state_deflect_followup_prob else (None, TOPIC_NONE)
                return (f"{apol} {d}".strip(),) + ((q,) if q else ()), None, bool(q)
            q, _ = self._next_stage_q(uid); return ((q,) if q else ()), None, bool(q)

        bt_bit = int(TOPIC_TO_BT[topic]) if topic < len(TOPIC_TO_BT) else 0
        if bt_bit != 0 and (told & bt_bit) != 0 and not force_answer:
            if topic == TOPIC_DOING and user_answered_doing:
                return (self._p(uid, "hru_reciprocal"),), None, False
            q, _ = self._next_stage_q(uid)
            return ((q,), None, True) if q else ((self._p(uid, "repeat_warn"),), (None if is_f else self.cfg.timers.disconnect_standard_s), False)

        row, r = self.cfg.topic_matrix_f32[1 if is_f else 0, topic], self._rand()
        can_force = force_answer and not (is_f and topic == TOPIC_NAME and not self.cfg.persona.female_can_share_name) and topic != TOPIC_INTIMATE
        mode = 0 if can_force else (0 if r < row[0] else (1 if r < row[1] else 2))
        if mode == 2:
            self._push_skip(uid, topic); q, _ = self._next_stage_q(uid); return ((q,) if q else ()), None, bool(q)
        if mode == 1:
            self._push_skip(uid, topic)
            ev = self._p(uid, EVADE_KEY_F[topic] if is_f else EVADE_KEY_M[topic])
            if topic == TOPIC_FROM and is_f and (int(self.store.confirmed_mask[uid]) & (1 << TOPIC_FROM)) == 0:
                self._set_pending(uid, TOPIC_FROM); return (ev + self._p(uid, "country_evade_append"),), None, True
            if topic in (TOPIC_FROM, TOPIC_NAME) or (topic == TOPIC_AGE and (int(self.store.confirmed_mask[uid]) & (1 << TOPIC_AGE)) != 0):
                q, _ = self._next_stage_q(uid); return (f"{apol} {ev}".strip(),) + ((q,) if q else ()), None, bool(q)
            return (f"{apol} {ev}".strip(),), None, False

        self.store.bot_told_mask[uid] |= np.uint32(bt_bit)
        if topic == TOPIC_FROM:
            cname, _ = self._ensure_bot_country(uid); pc = self.cfg.persona
            if is_f and pc.india_enabled and cname.lower() == pc.india_country.lower():
                return (f"{apol} {self._pick(uid, pc.india_bubble1)}".strip(), self._pick(uid, pc.india_bubble2)), pc.india_disconnect_s, False
            tmpls = pc.country_templates_female if is_f else pc.country_templates_male
            q, _ = self._next_stage_q(uid)
            return (f"{apol} {tmpls[self._randint(0, len(tmpls) - 1)].format(country=cname)}".strip(),) + ((q,) if q else ()), None, bool(q)
        if topic == TOPIC_AGE:
            age = self._ensure_bot_age(uid)
            if (int(self.store.confirmed_mask[uid]) & (1 << TOPIC_AGE)) == 0:
                self._set_pending(uid, TOPIC_AGE); return (f"{apol} {age}, wbu?".strip(),), None, True
            q, _ = self._next_stage_q(uid); return (f"{apol} {age}".strip(),) + ((q,) if q else ()), None, bool(q)
        if topic == TOPIC_NAME:
            nm, tmpls = self._ensure_bot_name(uid), self.cfg.persona.name_templates
            q, _ = self._next_stage_q(uid)
            return (f"{apol} {tmpls[self._randint(0, len(tmpls) - 1)].format(name=nm)}".strip(),) + ((q,) if q else ()), None, bool(q)
        if topic == TOPIC_DOING:
            if user_answered_doing or (int(self.store.confirmed_mask[uid]) & (1 << TOPIC_DOING)) != 0:
                ans = self._p(uid, "hru_reciprocal") if self._rand() < pr.hru_reciprocal_when_answered_prob else self._p(uid, "hru_ans_female" if is_f else "hru_ans_male")
                q, _ = self._next_stage_q(uid); return (f"{apol} {ans}".strip(),) + ((q,) if q else ()), None, bool(q)
            if self._rand() < pr.hru_bounce_when_unanswered_prob:
                self._set_pending(uid, TOPIC_DOING); return (f"{apol} {self._p(uid, 'hru_bounce_female' if is_f else 'hru_bounce_male')}".strip(),), None, True
            q, _ = self._next_stage_q(uid)
            return (f"{apol} {self._p(uid, 'hru_ans_female' if is_f else 'hru_ans_male')}".strip(),) + ((q,) if q else ()), None, bool(q)
        q, _ = self._next_stage_q(uid)
        return (f"{apol} {self._p(uid, ANS_KEY_F[topic] if is_f else ANS_KEY_M[topic])}".strip(),) + ((q,) if q else ()), None, bool(q)

    def _split_bounce_and_q(self, utt):
        s, lead_q = " ".join(utt.strip().split()), False
        if len(s) >= 2 and s[0] == "?" and any(c.isalnum() for c in s[1:]):
            lead_q, s = True, s.lstrip("? ").strip()
        clean_end, has_b, stem = s.lower().rstrip("?!., ").strip(), False, s
        for ph in TRAILING_BOUNCE_2W:
            if clean_end.endswith(" " + ph):
                has_b, stem = True, clean_end[: -(len(ph) + 1)].rstrip(", ").strip(); break
        if not has_b:
            toks = clean_end.replace(",", " ").split()
            if len(toks) >= 2 and toks[-1] in TRAILING_BOUNCE_1W:
                has_b, stem = True, " ".join(toks[:-1]).strip()
        return s, lead_q, has_b, stem

    def _react_and_advance(self, uid, topic, read_d, nlu_cls, nlu_lf, extra_val=""):
        is_f, pr = self._is_f(uid), self.cfg.probs
        if topic == TOPIC_FROM and self._rand() < pr.proactive_state_ask_prob:
            st_q = self._p(uid, "proactive_state_ask"); self._set_pending(uid, TOPIC_STATE)
            if is_f and self._rand() < pr.female_react_prob:
                return self._finish(uid, "reply", nlu_cls, nlu_lf, read_d, (self._p(uid, "country_react_female").format(country=extra_val or "there"), st_q), asked_q=True)
            return self._finish(uid, "reply", nlu_cls, nlu_lf, read_d, (st_q,), asked_q=True)
        if not is_f:
            q, _ = self._next_stage_q(uid)
            if not q: return self._finish(uid, "disconnect", nlu_cls, nlu_lf, read_d, (), self.cfg.timers.disconnect_standard_s)
            return self._finish(uid, "reply", nlu_cls, nlu_lf, read_d, (self._p(uid, "male_short_ack"), q) if self._rand() < pr.male_normal_ack_prob else (q,), asked_q=True)

        rx = ""
        if self._rand() < pr.female_react_prob:
            if topic == TOPIC_AGE:
                ua = int(self.store.user_age[uid])
                rx = self._p(uid, "age_react_female_young" if ua <= self.cfg.persona.user_young_max else "age_react_female_older").format(age=ua)
            elif topic == TOPIC_FROM:
                rx = self._p(uid, "country_react_female").format(country=extra_val or "there")
            else:
                rx = self._p(uid, "doing_react_female")
            if topic in (TOPIC_AGE, TOPIC_FROM) and (int(self.store.bot_told_mask[uid]) & int(BT_NTMU)) == 0 and self._rand() < pr.female_proactive_ntmu_prob:
                rx += self._p(uid, "ntmu_proactive_append"); self.store.bot_told_mask[uid] |= BT_NTMU
        want_q = (topic == TOPIC_AGE) or (not rx) or (self._rand() < pr.female_advance_after_non_age_prob)
        q, _ = self._next_stage_q(uid, prefer_age_branch=(topic == TOPIC_AGE)) if want_q else (None, TOPIC_NONE)
        if rx and q:
            return self._finish(uid, "reply", nlu_cls, nlu_lf, read_d, (f"{rx}. {q}",) if self._rand() < pr.female_merge_bubble_prob else (rx, q), asked_q=True)
        return self._finish(uid, "reply", nlu_cls, nlu_lf, read_d, (rx,)) if rx else self._advance_or_fallback(uid, read_d, nlu_cls, nlu_lf)

    def _handle_media(self, uid, event_type, read_d):
        is_f, pr, t = self._is_f(uid), self.cfg.probs, self.cfg.timers
        self.store.streaks[uid, ST_DRY] = self.store.streaks[uid, ST_QMARK] = np.uint8(0)
        ms = min(255, int(self.store.streaks[uid, ST_MEDIA]) + 1); self.store.streaks[uid, ST_MEDIA] = np.uint8(ms)
        if ms >= 3: return self._finish(uid, "silent", 0, 0, read_d, ())
        if ms == 2:
            r = self._rand()
            if r < pr.media_s2_silent: return self._finish(uid, "silent", 0, 0, read_d, ())
            if r < pr.media_s2_silent + pr.media_s2_next_q: return self._advance_or_fallback(uid, read_d, 0, 0)
            return self._finish(uid, "reply", 0, 0, read_d, (self._p(uid, "media_s2_female" if is_f else "media_s2_male"),))
        if event_type == "spoiler":
            if not is_f:
                return self._finish(uid, "disconnect", 0, 0, read_d, (), t.disconnect_standard_s) if self._rand() < pr.media_s1_spoiler_m_disc else self._finish(uid, "reply", 0, 0, read_d, (self._p(uid, "media_spoiler_male"),))
            if self._rand() < pr.media_s1_spoiler_f_bye: return self._finish(uid, "disconnect", 0, 0, read_d, (self._p(uid, "media_spoiler_female_bye"),), t.disconnect_spoiler_or_bored_bye_s)
            return self._finish(uid, "silent", 0, 0, read_d, ()) if self._rand() < pr.media_s1_spoiler_f_silent_rest else self._finish(uid, "reply", 0, 0, read_d, (self._p(uid, "media_spoiler_female_react"),))
        if event_type in ("voice", "audio"):
            if self._rand() < (pr.media_s1_voice_f_silent if is_f else pr.media_s1_voice_m_silent): return self._finish(uid, "silent", 0, 0, read_d, ())
            rx = self._p(uid, "media_voice_female" if is_f else "media_voice_male")
        elif event_type in ("photo", "video", "visual"):
            r = self._rand()
            if not is_f:
                return self._advance_or_fallback(uid, read_d, 0, 0) if r < pr.media_s1_visual_m_skip else self._advance_or_fallback(uid, read_d, 0, 0, (self._p(uid, "media_visual_male"),))
            if r < pr.media_s1_visual_f_silent: return self._finish(uid, "silent", 0, 0, read_d, ())
            if r < pr.media_s1_visual_f_silent + pr.media_s1_visual_f_skip: return self._advance_or_fallback(uid, read_d, 0, 0)
            early = int(self.store.turn_count[uid]) <= 3 and (int(self.store.bot_told_mask[uid]) & int(BT_EARLY_PIC)) == 0
            if early: self.store.bot_told_mask[uid] |= BT_EARLY_PIC
            rx = self._p(uid, "media_visual_female_early" if early else "media_visual_female_late")
        else:
            r, sil, skp = self._rand(), (pr.media_s1_sticker_f_silent if is_f else pr.media_s1_sticker_m_silent), (pr.media_s1_sticker_f_skip if is_f else pr.media_s1_sticker_m_skip)
            if r < sil: return self._finish(uid, "silent", 0, 0, read_d, ())
            if r < sil + skp: return self._advance_or_fallback(uid, read_d, 0, 0)
            rx = self._p(uid, "media_sticker_female" if is_f else "media_sticker_male")
        if is_f and int(self.store.pending_q[uid]) != TOPIC_NONE:
            if int(self.store.pending_retries[uid]) == 0 and self._rand() < pr.media_f_pending_nudge_prob:
                self.store.pending_retries[uid] = np.uint8(1)
                return self._finish(uid, "reply", 0, 0, read_d, (rx + self._p(uid, "media_female_nudge"),), asked_q=True)
            self.store.pending_q[uid] = np.uint8(0)
            if self._rand() < pr.media_f_next_q_prob: return self._advance_or_fallback(uid, read_d, 0, 0, (rx,))
        return self._finish(uid, "reply", 0, 0, read_d, (rx,))

    def process_turn(self, uid, utterances, event_type="text", emoji="", now_s=None):
        uid = int(uid) % MAX_USERS
        now_i = int(time.time() if now_s is None else now_s) & 0xFFFFFFFF
        self._ensure_session(uid, now_i)
        if event_type == "gift":
            return self._finish(uid, "reply", 0, 0, np.float32(0.8), (self._p(uid, "gift_thank_you").format(emoji=emoji or "gift"),))

        utts = (utterances,) if isinstance(utterances, str) else tuple(u for u in utterances if u and u.strip())
        joined = " ".join(utts).strip()
        read_d = self._read_delay_s(joined if event_type == "text" else "media")

        leaves, ages, ctries, genders, lead_q_any, bounce_any = [], [], [], [], False, False
        for u_txt in utts:
            clean_u, lq, hb, stem = self._split_bounce_and_q(u_txt)
            lead_q_any, bounce_any = lead_q_any or lq, bounce_any or hb
            lf, av, cv = vm_forward_pass(self.model, self.slot, memoryview(clean_u.encode("ascii", errors="ignore")[:64]))
            if lf == 0 and hb and stem:
                lf, av, cv = vm_forward_pass(self.model, self.slot, memoryview(stem.encode("ascii", errors="ignore")[:64]))
            if lf == 7:
                return self._finish(uid, "disconnect", 7, 7, np.float32(0.5), (self._p(uid, "underage_reject"),), self.cfg.timers.disconnect_standard_s)
            if lf == 75:
                self.start_chat(uid, now_s=now_i)
                return self._finish(uid, "reply", 99, 75, np.float32(0.2), (self._p(uid, "reset_chat"),))
            if "?" in clean_u and not hb:
                pq = int(self.store.pending_q[uid])
                if lf == 3 and pq != TOPIC_AGE: lf, av = 2, 0
                elif lf == 5 and pq != TOPIC_FROM: lf, cv = 4, 0
            leaves.append(lf)
            if av >= self.cfg.persona.user_min_adult: ages.append(av)
            if cv > 0: ctries.append(cv)
            if lf in (12, 14): genders.append(2)
            elif lf in (13, 15): genders.append(1)

        primary_lf = next((x for x in leaves if x != 0), 0)
        primary_cls = int(self.model.leaf_cls[primary_lf]) if primary_lf < self.model.num_leaves else 0
        self.store.turn_count[uid] = np.uint8(min(255, int(self.store.turn_count[uid]) + 1))
        turn_n, pt, pr, t = int(self.store.turn_count[uid]), int(self.store.persona_type[uid]), self.cfg.probs, self.cfg.timers

        # P1: Male Ghoster
        if pt == P_MALE_GHOSTER:
            if turn_n == 1 and (int(self.store.bot_told_mask[uid]) & int(BT_GENDER)) == 0 and not genders:
                tag = self._male_opener_tag(uid)
                return self._finish(uid, "reply", primary_cls, primary_lf, read_d, (tag,), asked_q=("?" in tag))
            if genders:
                self.store.user_gender[uid] = np.uint8(genders[0]); self._confirm_topic(uid, TOPIC_GENDER)
                pre = ()
                if (int(self.store.bot_told_mask[uid]) & int(BT_GENDER)) == 0:
                    self.store.bot_told_mask[uid] |= BT_GENDER; pre = (self._p(uid, "gender_reveal_male"),)
                if self._rand() < self.cfg.persona.ghoster_convert_prob:
                    self.store.persona_type[uid] = np.uint8(P_MALE_NORMAL)
                    q_tid = pr.ghoster_convert_q_pool.pick(self._rand())
                    q = self._pick(uid, self.cfg.male_stage_pools[q_tid]); self._set_pending(uid, q_tid)
                    return self._finish(uid, "reply", primary_cls, primary_lf, read_d, pre + (q,), asked_q=True)
                return self._finish(uid, "disconnect", primary_cls, primary_lf, read_d, pre, np.float32(t.ghoster_disconnect_pool.pick(self._rand())))
            return self._finish(uid, "disconnect", primary_cls, primary_lf, read_d, (), np.float32(t.ghoster_disconnect_pool.pick(self._rand())))

        # P2: Male Turn-1 Empty
        if pt == P_MALE_NORMAL and turn_n == 1 and event_type == "text" and not joined:
            return self._finish(uid, "reply", 0, 0, read_d, (self._male_opener_tag(uid),), asked_q=True)

        # P3: Identical Repeat Spam (3x) OR Max Turns
        if event_type == "text" and joined:
            mh = self._fnv32(joined.lower())
            if mh == self.store.last_msg_hash[uid]:
                self.store.streaks[uid, ST_IDENT] = np.uint8(min(255, int(self.store.streaks[uid, ST_IDENT]) + 1))
            else:
                self.store.last_msg_hash[uid], self.store.streaks[uid, ST_IDENT] = mh, np.uint8(1)
            if int(self.store.streaks[uid, ST_IDENT]) >= 3:
                return self._finish(uid, "disconnect", primary_cls, primary_lf, read_d, (self._p(uid, "repeat_spam_bye"),), t.disconnect_standard_s)
        if turn_n >= int(self.store.max_turns[uid]):
            return self._finish(uid, "disconnect", primary_cls, primary_lf, read_d, (self._p(uid, "max_turns_bye"),), t.disconnect_standard_s)

        # P4: Media
        if event_type != "text":
            return self._handle_media(uid, event_type, read_d)

        # P5: Pure Question Marks
        if primary_lf == 96:
            qs = min(255, int(self.store.streaks[uid, ST_QMARK]) + 1)
            self.store.streaks[uid, ST_QMARK], self.store.streaks[uid, ST_MEDIA] = np.uint8(qs), np.uint8(0)
            if qs == 1:
                if self._peek_skip(uid) != TOPIC_NONE and self._rand() < pr.qmark_s1_recall_prob:
                    b, ds, aq = self._dispatch_topic(uid, self._pop_skip(uid), force_answer=True)
                    return self._finish(uid, "disconnect" if ds else "reply", 55, 96, read_d, b, ds, asked_q=aq)
                if self._rand() < pr.qmark_s1_skip_group_prob:
                    r2 = self._rand()
                    if r2 < pr.qmark_s1_skip_disc: return self._bored_disconnect(uid, read_d, 55, 96)
                    if r2 < pr.qmark_s1_skip_disc + pr.qmark_s1_skip_next_q: return self._advance_or_fallback(uid, read_d, 55, 96)
                    return self._finish(uid, "silent", 55, 96, read_d, ())
                return self._finish(uid, "reply", 55, 96, read_d, (self._p(uid, "confused_reply"),))
            if qs == 2:
                if self._rand() < pr.qmark_s2_skip_group_prob:
                    r2 = self._rand()
                    if r2 < pr.qmark_s2_skip_disc: return self._bored_disconnect(uid, read_d, 55, 96)
                    if r2 < pr.qmark_s2_skip_disc + pr.qmark_s2_skip_next_q: return self._advance_or_fallback(uid, read_d, 55, 96)
                    return self._finish(uid, "silent", 55, 96, read_d, ())
                return self._finish(uid, "reply", 55, 96, read_d, (self._p(uid, "qmark_annoyed"),))
            if self._rand() < pr.qmark_s3_bored_disc_prob: return self._bored_disconnect(uid, read_d, 55, 96)
            return self._advance_or_fallback(uid, read_d, 55, 96) if self._rand() < np.float32(0.5) else self._finish(uid, "silent", 55, 96, read_d, ())

        self.store.streaks[uid, ST_MEDIA] = self.store.streaks[uid, ST_QMARK] = np.uint8(0)

        # P6: Hard Disconnect Intents (bye / rude)
        if primary_lf in (54, 72):
            return self._finish(uid, "disconnect", primary_cls, primary_lf, read_d, (self._p(uid, "max_turns_bye" if primary_lf == 54 else ("bored_bye_female" if self._is_f(uid) else "bored_bye_male")),), t.disconnect_standard_s)

        # P7: Private Chat / Social ID
        if any(lf in (43, 44) for lf in leaves):
            ps = min(255, int(self.store.streaks[uid, ST_PRIVATE]) + 1); self.store.streaks[uid, ST_PRIVATE] = np.uint8(ps)
            if ps >= 3: return self._finish(uid, "disconnect", 29, 43, read_d, (self._p(uid, "private_annoyed"),), t.disconnect_standard_s)
            r = self._rand()
            msg = self._p(uid, "private_annoyed") if (ps == 2 or r < pr.private_strike1_annoyed) else self._p(uid, "private_stay" if r < pr.private_strike1_annoyed + pr.private_strike1_stay else "private_maybe")
            return self._advance_or_fallback(uid, read_d, 29, 43, (msg,)) if self._rand() < pr.private_followup_prob else self._finish(uid, "reply", 29, 43, read_d, (msg,))

        # P8: User Stating Gender
        if genders:
            self.store.user_gender[uid] = np.uint8(genders[0]); self._confirm_topic(uid, TOPIC_GENDER)
            if ages: self.store.user_age[uid] = np.uint8(ages[0]); self._confirm_topic(uid, TOPIC_AGE)
            pre = ()
            if (int(self.store.bot_told_mask[uid]) & int(BT_GENDER)) == 0:
                self.store.bot_told_mask[uid] |= BT_GENDER
                pre = (self._p(uid, "gender_reveal_female" if self._is_f(uid) else "gender_reveal_male"),)
            pg = self._p(uid, "post_gender_female" if self._is_f(uid) else "post_gender_male"); lp = pg.lower()
            if "from" in lp: self._set_pending(uid, TOPIC_FROM)
            elif "old" in lp or "age" in lp: self._set_pending(uid, TOPIC_AGE)
            elif "doing" in lp or "wyd" in lp: self._set_pending(uid, TOPIC_DOING)
            elif "name" in lp: self._set_pending(uid, TOPIC_NAME)
            elif "married" in lp or "gf" in lp: self._set_pending(uid, TOPIC_MARRIED)
            elif "brings" in lp: self._set_pending(uid, TOPIC_LOOK)
            return self._finish(uid, "reply", primary_cls, primary_lf, read_d, pre + (pg,), asked_q=("?" in pg))

        # P9: Nice to Meet You
        lo_join = joined.lower()
        if primary_lf == 99 or (primary_lf == 1 and "meet" in lo_join):
            self.store.bot_told_mask[uid] |= BT_NTMU; msg = self._p(uid, "ntmu_reply")
            return self._advance_or_fallback(uid, read_d, 58, 99, (msg,)) if self._rand() < pr.ntmu_followup_prob else self._finish(uid, "reply", 58, 99, read_d, (msg,))

        # P10: User Insistence
        if any(lf == 97 for lf in leaves):
            if ages: self.store.user_age[uid] = np.uint8(ages[0]); self._confirm_topic(uid, TOPIC_AGE)
            if ctries: self.store.user_country[uid] = np.uint8(ctries[0]); self._confirm_topic(uid, TOPIC_FROM)
            ins = min(255, int(self.store.streaks[uid, ST_INSIST]) + 1); self.store.streaks[uid, ST_INSIST] = np.uint8(ins)
            if ins >= 3: return self._bored_disconnect(uid, read_d, 56, 97)
            if ins == 1:
                if self._rand() < pr.insist_s1_reply_prob:
                    sk = self._pop_skip(uid)
                    if sk != TOPIC_NONE:
                        ap = self._p(uid, "insist_apology_female" if self._is_f(uid) else "insist_apology_male") if self._rand() < pr.insist_s1_apology_prob else ""
                        b, ds, aq = self._dispatch_topic(uid, sk, force_answer=True, apol=ap)
                        return self._finish(uid, "disconnect" if ds else "reply", 56, 97, read_d, b, ds, asked_q=aq)
                    df = self._p(uid, "insist_deflect_female" if self._is_f(uid) else "insist_deflect_male")
                    return self._advance_or_fallback(uid, read_d, 56, 97, (df,)) if self._rand() < pr.insist_s1_deflect_followup_prob else self._finish(uid, "reply", 56, 97, read_d, (df,))
                r2 = self._rand()
                if r2 < pr.insist_s1_ignore_disc: return self._bored_disconnect(uid, read_d, 56, 97)
                return self._advance_or_fallback(uid, read_d, 56, 97) if r2 < pr.insist_s1_ignore_disc + pr.insist_s1_ignore_next_q else self._finish(uid, "silent", 56, 97, read_d, ())
            sk = self._pop_skip(uid)
            if sk != TOPIC_NONE and self._rand() < pr.insist_s2_answer_prob:
                b, ds, aq = self._dispatch_topic(uid, sk, force_answer=True)
                return self._finish(uid, "disconnect" if ds else "reply", 56, 97, read_d, b, ds, asked_q=aq)
            if self._rand() < pr.insist_s2_skip_group_prob:
                r2 = self._rand()
                if r2 < pr.insist_s2_skip_disc: return self._bored_disconnect(uid, read_d, 56, 97)
                return self._advance_or_fallback(uid, read_d, 56, 97) if r2 < pr.insist_s2_skip_disc + pr.insist_s2_skip_next_q else self._finish(uid, "silent", 56, 97, read_d, ())
            return self._finish(uid, "reply", 56, 97, read_d, (self._p(uid, "insist_pushback_female" if self._is_f(uid) else "insist_pushback_male"),))

        # P11: Horny / Sexual Check
        if any(lf == 101 for lf in leaves):
            hs = min(255, int(self.store.streaks[uid, ST_HORNY]) + 1); self.store.streaks[uid, ST_HORNY] = np.uint8(hs); is_f = self._is_f(uid)
            if hs >= 3:
                return self._finish(uid, "disconnect", 60, 101, read_d, (self._p(uid, "repeat_spam_bye"),), t.disconnect_standard_s) if is_f else self._finish(uid, "disconnect", 60, 101, read_d, (), self._uni(t.disconnect_horny_m3_min_s, t.disconnect_horny_m3_max_s))
            if hs == 2:
                if not is_f and self._rand() < pr.horny_m_s2_disc:
                    return self._finish(uid, "disconnect", 60, 101, read_d, (), self._uni(t.disconnect_horny_m2_min_s, t.disconnect_horny_m2_max_s))
                q, _ = self._next_stage_q(uid); return self._finish(uid, "reply", 60, 101, read_d, (q if q else self._p(uid, "repeat_warn"),), asked_q=bool(q))
            r = self._rand()
            if not is_f:
                if r < pr.horny_m_s1_skip: self._push_skip(uid, TOPIC_HORNY); return self._advance_or_fallback(uid, read_d, 60, 101)
                if r < pr.horny_m_s1_skip + pr.horny_m_s1_disc: return self._finish(uid, "disconnect", 60, 101, read_d, (), self._uni(t.disconnect_horny_m1_min_s, t.disconnect_horny_m1_max_s))
                msg = self._p(uid, "horny_male_reply")
                if int(self.store.user_gender[uid]) == 1 and self._rand() < pr.horny_m_s1_reply_male_user_disc:
                    return self._finish(uid, "disconnect", 60, 101, read_d, (msg,), self._uni(t.disconnect_horny_m1_male_user_min_s, t.disconnect_horny_m1_male_user_max_s))
                return self._advance_or_fallback(uid, read_d, 60, 101, (msg,)) if self._rand() < pr.horny_m_s1_followup else self._finish(uid, "reply", 60, 101, read_d, (msg,))
            if r < pr.horny_f_s1_skip: self._push_skip(uid, TOPIC_HORNY); return self._advance_or_fallback(uid, read_d, 60, 101)
            if r < pr.horny_f_s1_skip + pr.horny_f_s1_silent:
                q, _ = self._next_stage_q(uid); return self._finish(uid, "reply" if q else "silent", 60, 101, read_d, ((q,) if q else ()), asked_q=bool(q))
            msg = self._p(uid, "horny_female_reply" if any(w in self.cfg.val_kw.horny_statement_markers for w in lo_join.split()) else "horny_female_reply_ask")
            return self._advance_or_fallback(uid, read_d, 60, 101, (msg,)) if self._rand() < pr.horny_f_s1_followup else self._finish(uid, "reply", 60, 101, read_d, (msg,))

        # P12: Bounce-Only
        if primary_lf == 76 and len(leaves) == 1:
            self.store.streaks[uid, ST_DRY] = np.uint8(0)
            t_pick = int(self.store.pending_q[uid]) or int(self.store.last_topic[uid]) or int(self.store.last_confirmed_topic[uid]) or self._peek_skip(uid)
            if t_pick != TOPIC_NONE:
                b, ds, aq = self._dispatch_topic(uid, t_pick)
                if b or ds is not None: return self._finish(uid, "disconnect" if ds else "reply", 0, 76, read_d, b, ds, asked_q=aq)
            return self._advance_or_fallback(uid, read_d, 0, 76)

        # P13: Short Tokens (Confused -> Dry -> Yes -> No)
        raw_strip, clean_tok, vk = lo_join.strip(), lo_join.rstrip("?!., ").strip(), self.cfg.val_kw
        if len(leaves) == 1 and (primary_lf == 70 or clean_tok in vk.confused_tokens):
            return self._finish(uid, "reply", primary_cls, primary_lf, read_d, (self._p(uid, "confused_reply"),))
        if len(leaves) == 1 and (primary_lf == 100 or clean_tok in vk.dry_tokens or raw_strip in vk.dry_tokens):
            ds = min(255, int(self.store.streaks[uid, ST_DRY]) + 1); self.store.streaks[uid, ST_DRY] = np.uint8(ds)
            if ds >= 3: return self._finish(uid, "disconnect", primary_cls, primary_lf, read_d, (self._p(uid, "dry_bye"),), t.disconnect_standard_s)
            if self._is_f(uid) and int(self.store.pending_q[uid]) != TOPIC_NONE:
                return self._finish(uid, "reply", primary_cls, primary_lf, read_d, (self._p(uid, "dry_pending_react"),))
            return self._finish(uid, "reply", primary_cls, primary_lf, read_d, (self._p(uid, "dry_s1_no_q" if ds == 1 else "dry_s2_no_q").format(word=clean_tok or raw_strip or "ok"),))

        self.store.streaks[uid, ST_DRY] = np.uint8(0)
        if len(leaves) == 1 and (primary_lf in (45, 46) or clean_tok in vk.yes_tokens or clean_tok in vk.no_tokens):
            is_yes, pq = ((primary_lf == 45) or (clean_tok in vk.yes_tokens)), int(self.store.pending_q[uid])
            if self._is_f(uid) and pq != TOPIC_NONE:
                if pq in YN_ACCEPT_TOPICS:
                    self._confirm_topic(uid, pq); rx = self._p(uid, "pending_yn_react")
                    return self._advance_or_fallback(uid, read_d, primary_cls, primary_lf, (rx,)) if self._rand() < pr.yn_pending_followup_prob else self._finish(uid, "reply", primary_cls, primary_lf, read_d, (rx,))
                ret = int(self.store.pending_retries[uid]) + 1; self.store.pending_retries[uid] = np.uint8(ret)
                if ret == 1: return self._finish(uid, "reply", primary_cls, primary_lf, read_d, (self._p(uid, "pending_invalid_react"),))
                self.store.pending_q[uid] = np.uint8(0); return self._advance_or_fallback(uid, read_d, primary_cls, primary_lf)
            if not is_yes: return self._finish(uid, "reply", primary_cls, primary_lf, read_d, (self._p(uid, "generic_no"),))
            ds = min(255, int(self.store.streaks[uid, ST_DRY]) + 1); self.store.streaks[uid, ST_DRY] = np.uint8(ds)
            return self._finish(uid, "reply", primary_cls, primary_lf, read_d, (self._p(uid, "dry_s1_no_q" if ds == 1 else "dry_s2_no_q").format(word=clean_tok or "yes"),))

        # P14: Substantive Content Turn
        lead_bubbles = []
        if lead_q_any and self._peek_skip(uid) != TOPIC_NONE and self._rand() < pr.leading_q_recall_prob:
            b, ds, _ = self._dispatch_topic(uid, self._pop_skip(uid), force_answer=True)
            if ds is not None: return self._finish(uid, "disconnect", primary_cls, primary_lf, read_d, b, ds)
            lead_bubbles.extend(b)

        confirmed_topic, confirmed_extra = TOPIC_NONE, ""
        if ctries:
            cid = ctries[0]; self.store.user_country[uid] = np.uint8(cid); self._confirm_topic(uid, TOPIC_FROM)
            confirmed_topic, confirmed_extra = TOPIC_FROM, (COUNTRY_ID_NAMES[cid] if cid < len(COUNTRY_ID_NAMES) else "there")
        if ages:
            self.store.user_age[uid] = np.uint8(ages[0]); self._confirm_topic(uid, TOPIC_AGE)
            if confirmed_topic == TOPIC_NONE: confirmed_topic = TOPIC_AGE
        if any(lf in (17, 18, 19, 77) for lf in leaves):
            self.store.user_mood[uid] = np.uint8(1); self._confirm_topic(uid, TOPIC_DOING)
            if confirmed_topic == TOPIC_NONE: confirmed_topic = TOPIC_DOING
        if any(lf == 58 for lf in leaves):
            self.store.user_job[uid] = np.uint8(1); self._confirm_topic(uid, TOPIC_JOB)
            if confirmed_topic == TOPIC_NONE: confirmed_topic = TOPIC_JOB
        if any(lf in (40, 41, 42) for lf in leaves):
            self.store.user_rel[uid] = np.uint8(1); self._confirm_topic(uid, TOPIC_MARRIED)
            if confirmed_topic == TOPIC_NONE: confirmed_topic = TOPIC_MARRIED
        if any(lf in (23, 24, 25, 26) for lf in leaves):
            self._confirm_topic(uid, TOPIC_LOOK)
            if confirmed_topic == TOPIC_NONE: confirmed_topic = TOPIC_LOOK
        if any(lf == 28 for lf in leaves):
            self._confirm_topic(uid, TOPIC_NAME)
            if confirmed_topic == TOPIC_NONE: confirmed_topic = TOPIC_NAME

        if (bounce_any or 19 in leaves) and confirmed_topic != TOPIC_NONE:
            b, ds, aq = self._dispatch_topic(uid, confirmed_topic, user_answered_doing=(confirmed_topic == TOPIC_DOING))
            return self._finish(uid, "disconnect" if ds else "reply", primary_cls, primary_lf, read_d, tuple(lead_bubbles) + b, ds, asked_q=aq)

        for lf in leaves:
            if lf < 128 and LEAF_TO_Q_TOPIC[lf] != TOPIC_NONE:
                b, ds, aq = self._dispatch_topic(uid, LEAF_TO_Q_TOPIC[lf], user_yn_gender=(lf in (10, 11)), user_answered_doing=(confirmed_topic == TOPIC_DOING))
                return self._finish(uid, "disconnect" if ds else "reply", primary_cls, lf, read_d, tuple(lead_bubbles) + b, ds, asked_q=aq)

        if any(lf == 1 for lf in leaves) and confirmed_topic == TOPIC_NONE:
            if (int(self.store.bot_told_mask[uid]) & int(BT_GREETED)) != 0:
                return self._advance_or_fallback(uid, read_d, 1, 1, lead_bubbles)
            self.store.bot_told_mask[uid] |= BT_GREETED
            if self._rand() < pr.greet_reply_prob:
                gr = (joined.capitalize() if self._rand() < pr.greet_mirror_cap_prob else joined.lower()) if (2 <= len(joined) <= 12 and self._rand() < pr.greet_mirror_prob) else self._pick(uid, self.cfg.opening.female_greetings if self._is_f(uid) else self.cfg.opening.male_greetings)
                if self._rand() < pr.greet_followup_prob:
                    return self._advance_or_fallback(uid, read_d, 1, 1, tuple(lead_bubbles) + (gr,)) if self._is_f(uid) else self._finish(uid, "reply", 1, 1, read_d, tuple(lead_bubbles) + (gr, self._male_opener_tag(uid)), asked_q=True)
                return self._finish(uid, "reply", 1, 1, read_d, tuple(lead_bubbles) + (gr,))
            return self._advance_or_fallback(uid, read_d, 1, 1, lead_bubbles) if self._is_f(uid) else self._finish(uid, "reply", 1, 1, read_d, tuple(lead_bubbles) + (self._male_opener_tag(uid),), asked_q=True)

        if confirmed_topic != TOPIC_NONE:
            return self._react_and_advance(uid, confirmed_topic, read_d, primary_cls, primary_lf, confirmed_extra)

        if primary_lf != 0 and primary_lf < len(self.cfg.ext_replies) and self.cfg.ext_replies[primary_lf] is not None:
            ans = self._pick(uid, self.cfg.ext_replies[primary_lf])
            if self._rand() < pr.extended_intent_followup_prob and "?" not in ans:
                return self._advance_or_fallback(uid, read_d, primary_cls, primary_lf, tuple(lead_bubbles) + (ans,))
            return self._finish(uid, "reply", primary_cls, primary_lf, read_d, tuple(lead_bubbles) + (ans,), asked_q=("?" in ans))

        pq = int(self.store.pending_q[uid])
        if pq != TOPIC_NONE and joined:
            toks = [w.strip("?!.,'") for w in lo_join.split() if w.strip("?!.,'")]
            is_cq = ("?" in lo_join) or any(w in vk.counter_question_words for w in toks)
            is_filler = all(w in vk.false_text_fillers for w in toks) if toks else True
            valid_open = (not is_cq and not is_filler) and ((any(w in vk.job_keywords for w in toks)) if pq == TOPIC_JOB else (pq in OPEN_PENDING_TOPICS and (len(toks) >= 2 or len(clean_tok) >= 5)))
            if valid_open:
                self._confirm_topic(uid, pq); return self._react_and_advance(uid, pq, read_d, primary_cls, primary_lf)
            ret = int(self.store.pending_retries[uid]) + 1; self.store.pending_retries[uid] = np.uint8(ret)
            if self._is_f(uid) and ret >= 2 and pq in (TOPIC_JOB, TOPIC_DOING) and (int(self.store.extra_flags[uid]) & int(EF_BAD_ENG_ASKED)) == 0 and self._rand() < pr.bad_english_retry2_prob:
                self.store.extra_flags[uid] |= EF_BAD_ENG_ASKED
                return self._finish(uid, "reply", primary_cls, primary_lf, read_d, (self._p(uid, "bad_english_check"),), asked_q=True)
            if ret == 1: return self._finish(uid, "reply", primary_cls, primary_lf, read_d, (self._p(uid, "pending_invalid_react"),))
            self.store.pending_q[uid] = np.uint8(0)

        return self._advance_or_fallback(uid, read_d, primary_cls, primary_lf, lead_bubbles, allow_bad_eng=True)


if __name__ == "__main__":
    m = CompiledVMModel("model_weights.npz")
    c = load_behavior_config("bot_behavior_config.json")
    d = BehaviorDirector(m, c, seed=1337)
    allowed = (np.uint8, np.uint16, np.uint32, np.int32, np.float32, np.bool_)
    for s in d.store.__slots__:
        assert getattr(d.store, s).dtype in allowed, f"Forbidden dtype on {s}"
    # Zero-Fault Behavior Gate
    d.start_chat(1, now_s=10000); d.store.persona_type[1] = np.uint8(P_FEMALE)
    ob = d.build_opening_burst(1, now_s=10001)
    assert ob.action == "opening_burst" and 1 <= len(ob.bubbles) <= 4 and 23.0 <= float(ob.idle_timeout_s) <= 35.0
    r1 = d.process_turn(1, ["hi", "im 24", "from tunisia"], now_s=10005)
    assert int(d.store.user_age[1]) == 24 and int(d.store.user_country[1]) == 1 and r1.action == "reply"
    r_ua = d.process_turn(2, "im 15", now_s=10006)
    assert r_ua.action == "disconnect" and r_ua.nlu_class == 7
    d.start_chat(3, now_s=10010); d.store.persona_type[3] = np.uint8(P_FEMALE)
    for _ in range(3): r_dry = d.process_turn(3, "ok", now_s=10011)
    assert r_dry.action == "disconnect" and float(r_dry.disconnect_after_s) == 1.0
    d.start_chat(4, now_s=10020); d.store.persona_type[4] = np.uint8(P_FEMALE); d.store.bot_country[4] = np.uint8(c.persona.country_to_idx["India"])
    b_ind, ds_ind, _ = d._dispatch_topic(4, TOPIC_FROM, force_answer=True)
    assert len(b_ind) == 2 and float(ds_ind) == 4.0
    d.start_chat(5, now_s=10030); d.store.persona_type[5] = np.uint8(P_FEMALE)
    r_fast = d.process_turn(5, "Bcz you make it to fast.", now_s=10031)
    assert r_fast.nlu_leaf == 90 and r_fast.action == "reply"
    print(f"[BEHAVIOR ENGINE ZERO-FAULT OK] 50,000-user BehaviorStore: {d.store.total_nbytes() / 1048576.0:.2f} MB | ZERO float64/int64 | All gates passed!", flush=True)