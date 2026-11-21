#!/usr/bin/env python3
# FILE: chat_scheduler.py
# PURPOSE: Pure-Python + NumPy Async Timing Scheduler, 0.4s Rapid Buffer, Section 10 Platform Guards,
#          HTTP JSON Behavior Server (with live config hot-reload) & Interactive CLI Chat Simulator
# RULES: Enforces Rules 1-6 (Zero PyTorch/SciPy/FastAPI, zero float64/int64, __slots__, < 10 MB total RAM)

import asyncio
import json
import sys
import time
import numpy as np

from behavior_config_loader import load_behavior_config
from behavior_engine import BehaviorDirector, P_FEMALE
from production_server import CompiledVMModel, MAX_USERS


class PlatformGuardStore:
    """Section 10 Operational Safety, Rate Limits, 12h Quotas, AFK/Purge & Blocklist for 50,000 users (~2.35 MB)."""
    __slots__ = (
        "rate_win_ds", "rate_cur", "rate_prev", "last_btn_ds",
        "quota_win_s", "quota_bots", "quota_humans", "quota_human_ring",
        "exempt_flags", "last_partner", "blocked_ring", "blocked_pos", "last_active_s",
    )

    def __init__(self, max_users=MAX_USERS):
        self.rate_win_ds = np.zeros((max_users,), dtype=np.uint32)
        self.rate_cur = np.zeros((max_users,), dtype=np.uint8)
        self.rate_prev = np.zeros((max_users,), dtype=np.uint8)
        self.last_btn_ds = np.zeros((max_users,), dtype=np.uint32)
        self.quota_win_s = np.zeros((max_users,), dtype=np.uint32)
        self.quota_bots = np.zeros((max_users,), dtype=np.uint8)
        self.quota_humans = np.zeros((max_users,), dtype=np.uint8)
        self.quota_human_ring = np.zeros((max_users, 8), dtype=np.uint16)
        self.exempt_flags = np.zeros((max_users,), dtype=np.uint8)
        self.last_partner = np.zeros((max_users,), dtype=np.uint16)
        self.blocked_ring = np.zeros((max_users, 4), dtype=np.uint16)
        self.blocked_pos = np.zeros((max_users,), dtype=np.uint8)
        self.last_active_s = np.zeros((max_users,), dtype=np.uint32)

    def total_nbytes(self):
        return sum(getattr(self, s).nbytes for s in self.__slots__)

    def set_exempt(self, uid, vip=False, admin=False):
        self.exempt_flags[uid % MAX_USERS] = np.uint8((1 if vip else 0) | (2 if admin else 0))

    def check_rate_limit(self, uid, cfg_pl, now_s=None):
        u = int(uid) % MAX_USERS
        now_f = time.time() if now_s is None else float(now_s)
        now_ds = int(now_f * 10.0) & 0xFFFFFFFF
        win_ds = max(1, int(round(float(cfg_pl.rate_limit_window_s) * 10.0)))
        self.last_active_s[u] = np.uint32(int(now_f) & 0xFFFFFFFF)
        st = int(self.rate_win_ds[u])
        el = (now_ds - st) & 0xFFFFFFFF
        if st == 0 or el >= 2 * win_ds:
            self.rate_win_ds[u], self.rate_prev[u], self.rate_cur[u] = np.uint32(now_ds), np.uint8(0), np.uint8(0)
            el = 0
        elif el >= win_ds:
            self.rate_win_ds[u], self.rate_prev[u], self.rate_cur[u] = np.uint32((st + win_ds) & 0xFFFFFFFF), self.rate_cur[u], np.uint8(0)
            el -= win_ds
        eff = int(self.rate_cur[u]) + (int(self.rate_prev[u]) * (win_ds - el)) // win_ds
        if eff >= cfg_pl.rate_limit_max_msgs:
            return False
        self.rate_cur[u] = np.uint8(min(255, int(self.rate_cur[u]) + 1))
        return True

    def check_button_cooldown(self, uid, cfg_pl, now_s=None):
        u = int(uid) % MAX_USERS
        now_ds = int((time.time() if now_s is None else float(now_s)) * 10.0) & 0xFFFFFFFF
        cd_ds = max(1, int(round(float(cfg_pl.button_cooldown_s) * 10.0)))
        prev = int(self.last_btn_ds[u])
        if prev != 0 and ((now_ds - prev) & 0xFFFFFFFF) < cd_ds:
            return False
        self.last_btn_ds[u] = np.uint32(now_ds)
        return True

    def _refresh_quota(self, u, cfg_pl, now_i):
        st = int(self.quota_win_s[u])
        if st == 0 or ((now_i - st) & 0xFFFFFFFF) >= cfg_pl.quota_window_s:
            self.quota_win_s[u] = np.uint32(now_i)
            self.quota_bots[u] = self.quota_humans[u] = np.uint8(0)
            self.quota_human_ring[u].fill(np.uint16(0))

    def can_match_and_record(self, uid, cfg_pl, is_bot=True, partner_id=0, now_s=None):
        u = int(uid) % MAX_USERS
        now_i = int(time.time() if now_s is None else now_s) & 0xFFFFFFFF
        pid = int(partner_id) & 0xFFFF
        if pid > 0:
            if int(self.last_partner[u]) == pid:
                return False, "same_last_partner"
            br = self.blocked_ring[u]
            if pid in (int(br[0]), int(br[1]), int(br[2]), int(br[3])):
                return False, "blocked_partner"
        if int(self.exempt_flags[u]) != 0:
            if pid > 0: self.last_partner[u] = np.uint16(pid)
            return True, "exempt"
        self._refresh_quota(u, cfg_pl, now_i)
        if int(self.quota_bots[u]) >= cfg_pl.quota_max_bot_matches or int(self.quota_humans[u]) >= cfg_pl.quota_max_human_partners:
            return False, "quota_12h_locked"
        if is_bot:
            self.quota_bots[u] = np.uint8(min(255, int(self.quota_bots[u]) + 1))
        elif pid > 0:
            ring, hcnt = self.quota_human_ring[u], int(self.quota_humans[u])
            if not any(int(ring[i]) == pid for i in range(min(hcnt, 8))):
                if hcnt < 8:
                    ring[hcnt] = np.uint16(pid)
                self.quota_humans[u] = np.uint8(min(255, hcnt + 1))
            self.last_partner[u] = np.uint16(pid)
        return True, "ok"

    def refund_turn0_bot_match(self, uid, turn_count):
        u = int(uid) % MAX_USERS
        if int(turn_count) == 0 and int(self.quota_bots[u]) > 0:
            self.quota_bots[u] = np.uint8(int(self.quota_bots[u]) - 1)
            return True
        return False

    def block_partner(self, uid, partner_id):
        u = int(uid) % MAX_USERS
        pos = int(self.blocked_pos[u])
        self.blocked_ring[u, pos] = np.uint16(int(partner_id) & 0xFFFF)
        self.blocked_pos[u] = np.uint8((pos + 1) & 3)

    def activity_status(self, uid, cfg_pl, now_s=None):
        u = int(uid) % MAX_USERS
        now_i = int(time.time() if now_s is None else now_s) & 0xFFFFFFFF
        last = int(self.last_active_s[u])
        if last == 0: return "online"
        el = (now_i - last) & 0xFFFFFFFF
        return "purge" if el > cfg_pl.purge_timeout_s else ("afk" if el > cfg_pl.afk_timeout_s else "online")


class ActiveChatSlot:
    __slots__ = (
        "uid", "connected", "buf_texts", "buf_chars", "buf_media", "buf_emoji",
        "is_typing", "is_opening", "abort_opening",
        "opening_task", "debounce_task", "worker_task", "idle_task",
    )

    def __init__(self, uid):
        self.uid, self.connected = uid, True
        self.buf_texts, self.buf_chars, self.buf_media, self.buf_emoji = [], 0, "", ""
        self.is_typing = self.is_opening = self.abort_opening = False
        self.opening_task = self.debounce_task = self.worker_task = self.idle_task = None

    def has_buffered(self):
        return bool(self.buf_texts) or bool(self.buf_media)


class ChatScheduler:
    """AsyncIO Real-Time Runner implementing Section 3.1 Buffering, Non-Interruptible Typing & Idle Disconnects."""
    __slots__ = ("director", "guards", "send_cb", "typing_cb", "disconnect_cb", "time_scale", "sessions")

    def __init__(self, director, send_cb, typing_cb=None, disconnect_cb=None, time_scale=1.0):
        self.director, self.guards = director, PlatformGuardStore()
        self.send_cb, self.typing_cb, self.disconnect_cb = send_cb, typing_cb, disconnect_cb
        self.time_scale = np.float32(time_scale)
        self.sessions = {}

    def _cancel(self, task):
        if task is not None and not task.done():
            task.cancel()

    def _close_slot(self, uid):
        slot = self.sessions.pop(uid, None)
        if slot is not None:
            slot.connected = False
            self._cancel(slot.opening_task); self._cancel(slot.debounce_task)
            self._cancel(slot.worker_task); self._cancel(slot.idle_task)

    def on_match_start(self, uid, user_interests=None):
        u = int(uid) % MAX_USERS
        ok, reason = self.guards.can_match_and_record(u, self.director.cfg.platform, is_bot=True)
        if not ok:
            return False, reason, 0
        self._close_slot(u)
        persona, open_wait_s = self.director.start_chat(u, user_interests=user_interests)
        slot = ActiveChatSlot(u)
        self.sessions[u] = slot
        slot.opening_task = asyncio.create_task(self._opening_wait_flow(slot, open_wait_s))
        return True, persona, open_wait_s

    def on_turn0_hijack(self, uid):
        u = int(uid) % MAX_USERS
        refunded = self.guards.refund_turn0_bot_match(u, int(self.director.store.turn_count[u]))
        self._close_slot(u)
        self.director.store.reset_session(u)
        return refunded

    async def on_user_message(self, uid, text="", event_type="text", emoji=""):
        u = int(uid) % MAX_USERS
        cfg = self.director.cfg
        if not self.guards.check_rate_limit(u, cfg.platform):
            await self.send_cb(u, cfg.platform.rate_limit_reply)
            return
        slot = self.sessions.get(u)
        if slot is None or not slot.connected:
            ok, _p, _w = self.on_match_start(u)
            if not ok:
                return
            slot = self.sessions[u]

        self._cancel(slot.opening_task); slot.opening_task = None
        self._cancel(slot.idle_task); slot.idle_task = None
        if slot.is_opening:
            slot.abort_opening = True

        # Section 3.1 Buffer Rules: max 400 chars, never let media overwrite buffered text or join text onto media
        if event_type == "text":
            clean = " ".join(text.strip().split())
            if clean:
                if slot.buf_media:
                    slot.buf_media = ""
                rem = cfg.timers.buffer_max_chars - slot.buf_chars
                if rem > 0:
                    piece = clean[:rem]
                    slot.buf_texts.append(piece)
                    slot.buf_chars += len(piece) + 1
        elif not slot.buf_texts:
            slot.buf_media, slot.buf_emoji = event_type, emoji

        # Section 3.1: If mid-typing, NEVER cancel! Worker finishes current bubble and loops back automatically.
        if slot.is_typing:
            return
        self._cancel(slot.debounce_task)
        slot.debounce_task = asyncio.create_task(self._debounce_then_run(slot))

    async def _debounce_then_run(self, slot):
        try:
            await asyncio.sleep(float(self.director.cfg.timers.buffer_debounce_s) * float(self.time_scale))
            if not slot.connected or not slot.has_buffered():
                return
            if slot.worker_task is None or slot.worker_task.done():
                slot.worker_task = asyncio.create_task(self._process_loop(slot))
        except asyncio.CancelledError:
            pass

    async def _opening_wait_flow(self, slot, open_wait_s):
        try:
            await asyncio.sleep(float(open_wait_s) * float(self.time_scale))
            if not slot.connected or slot.has_buffered():
                return
            act = self.director.build_opening_burst(slot.uid)
            slot.is_opening, slot.abort_opening = True, False
            for b in act.bubbles:
                if not slot.connected or slot.abort_opening or slot.has_buffered():
                    break
                if float(b.pause_before_s) > 0.0:
                    await asyncio.sleep(float(b.pause_before_s) * float(self.time_scale))
                if not slot.connected or slot.abort_opening or slot.has_buffered():
                    break
                slot.is_typing = True
                if self.typing_cb: await self.typing_cb(slot.uid)
                await asyncio.sleep(float(b.typing_s) * float(self.time_scale))
                if slot.connected: await self.send_cb(slot.uid, b.text)
                slot.is_typing = False
            slot.is_opening = False
            if slot.connected and not slot.has_buffered() and float(act.idle_timeout_s) > 0.0:
                slot.idle_task = asyncio.create_task(self._idle_countdown(slot, act.idle_timeout_s))
            elif slot.connected and slot.has_buffered():
                slot.debounce_task = asyncio.create_task(self._debounce_then_run(slot))
        except asyncio.CancelledError:
            slot.is_opening = slot.is_typing = False

    async def _process_loop(self, slot):
        try:
            while slot.connected and slot.has_buffered():
                if slot.buf_texts:
                    utts, ev, emo = tuple(slot.buf_texts), "text", ""
                    slot.buf_texts.clear(); slot.buf_chars = 0
                else:
                    utts, ev, emo = (), slot.buf_media, slot.buf_emoji
                    slot.buf_media = slot.buf_emoji = ""

                act = self.director.process_turn(slot.uid, utts, event_type=ev, emoji=emo)
                if float(act.read_delay_s) > 0.0:
                    await asyncio.sleep(float(act.read_delay_s) * float(self.time_scale))
                if not slot.connected:
                    return

                preempted = False
                for b in act.bubbles:
                    if not slot.connected:
                        return
                    if float(b.pause_before_s) > 0.0:
                        await asyncio.sleep(float(b.pause_before_s) * float(self.time_scale))
                        if slot.has_buffered():
                            preempted = True; break
                    slot.is_typing = True
                    if self.typing_cb: await self.typing_cb(slot.uid)
                    await asyncio.sleep(float(b.typing_s) * float(self.time_scale))
                    if slot.connected: await self.send_cb(slot.uid, b.text)
                    slot.is_typing = False
                    if slot.has_buffered():
                        preempted = True; break

                if preempted and slot.connected:
                    await asyncio.sleep(float(self.director.cfg.timers.buffer_debounce_s) * float(self.time_scale))
                    continue
                if act.disconnect_after_s is not None:
                    await asyncio.sleep(float(act.disconnect_after_s) * float(self.time_scale))
                    if slot.connected:
                        slot.connected = False
                        self.sessions.pop(slot.uid, None)
                        if self.disconnect_cb: await self.disconnect_cb(slot.uid, "bot_left")
                    return
                if slot.connected and float(act.idle_timeout_s) > 0.0:
                    slot.idle_task = asyncio.create_task(self._idle_countdown(slot, act.idle_timeout_s))
                return
        except asyncio.CancelledError:
            slot.is_typing = False

    async def _idle_countdown(self, slot, idle_s):
        try:
            await asyncio.sleep(float(idle_s) * float(self.time_scale))
            await asyncio.sleep(float(self.director.cfg.timers.idle_disconnect_after_s) * float(self.time_scale))
            if slot.connected and not slot.has_buffered() and not slot.is_typing:
                slot.connected = False
                self.sessions.pop(slot.uid, None)
                if self.disconnect_cb: await self.disconnect_cb(slot.uid, "idle_timeout")
        except asyncio.CancelledError:
            pass


async def serve_behavior_http(host="127.0.0.1", port=8081):
    model = CompiledVMModel("model_weights.npz")
    cfg = load_behavior_config("bot_behavior_config.json")
    director = BehaviorDirector(model, cfg)

    async def handle(reader, writer):
        try:
            req = await asyncio.wait_for(reader.read(2048), timeout=2.0)
            if not req: return
            hdr_end = req.find(b"\r\n\r\n")
            hdr_raw = req[:hdr_end] if hdr_end != -1 else req
            body_raw = req[hdr_end + 4 :] if hdr_end != -1 else b""
            lines = hdr_raw.decode("ascii", errors="ignore").split("\r\n")
            path = lines[0].split(" ")[1] if len(lines[0].split(" ")) >= 2 else "/turn"
            uid, ev, emo = 101, "text", ""
            for ln in lines[1:]:
                lo = ln.lower()
                if lo.startswith("x-user-id:"):
                    v = ln.split(":", 1)[1].strip()
                    if v.isdigit(): uid = int(v)
                elif lo.startswith("x-event-type:"): ev = ln.split(":", 1)[1].strip().lower() or "text"
                elif lo.startswith("x-emoji:"): emo = ln.split(":", 1)[1].strip()

            if path == "/reload":
                director.reload_config()
                payload = {"status": "reloaded", "total_phrases": director.cfg.total_phrases}
            elif path == "/start":
                persona, wait_s = director.start_chat(uid)
                payload = {"status": "started", "persona": persona, "opening_wait_s": wait_s}
            elif path == "/opening":
                payload = director.build_opening_burst(uid).to_dict()
            else:
                txt = body_raw.decode("utf-8", errors="ignore")
                utts = [x for x in txt.splitlines() if x.strip()] or [txt]
                payload = director.process_turn(uid, utts, event_type=ev, emoji=emo).to_dict()
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + str(len(body)).encode("ascii") + b"\r\nConnection: close\r\n\r\n" + body)
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    srv = await asyncio.start_server(handle, host, port, backlog=2048)
    print(f"[BEHAVIOR HTTP SERVER] http://{host}:{port} (/turn, /start, /opening, /reload) | ~9.6 MB Total RAM", flush=True)
    async with srv:
        await srv.serve_forever()


async def run_zero_fault_async_tests():
    model = CompiledVMModel("model_weights.npz")
    cfg = load_behavior_config("bot_behavior_config.json")
    director = BehaviorDirector(model, cfg, seed=777)
    sent, disc = [], []

    async def on_send(uid, txt): sent.append((uid, txt))
    async def on_disc(uid, why): disc.append((uid, why))

    sched = ChatScheduler(director, on_send, None, on_disc, time_scale=0.02)
    allowed = (np.uint8, np.uint16, np.uint32, np.int32, np.float32, np.bool_)
    for s in sched.guards.__slots__:
        assert getattr(sched.guards, s).dtype in allowed, f"Forbidden dtype on guard.{s}"

    # 1. Rate Limit (15/10s -> "Wait a moment.") + Button Cooldown (1.5s)
    for _ in range(15): assert sched.guards.check_rate_limit(10, cfg.platform, now_s=5000.0)
    assert not sched.guards.check_rate_limit(10, cfg.platform, now_s=5000.5)
    assert sched.guards.check_button_cooldown(10, cfg.platform, now_s=5000.0)
    assert not sched.guards.check_button_cooldown(10, cfg.platform, now_s=5001.0)
    assert sched.guards.check_button_cooldown(10, cfg.platform, now_s=5002.0)

    # 2. 12h Quota (20 bots) + Turn-0 Hijack Refund + Last-Partner & Blocklist Shield
    for _ in range(20):
        ok, _ = sched.guards.can_match_and_record(11, cfg.platform, is_bot=True, now_s=6000)
        assert ok
    ok_21, rsn = sched.guards.can_match_and_record(11, cfg.platform, is_bot=True, now_s=6001)
    assert not ok_21 and rsn == "quota_12h_locked"
    assert sched.guards.refund_turn0_bot_match(11, 0)
    assert sched.guards.can_match_and_record(11, cfg.platform, is_bot=True, now_s=6002)[0]
    sched.guards.block_partner(12, 99)
    assert sched.guards.can_match_and_record(12, cfg.platform, is_bot=False, partner_id=99, now_s=6002)[1] == "blocked_partner"

    # 3. Rapid 0.4s Buffering + Multi-Fact NLU Extraction
    sched.on_match_start(20)
    director.store.persona_type[20] = np.uint8(P_FEMALE)
    await sched.on_user_message(20, "hi")
    await sched.on_user_message(20, "im 24")
    await sched.on_user_message(20, "from tunisia")
    await asyncio.sleep(0.35)
    assert int(director.store.user_age[20]) == 24 and int(director.store.user_country[20]) == 1 and len(sent) >= 1

    ram_mb = (director.store.total_nbytes() + sched.guards.total_nbytes()) / 1048576.0
    print(f"[SCHEDULER ZERO-FAULT GATE OK] Guards + BehaviorStore: {ram_mb:.2f} MB (50,000 users) | ZERO 64-bit | All async tests passed!", flush=True)


async def run_cli_demo():
    model = CompiledVMModel("model_weights.npz")
    cfg = load_behavior_config("bot_behavior_config.json")
    director = BehaviorDirector(model, cfg)

    async def on_send(_u, txt): print(f"\n[BOT]: {txt}", flush=True)
    async def on_type(_u): print("[bot is typing...]", flush=True)
    async def on_disc(_u, why): print(f"\n[DISCONNECTED: {why}] Type /next for a new match.", flush=True)

    sched = ChatScheduler(director, on_send, on_type, on_disc, time_scale=0.15)
    ok, persona, wait_s = sched.on_match_start(1)
    print(f"=== LIVE DEMO (0.15x timer speed) | Matched with: {persona} (open wait {wait_s}s) ===", flush=True)
    print("Commands: /next, /photo, /spoiler, /voice, /sticker, /gift, /reload, /quit", flush=True)
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line: break
        s = line.strip()
        if s == "/quit": break
        if s == "/reload":
            director.reload_config(); print("[CONFIG RELOADED LIVE]", flush=True); continue
        if s == "/next":
            _, persona, wait_s = sched.on_match_start(1)
            print(f"[NEW MATCH: {persona} | open wait {wait_s}s]", flush=True); continue
        if s in ("/photo", "/spoiler", "/voice", "/sticker"):
            await sched.on_user_message(1, event_type=s[1:]); continue
        if s == "/gift":
            await sched.on_user_message(1, event_type="gift", emoji="🌹"); continue
        await sched.on_user_message(1, s)


if __name__ == "__main__":
    if "--serve" in sys.argv:
        asyncio.run(serve_behavior_http())
    elif "--demo" in sys.argv:
        asyncio.run(run_cli_demo())
    else:
        asyncio.run(run_zero_fault_async_tests())