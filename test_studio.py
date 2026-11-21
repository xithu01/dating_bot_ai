#!/usr/bin/env python3
# FILE: test_studio.py
# PURPOSE: Ultra Telegram-Style Single-Server AI Test Studio (Steps 1-18 NLU + Compiled Behavior + Live Config Re-Pack)
# USAGE: Start ONLY `python production_server.py` (:8080), then run `python test_studio.py`
#        Open http://127.0.0.1:8090 in your browser.

import http.server
import json
import os
import socket
import socketserver
import time
import numpy as np

PORT = 8090
AI_HOST = "127.0.0.1"
AI_PORT = 8080
CONFIG_PATH = "bot_behavior_config.json"
WEIGHTS_PATH = "model_weights.npz"
ALLOWED_DTS = (np.int8, np.uint8, np.uint16, np.int32, np.uint32, np.float32, np.bool_)


def _raw_http_post(path, body_bytes, headers):
    hdr_lines = [f"POST {path} HTTP/1.1", f"Host: {AI_HOST}:{AI_PORT}"]
    for k, v in headers:
        hdr_lines.append(f"{k}: {v}")
    hdr_lines.append(f"Content-Length: {len(body_bytes)}")
    hdr_lines.append("Connection: close")
    req = ("\r\n".join(hdr_lines) + "\r\n\r\n").encode("ascii") + body_bytes

    t0 = time.perf_counter()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3.0)
    try:
        s.connect((AI_HOST, AI_PORT))
        s.sendall(req)
        chunks = []
        while True:
            c = s.recv(4096)
            if not c:
                break
            chunks.append(c)
    finally:
        s.close()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    raw = b"".join(chunks).decode("utf-8", errors="ignore")
    body = raw.split("\r\n\r\n", 1)[-1] if "\r\n\r\n" in raw else raw
    try:
        parsed = json.loads(body)
    except Exception:
        parsed = {"error": "parse_error", "raw": body[:200]}
    return elapsed_ms, parsed


def _normalize_response(mode, raw):
    if mode == "behavior" and "nlu" in raw:
        nlu = raw.get("nlu", {})
        persona = raw.get("persona", {})
        bubbles = raw.get("bubbles", [])
        action = raw.get("action", "reply")
        reply_joined = " | ".join(b.get("text", "") for b in bubbles) if bubbles else f"[{action}]"
        return {
            "mode": "behavior", "action": action,
            "class_id": int(nlu.get("class_id", -1)), "leaf_id": int(nlu.get("leaf_id", -1)),
            "intent": f"{action} (L{nlu.get('leaf_id', 0)})", "reply": reply_joined,
            "bubbles": bubbles, "persona": persona,
            "read_delay_s": raw.get("read_delay_s", 0.0), "idle_timeout_s": raw.get("idle_timeout_s", 0.0),
            "disconnect_after_s": raw.get("disconnect_after_s", None), "kbd": [],
        }
    return {
        "mode": "vm", "action": "reply",
        "class_id": int(raw.get("class_id", -1)), "leaf_id": -1,
        "intent": str(raw.get("intent", "unknown")), "reply": str(raw.get("reply", "")),
        "bubbles": [{"text": str(raw.get("reply", "")), "pause_before_s": 0.0, "typing_s": 0.0}],
        "persona": {"type": "vm_default", "vip": False, "interest": ""},
        "read_delay_s": 0.0, "idle_timeout_s": 0.0, "disconnect_after_s": None, "kbd": raw.get("kbd", []),
    }


def query_ai(text, uid=101, mode="behavior", event_type="text", emoji=""):
    uid_i = int(uid) & 0x7FFFFFFF
    if mode == "behavior":
        body_txt = "\n".join(p.strip() for p in text.split("||") if p.strip()) if event_type == "text" else ""
        hdrs = [("X-User-Id", str(uid_i)), ("X-Event-Type", event_type)]
        if emoji:
            hdrs.append(("X-Emoji", emoji))
        ms, raw = _raw_http_post("/turn", body_txt.encode("utf-8", errors="ignore"), hdrs)
        return ms, _normalize_response("behavior", raw)
    ms, raw = _raw_http_post("/", text.encode("utf-8", errors="ignore"), [("X-User-Id", str(uid_i))])
    return ms, _normalize_response("vm", raw)


def fast_repack_behavior_into_vault():
    """Repacks edited bot_behavior_config.json into model_weights.npz in ~0.08s (ZERO PyTorch import!)."""
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError("model_weights.npz not found! Run `python offline_train.py` first.")
    from offline_behavior_compiler import compile_behavior_brain
    beh_tbl = compile_behavior_brain(CONFIG_PATH)
    with np.load(WEIGHTS_PATH, allow_pickle=False) as arc:
        vault = {k: arc[k].copy() for k in arc.files}
    vault.update(beh_tbl)
    vault["vm_version_u8"] = np.array([15], dtype=np.uint8)
    for k, a in vault.items():
        if a.dtype not in ALLOWED_DTS:
            raise TypeError(f"Forbidden dtype {a.dtype} on '{k}'")
    np.savez(WEIGHTS_PATH, **vault)
    return int(beh_tbl["b_pool_len_u8"].shape[0])


def parse_jsonl_conversations(lines, max_convs=300):
    convs, cur_turns, cur_uid, cur_ts = [], [], "User", ""

    def flush():
        nonlocal cur_turns
        if cur_turns:
            convs.append({
                "id": 1000 + len(convs) + 1, "name": f"Log #{len(convs) + 1} ({cur_uid})",
                "category": "Imported Log", "uid": str(cur_uid), "time_str": cur_ts or "JSONL",
                "is_builtin": False, "steps": [{"text": t} for t in cur_turns[:40]],
            })
            cur_turns = []

    for raw_line in lines:
        if len(convs) >= max_convs:
            break
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            uid, sender, text, ts = str(obj.get("uid", cur_uid)), str(obj.get("sender", "")).lower(), str(obj.get("text", "")).strip(), obj.get("ts")
            if uid != cur_uid and cur_turns:
                flush()
            cur_uid = uid
            if ts and not cur_ts:
                try: cur_ts = time.strftime("%H:%M:%S", time.localtime(int(ts)))
                except Exception: cur_ts = str(ts)
            if sender == "user" and text: cur_turns.append(text)
            elif "disconnect" in text.lower() or text.lower() in ("next", "/next"):
                cur_turns.append("next"); flush(); cur_ts = ""
        else:
            if line.lower() in ("next", "/next", "disconnect"): cur_turns.append("next"); flush()
            else:
                cur_turns.append(line)
                if len(cur_turns) >= 15: flush()
    flush()
    return convs


HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ultra Dating AI Studio — Single-Server :8080 (Steps 1–18 + Behavior Brain)</title>
<style>
:root {
  --bg:#0b141a; --panel:#111b21; --card:#182229; --chat:#0e1621; --border:#222e35;
  --accent:#38bdf8; --user:#2b5278; --bot:#1e3a5f; --bot-rep:#6b2146; --bot-rst:#334155;
  --bot-disc:#5c1922; --text:#e9edef; --muted:#8696a0; --ok:#10b981; --bad:#ef4444; --warn:#f59e0b;
}
* { box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--text); margin:0; display:flex; height:100vh; overflow:hidden; }
.sidebar { width:335px; background:var(--panel); border-right:1px solid var(--border); display:flex; flex-direction:column; flex-shrink:0; }
.sidebar-header { padding:11px 14px; border-bottom:1px solid var(--border); background:#202c33; display:flex; justify-content:space-between; align-items:center; gap:6px; }
.sidebar-header h2 { margin:0; font-size:0.95rem; color:var(--accent); }
.sidebar-actions { padding:10px 12px; border-bottom:1px solid var(--border); display:flex; flex-direction:column; gap:6px; background:#152028; }
.sidebar-list { overflow-y:auto; flex:1; }
.side-item { padding:9px 13px; border-bottom:1px solid var(--border); cursor:pointer; }
.side-item:hover { background:#202c33; }
.side-item.active { background:#2a3942; border-left:4px solid var(--accent); }
.side-top { display:flex; justify-content:space-between; font-size:0.72rem; color:var(--muted); margin-bottom:2px; gap:6px; align-items:center; }
.side-title { font-size:0.84rem; font-weight:700; color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.side-seq { font-size:0.72rem; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }

.main { flex:1; display:flex; flex-direction:column; overflow:hidden; }
.toolbar { padding:9px 14px; background:#202c33; border-bottom:1px solid var(--border); display:flex; gap:7px; flex-wrap:wrap; align-items:center; }
input, select, button, textarea { padding:6px 10px; border-radius:6px; border:1px solid #374248; background:var(--panel); color:var(--text); font-size:0.8rem; }
button { cursor:pointer; font-weight:600; background:#0284c7; border-color:#0284c7; color:#fff; }
button:hover { background:#0369a1; }
.btn-green { background:#059669; border-color:#059669; } .btn-green:hover { background:#047857; }
.btn-amber { background:#d97706; border-color:#d97706; } .btn-amber:hover { background:#b45309; }
.btn-purple { background:#7c3aed; border-color:#7c3aed; } .btn-purple:hover { background:#6d28d9; }
.btn-pink { background:#db2777; border-color:#db2777; } .btn-pink:hover { background:#be185d; }
.btn-dark { background:#2a3942; border-color:#374248; } .btn-dark:hover { background:#374248; }
.content-scroll { flex:1; overflow-y:auto; padding:14px; display:flex; flex-direction:column; gap:14px; }

.smart-panel { background:linear-gradient(135deg,#0c192c 0%,#162436 100%); border:1px solid var(--accent); border-radius:12px; padding:13px 15px; display:flex; flex-direction:column; gap:10px; }
.smart-top { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; }
.smart-pct { font-size:2rem; font-weight:900; font-family:monospace; color:var(--ok); }
.smart-grade { font-size:0.82rem; font-weight:700; padding:4px 10px; border-radius:999px; background:#064e3b; color:#6ee7b7; border:1px solid var(--ok); }
.smart-cats { display:grid; grid-template-columns:repeat(auto-fit,minmax(138px,1fr)); gap:7px; }
.cat-card { background:#0b141a; border:1px solid #263545; border-radius:8px; padding:7px 9px; }
.cat-lbl { font-size:0.68rem; color:var(--muted); text-transform:uppercase; font-weight:700; }
.cat-val { font-size:0.96rem; font-weight:800; font-family:monospace; margin-top:2px; display:flex; justify-content:space-between; }
.pbar-bg { width:100%; height:6px; background:#222e35; border-radius:99px; overflow:hidden; margin-top:4px; }
.pbar-fill { height:100%; border-radius:99px; transition:width 0.25s; }

.chrono-panel { background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%); border:1px solid #334155; border-radius:12px; padding:14px; display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }
.chrono-panel.hidden { display:none; }
.tg-phone { width:425px; background:var(--chat); border:2px solid #334155; border-radius:16px; overflow:hidden; display:flex; flex-direction:column; height:540px; flex-shrink:0; box-shadow:0 10px 30px rgba(0,0,0,0.45); }
.tg-phone-header { background:#202c33; padding:9px 12px; display:flex; align-items:center; gap:9px; border-bottom:1px solid var(--border); }
.tg-avatar { width:34px; height:34px; border-radius:50%; background:#0284c7; display:flex; align-items:center; justify-content:center; font-size:1.1rem; }
.tg-status { font-size:0.73rem; color:var(--accent); font-weight:600; }
.tg-phone-body { flex:1; padding:11px; overflow-y:auto; display:flex; flex-direction:column; gap:7px; }
.tg-media-bar { padding:5px 8px; background:#17212b; border-top:1px solid var(--border); display:flex; gap:5px; flex-wrap:wrap; align-items:center; }
.tg-media-bar button { padding:4px 7px; font-size:0.72rem; }
.tg-phone-input { padding:7px 9px; background:#202c33; display:flex; gap:6px; border-top:1px solid var(--border); }
.tg-phone-input input { flex:1; }
.chrono-dashboard { flex:1; min-width:330px; display:flex; flex-direction:column; gap:9px; }
.stopwatch-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:7px; }
.sw-card { background:#0b141a; border:1px solid #334155; border-radius:9px; padding:8px 10px; }
.sw-label { font-size:0.67rem; color:var(--muted); text-transform:uppercase; }
.sw-val { font-size:1.18rem; font-weight:800; font-family:monospace; color:var(--accent); margin-top:2px; }
.sw-sub { font-size:0.7rem; color:#94a3b8; margin-top:2px; }
.mem-strip { background:#0b141a; border:1px solid #334155; border-radius:9px; padding:7px 11px; font-size:0.75rem; display:flex; gap:11px; flex-wrap:wrap; color:#cbd5e1; }
.mem-strip code { color:#38bdf8; font-weight:700; }

.studio-box { background:var(--card); border:1px solid #334155; border-radius:10px; padding:12px 14px; }
.studio-top { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:8px; }
.msg-steps-wrap { display:flex; gap:7px; flex-wrap:wrap; align-items:center; }
.msg-step-chip { display:flex; align-items:center; gap:4px; background:var(--panel); border:1px solid #374248; border-radius:8px; padding:3px 7px; }
.msg-step-chip input.step-txt { width:145px; border:none; background:transparent; padding:3px; font-weight:600; color:#fff; }
.msg-step-chip .del-step { background:transparent; border:none; color:#f87171; padding:2px 4px; }

.tg-chat { display:flex; flex-direction:column; gap:5px; }
.tg-bubble { max-width:91%; padding:7px 10px; border-radius:10px; font-size:0.83rem; line-height:1.34; word-wrap:break-word; }
.tg-bubble.user { background:var(--user); color:#fff; align-self:flex-end; border-bottom-right-radius:2px; }
.tg-bubble.bot { background:var(--bot); color:#fff; align-self:flex-start; border-bottom-left-radius:2px; border:1px solid #2a486b; }
.tg-bubble.bot-rst { background:var(--bot-rst); color:#e2e8f0; align-self:center; border:1px dashed #94a3b8; font-size:0.77rem; }
.tg-bubble.bot-disc { background:var(--bot-disc); color:#fecaca; align-self:flex-start; border:1px solid #ef4444; }
.tg-bubble.bot-sil { background:#17212b; color:#94a3b8; align-self:center; border:1px dotted #475569; font-size:0.75rem; font-style:italic; }
.tg-meta { display:flex; justify-content:space-between; align-items:center; gap:5px; font-size:0.66rem; color:rgba(255,255,255,0.78); margin-top:4px; flex-wrap:wrap; }
.kbd-row { display:flex; gap:5px; flex-wrap:wrap; margin-top:5px; }
.kbd-btn { background:var(--panel); border:1px solid var(--accent); color:#7dd3fc; padding:2px 8px; border-radius:5px; font-size:0.71rem; }
.badge { padding:2px 6px; border-radius:4px; font-size:0.67rem; font-weight:800; color:#fff; display:inline-block; background:#0284c7; }
.badge.c0 { background:#dc2626; } .badge.c1 { background:#16a34a; } .badge.c7 { background:#db2777; } .badge.c99 { background:#475569; }
.badge-pass { background:#065f46; color:#6ee7b7; border:1px solid var(--ok); padding:2px 7px; border-radius:99px; font-size:0.69rem; font-weight:700; }
.badge-fail { background:#7f1d1d; color:#fecaca; border:1px solid var(--bad); padding:2px 7px; border-radius:99px; font-size:0.69rem; font-weight:700; }
.badge-cnt { padding:2px 7px; border-radius:99px; font-size:0.7rem; background:#2a3942; color:#cbd5e1; font-weight:700; }
.time-badge { background:rgba(0,0,0,0.35); padding:1px 5px; border-radius:4px; color:#7dd3fc; font-family:monospace; }
.btn-poss { background:rgba(15,23,42,0.85); border:1px solid #fbbf24; color:#fde68a; padding:2px 6px; font-size:0.66rem; border-radius:5px; font-weight:700; }

.conv-header-card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:11px 14px; }
.turn-pills { display:flex; gap:5px; flex-wrap:wrap; align-items:center; margin-top:7px; }
.turn-pill { padding:3px 9px; border-radius:999px; background:#202c33; border:1px solid #374248; font-size:0.75rem; cursor:pointer; }
.turn-pill.active { background:var(--user); border-color:var(--accent); color:#fff; font-weight:700; }
.turn-pill.pass { border-color:var(--ok); } .turn-pill.fail { border-color:var(--bad); background:#450a0a; }
.turns-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(350px,1fr)); gap:11px; margin-top:11px; }
.tree-node { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:11px; display:flex; flex-direction:column; gap:7px; }
.tree-node.fail-card { border-color:var(--bad); background:#1c1317; }
.tree-node-top { display:flex; justify-content:space-between; align-items:center; font-size:0.78rem; font-weight:700; gap:7px; flex-wrap:wrap; }
.mem-mini { font-size:0.71rem; color:#94a3b8; background:#0b141a; border:1px solid var(--border); border-radius:6px; padding:4px 8px; display:flex; gap:8px; flex-wrap:wrap; }

.modal-backdrop { position:fixed; inset:0; background:rgba(0,0,0,0.82); backdrop-filter:blur(3px); display:none; align-items:center; justify-content:center; z-index:9999; padding:16px; }
.modal-backdrop.open { display:flex; }
.modal-box { background:var(--panel); border:1px solid var(--accent); border-radius:14px; width:100%; max-width:1060px; max-height:90vh; display:flex; flex-direction:column; overflow:hidden; box-shadow:0 20px 50px rgba(0,0,0,0.75); }
.modal-header { padding:13px 16px; background:#202c33; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }
.modal-history-strip { padding:8px 16px; background:#0b141a; border-bottom:1px solid var(--border); font-size:0.75rem; display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
.hist-chip-u { background:var(--user); color:#fff; padding:2px 7px; border-radius:6px; }
.modal-controls { padding:10px 16px; background:var(--card); border-bottom:1px solid var(--border); display:flex; gap:7px; flex-wrap:wrap; align-items:center; }
.modal-mem-bar { padding:7px 16px; background:var(--chat); border-bottom:1px solid var(--border); font-size:0.75rem; color:#94a3b8; display:flex; gap:13px; flex-wrap:wrap; }
.modal-body { padding:14px 16px; overflow-y:auto; flex:1; display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:11px; align-content:start; }
</style>
</head>
<body>

<!-- POSSIBILITIES MODAL -->
<div class="modal-backdrop" id="possModal" onclick="if(event.target===this) closePossModal()">
  <div class="modal-box">
    <div class="modal-header">
      <div>
        <strong style="font-size:1.02rem;color:var(--accent);" id="modalTitle">🎲 All AI Reply Possibilities</strong>
        <div style="font-size:0.74rem;color:var(--muted);">Replays exact prior turns across N independent X-User-Ids against production_server.py (:8080)</div>
      </div>
      <button type="button" class="btn-dark" onclick="closePossModal()">✕ Close</button>
    </div>
    <div class="modal-history-strip" id="modalHistoryStrip"></div>
    <div class="modal-controls">
      <input type="text" id="modalUserText" style="min-width:190px;font-weight:700;" onkeydown="if(event.key==='Enter') fetchTurnPossibilities()">
      <select id="modalPresetSelect" onchange="if(this.value){document.getElementById('modalUserText').value=this.value;fetchTurnPossibilities();}">
        <option value="">⚡ Presets...</option>
        <option value="hi">hi (C1)</option><option value="age?">age? (C2)</option>
        <option value="im 24">im 24 (C3)</option><option value="from?">from? (C4)</option>
        <option value="tuinsi">tuinsi (C5)</option><option value="24 from uk">24 from uk (C6)</option>
        <option value="im 15">im 15 (C7 Underage)</option><option value="how r u?">how r u? (C8)</option>
        <option value="m or f?">m or f? (C9)</option><option value="lm f u">lm f u (C12)</option>
        <option value="Chatting with u">Chatting with u (C16 L77)</option>
        <option value="m looking for girl , u">m looking for girl , u (C22)</option>
        <option value="whats ur name?">whats ur name? (C23)</option>
        <option value="are u a bot?">are u a bot? (C33)</option>
        <option value="Bcz you make it to fast.">Bcz you make it to fast. (C50)</option>
        <option value="?">? (C55)</option><option value="you didnt answer me">you didnt answer me (C56)</option>
        <option value="state?">state? (C57)</option><option value="h0rny">h0rny (C60)</option>
        <option value="Chicken">Chicken (C0 Trap)</option>
      </select>
      <select id="modalRepeatSend" onchange="fetchTurnPossibilities()">
        <option value="1" selected>Sent 1x</option><option value="2">Sent 2x</option><option value="3">Sent 3x</option>
      </select>
      <select id="modalIpMode" onchange="fetchTurnPossibilities()">
        <option value="vary" selected>Varying Users (See All %)</option><option value="same">Same User</option>
      </select>
      <label style="font-size:0.78rem;color:var(--muted);">Tries:</label>
      <input type="number" id="modalTriesInput" value="64" min="8" max="400" step="16" style="width:68px;text-align:center;font-weight:700;color:var(--accent);">
      <button type="button" class="btn-green" onclick="fetchTurnPossibilities()">🎲 Roll</button>
    </div>
    <div class="modal-mem-bar" id="modalMemBar"></div>
    <div class="modal-body" id="modalOutcomesGrid"></div>
  </div>
</div>

<!-- LIVE CONFIG EDITOR MODAL -->
<div class="modal-backdrop" id="cfgModal" onclick="if(event.target===this) closeConfigModal()">
  <div class="modal-box" style="max-width:960px;">
    <div class="modal-header">
      <div>
        <strong style="font-size:1.02rem;color:#fbbf24;">⚙️ Live Editor: bot_behavior_config.json → model_weights.npz</strong>
        <div style="font-size:0.74rem;color:var(--muted);">Edit any text, percentage, or timer — saves, re-packs model_weights.npz in 0.08s &amp; hot-reloads :8080 live!</div>
      </div>
      <div style="display:flex;gap:7px;">
        <button type="button" class="btn-green" onclick="saveAndReloadConfig()">💾 Save, Re-Pack &amp; Hot-Reload</button>
        <button type="button" class="btn-dark" onclick="closeConfigModal()">✕ Close</button>
      </div>
    </div>
    <div style="padding:8px 14px;background:#0b141a;font-size:0.76rem;color:#94a3b8;" id="cfgStatusMsg">Ready to edit.</div>
    <div style="padding:12px;flex:1;display:flex;">
      <textarea id="cfgTextarea" spellcheck="false" style="width:100%;height:65vh;font-family:monospace;font-size:0.78rem;line-height:1.4;background:#0b141a;color:#e2e8f0;resize:none;"></textarea>
    </div>
  </div>
</div>

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="sidebar-header">
    <div>
      <h2>🧪 Ultra AI Studio (:8080)</h2>
      <div style="font-size:0.71rem;color:var(--muted);" id="sidebarCount">Loading...</div>
    </div>
    <span id="apiStatusBadge" class="badge-cnt">⚡ Checking</span>
  </div>
  <div class="sidebar-actions">
    <button type="button" class="btn-green" style="width:100%;" onclick="runFullSmartnessBenchmark()">🧠 Run Full Audit (All 18 Steps)</button>
    <button type="button" class="btn-pink" style="width:100%;" onclick="runPersonaAndBehaviorAudit()">🎭 Persona % &amp; Behavior Audit</button>
    <div style="display:flex;gap:6px;">
      <button type="button" class="btn-amber" style="flex:1;font-size:0.75rem;" onclick="openConfigModal()">⚙️ Edit Brain</button>
      <button type="button" class="btn-dark" style="flex:1;font-size:0.75rem;" onclick="loadLocalJsonl()">📥 JSONL</button>
      <label class="btn-dark" style="flex:1;font-size:0.75rem;text-align:center;padding:6px 5px;border-radius:6px;cursor:pointer;border:1px solid #374248;font-weight:600;">
        📂 Upload<input type="file" id="fileUploadInput" accept=".txt,.json,.jsonl" style="display:none;" onchange="handleFileUpload(event)">
      </label>
    </div>
  </div>
  <div class="sidebar-list" id="sidebarList"></div>
</div>

<!-- MAIN -->
<div class="main">
  <div class="toolbar">
    <select id="backendMode" onchange="onModeChange()" title="Both modes run inside your SINGLE production_server.py on :8080!">
      <option value="behavior" selected>🎭 Behavior Brain (:8080/turn)</option>
      <option value="vm">⚡ Raw Instant NLU (:8080/)</option>
    </select>
    <input type="text" id="searchBox" placeholder="🔍 Filter scenarios, steps, intents..." oninput="applyFilters()" style="min-width:190px;">
    <select id="viewMode" onchange="renderSelectedConv()">
      <option value="all_turns_tree" selected>🌳 Per-Turn Cards</option>
      <option value="last_turn">🎯 Focus Single Turn</option>
      <option value="full_threads">📱 Full Transcript</option>
    </select>
    <select id="outcomeFilter" onchange="renderSelectedConv()">
      <option value="all">📊 All</option><option value="pass">✅ Pass</option><option value="fail">❌ Fail</option>
    </select>
    <button type="button" class="btn-amber" onclick="toggleChronoPanel()">📱 Phone &amp; HUD</button>
    <button type="button" class="btn-purple" onclick="runSelectedScenarioLive()">▶ Run Scenario</button>
  </div>

  <div class="content-scroll">
    <div class="smart-panel">
      <div class="smart-top">
        <div>
          <div style="font-size:0.72rem;color:var(--accent);text-transform:uppercase;font-weight:800;">🧠 Single-File Ultra AI Brain (103 Leaves + Compiled Behavior in model_weights.npz)</div>
          <div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;">
            <span class="smart-pct" id="smartOverallPct">--%</span>
            <span class="smart-grade" id="smartGrade">Click "Run Full Audit (All 18 Steps)"</span>
          </div>
        </div>
        <div style="text-align:right;font-size:0.78rem;color:#cbd5e1;">
          <div>Checks Passed: <strong id="smartPassedTxt" style="color:var(--ok);">0 / 0</strong></div>
          <div>Avg Latency: <strong id="smartAvgMs" style="color:var(--accent);">0.00 ms/req</strong></div>
          <div id="personaAuditSummary" style="color:#f472b6;font-size:0.74rem;margin-top:2px;"></div>
        </div>
      </div>
      <div class="pbar-bg" style="height:8px;"><div class="pbar-fill" id="smartOverallBar" style="width:0%;background:var(--ok);"></div></div>
      <div class="smart-cats" id="smartCatsGrid"></div>
    </div>

    <div class="chrono-panel" id="chronoPanel">
      <div class="tg-phone">
        <div class="tg-phone-header">
          <div class="tg-avatar" id="phoneAvatar">🤖</div>
          <div style="flex:1;">
            <div style="font-weight:700;font-size:0.88rem;" id="phoneTitle">Dating AI (:8080/turn)</div>
            <div class="tg-status" id="phoneStatus">checking...</div>
          </div>
          <select id="phoneSpeed" style="padding:3px 5px;font-size:0.72rem;">
            <option value="0">⚡ Instant</option><option value="0.15" selected>⏩ 0.15x</option><option value="1">⏱️ 1x Real</option>
          </select>
          <button type="button" class="btn-amber" style="padding:4px 8px;font-size:0.72rem;" onclick="resetLivePhone()">🔀 New</button>
        </div>
        <div class="tg-phone-body" id="phoneChatBody"></div>
        <div class="tg-media-bar">
          <span style="font-size:0.69rem;color:var(--muted);">Events:</span>
          <button type="button" class="btn-dark" onclick="triggerOpeningBurst()">💬 Open Burst</button>
          <button type="button" class="btn-dark" onclick="sendLiveEvent('photo')">📷 Photo</button>
          <button type="button" class="btn-dark" onclick="sendLiveEvent('spoiler')">🙈 Spoiler</button>
          <button type="button" class="btn-dark" onclick="sendLiveEvent('voice')">🎤 Voice</button>
          <button type="button" class="btn-dark" onclick="sendLiveEvent('sticker')">🎭 Sticker</button>
          <button type="button" class="btn-pink" onclick="sendLiveEvent('gift','🌹')">🌹 Gift</button>
        </div>
        <div class="tg-phone-input">
          <input type="text" id="liveChatInput" placeholder="Type msg (or burst: hi || im 24 || tunisia)..." onkeydown="if(event.key==='Enter') sendLivePhoneMsg()">
          <button type="button" onclick="sendLivePhoneMsg()">Send</button>
        </div>
      </div>

      <div class="chrono-dashboard">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">
          <strong style="font-size:0.95rem;color:var(--accent);">⚡ Live Latency, Persona &amp; Timing Inspector</strong>
          <div style="display:flex;gap:6px;">
            <button type="button" class="btn-green" style="font-size:0.74rem;" onclick="hotReloadNow()">🔄 Reload .npz</button>
            <button type="button" class="btn-dark" style="font-size:0.74rem;" onclick="clearLivePhoneOnly()">🧹 Clear</button>
          </div>
        </div>
        <div class="stopwatch-grid">
          <div class="sw-card"><div class="sw-label">⚡ API Latency</div><div class="sw-val" id="swLastMs">0.00ms</div><div class="sw-sub" id="swLastClass">No message</div></div>
          <div class="sw-card"><div class="sw-label">🎭 Bot Persona</div><div class="sw-val" id="swPersona" style="color:#f472b6;font-size:1.02rem;">Unrolled</div><div class="sw-sub" id="swVipInt">VIP: No</div></div>
          <div class="sw-card"><div class="sw-label">⏱️ Human Read+Type</div><div class="sw-val" id="swReadType" style="color:#10b981;font-size:1.02rem;">0.0s</div><div class="sw-sub" id="swBubbleCnt">0 bubbles</div></div>
          <div class="sw-card"><div class="sw-label">⏳ Idle / Leave</div><div class="sw-val" id="swIdleDisc" style="color:#f59e0b;font-size:1.02rem;">--</div><div class="sw-sub" id="swActionLbl">Ready</div></div>
        </div>
        <div class="mem-strip">
          <span>🧠 <strong>NLU:</strong> <code id="memLastIntent">None</code></span>
          <span>🔢 <strong>Turn:</strong> <code id="memMsgCnt">0</code></span>
          <span>🎂 <strong>Age:</strong> <code id="memAge">None</code></span>
          <span>🌍 <strong>Country:</strong> <code id="memCountry">None</code></span>
          <span>👤 <strong>UID:</strong> <code id="memUid">1001</code></span>
        </div>
        <div style="background:#0b141a;border:1px solid #334155;border-radius:9px;padding:9px 11px;max-height:245px;overflow-y:auto;">
          <div style="font-size:0.72rem;color:var(--muted);text-transform:uppercase;margin-bottom:5px;font-weight:700;">📊 Real-Time Inference &amp; Timing Log</div>
          <table style="width:100%;border-collapse:collapse;font-size:0.75rem;text-align:left;">
            <thead><tr style="border-bottom:1px solid var(--border);color:#94a3b8;">
              <th style="padding:3px;">#</th><th style="padding:3px;">Input</th><th style="padding:3px;">NLU</th><th style="padding:3px;">Action / Bubbles</th><th style="padding:3px;">Read/Idle</th>
            </tr></thead>
            <tbody id="chronoLogTable"></tbody>
          </table>
        </div>
      </div>
    </div>

    <div class="studio-box">
      <div class="studio-top">
        <div>
          <strong style="color:var(--accent);font-size:0.92rem;">🧪 Multi-Turn Batch Probability Simulator</strong>
          <span style="font-size:0.75rem;color:var(--muted);margin-left:5px;">Run any sequence across N users to inspect exact reply / evade / skip / disconnect %:</span>
        </div>
        <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
          <input type="text" id="quickArrayInput" placeholder="hi, f, im 24, from?, you didnt answer me" style="width:260px;">
          <button type="button" class="btn-dark" onclick="loadQuickArray()">📥 Load</button>
          <button type="button" class="btn-dark" onclick="addStepInput('')">➕ Msg</button>
          <select id="simRepeats" style="width:86px;"><option value="16">16x</option><option value="32" selected>32x</option><option value="64">64x</option><option value="100">100x</option></select>
          <button type="button" class="btn-green" onclick="runCustomBatchSimulation()">🚀 Simulate</button>
        </div>
      </div>
      <div class="msg-steps-wrap" id="stepsContainer"></div>
    </div>

    <div id="mainContent"></div>
  </div>
</div>

<script>
const CLASS_NAMES = {
  0:"not_valid", 1:"greeting", 2:"ask_age", 3:"tell_age", 4:"ask_country", 5:"tell_country",
  6:"age+country", 7:"underage", 8:"ask_hru", 9:"ask_gender_choice", 10:"ask_is_female",
  11:"ask_is_male", 12:"state_female", 13:"state_male", 14:"age+gender", 15:"ask_wyd",
  16:"status_pos", 17:"status_neg", 18:"status_ask_back", 19:"flirt", 20:"react_ack",
  21:"ask_looking", 22:"state_looking", 23:"ask_name", 24:"tell_name", 25:"ask_hobbies",
  26:"tell_hobbies", 27:"ask_rel", 28:"tell_rel", 29:"ask_private", 30:"yes_no_maybe",
  31:"ask_pic_app", 32:"offer_ref_pic", 33:"ask_is_bot", 34:"ask_is_real", 35:"bye_gn_brb",
  36:"work_study", 37:"language", 38:"ama_tod_bio", 39:"flirt_meet_aff", 40:"thanks_sorry",
  41:"ping_clarify", 42:"nsfw_rude", 43:"time_weather", 44:"tod_pick", 45:"vibe_choice",
  46:"type_ldr", 47:"life_zodiac", 48:"joke_vent", 49:"food_chat", 50:"fast_reply_reason",
  51:"reason_chat", 52:"trust_doubt", 53:"casual_bridge", 54:"ask_why", 55:"pure_qmark",
  56:"user_insist", 57:"ask_state_city", 58:"nice_to_meet_u", 59:"short_dry",
  60:"horny_prompt", 61:"ask_secret", 99:"reset"
};

const BUILTIN_SCENARIOS = [
  { id:1, name:"1️⃣ Step 1: Greetings, Typos & Reset", category:"Step 1 Greet", is_builtin:true, desc:"Stretched typos ('heeyy','hlo','whats up') and reset.",
    steps:[ {text:"hi",expect_class:1}, {text:"heeyy",expect_class:1}, {text:"hlo",expect_class:1}, {text:"whats up",expect_class:1}, {text:"nice to meet u",expect_class:1}, {text:"next",expect_class:99}, {text:"hey",expect_class:1} ] },
  { id:2, name:"2️⃣ Step 2: Age (18–80), Country & Combined", category:"Step 2 Age/Ctry", is_builtin:true, desc:"Age, country typos ('tuinsi','handi'), combined '24 from uk', bounds.",
    steps:[ {text:"age?",expect_class:2}, {text:"24",expect_class:3}, {text:"21yres",expect_class:3}, {text:"from?",expect_class:4}, {text:"tuinsi",expect_class:5}, {text:"handi",expect_class:5}, {text:"hi l'm 30 from tun",expect_class:6}, {text:"24 from uk",expect_class:6}, {text:"81",expect_class:0}, {text:"200",expect_class:0} ] },
  { id:3, name:"3️⃣ Step 3 & 5: HRU vs WYD vs Job Trap", category:"Step 3/5 HRU/WYD", is_builtin:true, desc:"'how are you' (C8) vs 'what are you doing' (C15) vs 'what do you do' (C0).",
    steps:[ {text:"how are you",expect_class:8}, {text:"hru",expect_class:8}, {text:"hor",expect_class:8}, {text:"wyd",expect_class:15}, {text:"what are you doing",expect_class:15}, {text:"what do you do",expect_class:0} ] },
  { id:4, name:"4️⃣ Step 4: Gender Statement-Over-Question", category:"Step 4 Gender", is_builtin:true, desc:"'m or f?', 'lm f u', 'im m u?', '22f', '24 m', '19 famle'.",
    steps:[ {text:"m or f?",expect_class:9}, {text:"m/f",expect_class:9}, {text:"are you a girl",expect_class:10}, {text:"are you a boy",expect_class:11}, {text:"lm f u",expect_class:12}, {text:"im m u?",expect_class:13}, {text:"22f",expect_class:14}, {text:"24 m",expect_class:14}, {text:"19 famle",expect_class:14} ] },
  { id:5, name:"5️⃣ Step 6: Status & Chatting-with-u (L77)", category:"Step 6 Status", is_builtin:true, desc:"Positive/negative status, 'nmu', and Leaf 77 'Chatting with u'.",
    steps:[ {text:"im good",expect_class:16}, {text:"just chilling in bed",expect_class:16}, {text:"Chatting with u",expect_class:16,expect_leaf:77}, {text:"texting u",expect_class:16,expect_leaf:77}, {text:"lm borad",expect_class:17}, {text:"nothing much",expect_class:17}, {text:"im good u?",expect_class:18}, {text:"nmu",expect_class:18} ] },
  { id:6, name:"6️⃣ Step 7: Flirt & Short Reactions", category:"Step 7 Flirt/Ack", is_builtin:true, desc:"Compliments and reactions ('haha','same','sam','for real').",
    steps:[ {text:"ur cute",expect_class:19}, {text:"love ur vibe",expect_class:19}, {text:"i like u",expect_class:19}, {text:"haha",expect_class:20}, {text:"same",expect_class:20}, {text:"sam",expect_class:20}, {text:"for real",expect_class:20} ] },
  { id:7, name:"7️⃣ Step 8: Looking For & Trap Shield", category:"Step 8 Looking", is_builtin:true, desc:"'m looking for girl , u', 'a gf', 'a bf' vs traps 'a bf here'/'a gf here'.",
    steps:[ {text:"what r u looking for?",expect_class:21}, {text:"looking for friends",expect_class:22}, {text:"m looking for girl , u",expect_class:22}, {text:"lm finding gril",expect_class:22}, {text:"a gf",expect_class:22}, {text:"a bf",expect_class:22}, {text:"a bf here",expect_class:0}, {text:"a gf here",expect_class:0} ] },
  { id:8, name:"8️⃣ Step 9: Names & Wildcard Unseen Name", category:"Step 9 Names", is_builtin:true, desc:"Catalog names ('hatem','sarah'), 'im sam', and wildcard 'zoltan'.",
    steps:[ {text:"whats ur name?",expect_class:23}, {text:"ur name?",expect_class:23}, {text:"hatem",expect_class:24}, {text:"my name is sarah",expect_class:24}, {text:"im sam",expect_class:24}, {text:"my name is zoltan",expect_class:24} ] },
  { id:9, name:"9️⃣ Step 10: Hobbies (All 5 Domains)", category:"Step 10 Hobbies", is_builtin:true, desc:"Music, games, anime, gym, and 'what do u do for fun?'.",
    steps:[ {text:"what music do u like?",expect_class:25}, {text:"what games do u play?",expect_class:25}, {text:"do u watch anime?",expect_class:25}, {text:"what do u do for fun?",expect_class:25}, {text:"i like rap music",expect_class:26}, {text:"i play valorant",expect_class:26}, {text:"i love anime",expect_class:26}, {text:"going to the gym",expect_class:26} ] },
  { id:10, name:"🔟 Step 11: Relationship & Negation Shield", category:"Step 11 Rel", is_builtin:true, desc:"'im not married' & 'i dont have a gf' -> single (L40).",
    steps:[ {text:"are you single or mariid",expect_class:27}, {text:"do u have a bf?",expect_class:27}, {text:"im single",expect_class:28}, {text:"im not married",expect_class:28,expect_leaf:40}, {text:"i dont have a gf",expect_class:28,expect_leaf:40}, {text:"im taken",expect_class:28}, {text:"lm mariid",expect_class:28} ] },
  { id:11, name:"1️⃣1️⃣ Step 12: Private / Socials & Y/N/M", category:"Step 12 Priv/YN", is_builtin:true, desc:"'can we talk privty', 'telgrm?', 'do u have snap?', 'yes plz', 'idk'.",
    steps:[ {text:"can we talk privty",expect_class:29}, {text:"telgrm?",expect_class:29}, {text:"do u have snap?",expect_class:29}, {text:"yes plz",expect_class:30}, {text:"i dont have snap",expect_class:30}, {text:"maybe",expect_class:30}, {text:"idk",expect_class:30} ] },
  { id:12, name:"1️⃣2️⃣ Step 13 & 14: Pics & Bot-vs-Real", category:"Step 13/14 Pic/Bot", is_builtin:true, desc:"Pics/appearance + 'are u a bot?' (C33) vs 'are u real?' (C34).",
    steps:[ {text:"send pic",expect_class:31}, {text:"what do u look like?",expect_class:31}, {text:"wanna see me?",expect_class:32}, {text:"u first",expect_class:32}, {text:"are u a bot?",expect_class:33}, {text:"bot or real?",expect_class:33}, {text:"are u real?",expect_class:34}, {text:"prove ur real",expect_class:34} ] },
  { id:13, name:"1️⃣3️⃣ Step 15: Unified Dating Pack (35–43)", category:"Step 15 Pack", is_builtin:true, desc:"gn, brb, work/study, language, truth or dare, flirt, thanks, wdym, nsfw, weather.",
    steps:[ {text:"gn",expect_class:35}, {text:"brb",expect_class:35}, {text:"do u work or study?",expect_class:36}, {text:"im a student",expect_class:36}, {text:"do u speak arabic?",expect_class:37}, {text:"my english is bad",expect_class:37}, {text:"truth or dare",expect_class:38}, {text:"do u like me?",expect_class:39}, {text:"thanks",expect_class:40}, {text:"wdym",expect_class:41}, {text:"send nudes",expect_class:42}, {text:"hows the weather",expect_class:43} ] },
  { id:14, name:"1️⃣4️⃣ Step 16: Games, Vibe, Zodiac, Jokes, Food", category:"Step 16 Deep", is_builtin:true, desc:"Classes 44–49 interactive dating pack.",
    steps:[ {text:"truth",expect_class:44}, {text:"give me a dare",expect_class:44}, {text:"im a night owl",expect_class:45}, {text:"whats ur type?",expect_class:46}, {text:"do u mind long distance?",expect_class:46}, {text:"do u have pets?",expect_class:47}, {text:"im a scorpio",expect_class:47}, {text:"tell me a joke",expect_class:48}, {text:"i had a bad day",expect_class:48}, {text:"whats ur fav food?",expect_class:49}, {text:"i love pizza",expect_class:49} ] },
  { id:15, name:"1️⃣5️⃣ Step 17: Natural English (50–54)", category:"Step 17 English", is_builtin:true, desc:"'Bcz you make it to fast.', 'bcz im bored', 'i believe u', 'really?', 'why do u ask?'.",
    steps:[ {text:"Bcz you make it to fast.",expect_class:50}, {text:"u type too fast",expect_class:50}, {text:"bcz im bored",expect_class:51}, {text:"i believe u",expect_class:52}, {text:"i dont believe u",expect_class:52}, {text:"really?",expect_class:52}, {text:"i guess",expect_class:53}, {text:"why do u ask?",expect_class:54} ] },
  { id:16, name:"1️⃣6️⃣ Step 18: Behavior Triggers (55–61)", category:"Step 18 Behavior", is_builtin:true, desc:"Pure '?', Insistence, State/City, NTMU, Dry, Horny/'h0rny', Secret.",
    steps:[ {text:"?",expect_class:55}, {text:"???",expect_class:55}, {text:"you didnt answer me",expect_class:56}, {text:"answer my question",expect_class:56}, {text:"state?",expect_class:57}, {text:"which city?",expect_class:57}, {text:"glad to meet u",expect_class:58}, {text:"hmm",expect_class:59}, {text:"idc",expect_class:59}, {text:"horny",expect_class:60}, {text:"h0rny",expect_class:60}, {text:"tell me a secret",expect_class:61} ] },
  { id:17, name:"1️⃣7️⃣ Multi-Turn Ask-Back & Anti-Loop", category:"Multi-Turn", is_builtin:true, desc:"'M'->'u', 'Chicken'->'u' (C0), 'Chatting with u'->'u', 'M'->'u'->'u'.",
    steps:[ {text:"M",expect_class:13}, {text:"u",expect_class_vm:9}, {text:"u",expect_class_vm:0}, {text:"Chicken",expect_class:0}, {text:"u",expect_class_vm:0}, {text:"Chatting with u",expect_class:16}, {text:"u",expect_class_vm:18} ] },
  { id:18, name:"1️⃣8️⃣ Underage 18+ Shield & Class 0 Traps", category:"Safety & Traps", is_builtin:true, desc:"Ages 1–17 -> C7 and gibberish/traps -> C0.",
    steps:[ {text:"15",expect_class:7}, {text:"im 16",expect_class:7}, {text:"17",expect_class:7}, {text:"Chicken",expect_class:0}, {text:"what",expect_class:0}, {text:"why not",expect_class:0}, {text:"who are you",expect_class:0}, {text:"hyw",expect_class:0}, {text:"m bi",expect_class:0}, {text:"asdfgh",expect_class:0} ] }
];

let allConvs = BUILTIN_SCENARIOS.map(c => ({ ...c, results:null, batchRuns:null }));
let filteredConvs = allConvs.slice(), selectedConvId = allConvs[0].id, selectedTurnIdx = null;
let serverOnline = false, liveUid = 1001, liveHistory = [], liveTurn = 0;
let liveMem = { lastIntent:"None", age:"None", country:"None" };
let modalPriorTexts = [], modalTurnNum = 1;
window._possReg = {}; let _possId = 0;

function esc(s) { return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function getMode() { return document.getElementById('backendMode').value; }
function classBadge(cid) {
  const n = CLASS_NAMES[cid] || `c${cid}`;
  const cls = [0,1,7,99].includes(cid) ? `c${cid}` : '';
  return `<span class="badge ${cls}">C${cid}: ${n}</span>`;
}

async function pingServer() {
  const badge = document.getElementById('apiStatusBadge');
  try { serverOnline = !!(await (await fetch('/api/ping')).json()).online; } catch (e) { serverOnline = false; }
  onModeChange(false);
  badge.innerText = serverOnline ? "🟢 :8080 Online" : "🔴 Start production_server.py";
  badge.style.background = serverOnline ? "#065f46" : "#7f1d1d";
  badge.style.color = serverOnline ? "#6ee7b7" : "#fecaca";
}

function onModeChange(doReset = true) {
  const m = getMode();
  document.getElementById('phoneTitle').innerText = m === 'behavior' ? 'Dating AI (:8080/turn)' : 'Dating AI (:8080/ Raw VM)';
  document.getElementById('phoneStatus').innerText = serverOnline ? `🟢 online — production_server.py (${m})` : '🔴 run: python production_server.py';
  if (doReset) resetLivePhone();
}

async function sendToAI(text, uid = liveUid, eventType = 'text', emoji = '') {
  return await (await fetch('/api/send', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ text, uid, mode: getMode(), event_type: eventType, emoji })
  })).json();
}

function registerPossBtn(turnNum, userText, priorTexts) {
  const id = ++_possId; window._possReg[id] = { turnNum, userText, priorTexts: priorTexts.slice() };
  return `<button type="button" class="btn-poss" onclick="triggerPoss(${id})">🎲 Possibilities</button>`;
}
function triggerPoss(id) { const it = window._possReg[id]; if (it) openPossModal(it.turnNum, it.userText, it.priorTexts); }

async function openPossModal(turnNum, userText, priorTexts) {
  modalTurnNum = turnNum; modalPriorTexts = priorTexts || [];
  document.getElementById('possModal').classList.add('open');
  document.getElementById('modalTitle').innerText = `🎲 All Possibilities at Turn #${turnNum} (${getMode().toUpperCase()})`;
  document.getElementById('modalUserText').value = userText || '';
  document.getElementById('modalPresetSelect').value = '';
  document.getElementById('modalRepeatSend').value = '1';
  document.getElementById('modalHistoryStrip').innerHTML = modalPriorTexts.length
    ? `<strong style="color:#38bdf8;">📜 Prior:</strong> ` + modalPriorTexts.map(t => `<span class="hist-chip-u">👤 ${esc(t)}</span>`).join(' → ')
    : `<span style="color:#64748b;">📜 Fresh Match Start</span>`;
  await fetchTurnPossibilities();
}
function closePossModal() { document.getElementById('possModal').classList.remove('open'); }

async function fetchTurnPossibilities() {
  const grid = document.getElementById('modalOutcomesGrid'), memBar = document.getElementById('modalMemBar');
  const txt = document.getElementById('modalUserText').value;
  const tries = parseInt(document.getElementById('modalTriesInput').value || '64', 10);
  const repeatSend = parseInt(document.getElementById('modalRepeatSend').value || '1', 10);
  const ipMode = document.getElementById('modalIpMode').value;
  grid.innerHTML = `<div style="padding:18px;color:var(--accent);">⏳ Running ${tries} live simulations...</div>`;
  try {
    const d = await (await fetch('/api/turn_possibilities', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ prior_texts: modalPriorTexts, text: txt, tries, repeat_send: repeatSend, ip_mode: ipMode, mode: getMode() })
    })).json();
    if (!d.ok) { grid.innerHTML = `<div style="color:#f87171;">Error: ${esc(d.error)}</div>`; return; }
    const uniqCls = [...new Set(d.outcomes.map(o => o.class_id))];
    memBar.innerHTML = `
      <span>🎯 <strong>Target:</strong> <code style="color:#fff;">"${esc(txt)}"</code> (${repeatSend}x)</span>
      <span>🎲 <strong>Unique Outcomes:</strong> <code style="color:#38bdf8;">${d.outcomes.length}</code></span>
      <span>⚡ <strong>Avg:</strong> <code style="color:#10b981;">${d.avg_ms.toFixed(2)}ms</code></span>
      <span>🛡️ <strong>NLU Stability:</strong> <code style="color:${uniqCls.length===1?'#10b981':'#f59e0b'};">${uniqCls.length===1?'100% Consistent ✅':'Split ('+uniqCls.join(',')+')'}</code></span>`;
    grid.innerHTML = d.outcomes.map(o => `
      <div class="tree-node">
        <div class="tree-node-top"><span>${classBadge(o.class_id)} <code>${esc(o.intent)}</code></span><span style="color:#38bdf8;font-size:0.9rem;">${o.pct}% (${o.count}/${d.tries})</span></div>
        <div class="pbar-bg"><div class="pbar-fill" style="width:${o.pct}%;background:#38bdf8;"></div></div>
        <div class="tg-bubble ${o.action==='disconnect'?'bot-disc':(o.action==='silent'?'bot-sil':'bot')}"><div>${esc(o.reply)}</div><div class="tg-meta"><span>avg ${o.avg_ms.toFixed(2)}ms</span><span>${o.action}</span></div></div>
      </div>`).join('');
  } catch (e) { grid.innerHTML = `<div style="color:#f87171;">Server connection failed!</div>`; }
}

async function openConfigModal() {
  document.getElementById('cfgModal').classList.add('open');
  document.getElementById('cfgStatusMsg').innerText = 'Loading bot_behavior_config.json...';
  const d = await (await fetch('/api/config')).json();
  if (!d.ok) { document.getElementById('cfgStatusMsg').innerText = '❌ ' + d.error; return; }
  document.getElementById('cfgTextarea').value = d.content;
  document.getElementById('cfgStatusMsg').innerText = '✅ Loaded — edit any text, percentage, or timer and click Save, Re-Pack & Hot-Reload.';
}
function closeConfigModal() { document.getElementById('cfgModal').classList.remove('open'); }
async function saveAndReloadConfig() {
  const content = document.getElementById('cfgTextarea').value, st = document.getElementById('cfgStatusMsg');
  try { JSON.parse(content); } catch (e) { st.innerText = '❌ Invalid JSON: ' + e.message; st.style.color = '#f87171'; return; }
  const d = await (await fetch('/api/config', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ content }) })).json();
  st.style.color = d.ok ? '#10b981' : '#f87171';
  st.innerText = d.ok ? `✅ Saved & Re-packed ${d.pools} pools into model_weights.npz in ${d.repack_ms}ms! ${d.reloaded ? 'Hot-reloaded :8080 live!' : 'Start production_server.py to apply.'}` : ('❌ ' + d.error);
}
async function hotReloadNow() {
  const d = await (await fetch('/api/reload', { method:'POST' })).json();
  alert(d.ok ? `✅ Reloaded model_weights.npz live (${d.leaves} leaves)!` : `⚠️ ${d.error}`);
}

function clearLivePhoneOnly() { document.getElementById('phoneChatBody').innerHTML = ''; document.getElementById('chronoLogTable').innerHTML = ''; }
async function resetLivePhone() {
  liveUid = 1000 + Math.floor(Math.random() * 40000);
  liveHistory = []; liveTurn = 0; liveMem = { lastIntent:"None", age:"None", country:"None" };
  clearLivePhoneOnly();
  document.getElementById('memUid').innerText = liveUid;
  const body = document.getElementById('phoneChatBody');
  if (getMode() === 'behavior') {
    const d = await (await fetch('/api/start_chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ uid: liveUid }) })).json();
    if (d.ok) {
      document.getElementById('swPersona').innerText = d.persona || 'rolled';
      document.getElementById('phoneAvatar').innerText = d.persona === 'female' ? '👩' : '👨';
      body.innerHTML = `<div class="tg-bubble bot-rst">✨ New Match (UID ${liveUid}): <strong>${esc(d.persona)}</strong> • Open wait: ${d.opening_wait_s}s</div>`;
    } else body.innerHTML = `<div class="tg-bubble bot-disc">⚠️ :8080 offline. Run: python production_server.py</div>`;
  } else await sendLiveText("next");
}

async function triggerOpeningBurst() {
  if (getMode() !== 'behavior') return alert("Switch to Behavior (:8080/turn) mode for Opening Bursts!");
  const d = await (await fetch('/api/opening', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ uid: liveUid }) })).json();
  if (!d.ok) return alert(d.error || "Failed");
  await renderAnimatedBotResponse(d.response, d.latency_ms, 0, "[Opening Burst]", []);
}
async function sendLiveEvent(evType, emoji = '') {
  if (getMode() !== 'behavior') return alert("Switch to Behavior (:8080/turn) mode for Media & Gift events!");
  await sendLiveText(emoji ? `[GIFT ${emoji}]` : `[MEDIA: ${evType.toUpperCase()}]`, evType, emoji);
}
async function sendLivePhoneMsg() {
  const inp = document.getElementById('liveChatInput'), txt = inp.value.trim();
  if (!txt) return; inp.value = ''; await sendLiveText(txt, 'text', '');
}
async function sendLiveText(txt, eventType = 'text', emoji = '') {
  const chatBody = document.getElementById('phoneChatBody'), prior = liveHistory.slice();
  liveTurn += 1;
  chatBody.insertAdjacentHTML('beforeend', `<div class="tg-bubble user"><div>${esc(txt)}</div><div class="tg-meta"><span>Turn #${liveTurn}</span><span>✓✓</span></div></div>`);
  chatBody.scrollTop = chatBody.scrollHeight;
  const data = await sendToAI(eventType === 'text' ? txt : '', liveUid, eventType, emoji);
  if (!data.ok) { chatBody.insertAdjacentHTML('beforeend', `<div class="tg-bubble bot-disc">⚠️ Offline: ${esc(data.error)}</div>`); return; }
  const r = data.response;
  if (r.class_id === 99) liveHistory = []; else if (eventType === 'text') liveHistory.push(txt);
  const nm = txt.match(/\b(1[89]|[2-7][0-9]|80)\b/);
  if ((r.class_id === 3 || r.class_id === 6 || r.class_id === 14) && nm) liveMem.age = nm[1];
  if (r.class_id === 5 || r.class_id === 6) liveMem.country = txt.slice(0, 18);
  liveMem.lastIntent = `C${r.class_id} (${CLASS_NAMES[r.class_id] || ''})`;
  await renderAnimatedBotResponse(r, data.latency_ms, liveTurn, txt, prior);
}

async function renderAnimatedBotResponse(r, ms, turnNum, userTxt, prior) {
  const chatBody = document.getElementById('phoneChatBody'), speed = parseFloat(document.getElementById('phoneSpeed').value || '0');
  const totType = (r.bubbles || []).reduce((a, b) => a + (b.typing_s || 0), 0);
  document.getElementById('swLastMs').innerText = `${ms.toFixed(2)}ms`;
  document.getElementById('swLastClass').innerText = `C${r.class_id} ${CLASS_NAMES[r.class_id] || ''}`;
  document.getElementById('swPersona').innerText = (r.persona && r.persona.type) ? r.persona.type : 'vm';
  document.getElementById('swVipInt').innerText = r.persona ? `VIP:${r.persona.vip ? 'Yes' : 'No'} | ${r.persona.interest || ''}` : '';
  document.getElementById('swReadType').innerText = `${(r.read_delay_s || 0).toFixed(1)}s + ${totType.toFixed(1)}s`;
  document.getElementById('swBubbleCnt').innerText = `${(r.bubbles || []).length} bubble(s)`;
  document.getElementById('swIdleDisc').innerText = r.disconnect_after_s !== null ? `Leave ${r.disconnect_after_s}s` : `Idle ${(r.idle_timeout_s || 0).toFixed(0)}s`;
  document.getElementById('swActionLbl').innerText = `Action: ${r.action}`;
  document.getElementById('memLastIntent').innerText = liveMem.lastIntent;
  document.getElementById('memMsgCnt').innerText = liveTurn;
  document.getElementById('memAge').innerText = liveMem.age;
  document.getElementById('memCountry').innerText = liveMem.country;

  const statusEl = document.getElementById('phoneStatus');
  if (speed > 0 && r.read_delay_s > 0) {
    statusEl.innerText = `👀 reading (${r.read_delay_s}s)...`;
    await new Promise(res => setTimeout(res, Math.min(r.read_delay_s * speed * 1000, 3500)));
  }
  const possBtn = turnNum > 0 ? registerPossBtn(turnNum, userTxt, prior) : '';
  if (!r.bubbles || !r.bubbles.length) {
    chatBody.insertAdjacentHTML('beforeend', `<div class="tg-bubble bot-sil">🤫 [${esc(r.action)}] ${r.disconnect_after_s !== null ? '— leaves in ' + r.disconnect_after_s + 's' : ''} ${possBtn}</div>`);
  } else {
    for (let i = 0; i < r.bubbles.length; i++) {
      const b = r.bubbles[i];
      if (speed > 0 && b.pause_before_s > 0) { statusEl.innerText = `⏸️ pausing (${b.pause_before_s}s)...`; await new Promise(res => setTimeout(res, Math.min(b.pause_before_s * speed * 1000, 2500))); }
      if (speed > 0 && b.typing_s > 0) { statusEl.innerText = `⌨️ typing (${b.typing_s}s)...`; await new Promise(res => setTimeout(res, Math.min(b.typing_s * speed * 1000, 2500))); }
      const kbdHtml = (i === r.bubbles.length - 1 && r.kbd && r.kbd.length) ? `<div class="kbd-row">${r.kbd.map(k => `<button type="button" class="kbd-btn" onclick="sendLiveText('${esc(k).replace(/'/g,"\\'")}')">${esc(k)}</button>`).join('')}</div>` : '';
      const bCls = r.class_id === 99 ? 'bot-rst' : (r.action === 'disconnect' ? 'bot-disc' : 'bot');
      chatBody.insertAdjacentHTML('beforeend', `<div class="tg-bubble ${bCls}"><div>${esc(b.text)}</div>${kbdHtml}<div class="tg-meta"><span>${classBadge(r.class_id)} ${b.typing_s ? '⌨️ '+b.typing_s+'s' : ''} ${b.pause_before_s ? '⏸️ '+b.pause_before_s+'s' : ''}</span><span>${i === r.bubbles.length - 1 ? possBtn : ''} <span class="time-badge">⚡ ${ms.toFixed(2)}ms</span></span></div></div>`);
      chatBody.scrollTop = chatBody.scrollHeight;
    }
  }
  if (r.disconnect_after_s !== null) chatBody.insertAdjacentHTML('beforeend', `<div class="tg-bubble bot-disc">🔌 Bot disconnected after ${r.disconnect_after_s}s — click "🔀 New".</div>`);
  statusEl.innerText = `🟢 online (${r.persona ? r.persona.type : 'ready'})`;
  chatBody.scrollTop = chatBody.scrollHeight;
  document.getElementById('chronoLogTable').insertAdjacentHTML('afterbegin', `<tr style="border-bottom:1px solid #1e293b;"><td style="padding:3px;">#${turnNum}</td><td style="padding:3px;color:#38bdf8;">${esc(userTxt)}</td><td style="padding:3px;">${classBadge(r.class_id)}</td><td style="padding:3px;">${esc(r.reply)} ${possBtn}</td><td style="padding:3px;color:#10b981;font-family:monospace;">${(r.read_delay_s||0).toFixed(1)}s / ${r.disconnect_after_s!==null?('L:'+r.disconnect_after_s+'s'):('I:'+(r.idle_timeout_s||0).toFixed(0)+'s')}</td></tr>`);
}

async function executeScenario(conv) {
  const results = [], mode = getMode(); let prior = [], uid = 20000 + (conv.id % 10000);
  if (mode === 'behavior') await fetch('/api/start_chat', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ uid }) });
  else await sendToAI("next", uid);
  for (let i = 0; i < conv.steps.length; i++) {
    const st = conv.steps[i], d = await sendToAI(st.text, uid);
    const r = d.ok ? d.response : { class_id:-1, leaf_id:-1, intent:"error", reply:d.error||"offline", bubbles:[], action:"error", kbd:[] };
    const expCls = (mode === 'vm' && st.expect_class_vm !== undefined) ? st.expect_class_vm : st.expect_class;
    const passed = expCls === undefined ? null : ((r.class_id === expCls) && (st.expect_leaf === undefined || r.leaf_id === -1 || r.leaf_id === st.expect_leaf));
    results.push({ turn:i+1, text:st.text, note:st.note||"", expect_class:expCls, resp:r, ms:d.ok?d.latency_ms:0, passed, priorTexts:prior.slice() });
    if (r.class_id === 99) prior = []; else prior.push(st.text);
  }
  conv.results = results; return results;
}
async function runSelectedScenarioLive() { const c = allConvs.find(x => x.id === selectedConvId); if (!c) return; await executeScenario(c); updateSmartnessBoard(); renderSidebar(); renderSelectedConv(); }
async function runFullSmartnessBenchmark() {
  await pingServer(); if (!serverOnline) return alert("Start `python production_server.py` on :8080 first!");
  for (const c of allConvs.filter(x => x.is_builtin)) await executeScenario(c);
  updateSmartnessBoard(); renderSidebar(); renderSelectedConv();
}
async function runPersonaAndBehaviorAudit() {
  await pingServer(); if (!serverOnline) return alert("Start `python production_server.py` on :8080 first!");
  const d = await (await fetch('/api/persona_audit', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ trials: 200 }) })).json();
  if (!d.ok) return alert(d.error);
  document.getElementById('personaAuditSummary').innerText = `🎭 200 Rolls: Female ${d.female_pct}% | Male Normal ${d.male_normal_pct}% | Ghoster ${d.male_ghoster_pct}% | Gates: ${d.strikes_ok}/${d.strikes_total}`;
  alert(`🎭 Single-Server Behavior & Persona Audit (200 fresh matches):\n• Female: ${d.female_pct}% (Target ~8%)\n• Male Normal: ${d.male_normal_pct}% (Target ~18.4%)\n• Male Ghoster: ${d.male_ghoster_pct}% (Target ~73.6%)\n• Opening Burst Valid: ${d.opening_ok}\n• Underage / Rapid Burst / Step 18 Gates: ${d.strikes_ok}/${d.strikes_total}\n• Avg Latency: ${d.avg_ms.toFixed(2)}ms`);
}

function updateSmartnessBoard() {
  const catStats = {}; let totPass = 0, totChecks = 0, totMs = 0, totCnt = 0;
  for (const c of allConvs) {
    if (!c.results) continue; const cat = c.category || "Other"; if (!catStats[cat]) catStats[cat] = { pass:0, total:0 };
    for (const r of c.results) { totMs += r.ms; totCnt++; if (r.passed !== null) { totChecks++; catStats[cat].total++; if (r.passed) { totPass++; catStats[cat].pass++; } } }
  }
  const pct = totChecks ? (totPass / totChecks) * 100 : 0;
  document.getElementById('smartOverallPct').innerText = totChecks ? `${pct.toFixed(1)}%` : '--%';
  document.getElementById('smartPassedTxt').innerText = `${totPass} / ${totChecks}`;
  document.getElementById('smartAvgMs').innerText = `${(totCnt ? totMs / totCnt : 0).toFixed(2)} ms/req`;
  const bar = document.getElementById('smartOverallBar'), grade = document.getElementById('smartGrade');
  bar.style.width = `${pct}%`; bar.style.background = pct >= 95 ? '#10b981' : (pct >= 80 ? '#f59e0b' : '#ef4444');
  grade.innerText = !totChecks ? 'Click "Run Full Audit (All 18 Steps)"' : (pct >= 98 ? '🧠 100% ULTRA SMART (Zero Faults)' : (pct >= 85 ? '🟢 SMART (Check Red Cards)' : '🔴 RUN python offline_train.py'));
  document.getElementById('smartCatsGrid').innerHTML = Object.entries(catStats).map(([cat, st]) => {
    const cp = st.total ? (st.pass / st.total) * 100 : 0, col = cp === 100 ? '#10b981' : (cp >= 75 ? '#f59e0b' : '#ef4444');
    return `<div class="cat-card"><div class="cat-lbl">${esc(cat)}</div><div class="cat-val" style="color:${col};"><span>${cp.toFixed(0)}%</span><span style="font-size:0.72rem;color:#94a3b8;">${st.pass}/${st.total}</span></div><div class="pbar-bg"><div class="pbar-fill" style="width:${cp}%;background:${col};"></div></div></div>`;
  }).join('');
}

function renderStepInputs(arr) {
  document.getElementById('stepsContainer').innerHTML = arr.map((t, i) => `<div class="msg-step-chip"><span style="font-size:0.7rem;color:var(--muted);">#${i+1}</span><input type="text" class="step-txt" value="${esc(t)}"><button type="button" class="del-step" onclick="removeStepInput(${i})">✕</button></div>`).join('');
}
function getStepsFromDOM() { return [...document.querySelectorAll('.step-txt')].map(el => el.value.trim()).filter(Boolean); }
function addStepInput(t='') { const s = getStepsFromDOM(); s.push(t); renderStepInputs(s); }
function removeStepInput(idx) { const s = getStepsFromDOM(); s.splice(idx, 1); renderStepInputs(s.length ? s : ['hi']); }
function loadQuickArray() { const p = document.getElementById('quickArrayInput').value.split(',').map(x=>x.trim()).filter(Boolean); if (p.length) renderStepInputs(p); }

async function runCustomBatchSimulation() {
  const msgs = getStepsFromDOM(); if (!msgs.length) return alert("Add at least 1 message!");
  const repeats = parseInt(document.getElementById('simRepeats').value || '32', 10);
  const d = await (await fetch('/api/run_batch', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ messages: msgs, repeats, mode: getMode() }) })).json();
  if (!d.ok) return alert(d.error || "Batch failed");
  const nc = { id: Date.now() % 100000, name:`🧪 Batch (${repeats}x): ${msgs.slice(0,3).join(' → ')}`, category:"Batch Sim", uid:`${repeats}x`, is_builtin:false, desc:`${repeats} runs across independent UIDs.`, steps: msgs.map(m => ({text:m})), results:null, batchRuns:d.per_turn };
  allConvs.unshift(nc); filteredConvs = allConvs.slice(); selectedConvId = nc.id; selectedTurnIdx = null; renderSidebar(); renderSelectedConv();
}
async function loadLocalJsonl() { const d = await (await fetch('/api/read_local')).json(); if (!d.ok) return alert(d.error); importConversations(d.convs); }
function handleFileUpload(e) {
  const f = e.target.files[0]; if (!f) return; const rd = new FileReader();
  rd.onload = async (ev) => { const d = await (await fetch('/api/parse_upload', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ filename: f.name, content: ev.target.result }) })).json(); if (d.ok) importConversations(d.convs); };
  rd.readAsText(f);
}
function importConversations(convs) {
  if (!convs || !convs.length) return alert("No valid user messages found.");
  allConvs = BUILTIN_SCENARIOS.map(b => allConvs.find(x => x.id === b.id) || { ...b, results:null, batchRuns:null }).concat(convs.map(c => ({ ...c, results:null, batchRuns:null })));
  filteredConvs = allConvs.slice(); selectedConvId = convs[0].id; renderSidebar(); renderSelectedConv();
}
function applyFilters() {
  const q = document.getElementById('searchBox').value.trim().toLowerCase();
  filteredConvs = allConvs.filter(c => !q || c.name.toLowerCase().includes(q) || (c.category||'').toLowerCase().includes(q) || c.steps.map(s=>s.text).join(' ').toLowerCase().includes(q));
  if (filteredConvs.length && !filteredConvs.some(c => c.id === selectedConvId)) selectedConvId = filteredConvs[0].id;
  renderSidebar(); renderSelectedConv();
}
function selectConv(id) { selectedConvId = id; selectedTurnIdx = null; renderSidebar(); renderSelectedConv(); }

function renderSidebar() {
  document.getElementById('sidebarCount').innerText = `${filteredConvs.length} scenarios / logs`;
  document.getElementById('sidebarList').innerHTML = filteredConvs.map(c => {
    let sb = `<span class="badge-cnt">${c.steps.length}</span>`;
    if (c.results) {
      const ch = c.results.filter(r => r.passed !== null);
      if (ch.length) { const p = ch.filter(r => r.passed).length; sb = p === ch.length ? `<span class="badge-pass">100% (${p}/${ch.length})</span>` : `<span class="badge-fail">${((p/ch.length)*100).toFixed(0)}% (${p}/${ch.length})</span>`; }
      else sb = `<span class="badge-pass">Ran ✓</span>`;
    } else if (c.batchRuns) sb = `<span class="badge-pass">${c.uid}</span>`;
    return `<div class="side-item ${c.id === selectedConvId ? 'active' : ''}" onclick="selectConv(${c.id})"><div class="side-top"><span>${esc(c.category)}</span>${sb}</div><div class="side-title">${esc(c.name)}</div><div class="side-seq">${esc(c.steps.map(s=>s.text).join(' → '))}</div></div>`;
  }).join('');
}

function renderSelectedConv() {
  window._possReg = {}; _possId = 0;
  const mainEl = document.getElementById('mainContent'), conv = filteredConvs.find(c => c.id === selectedConvId);
  if (!conv) { mainEl.innerHTML = ''; return; }
  const viewMode = document.getElementById('viewMode').value, outFilter = document.getElementById('outcomeFilter').value;
  const pillsHtml = conv.steps.map((s, idx) => {
    const r = conv.results ? conv.results[idx] : null;
    const cls = (r ? (r.passed === true ? 'pass' : (r.passed === false ? 'fail' : '')) : '') + ((viewMode==='last_turn' && selectedTurnIdx===idx+1)?' active':'');
    return `<span class="turn-pill ${cls}" onclick="selectedTurnIdx=${idx+1};document.getElementById('viewMode').value='last_turn';renderSelectedConv();">#${idx+1}: ${esc(s.text)}</span>`;
  }).join(' → ');

  let bodyHtml = '';
  if (conv.batchRuns) {
    bodyHtml = `<div class="turns-grid">` + conv.batchRuns.map((tObj, idx) => {
      const possBtn = registerPossBtn(idx + 1, tObj.text, conv.steps.slice(0, idx).map(x => x.text));
      return `<div class="tree-node"><div class="tree-node-top"><span>Turn #${idx+1}: 👤 "${esc(tObj.text)}"</span><span>${possBtn}</span></div>` +
        tObj.outcomes.map(o => `<div style="border-top:1px solid #222e35;padding-top:5px;"><div style="display:flex;justify-content:space-between;font-size:0.75rem;"><span>${classBadge(o.class_id)} <code>${esc(o.action)}</code></span><strong style="color:#38bdf8;">${o.pct}% (${o.count}x)</strong></div><div class="pbar-bg"><div class="pbar-fill" style="width:${o.pct}%;background:#38bdf8;"></div></div><div style="font-size:0.78rem;color:#cbd5e1;margin-top:3px;">🤖 ${esc(o.reply)}</div></div>`).join('') + `</div>`;
    }).join('') + `</div>`;
  } else if (!conv.results) {
    bodyHtml = `<div class="tree-node" style="margin-top:11px;text-align:center;padding:22px;"><div>Ready to run <strong>${conv.steps.length}</strong> turns against <code>127.0.0.1:8080</code> (${getMode().toUpperCase()}).</div><div style="margin-top:10px;"><button type="button" class="btn-green" onclick="runSelectedScenarioLive()">▶ Run Scenario Now</button></div></div>`;
  } else {
    let items = conv.results;
    if (viewMode === 'last_turn') items = items.filter(r => r.turn === (selectedTurnIdx || conv.results.length));
    if (outFilter === 'pass') items = items.filter(r => r.passed === true);
    if (outFilter === 'fail') items = items.filter(r => r.passed === false);
    bodyHtml = `<div class="turns-grid">` + items.map(r => {
      const possBtn = registerPossBtn(r.turn, r.text, r.priorTexts);
      const stB = r.passed === true ? `<span class="badge-pass">✅ PASS</span>` : (r.passed === false ? `<span class="badge-fail">❌ FAIL (Exp C${r.expect_class})</span>` : `<span class="badge-cnt">Info</span>`);
      const bCls = r.resp.action === 'disconnect' ? 'bot-disc' : (r.resp.action === 'silent' ? 'bot-sil' : 'bot');
      return `<div class="tree-node ${r.passed===false?'fail-card':''}"><div class="tree-node-top"><span>Turn #${r.turn} • ⚡ ${r.ms.toFixed(2)}ms</span><span style="display:flex;gap:5px;">${stB} ${possBtn}</span></div><div class="tg-chat"><div class="tg-bubble user">${esc(r.text)}</div><div class="tg-bubble ${bCls}"><div>${esc(r.resp.reply)}</div><div class="tg-meta"><span>${classBadge(r.resp.class_id)} ${esc(r.resp.intent)}</span><span>${r.resp.persona?esc(r.resp.persona.type):''}</span></div></div></div><div class="mem-mini"><span>Read: <strong>${(r.resp.read_delay_s||0).toFixed(1)}s</strong></span><span>Bubbles: <strong>${(r.resp.bubbles||[]).length}</strong></span><span>${r.resp.disconnect_after_s!==null?('Leave: <strong>'+r.resp.disconnect_after_s+'s</strong>'):('Idle: <strong>'+(r.resp.idle_timeout_s||0).toFixed(0)+'s</strong>')}</span></div></div>`;
    }).join('') + `</div>`;
  }
  mainEl.innerHTML = `<div class="conv-header-card"><div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;"><div><strong style="font-size:0.96rem;color:var(--accent);">${esc(conv.name)}</strong><div style="font-size:0.75rem;color:var(--muted);">${esc(conv.desc||'')}</div></div><button type="button" class="btn-green" onclick="runSelectedScenarioLive()">▶ Run Live</button></div><div class="turn-pills">${pillsHtml}</div></div>${bodyHtml}`;
}

function toggleChronoPanel() { document.getElementById('chronoPanel').classList.toggle('hidden'); }
renderStepInputs(["hi", "f", "im 24", "from?", "you didnt answer me", "Bcz you make it to fast."]);
renderSidebar(); renderSelectedConv();
pingServer().then(() => { if (serverOnline) { resetLivePhone(); runFullSmartnessBenchmark(); } });
</script>
</body>
</html>
"""


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class StudioHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, obj, code=200):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _ai_online(self):
        try:
            s = socket.create_connection((AI_HOST, AI_PORT), timeout=0.5)
            s.close()
            return True
        except Exception:
            return False

    def do_GET(self):
        if self.path == "/":
            raw = HTML_CONTENT.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/api/ping":
            self._send_json({"online": self._ai_online()})
        elif self.path == "/api/config":
            if not os.path.exists(CONFIG_PATH):
                self._send_json({"ok": False, "error": f"{CONFIG_PATH} not found."}); return
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                self._send_json({"ok": True, "content": f.read()})
        elif self.path == "/api/read_local":
            if not os.path.exists("bot_chat_history.jsonl"):
                self._send_json({"ok": False, "error": "bot_chat_history.jsonl not found."}); return
            with open("bot_chat_history.jsonl", "r", encoding="utf-8", errors="ignore") as f:
                self._send_json({"ok": True, "convs": parse_jsonl_conversations(f, 300)})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", errors="ignore")
        try: data = json.loads(body) if body else {}
        except Exception: self._send_json({"ok": False, "error": "Invalid JSON"}, 400); return

        if self.path == "/api/send":
            try:
                ms, resp = query_ai(data.get("text", ""), data.get("uid", 101), data.get("mode", "behavior"), data.get("event_type", "text"), data.get("emoji", ""))
                self._send_json({"ok": True, "latency_ms": ms, "response": resp})
            except Exception as e: self._send_json({"ok": False, "error": str(e)})

        elif self.path == "/api/start_chat":
            try:
                _, raw = _raw_http_post("/start", b"", [("X-User-Id", str(int(data.get("uid", 101)) & 0x7FFFFFFF))])
                self._send_json({"ok": True, **raw})
            except Exception as e: self._send_json({"ok": False, "error": str(e)})

        elif self.path == "/api/opening":
            try:
                ms, raw = _raw_http_post("/opening", b"", [("X-User-Id", str(int(data.get("uid", 101)) & 0x7FFFFFFF))])
                self._send_json({"ok": True, "latency_ms": ms, "response": _normalize_response("behavior", raw)})
            except Exception as e: self._send_json({"ok": False, "error": str(e)})

        elif self.path == "/api/reload":
            try:
                _, raw = _raw_http_post("/reload", b"", [])
                self._send_json({"ok": True, **raw})
            except Exception as e: self._send_json({"ok": False, "error": str(e)})

        elif self.path == "/api/config":
            try:
                content = data.get("content", "")
                json.loads(content)
                with open(CONFIG_PATH, "w", encoding="utf-8") as f: f.write(content)
                t0 = time.perf_counter()
                n_pools = fast_repack_behavior_into_vault()
                repack_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                reloaded = False
                if self._ai_online():
                    _raw_http_post("/reload", b"", [])
                    reloaded = True
                self._send_json({"ok": True, "pools": n_pools, "repack_ms": repack_ms, "reloaded": reloaded})
            except Exception as e: self._send_json({"ok": False, "error": str(e)})

        elif self.path == "/api/persona_audit":
            try:
                trials = max(20, min(int(data.get("trials", 200)), 500))
                cnts, tot_ms = {"female": 0, "male_normal": 0, "male_ghoster": 0}, 0.0
                for i in range(trials):
                    ms, raw = _raw_http_post("/start", b"", [("X-User-Id", str(30000 + i))])
                    tot_ms += ms
                    p = raw.get("persona", "male_ghoster")
                    cnts[p] = cnts.get(p, 0) + 1
                _, ob = _raw_http_post("/opening", b"", [("X-User-Id", "30000")])
                opening_ok = (ob.get("action") == "opening_burst") and (1 <= len(ob.get("bubbles", [])) <= 4)
                strikes_ok = 0
                _, r_ua = query_ai("im 15", 45001, "behavior")
                if r_ua["action"] == "disconnect" and r_ua["class_id"] == 7: strikes_ok += 1
                _raw_http_post("/start", b"", [("X-User-Id", "45002")])
                _, r_mb = query_ai("hi || im 24 || from tunisia", 45002, "behavior")
                if r_mb["class_id"] in (1, 3, 5, 6): strikes_ok += 1
                _, r_q = query_ai("?", 45003, "behavior")
                _, r_h = query_ai("h0rny", 45004, "behavior")
                if r_q["class_id"] == 55 and r_h["class_id"] == 60: strikes_ok += 1
                self._send_json({
                    "ok": True, "trials": trials, "avg_ms": tot_ms / trials, "opening_ok": opening_ok,
                    "female_pct": round((cnts["female"] / trials) * 100.0, 1),
                    "male_normal_pct": round((cnts["male_normal"] / trials) * 100.0, 1),
                    "male_ghoster_pct": round((cnts["male_ghoster"] / trials) * 100.0, 1),
                    "strikes_ok": strikes_ok, "strikes_total": 3,
                })
            except Exception as e: self._send_json({"ok": False, "error": str(e)})

        elif self.path == "/api/turn_possibilities":
            prior, target = data.get("prior_texts", []), data.get("text", "")
            tries, repeat_send = max(1, min(int(data.get("tries", 64)), 400)), max(1, min(int(data.get("repeat_send", 1)), 5))
            vary_u, mode = data.get("ip_mode", "vary") == "vary", data.get("mode", "behavior")
            groups, total_ms = {}, 0.0
            try:
                for i in range(tries):
                    uid = (10000 + i) if vary_u else 10000
                    if mode == "behavior": _raw_http_post("/start", b"", [("X-User-Id", str(uid))])
                    else: query_ai("next", uid, "vm")
                    for pt in prior: query_ai(pt, uid, mode)
                    last_ms, last_r = 0.0, {}
                    for _ in range(repeat_send): last_ms, last_r = query_ai(target, uid, mode)
                    total_ms += last_ms
                    key = (last_r.get("class_id", -1), last_r.get("action", ""), last_r.get("intent", ""), last_r.get("reply", ""))
                    if key not in groups: groups[key] = {"class_id": key[0], "action": key[1], "intent": key[2], "reply": key[3], "count": 0, "sum_ms": 0.0}
                    groups[key]["count"] += 1; groups[key]["sum_ms"] += last_ms
                outcomes = [{**g, "pct": round((g["count"] / tries) * 100.0, 1), "avg_ms": g["sum_ms"] / g["count"]} for g in sorted(groups.values(), key=lambda x: x["count"], reverse=True)]
                self._send_json({"ok": True, "tries": tries, "avg_ms": total_ms / tries, "outcomes": outcomes})
            except Exception as e: self._send_json({"ok": False, "error": str(e)})

        elif self.path == "/api/run_batch":
            msgs, repeats, mode = data.get("messages", []), max(1, min(int(data.get("repeats", 32)), 200)), data.get("mode", "behavior")
            per_turn, total_ms, total_calls = [{} for _ in msgs], 0.0, 0
            try:
                for i in range(repeats):
                    uid = 25000 + i
                    if mode == "behavior": _raw_http_post("/start", b"", [("X-User-Id", str(uid))])
                    else: query_ai("next", uid, "vm")
                    for t_idx, txt in enumerate(msgs):
                        ms, resp = query_ai(txt, uid, mode)
                        total_ms += ms; total_calls += 1
                        key = (resp.get("class_id", -1), resp.get("action", ""), resp.get("reply", ""))
                        grp = per_turn[t_idx]
                        if key not in grp: grp[key] = {"class_id": key[0], "action": key[1], "reply": key[2], "count": 0}
                        grp[key]["count"] += 1
                out = [{"text": txt, "outcomes": [{**it, "pct": round((it["count"] / repeats) * 100.0, 1)} for it in sorted(per_turn[i].values(), key=lambda x: x["count"], reverse=True)]} for i, txt in enumerate(msgs)]
                self._send_json({"ok": True, "avg_ms": total_ms / max(total_calls, 1), "per_turn": out})
            except Exception as e: self._send_json({"ok": False, "error": str(e)})

        elif self.path == "/api/parse_upload":
            self._send_json({"ok": True, "convs": parse_jsonl_conversations(data.get("content", "").splitlines(), 300)})
        else:
            self.send_response(404); self.end_headers()


if __name__ == "__main__":
    with ThreadingHTTPServer(("", PORT), StudioHandler) as httpd:
        print(f"[ULTRA AI TEST STUDIO] Open http://127.0.0.1:{PORT}")
        print(f" -> Connects to SINGLE production_server.py on http://{AI_HOST}:{AI_PORT}")
        httpd.serve_forever()