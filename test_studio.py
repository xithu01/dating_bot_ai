# FILE: test_studio.py
# PURPOSE: Ultra Telegram-Style AI Test Analyzer, Memory Inspector & Smartness Benchmark
# USAGE: Start `python production_server.py`, then run `python test_studio.py`
#        Open http://127.0.0.1:8090 in your browser.

import http.server
import socketserver
import socket
import json
import time
import os

PORT = 8090
AI_HOST = "127.0.0.1"
AI_PORT = 8080


def query_ai_server(text, src_ip=None):
    """Send a compact raw HTTP/1.1 POST to production_server.py (fits in 512-byte read)."""
    payload = text.encode("utf-8", errors="ignore")
    req = (
        f"POST / HTTP/1.1\r\nHost: {AI_HOST}:{AI_PORT}\r\n"
        f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
    ).encode("ascii") + payload

    t0 = time.perf_counter()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.5)
    try:
        if src_ip and src_ip != "127.0.0.1":
            try:
                s.bind((src_ip, 0))
            except OSError:
                pass
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
        parsed = {"class_id": -1, "is_greeting": False, "intent": "parse_error", "reply": body[:160], "kbd": []}
    return elapsed_ms, parsed


def parse_jsonl_conversations(lines, max_convs=300):
    """Parse large JSONL/TXT chat logs into multi-turn conversations split on Disconnect/next/UID."""
    convs = []
    cur_turns = []
    cur_uid = "User"
    cur_ts = ""

    def flush():
        nonlocal cur_turns
        if cur_turns:
            convs.append({
                "id": 1000 + len(convs) + 1,
                "name": f"Log #{len(convs) + 1} ({cur_uid})",
                "category": "Imported Log",
                "uid": str(cur_uid),
                "time_str": cur_ts or "JSONL",
                "is_builtin": False,
                "steps": [{"text": t} for t in cur_turns[:40]],
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
            uid = str(obj.get("uid", cur_uid))
            sender = str(obj.get("sender", "")).lower()
            text = str(obj.get("text", "")).strip()
            ts = obj.get("ts")
            if uid != cur_uid and cur_turns:
                flush()
            cur_uid = uid
            if ts and not cur_ts:
                try:
                    cur_ts = time.strftime("%H:%M:%S", time.localtime(int(ts)))
                except Exception:
                    cur_ts = str(ts)
            if sender == "user" and text:
                cur_turns.append(text)
            elif "disconnect" in text.lower() or text.lower() in ("next", "/next"):
                cur_turns.append("next")
                flush()
                cur_ts = ""
        else:
            if line.lower() in ("next", "/next", "disconnect"):
                cur_turns.append("next")
                flush()
            else:
                cur_turns.append(line)
                if len(cur_turns) >= 15:
                    flush()
    flush()
    return convs


HTML_CONTENT = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dating AI Analyzer — Smartness %, Memory Inspector & Possibilities</title>
<style>
:root {
    --bg-main:#0b141a; --bg-panel:#111b21; --bg-card:#182229; --bg-chat:#0e1621;
    --border:#222e35; --accent:#38bdf8; --user-bubble:#2b5278; --bot-bubble:#1e3a5f;
    --bot-repeat:#6b2146; --bot-reset:#334155; --text-main:#e9edef; --text-muted:#8696a0;
    --ok:#10b981; --bad:#ef4444; --warn:#f59e0b;
    --c0:#dc2626; --c1:#16a34a; --c2:#ca8a04; --c3:#ea580c; --c4:#9333ea;
    --c5:#0284c7; --c6:#0d9488; --c7:#db2777; --c8:#6366f1; --c99:#475569;
}
* { box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg-main); color:var(--text-main); margin:0; display:flex; height:100vh; overflow:hidden; }

/* Sidebar */
.sidebar { width:330px; background:var(--bg-panel); border-right:1px solid var(--border); display:flex; flex-direction:column; flex-shrink:0; }
.sidebar-header { padding:12px 14px; border-bottom:1px solid var(--border); background:#202c33; display:flex; justify-content:space-between; align-items:center; gap:8px; }
.sidebar-header h2 { margin:0; font-size:0.95rem; color:var(--accent); }
.sidebar-actions { padding:10px 12px; border-bottom:1px solid var(--border); display:flex; flex-direction:column; gap:7px; background:#152028; }
.sidebar-list { overflow-y:auto; flex:1; }
.side-item { padding:10px 14px; border-bottom:1px solid var(--border); cursor:pointer; transition:background 0.15s; }
.side-item:hover { background:#202c33; }
.side-item.active { background:#2a3942; border-left:4px solid var(--accent); }
.side-top { display:flex; justify-content:space-between; font-size:0.73rem; color:var(--text-muted); margin-bottom:3px; gap:6px; align-items:center; }
.side-title { font-size:0.86rem; font-weight:700; color:#fff; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.side-seq { font-size:0.73rem; color:var(--text-muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; margin-top:2px; }

/* Main & Toolbar */
.main { flex:1; display:flex; flex-direction:column; overflow:hidden; }
.toolbar { padding:10px 16px; background:#202c33; border-bottom:1px solid var(--border); display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
input, select, button { padding:7px 11px; border-radius:6px; border:1px solid #374248; background:var(--bg-panel); color:var(--text-main); font-size:0.82rem; }
button { cursor:pointer; font-weight:600; background:#0284c7; border-color:#0284c7; color:#fff; }
button:hover { background:#0369a1; }
button:disabled { opacity:0.5; cursor:not-allowed; }
.btn-green { background:#059669; border-color:#059669; } .btn-green:hover { background:#047857; }
.btn-amber { background:#d97706; border-color:#d97706; } .btn-amber:hover { background:#b45309; }
.btn-purple { background:#7c3aed; border-color:#7c3aed; } .btn-purple:hover { background:#6d28d9; }
.btn-dark { background:#2a3942; border-color:#374248; } .btn-dark:hover { background:#374248; }
.content-scroll { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:16px; }

/* Smartness Scoreboard */
.smart-panel { background:linear-gradient(135deg,#0c192c 0%,#162436 100%); border:1px solid #38bdf8; border-radius:12px; padding:14px 16px; display:flex; flex-direction:column; gap:12px; }
.smart-top { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; }
.smart-score-big { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; }
.smart-pct { font-size:2.1rem; font-weight:900; font-family:monospace; color:#10b981; }
.smart-grade { font-size:0.85rem; font-weight:700; padding:4px 10px; border-radius:999px; background:#064e3b; color:#6ee7b7; border:1px solid #10b981; }
.smart-cats { display:grid; grid-template-columns:repeat(auto-fit,minmax(148px,1fr)); gap:8px; }
.cat-card { background:#0b141a; border:1px solid #263545; border-radius:8px; padding:8px 10px; }
.cat-lbl { font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; font-weight:700; }
.cat-val { font-size:1.05rem; font-weight:800; font-family:monospace; margin-top:3px; display:flex; justify-content:space-between; align-items:center; }
.pbar-bg { width:100%; height:6px; background:#222e35; border-radius:99px; overflow:hidden; margin-top:5px; }
.pbar-fill { height:100%; border-radius:99px; transition:width 0.25s; }

/* Live Phone & Inspector */
.chrono-panel { background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%); border:1px solid #334155; border-radius:12px; padding:16px; display:flex; gap:18px; flex-wrap:wrap; align-items:flex-start; }
.chrono-panel.hidden { display:none; }
.tg-phone { width:410px; background:var(--bg-chat); border:2px solid #334155; border-radius:16px; overflow:hidden; display:flex; flex-direction:column; height:510px; flex-shrink:0; box-shadow:0 10px 30px rgba(0,0,0,0.45); }
.tg-phone-header { background:#202c33; padding:10px 14px; display:flex; align-items:center; gap:10px; border-bottom:1px solid var(--border); }
.tg-avatar { width:36px; height:36px; border-radius:50%; background:#0284c7; display:flex; align-items:center; justify-content:center; font-weight:bold; font-size:1.1rem; }
.tg-status { font-size:0.75rem; color:var(--accent); font-weight:600; }
.tg-phone-body { flex:1; padding:12px; overflow-y:auto; display:flex; flex-direction:column; gap:8px; }
.tg-phone-input { padding:8px 10px; background:#202c33; display:flex; gap:6px; border-top:1px solid var(--border); }
.tg-phone-input input { flex:1; background:var(--bg-panel); }
.chrono-dashboard { flex:1; min-width:320px; display:flex; flex-direction:column; gap:10px; }
.stopwatch-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(135px,1fr)); gap:8px; }
.sw-card { background:#0b141a; border:1px solid #334155; border-radius:9px; padding:9px 11px; }
.sw-label { font-size:0.69rem; color:var(--text-muted); text-transform:uppercase; }
.sw-val { font-size:1.3rem; font-weight:800; font-family:monospace; color:var(--accent); margin-top:2px; }
.sw-sub { font-size:0.71rem; color:#94a3b8; margin-top:2px; }
.mem-strip { background:#0b141a; border:1px solid #334155; border-radius:9px; padding:8px 12px; font-size:0.77rem; display:flex; gap:12px; flex-wrap:wrap; color:#cbd5e1; }
.mem-strip code { color:#38bdf8; font-weight:700; }

/* Batch Studio */
.studio-box { background:var(--bg-card); border:1px solid #334155; border-radius:10px; padding:14px 16px; }
.studio-top { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; margin-bottom:10px; }
.msg-steps-wrap { display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.msg-step-chip { display:flex; align-items:center; gap:4px; background:#111b21; border:1px solid #374248; border-radius:8px; padding:4px 8px; }
.msg-step-chip input.step-txt { width:145px; border:none; background:transparent; padding:4px; font-weight:600; color:#fff; }
.msg-step-chip .del-step { background:transparent; border:none; color:#f87171; padding:2px 5px; cursor:pointer; }

/* Bubbles & Badges */
.tg-chat { display:flex; flex-direction:column; gap:6px; }
.tg-bubble { max-width:90%; padding:8px 11px; border-radius:10px; font-size:0.84rem; line-height:1.35; word-wrap:break-word; }
.tg-bubble.user { background:var(--user-bubble); color:#fff; align-self:flex-end; border-bottom-right-radius:2px; }
.tg-bubble.bot { background:var(--bot-bubble); color:#fff; align-self:flex-start; border-bottom-left-radius:2px; border:1px solid #2a486b; }
.tg-bubble.bot-rep { background:var(--bot-repeat); color:#fff; align-self:flex-start; border-bottom-left-radius:2px; border:1px solid #9d174d; }
.tg-bubble.bot-rst { background:var(--bot-reset); color:#e2e8f0; align-self:center; border:1px dashed #94a3b8; font-size:0.79rem; }
.tg-bubble.bot-err { background:#450a0a; color:#fecaca; align-self:flex-start; border:1px solid #ef4444; }
.tg-meta { display:flex; justify-content:space-between; align-items:center; gap:6px; font-size:0.67rem; color:rgba(255,255,255,0.78); margin-top:5px; flex-wrap:wrap; }
.kbd-row { display:flex; gap:5px; flex-wrap:wrap; margin-top:6px; }
.kbd-btn { background:#111b21; border:1px solid #38bdf8; color:#7dd3fc; padding:3px 9px; border-radius:5px; font-size:0.72rem; cursor:pointer; }
.kbd-btn:hover { background:#0284c7; color:#fff; }
.badge { padding:2px 6px; border-radius:4px; font-size:0.68rem; font-weight:800; color:#fff; display:inline-block; }
.c0 { background:var(--c0); } .c1 { background:var(--c1); } .c2 { background:var(--c2); }
.c3 { background:var(--c3); } .c4 { background:var(--c4); } .c5 { background:var(--c5); }
.c6 { background:var(--c6); } .c7 { background:var(--c7); } .c8 { background:var(--c8); } .c99 { background:var(--c99); }
.badge-pass { background:#065f46; color:#6ee7b7; border:1px solid #10b981; padding:2px 7px; border-radius:99px; font-size:0.7rem; font-weight:700; }
.badge-fail { background:#7f1d1d; color:#fecaca; border:1px solid #ef4444; padding:2px 7px; border-radius:99px; font-size:0.7rem; font-weight:700; }
.badge-cnt { padding:2px 8px; border-radius:99px; font-size:0.72rem; background:#2a3942; color:#cbd5e1; font-weight:700; }
.time-badge { background:rgba(0,0,0,0.35); padding:1px 6px; border-radius:4px; color:#7dd3fc; font-family:monospace; }
.btn-poss { background:rgba(15,23,42,0.85); border:1px solid #fbbf24; color:#fde68a; padding:2px 7px; font-size:0.67rem; border-radius:5px; cursor:pointer; font-weight:700; }
.btn-poss:hover { background:#0284c7; color:#fff; border-color:#38bdf8; }

/* Selected Scenario Content */
.conv-header-card { background:var(--bg-card); border:1px solid var(--border); border-radius:10px; padding:12px 16px; }
.turn-pills { display:flex; gap:6px; flex-wrap:wrap; align-items:center; margin-top:8px; }
.turn-pill { padding:4px 10px; border-radius:999px; background:#202c33; border:1px solid #374248; font-size:0.78rem; cursor:pointer; }
.turn-pill.active { background:var(--user-bubble); border-color:var(--accent); color:#fff; font-weight:700; }
.turn-pill.pass { border-color:#10b981; }
.turn-pill.fail { border-color:#ef4444; background:#450a0a; }
.turns-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:12px; margin-top:12px; }
.tree-node { background:var(--bg-panel); border:1px solid var(--border); border-radius:10px; padding:12px; display:flex; flex-direction:column; gap:8px; }
.tree-node.fail-card { border-color:#ef4444; background:#1c1317; }
.tree-node-top { display:flex; justify-content:space-between; align-items:center; font-size:0.8rem; font-weight:700; gap:8px; flex-wrap:wrap; }
.mem-mini { font-size:0.72rem; color:#94a3b8; background:#0b141a; border:1px solid #222e35; border-radius:6px; padding:5px 8px; display:flex; gap:8px; flex-wrap:wrap; }

/* Modal */
.modal-backdrop { position:fixed; inset:0; background:rgba(0,0,0,0.82); backdrop-filter:blur(3px); display:none; align-items:center; justify-content:center; z-index:9999; padding:18px; }
.modal-backdrop.open { display:flex; }
.modal-box { background:#111b21; border:1px solid #38bdf8; border-radius:14px; width:100%; max-width:1040px; max-height:90vh; display:flex; flex-direction:column; overflow:hidden; box-shadow:0 20px 50px rgba(0,0,0,0.75); }
.modal-header { padding:14px 18px; background:#202c33; border-bottom:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }
.modal-history-strip { padding:9px 18px; background:#0b141a; border-bottom:1px solid var(--border); font-size:0.76rem; display:flex; gap:6px; flex-wrap:wrap; align-items:center; max-height:90px; overflow-y:auto; }
.hist-chip-u { background:#2b5278; color:#fff; padding:2px 8px; border-radius:6px; }
.modal-controls { padding:12px 18px; background:#182229; border-bottom:1px solid var(--border); display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.modal-mem-bar { padding:8px 18px; background:#0e1621; border-bottom:1px solid var(--border); font-size:0.76rem; color:#94a3b8; display:flex; gap:14px; flex-wrap:wrap; }
.modal-body { padding:16px 18px; overflow-y:auto; flex:1; display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:12px; align-content:start; }
</style>
</head>
<body>

<!-- POSSIBILITIES MODAL -->
<div class="modal-backdrop" id="possModal" onclick="if(event.target===this) closePossModal()">
  <div class="modal-box">
    <div class="modal-header">
      <div>
        <strong style="font-size:1.05rem;color:var(--accent);" id="modalTitle">🎲 All AI Reply Possibilities</strong>
        <div style="font-size:0.75rem;color:var(--text-muted);margin-top:2px;">Replays Exact Old Messages Before This Turn Across N Tries (Verifies Reply Pool Rotation &amp; Zero-Faux Consistency)</div>
      </div>
      <button type="button" class="btn-dark" onclick="closePossModal()">✕ Close</button>
    </div>
    <div class="modal-history-strip" id="modalHistoryStrip"></div>
    <div class="modal-controls">
      <label style="font-size:0.79rem;color:var(--text-muted);">Msg at Turn:</label>
      <input type="text" id="modalUserText" style="min-width:175px;font-weight:700;" onkeydown="if(event.key==='Enter') fetchTurnPossibilities()">
      <select id="modalPresetSelect" onchange="if(this.value){document.getElementById('modalUserText').value=this.value;fetchTurnPossibilities();}">
        <option value="">⚡ Presets...</option>
        <option value="hi">hi (Class 1)</option>
        <option value="heyy">heyy (Class 1)</option>
        <option value="age?">age? (Class 2)</option>
        <option value="im 24">im 24 (Class 3)</option>
        <option value="from?">from? (Class 4)</option>
        <option value="im from tuinsi">im from tuinsi (Class 5)</option>
        <option value="hi l'm 30 from tun">hi l'm 30 from tun (Class 6)</option>
        <option value="im 16">im 16 (Class 7 Underage)</option>
        <option value="how r u?">how r u? (Class 8 Step 3)</option>
        <option value="what!">what! (Class 0 Trap)</option>
        <option value="why?">why? (Class 0 Trap)</option>
        <option value="who are you">who are you (Class 0 Trap)</option>
        <option value="lol">lol (Class 0 Trap)</option>
        <option value="next">next (Class 99 Reset)</option>
      </select>
      <select id="modalRepeatSend" onchange="fetchTurnPossibilities()" title="Send message 1x, 2x, or 3x in a row to test repeat memory">
        <option value="1" selected>Sent 1x</option>
        <option value="2">Sent 2x (Repeat Memory)</option>
        <option value="3">Sent 3x (Spam)</option>
      </select>
      <select id="modalIpMode" onchange="fetchTurnPossibilities()" title="Varying IPs reveals all 32/8 rotating replies; Same IP verifies determinism">
        <option value="vary" selected>Varying Users (See All Replies %)</option>
        <option value="same">Same User + next Reset</option>
      </select>
      <label style="font-size:0.79rem;color:var(--text-muted);">Tries:</label>
      <input type="number" id="modalTriesInput" value="64" min="8" max="500" step="16" style="width:72px;text-align:center;font-weight:700;color:var(--accent);">
      <button type="button" class="btn-green" onclick="fetchTurnPossibilities()">🎲 Roll Possibilities</button>
    </div>
    <div class="modal-mem-bar" id="modalMemBar"></div>
    <div class="modal-body" id="modalOutcomesGrid"></div>
  </div>
</div>

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="sidebar-header">
    <div>
      <h2>🧪 AI Test Scenarios</h2>
      <div style="font-size:0.72rem;color:var(--text-muted);" id="sidebarCount">Loading...</div>
    </div>
    <span id="apiStatusBadge" class="badge-cnt">⚡ Checking</span>
  </div>
  <div class="sidebar-actions">
    <button type="button" class="btn-green" style="width:100%;" onclick="runFullSmartnessBenchmark()">🧠 Run Full Smartness Audit (All 10 Scenarios)</button>
    <div style="display:flex;gap:6px;">
      <button type="button" class="btn-dark" style="flex:1;font-size:0.76rem;" onclick="loadLocalJsonl()">📥 Load JSONL</button>
      <label class="btn-dark" style="flex:1;font-size:0.76rem;text-align:center;padding:7px 8px;border-radius:6px;cursor:pointer;border:1px solid #374248;font-weight:600;">
        📂 Upload File
        <input type="file" id="fileUploadInput" accept=".txt,.json,.jsonl" style="display:none;" onchange="handleFileUpload(event)">
      </label>
    </div>
  </div>
  <div class="sidebar-list" id="sidebarList"></div>
</div>

<!-- MAIN -->
<div class="main">
  <div class="toolbar">
    <input type="text" id="searchBox" placeholder="🔍 Filter scenarios, words, intents..." oninput="applyFilters()" style="min-width:210px;">
    <select id="viewMode" onchange="renderSelectedConv()">
      <option value="all_turns_tree" selected>🌳 Per-Turn Cards (Expected vs Actual + Memory)</option>
      <option value="last_turn">🎯 Focus: Selected Turn Only</option>
      <option value="full_threads">📱 Full Telegram Chat Transcript</option>
    </select>
    <select id="outcomeFilter" onchange="renderSelectedConv()">
      <option value="all">📊 All Turns</option>
      <option value="pass">✅ Passed Only</option>
      <option value="fail">❌ Failed / Faux Only</option>
    </select>
    <button type="button" class="btn-amber" onclick="toggleChronoPanel()">📱 Toggle Live Phone &amp; Memory HUD</button>
    <button type="button" class="btn-purple" onclick="runSelectedScenarioLive()">▶ Run Selected Scenario</button>
  </div>

  <div class="content-scroll">
    <!-- AI SMARTNESS PERCENTAGE SCOREBOARD -->
    <div class="smart-panel">
      <div class="smart-top">
        <div>
          <div style="font-size:0.74rem;color:var(--accent);text-transform:uppercase;font-weight:800;letter-spacing:0.5px;">🧠 AI Smartness &amp; Memory Accuracy Score</div>
          <div class="smart-score-big">
            <span class="smart-pct" id="smartOverallPct">--%</span>
            <span class="smart-grade" id="smartGrade">Click "Run Full Smartness Audit" to score your AI</span>
          </div>
        </div>
        <div style="text-align:right;font-size:0.79rem;color:#cbd5e1;">
          <div>Checks Passed: <strong id="smartPassedTxt" style="color:#10b981;">0 / 0</strong></div>
          <div>Avg AI Speed: <strong id="smartAvgMs" style="color:#38bdf8;">0.00 ms/req</strong></div>
        </div>
      </div>
      <div class="pbar-bg" style="height:9px;"><div class="pbar-fill" id="smartOverallBar" style="width:0%;background:#10b981;"></div></div>
      <div class="smart-cats" id="smartCatsGrid"></div>
    </div>

    <!-- LIVE TELEGRAM PHONE & MEMORY INSPECTOR -->
    <div class="chrono-panel" id="chronoPanel">
      <div class="tg-phone">
        <div class="tg-phone-header">
          <div class="tg-avatar">🤖</div>
          <div style="flex:1;">
            <div style="font-weight:700;font-size:0.9rem;">Dating Bot AI (127.0.0.1:8080)</div>
            <div class="tg-status" id="phoneStatus">🟢 online — live server connected</div>
          </div>
          <button type="button" class="btn-amber" style="padding:5px 9px;font-size:0.75rem;" onclick="resetLivePhone()" title="Send 'next' to wipe bot memory">🔀 New (next)</button>
        </div>
        <div class="tg-phone-body" id="phoneChatBody"></div>
        <div class="tg-phone-input">
          <input type="text" id="liveChatInput" placeholder="Type live msg (e.g. hi, im 24, tunisia, next)..." onkeydown="if(event.key==='Enter') sendLivePhoneMsg()">
          <button type="button" onclick="sendLivePhoneMsg()">Send</button>
        </div>
      </div>

      <div class="chrono-dashboard">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
          <strong style="font-size:1rem;color:var(--accent);">⚡ Live Latency &amp; Bot Memory Inspector</strong>
          <button type="button" class="btn-dark" onclick="clearLivePhoneOnly()">🧹 Clear Screen</button>
        </div>

        <div class="stopwatch-grid">
          <div class="sw-card">
            <div class="sw-label">⚡ Last Latency</div>
            <div class="sw-val" id="swLastMs">0.00ms</div>
            <div class="sw-sub" id="swLastClass">No message yet</div>
          </div>
          <div class="sw-card">
            <div class="sw-label">📊 Avg Latency</div>
            <div class="sw-val" id="swAvgMs" style="color:#10b981;">0.00ms</div>
            <div class="sw-sub" id="swTurnCnt">Turn 0</div>
          </div>
          <div class="sw-card">
            <div class="sw-label">🔁 Repeat Guard</div>
            <div class="sw-val" id="swRepeatArmed" style="color:#f59e0b;">OFF</div>
            <div class="sw-sub">Armed when last=Class 1</div>
          </div>
          <div class="sw-card">
            <div class="sw-label">👤 Saved Profile</div>
            <div class="sw-val" id="swProfileVal" style="color:#f472b6;font-size:1.05rem;">Empty</div>
            <div class="sw-sub">Age &amp; Country in RAM</div>
          </div>
        </div>

        <div class="mem-strip">
          <span>🧠 <strong>Last Intent:</strong> <code id="memLastIntent">None</code></span>
          <span>🔢 <strong>Msg Count:</strong> <code id="memMsgCnt">0</code></span>
          <span>🎂 <strong>Saved Age:</strong> <code id="memAge">None</code></span>
          <span>🌍 <strong>Saved Country:</strong> <code id="memCountry">None</code></span>
          <span>🛡️ <strong>Status:</strong> <code id="memStatus">Fresh Session</code></span>
        </div>

        <div style="background:#0b141a;border:1px solid #334155;border-radius:9px;padding:10px 12px;max-height:220px;overflow-y:auto;">
          <div style="font-size:0.74rem;color:var(--text-muted);text-transform:uppercase;margin-bottom:6px;font-weight:700;">📊 Real-Time Inference Log (Click 🎲 on any row to inspect all reply possibilities!)</div>
          <table style="width:100%;border-collapse:collapse;font-size:0.77rem;text-align:left;">
            <thead>
              <tr style="border-bottom:1px solid #222e35;color:#94a3b8;">
                <th style="padding:4px;">#</th>
                <th style="padding:4px;">User Input</th>
                <th style="padding:4px;">Class</th>
                <th style="padding:4px;">Intent / Bot Reply</th>
                <th style="padding:4px;">Latency</th>
              </tr>
            </thead>
            <tbody id="chronoLogTable"></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- MULTI-MESSAGE ARRAY BATCH SIMULATOR -->
    <div class="studio-box">
      <div class="studio-top">
        <div>
          <strong style="color:var(--accent);font-size:0.94rem;">🧪 Multi-Message Sequence &amp; Memory Batch Simulator</strong>
          <span style="font-size:0.77rem;color:var(--text-muted);margin-left:6px;">Test any custom chat sequence across N runs to inspect every reply variation %:</span>
        </div>
        <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
          <input type="text" id="quickArrayInput" placeholder="e.g. hi, hi, age?, im 24, tunisia, next, hi" style="width:270px;">
          <button type="button" class="btn-dark" onclick="loadQuickArray()">📥 Load Array</button>
          <button type="button" class="btn-dark" onclick="addStepInput('')">➕ Add Msg</button>
          <select id="simRepeats" style="width:92px;">
            <option value="16">16 Runs</option>
            <option value="32" selected>32 Runs</option>
            <option value="64">64 Runs</option>
            <option value="100">100 Runs</option>
          </select>
          <button type="button" class="btn-green" onclick="runCustomBatchSimulation()">🚀 Simulate Batch</button>
        </div>
      </div>
      <div class="msg-steps-wrap" id="stepsContainer"></div>
    </div>

    <!-- SELECTED SCENARIO / CONVERSATION VIEW -->
    <div id="mainContent"></div>
  </div>
</div>

<script>
const CLASS_NAMES = {
  0:"not_valid", 1:"greeting", 2:"ask_age", 3:"tell_age", 4:"ask_country",
  5:"tell_country", 6:"age+country", 7:"underage", 8:"ask_hru", 99:"reset"
};

// 10 BUILT-IN READY SCENARIOS COVERING MEMORY, REPEATS, AGE, COUNTRY, SAFETY, TRAPS & STEP 3
const BUILTIN_SCENARIOS = [
  {
    id: 1, name: "🧠 Memory 1: Double Greeting & Reset Wipe", category: "Memory & Repeat",
    uid: "MemorySuite", time_str: "Built-in", is_builtin: true,
    desc: "Proves bot remembers Class 1 repeat ('hi' -> 'hi') and wipes memory cleanly on 'next'.",
    steps: [
      { text:"next", expect_class:99, expect_intent:"system_reset", note:"Start clean" },
      { text:"hi", expect_class:1, expect_intent:"english_greeting", note:"1st hi -> normal greeting" },
      { text:"hi", expect_class:1, expect_intent:"english_greeting_repeat", note:"2nd hi -> MUST detect repeat memory!" },
      { text:"heyy", expect_class:1, expect_intent:"english_greeting_repeat", note:"3rd greeting -> still repeat memory!" },
      { text:"next", expect_class:99, expect_intent:"system_reset", note:"Wipe memory via next" },
      { text:"hi", expect_class:1, expect_intent:"english_greeting", note:"hi after next -> MUST be fresh greeting!" }
    ]
  },
  {
    id: 2, name: "🔁 Memory 2: Same Question & Broken Streak", category: "Memory & Repeat",
    uid: "MemorySuite", time_str: "Built-in", is_builtin: true,
    desc: "Tests asking the same question twice ('age?' -> 'age?') and breaking double-greeting streak.",
    steps: [
      { text:"next", expect_class:99, expect_intent:"system_reset" },
      { text:"hello", expect_class:1, expect_intent:"english_greeting" },
      { text:"age?", expect_class:2, expect_intent:"ask_age", note:"Ask age 1st time" },
      { text:"age?", expect_class:2, expect_intent:"ask_age", note:"Ask same question again" },
      { text:"hi", expect_class:1, expect_intent:"english_greeting", note:"Streak broken by age? -> fresh greeting" },
      { text:"hey", expect_class:1, expect_intent:"english_greeting_repeat", note:"Back-to-back greeting -> repeat!" }
    ]
  },
  {
    id: 3, name: "👋 Pure Greetings & Stretched Typos", category: "Greetings",
    uid: "Step1Suite", time_str: "Built-in", is_builtin: true,
    desc: "Checks canonicalized typos ('helo', 'hlo', 'ello', 'heeyy', 'hiyaaa') with clean resets.",
    steps: [
      { text:"next", expect_class:99 }, { text:"hello", expect_class:1, expect_intent:"english_greeting" },
      { text:"next", expect_class:99 }, { text:"hellooo", expect_class:1, expect_intent:"english_greeting" },
      { text:"next", expect_class:99 }, { text:"helo", expect_class:1, expect_intent:"english_greeting" },
      { text:"next", expect_class:99 }, { text:"hlo", expect_class:1, expect_intent:"english_greeting" },
      { text:"next", expect_class:99 }, { text:"ello", expect_class:1, expect_intent:"english_greeting" },
      { text:"next", expect_class:99 }, { text:"heeyy", expect_class:1, expect_intent:"english_greeting" },
      { text:"next", expect_class:99 }, { text:"hiyaaa", expect_class:1, expect_intent:"english_greeting" },
      { text:"next", expect_class:99 }, { text:"yo", expect_class:1, expect_intent:"english_greeting" },
      { text:"next", expect_class:99 }, { text:"sup", expect_class:1, expect_intent:"english_greeting" }
    ]
  },
  {
    id: 4, name: "🎂 Age Asking, Boundaries (18–80) & Invalid Numbers", category: "Age & Country",
    uid: "Step2Age", time_str: "Built-in", is_builtin: true,
    desc: "Tests ask_age, valid adult ages 18-80, and rejects 0, 81, 200, hi123.",
    steps: [
      { text:"next", expect_class:99 },
      { text:"how old are you", expect_class:2 },
      { text:"ur age?", expect_class:2 },
      { text:"18", expect_class:3 },
      { text:"im 24", expect_class:3 },
      { text:"l'm 30", expect_class:3 },
      { text:"hi im 22", expect_class:3 },
      { text:"80", expect_class:3 },
      { text:"0", expect_class:0, note:"Age 0 -> Reject" },
      { text:"81", expect_class:0, note:"Age > 80 -> Reject" },
      { text:"200", expect_class:0, note:"3 digits -> Reject" },
      { text:"hi123", expect_class:0, note:"Glued digits -> Reject" }
    ]
  },
  {
    id: 5, name: "🌍 Country, Cities, Slang & Typos", category: "Age & Country",
    uid: "Step2Country", time_str: "Built-in", is_builtin: true,
    desc: "Verifies ask_country and country typos ('tuinsi', 'tunsia', 'handi', 'ind', 'im from tuinsi').",
    steps: [
      { text:"next", expect_class:99 },
      { text:"from?", expect_class:4 },
      { text:"where are you from", expect_class:4 },
      { text:"where r u from", expect_class:4 },
      { text:"tunisia", expect_class:5 },
      { text:"tuinsi", expect_class:5 },
      { text:"tunsia", expect_class:5 },
      { text:"handi", expect_class:5 },
      { text:"ind", expect_class:5 },
      { text:"im from tuinsi", expect_class:5 },
      { text:"usa", expect_class:5 }
    ]
  },
  {
    id: 6, name: "⚡ Combined Single-Line Age + Country", category: "Combined",
    uid: "Step2Combo", time_str: "Built-in", is_builtin: true,
    desc: "Tests Class 6 single-line age+country extraction vs invalid age+country ('200 hello uk').",
    steps: [
      { text:"next", expect_class:99 },
      { text:"hi l'm 30 from tun", expect_class:6 },
      { text:"24 m tuinsi", expect_class:6 },
      { text:"hi im 22 from handi", expect_class:6 },
      { text:"30 from usa", expect_class:6 },
      { text:"200 hello  uk ", expect_class:0, note:"Invalid age 200 -> MUST reject!" }
    ]
  },
  {
    id: 7, name: "🔞 Underage 18+ Safety Shield (1–17)", category: "Underage Safety",
    uid: "SafetySuite", time_str: "Built-in", is_builtin: true,
    desc: "Ensures every age from 1 to 17 immediately returns Class 7 (underage_reject), even with country.",
    steps: [
      { text:"next", expect_class:99 },
      { text:"15", expect_class:7 },
      { text:"im 16", expect_class:7 },
      { text:"17", expect_class:7 },
      { text:"hi l'm 14 from tun", expect_class:7 },
      { text:"1", expect_class:7 }
    ]
  },
  {
    id: 8, name: "🛡️ Zero-Faux Hard Negatives & Traps", category: "Zero-Faux Traps",
    uid: "TrapSuite", time_str: "Built-in", is_builtin: true,
    desc: "Checks gibberish, non-English, 'what!', 'why?', 'who are you', 'lol', 'hi m' all reject as Class 0.",
    steps: [
      { text:"next", expect_class:99 },
      { text:"hyw", expect_class:0 },
      { text:"iphey", expect_class:0 },
      { text:"you in bed?", expect_class:0 },
      { text:"bonjour", expect_class:0 },
      { text:"salam", expect_class:0 },
      { text:"lol", expect_class:0 },
      { text:"what", expect_class:0 },
      { text:"what!", expect_class:0 },
      { text:"why?", expect_class:0 },
      { text:"why!", expect_class:0 },
      { text:"who are you", expect_class:0 },
      { text:"hi m", expect_class:0 }
    ]
  },
  {
    id: 9, name: "💬 Full Multi-Turn Dating Chat + Disconnect", category: "System Reset",
    uid: "FullFlow", time_str: "Built-in", is_builtin: true,
    desc: "End-to-end realistic dating conversation followed by disconnect reset and a brand-new chat.",
    steps: [
      { text:"next", expect_class:99 },
      { text:"heyy", expect_class:1, expect_intent:"english_greeting" },
      { text:"age?", expect_class:2 },
      { text:"im 25", expect_class:3 },
      { text:"where r u from", expect_class:4 },
      { text:"im from tunis", expect_class:5 },
      { text:"disconnect", expect_class:99, expect_intent:"system_reset" },
      { text:"hi", expect_class:1, expect_intent:"english_greeting", note:"Fresh chat after disconnect" }
    ]
  },
  {
    id: 10, name: "🚀 Step 3 Check: ask_how_are_you (Class 8)", category: "Step 3 HRU",
    uid: "Step3Ready", time_str: "Built-in", is_builtin: true,
    desc: "Checks Step 3 isolation: 'hi' (Class 1) -> 'how r u?' (Class 8), while 'what!'/'who are you' stay Class 0.",
    steps: [
      { text:"next", expect_class:99 },
      { text:"hi", expect_class:1, expect_intent:"english_greeting" },
      { text:"how r u?", expect_class:8, expect_intent:"ask_how_are_you", note:"Requires Step 3 trained" },
      { text:"how are you doing", expect_class:8, expect_intent:"ask_how_are_you" },
      { text:"hru", expect_class:8, expect_intent:"ask_how_are_you" },
      { text:"what!", expect_class:0, note:"Must NOT trigger HRU" },
      { text:"who are you", expect_class:0, note:"Must NOT trigger HRU" }
    ]
  }
];

let allConvs = BUILTIN_SCENARIOS.map(c => ({ ...c, results: null, batchRuns: null }));
let filteredConvs = allConvs.slice();
let selectedConvId = allConvs[0].id;
let selectedTurnIdx = null;
let serverOnline = false;

// Live Phone State
let liveHistory = [];
let liveLatencies = [];
let liveMem = { lastIntent: "None", lastClass: -1, msgCount: 0, age: "None", country: "None" };

// Possibilities Modal State
let modalPriorTexts = [];
let modalTurnNum = 1;
window._possReg = {};
let _possId = 0;

function escapeHtml(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function classBadge(cid) {
  const name = CLASS_NAMES[cid] || `c${cid}`;
  const cls = [0,1,2,3,4,5,6,7,8,99].includes(cid) ? `c${cid}` : 'c0';
  return `<span class="badge ${cls}">C${cid}: ${name}</span>`;
}

function extractLocalMemEffect(text, resp, mem) {
  const m = { ...mem };
  const cid = resp.class_id;
  if (cid === 99) {
    return { lastIntent: "system_reset", lastClass: 0, msgCount: 0, age: "None", country: "None" };
  }
  m.lastIntent = resp.intent || "unknown";
  m.lastClass = cid;
  m.msgCount = (m.msgCount || 0) + 1;
  const numMatch = text.match(/\b(1[89]|[2-7][0-9]|80)\b/);
  if ((cid === 3 || cid === 6) && numMatch) m.age = numMatch[1];
  if (cid === 5 || cid === 6) m.country = text.replace(/\b\d+\b/g, '').trim().slice(0, 18) || "Saved";
  return m;
}

function updateLiveMemUI(lastMs = null) {
  document.getElementById('memLastIntent').innerText = liveMem.lastIntent;
  document.getElementById('memMsgCnt').innerText = liveMem.msgCount;
  document.getElementById('memAge').innerText = liveMem.age;
  document.getElementById('memCountry').innerText = liveMem.country;
  document.getElementById('memStatus').innerText = liveMem.msgCount === 0 ? "Fresh / Wiped" : "Active Memory";
  document.getElementById('swTurnCnt').innerText = `Turn ${liveMem.msgCount}`;
  document.getElementById('swRepeatArmed').innerText = liveMem.lastClass === 1 ? "ARMED" : "OFF";
  document.getElementById('swRepeatArmed').style.color = liveMem.lastClass === 1 ? "#f59e0b" : "#64748b";
  document.getElementById('swProfileVal').innerText =
    (liveMem.age === "None" && liveMem.country === "None") ? "Empty" : `Age:${liveMem.age} | ${liveMem.country}`;

  if (lastMs !== null) {
    document.getElementById('swLastMs').innerText = `${lastMs.toFixed(2)}ms`;
    document.getElementById('swLastClass').innerText = `Class ${liveMem.lastClass} (${liveMem.lastIntent})`;
    const avg = liveLatencies.reduce((a,b)=>a+b,0) / (liveLatencies.length || 1);
    document.getElementById('swAvgMs').innerText = `${avg.toFixed(2)}ms`;
  }
}

async function pingServer() {
  const badge = document.getElementById('apiStatusBadge');
  try {
    const r = await fetch('/api/ping');
    const d = await r.json();
    serverOnline = !!d.online;
  } catch (e) { serverOnline = false; }

  if (serverOnline) {
    badge.innerText = "🟢 AI :8080 Online";
    badge.style.background = "#065f46"; badge.style.color = "#6ee7b7";
    document.getElementById('phoneStatus').innerText = "🟢 online — 127.0.0.1:8080 ready";
  } else {
    badge.innerText = "🔴 Start production_server.py";
    badge.style.background = "#7f1d1d"; badge.style.color = "#fecaca";
    document.getElementById('phoneStatus').innerText = "🔴 production_server.py offline on :8080";
  }
}

async function sendToAI(text, srcIp = "127.0.0.1") {
  const r = await fetch('/api/send', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ text, src_ip: srcIp })
  });
  return await r.json();
}

function registerPossBtn(turnNum, userText, priorTexts) {
  const id = ++_possId;
  window._possReg[id] = { turnNum, userText, priorTexts: priorTexts.slice() };
  return `<button type="button" class="btn-poss" onclick="triggerPoss(${id})">🎲 Possibilities</button>`;
}

function triggerPoss(id) {
  const it = window._possReg[id];
  if (it) openPossModal(it.turnNum, it.userText, it.priorTexts);
}

async function openPossModal(turnNum, userText, priorTexts) {
  modalTurnNum = turnNum;
  modalPriorTexts = priorTexts || [];
  document.getElementById('possModal').classList.add('open');
  document.getElementById('modalTitle').innerText = `🎲 All Bot Reply Possibilities at Turn #${turnNum}`;
  document.getElementById('modalUserText').value = userText || '';
  document.getElementById('modalPresetSelect').value = '';
  document.getElementById('modalRepeatSend').value = '1';

  const strip = document.getElementById('modalHistoryStrip');
  if (!modalPriorTexts.length) {
    strip.innerHTML = `<span style="color:#64748b;">📜 Prior History: (Fresh Start after 'next')</span>`;
  } else {
    strip.innerHTML = `<strong style="color:#38bdf8;">📜 Prior Turns:</strong> ` +
      modalPriorTexts.map(t => `<span class="hist-chip-u">👤 ${escapeHtml(t)}</span>`).join(' → ');
  }
  await fetchTurnPossibilities();
}

function closePossModal() {
  document.getElementById('possModal').classList.remove('open');
}

async function fetchTurnPossibilities() {
  const grid = document.getElementById('modalOutcomesGrid');
  const memBar = document.getElementById('modalMemBar');
  const txt = document.getElementById('modalUserText').value;
  const tries = parseInt(document.getElementById('modalTriesInput').value || '64', 10);
  const repeatSend = parseInt(document.getElementById('modalRepeatSend').value || '1', 10);
  const ipMode = document.getElementById('modalIpMode').value;

  grid.innerHTML = `<div style="padding:20px;color:var(--accent);">⏳ Running ${tries} live simulations against 127.0.0.1:8080...</div>`;
  try {
    const res = await fetch('/api/turn_possibilities', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ prior_texts: modalPriorTexts, text: txt, tries, repeat_send: repeatSend, ip_mode: ipMode })
    });
    const d = await res.json();
    if (!d.ok) {
      grid.innerHTML = `<div style="color:#f87171;">Error: ${escapeHtml(d.error)}</div>`;
      return;
    }

    const uniqueClasses = [...new Set(d.outcomes.map(o => o.class_id))];
    const consistency = uniqueClasses.length === 1 ? "100% Zero-Faux Consistent ✅" : `⚠️ Split Across ${uniqueClasses.length} Classes!`;
    memBar.innerHTML = `
      <span>🎯 <strong>Target:</strong> <code style="color:#fff;">"${escapeHtml(txt)}"</code> (Sent ${repeatSend}x)</span>
      <span>🎲 <strong>Unique Replies Found:</strong> <code style="color:#38bdf8;">${d.outcomes.length}</code></span>
      <span>⚡ <strong>Avg Speed:</strong> <code style="color:#10b981;">${d.avg_ms.toFixed(2)}ms</code></span>
      <span>🛡️ <strong>Class Stability:</strong> <code style="color:${uniqueClasses.length===1?'#10b981':'#f59e0b'};">${consistency}</code></span>
    `;

    grid.innerHTML = d.outcomes.map(o => `
      <div class="tree-node">
        <div class="tree-node-top">
          <span>${classBadge(o.class_id)} <code>${escapeHtml(o.intent)}</code></span>
          <span style="color:#38bdf8;font-size:0.95rem;">${o.pct}% (${o.count}/${d.tries})</span>
        </div>
        <div class="pbar-bg"><div class="pbar-fill" style="width:${o.pct}%;background:#38bdf8;"></div></div>
        <div class="tg-chat">
          <div class="tg-bubble ${o.intent==='english_greeting_repeat'?'bot-rep':(o.class_id===99?'bot-rst':'bot')}">
            <div>${escapeHtml(o.reply)}</div>
            <div class="tg-meta"><span>avg ${o.avg_ms.toFixed(2)}ms</span><span>kbd: ${escapeHtml(JSON.stringify(o.kbd||[]))}</span></div>
          </div>
        </div>
      </div>
    `).join('');
  } catch (e) {
    grid.innerHTML = `<div style="color:#f87171;">Connection failed. Ensure production_server.py is running!</div>`;
  }
}

// --- LIVE PHONE ---
function clearLivePhoneOnly() {
  document.getElementById('phoneChatBody').innerHTML = '';
  document.getElementById('chronoLogTable').innerHTML = '';
}

async function resetLivePhone() {
  liveHistory = [];
  liveLatencies = [];
  clearLivePhoneOnly();
  document.getElementById('phoneChatBody').innerHTML =
    `<div style="text-align:center;font-size:0.75rem;color:#8696a0;margin:4px 0;">🔄 Sending 'next' to wipe bot memory...</div>`;
  await sendLiveText("next");
}

async function sendLivePhoneMsg() {
  const inp = document.getElementById('liveChatInput');
  const txt = inp.value.trim();
  if (!txt) return;
  inp.value = '';
  await sendLiveText(txt);
}

async function sendLiveText(txt) {
  const chatBody = document.getElementById('phoneChatBody');
  const priorTexts = liveHistory.slice();
  const turnNum = liveHistory.length + 1;

  chatBody.insertAdjacentHTML('beforeend', `<div class="tg-bubble user"><div>${escapeHtml(txt)}</div><div class="tg-meta"><span>Turn #${turnNum}</span><span>✓✓</span></div></div>`);
  chatBody.scrollTop = chatBody.scrollHeight;

  const data = await sendToAI(txt, "127.0.0.1");
  if (!data.ok) {
    chatBody.insertAdjacentHTML('beforeend', `<div class="tg-bubble bot-err">⚠️ Server offline: ${escapeHtml(data.error)}</div>`);
    return;
  }

  const r = data.response;
  const ms = data.latency_ms;
  liveLatencies.push(ms);
  if (r.class_id === 99) liveHistory = []; else liveHistory.push(txt);
  liveMem = extractLocalMemEffect(txt, r, liveMem);
  updateLiveMemUI(ms);

  const possBtn = registerPossBtn(turnNum, txt, priorTexts);
  const bCls = r.class_id === 99 ? 'bot-rst' : (r.intent === 'english_greeting_repeat' ? 'bot-rep' : 'bot');
  const kbdHtml = (r.kbd && r.kbd.length)
    ? `<div class="kbd-row">${r.kbd.map(k => `<button type="button" class="kbd-btn" onclick="sendLiveText('${escapeHtml(k).replace(/'/g,"\\'")}')">${escapeHtml(k)}</button>`).join('')}</div>`
    : '';

  chatBody.insertAdjacentHTML('beforeend', `
    <div class="tg-bubble ${bCls}">
      <div>${escapeHtml(r.reply)}</div>
      ${kbdHtml}
      <div class="tg-meta">
        <span>${classBadge(r.class_id)} ${escapeHtml(r.intent)}</span>
        <span style="display:flex;gap:5px;align-items:center;">${possBtn} <span class="time-badge">⚡ ${ms.toFixed(2)}ms</span></span>
      </div>
    </div>
  `);
  chatBody.scrollTop = chatBody.scrollHeight;

  document.getElementById('chronoLogTable').insertAdjacentHTML('afterbegin', `
    <tr style="border-bottom:1px solid #1e293b;">
      <td style="padding:4px;">#${turnNum}</td>
      <td style="padding:4px;color:#38bdf8;">${escapeHtml(txt)}</td>
      <td style="padding:4px;">${classBadge(r.class_id)}</td>
      <td style="padding:4px;"><strong>${escapeHtml(r.intent)}</strong> — ${escapeHtml(r.reply)} ${possBtn}</td>
      <td style="padding:4px;color:#10b981;font-family:monospace;">${ms.toFixed(2)}ms</td>
    </tr>
  `);
}

async function replayScenarioOnPhone(conv) {
  document.getElementById('chronoPanel').classList.remove('hidden');
  liveHistory = [];
  liveLatencies = [];
  clearLivePhoneOnly();
  document.getElementById('phoneChatBody').insertAdjacentHTML('beforeend',
    `<div style="text-align:center;font-size:0.75rem;color:#38bdf8;margin:4px 0;">▶️ Replaying: ${escapeHtml(conv.name)}</div>`);
  for (const step of conv.steps) {
    await sendLiveText(step.text);
    await new Promise(r => setTimeout(r, 120));
  }
}

// --- SCENARIO EXECUTION & SMARTNESS SCORE ---
async function executeScenario(conv) {
  const results = [];
  let mem = { lastIntent: "None", lastClass: -1, msgCount: 0, age: "None", country: "None" };
  let prior = [];

  // Ensure clean start if scenario doesn't begin with next
  if (!conv.steps.length || conv.steps[0].text.toLowerCase() !== "next") {
    await sendToAI("next", "127.0.0.1");
  }

  for (let i = 0; i < conv.steps.length; i++) {
    const st = conv.steps[i];
    const d = await sendToAI(st.text, "127.0.0.1");
    const r = d.ok ? d.response : { class_id: -1, intent: "error", reply: d.error || "offline", kbd: [] };
    const ms = d.ok ? d.latency_ms : 0;

    let passed = null;
    if (st.expect_class !== undefined) {
      const classOk = r.class_id === st.expect_class;
      const intentOk = st.expect_intent ? (r.intent === st.expect_intent) : true;
      passed = classOk && intentOk;
    }

    const memBefore = { ...mem };
    mem = extractLocalMemEffect(st.text, r, mem);
    results.push({
      turn: i + 1, text: st.text, note: st.note || "",
      expect_class: st.expect_class, expect_intent: st.expect_intent || "",
      resp: r, ms, passed, memBefore, memAfter: { ...mem }, priorTexts: prior.slice()
    });
    if (r.class_id === 99) prior = []; else prior.push(st.text);
  }
  conv.results = results;
  return results;
}

async function runSelectedScenarioLive() {
  const conv = allConvs.find(c => c.id === selectedConvId);
  if (!conv) return;
  await executeScenario(conv);
  updateSmartnessBoard();
  renderSidebar();
  renderSelectedConv();
}

async function runFullSmartnessBenchmark() {
  await pingServer();
  if (!serverOnline) return alert("Start `python production_server.py` on port 8080 first!");
  for (const conv of allConvs.filter(c => c.is_builtin)) {
    await executeScenario(conv);
  }
  updateSmartnessBoard();
  renderSidebar();
  renderSelectedConv();
}

function updateSmartnessBoard() {
  const catStats = {};
  let totPass = 0, totChecks = 0, totMs = 0, totCount = 0;

  for (const conv of allConvs) {
    if (!conv.results) continue;
    const cat = conv.category || "Other";
    if (!catStats[cat]) catStats[cat] = { pass: 0, total: 0 };
    for (const r of conv.results) {
      totMs += r.ms; totCount++;
      if (r.passed !== null) {
        totChecks++; catStats[cat].total++;
        if (r.passed) { totPass++; catStats[cat].pass++; }
      }
    }
  }

  const pct = totChecks ? ((totPass / totChecks) * 100) : 0;
  const avgMs = totCount ? (totMs / totCount) : 0;
  document.getElementById('smartOverallPct').innerText = totChecks ? `${pct.toFixed(1)}%` : "--%";
  document.getElementById('smartPassedTxt').innerText = `${totPass} / ${totChecks}`;
  document.getElementById('smartAvgMs').innerText = `${avgMs.toFixed(2)} ms/req`;

  const bar = document.getElementById('smartOverallBar');
  const grade = document.getElementById('smartGrade');
  bar.style.width = `${pct}%`;

  if (!totChecks) {
    grade.innerText = 'Click "Run Full Smartness Audit" to score your AI';
  } else if (pct >= 95) {
    bar.style.background = '#10b981'; document.getElementById('smartOverallPct').style.color = '#10b981';
    grade.innerText = '🧠 ULTRA SMART (Production Ready)';
    grade.style.background = '#064e3b'; grade.style.color = '#6ee7b7'; grade.style.borderColor = '#10b981';
  } else if (pct >= 85) {
    bar.style.background = '#38bdf8'; document.getElementById('smartOverallPct').style.color = '#38bdf8';
    grade.innerText = '🟢 SMART (Minor Edge Gaps — Check Red Cards)';
    grade.style.background = '#0c4a6e'; grade.style.color = '#7dd3fc'; grade.style.borderColor = '#38bdf8';
  } else if (pct >= 70) {
    bar.style.background = '#f59e0b'; document.getElementById('smartOverallPct').style.color = '#f59e0b';
    grade.innerText = '🟡 LEARNING (Needs Step 3 / Dataset Tuning)';
    grade.style.background = '#78350f'; grade.style.color = '#fde68a'; grade.style.borderColor = '#f59e0b';
  } else {
    bar.style.background = '#ef4444'; document.getElementById('smartOverallPct').style.color = '#ef4444';
    grade.innerText = '🔴 NEEDS ATTENTION (High Faux Rate)';
    grade.style.background = '#7f1d1d'; grade.style.color = '#fecaca'; grade.style.borderColor = '#ef4444';
  }

  document.getElementById('smartCatsGrid').innerHTML = Object.entries(catStats).map(([cat, st]) => {
    const cp = st.total ? (st.pass / st.total) * 100 : 0;
    const col = cp === 100 ? '#10b981' : (cp >= 75 ? '#f59e0b' : '#ef4444');
    return `
      <div class="cat-card">
        <div class="cat-lbl">${escapeHtml(cat)}</div>
        <div class="cat-val" style="color:${col};"><span>${cp.toFixed(0)}%</span><span style="font-size:0.75rem;color:#94a3b8;">${st.pass}/${st.total}</span></div>
        <div class="pbar-bg"><div class="pbar-fill" style="width:${cp}%;background:${col};"></div></div>
      </div>`;
  }).join('');
}

// --- BATCH MULTI-MESSAGE SIMULATOR ---
function renderStepInputs(arr) {
  document.getElementById('stepsContainer').innerHTML = arr.map((txt, i) => `
    <div class="msg-step-chip">
      <span style="font-size:0.72rem;color:var(--text-muted);">#${i+1}</span>
      <input type="text" class="step-txt" value="${escapeHtml(txt)}" placeholder="Msg ${i+1}...">
      <button type="button" class="del-step" onclick="removeStepInput(${i})">✕</button>
    </div>
  `).join('');
}
function getStepsFromDOM() {
  return [...document.querySelectorAll('.step-txt')].map(el => el.value.trim()).filter(Boolean);
}
function addStepInput(t='') { const s = getStepsFromDOM(); s.push(t); renderStepInputs(s); }
function removeStepInput(idx) { const s = getStepsFromDOM(); s.splice(idx,1); renderStepInputs(s.length?s:['hi']); }
function loadQuickArray() {
  const parts = document.getElementById('quickArrayInput').value.split(',').map(x=>x.trim()).filter(Boolean);
  if (parts.length) renderStepInputs(parts);
}

async function runCustomBatchSimulation() {
  const msgs = getStepsFromDOM();
  if (!msgs.length) return alert("Add at least 1 message!");
  const repeats = parseInt(document.getElementById('simRepeats').value || '32', 10);

  const res = await fetch('/api/run_batch', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ messages: msgs, repeats })
  });
  const d = await res.json();
  if (!d.ok) return alert(d.error || "Batch failed");

  const newConv = {
    id: Date.now() % 100000,
    name: `🧪 Batch (${repeats}x): ${msgs.slice(0,3).join(' → ')}`,
    category: "Batch Sim", uid: `${repeats} Runs`, time_str: `${d.avg_ms.toFixed(2)}ms`,
    is_builtin: false, desc: `Simulated ${repeats} independent runs across varying users to capture all rotating reply variations.`,
    steps: msgs.map(m => ({ text: m })),
    results: null,
    batchRuns: d.per_turn
  };
  allConvs.unshift(newConv);
  filteredConvs = allConvs.slice();
  selectedConvId = newConv.id;
  selectedTurnIdx = null;
  renderSidebar();
  renderSelectedConv();
}

// --- FILE LOADING ---
async function loadLocalJsonl() {
  const r = await fetch('/api/read_local');
  const d = await r.json();
  if (!d.ok) return alert(d.error || "bot_chat_history.jsonl not found in folder");
  importConversations(d.convs);
}

function handleFileUpload(e) {
  const f = e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = async (ev) => {
    const r = await fetch('/api/parse_upload', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ filename: f.name, content: ev.target.result })
    });
    const d = await r.json();
    if (d.ok) importConversations(d.convs);
  };
  reader.readAsText(f);
}

function importConversations(convs) {
  if (!convs || !convs.length) return alert("No valid user messages found in file.");
  const enriched = convs.map(c => ({ ...c, results: null, batchRuns: null }));
  allConvs = BUILTIN_SCENARIOS.map(b => allConvs.find(x => x.id === b.id) || { ...b, results: null, batchRuns: null }).concat(enriched);
  filteredConvs = allConvs.slice();
  selectedConvId = enriched[0].id;
  renderSidebar();
  renderSelectedConv();
}

// --- RENDERING ---
function applyFilters() {
  const q = document.getElementById('searchBox').value.trim().toLowerCase();
  filteredConvs = allConvs.filter(c => {
    if (!q) return true;
    const seq = c.steps.map(s=>s.text).join(' ').toLowerCase();
    return c.name.toLowerCase().includes(q) || (c.category||'').toLowerCase().includes(q) || seq.includes(q);
  });
  if (filteredConvs.length && !filteredConvs.some(c => c.id === selectedConvId)) selectedConvId = filteredConvs[0].id;
  renderSidebar();
  renderSelectedConv();
}

function selectConv(id) {
  selectedConvId = id;
  selectedTurnIdx = null;
  renderSidebar();
  renderSelectedConv();
}

function renderSidebar() {
  document.getElementById('sidebarCount').innerText = `${filteredConvs.length} scenarios / logs`;
  document.getElementById('sidebarList').innerHTML = filteredConvs.map(c => {
    let scoreBadge = `<span class="badge-cnt">${c.steps.length} msgs</span>`;
    if (c.results) {
      const checked = c.results.filter(r => r.passed !== null);
      if (checked.length) {
        const p = checked.filter(r => r.passed).length;
        const pct = ((p / checked.length) * 100).toFixed(0);
        scoreBadge = p === checked.length
          ? `<span class="badge-pass">${pct}% (${p}/${checked.length})</span>`
          : `<span class="badge-fail">${pct}% (${p}/${checked.length})</span>`;
      } else {
        scoreBadge = `<span class="badge-pass">Ran ✓</span>`;
      }
    } else if (c.batchRuns) {
      scoreBadge = `<span class="badge-pass">${c.uid}</span>`;
    }
    const seq = c.steps.map(s => s.text).join(' → ');
    return `
      <div class="side-item ${c.id === selectedConvId ? 'active' : ''}" onclick="selectConv(${c.id})">
        <div class="side-top"><span>${escapeHtml(c.category)}</span>${scoreBadge}</div>
        <div class="side-title">${escapeHtml(c.name)}</div>
        <div class="side-seq">${escapeHtml(seq)}</div>
      </div>`;
  }).join('');
}

function renderSelectedConv() {
  window._possReg = {};
  _possId = 0;
  const mainEl = document.getElementById('mainContent');
  const conv = filteredConvs.find(c => c.id === selectedConvId);
  if (!conv) { mainEl.innerHTML = `<div style="padding:20px;color:var(--text-muted);">No scenario selected.</div>`; return; }

  const viewMode = document.getElementById('viewMode').value;
  const outFilter = document.getElementById('outcomeFilter').value;

  const pillsHtml = conv.steps.map((s, idx) => {
    const tNum = idx + 1;
    const resObj = conv.results ? conv.results[idx] : null;
    const stCls = resObj ? (resObj.passed === true ? 'pass' : (resObj.passed === false ? 'fail' : '')) : '';
    const actCls = (viewMode === 'last_turn' && selectedTurnIdx === tNum) ? 'active' : '';
    return `<span class="turn-pill ${stCls} ${actCls}" onclick="selectedTurnIdx=${tNum};document.getElementById('viewMode').value='last_turn';renderSelectedConv();">Msg ${tNum}: "${escapeHtml(s.text)}"</span>`;
  }).join(' → ');

  let bodyHtml = '';

  // Batch Multi-Run View
  if (conv.batchRuns) {
    bodyHtml = `<div class="turns-grid">` + conv.batchRuns.map((tObj, idx) => {
      const prior = conv.steps.slice(0, idx).map(x => x.text);
      const possBtn = registerPossBtn(idx + 1, tObj.text, prior);
      return `
        <div class="tree-node">
          <div class="tree-node-top">
            <span>Turn #${idx+1}: 👤 "${escapeHtml(tObj.text)}"</span>
            <span>${possBtn}</span>
          </div>
          ${tObj.outcomes.map(o => `
            <div style="border-top:1px solid #222e35;padding-top:6px;">
              <div style="display:flex;justify-content:space-between;font-size:0.77rem;">
                <span>${classBadge(o.class_id)} <code>${escapeHtml(o.intent)}</code></span>
                <strong style="color:#38bdf8;">${o.pct}% (${o.count}x)</strong>
              </div>
              <div class="pbar-bg"><div class="pbar-fill" style="width:${o.pct}%;background:#38bdf8;"></div></div>
              <div style="font-size:0.8rem;color:#cbd5e1;margin-top:4px;">🤖 "${escapeHtml(o.reply)}"</div>
            </div>
          `).join('')}
        </div>`;
    }).join('') + `</div>`;
  }
  // Not run yet
  else if (!conv.results) {
    bodyHtml = `
      <div class="tree-node" style="margin-top:12px;text-align:center;padding:24px;">
        <div style="font-size:0.95rem;margin-bottom:10px;">This scenario has <strong>${conv.steps.length}</strong> turns ready to test against <code>127.0.0.1:8080</code>.</div>
        <div style="display:flex;gap:10px;justify-content:center;">
          <button type="button" class="btn-green" onclick="runSelectedScenarioLive()">▶ Run This Scenario Now</button>
          <button type="button" class="btn-amber" onclick="replayScenarioOnPhone(allConvs.find(c=>c.id===selectedConvId))">📱 Play Live on Phone</button>
        </div>
      </div>`;
  }
  // Full Transcript Mode
  else if (viewMode === 'full_threads') {
    bodyHtml = `<div class="tree-node" style="margin-top:12px;"><div class="tg-chat">` + conv.results.map(r => {
      const possBtn = registerPossBtn(r.turn, r.text, r.priorTexts);
      const bCls = r.resp.class_id === 99 ? 'bot-rst' : (r.resp.intent === 'english_greeting_repeat' ? 'bot-rep' : 'bot');
      const status = r.passed === true ? `<span class="badge-pass">PASS</span>` : (r.passed === false ? `<span class="badge-fail">FAIL (Exp C${r.expect_class})</span>` : '');
      return `
        <div class="tg-bubble user"><div>${escapeHtml(r.text)}</div><div class="tg-meta"><span>Turn #${r.turn}</span><span>✓✓</span></div></div>
        <div class="tg-bubble ${bCls}">
          <div>${escapeHtml(r.resp.reply)}</div>
          <div class="tg-meta"><span>${classBadge(r.resp.class_id)} ${escapeHtml(r.resp.intent)} ${status}</span><span>${possBtn} <span class="time-badge">⚡ ${r.ms.toFixed(2)}ms</span></span></div>
        </div>`;
    }).join('') + `</div></div>`;
  }
  // Per-Turn Cards / Focus Mode
  else {
    let items = conv.results;
    if (viewMode === 'last_turn') {
      const tTarget = selectedTurnIdx || conv.results.length;
      items = items.filter(r => r.turn === tTarget);
    }
    if (outFilter === 'pass') items = items.filter(r => r.passed === true);
    if (outFilter === 'fail') items = items.filter(r => r.passed === false);

    bodyHtml = `<div class="turns-grid">` + items.map(r => {
      const possBtn = registerPossBtn(r.turn, r.text, r.priorTexts);
      const bCls = r.resp.class_id === 99 ? 'bot-rst' : (r.resp.intent === 'english_greeting_repeat' ? 'bot-rep' : 'bot');
      const stBadge = r.passed === true
        ? `<span class="badge-pass">✅ PASS</span>`
        : (r.passed === false ? `<span class="badge-fail">❌ FAIL (Exp C${r.expect_class}${r.expect_intent?' '+r.expect_intent:''})</span>` : `<span class="badge-cnt">Unscored</span>`);

      return `
        <div class="tree-node ${r.passed === false ? 'fail-card' : ''}">
          <div class="tree-node-top">
            <span>Turn #${r.turn} • ⚡ ${r.ms.toFixed(2)}ms</span>
            <span style="display:flex;gap:6px;align-items:center;">${stBadge} ${possBtn}</span>
          </div>
          ${r.note ? `<div style="font-size:0.73rem;color:#94a3b8;">💡 ${escapeHtml(r.note)}</div>` : ''}
          <div class="tg-chat">
            <div class="tg-bubble user"><div>${escapeHtml(r.text)}</div></div>
            <div class="tg-bubble ${bCls}">
              <div>${escapeHtml(r.resp.reply)}</div>
              <div class="tg-meta"><span>${classBadge(r.resp.class_id)} <code>${escapeHtml(r.resp.intent)}</code></span><span>kbd: ${escapeHtml(JSON.stringify(r.resp.kbd||[]))}</span></div>
            </div>
          </div>
          <div class="mem-mini">
            <span>🧠 After: <strong>${escapeHtml(r.memAfter.lastIntent)}</strong></span>
            <span>#Msgs: <strong>${r.memAfter.msgCount}</strong></span>
            <span>Age: <strong>${escapeHtml(r.memAfter.age)}</strong></span>
            <span>Ctry: <strong>${escapeHtml(r.memAfter.country)}</strong></span>
          </div>
        </div>`;
    }).join('') + `</div>`;
  }

  mainEl.innerHTML = `
    <div class="conv-header-card">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
        <div>
          <strong style="font-size:1rem;color:var(--accent);">${escapeHtml(conv.name)}</strong>
          <div style="font-size:0.77rem;color:var(--text-muted);margin-top:2px;">${escapeHtml(conv.desc || '')}</div>
        </div>
        <div style="display:flex;gap:7px;">
          <button type="button" class="btn-green" onclick="runSelectedScenarioLive()">▶ Run Live</button>
          <button type="button" class="btn-amber" onclick="replayScenarioOnPhone(allConvs.find(c=>c.id===selectedConvId))">📱 Replay on Phone</button>
        </div>
      </div>
      <div class="turn-pills">${pillsHtml}</div>
    </div>
    ${bodyHtml}
  `;
}

function toggleChronoPanel() {
  document.getElementById('chronoPanel').classList.toggle('hidden');
}

// Init
renderStepInputs(["hi", "hi", "age?", "im 24", "im from tuinsi", "next", "hi"]);
renderSidebar();
renderSelectedConv();
pingServer().then(() => { if (serverOnline) runFullSmartnessBenchmark(); });
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

    def do_GET(self):
        if self.path == "/":
            raw = HTML_CONTENT.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/api/ping":
            try:
                s = socket.create_connection((AI_HOST, AI_PORT), timeout=0.8)
                s.close()
                self._send_json({"online": True})
            except Exception:
                self._send_json({"online": False})
        elif self.path == "/api/read_local":
            if not os.path.exists("bot_chat_history.jsonl"):
                self._send_json({"ok": False, "error": "bot_chat_history.jsonl not found in current folder."})
                return
            try:
                with open("bot_chat_history.jsonl", "r", encoding="utf-8", errors="ignore") as f:
                    convs = parse_jsonl_conversations(f, max_convs=300)
                self._send_json({"ok": True, "convs": convs})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", errors="ignore")
        try:
            data = json.loads(body) if body else {}
        except Exception:
            self._send_json({"ok": False, "error": "Invalid JSON"}, 400)
            return

        if self.path == "/api/send":
            try:
                ms, resp = query_ai_server(data.get("text", ""), data.get("src_ip", "127.0.0.1"))
                self._send_json({"ok": True, "latency_ms": ms, "response": resp})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})

        elif self.path == "/api/turn_possibilities":
            prior = data.get("prior_texts", [])
            target = data.get("text", "")
            tries = max(1, min(int(data.get("tries", 64)), 500))
            repeat_send = max(1, min(int(data.get("repeat_send", 1)), 5))
            vary_ip = data.get("ip_mode", "vary") == "vary"

            groups = {}
            total_ms = 0.0
            try:
                for i in range(tries):
                    src_ip = f"127.0.0.{2 + (i % 240)}" if vary_ip else "127.0.0.1"
                    query_ai_server("next", src_ip)
                    for pt in prior:
                        query_ai_server(pt, src_ip)
                    last_ms, last_resp = 0.0, {}
                    for _ in range(repeat_send):
                        last_ms, last_resp = query_ai_server(target, src_ip)
                    total_ms += last_ms
                    key = (
                        last_resp.get("class_id", -1),
                        last_resp.get("intent", ""),
                        last_resp.get("reply", ""),
                    )
                    if key not in groups:
                        groups[key] = {
                            "class_id": key[0], "intent": key[1], "reply": key[2],
                            "kbd": last_resp.get("kbd", []), "count": 0, "sum_ms": 0.0,
                        }
                    groups[key]["count"] += 1
                    groups[key]["sum_ms"] += last_ms

                outcomes = []
                for g in sorted(groups.values(), key=lambda x: x["count"], reverse=True):
                    outcomes.append({
                        "class_id": g["class_id"],
                        "intent": g["intent"],
                        "reply": g["reply"],
                        "kbd": g["kbd"],
                        "count": g["count"],
                        "pct": round((g["count"] / tries) * 100.0, 1),
                        "avg_ms": g["sum_ms"] / g["count"],
                    })
                self._send_json({"ok": True, "tries": tries, "avg_ms": total_ms / tries, "outcomes": outcomes})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})

        elif self.path == "/api/run_batch":
            msgs = data.get("messages", [])
            repeats = max(1, min(int(data.get("repeats", 32)), 200))
            per_turn_groups = [{} for _ in msgs]
            total_ms, total_calls = 0.0, 0
            try:
                for i in range(repeats):
                    src_ip = f"127.0.0.{2 + (i % 240)}"
                    query_ai_server("next", src_ip)
                    for t_idx, txt in enumerate(msgs):
                        ms, resp = query_ai_server(txt, src_ip)
                        total_ms += ms
                        total_calls += 1
                        key = (resp.get("class_id", -1), resp.get("intent", ""), resp.get("reply", ""))
                        grp = per_turn_groups[t_idx]
                        if key not in grp:
                            grp[key] = {"class_id": key[0], "intent": key[1], "reply": key[2], "count": 0}
                        grp[key]["count"] += 1

                per_turn_out = []
                for t_idx, txt in enumerate(msgs):
                    items = sorted(per_turn_groups[t_idx].values(), key=lambda x: x["count"], reverse=True)
                    per_turn_out.append({
                        "text": txt,
                        "outcomes": [
                            {**it, "pct": round((it["count"] / repeats) * 100.0, 1)}
                            for it in items
                        ],
                    })
                self._send_json({
                    "ok": True,
                    "avg_ms": total_ms / max(total_calls, 1),
                    "per_turn": per_turn_out,
                })
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)})

        elif self.path == "/api/parse_upload":
            content = data.get("content", "")
            convs = parse_jsonl_conversations(content.splitlines(), max_convs=300)
            self._send_json({"ok": True, "convs": convs})

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    with ThreadingHTTPServer(("", PORT), StudioHandler) as httpd:
        print(f"[ULTRA AI ANALYZER] Running at http://127.0.0.1:{PORT}")
        print(f" -> Connects to production_server.py on http://{AI_HOST}:{AI_PORT}")
        httpd.serve_forever()