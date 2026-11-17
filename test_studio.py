# FILE: test_studio.py
# PURPOSE: High-Performance AI Stress-Test Studio (Handles Multiple Conversations)
# USAGE: Run `python test_studio.py` and open http://127.0.0.1:8090

import http.server
import socketserver
import socket
import json
import time
import os

PORT = 8090

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Dating Bot | Ultra Stress-Test Studio</title>
    <style>
        :root {
            --bg-dark: #0e1621; --bg-panel: #17212b; --bg-card: #202b36;
            --text-main: #f5f5f5; --text-muted: #7f91a4; --border: #38404a;
            --accent: #2b5278; --accent-hover: #3e6d99; --danger: #e53935; --success: #43a047;
            --bubble-user: #2b5278; --bubble-bot: #182533;
            --c0: #e53935; --c1: #43a047; --c2: #fdd835; --c3: #fb8c00;
            --c4: #8e24aa; --c5: #039be5; --c6: #00897b; --c7: #d81b60; --c99: #607d8b;
        }
        * { box-sizing: border-box; font-family: 'Segoe UI', sans-serif; }
        body { margin: 0; background: var(--bg-dark); color: var(--text-main); display: flex; height: 100vh; overflow: hidden; }
        
        /* Layout */
        #left-panel { width: 35%; border-right: 1px solid var(--border); display: flex; flex-direction: column; background: var(--bg-panel); }
        #right-panel { width: 65%; display: flex; flex-direction: column; overflow-y: auto; padding: 20px; gap: 15px; }
        
        /* Left: Chat */
        .header { padding: 15px; background: var(--bg-dark); border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
        .header h3 { margin: 0; font-size: 15px; }
        .btn { background: var(--accent); color: white; border: none; padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: bold; transition: 0.2s; }
        .btn:hover { filter: brightness(1.2); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        
        #chat-history { flex: 1; padding: 15px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
        .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px; font-size: 13px; word-wrap: break-word; }
        .msg.user { background: var(--bubble-user); align-self: flex-end; border-bottom-right-radius: 4px; }
        .msg.bot { background: var(--bubble-bot); align-self: flex-start; border-bottom-left-radius: 4px; border: 1px solid var(--border); }
        .meta { font-size: 11px; color: var(--text-muted); margin-top: 5px; display: block; }
        .chat-input { padding: 15px; background: var(--bg-dark); border-top: 1px solid var(--border); display: flex; gap: 10px; }
        .chat-input input { flex: 1; background: var(--bg-panel); border: 1px solid var(--border); color: white; padding: 10px; border-radius: 6px; outline: none; }
        
        /* Right: Dashboard Elements */
        .card { background: var(--bg-card); padding: 15px; border-radius: 8px; border: 1px solid var(--border); }
        .card-title { margin: 0 0 12px 0; font-size: 13px; color: var(--text-muted); text-transform: uppercase; font-weight: 700; letter-spacing: 0.5px; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        select, input[type="file"] { width: 100%; background: var(--bg-dark); color: white; border: 1px solid var(--border); padding: 9px; border-radius: 4px; margin-bottom: 10px; }
        
        /* Progress & Stats */
        .progress-wrap { background: var(--bg-dark); border-radius: 6px; height: 12px; overflow: hidden; margin: 10px 0; border: 1px solid var(--border); }
        #progress-bar { background: var(--success); height: 100%; width: 0%; transition: width 0.1s linear; }
        .stats-row { display: flex; justify-content: space-between; font-size: 12px; color: var(--text-muted); }
        .stat-val { font-weight: bold; color: white; }
        
        /* Class Counters */
        .class-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 10px; }
        .class-box { background: var(--bg-dark); border: 1px solid var(--border); padding: 10px; border-radius: 6px; text-align: center; }
        .class-box .lbl { font-size: 10px; color: var(--text-muted); display: block; margin-bottom: 4px; text-transform: uppercase; }
        .class-box .cnt { font-size: 18px; font-weight: bold; }
        .badge { padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; color: white; }
        .c0 { background: var(--c0); } .c1 { background: var(--c1); } .c2 { background: var(--c2); } .c3 { background: var(--c3); }
        .c4 { background: var(--c4); } .c5 { background: var(--c5); } .c6 { background: var(--c6); } .c7 { background: var(--c7); } .c99 { background: var(--c99); }
        
        /* Table (Capped at 100 rows to prevent lag) */
        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); }
        th { background: var(--bg-dark); color: var(--text-muted); position: sticky; top: 0; }
        .table-wrap { flex: 1; overflow-y: auto; background: var(--bg-card); border-radius: 8px; border: 1px solid var(--border); }
    </style>
</head>
<body>

<div id="left-panel">
    <div class="header">
        <h3 id="session-info">Test Chat Interface</h3>
        <button class="btn" style="background:var(--accent)" onclick="newChat()">🔄 New Chat</button>
    </div>
    <div id="chat-history"></div>
    <div class="chat-input">
        <input type="text" id="manual-input" placeholder="Type to test manually (or type 'next')..." onkeypress="if(event.key === 'Enter') sendManual()">
        <button class="btn" onclick="sendManual()">Send</button>
    </div>
</div>

<div id="right-panel">
    <div class="grid-2">
        <div class="card">
            <h4 class="card-title">1. Data Source</h4>
            <button class="btn" style="width:100%; margin-bottom:10px; background:#43a047;" onclick="loadLocalJsonl()">📥 Load bot_chat_history.jsonl</button>
            <input type="file" id="file-upload" accept=".txt,.json,.jsonl" onchange="handleFileUpload(event)">
            <div id="loaded-info" style="font-size:12px; color:var(--text-muted); margin-top:5px;">No file loaded.</div>
        </div>
        <div class="card">
            <h4 class="card-title">2. Test Execution</h4>
            <select id="test-mode">
                <option value="sequential">Mode 1: Sequential Replay</option>
                <option value="wordbyword">Mode 2: Word-by-Word Decomposition</option>
                <option value="repeat100">Mode 3: Repeat 100x (Rotation Test)</option>
                <option value="fuzzing">Mode 4: Fuzzing & Auto-Mutations</option>
            </select>
            <div style="display:flex; gap:10px;">
                <button class="btn" id="btn-run" style="flex:1; background:var(--success);" onclick="startTest()">▶ Run</button>
                <button class="btn" id="btn-pause" style="flex:1; background:#fb8c00;" onclick="togglePause()" disabled>⏸ Pause</button>
                <button class="btn" id="btn-stop" style="flex:1; background:var(--danger);" onclick="stopTest()" disabled>⏹ Stop</button>
            </div>
        </div>
    </div>

    <div class="card">
        <h4 class="card-title">Live Execution Dashboard</h4>
        <div class="stats-row">
            <span>Progress: <span id="stat-prog" class="stat-val">0 / 0</span></span>
            <span>Speed: <span id="stat-spd" class="stat-val">0.00 ms/req</span></span>
            <span>ETA: <span id="stat-eta" class="stat-val">0s</span></span>
            <span>Errors: <span id="stat-err" class="stat-val" style="color:var(--danger)">0</span></span>
        </div>
        <div class="progress-wrap"><div id="progress-bar"></div></div>
        
        <div class="class-grid">
            <div class="class-box"><span class="lbl c0 badge">0: Reject</span><span class="cnt" id="cnt-0">0</span></div>
            <div class="class-box"><span class="lbl c1 badge">1: Greet</span><span class="cnt" id="cnt-1">0</span></div>
            <div class="class-box"><span class="lbl c2 badge">2: Ask Age</span><span class="cnt" id="cnt-2">0</span></div>
            <div class="class-box"><span class="lbl c3 badge">3: Tell Age</span><span class="cnt" id="cnt-3">0</span></div>
            <div class="class-box"><span class="lbl c4 badge">4: Ask Ctry</span><span class="cnt" id="cnt-4">0</span></div>
            <div class="class-box"><span class="lbl c5 badge">5: Tell Ctry</span><span class="cnt" id="cnt-5">0</span></div>
            <div class="class-box"><span class="lbl c6 badge">6: Age+Ctry</span><span class="cnt" id="cnt-6">0</span></div>
            <div class="class-box"><span class="lbl c7 badge">7: Underage</span><span class="cnt" id="cnt-7">0</span></div>
            <div class="class-box"><span class="lbl c99 badge">99: Reset</span><span class="cnt" id="cnt-99">0</span></div>
        </div>
    </div>

    <div class="table-wrap">
        <table id="results-table">
            <thead><tr><th>Input Text</th><th>Class</th><th>Intent</th><th>Latency</th></tr></thead>
            <tbody id="results-body"></tbody>
        </table>
    </div>
</div>

<script>
    let loadedMessages = [];
    
    // Test Engine State
    let engine = {
        queue: [], idx: 0, running: false, paused: false,
        stats: { c0:0, c1:0, c2:0, c3:0, c4:0, c5:0, c6:0, c7:0, c99:0, err:0, totalMs:0, startTime:0 }
    };
    
    // Sends "next" directly to trigger Class 99 reset on the same IP
    function newChat() {
        document.getElementById('chat-history').innerHTML = '';
        addChatBubble('system', `🔄 Starting new conversation via 'next' command...`);
        sendDirect('next');
    }

    function addChatBubble(sender, text, aiData = null) {
        const chat = document.getElementById('chat-history');
        const div = document.createElement('div');
        div.className = `msg ${sender}`;
        
        let html = `<div>${text}</div>`;
        if (aiData && !aiData.error) {
            html += `<span class="meta">[Class ${aiData.response.class_id}] ${aiData.response.intent} (${aiData.latency_ms.toFixed(2)}ms)</span>`;
        } else if (aiData && aiData.error) {
            html += `<span class="meta" style="color:var(--danger)">Error: AI Server Down</span>`;
        }
        
        div.innerHTML = html;
        chat.appendChild(div);
        
        // BIG DATA DOM PROTECTOR: Remove old bubbles to prevent browser freeze
        while (chat.children.length > 50) { chat.removeChild(chat.firstChild); }
        chat.scrollTop = chat.scrollHeight;
    }

    function addTableRow(input, aiData) {
        const tbody = document.getElementById('results-body');
        const tr = document.createElement('tr');
        if (aiData.error) {
            tr.innerHTML = `<td>${input}</td><td colspan="3" style="color:var(--danger)">Connection Error</td>`;
            engine.stats.err++;
        } else {
            const r = aiData.response;
            tr.innerHTML = `<td>${input}</td><td><span class="badge c${r.class_id}">Class ${r.class_id}</span></td><td>${r.intent}</td><td>${aiData.latency_ms.toFixed(2)} ms</td>`;
            if (engine.stats[`c${r.class_id}`] !== undefined) {
                engine.stats[`c${r.class_id}`]++;
            }
            engine.stats.totalMs += aiData.latency_ms;
        }
        tbody.prepend(tr);
        
        // TABLE DOM PROTECTOR: Max 100 rows
        while (tbody.children.length > 100) { tbody.removeChild(tbody.lastChild); }
    }

    async function sendRequest(text) {
        try {
            const res = await fetch('/proxy', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: text }) });
            return await res.json();
        } catch (e) { return { error: "Failed" }; }
    }

    async function sendDirect(text) {
        addChatBubble('user', text);
        const data = await sendRequest(text);
        addTableRow(text, data);
        if (!data.error) addChatBubble('bot', data.response.reply, data);
    }

    function sendManual() {
        const input = document.getElementById('manual-input');
        if (!input.value.trim()) return;
        const txt = input.value.trim();
        input.value = '';
        sendDirect(txt);
    }

    // --- File Parsing (Now detects Disconnected and injects 'next') ---
    function parseLines(textData) {
        const lines = textData.split(/\\r?\\n/);
        let msgs = [];
        for (let line of lines) {
            line = line.trim();
            if (!line) continue;
            if (line.startsWith('{')) {
                try {
                    let obj = JSON.parse(line);
                    if (obj.sender === 'user' && obj.text) {
                        msgs.push(obj.text);
                    } 
                    // Automatically trigger a conversation reset when JSON logs a disconnect
                    else if (obj.text && (obj.text.includes("Disconnected") || obj.text.toLowerCase() === "next")) {
                        msgs.push("next");
                    }
                } catch(e) {} 
            } else {
                msgs.push(line);
            }
        }
        return msgs;
    }

    async function loadLocalJsonl() {
        const res = await fetch('/read_local');
        const data = await res.json();
        if (data.lines) {
            loadedMessages = parseLines(data.lines.join(''));
            document.getElementById('loaded-info').innerText = `✅ Loaded ${loadedMessages.length} user commands/messages safely.`;
        } else { alert(data.error || "File not found in project directory."); }
    }

    function handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            loadedMessages = parseLines(ev.target.result);
            document.getElementById('loaded-info').innerText = `✅ Loaded ${loadedMessages.length} messages/commands from ${file.name}.`;
        };
        reader.readAsText(file);
    }

    // --- High-Performance Test Engine ---
    function startTest() {
        if (loadedMessages.length === 0 && document.getElementById('manual-input').value) {
            loadedMessages = [document.getElementById('manual-input').value];
        }
        if (loadedMessages.length === 0) return alert("Load a file or type a message first!");
        
        document.getElementById('results-body').innerHTML = '';
        const mode = document.getElementById('test-mode').value;
        
        // Build Queue
        engine.queue = [];
        if (mode === 'sequential') { engine.queue = [...loadedMessages]; } 
        else if (mode === 'wordbyword') {
            for (let msg of loadedMessages) {
                if (msg.toLowerCase() === "next") { engine.queue.push("next"); continue; }
                let parts = msg.split(' ');
                for (let i = 1; i <= parts.length; i++) { engine.queue.push(parts.slice(0, i).join(' ')); }
            }
        } else if (mode === 'repeat100') {
            for (let i = 0; i < 100; i++) { engine.queue.push(loadedMessages[0]); }
        } else if (mode === 'fuzzing') {
            for (let msg of loadedMessages) {
                if (msg.toLowerCase() === "next") { engine.queue.push("next"); continue; }
                engine.queue.push(msg); engine.queue.push(msg + "!");
                engine.queue.push("   " + msg + "  "); engine.queue.push(msg.toUpperCase());
                engine.queue.push(msg + " 200");
                if (msg.includes('i')) engine.queue.push(msg.replace('i', 'iiiii'));
            }
        }

        // Reset Stats
        engine.idx = 0; engine.running = true; engine.paused = false;
        engine.stats = { c0:0, c1:0, c2:0, c3:0, c4:0, c5:0, c6:0, c7:0, c99:0, err:0, totalMs:0, startTime: performance.now() };
        
        document.getElementById('btn-run').disabled = true;
        document.getElementById('btn-pause').disabled = false;
        document.getElementById('btn-stop').disabled = false;
        
        processChunk(); 
    }

    function togglePause() {
        engine.paused = !engine.paused;
        document.getElementById('btn-pause').innerText = engine.paused ? "▶ Resume" : "⏸ Pause";
        if (!engine.paused) processChunk();
    }

    function stopTest() {
        engine.running = false;
        document.getElementById('btn-run').disabled = false;
        document.getElementById('btn-pause').disabled = true;
        document.getElementById('btn-stop').disabled = true;
    }

    async function processChunk() {
        if (!engine.running || engine.paused) return;
        
        let chunkLimit = 5;
        let processedInChunk = 0;

        while (processedInChunk < chunkLimit && engine.idx < engine.queue.length && engine.running && !engine.paused) {
            let item = engine.queue[engine.idx];
            
            if (engine.queue.length <= 200 || engine.idx % 50 === 0 || item === "next") {
                addChatBubble(item === "next" ? 'system' : 'user', `[${engine.idx+1}/${engine.queue.length}] ${item}`);
            }
            
            let data = await sendRequest(item);
            addTableRow(item, data);
            
            if (!data.error && (engine.queue.length <= 200 || engine.idx % 50 === 0 || item === "next")) {
                addChatBubble('bot', data.response.reply, data);
            }
            
            engine.idx++;
            processedInChunk++;
        }

        updateDashboard();

        if (engine.idx < engine.queue.length) {
            setTimeout(processChunk, 0); 
        } else {
            stopTest(); 
            addChatBubble('system', `✅ Test completed. Processed ${engine.queue.length} commands.`);
        }
    }

    function updateDashboard() {
        const qLen = engine.queue.length;
        const pct = qLen > 0 ? (engine.idx / qLen) * 100 : 0;
        
        document.getElementById('stat-prog').innerText = `${engine.idx} / ${qLen}`;
        document.getElementById('progress-bar').style.width = `${pct}%`;
        
        const avgMs = engine.idx > 0 ? (engine.stats.totalMs / engine.idx) : 0;
        document.getElementById('stat-spd').innerText = `${avgMs.toFixed(2)} ms/req`;
        
        if (avgMs > 0) {
            const remReqs = qLen - engine.idx;
            const etaSec = (remReqs * avgMs) / 1000;
            document.getElementById('stat-eta').innerText = etaSec > 60 ? `${(etaSec/60).toFixed(1)}m` : `${etaSec.toFixed(1)}s`;
        }
        
        document.getElementById('stat-err').innerText = engine.stats.err;
        const classIds = [0, 1, 2, 3, 4, 5, 6, 7, 99];
        for (let i of classIds) {
            if(document.getElementById(`cnt-${i}`)) {
                document.getElementById(`cnt-${i}`).innerText = engine.stats[`c${i}`];
            }
        }
    }
</script>
</body>
</html>
"""

class TestStudioHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/proxy':
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body.decode('utf-8'))
            text = data.get('text', '')

            try:
                t0 = time.perf_counter()
                
                # We always use the same IP now, sending "next" handles the memory wipe directly
                req = f"POST / HTTP/1.1\r\nHost: 127.0.0.1:8080\r\nContent-Length: {len(text.encode('utf-8'))}\r\nConnection: close\r\n\r\n{text}"
                
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.0) 
                s.connect(('127.0.0.1', 8080))
                s.sendall(req.encode('utf-8'))
                
                resp = b""
                while True:
                    chunk = s.recv(4096)
                    if not chunk: break
                    resp += chunk
                s.close()
                elapsed = (time.perf_counter() - t0) * 1000

                resp_str = resp.decode('utf-8', errors='ignore')
                body_str = resp_str.split('\r\n\r\n', 1)[-1] if '\r\n\r\n' in resp_str else resp_str
                
                try:
                    ai_json = json.loads(body_str)
                except:
                    ai_json = {"error": "Invalid JSON", "raw": body_str}

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"latency_ms": elapsed, "response": ai_json}).encode('utf-8'))

            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

        elif self.path == '/read_local':
            try:
                if os.path.exists('bot_chat_history.jsonl'):
                    with open('bot_chat_history.jsonl', 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"lines": lines}).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "File not found"}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), TestStudioHandler) as httpd:
        print(f"[ULTRA TEST STUDIO] Running on http://127.0.0.1:{PORT}")
        print("Leave this window open and open the link in your web browser.")
        httpd.serve_forever()