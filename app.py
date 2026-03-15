"""
pi-Lingual — Flask web server with Server-Sent Events
Serves the beautiful UI and streams translation events in real time.

Usage:
  python3 app.py
  Open: http://localhost:5000  (or http://<pi-ip>:5000 from another device)
"""

import json
import threading
import queue
import time
from flask import Flask, Response, render_template_string, request, jsonify
from translator_pi import (
    record_audio, transcribe, translate, speak,
    get_targets, normalise_lang, LANG_NAMES, RECORD_SECONDS,
    SILENCE_THRESHOLD, OLLAMA_MODEL
)
import numpy as np

app = Flask(__name__)

# Global state — one session at a time (single-user Pi device)
_session_lock = threading.Lock()
_session_active = False
_event_queue = queue.Queue()


def emit(event_type, data):
    """Push an SSE event into the queue."""
    _event_queue.put({"type": event_type, "data": data})


# ── SSE stream endpoint ───────────────────────────────────────────────
@app.route("/stream")
def stream():
    def generate():
        while True:
            try:
                event = _event_queue.get(timeout=30)
                yield f"data: {json.dumps(event)}\n\n"
            except queue.Empty:
                yield "data: {\"type\":\"ping\"}\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── Trigger translation ───────────────────────────────────────────────
@app.route("/translate", methods=["POST"])
def run_translation():
    global _session_active
    if not _session_lock.acquire(blocking=False):
        return jsonify({"error": "Already running"}), 409
    _session_active = True

    def worker():
        global _session_active
        try:
            # Recording
            emit("pipeline", {"node": "mic", "state": "active"})
            emit("terminal", {"cls": "l-dim", "text": "● recording " + str(RECORD_SECONDS) + "s — speak now"})
            emit("phase", {"phase": "recording"})

            rms_values = []
            def rms_cb(rms):
                rms_values.append(rms)
                emit("rms", {"rms": round(rms, 5)})

            audio = record_audio(callback=rms_cb)
            emit("pipeline", {"node": "mic", "state": "done"})
            emit("phase", {"phase": "processing"})

            # Silence check
            rms = float(np.sqrt(np.mean(audio ** 2)))
            if rms < SILENCE_THRESHOLD:
                emit("terminal", {"cls": "l-dim",
                    "text": f"too quiet (rms={rms:.5f}) — nothing detected"})
                emit("phase", {"phase": "idle"})
                emit("done", {})
                return

            # Transcribe
            emit("pipeline", {"node": "whisper", "state": "active"})
            emit("terminal", {"cls": "l-dim", "text": "◌ transcribing..."})
            text, detected = transcribe(audio)
            lang = normalise_lang(detected)
            emit("pipeline", {"node": "whisper", "state": "done"})
            emit("phase", {"phase": "idle"})

            lang_cls = {"en": "l-en", "hi": "l-hi", "sk": "l-sk"}.get(lang, "l-dim")
            emit("terminal", {"cls": "spacer", "text": ""})
            emit("terminal", {"cls": lang_cls, "tag": lang.upper(), "text": text})
            emit("terminal", {"cls": "spacer", "text": ""})

            # Translate + speak each target language
            for target in get_targets(lang):
                tgt_name = LANG_NAMES[target]
                tgt_cls  = {"en": "l-en", "hi": "l-hi", "sk": "l-sk"}.get(target, "l-dim")

                emit("pipeline", {"node": "ollama", "state": "active"})
                emit("terminal", {"cls": "l-dim", "text": f"translating → {tgt_name.lower()}"})

                try:
                    translation = translate(text, lang, target)
                except RuntimeError as e:
                    emit("terminal", {"cls": "l-dim", "text": f"error: {e}"})
                    emit("pipeline", {"node": "ollama", "state": ""})
                    continue

                emit("pipeline", {"node": "ollama", "state": "done"})
                emit("terminal", {"cls": tgt_cls, "tag": target.upper(), "text": translation})

                emit("pipeline", {"node": "tts", "state": "active"})
                emit("pipeline", {"node": "out", "state": "active"})
                emit("phase", {"phase": "speaking"})
                emit("terminal", {"cls": "l-speak", "text": f"♪ speaking {tgt_name.lower()}"})

                try:
                    speak(translation, target)
                except Exception as e:
                    emit("terminal", {"cls": "l-dim", "text": f"speak error: {e}"})

                emit("pipeline", {"node": "tts", "state": "done"})
                emit("pipeline", {"node": "out", "state": "done"})
                emit("phase", {"phase": "idle"})

            emit("terminal", {"cls": "spacer", "text": ""})
            emit("pipeline_reset", {})
            emit("done", {})

        except Exception as e:
            emit("terminal", {"cls": "l-dim", "text": f"error: {e}"})
            emit("phase", {"phase": "idle"})
            emit("pipeline_reset", {}),
            emit("done", {})
        finally:
            _session_active = False
            _session_lock.release()

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True})


# ── Status ────────────────────────────────────────────────────────────
@app.route("/status")
def status():
    return jsonify({
        "active": _session_active,
        "model": OLLAMA_MODEL,
    })


# ── Main page ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


# ── HTML (the beautiful UI, inline) ──────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>π-Lingual</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Fraunces:ital,opsz,wght@0,9..144,100;0,9..144,300;1,9..144,100;1,9..144,300&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:#f5f2ed; --bg2:#edeae3; --bg3:#e2dfd7;
  --ink:#1a1714; --ink2:#3d3a35; --ink3:#7a7670; --ink4:#b0aca5;
  --accent:#c8441a; --accent2:#e8602e;
  --blue:#2255cc; --blue2:#3d72f0;
  --teal:#0e8a74; --teal2:#12b096;
  --amber:#b87010;
  --rule:rgba(26,23,20,0.12); --rule2:rgba(26,23,20,0.06);
}
html,body { width:100%; min-height:100vh; background:var(--bg);
  font-family:'DM Mono',monospace; color:var(--ink2); }
body::after {
  content:''; position:fixed; inset:0; pointer-events:none; opacity:0.6;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
}
.stage { max-width:900px; margin:0 auto; padding:28px 24px; position:relative; z-index:1; }

/* masthead */
.masthead { display:flex; align-items:baseline; justify-content:space-between;
  padding:0 0 14px; border-bottom:1.5px solid var(--ink);
  animation:appear 0.4s ease 0.1s both; }
.masthead-pi { font-family:'Fraunces',serif; font-size:46px; font-weight:100;
  font-style:italic; color:var(--ink); letter-spacing:-1px; line-height:1; }
.masthead-pi span { color:var(--accent); }
.masthead-right { display:flex; flex-direction:column; align-items:flex-end; gap:3px; }
.mh-tag { font-size:9px; letter-spacing:2.5px; color:var(--ink4); text-transform:uppercase; }
.mh-langs { font-size:11px; color:var(--ink2); }
.mh-hw { font-size:9px; letter-spacing:1.5px; color:var(--teal); text-transform:uppercase; }

/* body cols */
.cols { display:grid; grid-template-columns:1fr 320px; gap:0;
  animation:appear 0.4s ease 0.25s both; }

/* terminal col */
.term-col { border-right:1px solid var(--rule); padding:18px 22px 18px 0; min-height:400px;
  display:flex; flex-direction:column; }
.col-header { display:flex; align-items:center; justify-content:space-between;
  margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid var(--rule2); }
.col-label { font-size:9px; letter-spacing:2px; color:var(--ink4); text-transform:uppercase; }
.col-meta  { font-size:9px; color:var(--teal); letter-spacing:0.5px; }
.term-body { font-size:12px; line-height:1.9; flex:1; overflow-y:auto;
  max-height:340px; scroll-behavior:smooth; }
.term-body .line { display:block; }
.l-dim    { color:var(--ink4); }
.l-ready  { color:var(--teal); }
.l-prompt { color:var(--accent); }
.l-en     { color:var(--amber); }
.l-hi     { color:var(--accent); }
.l-sk     { color:var(--blue); }
.l-speak  { color:var(--ink4); font-size:11px; padding-left:14px; }
.tag { display:inline-block; font-size:8px; font-weight:500; letter-spacing:1.5px;
  padding:1px 5px; text-transform:uppercase; border:1px solid currentColor; margin-right:6px;
  vertical-align:middle; }
.tag-en{color:var(--amber)} .tag-hi{color:var(--accent)} .tag-sk{color:var(--blue)}
.spacer { display:block; height:4px; }

/* right col */
.right-col { padding:18px 0 18px 22px; display:flex; flex-direction:column; gap:22px; }
.sec-label { font-size:8px; letter-spacing:2.5px; color:var(--ink4); text-transform:uppercase;
  margin-bottom:9px; display:flex; justify-content:space-between; align-items:center; }

/* waveform */
.wave-bars { height:46px; display:flex; align-items:center; gap:2px;
  border-bottom:1px solid var(--rule); padding-bottom:4px; }
.wbar { width:5px; background:var(--ink4); min-height:2px; transition:height 0.08s ease; }

/* meter */
.meter-track { height:3px; background:var(--bg3); overflow:hidden; }
.meter-fill  { height:100%; width:0%; background:var(--accent2); transition:width 0.08s ease; }

/* pipeline */
.pipe-item { display:flex; align-items:center; gap:9px; padding:6px 0;
  border-bottom:1px solid var(--rule2); transition:all 0.2s ease; }
.pipe-item:last-child { border:none; }
.pipe-sq { width:5px; height:5px; background:var(--ink4); flex-shrink:0;
  transition:background 0.2s ease; }
.pipe-name   { font-size:10px; color:var(--ink3); flex:1; transition:color 0.2s; }
.pipe-detail { font-size:9px; color:var(--ink4); transition:color 0.2s; }
.pipe-item.active .pipe-sq     { background:var(--accent2); }
.pipe-item.active .pipe-name   { color:var(--accent); }
.pipe-item.active .pipe-detail { color:var(--accent2); }
.pipe-item.done .pipe-sq     { background:var(--teal2); }
.pipe-item.done .pipe-name   { color:var(--teal); }
.pipe-item.done .pipe-detail { color:var(--teal2); }

/* stats */
.stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.stat-box { border-top:1.5px solid var(--ink); padding-top:5px; }
.stat-n { font-family:'Fraunces',serif; font-size:26px; font-weight:100;
  color:var(--ink); line-height:1; }
.stat-n span { font-style:italic; color:var(--accent); }
.stat-l { font-size:8px; letter-spacing:1.5px; text-transform:uppercase;
  color:var(--ink4); margin-top:2px; }

/* record button */
.btn-wrap { animation:appear 0.4s ease 0.4s both; }
.rec-btn {
  width:100%; padding:14px; border:1.5px solid var(--ink);
  background:var(--bg); color:var(--ink); font-family:'DM Mono',monospace;
  font-size:12px; letter-spacing:2px; text-transform:uppercase; cursor:pointer;
  display:flex; align-items:center; justify-content:center; gap:10px;
  transition:all 0.15s ease; margin-top:0;
}
.rec-btn:hover { background:var(--ink); color:var(--bg); }
.rec-btn:active { transform:scale(0.99); }
.rec-btn.recording { background:var(--accent); color:#fff; border-color:var(--accent); }
.rec-btn.recording:hover { background:var(--accent2); }
.rec-btn.busy { opacity:0.45; cursor:not-allowed; }
.rec-dot { width:8px; height:8px; background:currentColor; }
.rec-dot.pulse { animation:pdot 0.8s ease-in-out infinite; }
@keyframes pdot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.3;transform:scale(0.5)} }

/* footer */
.footer { display:flex; justify-content:space-between; padding:10px 0 0;
  border-top:1px solid var(--rule); animation:appear 0.4s ease 0.5s both; }
.footer span { font-size:8.5px; color:var(--ink4); letter-spacing:1px; }
.fbadge { border:1px solid var(--rule); padding:1px 6px; font-size:8px;
  letter-spacing:1px; color:var(--ink3); margin-left:6px; }

.cursor { display:inline-block; width:6px; height:12px; background:var(--accent);
  vertical-align:middle; margin-left:2px; animation:blink 1s step-end infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
@keyframes appear { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
</style>
</head>
<body>
<div class="stage">

  <div class="masthead">
    <div class="masthead-pi">π<span>-Lingual</span></div>
    <div class="masthead-right">
      <div class="mh-tag">Offline · Open Source · Edge AI</div>
      <div class="mh-langs">English &thinsp;/&thinsp; हिंदी &thinsp;/&thinsp; Slovenčina</div>
      <div class="mh-hw">Raspberry Pi 4 · 4GB</div>
    </div>
  </div>

  <div class="cols">
    <div class="term-col">
      <div class="col-header">
        <span class="col-label">Terminal</span>
        <span class="col-meta" id="model-tag">whisper:tiny · qwen2.5:1.5b · piper</span>
      </div>
      <div class="term-body" id="tbody">
        <span class="line l-ready">✓ pi-lingual ready</span>
        <span class="line l-dim">press record to speak</span>
        <span class="spacer"></span>
      </div>
    </div>

    <div class="right-col">

      <div>
        <div class="sec-label">
          <span>Audio input</span>
          <span id="rms-val">—</span>
        </div>
        <div class="wave-bars" id="wbars"></div>
      </div>

      <div>
        <div class="sec-label"><span>Signal level</span></div>
        <div class="meter-track"><div class="meter-fill" id="meter"></div></div>
      </div>

      <div>
        <div class="sec-label"><span>Pipeline</span></div>
        <div id="pipeline">
          <div class="pipe-item" id="pn-mic">
            <div class="pipe-sq"></div>
            <div class="pipe-name">Microphone</div>
            <div class="pipe-detail">ALSA · built-in</div>
          </div>
          <div class="pipe-item" id="pn-whisper">
            <div class="pipe-sq"></div>
            <div class="pipe-name">Whisper</div>
            <div class="pipe-detail">tiny · local</div>
          </div>
          <div class="pipe-item" id="pn-ollama">
            <div class="pipe-sq"></div>
            <div class="pipe-name">Ollama LLM</div>
            <div class="pipe-detail">qwen2.5:1.5b</div>
          </div>
          <div class="pipe-item" id="pn-tts">
            <div class="pipe-sq"></div>
            <div class="pipe-name">Piper / gTTS</div>
            <div class="pipe-detail">offline · local</div>
          </div>
          <div class="pipe-item" id="pn-out">
            <div class="pipe-sq"></div>
            <div class="pipe-name">Speaker</div>
            <div class="pipe-detail">aplay · ALSA</div>
          </div>
        </div>
      </div>

      <div class="stats-grid">
        <div class="stat-box">
          <div class="stat-n">3</div>
          <div class="stat-l">Languages</div>
        </div>
        <div class="stat-box">
          <div class="stat-n"><span>4</span>GB</div>
          <div class="stat-l">RAM total</div>
        </div>
        <div class="stat-box">
          <div class="stat-n">0</div>
          <div class="stat-l">Cloud calls</div>
        </div>
        <div class="stat-box">
          <div class="stat-n"><span>1.5</span>B</div>
          <div class="stat-l">Model params</div>
        </div>
      </div>

    </div>
  </div>

  <div class="btn-wrap">
    <button class="rec-btn" id="rec-btn" onclick="startTranslation()">
      <div class="rec-dot" id="rec-dot"></div>
      <span id="rec-label">Record</span>
    </button>
  </div>

  <div class="footer">
    <span>RASPBERRY PI 4 · ARM CORTEX-A72 · RASPBIAN BOOKWORM</span>
    <span>github.com/nageshydv/pi-lingual-edge
      <span class="fbadge">MIT</span>
      <span class="fbadge">OFFLINE</span>
    </span>
  </div>

</div>

<script>
// ── Waveform ──────────────────────────────────────────────────────────
const wbEl = document.getElementById('wbars');
const NBARS = 32;
const bars = [];
for (let i = 0; i < NBARS; i++) {
  const b = document.createElement('div');
  b.className = 'wbar';
  b.style.height = '2px';
  wbEl.appendChild(b);
  bars.push(b);
}

let wPhase = 'idle';
let wT = 0;
let latestRms = 0;

function animWave() {
  wT += 0.09;
  for (let i = 0; i < NBARS; i++) {
    let h = 2, col;
    if (wPhase === 'recording') {
      const scale = Math.min(latestRms * 300, 1);
      h = 2 + Math.abs(Math.sin(wT * 2.2 + i * 0.42) * 18 * scale
            + Math.sin(wT * 4.0 + i * 0.2) * 8 * scale
            + Math.random() * 4 * scale);
      h = Math.min(h, 40);
      col = `rgba(184,112,16,${0.2 + h/90})`;
    } else if (wPhase === 'processing') {
      h = 2 + Math.abs(Math.sin(wT * 0.5 + i * 0.6) * 3);
      col = 'rgba(26,23,20,0.1)';
    } else if (wPhase === 'speaking') {
      h = 2 + Math.abs(Math.sin(wT * 3.3 + i * 0.5) * 14
            + Math.sin(wT * 1.8 + i * 0.3) * 6);
      h = Math.min(h, 34);
      col = `rgba(14,138,116,${0.18 + h/80})`;
    } else {
      h = 2 + Math.abs(Math.sin(wT * 0.3 + i * 0.9) * 1.2);
      col = 'rgba(26,23,20,0.07)';
    }
    bars[i].style.height = h + 'px';
    bars[i].style.background = col;
  }
  requestAnimationFrame(animWave);
}
animWave();

// ── Pipeline ──────────────────────────────────────────────────────────
function pset(id, state) {
  const el = document.getElementById(id);
  if (el) el.className = 'pipe-item' + (state ? ' ' + state : '');
}
function preset() {
  ['pn-mic','pn-whisper','pn-ollama','pn-tts','pn-out'].forEach(id => pset(id, ''));
}

// ── Terminal ──────────────────────────────────────────────────────────
const tbody = document.getElementById('tbody');

function addLine(cls, text, tag) {
  const line = document.createElement('span');
  line.className = 'line ' + cls;
  if (tag) {
    const t = document.createElement('span');
    t.className = 'tag tag-' + tag.toLowerCase();
    t.textContent = tag;
    line.appendChild(t);
    line.appendChild(document.createTextNode(text));
  } else if (cls === 'spacer') {
    // empty spacer
  } else {
    line.textContent = text;
  }
  tbody.appendChild(line);
  tbody.scrollTop = tbody.scrollHeight;
}

// ── SSE listener ──────────────────────────────────────────────────────
let es = null;

function startSSE() {
  if (es) { es.close(); }
  es = new EventSource('/stream');
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === 'ping') return;

    if (ev.type === 'terminal') {
      addLine(ev.data.cls, ev.data.text, ev.data.tag);
    }
    else if (ev.type === 'phase') {
      wPhase = ev.data.phase;
      const m = document.getElementById('meter');
      m.style.background = ev.data.phase === 'speaking' ? '#12b096' :
                           ev.data.phase === 'recording' ? '#e8602e' : '#b0aca5';
    }
    else if (ev.type === 'rms') {
      latestRms = ev.data.rms;
      const pct = Math.min(ev.data.rms * 800, 100);
      document.getElementById('meter').style.width = pct + '%';
      document.getElementById('rms-val').textContent = ev.data.rms.toFixed(4);
    }
    else if (ev.type === 'pipeline') {
      pset('pn-' + ev.data.node, ev.data.state);
    }
    else if (ev.type === 'pipeline_reset') {
      preset();
    }
    else if (ev.type === 'done') {
      wPhase = 'idle';
      document.getElementById('meter').style.width = '0%';
      document.getElementById('rms-val').textContent = '—';
      const btn = document.getElementById('rec-btn');
      btn.className = 'rec-btn';
      btn.disabled = false;
      document.getElementById('rec-label').textContent = 'Record';
      document.getElementById('rec-dot').className = 'rec-dot';
      addLine('l-prompt', '⏎ ready — press record to speak');
      addLine('spacer', '');
    }
  };
}

startSSE();

// ── Record button ─────────────────────────────────────────────────────
function startTranslation() {
  const btn = document.getElementById('rec-btn');
  if (btn.disabled) return;
  btn.disabled = true;
  btn.className = 'rec-btn recording';
  document.getElementById('rec-label').textContent = 'Recording...';
  document.getElementById('rec-dot').className = 'rec-dot pulse';

  fetch('/translate', { method: 'POST' })
    .then(r => r.json())
    .then(d => {
      if (d.error) {
        btn.className = 'rec-btn';
        btn.disabled = false;
        document.getElementById('rec-label').textContent = 'Record';
        document.getElementById('rec-dot').className = 'rec-dot';
        addLine('l-dim', 'error: ' + d.error);
      } else {
        btn.className = 'rec-btn busy';
        document.getElementById('rec-label').textContent = 'Processing...';
        document.getElementById('rec-dot').className = 'rec-dot';
      }
    })
    .catch(() => {
      btn.className = 'rec-btn';
      btn.disabled = false;
      document.getElementById('rec-label').textContent = 'Record';
    });
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("=" * 52)
    print("  π-Lingual  —  Raspberry Pi 4")
    print("=" * 52)
    print("  Open in browser:  http://localhost:5000")
    print("  From other device: http://<pi-ip>:5000")
    print("  Ctrl+C to quit")
    print("=" * 52)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
