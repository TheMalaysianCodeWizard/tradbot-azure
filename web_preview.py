"""Browser chat page for the bot, built on Python's standard library.

    python web_preview.py      # then open http://127.0.0.1:8000

The Azure key stays on this local server (read from environment variables);
the browser only ever sees the bot's replies and the sentiment result.
"""

import json
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chatbot import ChatBot
from chatbot.config import DefaultConfig
from chatbot.sentiment import SentimentService

SERVICE = SentimentService()
BOT = ChatBot(sentiment=SERVICE)
SESSIONS: dict[str, dict] = {}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>TradBot</title>
<style>
body{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#f4f5f7;margin:0}
.wrap{max-width:760px;margin:24px auto;background:#fff;border:1px solid #dde;border-radius:12px;overflow:hidden}
header{padding:14px 18px;border-bottom:1px solid #eee}header b{font-size:18px}header div{color:#667;font-size:13px;margin-top:4px}
#log{height:470px;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
.m{max-width:78%;padding:9px 13px;border-radius:12px;white-space:pre-wrap;font-size:14px;line-height:1.4}
.u{align-self:flex-end;background:#2563eb;color:#fff}.b{align-self:flex-start;background:#eef0f4;color:#111}
.meta{font-size:11px;color:#889;margin-top:4px}
form{display:flex;border-top:1px solid #eee}input{flex:1;border:0;padding:14px;font-size:14px;outline:none}
button{border:0;background:#2563eb;color:#fff;padding:0 22px;font-size:14px;cursor:pointer}
</style></head><body><div class="wrap"><header><b>TradBot</b> &middot; traditional (non-LLM) chatbot
<div>Rules + TF-IDF matching, with sentiment from Azure AI Language. Type <i>help</i> to list capabilities.</div>
<div id="az" style="margin-top:6px;font-size:12px"></div></header>
<div id="log"></div><form id="f"><input id="i" autocomplete="off" placeholder="Type a message..."><button>Send</button></form></div>
<script>
const sid=crypto.randomUUID?crypto.randomUUID():String(Math.random());
const log=document.getElementById('log');
function add(cls,text,meta){const d=document.createElement('div');d.className='m '+cls;d.textContent=text;
 if(meta){const s=document.createElement('div');s.className='meta';s.textContent=meta;d.appendChild(s)}
 log.appendChild(d);log.scrollTop=log.scrollHeight}
async function send(text){add('u',text||' ');const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({message:text,session:sid})});const j=await r.json();
 const s=j.sentiment?` | Azure sentiment: ${j.sentiment}`:'';
 add('b',j.reply,`intent: ${j.intent??'none'} | confidence: ${j.confidence.toFixed(2)} | ${j.source}${s}`)}
fetch('/status').then(r=>r.json()).then(j=>{const e=document.getElementById('az');e.textContent=j.status;e.style.color=j.ok?'#15803d':'#b45309'});
document.getElementById('f').onsubmit=e=>{e.preventDefault();const i=document.getElementById('i');send(i.value);i.value=''};
window.sendMessage=send;
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/status":
            ok = SERVICE.available
            status = ("Connected to " + DefaultConfig.ENDPOINT_URI) if ok else ("Sentiment off: " + str(SERVICE.disabled_reason))
            return self._send(200, json.dumps({"ok": ok, "status": status}), "application/json")
        self._send(200, PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        if self.path != "/chat":
            return self._send(404, "{}", "application/json")
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 10_000)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            payload = {}
        session = SESSIONS.setdefault(str(payload.get("session") or uuid.uuid4()), {})
        reply = BOT.respond(payload.get("message"), session)
        s = reply.sentiment
        body = json.dumps({"reply": reply.text, "intent": reply.intent,
                           "confidence": reply.confidence, "source": reply.source,
                           "sentiment": s.summary() if s is not None else None})
        self._send(200, body, "application/json")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(DefaultConfig.describe())
    if not SERVICE.available:
        print(f"Sentiment analysis disabled: {SERVICE.disabled_reason}")
    print("TradBot running on http://127.0.0.1:8000  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
