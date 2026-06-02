import threading, urllib.request, time, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
BOX="http://localhost:8080"
latest={"jpg":b""}
state={"mode":"teleop"}   # cached locally so the browser never waits on the slow box

def cmd(path):
    """Forward a control command to the box off the request path (fire-and-forget)."""
    try: urllib.request.urlopen(BOX+path, timeout=3).read()
    except Exception: pass

def state_poller():
    """Refresh cached mode from the box in the background, never on a browser request."""
    while True:
        try:
            m=json.loads(urllib.request.urlopen(BOX+"/state", timeout=3).read()).get("mode")
            if m in ("teleop","policy"): state["mode"]=m
        except Exception: pass
        time.sleep(1.5)
threading.Thread(target=state_poller,daemon=True).start()
def puller():
    while True:
        try:
            r=urllib.request.urlopen(BOX+"/stream", timeout=10); buf=b""
            while True:
                c=r.read(16384)
                if not c: break
                buf+=c
                while True:
                    s=buf.find(b"\xff\xd8"); e=buf.find(b"\xff\xd9", s+2)
                    if s!=-1 and e!=-1: latest["jpg"]=buf[s:e+2]; buf=buf[e+2:]
                    else: break
                if len(buf)>4_000_000: buf=buf[-1_000_000:]
        except Exception: time.sleep(1)
threading.Thread(target=puller,daemon=True).start()
HTML=b"""<html><head><meta name=viewport content="width=device-width,initial-scale=1"><style>
body{background:#111;color:#ccc;font-family:sans-serif;text-align:center;margin:0}
img{max-width:99vw;display:block;margin:0 auto}
#controls{padding:10px}
button{font-size:17px;font-weight:bold;padding:12px 18px;margin:5px;border:0;border-radius:8px;color:#fff;cursor:pointer;background:#444;user-select:none;touch-action:none}
.mv{min-width:64px}
#reset{background:#c0392b}#toggle{background:#2980b9}#grip{background:#8e44ad}
.row{display:flex;justify-content:center;gap:6px}
</style></head>
<body>
<img id=v>
<div id=controls>
  <div class=row><button id=toggle tabindex=-1 onclick="snd('/mode');this.blur()">TOGGLE POLICY / TELEOP</button><button id=reset tabindex=-1 onclick="snd('/reset');this.blur()">RESET SCENE</button></div>
</div>
<script>
const v=document.getElementById('v');
// chained loader: fetch the next frame only after the current one renders,
// so slow frames never pile up / cancel each other (avoids stutter/freezes)
function nextFrame(){const im=new Image();im.onload=()=>{v.src=im.src;setTimeout(nextFrame,40)};im.onerror=()=>setTimeout(nextFrame,200);im.src='/frame?t='+Date.now()+Math.random()}
nextFrame();
function snd(p){fetch(p)}
const tg=document.getElementById('toggle');
function pollState(){fetch('/state').then(r=>r.json()).then(s=>{
  if(s.mode==='policy'){tg.textContent='POLICY - switch to teleop';tg.style.background='#d35400';}
  else if(s.mode==='teleop'){tg.textContent='TELEOP - switch to policy';tg.style.background='#27ae60';}
}).catch(()=>{})}
setInterval(pollState,700);pollState();
function hold(k){snd('/key?k='+k+'&d=1')}
function rel(k){snd('/key?k='+k+'&d=0')}
document.querySelectorAll('.mv').forEach(b=>{const k=b.dataset.k;
  b.addEventListener('mousedown',e=>{e.preventDefault();hold(k)});
  b.addEventListener('mouseup',e=>{e.preventDefault();rel(k)});
  b.addEventListener('mouseleave',()=>rel(k));
  b.addEventListener('touchstart',e=>{e.preventDefault();hold(k)},{passive:false});
  b.addEventListener('touchend',e=>{e.preventDefault();rel(k)},{passive:false});
});
const ks={'arrowleft':'left','arrowright':'right','arrowup':'up','arrowdown':'down','r':'r','f':'f'};
addEventListener('keydown',e=>{if(e.repeat)return;let k=ks[e.key.toLowerCase()];if(k){snd('/key?k='+k+'&d=1');e.preventDefault()}if(e.key===' '){snd('/grip');e.preventDefault()}if(e.key.toLowerCase()==='t'){snd('/mode');e.preventDefault()}if(e.key.toLowerCase()==='g'){snd('/reset');e.preventDefault()}});
addEventListener('keyup',e=>{let k=ks[e.key.toLowerCase()];if(k){snd('/key?k='+k+'&d=0');e.preventDefault()}});
</script></body></html>"""
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        if self.path=="/":
            self.send_response(200);self.send_header("Content-Type","text/html");self.end_headers();self.wfile.write(HTML)
        elif self.path.startswith("/frame"):
            j=latest["jpg"]
            try:
                self.send_response(200);self.send_header("Content-Type","image/jpeg");self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(j)));self.end_headers();self.wfile.write(j)
            except (BrokenPipeError, ConnectionResetError): pass
        elif self.path.startswith("/state"):
            body=('{"mode":"%s"}'%state["mode"]).encode()   # served from cache, instant
            try:
                self.send_response(200);self.send_header("Content-Type","application/json");self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError): pass
        elif self.path.startswith("/key") or self.path.startswith("/grip") or self.path.startswith("/mode") or self.path.startswith("/reset"):
            # update cached mode optimistically, forward to box in the background, reply now
            if self.path.startswith("/mode"): state["mode"]="policy" if state["mode"]=="teleop" else "teleop"
            elif self.path.startswith("/reset"): state["mode"]="teleop"
            threading.Thread(target=cmd,args=(self.path,),daemon=True).start()
            self.send_response(204);self.end_headers()
        else: self.send_response(404);self.end_headers()
print("PROXY on http://localhost:8090", flush=True)
ThreadingHTTPServer(("127.0.0.1",8090),H).serve_forever()
