import threading, urllib.request, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
BOX="http://localhost:8080"
latest={"jpg":b""}
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
HTML=b"""<html><head><style>body{background:#111;color:#ccc;font-family:sans-serif;text-align:center;margin:0}img{max-width:99vw}</style></head>
<body><div>T = toggle POLICY/TELEOP &mdash; LEFT external | RIGHT wrist &mdash; Arrows move EE, R/F up/down, Space gripper (Safari-friendly polling)</div>
<img id=v><script>
const v=document.getElementById('v');
setInterval(()=>{v.src='/frame?t='+Date.now()},80);
const ks={'arrowleft':'left','arrowright':'right','arrowup':'up','arrowdown':'down','r':'r','f':'f'};
function snd(p){fetch(p)}
addEventListener('keydown',e=>{if(e.repeat)return;let k=ks[e.key.toLowerCase()];if(k){snd('/key?k='+k+'&d=1');e.preventDefault()}if(e.key===' '){snd('/grip');e.preventDefault()}if(e.key.toLowerCase()==='t'){snd('/mode');e.preventDefault()}});
addEventListener('keyup',e=>{let k=ks[e.key.toLowerCase()];if(k){snd('/key?k='+k+'&d=0');e.preventDefault()}});
</script></body></html>"""
class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        if self.path=="/":
            self.send_response(200);self.send_header("Content-Type","text/html");self.end_headers();self.wfile.write(HTML)
        elif self.path.startswith("/frame"):
            j=latest["jpg"]; self.send_response(200);self.send_header("Content-Type","image/jpeg");self.send_header("Cache-Control","no-store");self.send_header("Content-Length",str(len(j)));self.end_headers();self.wfile.write(j)
        elif self.path.startswith("/key") or self.path.startswith("/grip") or self.path.startswith("/mode"):
            try: urllib.request.urlopen(BOX+self.path, timeout=2).read()
            except Exception: pass
            self.send_response(204);self.end_headers()
        else: self.send_response(404);self.end_headers()
print("PROXY on http://localhost:8090", flush=True)
ThreadingHTTPServer(("127.0.0.1",8090),H).serve_forever()
