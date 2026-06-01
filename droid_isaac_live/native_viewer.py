"""Native (pygame) teleop viewer for the DROID Isaac sim on the remote box.

Replaces the browser/proxy: one persistent connection for frames (/stream) and a
keep-alive connection for control commands, so there's no per-request tunnel
handshake (the ~0.8s latency) and no browser connection-pool freezes or stuck keys.

Run (after an SSH tunnel `localhost:8080 -> box:8080` is up):
    python3 native_viewer.py

Controls:  arrows = move EE in table plane,  R/F = up/down,  Space = gripper,
           T = toggle policy/teleop,  G = reset scene,  Esc/close = quit.
Mode is shown in the window title bar.
"""
import http.client, threading, io, time, json
import pygame

HOST, PORT = "localhost", 8080
SCALE = 1   # display scale for the combined (2-camera) frame

latest = {"jpg": None}
mode = {"v": "?"}
running = {"v": True}

# ---- frame stream (one persistent connection) -------------------------------
def frame_puller():
    while running["v"]:
        try:
            c = http.client.HTTPConnection(HOST, PORT, timeout=10)
            c.request("GET", "/stream"); r = c.getresponse()
            buf = b""
            while running["v"]:
                chunk = r.read(32768)
                if not chunk:
                    break
                buf += chunk
                while True:
                    s = buf.find(b"\xff\xd8"); e = buf.find(b"\xff\xd9", s + 2)
                    if s != -1 and e != -1:
                        latest["jpg"] = buf[s:e + 2]; buf = buf[e + 2:]
                    else:
                        break
                if len(buf) > 4_000_000:
                    buf = buf[-1_000_000:]
        except Exception:
            time.sleep(1)

# ---- control commands (one keep-alive connection, reused) -------------------
_cmd_lock = threading.Lock()
_cmd = {"c": None}
def _send(path):
    with _cmd_lock:
        for _ in range(2):
            try:
                if _cmd["c"] is None:
                    _cmd["c"] = http.client.HTTPConnection(HOST, PORT, timeout=5)
                _cmd["c"].request("GET", path)
                _cmd["c"].getresponse().read()
                return
            except Exception:
                try: _cmd["c"].close()
                except Exception: pass
                _cmd["c"] = None

def send(path):
    threading.Thread(target=_send, args=(path,), daemon=True).start()

def state_poller():
    while running["v"]:
        try:
            c = http.client.HTTPConnection(HOST, PORT, timeout=5)
            c.request("GET", "/state")
            mode["v"] = json.loads(c.getresponse().read()).get("mode", "?")
            c.close()
        except Exception:
            pass
        time.sleep(1.0)

threading.Thread(target=frame_puller, daemon=True).start()
threading.Thread(target=state_poller, daemon=True).start()
send("/key?k=warm&d=0")   # warm up the keep-alive command connection

# ---- pygame window ----------------------------------------------------------
pygame.init()
screen = pygame.display.set_mode((640, 180))
sized = {"wh": (640, 180)}
clock = pygame.time.Clock()
MOVE = {pygame.K_LEFT: 'left', pygame.K_RIGHT: 'right', pygame.K_UP: 'up',
        pygame.K_DOWN: 'down', pygame.K_r: 'r', pygame.K_f: 'f'}

while running["v"]:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running["v"] = False
        elif e.type == pygame.KEYDOWN:
            if e.key in MOVE: send("/key?k=%s&d=1" % MOVE[e.key])
            elif e.key == pygame.K_SPACE: send("/grip")
            elif e.key == pygame.K_t: send("/mode")
            elif e.key == pygame.K_g: send("/reset")
            elif e.key == pygame.K_ESCAPE: running["v"] = False
        elif e.type == pygame.KEYUP:
            if e.key in MOVE: send("/key?k=%s&d=0" % MOVE[e.key])

    j = latest["jpg"]
    if j:
        try:
            img = pygame.image.load(io.BytesIO(j))
            sw, sh = img.get_size()[0] * SCALE, img.get_size()[1] * SCALE
            if (sw, sh) != sized["wh"]:
                screen = pygame.display.set_mode((sw, sh)); sized["wh"] = (sw, sh)
            screen.blit(pygame.transform.scale(img, (sw, sh)), (0, 0))
            pygame.display.flip()
        except Exception:
            pass
    pygame.display.set_caption(
        "DROID teleop  |  MODE: %s  |  arrows/R/F move  Space gripper  T toggle  G reset"
        % mode["v"].upper())
    clock.tick(60)

running["v"] = False
pygame.quit()
