"""Native (pygame) teleop + eval viewer for the DROID Isaac sim on the remote box.

One persistent connection for frames (/stream) and a keep-alive connection for control
commands, so there's no per-request tunnel handshake and no browser-class freezes/stuck keys.

UI (below the camera view):
  - mode buttons:  REMOTE TELEOP / PI0.5 POLICY  (click to select; active one highlighted)
  - SUCCESS / FAIL / RESET buttons          (you score each run manually)
  - a live policy timer (from policy start to end)
  - a SCOREBOARD of trials 1..10: success time + green check / red x

Keyboard:  arrows/R/F move, Space gripper, T toggle, G reset, S success, X fail, Esc quit.

Run (after an SSH tunnel `localhost:8080 -> box:8080` is up):
    python3 native_viewer.py
"""
import http.client, threading, io, time, json, socket
import pygame
import numpy as np
import imageio

HOST, PORT = "116.127.115.27", 31755   # direct Vast-mapped port (no SSH tunnel)
SCALE = 1            # display scale for the combined (2-camera) frame
PANEL_H = 250        # height of the control + scoreboard panel below the cameras
N_TRIALS = 10        # scoreboard slots

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
                    _cmd["c"].connect()
                    try: _cmd["c"].sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    except Exception: pass
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
        time.sleep(0.6)

threading.Thread(target=frame_puller, daemon=True).start()
threading.Thread(target=state_poller, daemon=True).start()
send("/key?k=warm&d=0")   # warm up the keep-alive command connection

# ---- run / eval tracking ----------------------------------------------------
trials = []   # list of (duration_seconds, success_bool), one per scored run
run = {"active": False, "start": 0.0, "dur": 0.0, "scored": True}

def start_run():
    run.update(active=True, start=time.time(), dur=0.0, scored=False)

def stop_run():
    if run["active"]:
        run["dur"] = time.time() - run["start"]; run["active"] = False

def score(success):
    stop_run()                                  # freeze the timer if still running
    if not run["scored"] and run["dur"] > 0:
        trials.append((run["dur"], success))    # record the trial
        run["scored"] = True

def to_teleop():
    if mode["v"] == "policy": send("/mode")
def to_policy():
    if mode["v"] == "teleop": send("/mode")

# ---- panel self-recording (records the composited window, no OS permission) -
rec = {"w": None, "path": None, "last": 0.0}
def toggle_rec():
    if rec["w"] is None:
        path = "/tmp/droid_panel_%d.mp4" % int(time.time())
        try:
            rec["w"] = imageio.get_writer(path, fps=20, codec="libx264", quality=8, macro_block_size=16)
            rec["path"] = path; rec["last"] = 0.0
            print("REC START", path, flush=True)
        except Exception as ex:
            print("REC ERR", ex, flush=True); rec["w"] = None
    else:
        try: rec["w"].close()
        except Exception: pass
        print("REC SAVED", rec["path"], flush=True); rec["w"] = None

# ---- pygame window ----------------------------------------------------------
pygame.init()
GREEN, ORANGE, RED, GREY, DK = (39, 174, 96), (211, 84, 0), (192, 57, 43), (70, 70, 70), (25, 25, 25)
big = pygame.font.SysFont("Arial", 24, bold=True)
font = pygame.font.SysFont("Arial", 20, bold=True)
small = pygame.font.SysFont("Arial", 16)
screen = pygame.display.set_mode((640, 180 + PANEL_H))
sized = {"wh": (640, 180 + PANEL_H), "sw": 640, "sh": 180}
clock = pygame.time.Clock()
MOVE = {pygame.K_LEFT: 'left', pygame.K_RIGHT: 'right', pygame.K_UP: 'up',
        pygame.K_DOWN: 'down', pygame.K_r: 'r', pygame.K_f: 'f'}
buttons = {}
cells = []

def layout(sw, sh):
    buttons.clear(); cells.clear()
    by, pad, bh = sh, 12, 44
    buttons["teleop"] = pygame.Rect(pad, by + pad, 280, bh)
    buttons["policy"] = pygame.Rect(pad * 2 + 280, by + pad, 280, bh)
    buttons["success"] = pygame.Rect(pad, by + pad * 2 + bh, 150, bh)
    buttons["fail"] = pygame.Rect(pad * 2 + 150, by + pad * 2 + bh, 150, bh)
    buttons["reset"] = pygame.Rect(pad * 3 + 300, by + pad * 2 + bh, 150, bh)
    buttons["rec"] = pygame.Rect(pad * 4 + 450, by + pad * 2 + bh, 150, bh)
    # scoreboard row of N_TRIALS cells
    sy = by + pad * 3 + bh * 2 + 6
    cw = (sw - pad * 2) // N_TRIALS
    for i in range(N_TRIALS):
        cells.append(pygame.Rect(pad + i * cw, sy, cw - 4, 64))

def draw_btn(rect, label, bg):
    pygame.draw.rect(screen, bg, rect, border_radius=6)
    t = font.render(label, True, (255, 255, 255))
    screen.blit(t, (rect.x + (rect.w - t.get_width()) // 2, rect.y + (rect.h - t.get_height()) // 2))

def draw_check(r):
    cx, cy = r.centerx, r.bottom - 16
    pygame.draw.lines(screen, (255, 255, 255), False,
                      [(cx - 9, cy), (cx - 2, cy + 7), (cx + 11, cy - 9)], 4)

def draw_x(r):
    cx, cy = r.centerx, r.bottom - 16
    pygame.draw.line(screen, (255, 255, 255), (cx - 8, cy - 8), (cx + 8, cy + 8), 4)
    pygame.draw.line(screen, (255, 255, 255), (cx + 8, cy - 8), (cx - 8, cy + 8), 4)

def click(pos):
    if buttons.get("teleop") and buttons["teleop"].collidepoint(pos): to_teleop()
    elif buttons.get("policy") and buttons["policy"].collidepoint(pos): to_policy()
    elif buttons.get("success") and buttons["success"].collidepoint(pos): score(True); to_teleop()
    elif buttons.get("fail") and buttons["fail"].collidepoint(pos): score(False); to_teleop()
    elif buttons.get("reset") and buttons["reset"].collidepoint(pos): stop_run(); send("/reset")
    elif buttons.get("rec") and buttons["rec"].collidepoint(pos): toggle_rec()

prev_mode = None
while running["v"]:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running["v"] = False
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            click(e.pos)
        elif e.type == pygame.KEYDOWN:
            if e.key in MOVE: send("/key?k=%s&d=1" % MOVE[e.key])
            elif e.key == pygame.K_SPACE: send("/grip")
            elif e.key == pygame.K_t: send("/mode")
            elif e.key == pygame.K_g: stop_run(); send("/reset")
            elif e.key == pygame.K_s: score(True); to_teleop()
            elif e.key == pygame.K_x: score(False); to_teleop()
            elif e.key == pygame.K_c: toggle_rec()
            elif e.key == pygame.K_ESCAPE: running["v"] = False
        elif e.type == pygame.KEYUP:
            if e.key in MOVE: send("/key?k=%s&d=0" % MOVE[e.key])

    m = mode["v"]
    if m == "policy" and prev_mode != "policy":
        start_run()
    elif m == "teleop" and prev_mode == "policy":
        stop_run()
    prev_mode = m

    j = latest["jpg"]
    if j:
        try:
            img = pygame.image.load(io.BytesIO(j))
            sw, sh = img.get_size()[0] * SCALE, img.get_size()[1] * SCALE
            if (sw, sh + PANEL_H) != sized["wh"]:
                screen = pygame.display.set_mode((sw, sh + PANEL_H))
                sized.update(wh=(sw, sh + PANEL_H), sw=sw, sh=sh); layout(sw, sh)
            screen.fill(DK)
            screen.blit(pygame.transform.scale(img, (sw, sh)), (0, 0))
        except Exception:
            pass
    if not buttons:
        layout(sized["sw"], sized["sh"])

    # ---- mode + outcome buttons ----
    pol = (m == "policy")
    draw_btn(buttons["teleop"], "REMOTE TELEOP", GREEN if not pol else GREY)
    draw_btn(buttons["policy"], "PI0.5 POLICY", ORANGE if pol else GREY)
    draw_btn(buttons["success"], "SUCCESS", GREEN)
    draw_btn(buttons["fail"], "FAIL", RED)
    draw_btn(buttons["reset"], "RESET", DK)
    draw_btn(buttons["rec"], ("STOP REC" if rec["w"] else "REC"), RED if rec["w"] else (90, 60, 60))

    # ---- live timer / pending prompt (right of the buttons) ----
    tx = sized["sw"] - 360
    if run["active"]:
        screen.blit(big.render("POLICY RUNNING", True, ORANGE), (tx, sized["sh"] + 14))
        screen.blit(big.render("%.1f s" % (time.time() - run["start"]), True, ORANGE), (tx, sized["sh"] + 44))
    elif not run["scored"] and run["dur"] > 0:
        screen.blit(font.render("ran %.1f s" % run["dur"], True, (255, 230, 0)), (tx, sized["sh"] + 16))
        screen.blit(font.render("click SUCCESS or FAIL", True, (255, 230, 0)), (tx, sized["sh"] + 44))
    succ = sum(1 for _, s in trials if s)
    screen.blit(font.render("successes: %d / %d" % (succ, len(trials)), True, (220, 220, 220)),
                (tx, sized["sh"] + 74))

    # ---- scoreboard: trials 1..N ----
    shown = trials[-N_TRIALS:]
    base = len(trials) - len(shown)
    for i, cell in enumerate(cells):
        if i < len(shown):
            dur, ok = shown[i]
            pygame.draw.rect(screen, GREEN if ok else RED, cell, border_radius=6)
            screen.blit(small.render("T%d" % (base + i + 1), True, (255, 255, 255)), (cell.x + 6, cell.y + 4))
            ts = font.render("%.1fs" % dur, True, (255, 255, 255))
            screen.blit(ts, (cell.centerx - ts.get_width() // 2, cell.y + 22))
            (draw_check if ok else draw_x)(cell)
        else:
            pygame.draw.rect(screen, (45, 45, 45), cell, border_radius=6)
            screen.blit(small.render("T%d" % (i + 1), True, (110, 110, 110)), (cell.x + 6, cell.y + 4))

    pygame.display.flip()
    pygame.display.set_caption("DROID teleop / eval  |  MODE: %s%s" % (m.upper(), "  [REC]" if rec["w"] else ""))

    # record the composited panel at ~20 fps
    if rec["w"] is not None:
        now = time.time()
        if now - rec["last"] >= 0.05:
            try:
                rec["w"].append_data(np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2)))
                rec["last"] = now
            except Exception:
                pass
    clock.tick(60)

running["v"] = False
if rec["w"] is not None:
    try: rec["w"].close(); print("REC SAVED", rec["path"], flush=True)
    except Exception: pass
pygame.quit()
