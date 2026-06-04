"""Eval + reset loop UI for the real SO101 (pygame) — SO101 port of the DROID
native_viewer. One window:

  - LIVE camera view (both third-person feeds = exactly what MolmoAct sees, 640x480)
  - mode buttons:  TELEOP (our EE-IK keyboard control)  /  EVAL (MolmoAct policy)
  - SUCCESS / FAIL / RESET buttons  (score each eval run; RESET drives to home)
  - per-run timer + a SCOREBOARD of the last N trials (time + green check / red x)

Keyboard:  arrows/r/f move EE, e/d roll, space gripper, t toggle mode,
           g reset-to-home, s success, x fail, esc quit.

Arm + cameras are LOCAL (owned by a controller thread); only MolmoAct is remote
(via the tunnel on :8202). Run in a normal Terminal:
    .venv-lerobot/bin/python scripts/so101_eval_ui.py --prompt "reach for the pink ball, pick up the pink ball, and place the ball on the red plate"
"""
from __future__ import annotations
import argparse, os, sys, time, threading
import numpy as np
import cv2
import placo
import pygame
import imageio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from molmoact2 import adapter
from molmoact2.client import MolmoActClient, Observation

PORT = "/dev/tty.usbmodem58FD0169761"
URDF = os.path.join(REPO, "assets", "so101", "so101_new_calib.urdf")
ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINTS = ARM + ["gripper"]
CAM_IDX = [0, 1]
HOME = np.array([-0.53, -99.14, 91.41, 60.64, -3.60, 1.10], dtype=np.float32)  # follower deg
POS_STEP, ROLL_STEP = 0.004, np.deg2rad(3)
MAX_STEP_DEG = 3.0
WS_LO, WS_HI = np.array([0.08, -0.25, -0.02]), np.array([0.34, 0.25, 0.30])
GRIP_OPEN, GRIP_CLOSED = 45.0, 2.0
POS_W, ORI_W = 1.0, 0.6
N_TRIALS = 5           # scoreboard slots (5 rollouts)
MAX_RUN_S = 180.0      # 3-minute cap per eval run

# ---- placo IK (anti-flail) -------------------------------------------------
_k = placo.RobotWrapper(URDF); _s = placo.KinematicsSolver(_k); _s.mask_fbase(True)
_task = _s.add_frame_task("gripper_frame_link", np.eye(4))
_s.dt = 0.05; _s.enable_joint_limits(True); _s.enable_velocity_limits(True)
for _n in ARM: _k.set_velocity_limit(_n, np.deg2rad(120))
_s.add_regularization_task(1e-3)

def fk(qd):
    for n, a in zip(ARM, qd): _k.set_joint(n, float(a))
    _k.update_kinematics(); return np.array(_k.get_T_world_frame("gripper_frame_link"))
def ik(q0, T):
    for n, a in zip(ARM, q0): _k.set_joint(n, float(a))
    _task.T_world_frame = T; _task.configure("gripper_frame_link", "soft", POS_W, ORI_W)
    _s.solve(True); _k.update_kinematics(); return np.array([_k.get_joint(n) for n in ARM])
def Rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Controller(threading.Thread):
    """Owns the follower + cameras; runs teleop / eval / reset per `mode`."""
    def __init__(self, url, prompt, exec_steps, hz):
        super().__init__(daemon=True)
        self.url, self.prompt, self.exec_steps = url, prompt, exec_steps
        self.dt = 1.0 / hz
        self.mode = "idle"            # idle | teleop | eval | reset
        self.keys = set()             # held teleop keys
        self.frames_bgr = [None, None]
        self.lock = threading.Lock()
        self.running = True
        self.status = "starting"
        self._pending_grip_toggle = False
        self._req_mode = None

    def request_mode(self, m): self._req_mode = m

    def toggle_grip(self): self._pending_grip_toggle = True

    def run(self):
        caps = []
        for i in CAM_IDX:
            c = cv2.VideoCapture(i); c.set(cv2.CAP_PROP_FRAME_WIDTH, 640); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            caps.append(c)
        robot = SO101Follower(SO101FollowerConfig(port=PORT, id="blupe_follower",
                              max_relative_target=None, disable_torque_on_disconnect=False))
        robot.connect(calibrate=False)
        client = MolmoActClient(self.url, conv=adapter.LEROBOT_V21_COMPAT_DEG, timeout_s=120.0)

        def arm_deg():
            o = robot.get_observation(); return np.array([o[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
        def send(q6):
            try: robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, q6)})
            except Exception: pass

        q_cmd = arm_deg()                  # internal commanded joints (smooth)
        ee_pos = fk(np.deg2rad(q_cmd[:5]))[:3, 3].copy()
        R_home = fk(np.deg2rad(q_cmd[:5]))[:3, :3].copy()
        roll, grip = 0.0, float(q_cmd[5])
        chunk, cidx = None, 0
        try:
            while self.running:
                # always refresh frames (display + eval input)
                for n, c in enumerate(caps):
                    ok, f = c.read()
                    if ok:
                        with self.lock: self.frames_bgr[n] = f
                # mode transition: re-init from current pose
                if self._req_mode is not None:
                    self.mode = self._req_mode; self._req_mode = None
                    q_cmd = arm_deg(); ee_pos = fk(np.deg2rad(q_cmd[:5]))[:3, 3].copy()
                    roll = 0.0; chunk = None; cidx = 0

                if self.mode == "teleop":
                    k = self.keys
                    if pygame.K_UP in k:    ee_pos[0] += POS_STEP
                    if pygame.K_DOWN in k:  ee_pos[0] -= POS_STEP
                    if pygame.K_LEFT in k:  ee_pos[1] += POS_STEP
                    if pygame.K_RIGHT in k: ee_pos[1] -= POS_STEP
                    if pygame.K_r in k:     ee_pos[2] += POS_STEP
                    if pygame.K_f in k:     ee_pos[2] -= POS_STEP
                    if pygame.K_e in k:     roll += ROLL_STEP
                    if pygame.K_d in k:     roll -= ROLL_STEP
                    if self._pending_grip_toggle:
                        grip = GRIP_CLOSED if grip == GRIP_OPEN else GRIP_OPEN; self._pending_grip_toggle = False
                    ee_pos[:] = np.clip(ee_pos, WS_LO, WS_HI)
                    T = np.eye(4); psi = np.arctan2(ee_pos[1], ee_pos[0])
                    T[:3, :3] = Rz(psi) @ R_home @ Rz(roll); T[:3, 3] = ee_pos
                    q_ik = np.rad2deg(ik(np.deg2rad(q_cmd[:5]), T))
                    q_cmd[:5] = q_cmd[:5] + np.clip(q_ik - q_cmd[:5], -MAX_STEP_DEG, MAX_STEP_DEG)
                    q_cmd[5] = grip
                    send(q_cmd); self.status = "teleop"

                elif self.mode == "eval":
                    if chunk is None or cidx >= min(self.exec_steps, len(chunk)):
                        with self.lock:
                            fr = [self.frames_bgr[0], self.frames_bgr[1]]
                        if fr[0] is None or fr[1] is None:
                            time.sleep(self.dt); continue
                        imgs = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in fr]
                        st = arm_deg()
                        try:
                            chunk = client.act(Observation(images=imgs, state_rad=st, instruction=self.prompt))
                            cidx = 0; self.status = "eval (querying ok)"
                        except Exception as ex:
                            self.status = f"eval ERR: {type(ex).__name__}"; time.sleep(0.3); continue
                    tgt = chunk[cidx]; cidx += 1
                    q_cmd = q_cmd + np.clip(tgt - q_cmd, -MAX_STEP_DEG, MAX_STEP_DEG)
                    send(q_cmd)

                elif self.mode == "reset":
                    err = HOME - q_cmd
                    if np.abs(err).max() <= 2.0:
                        self.mode = "idle"; self.status = "idle (home)"
                    else:
                        q_cmd = q_cmd + np.clip(err, -MAX_STEP_DEG, MAX_STEP_DEG); send(q_cmd)
                else:
                    self.status = "idle"
                time.sleep(self.dt)
        finally:
            for c in caps:
                try: c.release()
                except Exception: pass
            try: robot.disconnect()
            except Exception: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8202")
    ap.add_argument("--prompt", default="reach for the pink ball, pick up the pink ball, and place the ball on the red plate")
    ap.add_argument("--exec-steps", type=int, default=30)
    ap.add_argument("--hz", type=float, default=10.0)
    args = ap.parse_args()

    ctrl = Controller(args.url, args.prompt, args.exec_steps, args.hz)
    ctrl.start()

    pygame.init()
    GREEN, ORANGE, RED, DK = (39, 174, 96), (211, 84, 0), (192, 57, 43), (25, 25, 25)
    font = pygame.font.SysFont("Arial", 18, bold=True); small = pygame.font.SysFont("Arial", 14)
    huge = pygame.font.SysFont("Arial", 52, bold=True)
    CAM_W, CAM_H, PANEL_H = 320, 240, 250
    W, H = CAM_W * 2, CAM_H + PANEL_H
    screen = pygame.display.set_mode((W, H)); clock = pygame.time.Clock()

    trials = []                                   # (duration, success)
    run = {"active": False, "start": 0.0, "dur": 0.0, "scored": True}
    def start_run(): run.update(active=True, start=time.time(), dur=0.0, scored=False)
    def stop_run():
        if run["active"]: run["dur"] = time.time() - run["start"]; run["active"] = False
    def score(ok):
        stop_run()
        if not run["scored"] and run["dur"] > 0:
            trials.append((run["dur"], ok)); run["scored"] = True

    rec = {"w": None, "path": None, "last": 0.0}   # window video recorder
    def toggle_rec():
        if rec["w"] is None:
            rec["path"] = "/tmp/so101_eval_%d.mp4" % int(time.time())
            try:
                rec["w"] = imageio.get_writer(rec["path"], fps=20, codec="libx264",
                                              quality=8, macro_block_size=16)
                print("REC START", rec["path"], flush=True)
            except Exception as ex:
                print("REC ERR", ex, flush=True); rec["w"] = None
        else:
            try: rec["w"].close(); print("REC SAVED", rec["path"], flush=True)
            except Exception: pass
            rec["w"] = None

    by = CAM_H
    btn = {
        "teleop": pygame.Rect(12, by + 10, 305, 40), "eval": pygame.Rect(323, by + 10, 305, 40),
        "success": pygame.Rect(12, by + 58, 118, 40), "fail": pygame.Rect(138, by + 58, 118, 40),
        "reset": pygame.Rect(264, by + 58, 118, 40), "rec": pygame.Rect(390, by + 58, 118, 40),
    }
    cells = []
    cw = (W - 24) // N_TRIALS
    for i in range(N_TRIALS): cells.append(pygame.Rect(12 + i * cw, by + 130, cw - 4, 64))

    def to_teleop(): ctrl.request_mode("teleop")
    def to_eval(): start_run(); ctrl.request_mode("eval")
    def do_reset(): stop_run(); ctrl.request_mode("reset")

    def draw_btn(r, label, bg):
        pygame.draw.rect(screen, bg, r, border_radius=6)
        t = font.render(label, True, (255, 255, 255))
        screen.blit(t, (r.centerx - t.get_width() // 2, r.centery - t.get_height() // 2))

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT: running = False
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                p = e.pos
                if btn["teleop"].collidepoint(p): to_teleop()
                elif btn["eval"].collidepoint(p): to_eval()
                elif btn["success"].collidepoint(p): score(True); to_teleop()
                elif btn["fail"].collidepoint(p): score(False); to_teleop()
                elif btn["reset"].collidepoint(p): do_reset()
                elif btn["rec"].collidepoint(p): toggle_rec()
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE: running = False
                elif e.key == pygame.K_t: to_eval() if ctrl.mode != "eval" else to_teleop()
                elif e.key == pygame.K_g: do_reset()
                elif e.key == pygame.K_s: score(True); to_teleop()
                elif e.key == pygame.K_x: score(False); to_teleop()
                elif e.key == pygame.K_c: toggle_rec()
                elif e.key == pygame.K_SPACE: ctrl.toggle_grip()
                else: ctrl.keys.add(e.key)
            elif e.type == pygame.KEYUP:
                ctrl.keys.discard(e.key)

        # 3-minute cap: auto-stop an eval run that runs too long
        if run["active"] and time.time() - run["start"] >= MAX_RUN_S:
            stop_run(); to_teleop()

        screen.fill(DK)
        # ---- camera views ----
        with ctrl.lock:
            frames = [None if f is None else f.copy() for f in ctrl.frames_bgr]
        for i, f in enumerate(frames):
            if f is not None:
                rgb = cv2.cvtColor(cv2.resize(f, (CAM_W, CAM_H)), cv2.COLOR_BGR2RGB)
                surf = pygame.image.frombuffer(rgb.tobytes(), (CAM_W, CAM_H), "RGB")
                screen.blit(surf, (i * CAM_W, 0))

        ev = (ctrl.mode == "eval")
        draw_btn(btn["teleop"], "TELEOP", GREEN if not ev else (70, 70, 70))
        draw_btn(btn["eval"], "EVAL (MolmoAct)", ORANGE if ev else (70, 70, 70))
        draw_btn(btn["success"], "SUCCESS", GREEN)
        draw_btn(btn["fail"], "FAIL", RED)
        draw_btn(btn["reset"], "RESET", (60, 60, 60))
        draw_btn(btn["rec"], "STOP REC" if rec["w"] else "REC", RED if rec["w"] else (90, 60, 60))

        # status line below the buttons
        succ = sum(1 for _, s in trials if s)
        if not run["scored"] and run["dur"] > 0:
            screen.blit(font.render("ran %.1fs — click SUCCESS or FAIL" % run["dur"], True, (255, 230, 0)), (12, by + 104))
        else:
            screen.blit(small.render("successes: %d / %d    mode: %s" % (succ, len(trials), ctrl.status),
                        True, (220, 220, 220)), (12, by + 104))

        # ---- BIG 3-minute countdown overlay (over the cameras) ----
        if run["active"]:
            remain = max(0.0, MAX_RUN_S - (time.time() - run["start"]))
            col = GREEN if remain > 60 else (ORANGE if remain > 20 else RED)
            pygame.draw.rect(screen, (50, 50, 50), (0, 0, W, 12))                          # track
            pygame.draw.rect(screen, col, (0, 0, int(W * remain / MAX_RUN_S), 12))         # shrinking fill
            mmss = "%d:%02d" % (int(remain) // 60, int(remain) % 60)
            t = huge.render(mmss, True, col)
            plate = pygame.Rect(W // 2 - t.get_width() // 2 - 14, 16, t.get_width() + 28, t.get_height() + 8)
            ov = pygame.Surface((plate.w, plate.h)); ov.set_alpha(190); ov.fill((0, 0, 0))
            screen.blit(ov, (plate.x, plate.y)); screen.blit(t, (plate.x + 14, plate.y + 4))

        shown = trials[-N_TRIALS:]; base = len(trials) - len(shown)
        for i, cell in enumerate(cells):
            if i < len(shown):
                dur, ok = shown[i]
                pygame.draw.rect(screen, GREEN if ok else RED, cell, border_radius=6)
                screen.blit(small.render("T%d %.1fs" % (base + i + 1, dur), True, (255, 255, 255)), (cell.x + 5, cell.y + 6))
            else:
                pygame.draw.rect(screen, (45, 45, 45), cell, border_radius=6)
        pygame.display.flip()
        pygame.display.set_caption("SO101 eval/teleop  |  MODE: %s%s" % (
            ctrl.mode.upper(), "  [REC]" if rec["w"] else ""))
        # record the composited window at ~20 fps
        if rec["w"] is not None and time.time() - rec["last"] >= 0.05:
            try:
                rec["w"].append_data(np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2)))
                rec["last"] = time.time()
            except Exception:
                pass
        clock.tick(60)

    if rec["w"] is not None:
        try: rec["w"].close(); print("REC SAVED", rec["path"], flush=True)
        except Exception: pass
    ctrl.running = False; time.sleep(0.3); pygame.quit()


if __name__ == "__main__":
    main()
