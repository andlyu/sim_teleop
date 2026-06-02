import argparse, threading, io, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np
from PIL import Image
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(ap)
args, _ = ap.parse_known_args(); args.headless = True; args.enable_cameras = True
app = AppLauncher(args).app
import sim_evals.environments  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym, torch
cfg = parse_env_cfg("DROID", device=args.device, num_envs=1, use_fabric=True); cfg.set_scene(1)
for _cn in ["external_cam", "wrist_cam"]:
    _c = getattr(cfg.scene, _cn); _c.width = 640; _c.height = 360   # crisper 224 for the policy, modest render cost
cfg.episode_length_s = 100000.0   # disable 30s auto-reset (interactive use)
env = gym.make("DROID", cfg=cfg)
obs, _ = env.reset(); obs, _ = env.reset()
u = env.unwrapped; robot = u.scene["robot"]; cam = u.scene["external_cam"]; wcam = u.scene["wrist_cam"]; dev = u.device
INSTRUCTION = "put the rubik's cube in the red bowl"

jn = list(robot.data.joint_names); bn = list(robot.data.body_names)
arm_ids = [jn.index("panda_joint%d" % i) for i in range(1, 8)]
cand = [n for n in bn if any(k in n.lower() for k in ["hand", "base_link", "tool", "gripper"])]
ee_idx = bn.index(cand[0] if cand else bn[-1])

def ee_world():
    return robot.data.body_pos_w[0, ee_idx].detach().cpu().numpy().astype(float)

def ee_quat():
    return robot.data.body_quat_w[0, ee_idx].detach().cpu().numpy().astype(float)  # (w,x,y,z)

def quat_err(q_des, q_cur):
    """World-frame rotation vector (axis*angle) that rotates current EE orientation to desired."""
    w1, x1, y1, z1 = q_des
    w2, x2, y2, z2 = q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]   # conjugate of current
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    if w < 0: w, x, y, z = -w, -x, -y, -z          # shortest path
    n = (x*x + y*y + z*z) ** 0.5
    if n < 1e-8: return np.zeros(3)
    return (2.0 * np.arctan2(n, w) / n) * np.array([x, y, z])

# Lock the EE to the home (downward-facing) orientation during teleop.
DOWN_QUAT = ee_quat().copy()

MOVE_KEYS = {'left', 'right', 'up', 'down', 'r', 'f'}
target = ee_world().copy()
held = set(); grip = {"v": 0.0}; mode = {"v": "teleop"}; frame = {"jpg": b"", "id": 0}; policy = {"c": None}; reset_req = {"v": False}
# per-loop timing (EMA, milliseconds) to find what makes a frame slow
timing = {"step": 0.0, "read": 0.0, "enc": 0.0, "infer": 0.0, "loop": 0.0, "fps": 0.0, "n": 0}
def _ema(k, v): timing[k] = v if timing["n"] < 3 else 0.9 * timing[k] + 0.1 * v
STEP = 0.01   # EE move increment per held step (smaller = finer/slower, easier to control)
arm_q = {"v": robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().copy()}  # cached arm target

def sync_teleop():
    """Resync the teleop target to the arm's current pose (on mode switch / reset)."""
    target[:] = ee_world()
    arm_q["v"] = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy().copy()

def teleop_targets():
    # Only the heavy Jacobian IK runs here — called only when a move key is held.
    if 'left' in held:  target[1] += STEP
    if 'right' in held: target[1] -= STEP
    if 'up' in held:    target[0] += STEP
    if 'down' in held:  target[0] -= STEP
    if 'r' in held:     target[2] += STEP
    if 'f' in held:     target[2] -= STEP
    J = robot.root_physx_view.get_jacobians()
    ji = ee_idx - 1 if J.shape[1] == len(bn) - 1 else ee_idx
    Jf = J[0, ji, :6, :][:, arm_ids].detach().cpu().numpy()      # full 6xN: linear(3)+angular(3)
    pos_err = target - ee_world()
    ori_err = quat_err(DOWN_QUAT, ee_quat())                     # hold downward orientation
    err = np.concatenate([pos_err, ori_err])
    dq = Jf.T @ np.linalg.solve(Jf @ Jf.T + 0.01*np.eye(6), err)
    q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
    return q + np.clip(dq, -0.15, 0.15)

def teleop_action():
    if held & MOVE_KEYS:                 # only recompute IK while actively moving
        arm_q["v"] = teleop_targets()
    a = torch.zeros((1, 8), device=dev)
    a[0, :7] = torch.tensor(arm_q["v"], dtype=torch.float32, device=dev)
    a[0, 7] = 1.0 if grip["v"] > 0.5 else 0.0
    return a

HTML = b"""<html><head><style>body{background:#111;color:#ccc;font-family:sans-serif;text-align:center;margin:0}img{max-width:99vw}</style></head>
<body><div>T = toggle POLICY/TELEOP &mdash; G = reset scene &mdash; Arrows move EE, R/F up/down, Space gripper</div>
<img src="/stream"><script>
const ks={'arrowleft':'left','arrowright':'right','arrowup':'up','arrowdown':'down','r':'r','f':'f'};
function snd(p){fetch(p)}
addEventListener('keydown',e=>{if(e.repeat)return;let k=ks[e.key.toLowerCase()];if(k){snd('/key?k='+k+'&d=1');e.preventDefault()}if(e.key===' '){snd('/grip');e.preventDefault()}if(e.key.toLowerCase()==='t'){snd('/mode');e.preventDefault()}if(e.key.toLowerCase()==='g'){snd('/reset');e.preventDefault()}});
addEventListener('keyup',e=>{let k=ks[e.key.toLowerCase()];if(k){snd('/key?k='+k+'&d=0');e.preventDefault()}});
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # keep-alive: native client reuses one connection (low-latency commands)
    def setup(self):
        super().setup()
        try:
            import socket as _s
            self.connection.setsockopt(_s.IPPROTO_TCP, _s.TCP_NODELAY, 1)   # no Nagle -> no extra RTT
        except Exception:
            pass
    def log_message(self, *a): pass
    def _no_content(self):
        self.send_response(204); self.send_header('Content-Length', '0'); self.end_headers()
    def do_GET(self):
        p = urlparse(self.path)
        if p.path == '/':
            self.send_response(200); self.send_header('Content-Type','text/html'); self.send_header('Content-Length', str(len(HTML))); self.end_headers(); self.wfile.write(HTML)
        elif p.path == '/key':
            q = parse_qs(p.query); k=q.get('k',[''])[0]; d=q.get('d',['0'])[0]
            (held.add(k) if d=='1' else held.discard(k)); self._no_content()
        elif p.path == '/grip':
            grip["v"] = 0.0 if grip["v"] > 0.5 else 1.0; self._no_content()
        elif p.path == '/mode':
            mode["v"] = "policy" if mode["v"] == "teleop" else "teleop"; self._no_content()
        elif p.path == '/reset':
            reset_req["v"] = True; self._no_content()
        elif p.path == '/state':
            body = b'{"mode":"%s"}' % mode["v"].encode()
            self.send_response(200); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        elif p.path == '/stream':
            self.send_response(200); self.send_header('Content-Type','multipart/x-mixed-replace; boundary=f'); self.end_headers()
            try:
                while True:
                    j = frame["jpg"]
                    if j: self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n\r\n'+j+b'\r\n')
                    time.sleep(0.05)
            except Exception: pass
        elif p.path == '/timing':
            body = ('{"step_ms":%.1f,"read_ms":%.1f,"enc_ms":%.1f,"infer_ms":%.1f,"loop_ms":%.1f,"fps":%.1f}'
                    % (timing["step"], timing["read"], timing["enc"], timing["infer"], timing["loop"], timing["fps"])).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            self.send_response(404); self.send_header('Content-Length', '0'); self.end_headers()

threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0",8080),H).serve_forever(), daemon=True).start()
print("TELEOP_READY_8080", flush=True)

prev_mode = mode["v"]
while app.is_running():
    if reset_req["v"]:
        reset_req["v"] = False
        held.clear()                                       # drop any stuck movement keys
        obs, _ = env.reset()
        hq = robot.data.default_joint_pos.clone()          # robot's home configuration
        robot.write_joint_state_to_sim(hq, torch.zeros_like(robot.data.default_joint_vel))
        arm_q["v"] = hq[0, arm_ids].detach().cpu().numpy().copy()   # hold arm at home
        target[:] = ee_world()
        grip["v"] = 0.0; mode["v"] = "teleop"
        if policy["c"] is not None:
            try: policy["c"].reset()
            except Exception: pass
        print("SCENE_RESET -> HOME", flush=True)
    # On entering policy mode, drop any buffered action chunk so the policy
    # re-plans from the CURRENT arm pose (no jump/discontinuity on handoff).
    if mode["v"] == "policy" and prev_mode != "policy" and policy["c"] is not None:
        try: policy["c"].reset(); print("POLICY_REPLAN_FROM_CURRENT", flush=True)
        except Exception: pass
    # On returning to teleop, start from where the arm currently is.
    if mode["v"] == "teleop" and prev_mode != "teleop":
        sync_teleop()
    prev_mode = mode["v"]
    t0 = time.perf_counter()
    act = None
    if mode["v"] == "policy":
        if policy["c"] is None:
            try:
                from sim_evals.inference.droid_jointpos import Client
                policy["c"] = Client(); policy["c"].reset(); print("POLICY_CONNECTED", flush=True)
            except Exception as e:
                print("POLICY_CONNECT_FAILED", e, flush=True); mode["v"] = "teleop"
        if policy["c"] is not None:
            try:
                ret = policy["c"].infer(obs, INSTRUCTION)
                act = torch.tensor(np.asarray(ret["action"], dtype=np.float32), device=dev).unsqueeze(0)
                sync_teleop()            # keep teleop target synced for smooth handoff
            except Exception as e:
                print("POLICY_INFER_ERR", e, flush=True); mode["v"] = "teleop"; act = None
    if act is None:
        act = teleop_action()
    t1 = time.perf_counter()
    obs, _, _, _, _ = env.step(act)
    t2 = time.perf_counter()
    ext = cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
    wr = wcam.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)[::-1, ::-1]
    t3 = time.perf_counter()
    hh = min(360, ext.shape[0], wr.shape[0])   # cap streamed view height; policy still gets full 720p
    def _fit(a): return np.asarray(Image.fromarray(a).resize((int(a.shape[1]*hh/a.shape[0]), hh)))
    combo = np.concatenate([_fit(ext), _fit(wr)], axis=1)
    img = Image.fromarray(combo)
    b = io.BytesIO(); img.save(b, format="JPEG", quality=55); frame["jpg"] = b.getvalue()
    t4 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000 if mode["v"] == "policy" else 0.0
    step_ms = (t2 - t1) * 1000; read_ms = (t3 - t2) * 1000; enc_ms = (t4 - t3) * 1000
    loop_ms = (t4 - t0) * 1000
    _ema("infer", infer_ms); _ema("step", step_ms); _ema("read", read_ms); _ema("enc", enc_ms)
    _ema("loop", loop_ms); _ema("fps", 1000.0 / loop_ms if loop_ms > 0 else 0.0)
    timing["n"] += 1
    if loop_ms > 120:   # log slow frames with the breakdown
        print("SLOW %.0fms (step=%.0f read=%.0f enc=%.0f infer=%.0f) mode=%s"
              % (loop_ms, step_ms, read_ms, enc_ms, infer_ms, mode["v"]), flush=True)
