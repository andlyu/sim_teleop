import argparse, threading, io, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np
from PIL import Image, ImageDraw
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(ap)
args, _ = ap.parse_known_args(); args.headless = True; args.enable_cameras = True
app = AppLauncher(args).app
import sim_evals.environments  # noqa
from isaaclab_tasks.utils import parse_env_cfg
import gymnasium as gym, torch
cfg = parse_env_cfg("DROID", device=args.device, num_envs=1, use_fabric=True); cfg.set_scene(1)
for _cn in ["external_cam", "wrist_cam"]:
    _c = getattr(cfg.scene, _cn); _c.width = 480; _c.height = 270
cfg.episode_length_s = 100000.0   # disable 30s auto-reset (interactive use)
env = gym.make("DROID", cfg=cfg)
obs, _ = env.reset(); obs, _ = env.reset()
u = env.unwrapped; robot = u.scene["robot"]; cam = u.scene["external_cam"]; wcam = u.scene["wrist_cam"]; dev = u.device
INSTRUCTION = "put the cube in the bowl"

jn = list(robot.data.joint_names); bn = list(robot.data.body_names)
arm_ids = [jn.index("panda_joint%d" % i) for i in range(1, 8)]
cand = [n for n in bn if any(k in n.lower() for k in ["hand", "base_link", "tool", "gripper"])]
ee_idx = bn.index(cand[0] if cand else bn[-1])

def ee_world():
    return robot.data.body_pos_w[0, ee_idx].detach().cpu().numpy().astype(float)

target = ee_world().copy()
held = set(); grip = {"v": 0.0}; mode = {"v": "teleop"}; frame = {"jpg": b""}; policy = {"c": None}
STEP = 0.012

def teleop_targets():
    if 'left' in held:  target[1] += STEP
    if 'right' in held: target[1] -= STEP
    if 'up' in held:    target[0] += STEP
    if 'down' in held:  target[0] -= STEP
    if 'r' in held:     target[2] += STEP
    if 'f' in held:     target[2] -= STEP
    J = robot.root_physx_view.get_jacobians()
    ji = ee_idx - 1 if J.shape[1] == len(bn) - 1 else ee_idx
    Jp = J[0, ji, :3, :][:, arm_ids].detach().cpu().numpy()
    dq = Jp.T @ np.linalg.solve(Jp @ Jp.T + 0.0064*np.eye(3), target - ee_world())
    q = robot.data.joint_pos[0, arm_ids].detach().cpu().numpy()
    return q + np.clip(dq, -0.06, 0.06)

def teleop_action():
    qt = teleop_targets(); a = torch.zeros((1, 8), device=dev)
    a[0, :7] = torch.tensor(qt, dtype=torch.float32, device=dev)
    a[0, 7] = 1.0 if grip["v"] > 0.5 else 0.0
    return a

HTML = b"""<html><head><style>body{background:#111;color:#ccc;font-family:sans-serif;text-align:center;margin:0}img{max-width:99vw}</style></head>
<body><div>T = toggle POLICY/TELEOP &mdash; Arrows move EE, R/F up/down, Space gripper</div>
<img src="/stream"><script>
const ks={'arrowleft':'left','arrowright':'right','arrowup':'up','arrowdown':'down','r':'r','f':'f'};
function snd(p){fetch(p)}
addEventListener('keydown',e=>{if(e.repeat)return;let k=ks[e.key.toLowerCase()];if(k){snd('/key?k='+k+'&d=1');e.preventDefault()}if(e.key===' '){snd('/grip');e.preventDefault()}if(e.key.toLowerCase()==='t'){snd('/mode');e.preventDefault()}});
addEventListener('keyup',e=>{let k=ks[e.key.toLowerCase()];if(k){snd('/key?k='+k+'&d=0');e.preventDefault()}});
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        p = urlparse(self.path)
        if p.path == '/':
            self.send_response(200); self.send_header('Content-Type','text/html'); self.end_headers(); self.wfile.write(HTML)
        elif p.path == '/key':
            q = parse_qs(p.query); k=q.get('k',[''])[0]; d=q.get('d',['0'])[0]
            (held.add(k) if d=='1' else held.discard(k)); self.send_response(204); self.end_headers()
        elif p.path == '/grip':
            grip["v"] = 0.0 if grip["v"] > 0.5 else 1.0; self.send_response(204); self.end_headers()
        elif p.path == '/mode':
            mode["v"] = "policy" if mode["v"] == "teleop" else "teleop"; self.send_response(204); self.end_headers()
        elif p.path == '/stream':
            self.send_response(200); self.send_header('Content-Type','multipart/x-mixed-replace; boundary=f'); self.end_headers()
            try:
                while True:
                    j = frame["jpg"]
                    if j: self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\n\r\n'+j+b'\r\n')
                    time.sleep(0.05)
            except Exception: pass
        else:
            self.send_response(404); self.end_headers()

threading.Thread(target=lambda: ThreadingHTTPServer(("0.0.0.0",8080),H).serve_forever(), daemon=True).start()
print("TELEOP_READY_8080", flush=True)

while app.is_running():
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
                target[:] = ee_world()   # keep teleop target synced for smooth handoff
            except Exception as e:
                print("POLICY_INFER_ERR", e, flush=True); mode["v"] = "teleop"; act = None
    if act is None:
        act = teleop_action()
    obs, _, _, _, _ = env.step(act)
    ext = cam.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)
    wr = wcam.data.output["rgb"][0].detach().cpu().numpy()[..., :3].astype(np.uint8)[::-1, ::-1]
    hh = min(ext.shape[0], wr.shape[0])
    def _fit(a): return np.asarray(Image.fromarray(a).resize((int(a.shape[1]*hh/a.shape[0]), hh)))
    combo = np.concatenate([_fit(ext), _fit(wr)], axis=1)
    img = Image.fromarray(combo)
    ImageDraw.Draw(img).text((10, 6), "MODE: " + mode["v"].upper() + "   (T toggles)", fill=(255, 230, 0))
    b = io.BytesIO(); img.save(b, format="JPEG", quality=55); frame["jpg"] = b.getvalue()
