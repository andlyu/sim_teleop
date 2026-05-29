"""Move the arm in a web UI; watch the gripper + side (+ top) cameras live.

The inverse of wrist_cam_tuner: there the camera moved and the arm was fixed.
Here the ARM moves (6 joint sliders) and the cameras are fixed, so you can see
exactly what each camera will observe as the arm is teleoperated.

Views rendered on every change:
  - TOP    third-person overview
  - SIDE   the external side camera the policy uses (from cameras.py); its
           POSITION (dx/dy/dz from the arm base) AND ORIENTATION (pan/tilt/roll)
           are adjustable live.
  - GRIPPER CAM  the wrist-mounted camera (what the arm "sees")

The wrist camera uses the tuned offset (roll 20, yaw 180) baked in below.

Run:
    .venv/bin/python scripts/arm_cam_view.py
then open http://localhost:8778
"""

import io
import sys
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image
import genesis as gs

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import so101_scene as S
import cameras as C

PORT = 8778
TP_RES = (360, 360)
WR_RES = (480, 360)

# Tuned wrist-cam offset from the web UI (degrees): roll 20, pitch 0, yaw 180.
WRIST_RPY_DEG = (20.0, 0.0, 180.0)
WRIST_XYZ = (0.0, 0.0, 0.0)
WRIST_FOV = 85

# Start pose (radians), clipped to limits at build.
START_POSE = np.array([0.0, 0.5, -0.5, 0.5, 0.0, 0.2])

# Side cam: POSITION is an offset from the arm base (on the table); ORIENTATION
# is pan/tilt/roll (deg) applied on top of "look at the workspace". All six are
# adjustable live in the UI. pan = swivel left/right, tilt = up/down, roll =
# rotate the image. (0,0,0) reproduces a level cam aimed at the workspace.
SIDE_OFFSET = np.array([0.410, -0.600, 0.390])  # tuned in UI (dx, dy, dz)
SIDE_RPY = np.array([0.0, 0.0, 0.0])            # (pan, tilt, roll) deg
SIDE_RANGES = {
    "dx": (-0.4, 0.6), "dy": (-0.6, 0.6), "dz": (0.0, 0.8),
    "pan": (-90.0, 90.0), "tilt": (-90.0, 90.0), "roll": (-180.0, 180.0),
}
SIDE_KEYS = ["dx", "dy", "dz", "pan", "tilt", "roll"]


def _rpy_to_T(xyz, rpy):
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz
    return T


def _rot(axis, ang, v):
    """Rodrigues rotation of vector v about unit-ish axis by ang radians."""
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return v
    axis = axis / n
    return (v * np.cos(ang)
            + np.cross(axis, v) * np.sin(ang)
            + axis * np.dot(axis, v) * (1 - np.cos(ang)))


def _aim(pos, base_lookat, pan_deg, tilt_deg, roll_deg):
    """Return (lookat, up) for a cam at `pos` aimed at `base_lookat`, with
    pan (yaw about world z), tilt (pitch about the cam's right axis), and roll
    (about the view axis) applied. (0,0,0) -> look straight at base_lookat, up=z.
    """
    pos = np.asarray(pos, float)
    d = np.asarray(base_lookat, float) - pos
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        d = np.array([1.0, 0.0, 0.0])
        dist = 1.0
    fwd = d / dist
    wup = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, wup)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)

    pan, tilt, roll = np.deg2rad([pan_deg, tilt_deg, roll_deg])
    # pan about world up, tilt about the (panned) right axis
    fwd = _rot(wup, pan, fwd)
    right = _rot(wup, pan, right)
    fwd = _rot(right, tilt, fwd)
    up = np.cross(right, fwd)
    up = _rot(fwd, roll, up)

    lookat = pos + dist * fwd
    return tuple(lookat), tuple(up)


STATE = {}


def build():
    gs.init(backend=gs.cpu, logging_level="error")
    scene, robot, cube = S.build_scene(show_viewer=False)

    # Side cam anchored to the ARM base on the table (not the floor). Offsets are
    # relative to the base so framing survives the arm being remounted.
    arm_base = np.array([0.0, 0.0, S.TABLE_TOP_Z])
    side_lookat = arm_base + np.array([S.CUBE_XY[0], 0.0, 0.02])  # at the workspace
    side_pos = arm_base + SIDE_OFFSET
    lookat0, up0 = _aim(side_pos, side_lookat, *SIDE_RPY)

    top = scene.add_camera(res=TP_RES, pos=(0.30, 0.0, 1.40), lookat=(0.30, 0, 0.80), fov=42, GUI=False)
    side = scene.add_camera(res=TP_RES, pos=tuple(side_pos), lookat=lookat0, up=up0, fov=45, GUI=False)
    wrist = scene.add_camera(res=WR_RES, fov=WRIST_FOV, near=0.01, far=10.0, GUI=False)

    scene.build()
    dofs = S.setup_control(robot)

    # Attach wrist cam to the camera_wrist link with the tuned offset.
    link = robot.get_link(C.WRIST_MOUNT_LINK)
    offset_T = _rpy_to_T(WRIST_XYZ, np.deg2rad(WRIST_RPY_DEG))
    wrist.attach(link, offset_T)

    limits = S.joint_limits()
    lowers = np.array([limits[n][0] for n in S.JOINT_NAMES])
    uppers = np.array([limits[n][1] for n in S.JOINT_NAMES])
    target = np.clip(START_POSE[: len(dofs)], lowers, uppers)

    for _ in range(150):
        robot.control_dofs_position(target, dofs)
        scene.step()
    wrist.move_to_attach()

    STATE.update(scene=scene, robot=robot, top=top, side=side, wrist=wrist,
                 dofs=dofs, target=target.copy(), lowers=lowers, uppers=uppers,
                 arm_base=arm_base, side_lookat=side_lookat,
                 side_offset=SIDE_OFFSET.astype(float).copy(),
                 side_rpy=SIDE_RPY.astype(float).copy())


def _png_b64(arr):
    im = Image.fromarray(np.asarray(arr)[..., :3].astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render(target, side_offset=None, side_rpy=None):
    st = STATE
    st["target"][:] = np.clip(target, st["lowers"], st["uppers"])
    if side_offset is not None:
        st["side_offset"][:] = side_offset
    if side_rpy is not None:
        st["side_rpy"][:] = side_rpy
    if side_offset is not None or side_rpy is not None:
        pos = st["arm_base"] + st["side_offset"]
        lookat, up = _aim(pos, st["side_lookat"], *st["side_rpy"])
        st["side"].set_pose(pos=tuple(pos), lookat=lookat, up=up)

    # Step a few times so the PD controller tracks toward the new target.
    for _ in range(40):
        st["robot"].control_dofs_position(st["target"], st["dofs"])
        st["scene"].step()
    st["wrist"].move_to_attach()
    return {
        "top": _png_b64(st["top"].render()[0]),
        "side": _png_b64(st["side"].render()[0]),
        "wrist": _png_b64(st["wrist"].render()[0]),
        "pose": [round(float(v), 4) for v in st["target"]],
        "side_vals": (
            [round(float(v), 4) for v in st["side_offset"]]
            + [round(float(v), 4) for v in st["side_rpy"]]
        ),
    }


def _page():
    names = S.JOINT_NAMES
    lim = S.joint_limits()
    rows = []
    for i, n in enumerate(names):
        lo, hi = lim[n]
        v = float(STATE["target"][i])
        rows.append(
            f'<div class=ctl><label>{i+1} {n}</label>'
            f'<input type=range data-i={i} min={lo:.4f} max={hi:.4f} step=0.01 value={v:.4f}>'
            f'<span>{v:.3f}</span></div>'
        )
    ctrls = "\n".join(rows)

    # Side-cam sliders: position (dx/dy/dz, m) + orientation (pan/tilt/roll, deg).
    vals = list(STATE["side_offset"]) + list(STATE["side_rpy"])
    srows = []
    for k, v in zip(SIDE_KEYS, vals):
        lo, hi = SIDE_RANGES[k]
        step = 0.01 if k in ("dx", "dy", "dz") else 1.0
        srows.append(
            f'<div class=ctl><label>side {k}</label>'
            f'<input type=range data-s={k} min={lo} max={hi} step={step} value={v:.4f}>'
            f'<span>{v:.3f}</span></div>'
        )
    sctrls = "\n".join(srows)

    return """<!doctype html><html><head><meta charset=utf-8><title>Arm Cam View</title>
<style>
body{font-family:system-ui,sans-serif;margin:12px;background:#1a1a1a;color:#eee}
.views{display:flex;gap:8px;flex-wrap:wrap}
.views figure{margin:0}.views img{border:1px solid #444;background:#000}
.views figcaption{font-size:12px;color:#9cf;text-align:center}
.ctl{display:flex;align-items:center;gap:8px;margin:4px 0}
.ctl label{width:130px}.ctl input[type=range]{width:300px}
.ctl span{width:64px;text-align:right;font-variant-numeric:tabular-nums}
h3{margin:8px 0 4px}
</style></head><body>
<h2>Arm Camera View — move the arm, see what the cameras see</h2>
<div class=views>
 <figure><img id=top width=360><figcaption>TOP (overview)</figcaption></figure>
 <figure><img id=side width=360><figcaption>SIDE (policy cam)</figcaption></figure>
 <figure><img id=wrist width=480><figcaption>GRIPPER CAM (what it sees)</figcaption></figure>
</div>
<h3>Joint targets (radians)</h3>
""" + ctrls + """
<h3>Side camera — position (m) &amp; orientation (deg)</h3>
""" + sctrls + """
<script>
const jinputs=[...document.querySelectorAll('input[data-i]')];
const sinputs=[...document.querySelectorAll('input[data-s]')];
let busy=false,pending=false;
function go(){ if(busy){pending=true;return;} busy=true;
 const q=jinputs.map((x,i)=>'j'+i+'='+x.value)
   .concat(sinputs.map(x=>x.dataset.s+'='+x.value)).join('&');
 fetch('/render?'+q).then(r=>r.json()).then(d=>{
  top.src='data:image/png;base64,'+d.top;
  side.src='data:image/png;base64,'+d.side;
  wrist.src='data:image/png;base64,'+d.wrist;
  jinputs.forEach((x,i)=>{x.nextElementSibling.textContent=d.pose[i].toFixed(3);});
  sinputs.forEach((x,i)=>{x.nextElementSibling.textContent=d.side_vals[i].toFixed(3);});
  busy=false; if(pending){pending=false;go();}
 }).catch(e=>{busy=false;});
}
jinputs.concat(sinputs).forEach(x=>x.oninput=go);
go();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_page().encode())
        elif u.path == "/render":
            q = parse_qs(u.query)
            target = np.array([float(q[f"j{i}"][0]) for i in range(len(STATE["dofs"]))])
            side_offset = side_rpy = None
            if "dx" in q:
                side_offset = np.array([float(q[k][0]) for k in ("dx", "dy", "dz")])
            if "pan" in q:
                side_rpy = np.array([float(q[k][0]) for k in ("pan", "tilt", "roll")])
            body = json.dumps(render(target, side_offset, side_rpy)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def main():
    print("Building Genesis scene (once)...")
    build()
    print(f"Ready -> open http://localhost:{PORT}")
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
