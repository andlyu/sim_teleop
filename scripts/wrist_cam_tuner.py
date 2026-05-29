"""Interactive wrist-camera tuner (local web UI).

Holds a live Genesis scene and serves a page with sliders for the wrist camera's
offset (x/y/z) and rotation (roll/pitch/yaw) relative to the camera_wrist link,
plus FOV. On every change it re-renders three views:

  - TOP  third-person, red sphere marking the camera position
  - SIDE third-person, red sphere marking the camera position
  - GRIPPER CAM (what the wrist camera actually sees)

and prints the resulting offset_T so a good pose can be baked into cameras.py.

Camera.fov is read-only after creation, so several wrist cams are pre-built at
discrete FOVs and the FOV slider snaps to the nearest one.

Run:
    .venv/bin/python scripts/wrist_cam_tuner.py
then open http://localhost:8777
"""

import sys
import io
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import numpy as np
import genesis as gs
from PIL import Image
from genesis.utils import geom as gu

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so101_scene as S  # noqa: E402

PORT = 8777
TP_RES = (360, 360)
WR_RES = (480, 360)
POSE = np.array([0.0, 1.0, -1.0, 0.5, 0.0, 0.2])
FOV_OPTIONS = list(range(40, 121, 10))

STATE = {}


def rpy_to_T(xyz, rpy):
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


def build():
    gs.init(backend=gs.cpu, logging_level="error")
    scene, robot, cube = S.build_scene(show_viewer=False)
    marker = scene.add_entity(
        gs.morphs.Sphere(radius=0.008, fixed=True, collision=False),
        surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
    )
    top = scene.add_camera(res=TP_RES, pos=(0.30, 0.0, 1.40), lookat=(0.30, 0, 0.80), fov=42, GUI=False)
    side = scene.add_camera(res=TP_RES, pos=(0.45, -0.45, 1.05), lookat=(0.30, 0, 0.86), fov=42, GUI=False)
    wrist = {f: scene.add_camera(res=WR_RES, fov=f, near=0.01, far=10.0, GUI=False) for f in FOV_OPTIONS}
    scene.build()
    dofs = S.setup_control(robot)
    target = POSE[: len(dofs)]
    for _ in range(150):
        robot.control_dofs_position(target, dofs)
        scene.step()
    STATE.update(scene=scene, robot=robot, cube=cube, marker=marker,
                 top=top, side=side, wrist=wrist, dofs=dofs, target=target,
                 link=robot.get_link("camera_wrist"))


def _png_b64(arr):
    im = Image.fromarray(arr[..., :3].astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _hold():
    """Hold the arm pose, step, and re-snap all wrist cams to the link."""
    st = STATE
    st["robot"].control_dofs_position(st["target"], st["dofs"])
    st["scene"].step()
    for cam in st["wrist"].values():
        cam.move_to_attach()


def render(params):
    st = STATE
    xyz = [params["x"], params["y"], params["z"]]
    rpy = [np.deg2rad(params["roll"]), np.deg2rad(params["pitch"]), np.deg2rad(params["yaw"])]
    offset_T = rpy_to_T(xyz, rpy)

    for cam in st["wrist"].values():
        cam.attach(st["link"], offset_T)
    _hold()

    # place red marker at the camera world position
    lp = np.asarray(st["link"].get_pos()).reshape(-1)
    lq = np.asarray(st["link"].get_quat()).reshape(-1)
    world_T = gu.trans_quat_to_T(lp, lq) @ offset_T
    st["marker"].set_pos(world_T[:3, 3])
    _hold()

    fov = min(FOV_OPTIONS, key=lambda f: abs(f - params["fov"]))
    wrist = st["wrist"][fov]

    return {
        "top": _png_b64(np.asarray(st["top"].render()[0])),
        "side": _png_b64(np.asarray(st["side"].render()[0])),
        "wrist": _png_b64(np.asarray(wrist.render()[0])),
        "fov": fov,
        "offset_local": np.round(offset_T, 6).tolist(),
    }


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Wrist Cam Tuner</title>
<style>
body{font-family:system-ui,sans-serif;margin:12px;background:#1a1a1a;color:#eee}
.views{display:flex;gap:8px;flex-wrap:wrap}
.views figure{margin:0}.views img{border:1px solid #444;background:#000}
.views figcaption{font-size:12px;color:#9cf;text-align:center}
.ctl{display:flex;align-items:center;gap:8px;margin:4px 0}
.ctl label{width:60px}.ctl input[type=range]{width:300px}
.ctl span{width:70px;text-align:right;font-variant-numeric:tabular-nums}
pre{background:#000;padding:8px;border:1px solid #333;font-size:11px;overflow:auto}
h3{margin:8px 0 4px}
</style></head><body>
<h2>Wrist Camera Tuner</h2>
<div class=views>
 <figure><img id=top width=360><figcaption>TOP (red = camera pos)</figcaption></figure>
 <figure><img id=side width=360><figcaption>SIDE (red = camera pos)</figcaption></figure>
 <figure><img id=wrist width=480><figcaption>GRIPPER CAM (what it sees)</figcaption></figure>
</div>
<h3>Offset from camera_wrist link</h3>
<div id=ctls></div>
<h3>offset_T (local) &nbsp; selected FOV: <span id=fovsel>-</span></h3>
<pre id=ot></pre>
<script>
const P={x:0,y:0,z:0,roll:20,pitch:0,yaw:180,fov:85};
const defs=[["x",-0.1,0.1,0.001],["y",-0.1,0.1,0.001],["z",-0.1,0.1,0.001],
 ["roll",-180,180,1],["pitch",-180,180,1],["yaw",-180,180,1],["fov",40,120,10]];
const c=document.getElementById('ctls');
defs.forEach(([k,mn,mx,st])=>{
 const row=document.createElement('div');row.className='ctl';
 row.innerHTML=`<label>${k}</label><input type=range min=${mn} max=${mx} step=${st} value=${P[k]}><span>${P[k]}</span>`;
 const inp=row.querySelector('input'),sp=row.querySelector('span');
 inp.oninput=()=>{P[k]=parseFloat(inp.value);sp.textContent=inp.value;go();};
 c.appendChild(row);
});
let busy=false,pending=false;
function go(){ if(busy){pending=true;return;} busy=true;
 const q=Object.entries(P).map(([k,v])=>k+'='+v).join('&');
 fetch('/render?'+q).then(r=>r.json()).then(d=>{
  top.src='data:image/png;base64,'+d.top;
  side.src='data:image/png;base64,'+d.side;
  wrist.src='data:image/png;base64,'+d.wrist;
  fovsel.textContent=d.fov;
  ot.textContent=JSON.stringify(d.offset_local);
  busy=false; if(pending){pending=false;go();}
 }).catch(e=>{busy=false;});
}
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
            self.wfile.write(PAGE.encode())
        elif u.path == "/render":
            q = parse_qs(u.query)
            params = {k: float(q[k][0]) for k in ("x", "y", "z", "roll", "pitch", "yaw", "fov")}
            body = json.dumps(render(params)).encode()
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
