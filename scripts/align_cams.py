"""Align the front + side cameras to the MolmoAct2 reference images (web UI).

The model card ships two sample inputs (assets/molmoact_ref/ref_top.png and
ref_side.png) showing the viewpoints MolmoAct2-SO100_101 was trained on. This
tool renders each of our two cameras side-by-side with its reference image and
gives full pose sliders (x/y/z offset from arm base + pan/tilt/roll + fov) so you
can drag until the rendered view matches the reference framing.

When matched, the printed pose block can be pasted into cameras.CAMERA_OFFSETS.

Run:
    .venv/bin/python scripts/align_cams.py
then open http://localhost:8779
"""

import io
import sys
import json
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import numpy as np
from PIL import Image
import genesis as gs

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import so101_scene as S
import cameras as C

PORT = 8779
RES = (480, 360)
REF = {
    "front": REPO / "assets" / "molmoact_ref" / "ref_top.png",
    "side": REPO / "assets" / "molmoact_ref" / "ref_side.png",
}

# Each camera: position offset from the arm base + aim (pan/tilt/roll) + fov.
# Seeded from the current cameras.py values; the UI lets you drag from here.
CAMS = {
    "front": dict(x=0.18, y=0.0, z=0.66, pan=0.0, tilt=-89.0, roll=0.0, fov=45.0),
    "side":  dict(x=0.41, y=-0.6, z=0.42, pan=-5.0, tilt=8.0, roll=2.0, fov=45.0),
}
RANGES = {
    "x": (-0.6, 0.8), "y": (-0.8, 0.8), "z": (0.0, 1.0),
    "pan": (-180, 180), "tilt": (-90, 90), "roll": (-180, 180), "fov": (20, 110),
}
KEYS = ["x", "y", "z", "pan", "tilt", "roll", "fov"]
FOV_OPTS = list(range(20, 111, 5))

# Resting pose (radians): arm folded down toward the table, gripper near the
# work surface — matches the lowered arm in the model-card reference images.
REST_POSE = np.array([0.0, 1.4, -1.4, 0.9, 0.0, 0.2])

STATE = {}


def build():
    gs.init(backend=gs.cpu, logging_level="error")
    scene, robot, cube = S.build_scene(show_viewer=False)
    arm_base = np.array([0.0, 0.0, S.TABLE_TOP_Z])
    # A camera per (name, fov-option), since fov is fixed at creation in Genesis.
    cams = {}
    for name in CAMS:
        cams[name] = {f: scene.add_camera(res=RES, fov=f, GUI=False) for f in FOV_OPTS}
    scene.build()
    dofs = S.setup_control(robot)
    # settle at a RESTING pose (arm curled down toward the table, like the
    # reference images) so the framing matches what the model card shows.
    # joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    target = np.clip(REST_POSE[:len(dofs)],
                     [S.joint_limits()[n][0] for n in S.JOINT_NAMES],
                     [S.joint_limits()[n][1] for n in S.JOINT_NAMES])
    lo = np.array([S.joint_limits()[n][0] for n in S.JOINT_NAMES])
    hi = np.array([S.joint_limits()[n][1] for n in S.JOINT_NAMES])
    for _ in range(150):
        robot.control_dofs_position(target, dofs)
        scene.step()
    STATE.update(scene=scene, robot=robot, dofs=dofs, lo=lo, hi=hi,
                 jtarget=target.copy(), cams=cams, arm_base=arm_base,
                 params={k: dict(v) for k, v in CAMS.items()})
    # preload reference images as base64
    STATE["ref_b64"] = {}
    for name, p in REF.items():
        STATE["ref_b64"][name] = base64.b64encode(p.read_bytes()).decode()


def _png_b64(arr):
    im = Image.fromarray(np.asarray(arr)[..., :3].astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_one(name, p):
    st = STATE
    pos = st["arm_base"] + np.array([p["x"], p["y"], p["z"]])
    # aim at the workspace center on the table, with pan/tilt/roll applied
    lookat0 = st["arm_base"] + np.array([0.18, 0.0, 0.02])
    lookat, up = C._aim(pos, lookat0, p["pan"], p["tilt"], p["roll"])
    fov = min(FOV_OPTS, key=lambda f: abs(f - p["fov"]))
    cam = st["cams"][name][fov]
    cam.set_pose(pos=tuple(pos), lookat=lookat, up=up)
    return _png_b64(cam.render()[0]), fov


def render(which, p):
    STATE["params"][which].update(p)
    img, fov = render_one(which, STATE["params"][which])
    return {"img": img, "fov": fov,
            "vals": {k: round(float(STATE["params"][which][k]), 4) for k in KEYS}}


def set_joints(jrad):
    """Drive the arm to new joint targets (radians) and re-settle."""
    st = STATE
    st["jtarget"][:] = np.clip(jrad, st["lo"], st["hi"])
    for _ in range(60):
        st["robot"].control_dofs_position(st["jtarget"], st["dofs"])
        st["scene"].step()
    # re-render both cameras at the new arm pose
    out = {}
    for name in CAMS:
        img, fov = render_one(name, st["params"][name])
        out[name] = {"img": img, "fov": fov}
    out["joints"] = [round(float(v), 4) for v in st["jtarget"]]
    return out


def _panel(name):
    p = CAMS[name]
    rows = []
    for k in KEYS:
        lo, hi = RANGES[k]
        step = 0.005 if k in ("x", "y", "z") else 1
        rows.append(
            f'<div class=ctl><label>{k}</label>'
            f'<input type=range data-cam={name} data-k={k} min={lo} max={hi} '
            f'step={step} value={p[k]}><span>{p[k]}</span></div>')
    return f"""
<div class=cam>
 <h3>{name.upper()} camera
   <button onclick="toggleOverlay('{name}')" id=ovbtn_{name}>Overlay</button>
 </h3>
 <div class=imgs>
   <figure><img id=ref_{name}><figcaption>REFERENCE (model card)</figcaption></figure>
   <figure><img id=ren_{name}><figcaption>OUR RENDER (fov <span id=fov_{name}>-</span>)</figcaption></figure>
   <figure class=stack id=ov_{name} style="display:none">
     <img id=ovref_{name} class=ovimg>
     <img id=ovren_{name} class="ovimg ovtop">
     <figcaption>OVERLAY (ref + render)</figcaption>
   </figure>
 </div>
 {''.join(rows)}
</div>"""


def page():
    return """<!doctype html><html><head><meta charset=utf-8><title>Align Cameras</title>
<style>
body{font-family:system-ui,sans-serif;margin:12px;background:#1a1a1a;color:#eee}
.cam{margin-bottom:20px;border-bottom:1px solid #333;padding-bottom:12px}
.imgs{display:flex;gap:10px}.imgs img{width:320px;border:1px solid #555;background:#000}
.imgs figcaption{font-size:12px;color:#9cf;text-align:center}
.ctl{display:flex;align-items:center;gap:8px;margin:3px 0}
.ctl label{width:80px}.ctl input[type=range]{width:260px}
.ctl span{width:60px;text-align:right;font-variant-numeric:tabular-nums}
h2{margin:4px 0}h3{margin:6px 0;color:#fc9}
button{background:#345;color:#eee;border:1px solid #678;padding:2px 8px;cursor:pointer}
.stack{position:relative}.ovimg{position:absolute;left:0;top:0}
.stack img:first-of-type{position:static}
.ovtop{opacity:0.5}
.joints{background:#222;padding:8px;margin-bottom:14px;border:1px solid #444}
</style></head><body>
<h2>Align cameras to MolmoAct2 reference images</h2>
<p>Drag until OUR RENDER matches the REFERENCE framing. Use <b>Overlay</b> to blend
ref+render. Move the arm with the joint sliders. Then copy the printed pose.</p>
<div class=joints>
 <h3>Arm joints (radians)</h3>
 <div id=jctls></div>
</div>
""" + _panel("front") + _panel("side") + """
<h3>Current poses (paste into cameras.CAMERA_OFFSETS)</h3>
<pre id=out style="background:#000;padding:8px;border:1px solid #333"></pre>
<script>
const cur={front:{},side:{}};
const JNAMES=""" + json.dumps(S.JOINT_NAMES) + """;
function setRender(name, b64, fov){
  document.getElementById('ren_'+name).src='data:image/png;base64,'+b64;
  document.getElementById('ovren_'+name).src='data:image/png;base64,'+b64;
  if(fov!==undefined) document.getElementById('fov_'+name).textContent=fov;
}
function refresh(name){
  const ins=[...document.querySelectorAll('input[data-cam='+name+']')];
  const q=ins.map(x=>x.dataset.k+'='+x.value).join('&');
  fetch('/render?cam='+name+'&'+q).then(r=>r.json()).then(d=>{
    setRender(name, d.img, d.fov);
    ins.forEach(x=>{x.nextElementSibling.textContent=d.vals[x.dataset.k];});
    cur[name]=d.vals; dump();
  });
}
function dump(){
  out.textContent='front: '+JSON.stringify(cur.front)+'\\nside:  '+JSON.stringify(cur.side);
}
function toggleOverlay(name){
  const ov=document.getElementById('ov_'+name);
  ov.style.display = ov.style.display==='none' ? 'inline-block' : 'none';
}
function setJoints(){
  const js=[...document.querySelectorAll('input[data-j]')];
  const q=js.map((x,i)=>'j'+i+'='+x.value).join('&');
  fetch('/joints?'+q).then(r=>r.json()).then(d=>{
    ['front','side'].forEach(n=>setRender(n, d[n].img, d[n].fov));
    js.forEach((x,i)=>{x.nextElementSibling.textContent=d.joints[i].toFixed(3);});
  });
}
// joint sliders, seeded from the settled rest pose
const JREST=""" + json.dumps([round(float(v), 4) for v in REST_POSE.tolist()]) + """;
const jc=document.getElementById('jctls');
JNAMES.forEach((n,i)=>{
  const v=(JREST[i]!==undefined?JREST[i]:0);
  const row=document.createElement('div'); row.className='ctl';
  row.innerHTML='<label>'+(i+1)+' '+n+'</label>'+
    '<input type=range data-j='+i+' min=-3.2 max=3.2 step=0.01 value='+v+'><span>'+v.toFixed(3)+'</span>';
  jc.appendChild(row);
});
document.querySelectorAll('input[data-j]').forEach(x=>x.oninput=setJoints);
// camera panels
['front','side'].forEach(name=>{
  document.querySelectorAll('input[data-cam='+name+']').forEach(x=>x.oninput=()=>refresh(name));
  fetch('/ref?cam='+name).then(r=>r.json()).then(d=>{
    document.getElementById('ref_'+name).src='data:image/png;base64,'+d.img;
    document.getElementById('ovref_'+name).src='data:image/png;base64,'+d.img;
  });
  refresh(name);
});
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            body = page().encode()
            ct = "text/html"
        elif u.path == "/ref":
            body = json.dumps({"img": STATE["ref_b64"][q["cam"][0]]}).encode()
            ct = "application/json"
        elif u.path == "/render":
            name = q["cam"][0]
            p = {k: float(q[k][0]) for k in KEYS if k in q}
            body = json.dumps(render(name, p)).encode()
            ct = "application/json"
        elif u.path == "/joints":
            n = len(STATE["dofs"])
            jr = [float(q[f"j{i}"][0]) for i in range(n)]
            body = json.dumps(set_joints(jr)).encode()
            ct = "application/json"
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.end_headers()
        self.wfile.write(body)


def main():
    print("Building scene (once)...")
    build()
    print(f"Ready -> open http://localhost:{PORT}")
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
