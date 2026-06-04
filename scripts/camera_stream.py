"""Tiny MJPEG camera streamer for aiming the real-arm cameras live.

Opens one or more OpenCV camera indices and serves them as MJPEG streams over
localhost so you can watch the views in a browser while physically repositioning
the cameras. Not part of the policy loop — just a setup/aiming aid.

    .venv-lerobot/bin/python scripts/camera_stream.py            # cams 0,1 on :8000
    .venv-lerobot/bin/python scripts/camera_stream.py --indices 0,1 --port 8000
    .venv-lerobot/bin/python scripts/camera_stream.py --rotate 0:90  # rotate cam0 by 90 CW

Open http://localhost:8000 in a browser. Ctrl-C to stop.
"""
from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

# index -> latest BGR frame (numpy array), guarded by LOCK
LATEST: dict[int, object] = {}
LOCK = threading.Lock()
ROTATE: dict[int, int] = {}  # index -> degrees CW (0/90/180/270)
STREAM_W = 640  # downscale width for the browser stream (keeps it light)


def _rotate(frame, deg: int):
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def grab_loop(index: int):
    """Continuously read frames from one camera into LATEST."""
    cap = cv2.VideoCapture(index)
    # Match the model-input format (native 640x480 / 4:3) so the preview shows
    # exactly what the policy receives.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"[cam {index}] FAILED to open")
        return
    print(f"[cam {index}] streaming")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        frame = _rotate(frame, ROTATE.get(index, 0))
        h, w = frame.shape[:2]
        if w > STREAM_W:
            frame = cv2.resize(frame, (STREAM_W, int(h * STREAM_W / w)))
        with LOCK:
            LATEST[index] = frame


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request logging
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "":
            self._page()
        elif self.path.startswith("/stream/"):
            try:
                idx = int(self.path.rsplit("/", 1)[1])
            except ValueError:
                self.send_error(404)
                return
            self._stream(idx)
        else:
            self.send_error(404)

    def _page(self):
        imgs = "".join(
            f'<div class="cam"><div class="lbl">index {i}</div>'
            f'<img src="/stream/{i}"></div>'
            for i in sorted(LATEST)
        ) or "<p>No cameras opened.</p>"
        html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>camera stream</title><style>
body{{background:#111;color:#ddd;font-family:system-ui;margin:0;padding:16px}}
.wrap{{display:flex;flex-wrap:wrap;gap:16px}}
.cam{{background:#000;border:1px solid #333;border-radius:8px;overflow:hidden}}
.lbl{{padding:6px 10px;font-size:13px;background:#1c1c1c}}
img{{display:block;width:640px;max-width:90vw}}
</style></head><body><div class=wrap>{imgs}</div></body></html>"""
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, idx: int):
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()
        try:
            while True:
                with LOCK:
                    frame = LATEST.get(idx)
                if frame is None:
                    time.sleep(0.03)
                    continue
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg.tobytes())
                self.wfile.write(b"\r\n")
                time.sleep(0.04)  # ~25 fps cap
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser closed the tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", default="0,1", help="comma-separated camera indices")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument(
        "--rotate", default="", help="per-cam rotation, e.g. '0:90,1:0' (degrees CW)"
    )
    args = ap.parse_args()

    for pair in filter(None, args.rotate.split(",")):
        i, deg = pair.split(":")
        ROTATE[int(i)] = int(deg)

    indices = [int(x) for x in args.indices.split(",") if x.strip() != ""]
    for i in indices:
        threading.Thread(target=grab_loop, args=(i,), daemon=True).start()
    time.sleep(1.5)  # let cameras warm up before serving

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"open http://localhost:{args.port}  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")


if __name__ == "__main__":
    main()
