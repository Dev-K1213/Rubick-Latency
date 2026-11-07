#!/usr/bin/env python3
"""
Camera latency tester (glass-to-glass) for a Rubik Pi + USB UVC camera
(e.g. ELP OV9712 / H264 / fisheye module).

How it works
------------
1. /clock  shows a big on-screen "binary clock" that encodes the Pi's own
   wall-clock time (in ms) every display frame. Open it on any screen
   (a laptop, phone, or the Pi's own monitor). The page syncs its clock to
   the Pi with an NTP-style ping, so the screen does not need to be the Pi.
2. The camera looks at that screen.
3. A background thread continuously reads the camera and stamps every frame
   with the Pi's time at the moment the frame arrives.
4. "Take picture" grabs the next frame, decodes the time shown in the picture
   (= when the light left the screen / the picture was "taken"), and computes

       latency = time frame was received - time shown in the frame

Usage
-----
    pip install flask opencv-python-headless numpy
    python3 rubik_latency.py --device /dev/video2        # see: v4l2-ctl --list-devices
    # dashboard:  http://<pi-ip>:5000/
    # clock page: http://<pi-ip>:5000/clock   (point the camera at this)

    python3 rubik_latency.py --selftest                   # test the decoder, no camera needed
"""
import argparse
import statistics
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify

# --------------------------------------------------------------------------
# Marker geometry. MUST match the JavaScript in CLOCK_HTML.
# --------------------------------------------------------------------------
#
# The marker is a row of tall vertical bars (one bar per bit), so every scanline
# of the camera image contains the whole timestamp. That makes it immune to
# rolling-shutter skew between rows. The time is sent as a Gray-coded counter
# of UNIT_MS-millisecond ticks: if exposure blur smears the fastest-changing
# bit, the decoded time is off by only one tick instead of being garbage.
FW, FH = 640, 400          # outer size of the black frame (units)
BORDER = 24                # black frame thickness
PAD = 16                   # white gap between frame and bars
HEADER = [1, 1, 0]         # sync pattern left of the data (asymmetric vs TRAILER,
TRAILER = [1, 0, 1]        # so a 180-degree rotated marker is detected)
NBITS = 24
COLS = len(HEADER) + NBITS + len(TRAILER)
CELL_W = (FW - 2 * (BORDER + PAD)) / COLS
UNIT_MS = 8.0              # one tick = 8 ms (finer than a 60 Hz display refresh)
MOD = 2 ** NBITS           # counter wraps every 2^24 * 8 ms ~ 37 hours


def encode_cols(units):
    g = units & (MOD - 1)
    g ^= g >> 1
    return HEADER + [(g >> (NBITS - 1 - i)) & 1 for i in range(NBITS)] + TRAILER


def cols_to_units(cols):
    if cols[:len(HEADER)] != HEADER or cols[-len(TRAILER):] != TRAILER:
        return None
    g = 0
    for b in cols[len(HEADER):len(HEADER) + NBITS]:
        g = (g << 1) | b
    v = 0
    while g:                # Gray -> binary
        v ^= g
        g >>= 1
    return v


# --------------------------------------------------------------------------
# Marker detection / decoding
# --------------------------------------------------------------------------
def order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)  # tl, tr, br, bl


def read_cols(warp):
    black = float(warp[6:BORDER - 6, 100:FW - 100].mean())
    white = float(warp[BORDER + 4:BORDER + PAD - 4, 100:FW - 100].mean())
    if white - black < 40:
        return None
    thr = (black + white) / 2
    y0, y1 = FH // 2 - 24, FH // 2 + 24          # read a band at mid-height
    hw = CELL_W * 0.3
    cols = []
    for c in range(COLS):
        cx = BORDER + PAD + (c + 0.5) * CELL_W
        m = float(warp[y0:y1, int(cx - hw):int(cx + hw) + 1].mean())
        cols.append(1 if m < thr else 0)
    return cols


def decode_frame(frame_bgr):
    """Return (units, quad): units = displayed time in UNIT_MS ticks (mod 2^24), or (None, None)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:12]
    dst = np.array([[0, 0], [FW - 1, 0], [FW - 1, FH - 1], [0, FH - 1]], np.float32)
    for c in cnts:
        if cv2.contourArea(c) < 2000:
            break
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = order_points(approx)
        M = cv2.getPerspectiveTransform(quad, dst)
        warp = cv2.warpPerspective(gray, M, (FW, FH))
        for w in (warp, cv2.rotate(warp, cv2.ROTATE_180)):
            cols = read_cols(w)
            if cols is None:
                continue
            val = cols_to_units(cols)
            if val is not None:
                return val, quad
    return None, None


# --------------------------------------------------------------------------
# Camera: background thread that stamps each frame on arrival
# --------------------------------------------------------------------------
class Camera:
    def __init__(self, device, width, height, fps, fourcc):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {device!r}. Try: v4l2-ctl --list-devices")
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cond = threading.Condition()
        self.frame, self.t_recv, self.seq = None, 0.0, 0
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def info(self):
        f = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "fourcc": "".join(chr((f >> (8 * i)) & 255) for i in range(4)),
        }

    def _loop(self):
        while self.running:
            if not self.cap.grab():            # returns when the driver hands us a frame
                time.sleep(0.01)
                continue
            t = time.time()                    # <- "received by the Pi"
            ok, frame = self.cap.retrieve()    # decode (MJPG) happens after the stamp
            if not ok:
                continue
            with self.cond:
                self.frame, self.t_recv, self.seq = frame, t, self.seq + 1
                self.cond.notify_all()

    def latest(self):
        with self.cond:
            return self.frame, self.t_recv, self.seq

    def wait_new(self, last_seq, timeout=2.0):
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq > last_seq, timeout):
                return None
            return self.frame, self.t_recv, self.seq


# --------------------------------------------------------------------------
# Flask app
# --------------------------------------------------------------------------
app = Flask(__name__)
cam = None
state_lock = threading.Lock()
results = []
failures = 0
last_jpg = None


def summarize(vals):
    if not vals:
        return None
    s = sorted(vals)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "median": statistics.median(s),
        "min": s[0],
        "max": s[-1],
        "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
        "p95": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))],
    }


def annotate(frame, quad, text, ok):
    img = frame.copy()
    color = (0, 200, 0) if ok else (0, 0, 255)
    if quad is not None:
        cv2.polylines(img, [quad.astype(np.int32)], True, color, 3)
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    _, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpg.tobytes()


@app.get("/")
def index():
    return Response(DASH_HTML, mimetype="text/html")


@app.get("/clock")
def clock():
    return Response(CLOCK_HTML, mimetype="text/html")


@app.get("/api/time")
def api_time():
    return jsonify(t=time.time() * 1000.0)


@app.get("/api/info")
def api_info():
    return jsonify(cam.info())


@app.post("/api/measure")
def api_measure():
    global failures, last_jpg
    _, _, seq0 = cam.latest()
    got = cam.wait_new(seq0, timeout=3.0)           # take the NEXT frame that arrives
    if got is None:
        return jsonify(ok=False, error="Timed out waiting for a new camera frame."), 504
    frame, t_recv, seq = got
    recv_ms = t_recv * 1000.0

    val, quad = decode_frame(frame)
    if val is None:
        with state_lock:
            failures += 1
            last_jpg = annotate(frame, None, "NO CLOCK PATTERN FOUND", False)
        return jsonify(ok=False, error="Couldn't find the clock pattern. Open /clock on a screen "
                                       "in front of the camera (marker should fill ~1/3 of the image, "
                                       "in focus, upright).")

    span = MOD * UNIT_MS
    lat = (recv_ms - (val + 0.5) * UNIT_MS) % span     # +0.5: middle of the tick
    if lat > span / 2:
        lat -= span
    if lat < 0 or lat > 10000:
        with state_lock:
            failures += 1
            last_jpg = annotate(frame, quad, f"IMPLAUSIBLE: {lat:.0f} ms", False)
        return jsonify(ok=False, error=f"Implausible latency ({lat:.0f} ms): clock sync problem or a "
                                       "stale /clock page. Reload /clock and try again.")

    with state_lock:
        results.append({"n": len(results) + 1, "time": time.strftime("%H:%M:%S"),
                        "latency_ms": round(lat, 1), "seq": seq})
        last_jpg = annotate(frame, quad, f"latency {lat:.1f} ms", True)
    return jsonify(ok=True, latency_ms=round(lat, 1), seq=seq)


@app.get("/api/results")
def api_results():
    with state_lock:
        vals = [r["latency_ms"] for r in results]
        return jsonify(results=results[-200:], stats=summarize(vals), failures=failures)


@app.post("/api/reset")
def api_reset():
    global failures
    with state_lock:
        results.clear()
        failures = 0
    return jsonify(ok=True)


@app.get("/api/last.jpg")
def api_last():
    with state_lock:
        data = last_jpg
    if data is None:
        return Response(status=404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/stream")
def stream():
    def gen():
        last = -1
        while True:
            got = cam.wait_new(last, 2.0)
            if got is None:
                continue
            frame, _, last = got
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (640, max(1, int(640 * h / w))))
            ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
DASH_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camera Latency Tester</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
 header{padding:14px 20px;background:#1b1b1b;border-bottom:1px solid #333}
 h1{margin:0;font-size:18px}.sub{color:#999;font-size:13px;margin-top:4px}
 main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px;padding:16px}
 .card{background:#1b1b1b;border:1px solid #333;border-radius:10px;padding:14px}
 .card h2{margin:0 0 10px;font-size:13px;color:#aaa;text-transform:uppercase;letter-spacing:.05em}
 img{width:100%;border-radius:6px;background:#000;display:block}
 button{background:#2d6cdf;color:#fff;border:0;border-radius:6px;padding:9px 14px;font-size:14px;cursor:pointer;margin:0 6px 6px 0}
 button.sec{background:#333}button:disabled{opacity:.5;cursor:default}
 input{width:60px;padding:8px;border-radius:6px;border:1px solid #444;background:#111;color:#eee}
 .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:10px 0}
 .stat{background:#111;border-radius:6px;padding:8px;text-align:center}
 .stat b{display:block;font-size:20px}.stat span{font-size:11px;color:#999}
 table{width:100%;border-collapse:collapse;font-size:13px}
 td,th{padding:4px 6px;text-align:left;border-bottom:1px solid #2a2a2a}
 #msg{min-height:20px;font-size:13px;color:#f0b34a;margin-top:6px}
 a{color:#6aa6ff}
</style></head><body>
<header><h1>Camera Latency Tester</h1>
<div class="sub">Camera: __INFO__ &middot; Point it at <a href="/clock" target="_blank">/clock</a> (open on a screen in front of the camera)</div></header>
<main>
 <div class="card"><h2>Live preview</h2><img src="/stream" alt="preview"></div>
 <div class="card"><h2>Last picture used</h2><img id="snap" alt="no measurement yet"></div>
 <div class="card"><h2>Measure</h2>
  <button id="one">Take 1 picture</button>
  <span><input id="n" type="number" value="20" min="1" max="500"> <button id="many">Run N pictures</button></span>
  <button class="sec" id="reset">Reset</button>
  <div id="msg"></div>
  <div class="stats" id="stats"></div>
 </div>
 <div class="card"><h2>Results (latest first)</h2>
  <table><thead><tr><th>#</th><th>Time</th><th>Latency (ms)</th></tr></thead><tbody id="rows"></tbody></table>
 </div>
</main>
<script>
const $=id=>document.getElementById(id);
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
let busy=false;
function setBusy(b){busy=b;['one','many','reset'].forEach(i=>$(i).disabled=b);}
async function refresh(){
  const j=await (await fetch('/api/results')).json();
  const s=j.stats, f=x=>x.toFixed(1);
  $('stats').innerHTML = s ? [['mean',s.mean],['median',s.median],['min',s.min],['max',s.max],
     ['p95',s.p95],['stdev',s.stdev]].map(([k,v])=>`<div class="stat"><b>${f(v)}</b><span>${k} ms</span></div>`).join('')
     + `<div class="stat"><b>${s.n}</b><span>samples</span></div><div class="stat"><b>${j.failures}</b><span>failed</span></div>`
     : '<div style="color:#888">No samples yet.</div>';
  $('rows').innerHTML=j.results.slice(-15).reverse().map(r=>`<tr><td>${r.n}</td><td>${r.time}</td><td>${r.latency_ms}</td></tr>`).join('');
}
async function once(){
  const r=await fetch('/api/measure',{method:'POST'}); const j=await r.json();
  $('snap').src='/api/last.jpg?'+Date.now();
  $('msg').textContent = j.ok ? '' : j.error;
  await refresh(); return j.ok;
}
$('one').onclick=async()=>{setBusy(true);try{await once();}finally{setBusy(false);}};
$('many').onclick=async()=>{
  setBusy(true);
  try{const n=Math.max(1,Math.min(500,+$('n').value||20));
    for(let i=0;i<n;i++){ $('msg').textContent=`Taking picture ${i+1}/${n}...`;
      const ok=await once(); if(!ok){ await sleep(500);} await sleep(200);} 
    $('msg').textContent='';
  }finally{setBusy(false);}
};
$('reset').onclick=async()=>{await fetch('/api/reset',{method:'POST'});$('msg').textContent='';await refresh();};
refresh();
</script></body></html>"""

CLOCK_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Latency clock</title>
<style>
 html,body{margin:0;background:#fff;color:#000;font-family:system-ui,sans-serif;height:100%}
 body{display:flex;flex-direction:column;align-items:center;justify-content:center}
 canvas{width:min(92vw,120vh);height:auto;display:block}
 #t{font-size:4vmin;font-variant-numeric:tabular-nums;margin-top:1vmin}
 #s{font-size:2vmin;color:#555;margin-top:.5vmin}
</style></head><body>
<canvas id="c" width="640" height="400"></canvas>
<div id="t">syncing...</div><div id="s"></div>
<script>
const FW=640,FH=400,BORDER=24,PAD=16,NBITS=24,UNIT=8;
const HEADER=[1,1,0],TRAILER=[1,0,1],COLS=3+NBITS+3;
const CW=(FW-2*(BORDER+PAD))/COLS;
const ctx=document.getElementById('c').getContext('2d');
const nowMs=()=>performance.timeOrigin+performance.now();
let offset=0, rtt=null;   // offset = server_ms - browser_ms

async function sync(){
  let best=null;
  for(let i=0;i<20;i++){
    const t0=nowMs();
    try{
      const j=await (await fetch('/api/time',{cache:'no-store'})).json();
      const t1=nowMs(), r=t1-t0;
      if(!best||r<best.r) best={r, off:j.t-(t0+t1)/2};
    }catch(e){}
    await new Promise(r=>setTimeout(r,40));
  }
  if(best){offset=best.off; rtt=best.r;
    document.getElementById('s').textContent=
      `clock offset ${best.off.toFixed(1)} ms, sync RTT ${best.r.toFixed(1)} ms (error about \u00b1${(best.r/2).toFixed(1)} ms). `+
      `Opening this page on the Pi itself gives the smallest error.`;}
}
function draw(){
  const t=Math.floor(nowMs()+offset);
  const u=Math.floor(t/UNIT)%(1<<NBITS), g=(u^(u>>>1))>>>0;   // Gray code
  const cols=HEADER.slice();
  for(let i=0;i<NBITS;i++) cols.push((g>>>(NBITS-1-i))&1);
  cols.push(...TRAILER);
  ctx.fillStyle='#000'; ctx.fillRect(0,0,FW,FH);
  ctx.fillStyle='#fff'; ctx.fillRect(BORDER,BORDER,FW-2*BORDER,FH-2*BORDER);
  ctx.fillStyle='#000';
  for(let c=0;c<COLS;c++)
    if(cols[c]) ctx.fillRect(Math.round(BORDER+PAD+c*CW),BORDER+PAD,Math.ceil(CW),FH-2*(BORDER+PAD));
  document.getElementById('t').textContent=(t%100000/1000).toFixed(3)+' s';
  requestAnimationFrame(draw);
}
sync().then(()=>{});
setInterval(sync,30000);
requestAnimationFrame(draw);
document.body.addEventListener('dblclick',()=>{document.documentElement.requestFullscreen&&document.documentElement.requestFullscreen();});
</script></body></html>"""


# --------------------------------------------------------------------------
# Self-test (no camera needed)
# --------------------------------------------------------------------------
def render_marker(units, blur_lsb=False):
    ox = oy = 80
    img = np.full((FH + 160, FW + 160, 3), 255, np.uint8)
    cv2.rectangle(img, (ox, oy), (ox + FW - 1, oy + FH - 1), (0, 0, 0), -1)
    cv2.rectangle(img, (ox + BORDER, oy + BORDER), (ox + FW - 1 - BORDER, oy + FH - 1 - BORDER),
                  (255, 255, 255), -1)
    cols = encode_cols(units)
    for c, bit in enumerate(cols):
        x0 = int(round(ox + BORDER + PAD + c * CELL_W))
        x1 = x0 + int(np.ceil(CELL_W)) - 1
        y0, y1 = oy + BORDER + PAD, oy + FH - 1 - BORDER - PAD
        if blur_lsb and c == COLS - len(TRAILER) - 1:      # fastest bit smeared to mid-gray
            cv2.rectangle(img, (x0, y0), (x1, y1), (128, 128, 128), -1)
        elif bit:
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)
    return img


def selftest(n=25):
    rng = np.random.default_rng(1)
    passed = 0
    for i in range(n):
        val = int(rng.integers(0, MOD))
        blur = i % 2 == 1                                   # odd cases: smeared LSB
        img = render_marker(val, blur)
        h, w = img.shape[:2]
        canvas = np.zeros((h + 200, w + 200, 3), np.uint8)          # dark monitor bezel
        canvas[100:100 + h, 100:100 + w] = img
        ch, cw = canvas.shape[:2]
        src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
        base = np.float32([[300, 100], [980, 100], [980, 620], [300, 620]])
        dst = base + rng.uniform(-50, 50, (4, 2)).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(canvas, M, (1280, 720), borderValue=(40, 40, 40))
        out = cv2.GaussianBlur(out, (5, 5), 0)
        out = np.clip(out.astype(np.int16) + rng.normal(0, 6, out.shape), 0, 255).astype(np.uint8)
        got, _ = decode_frame(out)
        good = got is not None and (abs(got - val) <= 1 if blur else got == val)
        passed += good
        print(f"[{'ok' if good else 'FAIL'}] {val} -> {got}")
    print(f"{passed}/{n} decoded")
    return 0 if passed == n else 1


# --------------------------------------------------------------------------
def main():
    global cam
    ap = argparse.ArgumentParser(description="Camera glass-to-glass latency tester")
    ap.add_argument("--device", default="0", help="video index (0) or path (/dev/video2)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fourcc", default="MJPG", help="MJPG (default) or YUYV; '' to leave unset")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--selftest", action="store_true", help="test the marker decoder and exit")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    device = int(args.device) if args.device.isdigit() else args.device
    cam = Camera(device, args.width, args.height, args.fps, args.fourcc)
    info = cam.info()
    print("Camera opened:", info)
    global DASH_HTML
    DASH_HTML = DASH_HTML.replace(
        "__INFO__", f"{info['width']}x{info['height']} @ {info['fps']:.0f} fps, {info['fourcc']}")
    print(f"Dashboard: http://{args.host}:{args.port}/   Clock page: http://<pi-ip>:{args.port}/clock")
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()