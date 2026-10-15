"""
Web version of cup_pixel_calibration.py (Track B calibration capture), served
as an MJPEG stream + small JSON API + browser UI instead of a cv2.imshow
window - built for the same reason as web_click_to_shoot.py: cv2.imshow over
X11 forwarding was too laggy for real use on this network. A browser tab over
plain HTTP on the LAN is far lighter.

All hardware/vision/fit logic is unchanged and imported from pong_common.py,
web_click_to_shoot.py (CameraWorker/draw_chip/wait_fresh_frame - reused, not
copy-pasted a third time), and cup_pixel_calibration.py (load_existing_samples/
fit_and_save - same fit that CLI script uses, so the two can't drift apart).

Flow: PAN auto-locks to the saved center position and never moves again
(same as the CLI script - this calibration doesn't need PAN motion). Tap the
cup in the video feed once to calibrate its color. Then, for each measured
distance: place the cup, type the distance (in) to its FRONT edge (near
wall, at table level - matches the 'base' pixel, not the cup's center) into
the box, press Capture. Repeat across ~6-8 distances spanning the working
range. Press Save Calibration to fit the curve and write it to
pixel_range_calibration.json - safe to press more than once as you add more
samples, each press re-fits and overwrites with the current full set.

Run with: odrive_env/bin/python web_cup_pixel_calibration.py
Then open http://<this machine's LAN IP>:5001/ from another device on the
same network.
"""
import time

import cv2
from odrive.enums import AxisState
from flask import Flask, Response, jsonify, request

from pong_common import (
    CAM_INDEX, PIVOT_TO_BASE_IN, WATCHDOG_FEED_INTERVAL,
    ColorCalibration, WatchdogFeeder, connect_pan, detect_target, lock_pan, median_read,
)
from web_click_to_shoot import CameraWorker, draw_chip, wait_fresh_frame, COLOR_CUP, COLOR_TEXT
from cup_pixel_calibration import fit_and_save, load_existing_samples

HOST = "0.0.0.0"
PORT = 5001
STREAM_FPS = 20
JPEG_QUALITY = 75

TAPE_OFFSET_IN = PIVOT_TO_BASE_IN
OFFSET_LABEL = "launcher base"


def make_web_row_reader(cam, calib):
    """Single-target (largest blob) row reader for the capture button - no
    tracker state needed since, unlike click_to_shoot's live aiming, the
    cup here just sits still at the measured distance for each capture."""
    def read_row():
        frame = wait_fresh_frame(cam)
        if frame is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        det = detect_target(hsv, calib)
        return det["base"][1] if det is not None else None
    return read_row


COLOR_GOOD = (88, 209, 48)     # BGR - same green as the HTML/CSS --green var
CENTERED_DEADBAND_PX = 15       # matches the on-screen "(centered)" threshold


def make_render_fn(calib):
    def render(frame):
        h, w = frame.shape[:2]
        frame_cx = w // 2
        det = detect_target(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), calib) \
            if calib.ranges is not None else None

        for yy in range(0, h, 16):
            cv2.line(frame, (frame_cx, yy), (frame_cx, min(yy + 8, h)), (90, 90, 90), 1)

        if det is not None:
            x, y, bw, bh = det["bbox"]
            cx, cy = det["centroid"]
            h_offset = cx - frame_cx
            centered = abs(h_offset) < CENTERED_DEADBAND_PX
            offset_color = COLOR_GOOD if centered else COLOR_CUP

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_CUP, 2, cv2.LINE_AA)
            bx, by = det["base"]
            cv2.circle(frame, (bx, by), 6, COLOR_CUP, -1, cv2.LINE_AA)
            cv2.line(frame, (frame_cx, cy), (cx, cy), offset_color, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 5, offset_color, -1, cv2.LINE_AA)

            draw_chip(frame, f"base row: {by}px", 10, 10, color=COLOR_TEXT)
            centered_suffix = " (centered)" if centered else ""
            draw_chip(frame, f"h-offset: {h_offset:+d}px{centered_suffix}", 10, 44, color=offset_color)
        elif calib.ranges is None:
            draw_chip(frame, "tap the cup to calibrate color", 10, 10, color=COLOR_TEXT)
        else:
            draw_chip(frame, "no target detected", 10, 10, color=COLOR_TEXT)

        return frame

    return render


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
<title>Cup Calibration</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #000000;
    --sheet: rgba(28, 28, 30, 0.78);
    --fill: rgba(120, 120, 128, 0.24);
    --fill-strong: rgba(255, 255, 255, 0.14);
    --separator: rgba(84, 84, 88, 0.6);
    --label: #ffffff;
    --label-2: rgba(235, 235, 245, 0.6);
    --label-3: rgba(235, 235, 245, 0.3);
    --blue: #0a84ff;
    --green: #30d158;
    --red: #ff453a;
    --orange: #ff9f0a;
  }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  html, body {
    height: 100%;
    margin: 0;
    overflow: hidden;
    overscroll-behavior: none;
    background: var(--bg);
    color: var(--label);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-variant-numeric: tabular-nums;
  }
  body {
    display: flex;
    flex-direction: column;
    height: 100dvh;
    padding: env(safe-area-inset-top) env(safe-area-inset-right) 0 env(safe-area-inset-left);
  }

  #stage {
    position: relative;
    flex: 1 1 auto;
    min-height: 0;
    background: #000;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #feed {
    max-width: 100%;
    max-height: 100%;
    display: block;
    cursor: crosshair;
    touch-action: manipulation;
    user-select: none;
  }

  .tap-marker {
    position: absolute;
    width: 64px; height: 64px;
    border: 1.5px solid var(--orange);
    border-radius: 14px;
    transform: translate(-50%, -50%) scale(1.3);
    opacity: 0;
    pointer-events: none;
    animation: tap-focus 0.6s ease-out forwards;
  }
  @keyframes tap-focus {
    0% { opacity: 0; transform: translate(-50%, -50%) scale(1.3); }
    15% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
    75% { opacity: 1; }
    100% { opacity: 0; transform: translate(-50%, -50%) scale(1); }
  }

  .pill {
    position: absolute;
    display: flex;
    align-items: baseline;
    gap: 5px;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(28, 28, 30, 0.65);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    backdrop-filter: blur(20px) saturate(180%);
    font-size: 13px;
    font-weight: 590;
    color: var(--label-2);
    pointer-events: none;
    white-space: nowrap;
  }
  .pill .num { color: var(--label); font-weight: 600; }
  #status-pill { top: calc(env(safe-area-inset-top) + 12px); left: 12px; }
  #status-pill.ready { color: var(--green); }
  #samples-pill { top: calc(env(safe-area-inset-top) + 12px); right: 12px; }

  #toast {
    position: absolute;
    left: 50%;
    bottom: 14px;
    transform: translate(-50%, 8px);
    max-width: 88%;
    padding: 9px 16px;
    border-radius: 12px;
    background: rgba(28, 28, 30, 0.82);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    backdrop-filter: blur(20px) saturate(180%);
    font-size: 13px;
    font-weight: 500;
    color: var(--label);
    text-align: center;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.25s ease, transform 0.25s ease;
  }
  #toast.show { opacity: 1; transform: translate(-50%, 0); }
  #toast.error { color: var(--red); }

  #sheet {
    flex: 0 0 auto;
    background: var(--sheet);
    -webkit-backdrop-filter: blur(30px) saturate(180%);
    backdrop-filter: blur(30px) saturate(180%);
    border-top: 0.5px solid var(--separator);
    border-radius: 20px 20px 0 0;
    padding: 14px 16px calc(10px + env(safe-area-inset-bottom));
  }

  .field-label {
    font-size: 12px;
    font-weight: 590;
    color: var(--label-2);
    margin-bottom: 6px;
  }
  .capture-row { display: flex; gap: 10px; margin-bottom: 12px; }
  #distance-input {
    flex: 1;
    background: var(--fill);
    border: none;
    border-radius: 12px;
    color: var(--label);
    font-size: 20px;
    font-weight: 600;
    font-family: inherit;
    padding: 12px 14px;
    text-align: right;
  }
  #distance-input::placeholder { color: var(--label-3); font-weight: 500; }
  #capture-btn {
    border: none;
    border-radius: 12px;
    background: var(--orange);
    color: #1a1a1a;
    font-size: 16px;
    font-weight: 650;
    font-family: inherit;
    padding: 0 22px;
    box-shadow: 0 2px 10px rgba(255, 159, 10, 0.3);
  }
  #capture-btn:disabled { opacity: 0.3; box-shadow: none; }

  .btn-row { display: flex; gap: 10px; }
  .tinted-btn {
    flex: 1;
    border: none;
    border-radius: 12px;
    background: var(--fill);
    color: var(--label);
    font-size: 15px;
    font-weight: 590;
    font-family: inherit;
    padding: 12px;
  }
  #save-btn { color: var(--green); }
  #disable-pan-btn { color: var(--red); flex: 0 0 auto; padding: 12px 16px; }

  button { -webkit-tap-highlight-color: transparent; }
  button:active { opacity: 0.7; }

  @keyframes flash-good { 0% { box-shadow: 0 0 0 0 rgba(48, 209, 88, 0.6); } 100% { box-shadow: 0 0 0 16px rgba(48, 209, 88, 0); } }
  @keyframes flash-bad { 0% { box-shadow: 0 0 0 0 rgba(255, 69, 58, 0.6); } 100% { box-shadow: 0 0 0 16px rgba(255, 69, 58, 0); } }
  .flash-good { animation: flash-good 0.5s ease-out; }
  .flash-bad { animation: flash-bad 0.5s ease-out; }
</style>
</head>
<body>
  <div id="stage">
    <img id="feed" src="/video_feed">
    <div id="status-pill" class="pill"><span id="status-text">Tap the cup to calibrate color</span></div>
    <div id="samples-pill" class="pill"><span class="num" id="samples-num">0</span><span>samples</span></div>
    <div id="toast"></div>
  </div>

  <div id="sheet">
    <div class="field-label">Distance to cup's front edge, from launcher base (in)</div>
    <div class="capture-row">
      <input id="distance-input" type="number" inputmode="decimal" step="0.1" placeholder="0.0">
      <button id="capture-btn">Capture</button>
    </div>
    <div class="btn-row">
      <button id="undo-btn" class="tinted-btn">Undo Last</button>
      <button id="save-btn" class="tinted-btn">Save Calibration</button>
      <button id="disable-pan-btn" class="tinted-btn">Disable PAN</button>
    </div>
  </div>

<script>
const feed = document.getElementById('feed');
const stage = document.getElementById('stage');

const toast = document.getElementById('toast');
let toastTimer = null;
function showToast(text, isError) {
  toast.textContent = text;
  toast.className = 'show' + (isError ? ' error' : '');
  clearTimeout(toastTimer);
  if (!isError) toastTimer = setTimeout(() => toast.classList.remove('show'), 3200);
}
function isErrorMessage(t) {
  const s = t.toLowerCase();
  return s.includes('fault') || s.includes('error') || s.includes('failed')
      || s.includes('no target') || s.includes('not a number') || s.includes('nothing to undo')
      || s.includes('need at least');
}
async function post(path, body) {
  try {
    const opts = {method: 'POST'};
    if (body) { opts.headers = {'Content-Type': 'application/json'}; opts.body = JSON.stringify(body); }
    const r = await fetch(path, opts);
    const j = await r.json();
    if (j.message) showToast(j.message, isErrorMessage(j.message));
    return j;
  } catch (e) {
    showToast('Connection lost', true);
  }
}

feed.addEventListener('click', (e) => {
  const rect = feed.getBoundingClientRect();
  const marker = document.createElement('div');
  marker.className = 'tap-marker';
  marker.style.left = (e.clientX - rect.left) + 'px';
  marker.style.top = (e.clientY - rect.top) + 'px';
  stage.appendChild(marker);
  marker.addEventListener('animationend', () => marker.remove());

  const scaleX = feed.naturalWidth / rect.width;
  const scaleY = feed.naturalHeight / rect.height;
  const x = Math.round((e.clientX - rect.left) * scaleX);
  const y = Math.round((e.clientY - rect.top) * scaleY);
  post('/click', {x, y});
});

const captureBtn = document.getElementById('capture-btn');
const distanceInput = document.getElementById('distance-input');
function flashResult(el, ok) {
  el.classList.remove('flash-good', 'flash-bad');
  void el.offsetWidth;
  el.classList.add(ok ? 'flash-good' : 'flash-bad');
}
captureBtn.addEventListener('click', async () => {
  const v = parseFloat(distanceInput.value);
  if (isNaN(v)) { showToast('Enter a distance first', true); return; }
  const j = await post('/capture', {distance_in: v});
  if (j) {
    const ok = !isErrorMessage(j.message || '');
    flashResult(captureBtn, ok);
    if (ok) distanceInput.value = '';
  }
});
distanceInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') captureBtn.click(); });

document.getElementById('undo-btn').addEventListener('click', () => post('/undo'));
document.getElementById('save-btn').addEventListener('click', async () => {
  const j = await post('/finish');
  if (j) flashResult(document.getElementById('save-btn'), !isErrorMessage(j.message || ''));
});
document.getElementById('disable-pan-btn').addEventListener('click', () => post('/disable_pan'));

const statusPill = document.getElementById('status-pill');
const statusText = document.getElementById('status-text');
const samplesNum = document.getElementById('samples-num');

async function pollStatus() {
  try {
    const r = await fetch('/status');
    const s = await r.json();
    samplesNum.textContent = s.sample_count;
    captureBtn.disabled = !s.detected;
    if (!s.calibrated) {
      statusText.textContent = 'Tap the cup to calibrate color';
      statusPill.classList.remove('ready');
    } else if (s.detected) {
      const centered = Math.abs(s.h_offset) < 15;
      statusText.textContent = 'row ' + s.base_row + 'px  •  '
        + (s.h_offset > 0 ? '+' : '') + s.h_offset + 'px'
        + (centered ? ' (centered)' : '');
      statusPill.classList.toggle('ready', centered);
    } else {
      statusText.textContent = 'No target detected';
      statusPill.classList.remove('ready');
    }
  } catch (e) {
    // transient poll failure - leave the last known state on screen
  }
}
setInterval(pollStatus, 400);
pollStatus();
</script>
</body>
</html>
"""


def main():
    print("=" * 60)
    print("TRACK B - pixel-row(cup base) -> range calibration capture (WEB)")
    print("=" * 60)
    print(f"Using offset {TAPE_OFFSET_IN:.2f}in ({OFFSET_LABEL} -> pivot) - tape now reads")
    print("directly from the base edge, so this is the only offset needed.")

    odrv_pan, axis_pan = connect_pan()
    pan_center = lock_pan(odrv_pan, axis_pan)
    if pan_center is None:
        axis_pan.requested_state = AxisState.IDLE
        return
    feeder = WatchdogFeeder(axis_pan, WATCHDOG_FEED_INTERVAL)
    feeder.start()

    calib = ColorCalibration()
    samples = load_existing_samples()
    if samples:
        print(f"Loaded {len(samples)} existing sample(s) - this session's captures will be added.")

    render_fn = make_render_fn(calib)
    cam = CameraWorker(CAM_INDEX, render_fn)
    if not cam.opened():
        print("Could not open camera.")
        feeder.stop()
        axis_pan.requested_state = AxisState.IDLE
        return
    cam.start()

    pan_disabled = {"value": False}

    def handle_click(x, y):
        frame, _ = cam.latest()
        if frame is None:
            return "No camera frame yet - try again in a moment."
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = calib.calibrate(hsv, x, y)
        return f"Color calibrated: H={hue:.0f} S={sat:.0f} V={val:.0f}."

    def handle_capture(distance_in):
        if calib.ranges is None:
            return "Tap the cup to calibrate its color first."
        read_row = make_web_row_reader(cam, calib)
        pixel_row = median_read(read_row, samples=5)
        if pixel_row is None:
            return "No target currently detected - can't capture."
        true_r_in = distance_in + TAPE_OFFSET_IN
        samples.append((pixel_row, distance_in, OFFSET_LABEL, true_r_in))
        return (f"Captured: row={pixel_row}px, {distance_in:.1f}in from {OFFSET_LABEL} "
                f"-> r={true_r_in:.2f}in ({len(samples)} total)")

    def handle_undo():
        if not samples:
            return "Nothing to undo."
        popped = samples.pop()
        return f"Undid sample at row {popped[0]}px ({len(samples)} remaining)."

    def handle_finish():
        ok, message = fit_and_save(list(samples))
        return message

    app = Flask(__name__)

    @app.route("/")
    def index():
        return INDEX_HTML

    @app.route("/video_feed")
    def video_feed():
        def generate():
            period = 1.0 / STREAM_FPS
            while True:
                jpeg = cam.latest_jpeg()
                if jpeg is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(period)
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/click", methods=["POST"])
    def click_route():
        data = request.get_json(force=True)
        message = handle_click(int(data["x"]), int(data["y"]))
        return jsonify({"message": message})

    @app.route("/capture", methods=["POST"])
    def capture_route():
        data = request.get_json(force=True)
        try:
            distance_in = float(data["distance_in"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"message": "Not a number - sample discarded."})
        message = handle_capture(distance_in)
        return jsonify({"message": message})

    @app.route("/undo", methods=["POST"])
    def undo_route():
        return jsonify({"message": handle_undo()})

    @app.route("/finish", methods=["POST"])
    def finish_route():
        return jsonify({"message": handle_finish()})

    @app.route("/disable_pan", methods=["POST"])
    def disable_pan_route():
        axis_pan.requested_state = AxisState.IDLE
        pan_disabled["value"] = True
        return jsonify({"message": "PAN disabled - safe to power off or adjust by hand."})

    @app.route("/status")
    def status_route():
        frame, _ = cam.latest()
        base_row = None
        h_offset = None
        detected = False
        if frame is not None and calib.ranges is not None:
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            det = detect_target(hsv, calib)
            if det is not None:
                detected = True
                base_row = det["base"][1]
                h_offset = det["centroid"][0] - w // 2
        return jsonify({
            "calibrated": calib.ranges is not None,
            "detected": detected,
            "base_row": base_row,
            "h_offset": h_offset,
            "sample_count": len(samples),
            "pan_disabled": pan_disabled["value"],
        })

    print(f"\nServing on http://{HOST}:{PORT}/ - open that from a browser on the same network.")
    print("Tap the cup to calibrate color, place it at a measured distance, type the")
    print("distance to its FRONT edge, press Capture. Repeat, then Save Calibration.\n")

    try:
        app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.stop()
        feeder.stop()
        print("Disabling PAN...")
        axis_pan.requested_state = AxisState.IDLE
        print("PAN disabled. Safe to power off.")


if __name__ == "__main__":
    main()
