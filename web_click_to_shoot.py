"""
Web version of click_to_shoot.py, served as an MJPEG stream + small JSON
API + a "Targeting HUD" browser UI instead of a cv2.imshow window. Built to
replace the X11-forwarded window (see recent commits about shrinking that
window / dropping buffered frames to cut per-frame SSH bytes) - a browser
tab over plain HTTP on the LAN is far lighter, and lets the same interface
work from a phone later without any X11 forwarding at all.

Two modes, switched client-side (the server doesn't track which one is
active - see /status's comment for why):
  - SEMI-AUTO: the original click-to-calibrate/aim/fire flow.
  - MANUAL: A/D jogs PAN (Shift=coarse, Alt=fine), W/S steps a stored swing
    angle (Shift=coarse, Alt=fine), SPACE launches at whatever that swing
    currently is - no cup or color calibration needed at all, for
    trial-and-error testing against a cup that hasn't been (or can't be)
    calibrated against.
ESC (E-STOP) and SPACE (fire) are global and always available regardless
of mode.

All hardware/vision logic is unchanged and imported from pong_common.py and
click_to_shoot.py (the bias constants and box-hit-test live there so the
two UIs can't drift apart).

Run with: odrive_env/bin/python web_click_to_shoot.py
Then open http://<this machine's LAN IP>:5000/ from another device on the
same network (e.g. your Mac). Only one browser tab should drive it at a
time - the hardware only supports one action in flight.
"""
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from odrive.enums import AxisState

from pong_common import (
    CAM_INDEX, MAX_RANGE_TURNS, PIVOT_TO_TAPE_START_IN, ROTATE_CODE,
    WATCHDOG_FEED_INTERVAL,
    ColorCalibration, WatchdogFeeder, calibrated_center, connect_launch, connect_pan,
    detect_all_targets, fire_shot, load_range_calibration, load_theta_calibration, lock_pan,
    median_read, move_pan_to, prepare_launch, solve_swing_for_range,
)
from click_to_shoot import RANGE_BIAS_CORRECTION_IN, THETA_BIAS_PX, find_target_in_box

HOST = "0.0.0.0"
PORT = 5000
STREAM_FPS = 20
JPEG_QUALITY = 75
# Same fixed small-step jog size validated in pan_axis_test.py (~7.2deg) -
# lock_pan() already puts PAN's trap_traj at JOG_VEL_LIMIT/ACCEL/DECEL, the
# same "slow crawl" speed that script used, so jog moves here inherit that
# safety margin automatically.
JOG_STEP_TURNS = 0.02
JOG_STEP_FINE_TURNS = JOG_STEP_TURNS / 2
JOG_STEP_COARSE_TURNS = JOG_STEP_TURNS * 3

# Manual swing-angle bound for the "set swing directly, no cup needed"
# testing mode. fire_shot() itself has no bound on swing_deg - every
# existing caller only ever reaches it through solve_swing_for_range(),
# which already refuses anything outside the calibrated 45-90deg curve. A
# free-typed swing box bypasses that guard, so it's clamped here to the
# actual tested envelope instead: 90deg is the same full swing
# launch_arm_control.py already runs routinely, and 30deg was already
# fired once during calibration (goes nearly straight up - a practical
# floor). Don't widen this without confirming by hand first.
MIN_MANUAL_SWING_DEG = 30.0
MAX_MANUAL_SWING_DEG = 90.0
# Manual mode's swing step per W/S press (held keys/buttons repeat via
# the same interval mechanism jog uses).
SWING_STEP_DEG = 5.0
SWING_STEP_FINE_DEG = SWING_STEP_DEG / 2
SWING_STEP_COARSE_DEG = SWING_STEP_DEG * 3
DEFAULT_MANUAL_SWING_DEG = 60.0

# BGR (OpenCV's channel order) equivalents of the HUD's cyan palette (see
# INDEX_HTML's :root colors) so the in-frame cup labels read as one system
# with the HTML HUD overlaid on top of the video.
COLOR_CUP = (255, 210, 20)      # #14d2ff-ish cyan - detected cup boxes/labels
COLOR_TEXT = (255, 245, 220)    # near-white with a cool tint, for the chip text
CHIP_BG = (18, 14, 4)           # near-black navy chip background

state_lock = threading.Lock()
action_lock = threading.Lock()
estop_event = threading.Event()
# Set for the duration of any hardware action (aiming or firing) - see
# CameraWorker._run below for why this exists: a TransportException from the
# ODrive USB link during a real fire traced back to this thread's continuous
# vision work (color convert + contour detection + JPEG encode, every frame,
# independent of whatever the action routes are doing) competing for CPU
# with the odrive library's own background comms thread. click_to_shoot.py
# never hit this because it's single-threaded - vision work was naturally
# paused for the whole duration of any move. Pausing the heavy per-frame
# work here for the duration of an action restores that same separation.
busy_event = threading.Event()


class CameraWorker:
    """Continuously grabs frames on its own thread and keeps the latest raw
    frame + a rendered/encoded JPEG ready to hand out. Both the MJPEG
    endpoint and the aiming logic read from here instead of touching
    cv2.VideoCapture themselves - a single cv2.VideoCapture isn't safe to
    read from two threads at once, and this also means the click handler's
    "wait for a fresh frame after a PAN move" check (see wait_fresh_frame
    below) doesn't have to fight the stream for camera access."""

    def __init__(self, cam_index, render_fn):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.render_fn = render_fn
        self.lock = threading.Lock()
        self.frame = None
        self.jpeg = None
        self.seq = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def opened(self):
        return self.cap.isOpened()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)
        self.cap.release()

    def _run(self):
        while not self.stop_event.is_set():
            self.cap.grab()
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            if ROTATE_CODE is not None:
                frame = cv2.rotate(frame, ROTATE_CODE)
            with self.lock:
                self.frame = frame
                self.seq += 1
            if busy_event.is_set():
                # Frame capture stays live (cheap, and aiming's own reads
                # depend on it via wait_fresh_frame) but detection/draw/
                # encode - the CPU-heavy part - is skipped so it can't steal
                # cycles from the ODrive comms thread during a real move.
                continue
            try:
                annotated = self.render_fn(frame.copy())
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    with self.lock:
                        self.jpeg = buf.tobytes()
            except Exception as e:
                # A transient ODrive read failure here (e.g. a brief power
                # blip - see the pan-angle label's own try/except above)
                # must not kill this whole thread the way it did before -
                # that permanently froze the video stream for the rest of
                # the session even though the hardware itself recovered
                # and later /click and /fire calls worked fine. This is
                # just a display label, not a motion command, so skipping
                # the frame is the right response, not an E-STOP.
                print(f"WARNING: frame render failed, skipping this frame: {e!r}")

    def latest(self):
        with self.lock:
            return self.frame, self.seq

    def latest_jpeg(self):
        with self.lock:
            return self.jpeg

    def set_busy_frame(self, text):
        """One-off cheap redraw so the stream shows *why* it's frozen
        instead of just going stale with no explanation."""
        frame, _ = self.latest()
        if frame is None:
            return
        frame = frame.copy()
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with self.lock:
                self.jpeg = buf.tobytes()


def wait_fresh_frame(cam, min_advance=2, timeout=2.0):
    """Blocks until at least `min_advance` new frames have been captured
    since this was called, so a read always reflects the scene after the
    last PAN move settled - same intent as click_to_shoot's single
    discarded-frame fix, just phrased against the background capture
    thread's sequence counter instead of a single extra cap.read()."""
    _, seq0 = cam.latest()
    start = time.time()
    while time.time() - start < timeout:
        frame, seq = cam.latest()
        if seq - seq0 >= min_advance:
            return frame
        time.sleep(0.01)
    frame, _ = cam.latest()
    return frame


def make_web_target_tracker(cam, calib, initial_x, frame_cx):
    """Same stateful nearest-to-last-seen tracker as click_to_shoot.py's
    make_target_tracker, adapted to read frames via wait_fresh_frame(cam)
    instead of calling cap.grab()/read() directly."""
    state = {"x": initial_x}

    def read_error():
        frame = wait_fresh_frame(cam)
        if frame is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        with state_lock:
            dets = detect_all_targets(hsv, calib)
        if not dets:
            return None
        best = min(dets, key=lambda d: abs(d["centroid"][0] - state["x"]))
        state["x"] = best["centroid"][0]
        return (best["centroid"][0] - frame_cx) - THETA_BIAS_PX

    def read_row():
        frame = wait_fresh_frame(cam)
        if frame is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        with state_lock:
            dets = detect_all_targets(hsv, calib)
        if not dets:
            return None
        best = min(dets, key=lambda d: abs(d["centroid"][0] - state["x"]))
        state["x"] = best["centroid"][0]
        return best["base"][1]

    return read_error, read_row


def draw_chip(frame, text, x, y, color=COLOR_TEXT, scale=0.55, thickness=1, bg_alpha=0.6):
    """Draws `text` on a small translucent chip instead of raw floating
    cv2.putText - matches the browser UI's panel/pill look and stays
    readable over any background scene. (x, y) is the chip's top-left
    corner. Only blends the chip's own small region (not the full frame)
    so this stays cheap to call several times per frame on a Pi 3B.
    Returns (width, height) so callers can stack chips without hardcoded
    pixel offsets."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 8, 6
    box_w, box_h = tw + 2 * pad_x, th + baseline + 2 * pad_y
    h, w = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w, x + box_w), min(h, y + box_h)
    if x1 > x0 and y1 > y0:
        roi = frame[y0:y1, x0:x1]
        bg = np.full_like(roi, CHIP_BG)
        cv2.addWeighted(bg, bg_alpha, roi, 1 - bg_alpha, 0, roi)
    cv2.putText(frame, text, (x + pad_x, y + pad_y + th), font, scale, color, thickness, cv2.LINE_AA)
    return box_w, box_h


def make_render_fn(calib, app_state):
    """Builds the per-frame overlay draw function handed to CameraWorker.

    Only draws things that must be pixel-locked to a specific detected blob
    in THIS exact frame (cup boxes + per-cup pixel-offset labels) - status
    text, pan angle, cup count, and E-STOP state used to be baked in here
    too, but those are now real HTML elements polled from /status instead
    (see INDEX_HTML's HUD readouts). That reads far crisper than
    JPEG-compressed cv2 text and updates independently of frame rate, and
    it's also how two different modes (SEMI-AUTO/MANUAL) can show different
    things without this server-side code needing to know which mode the
    browser is currently in."""

    def render(frame):
        h, w = frame.shape[:2]
        frame_cx = w // 2
        with state_lock:
            calibrated = calib.ranges is not None
            dets = detect_all_targets(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), calib) \
                if calibrated else []

        for d in dets:
            x, y, bw, bh = d["bbox"]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_CUP, 2, cv2.LINE_AA)
            h_offset = (d["centroid"][0] - frame_cx) - THETA_BIAS_PX
            draw_chip(frame, f"{h_offset:+d}px", x, max(0, y - 30),
                      color=COLOR_CUP, scale=0.45, bg_alpha=0.45)

        return frame

    return render


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
<title>Launch Control</title>
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
    /* System font stack, not a Google-Fonts/CDN import - resolves to actual
       San Francisco on Apple devices (and each platform's own equivalent
       elsewhere), so it keeps working with zero internet once this is
       off-grid. */
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-variant-numeric: tabular-nums;
  }
  body {
    display: flex;
    flex-direction: column;
    height: 100dvh;
    padding: env(safe-area-inset-top) env(safe-area-inset-right) 0 env(safe-area-inset-left);
  }

  /* ---------- Video stage: full-bleed, like a camera viewfinder ---------- */
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

  .reticle {
    position: absolute; top: 50%; left: 50%;
    width: 22px; height: 22px;
    transform: translate(-50%, -50%);
    pointer-events: none;
    opacity: 0.5;
  }
  .reticle::before, .reticle::after { content: ''; position: absolute; background: #fff; }
  .reticle::before { top: 50%; left: 3px; right: 3px; height: 1px; transform: translateY(-50%); }
  .reticle::after { left: 50%; top: 3px; bottom: 3px; width: 1px; transform: translateX(-50%); }

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
  #status-pill {
    top: calc(env(safe-area-inset-top) + 12px);
    left: 12px;
    max-width: 60%;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  #status-pill.ready { color: var(--green); }
  #status-pill .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor;
    flex: 0 0 auto;
  }
  #cups-pill { top: calc(env(safe-area-inset-top) + 12px); right: 12px; }

  #estop-banner {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    display: none;
    align-items: center;
    gap: 8px;
    padding: 12px 18px;
    border-radius: 14px;
    background: rgba(28, 28, 30, 0.85);
    -webkit-backdrop-filter: blur(20px);
    backdrop-filter: blur(20px);
    border: 0.5px solid rgba(255, 69, 58, 0.5);
    color: #fff;
    font-size: 15px;
    font-weight: 600;
    pointer-events: none;
  }
  #estop-banner.show { display: flex; }
  #estop-banner svg { color: var(--red); flex: 0 0 auto; }
  #estop-banner .sub { display: block; font-size: 12px; font-weight: 400; color: var(--label-2); }

  /* ---------- Toast: ephemeral action feedback, doesn't reserve layout space --- */
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

  /* ---------- Control sheet: frosted glass, like a Camera/Maps bottom card --- */
  #sheet {
    flex: 0 0 auto;
    background: var(--sheet);
    -webkit-backdrop-filter: blur(30px) saturate(180%);
    backdrop-filter: blur(30px) saturate(180%);
    border-top: 0.5px solid var(--separator);
    border-radius: 20px 20px 0 0;
    padding: 12px 16px calc(10px + env(safe-area-inset-bottom));
  }

  .segmented {
    position: relative;
    display: flex;
    background: var(--fill);
    border-radius: 9px;
    padding: 2px;
    height: 34px;
    margin-bottom: 14px;
  }
  .segmented .seg-indicator {
    position: absolute;
    top: 2px; bottom: 2px; left: 2px;
    width: calc(50% - 2px);
    background: var(--fill-strong);
    border-radius: 7px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
    transition: transform 0.28s cubic-bezier(0.25, 1, 0.5, 1);
  }
  .segmented.manual .seg-indicator { transform: translateX(100%); }
  .seg-btn {
    position: relative;
    flex: 1;
    background: none;
    border: none;
    padding: 0;
    color: var(--label-2);
    font-size: 13px;
    font-weight: 590;
    font-family: inherit;
    transition: color 0.2s ease;
  }
  .seg-btn.active { color: var(--label); }

  .mode-panel { display: none; }
  .mode-panel.active { display: block; }

  .tinted-btn {
    width: 100%;
    border: none;
    border-radius: 12px;
    background: var(--fill);
    color: var(--label);
    font-size: 15px;
    font-weight: 590;
    font-family: inherit;
    padding: 12px;
  }

  .axis-row { display: flex; gap: 10px; }
  .axis-card {
    flex: 1;
    background: var(--fill);
    border-radius: 14px;
    padding: 10px 12px 12px;
  }
  .axis-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
  }
  .axis-name {
    font-size: 12px;
    font-weight: 590;
    color: var(--label-2);
  }
  .axis-num {
    font-size: 20px;
    font-weight: 600;
    color: var(--label);
  }
  .stepper {
    display: flex;
    background: var(--fill-strong);
    border-radius: 9px;
    overflow: hidden;
    height: 40px;
  }
  .stepper button {
    flex: 1;
    background: none;
    border: none;
    color: var(--label);
    display: flex;
    align-items: center;
    justify-content: center;
    touch-action: manipulation;
    user-select: none;
  }
  .stepper button:first-child { border-right: 0.5px solid var(--separator); }
  .stepper button:active { background: rgba(255, 255, 255, 0.12); }
  .stepper svg { width: 18px; height: 18px; }

  button { -webkit-tap-highlight-color: transparent; }
  button:active { opacity: 0.7; }

  .action-row {
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 40px;
    padding-top: 16px;
  }
  .action-item { display: flex; flex-direction: column; align-items: center; gap: 6px; }
  .action-item .action-label { font-size: 12px; font-weight: 590; color: var(--label-2); }
  .circle-btn {
    border: none;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #fff;
    transition: transform 0.12s ease;
  }
  .circle-btn:active { transform: scale(0.92); }
  .circle-btn:disabled { opacity: 0.3; }
  #fire-btn {
    width: 76px; height: 76px;
    background: var(--orange);
    box-shadow: 0 2px 10px rgba(255, 159, 10, 0.35);
  }
  #fire-btn svg { width: 30px; height: 30px; }
  #estop-btn {
    width: 56px; height: 56px;
    background: var(--red);
    box-shadow: 0 2px 10px rgba(255, 69, 58, 0.35);
  }
  #estop-btn svg { width: 22px; height: 22px; }

  @keyframes flash-good { 0% { box-shadow: 0 0 0 0 rgba(48, 209, 88, 0.6); } 100% { box-shadow: 0 0 0 16px rgba(48, 209, 88, 0); } }
  @keyframes flash-bad { 0% { box-shadow: 0 0 0 0 rgba(255, 69, 58, 0.6); } 100% { box-shadow: 0 0 0 16px rgba(255, 69, 58, 0); } }
  .flash-good { animation: flash-good 0.5s ease-out; }
  .flash-bad { animation: flash-bad 0.5s ease-out; }
</style>
</head>
<body>
  <div id="stage">
    <img id="feed" src="/video_feed">
    <div class="reticle"></div>
    <div id="status-pill" class="pill"><span class="dot"></span><span id="status-text">Tap a cup to calibrate</span></div>
    <div id="cups-pill" class="pill"><span class="num" id="cups-num">0</span><span id="cups-word">cups</span></div>
    <div id="estop-banner">
      <svg viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="3"/></svg>
      <span>E-Stop active<span class="sub">Restart the server to resume</span></span>
    </div>
    <div id="toast"></div>
  </div>

  <div id="sheet">
    <div class="segmented" id="segmented">
      <div class="seg-indicator"></div>
      <button id="mode-semi" class="seg-btn active">Semi-Auto</button>
      <button id="mode-manual" class="seg-btn">Manual</button>
    </div>

    <div id="panel-semi" class="mode-panel active">
      <button id="reset" class="tinted-btn">Reset Color</button>
    </div>

    <div id="panel-manual" class="mode-panel">
      <div class="axis-row">
        <div class="axis-card">
          <div class="axis-head"><span class="axis-name">Pan</span><span class="axis-num" id="manual-pan-value">0.0&deg;</span></div>
          <div class="stepper">
            <button id="pan-left" aria-label="Pan left">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 6l-6 6 6 6"/></svg>
            </button>
            <button id="pan-right" aria-label="Pan right">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 6l6 6-6 6"/></svg>
            </button>
          </div>
        </div>
        <div class="axis-card">
          <div class="axis-head"><span class="axis-name">Swing</span><span class="axis-num" id="manual-swing-value">60.0&deg;</span></div>
          <div class="stepper">
            <button id="swing-down" aria-label="Decrease swing">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>
            </button>
            <button id="swing-up" aria-label="Increase swing">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 15l6-6 6 6"/></svg>
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="action-row">
      <div class="action-item">
        <button id="estop-btn" class="circle-btn" aria-label="E-Stop">
          <svg viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="3"/></svg>
        </button>
        <span class="action-label">Stop</span>
      </div>
      <div class="action-item">
        <button id="fire-btn" class="circle-btn" aria-label="Fire">
          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 3l7 15H5z"/></svg>
        </button>
        <span class="action-label">Fire</span>
      </div>
    </div>
  </div>

<script>
const feed = document.getElementById('feed');
const stage = document.getElementById('stage');
let mode = 'semi'; // 'semi' | 'manual'

// --- Toast (ephemeral, replaces a persistent log line) ------------------
const toast = document.getElementById('toast');
let toastTimer = null;
function showToast(text, isError) {
  toast.textContent = text;
  toast.className = 'show' + (isError ? ' error' : '');
  clearTimeout(toastTimer);
  if (!isError) toastTimer = setTimeout(() => toast.classList.remove('show'), 2600);
}

function isErrorMessage(t) {
  const s = t.toLowerCase();
  return s.includes('e-stop') || s.includes('fault') || s.includes('error') || s.includes('failed')
      || s.includes('must be') || s.includes('invalid') || s.includes('timed out');
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

// --- Mode switch ---------------------------------------------------------
const segmented = document.getElementById('segmented');
const modeSemiBtn = document.getElementById('mode-semi');
const modeManualBtn = document.getElementById('mode-manual');
const panelSemi = document.getElementById('panel-semi');
const panelManual = document.getElementById('panel-manual');

function setMode(m) {
  mode = m;
  segmented.classList.toggle('manual', m === 'manual');
  modeSemiBtn.classList.toggle('active', m === 'semi');
  modeManualBtn.classList.toggle('active', m === 'manual');
  panelSemi.classList.toggle('active', m === 'semi');
  panelManual.classList.toggle('active', m === 'manual');
}
modeSemiBtn.addEventListener('click', () => setMode('semi'));
modeManualBtn.addEventListener('click', () => setMode('manual'));

// --- Click-to-aim (Semi-Auto only), with a tap-to-focus style marker ----
feed.addEventListener('click', (e) => {
  if (mode !== 'semi') return;
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
document.getElementById('reset').addEventListener('click', () => post('/reset_calibration'));

// --- Fire / E-Stop (always available, endpoint depends on mode) ---------
const fireBtn = document.getElementById('fire-btn');
const estopBtn = document.getElementById('estop-btn');

function flashResult(el, ok) {
  el.classList.remove('flash-good', 'flash-bad');
  void el.offsetWidth; // restart animation if fired again quickly
  el.classList.add(ok ? 'flash-good' : 'flash-bad');
}

fireBtn.addEventListener('click', async () => {
  const j = await post(mode === 'manual' ? '/fire_manual' : '/fire');
  if (j) flashResult(fireBtn, !isErrorMessage(j.message || ''));
});
estopBtn.addEventListener('click', () => post('/estop'));

// --- Repeat-while-held helper, shared by pan jog and swing step ----------
const REPEAT_MS = 200;
function makeRepeater(fn) {
  let timer = null;
  let mag = 'med';
  return {
    start(m = 'med') { if (timer) return; mag = m; fn(mag); timer = setInterval(() => fn(mag), REPEAT_MS); },
    stop() { if (timer) { clearInterval(timer); timer = null; } },
  };
}
function bindHold(el, repeater) {
  el.addEventListener('pointerdown', (e) => { e.preventDefault(); repeater.start(); });
  ['pointerup', 'pointerleave', 'pointercancel'].forEach(evt => el.addEventListener(evt, repeater.stop));
}

const panLeft = makeRepeater((mag) => post('/jog', {dir: 'left', mag}));
const panRight = makeRepeater((mag) => post('/jog', {dir: 'right', mag}));
const swingUp = makeRepeater((mag) => post('/swing_step', {dir: 'up', mag}));
const swingDown = makeRepeater((mag) => post('/swing_step', {dir: 'down', mag}));
bindHold(document.getElementById('pan-left'), panLeft);
bindHold(document.getElementById('pan-right'), panRight);
bindHold(document.getElementById('swing-up'), swingUp);
bindHold(document.getElementById('swing-down'), swingDown);

// --- Keyboard: WASD only acts in Manual mode, so Semi-Auto's precise
// click-based aiming can't be nudged by an accidental key press. ESC/SPACE
// (e-stop/fire) are global regardless of mode. ---------------------------
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { post('/estop'); return; }
  if (e.key === ' ') { e.preventDefault(); post(mode === 'manual' ? '/fire_manual' : '/fire'); return; }
  if (e.key.toLowerCase() === 'r') { post('/reset_calibration'); return; }
  if (mode !== 'manual') return;
  const k = e.key.toLowerCase();
  if (!['a', 'd', 'w', 's'].includes(k)) return;
  e.preventDefault();
  const mag = e.shiftKey ? 'coarse' : (e.altKey ? 'fine' : 'med');
  if (k === 'a') panLeft.start(mag);
  else if (k === 'd') panRight.start(mag);
  else if (k === 'w') swingUp.start(mag);
  else if (k === 's') swingDown.start(mag);
});
document.addEventListener('keyup', (e) => {
  const k = e.key.toLowerCase();
  if (k === 'a') panLeft.stop();
  else if (k === 'd') panRight.stop();
  else if (k === 'w') swingUp.stop();
  else if (k === 's') swingDown.stop();
});

// --- Telemetry poll --------------------------------------------------------
const statusPill = document.getElementById('status-pill');
const statusText = document.getElementById('status-text');
const cupsNum = document.getElementById('cups-num');
const cupsWord = document.getElementById('cups-word');
const estopBanner = document.getElementById('estop-banner');
const manualPanValue = document.getElementById('manual-pan-value');
const manualSwingValue = document.getElementById('manual-swing-value');

function deg(v) { return (v === null || v === undefined) ? '--.-°' : v.toFixed(1) + '°'; }

async function pollStatus() {
  try {
    const r = await fetch('/status');
    const s = await r.json();
    estopBanner.classList.toggle('show', s.estop);
    fireBtn.disabled = s.estop;

    manualPanValue.textContent = deg(s.pan_deg);
    manualSwingValue.textContent = deg(s.manual_swing_deg);
    cupsNum.textContent = s.cup_count;
    cupsWord.textContent = s.cup_count === 1 ? 'cup' : 'cups';

    if (s.estop) {
      statusText.textContent = 'E-Stop active';
      statusPill.classList.remove('ready');
    } else if (s.target) {
      statusText.textContent = s.target.r_in !== null
        ? 'Ready — r=' + s.target.r_in.toFixed(1) + 'in'
        : 'Ready — swing=' + s.target.swing_deg.toFixed(1) + '°';
      statusPill.classList.add('ready');
    } else if (!s.calibrated) {
      statusText.textContent = 'Tap a cup to calibrate';
      statusPill.classList.remove('ready');
    } else {
      statusText.textContent = 'Tap a cup to aim';
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
    print("CLICK TO SHOOT - WEB")
    print("=" * 60)

    try:
        cal_pan_center, phi0, phi_of_e, valid_phi_range = load_theta_calibration()
    except FileNotFoundError:
        print("ERROR: theta_calibration.json not found - run theta_calibration.py first.")
        return
    try:
        row_to_r = load_range_calibration()
    except FileNotFoundError:
        print("ERROR: pixel_range_calibration.json not found - run cup_pixel_calibration.py first.")
        return

    odrv_pan, axis_pan = connect_pan()
    pan_center = lock_pan(odrv_pan, axis_pan)
    if pan_center is None:
        axis_pan.requested_state = AxisState.IDLE
        return
    if abs(pan_center - cal_pan_center) > 0.01:
        print(f"WARNING: current lock center ({pan_center:.4f}) differs from the one "
              f"theta_calibration.json was built with ({cal_pan_center:.4f}) by "
              f"{(pan_center - cal_pan_center) * 360:.1f}deg - aiming may be off. "
              "Consider recalibrating.")
    pan_feeder = WatchdogFeeder(axis_pan, WATCHDOG_FEED_INTERVAL)
    pan_feeder.start()

    odrv_launch, axis_launch = connect_launch()
    start_pos = prepare_launch(odrv_launch, axis_launch)
    if start_pos is None:
        pan_feeder.stop()
        axis_pan.requested_state = AxisState.IDLE
        axis_launch.requested_state = AxisState.IDLE
        return
    launch_feeder = WatchdogFeeder(axis_launch, WATCHDOG_FEED_INTERVAL)
    launch_feeder.start()

    calib = ColorCalibration()
    app_state = {"current_target": None, "manual_swing_deg": DEFAULT_MANUAL_SWING_DEG}

    render_fn = make_render_fn(calib, app_state)
    cam = CameraWorker(CAM_INDEX, render_fn)
    if not cam.opened():
        print("Could not open camera.")
        pan_feeder.stop()
        launch_feeder.stop()
        axis_pan.requested_state = AxisState.IDLE
        axis_launch.requested_state = AxisState.IDLE
        return
    cam.start()

    def check_abort():
        return estop_event.is_set()

    def handle_click(x, y):
        if estop_event.is_set():
            return "E-STOP is active - restart the server to resume."
        frame, _ = cam.latest()
        if frame is None:
            return "No camera frame yet - try again in a moment."
        h, w = frame.shape[:2]
        frame_cx = w // 2
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        with state_lock:
            calibrated = calib.ranges is not None
        if not calibrated:
            with state_lock:
                hue, sat, val = calib.calibrate(hsv, x, y)
            return (f"Color calibrated: H={hue:.0f} S={sat:.0f} V={val:.0f}. "
                    "Tap inside a box to aim there.")

        with state_lock:
            dets = detect_all_targets(hsv, calib)
        target = find_target_in_box(dets, x, y)
        if target is None:
            return "Tap inside one of the detected cup boxes to aim there."

        e0 = target["centroid"][0] - frame_cx
        row0 = target["base"][1]
        out_of_range = not (valid_phi_range[0]
                             <= (axis_pan.pos_estimate + phi0 - phi_of_e(e0)) - pan_center
                             <= valid_phi_range[1])

        read_error, read_row = make_web_target_tracker(cam, calib, target["centroid"][0], frame_cx)

        ok, final_err, moves = calibrated_center(
            axis_pan, read_error, pan_center, phi0, phi_of_e,
            max_offset_turns=MAX_RANGE_TURNS, deadband_px=4,
            move_tolerance=0.001, check_abort=check_abort)
        if ok == 'estop':
            axis_launch.requested_state = AxisState.IDLE
            return "E-STOP during centering."

        row_final = median_read(read_row, samples=5)
        if row_final is None:
            row_final = row0
        r_in = row_to_r(row_final)
        swing_deg = solve_swing_for_range((r_in - RANGE_BIAS_CORRECTION_IN) * 0.0254)
        if swing_deg is None:
            with state_lock:
                app_state["current_target"] = None
            return f"r={r_in:.1f}in is outside the calibrated launch range - not aiming there."

        with state_lock:
            app_state["current_target"] = {"r_in": r_in, "swing_deg": swing_deg}
        note = " (outside calibrated aiming range - may be less accurate)" if out_of_range else ""
        centering_note = "" if ok else f" (centering incomplete, final error {final_err}px)"
        return f"READY: r={r_in:.1f}in -> swing={swing_deg:.2f}deg{note}{centering_note}"

    def handle_fire():
        if estop_event.is_set():
            return "E-STOP is active - restart the server to resume."
        with state_lock:
            target = app_state["current_target"]
        if target is None:
            return "No target selected - tap a cup first."
        result = fire_shot(axis_launch, start_pos, target["swing_deg"], check_abort=check_abort)
        if result == 'estop':
            return "E-STOP during fire."
        if result != 'done':
            return f"Shot did not complete cleanly ({result})."
        return "Shot fired."

    def handle_jog(direction, mag="med"):
        if estop_event.is_set():
            return "E-STOP is active - restart the server to resume."
        step_by_mag = {"fine": JOG_STEP_FINE_TURNS, "coarse": JOG_STEP_COARSE_TURNS}
        base = step_by_mag.get(mag, JOG_STEP_TURNS)
        step = -base if direction == "left" else base
        target = axis_pan.pos_estimate + step
        clamped = max(pan_center - MAX_RANGE_TURNS, min(pan_center + MAX_RANGE_TURNS, target))
        if abs(clamped - target) > 1e-9:
            return f"At the {direction} jog limit ({MAX_RANGE_TURNS * 360:.0f}° from center)."
        result = move_pan_to(axis_pan, clamped, description="jog", check_abort=check_abort)
        if result == 'estop':
            return "E-STOP during jog."
        # A jog moves PAN away from wherever the last click aimed it, so
        # that aim (and its solved r/swing) no longer points at the same
        # spot - clearing it here stops a stale target from being fired
        # at the new pan angle by mistake.
        with state_lock:
            had_target = app_state["current_target"] is not None
            app_state["current_target"] = None
        note = " - target cleared, tap a cup to re-aim" if had_target else ""
        angle = (axis_pan.pos_estimate - pan_center) * 360
        if result != True:
            return f"Jog timed out (pan now {angle:+.1f}°){note}."
        return f"Pan {angle:+.1f}°{note}"

    def handle_swing_step(direction, mag="med"):
        # No hardware motion here - MANUAL mode's swing angle is just a
        # stored number that FIRE reads at the moment you press it, so
        # stepping it can't collide with an in-flight move the way jog/fire
        # can. Still clamped to the tested envelope (see MIN/MAX_MANUAL_
        # SWING_DEG's comment) since fire_shot() itself trusts the caller
        # completely.
        step_by_mag = {"fine": SWING_STEP_FINE_DEG, "coarse": SWING_STEP_COARSE_DEG}
        base = step_by_mag.get(mag, SWING_STEP_DEG)
        step = -base if direction == "down" else base
        with state_lock:
            new_val = max(MIN_MANUAL_SWING_DEG,
                           min(MAX_MANUAL_SWING_DEG, app_state["manual_swing_deg"] + step))
            app_state["manual_swing_deg"] = new_val
        return f"Swing {new_val:.1f}°"

    def handle_fire_manual():
        if estop_event.is_set():
            return "E-STOP is active - restart the server to resume."
        with state_lock:
            swing_deg = app_state["manual_swing_deg"]
        result = fire_shot(axis_launch, start_pos, swing_deg, check_abort=check_abort)
        if result == 'estop':
            return "E-STOP during fire."
        if result != 'done':
            return f"Shot did not complete cleanly ({result})."
        return f"Shot fired at {swing_deg:.1f}°."

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

    def run_action(label, fn):
        """Shared wrapper for /click and /fire: pauses the camera worker's
        heavy per-frame work for the duration (cheap insurance against CPU
        contention with the ODrive comms thread) and fails safe if the
        ODrive USB link drops mid-action.

        A real transport fault (confirmed via dmesg to be Pi under-voltage
        brownouts resetting the USB bus, not this script) leaves the
        Python-side connection object dead even after the physical device
        re-enumerates - every further call against it just times out
        forever rather than actually reconnecting. So instead of leaving
        the session up to uselessly retry (what happened before this fix:
        every subsequent /click and /fire timed out at the same stuck
        position), a transport error here trips estop_event, same as a
        manual E-STOP - the session is over and needs a restart once the
        arm's physical position has been checked by hand."""
        if not action_lock.acquire(blocking=False):
            return "Busy - previous action still in progress."
        busy_event.set()
        cam.set_busy_frame(label)
        try:
            return fn()
        except Exception as e:
            print(f"ERROR during {label.lower()}: {e!r}")
            estop_event.set()
            try:
                axis_pan.requested_state = AxisState.IDLE
                axis_launch.requested_state = AxisState.IDLE
            except Exception:
                pass  # connection may already be dead - nothing more to do here
            return (f"Hardware communication fault ({e}) - treating this as an E-STOP. "
                    "Check the arm by hand, then restart the server.")
        finally:
            busy_event.clear()
            action_lock.release()

    @app.route("/click", methods=["POST"])
    def click_route():
        data = request.get_json(force=True)
        x, y = int(data["x"]), int(data["y"])
        message = run_action("AIMING...", lambda: handle_click(x, y))
        return jsonify({"message": message})

    @app.route("/fire", methods=["POST"])
    def fire_route():
        message = run_action("FIRING...", handle_fire)
        return jsonify({"message": message})

    @app.route("/fire_manual", methods=["POST"])
    def fire_manual_route():
        message = run_action("FIRING...", handle_fire_manual)
        return jsonify({"message": message})

    @app.route("/jog", methods=["POST"])
    def jog_route():
        data = request.get_json(force=True)
        direction = data.get("dir")
        mag = data.get("mag", "med")
        if direction not in ("left", "right"):
            return jsonify({"message": "Invalid jog direction."}), 400
        message = run_action(f"JOG {direction.upper()}...", lambda: handle_jog(direction, mag))
        return jsonify({"message": message})

    @app.route("/swing_step", methods=["POST"])
    def swing_step_route():
        data = request.get_json(force=True)
        direction = data.get("dir")
        mag = data.get("mag", "med")
        if direction not in ("up", "down"):
            return jsonify({"message": "Invalid swing direction."}), 400
        message = handle_swing_step(direction, mag)
        return jsonify({"message": message})

    @app.route("/reset_calibration", methods=["POST"])
    def reset_calibration_route():
        with state_lock:
            calib.ranges = None
            app_state["current_target"] = None
        return jsonify({"message": "Color calibration reset - tap a cup to recalibrate."})

    @app.route("/estop", methods=["POST"])
    def estop_route():
        estop_event.set()
        axis_pan.requested_state = AxisState.IDLE
        axis_launch.requested_state = AxisState.IDLE
        return jsonify({"message": "E-STOP: both motors disabled. Restart the server to resume."})

    @app.route("/status")
    def status_route():
        """Polled by the browser's HUD readouts (~2.5Hz) - kept separate
        from the video stream so PAN angle/target/cup-count text renders as
        crisp real HTML instead of JPEG-compressed cv2 text, and so it can
        differ by mode without this server needing to track which mode the
        browser is in."""
        try:
            pan_deg = (axis_pan.pos_estimate - pan_center) * 360
        except Exception:
            pan_deg = None
        with state_lock:
            calibrated = calib.ranges is not None
            current_target = app_state["current_target"]
            manual_swing_deg = app_state["manual_swing_deg"]
        cup_count = 0
        if calibrated:
            frame, _ = cam.latest()
            if frame is not None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                with state_lock:
                    cup_count = len(detect_all_targets(hsv, calib))
        return jsonify({
            "estop": estop_event.is_set(),
            "calibrated": calibrated,
            "pan_deg": pan_deg,
            "manual_swing_deg": manual_swing_deg,
            "cup_count": cup_count,
            "target": current_target,
        })

    print(f"\nServing on http://{HOST}:{PORT}/ - open that from a browser on the same network.")
    print("SEMI-AUTO mode: tap a cup to calibrate color, then tap any cup to aim, then FIRE.")
    print("MANUAL mode: A/D (or on-screen buttons) jog PAN, W/S step the swing angle")
    print("(Shift=coarse, Alt=fine), SPACE launches at whatever swing is currently set.")
    print("ESC E-STOPs (disables both motors immediately) and ends the session (Ctrl+C also works).\n")

    try:
        app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.stop()
        pan_feeder.stop()
        launch_feeder.stop()
        print("Disabling PAN and LAUNCH...")
        axis_pan.requested_state = AxisState.IDLE
        axis_launch.requested_state = AxisState.IDLE
        print("Both motors disabled. Safe to power off.")


if __name__ == "__main__":
    main()
