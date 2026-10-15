"""
Track A calibration capture, browser version - the "aim by trial, then
record" method instead of the old "fire a fixed swing, measure landing with
slomo video, solve for exit speed" method. Place a cup at a measured
distance, jog the swing angle with the keyboard while firing test shots
until it lands in the cup, then record that (r, swing_deg) pair. Repeat
across several distances, then fit+save a direct r -> swing_deg curve.

This replaces the slomo-video speed/angle solve entirely for this dataset,
and any systematic real-world bias (the ~1.5in overshoot RANGE_BIAS_
CORRECTION_IN in click_to_shoot.py exists to correct for) gets absorbed
straight into the recorded swing angle instead of needing a separate
correction constant.

Tradeoff (deliberate, flat-table scoping): this
produces a table valid only for landing at table height (z=0), not a
physics model of the trajectory shape, so it can't generalize to elevated
targets the way pong_common.py's CAL/landing_range_m() is designed to. Fine
for now since height-targeting is explicitly out of scope until needed.

PAN defaults to its locked center each session, but can now be jogged too -
this is just a faster alternative to physically nudging the cup onto the
centerline (the r/swing_deg relationship this script is capturing is
radial - PAN angle doesn't factor into it, so jogging PAN mid-session
doesn't affect the validity of points recorded before or after). PAN holds
wherever it's last jogged to (ordinary closed-loop position control - it
never "goes loose" between jogs) until you jog it again. The h-offset
readout (same as web_cup_pixel_calibration.py) still shows how far the cup
sits from the frame's centerline, now for either nudging the cup OR jogging
PAN to match.

Run with: odrive_env/bin/python web_range_calibration.py
Then open http://<this machine's LAN IP>:5002/ from another device on the
same network.

Keyboard (page must have focus, not the distance input box):
  W/S              swing +/-1.0deg
  Shift+W/S        swing +/-5.0deg
  Alt+W/S          swing +/-0.5deg
  A/D              pan +/-2.0deg
  Shift+A/D        pan +/-10.0deg
  Alt+A/D          pan +/-0.5deg
  Space            fire at current swing
  Escape           E-STOP - disable both motors (safe to power off / adjust by hand)

Xbox controller (optional - auto-detected at startup, paired once over
Bluetooth, falls back silently to keyboard/web-only if none is found):
  D-pad left/right   pan +/-1.0deg per click
  D-pad up/down      swing +/-1.0deg per click
  LB + D-pad         +/-0.1deg per click (fine)
  LT + D-pad         +/-5.0deg per click (coarse)
  RT                 fire at current swing
  B                  E-STOP
  A                  quit the whole server (same as Ctrl+C in the terminal)
  Y / X              next / previous cup preset - drives PAN there and sets
                     swing_deg immediately, ready to fire

Cup presets (CUP_PRESETS below): pre-measured (pan_deg, swing_deg) pairs for
the current physical cup layout. Tap 1-4 in the web UI, or cycle with Y/X on
the controller, to aim at a specific cup - then RT (controller) or Space
(keyboard) to fire.
"""
import json
import threading
import time
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, request
from odrive.enums import AxisState

from pong_common import (
    CAM_INDEX, MAX_RANGE_TURNS, PIVOT_TO_BASE_IN, WATCHDOG_FEED_INTERVAL,
    ColorCalibration, WatchdogFeeder, connect_launch, connect_pan, detect_target,
    fire_shot, load_range_calibration, lock_pan, monotone_interpolant, move_pan_to, prepare_launch,
)
from web_click_to_shoot import (
    CameraWorker, DEFAULT_MANUAL_SWING_DEG, MAX_MANUAL_SWING_DEG, MIN_MANUAL_SWING_DEG,
    COLOR_CUP, COLOR_TEXT, action_lock, busy_event, draw_chip,
)
from web_cup_pixel_calibration import CENTERED_DEADBAND_PX, COLOR_GOOD

HOST = "0.0.0.0"
PORT = 5002
STREAM_FPS = 20
JPEG_QUALITY = 75
OUTPUT_JSON = "swing_range_calibration.json"

TAPE_OFFSET_IN = PIVOT_TO_BASE_IN
OFFSET_LABEL = "launcher base"

# Pre-measured cup presets for the current physical layout (2026-09-22) -
# 4 cups, triangle pointing at the launcher: 1=back-left, 2=back-center,
# 3=back-right, 4=mid-row-left. (pan_deg, swing_deg) from a real read at
# each cup, not derived from any calibration curve.
CUP_PRESETS = [
    {"label": "Cup 1 (back-left)", "pan_deg": -4.9, "swing_deg": 83.8},
    {"label": "Cup 2 (back-center)", "pan_deg": -2.3, "swing_deg": 83.7},
    {"label": "Cup 3 (back-right)", "pan_deg": 0.2, "swing_deg": 83.6},
    {"label": "Cup 4 (mid-left)", "pan_deg": -4.4, "swing_deg": 82.0},
]

SWING_STEP_FINE = 0.5
SWING_STEP_MED = 1.0
SWING_STEP_COARSE = 5.0

PAN_STEP_FINE = 0.5
PAN_STEP_MED = 2.0
PAN_STEP_COARSE = 10.0

# --- Xbox controller (optional - falls back to keyboard/web-only if absent) -
# Confirmed via xbox_controller_test.py on this exact Series X/S pad over
# Bluetooth (2026-09-22): button0=A, button1=B, axis4=RT/axis5=LT (rest -1.0,
# full pull +1.0) - swapped from the textbook USB SDL mapping (usually
# axis2=LT/axis5=RT), so don't trust that convention on this pad without
# re-checking. hat0=D-pad: (-1,0)=left (1,0)=right (0,1)=up (0,-1)=down
# (0,0)=released.
#
# Switched from analog-stick jogging to D-pad + modifier stepping
# (2026-09-22): the right stick idled at a real, persistent ~-0.85 on this
# pad instead of near 0, and even compensating for that measured center it
# still wasn't reliable enough - digital single-step-per-click is simpler
# and more predictable than fighting stick calibration further.
#
PAD_BUTTON_LB = 6              # confirmed 2026-09-22 - also non-standard on this pad
PAD_BUTTON_ESTOP = 1           # B
PAD_BUTTON_QUIT = 0            # A - shuts down the whole server (same as Ctrl+C), not just estop
# Indices confirmed 2026-09-22 (X=3, Y=4 - also non-standard on this pad,
# same as LB/RT/LT above).
PAD_BUTTON_PRESET_NEXT = 4     # Y - cycle to the next cup preset and move there
PAD_BUTTON_PRESET_PREV = 3     # X - cycle to the previous cup preset and move there
PAD_TRIGGER_FIRE = 4           # RT (swapped from typical USB mapping - see above)
PAD_TRIGGER_FIRE_THRESHOLD = 0.0   # halfway through the pull, rest=-1/full=+1
PAD_TRIGGER_LT = 5              # LT (coarse-step modifier)
PAD_TRIGGER_LT_THRESHOLD = 0.0
PAD_POLL_INTERVAL_S = 0.05

PAD_DPAD_STEP_DEG = 1.0            # plain click
PAD_DPAD_STEP_FINE_DEG = 0.1       # LB held
PAD_DPAD_STEP_COARSE_DEG = 5.0     # LT held


def make_controller_reader(handle_pan_step, handle_swing_step, handle_fire, handle_estop,
                            handle_preset_cycle):
    """Returns a thread target that polls an Xbox controller (if one is
    connected) and drives the same handler functions the keyboard/web UI
    already use - no separate jog logic, no HTTP roundtrip (same process).
    Missing pygame or no paired controller is not fatal - keyboard/web
    control keeps working either way, this just prints why and returns."""
    def run(stop_event):
        import os
        import signal
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        try:
            import pygame
        except ImportError:
            print("pygame not installed - controller input disabled (keyboard/web still work).")
            return

        pygame.display.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("No controller found - controller input disabled (keyboard/web still work).")
            return

        js = pygame.joystick.Joystick(0)
        js.init()
        print(f"Controller connected: {js.get_name()} - D-pad up/down=swing, left/right=pan "
              f"({PAD_DPAD_STEP_DEG:.1f}deg/click, hold LB={PAD_DPAD_STEP_FINE_DEG:.1f}deg, "
              f"hold LT={PAD_DPAD_STEP_COARSE_DEG:.1f}deg), RT=fire, B=E-STOP, A=quit server, "
              "Y/X=next/prev cup preset.")

        last_fire_pressed = False
        last_estop_btn = 0
        last_quit_btn = 0
        last_preset_next_btn = 0
        last_preset_prev_btn = 0
        last_hat = (0, 0)

        while not stop_event.is_set():
            pygame.event.pump()

            fire_pressed = js.get_axis(PAD_TRIGGER_FIRE) > PAD_TRIGGER_FIRE_THRESHOLD
            if fire_pressed and not last_fire_pressed:
                handle_fire()
            last_fire_pressed = fire_pressed

            estop_btn = js.get_button(PAD_BUTTON_ESTOP)
            if estop_btn and not last_estop_btn:
                handle_estop()
            last_estop_btn = estop_btn

            # A sends this process a real SIGINT - the exact same signal a
            # terminal Ctrl+C delivers - so it runs through main()'s normal
            # KeyboardInterrupt/finally shutdown path (motors disabled,
            # camera/feeders stopped) instead of just killing the process
            # and skipping cleanup.
            quit_btn = js.get_button(PAD_BUTTON_QUIT)
            if quit_btn and not last_quit_btn:
                print("\nController A pressed - shutting down (same as Ctrl+C).")
                os.kill(os.getpid(), signal.SIGINT)
            last_quit_btn = quit_btn

            preset_next_btn = js.get_button(PAD_BUTTON_PRESET_NEXT)
            if preset_next_btn and not last_preset_next_btn:
                handle_preset_cycle(1)
            last_preset_next_btn = preset_next_btn

            preset_prev_btn = js.get_button(PAD_BUTTON_PRESET_PREV)
            if preset_prev_btn and not last_preset_prev_btn:
                handle_preset_cycle(-1)
            last_preset_prev_btn = preset_prev_btn

            # Edge-triggered on the transition INTO a direction - one
            # physical click is one step, holding the D-pad does not repeat.
            # Release (back to (0,0)) doesn't itself trigger anything, and
            # re-pressing the same direction after releasing fires again.
            hat = js.get_hat(0)
            if hat != last_hat and hat != (0, 0):
                lb_held = js.get_button(PAD_BUTTON_LB)
                lt_held = js.get_axis(PAD_TRIGGER_LT) > PAD_TRIGGER_LT_THRESHOLD
                step = (PAD_DPAD_STEP_FINE_DEG if lb_held
                        else PAD_DPAD_STEP_COARSE_DEG if lt_held
                        else PAD_DPAD_STEP_DEG)
                hx, hy = hat
                if hx != 0:
                    handle_pan_step(step if hx > 0 else -step)
                if hy != 0:
                    handle_swing_step(step if hy > 0 else -step)
            last_hat = hat

            time.sleep(PAD_POLL_INTERVAL_S)

    return run


def load_existing_samples():
    """Same seed-from-disk pattern as cup_pixel_calibration.py's loader, so
    a second session adds to prior points instead of overwriting them."""
    try:
        with open(OUTPUT_JSON) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return [(s["r_in"], s["swing_deg"], s["measured_in"], s["measured_from"],
              s.get("vision_r_in"))
            for s in data.get("samples", [])]


def fit_and_save(samples, output_json=OUTPUT_JSON):
    """samples: list of (r_in, swing_deg, measured_in, measured_from,
    vision_r_in). vision_r_in is informational only (what Track B's pixel-
    row->distance calibration predicted for the cup at the moment this
    point was recorded, for cross-checking pixel_range_calibration.json
    against your tape measurement) - it plays no part in the fit itself,
    which only ever uses r_in/swing_deg. Returns (ok, message). Duplicate
    r_in (re-recording the same distance after adjusting further) keeps the
    LAST entry for that r_in, not an average - unlike Track B's pixel-row
    duplicates, a re-recorded swing angle at the same distance means the
    earlier one was wrong (bad centering, misjudged landing, etc.), not two
    independently-valid readings worth blending."""
    if len(samples) < 2:
        return False, f"Only {len(samples)} sample(s) - need at least 2 to fit a curve. Not saving."

    by_r = {}
    order = []
    for r_in, swing_deg, measured_in, measured_from, vision_r_in in samples:
        if r_in not in by_r:
            order.append(r_in)
        by_r[r_in] = (swing_deg, measured_in, measured_from, vision_r_in)
    dupes = len(samples) - len(order)

    rs = sorted(order)
    swings = [by_r[r][0] for r in rs]
    monotone_interpolant(rs, swings)  # sanity check it builds without error

    out = {
        "captured_at": datetime.now().isoformat(),
        "method": "direct empirical: swing angle jogged live until the ball lands in a cup "
                  "at the measured r, not solved from slomo-video exit speed/angle. Valid "
                  "only for landing at table height (z=0) - no trajectory-shape model like "
                  "pong_common.py's CAL, so it can't generalize to elevated targets. Not yet "
                  "wired into solve_swing_for_range() or any firing script - this file is "
                  "just the captured/fitted table so far. vision_r_in/vision_error_in are "
                  "Track B's pixel-row->distance prediction at record time and its miss from "
                  "this point's tape-measured r_in - a cross-check of pixel_range_calibration.json, "
                  "kept for reference only, not used by the fit.",
        "samples": [{"r_in": r, "swing_deg": by_r[r][0], "measured_in": by_r[r][1],
                     "measured_from": by_r[r][2], "vision_r_in": by_r[r][3],
                     "vision_error_in": (r - by_r[r][3]) if by_r[r][3] is not None else None}
                    for r in rs],
    }
    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved {len(rs)} point(s) to {output_json}.")
    print("r(in) -> swing(deg) table:")
    for r, s in zip(rs, swings):
        print(f"  {r:6.2f}in -> {s:6.2f}deg")

    summary = f"Saved {len(rs)} point(s) to {output_json}."
    if dupes:
        summary += f" ({dupes} re-recorded distance(s) - kept the latest.)"
    return True, summary


def make_render_fn(calib, app_state, row_to_r):
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
            cv2.line(frame, (frame_cx, cy), (cx, cy), offset_color, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 5, offset_color, -1, cv2.LINE_AA)

            centered_suffix = " (centered)" if centered else ""
            draw_chip(frame, f"h-offset: {h_offset:+d}px{centered_suffix}", 10, 10, color=offset_color)

            # Live cross-check of Track B's pixel-row->distance calibration
            # against whatever you actually tape-measure at this cup
            # placement - lets you sanity-check pixel_range_calibration.json
            # as you go, not the launch physics at all.
            if row_to_r is not None:
                base_row = det["base"][1]
                try:
                    vision_r_in = row_to_r(base_row)
                    draw_chip(frame, f"row {base_row}px -> vision r: {vision_r_in:.1f}in",
                              10, 44, color=COLOR_TEXT)
                except Exception:
                    pass
        elif calib.ranges is None:
            draw_chip(frame, "tap the cup to calibrate color", 10, 10, color=COLOR_TEXT)
        else:
            draw_chip(frame, "no target detected", 10, 10, color=COLOR_TEXT)

        draw_chip(frame, f"swing: {app_state['swing_deg']:.1f}deg", 10, 78, color=COLOR_TEXT)
        draw_chip(frame, f"pan: {app_state['pan_deg']:+.1f}deg", 10, 112, color=COLOR_TEXT)

        return frame

    return render


INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover">
<title>Range Calibration</title>
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
  .side-readout {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    padding: 14px 18px;
    border-radius: 20px;
    background: rgba(28, 28, 30, 0.65);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    backdrop-filter: blur(20px) saturate(180%);
    pointer-events: none;
  }
  .side-readout .side-label {
    font-size: 14px;
    font-weight: 650;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: var(--label-2);
  }
  .side-readout .side-num {
    font-size: clamp(36px, 8vw, 64px);
    font-weight: 700;
    color: var(--label);
    line-height: 1;
    font-variant-numeric: tabular-nums;
  }
  .side-readout .side-unit {
    font-size: clamp(16px, 3vw, 22px);
    font-weight: 600;
    color: var(--label-2);
  }
  #pan-readout { left: 12px; }
  #swing-readout { right: 12px; }
  #toast {
    position: absolute;
    left: 50%;
    top: calc(env(safe-area-inset-top) + 52px);
    transform: translate(-50%, -8px);
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
    padding: 12px 16px calc(10px + env(safe-area-inset-bottom));
  }

  .field-label { font-size: 12px; font-weight: 590; color: var(--label-2); margin-bottom: 6px; }

  .stepper-grid { display: flex; gap: 8px; margin-bottom: 12px; }
  .stepper-card {
    flex: 1;
    background: var(--fill);
    border-radius: 12px;
    padding: 8px;
    text-align: center;
  }
  .stepper-card .step-label { font-size: 11px; font-weight: 590; color: var(--label-2); margin-bottom: 6px; }
  .stepper-card .step-row { display: flex; gap: 6px; }
  .stepper-card button {
    flex: 1;
    border: none;
    border-radius: 8px;
    background: var(--fill-strong);
    color: var(--label);
    font-size: 15px;
    font-weight: 650;
    font-family: inherit;
    padding: 10px 0;
  }

  .capture-row { display: flex; gap: 10px; margin-bottom: 10px; }
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
  #record-btn {
    border: none;
    border-radius: 12px;
    background: var(--green);
    color: #0a0a0a;
    font-size: 16px;
    font-weight: 650;
    font-family: inherit;
    padding: 0 20px;
  }

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
  #fire-btn { background: var(--orange); color: #1a1a1a; flex: 1.4; }
  #save-btn { color: var(--green); }
  #estop-btn { color: var(--red); flex: 0 0 auto; padding: 12px 16px; }
  .preset-btn.active { color: var(--green); background: var(--fill-strong); }

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
    <div id="samples-pill" class="pill"><span class="num" id="samples-num">0</span><span>points</span></div>
    <div id="pan-readout" class="side-readout">
      <div class="side-label">Pan</div>
      <div><span class="side-num" id="pan-num">+0.0</span><span class="side-unit">&deg;</span></div>
    </div>
    <div id="swing-readout" class="side-readout">
      <div class="side-label">Swing</div>
      <div><span class="side-num" id="swing-num">60.0</span><span class="side-unit">&deg;</span></div>
    </div>
    <div id="toast"></div>
  </div>

  <div id="sheet">
    <div class="field-label">SWING</div>
    <div class="stepper-grid">
      <div class="stepper-card">
        <div class="step-label">0.5&deg;</div>
        <div class="step-row">
          <button data-step="-0.5">&minus;</button>
          <button data-step="0.5">+</button>
        </div>
      </div>
      <div class="stepper-card">
        <div class="step-label">1&deg;</div>
        <div class="step-row">
          <button data-step="-1">&minus;</button>
          <button data-step="1">+</button>
        </div>
      </div>
      <div class="stepper-card">
        <div class="step-label">5&deg;</div>
        <div class="step-row">
          <button data-step="-5">&minus;</button>
          <button data-step="5">+</button>
        </div>
      </div>
    </div>

    <div class="field-label">PAN</div>
    <div class="stepper-grid">
      <div class="stepper-card">
        <div class="step-label">0.5&deg;</div>
        <div class="step-row">
          <button data-pan-step="-0.5">&minus;</button>
          <button data-pan-step="0.5">+</button>
        </div>
      </div>
      <div class="stepper-card">
        <div class="step-label">2&deg;</div>
        <div class="step-row">
          <button data-pan-step="-2">&minus;</button>
          <button data-pan-step="2">+</button>
        </div>
      </div>
      <div class="stepper-card">
        <div class="step-label">10&deg;</div>
        <div class="step-row">
          <button data-pan-step="-10">&minus;</button>
          <button data-pan-step="10">+</button>
        </div>
      </div>
    </div>

    <div class="field-label">CUP PRESET</div>
    <div class="btn-row" id="preset-row" style="margin-bottom: 12px;">
      <button class="tinted-btn preset-btn" data-preset-idx="0">1</button>
      <button class="tinted-btn preset-btn" data-preset-idx="1">2</button>
      <button class="tinted-btn preset-btn" data-preset-idx="2">3</button>
      <button class="tinted-btn preset-btn" data-preset-idx="3">4</button>
    </div>

    <div class="capture-row">
      <input id="distance-input" type="number" inputmode="decimal" step="0.1" placeholder="measured r (in)">
      <button id="record-btn">Record</button>
    </div>
    <div class="btn-row">
      <button id="fire-btn" class="tinted-btn">Fire</button>
      <button id="undo-btn" class="tinted-btn">Undo</button>
      <button id="save-btn" class="tinted-btn">Save</button>
      <button id="estop-btn" class="tinted-btn">Stop</button>
    </div>
  </div>

<script>
const feed = document.getElementById('feed');
const stage = document.getElementById('stage');
const distanceInput = document.getElementById('distance-input');

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
      || s.includes('not a number') || s.includes('nothing to undo') || s.includes('need at least')
      || s.includes('e-stop') || s.includes('did not complete');
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

function flashResult(el, ok) {
  el.classList.remove('flash-good', 'flash-bad');
  void el.offsetWidth;
  el.classList.add(ok ? 'flash-good' : 'flash-bad');
}

document.querySelectorAll('[data-step]').forEach(btn => {
  btn.addEventListener('click', () => post('/swing_step', {delta: parseFloat(btn.dataset.step)}));
});
document.querySelectorAll('[data-pan-step]').forEach(btn => {
  btn.addEventListener('click', () => post('/pan_step', {delta: parseFloat(btn.dataset.panStep)}));
});

document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => post('/preset_goto', {idx: parseInt(btn.dataset.presetIdx)}));
});

const fireBtn = document.getElementById('fire-btn');
fireBtn.addEventListener('click', async () => {
  const j = await post('/fire');
  if (j) flashResult(fireBtn, !isErrorMessage(j.message || ''));
});

document.getElementById('record-btn').addEventListener('click', async () => {
  const v = parseFloat(distanceInput.value);
  if (isNaN(v)) { showToast('Enter the measured distance first', true); return; }
  const j = await post('/record', {distance_in: v});
  if (j) {
    const ok = !isErrorMessage(j.message || '');
    flashResult(document.getElementById('record-btn'), ok);
    if (ok) distanceInput.value = '';
  }
});
distanceInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.stopPropagation(); document.getElementById('record-btn').click(); } });

document.getElementById('undo-btn').addEventListener('click', () => post('/undo'));
document.getElementById('save-btn').addEventListener('click', async () => {
  const j = await post('/finish');
  if (j) flashResult(document.getElementById('save-btn'), !isErrorMessage(j.message || ''));
});
document.getElementById('estop-btn').addEventListener('click', () => post('/estop'));

// Keyboard: ignored while typing in the distance box (Enter there is
// handled separately above, everything else should just type normally).
document.addEventListener('keydown', (e) => {
  if (document.activeElement === distanceInput) return;
  if (e.key === 'Escape') { post('/estop'); return; }
  if (e.key === ' ') { e.preventDefault(); post('/fire'); return; }
  const k = e.key.toLowerCase();
  if (!['w', 's', 'a', 'd'].includes(k)) return;
  e.preventDefault();
  if (k === 'w' || k === 's') {
    let mag = e.shiftKey ? 5.0 : (e.altKey ? 0.5 : 1.0);
    if (k === 's') mag = -mag;
    post('/swing_step', {delta: mag});
  } else {
    let mag = e.shiftKey ? 10.0 : (e.altKey ? 0.5 : 2.0);
    if (k === 'a') mag = -mag;
    post('/pan_step', {delta: mag});
  }
});

const statusPill = document.getElementById('status-pill');
const statusText = document.getElementById('status-text');
const samplesNum = document.getElementById('samples-num');
const swingNum = document.getElementById('swing-num');
const panNum = document.getElementById('pan-num');

async function pollStatus() {
  try {
    const r = await fetch('/status');
    const s = await r.json();
    samplesNum.textContent = s.sample_count;
    swingNum.textContent = s.swing_deg.toFixed(1);
    panNum.textContent = (s.pan_deg >= 0 ? '+' : '') + s.pan_deg.toFixed(1);
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.classList.toggle('active', parseInt(btn.dataset.presetIdx) === s.preset_idx);
    });
    if (!s.calibrated) {
      statusText.textContent = 'Tap the cup to calibrate color';
      statusPill.classList.remove('ready');
    } else if (s.detected) {
      const centered = Math.abs(s.h_offset) < 15;
      const visionPart = s.vision_r_in !== null ? ('vision r: ' + s.vision_r_in.toFixed(1) + 'in  •  ') : '';
      statusText.textContent = visionPart + (s.h_offset > 0 ? '+' : '') + s.h_offset + 'px'
        + (centered ? ' (centered)' : ' - nudge cup to center');
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
    print("TRACK A - direct swing-angle -> range calibration capture (WEB)")
    print("=" * 60)
    print(f"Using offset {TAPE_OFFSET_IN:.2f}in ({OFFSET_LABEL} -> pivot).")
    print("PAN starts locked at center and holds wherever you last jog it to -")
    print("use the h-offset readout to nudge either the cup or PAN onto centerline.\n")

    odrv_pan, axis_pan = connect_pan()
    pan_center = lock_pan(odrv_pan, axis_pan)
    if pan_center is None:
        axis_pan.requested_state = AxisState.IDLE
        return
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

    try:
        row_to_r = load_range_calibration()
        print("Loaded pixel_range_calibration.json - vision r-value will be shown live "
              "for cross-checking against your tape measurements.")
    except FileNotFoundError:
        row_to_r = None
        print("NOTE: pixel_range_calibration.json not found - vision r-value won't be "
              "shown (Track A calibration still works fine without it).")

    calib = ColorCalibration()
    samples = load_existing_samples()
    if samples:
        print(f"Loaded {len(samples)} existing point(s) - this session's captures will be added.")

    app_state = {"swing_deg": DEFAULT_MANUAL_SWING_DEG, "pan_deg": 0.0, "preset_idx": None}
    render_fn = make_render_fn(calib, app_state, row_to_r)
    cam = CameraWorker(CAM_INDEX, render_fn)
    if not cam.opened():
        print("Could not open camera.")
        pan_feeder.stop()
        launch_feeder.stop()
        axis_pan.requested_state = AxisState.IDLE
        axis_launch.requested_state = AxisState.IDLE
        return
    cam.start()

    estop_event_local = {"tripped": False}

    def check_abort():
        return estop_event_local["tripped"]

    def handle_click(x, y):
        frame, _ = cam.latest()
        if frame is None:
            return "No camera frame yet - try again in a moment."
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, sat, val = calib.calibrate(hsv, x, y)
        return f"Color calibrated: H={hue:.0f} S={sat:.0f} V={val:.0f}."

    def handle_swing_step(delta):
        new_val = max(MIN_MANUAL_SWING_DEG, min(MAX_MANUAL_SWING_DEG, app_state["swing_deg"] + delta))
        app_state["swing_deg"] = new_val
        return f"Swing {new_val:.1f}deg"

    def handle_pan_step(delta):
        """Jogs PAN by delta degrees and leaves it there - ordinary closed-
        loop position control already holds the commanded target between
        calls, so there's nothing else needed to make this 'lock in' rather
        than drift; only an explicit jog (or E-STOP) ever changes it."""
        if estop_event_local["tripped"]:
            return "E-STOP is active - restart the server to resume."
        if not action_lock.acquire(blocking=False):
            return "Busy - previous action still in progress."
        busy_event.set()
        try:
            requested = app_state["pan_deg"] + delta
            clamped = max(-MAX_RANGE_TURNS * 360.0, min(MAX_RANGE_TURNS * 360.0, requested))
            target_turns = pan_center + clamped / 360.0
            result = move_pan_to(axis_pan, target_turns, description="pan jog", check_abort=check_abort)
            if result == 'estop':
                return "E-STOP during pan jog."
            app_state["pan_deg"] = clamped
            limit_note = " (at pan limit)" if abs(clamped - requested) > 1e-9 else ""
            if result != True:
                return f"Pan jog timed out (now {clamped:+.1f}deg){limit_note}."
            return f"Pan {clamped:+.1f}deg from center{limit_note}"
        except Exception as e:
            print(f"ERROR during pan jog: {e!r}")
            estop_event_local["tripped"] = True
            try:
                axis_pan.requested_state = AxisState.IDLE
                axis_launch.requested_state = AxisState.IDLE
            except Exception:
                pass
            return f"Hardware communication fault ({e}) - treating this as an E-STOP. Restart the server."
        finally:
            busy_event.clear()
            action_lock.release()

    def handle_goto_preset(idx):
        """Drives PAN to a preset's stored pan_deg and sets swing_deg to its
        stored value - one call readies both axes to fire at a specific
        pre-measured cup. Same 'set the number, LAUNCH only moves on Fire'
        pattern handle_swing_step already uses for swing."""
        if estop_event_local["tripped"]:
            return "E-STOP is active - restart the server to resume."
        if not (0 <= idx < len(CUP_PRESETS)):
            return "Invalid preset index."
        if not action_lock.acquire(blocking=False):
            return "Busy - previous action still in progress."
        busy_event.set()
        try:
            preset = CUP_PRESETS[idx]
            clamped = max(-MAX_RANGE_TURNS * 360.0, min(MAX_RANGE_TURNS * 360.0, preset["pan_deg"]))
            target_turns = pan_center + clamped / 360.0
            result = move_pan_to(axis_pan, target_turns, description="preset pan", check_abort=check_abort)
            if result == 'estop':
                return "E-STOP during preset move."
            app_state["pan_deg"] = clamped
            app_state["swing_deg"] = max(MIN_MANUAL_SWING_DEG, min(MAX_MANUAL_SWING_DEG, preset["swing_deg"]))
            app_state["preset_idx"] = idx
            if result != True:
                return f"{preset['label']}: pan move timed out (now {clamped:+.1f}deg)."
            return f"{preset['label']}: ready (pan {clamped:+.1f}deg, swing {app_state['swing_deg']:.1f}deg)"
        except Exception as e:
            print(f"ERROR during preset move: {e!r}")
            estop_event_local["tripped"] = True
            try:
                axis_pan.requested_state = AxisState.IDLE
                axis_launch.requested_state = AxisState.IDLE
            except Exception:
                pass
            return f"Hardware communication fault ({e}) - treating this as an E-STOP. Restart the server."
        finally:
            busy_event.clear()
            action_lock.release()

    def handle_preset_cycle(step):
        """step=+1/-1, wraps around. Immediately drives to the newly
        selected preset - matches the demo flow of one press meaning 'aim
        at the next cup', ready to fire right after."""
        current = app_state["preset_idx"]
        new_idx = 0 if current is None else (current + step) % len(CUP_PRESETS)
        return handle_goto_preset(new_idx)

    def handle_fire():
        if estop_event_local["tripped"]:
            return "E-STOP is active - restart the server to resume."
        if not action_lock.acquire(blocking=False):
            return "Busy - previous action still in progress."
        busy_event.set()
        try:
            result = fire_shot(axis_launch, start_pos, app_state["swing_deg"], check_abort=check_abort)
            if result == 'estop':
                return "E-STOP during fire."
            if result != 'done':
                return f"Shot did not complete cleanly ({result})."
            return f"Fired at {app_state['swing_deg']:.1f}deg."
        except Exception as e:
            print(f"ERROR during fire: {e!r}")
            estop_event_local["tripped"] = True
            try:
                axis_pan.requested_state = AxisState.IDLE
                axis_launch.requested_state = AxisState.IDLE
            except Exception:
                pass
            return f"Hardware communication fault ({e}) - treating this as an E-STOP. Restart the server."
        finally:
            busy_event.clear()
            action_lock.release()

    def handle_record(distance_in):
        true_r_in = distance_in + TAPE_OFFSET_IN
        swing_deg = app_state["swing_deg"]

        # Fresh vision read at the moment of recording, so the comparison
        # reflects the cup wherever it's actually sitting right now - a
        # cross-check of Track B's pixel-row->distance calibration
        # (pixel_range_calibration.json) against this tape measurement, not
        # anything to do with the launch physics.
        vision_r_in = None
        if row_to_r is not None and calib.ranges is not None:
            frame, _ = cam.latest()
            if frame is not None:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                det = detect_target(hsv, calib)
                if det is not None:
                    try:
                        vision_r_in = row_to_r(det["base"][1])
                    except Exception:
                        pass

        samples.append((true_r_in, swing_deg, distance_in, OFFSET_LABEL, vision_r_in))
        compare = ""
        if vision_r_in is not None:
            error_in = true_r_in - vision_r_in
            compare = f" (vision predicted {vision_r_in:.1f}in, off by {error_in:+.1f}in)"
        return (f"Recorded: {distance_in:.1f}in from {OFFSET_LABEL} -> r={true_r_in:.2f}in "
                f"at swing={swing_deg:.1f}deg{compare} ({len(samples)} total)")

    def handle_undo():
        if not samples:
            return "Nothing to undo."
        popped = samples.pop()
        return f"Undid point at r={popped[0]:.2f}in ({len(samples)} remaining)."

    def handle_finish():
        ok, message = fit_and_save(list(samples))
        return message

    def handle_estop():
        estop_event_local["tripped"] = True
        axis_pan.requested_state = AxisState.IDLE
        axis_launch.requested_state = AxisState.IDLE
        return "E-STOP: both motors disabled. Restart the server to resume."

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
        return jsonify({"message": handle_click(int(data["x"]), int(data["y"]))})

    @app.route("/swing_step", methods=["POST"])
    def swing_step_route():
        data = request.get_json(force=True)
        try:
            delta = float(data["delta"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"message": "Invalid step."}), 400
        return jsonify({"message": handle_swing_step(delta)})

    @app.route("/fire", methods=["POST"])
    def fire_route():
        return jsonify({"message": handle_fire()})

    @app.route("/pan_step", methods=["POST"])
    def pan_step_route():
        data = request.get_json(force=True)
        try:
            delta = float(data["delta"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"message": "Invalid step."}), 400
        return jsonify({"message": handle_pan_step(delta)})

    @app.route("/preset_goto", methods=["POST"])
    def preset_goto_route():
        data = request.get_json(force=True)
        try:
            idx = int(data["idx"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"message": "Invalid preset index."}), 400
        return jsonify({"message": handle_goto_preset(idx)})

    @app.route("/preset_next", methods=["POST"])
    def preset_next_route():
        return jsonify({"message": handle_preset_cycle(1)})

    @app.route("/preset_prev", methods=["POST"])
    def preset_prev_route():
        return jsonify({"message": handle_preset_cycle(-1)})

    @app.route("/record", methods=["POST"])
    def record_route():
        data = request.get_json(force=True)
        try:
            distance_in = float(data["distance_in"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"message": "Not a number - point discarded."})
        return jsonify({"message": handle_record(distance_in)})

    @app.route("/undo", methods=["POST"])
    def undo_route():
        return jsonify({"message": handle_undo()})

    @app.route("/finish", methods=["POST"])
    def finish_route():
        return jsonify({"message": handle_finish()})

    @app.route("/estop", methods=["POST"])
    def estop_route():
        return jsonify({"message": handle_estop()})

    @app.route("/status")
    def status_route():
        frame, _ = cam.latest()
        h_offset = None
        vision_r_in = None
        detected = False
        if frame is not None and calib.ranges is not None:
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            det = detect_target(hsv, calib)
            if det is not None:
                detected = True
                h_offset = det["centroid"][0] - w // 2
                if row_to_r is not None:
                    try:
                        vision_r_in = row_to_r(det["base"][1])
                    except Exception:
                        pass
        return jsonify({
            "calibrated": calib.ranges is not None,
            "detected": detected,
            "h_offset": h_offset,
            "vision_r_in": vision_r_in,
            "swing_deg": app_state["swing_deg"],
            "pan_deg": app_state["pan_deg"],
            "preset_idx": app_state["preset_idx"],
            "sample_count": len(samples),
        })

    stop_controller = threading.Event()
    controller_thread = threading.Thread(
        target=make_controller_reader(handle_pan_step, handle_swing_step, handle_fire, handle_estop,
                                       handle_preset_cycle),
        args=(stop_controller,), daemon=True)
    controller_thread.start()

    print(f"\nServing on http://{HOST}:{PORT}/ - open that from a browser on the same network.")
    print("Tap the cup to calibrate color, place it at a measured distance, nudge it to")
    print("center (h-offset), jog swing + fire until it lands in the cup, Record, repeat.\n")

    try:
        app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        stop_controller.set()
        cam.stop()
        pan_feeder.stop()
        launch_feeder.stop()
        print("Disabling PAN and LAUNCH...")
        axis_pan.requested_state = AxisState.IDLE
        axis_launch.requested_state = AxisState.IDLE
        print("Both motors disabled. Safe to power off.")


if __name__ == "__main__":
    main()
