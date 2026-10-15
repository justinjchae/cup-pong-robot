"""
Track B calibration-capture script: builds the pixel-row(of cup base)-to-range
calibration curve needed for vision-guided aiming. Shared
PAN-safety and click-to-calibrate detection logic lives in pong_common.py.

Flow: PAN auto-locks to the saved center position (same safety-bounded
raw-anchored pattern as range_test.py/combined_control.py - never moves
again after that, this script doesn't need PAN motion beyond the initial
lock). Camera opens with the mount-rotation correction applied. Click once
on the cup in the live window to calibrate its HSV color (same classical
contour/largest-blob detection as orange_tracker.py, but session-
parameterized instead of a hardcoded BLUE_LOWER/UPPER). Then:
  place the cup at a tape-measured distance
  -> press C to capture the detected cup-BASE (bottom of bounding box,
     NOT centroid - centroid bakes in a perspective-dependent bias)
     pixel row
  -> type the measured distance at the terminal
  -> repeat across ~6-8 distances spanning the working range (~2-8ft)
  -> press Q (in the video window) to fit a monotone-cubic
     pixel-row -> range curve and save it to OUTPUT_JSON.

Requires: pip install opencv-python numpy
Run with: odrive_env/bin/python cup_pixel_calibration.py
"""
import json
from datetime import datetime

import cv2
from odrive.enums import AxisState

from pong_common import (
    CAM_INDEX, ROTATE_CODE, HUE_TOL, PIVOT_TO_BASE_IN, PIVOT_TO_TAPE_START_IN,
    WATCHDOG_FEED_INTERVAL,
    ColorCalibration, WatchdogFeeder, connect_pan, detect_target, lock_pan,
    monotone_interpolant,
)

OUTPUT_JSON = "pixel_range_calibration.json"


def load_existing_samples():
    """Seeds this session with any samples already saved to OUTPUT_JSON, so
    a second calibration pass (e.g. filling in near-range points) adds to
    prior data instead of overwriting it. Tolerates the old schema (a plain
    'tape_in' field, implicitly tape-start-referenced) alongside the new
    'measured_in'/'measured_from' fields."""
    try:
        with open(OUTPUT_JSON) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    out = []
    for s in data.get("samples", []):
        measured_in = s.get("measured_in", s.get("tape_in"))
        measured_from = s.get("measured_from", "tape/ruler start" if "tape_in" in s else None)
        out.append((s["pixel_row"], measured_in, measured_from, s["distance_in"]))
    return out


def fit_and_save(samples, output_json=OUTPUT_JSON):
    """Shared fit+save logic for both this CLI script and
    web_cup_pixel_calibration.py, so the two can't drift apart (same
    discipline as pong_common.py's shared hardware/vision functions).
    samples is a list of (pixel_row, measured_in, measured_from, distance_in).
    Returns (ok, message) - ok=False means nothing was saved."""
    if len(samples) < 2:
        return False, f"Only {len(samples)} sample(s) - need at least 2 to fit a curve. Not saving."

    samples_sorted = sorted(samples, key=lambda s: s[0])
    rows = [s[0] for s in samples_sorted]
    true_rs = [s[3] for s in samples_sorted]

    # monotone_interpolant divides by consecutive-row spacing, so duplicate
    # rows (two captures landing on the same pixel, common near the close
    # end where sensitivity is high) need collapsing before the fit, not
    # just a warning - average their r values into one point.
    dupe_warnings = []
    if len(set(rows)) != len(rows):
        merged = {}
        for r, ri in zip(rows, true_rs):
            merged.setdefault(r, []).append(ri)
        dupes = {r: vs for r, vs in merged.items() if len(vs) > 1}
        for r, vs in dupes.items():
            msg = (f"{len(vs)} samples at row {r}px (r={vs}) - averaging to "
                   f"{sum(vs) / len(vs):.2f}in for the fit. Raw samples are still all saved.")
            print(f"\nWARNING: {msg}")
            dupe_warnings.append(msg)
        rows = sorted(merged)
        true_rs = [sum(merged[r]) / len(merged[r]) for r in rows]

    monotone_interpolant(rows, true_rs)  # sanity check it builds without error

    out = {
        "captured_at": datetime.now().isoformat(),
        "cam_index": CAM_INDEX,
        "rotate_code": "ROTATE_90_CLOCKWISE" if ROTATE_CODE == cv2.ROTATE_90_CLOCKWISE
                        else ("ROTATE_90_COUNTERCLOCKWISE" if ROTATE_CODE == cv2.ROTATE_90_COUNTERCLOCKWISE
                              else str(ROTATE_CODE)),
        "note": "distance_in is true pivot-referenced r (measured_in + this sample's offset "
                "for whichever point measured_from names); samples may mix reference points "
                "across sessions, distance_in is always the pivot-referenced value to use. "
                "measured_in is always measured to the cup's FRONT edge (near wall, at table "
                "level), matching the 'base' pixel (bottom of the cup's silhouette) - not the "
                "cup's center. This does not yet account for cup height/rim clearance - r is "
                "purely a ground-plane distance, z is assumed 0.",
        "samples": [{"pixel_row": r, "measured_in": m, "measured_from": mf, "distance_in": ri}
                     for r, m, mf, ri in samples_sorted],
    }
    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved {len(samples)} samples to {output_json}.")
    print("Pixel row -> true r(in) table:")
    for r, ri in zip(rows, true_rs):
        print(f"  {r:4d}px -> {ri:6.2f}in")

    summary = f"Saved {len(samples)} samples to {output_json}."
    if dupe_warnings:
        summary += f" ({len(dupe_warnings)} duplicate-row group(s) averaged.)"
    return True, summary


def choose_tape_offset():
    print("\nWhich point are tape/ruler distances measured from this session?")
    print(f"  1) Launcher base       (offset = {PIVOT_TO_BASE_IN:.2f}in)")
    print(f"  2) Tape/ruler start    (offset = {PIVOT_TO_TAPE_START_IN:.2f}in, same as range_test.py)")
    while True:
        choice = input("  Enter 1 or 2: ").strip()
        if choice == '1':
            return PIVOT_TO_BASE_IN, "launcher base"
        if choice == '2':
            return PIVOT_TO_TAPE_START_IN, "tape/ruler start"
        print("  Please enter 1 or 2.")


# --- Main ---------------------------------------------------------------
def main():
    print("=" * 60)
    print("TRACK B - pixel-row(cup base) -> range calibration capture")
    print("=" * 60)

    tape_offset_in, offset_label = choose_tape_offset()
    print(f"Using offset {tape_offset_in:.2f}in ({offset_label} -> pivot).")

    odrv_pan, axis_pan = connect_pan()
    lock_target = lock_pan(odrv_pan, axis_pan)
    if lock_target is None:
        axis_pan.requested_state = AxisState.IDLE
        return

    feeder = WatchdogFeeder(axis_pan, WATCHDOG_FEED_INTERVAL)
    feeder.start()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open camera.")
        feeder.stop()
        axis_pan.requested_state = AxisState.IDLE
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    calib = ColorCalibration()
    samples = load_existing_samples()  # list of (pixel_row, measured_in, measured_from, distance_in)
    if samples:
        print(f"Loaded {len(samples)} existing sample(s) from {OUTPUT_JSON} - "
              "this session's captures will be added to them.")
    latest_detection = {"result": None}
    mouse_state = {"frame": None}

    window = "Cup Calibration"
    cv2.namedWindow(window)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and userdata["frame"] is not None:
            hsv = cv2.cvtColor(userdata["frame"], cv2.COLOR_BGR2HSV)
            h, s, v = calib.calibrate(hsv, x, y)
            print(f"\nColor calibrated from click at ({x},{y}): "
                  f"H={h:.0f} S={s:.0f} V={v:.0f} (+/- H{HUE_TOL})")

    cv2.setMouseCallback(window, on_mouse, mouse_state)

    print("\nCheck the preview window: if it looks sideways or upside-down,")
    print("edit ROTATE_CODE in pong_common.py and rerun.")
    print("\nClick on the cup in the video window to calibrate its color.")
    print("Then, for each measured distance:")
    print("  place the cup -> press C to capture -> type the distance (in) to the cup's")
    print("  FRONT edge (near wall, at table level - matches the 'base' pixel below,")
    print("  which is the bottom of the cup's silhouette, i.e. its near-bottom edge, not")
    print("  its center) at this terminal")
    print("R - reset color calibration     U - undo last sample")
    print("Q or ESC (in video window) - finish, fit curve, and save\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break
            if ROTATE_CODE is not None:
                frame = cv2.rotate(frame, ROTATE_CODE)
            mouse_state["frame"] = frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            det = detect_target(hsv, calib)
            latest_detection["result"] = det

            h, w = frame.shape[:2]
            frame_cx = w // 2

            # Dashed vertical center line - not physically required (pixel-row
            # is horizontal-position-invariant since the camera only pans),
            # but keeps calibration conditions
            # consistent with how Phase 3 will read r (after theta-centering)
            # and avoids extra lens distortion near the frame edges.
            for yy in range(0, h, 16):
                cv2.line(frame, (frame_cx, yy), (frame_cx, min(yy + 8, h)), (180, 180, 180), 1)

            if det is not None:
                x, y, bw, bh = det["bbox"]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 255), 2)
                bx, by = det["base"]
                cv2.circle(frame, (bx, by), 6, (0, 0, 255), -1)
                cx, cy = det["centroid"]
                cv2.line(frame, (frame_cx, cy), (cx, cy), (0, 200, 255), 1)
                offset_px = cx - frame_cx
                offset_color = (0, 220, 0) if abs(offset_px) < 15 else (0, 200, 255)
                cv2.putText(frame, f"base row: {by}px", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"h-offset: {offset_px:+d}px {'(centered)' if abs(offset_px) < 15 else ''}",
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, offset_color, 2)
            else:
                msg = "no target detected" if calib.ranges else "click cup to calibrate color"
                cv2.putText(frame, msg, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

            cv2.putText(frame, f"samples: {len(samples)}", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                calib.ranges = None
                print("Color calibration reset - click the cup again.")
            elif key == ord('u'):
                if samples:
                    print(f"Undid sample: {samples.pop()}")
                else:
                    print("No samples to undo.")
            elif key == ord('c'):
                det = latest_detection["result"]
                if det is None:
                    print("No target currently detected - can't capture. "
                          "Check color calibration / cup placement.")
                    continue
                pixel_row = det["base"][1]
                print(f"\nCaptured pixel row {pixel_row}. Type the measured distance "
                      f"to the cup's FRONT edge (near wall, at table level) from the "
                      f"{offset_label} (inches) and press Enter:")
                raw = input("  measured distance to front edge (in): ").strip()
                try:
                    measured_in = float(raw)
                except ValueError:
                    print("  Not a number - sample discarded.")
                    continue
                true_r_in = measured_in + tape_offset_in
                samples.append((pixel_row, measured_in, offset_label, true_r_in))
                print(f"  Sample recorded: row={pixel_row}px -> measured {measured_in:.1f}in "
                      f"from {offset_label} -> true r {true_r_in:.2f}in ({len(samples)} total)")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        feeder.stop()
        print("Disabling PAN...")
        axis_pan.requested_state = AxisState.IDLE
        print("PAN disabled. Safe to power off.")

    ok, message = fit_and_save(samples)
    print(f"\n{message}")
    if ok:
        print("\nLoad this JSON's pixel_row/distance_in and pass into monotone_interpolant() "
              "(same function as range_test.py) in the Phase 3 targeting script.")


if __name__ == "__main__":
    main()
