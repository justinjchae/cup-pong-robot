"""
One-time pan pixel-offset -> angular-correction calibration. Replaces the
old approach in pan_aim.py of re-deriving this relationship live on every
single aim via probing, which turned out to be fighting real detection
noise at fine scales (the secant search
diverged on real hardware even after damping/re-probe fixes). Same "measure
it once, then look it up" pattern as Track A's launch calibration and
Track B's pixel-row-to-range curve - this is the pan-axis equivalent.

How it works: place a reference cup roughly centered in view, click to
calibrate its color, then press S. PAN then automatically sweeps through
evenly spaced absolute positions across its safety window (no manual
tape-measuring needed here, unlike the range calibration - PAN's own
commanded angle IS the measurement) and records the reference target's
pixel-x offset at each stop (median of a few frames, to reduce detection
noise). The resulting (angular-offset-from-lock-center, pixel-offset)
curve is saved to OUTPUT_JSON.

At use time: for a NEWLY detected target at pixel offset e while PAN sits
at absolute position p_now, the target's absolute pan angle is
    p_now + phi_of_e(0) - phi_of_e(e)
where phi_of_e is this calibration's fitted curve (see cup_map.py, which
uses this directly - no live probing needed).

Requires: pip install opencv-python numpy
Run with: odrive_env/bin/python theta_calibration.py
"""
import json
from datetime import datetime

import cv2
from odrive.enums import AxisState

from pong_common import (
    CAM_INDEX, ROTATE_CODE, HUE_TOL, MAX_RANGE_TURNS, WATCHDOG_FEED_INTERVAL,
    ColorCalibration, WatchdogFeeder, connect_pan, detect_target, lock_pan,
    median_read, monotone_interpolant, move_pan_to,
)

OUTPUT_JSON = "theta_calibration.json"
SWEEP_STEP_DEG = 3.0
SAMPLES_PER_STOP = 5
MIN_VALID_SAMPLES = 3


def trim_to_contiguous_region(samples, jump_factor=4.0, min_jump_px=100.0):
    """Real sweep data must be smooth and monotonic in pixel-offset as phi
    sweeps, since it's tracking one rigid physical point. A background
    object that happens to match the calibrated color (once the real
    target swings out of frame) shows up as a large discontinuity in the
    (phi, e) sequence. Keeps only the contiguous run of samples around
    phi=0 (guaranteed to be the real target - that's where color was
    calibrated) up to the first such jump in either direction, dropping
    everything past it. Returns (trimmed_samples, dropped_count)."""
    ordered = sorted(samples, key=lambda s: s[0])
    if len(ordered) < 3:
        return ordered, 0
    phis = [s[0] for s in ordered]
    zero_i = min(range(len(phis)), key=lambda i: abs(phis[i]))

    deltas = [abs(ordered[i + 1][1] - ordered[i][1]) for i in range(len(ordered) - 1)]
    median_delta = sorted(deltas)[len(deltas) // 2]
    threshold = max(jump_factor * median_delta, min_jump_px)

    hi = zero_i
    while hi + 1 < len(ordered) and abs(ordered[hi + 1][1] - ordered[hi][1]) <= threshold:
        hi += 1
    lo = zero_i
    while lo - 1 >= 0 and abs(ordered[lo][1] - ordered[lo - 1][1]) <= threshold:
        lo -= 1

    trimmed = ordered[lo:hi + 1]
    return trimmed, len(ordered) - len(trimmed)


def build_sweep_positions(pan_center):
    step_turns = SWEEP_STEP_DEG / 360.0
    n = int(round((2 * MAX_RANGE_TURNS) / step_turns))
    return [pan_center - MAX_RANGE_TURNS + i * step_turns for i in range(n + 1)]


def main():
    print("=" * 60)
    print("TRACK B - pan pixel-offset -> angle calibration (automatic sweep)")
    print("=" * 60)

    odrv_pan, axis_pan = connect_pan()
    pan_center = lock_pan(odrv_pan, axis_pan)
    if pan_center is None:
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
    mouse_state = {"frame": None}

    window = "Theta Calibration"
    cv2.namedWindow(window)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and userdata["frame"] is not None:
            hsv = cv2.cvtColor(userdata["frame"], cv2.COLOR_BGR2HSV)
            h, s, v = calib.calibrate(hsv, x, y)
            print(f"\nColor calibrated from click at ({x},{y}): "
                  f"H={h:.0f} S={s:.0f} V={v:.0f} (+/- H{HUE_TOL})")

    cv2.setMouseCallback(window, on_mouse, mouse_state)

    def grab_and_detect():
        ret, frame = cap.read()
        if not ret:
            return None, None
        if ROTATE_CODE is not None:
            frame = cv2.rotate(frame, ROTATE_CODE)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        det = detect_target(hsv, calib)
        return frame, det

    def read_offset():
        frame, det = grab_and_detect()
        if frame is None or det is None:
            return None
        return det["centroid"][0] - frame.shape[1] // 2

    print("\nPlace a reference cup roughly centered in view, click it to")
    print("calibrate its color, then press S to start the automatic sweep")
    print(f"({SWEEP_STEP_DEG}deg steps across +/-{MAX_RANGE_TURNS * 360:.0f}deg).")
    print("ESC at any time - E-STOP.   Q - quit without saving.\n")

    aborted = False
    samples = None  # list of (phi_turns, pixel_offset)

    try:
        while True:
            frame, det = grab_and_detect()
            if frame is None:
                print("Failed to read frame.")
                break
            mouse_state["frame"] = frame
            h, w = frame.shape[:2]
            frame_cx = w // 2

            for yy in range(0, h, 16):
                cv2.line(frame, (frame_cx, yy), (frame_cx, min(yy + 8, h)), (180, 180, 180), 1)
            if det is not None:
                x, y, bw, bh = det["bbox"]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 255), 2)
                cx, cy = det["centroid"]
                cv2.circle(frame, (cx, cy), 6, (0, 200, 255), -1)
                cv2.putText(frame, f"h-offset: {cx - frame_cx:+d}px", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            else:
                msg = "no target detected" if calib.ranges else "click cup to calibrate color"
                cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            cv2.putText(frame, "press S to start sweep" if calib.ranges else "",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quit without sweeping.")
                break

            elif key == 27:
                axis_pan.requested_state = AxisState.IDLE
                print("\n!!! E-STOP !!! PAN disabled.")
                break

            elif key == ord('s'):
                if calib.ranges is None:
                    print("Click the cup to calibrate color first.")
                    continue

                positions = build_sweep_positions(pan_center)
                print(f"\nSweeping {len(positions)} positions "
                      f"({positions[0]:.4f} to {positions[-1]:.4f} turns)...")
                collected = []
                for i, p in enumerate(positions):
                    move_pan_to(axis_pan, p, description=f"sweep point {i}")

                    # keep the window alive / watch for e-stop during the sweep
                    frame, _ = grab_and_detect()
                    if frame is not None:
                        cv2.putText(frame, f"sweeping... {i + 1}/{len(positions)}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                        cv2.imshow(window, frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:
                        axis_pan.requested_state = AxisState.IDLE
                        print("\n!!! E-STOP !!! Sweep aborted, PAN disabled.")
                        aborted = True
                        break

                    e = median_read(read_offset, samples=SAMPLES_PER_STOP, min_valid=MIN_VALID_SAMPLES)
                    phi = p - pan_center
                    if e is None:
                        print(f"  [{i + 1}/{len(positions)}] phi={phi * 360:+6.1f}deg  "
                              f"no reliable detection - skipped")
                        continue
                    collected.append((phi, e))
                    print(f"  [{i + 1}/{len(positions)}] phi={phi * 360:+6.1f}deg  offset={e:+.0f}px")

                if aborted:
                    break

                print("\nReturning PAN to locked center...")
                move_pan_to(axis_pan, pan_center, description="locked position")
                samples = collected
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        feeder.stop()
        print("Disabling PAN...")
        axis_pan.requested_state = AxisState.IDLE
        print("PAN disabled. Safe to power off.")

    if aborted or not samples:
        print("\nNo sweep data to save.")
        return
    if len(samples) < 4:
        print(f"\nOnly {len(samples)} usable point(s) - need more for a reliable curve. Not saving.")
        return

    samples, n_dropped = trim_to_contiguous_region(samples)
    if n_dropped:
        print(f"\nDropped {n_dropped} sample(s) outside the contiguous region around "
              f"phi=0 - likely a different (background) object picked up by the same "
              f"color match once the real target left frame. Kept phi "
              f"{samples[0][0] * 360:+.1f}deg to {samples[-1][0] * 360:+.1f}deg.")
    if len(samples) < 4:
        print(f"\nOnly {len(samples)} point(s) left after trimming - need more for a "
              "reliable curve. Not saving.")
        return

    # Fit phi as a function of pixel offset e (not the other way around) -
    # this is exactly the curve cup_map.py needs at use time. Duplicate e
    # values (possible near the noise floor) get averaged, same fix as the
    # range-calibration script's duplicate-row handling.
    by_e = {}
    for phi, e in samples:
        by_e.setdefault(e, []).append(phi)
    es = sorted(by_e)
    phis = [sum(by_e[e]) / len(by_e[e]) for e in es]
    if len(es) < 4:
        print(f"\nOnly {len(es)} distinct pixel-offset value(s) after merging duplicates - "
              "not enough for a reliable curve. Not saving.")
        return

    phi_of_e = monotone_interpolant(es, phis)
    phi0 = phi_of_e(0)
    valid_phi_range = [min(phis), max(phis)]

    out = {
        "captured_at": datetime.now().isoformat(),
        "cam_index": CAM_INDEX,
        "pan_center": pan_center,
        "phi0": phi0,
        "valid_phi_range": valid_phi_range,
        "note": "phi is pan offset in turns from the locked center (pan_center). "
                "For a new detection with pixel offset e while PAN is at absolute "
                "position p_now, its absolute target angle is "
                "p_now + phi0 - phi_of_e(e), reconstructing phi_of_e via "
                "monotone_interpolant(samples e-values, samples phi-values). "
                "valid_phi_range is the contiguous phi span the real target was "
                "actually tracked across - callers (cup_map.py) should not trust "
                "detections computed outside it.",
        "samples": [{"phi_turns": phi, "pixel_offset": e} for phi, e in zip(phis, es)],
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved {len(es)} points to {OUTPUT_JSON}.")
    print(f"Pixel-offset range covered: {es[0]:+.0f}px to {es[-1]:+.0f}px")
    print(f"Valid phi range: {valid_phi_range[0] * 360:+.1f}deg to {valid_phi_range[1] * 360:+.1f}deg")
    print(f"phi0 (offset from lock-center where the calibration curve reads zero "
          f"pixel error): {phi0 * 360:+.2f}deg")


if __name__ == "__main__":
    main()
