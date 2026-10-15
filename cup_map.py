"""
Builds a map of every same-colored cup's (theta, r) across the table -
the actual real-gameplay shape of the localization problem (several cups
of the same color, need each one's position, not just "center on the one
blob in frame"). Uses the two one-time calibrations instead of any live
per-target probing:
  - pixel_range_calibration.json (cup_pixel_calibration.py): base-row -> r
  - theta_calibration.json (theta_calibration.py): pixel-offset -> angle

Flow:
  1. PAN locks to center. Click the cup color once (same color for every
     cup in a real game).
  2. Coarse sweep across the PAN safety window, collecting every detected
     blob's pixel offset and the PAN position it was seen at. Converts
     each to an approximate absolute theta via the theta calibration -
     pure arithmetic, no iteration.
  3. Dedupes candidates seen in multiple overlapping frames (same cup,
     different pan angles) by clustering close theta estimates.
  4. For each cluster: pan directly (one open-loop move) to its estimated
     theta, run discrete_pan_aim as a cheap polish (starts near-centered,
     so this is usually 0-1 moves, not the multi-move hunt it was doing
     from a cold start), then read r from that now-centered frame's base
     row - this avoids any off-center perspective bias in the r reading.
  5. Saves/prints the final {theta_deg, r_in} map.

Requires: pip install opencv-python numpy
Run with: odrive_env/bin/python cup_map.py
"""
import json

import cv2
from odrive.enums import AxisState

from pong_common import (
    CAM_INDEX, ROTATE_CODE, HUE_TOL, MAX_RANGE_TURNS, RANGE_CAL_JSON, THETA_CAL_JSON,
    WATCHDOG_FEED_INTERVAL,
    ColorCalibration, WatchdogFeeder, connect_pan, detect_all_targets, discrete_pan_aim,
    load_range_calibration, load_theta_calibration, lock_pan, median_read, move_pan_to,
)

OUTPUT_JSON = "cup_map.json"

COARSE_STEP_DEG = 15.0     # sweep spacing for the initial find-everything pass
CLUSTER_TOL_DEG = 2.0      # merge raw detections into one cup if within this angle
SAMPLES_PER_READ = 5


def theta_from_offset(p_now, e, phi0, phi_of_e):
    return p_now + phi0 - phi_of_e(e)


def cluster_thetas(raw_thetas, tol_turns):
    raw_thetas = sorted(raw_thetas)
    clusters = []
    current = [raw_thetas[0]]
    for t in raw_thetas[1:]:
        if t - current[-1] <= tol_turns:
            current.append(t)
        else:
            clusters.append(current)
            current = [t]
    clusters.append(current)
    return [sum(c) / len(c) for c in clusters]


def main():
    print("=" * 60)
    print("CUP MAP - locate every cup's (theta, r) across the table")
    print("=" * 60)

    try:
        cal_pan_center, phi0, phi_of_e, valid_phi_range = load_theta_calibration()
    except FileNotFoundError:
        print(f"ERROR: {THETA_CAL_JSON} not found - run theta_calibration.py first.")
        return
    print(f"Theta calibration valid for phi in "
          f"[{valid_phi_range[0] * 360:+.1f}deg, {valid_phi_range[1] * 360:+.1f}deg] "
          f"- the coarse sweep will stay within this range.")
    try:
        row_to_r = load_range_calibration()
    except FileNotFoundError:
        print(f"ERROR: {RANGE_CAL_JSON} not found - run cup_pixel_calibration.py first.")
        return

    odrv_pan, axis_pan = connect_pan()
    pan_center = lock_pan(odrv_pan, axis_pan)
    if pan_center is None:
        axis_pan.requested_state = AxisState.IDLE
        return
    if abs(pan_center - cal_pan_center) > 0.01:
        print(f"WARNING: current lock center ({pan_center:.4f}) differs from the one "
              f"theta_calibration.json was built with ({cal_pan_center:.4f}) by "
              f"{(pan_center - cal_pan_center) * 360:.1f}deg - consider recalibrating.")

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

    window = "Cup Map"
    cv2.namedWindow(window)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and userdata["frame"] is not None:
            hsv = cv2.cvtColor(userdata["frame"], cv2.COLOR_BGR2HSV)
            h, s, v = calib.calibrate(hsv, x, y)
            print(f"\nColor calibrated from click at ({x},{y}): "
                  f"H={h:.0f} S={s:.0f} V={v:.0f} (+/- H{HUE_TOL})")

    cv2.setMouseCallback(window, on_mouse, mouse_state)

    def grab(rotate=True):
        ret, frame = cap.read()
        if not ret:
            return None
        if rotate and ROTATE_CODE is not None:
            frame = cv2.rotate(frame, ROTATE_CODE)
        return frame

    def detect_all_now():
        frame = grab()
        if frame is None:
            return None, []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return frame, detect_all_targets(hsv, calib)

    print("\nClick a cup in the video window to calibrate its color "
          "(same color assumed for every cup). Press S to start.")
    print("ESC - E-STOP     Q - quit without mapping\n")

    aborted = False
    clusters = None

    try:
        while True:
            frame, dets = detect_all_now()
            if frame is None:
                print("Failed to read frame.")
                break
            mouse_state["frame"] = frame
            h, w = frame.shape[:2]
            frame_cx = w // 2

            for d in dets:
                x, y, bw, bh = d["bbox"]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 255), 2)
            if not dets:
                msg = "no target detected" if calib.ranges else "click a cup to calibrate color"
                cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            else:
                cv2.putText(frame, f"{len(dets)} blob(s) in view", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(frame, "press S to map the table" if calib.ranges else "",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quit without mapping.")
                break

            elif key == 27:
                axis_pan.requested_state = AxisState.IDLE
                print("\n!!! E-STOP !!! PAN disabled.")
                break

            elif key == ord('s'):
                if calib.ranges is None:
                    print("Click a cup to calibrate color first.")
                    continue

                # --- Step 1: coarse sweep, collect raw theta candidates ---
                # Bounded by the theta calibration's own validated phi range,
                # not the full PAN safety window - sweeping further than the
                # calibration was actually measured across is exactly how the
                # background-contamination problem happened last time.
                sweep_lo = pan_center + valid_phi_range[0]
                sweep_hi = pan_center + valid_phi_range[1]
                step_turns = COARSE_STEP_DEG / 360.0
                n = max(1, int(round((sweep_hi - sweep_lo) / step_turns)))
                positions = [sweep_lo + i * (sweep_hi - sweep_lo) / n for i in range(n + 1)]

                print(f"\nCoarse sweep: {len(positions)} positions, {COARSE_STEP_DEG}deg apart...")
                raw_thetas = []
                for i, p in enumerate(positions):
                    move_pan_to(axis_pan, p, description=f"sweep point {i}")
                    frame, dets = detect_all_now()
                    if frame is not None:
                        cv2.putText(frame, f"sweeping... {i + 1}/{len(positions)}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                        cv2.imshow(window, frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:
                        axis_pan.requested_state = AxisState.IDLE
                        print("\n!!! E-STOP !!! Mapping aborted, PAN disabled.")
                        aborted = True
                        break
                    frame_cx = frame.shape[1] // 2 if frame is not None else None
                    for d in dets:
                        e = d["centroid"][0] - frame_cx
                        theta = theta_from_offset(p, e, phi0, phi_of_e)
                        # Defensive: a detection whose computed theta falls
                        # outside the calibration's own validated span isn't
                        # trustworthy (extrapolated past where phi_of_e was
                        # actually measured) - drop it rather than risk
                        # another phantom cup.
                        if not (sweep_lo - 0.01 <= theta <= sweep_hi + 0.01):
                            continue
                        raw_thetas.append(theta)
                    print(f"  [{i + 1}/{len(positions)}] pos={p:.4f} turns, {len(dets)} blob(s)")
                if aborted:
                    break

                if not raw_thetas:
                    print("\nNo cups detected anywhere in the sweep.")
                    move_pan_to(axis_pan, pan_center, description="locked position")
                    break

                # --- Step 2: dedupe overlapping detections into cups ---
                clusters = cluster_thetas(raw_thetas, CLUSTER_TOL_DEG / 360.0)
                print(f"\n{len(raw_thetas)} raw detection(s) -> {len(clusters)} candidate cup(s).")

                # --- Step 3: precision pass per cup ---
                results = []
                for ci, theta_guess in enumerate(clusters):
                    theta_guess = max(pan_center - MAX_RANGE_TURNS,
                                       min(pan_center + MAX_RANGE_TURNS, theta_guess))
                    print(f"\nCup {ci + 1}/{len(clusters)}: moving to estimated "
                          f"{theta_guess:.4f} turns...")
                    move_pan_to(axis_pan, theta_guess, description="cup precision aim")

                    def closest_detection():
                        """The blob nearest frame center, or None - used both to
                        drive the aim polish and for the final r reading, so a
                        stray second cup drifting into frame doesn't get picked
                        instead of the one we just centered on."""
                        frame = grab()
                        if frame is None:
                            return None, None
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        dets = detect_all_targets(hsv, calib)
                        if not dets:
                            return None, None
                        fcx = frame.shape[1] // 2
                        best = min(dets, key=lambda d: abs(d["centroid"][0] - fcx))
                        return best, fcx

                    def read_offset():
                        best, fcx = closest_detection()
                        return None if best is None else best["centroid"][0] - fcx

                    def read_base_row():
                        best, _ = closest_detection()
                        return None if best is None else best["base"][1]

                    ok, final_err, moves = discrete_pan_aim(
                        axis_pan, read_offset, pan_center, max_offset_turns=MAX_RANGE_TURNS)
                    if not ok:
                        print(f"  Could not precisely center this cup (final error "
                              f"{final_err}px) - recording with the coarse estimate.")

                    row = median_read(read_base_row, samples=SAMPLES_PER_READ)
                    final_pos = axis_pan.pos_estimate
                    theta_deg = (final_pos - pan_center) * 360
                    if row is None:
                        print("  Lost detection for the final r reading - skipping this cup.")
                        continue
                    r_in = row_to_r(row)
                    print(f"  theta={theta_deg:+.2f}deg  base_row={row:.0f}px  r={r_in:.1f}in")
                    results.append({"theta_deg": theta_deg, "pan_turns": final_pos,
                                     "base_row": row, "r_in": r_in})

                print("\nReturning PAN to locked center...")
                move_pan_to(axis_pan, pan_center, description="locked position")

                with open(OUTPUT_JSON, "w") as f:
                    json.dump({"cups": results}, f, indent=2)
                print(f"\nSaved {len(results)} cup(s) to {OUTPUT_JSON}:")
                for r in results:
                    print(f"  theta={r['theta_deg']:+6.2f}deg  r={r['r_in']:6.1f}in")
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


if __name__ == "__main__":
    main()
