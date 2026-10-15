"""
Click-to-shoot demo: click roughly on a cup in the live camera view (boxes
are drawn around every detected cup so you can see what you're selecting),
PAN aims there, then press SPACE to fire - press SPACE again to re-fire at
the same spot, or click a different cup to re-aim. Uses the two one-time
calibrations already built:
  - pixel_range_calibration.json (cup_pixel_calibration.py): base-row -> r
  - theta_calibration.json (theta_calibration.py): pixel-offset -> angle
and Track A's launch speed/angle curve (ported into pong_common.py from
range_test.py) to turn r into a swing angle.

Flow: click once to calibrate color (same as cup_map.py/cup_pixel_calibration.py),
then every click after that must land inside a drawn box, which selects
that blob as the target. Centering uses calibrated_center - it reapplies
the theta calibration's own pixel-offset -> angle formula directly to
each fresh error reading (usually converges in 1 move, since the
calibration alone gets very close) instead of live-probing for a gain,
which is what made earlier attempts noisy/divergent on real hardware. A
stateful tracker (make_target_tracker) keeps following the SAME blob by
proximity to its last-seen position, not "nearest to frame center", so a
hand or another color-matched object nearby can't hijack the aim.

Requires: pip install opencv-python numpy
Run with: odrive_env/bin/python click_to_shoot.py
"""
import cv2
from odrive.enums import AxisState

from pong_common import (
    CAM_INDEX, ROTATE_CODE, HUE_TOL, MAX_RANGE_TURNS, PIVOT_TO_TAPE_START_IN,
    WATCHDOG_FEED_INTERVAL,
    ColorCalibration, WatchdogFeeder, calibrated_center, connect_launch, connect_pan,
    detect_all_targets, fire_shot, load_range_calibration, load_theta_calibration, lock_pan,
    median_read, prepare_launch, solve_swing_for_range,
)


# Empirical corrections for two systematic (not random) biases found by
# comparing actual ball landing spots against the intended cup - both are
# separate from the pixel/vision calibrations, which are accurate on their
# own terms (the tape-measure r readout matches reality closely).
#
# RANGE_BIAS_CORRECTION_IN: the launcher lands consistently long - already
# noted in range_test.py's own validation ("~1 inch of range error, slight
# overshoot"). Subtracted from r before solving for the swing angle, so it
# solves for a slightly shorter target and the overshoot cancels out.
RANGE_BIAS_CORRECTION_IN = 1.5

# THETA_BIAS_PX: perfect vision-centering (pixel offset = 0) isn't
# guaranteed to be exactly where the ball actually flies straight - that
# depends on how precisely the camera's optical axis lines up with the
# launcher's true fire direction, which was never itself calibrated
# against real landing spots. Shifts the pixel offset calibrated_center
# treats as "centered" by this many px instead of 0, to compensate.
# Starts at 0 (unknown sign/magnitude) - tune it live:
#   1. Fire at a cup, note which way it misses and by how much (inches) at
#      the known r.
#   2. Try a signed guess here (start small, e.g. +/-10px) and re-test.
#   3. If the miss direction flips, you went too far / wrong sign - if it
#      shrinks, keep adjusting the same direction until it's centered.
THETA_BIAS_PX = 30


def find_target_in_box(dets, x, y):
    """Only a click landing strictly inside a drawn bounding box selects a
    target - no nearest-blob fallback, so a click outside every detected
    cup is simply ignored rather than snapping to something unintended."""
    for d in dets:
        bx, by, bw, bh = d["bbox"]
        if bx <= x <= bx + bw and by <= y <= by + bh:
            return d
    return None


def main():
    print("=" * 60)
    print("CLICK TO SHOOT")
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

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open camera.")
        pan_feeder.stop()
        launch_feeder.stop()
        axis_pan.requested_state = AxisState.IDLE
        axis_launch.requested_state = AxisState.IDLE
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    calib = ColorCalibration()
    window = "Click to Shoot"
    cv2.namedWindow(window)
    pending_click = {"xy": None}

    # Only the WINDOW is shrunk (over X11/SSH forwarding, bytes-per-frame is
    # the bottleneck) - all detection/centering math below still runs on
    # the full-resolution frame, so aim accuracy is unaffected. Clicks come
    # back in the shrunk window's pixel space, so scale them back up to
    # full-res coordinates before using them.
    DISPLAY_SCALE = 0.5

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click["xy"] = (int(x / DISPLAY_SCALE), int(y / DISPLAY_SCALE))

    cv2.setMouseCallback(window, on_mouse)

    def check_abort():
        return (cv2.waitKey(1) & 0xFF) == 27

    def grab():
        # Discard one buffered frame before reading - some V4L2/USB webcam
        # backends silently ignore CAP_PROP_BUFFERSIZE=1 and hand back a
        # frame that was captured before the last PAN move settled, which
        # shows up as two consecutive centering reads with the identical
        # pixel error even though the motor moved in between.
        cap.grab()
        ret, frame = cap.read()
        if not ret:
            return None
        if ROTATE_CODE is not None:
            frame = cv2.rotate(frame, ROTATE_CODE)
        return frame

    def make_target_tracker(initial_x):
        """Stateful blob tracker for one aim attempt: picks whichever
        detected blob is nearest to where THIS target was last seen, not
        nearest to frame center. A real scene can have more than one
        object matching the calibrated color (an arm, another reddish
        object) - "nearest to center" has no memory of which blob it was
        just following, so a competing blob passing near center hijacks
        the aim. Anchoring off the previous read instead keeps following
        the same physical object as PAN moves it across the frame."""
        state = {"x": initial_x}

        def read_error():
            frame = grab()
            if frame is None:
                return None
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dets = detect_all_targets(hsv, calib)
            if not dets:
                return None
            fcx = frame.shape[1] // 2
            best = min(dets, key=lambda d: abs(d["centroid"][0] - state["x"]))
            state["x"] = best["centroid"][0]
            return (best["centroid"][0] - fcx) - THETA_BIAS_PX

        def read_row():
            frame = grab()
            if frame is None:
                return None
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dets = detect_all_targets(hsv, calib)
            if not dets:
                return None
            best = min(dets, key=lambda d: abs(d["centroid"][0] - state["x"]))
            state["x"] = best["centroid"][0]
            return best["base"][1]

        return read_error, read_row

    current_target = None  # {'r_in', 'swing_deg'} once aimed and ready

    print("\nClick a cup to calibrate its color, then click any cup to aim at it")
    print("(boxes show every detected cup). SPACE - fire (repeatable).")
    print("ESC - E-STOP (both axes)     Q - quit\n")

    try:
        while True:
            frame = grab()
            if frame is None:
                print("Failed to read frame.")
                break
            h, w = frame.shape[:2]
            frame_cx = w // 2
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dets = detect_all_targets(hsv, calib) if calib.ranges is not None else []

            for yy in range(0, h, 16):
                cv2.line(frame, (frame_cx, yy), (frame_cx, min(yy + 8, h)), (180, 180, 180), 1)
            for d in dets:
                x, y, bw, bh = d["bbox"]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 255), 2)
                # Bias-corrected, same quantity calibrated_center actually
                # drives to zero - showing the raw (uncorrected) offset here
                # instead would make this label disagree with the terminal's
                # "error" once THETA_BIAS_PX is nonzero, which looks like a
                # bug but isn't (see THETA_BIAS_PX's comment above).
                h_offset = (d["centroid"][0] - frame_cx) - THETA_BIAS_PX
                cv2.putText(frame, f"{h_offset:+d}px", (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

            if calib.ranges is None:
                cv2.putText(frame, "click a cup to calibrate color", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            elif current_target is not None:
                cv2.putText(frame, f"READY r={current_target['r_in']:.1f}in - "
                                    f"press SPACE to fire", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
                tape_should_read = current_target["r_in"] - PIVOT_TO_TAPE_START_IN
                cv2.putText(frame, f"tape should read: {tape_should_read:.1f}in "
                                    f"(from tape start)", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
            else:
                cv2.putText(frame, "click a cup to aim", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            if calib.ranges is not None:
                cv2.putText(frame, f"{len(dets)} cup(s) detected", (10, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(frame, f"pan: {(axis_pan.pos_estimate - pan_center) * 360:+.1f}deg",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            display_frame = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
                                        interpolation=cv2.INTER_AREA)
            cv2.imshow(window, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == 27:
                axis_pan.requested_state = AxisState.IDLE
                axis_launch.requested_state = AxisState.IDLE
                print("\n!!! E-STOP !!! Both motors disabled.")
                break
            elif key == ord('r'):
                calib.ranges = None
                current_target = None
                print("Color calibration reset - click the cup again.")
            elif key == ord(' '):
                if current_target is None:
                    print("No target selected yet - click a cup first.")
                    continue
                result = fire_shot(axis_launch, start_pos, current_target["swing_deg"],
                                    check_abort=check_abort)
                if result == 'estop':
                    print("!!! E-STOP !!! Both motors disabled.")
                    axis_pan.requested_state = AxisState.IDLE
                    break
                elif result != 'done':
                    print(f"Shot did not complete cleanly ({result}).")

            click = pending_click["xy"]
            if click is None:
                continue
            pending_click["xy"] = None
            x, y = click

            if calib.ranges is None:
                # First click ever: calibrates color ONLY. Does not aim -
                # boxes start appearing on the next frame, and aiming only
                # happens on a click that lands inside one of them.
                hue, sat, val = calib.calibrate(hsv, x, y)
                print(f"\nColor calibrated from click at ({x},{y}): "
                      f"H={hue:.0f} S={sat:.0f} V={val:.0f} (+/- H{HUE_TOL}). "
                      "Click inside a box to aim there.")
                continue

            target = find_target_in_box(dets, x, y)
            if target is None:
                print("Click inside one of the detected cup boxes to aim there.")
                continue

            e0 = target["centroid"][0] - frame_cx
            row0 = target["base"][1]
            if not (valid_phi_range[0] <= (axis_pan.pos_estimate + phi0 - phi_of_e(e0)) - pan_center
                    <= valid_phi_range[1]):
                print(f"NOTE: click is outside the calibrated aiming range "
                      f"({valid_phi_range[0] * 360:+.1f}deg to {valid_phi_range[1] * 360:+.1f}deg) "
                      "- aim may be less accurate.")

            # Tracker seeded directly from the clicked blob - every read
            # during centering follows that SAME blob by proximity to its
            # last-seen position, not "nearest to frame center", so a hand
            # or another color-matched object can't hijack the aim.
            read_error, read_row = make_target_tracker(target["centroid"][0])

            print(f"\nCentering on click...")
            # move_tolerance much tighter than the default (0.005 turns/1.8deg,
            # a leftover from scripts that never needed pixel precision) -
            # at the camera's px/deg sensitivity, 1.8deg of "close enough"
            # settling slop alone can be worth far more than the 4px pixel
            # deadband, so the move can finish before the pixel error
            # actually gets there. PAN's real encoder noise floor is only
            # ~0.02deg, so 0.001 turns (0.36deg) has plenty of margin.
            ok, final_err, moves = calibrated_center(
                axis_pan, read_error, pan_center, phi0, phi_of_e,
                max_offset_turns=MAX_RANGE_TURNS, deadband_px=4,
                move_tolerance=0.001, check_abort=check_abort)
            if ok == 'estop':
                print("!!! E-STOP !!! Both motors disabled.")
                axis_launch.requested_state = AxisState.IDLE
                break
            if not ok:
                print(f"  Could not precisely center this cup (final error "
                      f"{final_err}px) - using the best-effort position.")

            row_final = median_read(read_row, samples=5)
            if row_final is None:
                row_final = row0
            r_in = row_to_r(row_final)
            # r_in itself stays the true (tape-matching) reading for display -
            # only the value fed into the launch solve gets the overshoot
            # correction, so the on-screen "tape should read" stays honest.
            swing_deg = solve_swing_for_range((r_in - RANGE_BIAS_CORRECTION_IN) * 0.0254)
            if swing_deg is None:
                print(f"r={r_in:.1f}in is outside the calibrated launch range - "
                      "not setting this as the active target.")
                current_target = None
                continue

            current_target = {"r_in": r_in, "swing_deg": swing_deg}
            print(f"READY: r={r_in:.1f}in -> swing={swing_deg:.2f}deg - press SPACE to fire")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pan_feeder.stop()
        launch_feeder.stop()
        print("Disabling PAN and LAUNCH...")
        axis_pan.requested_state = AxisState.IDLE
        axis_launch.requested_state = AxisState.IDLE
        print("Both motors disabled. Safe to power off.")


if __name__ == "__main__":
    main()
