"""
Track B pan-aiming test script: interactively exercises discrete_pan_aim()
(in pong_common.py) against the live camera + PAN axis. Replaces the
continuous-PD approach in orange_tracker.py, which oscillated
(latency-induced overshoot with the camera riding on the axis). This script is a test harness for that
function, not a new implementation - Phase 3's targeting pipeline should
call discrete_pan_aim() directly rather than duplicating this loop.

Flow: PAN locks to center (same as cup_pixel_calibration.py). Click the cup
to calibrate its color. Press A to run one discrete aim cycle (probe +
secant corrections, logged move-by-move to the terminal); press Z to
re-center PAN back to the lock position between tries; press Q to quit.

Requires: pip install opencv-python numpy
Run with: odrive_env/bin/python pan_aim.py
"""
import cv2
from odrive.enums import AxisState

from pong_common import (
    CAM_INDEX, ROTATE_CODE, HUE_TOL, WATCHDOG_FEED_INTERVAL,
    ColorCalibration, WatchdogFeeder, connect_pan, detect_target, discrete_pan_aim,
    lock_pan, move_pan_to,
)


def main():
    print("=" * 60)
    print("TRACK B - discrete pan-aiming test")
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
    latest = {"det": None, "frame_cx": None}

    window = "Pan Aim Test"
    cv2.namedWindow(window)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and userdata["frame"] is not None:
            hsv = cv2.cvtColor(userdata["frame"], cv2.COLOR_BGR2HSV)
            h, s, v = calib.calibrate(hsv, x, y)
            print(f"\nColor calibrated from click at ({x},{y}): "
                  f"H={h:.0f} S={s:.0f} V={v:.0f} (+/- H{HUE_TOL})")

    cv2.setMouseCallback(window, on_mouse, mouse_state)

    def grab_and_detect():
        """One fresh camera read + detection at PAN's current (at-rest)
        position. Used both for the live overlay and as discrete_pan_aim's
        read_error_px callback."""
        ret, frame = cap.read()
        if not ret:
            return None, None
        if ROTATE_CODE is not None:
            frame = cv2.rotate(frame, ROTATE_CODE)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        det = detect_target(hsv, calib)
        return frame, det

    def read_error_px():
        frame, det = grab_and_detect()
        if frame is None or det is None:
            return None
        frame_cx = frame.shape[1] // 2
        return det["centroid"][0] - frame_cx

    print("\nClick on the cup in the video window to calibrate its color.")
    print("A - run discrete pan-aim (probe + secant corrections, logged below)")
    print("Z - re-center PAN back to the locked position")
    print("R - reset color calibration")
    print("Q or ESC (in video window) - quit\n")

    try:
        while True:
            frame, det = grab_and_detect()
            if frame is None:
                print("Failed to read frame.")
                break
            mouse_state["frame"] = frame
            latest["det"] = det

            h, w = frame.shape[:2]
            frame_cx = w // 2
            latest["frame_cx"] = frame_cx

            for yy in range(0, h, 16):
                cv2.line(frame, (frame_cx, yy), (frame_cx, min(yy + 8, h)), (180, 180, 180), 1)

            if det is not None:
                x, y, bw, bh = det["bbox"]
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 200, 255), 2)
                cx, cy = det["centroid"]
                cv2.circle(frame, (cx, cy), 6, (0, 200, 255), -1)
                cv2.line(frame, (frame_cx, cy), (cx, cy), (0, 200, 255), 1)
                offset_px = cx - frame_cx
                offset_color = (0, 220, 0) if abs(offset_px) < 15 else (0, 200, 255)
                cv2.putText(frame, f"h-offset: {offset_px:+d}px {'(centered)' if abs(offset_px) < 15 else ''}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, offset_color, 2)
            else:
                msg = "no target detected" if calib.ranges else "click cup to calibrate color"
                cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

            cv2.putText(frame, f"pos: {axis_pan.pos_estimate:.4f} turns", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                calib.ranges = None
                print("Color calibration reset - click the cup again.")
            elif key == ord('z'):
                print(f"\nRe-centering PAN to {pan_center:.4f} turns...")
                move_pan_to(axis_pan, pan_center, description="locked position")
            elif key == ord('a'):
                if calib.ranges is None:
                    print("\nClick the cup to calibrate color before aiming.")
                    continue
                print("\nAiming...")
                ok, final_err, moves = discrete_pan_aim(axis_pan, read_error_px, pan_center)
                status = "CENTERED" if ok else "DID NOT CONVERGE"
                print(f"{status} after {moves} move(s), final error "
                      f"{final_err if final_err is not None else 'N/A'}px, "
                      f"final pos {axis_pan.pos_estimate:.4f} turns "
                      f"({(axis_pan.pos_estimate - pan_center) * 360:+.1f} deg from center)\n")

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
