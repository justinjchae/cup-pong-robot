# Requires: pip install opencv-python numpy
# Webcam index: change CAM_INDEX if your camera isn't detected on 0

import odrive
from odrive.enums import *
import cv2
import numpy as np
import time

# --- Tuning parameters ---
KP              = 0.00002  # proportional gain: turns nudged per pixel of error per frame
KD              = 0.00005  # derivative gain: dampens oscillation (opposes rapid error change)
DEADZONE_PX     = 25       # pixel dead zone — motor holds when target is this close to center
MAX_PAN_TURNS   = 0.25     # ±0.25 turns (±90°) hard limit from center position
VEL_LIMIT       = 1.0      # turns/sec
ACCEL_LIMIT     = 3.0      # turns/sec²
MIN_BLOB_AREA   = 500      # px² — blobs smaller than this are treated as noise
CAM_INDEX       = 0        # change if the wrong camera opens
COUNTDOWN_SEC   = 3        # seconds to wait before tracking starts

# Blue in HSV (OpenCV hue range 0–179)
# Painters tape is low-saturation so the saturation floor is kept low
BLUE_LOWER = np.array([100,  60,  50])
BLUE_UPPER = np.array([130, 255, 255])


def turns_to_degrees(turns):
    return turns * 360.0

def find_blue_centroid(frame):
    """Return (cx, cy) of the largest blue blob and its contour, or (None, None)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_BLOB_AREA:
        return None, None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), largest


def draw_overlay(frame, contour, centroid, frame_cx, error_px, offset_deg):
    """Draw tracking visuals onto the frame in-place."""
    h, w = frame.shape[:2]

    # Semi-transparent blue fill over the detected blob
    highlight = frame.copy()
    cv2.drawContours(highlight, [contour], -1, (200, 100, 0), thickness=cv2.FILLED)
    cv2.addWeighted(highlight, 0.35, frame, 0.65, 0, frame)

    # Contour outline
    cv2.drawContours(frame, [contour], -1, (255, 150, 0), 2)

    # Crosshair on centroid
    cx, cy = centroid
    cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (255, 150, 0), 2)
    cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (255, 150, 0), 2)
    cv2.circle(frame, centroid, 5, (255, 150, 0), -1)

    # Dashed vertical center line
    for y in range(0, h, 16):
        cv2.line(frame, (frame_cx, y), (frame_cx, min(y + 8, h)), (200, 200, 200), 1)

    # Line from center to centroid showing the error
    cv2.line(frame, (frame_cx, cy), (cx, cy), (0, 200, 255), 2)

    # HUD text
    cv2.putText(frame, f"Error:  {error_px:+d} px",      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 150, 0), 2)
    cv2.putText(frame, f"Motor: {offset_deg:.1f} deg",   (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 150, 0), 2)
    cv2.putText(frame, "TRACKING",                       (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 150, 0), 2)


def main():
    print("Finding ODrive...")
    odrv = odrive.find_any()
    print(f"Found ODrive: {odrv.serial_number}")

    print("Clearing errors...")
    odrv.clear_errors()

    # Position control with trapezoidal trajectory
    odrv.axis0.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
    odrv.axis0.controller.config.input_mode   = INPUT_MODE_TRAP_TRAJ
    odrv.axis0.trap_traj.config.vel_limit      = VEL_LIMIT
    odrv.axis0.trap_traj.config.accel_limit    = ACCEL_LIMIT
    odrv.axis0.trap_traj.config.decel_limit    = ACCEL_LIMIT

    # Record the current encoder position as the center reference before engaging.
    # All pan limits are expressed as offsets from this value.
    print("\nManually rotate the motor so the camera points straight ahead (center).")
    input("Press Enter to mark this as center...")

    center_pos = odrv.axis0.pos_vel_mapper.pos_rel
    odrv.axis0.controller.input_pos = center_pos
    print(f"Center marked at pos_rel = {center_pos:.4f} turns")

    odrv.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
    time.sleep(0.5)
    if odrv.axis0.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
        print(f"Failed to enable motor. State: {odrv.axis0.current_state}, "
              f"Errors: {odrv.axis0.active_errors}")
        return
    print(f"Motor engaged. Holding at center ({center_pos:.4f} turns).")

    print(f"Opening camera (index {CAM_INDEX})...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open webcam.")
        odrv.axis0.requested_state = AXIS_STATE_IDLE
        return

    # Keep only the most recent frame in the buffer to minimize latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_cx = frame_w // 2
    print(f"Frame width: {frame_w}px  |  center: {frame_cx}px")
    print("Tracking blue. Press 'q' in the video window to quit.")

    # Countdown so the user can place the blue marker in frame before tracking starts
    countdown_start = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        remaining = COUNTDOWN_SEC - int(time.time() - countdown_start)
        if remaining <= 0:
            break
        h, w = frame.shape[:2]
        cv2.putText(frame, "Place blue marker in frame...",
                    (w // 2 - 220, h // 2 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 150, 0), 2)
        cv2.putText(frame, str(remaining),
                    (w // 2 - 20, h // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 150, 0), 4)
        cv2.imshow("Blue Tracker", frame)
        cv2.waitKey(1)

    # Flush stale frames that built up during the countdown
    for _ in range(5):
        cap.read()

    target_turns  = center_pos  # current commanded motor position
    prev_error_px = 0           # previous frame error for derivative term

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            centroid, contour = find_blue_centroid(frame)
            h, w = frame.shape[:2]

            if centroid is not None:
                cx, cy = centroid
                error_px = cx - frame_cx  # positive = target is right of center

                if abs(error_px) > DEADZONE_PX:
                    # PD controller: proportional nudge + derivative damping
                    d_error = error_px - prev_error_px
                    delta = KP * error_px + KD * d_error
                    target_turns += delta
                    target_turns = max(center_pos - MAX_PAN_TURNS,
                                       min(center_pos + MAX_PAN_TURNS, target_turns))
                    odrv.axis0.controller.input_pos = target_turns

                prev_error_px = error_px
                offset_deg = turns_to_degrees(target_turns - center_pos)
                draw_overlay(frame, contour, centroid, frame_cx, error_px, offset_deg)

            else:
                prev_error_px = 0
                for y in range(0, h, 16):
                    cv2.line(frame, (frame_cx, y), (frame_cx, min(y + 8, h)), (200, 200, 200), 1)
                cv2.putText(frame, "No blue detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imshow("Blue Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print("Shutting down...")
        cap.release()
        cv2.destroyAllWindows()
        odrv.axis0.requested_state = AXIS_STATE_IDLE
        print("Motor disabled. Done.")


if __name__ == "__main__":
    main()
