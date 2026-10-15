"""
Shared hardware + vision utilities for the cup-pong robot's vision scripts
(cup_pixel_calibration.py, pan_aim.py, and eventually the Phase 3 targeting
pipeline). Centralizes the PAN-axis safety pattern (serial-locked connect,
raw-anchored lock-to-center, watchdog feeding) and the click-to-calibrate
cup detection so every script uses the exact same validated logic instead
of copy-pasted (and potentially drifting) versions - this keeps computation
in clean, separately-callable functions.
"""
import glob
import json
import math
import threading
import time

import cv2
import numpy as np
import odrive
from odrive.enums import AxisState, ControlMode, InputMode

# --- Identity -----------------------------------------------------------
PAN_SERIAL = "3970355B3231"
LAUNCH_SERIAL = "394A355F3231"

# --- Pan axis tuning (turns) - validated values, shared across scripts -----
PAN_LOCK_RAW = 0.1091
# +/-60deg from the locked position - widened from the original +/-45deg
# (still used by combined_control.py/pan_axis_test.py/range_test.py, left
# alone there) specifically for the cup-mapping sweep, which needs more
# reach across the table. Confirmed 2026-09-20 after checking cable slack.
MAX_RANGE_TURNS = 60.0 / 360.0
PAN_CONTROLLER_VEL_LIMIT = 0.6
PAN_POS_GAIN = 40.0
PAN_VEL_GAIN = 0.35
PAN_VEL_INTEGRATOR_GAIN = 0.20
JOG_VEL_LIMIT = 0.3
JOG_ACCEL_LIMIT = 1.0
JOG_DECEL_LIMIT = 1.0

WATCHDOG_FEED_INTERVAL = 0.2
POSITION_TOLERANCE = 0.005
MOVE_TIMEOUT = 5.0

# --- Camera ---------------------------------------------------------------
def _resolve_camera_device(fallback_index=0):
    """
    Resolve the Opal C1 by its stable USB by-id path instead of a raw
    /dev/videoN index, which shifts whenever the camera re-enumerates (seen
    2026-09-22: a plain USB re-enumeration, no undervoltage, bumped it from
    video0 to video1/video2 and broke every script hardcoded to index 0).
    Falls back to fallback_index if the by-id symlink isn't present.
    """
    matches = sorted(glob.glob("/dev/v4l/by-id/*Opal_C1*video-index0"))
    return matches[0] if matches else fallback_index


CAM_INDEX = _resolve_camera_device()
# Confirmed 2026-09-20 on this mount: no rotation correction needed.
ROTATE_CODE = None
# Measured 2026-09-21 on the new rig: the camera's optical center sits 5mm to
# the right of PAN's rotation axis (camera looking outward). Not yet folded
# into any geometry math - theta_calibration.py/cup_pixel_calibration.py are
# both fully empirical (fit directly from real tape/sweep data at the real
# camera position), so a fixed offset this small gets absorbed into those
# curves automatically for whatever range they're calibrated across. Revisit
# only if real shots show a range-dependent aiming bias (worse close than
# far) consistent with an uncorrected parallax term.
CAMERA_PIVOT_OFFSET_MM = 5.0

MIN_BLOB_AREA = 500    # px^2 - tune to the cup's apparent size at typical range
MIN_BLOB_WIDTH_PX = 30   # rejects thin/small noise specks area alone can miss
MIN_BLOB_HEIGHT_PX = 30
HUE_TOL = 12          # +/- on OpenCV's 0-179 hue scale
# Widened 2026-09-21 (were 60/60/40/40) - shadows on part of a cup mostly
# drop Value (and somewhat Saturation) while Hue stays put; the old margins
# excluded shadowed cup surface from the mask, shrinking the detected
# bounding box below the cup's real extent. Only the lower bound moves
# (upper stays 255), so this only makes the mask MORE tolerant of dark
# pixels near the calibrated hue - shouldn't increase false positives from
# unrelated dark objects since hue still has to match.
SAT_MARGIN = 90
VAL_MARGIN = 90
SAT_FLOOR = 25
VAL_FLOOR = 25
SAMPLE_RADIUS = 6     # px, neighborhood around the click to sample HSV from

# Tape/ruler distances aren't measured from the pivot (r=0) - offset to the
# physical point operators actually measure from, same setup as
# range_test.py's TAPE_OFFSET_IN. Updated 2026-09-21: tape 0 now reads
# directly from the base edge (new table setup), so tape-start and
# launcher-base are the same physical point - confirmed still 50mm on the
# rebuilt rig, so PIVOT_TO_TAPE_START_IN is just an alias for
# PIVOT_TO_BASE_IN now (previously a separate ~56in-further point).
PIVOT_TO_BASE_IN = 50.0 / 25.4          # pivot -> launcher base / tape start, ~50mm
PIVOT_TO_TAPE_START_IN = PIVOT_TO_BASE_IN


# --- Monotone cubic interpolation (Fritsch-Carlson), used by range_test.py,
# cup_pixel_calibration.py, and the pixel-row curve loaded here too ---------
def monotone_interpolant(xs, ys):
    n = len(xs)
    dxs, dys, ms = [], [], []
    for i in range(n - 1):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        dxs.append(dx); dys.append(dy); ms.append(dy / dx)

    c1s = [ms[0]]
    for i in range(len(dxs) - 1):
        m0, m1 = ms[i], ms[i + 1]
        if m0 * m1 <= 0:
            c1s.append(0.0)
        else:
            dx0, dx1 = dxs[i], dxs[i + 1]
            common = dx0 + dx1
            c1s.append(3 * common / ((common + dx1) / m0 + (common + dx0) / m1))
    c1s.append(ms[-1])

    c2s, c3s = [], []
    for i in range(len(c1s) - 1):
        c1, m, invDx = c1s[i], ms[i], 1.0 / dxs[i]
        common = c1 + c1s[i + 1] - 2 * m
        c2s.append((m - c1 - common) * invDx)
        c3s.append(common * invDx * invDx)

    def f(x):
        xc = max(xs[0], min(xs[-1], x))
        i = 0
        for k in range(len(xs) - 1):
            if xc >= xs[k]:
                i = k
        diff = xc - xs[i]
        return ys[i] + c1s[i] * diff + c2s[i] * diff ** 2 + c3s[i] * diff ** 3

    return f


# --- Color calibration (click-to-calibrate HSV) -----------------------------
class ColorCalibration:
    """Holds the session's click-derived HSV detection range. Handles hue
    wraparound (e.g. red cups sit near both ends of OpenCV's 0-179 scale)
    by keeping two ranges and OR-ing their masks when needed."""

    def __init__(self):
        self.ranges = None  # list of (lower, upper) np arrays, or None

    def calibrate(self, hsv_frame, x, y):
        h, w = hsv_frame.shape[:2]
        x0, x1 = max(0, x - SAMPLE_RADIUS), min(w, x + SAMPLE_RADIUS + 1)
        y0, y1 = max(0, y - SAMPLE_RADIUS), min(h, y + SAMPLE_RADIUS + 1)
        patch = hsv_frame[y0:y1, x0:x1].reshape(-1, 3)
        med_h, med_s, med_v = np.median(patch, axis=0)

        # int + explicit uint8 below: np.median returns float64, and mixing
        # that with plain Python ints (179, 255) made np.array() pick a
        # different dtype for the lower vs upper bound - cv2.inRange requires
        # both bounds to share a dtype, so this was throwing "lb.type() ==
        # ub.type()" at runtime.
        s_lo = int(max(SAT_FLOOR, med_s - SAT_MARGIN))
        v_lo = int(max(VAL_FLOOR, med_v - VAL_MARGIN))
        s_hi, v_hi = 255, 255

        h_lo, h_hi = int(med_h - HUE_TOL), int(med_h + HUE_TOL)
        if h_lo < 0:
            ranges = [
                (0, s_lo, v_lo, h_hi, s_hi, v_hi),
                (180 + h_lo, s_lo, v_lo, 179, s_hi, v_hi),
            ]
        elif h_hi > 179:
            ranges = [
                (h_lo, s_lo, v_lo, 179, s_hi, v_hi),
                (0, s_lo, v_lo, h_hi - 180, s_hi, v_hi),
            ]
        else:
            ranges = [(h_lo, s_lo, v_lo, h_hi, s_hi, v_hi)]

        self.ranges = [
            (np.array([lh, ls, lv], dtype=np.uint8), np.array([uh, us, uv], dtype=np.uint8))
            for lh, ls, lv, uh, us, uv in ranges
        ]

        return med_h, med_s, med_v

    def mask(self, hsv_frame):
        if self.ranges is None:
            return None
        m = cv2.inRange(hsv_frame, *self.ranges[0])
        for lower, upper in self.ranges[1:]:
            m = cv2.bitwise_or(m, cv2.inRange(hsv_frame, lower, upper))
        return m


def detect_target(hsv_frame, calib, min_blob_area=MIN_BLOB_AREA):
    """Returns dict(base=(x,y), centroid=(x,y), bbox=(x,y,w,h)) or None.
    base = bottom-center of the bounding box, used for the range estimate
    (NOT centroid - centroid bakes in a perspective-dependent bias).
    centroid is used for x-centering."""
    results = detect_all_targets(hsv_frame, calib, min_blob_area)
    return results[0] if results else None


def _contour_result(contour):
    x, y, w, h = cv2.boundingRect(contour)
    base = (x + w // 2, y + h)
    M = cv2.moments(contour)
    centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] else base
    return {"base": base, "centroid": centroid, "bbox": (x, y, w, h), "contour": contour}


def detect_all_targets(hsv_frame, calib, min_blob_area=MIN_BLOB_AREA,
                        min_width_px=MIN_BLOB_WIDTH_PX, min_height_px=MIN_BLOB_HEIGHT_PX):
    """Like detect_target, but returns every blob passing the color+size
    filter (largest first), not just the biggest one. Needed for real
    gameplay: several same-colored cups can be in frame at once, and each
    one needs its own (theta, r), so the single-largest-blob approach
    doesn't generalize.

    Filters on bounding-box width/height as well as area - a thin sliver
    of color-matched noise (a reflection, an edge) can clear the area
    threshold by being long and thin without looking anything like a cup."""
    mask = calib.mask(hsv_frame)
    if mask is None:
        return []
    denoise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, denoise_kernel)
    # A glossy cup under uneven light doesn't just get darker in shadow - it
    # also desaturates toward white at specular highlights, and a
    # near-white pixel's hue is essentially noise, so no HSV threshold
    # tuning can reliably classify it as "cup-colored" (see 2026-09-21
    # session: widening VAL margins fixed shadow dropout but then exposed
    # highlight dropout on the SAME cup - two symptoms of one cause). A
    # highlight/shadow band that fully cuts through the cup's color mask
    # splits it into separate contours, and only the largest survives below
    # - dropping the other side entirely. Closing with a much bigger kernel
    # bridges that gap geometrically (fill space surrounded by matched
    # pixels) regardless of why the pixels in between failed the color
    # test. Sized for a typical highlight streak, not a whole separate
    # object - if two genuinely different cups ever sit closer than this in
    # frame, shrink it; if a highlight/shadow band is still splitting a
    # single cup, grow it.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, denoise_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        if cv2.contourArea(c) < min_blob_area:
            continue
        _, _, w, h = cv2.boundingRect(c)
        if w < min_width_px or h < min_height_px:
            continue
        valid.append(c)
    valid.sort(key=cv2.contourArea, reverse=True)
    return [_contour_result(c) for c in valid]


def median_read(read_value_fn, samples=5, min_valid=3):
    """Calls read_value_fn() (returning an int/float or None) up to `samples`
    times and returns the median of the valid reads, or None if fewer than
    min_valid succeeded. Real detection noise (lighting flicker, contour-edge
    jitter) was large enough on real hardware to destabilize the live pan-
    aiming secant search - a few frames' median is cheap insurance for any
    one-off reading (calibration sweep points, cup-map captures) that isn't
    itself part of a tight per-frame control loop."""
    import statistics
    vals = [v for v in (read_value_fn() for _ in range(samples)) if v is not None]
    if len(vals) < min_valid:
        return None
    return statistics.median(vals)


# --- Hardware plumbing -------------------------------------------------------
class WatchdogFeeder:
    def __init__(self, axis, interval=WATCHDOG_FEED_INTERVAL):
        self._axis = axis
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._axis.watchdog_feed()
            except Exception:
                pass
            self._stop_event.wait(self._interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=1.0)


def connect_axis(serial, label):
    print(f"Connecting to {label} axis (S/N {serial})...")
    odrv = odrive.find_any(serial_number=serial, timeout=10)
    axis = odrv.axis0
    if axis.current_state != AxisState.IDLE:
        print(f"  {label} axis was not idle - forcing IDLE.")
        axis.requested_state = AxisState.IDLE
        time.sleep(0.3)
    odrv.clear_errors()
    return odrv, axis


def connect_pan():
    return connect_axis(PAN_SERIAL, "PAN")


def move_pan_to(axis, position, tolerance=POSITION_TOLERANCE, timeout=MOVE_TIMEOUT,
                 description="", check_abort=None):
    """Commands an axis to an absolute position and blocks until settled (or
    timed out). Generic despite the name - also used for LAUNCH moves, since
    the "command + block until fully settled" pattern is identical for both
    axes. Used both for the initial lock and later discrete aiming
    corrections - always waits for the move to fully finish before
    returning, which is the whole point of "discrete" aiming (no reacting
    to a frame while the axis is still moving, see vision-and-gui-design
    memory's oscillation diagnosis).

    check_abort, if given, is polled each loop iteration; when it returns
    True the axis is immediately set to IDLE (e-stop) and this returns
    'estop' instead of True/False."""
    axis.controller.input_pos = position
    start = time.time()
    while abs(axis.pos_estimate - position) > tolerance:
        if check_abort is not None and check_abort():
            axis.requested_state = AxisState.IDLE
            return 'estop'
        if time.time() - start > timeout:
            print(f"  WARNING: timed out moving to {description or position} "
                  f"(error={abs(axis.pos_estimate - position):.4f} turns)")
            return False
        time.sleep(0.02)
    return True


def lock_pan(odrv, axis):
    """Auto-drives PAN to the saved locked position, raw-anchored and
    safety-bounded (pos_estimate isn't power-cycle-stable,
    onboard_encoder0.raw is). Aborts (returns None)
    rather than moving if the correction looks implausible."""
    pan_now = axis.pos_estimate
    raw_diff = PAN_LOCK_RAW - odrv.onboard_encoder0.raw
    correction = raw_diff - round(raw_diff)
    if abs(correction) > MAX_RANGE_TURNS:
        print(f"ERROR: PAN correction is {abs(correction) * 360:.1f} deg - farther than "
              f"the {MAX_RANGE_TURNS * 360:.0f} deg safety window. Manually move PAN "
              "closer to the locked position before running this script.")
        return None

    lock_target = pan_now + correction
    axis.controller.config.control_mode = ControlMode.POSITION_CONTROL
    axis.controller.config.input_mode = InputMode.TRAP_TRAJ
    axis.controller.config.pos_gain = PAN_POS_GAIN
    axis.controller.config.vel_gain = PAN_VEL_GAIN
    axis.controller.config.vel_integrator_gain = PAN_VEL_INTEGRATOR_GAIN
    axis.controller.config.vel_limit = PAN_CONTROLLER_VEL_LIMIT
    axis.trap_traj.config.vel_limit = JOG_VEL_LIMIT
    axis.trap_traj.config.accel_limit = JOG_ACCEL_LIMIT
    axis.trap_traj.config.decel_limit = JOG_DECEL_LIMIT
    axis.controller.input_pos = pan_now

    print("Engaging PAN closed-loop control...")
    axis.requested_state = AxisState.CLOSED_LOOP_CONTROL
    time.sleep(0.3)
    if axis.current_state != AxisState.CLOSED_LOOP_CONTROL:
        print(f"ERROR: PAN did not reach closed loop control. "
              f"active_errors={axis.active_errors} disarm_reason={axis.disarm_reason}")
        axis.requested_state = AxisState.IDLE
        return None

    print(f"  Moving PAN to locked position ({lock_target:.4f} turns)...")
    move_pan_to(axis, lock_target, description="locked position")
    print("  PAN locked.")
    return lock_target


# --- Calibrated centering (preferred whenever a theta calibration exists) --
CENTER_DEADBAND_PX = 10
CENTER_MAX_ITERS = 3


def calibrated_center(axis, read_error_px, pan_center, phi0, phi_of_e,
                       max_offset_turns=MAX_RANGE_TURNS, deadband_px=CENTER_DEADBAND_PX,
                       max_iters=CENTER_MAX_ITERS, move_tolerance=POSITION_TOLERANCE,
                       move_timeout=MOVE_TIMEOUT, check_abort=None, log=print):
    """Centers on a target using the theta calibration directly, instead of
    live-probing for a pixel-per-turn gain (see discrete_pan_aim below).
    phi_of_e is already a validated, stable pixel-offset -> angle mapping
    (built once by theta_calibration.py from real sweep data) - so each
    iteration just recomputes the exact correction from the current pixel
    error via that mapping, rather than numerically re-deriving a local
    slope from noisy small probe moves. Far less prone to the noise-driven
    divergence discrete_pan_aim showed on real hardware. Usually converges in 1-2 moves: applying the calibration
    once already gets close (that's its whole job), a second corrects any
    small residual from measurement noise.

    read_error_px() must return the tracked target's pixel x-offset from
    frame center (int, + = right of center), or None if lost - pass a
    stateful tracker (see click_to_shoot.py's make_target_tracker) so a
    different color-matched object can't hijack the read.

    Returns (success: bool, final_error_px: int or None, moves_made: int).
    """
    moves = 0
    e = read_error_px()
    if e is None:
        log("  No target detected - can't center.")
        return False, None, 0
    log(f"  iter 0: pos={axis.pos_estimate:.4f} turns, error={e:+d}px")
    if abs(e) <= deadband_px:
        log("  Already centered.")
        return True, e, 0

    for i in range(1, max_iters + 1):
        p_now = axis.pos_estimate
        target = p_now + phi0 - phi_of_e(e)
        target = max(pan_center - max_offset_turns, min(pan_center + max_offset_turns, target))
        result = move_pan_to(axis, target, move_tolerance, move_timeout,
                              "calibrated center", check_abort=check_abort)
        if result == 'estop':
            return 'estop', e, moves
        if result != True:
            return False, e, moves
        moves += 1
        e = read_error_px()
        if e is None:
            log("  Lost target mid-correction - aborting.")
            return False, None, moves
        log(f"  iter {i}: pos={target:.4f} turns, error={e:+d}px")
        if abs(e) <= deadband_px:
            log(f"  Centered after {moves} move(s).")
            return True, e, moves

    log(f"  Did not reach deadband within {moves} move(s) (final error {e:+d}px).")
    return False, e, moves


# --- Discrete pan-aiming - live-probing fallback for when no theta
# calibration exists yet (see calibrated_center above, which should be
# preferred whenever theta_calibration.json is available) -----------------
PROBE_TURNS = 0.015     # ~5.4deg, initial probe step for the live gain measurement
MIN_PROBE_PX = 4        # minimum pixel shift/difference needed to trust a gain reading
AIM_DEADZONE_PX = 8
AIM_MAX_ITERS = 8
# The pixel-vs-pan-angle relationship isn't actually linear (roughly a
# tangent curve - steeper away from center), so a secant step computed from
# a small local probe reliably overshoots when extrapolated over a large
# correction. Damping trades an extra iteration or two for not overshooting
# into (or past) the safety window - found empirically after real-hardware
# runs kept landing near the 45deg limit off a single raw secant step.
AIM_DAMPING = 0.7


def _probe(axis, read_error_px, p_ref, e_ref, pan_center, max_offset_turns,
           probe_turns, min_probe_px, move_tolerance, move_timeout, log):
    """Nudges PAN by a small (growing) step from p_ref until the pixel error
    reads measurably different from e_ref, to get a fresh local (position,
    error) pair for the secant slope. Used both for the very first probe
    and mid-aim, when two consecutive readings are too close (within pixel
    noise) to divide by safely. Returns (p_new, e_new) or None on failure."""
    probe = probe_turns
    for _attempt in range(3):
        p_new = p_ref + probe
        if abs(p_new - pan_center) > max_offset_turns:
            p_new = p_ref - probe
        if abs(p_new - pan_center) > max_offset_turns:
            log("  Can't probe without exceeding the PAN safety window - aborting.")
            return None
        if not move_pan_to(axis, p_new, move_tolerance, move_timeout, "aim probe"):
            return None
        e_new = read_error_px()
        if e_new is None:
            log("  Lost target during probe - aborting.")
            return None
        if abs(e_new - e_ref) >= min_probe_px:
            return p_new, e_new
        probe *= 2
    log("  No measurable pixel response to probing - aborting "
        "(check color calibration / PAN motion).")
    return None


def discrete_pan_aim(axis, read_error_px, pan_center, max_offset_turns=MAX_RANGE_TURNS,
                      deadzone_px=AIM_DEADZONE_PX, max_iters=AIM_MAX_ITERS,
                      probe_turns=PROBE_TURNS, min_probe_px=MIN_PROBE_PX, damping=AIM_DAMPING,
                      move_tolerance=POSITION_TOLERANCE, move_timeout=MOVE_TIMEOUT, log=print):
    """Discrete look -> compute -> move -> wait -> re-check pan-aiming -
    replaces the continuous-PD approach in orange_tracker.py, which
    oscillated from reacting to frames while PAN was still moving
    (latency-induced overshoot plus eye-in-hand coupling). Every move here fully settles
    (via move_pan_to's blocking wait) before the next frame is read.

    Rather than assuming the camera's field of view to convert a pixel
    error into a turns correction, this measures the live pixel-per-turn
    gain with a small probe move - same "measure it empirically, don't
    model the geometry" approach Track A's launch calibration used - then converges with a damped secant-
    method correction using that measured gain. Damping and a re-probe
    fallback (instead of dividing by a near-zero pixel difference) guard
    against the real nonlinearity in pixel-vs-angle causing overshoot when
    a small local probe is extrapolated over a large correction.

    read_error_px() must return the detected centroid's x-offset from
    frame center in pixels (int, + = right of center) when called with
    PAN at rest at its current commanded position, or None if no target
    is detected.

    Returns (success: bool, final_error_px: int or None, moves_made: int).
    """
    p0 = axis.pos_estimate
    e0 = read_error_px()
    if e0 is None:
        log("  No target detected - can't aim.")
        return False, None, 0
    log(f"  iter 0: pos={p0:.4f} turns, error={e0:+d}px")
    if abs(e0) <= deadzone_px:
        log("  Already centered.")
        return True, e0, 0

    probed = _probe(axis, read_error_px, p0, e0, pan_center, max_offset_turns,
                     probe_turns, min_probe_px, move_tolerance, move_timeout, log)
    if probed is None:
        return False, e0, 0
    p1, e1 = probed
    moves = 1
    log(f"  iter 1: pos={p1:.4f} turns, error={e1:+d}px "
        f"(measured gain ~{(e1 - e0) / (p1 - p0):.0f}px/turn)")

    p_prev, e_prev = p0, e0
    p_cur, e_cur = p1, e1
    i = 2
    while i <= max_iters:
        if abs(e_cur) <= deadzone_px:
            log(f"  Centered after {moves} move(s).")
            return True, e_cur, moves

        denom = e_cur - e_prev
        if abs(denom) < min_probe_px:
            log(f"  Last two readings too close to trust ({e_prev:+d}px, {e_cur:+d}px) "
                f"- re-probing near the current position.")
            reprobed = _probe(axis, read_error_px, p_cur, e_cur, pan_center, max_offset_turns,
                               probe_turns, min_probe_px, move_tolerance, move_timeout, log)
            if reprobed is None:
                return False, e_cur, moves
            moves += 1
            p_prev, e_prev = p_cur, e_cur
            p_cur, e_cur = reprobed
            i += 1
            continue

        raw_delta = -e_cur * (p_cur - p_prev) / denom
        p_next = p_cur + damping * raw_delta
        if abs(p_next - pan_center) > max_offset_turns:
            log(f"  Next correction ({p_next:.4f} turns) would exceed the PAN "
                f"safety window - aborting. Target may be out of the aimable range.")
            return False, e_cur, moves
        if not move_pan_to(axis, p_next, move_tolerance, move_timeout, "aim correction"):
            return False, e_cur, moves
        e_next = read_error_px()
        if e_next is None:
            log("  Lost target mid-correction - aborting.")
            return False, e_cur, moves
        moves += 1
        log(f"  iter {i}: pos={p_next:.4f} turns, error={e_next:+d}px")
        p_prev, e_prev = p_cur, e_cur
        p_cur, e_cur = p_next, e_next
        i += 1

    log(f"  Did not converge within {moves} moves (final error {e_cur:+d}px).")
    return abs(e_cur) <= deadzone_px, e_cur, moves


# --- Launch axis (ported from range_test.py - exact validated values,
# reused rather than re-derived) --------------------------------
LAUNCH_HARD_STOP_RAW = 0.2073          # re-captured 2026-09-21 for the new (larger-diameter)
                                        # arm - hard stop itself didn't move, but the arm now
                                        # rests slightly higher against it (was 0.2036)
HARD_STOP_BUFFER_DEG = 10.0             # reduced 2026-09-21 (was 15.0) for more accel range
HARD_STOP_BUFFER_TURNS = HARD_STOP_BUFFER_DEG / 360.0
LAUNCH_HARD_STOP_MAX_CORRECTION_TURNS = 0.083

LAUNCH_TRAP_ACCEL_LIMIT = 100.0
LAUNCH_TRAP_DECEL_LIMIT = 100.0
LAUNCH_VEL_LIMIT = 30.0
RETURN_VEL_LIMIT = 1.5     # halved 2026-09-21 (was 3.0) - arm was hitting the hard stop on
                           # return, likely worse now that HARD_STOP_BUFFER_DEG shrank the
                           # margin available to absorb any overshoot
LAUNCH_CONTROLLER_VEL_LIMIT = LAUNCH_VEL_LIMIT * 1.5
LAUNCH_POS_GAIN = 40.0    # 2026-09-22: tried both directions to chase a small resting-
                           # position rattle (ball against the rails) - 30/vel_gain=0.45
                           # (softer) made it worse, not better; 55/vel_gain=0.48 (stiffer)
                           # made it much louder. Neither changed scatter either way. Not a
                           # gain issue - reverted to this baseline (the best-sounding of
                           # the three) and moved on rather than keep probing what's likely
                           # a resonance interaction, not something PID tuning fixes.
LAUNCH_VEL_GAIN = 0.35    # reverted alongside pos_gain above - see that comment
LAUNCH_VEL_INTEGRATOR_GAIN = 0.12
LAUNCH_HOLD_SECONDS = 1.0

# Anchored-release scheme (2026-09-22): instead of a fixed start_pos and a
# variable end (swing_deg) - which tangles release angle, release timing,
# and exit speed together in the hard-to-predict way this whole session
# fought - fix the ANGLE at which deceleration begins (a "virtual hard
# stop") and vary start_pos per shot so pure acceleration reaches exactly
# the target vel_limit right as the arm arrives there. 90deg gives comfortable
# headroom for accel_distance=v^2/(2*accel) up to ~vel_limit=7 turns/s at the
# default accel_limit=100 (well above anything fired so far this session,
# whose peak was ~5.27) before hitting the near-hard-stop safety margin -
# NOT derived from any real release-point measurement, just chosen so a
# useful vel_limit range is actually reachable. Revisit with slow-mo footage
# of the anchored scheme once it's running on hardware.
LAUNCH_RELEASE_ANCHOR_DEG_FROM_START = 90.0
# Minimum clearance kept between a shot's computed start_pos (which moves
# further back for higher vel_limit targets) and the near hard stop -
# reuses the same buffer already validated for the resting start position.
LAUNCH_MIN_START_BUFFER_TURNS = HARD_STOP_BUFFER_TURNS

# Far-side hard stop (swinging this far would hit the camera). NOT anchored
# via its own raw value the way the near stop is - that only works when the
# target is within +/-0.5 turn of the arm's current position (fine for the
# near stop/PAN lock, both always nearby), and this stop is ~263deg away,
# so the nearest-wrap correction silently picks the wrong modular branch
# (confirmed 2026-09-21: it resolved to a position BEHIND start_pos, which
# is physically impossible - caught by max_safe_launch_pos()'s own check
# before ever commanding a move). Anchored instead as a fixed separation
# from the near stop (where the nearest-wrap math IS valid), measured in
# one unbroken launch_read_position.py session so both readings shared a
# pos_estimate reference: near pos_estimate=0.9801/raw=0.2075, far
# pos_estimate=1.7111/raw=0.9385 -> separation = 0.7310 turns (263.2deg).
LAUNCH_STOP_SEPARATION_TURNS = 0.7310
LAUNCH_FAR_HARD_STOP_BUFFER_DEG = 20.0
LAUNCH_FAR_HARD_STOP_BUFFER_TURNS = LAUNCH_FAR_HARD_STOP_BUFFER_DEG / 360.0

# Geometry + reference-shot calibration (corrected pivot height)
PIVOT_M = 0.2364    # updated 2026-09-21 for the new arm/table redesign (was 0.175 -
                    # base raised 61.4mm)
ARM_M = 0.180       # updated 2026-09-21 for the new arm/table redesign (was 0.147,
                    # pivot to ball-rest center)
G = 9.81
# STALE as of the 2026-09-21 geometry change above - this speed/angle table was
# fit by firing real shots on the OLD 0.147m arm / 0.175m pivot and does not
# transfer to the new geometry (exit speed
# and release geometry both scale with arm length, and the release point was
# already shown empirically not to follow a fixed geometric model even for a
# single arm). landing_range_m()/solve_swing_for_range() will silently give
# wrong answers until Track A (fire shots at each swing angle, measure landing
# distance + slomo flight time, re-solve for speed/angle) is redone on the new
# arm and this table is replaced.
CAL = [
    (45.0, 2.138, 69.15),
    (60.0, 2.834, 44.70),
    (75.0, 3.838, 35.14),
    (90.0, 4.880, 26.00),
]
_swings = [c[0] for c in CAL]
speed_fn = monotone_interpolant(_swings, [c[1] for c in CAL])
angle_fn = monotone_interpolant(_swings, [c[2] for c in CAL])


def release_height_m(angle_deg):
    return PIVOT_M + ARM_M * math.cos(math.radians(angle_deg))


def landing_range_m(swing_deg):
    speed = speed_fn(swing_deg)
    angle_deg = angle_fn(swing_deg)
    rad = math.radians(angle_deg)
    vx = speed * math.cos(rad)
    vy = speed * math.sin(rad)
    h = release_height_m(angle_deg)
    t_land = (vy + math.sqrt(vy * vy + 2 * G * h)) / G
    return vx * t_land


def solve_swing_for_range(target_r_m):
    """Returns the swing angle (deg) whose calibrated trajectory lands at
    target_r_m, or None if that's outside the calibrated 45-90deg swing
    range (see landing_range_m(45)/landing_range_m(90) for the reachable
    bounds)."""
    lo, hi = _swings[0], _swings[-1]
    r_lo, r_hi = landing_range_m(lo), landing_range_m(hi)
    if target_r_m < r_lo or target_r_m > r_hi:
        return None
    for _ in range(30):
        mid = (lo + hi) / 2
        if landing_range_m(mid) < target_r_m:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def connect_launch():
    return connect_axis(LAUNCH_SERIAL, "LAUNCH")


def prepare_launch(odrv, axis):
    """Computes the hard-stop position from onboard_encoder0.raw (same
    safety-bounded pattern as lock_pan), engages closed-loop control
    holding the current position, then moves to the buffered start
    position. Returns start_pos, or None on failure (axis left IDLE)."""
    launch_now = axis.pos_estimate
    raw_diff = LAUNCH_HARD_STOP_RAW - odrv.onboard_encoder0.raw
    correction = raw_diff - round(raw_diff)
    if abs(correction) > LAUNCH_HARD_STOP_MAX_CORRECTION_TURNS:
        print(f"ERROR: LAUNCH hard-stop correction is {abs(correction) * 360:.1f} deg - "
              f"farther than the {LAUNCH_HARD_STOP_MAX_CORRECTION_TURNS * 360:.0f} deg "
              "sanity bound. Check the arm by hand before proceeding.")
        return None

    hard_stop_pos = launch_now + correction
    start_pos = hard_stop_pos + HARD_STOP_BUFFER_TURNS

    axis.controller.config.control_mode = ControlMode.POSITION_CONTROL
    axis.controller.config.input_mode = InputMode.TRAP_TRAJ
    axis.controller.config.pos_gain = LAUNCH_POS_GAIN
    axis.controller.config.vel_gain = LAUNCH_VEL_GAIN
    axis.controller.config.vel_integrator_gain = LAUNCH_VEL_INTEGRATOR_GAIN
    axis.controller.config.vel_limit = LAUNCH_CONTROLLER_VEL_LIMIT
    axis.trap_traj.config.vel_limit = RETURN_VEL_LIMIT
    axis.trap_traj.config.accel_limit = LAUNCH_TRAP_ACCEL_LIMIT
    axis.trap_traj.config.decel_limit = LAUNCH_TRAP_DECEL_LIMIT
    axis.controller.input_pos = launch_now

    print("Engaging LAUNCH closed-loop control, holding current position...")
    axis.requested_state = AxisState.CLOSED_LOOP_CONTROL
    time.sleep(0.3)
    if axis.current_state != AxisState.CLOSED_LOOP_CONTROL:
        print(f"ERROR: LAUNCH did not reach closed loop control. "
              f"active_errors={axis.active_errors} disarm_reason={axis.disarm_reason}")
        axis.requested_state = AxisState.IDLE
        return None

    print(f"  Moving LAUNCH to start position ({start_pos:.4f} turns)...")
    move_pan_to(axis, start_pos, description="launch start position")
    return start_pos


def max_safe_launch_pos(odrv, axis):
    """Position beyond which LAUNCH must not be commanded -
    LAUNCH_FAR_HARD_STOP_BUFFER_DEG short of the far-side (camera) hard
    stop. Anchors the NEAR hard stop first (raw-encoder anchoring, valid
    here since it's always within +/-0.5 turn of the arm's current
    position - pos_estimate isn't power-cycle-stable, onboard_encoder0.raw
    is), then adds the fixed, empirically
    measured LAUNCH_STOP_SEPARATION_TURNS to reach the far stop - see that
    constant's comment for why the far stop can't be anchored the same
    direct way. Callers extending swing_deg beyond the default 90deg (e.g.
    to test a gentler decel_limit while preserving peak velocity) must
    check their target against this before firing."""
    now = axis.pos_estimate
    near_raw_diff = LAUNCH_HARD_STOP_RAW - odrv.onboard_encoder0.raw
    near_correction = near_raw_diff - round(near_raw_diff)
    near_hard_stop_pos = now + near_correction
    far_hard_stop_pos = near_hard_stop_pos + LAUNCH_STOP_SEPARATION_TURNS
    return far_hard_stop_pos - LAUNCH_FAR_HARD_STOP_BUFFER_TURNS


THETA_CAL_JSON = "theta_calibration.json"
RANGE_CAL_JSON = "pixel_range_calibration.json"


def load_theta_calibration(path=THETA_CAL_JSON):
    """Returns (pan_center, phi0, phi_of_e, valid_phi_range). See
    theta_calibration.py for what these mean. valid_phi_range is the
    contiguous phi span the calibration target was actually tracked
    across - callers should not trust detections computed outside it
    (background contamination once the target left frame)."""
    with open(path) as f:
        data = json.load(f)
    samples = data["samples"]
    es = [s["pixel_offset"] for s in samples]
    phis = [s["phi_turns"] for s in samples]
    order = sorted(range(len(es)), key=lambda i: es[i])
    es = [es[i] for i in order]
    phis = [phis[i] for i in order]
    phi_of_e = monotone_interpolant(es, phis)
    valid_phi_range = tuple(data.get("valid_phi_range", (min(phis), max(phis))))
    return data["pan_center"], data["phi0"], phi_of_e, valid_phi_range


def load_range_calibration(path=RANGE_CAL_JSON):
    """Returns row_to_r(pixel_row) -> distance_in, fit from
    cup_pixel_calibration.py's saved samples (duplicate rows averaged)."""
    with open(path) as f:
        data = json.load(f)
    by_row = {}
    for s in data["samples"]:
        by_row.setdefault(s["pixel_row"], []).append(s["distance_in"])
    rows = sorted(by_row)
    rs = [sum(by_row[r]) / len(by_row[r]) for r in rows]
    return monotone_interpolant(rows, rs)


def fire_shot(axis, start_pos, swing_deg, check_abort=None,
              swing_accel_limit=None, swing_decel_limit=None):
    """Swing -> hold -> return sequence, same as range_test.py. Returns
    'done', 'estop', or 'timeout'.

    swing_accel_limit/swing_decel_limit override LAUNCH_TRAP_ACCEL_LIMIT/
    LAUNCH_TRAP_DECEL_LIMIT for the swing (launch) move only - the return
    move always uses the standard constants (RETURN_VEL_LIMIT was already
    tuned down once to stop it hitting the hard stop
    - not something to vary casually). Callers that widen swing_deg to
    compensate for a gentler swing_decel_limit (see max_safe_launch_pos)
    must check start_pos + swing_deg/360 against it before calling."""
    swing_turns = swing_deg / 360.0
    accel = LAUNCH_TRAP_ACCEL_LIMIT if swing_accel_limit is None else swing_accel_limit
    decel = LAUNCH_TRAP_DECEL_LIMIT if swing_decel_limit is None else swing_decel_limit
    print("\nLAUNCH")
    axis.trap_traj.config.vel_limit = LAUNCH_VEL_LIMIT
    axis.trap_traj.config.accel_limit = accel
    axis.trap_traj.config.decel_limit = decel
    result = move_pan_to(axis, start_pos + swing_turns, description="launch position",
                          check_abort=check_abort)
    if result != True:
        return result if result == 'estop' else 'timeout'

    print(f"  Holding for {LAUNCH_HOLD_SECONDS:.1f}s...")
    hold_start = time.time()
    while time.time() - hold_start < LAUNCH_HOLD_SECONDS:
        if check_abort is not None and check_abort():
            axis.requested_state = AxisState.IDLE
            return 'estop'
        time.sleep(0.02)

    print("Returning to start")
    axis.trap_traj.config.vel_limit = RETURN_VEL_LIMIT
    axis.trap_traj.config.accel_limit = LAUNCH_TRAP_ACCEL_LIMIT
    axis.trap_traj.config.decel_limit = LAUNCH_TRAP_DECEL_LIMIT
    result = move_pan_to(axis, start_pos, description="start position", check_abort=check_abort)
    if result != True:
        return result if result == 'estop' else 'timeout'
    return 'done'


def compute_anchored_launch_params(odrv, axis, vel_limit,
                                    accel_limit=None, decel_limit=None):
    """Computes start_pos/final_target for the anchored-release scheme:
    deceleration always begins at the same physical arm angle (the
    "virtual hard stop", LAUNCH_RELEASE_ANCHOR_DEG_FROM_START past the near
    hard stop + buffer) regardless of vel_limit, by solving for however far
    back the arm needs to start so pure acceleration reaches exactly
    vel_limit right at that anchor (v^2 = 2*a*d kinematics).

    Returns a dict with start_pos, release_anchor_pos, final_target, or
    None (printing why) if the computed start_pos or final_target would
    violate either hard-stop safety margin - callers must not fire without
    checking this."""
    accel = LAUNCH_TRAP_ACCEL_LIMIT if accel_limit is None else accel_limit
    decel = LAUNCH_TRAP_DECEL_LIMIT if decel_limit is None else decel_limit

    now = axis.pos_estimate
    near_raw_diff = LAUNCH_HARD_STOP_RAW - odrv.onboard_encoder0.raw
    near_correction = near_raw_diff - round(near_raw_diff)
    near_hard_stop_pos = now + near_correction
    release_anchor_pos = (near_hard_stop_pos + HARD_STOP_BUFFER_TURNS
                           + LAUNCH_RELEASE_ANCHOR_DEG_FROM_START / 360.0)

    accel_distance = vel_limit ** 2 / (2 * accel)
    decel_distance = vel_limit ** 2 / (2 * decel)
    start_pos = release_anchor_pos - accel_distance
    final_target = release_anchor_pos + decel_distance

    min_start_pos = near_hard_stop_pos + LAUNCH_MIN_START_BUFFER_TURNS
    if start_pos < min_start_pos:
        print(f"ERROR: vel_limit={vel_limit:.3f} needs start_pos={start_pos:.4f} turns, "
              f"which is closer than {LAUNCH_MIN_START_BUFFER_TURNS * 360:.0f}deg to the "
              f"near hard stop (min safe start_pos={min_start_pos:.4f}). Try a lower "
              "vel_limit or a higher accel_limit.")
        return None

    safe_max = max_safe_launch_pos(odrv, axis)
    if final_target > safe_max:
        print(f"ERROR: vel_limit={vel_limit:.3f} needs final_target={final_target:.4f} turns, "
              f"past the far hard-stop safety margin (safe max={safe_max:.4f}). Try a lower "
              "vel_limit or a higher decel_limit.")
        return None

    return {
        "start_pos": start_pos,
        "release_anchor_pos": release_anchor_pos,
        "final_target": final_target,
        "vel_limit": vel_limit,
        "accel_limit": accel,
        "decel_limit": decel,
    }


def fire_shot_anchored(axis, ready_pos, params, check_abort=None, on_windup_done=None):
    """Anchored-release swing -> hold -> return, using params from
    compute_anchored_launch_params(). Unlike fire_shot() (fixed start,
    variable end), this has a variable start_pos per shot (the windup),
    so there's an extra move first: from wherever the arm currently rests
    (ready_pos, from prepare_launch) back to this shot's start_pos, before
    the actual accel/decel launch move to final_target. Returns 'done',
    'estop', or 'timeout'.

    on_windup_done, if given, is called with no arguments right after the
    windup move completes and right before the launch move is commanded -
    lets a caller (e.g. a telemetry sampler) start timing exactly at the
    start of the accel/decel move instead of including the windup's own
    motion, which isn't the interesting part to log."""
    print("\nWinding up to start_pos...")
    axis.trap_traj.config.vel_limit = RETURN_VEL_LIMIT
    axis.trap_traj.config.accel_limit = LAUNCH_TRAP_ACCEL_LIMIT
    axis.trap_traj.config.decel_limit = LAUNCH_TRAP_DECEL_LIMIT
    result = move_pan_to(axis, params["start_pos"], description="windup position",
                          check_abort=check_abort)
    if result != True:
        return result if result == 'estop' else 'timeout'

    if on_windup_done is not None:
        on_windup_done()

    print("LAUNCH (anchored release)")
    axis.trap_traj.config.vel_limit = params["vel_limit"]
    axis.trap_traj.config.accel_limit = params["accel_limit"]
    axis.trap_traj.config.decel_limit = params["decel_limit"]
    result = move_pan_to(axis, params["final_target"], description="final target",
                          check_abort=check_abort)
    if result != True:
        return result if result == 'estop' else 'timeout'

    print(f"  Holding for {LAUNCH_HOLD_SECONDS:.1f}s...")
    hold_start = time.time()
    while time.time() - hold_start < LAUNCH_HOLD_SECONDS:
        if check_abort is not None and check_abort():
            axis.requested_state = AxisState.IDLE
            return 'estop'
        time.sleep(0.02)

    print("Returning to ready position")
    axis.trap_traj.config.vel_limit = RETURN_VEL_LIMIT
    axis.trap_traj.config.accel_limit = LAUNCH_TRAP_ACCEL_LIMIT
    axis.trap_traj.config.decel_limit = LAUNCH_TRAP_DECEL_LIMIT
    result = move_pan_to(axis, ready_pos, description="ready position", check_abort=check_abort)
    if result != True:
        return result if result == 'estop' else 'timeout'
    return 'done'
