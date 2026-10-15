"""
Quick range-accuracy test: enter a tape-measure distance, the script solves
for the swing angle the fitted calibration curve says should land there
(z=0, same height as the launcher base), fires it, then lets you enter a
new distance or press SPACE to repeat the same one - no restart needed.
PAN auto-locks to a saved position on startup (same as combined_control.py)
and can still be jogged with A/D (Shift=coarse, Ctrl=fine) while you work.

Hard stop / pan-lock positions are both auto-computed from
onboard_encoder0.raw - no manual placement needed, survives power cycles.

Calibration data below uses the CORRECTED pivot height (175mm, not the
original mis-measured 100mm) re-solved from the four
Track A reference shots.
"""
import math
import queue
import sys
import threading
import time
import tty
import termios

import odrive
from odrive.enums import AxisState, ControlMode, InputMode

# --- Identity ---------------------------------------------------------------
LAUNCH_SERIAL = "394A355F3231"
PAN_SERIAL = "3970355B3231"

# --- Geometry -----------------------------------------------------------
PIVOT_M = 0.2364    # updated 2026-09-21 for the new arm/table redesign (was 0.175 -
                    # base raised 61.4mm)
ARM_M = 0.180       # updated 2026-09-21 for the new arm/table redesign (was 0.147,
                    # pivot to ball-rest center)
G = 9.81
TAPE_OFFSET_IN = 50.0 / 25.4   # pivot -> base edge = tape 0 on the new table setup
                                # (tape now reads directly from the base edge, no more
                                # separate tape-start point ~56in further out); confirmed
                                # still 50mm on the rebuilt rig, 2026-09-21

# --- Calibration data (swing deg, speed m/s, angle deg above horizontal) ---
# STALE as of the 2026-09-21 geometry change above - fit on the OLD 0.147m arm /
# 0.175m pivot, does not transfer to the new geometry. Do not trust
# solve_swing_for_range()/landing_range_m() output until Track A is redone
# (fire shots at each swing angle, measure landing distance + slomo flight
# time, re-solve for speed/angle) on the new arm and this table is replaced.
CAL = [
    (45.0, 2.138, 69.15),
    (60.0, 2.834, 44.70),
    (75.0, 3.838, 35.14),
    (90.0, 4.880, 26.00),
]

# --- Launch axis tuning (turns) - same validated values as combined_control.py
HARD_STOP_BUFFER_DEG = 10.0     # reduced 2026-09-21 (was 15.0) for more accel range
HARD_STOP_BUFFER_TURNS = HARD_STOP_BUFFER_DEG / 360.0
LAUNCH_HARD_STOP_RAW = 0.2073   # re-captured 2026-09-21 for the new (larger-diameter) arm -
                                # hard stop itself didn't move (was 0.2036)
LAUNCH_HARD_STOP_MAX_CORRECTION_TURNS = 0.083

LAUNCH_TRAP_ACCEL_LIMIT = 100.0
LAUNCH_TRAP_DECEL_LIMIT = 100.0
LAUNCH_VEL_LIMIT = 30.0
RETURN_VEL_LIMIT = 1.5     # halved 2026-09-21 (was 3.0), arm was hitting the hard stop
LAUNCH_CONTROLLER_VEL_LIMIT = LAUNCH_VEL_LIMIT * 1.5
LAUNCH_POS_GAIN = 40.0
LAUNCH_VEL_GAIN = 0.35
LAUNCH_VEL_INTEGRATOR_GAIN = 0.12
LAUNCH_HOLD_SECONDS = 1.0

# --- Pan axis tuning (turns) - same validated values as combined_control.py
PAN_LOCK_RAW = 0.1091
JOG_STEP_TURNS = 0.02
JOG_STEP_FINE_TURNS = JOG_STEP_TURNS / 2     # Ctrl+A/D - finer nudge
JOG_STEP_COARSE_TURNS = JOG_STEP_TURNS * 3   # Shift+A/D - bigger jump
MAX_RANGE_TURNS = 0.125
JOG_VEL_LIMIT = 0.3
JOG_ACCEL_LIMIT = 1.0
JOG_DECEL_LIMIT = 1.0
PAN_CONTROLLER_VEL_LIMIT = 0.6
PAN_POS_GAIN = 40.0
PAN_VEL_GAIN = 0.35
PAN_VEL_INTEGRATOR_GAIN = 0.20

WATCHDOG_FEED_INTERVAL = 0.2
POSITION_TOLERANCE = 0.005
MOVE_TIMEOUT = 5.0


# --- Monotone cubic interpolation (Fritsch-Carlson) ---
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


# --- Hardware plumbing (same pattern as combined_control.py) ---------------
class WatchdogFeeder:
    def __init__(self, axis, interval):
        self._axis = axis
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._axis.watchdog_feed()
            except Exception:
                pass
            self._stop.wait(self._interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


def keyboard_reader(key_queue, stop_event):
    while not stop_event.is_set():
        ch = sys.stdin.read(1)
        if ch:
            key_queue.put(ch)


def set_trap_traj_limits(axis, vel, accel, decel):
    axis.trap_traj.config.vel_limit = vel
    axis.trap_traj.config.accel_limit = accel
    axis.trap_traj.config.decel_limit = decel


def estop_all(axes):
    for axis in axes:
        axis.requested_state = AxisState.IDLE


def move_to_position(axis, position, description, key_queue, all_axes):
    print(f"  Moving to {description} ({position:.4f} turns / {position * 360:.1f} deg)...")
    axis.controller.input_pos = position
    start_time = time.time()
    while True:
        try:
            key = key_queue.get_nowait()
        except queue.Empty:
            key = None
        if key == '\x1b':
            estop_all(all_axes)
            print("\n!!! E-STOP !!! All motors disabled immediately.")
            return 'estop'
        error = abs(axis.pos_estimate - position)
        if error < POSITION_TOLERANCE:
            print(f"  Reached {description}.")
            return 'reached'
        if time.time() - start_time > MOVE_TIMEOUT:
            print(f"  WARNING: timed out moving to {description} (error={error:.4f} turns)")
            return 'timeout'
        time.sleep(0.02)


def hold_for(axis, duration, key_queue, all_axes):
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            key = key_queue.get_nowait()
        except queue.Empty:
            key = None
        if key == '\x1b':
            estop_all(all_axes)
            print("\n!!! E-STOP !!! All motors disabled immediately.")
            return 'estop'
        time.sleep(0.02)
    return 'done'


def connect(serial, label):
    print(f"Connecting to {label} axis (S/N {serial})...")
    odrv = odrive.find_any(serial_number=serial, timeout=10)
    axis = odrv.axis0
    if axis.current_state != AxisState.IDLE:
        print(f"  {label} axis was not idle - forcing IDLE.")
        axis.requested_state = AxisState.IDLE
        time.sleep(0.3)
    odrv.clear_errors()
    return odrv, axis


def fire_shot(axis, start_pos, swing_turns, key_queue, all_axes):
    """Runs the whole launch/hold/return sequence, watching the shared
    key_queue for an instant SPACE e-stop."""
    print("\nLAUNCH")
    set_trap_traj_limits(axis, LAUNCH_VEL_LIMIT, LAUNCH_TRAP_ACCEL_LIMIT, LAUNCH_TRAP_DECEL_LIMIT)
    result = move_to_position(axis, start_pos + swing_turns, "launch position", key_queue, all_axes)
    if result == 'estop':
        return 'estop'

    print(f"  Holding for {LAUNCH_HOLD_SECONDS:.1f}s...")
    if hold_for(axis, LAUNCH_HOLD_SECONDS, key_queue, all_axes) == 'estop':
        return 'estop'

    print("Returning to start")
    set_trap_traj_limits(axis, RETURN_VEL_LIMIT, LAUNCH_TRAP_ACCEL_LIMIT, LAUNCH_TRAP_DECEL_LIMIT)
    if move_to_position(axis, start_pos, "start position", key_queue, all_axes) == 'estop':
        return 'estop'
    return 'done'


NUMBER_CHARS = set("0123456789.-")


def main():
    print("=" * 55)
    print("RANGE TEST - LAUNCH (distance) + PAN (locked, jog-able)")
    print("=" * 55)

    odrv_launch, axis_launch = connect(LAUNCH_SERIAL, "LAUNCH")
    odrv_pan, axis_pan = connect(PAN_SERIAL, "PAN")
    all_axes = [axis_launch, axis_pan]

    # --- LAUNCH: hard stop position computed from onboard_encoder0.raw ---
    launch_now = axis_launch.pos_estimate
    launch_raw_diff = LAUNCH_HARD_STOP_RAW - odrv_launch.onboard_encoder0.raw
    launch_correction = launch_raw_diff - round(launch_raw_diff)
    if abs(launch_correction) > LAUNCH_HARD_STOP_MAX_CORRECTION_TURNS:
        print(f"ERROR: LAUNCH hard-stop correction is {abs(launch_correction) * 360:.1f} deg - "
              f"farther than the {LAUNCH_HARD_STOP_MAX_CORRECTION_TURNS * 360:.0f} deg sanity bound. "
              "Check the arm by hand before proceeding.")
        return

    hard_stop_pos = launch_now + launch_correction
    start_pos = hard_stop_pos + HARD_STOP_BUFFER_TURNS
    print(f"  LAUNCH hard stop: {hard_stop_pos:.4f} turns   Start: {start_pos:.4f} turns")

    axis_launch.controller.config.control_mode = ControlMode.POSITION_CONTROL
    axis_launch.controller.config.input_mode = InputMode.TRAP_TRAJ
    axis_launch.controller.config.pos_gain = LAUNCH_POS_GAIN
    axis_launch.controller.config.vel_gain = LAUNCH_VEL_GAIN
    axis_launch.controller.config.vel_integrator_gain = LAUNCH_VEL_INTEGRATOR_GAIN
    axis_launch.controller.config.vel_limit = LAUNCH_CONTROLLER_VEL_LIMIT
    set_trap_traj_limits(axis_launch, RETURN_VEL_LIMIT, LAUNCH_TRAP_ACCEL_LIMIT, LAUNCH_TRAP_DECEL_LIMIT)
    axis_launch.controller.input_pos = launch_now

    print("Engaging LAUNCH closed-loop control, holding current position...")
    axis_launch.requested_state = AxisState.CLOSED_LOOP_CONTROL
    time.sleep(0.3)
    if axis_launch.current_state != AxisState.CLOSED_LOOP_CONTROL:
        print(f"ERROR: LAUNCH did not reach closed loop control. "
              f"active_errors={axis_launch.active_errors} disarm_reason={axis_launch.disarm_reason}")
        axis_launch.requested_state = AxisState.IDLE
        return

    launch_feeder = WatchdogFeeder(axis_launch, WATCHDOG_FEED_INTERVAL)
    launch_feeder.start()

    # --- PAN: locked position computed from onboard_encoder0.raw ---
    pan_now = axis_pan.pos_estimate
    pan_raw_diff = PAN_LOCK_RAW - odrv_pan.onboard_encoder0.raw
    pan_correction = pan_raw_diff - round(pan_raw_diff)
    if abs(pan_correction) > MAX_RANGE_TURNS:
        print(f"ERROR: PAN correction is {abs(pan_correction) * 360:.1f} deg - "
              f"farther than the {MAX_RANGE_TURNS * 360:.0f} deg safety window. "
              "Manually move PAN closer to the locked position before running this script.")
        axis_launch.requested_state = AxisState.IDLE
        launch_feeder.stop()
        return

    pan_lock_target = pan_now + pan_correction
    print(f"  PAN locked position: {pan_lock_target:.4f} turns")

    odrv_pan.clear_errors()
    axis_pan.controller.config.control_mode = ControlMode.POSITION_CONTROL
    axis_pan.controller.config.input_mode = InputMode.TRAP_TRAJ
    axis_pan.controller.config.pos_gain = PAN_POS_GAIN
    axis_pan.controller.config.vel_gain = PAN_VEL_GAIN
    axis_pan.controller.config.vel_integrator_gain = PAN_VEL_INTEGRATOR_GAIN
    axis_pan.controller.config.vel_limit = PAN_CONTROLLER_VEL_LIMIT
    set_trap_traj_limits(axis_pan, JOG_VEL_LIMIT, JOG_ACCEL_LIMIT, JOG_DECEL_LIMIT)
    axis_pan.controller.input_pos = pan_now

    print("Engaging PAN closed-loop control, holding current position...")
    axis_pan.requested_state = AxisState.CLOSED_LOOP_CONTROL
    time.sleep(0.3)
    if axis_pan.current_state != AxisState.CLOSED_LOOP_CONTROL:
        print(f"ERROR: PAN did not reach closed loop control. "
              f"active_errors={axis_pan.active_errors} disarm_reason={axis_pan.disarm_reason}")
        axis_pan.requested_state = AxisState.IDLE
        axis_launch.requested_state = AxisState.IDLE
        launch_feeder.stop()
        return

    pan_feeder = WatchdogFeeder(axis_pan, WATCHDOG_FEED_INTERVAL)
    pan_feeder.start()

    pan_center = pan_lock_target
    pan_min = pan_center - MAX_RANGE_TURNS
    pan_max = pan_center + MAX_RANGE_TURNS
    pan_target = pan_center

    # One persistent raw-mode reader for the whole interactive session -
    # restarting this per shot leaks a thread blocked on the next keystroke,
    # which then steals the first character(s) of the next distance entry.
    key_queue = queue.Queue()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    stop_reader = threading.Event()
    tty.setraw(fd)
    reader_thread = threading.Thread(target=keyboard_reader, args=(key_queue, stop_reader), daemon=True)
    reader_thread.start()

    try:
        print("Moving LAUNCH to start position...")
        if move_to_position(axis_launch, start_pos, "launch start position", key_queue, all_axes) == 'estop':
            return
        print("Moving PAN to locked position...")
        if move_to_position(axis_pan, pan_lock_target, "pan locked position", key_queue, all_axes) == 'estop':
            return

        print("\n" + "-" * 55)
        print("Type digits for a tape distance (in), SPACE to fire it")
        print("  (blank SPACE repeats the last distance).")
        print("A / D  - jog PAN one step, any time (Shift=coarse, Ctrl=fine)")
        print("ESC    - E-STOP (disables both motors immediately)")
        print("Q      - quit")
        print("-" * 55 + "\n")

        last_tape_in = None
        buf = ""

        def prompt_line():
            p = "Tape distance (in)"
            p += f" [{last_tape_in:.1f}]" if last_tape_in is not None else ""
            print(f"\n{p}: {buf}", end="", flush=True)

        prompt_line()

        while True:
            ch = key_queue.get()

            if ch == '\x1b':
                estop_all(all_axes)
                print("\n!!! E-STOP !!! All motors disabled immediately.")
                return

            elif ch.lower() == 'q':
                print()
                break

            elif ch in ('\x01', '\x04') or ch in ('A', 'D') or ch.lower() in ('a', 'd'):
                if ch in ('\x01', '\x04'):              # Ctrl+A / Ctrl+D - fine
                    step = -JOG_STEP_FINE_TURNS if ch == '\x01' else JOG_STEP_FINE_TURNS
                elif ch in ('A', 'D'):                   # Shift+A / Shift+D - coarse
                    step = -JOG_STEP_COARSE_TURNS if ch == 'A' else JOG_STEP_COARSE_TURNS
                else:                                    # plain - base step
                    step = -JOG_STEP_TURNS if ch.lower() == 'a' else JOG_STEP_TURNS
                proposed = pan_target + step
                print()
                if proposed < pan_min or proposed > pan_max:
                    print(f"BLOCKED: would exceed PAN's allowed range "
                          f"({pan_min:.4f} to {pan_max:.4f} turns). Not moving.")
                else:
                    pan_target = proposed
                    direction = "CCW" if step < 0 else "CW"
                    print(f"PAN jog {direction} -> {pan_target:.4f} turns "
                          f"({(pan_target - pan_center) * 360:+.1f} deg from center)")
                    if move_to_position(axis_pan, pan_target, "pan jog target", key_queue, all_axes) == 'estop':
                        return
                prompt_line()

            elif ch in ('\x7f', '\x08'):
                if buf:
                    buf = buf[:-1]
                prompt_line()

            elif ch == '\x03':
                print()
                break

            elif ch == ' ':
                print()
                if buf:
                    try:
                        last_tape_in = float(buf)
                    except ValueError:
                        print("Not a number - try again.")
                        buf = ""
                        prompt_line()
                        continue
                    buf = ""
                if last_tape_in is None:
                    print("Enter a distance first.")
                    prompt_line()
                    continue

                true_r_in = last_tape_in + TAPE_OFFSET_IN
                true_r_m = true_r_in * 0.0254
                swing = solve_swing_for_range(true_r_m)
                if swing is None:
                    print(f"  {true_r_in:.1f}in true range is outside the calibrated "
                          f"45-90deg swing range ({landing_range_m(45) * 39.3701:.1f}in - "
                          f"{landing_range_m(90) * 39.3701:.1f}in). Try a different distance.")
                    prompt_line()
                    continue

                print(f"  Tape {last_tape_in:.1f}in -> true range {true_r_in:.1f}in -> swing {swing:.2f} deg")
                if fire_shot(axis_launch, start_pos, swing / 360.0, key_queue, all_axes) == 'estop':
                    return
                prompt_line()

            elif ch in NUMBER_CHARS:
                buf += ch
                prompt_line()

            # other keys ignored

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        stop_reader.set()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        launch_feeder.stop()
        pan_feeder.stop()
        print("Disabling both motors...")
        axis_launch.requested_state = AxisState.IDLE
        axis_pan.requested_state = AxisState.IDLE
        time.sleep(0.3)
        print("Both motors disabled. Safe to power off.\n")


if __name__ == "__main__":
    main()
