"""
Combined pan + launch control - jog the pan axis to aim, then launch.

Two independent ODrive S1 axes are armed and held simultaneously. Neither
axis requires manual positioning anymore - both compute their reference
position each run from onboard_encoder0.raw (a genuine absolute
magnetic-sensor reading, stable across power cycles), captured once with pan_read_position.py /
launch_read_position.py:
  - LAUNCH axis: hard stop position computed from LAUNCH_HARD_STOP_RAW, with
    a 90-degree default launch/return swing - overridable per run (see
    below) for Phase 2 Track A stop-angle sweep testing.
  - PAN axis: locked position computed from PAN_LOCK_RAW, then same
    small-step jog tool as pan_axis_test.py, hard-limited to a +/-45 degree
    window around that locked position.

Setup flow:
  0. Prompted for this session's swing distance in degrees (blank = default
     90). Only values <= 90 are known-safe without further checking; going
     past 90 hasn't been verified for mechanical clearance.
  1. LAUNCH arms holding wherever it physically is (zero error, no jump),
     then auto-drives to the computed start position - no manual placement
     at the hard stop needed. If the computed correction is implausibly
     large (> LAUNCH_HARD_STOP_MAX_CORRECTION_TURNS), the script aborts
     rather than trusting it.
  2. PAN arms holding wherever it physically is (zero error, no jump), then
     auto-drives to the locked position (computed from PAN_LOCK_RAW) - same
     idea, works across power cycles. If PAN is found more than +/-45 deg
     from that position, the script aborts instead of swinging that far
     automatically.
  3. Controls:
       SPACE - launch: swings to launch position, holds LAUNCH_HOLD_SECONDS,
               then auto-returns to start - no second keypress needed
       A     - jog PAN one step counter-clockwise
       D     - jog PAN one step clockwise
       Shift+A/D - bigger PAN jog step   Ctrl+A/D - finer PAN jog step
       ESC   - E-STOP: disables BOTH motors immediately, even mid-move, exits
       Q     - quit gracefully, disables both motors

This script is the only one that talks to both ODrives at once - it always
addresses each one by its own serial number, never a bare find_any().
"""
import queue
import sys
import threading
import time
import tty
import termios

import odrive
from odrive.enums import AxisState, ControlMode, InputMode

# --- Identity -------------------------------------------------------------
LAUNCH_SERIAL = "394A355F3231"
PAN_SERIAL = "3970355B3231"

# --- Launch axis motion tuning (turns) ------------------------------------
# Carried over from launch_arm_control.py, already validated on this rig.
LAUNCH_OFFSET_TURNS = 0.25          # 90 degrees, measured from the actual start (not the hard stop) -
                                     # DEFAULT only; overridable at runtime for Phase 2 Track A stop-angle
                                     # sweep testing, see swing-degrees prompt at the top of main()
HARD_STOP_BUFFER_DEG = 10.0         # start position is this many degrees off the hard stop -
                                     # reduced 2026-09-21 (was 15.0) for more accel range
HARD_STOP_BUFFER_TURNS = HARD_STOP_BUFFER_DEG / 360.0

# Hard stop position, auto-computed each run from onboard_encoder0.raw (same
# power-cycle-stable anchoring trick as PAN_LOCK_RAW below) instead of
# requiring you to manually place the arm at the hard stop every run.
# Captured once with launch_read_position.py.
LAUNCH_HARD_STOP_RAW = 0.2073       # re-captured 2026-09-21 for the new (larger-diameter) arm -
                                     # hard stop itself didn't move, arm rests slightly higher
                                     # against it now (was 0.2036, captured 2026-09-19)
# Sanity bound on the computed correction - not a hard mechanical safety
# limit like PAN's wiring constraint, just a "something's probably wrong,
# stop and look" check (e.g. arm bumped while idle between runs).
LAUNCH_HARD_STOP_MAX_CORRECTION_TURNS = 0.083  # ~30 deg

LAUNCH_TRAP_ACCEL_LIMIT = 100.0
LAUNCH_TRAP_DECEL_LIMIT = 100.0
LAUNCH_VEL_LIMIT = 30.0             # turns/s, fast flick toward the launch position
RETURN_VEL_LIMIT = 1.5              # turns/s, slow/gentle move back to start - halved
                                     # 2026-09-21 (was 3.0), arm was hitting the hard stop
LAUNCH_HOLD_SECONDS = 1.0           # dwell at launch position before auto-returning to start
LAUNCH_CONTROLLER_VEL_LIMIT = LAUNCH_VEL_LIMIT * 1.5  # outer safety ceiling, must stay above LAUNCH_VEL_LIMIT
LAUNCH_POS_GAIN = 40.0               # raised from 20.0 to match PAN - "overshoot then slow creep
                                      # back" meant the residual-error correction was too weak given
                                      # the now-tighter damping below, not that damping itself was bad
# Same live-read defaults (0.1667 / 0.3333) this axis started at as PAN did
# before its fix - live-checked, same motor/hardware. Settling oscillation
# after the fast recoil impulse (launch flick, and the reset move) is the
# same slow-hunting signature as PAN's: integral gain too dominant relative
# to damping. Same rebalance applied - more vel_gain damping, less
# vel_integrator_gain windup - without touching pos_gain or the swing-speed
# constants above.
LAUNCH_VEL_GAIN = 0.35
# Dropped further from 0.20 - minor hunting persisted at 0.20, still the
# windup signature (slow, not a fast buzz), so cutting integral action
# again rather than touching vel_gain (which governs damping, not this).
LAUNCH_VEL_INTEGRATOR_GAIN = 0.12

# --- Pan axis motion tuning (turns) ---------------------------------------
# Carried over from pan_axis_test.py, already validated on this rig - do not
# widen without re-testing, wiring from both axes crosses this joint.
#
# pos_estimate was found to NOT survive a power cycle for the same physical
# position (pos_vel_mapper runs in RELATIVE_MODE on this
# hardware) - so the locked position is anchored instead to
# onboard_encoder0.raw, a genuinely absolute magnetic-sensor reading that IS
# stable across power cycles. Captured live with pan_read_position.py, which
# prints both raw and pos_estimate at your hand-aligned position.
PAN_LOCK_RAW = 0.1091                # onboard_encoder0.raw fraction (0-1 turn) at the desired
                                     # locked position, captured 2026-09-21 via pan_read_position.py
                                     # on the new dedicated test stand
JOG_STEP_TURNS = 0.02               # ~7.2 degrees per keypress
JOG_STEP_FINE_TURNS = JOG_STEP_TURNS / 2     # Ctrl+A/D - finer nudge
JOG_STEP_COARSE_TURNS = JOG_STEP_TURNS * 3   # Shift+A/D - bigger jump
MAX_RANGE_TURNS = 0.125             # +/-45 degrees from the locked position - hard software limit
JOG_VEL_LIMIT = 0.3                 # turns/s
JOG_ACCEL_LIMIT = 1.0
JOG_DECEL_LIMIT = 1.0
# Wider than JOG_VEL_LIMIT*1.5: this is the disarm ceiling for the *holding*
# loop's corrective velocity command (see PAN_POS_GAIN below), not the speed
# of commanded jog/launch moves - those still obey trap_traj's JOG_VEL_LIMIT.
# Without this headroom, a raised PAN_POS_GAIN would make a moderate hand
# push trip a VELOCITY_LIMIT_VIOLATION disarm.
PAN_CONTROLLER_VEL_LIMIT = 0.6
# Raised from the ODrive's live default (20.0) - felt "gummy"/easy to push
# by hand at that value.
PAN_POS_GAIN = 40.0
# vel_gain/vel_integrator_gain started at the ODrive's live-read defaults
# (0.1667 / 0.3333) but doubling pos_gain above with damping left unchanged
# produced a slow "hunting" oscillation around the lock position (classic
# cascade-loop symptom: outer loop got more aggressive, inner loop didn't
# get more damping to match). vel_gain raised ~2x in proportion to pos_gain
# to restore stability margin; vel_integrator_gain pulled down since a slow
# (not fast/buzzy) oscillation points at integral windup specifically as the
# cause, not insufficient proportional damping alone.
PAN_VEL_GAIN = 0.35
PAN_VEL_INTEGRATOR_GAIN = 0.20

# --- Shared ----------------------------------------------------------------
WATCHDOG_FEED_INTERVAL = 0.2        # seconds; must be well under axis config.watchdog_timeout
POSITION_TOLERANCE = 0.005          # turns
MOVE_TIMEOUT = 5.0                  # seconds


def keyboard_reader(key_queue, stop_event):
    """Background thread continuously pushing raw keypresses onto key_queue,
    so a SPACE e-stop is seen immediately even mid-move."""
    while not stop_event.is_set():
        ch = sys.stdin.read(1)
        if ch:
            key_queue.put(ch)


class WatchdogFeeder:
    """Feeds one axis's watchdog on a background thread so blocking waits in
    the main thread don't trip its watchdog timeout while armed."""

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


def set_trap_traj_limits(axis, vel, accel, decel):
    axis.trap_traj.config.vel_limit = vel
    axis.trap_traj.config.accel_limit = accel
    axis.trap_traj.config.decel_limit = decel


def estop_all(axes):
    for axis in axes:
        axis.requested_state = AxisState.IDLE


def move_to_position(moving_axis, position, description, key_queue, all_axes):
    """Returns 'reached', 'timeout', or 'estop'. On estop, every axis in
    all_axes is disabled immediately, not just moving_axis."""
    print(f"  Moving to {description} ({position:.4f} turns / {position * 360:.1f} deg)...")
    moving_axis.controller.input_pos = position

    start_time = time.time()
    while True:
        try:
            key = key_queue.get_nowait()
        except queue.Empty:
            key = None

        if key == '\x1b':
            estop_all(all_axes)
            print("\n!!! E-STOP !!! Both motors disabled immediately.")
            return 'estop'

        current_pos = moving_axis.pos_estimate
        error = abs(current_pos - position)

        if error < POSITION_TOLERANCE:
            print(f"  Reached {description}.")
            return 'reached'

        if time.time() - start_time > MOVE_TIMEOUT:
            print(f"  WARNING: timed out moving to {description} (error={error:.4f} turns)")
            return 'timeout'

        time.sleep(0.02)


def hold_for(duration, key_queue, all_axes):
    """Blocks for `duration` seconds while still watching for a SPACE
    e-stop. Returns 'estop' or 'done'."""
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            key = key_queue.get_nowait()
        except queue.Empty:
            key = None

        if key == '\x1b':
            estop_all(all_axes)
            print("\n!!! E-STOP !!! Both motors disabled immediately.")
            return 'estop'

        time.sleep(0.02)
    return 'done'


def connect(serial, label):
    print(f"Connecting to {label} axis (S/N {serial})...")
    odrv = odrive.find_any(serial_number=serial, timeout=10)
    print(f"  Connected. serial={odrv.serial_number:012X} "
          f"hw={odrv.hw_version_major}.{odrv.hw_version_minor} "
          f"fw={odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}")
    axis = odrv.axis0
    if axis.current_state != AxisState.IDLE:
        print(f"  {label} axis was not idle - forcing IDLE.")
        axis.requested_state = AxisState.IDLE
        time.sleep(0.3)
    odrv.clear_errors()
    return odrv, axis


def main():
    print("=" * 55)
    print("COMBINED PAN + LAUNCH CONTROL")
    print("=" * 55)

    # Swing distance for this session only - lets Phase 2 Track A (and this
    # release-angle check) test different stop angles without editing the
    # script. Only ever LESS than the validated 90 deg default is safe to
    # assume with no further checking; going past 90 deg hasn't been
    # verified for mechanical clearance, so treat that as a separate,
    # cautious step-by-step check, not a default part of this sweep.
    swing_input = input(f"Swing distance for this session in degrees "
                         f"(default {LAUNCH_OFFSET_TURNS * 360:.0f}): ").strip()
    launch_offset_turns = (float(swing_input) / 360.0) if swing_input else LAUNCH_OFFSET_TURNS
    print(f"  Using {launch_offset_turns * 360:.1f} deg swing for this session.")

    odrv_launch, axis_launch = connect(LAUNCH_SERIAL, "LAUNCH")
    odrv_pan, axis_pan = connect(PAN_SERIAL, "PAN")
    all_axes = [axis_launch, axis_pan]

    # --- LAUNCH setup: hard stop position computed from onboard_encoder0.raw
    # (same power-cycle-stable anchoring as PAN) - no manual placement needed ---
    print("\n=== LAUNCH AXIS SETUP ===")
    launch_now = axis_launch.pos_estimate
    current_raw = odrv_launch.onboard_encoder0.raw
    raw_diff = LAUNCH_HARD_STOP_RAW - current_raw
    hard_stop_correction = raw_diff - round(raw_diff)   # wrap to (-0.5, 0.5] turns

    print(f"  Current LAUNCH position: {launch_now:.4f} turns ({launch_now * 360:.1f} deg).")
    print(f"  onboard_encoder0.raw: {current_raw:.4f} (hard stop target {LAUNCH_HARD_STOP_RAW:.4f}) "
          f"-> correction {hard_stop_correction * 360:+.1f} deg")

    if abs(hard_stop_correction) > LAUNCH_HARD_STOP_MAX_CORRECTION_TURNS:
        print(f"ERROR: computed hard-stop correction is {abs(hard_stop_correction) * 360:.1f} deg - "
              f"farther than the {LAUNCH_HARD_STOP_MAX_CORRECTION_TURNS * 360:.0f} deg sanity bound. "
              "Something's likely off (arm bumped, wrong LAUNCH_HARD_STOP_RAW, etc.) - check by hand "
              "before proceeding rather than trusting this correction.")
        return

    hard_stop_pos = launch_now + hard_stop_correction
    start_pos = hard_stop_pos + HARD_STOP_BUFFER_TURNS
    launch_pos = start_pos + launch_offset_turns
    print(f"  Hard stop: {hard_stop_pos:.4f} turns ({hard_stop_pos * 360:.1f} deg)")
    print(f"  Start (stop + {HARD_STOP_BUFFER_DEG:.0f} deg): {start_pos:.4f} turns ({start_pos * 360:.1f} deg)")
    print(f"  Launch (start + {launch_offset_turns * 360:.1f} deg): {launch_pos:.4f} turns ({launch_pos * 360:.1f} deg)")
    odrv_launch.clear_errors()

    axis_launch.controller.config.control_mode = ControlMode.POSITION_CONTROL
    axis_launch.controller.config.input_mode = InputMode.TRAP_TRAJ
    axis_launch.controller.config.pos_gain = LAUNCH_POS_GAIN
    axis_launch.controller.config.vel_gain = LAUNCH_VEL_GAIN
    axis_launch.controller.config.vel_integrator_gain = LAUNCH_VEL_INTEGRATOR_GAIN
    axis_launch.controller.config.vel_limit = LAUNCH_CONTROLLER_VEL_LIMIT
    set_trap_traj_limits(axis_launch, RETURN_VEL_LIMIT, LAUNCH_TRAP_ACCEL_LIMIT, LAUNCH_TRAP_DECEL_LIMIT)
    # Arm holding exactly where the arm physically is right now - zero
    # position error on engage (it's likely resting near start_pos from the
    # previous run's auto-return, not at the hard stop - that's fine, the
    # hard stop is now a computed reference, not somewhere we move to and
    # rest at). The move to start_pos is issued explicitly afterward, once
    # armed and the e-stop-capable keyboard reader is running.
    axis_launch.controller.input_pos = launch_now

    print("Engaging LAUNCH closed-loop control, holding current position...")
    axis_launch.requested_state = AxisState.CLOSED_LOOP_CONTROL
    time.sleep(0.3)

    if axis_launch.current_state != AxisState.CLOSED_LOOP_CONTROL:
        print(f"ERROR: LAUNCH did not reach closed loop control. "
              f"current_state={axis_launch.current_state} active_errors={axis_launch.active_errors} "
              f"disarm_reason={axis_launch.disarm_reason}")
        axis_launch.requested_state = AxisState.IDLE
        return

    print(f"  LAUNCH engaged (pos_estimate={axis_launch.pos_estimate:.4f} turns).")
    launch_feeder = WatchdogFeeder(axis_launch, WATCHDOG_FEED_INTERVAL)
    launch_feeder.start()

    # --- PAN setup: arm holding wherever it currently is (zero jump), then
    # explicitly auto-drive to the saved locked position below. The target is
    # computed fresh each run from onboard_encoder0.raw (stable across power
    # cycles) rather than trusting pos_estimate's absolute value directly. ---
    print("\n=== PAN AXIS SETUP ===")
    pan_now = axis_pan.pos_estimate
    current_raw = odrv_pan.onboard_encoder0.raw
    raw_diff = PAN_LOCK_RAW - current_raw
    pan_correction = raw_diff - round(raw_diff)   # wrap to (-0.5, 0.5] turns
    pan_lock_target = pan_now + pan_correction

    print(f"  Current PAN position: {pan_now:.4f} turns ({pan_now * 360:.1f} deg).")
    print(f"  onboard_encoder0.raw: {current_raw:.4f} (target {PAN_LOCK_RAW:.4f}) "
          f"-> correction {pan_correction * 360:+.1f} deg")
    print(f"  Locked position (this session): {pan_lock_target:.4f} turns ({pan_lock_target * 360:.1f} deg).")

    if abs(pan_correction) > MAX_RANGE_TURNS:
        print(f"ERROR: PAN is {abs(pan_correction) * 360:.1f} deg from the locked "
              f"position - farther than the {MAX_RANGE_TURNS * 360:.0f} deg safety window. "
              "Manually move PAN closer to the locked position before running this script.")
        axis_launch.requested_state = AxisState.IDLE
        launch_feeder.stop()
        return

    odrv_pan.clear_errors()
    axis_pan.controller.config.control_mode = ControlMode.POSITION_CONTROL
    axis_pan.controller.config.input_mode = InputMode.TRAP_TRAJ
    axis_pan.controller.config.pos_gain = PAN_POS_GAIN
    axis_pan.controller.config.vel_gain = PAN_VEL_GAIN
    axis_pan.controller.config.vel_integrator_gain = PAN_VEL_INTEGRATOR_GAIN
    axis_pan.controller.config.vel_limit = PAN_CONTROLLER_VEL_LIMIT
    set_trap_traj_limits(axis_pan, JOG_VEL_LIMIT, JOG_ACCEL_LIMIT, JOG_DECEL_LIMIT)
    # Arm holding exactly where PAN physically is - zero position error on
    # engage. The move to pan_lock_target is issued explicitly afterward,
    # once armed and the e-stop-capable keyboard reader is running.
    axis_pan.controller.input_pos = pan_now

    print("Engaging PAN closed-loop control, holding current position...")
    axis_pan.requested_state = AxisState.CLOSED_LOOP_CONTROL
    time.sleep(0.3)

    if axis_pan.current_state != AxisState.CLOSED_LOOP_CONTROL:
        print(f"ERROR: PAN did not reach closed loop control. "
              f"current_state={axis_pan.current_state} active_errors={axis_pan.active_errors} "
              f"disarm_reason={axis_pan.disarm_reason}")
        axis_pan.requested_state = AxisState.IDLE
        axis_launch.requested_state = AxisState.IDLE
        launch_feeder.stop()
        return

    print(f"  PAN engaged (pos_estimate={axis_pan.pos_estimate:.4f} turns).")
    pan_feeder = WatchdogFeeder(axis_pan, WATCHDOG_FEED_INTERVAL)
    pan_feeder.start()

    pan_center = pan_lock_target
    pan_min = pan_center - MAX_RANGE_TURNS
    pan_max = pan_center + MAX_RANGE_TURNS
    print(f"  Allowed jog range around lock: {pan_min:.4f} to {pan_max:.4f} turns "
          f"({pan_min * 360:.1f} to {pan_max * 360:.1f} deg)")

    # --- Shared keyboard reader (started only now - everything above used
    # normal input(), which can't coexist with a raw-mode background reader) ---
    key_queue = queue.Queue()
    stop_reader = threading.Event()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)
    reader_thread = threading.Thread(target=keyboard_reader, args=(key_queue, stop_reader), daemon=True)
    reader_thread.start()

    pan_target = pan_center

    try:
        print("Moving LAUNCH to the start position...")
        result = move_to_position(axis_launch, start_pos, "launch start position", key_queue, all_axes)
        if result == 'estop':
            return

        print("Moving PAN to the locked position...")
        result = move_to_position(axis_pan, pan_lock_target, "pan locked position", key_queue, all_axes)
        if result == 'estop':
            return

        print("\n" + "-" * 55)
        print("READY")
        print("-" * 55)
        print("  SPACE - launch (auto-holds, then returns), LAUNCH axis")
        print("  A     - jog PAN one step counter-clockwise")
        print("  D     - jog PAN one step clockwise")
        print("  Shift+A/D - bigger PAN jog step   Ctrl+A/D - finer PAN jog step")
        print("  ESC   - E-STOP (disables BOTH motors immediately, even mid-move)")
        print("  Q     - quit and disable both motors")
        print("-" * 55 + "\n")

        while True:
            try:
                key = key_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if key == '\x1b':
                estop_all(all_axes)
                print("\n!!! E-STOP !!! Both motors disabled immediately.")
                break

            elif key == ' ':
                print("\nLAUNCH")
                set_trap_traj_limits(axis_launch, LAUNCH_VEL_LIMIT, LAUNCH_TRAP_ACCEL_LIMIT, LAUNCH_TRAP_DECEL_LIMIT)
                result = move_to_position(axis_launch, start_pos + launch_offset_turns,
                                           "launch position", key_queue, all_axes)
                if result == 'estop':
                    break

                print(f"  Holding at launch position for {LAUNCH_HOLD_SECONDS:.1f}s...")
                hold_result = hold_for(LAUNCH_HOLD_SECONDS, key_queue, all_axes)
                if hold_result == 'estop':
                    break

                print("\nReturning to start")
                set_trap_traj_limits(axis_launch, RETURN_VEL_LIMIT, LAUNCH_TRAP_ACCEL_LIMIT, LAUNCH_TRAP_DECEL_LIMIT)
                result = move_to_position(axis_launch, start_pos, "start position", key_queue, all_axes)
                if result == 'estop':
                    break

            elif key in ('\x01', '\x04') or key in ('A', 'D') or key.lower() in ('a', 'd'):
                if key in ('\x01', '\x04'):            # Ctrl+A / Ctrl+D - fine
                    step = -JOG_STEP_FINE_TURNS if key == '\x01' else JOG_STEP_FINE_TURNS
                elif key in ('A', 'D'):                 # Shift+A / Shift+D - coarse
                    step = -JOG_STEP_COARSE_TURNS if key == 'A' else JOG_STEP_COARSE_TURNS
                else:                                   # plain - base step
                    step = -JOG_STEP_TURNS if key.lower() == 'a' else JOG_STEP_TURNS
                proposed = pan_target + step

                if proposed < pan_min or proposed > pan_max:
                    print(f"\nBLOCKED: would exceed PAN's allowed range "
                          f"({pan_min:.4f} to {pan_max:.4f} turns). Not moving.")
                    continue

                pan_target = proposed
                direction = "CCW" if step < 0 else "CW"
                print(f"\nPAN jog {direction} -> target {pan_target:.4f} turns "
                      f"({pan_target * 360:.1f} deg, {(pan_target - pan_center) * 360:+.1f} from center)")
                result = move_to_position(axis_pan, pan_target, "pan jog target", key_queue, all_axes)

                if result == 'estop':
                    break

            elif key.lower() == 'q':
                print("\nShutting down...")
                break

            elif key == '\x03':  # Ctrl+C
                print("\nInterrupted!")
                break

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
