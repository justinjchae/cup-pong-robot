"""
Pan axis test - small-step jog tool.

This axis has wiring from BOTH the pan and launch axes running through/across
it, so an overshoot here risks yanking or twisting cables on the whole rig.
Unlike the launch script, this does NOT do one big swing - it moves in small
fixed steps per keypress, and refuses to go past a hard total-range limit no
matter how many times you press a key.

Flow:
  1. Motor starts IDLE. Current position is read and printed.
  2. Press ENTER to arm and hold at the current position (the "center").
  3. Press 'a' to jog one step counter-clockwise, 'd' to jog one step
     clockwise. Each step is JOG_STEP_TURNS (Shift+A/D = JOG_STEP_COARSE_TURNS,
     Ctrl+A/D = JOG_STEP_FINE_TURNS), and motion is intentionally slow.
  4. ESC is e-stop: disables the motor immediately, even mid-step, and
     ends the program. Q quits gracefully between moves.

Only ever talks to the PAN ODrive by serial number. The LAUNCH axis ODrive
is never connected to or touched by this script.
"""
import queue
import sys
import threading
import time
import tty
import termios

import odrive
from odrive.enums import AxisState, ControlMode, InputMode

# --- Identity -----------------------------------------------------------
PAN_SERIAL = "3970355B3231"  # pan axis ONLY. Launch axis (394A355F3231) is never touched.

# --- Motion tuning (turns) -----------------------------------------------
# Direct drive, no gearing: motor-shaft turns == pan degrees, 1:1.
JOG_STEP_TURNS = 0.02        # ~7.2 degrees per keypress - deliberately small for a first test
JOG_STEP_FINE_TURNS = JOG_STEP_TURNS / 2     # Ctrl+A/D - finer nudge
JOG_STEP_COARSE_TURNS = JOG_STEP_TURNS * 3   # Shift+A/D - bigger jump
MAX_RANGE_TURNS = 0.125      # +/- 45 degrees from wherever you arm it - hard software limit

JOG_VEL_LIMIT = 0.3          # turns/s - slow crawl, this is a wire-safety-critical axis
JOG_ACCEL_LIMIT = 1.0
JOG_DECEL_LIMIT = 1.0

# controller.config.vel_limit is a separate, outer safety ceiling the axis
# will DISARM if exceeded (independent of trap_traj.config.vel_limit). Must
# stay comfortably above JOG_VEL_LIMIT or a jog move will fault.
CONTROLLER_VEL_LIMIT = JOG_VEL_LIMIT * 1.5

POS_GAIN = 20.0

WATCHDOG_FEED_INTERVAL = 0.2  # seconds; must be well under axis config.watchdog_timeout
POSITION_TOLERANCE = 0.005    # turns
MOVE_TIMEOUT = 5.0            # seconds


def keyboard_reader(key_queue, stop_event):
    """Background thread continuously pushing raw keypresses onto key_queue,
    so a SPACE e-stop is seen immediately even mid-move."""
    while not stop_event.is_set():
        ch = sys.stdin.read(1)
        if ch:
            key_queue.put(ch)


class WatchdogFeeder:
    """Feeds the axis watchdog on a background thread so blocking waits in
    the main thread don't trip the axis's watchdog timeout while armed."""

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


def move_to_position(axis, position, description, key_queue):
    """Returns 'reached', 'timeout', or 'estop'."""
    print(f"  Moving to {description} ({position:.4f} turns / {position * 360:.1f} deg)...")
    axis.controller.input_pos = position

    start_time = time.time()
    while True:
        try:
            key = key_queue.get_nowait()
        except queue.Empty:
            key = None

        if key == '\x1b':
            axis.requested_state = AxisState.IDLE
            print("\n!!! E-STOP !!! Motor disabled immediately.")
            return 'estop'

        current_pos = axis.pos_estimate
        error = abs(current_pos - position)

        if error < POSITION_TOLERANCE:
            print(f"  Reached {description}.")
            return 'reached'

        if time.time() - start_time > MOVE_TIMEOUT:
            print(f"  WARNING: timed out moving to {description} (error={error:.4f} turns)")
            return 'timeout'

        time.sleep(0.02)


def main():
    print("=" * 55)
    print("PAN AXIS TEST (jog mode)")
    print("=" * 55)
    print("CAUTION: wiring from both axes runs across this joint.")
    print(f"Range is hard-limited to +/-{MAX_RANGE_TURNS * 360:.0f} deg from wherever you arm it.")
    print(f"Connecting to PAN axis only (S/N {PAN_SERIAL})...")
    odrv = odrive.find_any(serial_number=PAN_SERIAL, timeout=10)
    print(f"Connected. serial={odrv.serial_number:012X} "
          f"hw={odrv.hw_version_major}.{odrv.hw_version_minor} "
          f"fw={odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}")

    axis = odrv.axis0

    if axis.current_state != AxisState.IDLE:
        print("Axis was not idle - forcing IDLE.")
        axis.requested_state = AxisState.IDLE
        time.sleep(0.3)

    odrv.clear_errors()

    start_pos = axis.pos_estimate
    min_pos = start_pos - MAX_RANGE_TURNS
    max_pos = start_pos + MAX_RANGE_TURNS
    print(f"\nCurrent position: {start_pos:.4f} turns ({start_pos * 360:.1f} deg).")
    print(f"Allowed range once armed: {min_pos:.4f} to {max_pos:.4f} turns "
          f"({min_pos * 360:.1f} to {max_pos * 360:.1f} deg).")
    input("Press ENTER to arm and hold here as the test center...")

    axis.controller.config.control_mode = ControlMode.POSITION_CONTROL
    axis.controller.config.input_mode = InputMode.TRAP_TRAJ
    axis.controller.config.pos_gain = POS_GAIN
    axis.controller.config.vel_limit = CONTROLLER_VEL_LIMIT
    axis.trap_traj.config.vel_limit = JOG_VEL_LIMIT
    axis.trap_traj.config.accel_limit = JOG_ACCEL_LIMIT
    axis.trap_traj.config.decel_limit = JOG_DECEL_LIMIT
    axis.controller.input_pos = start_pos

    print("Engaging closed-loop control, holding center position...")
    axis.requested_state = AxisState.CLOSED_LOOP_CONTROL
    time.sleep(0.3)

    if axis.current_state != AxisState.CLOSED_LOOP_CONTROL:
        print(f"ERROR: motor did not reach closed loop control. "
              f"current_state={axis.current_state} active_errors={axis.active_errors} "
              f"disarm_reason={axis.disarm_reason}")
        axis.requested_state = AxisState.IDLE
        return

    print(f"Motor engaged and holding center (pos_estimate={axis.pos_estimate:.4f} turns).")

    feeder = WatchdogFeeder(axis, WATCHDOG_FEED_INTERVAL)
    feeder.start()

    key_queue = queue.Queue()
    stop_reader = threading.Event()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    current_target = start_pos
    print("\n" + "-" * 55)
    print("READY")
    print("-" * 55)
    print("  A     - jog one step counter-clockwise")
    print("  D     - jog one step clockwise")
    print("  Shift+A/D - bigger jog step   Ctrl+A/D - finer jog step")
    print("  ESC   - E-STOP (disables motor immediately, even mid-move)")
    print("  Q     - quit and disable motor")
    print("-" * 55 + "\n")

    tty.setraw(fd)
    reader_thread = threading.Thread(target=keyboard_reader, args=(key_queue, stop_reader), daemon=True)
    reader_thread.start()

    try:
        while True:
            try:
                key = key_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if key == '\x1b':
                axis.requested_state = AxisState.IDLE
                print("\n!!! E-STOP !!! Motor disabled immediately.")
                break

            step = None
            if key in ('\x01', '\x04'):          # Ctrl+A / Ctrl+D - fine
                step = -JOG_STEP_FINE_TURNS if key == '\x01' else JOG_STEP_FINE_TURNS
            elif key in ('A', 'D'):               # Shift+A / Shift+D - coarse
                step = -JOG_STEP_COARSE_TURNS if key == 'A' else JOG_STEP_COARSE_TURNS
            elif key.lower() in ('a', 'd'):        # plain - base step
                step = -JOG_STEP_TURNS if key.lower() == 'a' else JOG_STEP_TURNS

            if step is not None:
                proposed = current_target + step

                if proposed < min_pos or proposed > max_pos:
                    print(f"\nBLOCKED: would exceed allowed range "
                          f"({min_pos:.4f} to {max_pos:.4f} turns). Not moving.")
                    continue

                current_target = proposed
                direction = "CCW" if step < 0 else "CW"
                print(f"\nJog {direction} -> target {current_target:.4f} turns "
                      f"({current_target * 360:.1f} deg, {(current_target - start_pos) * 360:+.1f} from center)")
                result = move_to_position(axis, current_target, "jog target", key_queue)

                if result == 'estop':
                    break
                continue

            if key.lower() == 'q':
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
        feeder.stop()
        print("Disabling motor...")
        axis.requested_state = AxisState.IDLE
        time.sleep(0.3)
        print("Motor disabled. Safe to power off.\n")


if __name__ == "__main__":
    main()
