"""
Launch axis control - catapult-style test script.

Flow:
  1. Motor starts IDLE (freewheeling) so you can manually position the arm.
  2. Press ENTER to lock in that position as the "start" / ready position (zero).
  3. Motor engages closed-loop control and holds the start position.
  4. Press SPACE to launch (fast move to the launch position).
  5. Press SPACE again to return to the start position (slow, gentle move).
  6. Repeat step 4-5 as needed.

  ESC is a dedicated e-stop: it disables the motor immediately, even in the
  middle of a launch/return move, and ends the program. Q quits gracefully
  (only acts between moves, motor disabled).

Only ever talks to the LAUNCH/tilt ODrive by serial number. The PAN axis ODrive
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
LAUNCH_SERIAL = "394A355F3231"  # launch/tilt axis ONLY. Pan axis (3970355B3231) is never touched.

# --- Motion tuning (turns) -----------------------------------------------
# 1.0 turn = 360 degrees. Direct drive, no gearing: horizontal -> vertical
# is a quarter turn.
LAUNCH_OFFSET_TURNS = 0.25   # 90 degrees: start to launch (measured from the actual start, not the hard stop)

# The position you set by hand is a mechanical hard stop, not the start
# position itself. The real start is this many degrees off the stop, so the
# return move never drives the arm back into the stop.
HARD_STOP_BUFFER_DEG = 10.0     # reduced 2026-09-21 (was 15.0) for more accel range
HARD_STOP_BUFFER_TURNS = HARD_STOP_BUFFER_DEG / 360.0

# Trapezoidal trajectory limits, carried over from the previous Launch.py
# that was already tuned and worked on this same rig. Only vel_limit toggles
# between the launch shot and the return; accel/decel stay fixed.
TRAP_ACCEL_LIMIT = 100.0
TRAP_DECEL_LIMIT = 100.0

LAUNCH_VEL_LIMIT = 30.0      # turns/s, fast flick toward the launch position
RETURN_VEL_LIMIT = 1.5       # turns/s, slow/gentle move back to start - halved
                              # 2026-09-21 (was 3.0), arm was hitting the hard stop

# controller.config.vel_limit is a separate, outer safety ceiling the axis
# will DISARM if exceeded (independent of trap_traj.config.vel_limit above).
# Must stay comfortably above LAUNCH_VEL_LIMIT or the launch move will fault.
CONTROLLER_VEL_LIMIT = LAUNCH_VEL_LIMIT * 1.5

POS_GAIN = 20.0

WATCHDOG_FEED_INTERVAL = 0.2  # seconds; must be well under axis config.watchdog_timeout
POSITION_TOLERANCE = 0.01     # turns
MOVE_TIMEOUT = 5.0            # seconds


def keyboard_reader(key_queue, stop_event):
    """Runs on a background thread, continuously pushing raw keypresses onto
    key_queue. Kept separate from the move-polling loop so a SPACE e-stop can
    be seen immediately even while a launch/return move is in progress."""
    while not stop_event.is_set():
        ch = sys.stdin.read(1)
        if ch:
            key_queue.put(ch)


class WatchdogFeeder:
    """Feeds the axis watchdog on a background thread so blocking input()
    calls in the main thread don't trip the axis's watchdog timeout while
    the motor is armed and holding position."""

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


def move_to_position(axis, position, description, key_queue):
    """Returns 'reached', 'timeout', or 'estop'."""
    print(f"  Moving to {description} ({position:.4f} turns)...")
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
    print("LAUNCH ARM CONTROL")
    print("=" * 55)
    print(f"Connecting to LAUNCH axis only (S/N {LAUNCH_SERIAL})...")
    odrv = odrive.find_any(serial_number=LAUNCH_SERIAL, timeout=10)
    print(f"Connected. serial={odrv.serial_number:012X} "
          f"hw={odrv.hw_version_major}.{odrv.hw_version_minor} "
          f"fw={odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision}")

    axis = odrv.axis0

    # Make sure the arm is freewheeling so it can be positioned by hand.
    if axis.current_state != AxisState.IDLE:
        print("Axis was not idle - forcing IDLE so the arm can be moved by hand.")
        axis.requested_state = AxisState.IDLE
        time.sleep(0.3)

    odrv.clear_errors()

    print("\nStep 1: Manually move the launch arm to your HARD STOP (fully rested against it).")
    input("         Press ENTER when it's against the stop...")

    # pos_vel_mapper.set_abs_pos(0) only zeros pos_abs - it does NOT affect
    # pos_rel or pos_estimate, which is what the position controller actually
    # uses for feedback (confirmed by live testing on this rig). So instead
    # of trying to re-zero the encoder, just record the current position in
    # software and command everything as offsets from it.
    hard_stop_pos = axis.pos_estimate
    # The real start is a few degrees off the stop, in the same direction as
    # the launch swing, so the return move settles just short of the stop
    # instead of driving into it every cycle.
    start_pos = hard_stop_pos + HARD_STOP_BUFFER_TURNS
    launch_pos = start_pos + LAUNCH_OFFSET_TURNS
    print(f"Hard stop: {hard_stop_pos:.4f} turns ({hard_stop_pos * 360:.1f} deg)")
    print(f"Start (stop + {HARD_STOP_BUFFER_DEG:.0f} deg): {start_pos:.4f} turns ({start_pos * 360:.1f} deg)")
    print(f"Launch (start + 90 deg): {launch_pos:.4f} turns ({launch_pos * 360:.1f} deg)")
    odrv.clear_errors()

    axis.controller.config.control_mode = ControlMode.POSITION_CONTROL
    axis.controller.config.input_mode = InputMode.TRAP_TRAJ
    axis.controller.config.pos_gain = POS_GAIN
    axis.controller.config.vel_limit = CONTROLLER_VEL_LIMIT
    set_trap_traj_limits(axis, RETURN_VEL_LIMIT, TRAP_ACCEL_LIMIT, TRAP_DECEL_LIMIT)
    # Arm holding exactly where the arm physically is (the hard stop) - zero
    # position error on engage. A pre-arm input_pos write doesn't reliably
    # turn into a move once armed, so the move to start_pos is issued
    # explicitly below, after arming, using the same move_to_position() path
    # already proven to work for jogs and launches.
    axis.controller.input_pos = hard_stop_pos

    print("Engaging closed-loop control, holding at the hard stop...")
    axis.requested_state = AxisState.CLOSED_LOOP_CONTROL
    time.sleep(0.3)

    if axis.current_state != AxisState.CLOSED_LOOP_CONTROL:
        print(f"ERROR: motor did not reach closed loop control. "
              f"current_state={axis.current_state} active_errors={axis.active_errors} "
              f"disarm_reason={axis.disarm_reason}")
        axis.requested_state = AxisState.IDLE
        return

    print(f"Motor engaged (pos_estimate={axis.pos_estimate:.4f} turns).")

    feeder = WatchdogFeeder(axis, WATCHDOG_FEED_INTERVAL)
    feeder.start()

    key_queue = queue.Queue()
    stop_reader = threading.Event()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    tty.setraw(fd)
    reader_thread = threading.Thread(target=keyboard_reader, args=(key_queue, stop_reader), daemon=True)
    reader_thread.start()

    try:
        print("Moving off the hard stop to the start position...")
        result = move_to_position(axis, start_pos, "start position", key_queue)
        if result == 'estop':
            return

        at_start = True
        print("\n" + "-" * 55)
        print("READY")
        print("-" * 55)
        print("  SPACE - launch / return (toggles)")
        print("  ESC   - E-STOP (disables motor immediately, even mid-move)")
        print("  Q     - quit and disable motor")
        print("-" * 55 + "\n")

        while True:
            try:
                key = key_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if key == '\x1b':
                axis.requested_state = AxisState.IDLE
                print("\n!!! E-STOP !!! Motor disabled immediately.")
                break

            elif key == ' ':
                if at_start:
                    print("\nLAUNCH")
                    set_trap_traj_limits(axis, LAUNCH_VEL_LIMIT, TRAP_ACCEL_LIMIT, TRAP_DECEL_LIMIT)
                    result = move_to_position(axis, start_pos + LAUNCH_OFFSET_TURNS, "launch position", key_queue)
                else:
                    print("\nReturning to start")
                    set_trap_traj_limits(axis, RETURN_VEL_LIMIT, TRAP_ACCEL_LIMIT, TRAP_DECEL_LIMIT)
                    result = move_to_position(axis, start_pos, "start position", key_queue)

                if result == 'estop':
                    break
                at_start = not at_start

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
        feeder.stop()
        print("Disabling motor...")
        axis.requested_state = AxisState.IDLE
        time.sleep(0.3)
        print("Motor disabled. Safe to power off.\n")


if __name__ == "__main__":
    main()
