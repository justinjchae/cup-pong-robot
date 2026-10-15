"""
Flywheel launch-axis jog/characterization tool.

The catapult arm on the LAUNCH axis has been replaced with a 91mm flywheel.
Physical structure (motor, ODrive, mount) is otherwise unchanged. With a
fixed angled hood at the back, ball speed - and therefore range - is now set
purely by flywheel RPM, so this script exists to find that relationship:
spin the flywheel to a target velocity, feed a ball in by hand, see how far
it goes, adjust, repeat.

Runs the axis in VELOCITY_CONTROL / VEL_RAMP input mode: each jog step (or
the initial spin-up) ramps smoothly to the new target at a fixed
turns/s^2 rate instead of snapping the motor straight to a new speed and
spiking current.

PAN is also auto-locked to its saved position at startup (same
connect_pan()/lock_pan() used by the other scripts) and held there for the
whole session, so shots land on the same bearing every time instead of
wherever PAN happened to be left - it's just locked, never jogged, from
this script.

There's no hard-stop/position concept here (unlike the old catapult arm) -
the flywheel just spins in one direction, so this is a much simpler
velocity dial, not a position-anchored move.

Controls (raw terminal input, single keypress, no ENTER needed):
  +/-       - jog target velocity +/- 1 turn/s (fine)
  ]/[       - jog target velocity +/- 5 turns/s (coarse)
  SPACE     - toggle spin: engage closed-loop control and ramp up to the
              current target velocity, or ramp back down to 0 and idle
              the motor (so it's not left energized between test shots)
  r         - reset target velocity to 0 (ramps down if currently spinning)
  ESC       - E-STOP: disables the motor immediately, even mid-ramp
  q         - quit gracefully (ramps down to 0, then disables)

The status line prints target vs. measured velocity (turns/s, RPM, and
flywheel surface speed in m/s) plus motor current continuously, so you can
correlate whatever you see the ball do with a specific number.
"""
import math
import queue
import sys
import threading
import time
import termios
import tty

from odrive.enums import AxisState, ControlMode, InputMode

from pong_common import (
    connect_axis, connect_pan, lock_pan, LAUNCH_SERIAL, WatchdogFeeder,
    LAUNCH_VEL_GAIN, LAUNCH_VEL_INTEGRATOR_GAIN,
)

# --- Flywheel geometry -----------------------------------------------------
FLYWHEEL_DIAMETER_MM = 91.0
FLYWHEEL_CIRCUMFERENCE_M = math.pi * FLYWHEEL_DIAMETER_MM / 1000.0

# --- Jog tuning --------------------------------------------------------------
JOG_STEP_FINE = 1.0      # turns/s per +/- press
JOG_STEP_COARSE = 5.0    # turns/s per [/] press

# Safety ceiling on how fast this script will ever let you jog to - well
# under CONTROLLER_VEL_LIMIT below, which is the ODrive's own disarm
# threshold. Raise via --max-vel once you know the flywheel handles it fine;
# starting conservative since no one's spun this flywheel before.
DEFAULT_MAX_JOG_VEL = 60.0     # turns/s (3600 RPM)
CONTROLLER_VEL_LIMIT_MULT = 1.5  # same headroom convention as LAUNCH_CONTROLLER_VEL_LIMIT in pong_common

# How fast the ODrive's own VEL_RAMP mode ramps the setpoint toward
# input_vel, for both spin-up and spin-down. Conservative starting point -
# raise if spin-up feels too slow, lower if current spikes on a big jog step.
DEFAULT_RAMP_RATE = 20.0       # turns/s^2

# vel_gain/vel_integrator_gain are the same inner velocity-loop gains that
# were tuned for the catapult arm (pos_gain, the OUTER loop, doesn't apply
# in VELOCITY_CONTROL mode at all). Reused as a starting point since it's
# the same motor/axis - but the arm's inertia and a flywheel's inertia
# aren't the same, so revisit if speed sags noticeably under ball contact
# or if it hunts/oscillates at a held speed (raise vel_gain /
# lower vel_integrator_gain for hunting, raise pos_gain for overshoot-then-creep).
VEL_GAIN = LAUNCH_VEL_GAIN
VEL_INTEGRATOR_GAIN = LAUNCH_VEL_INTEGRATOR_GAIN

WATCHDOG_FEED_INTERVAL = 0.2
VEL_SETTLE_TOLERANCE = 1.0     # turns/s, "close enough" to 0 before idling after a ramp-down
STATUS_INTERVAL = 0.1          # seconds between status-line refreshes


def keyboard_reader(key_queue, stop_event):
    """Background thread continuously pushing raw keypresses onto key_queue -
    kept separate from the main loop so ESC is seen immediately even mid-ramp
    (see pong_common's keyboard-reader-thread-leak gotcha: one persistent
    reader for the script's whole lifetime, never restarted per action)."""
    while not stop_event.is_set():
        ch = sys.stdin.read(1)
        if ch:
            key_queue.put(ch)


def status_line(target_vel, axis, spinning):
    measured = axis.vel_estimate
    iq = axis.motor.foc.Iq_measured
    state = "SPINNING" if spinning else "   IDLE "
    return (
        f"\r{state}  target={target_vel:5.1f} turns/s ({target_vel * 60:6.0f} RPM, "
        f"{target_vel * FLYWHEEL_CIRCUMFERENCE_M:5.2f} m/s)   "
        f"measured={measured:5.1f} turns/s ({measured * 60:6.0f} RPM, "
        f"{measured * FLYWHEEL_CIRCUMFERENCE_M:5.2f} m/s)   Iq={iq:5.2f}A   "
    )


def ramp_down_and_idle(axis, key_queue, ramp_rate, pan_axis=None):
    """Ramps input_vel to 0 and waits (polling for ESC) until vel_estimate
    settles near 0, then idles the motor (and PAN, if an ESC hits mid-ramp).
    Returns 'done' or 'estop'."""
    axis.controller.input_vel = 0.0
    timeout = max(3.0, abs(axis.vel_estimate) / ramp_rate + 2.0)
    start = time.time()
    while abs(axis.vel_estimate) > VEL_SETTLE_TOLERANCE:
        try:
            key = key_queue.get_nowait()
        except queue.Empty:
            key = None
        if key == '\x1b':
            axis.requested_state = AxisState.IDLE
            if pan_axis is not None:
                pan_axis.requested_state = AxisState.IDLE
            return 'estop'
        if time.time() - start > timeout:
            break
        time.sleep(0.02)
    axis.requested_state = AxisState.IDLE
    return 'done'


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-vel", type=float, default=DEFAULT_MAX_JOG_VEL,
                         help=f"safety ceiling on jog target velocity, turns/s (default: {DEFAULT_MAX_JOG_VEL})")
    parser.add_argument("--ramp-rate", type=float, default=DEFAULT_RAMP_RATE,
                         help=f"VEL_RAMP rate, turns/s^2 (default: {DEFAULT_RAMP_RATE})")
    parser.add_argument("--current-limit", type=float, default=None,
                         help="override config.motor.current_soft_max, amps (default: leave "
                              "whatever's already configured on the ODrive unchanged). Raise "
                              "this if measured velocity won't track target under load - but "
                              "unlike the old catapult's <1s bursts, a flywheel draws current "
                              "continuously, so watch/feel the motor for heat as you push this up.")
    args = parser.parse_args()

    max_jog_vel = args.max_vel
    ramp_rate = args.ramp_rate
    controller_vel_limit = max_jog_vel * CONTROLLER_VEL_LIMIT_MULT

    print("=" * 55)
    print("FLYWHEEL VELOCITY JOG")
    print("=" * 55)
    print(f"Flywheel diameter: {FLYWHEEL_DIAMETER_MM:.0f}mm "
          f"(circumference {FLYWHEEL_CIRCUMFERENCE_M:.4f}m)")
    print(f"Jog range: 0 to {max_jog_vel:.1f} turns/s, ramp rate {ramp_rate:.1f} turns/s^2")

    pan_odrv, pan_axis = connect_pan()
    pan_lock_target = lock_pan(pan_odrv, pan_axis)
    if pan_lock_target is None:
        print("ERROR: could not lock PAN, aborting.")
        sys.exit(1)

    # Feed PAN's watchdog immediately, before connecting to/configuring
    # LAUNCH - otherwise PAN sits armed with nothing feeding it while that
    # runs, and trips its watchdog fault before the flywheel is even ready
    # (same ordering fire_telemetry_logger.py uses).
    pan_feeder = WatchdogFeeder(pan_axis, WATCHDOG_FEED_INTERVAL)
    pan_feeder.start()

    odrv, axis = connect_axis(LAUNCH_SERIAL, "LAUNCH")

    current_hard_max = axis.config.motor.current_hard_max
    if args.current_limit is not None:
        if args.current_limit >= current_hard_max:
            print(f"ERROR: --current-limit {args.current_limit:.1f}A must stay below "
                  f"current_hard_max ({current_hard_max:.1f}A), or the axis will fault. Aborting.")
            pan_feeder.stop()
            pan_axis.requested_state = AxisState.IDLE
            sys.exit(1)
        axis.config.motor.current_soft_max = args.current_limit
    print(f"LAUNCH current_soft_max={axis.config.motor.current_soft_max:.1f}A "
          f"(current_hard_max={current_hard_max:.1f}A) - if measured velocity won't track "
          f"target under load, this is the likely ceiling; re-run with --current-limit to raise it.")

    axis.controller.config.control_mode = ControlMode.VELOCITY_CONTROL
    axis.controller.config.input_mode = InputMode.VEL_RAMP
    axis.controller.config.vel_gain = VEL_GAIN
    axis.controller.config.vel_integrator_gain = VEL_INTEGRATOR_GAIN
    axis.controller.config.vel_limit = controller_vel_limit
    axis.controller.config.vel_ramp_rate = ramp_rate
    axis.controller.input_vel = 0.0

    feeder = WatchdogFeeder(axis, WATCHDOG_FEED_INTERVAL)
    feeder.start()

    key_queue = queue.Queue()
    stop_reader = threading.Event()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    tty.setraw(fd)
    reader_thread = threading.Thread(target=keyboard_reader, args=(key_queue, stop_reader), daemon=True)
    reader_thread.start()

    target_vel = 0.0
    spinning = False

    try:
        print("\n" + "-" * 55)
        print("READY (motor idle)")
        print("-" * 55)
        print("  +/-  jog +/-1 turn/s     ]/[  jog +/-5 turns/s")
        print("  SPACE  toggle spin       r  reset target to 0")
        print("  ESC  E-STOP              q  quit")
        print("-" * 55 + "\n")

        last_status = 0.0
        while True:
            try:
                key = key_queue.get(timeout=0.02)
            except queue.Empty:
                key = None

            if key == '\x1b':
                axis.requested_state = AxisState.IDLE
                pan_axis.requested_state = AxisState.IDLE
                print("\n\n!!! E-STOP !!! Motor disabled immediately.")
                break

            elif key in ('+', '='):
                target_vel = min(max_jog_vel, target_vel + JOG_STEP_FINE)
                if spinning:
                    axis.controller.input_vel = target_vel

            elif key == '-':
                target_vel = max(0.0, target_vel - JOG_STEP_FINE)
                if spinning:
                    axis.controller.input_vel = target_vel

            elif key == ']':
                target_vel = min(max_jog_vel, target_vel + JOG_STEP_COARSE)
                if spinning:
                    axis.controller.input_vel = target_vel

            elif key == '[':
                target_vel = max(0.0, target_vel - JOG_STEP_COARSE)
                if spinning:
                    axis.controller.input_vel = target_vel

            elif key == 'r':
                target_vel = 0.0
                if spinning:
                    axis.controller.input_vel = target_vel

            elif key == ' ':
                if not spinning:
                    print(f"\nSpinning up to {target_vel:.1f} turns/s...")
                    axis.controller.input_vel = 0.0
                    axis.requested_state = AxisState.CLOSED_LOOP_CONTROL
                    time.sleep(0.3)
                    if axis.current_state != AxisState.CLOSED_LOOP_CONTROL:
                        print(f"ERROR: LAUNCH did not reach closed loop control. "
                              f"active_errors={axis.active_errors} disarm_reason={axis.disarm_reason}")
                        axis.requested_state = AxisState.IDLE
                        pan_axis.requested_state = AxisState.IDLE
                        break
                    axis.controller.input_vel = target_vel
                    spinning = True
                else:
                    print("\nRamping down to stop...")
                    result = ramp_down_and_idle(axis, key_queue, ramp_rate, pan_axis)
                    spinning = False
                    if result == 'estop':
                        print("\n!!! E-STOP !!! Motor disabled immediately.")
                        break
                    print("Stopped (motor idle).")

            elif key is not None and key.lower() == 'q':
                print("\nShutting down...")
                if spinning:
                    print("Ramping down to stop first...")
                    ramp_down_and_idle(axis, key_queue, ramp_rate, pan_axis)
                break

            elif key == '\x03':  # Ctrl+C
                print("\nInterrupted!")
                break

            now = time.time()
            if now - last_status > STATUS_INTERVAL:
                sys.stdout.write(status_line(target_vel, axis, spinning))
                sys.stdout.flush()
                last_status = now

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        stop_reader.set()
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        feeder.stop()
        pan_feeder.stop()
        print("\nDisabling motor...")
        axis.requested_state = AxisState.IDLE
        pan_axis.requested_state = AxisState.IDLE
        time.sleep(0.3)
        print("LAUNCH and PAN disabled. Safe to power off.\n")


if __name__ == "__main__":
    main()
