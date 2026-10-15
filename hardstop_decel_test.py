"""
One-off test: does a real mechanical hard stop give more consistent ball
release than the software trapezoidal decel profile? Commands LAUNCH toward
a target deliberately PAST the newly-installed stop at full launch velocity/
accel (LAUNCH_VEL_LIMIT/LAUNCH_TRAP_ACCEL_LIMIT, same as a real shot) - the
position controller is still actively driving forward when the arm
physically can't move any further, which shows up as a current spike
(Iq_measured). The moment that spike crosses STALL_CURRENT_THRESHOLD_A, this
treats it as "hit the stop" and immediately reverses to a controlled return
move, rather than holding torque into the obstruction waiting for a position
error that will never resolve.

LAUNCH_DECEL_STOP_RAW below is a SEPARATE constant from pong_common's
LAUNCH_HARD_STOP_RAW - that one is the launcher's original near/rest-position
stop (still used by prepare_launch() to compute start_pos) and stays
untouched. This is a second, new stop further into the swing, just for this
test.

STALL_CURRENT_THRESHOLD_A is a first guess - the live status line prints
current every tick, so watch it on your first press and tune the constant if
it trips too early/late. An existing full-swing trace at decel_limit=50
(launch_telemetry_decel50_20260921_225126.csv) peaked at ~9.2A during
ordinary commanded accel/decel, so 15A leaves real headroom above normal
operation before calling it a stall.

Only ever talks to the LAUNCH axis by serial number (via pong_common's
connect_launch/prepare_launch) - PAN is never connected to or touched.

Run with: odrive_env/bin/python hardstop_decel_test.py
"""
import queue
import sys
import threading
import time
import tty
import termios

from odrive.enums import AxisState

from pong_common import (
    LAUNCH_TRAP_ACCEL_LIMIT, LAUNCH_TRAP_DECEL_LIMIT, LAUNCH_VEL_LIMIT, POSITION_TOLERANCE,
    RETURN_VEL_LIMIT, WATCHDOG_FEED_INTERVAL,
    WatchdogFeeder, connect_launch, max_safe_launch_pos, move_pan_to, prepare_launch,
)

# --- New physical stop, captured via launch_read_position.py -------------
LAUNCH_DECEL_STOP_RAW = 0.4232          # onboard_encoder0.raw at the new stop
                                         # (pos_estimate was 0.2156 turns / 77.61deg
                                         # at capture time, 2026-09-22)
LAUNCH_DECEL_STOP_MAX_CORRECTION_TURNS = 0.5   # sanity bound on the raw-anchored
                                                # correction - wide since this stop
                                                # could legitimately be far from
                                                # wherever LAUNCH happens to rest

# How far past the new stop to aim the commanded target, so the controller is
# still actively driving forward (not coincidentally arriving under its own
# decel) at the moment of contact. The actual commanded target is clamped
# against max_safe_launch_pos() regardless, so a missing/slipped stop can't
# send the arm past the already-validated far (camera-side) stop margin.
OVERSHOOT_BUFFER_DEG = 15.0

STALL_CURRENT_THRESHOLD_A = 15.0   # tune from the live Iq printout - see docstring.
                                    # First real run (2026-09-22) showed peak |Iq| only
                                    # reaches ~9.8A when actually stalled against the
                                    # stop - too close to the ~9.2A this hardware already
                                    # shows during ordinary accel to give current enough
                                    # separation margin on its own. Kept as a secondary/
                                    # backup trigger; velocity collapse below is the
                                    # primary one now.
STALL_PEAK_VEL_NOISE_FLOOR = 1.0   # peak vel this strike must clear before the collapse
                                    # check even engages - just filters sensor noise/settling
                                    # creep right after arming, NOT a "reached full speed"
                                    # requirement. A fixed "was moving" floor (tried 5.0
                                    # turns/s first) turned out wrong: this stop is only
                                    # 67deg into the swing, not enough distance for the
                                    # trapezoidal profile to get anywhere near
                                    # LAUNCH_VEL_LIMIT (30) before impact - the first real
                                    # run peaked at only 4.65 turns/s and clearly stalled
                                    # (collapsed to -0.02), but never tripped that gate.
STALL_VEL_FRACTION = 0.2           # "stopped" = current |vel| has dropped to this fraction
                                    # of whatever peak was actually reached - self-calibrating
                                    # to the real achievable speed instead of a fixed number
STALL_VEL_ABS_FLOOR = 0.5          # ...but never looser than this many turns/s, so a small
                                    # peak_vel doesn't make the fractional check too twitchy
STALL_TIMEOUT_S = 2.0              # safety net if neither signal ever trips - don't hold
                                    # torque into an obstruction forever


def keyboard_reader(key_queue, stop_event):
    while not stop_event.is_set():
        ch = sys.stdin.read(1)
        if ch:
            key_queue.put(ch)


def compute_decel_stop_pos(odrv, axis):
    """Same raw-anchored pattern as pong_common's prepare_launch/
    max_safe_launch_pos - pos_estimate isn't power-cycle-stable,
    onboard_encoder0.raw is."""
    now = axis.pos_estimate
    raw_diff = LAUNCH_DECEL_STOP_RAW - odrv.onboard_encoder0.raw
    correction = raw_diff - round(raw_diff)
    if abs(correction) > LAUNCH_DECEL_STOP_MAX_CORRECTION_TURNS:
        print(f"ERROR: computed decel-stop correction is {abs(correction) * 360:.1f} deg - "
              f"farther than the {LAUNCH_DECEL_STOP_MAX_CORRECTION_TURNS * 360:.0f} deg "
              "sanity bound. Check LAUNCH_DECEL_STOP_RAW or the arm's position by hand.")
        return None
    return now + correction


def fire_into_hardstop(axis, start_pos, target_pos, key_queue):
    """Drives toward target_pos at full launch velocity, watching vel/Iq and
    the key_queue every tick. Returns one of:
      ('hit', peak_iq, impact_t, impact_pos)        - stop contact detected
      ('reached_target', peak_iq, None, None)        - got to target_pos with
                                                         no stall detected (stop
                                                         missing or farther out)
      ('estop', peak_iq, None, None)
      ('timeout', peak_iq, None, None)                - neither signal tripped in
                                                          STALL_TIMEOUT_S; still attempts a
                                                          controlled return to start_pos,
                                                          only falls back to IDLE if that
                                                          return itself doesn't complete
    Always ends with LAUNCH back at start_pos, or IDLE only if estop or a return move
    itself fails.

    Primary stall signal is velocity collapse, self-calibrated to whatever
    peak velocity this strike actually reached (STALL_PEAK_VEL_NOISE_FLOOR/
    STALL_VEL_FRACTION/STALL_VEL_ABS_FLOOR) rather than a fixed speed -
    current alone didn't separate cleanly from normal-operation current on
    this hardware (see the constants' comments), so it's kept only as a
    secondary/backup trigger."""
    print(f"\nSTRIKE: commanding {target_pos:.4f} turns ({target_pos * 360:.1f} deg) at "
          f"full launch velocity, watching for a velocity collapse (or |Iq| > "
          f"{STALL_CURRENT_THRESHOLD_A:.1f}A as backup)...")
    axis.trap_traj.config.vel_limit = LAUNCH_VEL_LIMIT
    axis.trap_traj.config.accel_limit = LAUNCH_TRAP_ACCEL_LIMIT
    axis.trap_traj.config.decel_limit = LAUNCH_TRAP_DECEL_LIMIT
    axis.controller.input_pos = target_pos

    start_time = time.time()
    peak_iq = 0.0
    peak_vel = 0.0
    last_print = 0.0

    while True:
        try:
            key = key_queue.get_nowait()
        except queue.Empty:
            key = None
        if key == '\x1b':
            axis.requested_state = AxisState.IDLE
            print("\n!!! E-STOP !!! Motor disabled immediately.")
            return 'estop', peak_iq, None, None

        iq = axis.motor.foc.Iq_measured
        vel = axis.vel_estimate
        peak_iq = max(peak_iq, abs(iq))
        peak_vel = max(peak_vel, abs(vel))
        now = time.time()
        if now - last_print > 0.05:
            print(f"\r  t={now - start_time:5.3f}s  pos={axis.pos_estimate:.4f}  "
                  f"vel={vel:+6.2f} (peak {peak_vel:5.2f})  Iq={iq:+6.2f}A (peak {peak_iq:5.2f})   ",
                  end="", flush=True)
            last_print = now

        was_moving = peak_vel > STALL_PEAK_VEL_NOISE_FLOOR
        stopped_now = abs(vel) < max(STALL_VEL_ABS_FLOOR, STALL_VEL_FRACTION * peak_vel)
        stalled = was_moving and stopped_now
        current_spike = abs(iq) > STALL_CURRENT_THRESHOLD_A

        if stalled or current_spike:
            impact_pos = axis.pos_estimate
            impact_t = now - start_time
            reason = "velocity collapse" if stalled else "current spike"
            print(f"\n  HIT ({reason}) at t={impact_t:.3f}s, pos={impact_pos:.4f} turns "
                  f"({impact_pos * 360:.1f} deg), vel={vel:+.2f} (peak {peak_vel:.2f}), "
                  f"Iq={iq:+.2f}A (peak {peak_iq:.2f}A) - returning to start.")
            axis.trap_traj.config.vel_limit = RETURN_VEL_LIMIT
            axis.trap_traj.config.accel_limit = LAUNCH_TRAP_ACCEL_LIMIT
            axis.trap_traj.config.decel_limit = LAUNCH_TRAP_DECEL_LIMIT
            result = move_pan_to(axis, start_pos, description="start position")
            if result != True:
                return 'timeout', peak_iq, impact_t, impact_pos
            return 'hit', peak_iq, impact_t, impact_pos

        if abs(axis.pos_estimate - target_pos) < POSITION_TOLERANCE:
            print("\n  Reached the commanded target with no stall detected - the stop is "
                  "farther out than expected, or missing. Returning to start.")
            move_pan_to(axis, start_pos, description="start position")
            return 'reached_target', peak_iq, None, None

        if now - start_time > STALL_TIMEOUT_S:
            print(f"\n  TIMEOUT ({STALL_TIMEOUT_S}s) waiting for stop contact or target - "
                  "neither signal tripped. Returning to start under control rather than "
                  "leaving the arm powered mid-swing.")
            axis.trap_traj.config.vel_limit = RETURN_VEL_LIMIT
            axis.trap_traj.config.accel_limit = LAUNCH_TRAP_ACCEL_LIMIT
            axis.trap_traj.config.decel_limit = LAUNCH_TRAP_DECEL_LIMIT
            result = move_pan_to(axis, start_pos, description="start position")
            if result != True:
                print("  Return move itself did not complete cleanly - disabling motor.")
                axis.requested_state = AxisState.IDLE
            return 'timeout', peak_iq, None, None

        time.sleep(0.005)


def main():
    print("=" * 60)
    print("HARDSTOP DECEL TEST - LAUNCH only, one-off experiment")
    print("=" * 60)
    print("Fires at full launch velocity into the new physical stop, detects")
    print("contact via a velocity collapse (current spike as backup), then returns")
    print("to start.\n")

    odrv, axis = connect_launch()
    start_pos = prepare_launch(odrv, axis)
    if start_pos is None:
        return

    decel_stop_pos = compute_decel_stop_pos(odrv, axis)
    if decel_stop_pos is None:
        axis.requested_state = AxisState.IDLE
        return

    safe_ceiling = max_safe_launch_pos(odrv, axis)
    target_pos = min(decel_stop_pos + OVERSHOOT_BUFFER_DEG / 360.0, safe_ceiling)
    if target_pos <= decel_stop_pos:
        print(f"ERROR: the far-stop safety ceiling ({safe_ceiling:.4f} turns) leaves no "
              f"room past the new stop ({decel_stop_pos:.4f} turns) for an overshoot "
              "target. Refusing to run - check LAUNCH_DECEL_STOP_RAW.")
        axis.requested_state = AxisState.IDLE
        return

    print(f"  Start position:      {start_pos:.4f} turns ({start_pos * 360:.1f} deg)")
    print(f"  New stop position:   {decel_stop_pos:.4f} turns ({decel_stop_pos * 360:.1f} deg) "
          f"- {(decel_stop_pos - start_pos) * 360:.1f} deg of swing")
    print(f"  Commanded target:    {target_pos:.4f} turns ({target_pos * 360:.1f} deg, "
          f"{OVERSHOOT_BUFFER_DEG:.0f} deg past the stop)")
    print(f"  Far-stop safety cap: {safe_ceiling:.4f} turns\n")

    feeder = WatchdogFeeder(axis, WATCHDOG_FEED_INTERVAL)
    feeder.start()

    key_queue = queue.Queue()
    stop_reader = threading.Event()
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    tty.setraw(fd)
    reader_thread = threading.Thread(target=keyboard_reader, args=(key_queue, stop_reader), daemon=True)
    reader_thread.start()

    results = []

    try:
        print("-" * 55)
        print("READY")
        print("  SPACE - fire into the stop at full launch velocity")
        print("  ESC   - E-STOP")
        print("  Q     - quit")
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
                result, peak_iq, impact_t, impact_pos = fire_into_hardstop(
                    axis, start_pos, target_pos, key_queue)
                results.append((result, peak_iq, impact_t, impact_pos))
                if result == 'estop':
                    break
                # drain anything buffered during the strike (e.g. accidental
                # repeats) so it isn't misread as the next command
                while not key_queue.empty():
                    try:
                        key_queue.get_nowait()
                    except queue.Empty:
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
        feeder.stop()
        print("Disabling motor...")
        axis.requested_state = AxisState.IDLE
        time.sleep(0.3)
        print("Motor disabled. Safe to power off.\n")

        hits = [r for r in results if r[0] == 'hit']
        if hits:
            print(f"{len(hits)} hit(s) out of {len(results)} shot(s):")
            for i, (_, peak_iq, impact_t, impact_pos) in enumerate(hits, 1):
                print(f"  #{i}: t={impact_t:.3f}s  pos={impact_pos:.4f} turns "
                      f"({impact_pos * 360:.1f} deg)  peak Iq={peak_iq:.2f}A")
            if len(hits) > 1:
                positions = [h[3] for h in hits]
                spread_deg = (max(positions) - min(positions)) * 360
                print(f"  Impact-position spread across hits: {spread_deg:.2f} deg")


if __name__ == "__main__":
    main()
