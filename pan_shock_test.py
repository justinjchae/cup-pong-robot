"""
PAN shock/recoil test - lateral-scatter diagnosis follow-up (2026-09-21
session): locks PAN, fires the LAUNCH axis repeatedly (swing -> hold ->
return, same as a real shot), and checks whether PAN's own raw encoder
reading has shifted after each firing. Tests the hypothesis that firing
recoil/shock is knocking PAN off its locked position.

IMPORTANT LIMITATION: onboard_encoder0.raw is read on PAN's own motor shaft.
If recoil causes the PAN *motor* to slip/desync from its commanded position,
this test will catch it - PAN's closed-loop controller can't correct a raw
encoder value it doesn't know is wrong. But if recoil instead flexes the
frame/mount/camera bracket *without* the PAN motor coupling itself slipping,
this raw-encoder check will read clean (PAN motor still thinks it's exactly
where it should be) even though the real boresight moved - that failure mode
is invisible to this test and would need a vision-based check (e.g. does the
camera's view of a fixed reference point shift after firing) to catch.

Connects to and arms BOTH axes. Every trial fires a real shot - treat this
like normal operation (clear the table in front of the arm) and check the
run print for the e-stop key (SPACE, same as the other control scripts) if
tty raw mode is set up; otherwise Ctrl+C aborts between trials.
"""
import argparse
import math
import statistics
import sys
import time

from odrive.enums import AxisState

from pong_common import (
    connect_pan, lock_pan,
    connect_launch, prepare_launch, fire_shot,
    WatchdogFeeder,
)

DEFAULT_TRIALS = 10
DEFAULT_SWING_DEG = 90.0
DEFAULT_RANGE_IN = 60.0
SETTLE_TIME = 0.3  # seconds after the return move before reading PAN's encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--swing-deg", type=float, default=DEFAULT_SWING_DEG)
    parser.add_argument("--range-in", type=float, default=DEFAULT_RANGE_IN,
                         help="reference range for the equivalent-lateral-inches figure")
    args = parser.parse_args()

    print("=" * 55)
    print("PAN SHOCK/RECOIL TEST (fires LAUNCH each trial)")
    print("=" * 55)
    print("Make sure the launch path is clear before continuing.")
    input("Press ENTER to begin...")

    pan_odrv, pan_axis = connect_pan()
    pan_lock_target = lock_pan(pan_odrv, pan_axis)
    if pan_lock_target is None:
        print("ERROR: could not lock PAN, aborting.")
        sys.exit(1)

    # Feed PAN's watchdog immediately, before connecting to/engaging LAUNCH -
    # otherwise PAN sits armed with nothing feeding it while connect_launch()/
    # prepare_launch() run, and trips its watchdog fault before we ever fire.
    pan_feeder = WatchdogFeeder(pan_axis)
    pan_feeder.start()

    launch_odrv, launch_axis = connect_launch()
    launch_start_pos = prepare_launch(launch_odrv, launch_axis)
    if launch_start_pos is None:
        print("ERROR: could not prepare LAUNCH, aborting.")
        pan_feeder.stop()
        pan_axis.requested_state = AxisState.IDLE
        sys.exit(1)

    launch_feeder = WatchdogFeeder(launch_axis)
    launch_feeder.start()

    deltas = []
    try:
        baseline_raw = pan_odrv.onboard_encoder0.raw
        print(f"\nPAN baseline raw encoder: {baseline_raw:.5f} turns\n")

        for i in range(args.trials):
            before_raw = pan_odrv.onboard_encoder0.raw

            result = fire_shot(launch_axis, launch_start_pos, args.swing_deg)
            if result != 'done':
                print(f"  trial {i + 1}: fire_shot returned {result!r}, stopping.")
                break

            time.sleep(SETTLE_TIME)
            after_raw = pan_odrv.onboard_encoder0.raw
            delta = after_raw - before_raw
            deltas.append(delta)
            print(f"  trial {i + 1:2d}: PAN before={before_raw:.5f}  after={after_raw:.5f}  "
                  f"delta={delta:+.5f} turns ({delta * 360:+.3f} deg)")

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        pan_feeder.stop()
        launch_feeder.stop()
        pan_axis.requested_state = AxisState.IDLE
        launch_axis.requested_state = AxisState.IDLE
        print("\nBoth axes disabled.")

    if len(deltas) < 2:
        print("Not enough successful trials to compute stats.")
        return

    abs_deltas = [abs(d) for d in deltas]
    net_drift = sum(deltas)  # signed sum - reveals a one-directional walk vs random jitter
    max_abs_turns = max(abs_deltas)
    stdev_turns = statistics.pstdev(deltas)
    max_abs_deg = max_abs_turns * 360.0
    stdev_deg = stdev_turns * 360.0
    lateral_max_in = args.range_in * math.tan(math.radians(max_abs_deg))
    lateral_stdev_in = args.range_in * math.tan(math.radians(stdev_deg))

    print("\n" + "-" * 55)
    print(f"{len(deltas)} successful trials")
    print(f"  largest single per-shot shift: {max_abs_turns:.5f} turns = {max_abs_deg:.3f} deg")
    print(f"  per-shot shift stdev:          {stdev_turns:.5f} turns = {stdev_deg:.3f} deg")
    print(f"  net drift over all trials:     {net_drift:.5f} turns = {net_drift * 360:.3f} deg")
    print(f"  equivalent lateral (largest shift) at {args.range_in:.0f}in: {lateral_max_in:.2f} in")
    print(f"  equivalent lateral (stdev)         at {args.range_in:.0f}in: {lateral_stdev_in:.2f} in")
    print("-" * 55)
    print("Compare against the ~3.4in center-to-center fire-test lateral scatter.")
    print("A large net drift (one direction, growing) suggests the PAN coupling is")
    print("creeping/slipping under repeated recoil. Large but non-drifting per-shot")
    print("shifts suggest the motor briefly loses sync each shot then re-settles.")
    print("A clean (near-zero) result here does NOT rule out recoil - see the")
    print("module docstring: it only catches motor-side slip, not frame/mount flex")
    print("downstream of the encoder. If clean, a camera-based before/after check")
    print("of a fixed reference point is the next thing to try.")


if __name__ == "__main__":
    main()
