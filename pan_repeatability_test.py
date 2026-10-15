"""
PAN axis repeatability test - lateral-scatter diagnosis, step 3 (2026-09-21
debugging session): drives PAN away and back to the SAME locked position
repeatedly, with no firing at all, and logs the actual raw encoder value
reached each time.

Purpose: isolate whether PAN's own positioning is repeatable, decoupled from
everything downstream (launch mechanism, ball seating/release). Note the
existing fixed-swing/fixed-PAN scatter test
never re-commanded PAN between shots, so it couldn't have been measuring this
in the first place - this fills that gap and also covers the case that
actually matters for live play, where PAN re-aims before every shot.

Never touches the LAUNCH axis.
"""
import argparse
import math
import statistics
import sys
import time

from odrive.enums import AxisState

from pong_common import connect_pan, lock_pan, move_pan_to, WatchdogFeeder

DEFAULT_TRIALS = 20
DEFAULT_AWAY_DEG = 20.0   # excursion before returning - representative of a
                          # typical vision-aiming correction, not a fixed constant
DEFAULT_RANGE_IN = 60.0   # for converting angular spread to equivalent lateral inches
SETTLE_TIME = 0.3         # seconds to let the axis fully settle before reading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--away-deg", type=float, default=DEFAULT_AWAY_DEG,
                         help="how far PAN moves away before returning, each trial")
    parser.add_argument("--range-in", type=float, default=DEFAULT_RANGE_IN,
                         help="reference range for the equivalent-lateral-inches figure")
    args = parser.parse_args()

    away_turns = args.away_deg / 360.0

    print("=" * 55)
    print("PAN REPEATABILITY TEST (no firing, PAN axis only)")
    print("=" * 55)

    odrv, axis = connect_pan()
    lock_target = lock_pan(odrv, axis)
    if lock_target is None:
        print("ERROR: could not lock PAN, aborting.")
        sys.exit(1)

    feeder = WatchdogFeeder(axis)
    feeder.start()

    raws = []
    try:
        for i in range(args.trials):
            away_result = move_pan_to(
                axis, lock_target + away_turns,
                description=f"trial {i + 1}/{args.trials} away")
            if away_result is not True:
                print(f"  trial {i + 1}: away move failed ({away_result}), skipping")
                continue

            return_result = move_pan_to(
                axis, lock_target,
                description=f"trial {i + 1}/{args.trials} return")
            if return_result is not True:
                print(f"  trial {i + 1}: return move failed ({return_result}), skipping")
                continue

            time.sleep(SETTLE_TIME)
            raw = odrv.onboard_encoder0.raw
            pos = axis.pos_estimate
            raws.append(raw)
            print(f"  trial {i + 1:2d}: raw={raw:.5f}  pos_estimate={pos:.5f}  "
                  f"(target={lock_target:.5f})")

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        feeder.stop()
        axis.requested_state = AxisState.IDLE
        print("\nPAN disabled.")

    if len(raws) < 2:
        print("Not enough successful trials to compute stats.")
        return

    spread_turns = max(raws) - min(raws)
    stdev_turns = statistics.pstdev(raws)
    spread_deg = spread_turns * 360.0
    stdev_deg = stdev_turns * 360.0
    lateral_spread_in = args.range_in * math.tan(math.radians(spread_deg))
    lateral_stdev_in = args.range_in * math.tan(math.radians(stdev_deg))

    print("\n" + "-" * 55)
    print(f"{len(raws)} successful trials")
    print(f"  raw encoder spread (max-min): {spread_turns:.5f} turns = {spread_deg:.3f} deg")
    print(f"  raw encoder stdev:            {stdev_turns:.5f} turns = {stdev_deg:.3f} deg")
    print(f"  equivalent lateral spread at {args.range_in:.0f}in: {lateral_spread_in:.2f} in")
    print(f"  equivalent lateral stdev  at {args.range_in:.0f}in: {lateral_stdev_in:.2f} in")
    print("-" * 55)
    print("Compare against the ~3.4in center-to-center fire-test lateral scatter.")
    print("If this spread is much smaller, PAN positioning is not a meaningful")
    print("lateral-scatter contributor on its own - look downstream (ball")
    print("seating/release, launch mechanism) instead. It can still stack with")
    print("a downstream source during live play, where PAN re-aims every shot.")


if __name__ == "__main__":
    main()
