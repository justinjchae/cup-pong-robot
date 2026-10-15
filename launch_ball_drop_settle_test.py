"""
LAUNCH start-position settle test (2026-09-21 session): tests the
hypothesis that dropping the ball onto the arm at the start position knocks
it off target, and that firing before it fully settles back could be
introducing shot-to-shot start-position variance - a plausible source of the
measured lateral/range scatter since it's operator-timing-dependent
(inherently intermittent, not a constant bias).

Holds LAUNCH at 'start' (via prepare_launch, same as real operation), then
for each trial: prompts you to drop the ball as you normally would, and
samples axis.pos_estimate at high rate for a few seconds afterward. Reports,
per trial: peak deviation from start_pos right after the drop, how long it
took to settle back within the same tolerance move_pan_to() uses everywhere
else in this project (POSITION_TOLERANCE), and the final position at the end
of the window - across trials, that final-position spread answers "does the
arm actually return to the exact same start position every time."

LIMITATION: this only sees the LAUNCH axis's own commanded position (motor/
encoder side). If the ball itself is still settling into its resting pocket
in the arm after the arm has already stopped correcting, this won't catch
that - it answers "is the arm back where it should be," not "has the ball
fully stopped moving inside the arm."

Never touches the PAN axis.
"""
import argparse
import statistics
import sys
import time

from odrive.enums import AxisState

from pong_common import connect_launch, prepare_launch, WatchdogFeeder, POSITION_TOLERANCE

DEFAULT_TRIALS = 5
SAMPLE_HZ = 100
WINDOW_SECONDS = 5.0
DWELL_SECONDS = 0.2  # must stay within tolerance for this long to count as "settled"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--window", type=float, default=WINDOW_SECONDS,
                         help="seconds to sample after each ball drop")
    args = parser.parse_args()

    print("=" * 55)
    print("LAUNCH BALL-DROP SETTLE TEST")
    print("=" * 55)

    odrv, axis = connect_launch()
    start_pos = prepare_launch(odrv, axis)
    if start_pos is None:
        print("ERROR: could not prepare LAUNCH, aborting.")
        sys.exit(1)

    feeder = WatchdogFeeder(axis)
    feeder.start()

    interval = 1.0 / SAMPLE_HZ
    dwell_samples = max(1, int(DWELL_SECONDS / interval))
    final_deviations = []

    try:
        for i in range(args.trials):
            input(f"\nTrial {i + 1}/{args.trials}: confirm arm is EMPTY at start "
                  f"position, then press ENTER, then drop the ball immediately "
                  f"after (sampling starts on ENTER)...")

            t0 = time.time()
            samples = []  # (t, pos)
            n = int(args.window * SAMPLE_HZ)
            for _ in range(n):
                t = time.time() - t0
                pos = axis.pos_estimate
                samples.append((t, pos))
                time.sleep(interval)

            deviations = [pos - start_pos for _, pos in samples]
            abs_devs = [abs(d) for d in deviations]
            peak_idx = max(range(len(abs_devs)), key=lambda k: abs_devs[k])
            peak_dev = deviations[peak_idx]
            peak_t = samples[peak_idx][0]

            settle_t = None
            for k in range(len(deviations) - dwell_samples):
                window = abs_devs[k:k + dwell_samples]
                if max(window) < POSITION_TOLERANCE:
                    settle_t = samples[k][0]
                    break

            final_dev = deviations[-1]
            final_deviations.append(final_dev)

            print(f"  peak deviation: {peak_dev:+.5f} turns ({peak_dev * 360:+.2f} deg) "
                  f"at t={peak_t:.2f}s")
            if settle_t is not None:
                print(f"  settled (within {POSITION_TOLERANCE} turns for "
                      f"{DWELL_SECONDS:.1f}s) at t={settle_t:.2f}s")
            else:
                print(f"  did NOT settle within the {args.window:.1f}s window - "
                      f"still moving/oscillating at window end")
            print(f"  final deviation at t={samples[-1][0]:.2f}s: {final_dev:+.5f} turns "
                  f"({final_dev * 360:+.2f} deg)")

            if i < args.trials - 1:
                input("  Remove the ball, then press ENTER to continue to the next trial...")

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        feeder.stop()
        axis.requested_state = AxisState.IDLE
        print("\nLAUNCH disabled.")

    if len(final_deviations) < 2:
        print("Not enough successful trials to compute stats.")
        return

    spread_turns = max(final_deviations) - min(final_deviations)
    stdev_turns = statistics.pstdev(final_deviations)

    print("\n" + "-" * 55)
    print(f"{len(final_deviations)} trials")
    print(f"  final-position spread (max-min): {spread_turns:.5f} turns "
          f"({spread_turns * 360:.2f} deg)")
    print(f"  final-position stdev:            {stdev_turns:.5f} turns "
          f"({stdev_turns * 360:.2f} deg)")
    print("-" * 55)
    print("If the final-position spread across trials is small (well under a")
    print("degree), the arm reliably returns to the same start position after")
    print("a ball drop, regardless of how long you wait - premature firing")
    print("wouldn't matter much. If it's large, or if several trials didn't")
    print("settle within the window, that supports the hypothesis: firing")
    print("before it's done settling would fire from a different start")
    print("position each time.")


if __name__ == "__main__":
    main()
