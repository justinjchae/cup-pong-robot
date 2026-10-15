"""
Ball-weight-via-current feasibility check (2026-09-21 debugging session).

Holds the LAUNCH arm at its 'start' position (near-horizontal - see
launch_arm_control.py's "horizontal -> vertical" swing description, and
release_height_m()'s angle convention in pong_common.py, where gravity
torque on a ball at ARM_M is m*g*ARM_M*sin(theta-from-vertical), maximized
right at horizontal) with no ball, samples axis.motor.torque_estimate to
get a baseline, then does the same with a ball resting in the arm, and
reports whether the difference is big enough to read reliably above the
holding-current noise floor.

This is a go/no-go signal check, not a calibrated scale - it answers "can we
detect a ball's weight this way at all" before building any current->mass
lookup table.

Never touches the PAN axis.
"""
import statistics
import sys
import time

from odrive.enums import AxisState

from pong_common import connect_launch, prepare_launch, WatchdogFeeder, ARM_M, G

MOTOR_KV = 90.0  # motor's rated value - compared against the live-read
                 # config below rather than trusted blindly
SAMPLE_HZ = 50
SAMPLE_SECONDS = 3.0
PING_PONG_BALL_G = 2.7  # for the reference prediction printed up front


def sample_torque(axis, duration_s, hz):
    interval = 1.0 / hz
    n = int(duration_s * hz)
    values = []
    for _ in range(n):
        values.append(axis.motor.torque_estimate)
        time.sleep(interval)
    return values


def main():
    print("=" * 55)
    print("BALL WEIGHT VIA CURRENT - FEASIBILITY CHECK")
    print("=" * 55)

    odrv, axis = connect_launch()

    live_kt = axis.config.motor.torque_constant
    predicted_kt = 8.27 / MOTOR_KV
    print(f"\nLive torque_constant from ODrive config: {live_kt:.5f} Nm/A")
    print(f"Predicted from stated {MOTOR_KV:.0f}KV (8.27/KV):  {predicted_kt:.5f} Nm/A")
    if abs(live_kt - predicted_kt) / predicted_kt > 0.05:
        print("  WARNING: more than 5% off - double check the KV value or the "
              "ODrive's motor config before trusting the estimate below.")
    else:
        print("  Matches - KV confirmed.")

    predicted_torque_nm = (PING_PONG_BALL_G / 1000.0) * G * ARM_M
    predicted_delta_a = predicted_torque_nm / live_kt
    print(f"\nPredicted torque from a {PING_PONG_BALL_G:.1f}g ball at ARM_M="
          f"{ARM_M:.3f}m, held horizontal: {predicted_torque_nm:.5f} Nm "
          f"(~{predicted_delta_a * 1000:.0f} mA equivalent current)")

    start_pos = prepare_launch(odrv, axis)
    if start_pos is None:
        print("ERROR: could not prepare LAUNCH, aborting.")
        sys.exit(1)

    feeder = WatchdogFeeder(axis)
    feeder.start()

    try:
        input("\nConfirm the arm is EMPTY (no ball) at the start position, "
              "then press ENTER to sample baseline...")
        print(f"Sampling baseline for {SAMPLE_SECONDS:.0f}s...")
        baseline = sample_torque(axis, SAMPLE_SECONDS, SAMPLE_HZ)

        input("\nPlace the ball in the arm now (resting, won't roll away), "
              "then press ENTER to sample with ball...")
        print(f"Sampling with ball for {SAMPLE_SECONDS:.0f}s...")
        with_ball = sample_torque(axis, SAMPLE_SECONDS, SAMPLE_HZ)

    except KeyboardInterrupt:
        print("\nInterrupted!")
        feeder.stop()
        axis.requested_state = AxisState.IDLE
        return

    feeder.stop()
    axis.requested_state = AxisState.IDLE
    print("\nLAUNCH disabled.")

    base_mean = statistics.mean(baseline)
    base_stdev = statistics.pstdev(baseline)
    ball_mean = statistics.mean(with_ball)
    ball_stdev = statistics.pstdev(with_ball)
    delta_nm = ball_mean - base_mean
    pooled_stdev = ((base_stdev ** 2 + ball_stdev ** 2) / 2) ** 0.5
    snr = abs(delta_nm) / pooled_stdev if pooled_stdev > 0 else float("inf")

    print("\n" + "-" * 55)
    print(f"baseline: mean={base_mean:.5f} Nm  stdev={base_stdev:.5f} Nm  (n={len(baseline)})")
    print(f"w/ ball:  mean={ball_mean:.5f} Nm  stdev={ball_stdev:.5f} Nm  (n={len(with_ball)})")
    print(f"delta:    {delta_nm:+.5f} Nm  ({delta_nm / live_kt * 1000:+.1f} mA equivalent)")
    print(f"SNR (|delta| / pooled stdev): {snr:.2f}")
    print("-" * 55)

    # Rough sanity-check mass estimate, ASSUMING start is exactly horizontal
    # (theta=90deg from vertical) - real angle unconfirmed, treat as approximate.
    inferred_mass_g = (delta_nm / (G * ARM_M)) * 1000.0 if delta_nm > 0 else float("nan")
    print(f"Approx inferred mass (assumes start = exactly horizontal): "
          f"{inferred_mass_g:.2f} g")

    if snr > 3:
        print("\nSNR > 3: signal is clearly above the noise floor - a current->mass "
              "lookup table is plausible with this setup.")
    elif snr > 1:
        print("\nSNR 1-3: marginal - detectable but noisy. More samples per "
              "reading (longer averaging window) or repeated trials would help "
              "before trusting a lookup table built from single readings.")
    else:
        print("\nSNR < 1: not reliably detectable above the holding-current "
              "noise floor at this sample size. Would need a different "
              "approach (longer averaging, a stiffer arm angle, or a real "
              "load cell) rather than current sensing.")


if __name__ == "__main__":
    main()
