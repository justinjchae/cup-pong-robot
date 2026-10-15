"""
LAUNCH firing telemetry logger (2026-09-21 session): with every static/
resting-position check (PAN repeatability, firing recoil, ball-drop settle)
now ruled out as a scatter source, this checks the one thing none of those
could see - whether the arm's actual DYNAMIC motion during the fast launch
stroke itself is consistent shot to shot. A statically rigid arm can still
flex transiently under acceleration, and the motor's velocity/current
profile could vary run to run even commanding the same target.

Fires real shots (via pong_common.fire_shot - the same function the actual
control scripts use, so the logged motion is the real production firing
sequence, not a reimplementation) while a background thread samples
pos_estimate, vel_estimate, and motor current (Iq_measured) at whatever
rate the USB link allows (measured ~120Hz on this rig - each sample's three
reads are staggered by a few ms, negligible next to the ~100s-of-ms swing
duration).

Saves one CSV with all trials (trial, t_s, pos_turns, vel_turns_per_s,
Iq_A) for external plotting/overlay, and prints a same-run summary (peak
velocity/current and their timing, per trial + cross-trial spread) so you
get an immediate signal before opening anything.

Locks PAN to its normal locked position first (same as real operation) and
holds it there for the whole run - only LAUNCH telemetry is sampled/logged,
but PAN is armed and locked throughout so the firing conditions (and any
PAN-LAUNCH coupling, e.g. recoil transmitted through the shared mount) match
real shots rather than an isolated LAUNCH-only bench test.

--decel-limit lets you test a gentler deceleration on the SWING (launch)
move specifically - the hypothesis, from watching slow-mo, that the
ball separates right around the accel->decel transition and the abrupt
decel jerk right at release could be kicking it inconsistently. Because
trap_traj must cover a fixed total swing distance under accel-then-decel,
softening decel alone (same swing_deg) would also lower peak/exit velocity
- entangling any scatter change with a range change. So unless --swing-deg
is given explicitly, this AUTO-COMPUTES a longer swing_deg that keeps the
accel phase (and therefore peak velocity, and the arm angle at peak) exactly
matching the default 90deg/equal-accel-decel baseline - only the decel half
gets stretched out. The result is checked against max_safe_launch_pos()
(the far/camera-side hard stop, 2026-09-21, 20deg buffer) before ever
being commanded - refuses to fire rather than risk hitting the camera.
"""
import argparse
import csv
import os
import math
import statistics
import sys
import threading
import time

from odrive.enums import AxisState

from pong_common import (
    connect_launch, prepare_launch, fire_shot, WatchdogFeeder,
    connect_pan, lock_pan, max_safe_launch_pos,
    LAUNCH_TRAP_ACCEL_LIMIT, LAUNCH_TRAP_DECEL_LIMIT,
)

DEFAULT_TRIALS = 5
# 70deg, NOT 90 - this is the actual swing angle used for
# every real scatter test (the 4-6in lateral/range baseline numbers), even
# though launch_arm_control.py's own default swing is 90deg. Confirmed
# 2026-09-21 after a decel=50/90deg-referenced test shot flew past the
# table - it had been tuned to preserve peak velocity at the WRONG (90deg)
# reference, well above what the 70deg baseline shots actually fly at.
DEFAULT_SWING_DEG = 70.0
POST_FIRE_BUFFER_S = 0.3  # keep sampling a bit after fire_shot returns, to catch final settling


def baseline_peak_velocity():
    """Peak velocity of the default (70deg) swing under the stock (equal)
    accel/decel limits - the pure-triangular-profile result. Used as the
    reference to preserve when computing a longer swing_deg for a softened
    decel_limit, so a decel comparison test actually matches the exit speed
    of the real scatter-test shots instead of some other swing angle."""
    baseline_turns = DEFAULT_SWING_DEG / 360.0
    return math.sqrt(2 * baseline_turns / (1 / LAUNCH_TRAP_ACCEL_LIMIT + 1 / LAUNCH_TRAP_DECEL_LIMIT))


def swing_deg_for_decel(decel_limit):
    """Swing angle (deg) that keeps peak velocity at the baseline value
    (same accel phase, same exit speed) while giving decel_limit however
    much extra distance it needs to ramp down gently."""
    v_peak = baseline_peak_velocity()
    swing_turns = (v_peak ** 2) / (2 * LAUNCH_TRAP_ACCEL_LIMIT) + (v_peak ** 2) / (2 * decel_limit)
    return swing_turns * 360.0


class Sampler:
    def __init__(self, axis, t0):
        self._axis = axis
        self._t0 = t0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.rows = []

    def _run(self):
        while not self._stop.is_set():
            t = time.time() - self._t0
            pos = self._axis.pos_estimate
            vel = self._axis.vel_estimate
            iq = self._axis.motor.foc.Iq_measured
            self.rows.append((t, pos, vel, iq))

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)


def summarize_trial(rows, start_pos, label):
    ts = [r[0] for r in rows]
    poss = [r[1] for r in rows]
    vels = [r[2] for r in rows]
    iqs = [r[3] for r in rows]

    peak_vel_idx = max(range(len(vels)), key=lambda k: abs(vels[k]))
    peak_iq_idx = max(range(len(iqs)), key=lambda k: abs(iqs[k]))
    peak_pos = max(poss, key=lambda p: abs(p - start_pos))

    print(f"  {label}: {len(rows)} samples over {ts[-1]:.2f}s")
    print(f"    peak |vel|: {abs(vels[peak_vel_idx]):.3f} turns/s at t={ts[peak_vel_idx]:.3f}s")
    print(f"    peak |Iq|:  {abs(iqs[peak_iq_idx]):.3f} A at t={ts[peak_iq_idx]:.3f}s")
    print(f"    peak pos reached: {peak_pos:.4f} turns "
          f"(target extent = start + swing)")
    print(f"    final pos: {poss[-1]:.4f} turns (deviation from start: "
          f"{poss[-1] - start_pos:+.5f} turns)")

    return {
        "peak_vel": abs(vels[peak_vel_idx]),
        "peak_vel_t": ts[peak_vel_idx],
        "peak_iq": abs(iqs[peak_iq_idx]),
        "peak_iq_t": ts[peak_iq_idx],
        "peak_pos": peak_pos,
        "final_dev": poss[-1] - start_pos,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--decel-limit", type=float, default=LAUNCH_TRAP_DECEL_LIMIT,
                         help="swing-phase decel limit, turns/s^2 (default: stock value, "
                              "no behavior change)")
    parser.add_argument("--swing-deg", type=float, default=None,
                         help="override the auto-computed swing angle (advanced - normally "
                              "left unset so peak velocity is preserved automatically)")
    parser.add_argument("--out", default=None, help="CSV output path (default: timestamped)")
    args = parser.parse_args()

    swing_deg = args.swing_deg if args.swing_deg is not None else swing_deg_for_decel(args.decel_limit)
    out_path = args.out or os.path.join("data", "launch_telemetry", time.strftime(
        f"launch_telemetry_decel{args.decel_limit:.0f}_%Y%m%d_%H%M%S.csv"))

    print("=" * 55)
    print("LAUNCH FIRING TELEMETRY LOGGER")
    print("=" * 55)
    print("Fires real shots - clear the launch path.")
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

    odrv, axis = connect_launch()
    start_pos = prepare_launch(odrv, axis)
    if start_pos is None:
        print("ERROR: could not prepare LAUNCH, aborting.")
        pan_feeder.stop()
        pan_axis.requested_state = AxisState.IDLE
        sys.exit(1)

    feeder = WatchdogFeeder(axis)
    feeder.start()

    safe_max = max_safe_launch_pos(odrv, axis)
    swing_turns = swing_deg / 360.0
    target_pos = start_pos + swing_turns
    print(f"\ndecel_limit={args.decel_limit:.1f} turns/s^2 (accel unchanged at "
          f"{LAUNCH_TRAP_ACCEL_LIMIT:.1f}) -> swing={swing_deg:.2f} deg "
          f"(target peak velocity ~{baseline_peak_velocity():.3f} turns/s, matching "
          f"the stock {DEFAULT_SWING_DEG:.0f}deg/equal-accel-decel baseline)")
    print(f"target position {target_pos:.4f} turns vs safe max {safe_max:.4f} turns "
          f"(margin {(safe_max - target_pos) * 360:.1f} deg)")
    if target_pos > safe_max:
        print("ERROR: this swing would go past the far-side hard-stop safety margin - "
              "refusing to fire. Try a less extreme --decel-limit or a smaller manual "
              "--swing-deg.")
        feeder.stop()
        pan_feeder.stop()
        axis.requested_state = AxisState.IDLE
        pan_axis.requested_state = AxisState.IDLE
        sys.exit(1)

    all_rows = []  # (trial, t, pos, vel, iq)
    trial_summaries = []

    try:
        for i in range(args.trials):
            input(f"\nTrial {i + 1}/{args.trials}: press ENTER to fire...")

            t0 = time.time()
            sampler = Sampler(axis, t0)
            sampler.start()

            result = fire_shot(axis, start_pos, swing_deg,
                                swing_accel_limit=LAUNCH_TRAP_ACCEL_LIMIT,
                                swing_decel_limit=args.decel_limit)

            time.sleep(POST_FIRE_BUFFER_S)
            sampler.stop()

            if result != 'done':
                print(f"  fire_shot returned {result!r} - logging what was captured, "
                      f"but treat this trial with caution.")

            for t, pos, vel, iq in sampler.rows:
                all_rows.append((i + 1, t, pos, vel, iq))

            summary = summarize_trial(sampler.rows, start_pos, f"trial {i + 1}")
            trial_summaries.append(summary)

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        feeder.stop()
        pan_feeder.stop()
        axis.requested_state = AxisState.IDLE
        pan_axis.requested_state = AxisState.IDLE
        print("\nLAUNCH and PAN disabled.")

    if all_rows:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "t_s", "pos_turns", "vel_turns_per_s", "Iq_A"])
            writer.writerows(all_rows)
        print(f"\nSaved {len(all_rows)} samples across {len(trial_summaries)} trials to {out_path}")

    if len(trial_summaries) < 2:
        print("Not enough successful trials to compute cross-trial stats.")
        return

    peak_vels = [s["peak_vel"] for s in trial_summaries]
    peak_vel_ts = [s["peak_vel_t"] for s in trial_summaries]
    peak_iqs = [s["peak_iq"] for s in trial_summaries]
    peak_poss = [s["peak_pos"] for s in trial_summaries]

    print("\n" + "-" * 55)
    print(f"{len(trial_summaries)} trials - cross-trial consistency")
    print(f"  peak velocity:     mean={statistics.mean(peak_vels):.3f}  "
          f"stdev={statistics.pstdev(peak_vels):.3f} turns/s")
    print(f"  peak velocity timing: mean={statistics.mean(peak_vel_ts):.3f}  "
          f"stdev={statistics.pstdev(peak_vel_ts):.3f} s")
    print(f"  peak current:      mean={statistics.mean(peak_iqs):.3f}  "
          f"stdev={statistics.pstdev(peak_iqs):.3f} A")
    print(f"  peak position reached: mean={statistics.mean(peak_poss):.4f}  "
          f"stdev={statistics.pstdev(peak_poss):.5f} turns "
          f"({statistics.pstdev(peak_poss) * 360:.3f} deg)")
    print("-" * 55)
    print("Large stdev in peak velocity/current relative to their means points to")
    print("inconsistent motor execution shot-to-shot (control or power-side).")
    print("Large stdev in peak position reached (the actual swing extent) with")
    print("tight velocity/current stdev would instead point toward mechanical")
    print("flex/slip during the swing that the controller can't see or correct")
    print(f"for in time. Open {out_path} to overlay the raw pos/vel/Iq traces")
    print("across trials directly if the summary numbers don't make it clear.")


if __name__ == "__main__":
    main()
