"""
One-time PAN axis encoder offset calibration - establishes the relationship
between the onboard absolute magnetic encoder's raw reading and the axis's
usable position (pos_vel_mapper.config.offset), so pos_abs becomes valid and
(once saved) position is tied to a real physical reference that survives
power cycles - unlike pos_estimate/pos_rel, which was found to reset its
zero-reference every boot.

Only ever talks to the PAN ODrive by serial number.

SAFETY: AxisState.ENCODER_OFFSET_CALIBRATION physically spins the motor.
The ODrive's default calibration_lockin.ramp_distance is 0.5 turns (180 deg)
- 4x past this axis's established +/-45 deg safe window (wiring crosses this
joint). This script reduces ramp_distance/vel/accel to a small, safe motion
(well inside +/-45 deg) before running calibration - the motor has 20 pole
pairs, so one electrical period is only 18 deg mechanical, well covered by
the reduced distance used here.
"""
import threading
import time

import odrive
from odrive.enums import AxisState

PAN_SERIAL = "3970355B3231"

# Reduced lock-in motion - comfortably inside the established +/-45 deg
# (0.125 turn) safety window, still >1 electrical period (20 pole pairs ->
# 18 deg/period) for a reliable calibration.
SAFE_RAMP_DISTANCE = 0.06   # turns (~21.6 deg)
SAFE_VEL = 0.3              # turns/s - matches established-safe JOG_VEL_LIMIT
SAFE_ACCEL = 1.0            # turns/s^2 - matches established-safe JOG_ACCEL_LIMIT

WATCHDOG_FEED_INTERVAL = 0.2


class WatchdogFeeder:
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


def main():
    print(f"Connecting to PAN axis (S/N {PAN_SERIAL})...")
    odrv = odrive.find_any(serial_number=PAN_SERIAL, timeout=10)
    axis = odrv.axis0

    print(f"  current_state={axis.current_state} active_errors={axis.active_errors}")
    odrv.clear_errors()

    lockin = axis.config.calibration_lockin
    print(f"\nOriginal lock-in: current={lockin.current} ramp_time={lockin.ramp_time} "
          f"ramp_distance={lockin.ramp_distance} ({lockin.ramp_distance * 360:.1f} deg) "
          f"accel={lockin.accel} vel={lockin.vel}")

    if lockin.ramp_distance > 0.125:
        lockin.ramp_distance = SAFE_RAMP_DISTANCE
        lockin.vel = SAFE_VEL
        lockin.accel = SAFE_ACCEL
        print(f"Reduced lock-in: ramp_distance={lockin.ramp_distance} "
              f"({lockin.ramp_distance * 360:.1f} deg) accel={lockin.accel} vel={lockin.vel}")
    else:
        print("Lock-in distance already within safe window - leaving as-is.")

    pvm = axis.pos_vel_mapper
    pos_before = axis.pos_estimate
    abs_before = pvm.pos_abs
    print(f"\nBefore calibration: pos_estimate={pos_before:.4f} turns  pos_abs={abs_before}")

    feeder = WatchdogFeeder(axis, WATCHDOG_FEED_INTERVAL)
    feeder.start()

    print("\nRunning ENCODER_OFFSET_CALIBRATION (motor will move briefly, small motion)...")
    axis.requested_state = AxisState.ENCODER_OFFSET_CALIBRATION

    start_time = time.time()
    while axis.current_state != AxisState.IDLE:
        if time.time() - start_time > 10.0:
            print("TIMEOUT waiting for calibration to return to IDLE.")
            break
        time.sleep(0.1)

    feeder.stop()

    print(f"\nFinished. current_state={axis.current_state}")
    print(f"procedure_result={axis.procedure_result}")
    print(f"active_errors={axis.active_errors} disarm_reason={axis.disarm_reason}")

    pos_after = axis.pos_estimate
    abs_after = pvm.pos_abs
    print(f"\nAfter calibration: pos_estimate={pos_after:.4f} turns  pos_abs={abs_after}")
    print(f"pos_vel_mapper.config.offset_valid = {pvm.config.offset_valid}")
    print(f"pos_vel_mapper.config.offset       = {pvm.config.offset}")

    # Safety: force IDLE regardless of outcome.
    axis.requested_state = AxisState.IDLE


if __name__ == "__main__":
    main()
