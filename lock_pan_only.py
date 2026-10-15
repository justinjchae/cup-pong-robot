"""
Headless PAN lock-and-hold - no keyboard interaction, no jog. Connects to
PAN, drives it to the saved locked position (pong_common.lock_pan), then
just holds there indefinitely, feeding the watchdog, until Ctrl+C/killed.

Exists for sessions where PAN just needs to sit locked and out of the way
(not freewheeling/idle) while some other script - e.g. hardstop_decel_test.py
- exercises LAUNCH only in its own process. Never touches LAUNCH.

Run with: odrive_env/bin/python lock_pan_only.py
"""
import time

from odrive.enums import AxisState

from pong_common import WATCHDOG_FEED_INTERVAL, WatchdogFeeder, connect_pan, lock_pan


def main():
    odrv, axis = connect_pan()
    pan_center = lock_pan(odrv, axis)
    if pan_center is None:
        axis.requested_state = AxisState.IDLE
        return

    feeder = WatchdogFeeder(axis, WATCHDOG_FEED_INTERVAL)
    feeder.start()
    print(f"PAN locked and holding at {pan_center:.4f} turns ({pan_center * 360:.1f} deg). "
          "Ctrl+C to release.\n")

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nReleasing PAN...")
    finally:
        feeder.stop()
        axis.requested_state = AxisState.IDLE
        print("PAN disabled.\n")


if __name__ == "__main__":
    main()
