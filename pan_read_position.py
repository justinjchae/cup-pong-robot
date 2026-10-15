"""
Read-only PAN axis position monitor - no motion, motor stays idle/unpowered
so you can freely hand-rotate the arm to find your desired "locked" position.

Connects ONLY to the PAN axis (S/N 3970355B3231), never touches LAUNCH.

Usage:
  odrive_env/bin/python pan_read_position.py

Hand-position the pan axis to wherever you want the repeatable "locked"
test position to be, watch the live readout settle, then Ctrl+C and note
BOTH values:
  - raw          -> save as PAN_LOCK_RAW in combined_control.py. This is
                     the number that actually matters: onboard_encoder0.raw
                     is a genuine absolute magnetic-sensor reading (MA702
                     chip) that survives power cycles for the same physical
                     position.
  - pos_estimate -> informational only. This does NOT survive a power
                     cycle for the same physical position - pos_vel_mapper
                     runs in RELATIVE_MODE on this hardware, its zero
                     reference gets re-established every boot.
"""
import time

import odrive

PAN_SERIAL = "3970355B3231"


def main():
    print(f"Connecting to PAN axis (S/N {PAN_SERIAL})...")
    odrv = odrive.find_any(serial_number=PAN_SERIAL, timeout=10)
    axis = odrv.axis0
    print(f"  Connected. fw={odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision} "
          f"current_state={axis.current_state}")
    print("\nMotor is NOT being commanded - safe to hand-rotate freely.")
    print("Move PAN to your desired locked position and watch it settle below.")
    print("Press Ctrl+C when you're done reading.\n")

    try:
        while True:
            pos = axis.pos_estimate
            raw = odrv.onboard_encoder0.raw
            print(f"\rpos_estimate = {pos:8.4f} turns ({pos * 360:7.2f} deg)   "
                  f"raw = {raw:8.4f}   ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pos = axis.pos_estimate
        raw = odrv.onboard_encoder0.raw
        print(f"\n\nFinal reading: pos_estimate = {pos:.4f} turns ({pos * 360:.2f} deg)   "
              f"raw = {raw:.4f}  <-- save this as PAN_LOCK_RAW")


if __name__ == "__main__":
    main()
