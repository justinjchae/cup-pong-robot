"""
Read-only LAUNCH axis position monitor - no motion, motor stays idle/
unpowered so you can freely hand-rotate the arm to find the hard stop.

Connects ONLY to the LAUNCH axis (S/N 394A355F3231), never touches PAN.

Usage:
  odrive_env/bin/python launch_read_position.py

Hand-position the launch arm fully against its hard stop, watch the live
readout settle, then Ctrl+C and note BOTH values:
  - raw          -> save as LAUNCH_HARD_STOP_RAW in combined_control.py.
                     onboard_encoder0.raw is a genuine absolute
                     magnetic-sensor reading (MA702 chip) that survives
                     power cycles for the same physical position (found
                     on the PAN axis, same hardware applies here).
  - pos_estimate -> informational only. Does NOT survive a power cycle
                     for the same physical position.
"""
import time

import odrive

LAUNCH_SERIAL = "394A355F3231"


def main():
    print(f"Connecting to LAUNCH axis (S/N {LAUNCH_SERIAL})...")
    odrv = odrive.find_any(serial_number=LAUNCH_SERIAL, timeout=10)
    axis = odrv.axis0
    print(f"  Connected. fw={odrv.fw_version_major}.{odrv.fw_version_minor}.{odrv.fw_version_revision} "
          f"current_state={axis.current_state}")
    print("\nMotor is NOT being commanded - safe to hand-rotate freely.")
    print("Move the launch arm fully against its hard stop and watch it settle below.")
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
              f"raw = {raw:.4f}  <-- save this as LAUNCH_HARD_STOP_RAW")


if __name__ == "__main__":
    main()
