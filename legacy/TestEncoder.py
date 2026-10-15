import odrive
from odrive.enums import *
import time

def degrees_to_turns(degrees):
    return degrees / 360.0

def turns_to_degrees(turns):
    return turns * 360.0

print("Searching for ODrive S1...")
odrv0 = odrive.find_any()
print(f"Found ODrive: {odrv0.serial_number}")

# --- STARTUP ---
print("\nClearing errors...")
odrv0.clear_errors()

print("\nPosition the motor shaft at your 0° reference point.")
input("Manually move it now, then press Enter...")

# Zero the encoder at current position
odrv0.axis0.pos_vel_mapper.set_abs_pos(0)
time.sleep(0.1)

# Sync controller to zeroed position
odrv0.axis0.controller.input_pos = odrv0.axis0.pos_vel_mapper.pos_rel
print(f"✓ Zeroed. Current pos_estimate: {turns_to_degrees(odrv0.axis0.pos_vel_mapper.pos_rel):.4f}°")

# --- CONFIGURE TRAP TRAJ ---
odrv0.axis0.controller.config.input_mode = INPUT_MODE_TRAP_TRAJ
odrv0.axis0.trap_traj.config.vel_limit   = 2.0    # turns/sec — slow for testing
odrv0.axis0.trap_traj.config.accel_limit = 5.0    # turns/sec²
odrv0.axis0.trap_traj.config.decel_limit = 5.0

odrv0.axis0.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL

# Engage closed-loop control
odrv0.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
time.sleep(0.5)
print("✓ Motor engaged in closed-loop position control.")

# --- ENCODER RESOLUTION TEST LOOP ---
print("\n=== Encoder Resolution Test ===")
print("Type an angle in degrees and press Enter. The motor will go there.")
print("Commands: a number (e.g. 45), 'r' to read current pos, 'z' to re-zero, 'q' to quit\n")

try:
    while True:
        raw = input("Target angle (°): ").strip().lower()

        if raw == 'q':
            break

        elif raw == 'r':
            pos_turns = odrv0.axis0.pos_vel_mapper.pos_rel
            pos_deg   = turns_to_degrees(pos_turns)
            target_deg = turns_to_degrees(odrv0.axis0.controller.input_pos)
            error_deg  = pos_deg - target_deg
            print(f"  Actual:  {pos_deg:.4f}°  ({pos_turns:.6f} turns)")
            print(f"  Target:  {target_deg:.4f}°")
            print(f"  Error:   {error_deg:.4f}°")

        elif raw == 'z':
            odrv0.axis0.pos_vel_mapper.set_abs_pos(0)
            time.sleep(0.1)
            odrv0.axis0.controller.input_pos = 0.0
            print("✓ Re-zeroed at current position.")

        else:
            try:
                target_deg   = float(raw)
                target_turns = degrees_to_turns(target_deg)
                odrv0.axis0.controller.input_pos = target_turns
                print(f"  → Moving to {target_deg:.2f}° ({target_turns:.6f} turns)...")

                # Wait for motion to settle, then report actual position
                time.sleep(0.1)
                deadline = time.time() + 5.0
                while time.time() < deadline:
                    vel = abs(odrv0.axis0.vel_estimate)
                    if vel < 0.02:   # turns/sec — considered settled
                        break
                    time.sleep(0.05)

                actual_deg = turns_to_degrees(odrv0.axis0.pos_vel_mapper.pos_rel)
                error_deg  = actual_deg - target_deg
                print(f"  ✓ Settled at {actual_deg:.4f}°  (error: {error_deg:+.4f}°)")

                # Check for ODrive errors
                if odrv0.axis0.active_errors != 0:
                    print(f"  ⚠ Active errors: {odrv0.axis0.active_errors}")

            except ValueError:
                print("  Invalid input. Enter a number, 'r', 'z', or 'q'.")

except KeyboardInterrupt:
    print("\nInterrupted.")

finally:
    print("Disabling motor...")
    odrv0.axis0.requested_state = AXIS_STATE_IDLE
    print("Done.")