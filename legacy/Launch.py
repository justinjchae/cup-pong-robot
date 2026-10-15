import odrive
from odrive.enums import *
import time

def degrees_to_turns(degrees):
    return degrees / 360.0

print("Searching for ODrive...")
odrv0 = odrive.find_any()

# --- SAFE STARTUP SEQUENCE ---
print("Clearing old errors...")
odrv0.clear_errors()

print("\nStep 1: Position the arm at your desired starting point (0°)")
input("Manually move the arm now and press Enter...")

# 1. Zero the encoder
odrv0.axis0.pos_vel_mapper.set_abs_pos(0)
time.sleep(0.1) 

# 2. Sync the controller's goal to the current position to prevent the "jump"
odrv0.axis0.controller.input_pos = odrv0.axis0.pos_estimate
print(f"✓ Controller synced to: {odrv0.axis0.pos_estimate} turns")

# 3. Configure Trapezoidal Trajectory for the launch
odrv0.axis0.controller.config.input_mode = INPUT_MODE_TRAP_TRAJ
odrv0.axis0.trap_traj.config.vel_limit = 20.0   # Adjust for distance
odrv0.axis0.trap_traj.config.accel_limit = 100.0 # Adjust for "snap"
odrv0.axis0.trap_traj.config.decel_limit = 100.0 # Adjust for braking

# 4. Enable motor
odrv0.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
print("✓ Motor engaged and holding position.")

# --- LAUNCH LOOP ---
print("\n=== Launcher Ready ===")
at_zero = True

try:
    while True:
        cmd = input("Press Enter to FLICK (or 'q' to quit): ").lower()
        if cmd == 'q': break
        
        if at_zero:
            print("Launching to 90°...")
            # Set high speed for the shot
            odrv0.axis0.trap_traj.config.vel_limit = 30.0 
            odrv0.axis0.controller.input_pos = degrees_to_turns(90)
            at_zero = False
        else:
            print("Resetting to 0°...")
            # Set slow speed for a safe return
            odrv0.axis0.trap_traj.config.vel_limit = 3.0
            odrv0.axis0.controller.input_pos = degrees_to_turns(0)
            at_zero = True
            
except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("Disabling motor and exiting...")
    odrv0.axis0.requested_state = AXIS_STATE_IDLE