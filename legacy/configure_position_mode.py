import odrive
from odrive.enums import *
import math

print("Finding ODrive...")
odrv = odrive.find_any()
print(f"Found ODrive: {odrv.serial_number}")

# Update controller settings for position control
print("Configuring position control mode...")
odrv.axis0.controller.config.control_mode = ControlMode.POSITION_CONTROL
odrv.axis0.controller.config.input_mode = InputMode.TRAP_TRAJ

# Set position control gains (these are starting values, you may need to tune)
odrv.axis0.controller.config.pos_gain = 20.0  # Position proportional gain
odrv.axis0.controller.config.vel_gain = 0.16  # Velocity proportional gain
odrv.axis0.controller.config.vel_integrator_gain = 0.32  # Velocity integral gain

# Trajectory limits for smooth motion
odrv.axis0.trap_traj.config.vel_limit = 15  # Max velocity (turns/sec)
odrv.axis0.trap_traj.config.accel_limit = 20  # Max acceleration (turns/sec²)
odrv.axis0.trap_traj.config.decel_limit = 20  # Max deceleration (turns/sec²)

print("Saving configuration...")
odrv.save_configuration()
print("Configuration saved! Please wait for ODrive to reboot...")
print("Unplug and replug the ODrive after the reboot completes.")
