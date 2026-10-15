import odrive
from odrive.enums import *

odrv0 = odrive.find_any()

print("=== Reverting to previous working settings ===")

# Settings from before ping pong ball optimization
odrv0.axis0.controller.config.pos_gain = 100
odrv0.axis0.controller.config.vel_limit = 10
odrv0.axis0.controller.config.vel_integrator_gain = 0

# Make sure we're in passthrough mode
odrv0.axis0.controller.config.input_mode = INPUT_MODE_PASSTHROUGH

print(f"Position gain: {odrv0.axis0.controller.config.pos_gain}")
print(f"Velocity limit: {odrv0.axis0.controller.config.vel_limit}")
print(f"Velocity integrator gain: {odrv0.axis0.controller.config.vel_integrator_gain}")

# What's the current vel_gain?
print(f"Velocity gain: {odrv0.axis0.controller.config.vel_gain}")

odrv0.save_configuration()
print("\nSaved! Rebooting...")
odrv0.reboot()