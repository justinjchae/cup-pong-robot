import odrive
from odrive.enums import *
import time
import sys
import tty
import termios

print("Finding ODrive...")
odrv = odrive.find_any()
print(f"Found ODrive: {odrv.serial_number}")

# Define positions (in turns - ODrive uses turns as the unit)
# 1 full rotation = 1 turn
# 90 degrees = 0.25 turns
POS1 = -0.125  # -45 degrees below parallel
POS2 = 0.125   # +45 degrees above parallel

def clear_errors():
    """Clear any errors on the axis"""
    odrv.clear_errors()
    time.sleep(0.1)

def enable_motor():
    """Put motor into closed loop control"""
    print("Enabling motor...")
    odrv.axis0.requested_state = AxisState.CLOSED_LOOP_CONTROL
    time.sleep(0.5)
    
    # Check if motor is in closed loop
    if odrv.axis0.current_state != AxisState.CLOSED_LOOP_CONTROL:
        print(f"Error: Could not enable motor. Current state: {odrv.axis0.current_state}")
        print(f"Error code: {odrv.axis0.active_errors}")
        return False
    
    print("Motor enabled successfully!")
    return True

def move_to_position(position, description):
    """Move to a specified position"""
    print(f"Moving to {description} (position: {position} turns)...")
    odrv.axis0.controller.input_pos = position
    
    # Wait for movement to complete (position reached within tolerance)
    timeout = 5.0  # seconds
    start_time = time.time()
    
    while True:
        current_pos = odrv.axis0.pos_vel_mapper.pos_rel
        error = abs(current_pos - position)
        
        if error < 0.01:  # Within 0.01 turns (~3.6 degrees)
            print(f"Reached {description}!")
            break
        
        if time.time() - start_time > timeout:
            print(f"Warning: Timeout waiting for position. Current error: {error:.4f} turns")
            break
        
        time.sleep(0.05)

def get_single_keypress():
    """Get a single keypress without requiring Enter"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    print("\n" + "="*50)
    print("PING PONG LAUNCHER CONTROL")
    print("="*50)
    
    # Clear any existing errors
    clear_errors()
    
    # Enable the motor
    if not enable_motor():
        print("Failed to enable motor. Exiting.")
        return
    
    # SET CURRENT POSITION AS ZERO
    print("\nSetting current position as zero reference...")
    odrv.axis0.set_abs_pos(0)
    time.sleep(0.2)
    print(f"Zero set! Current position is now 0.000 turns")

    # Move to starting position
    print("\nMoving to starting position (POS1)...")
    move_to_position(POS1, "POS1 (ready position)")
    
    print("\n" + "-"*50)
    print("READY TO LAUNCH!")
    print("-"*50)
    print("Commands:")
    print("  SPACE - Launch (move to POS2)")
    print("  R     - Return to POS1 (ready position)")
    print("  Q     - Quit and disable motor")
    print("-"*50 + "\n")
    
    try:
        while True:
            key = get_single_keypress().lower()
            
            if key == ' ':
                print("\n🚀 LAUNCHING!")
                move_to_position(POS2, "POS2 (launch position)")
                print("Press 'R' to return to ready position, or 'Q' to quit.\n")
            
            elif key == 'r':
                print("\n↩️  Returning to ready position...")
                move_to_position(POS1, "POS1 (ready position)")
                print("Ready to launch! Press SPACE.\n")
            
            elif key == 'q':
                print("\n\nShutting down...")
                break
            
            elif key == '\x03':  # Ctrl+C
                print("\n\nInterrupted!")
                break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted!")
    
    finally:
        # Disable motor and return to idle
        print("Disabling motor...")
        odrv.axis0.requested_state = AxisState.IDLE
        time.sleep(0.5)
        print("Motor disabled. Safe to power off.")
        print("\nGoodbye!\n")

if __name__ == "__main__":
    main()
