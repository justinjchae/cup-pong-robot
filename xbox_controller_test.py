"""
Diagnostic-only: prints live axis/button/hat values from the paired Xbox
controller so we can confirm exact indices before wiring it into jog/fire
logic. No hardware (ODrive) touched at all - pure input readout.

Run with: odrive_env/bin/python xbox_controller_test.py
Ctrl+C to quit.
"""
import os
import time

# Headless - no display or audio device on this Pi. Only init the joystick
# subsystem (not pygame.init(), which also probes audio/video and can hang
# for a while looking for an ALSA device that isn't there).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import pygame


def main():
    pygame.display.init()
    pygame.joystick.init()

    if pygame.joystick.get_count() == 0:
        print("No joystick found. Is the controller connected? Check with: "
              "bluetoothctl info 98:7A:14:D1:C2:B8")
        return

    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Found: {js.get_name()}")
    print(f"  axes={js.get_numaxes()}  buttons={js.get_numbuttons()}  hats={js.get_numhats()}")
    print("\nMove sticks / press buttons / press D-pad - watching for changes. Ctrl+C to quit.\n")

    last_axes = [0.0] * js.get_numaxes()
    last_buttons = [0] * js.get_numbuttons()
    last_hats = [(0, 0)] * js.get_numhats()

    try:
        while True:
            pygame.event.pump()

            axes = [js.get_axis(i) for i in range(js.get_numaxes())]
            buttons = [js.get_button(i) for i in range(js.get_numbuttons())]
            hats = [js.get_hat(i) for i in range(js.get_numhats())]

            for i, (a, la) in enumerate(zip(axes, last_axes)):
                if abs(a - la) > 0.05:
                    print(f"  axis[{i}] = {a:+.3f}")
            for i, (b, lb) in enumerate(zip(buttons, last_buttons)):
                if b != lb:
                    print(f"  button[{i}] = {b}")
            for i, (h, lh) in enumerate(zip(hats, last_hats)):
                if h != lh:
                    print(f"  hat[{i}] = {h}")

            last_axes, last_buttons, last_hats = axes, buttons, hats
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
