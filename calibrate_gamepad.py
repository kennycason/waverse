#!/usr/bin/env python3
"""
Interactive gamepad calibration tool.
Asks user to perform each action and records the mappings.
Saves to ~/.waverse_gamepad.json
"""

import pygame
from pygame.locals import *
import json
import os
import time

SAVE_FILE = os.path.expanduser("~/.waverse_gamepad.json")


def init_pygame_with_display():
    """Initialize pygame with a display window (required for gamepad on macOS)."""
    pygame.init()
    # Create a small window - required for joystick detection on macOS
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Gamepad Calibration")
    
    # Fill with instructions
    screen.fill((40, 40, 50))
    font = pygame.font.Font(None, 24)
    lines = [
        "GAMEPAD CALIBRATION",
        "",
        "Follow prompts in terminal.",
        "Keep this window focused.",
        "",
        "Press ESC to cancel."
    ]
    y = 30
    for line in lines:
        text = font.render(line, True, (200, 200, 220))
        rect = text.get_rect(centerx=200, y=y)
        screen.blit(text, rect)
        y += 28
    pygame.display.flip()
    
    # Now init joystick
    pygame.joystick.quit()
    pygame.joystick.init()
    
    # Pump events to detect controllers
    for _ in range(10):
        pygame.event.pump()
        time.sleep(0.1)
    
    return screen


def check_quit():
    """Check for quit events."""
    for event in pygame.event.get():
        if event.type == QUIT:
            return True
        if event.type == KEYDOWN and event.key == K_ESCAPE:
            return True
    return False


def wait_for_axis_movement(gamepad, prompt, direction="any"):
    """Wait for user to move an axis and return which axis moved."""
    print(f"\n>>> {prompt}")
    print("    (waiting for input...)")
    
    pygame.event.pump()
    time.sleep(0.3)
    
    # Record baseline values
    baseline = {}
    for i in range(gamepad.get_numaxes()):
        baseline[i] = gamepad.get_axis(i)
    
    # Wait for significant movement
    while True:
        if check_quit():
            return None, None
        pygame.event.pump()
        for i in range(gamepad.get_numaxes()):
            current = gamepad.get_axis(i)
            delta = current - baseline[i]
            
            if direction == "positive" and delta > 0.5:
                print(f"    Detected: Axis {i} (value: {current:+.2f})")
                time.sleep(0.3)
                return i, False  # axis, not inverted
            elif direction == "negative" and delta < -0.5:
                print(f"    Detected: Axis {i} (value: {current:+.2f})")
                time.sleep(0.3)
                return i, False
            elif direction == "any" and abs(delta) > 0.5:
                inverted = delta < 0
                print(f"    Detected: Axis {i} (value: {current:+.2f}, inverted: {inverted})")
                time.sleep(0.3)
                return i, inverted
        
        time.sleep(0.05)


def wait_for_button(gamepad, prompt):
    """Wait for user to press a button and return which button."""
    print(f"\n>>> {prompt}")
    print("    (waiting for button press...)")
    
    # Wait for all buttons to be released first
    pygame.event.pump()
    time.sleep(0.2)
    
    while True:
        if check_quit():
            return None
        pygame.event.pump()
        for i in range(gamepad.get_numbuttons()):
            if gamepad.get_button(i):
                print(f"    Detected: Button {i}")
                # Wait for release
                while gamepad.get_button(i):
                    pygame.event.pump()
                    time.sleep(0.05)
                time.sleep(0.2)
                return i
        time.sleep(0.05)


def wait_for_trigger(gamepad, prompt):
    """Wait for trigger press - could be axis or button."""
    print(f"\n>>> {prompt}")
    print("    (press the trigger fully...)")
    
    pygame.event.pump()
    time.sleep(0.3)
    
    # Record baseline
    baseline = {}
    for i in range(gamepad.get_numaxes()):
        baseline[i] = gamepad.get_axis(i)
    
    while True:
        if check_quit():
            return None, None, None
        pygame.event.pump()
        
        # Check axes
        for i in range(gamepad.get_numaxes()):
            current = gamepad.get_axis(i)
            delta = current - baseline[i]
            if abs(delta) > 0.5:
                print(f"    Detected: Axis {i} (baseline: {baseline[i]:+.2f}, now: {current:+.2f})")
                time.sleep(0.3)
                return ("axis", i, baseline[i])
        
        # Check buttons
        for i in range(gamepad.get_numbuttons()):
            if gamepad.get_button(i):
                print(f"    Detected: Button {i}")
                while gamepad.get_button(i):
                    pygame.event.pump()
                    time.sleep(0.05)
                time.sleep(0.2)
                return ("button", i, 0)
        
        time.sleep(0.05)


def wait_for_dpad_or_button(gamepad, prompt):
    """Wait for D-pad input - could be hat or buttons."""
    print(f"\n>>> {prompt}")
    print("    (press and hold...)")
    
    pygame.event.pump()
    time.sleep(0.2)
    
    while True:
        if check_quit():
            return None
        pygame.event.pump()
        
        # Check hats (D-pad on many controllers)
        for i in range(gamepad.get_numhats()):
            hat = gamepad.get_hat(i)
            if hat != (0, 0):
                direction = None
                if hat[1] == 1:
                    direction = "up"
                elif hat[1] == -1:
                    direction = "down"
                elif hat[0] == -1:
                    direction = "left"
                elif hat[0] == 1:
                    direction = "right"
                if direction:
                    print(f"    Detected: Hat {i} ({direction})")
                    # Wait for release
                    while gamepad.get_hat(i) != (0, 0):
                        pygame.event.pump()
                        time.sleep(0.05)
                    time.sleep(0.2)
                    return {"type": "hat", "hat": i, "direction": direction}
        
        # Check buttons
        for i in range(gamepad.get_numbuttons()):
            if gamepad.get_button(i):
                print(f"    Detected: Button {i}")
                while gamepad.get_button(i):
                    pygame.event.pump()
                    time.sleep(0.05)
                time.sleep(0.2)
                return {"type": "button", "button": i}
        
        time.sleep(0.05)


def main():
    print("=" * 60)
    print("  WAVERSE GAMEPAD CALIBRATION")
    print("=" * 60)
    print("\nInitializing (a window will appear)...")
    
    screen = init_pygame_with_display()
    
    print(f"Joysticks found: {pygame.joystick.get_count()}")
    
    if pygame.joystick.get_count() == 0:
        print("\nERROR: No gamepad detected!")
        print("Make sure your controller is connected and try again.")
        print("(Keep the pygame window focused)")
        time.sleep(3)
        pygame.quit()
        return
    
    gamepad = pygame.joystick.Joystick(0)
    gamepad.init()
    
    print(f"\nDetected: {gamepad.get_name()}")
    print(f"  Axes: {gamepad.get_numaxes()}")
    print(f"  Buttons: {gamepad.get_numbuttons()}")
    
    # Print current axis values
    print("\n  Current axis values:")
    for i in range(gamepad.get_numaxes()):
        val = gamepad.get_axis(i)
        print(f"    Axis {i}: {val:+.3f}")
    
    print("\n" + "-" * 60)
    print("Follow the prompts below. Move sticks/press buttons as asked.")
    print("Release controls between each prompt.")
    print("Press ESC in the pygame window to cancel.")
    print("-" * 60)
    
    mappings = {
        "name": gamepad.get_name(),
        "num_axes": gamepad.get_numaxes(),
        "num_buttons": gamepad.get_numbuttons(),
    }
    
    # Left stick
    print("\n[LEFT STICK CALIBRATION]")
    
    ls_x, ls_x_inv = wait_for_axis_movement(gamepad, "Move LEFT STICK to the RIGHT", "positive")
    if ls_x is None:
        print("Cancelled.")
        pygame.quit()
        return
    time.sleep(0.5)
    
    ls_y, ls_y_inv = wait_for_axis_movement(gamepad, "Move LEFT STICK DOWN", "positive")
    if ls_y is None:
        print("Cancelled.")
        pygame.quit()
        return
    
    mappings["left_stick_x"] = ls_x
    mappings["left_stick_x_inverted"] = ls_x_inv
    mappings["left_stick_y"] = ls_y
    mappings["left_stick_y_inverted"] = ls_y_inv
    
    # Right stick
    print("\n[RIGHT STICK CALIBRATION]")
    time.sleep(0.5)
    
    rs_x, rs_x_inv = wait_for_axis_movement(gamepad, "Move RIGHT STICK to the RIGHT", "positive")
    if rs_x is None:
        print("Cancelled.")
        pygame.quit()
        return
    time.sleep(0.5)
    
    rs_y, rs_y_inv = wait_for_axis_movement(gamepad, "Move RIGHT STICK DOWN", "positive")
    if rs_y is None:
        print("Cancelled.")
        pygame.quit()
        return
    
    mappings["right_stick_x"] = rs_x
    mappings["right_stick_x_inverted"] = rs_x_inv
    mappings["right_stick_y"] = rs_y
    mappings["right_stick_y_inverted"] = rs_y_inv
    
    # Triggers
    print("\n[TRIGGER CALIBRATION]")
    time.sleep(0.5)
    
    l2_type, l2_id, l2_baseline = wait_for_trigger(gamepad, "Press L2 (left trigger)")
    if l2_type is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["l2_type"] = l2_type
    mappings["l2_id"] = l2_id
    mappings["l2_baseline"] = l2_baseline
    
    time.sleep(0.5)
    
    r2_type, r2_id, r2_baseline = wait_for_trigger(gamepad, "Press R2 (right trigger)")
    if r2_type is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["r2_type"] = r2_type
    mappings["r2_id"] = r2_id
    mappings["r2_baseline"] = r2_baseline
    
    # Stick buttons
    print("\n[STICK BUTTON CALIBRATION]")
    time.sleep(0.5)
    
    l3 = wait_for_button(gamepad, "Click LEFT STICK (L3)")
    if l3 is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["l3"] = l3
    
    r3 = wait_for_button(gamepad, "Click RIGHT STICK (R3)")
    if r3 is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["r3"] = r3
    
    # Shoulder buttons
    print("\n[SHOULDER BUTTON CALIBRATION]")
    time.sleep(0.5)
    
    l1 = wait_for_button(gamepad, "Press L1 (left shoulder)")
    if l1 is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["l1"] = l1
    
    r1 = wait_for_button(gamepad, "Press R1 (right shoulder)")
    if r1 is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["r1"] = r1
    
    # Face buttons
    print("\n[FACE BUTTON CALIBRATION]")
    time.sleep(0.5)
    
    a_btn = wait_for_button(gamepad, "Press A (bottom button)")
    if a_btn is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["a"] = a_btn
    
    b_btn = wait_for_button(gamepad, "Press B (right button)")
    if b_btn is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["b"] = b_btn
    
    x_btn = wait_for_button(gamepad, "Press X (left button)")
    if x_btn is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["x"] = x_btn
    
    y_btn = wait_for_button(gamepad, "Press Y (top button)")
    if y_btn is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["y"] = y_btn
    
    # Start/Select
    print("\n[MENU BUTTON CALIBRATION]")
    time.sleep(0.5)
    
    start = wait_for_button(gamepad, "Press START")
    if start is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["start"] = start
    
    select = wait_for_button(gamepad, "Press SELECT/BACK")
    if select is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["select"] = select
    
    # D-Pad (could be buttons or hat)
    print("\n[D-PAD CALIBRATION]")
    print("    D-Pad may be buttons or a 'hat' - we'll detect both.")
    time.sleep(0.5)
    
    dpad_up = wait_for_dpad_or_button(gamepad, "Press D-PAD UP")
    if dpad_up is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["dpad_up"] = dpad_up
    
    dpad_down = wait_for_dpad_or_button(gamepad, "Press D-PAD DOWN")
    if dpad_down is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["dpad_down"] = dpad_down
    
    dpad_left = wait_for_dpad_or_button(gamepad, "Press D-PAD LEFT")
    if dpad_left is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["dpad_left"] = dpad_left
    
    dpad_right = wait_for_dpad_or_button(gamepad, "Press D-PAD RIGHT")
    if dpad_right is None:
        print("Cancelled.")
        pygame.quit()
        return
    mappings["dpad_right"] = dpad_right
    
    # Done!
    print("\n" + "=" * 60)
    print("  CALIBRATION COMPLETE!")
    print("=" * 60)
    print("\nYour mappings:")
    print(json.dumps(mappings, indent=2))
    
    # Save
    with open(SAVE_FILE, 'w') as f:
        json.dump(mappings, f, indent=2)
    print(f"\nSaved to: {SAVE_FILE}")
    
    pygame.quit()
    print("\nYou can now run the game - it will use these mappings!")


if __name__ == "__main__":
    main()

