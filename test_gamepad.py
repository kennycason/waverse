#!/usr/bin/env python3
"""Simple gamepad test script to identify axis and button mappings."""

import pygame
import time

pygame.init()
pygame.joystick.init()

print("=" * 60)
print("GAMEPAD TESTER")
print("=" * 60)

if pygame.joystick.get_count() == 0:
    print("No gamepad detected! Please connect a controller.")
    pygame.quit()
    exit(1)

gamepad = pygame.joystick.Joystick(0)
gamepad.init()

print(f"Detected: {gamepad.get_name()}")
print(f"Axes: {gamepad.get_numaxes()}")
print(f"Buttons: {gamepad.get_numbuttons()}")
print(f"Hats: {gamepad.get_numhats()}")
print("=" * 60)
print("\nMove sticks and press buttons to see values.")
print("Press Ctrl+C to exit.\n")

try:
    while True:
        pygame.event.pump()  # Process events
        
        # Print axes
        axes = []
        for i in range(gamepad.get_numaxes()):
            val = gamepad.get_axis(i)
            # Highlight non-zero values
            if abs(val) > 0.1:
                axes.append(f"\033[92m{i}:{val:+.2f}\033[0m")  # Green
            else:
                axes.append(f"{i}:{val:+.2f}")
        
        # Print buttons
        buttons = []
        for i in range(gamepad.get_numbuttons()):
            if gamepad.get_button(i):
                buttons.append(str(i))
        
        # Print hats (d-pad)
        hats = []
        for i in range(gamepad.get_numhats()):
            hat = gamepad.get_hat(i)
            if hat != (0, 0):
                hats.append(f"{i}:{hat}")
        
        print(f"\rAxes: {' '.join(axes)} | Btns: {','.join(buttons) if buttons else '-'} | Hats: {','.join(hats) if hats else '-'}    ", end="", flush=True)
        
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n\nExiting...")
    pygame.quit()

