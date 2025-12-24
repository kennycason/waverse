"""
Terrain Explorer - Based directly on the working andrew_maps explorer.

Simplified and working, just with wave-based generation instead of GAN.
"""

import pygame
from pygame.locals import *
import numpy as np
import math
import time
import json
import os
import random

from .chunk_dna import ChunkDNAManager, ChunkDNA, TerrainPalette
from .climate import ClimateManager, WeatherRenderer
from .structures import StructureManager, Structure
from .life import LifeSimulator

try:
    from OpenGL.GL import *
    from OpenGL.GL import GLubyte
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not found.")

# ModernGL for high-performance rendering (optional)
try:
    import moderngl
    from pyglm import glm  # Avoid deprecation warning
    from .modern_integration import ModernWorldRenderer
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False
    print("Note: ModernGL not available, using legacy renderer.")

from .world import WorldConfig, ChunkManager, CHUNK_SIZE, TILE_SCALE, get_height
from .flora import FloraManager
from .sky import SkySystem
from .animals import AnimalManager
from .chunk_worker import ChunkWorker
from .microscope_view import MicroscopeView
from . import micro_life
from .micro_life import MicroDNA, MICRO_TEMPLATES, TERRAIN_POPULATIONS


# =============================================================================
# RENDER CONFIG - Easy tuning for performance vs quality tradeoffs
# =============================================================================
class RenderConfig:
    """Centralized render settings for easy tweaking."""
    
    # Flora (plants/trees)
    FLORA_RENDER_RADIUS = 22  # Chunks (legacy renderer)
    FLORA_MAX_NEW_PER_FRAME = 3  # Limit chunk loading per frame
    
    # Animals
    ANIMAL_RENDER_HEIGHT_LIMIT = 150  # Don't render if camera this high
    ANIMAL_UPDATE_INTERVAL = 2  # Frames between updates
    
    # Structures (buildings)
    STRUCTURE_SPAWN_RADIUS = 12  # Chunks
    
    # Modern renderer (GPU instanced)
    MODERN_TERRAIN_DISTANCE = 16  # Chunks
    MODERN_FLORA_DISTANCE = 14  # Chunks  
    MODERN_ANIMAL_DISTANCE = 12  # Chunks
    
    # Debug
    NO_FLORA_RENDER = False
    NO_ANIMAL_RENDER = False
    NO_ANIMAL_UPDATE = False


# =============================================================================
# PERFORMANCE MONITOR - Press F3 to dump stats, F4 for overlay
# =============================================================================
class PerfMonitor:
    """Lightweight performance monitoring - only tracks on demand."""
    
    def __init__(self):
        self.enabled = False  # Full tracking off by default
        self.show_overlay = False  # Mini HUD overlay
        
        # Ring buffers for rolling averages (last 60 frames)
        self.frame_times = []
        self.life_times = []
        self.render_times = []
        self.flora_times = []
        self.animal_times = []
        self.chunk_times = []
        self.max_samples = 60
        
        # Current frame timing (temp storage)
        self.current = {}
        
        # Counts
        self.plant_count = 0
        self.animal_count = 0
        self.egg_count = 0
        self.chunk_count = 0
        self.structure_count = 0
        
        # Slow frame detection
        self.slow_frames = 0
        self.slow_threshold_ms = 33.0  # > 30fps = slow
        
        # Last dump time (avoid spam)
        self.last_dump = 0
    
    def start_frame(self):
        """Start timing a frame."""
        self.current['frame_start'] = time.perf_counter()
        
    def end_frame(self):
        """End frame timing and record."""
        if 'frame_start' not in self.current:
            return
        frame_ms = (time.perf_counter() - self.current['frame_start']) * 1000
        self.frame_times.append(frame_ms)
        if len(self.frame_times) > self.max_samples:
            self.frame_times.pop(0)
        
        # Track slow frames
        if frame_ms > self.slow_threshold_ms:
            self.slow_frames += 1
    
    def time_section(self, name: str):
        """Context manager to time a section."""
        return _PerfSection(self, name)
    
    def record_time(self, name: str, start_time: float):
        """Record time for a named section."""
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        buffer = getattr(self, f'{name}_times', None)
        if buffer is not None:
            buffer.append(elapsed_ms)
            if len(buffer) > self.max_samples:
                buffer.pop(0)
    
    def update_counts(self, plants: int, animals: int, eggs: int, chunks: int, structures: int):
        """Update entity counts."""
        self.plant_count = plants
        self.animal_count = animals
        self.egg_count = eggs
        self.chunk_count = chunks
        self.structure_count = structures
    
    def get_avg(self, times: list) -> float:
        """Get average from buffer."""
        return sum(times) / len(times) if times else 0.0
    
    def get_max(self, times: list) -> float:
        """Get max from buffer."""
        return max(times) if times else 0.0
    
    def dump_stats(self):
        """Print detailed performance stats."""
        now = time.time()
        if now - self.last_dump < 0.5:  # Cooldown
            return
        self.last_dump = now
        
        print("\n" + "=" * 70)
        print("  PERFORMANCE STATS (last 60 frames)")
        print("=" * 70)
        
        # Frame times
        avg_frame = self.get_avg(self.frame_times)
        max_frame = self.get_max(self.frame_times)
        fps = 1000 / avg_frame if avg_frame > 0 else 0
        print(f"  FRAME TIME: avg={avg_frame:.1f}ms  max={max_frame:.1f}ms  fps={fps:.0f}")
        print(f"  SLOW FRAMES: {self.slow_frames} (>{self.slow_threshold_ms:.0f}ms)")
        
        # Section breakdown
        print("\n  BREAKDOWN (avg/max ms):")
        print(f"    Life Sim:   {self.get_avg(self.life_times):6.1f} / {self.get_max(self.life_times):.1f}")
        print(f"    Render:     {self.get_avg(self.render_times):6.1f} / {self.get_max(self.render_times):.1f}")
        print(f"    Flora:      {self.get_avg(self.flora_times):6.1f} / {self.get_max(self.flora_times):.1f}")
        print(f"    Animals:    {self.get_avg(self.animal_times):6.1f} / {self.get_max(self.animal_times):.1f}")
        print(f"    Chunks:     {self.get_avg(self.chunk_times):6.1f} / {self.get_max(self.chunk_times):.1f}")
        
        # Entity counts
        print("\n  ENTITY COUNTS:")
        print(f"    Plants:     {self.plant_count:6d}")
        print(f"    Animals:    {self.animal_count:6d}")
        print(f"    Eggs:       {self.egg_count:6d}")
        print(f"    Chunks:     {self.chunk_count:6d}")
        print(f"    Structures: {self.structure_count:6d}")
        
        # Suggestions - realistic thresholds
        print("\n  POTENTIAL ISSUES:")
        issues_found = False
        if self.get_avg(self.life_times) > 15:
            print(f"    ⚠ Life sim slow ({self.get_avg(self.life_times):.1f}ms)")
            issues_found = True
        if self.plant_count > 12000:
            print(f"    ⚠ High plant count ({self.plant_count}) - cleanup running?")
            issues_found = True
        if self.animal_count > 850:
            print(f"    ⚠ High animal count ({self.animal_count})")
            issues_found = True
        if self.get_avg(self.flora_times) > 30:
            print(f"    ⚠ Flora render slow ({self.get_avg(self.flora_times):.1f}ms)")
            issues_found = True
        if self.get_max(self.frame_times) > 200:
            print(f"    ⚠ Frame spikes ({self.get_max(self.frame_times):.0f}ms max) - loading new areas?")
            issues_found = True
        if not issues_found:
            print("    ✓ Performance looks good!")
        
        print("=" * 70 + "\n")
    
    def toggle_overlay(self):
        """Toggle mini performance overlay."""
        self.show_overlay = not self.show_overlay
        print(f"  Perf overlay: {'ON' if self.show_overlay else 'OFF'}")
    
    def reset_slow_frames(self):
        """Reset slow frame counter."""
        self.slow_frames = 0


class _PerfSection:
    """Context manager for timing sections."""
    def __init__(self, monitor: PerfMonitor, name: str):
        self.monitor = monitor
        self.name = name
        self.start = 0
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        self.monitor.record_time(self.name, self.start)


# Configuration - open world style
HEIGHT_SCALE = 3.5
TERRAIN_SCALE = TILE_SCALE
BASE_MOVE_SPEED = 0.8  # Base movement speed (adjustable with 1-5 keys)
MOUSE_SENSITIVITY = 0.06
CHUNK_RENDER_DISTANCE = 20  # Massive view distance!
PLAYER_HEIGHT = 2.5  # Eye height above ground (shorter = world feels bigger, fits through doors)
WALK_SMOOTH_SPEED = 0.4  # Faster terrain following

# Speed levels - more granular at low end for smooth walking
SPEED_LEVELS = [
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0
]  # 7 speed levels - L2 cycles through these (max 2x)

# LOD settings - aggressive LOD for huge view distance
LOD_FULL_DISTANCE = 5      # Full detail within this range
LOD_HALF_DISTANCE = 10     # Half detail within this range  
LOD_QUARTER_DISTANCE = 15  # Quarter detail within this range
# Beyond that = 1/8th detail


# Waverse data directory - all user data goes here
WAVERSE_DIR = os.path.expanduser("~/.waverse")
SAVE_FILE = os.path.join(WAVERSE_DIR, "config.json")
DNA_LOGS_DIR = os.path.join(WAVERSE_DIR, "dna_logs")
SCREENSHOTS_DIR = os.path.join(WAVERSE_DIR, "screenshots")
CHUNK_CACHE_DIR = os.path.join(WAVERSE_DIR, "chunks")

# Ensure directories exist
os.makedirs(WAVERSE_DIR, exist_ok=True)
os.makedirs(DNA_LOGS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(CHUNK_CACHE_DIR, exist_ok=True)


# Gamepad/Controller configuration
GAMEPAD_CONFIG_FILE = os.path.expanduser("~/.waverse_gamepad.json")


class GamepadConfig:
    """Gamepad button and axis mappings - loaded from calibration file."""
    
    # Default mappings (Logitech Dual Action style)
    L_STICK_X = 0
    L_STICK_Y = 1
    L_STICK_X_INV = False
    L_STICK_Y_INV = False
    
    R_STICK_X = 3
    R_STICK_Y = 4
    R_STICK_X_INV = False
    R_STICK_Y_INV = False
    
    L2_TYPE = "axis"  # "axis" or "button"
    L2_ID = 2
    L2_BASELINE = -1.0
    
    R2_TYPE = "axis"
    R2_ID = 5
    R2_BASELINE = -1.0
    
    L3 = 6
    R3 = 7
    L1 = 4
    R1 = 5
    
    A = 1
    B = 2
    X = 0
    Y = 3
    
    START = 9
    SELECT = 8
    
    # D-pad (can be hat or buttons)
    DPAD_UP = {"type": "button", "button": 11}
    DPAD_DOWN = {"type": "button", "button": 12}
    DPAD_LEFT = {"type": "button", "button": 13}
    DPAD_RIGHT = {"type": "button", "button": 14}
    
    DEADZONE = 0.25
    
    @classmethod
    def load_from_file(cls):
        """Load mappings from calibration file if it exists."""
        if os.path.exists(GAMEPAD_CONFIG_FILE):
            try:
                with open(GAMEPAD_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                
                cls.L_STICK_X = data.get("left_stick_x", cls.L_STICK_X)
                cls.L_STICK_Y = data.get("left_stick_y", cls.L_STICK_Y)
                cls.L_STICK_X_INV = data.get("left_stick_x_inverted", False)
                cls.L_STICK_Y_INV = data.get("left_stick_y_inverted", False)
                
                cls.R_STICK_X = data.get("right_stick_x", cls.R_STICK_X)
                cls.R_STICK_Y = data.get("right_stick_y", cls.R_STICK_Y)
                cls.R_STICK_X_INV = data.get("right_stick_x_inverted", False)
                cls.R_STICK_Y_INV = data.get("right_stick_y_inverted", False)
                
                cls.L2_TYPE = data.get("l2_type", cls.L2_TYPE)
                cls.L2_ID = data.get("l2_id", cls.L2_ID)
                cls.L2_BASELINE = data.get("l2_baseline", cls.L2_BASELINE)
                
                cls.R2_TYPE = data.get("r2_type", cls.R2_TYPE)
                cls.R2_ID = data.get("r2_id", cls.R2_ID)
                cls.R2_BASELINE = data.get("r2_baseline", cls.R2_BASELINE)
                
                cls.L3 = data.get("l3", cls.L3)
                cls.R3 = data.get("r3", cls.R3)
                cls.L1 = data.get("l1", cls.L1)
                cls.R1 = data.get("r1", cls.R1)
                
                cls.A = data.get("a", cls.A)
                cls.B = data.get("b", cls.B)
                cls.X = data.get("x", cls.X)
                cls.Y = data.get("y", cls.Y)
                
                cls.START = data.get("start", cls.START)
                cls.SELECT = data.get("select", cls.SELECT)
                
                cls.DPAD_UP = data.get("dpad_up", cls.DPAD_UP)
                cls.DPAD_DOWN = data.get("dpad_down", cls.DPAD_DOWN)
                cls.DPAD_LEFT = data.get("dpad_left", cls.DPAD_LEFT)
                cls.DPAD_RIGHT = data.get("dpad_right", cls.DPAD_RIGHT)
                
                print(f"  Loaded gamepad config from {GAMEPAD_CONFIG_FILE}")
                return True
            except Exception as e:
                print(f"  Warning: Could not load gamepad config: {e}")
                return False
        return False
    
    @staticmethod
    def apply_deadzone(value: float, deadzone: float = 0.25) -> float:
        """Apply deadzone to analog stick value."""
        if abs(value) < deadzone:
            return 0.0
        sign = 1 if value > 0 else -1
        return sign * (abs(value) - deadzone) / (1.0 - deadzone)


class GamepadManager:
    """Manages gamepad input with hotplug support."""
    
    def __init__(self):
        self.gamepad = None
        self.name = "None"
        self.debug_mode = False
        self.debug_cooldown = 0
        self.config_loaded = False
        pygame.joystick.init()
        self._detect_gamepad()
    
    def _detect_gamepad(self):
        """Detect and initialize first available gamepad."""
        pygame.joystick.quit()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count > 0:
            self.gamepad = pygame.joystick.Joystick(0)
            self.gamepad.init()
            self.name = self.gamepad.get_name()
            print(f"  Gamepad detected: {self.name}")
            print(f"    Axes: {self.gamepad.get_numaxes()}")
            print(f"    Buttons: {self.gamepad.get_numbuttons()}")
            
            # Try to load calibration
            self.config_loaded = GamepadConfig.load_from_file()
            if not self.config_loaded:
                print("    Run 'python calibrate_gamepad.py' to calibrate!")
            
            # Print axis info
            num_axes = self.gamepad.get_numaxes()
            print(f"    Axes: {num_axes} | L-Stick: {GamepadConfig.L_STICK_X},{GamepadConfig.L_STICK_Y} | R-Stick: {GamepadConfig.R_STICK_X},{GamepadConfig.R_STICK_Y}")
            # Print raw axis values for diagnostics
            if num_axes > 0:
                raw_vals = [f"{i}:{self.gamepad.get_axis(i):.2f}" for i in range(min(num_axes, 6))]
                print(f"    Raw axis values: {' '.join(raw_vals)}")
            print("    Press G to toggle gamepad debug mode")
            return True
        else:
            if self.gamepad is None:
                print("  No gamepad detected (keyboard only)")
            self.gamepad = None
            self.name = "None"
            return False
    
    def check_hotplug(self):
        """Check for newly connected gamepads (call periodically)."""
        if self.gamepad is None:
            found = self._detect_gamepad()
            if found:
                print(f"  [Gamepad] Hotplug detected: {self.name}")
            return found
        return True
    
    def is_connected(self) -> bool:
        return self.gamepad is not None
    
    def get_axis_raw(self, axis: int) -> float:
        """Get raw axis value without deadzone."""
        if not self.gamepad:
            return 0.0
        if axis >= self.gamepad.get_numaxes():
            return 0.0
        return self.gamepad.get_axis(axis)
    
    def get_axis(self, axis: int, inverted: bool = False, deadzone: float = 0.25) -> float:
        """Get axis value with deadzone applied and optional inversion."""
        if not self.gamepad:
            return 0.0
        if axis >= self.gamepad.get_numaxes():
            return 0.0
        value = self.gamepad.get_axis(axis)
        if inverted:
            value = -value
        return GamepadConfig.apply_deadzone(value, deadzone)
    
    def get_button(self, button: int) -> bool:
        """Get button state."""
        if not self.gamepad:
            return False
        if button >= self.gamepad.get_numbuttons():
            return False
        return self.gamepad.get_button(button)
    
    def get_movement(self) -> tuple:
        """Get movement from left stick (forward, right)."""
        if not self.gamepad:
            return (0.0, 0.0)
        x = self.get_axis(GamepadConfig.L_STICK_X, GamepadConfig.L_STICK_X_INV)
        y = self.get_axis(GamepadConfig.L_STICK_Y, GamepadConfig.L_STICK_Y_INV)
        # Forward is -Y on stick (up = negative Y), right is +X
        return (-y, x)
    
    def get_look(self) -> tuple:
        """Get look from right stick (yaw, pitch)."""
        # Validate axes exist
        if not self.gamepad:
            return (0.0, 0.0)
        num_axes = self.gamepad.get_numaxes()
        if GamepadConfig.R_STICK_X >= num_axes or GamepadConfig.R_STICK_Y >= num_axes:
            return (0.0, 0.0)
        
        # Get raw values and apply baseline correction
        raw_x = self.get_axis_raw(GamepadConfig.R_STICK_X)
        raw_y = self.get_axis_raw(GamepadConfig.R_STICK_Y)
        
        # Detect if axis has a stuck value (trigger or misconfigured axis)
        # Real sticks center near 0, triggers rest at -1 or 1
        # Also check for stuck intermediate values (drift)
        if not hasattr(self, '_r_stick_baseline'):
            # Record baseline on first call (controller at rest)
            self._r_stick_baseline = (raw_x, raw_y)
            if abs(raw_x) > 0.5 or abs(raw_y) > 0.5:
                print(f"  [GAMEPAD] Warning: Right stick axes have unusual resting values: X={raw_x:.2f} Y={raw_y:.2f}")
                print(f"  [GAMEPAD] This may indicate incorrect axis mapping. Press G for debug, or reconfigure controller.")
        
        # Subtract baseline (handles triggers mapped as stick)
        baseline_x, baseline_y = self._r_stick_baseline
        
        # If baseline is extreme (likely a trigger), disable this axis
        if abs(baseline_x) > 0.8:
            adjusted_x = 0.0
        else:
            adjusted_x = raw_x - baseline_x
        
        if abs(baseline_y) > 0.8:
            adjusted_y = 0.0
        else:
            adjusted_y = raw_y - baseline_y
        
        # Apply deadzone to adjusted values
        x = GamepadConfig.apply_deadzone(adjusted_x, 0.25)
        y = GamepadConfig.apply_deadzone(adjusted_y, 0.25)
        
        if GamepadConfig.R_STICK_X_INV:
            x = -x
        if GamepadConfig.R_STICK_Y_INV:
            y = -y
        
        # yaw from horizontal (X), pitch from vertical (Y)
        # Invert X so pushing stick right = look right (standard)
        # Invert Y so pushing stick up = look up (positive pitch)
        return (-x, -y)
    
    def get_vertical(self) -> float:
        """Get vertical movement from L3/R3 buttons."""
        up = 1.0 if self.get_button(GamepadConfig.R3) else 0.0
        down = -1.0 if self.get_button(GamepadConfig.L3) else 0.0
        return up + down
    
    def get_trigger(self, trigger_type: str, trigger_id: int, baseline: float) -> float:
        """Get trigger value normalized to 0-1."""
        if trigger_type == "button":
            return 1.0 if self.get_button(trigger_id) else 0.0
        else:
            raw = self.get_axis_raw(trigger_id)
            # Normalize based on baseline
            # If baseline is -1, range is -1 to 1, so normalize to 0-1
            if baseline < -0.5:
                return (raw + 1.0) / 2.0
            else:
                # Baseline is ~0, so just use raw value clamped
                return max(0.0, raw)
    
    def get_triggers(self) -> tuple:
        """Get L2/R2 trigger values (normalized 0-1)."""
        l2 = self.get_trigger(GamepadConfig.L2_TYPE, GamepadConfig.L2_ID, GamepadConfig.L2_BASELINE)
        r2 = self.get_trigger(GamepadConfig.R2_TYPE, GamepadConfig.R2_ID, GamepadConfig.R2_BASELINE)
        return (l2, r2)
    
    def get_dpad(self, dpad_config: dict) -> bool:
        """Check if a D-pad direction is pressed."""
        if not self.gamepad:
            return False
        
        if dpad_config.get("type") == "hat":
            hat_id = dpad_config.get("hat", 0)
            direction = dpad_config.get("direction", "")
            if hat_id >= self.gamepad.get_numhats():
                return False
            hat = self.gamepad.get_hat(hat_id)
            if direction == "up":
                return hat[1] == 1
            elif direction == "down":
                return hat[1] == -1
            elif direction == "left":
                return hat[0] == -1
            elif direction == "right":
                return hat[0] == 1
        elif dpad_config.get("type") == "button":
            return self.get_button(dpad_config.get("button", -1))
        
        return False
    
    def print_debug(self):
        """Print all axis and button values for debugging."""
        if not self.gamepad or self.debug_cooldown > 0:
            return
        self.debug_cooldown = 30  # Print every 0.5 seconds
        
        axes = []
        for i in range(self.gamepad.get_numaxes()):
            axes.append(f"{i}:{self.get_axis_raw(i):+.2f}")
        
        buttons = []
        for i in range(min(16, self.gamepad.get_numbuttons())):
            if self.get_button(i):
                buttons.append(str(i))
        
        print(f"  Axes: {' '.join(axes)} | Buttons: {','.join(buttons) if buttons else 'none'}")


def save_position(camera, seed: int):
    """Save current position to config file."""
    data = {
        "seed": seed,
        "x": camera.x,
        "y": camera.y,
        "z": camera.z,
        "yaw": camera.yaw,
        "pitch": camera.pitch,
        "flying": camera.flying,
        "speed_level": camera.speed_level,
    }
    try:
        with open(SAVE_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Position saved to {SAVE_FILE}")
    except Exception as e:
        print(f"  Error saving: {e}")


def request_screenshot(camera):
    """Request a screenshot - will be captured on next frame without status text."""
    if camera:
        camera.screenshot_pending = True
        # Hide status temporarily so it doesn't appear in screenshot
        camera._saved_status = camera.status_message
        camera._saved_timer = camera.status_timer
        camera.status_message = ""
        camera.status_timer = 0
        print("  Screenshot requested (capturing next frame...)")


def capture_pending_screenshot(camera):
    """Actually capture the screenshot if one is pending. Call AFTER rendering."""
    if not camera or not camera.screenshot_pending:
        return None
    
    camera.screenshot_pending = False
    
    import os
    from datetime import datetime
    
    # Use waverse screenshots directory
    screenshots_dir = SCREENSHOTS_DIR
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(screenshots_dir, f"screenshot_{timestamp}.png")
    
    # Get the current display size
    display = pygame.display.get_surface()
    width, height = display.get_size()
    
    # Read pixels from OpenGL
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    
    # Convert to pygame surface (OpenGL gives us bottom-up, so flip)
    surface = pygame.Surface((width, height))
    raw = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
    # Flip vertically (OpenGL origin is bottom-left)
    raw = np.flipud(raw)
    # Convert RGB to pygame surface
    pygame.surfarray.blit_array(surface, np.transpose(raw, (1, 0, 2)))
    
    # Save the screenshot
    pygame.image.save(surface, filename)
    print(f"  [Screenshot] Saved: {filename}")
    
    # Set status message
    camera.set_status(f"SAVED: {filename}", 3.0)
    
    return filename


def take_screenshot(camera=None):
    """Request a screenshot - will be captured on next frame without status text."""
    if camera:
        request_screenshot(camera)
    else:
        # Fallback for no camera - immediate capture (legacy)
        import os
        from datetime import datetime
        
        screenshots_dir = SCREENSHOTS_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(screenshots_dir, f"screenshot_{timestamp}.png")
        
        display = pygame.display.get_surface()
        width, height = display.get_size()
        
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        
        surface = pygame.Surface((width, height))
        raw = np.frombuffer(pixels, dtype=np.uint8).reshape((height, width, 3))
        raw = np.flipud(raw)
        pygame.surfarray.blit_array(surface, np.transpose(raw, (1, 0, 2)))
        
        pygame.image.save(surface, filename)
        print(f"  [Screenshot] Saved: {filename}")
        return filename


def render_entity_to_png(entity, entity_type: str, filename: str, size: int = 512):
    """Render a plant or animal to a PNG with transparent background.
    
    Note: This uses legacy OpenGL and will not work on macOS Core profile.
    On macOS, PNG rendering is skipped.
    """
    import platform
    from waverse.flora import PlantRenderer, PlantInstance
    from waverse.animals import AnimalRenderer
    
    # Skip PNG rendering on macOS Core profile - legacy GL calls don't work
    # On macOS, if ModernGL is available we're using Core profile which doesn't support legacy GL
    if platform.system() == 'Darwin' and MODERNGL_AVAILABLE:
        print(f"  [DNA Logger] PNG rendering skipped (macOS Core profile)")
        return
    
    # Save current OpenGL state
    viewport = glGetIntegerv(GL_VIEWPORT)
    
    try:
        # Disable fog for clean render (may fail in Core profile, that's ok)
        try:
            glDisable(GL_FOG)
        except Exception:
            pass  # Core profile doesn't have fog
        
        # Set up a square viewport for rendering
        glViewport(0, 0, size, size)
        
        # Clear to transparent background
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up orthographic projection
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        
        # Determine entity size for proper framing
        if entity_type == "plant":
            entity_height = entity.dna.height_gene.value * entity.scale
            entity_width = max(entity.dna.width_gene.value * entity.scale * 2, entity_height * 0.5)
            # Plants grow UP from ground, so center is at height/2
            center_y = entity_height * 0.4
            # Bias view upward for plants
            view_bottom_mult = 0.3
            view_top_mult = 1.7
        else:
            # Animals - estimate size from body segments
            entity_height = 3.0
            entity_width = 3.0
            if hasattr(entity.dna, 'body_segments') and entity.dna.body_segments:
                # size is a tuple (width, height, depth), sum the max dimensions
                total_size = sum(max(seg.size) if isinstance(seg.size, tuple) else seg.size 
                               for seg in entity.dna.body_segments)
                entity_height = total_size * 2.5  # More generous height estimate
                entity_width = total_size * 3.0
            # Animals body is mostly above origin, legs go down
            center_y = entity_height * 0.3  # Look at middle of body (above legs)
            # View needs more room below for legs/tentacles
            view_bottom_mult = 1.8  # Much more room below
            view_top_mult = 0.8    # Less room above (body doesn't extend much up)
        
        # Add padding - make view big enough
        view_size = max(entity_height, entity_width, 5.0) * 2.0
        half_size = view_size / 2
        
        # Orthographic projection - different for plants vs animals
        glOrtho(-half_size, half_size, -half_size * view_bottom_mult, half_size * view_top_mult, -100, 100)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Look at entity from the front-right, slightly above
        gluLookAt(
            half_size * 0.7, half_size * 0.4, half_size * 0.7,  # Eye position
            0, center_y, 0,  # Look at center of entity (different for plants vs animals)
            0, 1, 0   # Up vector
        )
        
        # Simple lighting
        glDisable(GL_LIGHTING)  # Use simple colors for now
        
        # Render the entity at origin
        if entity_type == "plant":
            # Create a temporary plant instance at origin
            temp_plant = PlantInstance(
                x=0, y=0, z=0,
                dna=entity.dna,
                scale=entity.scale,
                rotation=45  # Rotate a bit for better view
            )
            PlantRenderer.draw_full(temp_plant)
        else:
            # Render animal
            AnimalRenderer.draw_full(entity, at_origin=True)
        
        glFlush()  # Make sure rendering is complete
        
        # Read pixels with alpha
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(0, 0, size, size, GL_RGBA, GL_UNSIGNED_BYTE)
        
        # Convert to numpy array and flip vertically
        raw = np.frombuffer(pixels, dtype=np.uint8).reshape((size, size, 4))
        raw = np.flipud(raw)
        
        # Create pygame surface with alpha and copy pixels
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        for y in range(size):
            for x in range(size):
                r, g, b, a = raw[y, x]
                # Make black pixels transparent
                if r == 0 and g == 0 and b == 0:
                    a = 0
                surface.set_at((x, y), (r, g, b, a))
        
        # Save as PNG (preserves transparency)
        pygame.image.save(surface, filename)
        
        # Restore matrix state
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        
    finally:
        # ALWAYS restore viewport to prevent rendering issues
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3])
        
        # Restore fog and clear color (may fail in Core profile, that's ok)
        try:
            glEnable(GL_FOG)
        except Exception:
            pass  # Core profile doesn't have fog
        glClearColor(0.5, 0.7, 1.0, 1.0)


def log_dna_at_cursor(camera, flora_manager, animal_manager):
    """Find and log the DNA of the nearest plant or animal at the cursor (camera look direction)."""
    import os
    from datetime import datetime
    from dataclasses import asdict
    
    # Get camera look direction - use Camera's method for consistency
    look_x, look_y, look_z = camera.get_forward_vector()
    
    # Search distance and hit radius
    max_dist = 100.0  # Max distance to search
    hit_radius = 5.0  # How close the ray must pass to the entity center (increased for easier selection)
    
    best_entity = None
    best_type = None
    best_ray_dist = float('inf')  # Distance along ray to closest point
    best_perp_dist = float('inf')  # Perpendicular distance from ray
    
    def check_entity(entity, entity_type):
        """Check if ray passes close to entity. Returns (ray_dist, perp_dist) or None."""
        nonlocal best_entity, best_type, best_ray_dist, best_perp_dist
        
        # Vector from camera to entity
        dx = entity.x - camera.x
        dy = entity.y - camera.y
        dz = entity.z - camera.z
        
        # Project onto look direction (distance along ray)
        ray_dist = dx * look_x + dy * look_y + dz * look_z
        
        # Must be in front of camera and within range
        if ray_dist < 0 or ray_dist > max_dist:
            return
        
        # Closest point on ray to entity
        closest_x = camera.x + look_x * ray_dist
        closest_y = camera.y + look_y * ray_dist
        closest_z = camera.z + look_z * ray_dist
        
        # Perpendicular distance from ray to entity
        perp_dx = entity.x - closest_x
        perp_dy = entity.y - closest_y
        perp_dz = entity.z - closest_z
        perp_dist = math.sqrt(perp_dx*perp_dx + perp_dy*perp_dy + perp_dz*perp_dz)
        
        # Check if within hit radius and closer than current best
        if perp_dist < hit_radius:
            # Prioritize by perpendicular distance first (most accurate hit), then ray distance
            if perp_dist < best_perp_dist or (perp_dist < best_perp_dist + 0.5 and ray_dist < best_ray_dist):
                best_entity = entity
                best_type = entity_type
                best_ray_dist = ray_dist
                best_perp_dist = perp_dist
    
    # Search in nearby chunks
    cam_cx = int(camera.x // 32)
    cam_cz = int(camera.z // 32)
    
    for dcx in range(-2, 3):
        for dcz in range(-2, 3):
            cx, cz = cam_cx + dcx, cam_cz + dcz
            
            # Check plants
            if (cx, cz) in flora_manager.chunk_plants:
                for plant in flora_manager.chunk_plants[(cx, cz)]:
                    check_entity(plant, "plant")
            
            # Check animals
            if (cx, cz) in animal_manager.chunk_animals:
                for animal in animal_manager.chunk_animals[(cx, cz)]:
                    check_entity(animal, "animal")
    
    if best_entity is None:
        print("  [DNA Logger] No plant or animal found at cursor (aim at something within 100 units)")
        return None
    
    # Create DNA log directory
    dna_log_dir = DNA_LOGS_DIR
    os.makedirs(dna_log_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(dna_log_dir, f"{best_type}_dna_{timestamp}.json")
    
    # Convert DNA to dict and save with location info
    try:
        dna_dict = asdict(best_entity.dna)
        
        # Calculate chunk ID from entity position
        entity_cx = int(best_entity.x // 32)
        entity_cz = int(best_entity.z // 32)
        
        # Create full log entry with location context
        log_entry = {
            "entity_type": best_type,
            "position": {
                "x": round(best_entity.x, 2),
                "y": round(best_entity.y, 2),
                "z": round(best_entity.z, 2)
            },
            "chunk": {
                "cx": entity_cx,
                "cz": entity_cz,
                "id": f"({entity_cx}, {entity_cz})"
            },
            "camera_position": {
                "x": round(camera.x, 2),
                "y": round(camera.y, 2),
                "z": round(camera.z, 2)
            },
            "distance": round(best_ray_dist, 2),
            "dna": dna_dict
        }
        
        dna_string = json.dumps(log_entry, indent=2, default=str)
        
        with open(filename, "w") as f:
            f.write(dna_string)
        
        # Get type name
        if best_type == "plant":
            type_name = best_entity.dna.plant_type
        else:
            type_name = best_entity.dna.animal_type
        
        # Render the entity to a PNG with transparent background
        png_filename = filename.replace(".json", ".png")
        render_entity_to_png(best_entity, best_type, png_filename)
        
        # Print full info
        print(f"\n  ========== LOG {best_type.upper()} DNA ==========")
        print(f"  Type: {type_name}")
        print(f"  Position: ({best_entity.x:.1f}, {best_entity.y:.1f}, {best_entity.z:.1f})")
        print(f"  Chunk: ({entity_cx}, {entity_cz})")
        print(f"  Distance: {best_ray_dist:.1f} units")
        print(f"  Accuracy: {best_perp_dist:.2f} (0 = perfect aim)")
        print(f"  Saved: {filename}")
        print(f"  Image: {png_filename}")
        print(f"  ----------------------------------------")
        # Print compact DNA string (single line, truncated)
        compact_dna = json.dumps(dna_dict, default=str)
        if len(compact_dna) > 300:
            print(f"  DNA: {compact_dna[:300]}...")
        else:
            print(f"  DNA: {compact_dna}")
        print(f"  ==========================================\n")
        
        # Set status message
        camera.set_status(f"SCANNED {best_type.upper()} DNA - saved to log", 3.0)
        
        return filename
    except Exception as e:
        print(f"  Error logging DNA: {e}")
        return None


def cut_tree_at_cursor(camera, flora_manager, chunk_manager, dropped_items_list, debris_list=None, modern_renderer=None, fallen_trees_list=None):
    """Cut down the nearest tree at the cursor (camera look direction).
    
    Creates dropped items (wood) that can be picked up.
    Spawns debris particles for visual effect (if debris_list provided).
    Spawns a fallen tree trunk that tips over (if fallen_trees_list provided).
    Returns the number of items created, or 0 if no tree found.
    """
    # Get camera look direction
    look_x, look_y, look_z = camera.get_forward_vector()
    
    # Search for nearest tree/plant in look direction
    max_dist = 15.0  # Max cutting distance
    hit_radius = 3.0  # How close the ray must pass
    
    best_plant = None
    best_plant_chunk = None
    best_dist = float('inf')
    
    # Search plants in nearby chunks
    cam_cx = int(camera.x // 32)
    cam_cz = int(camera.z // 32)
    
    for dcx in range(-1, 2):
        for dcz in range(-1, 2):
            cx, cz = cam_cx + dcx, cam_cz + dcz
            chunk_key = (cx, cz)
            
            plants = flora_manager.chunk_plants.get(chunk_key, [])
            
            for plant in plants:
                # Vector from camera to plant
                dx = plant.x - camera.x
                dy = plant.y - camera.y
                dz = plant.z - camera.z
                
                # Project onto look direction
                ray_dist = dx * look_x + dy * look_y + dz * look_z
                
                if ray_dist < 1.0 or ray_dist > max_dist:
                    continue
                
                # Closest point on ray to plant
                closest_x = camera.x + look_x * ray_dist
                closest_y = camera.y + look_y * ray_dist
                closest_z = camera.z + look_z * ray_dist
                
                # Perpendicular distance
                perp_dist = math.sqrt(
                    (plant.x - closest_x)**2 + 
                    (plant.y - closest_y)**2 + 
                    (plant.z - closest_z)**2
                )
                
                if perp_dist < hit_radius and ray_dist < best_dist:
                    best_plant = plant
                    best_plant_chunk = chunk_key
                    best_dist = ray_dist
    
    if best_plant is None:
        camera.set_status("No tree in range", 1.5)
        return 0
    
    # Determine wood yield based on plant size/type
    dna = best_plant.dna
    plant_height = getattr(dna, 'height_gene', None)
    if plant_height and hasattr(plant_height, 'value'):
        height = plant_height.value
    else:
        height = 1.0
    
    # Get plant colors for debris
    trunk_color = (0.55, 0.35, 0.15)  # Default brown
    leaf_color = (0.2, 0.6, 0.2)  # Default green
    
    # Try to extract colors from DNA
    if hasattr(dna, 'trunk_color') and dna.trunk_color:
        tc = dna.trunk_color
        if hasattr(tc, 'r'):
            trunk_color = (tc.r, tc.g, tc.b)
        elif isinstance(tc, (list, tuple)) and len(tc) >= 3:
            trunk_color = (tc[0], tc[1], tc[2])
    
    if hasattr(dna, 'leaf_color') and dna.leaf_color:
        lc = dna.leaf_color
        if hasattr(lc, 'r'):
            leaf_color = (lc.r, lc.g, lc.b)
        elif isinstance(lc, (list, tuple)) and len(lc) >= 3:
            leaf_color = (lc[0], lc[1], lc[2])
    
    # Spawn debris particles (visual effect)
    if debris_list is not None:
        num_debris = int(8 + height * 4)  # More debris for bigger trees
        for i in range(num_debris):
            # Spawn throughout the tree volume
            debris_y = best_plant.y + np.random.random() * height
            debris_x = best_plant.x + (np.random.random() - 0.5) * 2.0
            debris_z = best_plant.z + (np.random.random() - 0.5) * 2.0
            
            # Mix of trunk and leaf debris
            if np.random.random() < 0.3:
                color = trunk_color
                dtype = "wood"
            else:
                color = leaf_color
                dtype = "leaf"
            
            debris = Debris(debris_x, debris_y, debris_z, dtype, color)
            debris_list.append(debris)
    
    # Bigger trees give more wood (1-5 logs)
    wood_count = min(5, max(1, int(height * 1.5)))
    
    # Add wood directly to inventory (simpler UX for now)
    wood_item = Item("wood", "Wood", count=wood_count, max_stack=99,
                    description="Wood from a tree. Used for crafting.")
    camera.add_item(wood_item)
    print(f"        Added {wood_count} wood to inventory. Inventory now has {len(camera.inventory)} items.")
    
    # Create fallen tree trunk effect
    if fallen_trees_list is not None:
        fallen = FallenTree(
            best_plant.x,
            best_plant.y,
            best_plant.z,
            height=height,
            trunk_color=trunk_color
        )
        fallen_trees_list.append(fallen)
    
    # Remove the plant from flora manager
    if best_plant_chunk in flora_manager.chunk_plants:
        flora_manager.chunk_plants[best_plant_chunk] = [
            p for p in flora_manager.chunk_plants[best_plant_chunk] 
            if p is not best_plant
        ]
        # Invalidate display list so it re-renders (legacy)
        if best_plant_chunk in flora_manager.display_lists:
            del flora_manager.display_lists[best_plant_chunk]
        # Invalidate modern renderer flora cache
        if modern_renderer and hasattr(modern_renderer, 'flora'):
            cx, cz = best_plant_chunk
            if (cx, cz) in modern_renderer.loaded_flora_chunks:
                modern_renderer.loaded_flora_chunks.discard((cx, cz))
                # Clear the batch for this chunk type
                modern_renderer.flora.clear_instances()
    
    plant_type = getattr(dna, 'plant_type', 'plant')
    if hasattr(plant_type, 'name'):
        plant_type = plant_type.name
    
    camera.set_status(f"CUT {plant_type} - {wood_count} wood!", 2.0)
    print(f"  [CUT] Cut {plant_type} at ({best_plant.x:.1f}, {best_plant.z:.1f}) - {wood_count} logs")
    print(f"        Dropped items: {len(dropped_items_list)}, Debris: {len(debris_list) if debris_list else 0}")
    
    return wood_count


def pickup_nearby_items(camera, dropped_items_list, pickup_radius: float = 3.0):
    """Pick up dropped items near the camera and add to inventory.
    
    Returns dict of item_type -> count picked up.
    """
    picked_up = {}
    remaining = []
    
    for dropped in dropped_items_list:
        dist = dropped.distance_to(camera.x, camera.y, camera.z)
        if dist < pickup_radius and dropped.on_ground:
            item = dropped.item
            # Add to inventory
            camera.add_item(Item(item.item_type, item.name, item.count, 
                                item.max_stack, item.description, item.properties.copy()))
            # Track what we picked up
            picked_up[item.item_type] = picked_up.get(item.item_type, 0) + item.count
        else:
            remaining.append(dropped)
    
    dropped_items_list.clear()
    dropped_items_list.extend(remaining)
    
    if picked_up:
        items_str = ", ".join(f"{count} {itype}" for itype, count in picked_up.items())
        camera.set_status(f"Picked up {items_str}!", 1.5)
        print(f"  [PICKUP] Collected: {picked_up}")
    
    return picked_up


def get_terrain_type_at_cursor(camera, chunk_manager, flora_manager):
    """Determine what terrain type is at the camera's cursor (look direction).
    
    Returns: "water", "plant", or "ground"
    """
    # Get camera look direction
    look_x, look_y, look_z = camera.get_forward_vector()
    
    # Check if we hit a plant first
    max_dist = 15.0
    hit_radius = 3.0
    
    cam_cx = int(camera.x // 32)
    cam_cz = int(camera.z // 32)
    
    for dcx in range(-1, 2):
        for dcz in range(-1, 2):
            cx, cz = cam_cx + dcx, cam_cz + dcz
            chunk_key = (cx, cz)
            plants = flora_manager.chunk_plants.get(chunk_key, [])
            
            for plant in plants:
                dx = plant.x - camera.x
                dy = plant.y - camera.y
                dz = plant.z - camera.z
                ray_dist = dx * look_x + dy * look_y + dz * look_z
                
                if ray_dist < 1.0 or ray_dist > max_dist:
                    continue
                
                closest_x = camera.x + look_x * ray_dist
                closest_y = camera.y + look_y * ray_dist
                closest_z = camera.z + look_z * ray_dist
                
                perp_dist = math.sqrt(
                    (plant.x - closest_x)**2 + 
                    (plant.y - closest_y)**2 + 
                    (plant.z - closest_z)**2
                )
                
                if perp_dist < hit_radius:
                    return "plant"
    
    # Check terrain at look point
    # Find where the look ray hits terrain
    check_dist = 5.0
    check_x = camera.x + look_x * check_dist
    check_z = camera.z + look_z * check_dist
    
    # Get terrain height at that point using ChunkManager
    terrain_height = chunk_manager.get_height_at(check_x, check_z)
    
    # Check if water (terrain height at or below sea level)
    SEA_LEVEL = 0.0  # Adjust if your sea level is different
    if terrain_height <= SEA_LEVEL:
        return "water"
    
    return "ground"


def populate_microscope_world(world, terrain_type: str):
    """Populate a MicroscopeWorld with organisms based on terrain type.
    
    Args:
        world: MicroscopeWorld instance to populate
        terrain_type: "water", "plant", or "ground"
    """
    import random
    
    # Get terrain-specific population from micro_life.py
    population = TERRAIN_POPULATIONS.get(terrain_type, TERRAIN_POPULATIONS["ground"])
    
    # Add organisms based on population counts
    total = 0
    for template_name, count in population.items():
        template_func = MICRO_TEMPLATES.get(template_name)
        if template_func:
            for i in range(count):
                # Pass unique seed for variety
                dna = template_func(seed=random.randint(0, 999999))
                x = world.rng.random() * world.width
                y = world.rng.random() * world.height
                world.add_organism(x, y, dna)
                total += 1
    
    # Add some floating debris/particles (non-living)
    num_debris = random.randint(20, 40)
    for _ in range(num_debris):
        # Create tiny static "organic debris" particles
        debris_dna = MicroDNA(
            species_id=random.randint(0, 999999),
            base_size=0.05 + random.random() * 0.15,
            membrane=micro_life.MembranGene(
                shape="circle",
                color=(0.6 + random.random() * 0.2,
                       0.55 + random.random() * 0.2,
                       0.4 + random.random() * 0.2),
                transparency=0.5 + random.random() * 0.3,
            ),
            organelles=[],  # No internal structure - just debris
            movement=micro_life.MovementGene(
                pattern=micro_life.MovementPattern.DRIFT,
                speed=0.02,
            ),
        )
        x = world.rng.random() * world.width
        y = world.rng.random() * world.height
        world.add_organism(x, y, debris_dna)
    
    print(f"  [MICRO] Populated world with {total} organisms + {num_debris} debris for {terrain_type}")


def scan_microorganism_at_cursor(camera, modern_renderer):
    """Scan and log the DNA of a microorganism at the microscope cursor.
    
    Similar to scanning plants/animals but for microorganisms.
    """
    import json
    import os
    from datetime import datetime
    
    if not modern_renderer or not hasattr(modern_renderer, 'microscope_view'):
        camera.set_status("No microscope view", 1.5)
        return None
    
    microscope = modern_renderer.microscope_view
    org = microscope.get_organism_at_cursor()
    
    if org is None:
        camera.set_status("No organism at cursor", 1.5)
        return None
    
    dna = org.dna
    
    # Build DNA data for saving
    dna_data = {
        "type": "microorganism",
        "species_id": dna.species_id,
        "generation": dna.generation,
        "base_size": float(dna.base_size),
        "terrain": camera.microscope_terrain_type,
        "membrane": {
            "shape": dna.membrane.shape,
            "color": list(dna.membrane.color),
            "has_wall": dna.membrane.has_wall,
            "transparency": float(dna.membrane.transparency),
        },
        "organelles": [
            {
                "type": o.organelle_type.name,
                "count": int(o.count),
                "size": float(o.size),
                "color": list(o.color),
            }
            for o in dna.organelles
        ],
        "movement": {
            "pattern": dna.movement.pattern.name,
            "speed": float(dna.movement.speed),
        },
        "metabolism": {
            "is_predator": dna.metabolism.is_predator,
            "is_photosynthetic": dna.metabolism.is_photosynthetic,
        },
        "glow": float(dna.glow),
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save to ~/.waverse/dna_logs/
    log_dir = os.path.expanduser("~/.waverse/dna_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"micro_dna_{timestamp}.json"
    filepath = os.path.join(log_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(dna_data, f, indent=2)
        
        # Get organism "name" based on movement pattern and features
        org_type = "Microbe"
        if dna.metabolism.is_photosynthetic:
            org_type = "Algae"
        elif dna.metabolism.is_predator:
            org_type = "Predator"
        elif dna.membrane.has_wall:
            org_type = "Bacteria"
        elif dna.membrane.shape == "amoeba":
            org_type = "Amoeba"
        
        camera.set_status(f"SCANNED: {org_type} (species {dna.species_id})", 2.5)
        print(f"  [MICRO SCAN] Logged {org_type} DNA to {filename}")
        
        # Select the scanned organism for visual feedback
        microscope.selected = org
        
        return filepath
        
    except Exception as e:
        print(f"  [MICRO SCAN] Error saving DNA: {e}")
        camera.set_status("Error saving DNA", 1.5)
        return None


def _set_terrain_height_at_world_pos(chunk_manager, world_x, world_z, delta, modified_chunks):
    """Set terrain height at a world position, updating ALL chunks that share this vertex.
    
    Chunk boundaries share vertices - a vertex at (32*TILE_SCALE, z) is stored in both:
    - chunk(0, cz).heightmap[z, 32] (right edge)
    - chunk(1, cz).heightmap[z, 0] (left edge)
    
    We must update ALL chunks that contain this vertex to prevent rips.
    """
    # Chunk world size (32 tiles * TILE_SCALE per tile)
    chunk_world_size = CHUNK_SIZE * TILE_SCALE
    
    # Find the primary chunk this point is in (use floor for correct negative handling)
    primary_cx = int(np.floor(world_x / chunk_world_size))
    primary_cz = int(np.floor(world_z / chunk_world_size))
    
    chunk = chunk_manager.get_chunk(primary_cx, primary_cz)
    if chunk is None:
        return
    
    h, w = chunk.heightmap.shape  # Should be 33x33 (CHUNK_SIZE+1)
    
    # Calculate local position within chunk (chunk origin is at cx * chunk_world_size)
    local_x_float = (world_x - primary_cx * chunk_world_size) / TILE_SCALE
    local_z_float = (world_z - primary_cz * chunk_world_size) / TILE_SCALE
    
    # Round to nearest integer vertex
    local_x = int(round(local_x_float))
    local_z = int(round(local_z_float))
    
    # Clamp to valid range
    local_x = max(0, min(w - 1, local_x))
    local_z = max(0, min(h - 1, local_z))
    
    # Modify the primary chunk
    chunk = chunk_manager.get_chunk(primary_cx, primary_cz)
    if chunk is not None:
        chunk.heightmap[local_z, local_x] += delta
        modified_chunks.add((primary_cx, primary_cz))
        new_height = chunk.heightmap[local_z, local_x]
    else:
        return
    
    # If on left edge (local_x == 0), also update right edge of chunk to the left
    if local_x == 0:
        left_chunk = chunk_manager.get_chunk(primary_cx - 1, primary_cz)
        if left_chunk is not None:
            left_chunk.heightmap[local_z, w - 1] = new_height
            modified_chunks.add((primary_cx - 1, primary_cz))
    
    # If on right edge (local_x == w-1), also update left edge of chunk to the right
    if local_x == w - 1:
        right_chunk = chunk_manager.get_chunk(primary_cx + 1, primary_cz)
        if right_chunk is not None:
            right_chunk.heightmap[local_z, 0] = new_height
            modified_chunks.add((primary_cx + 1, primary_cz))
    
    # If on top edge (local_z == 0), also update bottom edge of chunk above
    if local_z == 0:
        top_chunk = chunk_manager.get_chunk(primary_cx, primary_cz - 1)
        if top_chunk is not None:
            top_chunk.heightmap[h - 1, local_x] = new_height
            modified_chunks.add((primary_cx, primary_cz - 1))
    
    # If on bottom edge (local_z == h-1), also update top edge of chunk below
    if local_z == h - 1:
        bottom_chunk = chunk_manager.get_chunk(primary_cx, primary_cz + 1)
        if bottom_chunk is not None:
            bottom_chunk.heightmap[0, local_x] = new_height
            modified_chunks.add((primary_cx, primary_cz + 1))
    
    # Handle corners (touch up to 4 chunks)
    if local_x == 0 and local_z == 0:
        corner = chunk_manager.get_chunk(primary_cx - 1, primary_cz - 1)
        if corner is not None:
            corner.heightmap[h - 1, w - 1] = new_height
            modified_chunks.add((primary_cx - 1, primary_cz - 1))
    
    if local_x == w - 1 and local_z == 0:
        corner = chunk_manager.get_chunk(primary_cx + 1, primary_cz - 1)
        if corner is not None:
            corner.heightmap[h - 1, 0] = new_height
            modified_chunks.add((primary_cx + 1, primary_cz - 1))
    
    if local_x == 0 and local_z == h - 1:
        corner = chunk_manager.get_chunk(primary_cx - 1, primary_cz + 1)
        if corner is not None:
            corner.heightmap[0, w - 1] = new_height
            modified_chunks.add((primary_cx - 1, primary_cz + 1))
    
    if local_x == w - 1 and local_z == h - 1:
        corner = chunk_manager.get_chunk(primary_cx + 1, primary_cz + 1)
        if corner is not None:
            corner.heightmap[0, 0] = new_height
            modified_chunks.add((primary_cx + 1, primary_cz + 1))


def modify_terrain_at_cursor(camera, chunk_manager, mode: str = "MINE", radius: int = 1, amount: float = 2.0, set_status: bool = True):
    """Modify terrain at cursor position (MINE = lower, FILL = raise) with circular brush.
    
    Args:
        camera: Camera object
        chunk_manager: ChunkManager for terrain access
        mode: "MINE" to lower terrain, "FILL" to raise terrain
        radius: Brush radius in tiles (1, 2, 4, 8)
        amount: How much to modify (default 2.0 units)
        set_status: Whether to show status message
        
    Returns:
        Set of (cx, cz) chunk keys that were modified, or None if no terrain hit
    """
    # Get look direction
    look_x, look_y, look_z = camera.get_forward_vector()
    
    # Ray march to find terrain intersection
    max_dist = 50.0
    step = 0.5
    
    for t in np.arange(step, max_dist, step):
        # Point along ray
        px = camera.x + look_x * t
        py = camera.y + look_y * t
        pz = camera.z + look_z * t
        
        # Get chunk and local coordinates
        cx = int(px // 32)
        cz = int(pz // 32)
        
        chunk = chunk_manager.get_chunk(cx, cz)
        if chunk is None:
            continue
        
        # Local coordinates within chunk
        local_x = int((px - chunk.world_x) / TILE_SCALE)
        local_z = int((pz - chunk.world_z) / TILE_SCALE)
        
        # Bounds check
        h, w = chunk.heightmap.shape
        if 0 <= local_x < w and 0 <= local_z < h:
            terrain_height = chunk.heightmap[local_z, local_x] * HEIGHT_SCALE
            
            # Check if ray is at or below terrain
            if py <= terrain_height:
                # Found intersection! Modify terrain with circular smooth falloff
                # Brush radius maps: 1->3, 2->5, 4->9, 8->17 for smoother dents
                brush_radius = radius * 2 + 1
                delta_base = (-amount if mode == "MINE" else amount) / HEIGHT_SCALE
                
                # Track all chunks that get modified
                modified_chunks = set()
                
                # Apply Gaussian-like falloff over brush area using WORLD coordinates
                for dz in range(-brush_radius, brush_radius + 1):
                    for dx in range(-brush_radius, brush_radius + 1):
                        # Calculate distance-based falloff (smooth Gaussian-like)
                        dist = math.sqrt(dx * dx + dz * dz)
                        if dist > brush_radius:
                            continue
                        
                        # Smooth falloff: 1.0 at center, 0 at edge
                        falloff = (1.0 - (dist / brush_radius)) ** 2
                        delta = delta_base * falloff
                        
                        # Calculate world position for this brush point
                        world_x = px + dx * TILE_SCALE
                        world_z = pz + dz * TILE_SCALE
                        
                        # Modify terrain at this world position (handles all chunk boundaries)
                        _set_terrain_height_at_world_pos(chunk_manager, world_x, world_z, delta, modified_chunks)
                
                action = "MINE" if mode == "MINE" else "FILL"
                print(f"  [{action}] Modified terrain at ({px:.1f}, {pz:.1f}) - {len(modified_chunks)} chunks affected")
                if set_status:
                    camera.set_status(f"{action} at ({px:.0f}, {pz:.0f})", 1.5)
                
                # Return all modified chunks so they can be re-rendered
                return modified_chunks
    
    print(f"  No terrain in range")
    return None


def load_position(seed: int) -> dict:
    """Load saved position if it exists and matches seed."""
    try:
        if os.path.exists(SAVE_FILE):
            with open(SAVE_FILE, "r") as f:
                data = json.load(f)
            if data.get("seed") == seed:
                return data
            else:
                print(f"  Save file is for different seed ({data.get('seed')}), starting fresh")
    except Exception as e:
        print(f"  Could not load save: {e}")
    return None


def warp_to_new_universe(camera, chunk_manager, chunk_renderer, flora_manager, 
                         animal_manager, chunk_dna_manager, climate_manager,
                         structure_manager, config, life_simulator=None,
                         force_biome=None):
    """
    Warp to a far-away location with completely fresh DNA AND terrain - a new waverse!
    
    Args:
        force_biome: Optional. Force warp to a specific exotic biome:
                     'psychedelic', 'hellfire', 'shadow', 'crystal', 'void'
    """
    print("=" * 60)
    if force_biome:
        print(f"  WARPING TO {force_biome.upper()} REALM...")
    else:
        print("  WARPING TO NEW WAVERSE...")
    print("=" * 60)
    
    # Pick a random far-away location (1000-5000 chunks away)
    import random as rnd
    angle = rnd.random() * 2 * math.pi
    distance = rnd.randint(1000, 5000) * 32  # In world units (chunks * chunk_size)
    
    new_x = camera.x + math.cos(angle) * distance
    new_z = camera.z + math.sin(angle) * distance
    
    # Generate a new random seed for this universe (terrain + DNA)
    new_universe_seed = rnd.randint(0, 2**31)
    
    # Change the TERRAIN seed too for completely new world shape!
    config.seed = new_universe_seed
    chunk_manager.config.seed = new_universe_seed
    
    print(f"  Distance: {distance/32:.0f} chunks")
    print(f"  New position: ({new_x:.0f}, {new_z:.0f})")
    print(f"  Universe seed: {new_universe_seed}")
    print(f"  (New terrain shape + new DNA!)")
    
    # Clear all caches to force fresh generation
    chunk_manager.chunks.clear()
    chunk_renderer.display_lists.clear()
    flora_manager.chunk_plants.clear()
    flora_manager.display_lists.clear()
    # Clear cached DNA values from old animals before clearing
    for animal in animal_manager.animals:
        for attr in ('_cached_growth', '_cached_metabolism', '_cached_type'):
            if hasattr(animal, attr):
                delattr(animal, attr)
    animal_manager.animals.clear()
    animal_manager.chunk_animals.clear()
    structure_manager.structures.clear()
    structure_manager.spawned_chunks.clear()
    
    # Reset DNA managers with new random base DNA
    # This creates completely new evolutionary starting points
    chunk_dna_manager.chunk_dna.clear()
    chunk_dna_manager.seed = new_universe_seed
    
    climate_manager.biomes.clear()
    climate_manager.regional_weather.clear()
    climate_manager.seed = new_universe_seed
    # Reset current weather to safe defaults to prevent HUD rendering bugs
    from waverse.climate import WeatherState, BiomeDNA
    climate_manager.current_weather = WeatherState()
    
    # === FORCE EXOTIC BIOME if requested ===
    if force_biome and force_biome in ('psychedelic', 'hellfire', 'shadow', 'crystal', 'void'):
        print(f"  Entering {force_biome.upper()} realm!")
        climate_manager.current_biome = BiomeDNA(
            special_biome=force_biome,
            chaos_factor=0.6 + rnd.random() * 0.4,  # High chaos!
            temperature=0.5,
            humidity=0.5
        )
        # Pre-populate surrounding chunks with this biome
        for dx in range(-5, 6):
            for dz in range(-5, 6):
                cx = int(new_x // 32) + dx
                cz = int(new_z // 32) + dz
                climate_manager.biomes[(cx, cz)] = BiomeDNA(
                    special_biome=force_biome,
                    chaos_factor=0.5 + rnd.random() * 0.5,
                    temperature=0.5 + rnd.random() * 0.2 - 0.1,
                    humidity=0.5 + rnd.random() * 0.2 - 0.1
                )
    else:
        climate_manager.current_biome = BiomeDNA()
    climate_manager.lightning_flash = 0.0
    
    # Reset life simulator and pause it briefly to let world settle
    if life_simulator:
        life_simulator.plant_life.clear()
        life_simulator.animal_life.clear()
        life_simulator.eggs.clear()
        life_simulator.pending_births.clear()
        life_simulator.pending_plants.clear()
        life_simulator.chunks_needing_refresh.clear()
        life_simulator.seed = new_universe_seed
        life_simulator.rng = random.Random(new_universe_seed)
        # Pause life sim for 3 seconds to let chunks load
        life_simulator.update_timer = -3.0
    
    # Reset flora DNA pool with new mutations
    flora_manager.dna_pool.seed = new_universe_seed
    flora_manager.dna_pool.rng = np.random.default_rng(new_universe_seed)
    flora_manager.dna_pool.chunk_dna.clear()
    flora_manager.dna_pool.templates = flora_manager.dna_pool._generate_templates()
    flora_manager.world_seed = new_universe_seed
    
    # Reset animal species templates with new seed
    animal_manager.world_seed = new_universe_seed
    animal_manager.species_templates = animal_manager._generate_species()
    
    # Teleport camera
    camera.x = new_x
    camera.z = new_z
    camera.y = 100  # Start high, will settle to terrain
    # Preserve current fly/walk mode - don't change camera.flying
    
    # Set status message
    camera.set_status(f"WARPED to ({new_x:.0f}, {new_z:.0f}) - New Waverse!", 4.0)
    
    print("  Welcome to a new waverse!")
    print("  The DNA here evolved completely independently.")
    print("=" * 60)


def height_to_color(h: float) -> tuple:
    """Convert height to terrain color - adjusted for better distribution."""
    # Scale heights more reasonably (divide by height scale factor)
    h = h / HEIGHT_SCALE
    
    if h < -8:
        return (0.03, 0.12, 0.35)  # Deep ocean
    elif h < -4:
        t = (h + 8) / 4
        return (0.03 + t*0.05, 0.12 + t*0.1, 0.35 + t*0.15)  # Ocean
    elif h < -1:
        t = (h + 4) / 3
        return (0.08 + t*0.1, 0.22 + t*0.2, 0.5 + t*0.15)  # Shallow water
    elif h < 0:
        return (0.18, 0.42, 0.65)  # Coast water
    elif h < 1:
        return (0.82, 0.76, 0.55)  # Beach/sand
    elif h < 4:
        t = (h - 1) / 3
        return (0.35 - t*0.08, 0.55 + t*0.05, 0.25 - t*0.02)  # Grassland
    elif h < 8:
        t = (h - 4) / 4
        return (0.27 - t*0.05, 0.60 - t*0.1, 0.23 - t*0.03)  # Forest
    elif h < 15:
        t = (h - 8) / 7
        return (0.22 + t*0.2, 0.50 - t*0.15, 0.20 + t*0.05)  # Hills
    elif h < 25:
        t = (h - 15) / 10
        return (0.42 + t*0.15, 0.35 + t*0.05, 0.25 + t*0.15)  # Mountain base
    elif h < 40:
        t = (h - 25) / 15
        return (0.57 + t*0.1, 0.40 + t*0.15, 0.40 + t*0.15)  # Mountain rock
    elif h < 60:
        t = (h - 40) / 20
        return (0.67 + t*0.13, 0.55 + t*0.2, 0.55 + t*0.2)  # High mountain
    else:
        t = min(1, (h - 60) / 30)
        return (0.80 + t*0.15, 0.75 + t*0.2, 0.75 + t*0.2)  # Snow caps


# Tool types
class ToolType:
    SCAN = "SCAN"   # Scan/log DNA of plants and animals
    MINE = "MINE"   # Mine/lower terrain (L1 = single point)
    FILL = "FILL"   # Fill/raise terrain (inverse of MINE)
    CUT = "CUT"     # Cut down trees for wood
    MICRO = "MICRO" # Microscope - view microscopic life
    
    ALL_TOOLS = [SCAN, MINE, FILL, CUT, MICRO]


class Item:
    """
    A generic item that can be in inventory or dropped in the world.
    
    Items can be:
    - Stackable resources (wood, stone, fiber)
    - Tools with durability (pickaxe, axe)
    - Special items (genetic sequencer, DNA samples)
    - Consumables
    """
    
    def __init__(self, 
                 item_type: str,
                 name: str = None,
                 count: int = 1,
                 max_stack: int = 99,
                 description: str = "",
                 properties: dict = None):
        self.item_type = item_type  # e.g., "wood", "tool:scan", "dna_sample"
        self.name = name or item_type.replace("_", " ").title()
        self.count = count
        self.max_stack = max_stack
        self.description = description
        self.properties = properties or {}  # Extra data (durability, dna, color, etc.)
    
    def is_stackable(self) -> bool:
        """Check if this item can stack with others of same type."""
        return self.max_stack > 1
    
    def can_stack_with(self, other: 'Item') -> bool:
        """Check if this item can stack with another item."""
        if not self.is_stackable() or not other.is_stackable():
            return False
        return self.item_type == other.item_type
    
    def add_count(self, amount: int) -> int:
        """Add to stack, returns overflow (amount that didn't fit)."""
        space = self.max_stack - self.count
        to_add = min(amount, space)
        self.count += to_add
        return amount - to_add
    
    def __repr__(self):
        return f"Item({self.item_type}, x{self.count})"


class DroppedItem:
    """An item dropped in the world that can be picked up."""
    
    def __init__(self, x: float, y: float, z: float, item: Item):
        self.x = x
        self.y = y
        self.z = z
        self.item = item  # The Item object this drop contains
        self.velocity_y = 0.0  # For falling/bouncing
        self.on_ground = False
        self.lifetime = 300.0  # Seconds before despawning
        self.rotation = np.random.random() * 360  # Random rotation
        self.scale = 0.3 + np.random.random() * 0.2  # Slight size variation
    
    def update(self, dt: float, terrain_height: float):
        """Update physics (falling, bouncing)."""
        self.lifetime -= dt
        
        if not self.on_ground:
            # Gravity
            self.velocity_y -= 20.0 * dt
            self.y += self.velocity_y * dt
            
            # Ground collision
            ground_y = terrain_height + 0.3  # Slight offset so it sits on ground
            if self.y <= ground_y:
                self.y = ground_y
                if abs(self.velocity_y) > 2.0:
                    self.velocity_y *= -0.3  # Bounce
                else:
                    self.velocity_y = 0
                    self.on_ground = True
    
    def distance_to(self, x: float, y: float, z: float) -> float:
        """Distance from a point."""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2 + (self.z - z)**2)


class Debris:
    """
    Visual-only debris particle that flies out and fades away.
    Used for tree cutting effects, explosions, etc.
    """
    
    def __init__(self, x: float, y: float, z: float, 
                 debris_type: str = "wood",
                 color: tuple = (0.55, 0.35, 0.15)):
        self.x = x
        self.y = y
        self.z = z
        self.debris_type = debris_type
        self.color = color
        
        # Random velocity (fly outward)
        angle = np.random.random() * 2 * math.pi
        speed = 3.0 + np.random.random() * 5.0
        self.vx = math.cos(angle) * speed
        self.vy = 5.0 + np.random.random() * 8.0  # Upward burst
        self.vz = math.sin(angle) * speed
        
        # Spin
        self.rotation = np.random.random() * 360
        self.spin = (np.random.random() - 0.5) * 720  # Degrees per second
        
        # Size and lifetime
        self.scale = 0.1 + np.random.random() * 0.3
        self.lifetime = 1.5 + np.random.random() * 1.0  # 1.5-2.5 seconds
        self.max_lifetime = self.lifetime
        self.alpha = 1.0
    
    def update(self, dt: float, terrain_height: float):
        """Update physics and fade."""
        self.lifetime -= dt
        
        # Fade out over time
        self.alpha = max(0, self.lifetime / self.max_lifetime)
        
        # Apply gravity
        self.vy -= 15.0 * dt
        
        # Move
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        
        # Spin
        self.rotation += self.spin * dt
        
        # Ground collision (just stop, no bounce)
        ground_y = terrain_height + 0.05
        if self.y < ground_y:
            self.y = ground_y
            self.vy = 0
            self.vx *= 0.8  # Friction
            self.vz *= 0.8
    
    def is_alive(self) -> bool:
        return self.lifetime > 0


class FallenTree:
    """
    A fallen tree trunk that tips over and lays on the ground.
    Visual effect that fades away after a while.
    """
    
    def __init__(self, x: float, y: float, z: float, 
                 height: float = 3.0,
                 trunk_color: tuple = (0.55, 0.35, 0.15)):
        self.x = x
        self.y = y
        self.z = z
        self.height = height
        self.trunk_color = trunk_color
        
        # Fall direction (random)
        self.fall_angle = np.random.random() * 360
        
        # Tilt animation
        self.tilt = 0.0  # Current tilt (0 = standing, 90 = fallen)
        self.tilt_speed = 60.0 + np.random.random() * 40.0  # Degrees per second
        self.fallen = False
        
        # Trunk dimensions
        self.radius = 0.3 + height * 0.05
        
        # Lifetime
        self.lifetime = 8.0  # Fade after 8 seconds
        self.max_lifetime = self.lifetime
        self.alpha = 1.0
    
    def update(self, dt: float, terrain_height: float):
        """Update falling animation and fade."""
        # Tipping over
        if not self.fallen:
            self.tilt += self.tilt_speed * dt
            if self.tilt >= 85:
                self.tilt = 85
                self.fallen = True
        
        # Start fading after fallen
        if self.fallen:
            self.lifetime -= dt
            # Fade out in last 2 seconds
            if self.lifetime < 2.0:
                self.alpha = max(0, self.lifetime / 2.0)
    
    def is_alive(self) -> bool:
        return self.lifetime > 0


class Camera:
    """Camera with walking, jumping, flying, and auto-fly modes."""
    
    def __init__(self):
        self.x = 0
        self.y = 40
        self.z = 0
        self.yaw = 0
        self.pitch = -20
        self.flying = False  # Start in walking mode
        self.swimming = False  # In water, swim mode (fly but capped at surface)
        self.target_y = 40
        self.speed_level = 2  # Default speed (0.75x) - index into SPEED_LEVELS
        
        # Velocity tracking for HUD display
        self.prev_x = 0
        self.prev_y = 40
        self.prev_z = 0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.velocity_z = 0.0
        
        # Jump/fall physics
        self.jumping = False
        self.falling = False  # Triggered by steep drops (cliff/building edge)
        self.jump_velocity = 0.0
        self.gravity = 20.0  # Normal gravity (used when falling)
        # Super Metroid style variable jump:
        # Super Metroid style variable jump:
        # - Holding jump = low gravity, you rise much higher and longer
        # - Release jump = high gravity immediately cuts your ascent
        # - Tap = short hop (~1 unit), Hold = full jump (~3-4 units)
        self.gravity_held = 12.0  # Moderate gravity while holding
        self.gravity_released = 60.0  # VERY high gravity when released (immediate stop)
        self.jump_strength = 4.0  # Lower initial impulse for reasonable max height
        self.jump_held = False  # Is jump button currently held?
        self.last_ground_y = 0.0  # Track previous ground height for cliff detection
        
        # Auto-fly / Tour mode: 0=off, 1=wander, 2=showcase
        self.auto_fly_mode = 0
        
        # Wander mode state - movement direction (independent of view)
        self.wander_timer = 0.0
        self.wander_direction = 0.0  # Radians - actual movement direction
        
        # Tour mode height and view
        self.auto_fly_height = 40.0  # Height above terrain (above trees)
        self.view_yaw_offset = 0.0   # Look offset from movement direction
        self.view_pitch = -5.0       # View pitch (default slight down)
        self.view_return_timer = 0.0 # Timer for returning to forward
        
        # Showcase mode state - fly to interesting items
        self.showcase_target = None       # (x, y, z, entity_type, entity)
        self.showcase_phase = 0           # 0=traveling, 1=viewing
        self.showcase_view_timer = 0.0    # Time to view current target
        self.showcase_arc_progress = 0.0  # 0.0 to 1.0 along arc
        self.showcase_start_pos = None    # (x, y, z) starting position
        self.showcase_arc_height = 0.0    # Peak height of arc
        self.showcase_view_distance = 15.0  # Distance to stay from target
        self.showcase_orbit_angle = 0.0   # Current angle orbiting target
        self.showcase_target_yaw = 0.0    # For smooth camera transitions
        self.showcase_target_pitch = 0.0
        self.showcase_explore_dir = 0.0   # Direction to bias exploration (radians)
        self.showcase_origin = None       # (x, z) starting point for outward exploration
        self.showcase_total_distance = 0.0  # Track how far we've traveled
        self.showcase_is_structure = False  # Track if current target is a building
        self.showcase_pan_direction = 1   # -1 = pan left, +1 = pan right
        self.showcase_pan_speed = 8.0     # degrees per second
        self.showcase_pan_total = 30.0    # total degrees to pan
        
        # Status message display
        self.status_message = ""
        self.status_timer = 0.0  # Seconds remaining to show message
        
        # Screenshot pending flag - when True, hide status and capture on next frame
        self.screenshot_pending = False
        
        # Tool system
        self.current_tool_index = 0  # Index into ToolType.ALL_TOOLS
        self.current_tool = ToolType.SCAN
        self.tool_radius = 1  # Radius for MINE/FILL tools (1, 2, 4, 8)
        self.tool_radius_sizes = [1, 2, 4, 8]  # Available radius sizes
        
        # Running mode
        self.is_running = False  # B button held = double speed
        
        # Markers for compass (list of (x, z, color, name) tuples)
        self.markers = []
        self.max_markers = 10
        
        # Menu state
        self.menu_open = False
        self.menu_tab = 0  # 0 = Inventory, 1 = Log, 2 = Controls
        self.log_index = 0  # Currently selected log item
        self.log_items = []  # List of logged DNA files
        self.log_filter = 0  # 0 = ALL, 1 = PLANTS, 2 = ANIMALS
        self.log_filter_names = ["ALL", "PLANTS", "ANIMALS"]
        self.inventory_index = 0  # Selected item in inventory grid
        self.inventory_cols = 2   # Columns in inventory grid
        self.inventory_count = 6  # Total inventory items (tools count)
        
        # Microscope mode
        self.microscope_active = False
        self.microscope_terrain_type = "water"  # What was sampled: water, plant, ground
        
        # Inventory: list of Item objects
        # Starts with default tools (non-stackable, always present)
        self.inventory: list = []
        self._init_default_inventory()
        
        # Scroll acceleration state
        self.scroll_hold_time = 0.0  # How long scroll direction held
        self.scroll_direction = 0    # -1 = up, 1 = down, 0 = none
        self.scroll_cooldown = 0.0   # Time until next scroll step
        self.favorites = set()  # Set of favorite log filenames
        self._load_favorites()
    
    def _init_default_inventory(self):
        """Initialize inventory with default tools."""
        self.inventory = [
            Item("tool:scan", "Scanner", max_stack=1, 
                 description="Scan plants and animals to log their DNA"),
            Item("tool:mine", "Mining Tool", max_stack=1,
                 description="Lower terrain by mining"),
            Item("tool:fill", "Fill Tool", max_stack=1,
                 description="Raise terrain by filling"),
            Item("tool:cut", "Axe", max_stack=1,
                 description="Cut trees for wood"),
        ]
    
    def add_item(self, item: 'Item') -> bool:
        """
        Add an item to inventory. Stacks with existing items if possible.
        Returns True if item was added, False if inventory full.
        """
        print(f"  [INVENTORY] Adding {item.count}x {item.name} (type: {item.item_type})")
        
        # Try to stack with existing items first
        if item.is_stackable():
            for inv_item in self.inventory:
                if inv_item.can_stack_with(item):
                    overflow = inv_item.add_count(item.count)
                    print(f"  [INVENTORY] Stacked with existing. Now: {inv_item.count}x")
                    if overflow == 0:
                        return True
                    item.count = overflow  # Continue with remainder
        
        # Add as new slot (no limit for now, could add max inventory size later)
        self.inventory.append(item)
        print(f"  [INVENTORY] Added new slot. Total items: {len(self.inventory)}")
        
        # Debug: print all non-tool items
        non_tools = [i for i in self.inventory if not i.item_type.startswith('tool:')]
        print(f"  [INVENTORY] Non-tool items: {[(i.name, i.count) for i in non_tools]}")
        return True
    
    def get_item_count(self, item_type: str) -> int:
        """Get total count of an item type in inventory."""
        total = 0
        for item in self.inventory:
            if item.item_type == item_type:
                total += item.count
        return total
    
    def remove_item(self, item_type: str, count: int = 1) -> int:
        """Remove items from inventory. Returns amount actually removed."""
        removed = 0
        to_remove = []
        
        for item in self.inventory:
            if item.item_type == item_type and removed < count:
                take = min(item.count, count - removed)
                item.count -= take
                removed += take
                if item.count <= 0:
                    to_remove.append(item)
        
        for item in to_remove:
            self.inventory.remove(item)
        
        return removed
    
    def set_status(self, message: str, duration: float = 3.0):
        """Set a status message to display at bottom of screen."""
        self.status_message = message
        self.status_timer = duration
    
    def update_status(self, dt: float):
        """Update status message timer."""
        if self.status_timer > 0:
            self.status_timer -= dt
            if self.status_timer <= 0:
                self.status_message = ""
    
    def toggle_menu(self):
        """Toggle menu open/closed."""
        was_open = self.menu_open
        self.menu_open = not self.menu_open
        if self.menu_open:
            # Refresh log items when opening menu
            self.refresh_log_items()
            print(f"  [Menu] Opened (was: {was_open})")
        else:
            # Reset scroll state when closing
            self.scroll_hold_time = 0.0
            self.scroll_direction = 0
            self.scroll_cooldown = 0.0
            print(f"  [Menu] Closed (was: {was_open})")
    
    def refresh_log_items(self):
        """Refresh list of logged DNA items, filtered and sorted by time (newest first)."""
        all_items = []
        if os.path.exists(DNA_LOGS_DIR):
            for f in os.listdir(DNA_LOGS_DIR):
                if f.endswith('.json'):
                    # Apply filter
                    if self.log_filter == 0:  # ALL
                        all_items.append(f)
                    elif self.log_filter == 1 and f.startswith('plant_'):  # PLANTS
                        all_items.append(f)
                    elif self.log_filter == 2 and f.startswith('animal_'):  # ANIMALS
                        all_items.append(f)
        
        # Sort by timestamp (newest first) - filename format: type_dna_YYYYMMDD_HHMMSS.json
        def get_timestamp(filename):
            # Extract timestamp from filename like "plant_dna_20251223_033251.json"
            parts = filename.replace('.json', '').split('_')
            if len(parts) >= 4:
                return parts[2] + parts[3]  # "20251223033251"
            return "0"  # Fallback for malformed names
        
        all_items.sort(key=get_timestamp, reverse=True)  # Newest first
        
        # Put favorites at top (but still sorted by time within favorites)
        favorites = [f for f in all_items if f in self.favorites]
        non_favorites = [f for f in all_items if f not in self.favorites]
        self.log_items = favorites + non_favorites
        self.log_index = min(self.log_index, max(0, len(self.log_items) - 1))
    
    def menu_navigate_vertical(self, direction: int):
        """Navigate vertically in menu (direction: -1 = up, 1 = down)."""
        if self.menu_tab == 0:  # Inventory tab - 2D grid navigation
            # Move up/down by number of columns
            new_idx = self.inventory_index + (direction * self.inventory_cols)
            if 0 <= new_idx < self.inventory_count:
                self.inventory_index = new_idx
        elif self.menu_tab == 1:  # Log tab
            self.log_index = max(0, min(len(self.log_items) - 1, self.log_index + direction))
    
    def menu_navigate_horizontal(self, direction: int):
        """Navigate horizontally in menu (direction: -1 = left, 1 = right)."""
        if self.menu_tab == 0:  # Inventory tab - 2D grid navigation
            # Calculate current row/col
            row = self.inventory_index // self.inventory_cols
            col = self.inventory_index % self.inventory_cols
            
            # Move left/right within row
            new_col = col + direction
            if 0 <= new_col < self.inventory_cols:
                new_idx = row * self.inventory_cols + new_col
                if new_idx < self.inventory_count:
                    self.inventory_index = new_idx
        elif self.menu_tab == 1:  # Log tab - filter change
            if direction < 0:
                self.log_filter_prev()
            else:
                self.log_filter_next()
    
    def menu_navigate(self, direction: int):
        """Navigate in menu (direction: -1 = up, 1 = down). Kept for compatibility."""
        self.menu_navigate_vertical(direction)
    
    def update_scroll(self, dt: float, is_scrolling_up: bool, is_scrolling_down: bool):
        """Update accelerated scrolling. Returns number of items to scroll.
        Tap = always moves one item. Hold = accelerates over time."""
        # Determine current scroll direction
        new_direction = 0
        if is_scrolling_up:
            new_direction = -1
        elif is_scrolling_down:
            new_direction = 1
        
        # If direction changed or stopped, reset
        if new_direction != self.scroll_direction:
            self.scroll_direction = new_direction
            self.scroll_hold_time = 0.0
            self.scroll_cooldown = 0.0
            # Immediate first scroll on tap (always moves one item)
            if new_direction != 0:
                self.scroll_cooldown = 0.35  # Delay before acceleration kicks in
                return new_direction  # Scroll 1 item immediately
            return 0
        
        # Not scrolling
        if new_direction == 0:
            return 0
        
        # Accumulate hold time
        self.scroll_hold_time += dt
        self.scroll_cooldown -= dt
        
        # Only scroll when cooldown expires (acceleration for holding)
        if self.scroll_cooldown > 0:
            return 0
        
        # Calculate scroll speed based on hold time (acceleration)
        # 0-0.5s: slow (1 item every 0.25s)
        # 0.5-1.5s: medium (1 item every 0.15s)
        # 1.5-3s: fast (1 item every 0.1s)
        # 3-5s: faster (2 items every 0.08s)
        # 5s+: very fast (3 items every 0.06s)
        if self.scroll_hold_time < 0.5:
            scroll_amount = 1
            self.scroll_cooldown = 0.25
        elif self.scroll_hold_time < 1.5:
            scroll_amount = 1
            self.scroll_cooldown = 0.15
        elif self.scroll_hold_time < 3.0:
            scroll_amount = 1
            self.scroll_cooldown = 0.1
        elif self.scroll_hold_time < 5.0:
            scroll_amount = 2
            self.scroll_cooldown = 0.08
        else:
            scroll_amount = 3
            self.scroll_cooldown = 0.06
        
        return scroll_amount * new_direction
    
    def menu_switch_tab(self, direction: int):
        """Switch menu tab (direction: -1 = left, 1 = right)."""
        self.menu_tab = (self.menu_tab + direction) % 3  # 3 tabs: Inventory, Log, Controls
    
    def log_filter_prev(self):
        """Switch to previous log filter (ALL, PLANTS, ANIMALS)."""
        self.log_filter = (self.log_filter - 1) % 3
        self.refresh_log_items()
        print(f"  Log filter: {self.log_filter_names[self.log_filter]}")
    
    def log_filter_next(self):
        """Switch to next log filter (ALL, PLANTS, ANIMALS)."""
        self.log_filter = (self.log_filter + 1) % 3
        self.refresh_log_items()
        print(f"  Log filter: {self.log_filter_names[self.log_filter]}")
    
    def _load_favorites(self):
        """Load favorites from disk."""
        favorites_file = os.path.join(WAVERSE_DIR, "favorites.json")
        try:
            if os.path.exists(favorites_file):
                with open(favorites_file, "r") as f:
                    self.favorites = set(json.load(f))
        except Exception as e:
            print(f"  Could not load favorites: {e}")
            self.favorites = set()
    
    def _save_favorites(self):
        """Save favorites to disk."""
        favorites_file = os.path.join(WAVERSE_DIR, "favorites.json")
        try:
            os.makedirs(WAVERSE_DIR, exist_ok=True)
            with open(favorites_file, "w") as f:
                json.dump(list(self.favorites), f)
        except Exception as e:
            print(f"  Could not save favorites: {e}")
    
    def toggle_log_favorite(self):
        """Toggle favorite status of currently selected log item."""
        if 0 <= self.log_index < len(self.log_items):
            item = self.log_items[self.log_index]
            if item in self.favorites:
                self.favorites.discard(item)
                self.set_status(f"Removed from favorites", 1.5)
            else:
                self.favorites.add(item)
                self.set_status(f"Added to favorites ★", 1.5)
            self._save_favorites()
            self.refresh_log_items()  # Re-sort with favorites at top
    
    def delete_log_favorite(self):
        """Remove currently selected log item from favorites."""
        if 0 <= self.log_index < len(self.log_items):
            item = self.log_items[self.log_index]
            if item in self.favorites:
                self.favorites.discard(item)
                self._save_favorites()
                self.refresh_log_items()
                self.set_status(f"Removed favorite", 1.5)
    
    def next_tool(self):
        """Switch to next tool (R1)."""
        self.current_tool_index = (self.current_tool_index + 1) % len(ToolType.ALL_TOOLS)
        self.current_tool = ToolType.ALL_TOOLS[self.current_tool_index]
        print(f"  Tool: {self.current_tool}")
    
    def prev_tool(self):
        """Switch to previous tool."""
        self.current_tool_index = (self.current_tool_index - 1) % len(ToolType.ALL_TOOLS)
        self.current_tool = ToolType.ALL_TOOLS[self.current_tool_index]
        print(f"  Tool: {self.current_tool}")
    
    def cycle_tool_radius(self):
        """Cycle through tool radius sizes (for MINE/FILL)."""
        current_idx = self.tool_radius_sizes.index(self.tool_radius) if self.tool_radius in self.tool_radius_sizes else 0
        next_idx = (current_idx + 1) % len(self.tool_radius_sizes)
        self.tool_radius = self.tool_radius_sizes[next_idx]
        self.set_status(f"Tool radius: {self.tool_radius}", 2.0)
        print(f"  Tool radius: {self.tool_radius}")
    
    def rotate(self, dx, dy):
        if self.auto_fly_mode > 0:
            # In tour mode, adjust view offset from movement direction
            self.view_yaw_offset += dx * MOUSE_SENSITIVITY
            self.view_pitch -= dy * MOUSE_SENSITIVITY
            self.view_pitch = max(-60, min(60, self.view_pitch))
            self.view_return_timer = 2.0  # Seconds before returning to forward
        else:
            self.yaw += dx * MOUSE_SENSITIVITY
            self.pitch -= dy * MOUSE_SENSITIVITY
            # Allow full vertical loops (like a fighter jet loop-de-loop)
            while self.pitch > 180: self.pitch -= 360
            while self.pitch < -180: self.pitch += 360
    
    def rotate_keyboard(self, dyaw, dpitch):
        if self.auto_fly_mode > 0:
            # In tour mode, adjust view offset from movement direction
            self.view_yaw_offset += dyaw
            self.view_pitch += dpitch
            self.view_pitch = max(-60, min(60, self.view_pitch))
            self.view_return_timer = 2.0
        else:
            self.yaw += dyaw
            self.pitch += dpitch
            # Allow full vertical loops (like a fighter jet loop-de-loop)
            while self.pitch > 180: self.pitch -= 360
            while self.pitch < -180: self.pitch += 360
    
    def get_terrain_height(self, chunk_manager: ChunkManager) -> float:
        """Get terrain height at current position."""
        return chunk_manager.get_height_at(self.x, self.z) * HEIGHT_SCALE
    
    def move(self, forward, right, up, chunk_manager: ChunkManager, structure_manager=None, water_level: float = 0.0):
        # Get current speed based on level, with running multiplier
        speed = BASE_MOVE_SPEED * SPEED_LEVELS[self.speed_level]
        if self.is_running:
            speed *= 2.0  # Double speed while running (B held)
        
        # Swimming is slower
        if self.swimming:
            speed *= 0.7
        
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        if self.flying or self.swimming:
            # Flying/swimming: full 3D movement like a fighter jet
            # Forward vector based on yaw AND pitch
            forward_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
            forward_y = math.sin(pitch_rad)
            forward_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
            
            # Right vector is perpendicular to forward in XZ plane
            # When upside down (|pitch| > 90), the right direction should flip
            # to maintain intuitive controls (push stick right = roll right)
            cos_pitch = math.cos(pitch_rad)
            # Determine if we're "upside down" - when looking more than 90 deg up or down
            upside_down = abs(self.pitch) > 90
            right_flip = -1.0 if upside_down else 1.0
            
            right_x = math.cos(yaw_rad) * right_flip
            right_z = -math.sin(yaw_rad) * right_flip
            
            # Up vector for the camera (perpendicular to both forward and right)
            # This determines which way is "up" when we press the up button
            # cross(right, forward) = up
            up_x = right_z * forward_y - 0 * forward_z  # right.z * forward.y - right.y * forward.z
            up_y = 0 * forward_z - right_x * forward_x + right_z * forward_z + right_x * forward_z  # Simplified to cos(pitch)
            up_y = math.cos(pitch_rad) * right_flip  # Vertical component of up
            up_z = right_x * forward_y - 0 * forward_x  # right.x * forward.y - right.y * forward.x
        else:
            # Walking: move along ground plane only
            forward_x = -math.sin(yaw_rad)
            forward_y = 0
            forward_z = -math.cos(yaw_rad)
            
            right_x = math.cos(yaw_rad)
            right_z = -math.sin(yaw_rad)
            up_x, up_y, up_z = 0, 1, 0  # Walking - up is always world up
        
        # Calculate new position
        # Include up vector's XZ components for full 3D flight
        if self.flying or self.swimming:
            new_x = self.x + (forward * forward_x + right * right_x + up * up_x) * speed
            new_z = self.z + (forward * forward_z + right * right_z + up * up_z) * speed
        else:
            new_x = self.x + (forward * forward_x + right * right_x) * speed
            new_z = self.z + (forward * forward_z + right * right_z) * speed
        
        # Player collision box: ~1m wide, PLAYER_HEIGHT tall (like a rod)
        PLAYER_RADIUS = 1.0  # ~2ft wide collision box
        
        # Check structure collision at new position
        blocked = False
        structure_floor = None
        if structure_manager:
            blocked, structure_floor = structure_manager.check_collision(
                new_x, self.y, new_z, 
                radius=PLAYER_RADIUS, 
                player_height=PLAYER_HEIGHT
            )
        
        # Apply horizontal movement if not blocked
        if not blocked:
            self.x = new_x
            self.z = new_z
        
        # Calculate ground/floor height
        terrain_h = self.get_terrain_height(chunk_manager)
        terrain_ground = terrain_h + PLAYER_HEIGHT
        
        # Effective ground = highest of terrain or structure floor
        if structure_floor is not None:
            effective_ground = max(terrain_ground, structure_floor + PLAYER_HEIGHT)
        else:
            effective_ground = terrain_ground
        
        # Water surface height (scaled)
        water_surface = water_level * HEIGHT_SCALE + PLAYER_HEIGHT
        
        # Check if we should enter/exit swimming mode
        if not self.flying:
            # Enter swimming if in water (below water surface and terrain is underwater)
            if terrain_h < water_level * HEIGHT_SCALE and self.y <= water_surface + 1.0:
                if not self.swimming:
                    self.swimming = True
                    self.set_status("Swimming", 1.5)
            # Exit swimming if terrain is above water
            elif terrain_h >= water_level * HEIGHT_SCALE:
                if self.swimming:
                    self.swimming = False
        
        if self.flying or self.swimming:
            # Calculate new Y position - use the up vector for proper 3D flight
            # When flying, "up" input moves you perpendicular to your look direction
            new_y = self.y + forward * forward_y * speed + up * up_y * speed
            
            # Swimming: cap at water surface
            if self.swimming and new_y > water_surface:
                new_y = water_surface
            
            # Check for collision at new height (both UP and DOWN movement)
            vertical_blocked = False
            if structure_manager and up != 0:
                vertical_blocked, floor_at_new = structure_manager.check_collision(
                    self.x, new_y, self.z,
                    radius=PLAYER_RADIUS,
                    player_height=PLAYER_HEIGHT
                )
                # Also check if we're trying to go through a floor
                if up < 0 and floor_at_new is not None:
                    # Going down - don't go below the floor we're standing on
                    if new_y < floor_at_new + PLAYER_HEIGHT:
                        new_y = floor_at_new + PLAYER_HEIGHT
                        vertical_blocked = False  # Not blocked, just clamped
            
            if not vertical_blocked:
                self.y = new_y
            
            # Recheck floor at new position
            if structure_manager:
                _, new_floor = structure_manager.check_collision(
                    self.x, self.y, self.z,
                    radius=PLAYER_RADIUS,
                    player_height=PLAYER_HEIGHT
                )
                if new_floor is not None:
                    effective_ground = max(terrain_ground, new_floor + PLAYER_HEIGHT)
            
            # Land when pressing down and at ground level (not when swimming)
            if up < 0 and self.y <= effective_ground + 0.5 and not self.swimming:
                self.y = effective_ground
                self.flying = False
                self.target_y = effective_ground
            
            # Swimming: exit water by pressing up when above surface and on land
            if self.swimming and up > 0 and terrain_h >= water_level * HEIGHT_SCALE:
                self.swimming = False
                self.y = effective_ground
            
            # Don't go below ground/floor
            if self.y < effective_ground:
                self.y = effective_ground
        else:
            # Walking/jumping/falling mode
            self.target_y = effective_ground
            
            if self.jumping or self.falling:
                # Variable gravity for dynamic jumping (Mario-style)
                # - Holding jump while rising = low gravity (higher jump)
                # - Released jump while rising = high gravity (cut jump short)
                # - Falling = normal gravity
                if self.jump_velocity > 0:
                    # Rising - gravity depends on whether jump is held
                    if self.jump_held:
                        current_gravity = self.gravity_held  # Float up longer
                    else:
                        current_gravity = self.gravity_released  # Cut jump short
                else:
                    # Falling - use normal gravity
                    current_gravity = self.gravity
                
                self.jump_velocity -= current_gravity * 0.016
                new_y = self.y + self.jump_velocity
                
                # Check for ceiling collision when jumping UP
                ceiling_hit = False
                if self.jump_velocity > 0 and structure_manager:
                    ceiling_hit, _ = structure_manager.check_collision(
                        self.x, new_y, self.z,
                        radius=PLAYER_RADIUS,
                        player_height=PLAYER_HEIGHT
                    )
                
                if ceiling_hit:
                    # Hit head on ceiling - stop upward momentum
                    self.jump_velocity = 0.0
                else:
                    self.y = new_y
                
                # Landed?
                if self.y <= effective_ground:
                    self.y = effective_ground
                    self.jumping = False
                    self.falling = False
                    self.jump_velocity = 0.0
                    self.jump_held = False
                    self.last_ground_y = effective_ground
            else:
                # Walking - follow terrain/floor smoothly
                diff = self.target_y - self.y
                
                # Check for steep drop (cliff/building edge)
                # If ground dropped significantly below our feet, trigger falling
                ground_drop = self.last_ground_y - effective_ground
                CLIFF_THRESHOLD = 4.0  # Must be a significant drop (forgiving)
                
                if ground_drop > CLIFF_THRESHOLD and self.y > effective_ground + 1.0:
                    # We walked off something steep! Start falling naturally
                    self.falling = True
                    self.jump_velocity = 0.0  # Start with no upward velocity
                elif diff < 0:
                    # Going UP - snap/smooth follow (hills)
                    if abs(diff) > 2:
                        self.y += diff * 0.5  # Fast snap up
                    else:
                        self.y += diff * WALK_SMOOTH_SPEED  # Smooth follow up
                    self.last_ground_y = effective_ground
                else:
                    # Going DOWN - gentle slopes follow terrain, steep drops fall
                    if diff < 2.0:
                        # Gentle slope - smooth follow
                        self.y += diff * WALK_SMOOTH_SPEED
                        self.last_ground_y = effective_ground
                    elif diff < CLIFF_THRESHOLD:
                        # Moderate slope - faster follow
                        self.y += diff * 0.3
                        self.last_ground_y = effective_ground
                    else:
                        # Very steep - we're above ground, trigger falling
                        self.falling = True
                        self.jump_velocity = 0.0
            
            # HARD FLOOR: Never let camera go below terrain on steep hills
            # Buffer increased to 2.5 to prevent seeing through ground on slopes
            # (combined with near clip plane of 1.5, this keeps terrain visible)
            slope_buffer = 2.5
            min_height = terrain_h + PLAYER_HEIGHT + slope_buffer
            if self.y < min_height:
                self.y = min_height
                self.falling = False
                self.jumping = False
                self.jump_velocity = 0.0
            
            # Fly when pressing up (H key or R3)
            if up > 0:
                self.flying = True
                self.jumping = False
                self.y += speed
        
        # Update velocity tracking for HUD
        self.velocity_x = self.x - self.prev_x
        self.velocity_y = self.y - self.prev_y
        self.velocity_z = self.z - self.prev_z
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_z = self.z
    
    def get_speed(self) -> float:
        """Get current movement speed (horizontal)."""
        return math.sqrt(self.velocity_x**2 + self.velocity_z**2) * 60  # Per second
    
    def get_speed_3d(self) -> float:
        """Get current 3D movement speed."""
        return math.sqrt(self.velocity_x**2 + self.velocity_y**2 + self.velocity_z**2) * 60
    
    def get_speed_factor(self) -> float:
        """Get normalized speed factor (0.0 = still, 1.0 = very fast)."""
        speed = self.get_speed()
        # Normalize: walking ~2-5, running ~10, flying fast ~50+
        return min(1.0, speed / 30.0)
    
    def jump(self, held: bool = True):
        """Start a jump if on the ground. held=True for variable jump height."""
        if not self.flying and not self.jumping and not self.falling:
            self.jumping = True
            self.falling = False
            self.jump_velocity = self.jump_strength
            self.jump_held = held
    
    def update_jump_held(self, held: bool):
        """Update whether jump button is being held (for variable jump height)."""
        self.jump_held = held
    
    def add_marker(self):
        """Add a marker at current position, using next available letter."""
        # Cycle through colors for different markers
        colors = [
            (0.2, 0.6, 1.0),   # Blue
            (0.3, 1.0, 0.4),   # Green
            (1.0, 1.0, 0.2),   # Yellow
            (1.0, 0.5, 0.0),   # Orange
            (0.9, 0.2, 0.9),   # Magenta
            (0.2, 1.0, 1.0),   # Cyan
            (1.0, 0.6, 0.6),   # Light red
        ]
        
        # Find first unused letter
        labels = "ABCDEFGHIJ"
        used_labels = {m[3] for m in self.markers}  # Get labels from existing markers
        label = "?"
        for l in labels:
            if l not in used_labels:
                label = l
                break
        
        # Color based on letter index
        label_idx = labels.index(label) if label in labels else 0
        color = colors[label_idx % len(colors)]
        
        self.markers.append((self.x, self.z, color, label))
        
        # Limit number of markers
        if len(self.markers) > self.max_markers:
            self.markers.pop(0)
        
        print(f"  Marker {label} set at ({self.x:.1f}, {self.z:.1f})")
        return label
    
    def clear_markers(self):
        """Clear all markers."""
        self.markers.clear()
        print("  All markers cleared")
    
    def toggle_auto_fly(self):
        """Cycle tour modes: off -> wander -> showcase -> off."""
        self.auto_fly_mode = (self.auto_fly_mode + 1) % 3  # Cycle 0 -> 1 -> 2 -> 0
        
        if self.auto_fly_mode == 0:
            print("=" * 40)
            print("  TOUR MODE: OFF")
            print("=" * 40)
            self.set_status("Tour mode: OFF")
            # Reset states that tour mode may have modified
            self.view_yaw_offset = 0.0
            self.view_pitch = -5.0
            self.view_return_timer = 0.0
            self.showcase_target = None
            self.showcase_phase = 0
            # Reset pitch to a reasonable value (extreme pitch causes no horizontal movement in flying mode)
            if abs(self.pitch) > 60:
                self.pitch = -10.0
            # Note: keep flying state as-is (user can land with F)
        elif self.auto_fly_mode == 1:
            print("=" * 40)
            print("  TOUR MODE: WANDER")
            print("  Auto-flying, random exploration")
            print("  Look around freely - returns to forward")
            print("  Press X for SHOWCASE mode")
            print("=" * 40)
            self.set_status("Tour mode: WANDER")
            # Start wandering in current facing direction
            self.wander_direction = math.radians(self.yaw)
            self.wander_timer = 30 + random.random() * 30
            self.flying = True
            # Reset view to forward
            self.view_yaw_offset = 0.0
            self.view_pitch = -5.0
            self.view_return_timer = 0.0
        else:  # mode == 2 (showcase)
            print("=" * 40)
            print("  TOUR MODE: SHOWCASE")
            print("  Flying to interesting plants/animals/structures")
            print("  Exploring in consistent direction!")
            print("  Press X to turn OFF")
            print("=" * 40)
            self.set_status("Tour mode: SHOWCASE")
            self.flying = True
            self.showcase_target = None  # Will be set on first update
            self.showcase_phase = 0
            self.showcase_arc_progress = 0.0
            # Set the origin point - we'll always move OUTWARD from here
            self.showcase_origin = (self.x, self.z)
            self.showcase_explore_dir = math.radians(self.yaw)  # Initial direction
            self.showcase_total_distance = 0.0
            self.showcase_is_structure = False
            print(f"  Showcase origin: ({int(self.x)}, {int(self.z)})")
    
    def update_auto_fly(self, dt: float, chunk_manager: ChunkManager, 
                        flora_manager=None, animal_manager=None, structure_manager=None):
        """Update tour mode - movement and view are independent."""
        if self.auto_fly_mode == 0:
            return 0, 0, 0
        
        dt_seconds = dt / 60.0
        # Tour mode is slower to let chunks load - use lower multiplier
        speed_mult = SPEED_LEVELS[self.speed_level] * 1.0  # Reduced from 3.0
        
        if self.auto_fly_mode == 1:
            # === WANDER MODE ===
            return self._update_wander_mode(dt_seconds, speed_mult)
        else:
            # === SHOWCASE MODE ===
            return self._update_showcase_mode(dt_seconds, speed_mult, chunk_manager,
                                              flora_manager, animal_manager, structure_manager)
    
    def _update_wander_mode(self, dt_seconds: float, speed_mult: float):
        """Wander mode - random direction changes."""
        # === MOVEMENT (independent of view) ===
        # Random direction changes
        self.wander_timer -= dt_seconds
        if self.wander_timer <= 0:
            turn_angle = (random.random() - 0.5) * 60  # ±30 degrees
            self.wander_direction += math.radians(turn_angle)
            self.wander_timer = 30 + random.random() * 30
            print(f"  Tour: turning {turn_angle:.0f}° (next in {self.wander_timer:.0f}s)")
        
        # === VIEW (can look around, returns to forward) ===
        # Update return timer
        if self.view_return_timer > 0:
            self.view_return_timer -= dt_seconds
        
        # If not actively looking, smoothly return to forward
        if self.view_return_timer <= 0:
            # Decay yaw offset toward 0
            self.view_yaw_offset *= 0.97
            if abs(self.view_yaw_offset) < 0.5:
                self.view_yaw_offset = 0
            # Decay pitch toward default (-5)
            self.view_pitch = self.view_pitch * 0.97 + (-5.0) * 0.03
        
        # Set camera view = movement direction + view offset
        movement_yaw_deg = math.degrees(self.wander_direction)
        self.yaw = movement_yaw_deg + self.view_yaw_offset
        self.pitch = self.view_pitch
        
        # === ACTUALLY MOVE along wander_direction (not yaw!) ===
        # Calculate movement vector from wander_direction
        move_x = -math.sin(self.wander_direction)
        move_z = -math.cos(self.wander_direction)
        
        # Very slow movement to let chunks load (reduced from 0.25)
        move_speed = speed_mult * 0.2 * dt_seconds * 60
        self.x += move_x * move_speed
        self.z += move_z * move_speed
        
        return 0, 0, 0  # We moved directly, don't use forward/right/up
    
    def _update_showcase_mode(self, dt_seconds: float, speed_mult: float, chunk_manager: ChunkManager,
                              flora_manager, animal_manager, structure_manager):
        """Showcase mode - fly to interesting items in arcs."""
        
        # === PICK A NEW TARGET if needed ===
        if self.showcase_target is None or self.showcase_phase > 1:
            self._pick_showcase_target(chunk_manager, flora_manager, animal_manager, structure_manager)
            if self.showcase_target is None:
                # No targets found, fall back to wander
                return self._update_wander_mode(dt_seconds, speed_mult)
        
        target_x, target_y, target_z, entity_type, entity = self.showcase_target
        
        if self.showcase_phase == 0:
            # === TRAVELING PHASE - fly in arc toward viewing position (not target itself) ===
            self.showcase_arc_progress += dt_seconds * 0.08 * speed_mult  # Very slow for chunk loading
            
            # Calculate viewing position (offset from target)
            # We orbit at showcase_view_distance from target
            view_x = target_x + math.sin(self.showcase_orbit_angle) * self.showcase_view_distance
            view_z = target_z + math.cos(self.showcase_orbit_angle) * self.showcase_view_distance
            view_y = target_y + 3.0  # Slightly above target
            
            if self.showcase_arc_progress >= 1.0:
                # Arrived! Switch to viewing with pan
                self.showcase_phase = 1
                self.showcase_arc_progress = 1.0
                
                # Set up pan: random direction, random angle (5-30 degrees)
                self.showcase_pan_direction = random.choice([-1, 1])
                self.showcase_pan_total = 5.0 + random.random() * 25.0  # 5-30 degrees
                view_time = 3.0 + random.random() * 2.0  # 3-5 seconds viewing
                self.showcase_view_timer = view_time
                # Slower pan for smaller angles (same view time)
                self.showcase_pan_speed = self.showcase_pan_total / view_time
                
                print(f"  Showcase: viewing {entity_type} (pan {self.showcase_pan_total:.0f}°)")
            
            # Calculate position along arc (parabolic)
            t = self.showcase_arc_progress
            start_x, start_y, start_z = self.showcase_start_pos
            
            # Horizontal: linear interpolation toward viewing position
            self.x = start_x + (view_x - start_x) * t
            self.z = start_z + (view_z - start_z) * t
            
            # Vertical: parabolic arc (peaks at t=0.5)
            base_y = start_y + (view_y - start_y) * t
            arc_offset = self.showcase_arc_height * 4 * t * (1 - t)  # Parabola: 0 at t=0,1; max at t=0.5
            self.y = base_y + arc_offset
            
            # Look toward target (not viewing position) - SMOOTH transition
            dx = target_x - self.x
            dy = target_y - self.y
            dz = target_z - self.z
            dist = math.sqrt(dx*dx + dz*dz)
            
            # Calculate target camera angles
            self.showcase_target_yaw = math.degrees(math.atan2(-dx, -dz))
            raw_pitch = math.degrees(math.atan2(dy, dist)) if dist > 0.1 else 0
            # Clamp pitch to avoid looking too far up/down
            self.showcase_target_pitch = max(-25, min(15, raw_pitch))
            
            # Smooth interpolation toward target angles
            # Handle yaw wraparound
            yaw_diff = self.showcase_target_yaw - self.yaw
            while yaw_diff > 180: yaw_diff -= 360
            while yaw_diff < -180: yaw_diff += 360
            self.yaw += yaw_diff * 0.08  # Smooth factor
            
            self.pitch += (self.showcase_target_pitch - self.pitch) * 0.08
            
        elif self.showcase_phase == 1:
            # === VIEWING PHASE - stay in place, pan camera left or right ===
            self.showcase_view_timer -= dt_seconds
            
            if self.showcase_view_timer <= 0:
                # Done viewing, pick new target
                self.showcase_phase = 2  # Will trigger new target on next update
                print(f"  Showcase: moving to next...")
            
            # Pan the camera (rotate yaw) - direction and speed set when arrived
            # showcase_pan_direction: -1 = left, +1 = right
            # showcase_pan_speed: degrees per second (slower for smaller angles)
            self.yaw += self.showcase_pan_direction * self.showcase_pan_speed * dt_seconds
            
            # Smoothly level out pitch while viewing
            target_pitch = -5.0  # Slight downward look
            self.pitch += (target_pitch - self.pitch) * 0.05
        
        return 0, 0, 0
    
    def _pick_showcase_target(self, chunk_manager: ChunkManager, 
                              flora_manager, animal_manager, structure_manager):
        """Pick a target that is FURTHER from origin - always exploring outward."""
        candidates = []
        
        # Calculate current distance from origin
        if self.showcase_origin is None:
            self.showcase_origin = (self.x, self.z)
        origin_x, origin_z = self.showcase_origin
        current_dist_from_origin = math.sqrt((self.x - origin_x)**2 + (self.z - origin_z)**2)
        
        # Update exploration direction: from origin through current position
        if current_dist_from_origin > 100:
            # Direction from origin to current position = outward direction
            self.showcase_explore_dir = math.atan2(self.x - origin_x, self.z - origin_z)
        
        cx, cz = self.get_chunk_pos()
        search_range = 15  # Search much further for distant targets
        
        # Collect plants
        if flora_manager:
            for dcx in range(-search_range, search_range + 1):
                for dcz in range(-search_range, search_range + 1):
                    key = (cx + dcx, cz + dcz)
                    if key in flora_manager.chunk_plants:
                        for plant in flora_manager.chunk_plants[key]:
                            # Skip tiny plants
                            plant_height = getattr(plant.dna, 'height_gene', None)
                            if plant_height and plant_height.value > 2.0:
                                dist = math.sqrt((plant.x - self.x)**2 + (plant.z - self.z)**2)
                                if 500 < dist < 3000:  # 5x further!
                                    h = plant_height.value if plant_height else 3.0
                                    candidates.append((plant.x, plant.y + h * 0.5, plant.z, 
                                                      "plant", plant, dist))
        
        # Collect animals
        if animal_manager:
            for dcx in range(-search_range, search_range + 1):
                for dcz in range(-search_range, search_range + 1):
                    key = (cx + dcx, cz + dcz)
                    if key in animal_manager.chunk_animals:
                        for animal in animal_manager.chunk_animals[key]:
                            dist = math.sqrt((animal.x - self.x)**2 + (animal.z - self.z)**2)
                            if 500 < dist < 3000:  # 5x further!
                                candidates.append((animal.x, animal.y + 2.0, animal.z, 
                                                  "animal", animal, dist))
        
        # Collect structures
        if structure_manager:
            for structure in structure_manager.structures:
                # Find center of structure
                if structure.floors:
                    floor = structure.floors[0]
                    struct_x = floor.x
                    struct_z = floor.z
                    struct_y = floor.y + 15  # Higher above for better view
                    dist = math.sqrt((struct_x - self.x)**2 + (struct_z - self.z)**2)
                    if 500 < dist < 4000:  # 5x further!
                        candidates.append((struct_x, struct_y, struct_z, "structure", structure, dist))
        
        if not candidates:
            self.showcase_target = None
            return
        
        # STRICT filtering: only consider targets within 45 degrees of where we're facing
        forward_candidates = []
        for cand in candidates:
            tx, ty, tz, etype, ent, dist = cand
            
            # Calculate direction to this target
            dx = tx - self.x
            dz = tz - self.z
            target_dir = math.atan2(dx, dz)
            
            # How well does it align with our facing direction?
            dir_diff = abs(target_dir - self.showcase_explore_dir)
            while dir_diff > math.pi: dir_diff = abs(dir_diff - 2 * math.pi)
            
            # Only keep targets within 45 degrees of forward
            if dir_diff < math.radians(45):
                forward_candidates.append((cand, dist, dir_diff))
        
        # If no forward targets, widen to 90 degrees
        if not forward_candidates:
            for cand in candidates:
                tx, ty, tz, etype, ent, dist = cand
                dx = tx - self.x
                dz = tz - self.z
                target_dir = math.atan2(dx, dz)
                dir_diff = abs(target_dir - self.showcase_explore_dir)
                while dir_diff > math.pi: dir_diff = abs(dir_diff - 2 * math.pi)
                if dir_diff < math.radians(90):
                    forward_candidates.append((cand, dist, dir_diff))
        
        # If still nothing, take anything
        if not forward_candidates:
            forward_candidates = [(c, c[5], 0) for c in candidates]
        
        # Score remaining candidates: STRONGLY prefer targets FURTHER from origin
        scored = []
        origin_x, origin_z = self.showcase_origin if self.showcase_origin else (self.x, self.z)
        current_dist_from_origin = math.sqrt((self.x - origin_x)**2 + (self.z - origin_z)**2)
        
        for cand, dist, dir_diff in forward_candidates:
            tx, ty, tz = cand[0], cand[1], cand[2]
            
            # How far is this target from origin?
            target_dist_from_origin = math.sqrt((tx - origin_x)**2 + (tz - origin_z)**2)
            
            # CRITICAL: Is this target FURTHER from origin than we are?
            outward_progress = target_dist_from_origin - current_dist_from_origin
            
            # Skip targets that would take us backward (closer to origin)
            if outward_progress < -100:  # Allow small backtracking (100 units)
                continue
            
            # Score heavily based on outward progress
            outward_score = max(0, outward_progress / 1000.0)  # Huge bonus for going outward
            direction_score = 1.0 - (dir_diff / math.radians(90))  # Less strict on direction
            
            score = max(0.1, (outward_score * 5.0) + (direction_score * 1.0))
            
            # Slight bonus for structures
            if cand[3] == "structure":
                score += 0.3
            
            scored.append((cand, score))
        
        # If all targets were rejected (going backward), pick the one that goes most outward
        if not scored and forward_candidates:
            # Fall back: pick the one with best outward progress even if negative
            best_cand = None
            best_progress = -99999
            for cand, dist, dir_diff in forward_candidates:
                tx, tz = cand[0], cand[2]
                target_dist_from_origin = math.sqrt((tx - origin_x)**2 + (tz - origin_z)**2)
                outward_progress = target_dist_from_origin - current_dist_from_origin
                if outward_progress > best_progress:
                    best_progress = outward_progress
                    best_cand = cand
            if best_cand:
                scored.append((best_cand, 1.0))
        
        # Weighted random selection
        total_score = sum(s for _, s in scored)
        if total_score <= 0:
            target_full = forward_candidates[0][0] if forward_candidates else random.choice(candidates)
        else:
            r = random.random() * total_score
            cumulative = 0
            target_full = scored[0][0]
            for cand, score in scored:
                cumulative += score
                if r <= cumulative:
                    target_full = cand
                    break
        
        # Unpack target (now has 6 elements including dist)
        target_x, target_y, target_z, entity_type, entity, target_dist = target_full
        self.showcase_target = (target_x, target_y, target_z, entity_type, entity)
        self.showcase_start_pos = (self.x, self.y, self.z)
        self.showcase_phase = 0
        self.showcase_arc_progress = 0.0
        self.showcase_is_structure = (entity_type == "structure")
        
        # Track total distance traveled
        target_dist_from_origin = math.sqrt((target_x - origin_x)**2 + (target_z - origin_z)**2)
        self.showcase_total_distance = target_dist_from_origin
        
        # Calculate arc height based on distance
        horiz_dist = math.sqrt((target_x - self.x)**2 + (target_z - self.z)**2)
        height_diff = abs(target_y - self.y)
        self.showcase_arc_height = max(20.0, horiz_dist * 0.25, height_diff * 0.5)
        
        # Set initial orbit angle based on approach direction
        dx = self.x - target_x
        dz = self.z - target_z
        self.showcase_orbit_angle = math.atan2(dx, dz)
        
        # Vary viewing distance based on entity type
        if entity_type == "structure":
            self.showcase_view_distance = 35.0  # Much further for buildings - stay OUTSIDE
        elif entity_type == "plant":
            self.showcase_view_distance = 7.0   # Plants
        else:
            self.showcase_view_distance = 9.0   # Animals
        
        # Show progress from origin
        origin_x, origin_z = self.showcase_origin if self.showcase_origin else (self.x, self.z)
        dist_from_origin = math.sqrt((target_x - origin_x)**2 + (target_z - origin_z)**2)
        print(f"  Showcase: flying to {entity_type} ({horiz_dist:.0f}m away, {dist_from_origin:.0f}m from origin)")
        self.set_status(f"Flying to {entity_type}... ({int(dist_from_origin)}m explored)")
    
    def maintain_auto_fly_height(self, chunk_manager: ChunkManager):
        """Keep camera at consistent height above terrain during tour mode (wander only)."""
        # Only maintain height in wander mode (mode 1), showcase handles its own height
        if self.auto_fly_mode != 1:
            return
        
        terrain_h = self.get_terrain_height(chunk_manager)
        target_y = terrain_h + self.auto_fly_height
        
        # Calculate height difference
        diff = target_y - self.y
        
        # Smooth height adjustment with velocity limits
        MAX_RISE_SPEED = 1.0    # Rise quickly to clear hills
        MAX_FALL_SPEED = 0.3    # Fall slowly over cliffs
        
        if diff > 0:
            change = min(diff * 0.2, MAX_RISE_SPEED)
        else:
            change = max(diff * 0.15, -MAX_FALL_SPEED)
        
        self.y += change
        
        # Hard floor - never clip through terrain
        min_height = terrain_h + 8
        if self.y < min_height:
            self.y = min_height
    
    def set_speed(self, level: int):
        """Set speed level (0 to len(SPEED_LEVELS)-1)."""
        self.speed_level = max(0, min(len(SPEED_LEVELS) - 1, level))
    
    def apply(self):
        glRotatef(-self.pitch, 1, 0, 0)
        glRotatef(-self.yaw, 0, 1, 0)
        glTranslatef(-self.x, -self.y, -self.z)
    
    def get_chunk_pos(self) -> tuple:
        chunk_world_size = CHUNK_SIZE * TERRAIN_SCALE
        return (int(self.x // chunk_world_size), int(self.z // chunk_world_size))
    
    def get_pos(self) -> tuple:
        return (self.x, self.y, self.z)
    
    def get_forward_vector(self) -> tuple:
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        return (
            -math.sin(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            -math.cos(yaw_rad) * math.cos(pitch_rad)
        )


class ChunkRenderer:
    """Renders chunks using display lists with LOD for distant terrain."""
    
    def __init__(self, chunk_manager: ChunkManager, chunk_dna_manager: ChunkDNAManager = None):
        self.chunk_manager = chunk_manager
        self.chunk_dna_manager = chunk_dna_manager
        self.display_lists = {}  # (cx, cz) -> (display_list_id, lod_level)
        self.camera_chunk = (0, 0)
        self._current_palette = TerrainPalette()  # Default palette
    
    def create_chunk_display_list(self, cx: int, cz: int, lod_step: int = 1) -> int:
        """Create a display list for a chunk with LOD support."""
        chunk = self.chunk_manager.get_chunk(cx, cz)
        heightmap = chunk.heightmap
        h, w = heightmap.shape
        
        # Get chunk-specific color palette from DNA
        if self.chunk_dna_manager:
            chunk_dna = self.chunk_dna_manager.get_dna(cx, cz)
            palette = chunk_dna.palette
        else:
            palette = self._current_palette
        
        chunk_list = glGenLists(1)
        glNewList(chunk_list, GL_COMPILE)
        glBegin(GL_QUADS)
        
        # World offset for this chunk
        world_x_offset = chunk.world_x
        world_z_offset = chunk.world_z
        
        # Step controls LOD - 1 = full detail, 2 = half, 4 = quarter
        step = lod_step
        
        for z in range(0, h - step, step):
            for x in range(0, w - step, step):
                # Heights at 4 corners (with LOD step)
                h00 = heightmap[z, x]
                h10 = heightmap[z, min(x + step, w - 1)]
                h01 = heightmap[min(z + step, h - 1), x]
                h11 = heightmap[min(z + step, h - 1), min(x + step, w - 1)]
                
                # World positions (scaled by LOD step)
                wx0 = world_x_offset + x * TERRAIN_SCALE
                wx1 = world_x_offset + (x + step) * TERRAIN_SCALE
                wz0 = world_z_offset + z * TERRAIN_SCALE
                wz1 = world_z_offset + (z + step) * TERRAIN_SCALE
                
                # Scaled heights
                y00 = h00 * HEIGHT_SCALE
                y10 = h10 * HEIGHT_SCALE
                y01 = h01 * HEIGHT_SCALE
                y11 = h11 * HEIGHT_SCALE
                
                # Color from average height using chunk's palette
                avg_raw_h = (h00 + h10 + h01 + h11) / 4
                col = palette.get_color(avg_raw_h)
                
                # Normal calculation
                dx = (h10 - h00 + h11 - h01) * HEIGHT_SCALE
                dz = (h01 - h00 + h11 - h10) * HEIGHT_SCALE
                nx, ny, nz = -dx, 2 * TERRAIN_SCALE * step, -dz
                length = math.sqrt(nx*nx + ny*ny + nz*nz)
                if length > 0:
                    nx, ny, nz = nx/length, ny/length, nz/length
                
                glNormal3f(nx, ny, nz)
                glColor3f(*col)
                
                # Quad vertices
                glVertex3f(wx0, y00, wz0)
                glVertex3f(wx1, y10, wz0)
                glVertex3f(wx1, y11, wz1)
                glVertex3f(wx0, y01, wz1)
        
        glEnd()
        glEndList()
        
        return chunk_list
    
    def get_lod_for_distance(self, dist: int) -> int:
        """Get LOD step based on distance from camera."""
        if dist <= LOD_FULL_DISTANCE:
            return 1  # Full detail (1024 quads)
        elif dist <= LOD_HALF_DISTANCE:
            return 2  # Half detail (256 quads)
        elif dist <= LOD_QUARTER_DISTANCE:
            return 4  # Quarter detail (64 quads)
        else:
            return 8  # 1/8th detail (16 quads) - for horizon
    
    def update_chunks(self, camera_chunk: tuple, force_all: bool = False):
        """Update which chunks are loaded with LOD support."""
        cx, cz = camera_chunk
        self.camera_chunk = camera_chunk
        
        # Calculate all needed chunks with their LOD levels
        needed = {}  # pos -> lod_step
        for dz in range(-CHUNK_RENDER_DISTANCE, CHUNK_RENDER_DISTANCE + 1):
            for dx in range(-CHUNK_RENDER_DISTANCE, CHUNK_RENDER_DISTANCE + 1):
                pos = (cx + dx, cz + dz)
                dist = max(abs(dx), abs(dz))
                lod = self.get_lod_for_distance(dist)
                needed[pos] = lod
        
        # Remove chunks that are too far
        buffer_dist = CHUNK_RENDER_DISTANCE + 2
        to_remove = []
        for pos in self.display_lists:
            if abs(pos[0] - cx) > buffer_dist or abs(pos[1] - cz) > buffer_dist:
                to_remove.append(pos)
        
        for pos in to_remove:
            glDeleteLists(self.display_lists[pos][0], 1)
            del self.display_lists[pos]
        
        # Check for LOD changes or missing chunks
        to_update = []
        for pos, target_lod in needed.items():
            if pos not in self.display_lists:
                to_update.append((pos, target_lod))
            elif self.display_lists[pos][1] != target_lod:
                # LOD changed - need to regenerate (but lower priority)
                dist = max(abs(pos[0] - cx), abs(pos[1] - cz))
                if dist <= LOD_FULL_DISTANCE + 1:  # Only update LOD for nearby
                    to_update.append((pos, target_lod))
        
        if not to_update:
            return
        
        # Sort by distance (closest first)
        to_update.sort(key=lambda p: max(abs(p[0][0] - cx), abs(p[0][1] - cz)))
        
        if force_all:
            # Load everything at startup
            for i, (pos, lod) in enumerate(to_update):
                if pos in self.display_lists:
                    glDeleteLists(self.display_lists[pos][0], 1)
                display_list = self.create_chunk_display_list(pos[0], pos[1], lod)
                self.display_lists[pos] = (display_list, lod)
                if (i + 1) % 20 == 0:
                    print(f"    Loading chunks: {i + 1}/{len(to_update)}")
        else:
            # Load up to 3 chunks per frame
            for pos, lod in to_update[:3]:
                if pos in self.display_lists:
                    glDeleteLists(self.display_lists[pos][0], 1)
                display_list = self.create_chunk_display_list(pos[0], pos[1], lod)
                self.display_lists[pos] = (display_list, lod)
    
    def render(self):
        """Render all loaded chunks."""
        for display_list, lod in self.display_lists.values():
            glCallList(display_list)
    
    def stop(self):
        """Cleanup."""
        pass


class Minimap:
    """Simple minimap."""
    
    def __init__(self, chunk_manager: ChunkManager, size: int = 180):
        self.chunk_manager = chunk_manager
        self.size = size
        self.texture_id = None
        self.last_pos = (0, 0)
    
    def update(self, center_x: float, center_z: float, radius: float = 200):
        """Update minimap texture."""
        samples = 64
        step = radius * 2 / samples
        
        pixels = []
        for j in range(samples):
            for i in range(samples):
                wx = center_x - radius + i * step
                wz = center_z - radius + j * step
                h = self.chunk_manager.get_height_at(wx, wz) * HEIGHT_SCALE
                
                r, g, b = height_to_color(h)
                pixels.extend([int(r * 255), int(g * 255), int(b * 255)])
        
        pixel_data = (GLubyte * len(pixels))(*pixels)
        
        if self.texture_id:
            glDeleteTextures([self.texture_id])
        
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, samples, samples, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, pixel_data)
        
        self.last_pos = (center_x, center_z)
    
    def draw(self, display: tuple, camera_x: float, camera_z: float):
        """Draw minimap overlay."""
        if not self.texture_id:
            return
        
        margin = 10
        
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, display[0], display[1], 0, -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Border
        glColor4f(0, 0, 0, 0.7)
        glBegin(GL_QUADS)
        glVertex2f(margin - 3, margin - 3)
        glVertex2f(margin + self.size + 3, margin - 3)
        glVertex2f(margin + self.size + 3, margin + self.size + 3)
        glVertex2f(margin - 3, margin + self.size + 3)
        glEnd()
        
        # Map
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glColor4f(1, 1, 1, 0.9)
        
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(margin, margin)
        glTexCoord2f(1, 0); glVertex2f(margin + self.size, margin)
        glTexCoord2f(1, 1); glVertex2f(margin + self.size, margin + self.size)
        glTexCoord2f(0, 1); glVertex2f(margin, margin + self.size)
        glEnd()
        
        glDisable(GL_TEXTURE_2D)
        
        # Player marker (center)
        px = margin + self.size / 2
        py = margin + self.size / 2
        glColor4f(1, 0, 0, 1)
        glBegin(GL_QUADS)
        glVertex2f(px - 5, py - 5)
        glVertex2f(px + 5, py - 5)
        glVertex2f(px + 5, py + 5)
        glVertex2f(px - 5, py + 5)
        glEnd()
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


def render_dropped_items(dropped_items: list, camera_x: float, camera_y: float, camera_z: float):
    """Render dropped items as small cylinder logs (for wood) or other shapes."""
    if not dropped_items:
        return
    
    # Culling distance
    max_render_dist = 100.0
    
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    for dropped in dropped_items:
        dx = dropped.x - camera_x
        dy = dropped.y - camera_y
        dz = dropped.z - camera_z
        dist_sq = dx*dx + dy*dy + dz*dz
        
        if dist_sq > max_render_dist * max_render_dist:
            continue
        
        glPushMatrix()
        glTranslatef(dropped.x, dropped.y, dropped.z)
        glRotatef(dropped.rotation, 0, 1, 0)  # Random rotation
        glRotatef(90, 0, 0, 1)  # Lay flat horizontally
        
        # Draw based on item type
        item_type = dropped.item.item_type
        
        if item_type == "wood":
            # Draw 3 small cylinder logs
            log_radius = 0.15 * dropped.scale
            log_length = 0.5 * dropped.scale
            
            # Brown wood color
            glColor3f(0.55, 0.35, 0.15)
            
            for i, offset in enumerate([(-0.2, 0), (0.15, 0.1), (0, -0.15)]):
                glPushMatrix()
                glTranslatef(offset[0] * dropped.scale, offset[1] * dropped.scale, 0)
                
                # Draw cylinder using quads
                segments = 8
                for j in range(segments):
                    angle1 = (j / segments) * 2 * math.pi
                    angle2 = ((j + 1) / segments) * 2 * math.pi
                    
                    x1, y1 = math.cos(angle1) * log_radius, math.sin(angle1) * log_radius
                    x2, y2 = math.cos(angle2) * log_radius, math.sin(angle2) * log_radius
                    
                    # Cylinder side
                    glBegin(GL_QUADS)
                    glNormal3f(math.cos(angle1), math.sin(angle1), 0)
                    glVertex3f(x1, y1, -log_length/2)
                    glVertex3f(x1, y1, log_length/2)
                    glNormal3f(math.cos(angle2), math.sin(angle2), 0)
                    glVertex3f(x2, y2, log_length/2)
                    glVertex3f(x2, y2, -log_length/2)
                    glEnd()
                
                # Draw end caps (lighter wood color for cross-section)
                glColor3f(0.75, 0.55, 0.35)
                for z_end in [-log_length/2, log_length/2]:
                    glBegin(GL_TRIANGLE_FAN)
                    glNormal3f(0, 0, 1 if z_end > 0 else -1)
                    glVertex3f(0, 0, z_end)
                    for j in range(segments + 1):
                        angle = (j / segments) * 2 * math.pi
                        glVertex3f(math.cos(angle) * log_radius, math.sin(angle) * log_radius, z_end)
                    glEnd()
                glColor3f(0.55, 0.35, 0.15)  # Reset to bark color
                
                glPopMatrix()
        else:
            # Generic item: draw a small glowing cube
            size = 0.3 * dropped.scale
            glColor3f(0.8, 0.8, 0.2)  # Yellow glow
            glBegin(GL_QUADS)
            # Front
            glNormal3f(0, 0, 1)
            glVertex3f(-size, -size, size)
            glVertex3f(size, -size, size)
            glVertex3f(size, size, size)
            glVertex3f(-size, size, size)
            # Back
            glNormal3f(0, 0, -1)
            glVertex3f(-size, -size, -size)
            glVertex3f(-size, size, -size)
            glVertex3f(size, size, -size)
            glVertex3f(size, -size, -size)
            # Top
            glNormal3f(0, 1, 0)
            glVertex3f(-size, size, -size)
            glVertex3f(-size, size, size)
            glVertex3f(size, size, size)
            glVertex3f(size, size, -size)
            # Bottom
            glNormal3f(0, -1, 0)
            glVertex3f(-size, -size, -size)
            glVertex3f(size, -size, -size)
            glVertex3f(size, -size, size)
            glVertex3f(-size, -size, size)
            glEnd()
        
        glPopMatrix()
    
    glDisable(GL_COLOR_MATERIAL)


def render_debris(debris_particles: list, camera_x: float, camera_y: float, camera_z: float):
    """Render debris particles (fading visual effects from cutting trees, etc.)."""
    if not debris_particles:
        return
    
    # Culling distance
    max_render_dist = 80.0
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    for debris in debris_particles:
        dx = debris.x - camera_x
        dy = debris.y - camera_y
        dz = debris.z - camera_z
        dist_sq = dx*dx + dy*dy + dz*dz
        
        if dist_sq > max_render_dist * max_render_dist:
            continue
        
        if debris.alpha <= 0:
            continue
        
        glPushMatrix()
        glTranslatef(debris.x, debris.y, debris.z)
        glRotatef(debris.rotation, 0, 1, 0)
        glRotatef(debris.rotation * 0.7, 1, 0, 0)  # Tumble
        
        # Color with alpha fade
        r, g, b = debris.color
        glColor4f(r, g, b, debris.alpha)
        
        # Draw as small box or flat quad
        size = debris.scale
        
        if debris.debris_type == "leaf":
            # Flat quad for leaves
            glBegin(GL_QUADS)
            glNormal3f(0, 1, 0)
            glVertex3f(-size, 0, -size * 0.6)
            glVertex3f(size, 0, -size * 0.6)
            glVertex3f(size, 0, size * 0.6)
            glVertex3f(-size, 0, size * 0.6)
            glEnd()
        else:
            # Small cube for wood chips
            half = size * 0.5
            glBegin(GL_QUADS)
            # Front
            glNormal3f(0, 0, 1)
            glVertex3f(-half, -half, half)
            glVertex3f(half, -half, half)
            glVertex3f(half, half, half)
            glVertex3f(-half, half, half)
            # Back
            glNormal3f(0, 0, -1)
            glVertex3f(-half, -half, -half)
            glVertex3f(-half, half, -half)
            glVertex3f(half, half, -half)
            glVertex3f(half, -half, -half)
            # Top
            glNormal3f(0, 1, 0)
            glVertex3f(-half, half, -half)
            glVertex3f(-half, half, half)
            glVertex3f(half, half, half)
            glVertex3f(half, half, -half)
            # Bottom
            glNormal3f(0, -1, 0)
            glVertex3f(-half, -half, -half)
            glVertex3f(half, -half, -half)
            glVertex3f(half, -half, half)
            glVertex3f(-half, -half, half)
            glEnd()
        
        glPopMatrix()
    
    glDisable(GL_COLOR_MATERIAL)
    glDisable(GL_BLEND)


def render_fallen_trees(fallen_trees: list, camera_x: float, camera_y: float, camera_z: float):
    """Render fallen tree trunks that tip over after cutting."""
    if not fallen_trees:
        return
    
    max_render_dist = 100.0
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    for fallen in fallen_trees:
        dx = fallen.x - camera_x
        dy = fallen.y - camera_y
        dz = fallen.z - camera_z
        dist_sq = dx*dx + dy*dy + dz*dz
        
        if dist_sq > max_render_dist * max_render_dist:
            continue
        
        if fallen.alpha <= 0:
            continue
        
        glPushMatrix()
        glTranslatef(fallen.x, fallen.y, fallen.z)
        
        # Rotate to fall direction
        glRotatef(fallen.fall_angle, 0, 1, 0)
        # Tilt over
        glRotatef(fallen.tilt, 1, 0, 0)
        
        # Trunk color with alpha
        r, g, b = fallen.trunk_color
        glColor4f(r, g, b, fallen.alpha)
        
        # Draw trunk as a cylinder
        radius = fallen.radius
        height = fallen.height
        segments = 8
        
        # Draw cylinder using quads
        for i in range(segments):
            angle1 = (i / segments) * 2 * math.pi
            angle2 = ((i + 1) / segments) * 2 * math.pi
            
            x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
            x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
            
            glBegin(GL_QUADS)
            glNormal3f(math.cos(angle1), 0, math.sin(angle1))
            glVertex3f(x1, 0, z1)
            glVertex3f(x1, height, z1)
            glNormal3f(math.cos(angle2), 0, math.sin(angle2))
            glVertex3f(x2, height, z2)
            glVertex3f(x2, 0, z2)
            glEnd()
        
        # Top cap (lighter wood color)
        glColor4f(r * 1.3, g * 1.3, b * 1.2, fallen.alpha)
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(0, 1, 0)
        glVertex3f(0, height, 0)
        for i in range(segments + 1):
            angle = (i / segments) * 2 * math.pi
            glVertex3f(math.cos(angle) * radius, height, math.sin(angle) * radius)
        glEnd()
        
        glPopMatrix()
    
    glDisable(GL_COLOR_MATERIAL)
    glDisable(GL_BLEND)


def render_water(camera_x: float, camera_z: float, water_level: float = 0.0):
    """Render water plane centered on camera at water level height."""
    size = 2000  # Large enough to cover visible area
    
    # Scale water level
    water_y = water_level * HEIGHT_SCALE
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_LIGHTING)
    
    glNormal3f(0, 1, 0)
    glColor4f(0.08, 0.30, 0.55, 0.75)  # Deep blue, translucent
    
    # Center water on camera position
    glBegin(GL_QUADS)
    glVertex3f(camera_x - size, water_y, camera_z - size)
    glVertex3f(camera_x + size, water_y, camera_z - size)
    glVertex3f(camera_x + size, water_y, camera_z + size)
    glVertex3f(camera_x - size, water_y, camera_z + size)
    glEnd()
    
    # Add subtle underwater tint when below water
    glEnable(GL_LIGHTING)
    glDisable(GL_BLEND)


def draw_crosshair(display: tuple):
    """Draw crosshair."""
    cx, cy = display[0] // 2, display[1] // 2
    size = 12
    
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], display[1], 0, -1, 1)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    
    glColor3f(0, 0, 0)
    glLineWidth(3)
    glBegin(GL_LINES)
    glVertex2f(cx - size, cy); glVertex2f(cx + size, cy)
    glVertex2f(cx, cy - size); glVertex2f(cx, cy + size)
    glEnd()
    
    glColor3f(1, 1, 1)
    glLineWidth(1.5)
    glBegin(GL_LINES)
    glVertex2f(cx - size, cy); glVertex2f(cx + size, cy)
    glVertex2f(cx, cy - size); glVertex2f(cx, cy + size)
    glEnd()
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


# Text rendering cache for OpenGL
_text_textures = {}

# Entity cache for log preview (caches reconstructed DNA and entities)
_log_entity_cache = {}
_preview_rotation = 0.0  # Animation rotation angle


def _reconstruct_gene(d: dict):
    """Reconstruct a Gene from a dict."""
    from waverse.dna import Gene
    return Gene(
        value=d['value'],
        min_val=d['min_val'],
        max_val=d['max_val'],
        mutation_rate=d['mutation_rate']
    )


def _reconstruct_color_gene(d: dict):
    """Reconstruct a ColorGene from a dict."""
    from waverse.dna import ColorGene
    return ColorGene(
        r=d['r'],
        g=d['g'],
        b=d['b'],
        mutation_rate=d.get('mutation_rate', 0.15)
    )


def _reconstruct_segment_gene(d: dict):
    """Reconstruct a SegmentGene from a dict."""
    from waverse.dna import SegmentGene
    return SegmentGene(
        length=d['length'],
        width=d['width'],
        taper=d['taper'],
        curve=d['curve'],
        twist=d['twist']
    )


def _reconstruct_plant_dna(d: dict):
    """Reconstruct PlantDNA from a dict (loaded from JSON)."""
    from waverse.dna import PlantDNA
    
    return PlantDNA(
        plant_type=d.get('plant_type', 'tree'),
        species_id=d.get('species_id', 0),
        generation=d.get('generation', 0),
        height_gene=_reconstruct_gene(d['height_gene']),
        width_gene=_reconstruct_gene(d['width_gene']),
        trunk_segments=[_reconstruct_segment_gene(s) for s in d.get('trunk_segments', [])],
        branch_count=int(d.get('branch_count', 4)),
        branch_angle=d.get('branch_angle', 0.4),
        branch_spread=d.get('branch_spread', 1.0),
        branch_height=d.get('branch_height', 0.6),
        branch_segments=[_reconstruct_segment_gene(s) for s in d.get('branch_segments', [])],
        sub_branch_chance=d.get('sub_branch_chance', 0.3),
        leaf_density=d.get('leaf_density', 0.7),
        leaf_size=d.get('leaf_size', 0.5),
        leaf_shape=d.get('leaf_shape', 'round'),
        canopy_shape=d.get('canopy_shape', 'dome'),
        canopy_spread=d.get('canopy_spread', 0.5),
        has_flowers=d.get('has_flowers', False),
        flower_size=d.get('flower_size', 0.0),
        has_fruit=d.get('has_fruit', False),
        fruit_size=d.get('fruit_size', 0.0),
        has_glow=d.get('has_glow', False),
        glow_intensity=d.get('glow_intensity', 0.0),
        has_thorns=d.get('has_thorns', False),
        trunk_color=_reconstruct_color_gene(d['trunk_color']),
        leaf_color=_reconstruct_color_gene(d['leaf_color']),
        flower_color=_reconstruct_color_gene(d['flower_color']),
        glow_color=_reconstruct_color_gene(d['glow_color']),
        asymmetry=d.get('asymmetry', 0.1),
        droop=d.get('droop', 0.0),
        wind_sway=d.get('wind_sway', 0.3),
        recursive_depth=d.get('recursive_depth', 1),
        growth_direction=d.get('growth_direction', 0.0),
        spiral_factor=d.get('spiral_factor', 0.0),
        bulb_count=d.get('bulb_count', 0),
        bulb_size=d.get('bulb_size', 0.0),
        bark_texture=d.get('bark_texture', 'smooth'),
        surface_bumps=d.get('surface_bumps', 0.0),
        has_moss=d.get('has_moss', False),
        moss_density=d.get('moss_density', 0.0),
        secondary_trunk_color=_reconstruct_color_gene(d.get('secondary_trunk_color', {'r': 0.3, 'g': 0.22, 'b': 0.12})),
        tip_color=_reconstruct_color_gene(d.get('tip_color', {'r': 0.4, 'g': 0.6, 'b': 0.3})),
        fruit_color=_reconstruct_color_gene(d.get('fruit_color', {'r': 0.8, 'g': 0.2, 'b': 0.2})),
        moss_color=_reconstruct_color_gene(d.get('moss_color', {'r': 0.2, 'g': 0.4, 'b': 0.15})),
        bioluminescent=d.get('bioluminescent', False),
        crystal_growth=d.get('crystal_growth', False),
        spore_pods=d.get('spore_pods', False),
        tendrils=d.get('tendrils', 0),
        root_exposure=d.get('root_exposure', 0.0),
        upside_down=d.get('upside_down', False),
        lean_angle=d.get('lean_angle', 0.0),
    )


def _reconstruct_body_segment(d: dict):
    """Reconstruct a BodySegment from a dict."""
    from waverse.animal_dna import BodySegment
    size = d.get('size', (1, 1, 1))
    if isinstance(size, list):
        size = tuple(size)
    return BodySegment(
        size=size,
        shape=d.get('shape', 'sphere'),
        offset=tuple(d.get('offset', (0, 0, 0))) if isinstance(d.get('offset'), list) else d.get('offset', (0, 0, 0)),
        color_index=d.get('color_index', 0),
    )


def _reconstruct_limb(d: dict):
    """Reconstruct a Limb from a dict."""
    from waverse.animal_dna import Limb
    return Limb(
        limb_type=d.get('limb_type', 'leg'),
        segments=d.get('segments', 2),
        segment_length=d.get('segment_length', 1.0),
        segment_width=d.get('segment_width', 0.2),
        attachment_point=tuple(d.get('attachment_point', (0, 0, 0))) if isinstance(d.get('attachment_point'), list) else d.get('attachment_point', (0, 0, 0)),
        attachment_angle=d.get('attachment_angle', 0.0),
        spread_angle=d.get('spread_angle', 0.0),
        mirror=d.get('mirror', True),
        animation_phase_offset=d.get('animation_phase_offset', 0.0),
    )


def _reconstruct_feature(d: dict):
    """Reconstruct a Feature from a dict."""
    from waverse.animal_dna import Feature
    return Feature(
        feature_type=d.get('feature_type', 'eye'),
        size=d.get('size', 0.3),
        position=tuple(d.get('position', (0, 0, 0))) if isinstance(d.get('position'), list) else d.get('position', (0, 0, 0)),
        color_index=d.get('color_index', 0),
        count=d.get('count', 2),
        spread=d.get('spread', 0.3),
    )


def _reconstruct_animal_dna(d: dict):
    """Reconstruct AnimalDNA from a dict (loaded from JSON)."""
    from waverse.animal_dna import AnimalDNA
    
    return AnimalDNA(
        animal_type=d.get('animal_type', 'mammal'),
        species_id=d.get('species_id', 0),
        generation=d.get('generation', 0),
        body_segments=[_reconstruct_body_segment(s) for s in d.get('body_segments', [])],
        limbs=[_reconstruct_limb(l) for l in d.get('limbs', [])],
        features=[_reconstruct_feature(f) for f in d.get('features', [])],
        movement_style=d.get('movement_style', 'walk'),
        movement_speed=d.get('movement_speed', 1.0),
        animation_speed=d.get('animation_speed', 1.0),
        primary_color=_reconstruct_color_gene(d.get('primary_color', {'r': 0.5, 'g': 0.4, 'b': 0.3})),
        secondary_color=_reconstruct_color_gene(d.get('secondary_color', {'r': 0.6, 'g': 0.5, 'b': 0.4})),
        accent_color=_reconstruct_color_gene(d.get('accent_color', {'r': 0.2, 'g': 0.2, 'b': 0.2})),
        eye_color=_reconstruct_color_gene(d.get('eye_color', {'r': 0.1, 'g': 0.1, 'b': 0.1})),
        pattern_type=d.get('pattern_type', 'solid'),
        pattern_scale=d.get('pattern_scale', 1.0),
        has_tail=d.get('has_tail', False),
        tail_length=d.get('tail_length', 0.0),
        tail_segments=d.get('tail_segments', 3),
        has_shell=d.get('has_shell', False),
        shell_coverage=d.get('shell_coverage', 0.0),
        has_spikes=d.get('has_spikes', False),
        spike_density=d.get('spike_density', 0.0),
        has_glow=d.get('has_glow', False),
        glow_intensity=d.get('glow_intensity', 0.0),
        glow_color=_reconstruct_color_gene(d.get('glow_color', {'r': 0.5, 'g': 0.8, 'b': 0.5})),
        overall_scale=d.get('overall_scale', 1.0),
    )


def _load_log_entity(json_path: str):
    """Load and reconstruct an entity from a DNA log JSON file.
    
    Returns: (entity_type, entity) or (None, None) on error.
    entity is a PlantInstance or AnimalInstance.
    """
    global _log_entity_cache
    
    if json_path in _log_entity_cache:
        return _log_entity_cache[json_path]
    
    try:
        with open(json_path, 'r') as f:
            log_entry = json.load(f)
        
        entity_type = log_entry.get('entity_type', 'plant')
        dna_dict = log_entry.get('dna', log_entry)  # Support both wrapped and raw DNA
        
        if entity_type == 'plant':
            dna = _reconstruct_plant_dna(dna_dict)
            from waverse.flora import PlantInstance
            entity = PlantInstance(
                x=0, y=0, z=0,
                dna=dna,
                scale=1.0,
                rotation=0
            )
        else:
            dna = _reconstruct_animal_dna(dna_dict)
            from waverse.animals import AnimalInstance
            entity = AnimalInstance(
                x=0, y=0, z=0,
                dna=dna,
                rotation=0,
                anim_time=0.0,
                anim_phase=0.0
            )
        
        _log_entity_cache[json_path] = (entity_type, entity)
        return entity_type, entity
        
    except Exception as e:
        print(f"  Error loading DNA log: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def draw_entity_preview(json_path: str, x: float, y: float, size: float, dt: float = 0.016):
    """Render an entity preview in a small 3D viewport.
    
    The entity slowly rotates for a more natural, animated look.
    Can be called standalone (doesn't require being inside 2D ortho mode).
    
    Note: Uses legacy OpenGL - won't work on macOS Core profile.
    """
    import platform
    
    # Skip on macOS Core profile - legacy GL calls don't work
    if platform.system() == 'Darwin' and MODERNGL_AVAILABLE:
        # Can't render preview with legacy GL on Core profile
        return
    
    global _preview_rotation
    from waverse.flora import PlantRenderer, PlantInstance
    from waverse.animals import AnimalRenderer
    
    # Update rotation animation (slow spin)
    _preview_rotation += dt * 15.0  # 15 degrees per second
    if _preview_rotation > 360:
        _preview_rotation -= 360
    
    # Load entity from cache or file
    entity_type, entity = _load_log_entity(json_path)
    if entity is None:
        return
    
    # Save viewport and set up preview viewport
    viewport = glGetIntegerv(GL_VIEWPORT)
    preview_size = int(size)
    preview_x = int(x)
    preview_y = int(viewport[3] - y - size)  # Flip Y for OpenGL
    glViewport(preview_x, preview_y, preview_size, preview_size)
    
    # Set up 3D projection for preview
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    
    # Determine entity dimensions for framing
    if entity_type == "plant":
        entity_height = entity.dna.height_gene.value * entity.scale
        entity_width = max(entity.dna.width_gene.value * entity.scale * 2, entity_height * 0.5)
        center_y = entity_height * 0.4
        view_bottom_mult = 0.3
        view_top_mult = 1.7
    else:
        entity_height = 3.0
        entity_width = 3.0
        if hasattr(entity.dna, 'body_segments') and entity.dna.body_segments:
            total_size = sum(max(seg.size) if isinstance(seg.size, tuple) else seg.size 
                           for seg in entity.dna.body_segments)
            entity_height = total_size * 2.5
            entity_width = total_size * 3.0
        center_y = entity_height * 0.3
        view_bottom_mult = 1.8
        view_top_mult = 0.8
    
    view_size = max(entity_height, entity_width, 5.0) * 2.0
    half_size = view_size / 2
    
    glOrtho(-half_size, half_size, -half_size * view_bottom_mult, half_size * view_top_mult, -100, 100)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    # Camera position - orbits around the entity
    cam_dist = half_size * 1.0
    cam_angle = math.radians(_preview_rotation)
    cam_x = math.sin(cam_angle) * cam_dist
    cam_z = math.cos(cam_angle) * cam_dist
    cam_y = half_size * 0.3
    
    gluLookAt(
        cam_x, cam_y, cam_z,  # Eye position (orbiting)
        0, center_y, 0,       # Look at center of entity
        0, 1, 0               # Up vector
    )
    
    # Clear just the preview viewport area (depth only, keep color from background)
    glClear(GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)
    
    # Render the entity at origin
    if entity_type == "plant":
        # Update animation phase for plants
        entity.rotation = 0  # Let camera do the rotating
        PlantRenderer.draw_full(entity)
    else:
        # Update animation for animals
        entity.anim_time += dt
        entity.anim_phase = math.sin(entity.anim_time * 2.0) * 0.5 + 0.5
        AnimalRenderer.draw_full(entity, at_origin=True)
    
    glFlush()
    
    # Restore viewport
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3])
    
    # Restore matrices
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def _draw_text(font, text: str, x: float, y: float, color: tuple):
    """Draw text at position using pygame font and OpenGL texture."""
    global _text_textures
    
    # Create cache key
    cache_key = (text, color)
    
    if cache_key not in _text_textures:
        # Render text to surface
        text_surface = font.render(text, True, color)
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        width, height = text_surface.get_size()
        
        # Create OpenGL texture
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        _text_textures[cache_key] = (texture_id, width, height)
    
    texture_id, width, height = _text_textures[cache_key]
    
    # Draw textured quad
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glColor4f(1, 1, 1, 1)
    
    glBegin(GL_QUADS)
    glTexCoord2f(0, 0); glVertex2f(x, y + height)
    glTexCoord2f(1, 0); glVertex2f(x + width, y + height)
    glTexCoord2f(1, 1); glVertex2f(x + width, y)
    glTexCoord2f(0, 1); glVertex2f(x, y)
    glEnd()
    
    glDisable(GL_TEXTURE_2D)


def _draw_number(x: float, y: float, value: float):
    """Draw a number using simple line segments (fallback when no font)."""
    # Format number
    text = f"{value:.1f}"
    char_w = 8
    
    for i, char in enumerate(text):
        cx = x + i * char_w
        _draw_digit(cx, y, char)


def _draw_digit(x: float, y: float, char: str):
    """Draw a single digit/character using line segments."""
    # 7-segment style digits
    h = 10
    w = 6
    
    segments = {
        '0': [(0,0,w,0), (w,0,w,h/2), (w,h/2,w,h), (w,h,0,h), (0,h,0,h/2), (0,h/2,0,0)],
        '1': [(w/2,0,w/2,h)],
        '2': [(0,0,w,0), (w,0,w,h/2), (w,h/2,0,h/2), (0,h/2,0,h), (0,h,w,h)],
        '3': [(0,0,w,0), (w,0,w,h/2), (w,h/2,0,h/2), (w,h/2,w,h), (w,h,0,h)],
        '4': [(0,0,0,h/2), (0,h/2,w,h/2), (w,0,w,h)],
        '5': [(w,0,0,0), (0,0,0,h/2), (0,h/2,w,h/2), (w,h/2,w,h), (w,h,0,h)],
        '6': [(w,0,0,0), (0,0,0,h), (0,h,w,h), (w,h,w,h/2), (w,h/2,0,h/2)],
        '7': [(0,0,w,0), (w,0,w,h)],
        '8': [(0,0,w,0), (w,0,w,h), (w,h,0,h), (0,h,0,0), (0,h/2,w,h/2)],
        '9': [(w,h,w,0), (w,0,0,0), (0,0,0,h/2), (0,h/2,w,h/2)],
        '.': [(w/2-1,h-2,w/2+1,h)],
        '-': [(0,h/2,w,h/2)],
        ' ': [],
    }
    
    segs = segments.get(char, [])
    glBegin(GL_LINES)
    for seg in segs:
        glVertex2f(x + seg[0], y + seg[1])
        glVertex2f(x + seg[2], y + seg[3])
    glEnd()


def draw_menu(display: tuple, camera: Camera, hud_font=None):
    """Draw the menu overlay when open."""
    if not camera.menu_open:
        return
    
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], display[1], 0, -1, 1)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Semi-transparent background overlay
    glColor4f(0, 0, 0, 0.7)
    glBegin(GL_QUADS)
    glVertex2f(0, 0)
    glVertex2f(display[0], 0)
    glVertex2f(display[0], display[1])
    glVertex2f(0, display[1])
    glEnd()
    
    # Menu box - wider to show images
    menu_w = min(800, display[0] - 50)
    menu_h = min(500, display[1] - 50)
    menu_x = (display[0] - menu_w) / 2
    menu_y = (display[1] - menu_h) / 2
    
    # Menu background
    glColor4f(0.15, 0.15, 0.2, 0.95)
    glBegin(GL_QUADS)
    glVertex2f(menu_x, menu_y)
    glVertex2f(menu_x + menu_w, menu_y)
    glVertex2f(menu_x + menu_w, menu_y + menu_h)
    glVertex2f(menu_x, menu_y + menu_h)
    glEnd()
    
    # Menu border
    glColor4f(0.4, 0.6, 0.8, 1.0)
    glLineWidth(2)
    glBegin(GL_LINE_LOOP)
    glVertex2f(menu_x, menu_y)
    glVertex2f(menu_x + menu_w, menu_y)
    glVertex2f(menu_x + menu_w, menu_y + menu_h)
    glVertex2f(menu_x, menu_y + menu_h)
    glEnd()
    
    # Tab bar
    tab_h = 35
    tab_names = ["INVENTORY", "LOG", "CONTROLS"]
    tab_w = menu_w / len(tab_names)
    
    for i, name in enumerate(tab_names):
        tx = menu_x + i * tab_w
        
        # Tab background
        if i == camera.menu_tab:
            glColor4f(0.3, 0.5, 0.7, 1.0)  # Selected
        else:
            glColor4f(0.2, 0.25, 0.3, 1.0)  # Unselected
        
        glBegin(GL_QUADS)
        glVertex2f(tx + 2, menu_y + 2)
        glVertex2f(tx + tab_w - 2, menu_y + 2)
        glVertex2f(tx + tab_w - 2, menu_y + tab_h)
        glVertex2f(tx + 2, menu_y + tab_h)
        glEnd()
        
        # Tab text
        if hud_font:
            _draw_text(hud_font, name, tx + tab_w/2 - 30, menu_y + 12, (255, 255, 255))
    
    # Content area
    content_y = menu_y + tab_h + 10
    content_h = menu_h - tab_h - 20
    
    if camera.menu_tab == 0:  # Inventory
        if hud_font:
            _draw_text(hud_font, "Inventory (Coming Soon)", menu_x + 20, content_y, (200, 200, 200))
    
    elif camera.menu_tab == 1:  # Log
        # Draw filter header
        filter_y = content_y
        if hud_font:
            filter_text = f"Filter: [{camera.log_filter_names[camera.log_filter]}]  (D-PAD L/R to change)"
            _draw_text(hud_font, filter_text, menu_x + 20, filter_y, (150, 200, 255))
        content_y += 25
        content_h -= 25
        
        if len(camera.log_items) == 0:
            if hud_font:
                _draw_text(hud_font, "No DNA logs yet. Use SCAN tool (R or R1) to log creatures!", 
                          menu_x + 20, content_y, (200, 200, 200))
        else:
            # Split: list on left (narrower), image preview on right (wider)
            list_width = menu_w * 0.4
            image_x = menu_x + list_width + 10
            image_size = min(content_h - 10, menu_w * 0.55)  # Wider image
            
            # Show list of log items on left
            item_h = 24
            visible_items = int(content_h / item_h)
            start_idx = max(0, camera.log_index - visible_items // 2)
            
            for i, item in enumerate(camera.log_items[start_idx:start_idx + visible_items]):
                iy = content_y + i * item_h
                
                # Highlight selected item
                if start_idx + i == camera.log_index:
                    glColor4f(0.3, 0.5, 0.7, 0.8)
                    glBegin(GL_QUADS)
                    glVertex2f(menu_x + 10, iy - 2)
                    glVertex2f(menu_x + list_width - 10, iy - 2)
                    glVertex2f(menu_x + list_width - 10, iy + item_h - 4)
                    glVertex2f(menu_x + 10, iy + item_h - 4)
                    glEnd()
                
                # Item text with star for favorites
                if hud_font:
                    is_favorite = item in camera.favorites
                    parts = item.replace('.json', '').split('_')
                    if len(parts) >= 3:
                        entity_type = parts[0]
                        date = parts[2] if len(parts) > 2 else ""
                        time = parts[3] if len(parts) > 3 else ""
                        display_text = f"{entity_type.upper()} - {date} {time}"
                    else:
                        display_text = item
                    
                    # Add star for favorites
                    if is_favorite:
                        display_text = "★ " + display_text
                    
                    color = (255, 215, 0) if is_favorite else (100, 200, 255) if start_idx + i == camera.log_index else (180, 180, 180)
                    _draw_text(hud_font, display_text, menu_x + 20, iy, color)
            
            # Draw live 3D entity preview on right side
            if 0 <= camera.log_index < len(camera.log_items):
                selected_item = camera.log_items[camera.log_index]
                json_path = os.path.join(DNA_LOGS_DIR, selected_item)
                
                if os.path.exists(json_path):
                    # Use dt from frame time (approximate at 60fps)
                    draw_entity_preview(json_path, image_x, content_y, image_size, dt=0.016)
    
    elif camera.menu_tab == 2:  # Controls
        if hud_font:
            line_h = 20
            left_col = menu_x + 20
            right_col = menu_x + menu_w / 2 + 20
            y = content_y
            
            # Keyboard controls (left column)
            _draw_text(hud_font, "=== KEYBOARD ===", left_col, y, (100, 200, 255))
            y += line_h + 5
            
            controls_kb = [
                ("WASD / Arrows", "Move"),
                ("IJKL", "Look around"),
                ("H / Space", "Go up / Jump"),
                ("F / Shift", "Go down"),
                ("[ / -", "Slower speed"),
                ("] / =", "Faster speed"),
                ("B (hold)", "Run (2x speed)"),
                ("R", "Use tool (ACTION)"),
                (", / .", "Switch tool"),
                ("T", "Cycle tool radius"),
                ("P", "Screenshot"),
                ("Tab", "Menu"),
                ("X", "Tour mode"),
                ("N", "Warp to new world"),
                ("M", "Set marker"),
                ("C", "Clear markers"),
                ("O", "Save position"),
            ]
            
            for key, action in controls_kb:
                _draw_text(hud_font, f"{key}: {action}", left_col, y, (200, 200, 200))
                y += line_h
            
            # Gamepad controls (right column)
            y = content_y
            _draw_text(hud_font, "=== GAMEPAD ===", right_col, y, (100, 200, 255))
            y += line_h + 5
            
            controls_gp = [
                ("Left Stick", "Move"),
                ("Right Stick", "Look around"),
                ("A", "Action/Talk (future)"),
                ("B", "Jump (hold=higher)"),
                ("X", "Toggle fly/walk"),
                ("Y", "Screenshot"),
                ("L2 / R2", "Speed down / up"),
                ("R1", "Use tool"),
                ("L3 (hold)", "Run (2x speed)"),
                ("D-PAD ←→", "Cycle tools"),
                ("D-PAD ↑↓", "Tool radius / Menu nav"),
                ("START", "Menu"),
                ("SELECT", "Warp to new world"),
            ]
            
            for btn, action in controls_gp:
                _draw_text(hud_font, f"{btn}: {action}", right_col, y, (200, 200, 200))
                y += line_h
    
    # Help text at bottom (different per tab)
    if hud_font:
        help_y = menu_y + menu_h - 25
        if camera.menu_tab == 1:  # LOG tab
            help_text = "UP/DN: Scroll | L/R: Filter | A: Favorite | SEL: Unfav | L1/R1: Tabs | START: Close"
        else:
            help_text = "UP/DN: Navigate | L1/R1: Tabs | START: Close"
        _draw_text(hud_font, help_text, menu_x + 20, help_y, (150, 150, 150))
    
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def draw_perf_overlay(display: tuple, perf: PerfMonitor, hud_font=None):
    """Draw mini performance overlay in top-right corner."""
    if not perf.show_overlay:
        return
    
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], display[1], 0, -1, 1)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Background box
    box_w, box_h = 180, 140
    box_x = display[0] - box_w - 10
    box_y = 10
    
    glColor4f(0.0, 0.0, 0.0, 0.7)
    glBegin(GL_QUADS)
    glVertex2f(box_x, box_y)
    glVertex2f(box_x + box_w, box_y)
    glVertex2f(box_x + box_w, box_y + box_h)
    glVertex2f(box_x, box_y + box_h)
    glEnd()
    
    if hud_font:
        y = box_y + 8
        line_h = 16
        
        # FPS
        avg_frame = perf.get_avg(perf.frame_times)
        fps = 1000 / avg_frame if avg_frame > 0 else 0
        color = (0, 255, 0) if fps >= 50 else (255, 255, 0) if fps >= 30 else (255, 100, 100)
        _draw_text(hud_font, f"FPS: {fps:.0f} ({avg_frame:.1f}ms)", box_x + 8, y, color)
        y += line_h
        
        # Life sim
        life_avg = perf.get_avg(perf.life_times)
        color = (0, 255, 0) if life_avg < 5 else (255, 255, 0) if life_avg < 15 else (255, 100, 100)
        _draw_text(hud_font, f"Life: {life_avg:.1f}ms", box_x + 8, y, color)
        y += line_h
        
        # Render
        render_avg = perf.get_avg(perf.render_times)
        _draw_text(hud_font, f"Render: {render_avg:.1f}ms", box_x + 8, y, (200, 200, 200))
        y += line_h
        
        # Entity counts
        _draw_text(hud_font, f"Plants: {perf.plant_count}", box_x + 8, y, (100, 200, 100))
        y += line_h
        _draw_text(hud_font, f"Animals: {perf.animal_count}", box_x + 8, y, (200, 150, 100))
        y += line_h
        _draw_text(hud_font, f"Chunks: {perf.chunk_count}", box_x + 8, y, (100, 150, 200))
        y += line_h
        
        # Slow frames
        color = (0, 255, 0) if perf.slow_frames < 5 else (255, 255, 0) if perf.slow_frames < 20 else (255, 100, 100)
        _draw_text(hud_font, f"Slow: {perf.slow_frames}", box_x + 8, y, color)
    
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def draw_hud(display: tuple, camera: Camera, sky: SkySystem = None, climate: ClimateManager = None,
              hud_font=None):
    """Draw HUD."""
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], display[1], 0, -1, 1)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # ========== TOP BAR: Coordinates + Compass ==========
    # All elements use same top margin (10) as minimap
    top_margin = 10
    top_bar_h = 36
    
    # Coordinates background (left side, after minimap ~190px)
    # Expanded to show X, Y, Z positions and dx, dy, dz velocities
    coord_x = 200
    coord_w = 220  # Wider to fit position + velocity columns
    coord_h = 52   # Taller for 3 rows (X, Y, Z)
    glColor4f(0, 0, 0, 0.6)
    glBegin(GL_QUADS)
    glVertex2f(coord_x, top_margin)
    glVertex2f(coord_x + coord_w, top_margin)
    glVertex2f(coord_x + coord_w, top_margin + coord_h)
    glVertex2f(coord_x, top_margin + coord_h)
    glEnd()
    
    # Draw coordinate text using pygame font if available
    # Format: X: 123.4  dx: 0.5
    #         Y: 567.8  dy: 0.0
    #         Z:  45.2  dz: 0.3
    if hud_font:
        # Scale velocity for display (multiply by ~60 for per-second)
        vx = camera.velocity_x * 60
        vy = camera.velocity_y * 60
        vz = camera.velocity_z * 60
        
        # Position column
        x_text = f"{camera.x:7.1f}"
        y_text = f"{camera.z:7.1f}"  # Z is the "Y" in top-down view
        z_text = f"{camera.y:7.1f}"  # Y is height, shown as Z
        
        # Velocity column
        dx_text = f"∂X: {vx:+5.1f}"
        dy_text = f"∂Y: {vz:+5.1f}"
        dz_text = f"∂Z: {vy:+5.1f}"
        
        _draw_text(hud_font, x_text, coord_x + 5, top_margin + 4, (255, 180, 100))
        _draw_text(hud_font, dx_text, coord_x + 115, top_margin + 4, (200, 150, 80))
        _draw_text(hud_font, y_text, coord_x + 5, top_margin + 18, (100, 200, 255))
        _draw_text(hud_font, dy_text, coord_x + 115, top_margin + 18, (80, 160, 200))
        _draw_text(hud_font, z_text, coord_x + 5, top_margin + 32, (150, 255, 150))
        _draw_text(hud_font, dz_text, coord_x + 115, top_margin + 32, (120, 200, 120))
    else:
        # Fallback: draw simple coordinate indicators
        glColor4f(1.0, 0.7, 0.4, 1.0)
        _draw_number(coord_x + 10, 18, camera.x)
        glColor4f(0.4, 0.8, 1.0, 1.0)
        _draw_number(coord_x + 10, 32, camera.z)
    
    # Compass background (adjusted position for wider coord box)
    compass_x = coord_x + coord_w + 10
    compass_w = 350  # Slightly narrower to fit
    compass_y = top_margin  # Same top margin
    compass_h = top_bar_h
    compass_center_y = compass_y + compass_h / 2  # Vertical center of compass
    
    glColor4f(0, 0, 0, 0.6)
    glBegin(GL_QUADS)
    glVertex2f(compass_x, compass_y)
    glVertex2f(compass_x + compass_w, compass_y)
    glVertex2f(compass_x + compass_w, compass_y + compass_h)
    glVertex2f(compass_x, compass_y + compass_h)
    glEnd()
    
    # Compass center line
    compass_center = compass_x + compass_w / 2
    glColor4f(0.5, 0.5, 0.5, 0.8)
    glBegin(GL_LINES)
    glVertex2f(compass_x + 10, compass_center_y)
    glVertex2f(compass_x + compass_w - 10, compass_center_y)
    glEnd()
    
    # Center tick mark
    glColor4f(1.0, 1.0, 1.0, 0.9)
    glBegin(GL_LINES)
    glVertex2f(compass_center, compass_center_y - 6)
    glVertex2f(compass_center, compass_center_y + 6)
    glEnd()
    
    # North and South markers - inverted so turning right moves markers left
    yaw = camera.yaw
    
    # Helper to draw compass direction marker
    def draw_compass_marker(direction_angle, letter, fill_color, outline_color):
        diff = direction_angle - yaw
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        
        if abs(diff) < 90:
            # INVERTED: negative sign so turning right moves marker left
            pos_x = compass_center - (diff / 90) * (compass_w / 2 - 30)
            cy = compass_center_y  # Use compass vertical center
            radius = 12
            
            # Filled circle
            glColor4f(*fill_color)
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(pos_x, cy)
            for i in range(13):
                angle = (i / 12) * 2 * math.pi
                glVertex2f(pos_x + math.cos(angle) * radius, cy + math.sin(angle) * radius)
            glEnd()
            
            # Outline
            glColor4f(*outline_color)
            glBegin(GL_LINE_LOOP)
            for i in range(12):
                angle = (i / 12) * 2 * math.pi
                glVertex2f(pos_x + math.cos(angle) * radius, cy + math.sin(angle) * radius)
            glEnd()
            
            # Letter
            if letter == "N":
                glBegin(GL_LINES)
                glVertex2f(pos_x - 4, cy + 4)
                glVertex2f(pos_x - 4, cy - 4)
                glVertex2f(pos_x - 4, cy - 4)
                glVertex2f(pos_x + 4, cy + 4)
                glVertex2f(pos_x + 4, cy + 4)
                glVertex2f(pos_x + 4, cy - 4)
                glEnd()
            elif letter == "S":
                # Draw "S" shape
                glBegin(GL_LINE_STRIP)
                glVertex2f(pos_x + 3, cy - 4)
                glVertex2f(pos_x - 3, cy - 4)
                glVertex2f(pos_x - 3, cy)
                glVertex2f(pos_x + 3, cy)
                glVertex2f(pos_x + 3, cy + 4)
                glVertex2f(pos_x - 3, cy + 4)
                glEnd()
    
    # North (yaw 0) - red
    draw_compass_marker(0, "N", (1.0, 0.3, 0.3, 0.9), (1, 1, 1, 1))
    # South (yaw 180) - blue
    draw_compass_marker(180, "S", (0.3, 0.5, 0.9, 0.9), (1, 1, 1, 1))
    
    # Draw markers on compass (labeled A, B, C...)
    marker_labels = "ABCDEFGHIJ"
    cy = compass_center_y  # Vertical center
    for idx, (mx, mz, mcolor, mname) in enumerate(camera.markers):
        # Calculate angle to marker
        dx = mx - camera.x
        dz = mz - camera.z
        marker_angle = math.degrees(math.atan2(-dx, -dz))  # Angle to marker
        
        # Difference from current yaw
        diff = marker_angle - yaw
        while diff > 180: diff -= 360
        while diff < -180: diff += 360
        
        # Show if within range
        if abs(diff) < 90:
            # INVERTED: negative sign for consistent compass behavior
            pos_x = compass_center - (diff / 90) * (compass_w / 2 - 30)
            
            # Colored marker circle
            glColor4f(*mcolor, 0.9)
            radius = 10
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(pos_x, cy)
            for j in range(13):
                angle = (j / 12) * 2 * math.pi
                glVertex2f(pos_x + math.cos(angle) * radius, cy + math.sin(angle) * radius)
            glEnd()
            
            # White outline
            glColor4f(1, 1, 1, 1)
            glBegin(GL_LINE_LOOP)
            for j in range(12):
                angle = (j / 12) * 2 * math.pi
                glVertex2f(pos_x + math.cos(angle) * radius, cy + math.sin(angle) * radius)
            glEnd()
            
            # Draw letter (A, B, C...) - simplified letter shapes
            label = marker_labels[idx] if idx < len(marker_labels) else "?"
            if label == "A":
                glBegin(GL_LINES)
                glVertex2f(pos_x, cy - 5)
                glVertex2f(pos_x - 4, cy + 5)
                glVertex2f(pos_x, cy - 5)
                glVertex2f(pos_x + 4, cy + 5)
                glVertex2f(pos_x - 2, cy + 1)
                glVertex2f(pos_x + 2, cy + 1)
                glEnd()
            elif label == "B":
                glBegin(GL_LINES)
                glVertex2f(pos_x - 3, cy - 5)
                glVertex2f(pos_x - 3, cy + 5)
                glVertex2f(pos_x - 3, cy - 5)
                glVertex2f(pos_x + 2, cy - 5)
                glVertex2f(pos_x - 3, cy)
                glVertex2f(pos_x + 2, cy)
                glVertex2f(pos_x - 3, cy + 5)
                glVertex2f(pos_x + 2, cy + 5)
                glEnd()
            elif label == "C":
                glBegin(GL_LINE_STRIP)
                glVertex2f(pos_x + 3, cy - 4)
                glVertex2f(pos_x - 2, cy - 4)
                glVertex2f(pos_x - 3, cy)
                glVertex2f(pos_x - 2, cy + 4)
                glVertex2f(pos_x + 3, cy + 4)
                glEnd()
            else:
                # Simple dot for other letters
                glBegin(GL_QUADS)
                glVertex2f(pos_x - 2, cy - 2)
                glVertex2f(pos_x + 2, cy - 2)
                glVertex2f(pos_x + 2, cy + 2)
                glVertex2f(pos_x - 2, cy + 2)
                glEnd()
    
    # Info box - match minimap height (180) and top margin (10)
    has_climate = climate is not None
    box_w = 180
    box_h = 180  # Match minimap height
    box_x = display[0] - box_w - 10
    box_y = 10  # Same top margin as minimap
    
    glColor4f(0, 0, 0, 0.6)
    glBegin(GL_QUADS)
    glVertex2f(box_x, box_y)
    glVertex2f(box_x + box_w, box_y)
    glVertex2f(box_x + box_w, box_y + box_h)
    glVertex2f(box_x, box_y + box_h)
    glEnd()
    
    # Mode bar
    mode_y = box_y + 10
    if camera.flying:
        glColor4f(0.3, 0.6, 1.0, 0.9)
    else:
        glColor4f(0.3, 0.9, 0.3, 0.9)
    
    glBegin(GL_QUADS)
    glVertex2f(box_x + 10, mode_y)
    glVertex2f(box_x + box_w - 10, mode_y)
    glVertex2f(box_x + box_w - 10, mode_y + 20)
    glVertex2f(box_x + 10, mode_y + 20)
    glEnd()
    
    # Height bar
    bar_y = mode_y + 30
    glColor4f(0.5, 0.5, 0.5, 0.8)
    glBegin(GL_QUADS)
    glVertex2f(box_x + 10, bar_y)
    glVertex2f(box_x + box_w - 10, bar_y)
    glVertex2f(box_x + box_w - 10, bar_y + 8)
    glVertex2f(box_x + 10, bar_y + 8)
    glEnd()
    
    # Clamp height_pct to prevent rendering bugs on bad values
    try:
        height_pct = min(1.0, max(0.0, float(camera.y) / 100.0))
        if not (0.0 <= height_pct <= 1.0):  # Catch NaN
            height_pct = 0.5
    except:
        height_pct = 0.5
    
    glColor4f(1, 0.8, 0.2, 0.9)
    glBegin(GL_QUADS)
    glVertex2f(box_x + 10, bar_y)
    glVertex2f(box_x + 10 + height_pct * (box_w - 20), bar_y)
    glVertex2f(box_x + 10 + height_pct * (box_w - 20), bar_y + 8)
    glVertex2f(box_x + 10, bar_y + 8)
    glEnd()
    
    # Time of day bar (if sky system provided)
    if sky:
        time_y = bar_y + 18
        
        # Day/night gradient bar
        glBegin(GL_QUADS)
        # Left half (night) - dark blue
        glColor4f(0.1, 0.1, 0.3, 0.8)
        glVertex2f(box_x + 10, time_y)
        glColor4f(0.4, 0.5, 0.7, 0.8)
        glVertex2f(box_x + box_w/2, time_y)
        glVertex2f(box_x + box_w/2, time_y + 8)
        glColor4f(0.1, 0.1, 0.3, 0.8)
        glVertex2f(box_x + 10, time_y + 8)
        glEnd()
        
        glBegin(GL_QUADS)
        # Right half (day) - light to dark
        glColor4f(0.4, 0.5, 0.7, 0.8)
        glVertex2f(box_x + box_w/2, time_y)
        glColor4f(0.1, 0.1, 0.3, 0.8)
        glVertex2f(box_x + box_w - 10, time_y)
        glVertex2f(box_x + box_w - 10, time_y + 8)
        glColor4f(0.4, 0.5, 0.7, 0.8)
        glVertex2f(box_x + box_w/2, time_y + 8)
        glEnd()
        
        # Time marker - clamp to prevent rendering bugs
        try:
            time_pct = min(1.0, max(0.0, float(sky.time)))
            if not (0.0 <= time_pct <= 1.0):  # Catch NaN
                time_pct = 0.5
        except:
            time_pct = 0.5
        marker_x = box_x + 10 + time_pct * (box_w - 20)
        if sky.is_night():
            glColor4f(0.9, 0.9, 1.0, 1.0)  # Moon color
        else:
            glColor4f(1.0, 0.9, 0.3, 1.0)  # Sun color
        
        glBegin(GL_TRIANGLES)
        glVertex2f(marker_x, time_y - 3)
        glVertex2f(marker_x - 4, time_y + 4)
        glVertex2f(marker_x + 4, time_y + 4)
        glEnd()
    
    # Climate/weather info
    if climate:
        climate_y = (bar_y + 38) if sky else (bar_y + 18)
        biome = climate.current_biome
        weather = climate.current_weather
        
        # Biome indicator bar (temperature gradient)
        glBegin(GL_QUADS)
        # Cold to hot gradient
        glColor4f(0.3, 0.5, 1.0, 0.8)  # Cold blue
        glVertex2f(box_x + 10, climate_y)
        glColor4f(1.0, 0.5, 0.2, 0.8)  # Hot orange
        glVertex2f(box_x + box_w - 10, climate_y)
        glVertex2f(box_x + box_w - 10, climate_y + 6)
        glColor4f(0.3, 0.5, 1.0, 0.8)
        glVertex2f(box_x + 10, climate_y + 6)
        glEnd()
        
        # Temperature marker - clamp to prevent rendering bugs
        try:
            temp_val = min(1.0, max(0.0, float(biome.temperature)))
            if not (0.0 <= temp_val <= 1.0):  # Catch NaN
                temp_val = 0.5
        except:
            temp_val = 0.5
        temp_x = box_x + 10 + temp_val * (box_w - 20)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_TRIANGLES)
        glVertex2f(temp_x, climate_y - 2)
        glVertex2f(temp_x - 3, climate_y + 3)
        glVertex2f(temp_x + 3, climate_y + 3)
        glEnd()
        
        # Weather bar (precipitation)
        weather_y = climate_y + 14
        glColor4f(0.4, 0.4, 0.4, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(box_x + 10, weather_y)
        glVertex2f(box_x + box_w - 10, weather_y)
        glVertex2f(box_x + box_w - 10, weather_y + 6)
        glVertex2f(box_x + 10, weather_y + 6)
        glEnd()
        
        # Precipitation fill - clamp to prevent rendering bugs
        try:
            precip_val = min(1.0, max(0.0, float(weather.precipitation)))
            if not (0.0 <= precip_val <= 1.0):  # Catch NaN
                precip_val = 0.0
        except:
            precip_val = 0.0
        
        if precip_val > 0.05:
            if weather.precipitation_type == "snow":
                glColor4f(0.9, 0.95, 1.0, 0.9)  # White for snow
            else:
                glColor4f(0.4, 0.6, 0.9, 0.9)  # Blue for rain
            
            precip_w = precip_val * (box_w - 20)
            glBegin(GL_QUADS)
            glVertex2f(box_x + 10, weather_y)
            glVertex2f(box_x + 10 + precip_w, weather_y)
            glVertex2f(box_x + 10 + precip_w, weather_y + 6)
            glVertex2f(box_x + 10, weather_y + 6)
            glEnd()
        
        # Cloud cover bar
        cloud_y = weather_y + 10
        glColor4f(0.3, 0.3, 0.35, 0.8)
        glBegin(GL_QUADS)
        glVertex2f(box_x + 10, cloud_y)
        glVertex2f(box_x + box_w - 10, cloud_y)
        glVertex2f(box_x + box_w - 10, cloud_y + 4)
        glVertex2f(box_x + 10, cloud_y + 4)
        glEnd()
        
        # Cloud cover fill - CLAMP to prevent the white line bug!
        try:
            cloud_val = min(1.0, max(0.0, float(weather.cloud_cover)))
            if not (0.0 <= cloud_val <= 1.0):  # Catch NaN
                cloud_val = 0.3
        except:
            cloud_val = 0.3
        
        cloud_w = cloud_val * (box_w - 20)
        glColor4f(0.8, 0.8, 0.85, 0.9)
        glBegin(GL_QUADS)
        glVertex2f(box_x + 10, cloud_y)
        glVertex2f(box_x + 10 + cloud_w, cloud_y)
        glVertex2f(box_x + 10 + cloud_w, cloud_y + 4)
        glVertex2f(box_x + 10, cloud_y + 4)
        glEnd()
    
    # ========== TOOL DISPLAY (to the right of compass) ==========
    tool_x = compass_x + compass_w + 15
    tool_w = 80
    tool_h = top_bar_h  # Match compass height
    tool_y = top_margin  # Same top margin
    
    # Background
    glColor4f(0, 0, 0, 0.7)
    glBegin(GL_QUADS)
    glVertex2f(tool_x, tool_y)
    glVertex2f(tool_x + tool_w, tool_y)
    glVertex2f(tool_x + tool_w, tool_y + tool_h)
    glVertex2f(tool_x, tool_y + tool_h)
    glEnd()
    
    # Tool-specific color
    if camera.current_tool == ToolType.SCAN:
        glColor4f(0.2, 0.8, 1.0, 0.9)  # Cyan for scan
    elif camera.current_tool == ToolType.MINE:
        glColor4f(1.0, 0.6, 0.2, 0.9)  # Orange for mine
    elif camera.current_tool == ToolType.FILL:
        glColor4f(0.4, 1.0, 0.4, 0.9)  # Green for fill
    else:
        glColor4f(0.8, 0.8, 0.8, 0.9)  # Gray default
    
    # Tool indicator bar
    glBegin(GL_QUADS)
    glVertex2f(tool_x + 5, tool_y + 3)
    glVertex2f(tool_x + tool_w - 5, tool_y + 3)
    glVertex2f(tool_x + tool_w - 5, tool_y + 7)
    glVertex2f(tool_x + 5, tool_y + 7)
    glEnd()
    
    # Draw tool name with font if available
    if hud_font:
        _draw_text(hud_font, camera.current_tool, tool_x + 10, tool_y + 16, (255, 255, 255))
    
    # Draw status message at bottom-left of screen
    if camera.status_message and camera.status_timer > 0 and hud_font:
        # Fade out in last 0.5 seconds
        alpha = min(1.0, camera.status_timer / 0.5) if camera.status_timer < 0.5 else 1.0
        msg_color = (255, 255, 255)
        msg_x = 20  # Bottom-left
        msg_y = display[1] - 40
        _draw_text(hud_font, camera.status_message, msg_x, msg_y, msg_color)
    
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def setup_opengl():
    """Setup OpenGL - open world style with distant view."""
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    glLightfv(GL_LIGHT0, GL_POSITION, (0.4, 1.0, 0.3, 0.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.95, 0.88, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.45, 0.45, 0.5, 1.0))
    
    # Sky blue
    glClearColor(0.52, 0.75, 0.95, 1.0)
    
    # Atmospheric fog for massive view distance
    glEnable(GL_FOG)
    glFogfv(GL_FOG_COLOR, (0.6, 0.78, 0.95, 1.0))
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogf(GL_FOG_START, 400)   # Fog starts very far
    glFogf(GL_FOG_END, 1500)    # Fog fades to horizon


def run_explorer(config: WorldConfig = None, precompute_chunks: int = 0, debug_flags: dict = None):
    """Main explorer loop.
    
    Args:
        config: World configuration
        precompute_chunks: Number of chunks to precompute flora/animals for before starting
        debug_flags: Optional dict with debug toggles:
            - no_flora_render: Skip rendering flora
            - no_flora_update: Skip flora life sim
            - no_animal_render: Skip rendering animals
            - no_animal_update: Skip animal AI updates
    """
    # Parse debug flags
    debug_flags = debug_flags or {}
    NO_FLORA_RENDER = debug_flags.get('no_flora_render', False)
    NO_FLORA_UPDATE = debug_flags.get('no_flora_update', False)
    NO_ANIMAL_RENDER = debug_flags.get('no_animal_render', False)
    NO_ANIMAL_UPDATE = debug_flags.get('no_animal_update', False)
    MAX_FLORA = debug_flags.get('max_flora')
    MAX_ANIMALS = debug_flags.get('max_animals')
    
    # Apply max flora/animals if specified (will be set on managers after they're created)
    if MAX_FLORA is not None:
        print(f"  [CONFIG] max-flora={MAX_FLORA}")
    if MAX_ANIMALS is not None:
        from waverse.life import LifeConfig
        LifeConfig.MAX_TOTAL_ANIMALS = MAX_ANIMALS
        print(f"  [CONFIG] max-animals={MAX_ANIMALS}")
    
    if any([NO_FLORA_RENDER, NO_FLORA_UPDATE, NO_ANIMAL_RENDER, NO_ANIMAL_UPDATE]):
        print("  [DEBUG FLAGS]", end="")
        if NO_FLORA_RENDER: print(" no-flora-render", end="")
        if NO_FLORA_UPDATE: print(" no-flora-update", end="")
        if NO_ANIMAL_RENDER: print(" no-animal-render", end="")
        if NO_ANIMAL_UPDATE: print(" no-animal-update", end="")
        print()
    if not OPENGL_AVAILABLE:
        print("Error: OpenGL not available!")
        return
    
    if config is None:
        config = WorldConfig.create_default()
    
    # Check if modern renderer is requested
    USE_MODERN_RENDERER = debug_flags.get('modern_renderer', False) and MODERNGL_AVAILABLE
    ENABLE_WAVES = debug_flags.get('enable_waves', False)
    
    pygame.init()
    display = (1400, 800)
    
    # On macOS, we can only use:
    # - Legacy OpenGL 2.1 (default, supports glBegin/glEnd) 
    # - Core OpenGL 3.2-4.1 (no legacy support)
    # macOS does NOT support Compatibility profile for OpenGL 3.2+
    # We'll try to detect the available version and work with it
    if USE_MODERN_RENDERER:
        import platform
        if platform.system() == 'Darwin':
            # macOS: Try Core 4.1 first, but we'll need to disable legacy for full modern
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FLAGS, pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG)
            print("  [RENDERER] macOS: Requesting OpenGL 4.1 Core profile...")
        else:
            # Linux/Windows: Try Compatibility profile to mix modern+legacy
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_COMPATIBILITY)
            print("  [RENDERER] Requesting OpenGL 3.3 Compatibility profile...")
    
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption(f"Waverse - {config.name}")
    
    if USE_MODERN_RENDERER:
        print("  [RENDERER] ModernGL hybrid mode (terrain+flora=modern, HUD=legacy)")
    
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    
    # Initialize ModernGL FIRST if enabled (before any legacy OpenGL calls)
    # This must happen right after display creation while context is fresh
    modern_renderer = None
    modern_ctx = None
    if USE_MODERN_RENDERER:
        try:
            import platform
            # On macOS with Core profile, we need to init ModernGL before legacy GL calls
            # because Core profile doesn't support legacy functions
            modern_ctx = moderngl.create_context()
            gl_version = modern_ctx.version_code
            print(f"  [RENDERER] ModernGL context created (OpenGL {gl_version})")
            
            if gl_version < 330:
                raise RuntimeError(f"OpenGL {gl_version} < 330, need 3.3+ for shaders")
            
            modern_renderer = ModernWorldRenderer(modern_ctx, enable_waves=ENABLE_WAVES)
            modern_renderer.set_chunk_params(CHUNK_SIZE, TILE_SCALE, HEIGHT_SCALE)
            modern_renderer.set_screen_size(display[0], display[1])
            print(f"  [RENDERER] ModernGL initialized successfully!")
            
            # Check if we're on macOS with Core profile - legacy GL won't work
            if platform.system() == 'Darwin':
                print("  [RENDERER] Note: macOS Core profile - some legacy effects disabled")
                
        except Exception as e:
            print(f"  [RENDERER] ModernGL failed: {e}")
            import traceback
            traceback.print_exc()
            print(f"  [RENDERER] Falling back to legacy renderer")
            USE_MODERN_RENDERER = False
            modern_renderer = None
            modern_ctx = None
            
            # On macOS, we need to recreate the display with legacy profile
            # because Core profile doesn't support glGenLists etc.
            if platform.system() == 'Darwin':
                print("  [RENDERER] Recreating display with legacy OpenGL for fallback...")
                pygame.display.quit()
                pygame.display.init()
                # Reset to default (legacy) OpenGL - don't set any profile
                pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 2)
                pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
                pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
                pygame.display.set_caption(f"Waverse - {config.name}")
    
    # Initialize font for HUD text
    pygame.font.init()
    try:
        hud_font = pygame.font.SysFont("monospace", 14)
    except:
        hud_font = pygame.font.Font(None, 16)
    
    # Initialize gamepad
    gamepad = GamepadManager()
    
    # Legacy OpenGL setup (needed for HUD/structures/weather)
    # On macOS Core profile some of these may not work but we try anyway
    try:
        setup_opengl()
        glMatrixMode(GL_PROJECTION)
        gluPerspective(75, display[0]/display[1], 1.0, 2000)
        glMatrixMode(GL_MODELVIEW)
    except Exception as e:
        print(f"  [RENDERER] Legacy OpenGL setup warning: {e}")
    
    # Create world with background chunk worker
    chunk_manager = ChunkManager(config)
    chunk_worker = ChunkWorker(config, preload_radius=30)
    chunk_manager.set_worker(chunk_worker)
    chunk_worker.start()
    
    # Chunk DNA for terrain colors and sky - evolves across chunks
    chunk_dna_manager = ChunkDNAManager(config.seed)
    
    chunk_renderer = ChunkRenderer(chunk_manager, chunk_dna_manager)
    flora_manager = FloraManager(config.seed)
    if MAX_FLORA is not None:
        flora_manager.MAX_TOTAL_PLANTS = MAX_FLORA
    animal_manager = AnimalManager(config.seed)
    
    # Dropped items in the world (wood, stone, etc.)
    dropped_items: list = []
    # Debris particles (visual-only, fading)
    debris_particles: list = []
    # Fallen trees (visual, fading tree trunks)
    fallen_trees: list = []
    sky = SkySystem(chunk_dna_manager)
    water_level = config.water_level
    minimap = Minimap(chunk_manager)
    
    # Climate/weather system
    climate_manager = ClimateManager(config.seed)
    weather_renderer = WeatherRenderer()
    
    # Life simulation system (growth, death, reproduction)
    life_simulator = LifeSimulator(config.seed)
    
    # Structure system (buildings, etc.)
    structure_manager = StructureManager(config.seed)
    
    # Performance monitor
    perf = PerfMonitor()
    
    # Camera - try to load saved position
    camera = Camera()
    saved = load_position(config.seed)
    if saved:
        camera.x = saved.get("x", 0)
        camera.y = saved.get("y", 40)
        camera.z = saved.get("z", 0)
        camera.yaw = saved.get("yaw", 0)
        camera.pitch = saved.get("pitch", -20)
        # ALWAYS start in walking mode - flying is for tour/explore only
        camera.flying = False
        saved_speed = saved.get("speed_level", 2)
        # If saved speed is out of range (from old config), reset to default
        camera.speed_level = saved_speed if 0 <= saved_speed < len(SPEED_LEVELS) else 2
        print(f"  Loaded position: ({camera.x:.0f}, {camera.y:.0f}, {camera.z:.0f})")
        print(f"  (Starting in WALK mode)")
    else:
        # New game - start in walking mode at spawn
        camera.y = chunk_manager.get_height_at(0, 0) * HEIGHT_SCALE + 30
        camera.flying = False  # Walking mode
        camera.speed_level = 2  # 0.75x speed (comfortable walking pace)
    
    # Pre-load chunks around camera position
    start_chunk = camera.get_chunk_pos()
    print("  Loading terrain (this may take a moment on first run)...")
    
    if USE_MODERN_RENDERER and modern_renderer:
        # Modern renderer handles terrain via VBOs, not display lists
        # Just call update once - it loads chunks based on camera position
        modern_renderer.update_chunks_around_camera(
            camera, chunk_manager, flora_manager, animal_manager, structure_manager, climate_manager
        )
        print(f"  Loaded chunks via modern renderer!")
    else:
        # Legacy: use display lists
        chunk_renderer.update_chunks(start_chunk, force_all=True)
        total_chunks = len(chunk_renderer.display_lists)
        print(f"  Loaded {total_chunks} chunks!")
    
    # Precompute flora/animals if requested (for performance testing)
    # Skip in modern mode - modern renderer handles loading dynamically
    if precompute_chunks > 0 and not USE_MODERN_RENDERER:
        import math as _math
        # Calculate radius needed for requested chunk count
        # For a square grid: (2r+1)^2 = num_chunks, so r = (sqrt(num_chunks) - 1) / 2
        precompute_radius = int(_math.ceil((_math.sqrt(precompute_chunks) - 1) / 2))
        precompute_radius = max(precompute_radius, 30)  # At least the default load radius
        
        print(f"\n  Precomputing {precompute_chunks} chunks (radius={precompute_radius})...")
        
        # Generate chunks in expanding spiral from start position
        scx, scz = start_chunk
        precomputed = 0
        
        # Generate in square rings outward
        for ring in range(precompute_radius + 1):
            if precomputed >= precompute_chunks:
                break
            
            # Generate all chunks at this ring distance
            for dx in range(-ring, ring + 1):
                for dz in range(-ring, ring + 1):
                    # Only process chunks on the edge of this ring (or ring 0)
                    if ring == 0 or abs(dx) == ring or abs(dz) == ring:
                        if precomputed >= precompute_chunks:
                            break
                        
                        cx, cz = scx + dx, scz + dz
                        
                        # Get or generate the terrain chunk
                        chunk = chunk_manager.get_chunk(cx, cz)
                        if chunk is None:
                            continue
                        
                        # Get biome for this chunk
                        biome_name = None
                        if climate_manager:
                            biome_dna = climate_manager.get_biome(cx, cz)
                            if biome_dna:
                                biome_name = biome_dna.get_biome_name()
                        
                        # Generate flora (biome-specific density/types)
                        flora_manager.get_plants_for_chunk(
                            cx, cz, chunk.heightmap, chunk.world_x, chunk.world_z, TILE_SCALE, HEIGHT_SCALE,
                            biome_name=biome_name
                        )
                        # Create flora display lists
                        plants = flora_manager.chunk_plants.get((cx, cz), [])
                        if plants:
                            flora_manager.create_display_lists(cx, cz, plants)
                        # Spawn animals (biome-specific density/types)
                        animal_manager.spawn_animals_for_chunk(
                            cx, cz, chunk.heightmap, chunk.world_x, chunk.world_z, TILE_SCALE, HEIGHT_SCALE,
                            biome_name=biome_name
                        )
                        # Spawn structures
                        structure_manager.spawn_random_buildings(
                            cx, cz, chunk.world_x, chunk.world_z, chunk.heightmap, HEIGHT_SCALE, TILE_SCALE
                        )
                        precomputed += 1
                        
                        if precomputed % 500 == 0:
                            print(f"    Precomputed {precomputed}/{precompute_chunks}...")
        
        total_plants = sum(len(p) for p in flora_manager.chunk_plants.values())
        print(f"  Precomputed {precomputed} chunks: {total_plants} plants, {len(animal_manager.animals)} animals")
    
    # Initial minimap
    minimap.update(0, 0)
    
    print("\n" + "=" * 60)
    print(f"  WAVERSE - {config.name}")
    print("=" * 60)
    print("  KEYBOARD:")
    print("    Movement: WASD/Arrows | H/Space=Up F/Shift=Down | B(hold)=Run")
    print("    Camera: IJKL or Right-Click+Mouse")
    print("    Speed: [/- = slower | ]/= = faster")
    print("    R=ACTION | ,/.=Switch Tool | T=Tool Size | Tab=Menu | P=Screenshot | X=Tour | N=Warp")
    print("    F3=Perf Stats | F4=Perf Overlay")
    if gamepad.is_connected():
        print(f"  GAMEPAD ({gamepad.name}):")
        print(f"    Config: L-Stick X={GamepadConfig.L_STICK_X} Y={GamepadConfig.L_STICK_Y}")
        print(f"    Config: R-Stick X={GamepadConfig.R_STICK_X} Y={GamepadConfig.R_STICK_Y}")
        print("    Left Stick=Move | Right Stick=Look")
        print("    A=Action | B=Jump | X=Fly/Walk | Y=Screenshot")
        print("    L2/R2=Speed | R1=Tool | DPAD=Cycle | L3=Run | START=Menu | SELECT=Warp")
    print("=" * 60 + "\n")
    
    clock = pygame.time.Clock()
    running = True
    mouse_look = False
    last_chunk = None
    frame_count = 0
    
    # Gamepad speed change cooldown
    gamepad_speed_cooldown = 0
    
    # Menu toggle handled via JOYBUTTONDOWN event
    
    while running:
        perf.start_frame()
        frame_count += 1
        dt = clock.tick(60) / 16.67
        # Cap dt to prevent extreme values causing fast camera spinning
        dt = min(dt, 4.0)  # Max ~15 fps worth of movement per frame
        
        # Update gamepad speed cooldown
        if gamepad_speed_cooldown > 0:
            gamepad_speed_cooldown -= dt
        
        # Update status message timer
        camera.update_status(dt / 60.0)  # Convert frame-time to seconds
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if camera.microscope_active:
                        # Exit microscope mode
                        camera.microscope_active = False
                        camera.set_status("Exited microscope", 1.5)
                    else:
                        running = False
                # Speed controls: [ and ] or - and = to decrease/increase
                elif event.key in (pygame.K_LEFTBRACKET, pygame.K_MINUS):
                    new_level = max(0, camera.speed_level - 1)
                    if new_level != camera.speed_level:
                        camera.set_speed(new_level)
                        print(f"  Speed: {SPEED_LEVELS[new_level]:.2f}x")
                elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_EQUALS):
                    new_level = min(len(SPEED_LEVELS) - 1, camera.speed_level + 1)
                    if new_level != camera.speed_level:
                        camera.set_speed(new_level)
                        print(f"  Speed: {SPEED_LEVELS[new_level]:.2f}x")
                elif event.key == pygame.K_o:  # O for Origin/Save
                    save_position(camera, config.seed)
                elif event.key == pygame.K_p:  # P for Picture/Screenshot
                    take_screenshot(camera)
                elif event.key == pygame.K_TAB:  # Tab = toggle menu
                    print("  [KEYDOWN] Tab pressed")
                    camera.toggle_menu()
                elif event.key == pygame.K_r:  # R = use current tool (ACTION)
                    if camera.current_tool == ToolType.SCAN:
                        log_dna_at_cursor(camera, flora_manager, animal_manager)
                    elif camera.current_tool == ToolType.MINE:
                        modified_chunks = modify_terrain_at_cursor(camera, chunk_manager, "MINE", camera.tool_radius)
                        if modified_chunks:
                            # Invalidate all modified chunk display lists to force re-render
                            for chunk_key in modified_chunks:
                                if chunk_key in chunk_renderer.display_lists:
                                    del chunk_renderer.display_lists[chunk_key]
                                # Also invalidate modern renderer chunks
                                if USE_MODERN_RENDERER and modern_renderer:
                                    modern_renderer.invalidate_terrain_chunk(chunk_key[0], chunk_key[1])
                    elif camera.current_tool == ToolType.FILL:
                        modified_chunks = modify_terrain_at_cursor(camera, chunk_manager, "FILL", camera.tool_radius)
                        if modified_chunks:
                            for chunk_key in modified_chunks:
                                if chunk_key in chunk_renderer.display_lists:
                                    del chunk_renderer.display_lists[chunk_key]
                                # Also invalidate modern renderer chunks
                                if USE_MODERN_RENDERER and modern_renderer:
                                    modern_renderer.invalidate_terrain_chunk(chunk_key[0], chunk_key[1])
                    elif camera.current_tool == ToolType.CUT:
                        cut_tree_at_cursor(camera, flora_manager, chunk_manager, dropped_items, debris_particles, modern_renderer, fallen_trees)
                    elif camera.current_tool == ToolType.MICRO:
                        if camera.microscope_active:
                            # Already in microscope - scan organism at cursor
                            scan_microorganism_at_cursor(camera, modern_renderer)
                        else:
                            # Activate microscope view - sample what's at cursor
                            terrain_type = get_terrain_type_at_cursor(camera, chunk_manager, flora_manager)
                            camera.microscope_active = True
                            camera.microscope_terrain_type = terrain_type
                            # Initialize microscope world with organisms based on terrain
                            if modern_renderer and hasattr(modern_renderer, 'microscope_view'):
                                modern_renderer.microscope_view.world.clear()
                                modern_renderer.microscope_view.set_terrain(terrain_type)
                                populate_microscope_world(modern_renderer.microscope_view.world, terrain_type)
                            camera.set_status(f"Microscope: viewing {terrain_type} life")
                elif event.key == pygame.K_t:  # T = cycle tool radius (for MINE/FILL)
                    if camera.current_tool in (ToolType.MINE, ToolType.FILL):
                        camera.cycle_tool_radius()
                elif event.key == pygame.K_COMMA:  # , = previous tool
                    camera.prev_tool()
                    if USE_MODERN_RENDERER and modern_renderer:
                        modern_renderer.hud_prev_tool()
                elif event.key == pygame.K_PERIOD:  # . = next tool
                    camera.next_tool()
                    if USE_MODERN_RENDERER and modern_renderer:
                        modern_renderer.hud_next_tool()
                elif event.key == pygame.K_g:
                    gamepad.debug_mode = not gamepad.debug_mode
                    print(f"  Gamepad debug: {'ON' if gamepad.debug_mode else 'OFF'}")
                elif event.key == pygame.K_x:
                    camera.toggle_auto_fly()
                elif event.key == pygame.K_m:
                    camera.add_marker()
                elif event.key == pygame.K_c:
                    camera.clear_markers()
                elif event.key == pygame.K_n:
                    # N = New location (warp to new waverse)
                    # Check if Shift is held for exotic biome warps!
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        # Shift+N = random exotic biome!
                        exotic_biomes = ['psychedelic', 'hellfire', 'shadow', 'crystal', 'void']
                        force_biome = random.choice(exotic_biomes)
                        warp_to_new_universe(camera, chunk_manager, chunk_renderer, 
                                            flora_manager, animal_manager,
                                            chunk_dna_manager, climate_manager,
                                            structure_manager, config, life_simulator,
                                            force_biome=force_biome)
                    else:
                        warp_to_new_universe(camera, chunk_manager, chunk_renderer, 
                                            flora_manager, animal_manager,
                                            chunk_dna_manager, climate_manager,
                                            structure_manager, config, life_simulator)
                # Number keys 1-5 with Ctrl = warp to specific exotic biomes
                elif event.key == pygame.K_1 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    warp_to_new_universe(camera, chunk_manager, chunk_renderer, 
                                        flora_manager, animal_manager,
                                        chunk_dna_manager, climate_manager,
                                        structure_manager, config, life_simulator,
                                        force_biome='psychedelic')
                elif event.key == pygame.K_2 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    warp_to_new_universe(camera, chunk_manager, chunk_renderer, 
                                        flora_manager, animal_manager,
                                        chunk_dna_manager, climate_manager,
                                        structure_manager, config, life_simulator,
                                        force_biome='hellfire')
                elif event.key == pygame.K_3 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    warp_to_new_universe(camera, chunk_manager, chunk_renderer, 
                                        flora_manager, animal_manager,
                                        chunk_dna_manager, climate_manager,
                                        structure_manager, config, life_simulator,
                                        force_biome='shadow')
                elif event.key == pygame.K_4 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    warp_to_new_universe(camera, chunk_manager, chunk_renderer, 
                                        flora_manager, animal_manager,
                                        chunk_dna_manager, climate_manager,
                                        structure_manager, config, life_simulator,
                                        force_biome='crystal')
                elif event.key == pygame.K_5 and pygame.key.get_mods() & pygame.KMOD_CTRL:
                    warp_to_new_universe(camera, chunk_manager, chunk_renderer, 
                                        flora_manager, animal_manager,
                                        chunk_dna_manager, climate_manager,
                                        structure_manager, config, life_simulator,
                                        force_biome='void')
                elif event.key == pygame.K_F3:
                    # F3 = dump performance stats (like Minecraft debug)
                    perf.dump_stats()
                elif event.key == pygame.K_F4:
                    # F4 = toggle perf overlay
                    perf.toggle_overlay()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    mouse_look = True
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    mouse_look = False
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)
            elif event.type == pygame.MOUSEMOTION and mouse_look:
                camera.rotate(*event.rel)
            elif event.type == pygame.JOYBUTTONDOWN:
                # Handle joystick button press events (more reliable than polling)
                print(f"  [JOYBUTTONDOWN] button={event.button} (START is {GamepadConfig.START})")
                if event.button == GamepadConfig.START:
                    if camera.microscope_active:
                        # Exit microscope mode
                        camera.microscope_active = False
                        camera.set_status("Exited microscope", 1.5)
                    else:
                        camera.toggle_menu()
                elif event.button == 1:  # B button - exit microscope
                    if camera.microscope_active:
                        camera.microscope_active = False
                        camera.set_status("Exited microscope", 1.5)
        
        # Keyboard input
        keys = pygame.key.get_pressed()
        forward = right = up = 0
        
        # Ignore movement when system modifiers held (Cmd+Shift+Ctrl+4 for screenshots)
        mods = pygame.key.get_mods()
        system_mod_held = (mods & pygame.KMOD_META) or (mods & pygame.KMOD_CTRL)
        
        # Only allow world movement/look when menu is closed and no system mods
        if not camera.menu_open and not system_mod_held:
            speed = 2.5 if keys[pygame.K_LALT] else 1.0
            
            if keys[pygame.K_w] or keys[pygame.K_UP]: forward += dt * speed
            if keys[pygame.K_s] or keys[pygame.K_DOWN]: forward -= dt * speed
            if keys[pygame.K_a] or keys[pygame.K_LEFT]: right -= dt * speed
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]: right += dt * speed
            
            # H/F for height - in tour mode adjusts target height, otherwise moves up/down
            if camera.auto_fly_mode > 0:
                if keys[pygame.K_h]:
                    camera.auto_fly_height = min(100, camera.auto_fly_height + 0.3)
                if keys[pygame.K_f]:
                    camera.auto_fly_height = max(15, camera.auto_fly_height - 0.3)
            else:
                if keys[pygame.K_h]: up += dt * speed
                if keys[pygame.K_f]: up -= dt * speed
            
            # Keyboard look
            look_speed = dt * 0.8
            if keys[pygame.K_i]: camera.rotate_keyboard(0, look_speed * 1.2)
            if keys[pygame.K_k]: camera.rotate_keyboard(0, -look_speed * 1.2)
            if keys[pygame.K_j]: camera.rotate_keyboard(look_speed * 1.5, 0)
            if keys[pygame.K_l]: camera.rotate_keyboard(-look_speed * 1.5, 0)
        else:
            # Menu keyboard navigation
            if keys[pygame.K_UP]: camera.menu_navigate(-1)
            if keys[pygame.K_DOWN]: camera.menu_navigate(1)
            if keys[pygame.K_LEFT]: camera.menu_switch_tab(-1)
            if keys[pygame.K_RIGHT]: camera.menu_switch_tab(1)
        
        # Gamepad input (check for hotplug every 30 frames - more responsive)
        if frame_count % 30 == 0:
            gamepad.check_hotplug()
        
        if gamepad.is_connected():
            # Debug mode - print axis values
            if gamepad.debug_mode:
                gamepad.debug_cooldown = max(0, gamepad.debug_cooldown - 1)
                gamepad.print_debug()
            
            # Menu navigation when menu is open
            if camera.menu_open:
                gp_move = gamepad.get_movement()
                
                # Accelerated scrolling for UP/DOWN (DPAD or Left Stick)
                is_scrolling_up = gp_move[0] < -0.5 or gamepad.get_dpad(GamepadConfig.DPAD_UP)
                is_scrolling_down = gp_move[0] > 0.5 or gamepad.get_dpad(GamepadConfig.DPAD_DOWN)
                
                scroll_amount = camera.update_scroll(dt / 60.0, is_scrolling_up, is_scrolling_down)
                if scroll_amount != 0:
                    camera.menu_navigate(scroll_amount)
                
                # Other menu controls use cooldown
                if gamepad_speed_cooldown <= 0:
                    # DPAD LEFT/RIGHT = navigate horizontally (grid in inventory, filter in log)
                    if gamepad.get_dpad(GamepadConfig.DPAD_LEFT):
                        camera.menu_navigate_horizontal(-1)
                        gamepad_speed_cooldown = 15
                    elif gamepad.get_dpad(GamepadConfig.DPAD_RIGHT):
                        camera.menu_navigate_horizontal(1)
                        gamepad_speed_cooldown = 15
                    
                    # L1/R1 = switch tabs
                    if gamepad.get_button(GamepadConfig.L1):
                        camera.menu_switch_tab(-1)
                        gamepad_speed_cooldown = 15
                    elif gamepad.get_button(GamepadConfig.R1):
                        camera.menu_switch_tab(1)
                        gamepad_speed_cooldown = 15
                    
                    # A = equip tool (inventory) or favorite log item (log)
                    if gamepad.get_button(GamepadConfig.A):
                        if camera.menu_tab == 0:  # Inventory - equip selected tool
                            # Only equip if it's an implemented tool (first 3)
                            if camera.inventory_index < len(ToolType.ALL_TOOLS):
                                camera.current_tool_index = camera.inventory_index
                                camera.current_tool = ToolType.ALL_TOOLS[camera.inventory_index]
                                camera.set_status(f"EQUIPPED: {camera.current_tool}", 2.0)
                                if USE_MODERN_RENDERER and modern_renderer:
                                    modern_renderer.hud_set_tool(camera.current_tool)
                            else:
                                camera.set_status("TOOL NOT YET AVAILABLE", 2.0)
                            gamepad_speed_cooldown = 20
                        elif camera.menu_tab == 1:  # Log - favorite
                            camera.toggle_log_favorite()
                            gamepad_speed_cooldown = 20
                    
                    # SELECT = delete favorite (in LOG tab)
                    if gamepad.get_button(GamepadConfig.SELECT) and camera.menu_tab == 1:
                        camera.delete_log_favorite()
                        gamepad_speed_cooldown = 20
            
            # Left stick = movement (WASD) - only when menu closed
            if camera.microscope_active and modern_renderer:
                # Microscope controls
                gp_move = gamepad.get_movement()
                pan_speed = 0.3  # Slower for precise microscope control
                
                # Left stick = pan
                if abs(gp_move[0]) > 0.1 or abs(gp_move[1]) > 0.1:
                    modern_renderer.microscope_view.pan(
                        gp_move[1] * pan_speed,  # X = left/right
                        gp_move[0] * pan_speed   # Y = up/down (not inverted)
                    )
                
                # L1/R1 = zoom
                if gamepad.get_button(GamepadConfig.L1):
                    modern_renderer.microscope_view.zoom_out(1.02)
                if gamepad.get_button(GamepadConfig.R1):
                    modern_renderer.microscope_view.zoom_in(1.02)
                
                # A button = select organism at center
                if gamepad.get_button(GamepadConfig.A) and gamepad_speed_cooldown <= 0:
                    org = modern_renderer.microscope_view.select_at(
                        modern_renderer.microscope_view.width // 2,
                        modern_renderer.microscope_view.height // 2
                    )
                    if org:
                        camera.set_status(f"Selected: {org.dna.species_id}", 2.0)
                    gamepad_speed_cooldown = 15
            elif not camera.menu_open:
                gp_move = gamepad.get_movement()
                gp_speed = SPEED_LEVELS[camera.speed_level]
                # Debug: print if movement detected (commented out)
                # if abs(gp_move[0]) > 0.1 or abs(gp_move[1]) > 0.1:
                #     if frame_count % 30 == 0:  # Print every 0.5 sec
                #         print(f"  [GP] L-Stick: fwd={gp_move[0]:.2f} right={gp_move[1]:.2f}")
                forward += gp_move[0] * dt * gp_speed
                right += gp_move[1] * dt * gp_speed
            
            # Right stick = look (IJKL) - only when menu closed
            if not camera.menu_open:
                gp_look = gamepad.get_look()
                # Debug: Log if look input is non-zero (every 2 seconds)
                if USE_MODERN_RENDERER and (abs(gp_look[0]) > 0.05 or abs(gp_look[1]) > 0.05):
                    if frame_count % 120 == 0:
                        print(f"  [MODERN DEBUG] look input: yaw={gp_look[0]:.3f} pitch={gp_look[1]:.3f} dt={dt:.2f}")
                gamepad_look_speed = dt * 2.5
                camera.rotate_keyboard(gp_look[0] * gamepad_look_speed, gp_look[1] * gamepad_look_speed)
                
                # In fly mode, vertical movement comes from look direction + forward
                # No more L3/R3 for up/down - more intuitive controls
            
            # L2 = Speed down, R2 = Speed up
            l2_val, r2_val = gamepad.get_triggers()
            
            if camera.auto_fly_mode > 0:
                # TOUR MODE: L2/R2 adjust flight height
                if l2_val > 0.3:
                    camera.auto_fly_height -= l2_val * 0.5
                    camera.auto_fly_height = max(15, camera.auto_fly_height)
                if r2_val > 0.3:
                    camera.auto_fly_height += r2_val * 0.5
                    camera.auto_fly_height = min(100, camera.auto_fly_height)
            else:
                # NORMAL MODE:
                # L2 = speed down
                if gamepad_speed_cooldown <= 0 and l2_val > 0.7:
                    new_level = max(0, camera.speed_level - 1)
                    if new_level != camera.speed_level:
                        camera.set_speed(new_level)
                        camera.set_status(f"Speed: {SPEED_LEVELS[new_level]:.2f}x", 1.0)
                        print(f"  Speed: {SPEED_LEVELS[new_level]:.2f}x")
                    gamepad_speed_cooldown = 15
                
                # R2 = speed up
                if gamepad_speed_cooldown <= 0 and r2_val > 0.7:
                    new_level = min(len(SPEED_LEVELS) - 1, camera.speed_level + 1)
                    if new_level != camera.speed_level:
                        camera.set_speed(new_level)
                        camera.set_status(f"Speed: {SPEED_LEVELS[new_level]:.2f}x", 1.0)
                        print(f"  Speed: {SPEED_LEVELS[new_level]:.2f}x")
                    gamepad_speed_cooldown = 15
            
            # L3 = Run (hold for 2x speed) - only when menu closed
            if not camera.menu_open:
                if gamepad.get_button(GamepadConfig.L3):
                    camera.is_running = True
                else:
                    camera.is_running = False
            
            # X button = Toggle fly/walk mode - only when menu closed
            if not camera.menu_open and gamepad.get_button(GamepadConfig.X) and gamepad_speed_cooldown <= 0:
                camera.flying = not camera.flying
                mode_name = "FLY" if camera.flying else "WALK"
                camera.set_status(f"Mode: {mode_name}", 1.5)
                print(f"  Mode: {mode_name}")
                gamepad_speed_cooldown = 20
            
            # B button = JUMP - only when menu closed
            # Variable jump: hold for higher jump, tap for short hop
            b_pressed = gamepad.get_button(GamepadConfig.B)
            if not camera.menu_open:
                if b_pressed:
                    camera.jump()  # Start jump if grounded
                    camera.update_jump_held(True)  # Held = higher jump
                else:
                    camera.update_jump_held(False)  # Released = cut jump short
            
            # A button = Action/Talk (currently unused, placeholder for future)
            # if not camera.menu_open and gamepad.get_button(GamepadConfig.A) and gamepad_speed_cooldown <= 0:
            #     # Future: interact with NPCs, objects, etc.
            #     pass
            
            # R1 = Use tool (SCAN/MINE/FILL/CUT) - only when menu closed
            if not camera.menu_open and gamepad_speed_cooldown <= 0:
                if gamepad.get_button(GamepadConfig.R1):
                    if camera.current_tool == ToolType.SCAN:
                        log_dna_at_cursor(camera, flora_manager, animal_manager)
                    elif camera.current_tool == ToolType.MINE:
                        modified_chunks = modify_terrain_at_cursor(camera, chunk_manager, "MINE", camera.tool_radius)
                        if modified_chunks:
                            for chunk_key in modified_chunks:
                                if chunk_key in chunk_renderer.display_lists:
                                    del chunk_renderer.display_lists[chunk_key]
                                # Also invalidate modern renderer chunks
                                if USE_MODERN_RENDERER and modern_renderer:
                                    modern_renderer.invalidate_terrain_chunk(chunk_key[0], chunk_key[1])
                    elif camera.current_tool == ToolType.FILL:
                        modified_chunks = modify_terrain_at_cursor(camera, chunk_manager, "FILL", camera.tool_radius)
                        if modified_chunks:
                            for chunk_key in modified_chunks:
                                if chunk_key in chunk_renderer.display_lists:
                                    del chunk_renderer.display_lists[chunk_key]
                                # Also invalidate modern renderer chunks
                                if USE_MODERN_RENDERER and modern_renderer:
                                    modern_renderer.invalidate_terrain_chunk(chunk_key[0], chunk_key[1])
                    elif camera.current_tool == ToolType.CUT:
                        cut_tree_at_cursor(camera, flora_manager, chunk_manager, dropped_items, debris_particles, modern_renderer, fallen_trees)
                    elif camera.current_tool == ToolType.MICRO:
                        if camera.microscope_active:
                            # Already in microscope - scan organism at cursor
                            scan_microorganism_at_cursor(camera, modern_renderer)
                        else:
                            # Activate microscope view
                            terrain_type = get_terrain_type_at_cursor(camera, chunk_manager, flora_manager)
                            camera.microscope_active = True
                            camera.microscope_terrain_type = terrain_type
                            if modern_renderer and hasattr(modern_renderer, 'microscope_view'):
                                modern_renderer.microscope_view.world.clear()
                                modern_renderer.microscope_view.set_terrain(terrain_type)
                                populate_microscope_world(modern_renderer.microscope_view.world, terrain_type)
                            camera.set_status(f"Microscope: viewing {terrain_type} life")
                    gamepad_speed_cooldown = 15
            
            # DPAD = Tool cycling (left/right) and tool radius (up/down)
            if not camera.menu_open and gamepad_speed_cooldown <= 0:
                if gamepad.get_dpad(GamepadConfig.DPAD_LEFT):
                    camera.prev_tool()
                    if USE_MODERN_RENDERER and modern_renderer:
                        modern_renderer.hud_prev_tool()
                    gamepad_speed_cooldown = 15
                elif gamepad.get_dpad(GamepadConfig.DPAD_RIGHT):
                    camera.next_tool()
                    if USE_MODERN_RENDERER and modern_renderer:
                        modern_renderer.hud_next_tool()
                    gamepad_speed_cooldown = 15
                elif gamepad.get_dpad(GamepadConfig.DPAD_UP) or gamepad.get_dpad(GamepadConfig.DPAD_DOWN):
                    if camera.current_tool in (ToolType.MINE, ToolType.FILL):
                        camera.cycle_tool_radius()
                        gamepad_speed_cooldown = 15
            
            # Y button = screenshot
            if gamepad.get_button(GamepadConfig.Y) and gamepad_speed_cooldown <= 0:
                take_screenshot(camera)
                gamepad_speed_cooldown = 30
            
            # START button handled via JOYBUTTONDOWN event for reliability
            
            # SELECT button = warp to new waverse
            if gamepad.get_button(GamepadConfig.SELECT) and gamepad_speed_cooldown <= 0:
                warp_to_new_universe(camera, chunk_manager, chunk_renderer, 
                                    flora_manager, animal_manager,
                                    chunk_dna_manager, climate_manager,
                                    structure_manager, config, life_simulator)
                gamepad_speed_cooldown = 60  # Longer cooldown for warp
            
        
        # Keyboard space = jump (in walk mode) - only when menu closed
        # Variable jump: hold for higher jump, tap for short hop
        # Only update jump_held from keyboard if NOT using gamepad for jump
        if not camera.menu_open and not camera.flying:
            if keys[pygame.K_SPACE]:
                camera.jump()  # Start jump if grounded
                camera.update_jump_held(True)  # Held = higher jump
            elif not (gamepad and gamepad.get_button(GamepadConfig.B)):
                # Only release if gamepad B isn't being held either
                camera.update_jump_held(False)  # Released = cut jump short
        
        # Keyboard B = run (double speed while held) - OR with gamepad B
        if keys[pygame.K_b]:
            camera.is_running = True
        # Don't reset if keyboard B not pressed - gamepad B may have set it
        
        # Auto-fly mode - overrides manual movement
        if camera.auto_fly_mode > 0:
            auto_forward, auto_right, auto_up = camera.update_auto_fly(
                dt, chunk_manager, flora_manager, animal_manager, structure_manager)
            forward += auto_forward
            right += auto_right
            up += auto_up
        
        camera.move(forward, right, up, chunk_manager, structure_manager, water_level)
        
        # Maintain height during auto-fly
        camera.maintain_auto_fly_height(chunk_manager)
        
        # Update chunks (stream in new ones as we move)
        current_chunk = camera.get_chunk_pos()
        if not USE_MODERN_RENDERER:
            chunk_renderer.update_chunks(current_chunk)
        chunk_worker.update_player_position(current_chunk[0], current_chunk[1])
        last_chunk = current_chunk
        
        # Update minimap occasionally
        lx, lz = minimap.last_pos
        if abs(camera.x - lx) > 100 or abs(camera.z - lz) > 100:
            minimap.update(camera.x, camera.z)
        
        # Update sky/time - also update sky DNA based on current chunk
        sky.update(dt)
        sky.update_for_chunk(*current_chunk)
        
        # Update climate/weather
        climate_manager.update(dt, *current_chunk)
        weather_renderer.update(dt, camera.x, camera.y, camera.z,
                               climate_manager.current_weather,
                               climate_manager.current_biome)
        
        # Update life simulation (growth, death, reproduction)
        is_raining = climate_manager.current_weather in ('rain', 'heavy_rain', 'storm')
        is_day = sky.time > 0.25 and sky.time < 0.75
        sun_intensity = max(0, math.sin(sky.time * math.pi * 2 - math.pi/2)) if is_day else 0
        
        # Build chunk lookups for life sim
        plants_by_chunk = {}
        animals_by_chunk = {}
        total_plants = 0
        total_animals = 0
        for (cx, cz), plants in flora_manager.chunk_plants.items():
            plants_by_chunk[(cx, cz)] = plants
            total_plants += len(plants)
        for (cx, cz), animals in animal_manager.chunk_animals.items():
            animals_by_chunk[(cx, cz)] = animals
            total_animals += len(animals)
        
        _life_start = time.perf_counter()
        # Graduated life simulation based on speed
        # speed_factor: 0.0 = still, 0.3 = walking, 0.5 = running, 0.8 = flying fast, 1.0 = max
        speed_factor = camera.get_speed_factor()
        
        # Determine update frequency based on speed (graduated stages)
        # Normal: every frame, Fast: every 2nd, Faster: every 4th, Max: every 8th
        if speed_factor < 0.6:
            life_update_interval = 1  # Every frame (normal walking/running)
        elif speed_factor < 0.75:
            life_update_interval = 2  # Every 2nd frame (fast running)
        elif speed_factor < 0.9:
            life_update_interval = 4  # Every 4th frame (very fast/flying)
        else:
            life_update_interval = 8  # Every 8th frame (max speed flying)
        
        should_update_life = (frame_count % life_update_interval == 0)
        
        if not (NO_FLORA_UPDATE and NO_ANIMAL_UPDATE) and should_update_life:
            # Scale dt to compensate for skipped frames
            effective_dt = dt * life_update_interval
            life_simulator.update(
                effective_dt,
                camera.x, camera.z,
                is_raining, is_day, sun_intensity,
                plants_by_chunk, animals_by_chunk,
                chunk_size=CHUNK_SIZE * TERRAIN_SCALE,
                skip_flora=NO_FLORA_UPDATE,
                skip_animals=NO_ANIMAL_UPDATE
            )
        perf.record_time('life', _life_start)
        
        # Refresh display lists for chunks where plants changed (eaten/died)
        # Skip in modern mode - no display lists used
        if not NO_FLORA_UPDATE and not (USE_MODERN_RENDERER and modern_renderer):
            for chunk_key in life_simulator.chunks_needing_refresh:
                if chunk_key in flora_manager.display_lists:
                    # Delete old display lists
                    for dl in flora_manager.display_lists[chunk_key]:
                        glDeleteLists(dl, 1)
                    del flora_manager.display_lists[chunk_key]
        life_simulator.chunks_needing_refresh.clear()
        
        # Process pending births - spawn animals from hatched eggs
        if not NO_ANIMAL_UPDATE:
            for chunk_key, births in list(life_simulator.pending_births.items()):
                for birth in births:
                    # Spawn new animal at the egg's position (on ground)
                    animal_manager.spawn_baby_animal(
                        birth['x'], 
                        birth['y'],  # Egg was on ground
                        birth['z'],
                        birth.get('parent_dna', {})
                    )
            life_simulator.pending_births.clear()
        
        # Render
        _render_start = time.perf_counter()
        cam_pos = camera.get_pos()
        
        # Use modern renderer for terrain/flora if available
        if USE_MODERN_RENDERER and modern_renderer:
            # Modern renderer handles everything via shaders
            # Just need to sync camera and time/weather
            modern_renderer.set_camera_from_waverse(camera, display[0]/display[1])
            modern_renderer.set_time_of_day(getattr(sky, 'time', 0.5))
            modern_renderer.set_lighting(fog_start=2000.0, fog_end=6000.0)  # Extended for far render
            modern_renderer.set_water_level(water_level * HEIGHT_SCALE)  # Convert to world units
            modern_renderer.set_hud_yaw(camera.yaw)  # Update compass
            modern_renderer.hud_set_tool(camera.current_tool)  # Sync tool selection
            modern_renderer.sync_camera_menu(camera)  # Sync menu state
            
            # Sync weather with biome
            weather = climate_manager.current_weather
            weather_type = weather.precipitation_type if hasattr(weather, 'precipitation_type') else 'none'
            intensity = weather.precipitation if hasattr(weather, 'precipitation') else 0.0
            current_biome = climate_manager.current_biome.get_biome_name() if climate_manager.current_biome else None
            modern_renderer.set_weather(weather_type, intensity, biome=current_biome)
            
            # Update chunk loading based on camera position
            modern_renderer.update_chunks_around_camera(
                camera, chunk_manager, flora_manager, animal_manager, structure_manager, climate_manager
            )
            
            # Sync dropped items, debris, and fallen trees for rendering
            modern_renderer.set_wood_chunks(dropped_items)
            modern_renderer.set_debris_particles(debris_particles)
            modern_renderer.set_fallen_trees(fallen_trees)
            
            # Render everything via modern renderer (or microscope if active)
            # Convert normalized dt (1.0 = 60fps) to seconds for renderer
            dt_seconds = dt * 0.01667  # dt * (1/60)
            modern_renderer.render(dt=dt_seconds, microscope_active=camera.microscope_active)
            
            # Render entity preview in menu (uses legacy GL after modern render)
            preview_info = modern_renderer.get_menu_preview_info()
            if preview_info[0]:  # should_render
                _, px, py, psize, json_filename = preview_info
                json_path = os.path.join(DNA_LOGS_DIR, json_filename)
                if os.path.exists(json_path):
                    draw_entity_preview(json_path, px, py, psize, dt=dt)
        else:
            # Set sky color and fog based on time of day (now with DNA tint)
            sky_color = sky.get_sky_color()
            glClearColor(*sky_color, 1.0)
            sky.apply_fog()
            sky.apply_lighting()
            
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            camera.apply()
            # Legacy: Render sky (sun/moon/stars)
            sky.render(cam_pos[0], cam_pos[1], cam_pos[2])
            
            # Legacy: Render clouds with DNA-based visuals
            weather_renderer.render_clouds(cam_pos[0], cam_pos[1], cam_pos[2],
                                           climate_manager.current_weather.cloud_cover,
                                           climate_manager.time,
                                           climate_manager.current_weather.visual_dna)
        
        _chunk_start = time.perf_counter()
        
        # Continue rendering for legacy mode
        if USE_MODERN_RENDERER and modern_renderer:
            # Modern renderer already rendered everything above
            # Just record timing
            perf.record_time('chunk', _chunk_start)
            perf.record_time('flora', _chunk_start)  # Combined in modern renderer
        else:
            # Legacy terrain rendering
            chunk_renderer.render()
            perf.record_time('chunk', _chunk_start)
            
            # Legacy flora rendering
            _flora_start = time.perf_counter()
            
            # Fixed render radius - consistent to avoid flickering
            FLORA_RENDER_RADIUS = RenderConfig.FLORA_RENDER_RADIUS
            height_above_ground = camera.y - chunk_manager.get_height_at(camera.x, camera.z)
            
            MAX_NEW_CHUNKS_PER_FRAME = 3
            cam_cx, cam_cz = current_chunk
            cam_x, cam_z = cam_pos[0], cam_pos[2]
            new_chunks = 0
            
            # Sort chunks by distance from camera (radial order, not stripes)
            chunks_by_distance = []
            for (cx, cz), (display_list, lod) in chunk_renderer.display_lists.items():
                dx, dz = cx - cam_cx, cz - cam_cz
                if abs(dx) > FLORA_RENDER_RADIUS or abs(dz) > FLORA_RENDER_RADIUS:
                    continue
                dist_sq = dx * dx + dz * dz
                chunks_by_distance.append((dist_sq, cx, cz, display_list, lod))
            
            # Process closest chunks first
            chunks_by_distance.sort(key=lambda x: x[0])
            
            for dist_sq, cx, cz, display_list, lod in chunks_by_distance:
                key = (cx, cz)
                
                needs_generation = key not in flora_manager.chunk_plants
                if needs_generation:
                    if new_chunks >= MAX_NEW_CHUNKS_PER_FRAME:
                        continue
                    new_chunks += 1
                
                chunk = chunk_manager.get_chunk(cx, cz)
                
                if not NO_FLORA_RENDER:
                    flora_manager.render_chunk_flora(
                        cx, cz, cam_x, cam_z,
                        chunk.heightmap, chunk.world_x, chunk.world_z, TILE_SCALE, HEIGHT_SCALE,
                        height_above_ground=height_above_ground, speed_factor=speed_factor
                    )
                
                if key not in animal_manager.chunk_animals:
                    # Get biome for this chunk
                    biome_name = None
                    if climate_manager:
                        biome_dna = climate_manager.get_biome(cx, cz)
                        if biome_dna:
                            biome_name = biome_dna.get_biome_name()
                    animal_manager.spawn_animals_for_chunk(
                        cx, cz, chunk.heightmap, chunk.world_x, chunk.world_z, TILE_SCALE, HEIGHT_SCALE,
                        biome_name=biome_name
                    )
                if needs_generation:
                    structure_manager.spawn_random_buildings(
                        cx, cz, chunk.world_x, chunk.world_z, chunk.heightmap, HEIGHT_SCALE, TILE_SCALE
                    )
            
            perf.record_time('flora', _flora_start)
        
        # Always spawn animals/structures even with modern renderer
        if USE_MODERN_RENDERER and modern_renderer:
            cam_cx, cam_cz = current_chunk
            # Build list sorted by distance (radial order)
            spawn_chunks = []
            for dx in range(-12, 13):
                for dz in range(-12, 13):
                    dist_sq = dx * dx + dz * dz
                    if dist_sq <= 12 * 12:  # Circular, not square
                        spawn_chunks.append((dist_sq, cam_cx + dx, cam_cz + dz))
            spawn_chunks.sort(key=lambda x: x[0])
            
            for _, cx, cz in spawn_chunks:
                key = (cx, cz)
                chunk = chunk_manager.chunks.get(key)
                if chunk:
                    # Get biome for this chunk
                    biome_name = None
                    if climate_manager:
                        biome_dna = climate_manager.get_biome(cx, cz)
                        if biome_dna:
                            biome_name = biome_dna.get_biome_name()
                    
                    if key not in animal_manager.chunk_animals:
                        animal_manager.spawn_animals_for_chunk(
                            cx, cz, chunk.heightmap, chunk.world_x, chunk.world_z, TILE_SCALE, HEIGHT_SCALE,
                            biome_name=biome_name
                        )
                    if key not in flora_manager.chunk_plants:
                        structure_manager.spawn_random_buildings(
                            cx, cz, chunk.world_x, chunk.world_z, chunk.heightmap, HEIGHT_SCALE, TILE_SCALE
                        )
        
        # Update animals every few frames for performance
        _animal_start = time.perf_counter()
        # Graduated animal update frequency based on speed
        if speed_factor < 0.6:
            animal_update_interval = 3  # Every 3rd frame (normal)
        elif speed_factor < 0.8:
            animal_update_interval = 5  # Every 5th frame (fast)
        else:
            animal_update_interval = 8  # Every 8th frame (very fast)
        
        if not NO_ANIMAL_UPDATE and frame_count % animal_update_interval == 0:
            # Ground height function for animals to stay on terrain
            def get_ground_height(x, z):
                return chunk_manager.get_height_at(x, z) * HEIGHT_SCALE
            animal_manager.update(dt, cam_pos, get_ground_height)
        
        # Update dropped items physics (falling, bouncing)
        for dropped in dropped_items[:]:
            terrain_h = chunk_manager.get_height_at(dropped.x, dropped.z) * HEIGHT_SCALE
            dropped.update(dt, terrain_h)
            # Remove expired items
            if dropped.lifetime <= 0:
                dropped_items.remove(dropped)
        
        # Update debris particles (visual effects, fade out)
        for debris in debris_particles[:]:
            terrain_h = chunk_manager.get_height_at(debris.x, debris.z) * HEIGHT_SCALE
            debris.update(dt, terrain_h)
            if not debris.is_alive():
                debris_particles.remove(debris)
        
        # Update fallen trees (tipping animation, fade out)
        for fallen in fallen_trees[:]:
            terrain_h = chunk_manager.get_height_at(fallen.x, fallen.z) * HEIGHT_SCALE
            fallen.update(dt, terrain_h)
            if not fallen.is_alive():
                fallen_trees.remove(fallen)
        
        # Auto-pickup items when walking over them
        pickup_nearby_items(camera, dropped_items, pickup_radius=2.5)
        
        # Skip legacy rendering in modern mode (modern renderer handles all of this)
        if not (USE_MODERN_RENDERER and modern_renderer):
            # Render animals (skip when very high up - they're invisible anyway)
            if not NO_ANIMAL_RENDER and height_above_ground < 150:
                animal_manager.render(cam_pos[0], cam_pos[1], cam_pos[2])
            perf.record_time('animal', _animal_start)
            
            # Render eggs from life simulation
            life_simulator.render_eggs(cam_pos[0], cam_pos[2])
            
            # Render structures (buildings)
            structure_manager.render(cam_pos[0], cam_pos[1], cam_pos[2])
            
            # Render water plane at water level, centered on camera
            render_water(camera.x, camera.z, water_level)
            
            # Render dropped items
            render_dropped_items(dropped_items, camera.x, camera.y, camera.z)
            
            # Render debris particles (fading visual effects)
            render_debris(debris_particles, camera.x, camera.y, camera.z)
            
            # Render fallen tree trunks (tipping over animation)
            render_fallen_trees(fallen_trees, camera.x, camera.y, camera.z)
            
            # Render weather effects (rain/snow particles, lightning)
            weather_renderer.render(cam_pos[0], cam_pos[1], cam_pos[2],
                                   climate_manager.current_weather,
                                   climate_manager.lightning_flash)
        else:
            perf.record_time('animal', _animal_start)
        
        # Cleanup distant chunks occasionally (always run, not rendering-related)
        if frame_count % 60 == 0:
            flora_manager.cleanup_distant_chunks(current_chunk[0], current_chunk[1])
            animal_manager.cleanup_distant_chunks(current_chunk[0], current_chunk[1])
            chunk_dna_manager.cleanup_distant(current_chunk[0], current_chunk[1])
            climate_manager.cleanup_distant(current_chunk[0], current_chunk[1])
            structure_manager.cleanup_distant(current_chunk[0], current_chunk[1])
        
        perf.record_time('render', _render_start)
        
        # Update perf counts (every 30 frames to save CPU)
        if frame_count % 30 == 0:
            perf.update_counts(
                total_plants,
                total_animals,
                len(life_simulator.eggs),
                len(chunk_renderer.display_lists),
                len(structure_manager.structures)
            )
        
        # HUD/overlay rendering (uses legacy OpenGL, skip in modern mode on macOS Core profile)
        if not (USE_MODERN_RENDERER and modern_renderer):
            draw_crosshair(display)
            draw_hud(display, camera, sky, climate_manager, hud_font)
            draw_menu(display, camera, hud_font)  # Draw menu overlay if open
            minimap.draw(display, camera.x, camera.z)
            
            # Perf overlay (if enabled)
            if perf.show_overlay:
                draw_perf_overlay(display, perf, hud_font)
        # TODO: Add modern HUD rendering for Core profile
        
        # Capture pending screenshot (after render, before flip)
        capture_pending_screenshot(camera)
        
        pygame.display.flip()
        perf.end_frame()
    
    # Cleanup
    chunk_worker.stop()
    chunk_renderer.stop()
    
    # Cleanup modern renderer
    if modern_renderer:
        modern_renderer.cleanup()
    
    pygame.quit()


if __name__ == '__main__':
    run_explorer()

