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

try:
    from OpenGL.GL import *
    from OpenGL.GL import GLubyte
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not found.")

from .world import WorldConfig, ChunkManager, CHUNK_SIZE, TILE_SCALE, get_height
from .flora import FloraManager
from .sky import SkySystem
from .animals import AnimalManager
from .chunk_worker import ChunkWorker


# Configuration - open world style
HEIGHT_SCALE = 3.5
TERRAIN_SCALE = TILE_SCALE
BASE_MOVE_SPEED = 0.8  # Base movement speed (adjustable with 1-5 keys)
MOUSE_SENSITIVITY = 0.06
CHUNK_RENDER_DISTANCE = 20  # Massive view distance!
PLAYER_HEIGHT = 2.5  # Eye height above ground (shorter = world feels bigger, fits through doors)
WALK_SMOOTH_SPEED = 0.4  # Faster terrain following

# Speed levels (keys: ` 1 2 3 4 5)
SPEED_LEVELS = [0.05, 0.15, 0.3, 0.6, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]  # Ultra-slow to insane

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
            return self._detect_gamepad()
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
        x = self.get_axis(GamepadConfig.L_STICK_X, GamepadConfig.L_STICK_X_INV)
        y = self.get_axis(GamepadConfig.L_STICK_Y, GamepadConfig.L_STICK_Y_INV)
        # Forward is -Y on stick (up = negative Y), right is +X
        return (-y, x)
    
    def get_look(self) -> tuple:
        """Get look from right stick (yaw, pitch)."""
        x = self.get_axis(GamepadConfig.R_STICK_X, GamepadConfig.R_STICK_X_INV)
        y = self.get_axis(GamepadConfig.R_STICK_Y, GamepadConfig.R_STICK_Y_INV)
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


def take_screenshot():
    """Take a screenshot of the current view including HUD."""
    import os
    from datetime import datetime
    
    print("  Taking screenshot...")
    
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
    
    return filename


def render_entity_to_png(entity, entity_type: str, filename: str, size: int = 512):
    """Render a plant or animal to a PNG with transparent background."""
    from waverse.flora import PlantRenderer, PlantInstance
    from waverse.animals import AnimalRenderer
    
    # Save current OpenGL state
    viewport = glGetIntegerv(GL_VIEWPORT)
    
    # Disable fog for clean render
    glDisable(GL_FOG)
    
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
    else:
        # Animals - estimate size from body segments
        entity_height = 3.0
        entity_width = 3.0
        if hasattr(entity.dna, 'body_segments') and entity.dna.body_segments:
            total_size = sum(seg.size for seg in entity.dna.body_segments)
            entity_height = total_size * 1.5
            entity_width = total_size * 2
    
    # Add padding - make view big enough
    view_size = max(entity_height, entity_width, 5.0) * 2.0
    half_size = view_size / 2
    
    # Orthographic projection - center the view
    glOrtho(-half_size, half_size, -half_size * 0.3, half_size * 1.7, -100, 100)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    # Look at entity from the front-right, slightly above
    gluLookAt(
        half_size * 0.7, half_size * 0.5, half_size * 0.7,  # Eye position
        0, entity_height * 0.4, 0,  # Look at center of entity
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
    
    # Restore state
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glViewport(viewport[0], viewport[1], viewport[2], viewport[3])
    
    # Restore fog and clear color
    glEnable(GL_FOG)
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
    
    # Convert DNA to dict and save
    try:
        dna_dict = asdict(best_entity.dna)
        dna_string = json.dumps(dna_dict, indent=2, default=str)
        
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
        
        return filename
    except Exception as e:
        print(f"  Error logging DNA: {e}")
        return None


def modify_terrain_at_cursor(camera, chunk_manager, mode: str = "MINE", amount: float = 2.0):
    """Modify terrain at cursor position (MINE = lower, FILL = raise).
    
    Args:
        camera: Camera object
        chunk_manager: ChunkManager for terrain access
        mode: "MINE" to lower terrain, "FILL" to raise terrain
        amount: How much to modify (default 2.0 units)
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
        local_x = int((px - chunk.world_x) / chunk.tile_size)
        local_z = int((pz - chunk.world_z) / chunk.tile_size)
        
        # Bounds check
        h, w = chunk.heightmap.shape
        if 0 <= local_x < w and 0 <= local_z < h:
            terrain_height = chunk.heightmap[local_z, local_x] * HEIGHT_SCALE
            
            # Check if ray is at or below terrain
            if py <= terrain_height:
                # Found intersection! Modify terrain
                if mode == "MINE":
                    # Lower terrain
                    delta = -amount / HEIGHT_SCALE
                    chunk.heightmap[local_z, local_x] += delta
                    
                    # Also affect neighbors slightly for smoother result
                    for dz in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            nx, nz = local_x + dx, local_z + dz
                            if 0 <= nx < w and 0 <= nz < h and (dx != 0 or dz != 0):
                                chunk.heightmap[nz, nx] += delta * 0.3
                    
                    print(f"  [MINE] Lowered terrain at ({px:.1f}, {pz:.1f})")
                    
                elif mode == "FILL":
                    # Raise terrain
                    delta = amount / HEIGHT_SCALE
                    chunk.heightmap[local_z, local_x] += delta
                    
                    # Also affect neighbors slightly
                    for dz in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            nx, nz = local_x + dx, local_z + dz
                            if 0 <= nx < w and 0 <= nz < h and (dx != 0 or dz != 0):
                                chunk.heightmap[nz, nx] += delta * 0.3
                    
                    print(f"  [FILL] Raised terrain at ({px:.1f}, {pz:.1f})")
                
                # Mark chunk as needing re-render
                chunk_key = (cx, cz)
                # The chunk renderer will need to regenerate the display list
                # We'll handle this by removing from the cache
                return (cx, cz)
    
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
                         structure_manager, config):
    """Warp to a far-away location with completely fresh DNA AND terrain - a new universe!"""
    print("=" * 60)
    print("  WARPING TO NEW UNIVERSE...")
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
    camera.flying = True
    
    print("  Welcome to a new universe!")
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
    
    ALL_TOOLS = [SCAN, MINE, FILL]


class Camera:
    """Camera with walking, jumping, flying, and auto-fly modes."""
    
    def __init__(self):
        self.x = 0
        self.y = 40
        self.z = 0
        self.yaw = 0
        self.pitch = -20
        self.flying = True
        self.target_y = 40
        self.speed_level = 4  # Default speed (1.0x) - index into SPEED_LEVELS
        
        # Jump physics
        self.jumping = False
        self.jump_velocity = 0.0
        self.gravity = 2.5  # Gravity acceleration (faster fall)
        self.jump_strength = 0.5  # Initial jump velocity
        
        # Auto-fly / Tour mode: 0=off, 1=wander
        self.auto_fly_mode = 0
        
        # Wander mode state - movement direction (independent of view)
        self.wander_timer = 0.0
        self.wander_direction = 0.0  # Radians - actual movement direction
        
        # Tour mode height and view
        self.auto_fly_height = 40.0  # Height above terrain (above trees)
        self.view_yaw_offset = 0.0   # Look offset from movement direction
        self.view_pitch = -5.0       # View pitch (default slight down)
        self.view_return_timer = 0.0 # Timer for returning to forward
        
        # Tool system
        self.current_tool_index = 0  # Index into ToolType.ALL_TOOLS
        self.current_tool = ToolType.SCAN
        
        # Markers for compass (list of (x, z, color, name) tuples)
        self.markers = []
        
        # Menu state
        self.menu_open = False
        self.menu_tab = 0  # 0 = Inventory, 1 = Log
        self.log_index = 0  # Currently selected log item
        self.log_items = []  # List of logged DNA files
    
    def toggle_menu(self):
        """Toggle menu open/closed."""
        self.menu_open = not self.menu_open
        if self.menu_open:
            # Refresh log items when opening menu
            self.refresh_log_items()
            print("  [Menu] Opened")
        else:
            print("  [Menu] Closed")
    
    def refresh_log_items(self):
        """Refresh list of logged DNA items."""
        self.log_items = []
        if os.path.exists(DNA_LOGS_DIR):
            for f in sorted(os.listdir(DNA_LOGS_DIR), reverse=True):
                if f.endswith('.json'):
                    self.log_items.append(f)
        self.log_index = min(self.log_index, max(0, len(self.log_items) - 1))
    
    def menu_navigate(self, direction: int):
        """Navigate in menu (direction: -1 = left/up, 1 = right/down)."""
        if self.menu_tab == 1:  # Log tab
            self.log_index = max(0, min(len(self.log_items) - 1, self.log_index + direction))
    
    def menu_switch_tab(self, direction: int):
        """Switch menu tab (direction: -1 = left, 1 = right)."""
        self.menu_tab = (self.menu_tab + direction) % 2  # 2 tabs: Inventory, Log
    
    def add_marker(self):
        """Add a marker at current position."""
        colors = [(0.2, 0.6, 1.0), (0.2, 1.0, 0.4), (1.0, 0.8, 0.2), 
                  (1.0, 0.4, 0.4), (0.8, 0.4, 1.0), (1.0, 0.6, 0.2)]
        color = colors[len(self.markers) % len(colors)]
        name = chr(ord('A') + len(self.markers))
        self.markers.append((self.x, self.z, color, name))
        print(f"  Marker {name} set at ({self.x:.1f}, {self.z:.1f})")
    
    def clear_markers(self):
        """Clear all markers."""
        self.markers.clear()
        print("  Markers cleared")
    
    def next_tool(self):
        """Switch to next tool (R1)."""
        self.current_tool_index = (self.current_tool_index + 1) % len(ToolType.ALL_TOOLS)
        self.current_tool = ToolType.ALL_TOOLS[self.current_tool_index]
        print(f"  Tool: {self.current_tool}")
    
    def prev_tool(self):
        """Switch to previous tool (L1)."""
        self.current_tool_index = (self.current_tool_index - 1) % len(ToolType.ALL_TOOLS)
        self.current_tool = ToolType.ALL_TOOLS[self.current_tool_index]
        print(f"  Tool: {self.current_tool}")
        
        # Waypoint markers (for compass)
        self.markers = []  # List of (x, z, color, name) tuples
        self.max_markers = 10  # Limit markers
    
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
            self.pitch = max(-89, min(89, self.pitch))
    
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
            self.pitch = max(-89, min(89, self.pitch))
    
    def get_terrain_height(self, chunk_manager: ChunkManager) -> float:
        """Get terrain height at current position."""
        return chunk_manager.get_height_at(self.x, self.z) * HEIGHT_SCALE
    
    def move(self, forward, right, up, chunk_manager: ChunkManager, structure_manager=None):
        # Get current speed based on level
        speed = BASE_MOVE_SPEED * SPEED_LEVELS[self.speed_level]
        
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        if self.flying:
            forward_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
            forward_y = math.sin(pitch_rad)
            forward_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        else:
            # Walking: move along ground plane only
            forward_x = -math.sin(yaw_rad)
            forward_y = 0
            forward_z = -math.cos(yaw_rad)
        
        right_x = math.cos(yaw_rad)
        right_z = -math.sin(yaw_rad)
        
        # Calculate new position
        new_x = self.x + (forward * forward_x + right * right_x) * speed
        new_z = self.z + (forward * forward_z + right * right_z) * speed
        
        # Check structure collision at new position
        blocked = False
        structure_floor = None
        if structure_manager:
            blocked, structure_floor = structure_manager.check_collision(new_x, self.y, new_z)
        
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
        
        if self.flying:
            # Calculate new Y position
            new_y = self.y + forward * forward_y * speed + up * speed
            
            # Check for ceiling collision at new height
            ceiling_blocked = False
            if structure_manager and up > 0:
                ceiling_blocked, _ = structure_manager.check_collision(self.x, new_y, self.z)
            
            if not ceiling_blocked:
                self.y = new_y
            
            # Recheck floor at new position
            if structure_manager:
                _, new_floor = structure_manager.check_collision(self.x, self.y, self.z)
                if new_floor is not None:
                    effective_ground = max(terrain_ground, new_floor + PLAYER_HEIGHT)
            
            # Land when pressing down and at ground level
            if up < 0 and self.y <= effective_ground + 0.5:
                self.y = effective_ground
                self.flying = False
                self.target_y = effective_ground
            
            # Don't go below ground/floor
            if self.y < effective_ground:
                self.y = effective_ground
        else:
            # Walking/jumping mode
            self.target_y = effective_ground
            
            if self.jumping:
                # Apply jump physics
                self.jump_velocity -= self.gravity * 0.016  # Gravity
                self.y += self.jump_velocity
                
                # Landed?
                if self.y <= effective_ground:
                    self.y = effective_ground
                    self.jumping = False
                    self.jump_velocity = 0.0
            else:
                # Follow terrain/floor smoothly
                diff = self.target_y - self.y
                
                # Fast catch-up if far from terrain, smooth otherwise
                if abs(diff) > 2:
                    self.y += diff * 0.5  # Fast snap
                else:
                    self.y += diff * WALK_SMOOTH_SPEED  # Smooth follow
            
            # Fly when pressing up (H key or R3)
            if up > 0:
                self.flying = True
                self.jumping = False
                self.y += speed
    
    def jump(self):
        """Start a jump if on the ground and not already jumping."""
        if not self.flying and not self.jumping:
            self.jumping = True
            self.jump_velocity = self.jump_strength
    
    def add_marker(self):
        """Add a marker at current position."""
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
        
        # Use letters A, B, C...
        labels = "ABCDEFGHIJ"
        marker_idx = len(self.markers)
        label = labels[marker_idx] if marker_idx < len(labels) else "?"
        color = colors[marker_idx % len(colors)]
        
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
        """Toggle tour/wander mode on/off."""
        self.auto_fly_mode = 1 - self.auto_fly_mode  # Toggle 0 <-> 1
        
        if self.auto_fly_mode == 0:
            print("=" * 40)
            print("  TOUR MODE: OFF")
            print("=" * 40)
        else:
            print("=" * 40)
            print("  TOUR MODE: ON")
            print("  Auto-flying, random exploration")
            print("  Look around freely - returns to forward")
            print("  Press X to turn OFF")
            print("=" * 40)
            # Start wandering in current facing direction
            self.wander_direction = math.radians(self.yaw)
            self.wander_timer = 30 + random.random() * 30
            self.flying = True
            # Reset view to forward
            self.view_yaw_offset = 0.0
            self.view_pitch = -5.0
            self.view_return_timer = 0.0
    
    def update_auto_fly(self, dt: float, chunk_manager: ChunkManager):
        """Update tour mode - movement and view are independent."""
        if self.auto_fly_mode == 0:
            return 0, 0, 0
        
        dt_seconds = dt / 60.0
        speed_mult = SPEED_LEVELS[self.speed_level] * 3.0
        
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
        
        move_speed = speed_mult * 0.5 * dt_seconds * 60  # Convert back to per-frame
        self.x += move_x * move_speed
        self.z += move_z * move_speed
        
        return 0, 0, 0  # We moved directly, don't use forward/right/up
    
    def maintain_auto_fly_height(self, chunk_manager: ChunkManager):
        """Keep camera at consistent height above terrain during tour mode."""
        if self.auto_fly_mode == 0:
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
        """Set speed level (0-4, corresponds to keys 1-5)."""
        self.speed_level = max(0, min(4, level))
    
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
    
    # Menu box
    menu_w = min(600, display[0] - 100)
    menu_h = min(400, display[1] - 100)
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
    tab_names = ["INVENTORY", "LOG"]
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
        if len(camera.log_items) == 0:
            if hud_font:
                _draw_text(hud_font, "No DNA logs yet. Use SCAN tool to log creatures!", 
                          menu_x + 20, content_y, (200, 200, 200))
        else:
            # Show list of log items
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
                    glVertex2f(menu_x + menu_w - 10, iy - 2)
                    glVertex2f(menu_x + menu_w - 10, iy + item_h - 4)
                    glVertex2f(menu_x + 10, iy + item_h - 4)
                    glEnd()
                
                # Item text
                if hud_font:
                    # Parse filename for type and timestamp
                    parts = item.replace('.json', '').split('_')
                    if len(parts) >= 3:
                        entity_type = parts[0]  # plant or animal
                        date = parts[2] if len(parts) > 2 else ""
                        time = parts[3] if len(parts) > 3 else ""
                        display_text = f"{entity_type.upper()} - {date} {time}"
                    else:
                        display_text = item
                    
                    color = (100, 200, 255) if start_idx + i == camera.log_index else (180, 180, 180)
                    _draw_text(hud_font, display_text, menu_x + 20, iy, color)
    
    # Help text at bottom
    if hud_font:
        help_y = menu_y + menu_h - 25
        _draw_text(hud_font, "D-Pad/Stick: Navigate | START: Close", 
                  menu_x + 20, help_y, (150, 150, 150))
    
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
    top_bar_h = 36
    
    # Coordinates background (left side, after minimap ~190px)
    coord_x = 200
    coord_w = 130  # Sized for ~12 digit coordinates
    glColor4f(0, 0, 0, 0.6)
    glBegin(GL_QUADS)
    glVertex2f(coord_x, 10)
    glVertex2f(coord_x + coord_w, 10)
    glVertex2f(coord_x + coord_w, 10 + top_bar_h)
    glVertex2f(coord_x, 10 + top_bar_h)
    glEnd()
    
    # Draw coordinate text using pygame font if available
    if hud_font:
        # X coordinate
        x_text = f"X: {camera.x:.1f}"
        y_text = f"Y: {camera.z:.1f}"  # Z is the "Y" in top-down view
        
        _draw_text(hud_font, x_text, coord_x + 10, 14, (255, 180, 100))
        _draw_text(hud_font, y_text, coord_x + 10, 28, (100, 200, 255))
    else:
        # Fallback: draw simple coordinate indicators
        glColor4f(1.0, 0.7, 0.4, 1.0)
        _draw_number(coord_x + 10, 18, camera.x)
        glColor4f(0.4, 0.8, 1.0, 1.0)
        _draw_number(coord_x + 10, 32, camera.z)
    
    # Compass background (2x wider)
    compass_x = coord_x + coord_w + 10
    compass_w = 400  # 2x wider
    glColor4f(0, 0, 0, 0.6)
    glBegin(GL_QUADS)
    glVertex2f(compass_x, 10)
    glVertex2f(compass_x + compass_w, 10)
    glVertex2f(compass_x + compass_w, 10 + top_bar_h)
    glVertex2f(compass_x, 10 + top_bar_h)
    glEnd()
    
    # Compass center line
    compass_center = compass_x + compass_w / 2
    glColor4f(0.5, 0.5, 0.5, 0.8)
    glBegin(GL_LINES)
    glVertex2f(compass_x + 10, 28)
    glVertex2f(compass_x + compass_w - 10, 28)
    glEnd()
    
    # Center tick mark
    glColor4f(1.0, 1.0, 1.0, 0.9)
    glBegin(GL_LINES)
    glVertex2f(compass_center, 22)
    glVertex2f(compass_center, 34)
    glEnd()
    
    # North marker only - red circle with "N"
    yaw = camera.yaw
    
    # North direction (yaw 0)
    north_diff = 0 - yaw
    while north_diff > 180: north_diff -= 360
    while north_diff < -180: north_diff += 360
    
    # Only show if within view range (-90 to +90 degrees)
    if abs(north_diff) < 90:
        pos_x = compass_center + (north_diff / 90) * (compass_w / 2 - 30)
        radius = 12
        
        # Red filled circle
        glColor4f(1.0, 0.3, 0.3, 0.9)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(pos_x, 28)
        for i in range(13):
            angle = (i / 12) * 2 * math.pi
            glVertex2f(pos_x + math.cos(angle) * radius, 28 + math.sin(angle) * radius)
        glEnd()
        
        # White outline
        glColor4f(1, 1, 1, 1)
        glBegin(GL_LINE_LOOP)
        for i in range(12):
            angle = (i / 12) * 2 * math.pi
            glVertex2f(pos_x + math.cos(angle) * radius, 28 + math.sin(angle) * radius)
        glEnd()
        
        # "N" letter inside (simple shape)
        glBegin(GL_LINES)
        # Left vertical
        glVertex2f(pos_x - 4, 32)
        glVertex2f(pos_x - 4, 24)
        # Diagonal
        glVertex2f(pos_x - 4, 24)
        glVertex2f(pos_x + 4, 32)
        # Right vertical
        glVertex2f(pos_x + 4, 32)
        glVertex2f(pos_x + 4, 24)
        glEnd()
    
    # Draw markers on compass (labeled A, B, C...)
    marker_labels = "ABCDEFGHIJ"
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
            pos_x = compass_center + (diff / 90) * (compass_w / 2 - 30)
            
            # Colored marker circle
            glColor4f(*mcolor, 0.9)
            radius = 10
            glBegin(GL_TRIANGLE_FAN)
            glVertex2f(pos_x, 28)
            for j in range(13):
                angle = (j / 12) * 2 * math.pi
                glVertex2f(pos_x + math.cos(angle) * radius, 28 + math.sin(angle) * radius)
            glEnd()
            
            # White outline
            glColor4f(1, 1, 1, 1)
            glBegin(GL_LINE_LOOP)
            for j in range(12):
                angle = (j / 12) * 2 * math.pi
                glVertex2f(pos_x + math.cos(angle) * radius, 28 + math.sin(angle) * radius)
            glEnd()
            
            # Draw letter (A, B, C...) - simplified letter shapes
            label = marker_labels[idx] if idx < len(marker_labels) else "?"
            if label == "A":
                glBegin(GL_LINES)
                glVertex2f(pos_x, 23)
                glVertex2f(pos_x - 4, 33)
                glVertex2f(pos_x, 23)
                glVertex2f(pos_x + 4, 33)
                glVertex2f(pos_x - 2, 29)
                glVertex2f(pos_x + 2, 29)
                glEnd()
            elif label == "B":
                glBegin(GL_LINES)
                glVertex2f(pos_x - 3, 23)
                glVertex2f(pos_x - 3, 33)
                glVertex2f(pos_x - 3, 23)
                glVertex2f(pos_x + 2, 23)
                glVertex2f(pos_x - 3, 28)
                glVertex2f(pos_x + 2, 28)
                glVertex2f(pos_x - 3, 33)
                glVertex2f(pos_x + 2, 33)
                glEnd()
            elif label == "C":
                glBegin(GL_LINE_STRIP)
                glVertex2f(pos_x + 3, 24)
                glVertex2f(pos_x - 2, 24)
                glVertex2f(pos_x - 3, 28)
                glVertex2f(pos_x - 2, 32)
                glVertex2f(pos_x + 3, 32)
                glEnd()
            else:
                # Simple dot for other letters
                glBegin(GL_QUADS)
                glVertex2f(pos_x - 2, 26)
                glVertex2f(pos_x + 2, 26)
                glVertex2f(pos_x + 2, 30)
                glVertex2f(pos_x - 2, 30)
                glEnd()
    
    # Info box
    has_climate = climate is not None
    box_w = 180
    box_h = 70
    if sky:
        box_h += 20
    if has_climate:
        box_h += 40
    box_x = display[0] - box_w - 10
    box_y = 10
    
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
    
    height_pct = min(1, max(0, camera.y / 100))
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
        
        # Time marker
        time_pct = sky.time
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
        
        # Temperature marker
        temp_x = box_x + 10 + biome.temperature * (box_w - 20)
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
        
        # Precipitation fill
        if weather.precipitation > 0.05:
            if weather.precipitation_type == "snow":
                glColor4f(0.9, 0.95, 1.0, 0.9)  # White for snow
            else:
                glColor4f(0.4, 0.6, 0.9, 0.9)  # Blue for rain
            
            precip_w = weather.precipitation * (box_w - 20)
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
        
        cloud_w = weather.cloud_cover * (box_w - 20)
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
    tool_h = 28
    
    # Background
    glColor4f(0, 0, 0, 0.7)
    glBegin(GL_QUADS)
    glVertex2f(tool_x, 15)
    glVertex2f(tool_x + tool_w, 15)
    glVertex2f(tool_x + tool_w, 15 + tool_h)
    glVertex2f(tool_x, 15 + tool_h)
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
    glVertex2f(tool_x + 5, 18)
    glVertex2f(tool_x + tool_w - 5, 18)
    glVertex2f(tool_x + tool_w - 5, 22)
    glVertex2f(tool_x + 5, 22)
    glEnd()
    
    # Draw tool name with font if available
    if hud_font:
        _draw_text(hud_font, camera.current_tool, tool_x + 10, 26, (255, 255, 255))
    
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


def run_explorer(config: WorldConfig = None):
    """Main explorer loop."""
    if not OPENGL_AVAILABLE:
        print("Error: OpenGL not available!")
        return
    
    if config is None:
        config = WorldConfig.create_default()
    
    pygame.init()
    display = (1400, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption(f"Waverse - {config.name}")
    
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    
    # Initialize font for HUD text
    pygame.font.init()
    try:
        hud_font = pygame.font.SysFont("monospace", 14)
    except:
        hud_font = pygame.font.Font(None, 16)
    
    # Initialize gamepad
    gamepad = GamepadManager()
    
    setup_opengl()
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(75, display[0]/display[1], 0.5, 2000)  # Wide FOV, horizon-level clip plane
    glMatrixMode(GL_MODELVIEW)
    
    # Create world with background chunk worker
    chunk_manager = ChunkManager(config)
    chunk_worker = ChunkWorker(config, preload_radius=30)
    chunk_manager.set_worker(chunk_worker)
    chunk_worker.start()
    
    # Chunk DNA for terrain colors and sky - evolves across chunks
    chunk_dna_manager = ChunkDNAManager(config.seed)
    
    chunk_renderer = ChunkRenderer(chunk_manager, chunk_dna_manager)
    flora_manager = FloraManager(config.seed)
    animal_manager = AnimalManager(config.seed)
    sky = SkySystem(chunk_dna_manager)
    water_level = config.water_level
    minimap = Minimap(chunk_manager)
    
    # Climate/weather system
    climate_manager = ClimateManager(config.seed)
    weather_renderer = WeatherRenderer()
    
    # Structure system (buildings, etc.)
    structure_manager = StructureManager(config.seed)
    
    # Camera - try to load saved position
    camera = Camera()
    saved = load_position(config.seed)
    if saved:
        camera.x = saved.get("x", 0)
        camera.y = saved.get("y", 40)
        camera.z = saved.get("z", 0)
        camera.yaw = saved.get("yaw", 0)
        camera.pitch = saved.get("pitch", -20)
        camera.flying = saved.get("flying", True)
        camera.speed_level = saved.get("speed_level", 2)
        print(f"  Loaded position: ({camera.x:.0f}, {camera.y:.0f}, {camera.z:.0f})")
    else:
        camera.y = chunk_manager.get_height_at(0, 0) * HEIGHT_SCALE + 30
    
    # Pre-load chunks around camera position
    start_chunk = camera.get_chunk_pos()
    print("  Loading terrain (this may take a moment on first run)...")
    chunk_renderer.update_chunks(start_chunk, force_all=True)
    total_chunks = len(chunk_renderer.display_lists)
    print(f"  Loaded {total_chunks} chunks!")
    
    # Initial minimap
    minimap.update(0, 0)
    
    print("\n" + "=" * 60)
    print(f"  WAVERSE - {config.name}")
    print("=" * 60)
    print("  KEYBOARD:")
    print("    Movement: WASD/Arrows | H/Space=Up F/Shift=Down")
    print("    Camera: IJKL or Right-Click+Mouse")
    print("    Speed: [/- = slower | ]/= = faster")
    print("    P=Screenshot | B=Use Tool | Tab=Menu | ,/.=Switch Tool | X=Tour | N=Warp")
    if gamepad.is_connected():
        print(f"  GAMEPAD ({gamepad.name}):")
        print("    Left Stick=Move | Right Stick=Look")
        print("    L3=Down | R3=Up | L2/R2=Speed | A=Jump | B=Use Tool | Y=Screenshot")
        print("    L1/R1=Switch Tool | START=Menu | SELECT=Warp")
    print("=" * 60 + "\n")
    
    clock = pygame.time.Clock()
    running = True
    mouse_look = False
    last_chunk = None
    frame_count = 0
    
    # Gamepad speed change cooldown
    gamepad_speed_cooldown = 0
    
    while running:
        frame_count += 1
        dt = clock.tick(60) / 16.67
        
        # Update gamepad speed cooldown
        if gamepad_speed_cooldown > 0:
            gamepad_speed_cooldown -= dt
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
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
                elif event.key == pygame.K_s:
                    save_position(camera, config.seed)
                elif event.key == pygame.K_p:  # P for Picture/Screenshot
                    take_screenshot()
                elif event.key == pygame.K_TAB:  # Tab = toggle menu
                    camera.toggle_menu()
                elif event.key == pygame.K_b:  # B = use current tool
                    if camera.current_tool == ToolType.SCAN:
                        log_dna_at_cursor(camera, flora_manager, animal_manager)
                    elif camera.current_tool == ToolType.MINE:
                        result = modify_terrain_at_cursor(camera, chunk_manager, "MINE")
                        if result:
                            # Invalidate chunk display list to force re-render
                            if result in chunk_renderer.display_lists:
                                del chunk_renderer.display_lists[result]
                    elif camera.current_tool == ToolType.FILL:
                        result = modify_terrain_at_cursor(camera, chunk_manager, "FILL")
                        if result:
                            if result in chunk_renderer.display_lists:
                                del chunk_renderer.display_lists[result]
                elif event.key == pygame.K_COMMA:  # , = previous tool
                    camera.prev_tool()
                elif event.key == pygame.K_PERIOD:  # . = next tool
                    camera.next_tool()
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
                    # N = New location (warp to new universe)
                    warp_to_new_universe(camera, chunk_manager, chunk_renderer, 
                                        flora_manager, animal_manager,
                                        chunk_dna_manager, climate_manager,
                                        structure_manager, config)
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
        
        # Keyboard input
        keys = pygame.key.get_pressed()
        forward = right = up = 0
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
        
        # Gamepad input (check for hotplug every 60 frames)
        if frame_count % 60 == 0:
            gamepad.check_hotplug()
        
        if gamepad.is_connected():
            # Debug mode - print axis values
            if gamepad.debug_mode:
                gamepad.debug_cooldown = max(0, gamepad.debug_cooldown - 1)
                gamepad.print_debug()
            
            # Left stick = movement (WASD)
            gp_move = gamepad.get_movement()
            forward += gp_move[0] * dt * speed
            right += gp_move[1] * dt * speed
            
            # Right stick = look (IJKL)
            # gp_look returns (yaw, pitch) with proper signs
            gp_look = gamepad.get_look()
            gamepad_look_speed = dt * 2.5
            camera.rotate_keyboard(gp_look[0] * gamepad_look_speed, gp_look[1] * gamepad_look_speed)
            
            # L3 = go down, R3 = go up
            up += gamepad.get_vertical() * dt * speed
            
            # L2/R2 behavior depends on mode
            l2_val, r2_val = gamepad.get_triggers()
            
            if camera.auto_fly_mode > 0:
                # TOUR MODE: L2/R2 adjust flight height
                if l2_val > 0.3:  # L2 = lower height
                    camera.auto_fly_height -= l2_val * 0.5
                    camera.auto_fly_height = max(15, camera.auto_fly_height)
                if r2_val > 0.3:  # R2 = raise height
                    camera.auto_fly_height += r2_val * 0.5
                    camera.auto_fly_height = min(100, camera.auto_fly_height)
            else:
                # NORMAL MODE: L2/R2 for speed changes (with cooldown)
                if gamepad_speed_cooldown <= 0:
                    if l2_val > 0.7:  # L2 = decrease speed
                        new_level = max(0, camera.speed_level - 1)
                        if new_level != camera.speed_level:
                            camera.set_speed(new_level)
                            print(f"  Speed: {SPEED_LEVELS[new_level]:.2f}x")
                            gamepad_speed_cooldown = 20  # ~0.33 seconds
                    elif r2_val > 0.7:  # R2 = increase speed
                        new_level = min(len(SPEED_LEVELS) - 1, camera.speed_level + 1)
                        if new_level != camera.speed_level:
                            camera.set_speed(new_level)
                            print(f"  Speed: {SPEED_LEVELS[new_level]:.2f}x")
                            gamepad_speed_cooldown = 20
            
            # A button = jump (in walk mode)
            if gamepad.get_button(GamepadConfig.A):
                camera.jump()
            
            # B button = use current tool
            if gamepad.get_button(GamepadConfig.B) and gamepad_speed_cooldown <= 0:
                if camera.current_tool == ToolType.SCAN:
                    log_dna_at_cursor(camera, flora_manager, animal_manager)
                elif camera.current_tool == ToolType.MINE:
                    result = modify_terrain_at_cursor(camera, chunk_manager, "MINE")
                    if result and result in chunk_renderer.display_lists:
                        del chunk_renderer.display_lists[result]
                elif camera.current_tool == ToolType.FILL:
                    result = modify_terrain_at_cursor(camera, chunk_manager, "FILL")
                    if result and result in chunk_renderer.display_lists:
                        del chunk_renderer.display_lists[result]
                gamepad_speed_cooldown = 20  # Faster for terrain tools
            
            # L1 = previous tool, R1 = next tool
            if gamepad.get_button(GamepadConfig.L1) and gamepad_speed_cooldown <= 0:
                camera.prev_tool()
                gamepad_speed_cooldown = 15
            if gamepad.get_button(GamepadConfig.R1) and gamepad_speed_cooldown <= 0:
                camera.next_tool()
                gamepad_speed_cooldown = 15
            
            # Y button = screenshot
            if gamepad.get_button(GamepadConfig.Y) and gamepad_speed_cooldown <= 0:
                take_screenshot()
                gamepad_speed_cooldown = 30
            
            # START button = toggle menu
            if gamepad.get_button(GamepadConfig.START) and gamepad_speed_cooldown <= 0:
                camera.toggle_menu()
                gamepad_speed_cooldown = 20
            
            # SELECT button = warp to new universe
            if gamepad.get_button(GamepadConfig.SELECT) and gamepad_speed_cooldown <= 0:
                warp_to_new_universe(camera, chunk_manager, chunk_renderer, 
                                    flora_manager, animal_manager,
                                    chunk_dna_manager, climate_manager,
                                    structure_manager, config)
                gamepad_speed_cooldown = 60  # Longer cooldown for warp
            
            # Menu navigation with D-pad or left stick when menu is open
            if camera.menu_open and gamepad_speed_cooldown <= 0:
                # D-pad left/right = switch tabs
                if gamepad.get_dpad(GamepadConfig.DPAD_LEFT):
                    camera.menu_switch_tab(-1)
                    gamepad_speed_cooldown = 15
                elif gamepad.get_dpad(GamepadConfig.DPAD_RIGHT):
                    camera.menu_switch_tab(1)
                    gamepad_speed_cooldown = 15
                # D-pad up/down = navigate items
                elif gamepad.get_dpad(GamepadConfig.DPAD_UP):
                    camera.menu_navigate(-1)
                    gamepad_speed_cooldown = 10
                elif gamepad.get_dpad(GamepadConfig.DPAD_DOWN):
                    camera.menu_navigate(1)
                    gamepad_speed_cooldown = 10
                # Left stick also works for navigation
                gp_move = gamepad.get_movement()
                if abs(gp_move[0]) > 0.5:  # Forward/back = up/down
                    camera.menu_navigate(-1 if gp_move[0] > 0 else 1)
                    gamepad_speed_cooldown = 10
                if abs(gp_move[1]) > 0.5:  # Left/right = switch tabs
                    camera.menu_switch_tab(1 if gp_move[1] > 0 else -1)
                    gamepad_speed_cooldown = 15
        
        # Keyboard space = jump (in walk mode)
        if keys[pygame.K_SPACE] and not camera.flying:
            camera.jump()
        
        # Auto-fly mode - overrides manual movement
        if camera.auto_fly_mode > 0:
            auto_forward, auto_right, auto_up = camera.update_auto_fly(dt, chunk_manager)
            forward += auto_forward
            right += auto_right
            up += auto_up
        
        camera.move(forward, right, up, chunk_manager, structure_manager)
        
        # Maintain height during auto-fly
        camera.maintain_auto_fly_height(chunk_manager)
        
        # Update chunks (stream in new ones as we move)
        current_chunk = camera.get_chunk_pos()
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
        
        # Set sky color and fog based on time of day (now with DNA tint)
        sky_color = sky.get_sky_color()
        glClearColor(*sky_color, 1.0)
        sky.apply_fog()
        sky.apply_lighting()
        
        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        camera.apply()
        
        # Render sky (sun/moon/stars)
        cam_pos = camera.get_pos()
        sky.render(cam_pos[0], cam_pos[1], cam_pos[2])
        
        # Render clouds
        weather_renderer.render_clouds(cam_pos[0], cam_pos[1], cam_pos[2],
                                       climate_manager.current_weather.cloud_cover,
                                       climate_manager.time)
        
        chunk_renderer.render()
        
        # Render flora with LOD (cam_pos already set above)
        for (cx, cz), (display_list, lod) in chunk_renderer.display_lists.items():
            chunk = chunk_manager.get_chunk(cx, cz)
            flora_manager.render_chunk_flora(
                cx, cz, cam_pos[0], cam_pos[2],
                chunk.heightmap, chunk.world_x, chunk.world_z, TILE_SCALE, HEIGHT_SCALE
            )
            # Spawn animals for this chunk if not already done
            animal_manager.spawn_animals_for_chunk(
                cx, cz, chunk.heightmap, chunk.world_x, chunk.world_z, TILE_SCALE, HEIGHT_SCALE
            )
            # Maybe spawn buildings
            structure_manager.spawn_random_buildings(
                cx, cz, chunk.world_x, chunk.world_z, chunk.heightmap, HEIGHT_SCALE, TILE_SCALE
            )
        
        # Update animals every few frames for performance
        if frame_count % 3 == 0:  # Update AI every 3rd frame
            animal_manager.update(dt, cam_pos, None)  # dt is frame count, normalized in update
        
        # Render animals
        animal_manager.render(cam_pos[0], cam_pos[1], cam_pos[2])
        
        # Render structures (buildings)
        structure_manager.render(cam_pos[0], cam_pos[1], cam_pos[2])
        
        # Cleanup distant chunks occasionally
        if frame_count % 60 == 0:
            animal_manager.cleanup_distant_chunks(current_chunk[0], current_chunk[1])
            chunk_dna_manager.cleanup_distant(current_chunk[0], current_chunk[1])
            climate_manager.cleanup_distant(current_chunk[0], current_chunk[1])
            structure_manager.cleanup_distant(current_chunk[0], current_chunk[1])
        
        # Render water plane at water level, centered on camera
        render_water(camera.x, camera.z, water_level)
        
        # Render weather effects (rain/snow particles, lightning)
        weather_renderer.render(cam_pos[0], cam_pos[1], cam_pos[2],
                               climate_manager.current_weather,
                               climate_manager.lightning_flash)
        
        draw_crosshair(display)
        draw_hud(display, camera, sky, climate_manager, hud_font)
        draw_menu(display, camera, hud_font)  # Draw menu overlay if open
        minimap.draw(display, camera.x, camera.z)
        
        pygame.display.flip()
    
    # Cleanup
    chunk_worker.stop()
    chunk_renderer.stop()
    pygame.quit()


if __name__ == '__main__':
    run_explorer()

