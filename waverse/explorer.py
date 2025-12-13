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
PLAYER_HEIGHT = 3.5  # Eye height above ground (higher to avoid seeing through terrain)
WALK_SMOOTH_SPEED = 0.4  # Faster terrain following

# Speed levels (1-5 keys)
SPEED_LEVELS = [0.3, 0.6, 1.0, 2.0, 4.0]  # Slow to fast

# LOD settings - aggressive LOD for huge view distance
LOD_FULL_DISTANCE = 5      # Full detail within this range
LOD_HALF_DISTANCE = 10     # Half detail within this range  
LOD_QUARTER_DISTANCE = 15  # Quarter detail within this range
# Beyond that = 1/8th detail


SAVE_FILE = os.path.expanduser("~/.waverse.json")


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


class Camera:
    """Camera with walking and flying modes."""
    
    def __init__(self):
        self.x = 0
        self.y = 40
        self.z = 0
        self.yaw = 0
        self.pitch = -20
        self.flying = True
        self.target_y = 40
        self.speed_level = 2  # Default speed (1.0x)
    
    def rotate(self, dx, dy):
        self.yaw += dx * MOUSE_SENSITIVITY
        self.pitch -= dy * MOUSE_SENSITIVITY
        self.pitch = max(-89, min(89, self.pitch))
    
    def rotate_keyboard(self, dyaw, dpitch):
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
            # Walking mode: follow terrain/floor smoothly
            self.target_y = effective_ground
            diff = self.target_y - self.y
            
            # Fast catch-up if far from terrain, smooth otherwise
            if abs(diff) > 2:
                self.y += diff * 0.5  # Fast snap
            else:
                self.y += diff * WALK_SMOOTH_SPEED  # Smooth follow
            
            # Jump/fly when pressing up
            if up > 0:
                self.flying = True
                self.y += speed
    
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


def draw_hud(display: tuple, camera: Camera, sky: SkySystem = None, climate: ClimateManager = None):
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
    
    glBegin(GL_QUADS)  # Dummy begin to match the end below
    glEnd()
    
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
    print("    Speed: 1=Slow 2 3=Normal 4 5=Fast | S=Save | ESC=Exit")
    if gamepad.is_connected():
        print(f"  GAMEPAD ({gamepad.name}):")
        print("    Left Stick=Move | Right Stick=Look")
        print("    L3=Down | R3=Up | L2/R2=Change Speed")
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
                # Speed controls: 1-5 keys
                elif event.key == pygame.K_1:
                    camera.set_speed(0)
                    print("  Speed: 1 (Slow)")
                elif event.key == pygame.K_2:
                    camera.set_speed(1)
                    print("  Speed: 2")
                elif event.key == pygame.K_3:
                    camera.set_speed(2)
                    print("  Speed: 3 (Normal)")
                elif event.key == pygame.K_4:
                    camera.set_speed(3)
                    print("  Speed: 4")
                elif event.key == pygame.K_5:
                    camera.set_speed(4)
                    print("  Speed: 5 (Fast)")
                elif event.key == pygame.K_s:
                    save_position(camera, config.seed)
                elif event.key == pygame.K_g:
                    gamepad.debug_mode = not gamepad.debug_mode
                    print(f"  Gamepad debug: {'ON' if gamepad.debug_mode else 'OFF'}")
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
            
            # L2/R2 for speed changes (with cooldown to prevent rapid changes)
            # Use normalized trigger values (0-1 range)
            if gamepad_speed_cooldown <= 0:
                l2_val, r2_val = gamepad.get_triggers()
                
                if l2_val > 0.7:  # L2 = decrease speed
                    new_level = max(0, camera.speed_level - 1)
                    if new_level != camera.speed_level:
                        camera.set_speed(new_level)
                        print(f"  Speed: {new_level + 1}")
                        gamepad_speed_cooldown = 20  # ~0.33 seconds
                elif r2_val > 0.7:  # R2 = increase speed
                    new_level = min(4, camera.speed_level + 1)
                    if new_level != camera.speed_level:
                        camera.set_speed(new_level)
                        print(f"  Speed: {new_level + 1}")
                        gamepad_speed_cooldown = 20
        
        camera.move(forward, right, up, chunk_manager, structure_manager)
        
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
        draw_hud(display, camera, sky, climate_manager)
        minimap.draw(display, camera.x, camera.z)
        
        pygame.display.flip()
    
    # Cleanup
    chunk_worker.stop()
    chunk_renderer.stop()
    pygame.quit()


if __name__ == '__main__':
    run_explorer()

