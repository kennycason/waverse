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
    
    def move(self, forward, right, up, chunk_manager: ChunkManager):
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
        
        self.x += (forward * forward_x + right * right_x) * speed
        self.z += (forward * forward_z + right * right_z) * speed
        
        terrain_h = self.get_terrain_height(chunk_manager)
        ground_height = terrain_h + PLAYER_HEIGHT
        
        if self.flying:
            self.y += forward * forward_y * speed
            self.y += up * speed
            
            # Land when pressing down and at ground level
            if up < 0 and self.y <= ground_height + 0.5:
                self.y = ground_height
                self.flying = False
                self.target_y = ground_height
            
            # Don't go below ground
            if self.y < ground_height:
                self.y = ground_height
        else:
            # Walking mode: follow terrain smoothly
            self.target_y = ground_height
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


def create_water_plane() -> int:
    """Create water plane display list."""
    water_list = glGenLists(1)
    glNewList(water_list, GL_COMPILE)
    
    size = 2000
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glNormal3f(0, 1, 0)
    glColor4f(0.1, 0.35, 0.6, 0.7)
    
    glBegin(GL_QUADS)
    glVertex3f(-size, 0, -size)
    glVertex3f(size, 0, -size)
    glVertex3f(size, 0, size)
    glVertex3f(-size, 0, size)
    glEnd()
    
    glDisable(GL_BLEND)
    glEndList()
    
    return water_list


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


def draw_hud(display: tuple, camera: Camera, sky: SkySystem = None):
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
    box_w, box_h = 180, 90 if sky else 70
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
    water_list = create_water_plane()
    minimap = Minimap(chunk_manager)
    
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
    print("  MOVEMENT: WASD/Arrows | H/Space=Up F/Shift=Down")
    print("  CAMERA: IJKL or Right-Click+Mouse")
    print("  SPEED: 1=Slow 2 3=Normal 4 5=Fast")
    print("  SAVE: S (saves position to ~/.waverse.json)")
    print("  EXIT: ESC")
    print("=" * 60 + "\n")
    
    clock = pygame.time.Clock()
    running = True
    mouse_look = False
    last_chunk = None
    frame_count = 0
    
    while running:
        frame_count += 1
        dt = clock.tick(60) / 16.67
        
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
        
        camera.move(forward, right, up, chunk_manager)
        
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
        
        # Update animals every few frames for performance
        if frame_count % 3 == 0:  # Update AI every 3rd frame
            animal_manager.update(dt / 60.0, cam_pos, None)  # Skip ground height for perf
        
        # Render animals
        animal_manager.render(cam_pos[0], cam_pos[1], cam_pos[2])
        
        # Cleanup distant chunks occasionally
        if frame_count % 60 == 0:
            animal_manager.cleanup_distant_chunks(current_chunk[0], current_chunk[1])
            chunk_dna_manager.cleanup_distant(current_chunk[0], current_chunk[1])
        
        glCallList(water_list)
        
        draw_crosshair(display)
        draw_hud(display, camera, sky)
        minimap.draw(display, camera.x, camera.z)
        
        pygame.display.flip()
    
    # Cleanup
    chunk_worker.stop()
    chunk_renderer.stop()
    pygame.quit()


if __name__ == '__main__':
    run_explorer()

