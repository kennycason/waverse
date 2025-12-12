"""
OpenGL Renderer - Renders the Waverse world.

Based on the terrain explorer from andrew_maps, adapted for
the new wave-based DNA terrain system.
"""

import pygame
from pygame.locals import *
import numpy as np
import math
import time
from typing import Dict, Tuple, Optional, List, Any, TYPE_CHECKING

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: PyOpenGL not found. Renderer disabled.")

from .mesh import MeshData, generate_chunk_mesh, generate_water_mesh, height_to_color
from .chunk import Chunk, ChunkManager

if TYPE_CHECKING:
    from .dna import WaveDNA


# Configuration
MOVE_SPEED = 1.2
MOUSE_SENSITIVITY = 0.06
PLAYER_HEIGHT = 3.0
WALK_SMOOTH_SPEED = 0.15
RENDER_DISTANCE = 3  # Chunks to render in each direction


class Camera:
    """First-person camera with fly/walk modes."""
    
    def __init__(self, x: float = 0, y: float = 40, z: float = 0):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = 0.0
        self.pitch = -20.0
        self.flying = True
        self.target_y = y
    
    def rotate(self, dx: float, dy: float):
        """Rotate camera from mouse movement."""
        self.yaw += dx * MOUSE_SENSITIVITY
        self.pitch -= dy * MOUSE_SENSITIVITY
        self.pitch = max(-89, min(89, self.pitch))
    
    def rotate_keyboard(self, dyaw: float, dpitch: float):
        """Rotate camera from keyboard."""
        self.yaw += dyaw
        self.pitch += dpitch
        self.pitch = max(-89, min(89, self.pitch))
    
    def get_forward_vector(self) -> Tuple[float, float, float]:
        """Get the forward direction vector."""
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        fx = -math.sin(yaw_rad) * math.cos(pitch_rad)
        fy = math.sin(pitch_rad)
        fz = -math.cos(yaw_rad) * math.cos(pitch_rad)
        
        return (fx, fy, fz)
    
    def move(self, forward: float, right: float, up: float,
             chunk_manager: Optional[ChunkManager] = None):
        """Move the camera."""
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)
        
        if self.flying:
            # Full 3D movement in fly mode
            forward_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
            forward_y = math.sin(pitch_rad)
            forward_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        else:
            # Horizontal movement only in walk mode
            forward_x = -math.sin(yaw_rad)
            forward_y = 0
            forward_z = -math.cos(yaw_rad)
        
        right_x = math.cos(yaw_rad)
        right_z = -math.sin(yaw_rad)
        
        # Apply horizontal movement
        self.x += (forward * forward_x + right * right_x) * MOVE_SPEED
        self.z += (forward * forward_z + right * right_z) * MOVE_SPEED
        
        # Get terrain height at new position
        terrain_h = 0
        if chunk_manager:
            terrain_h = chunk_manager.get_height_at(self.x, self.z)
        min_height = terrain_h + PLAYER_HEIGHT
        
        if self.flying:
            # In fly mode, apply vertical movement from forward pitch
            self.y += forward * forward_y * MOVE_SPEED
            
            # H raises us up, F lowers us
            self.y += up * MOVE_SPEED
            
            # If pressing F (down) and we hit ground, switch to walk mode
            if up < 0 and self.y <= min_height:
                self.y = min_height
                self.flying = False
                self.target_y = min_height
            
            # Enforce minimum height even in fly mode
            if self.y < min_height:
                self.y = min_height
        else:
            # Walk mode - smoothly follow terrain
            self.target_y = min_height
            self.y += (self.target_y - self.y) * WALK_SMOOTH_SPEED
            
            # H key lifts off into fly mode
            if up > 0:
                self.flying = True
                self.y += up * MOVE_SPEED
    
    def apply(self):
        """Apply camera transformation to OpenGL."""
        glRotatef(-self.pitch, 1, 0, 0)
        glRotatef(-self.yaw, 0, 1, 0)
        glTranslatef(-self.x, -self.y, -self.z)
    
    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class ChunkRenderer:
    """
    Renders chunks using OpenGL display lists for performance.
    """
    
    def __init__(self, chunk_manager: ChunkManager, water_level: float = 0.0):
        self.chunk_manager = chunk_manager
        self.water_level = water_level
        
        # Display list cache: chunk coords -> display list ID
        self.display_lists: Dict[Tuple[int, int], int] = {}
        
        # Track which chunks need rebuilding
        self.dirty_chunks: set = set()
        
        # Water plane display list
        self.water_list: Optional[int] = None
    
    def get_or_create_display_list(self, cx: int, cz: int) -> Optional[int]:
        """Get or create a display list for a chunk."""
        key = (cx, cz)
        
        # Check if we have a valid cached list
        if key in self.display_lists and key not in self.dirty_chunks:
            return self.display_lists[key]
        
        # Get the chunk
        chunk = self.chunk_manager.get_chunk(cx, cz)
        if not chunk.generated:
            return None
        
        # Generate mesh
        mesh = generate_chunk_mesh(chunk, self.water_level, use_merging=True)
        if mesh.vertex_count == 0:
            return None
        
        # Delete old display list if exists
        if key in self.display_lists:
            glDeleteLists(self.display_lists[key], 1)
        
        # Create new display list
        list_id = glGenLists(1)
        glNewList(list_id, GL_COMPILE)
        
        self._render_mesh(mesh)
        
        glEndList()
        
        self.display_lists[key] = list_id
        self.dirty_chunks.discard(key)
        
        return list_id
    
    def _render_mesh(self, mesh: MeshData):
        """Render a mesh immediately (used inside display list)."""
        if mesh.vertices is None or len(mesh.vertices) == 0:
            return
        
        if mesh.primitive == 'quads':
            glBegin(GL_QUADS)
        elif mesh.primitive == 'triangles':
            glBegin(GL_TRIANGLES)
        else:
            glBegin(GL_LINES)
        
        for i in range(len(mesh.vertices)):
            if mesh.normals is not None:
                glNormal3fv(mesh.normals[i])
            if mesh.colors is not None:
                if len(mesh.colors[i]) == 4:
                    glColor4fv(mesh.colors[i])
                else:
                    glColor3fv(mesh.colors[i])
            glVertex3fv(mesh.vertices[i])
        
        glEnd()
    
    def render_chunks(self, camera: Camera, render_distance: int = RENDER_DISTANCE):
        """Render all chunks around the camera."""
        cx, cz = self.chunk_manager.world_to_chunk(camera.x, camera.z)
        
        rendered = 0
        for dz in range(-render_distance, render_distance + 1):
            for dx in range(-render_distance, render_distance + 1):
                list_id = self.get_or_create_display_list(cx + dx, cz + dz)
                if list_id:
                    glCallList(list_id)
                    rendered += 1
        
        return rendered
    
    def render_water(self, camera: Camera, size: float = 2000):
        """Render the water plane."""
        if self.water_list is None:
            self.water_list = glGenLists(1)
            glNewList(self.water_list, GL_COMPILE)
            
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glNormal3f(0, 1, 0)
            glColor4f(0.15, 0.35, 0.55, 0.7)
            
            # Counter-clockwise when viewed from above
            glBegin(GL_QUADS)
            glVertex3f(-size, self.water_level, -size)  # NW
            glVertex3f(-size, self.water_level, size)   # SW
            glVertex3f(size, self.water_level, size)    # SE
            glVertex3f(size, self.water_level, -size)   # NE
            glEnd()
            
            glDisable(GL_BLEND)
            glEndList()
        
        glCallList(self.water_list)
    
    def mark_chunk_dirty(self, cx: int, cz: int):
        """Mark a chunk for rebuilding."""
        self.dirty_chunks.add((cx, cz))
    
    def cleanup(self):
        """Delete all display lists."""
        for list_id in self.display_lists.values():
            glDeleteLists(list_id, 1)
        self.display_lists.clear()
        
        if self.water_list:
            glDeleteLists(self.water_list, 1)
            self.water_list = None


class Minimap:
    """2D overview minimap."""
    
    def __init__(self, chunk_manager: ChunkManager, size: int = 180):
        self.chunk_manager = chunk_manager
        self.size = size
        self.texture_id: Optional[int] = None
        self.last_update_pos = (0, 0)
        self.update_threshold = 50  # Pixels before update
    
    def update_texture(self, center_x: float, center_z: float, radius: float = 200):
        """Update minimap texture around position."""
        # Sample terrain in a grid
        samples = 128
        step = radius * 2 / samples
        
        heights = np.zeros((samples, samples), dtype=np.float32)
        
        for i in range(samples):
            for j in range(samples):
                wx = center_x - radius + i * step
                wz = center_z - radius + j * step
                heights[j, i] = self.chunk_manager.get_height_at(wx, wz)
        
        # Convert to colors
        pixels = []
        h_min, h_max = heights.min(), heights.max()
        water_level = self.chunk_manager.dna.water_level
        
        for j in range(samples):
            for i in range(samples):
                r, g, b = height_to_color(heights[j, i], water_level)
                pixels.extend([int(r * 255), int(g * 255), int(b * 255)])
        
        # Create texture
        from OpenGL.GL import GLubyte
        pixel_data = (GLubyte * len(pixels))(*pixels)
        
        if self.texture_id:
            glDeleteTextures([self.texture_id])
        
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, samples, samples, 0, 
                     GL_RGB, GL_UNSIGNED_BYTE, pixel_data)
        
        self.last_update_pos = (center_x, center_z)
    
    def draw(self, display: Tuple[int, int], camera: Camera):
        """Draw the minimap overlay."""
        if self.texture_id is None:
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
        
        # Map texture
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
        glVertex2f(px - 4, py - 4)
        glVertex2f(px + 4, py - 4)
        glVertex2f(px + 4, py + 4)
        glVertex2f(px - 4, py + 4)
        glEnd()
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


def draw_crosshair(display: Tuple[int, int]):
    """Draw a simple crosshair."""
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


def draw_hud(display: Tuple[int, int], camera: Camera, 
             stats: Dict[str, Any], t: float):
    """Draw HUD overlay with stats."""
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
    
    # Info box (top right)
    box_w, box_h = 180, 90
    box_x = display[0] - box_w - 10
    box_y = 10
    
    glColor4f(0, 0, 0, 0.6)
    glBegin(GL_QUADS)
    glVertex2f(box_x, box_y)
    glVertex2f(box_x + box_w, box_y)
    glVertex2f(box_x + box_w, box_y + box_h)
    glVertex2f(box_x, box_y + box_h)
    glEnd()
    
    # Mode indicator bar
    mode_y = box_y + 10
    if camera.flying:
        glColor4f(0.3, 0.6, 1.0, 0.9)  # Blue for flying
    else:
        glColor4f(0.3, 0.9, 0.3, 0.9)  # Green for walking
    
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
    glVertex2f(box_x + box_w - 10, bar_y + 10)
    glVertex2f(box_x + 10, bar_y + 10)
    glEnd()
    
    # Height fill
    height_pct = min(1, max(0, camera.y / 100))
    glColor4f(1, 0.8, 0.2, 0.9)
    glBegin(GL_QUADS)
    glVertex2f(box_x + 10, bar_y)
    glVertex2f(box_x + 10 + height_pct * (box_w - 20), bar_y)
    glVertex2f(box_x + 10 + height_pct * (box_w - 20), bar_y + 10)
    glVertex2f(box_x + 10, bar_y + 10)
    glEnd()
    
    # Chunk count bar
    chunk_y = bar_y + 20
    chunk_pct = min(1, stats.get('loaded_chunks', 0) / stats.get('cache_size', 64))
    glColor4f(0.4, 0.4, 0.6, 0.8)
    glBegin(GL_QUADS)
    glVertex2f(box_x + 10, chunk_y)
    glVertex2f(box_x + 10 + chunk_pct * (box_w - 20), chunk_y)
    glVertex2f(box_x + 10 + chunk_pct * (box_w - 20), chunk_y + 8)
    glVertex2f(box_x + 10, chunk_y + 8)
    glEnd()
    
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


def setup_opengl(water_level: float = 0.0):
    """Initialize OpenGL settings."""
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_NORMALIZE)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    # Disable backface culling for now (terrain is single-sided but we want to see it from both sides during dev)
    glDisable(GL_CULL_FACE)
    
    # Light settings
    glLightfv(GL_LIGHT0, GL_POSITION, (0.4, 1.0, 0.3, 0.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 0.95, 0.9, 1.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.4, 0.4, 0.45, 1.0))
    
    # Sky color
    glClearColor(0.55, 0.75, 0.92, 1.0)
    
    # Fog
    glEnable(GL_FOG)
    glFogfv(GL_FOG_COLOR, (0.55, 0.72, 0.88, 1.0))
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogf(GL_FOG_START, 80)
    glFogf(GL_FOG_END, 350)


class WaverseRenderer:
    """
    Main renderer class that manages the whole rendering pipeline.
    """
    
    def __init__(self, dna: "WaveDNA", display_size: Tuple[int, int] = (1400, 800)):
        self.dna = dna
        self.display_size = display_size
        
        # Initialize pygame and OpenGL
        pygame.init()
        pygame.display.set_mode(display_size, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Waverse - Wave-Based World Explorer")
        
        pygame.mouse.set_visible(True)
        pygame.event.set_grab(False)
        
        setup_opengl(dna.water_level)
        
        glMatrixMode(GL_PROJECTION)
        gluPerspective(65, display_size[0] / display_size[1], 0.5, 500)
        glMatrixMode(GL_MODELVIEW)
        
        # Create chunk manager
        self.chunk_manager = ChunkManager(dna, cache_size=64, generate_caves=bool(dna.cave_layers))
        
        # Create chunk renderer
        self.chunk_renderer = ChunkRenderer(self.chunk_manager, dna.water_level)
        
        # Create camera
        self.camera = Camera(x=0, y=50, z=0)
        
        # Create minimap
        self.minimap = Minimap(self.chunk_manager)
        
        # State
        self.running = True
        self.mouse_look = False
        self.clock = pygame.time.Clock()
        self.start_time = time.time()
        
        # Preload chunks around spawn
        self.chunk_manager.preload_around(0, 0, RENDER_DISTANCE)
        
        # Set initial camera height
        spawn_height = self.chunk_manager.get_height_at(0, 0)
        self.camera.y = spawn_height + 30
        
        # Initial minimap
        self.minimap.update_texture(0, 0)
    
    def handle_events(self):
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # Right click
                    self.mouse_look = True
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
                elif event.button == 1:  # Left click (dig)
                    self._handle_dig()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    self.mouse_look = False
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)
            
            elif event.type == pygame.MOUSEMOTION and self.mouse_look:
                self.camera.rotate(*event.rel)
    
    def _handle_dig(self):
        """Handle digging at camera position."""
        # For now, dig at current position
        self.chunk_manager.dig_at(self.camera.x, self.camera.z, 1.0)
        
        # Mark chunk for rebuild
        cx, cz = self.chunk_manager.world_to_chunk(self.camera.x, self.camera.z)
        self.chunk_renderer.mark_chunk_dirty(cx, cz)
    
    def handle_input(self, dt: float):
        """Process keyboard input."""
        keys = pygame.key.get_pressed()
        
        forward = right = up = 0
        speed = 2.5 if keys[pygame.K_LALT] else 1.0
        
        if keys[pygame.K_w] or keys[pygame.K_UP]: forward += dt * speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: forward -= dt * speed
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: right -= dt * speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: right += dt * speed
        if keys[pygame.K_h] or keys[pygame.K_SPACE]: up += dt * speed
        if keys[pygame.K_f] or keys[pygame.K_LSHIFT]: up -= dt * speed
        
        # Keyboard look (same speed as reference)
        look_speed = dt * 0.8
        if keys[pygame.K_i]: self.camera.rotate_keyboard(0, look_speed * 1.2)
        if keys[pygame.K_k]: self.camera.rotate_keyboard(0, -look_speed * 1.2)
        if keys[pygame.K_j]: self.camera.rotate_keyboard(look_speed * 1.5, 0)
        if keys[pygame.K_l]: self.camera.rotate_keyboard(-look_speed * 1.5, 0)
        
        self.camera.move(forward, right, up, self.chunk_manager)
    
    def update(self, dt: float):
        """Update game state."""
        # Preload chunks around camera
        self.chunk_manager.preload_around(self.camera.x, self.camera.z, RENDER_DISTANCE)
        
        # Update minimap occasionally
        last_x, last_z = self.minimap.last_update_pos
        if abs(self.camera.x - last_x) > 50 or abs(self.camera.z - last_z) > 50:
            self.minimap.update_texture(self.camera.x, self.camera.z)
    
    def render(self):
        """Render the scene."""
        t = time.time() - self.start_time
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        self.camera.apply()
        
        # Render terrain chunks
        self.chunk_renderer.render_chunks(self.camera, RENDER_DISTANCE)
        
        # Render water
        self.chunk_renderer.render_water(self.camera)
        
        # UI overlays
        draw_crosshair(self.display_size)
        draw_hud(self.display_size, self.camera, 
                 self.chunk_manager.get_stats(), t)
        self.minimap.draw(self.display_size, self.camera)
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop."""
        print("\n" + "=" * 60)
        print("  WAVERSE - Wave-Based World Explorer")
        print("=" * 60)
        print(f"  World: {self.dna.name}")
        print(f"  Seed: {self.dna.seed}")
        print("=" * 60)
        print("  CONTROLS:")
        print("  WASD/Arrows: Move | H/Space: Up | F/Shift: Down")
        print("  Right-Click + Mouse: Look around")
        print("  IJKL: Keyboard look | Left-Click: Dig")
        print("  Alt: Speed boost | ESC: Exit")
        print("=" * 60 + "\n")
        
        while self.running:
            dt = self.clock.tick(60) / 16.67
            
            self.handle_events()
            self.handle_input(dt)
            self.update(dt)
            self.render()
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.chunk_renderer.cleanup()
        pygame.quit()

