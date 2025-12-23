"""
Modern Terrain Renderer - VBO-based terrain chunk rendering.

Converts existing heightmap data to GPU-optimized VBOs for fast rendering.
"""

import numpy as np
import moderngl
try:
    from pyglm import glm
except ImportError:
    import glm
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


# =============================================================================
# SHADERS
# =============================================================================

TERRAIN_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

out vec3 v_normal;
out vec3 v_color;
out vec3 v_world_pos;
out float v_height;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec3 u_camera_pos;

void main() {
    v_world_pos = in_position;
    v_normal = in_normal;
    v_color = in_color;
    v_height = in_position.y;
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
}
"""

TERRAIN_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_normal;
in vec3 v_color;
in vec3 v_world_pos;
in float v_height;

out vec4 fragColor;

uniform vec3 u_light_dir;
uniform vec3 u_ambient;
uniform float u_fog_start;
uniform float u_fog_end;
uniform vec3 u_fog_color;
uniform vec3 u_camera_pos;
uniform float u_water_level;

void main() {
    // Normalize vectors
    vec3 normal = normalize(v_normal);
    vec3 light = normalize(u_light_dir);
    
    // Diffuse lighting with slight wrap for softer shadows
    float ndotl = dot(normal, light);
    float diff = max(ndotl * 0.5 + 0.5, 0.0);  // Wrap lighting
    
    // Base color
    vec3 base_color = v_color;
    
    // Underwater tint
    if (v_height < u_water_level) {
        float depth = (u_water_level - v_height) / 10.0;
        base_color = mix(base_color, vec3(0.2, 0.3, 0.5), clamp(depth, 0.0, 0.7));
    }
    
    // Apply lighting
    vec3 lit_color = base_color * (u_ambient + diff * 0.6);
    
    // Height-based atmospheric scattering (higher = bluer)
    float height_factor = clamp(v_height / 100.0, 0.0, 1.0);
    lit_color = mix(lit_color, lit_color * vec3(0.9, 0.95, 1.1), height_factor * 0.3);
    
    // Distance fog
    float dist = length(v_world_pos - u_camera_pos);
    float fog_factor = clamp((dist - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);
    fog_factor = fog_factor * fog_factor;  // Quadratic falloff for nicer look
    vec3 final_color = mix(lit_color, u_fog_color, fog_factor);
    
    fragColor = vec4(final_color, 1.0);
}
"""


# =============================================================================
# CHUNK MESH
# =============================================================================

@dataclass
class TerrainChunkMesh:
    """GPU mesh for a terrain chunk."""
    cx: int
    cz: int
    vbo: Optional[moderngl.Buffer] = None
    vao: Optional[moderngl.VertexArray] = None
    vertex_count: int = 0
    triangle_count: int = 0
    center_x: float = 0.0
    center_z: float = 0.0
    
    def release(self):
        if self.vbo:
            self.vbo.release()
            self.vbo = None
        if self.vao:
            self.vao.release()
            self.vao = None


# =============================================================================
# BIOME COLORS (from existing code)
# =============================================================================

BIOME_COLORS = {
    # Lowercase keys (legacy)
    'ocean': (0.15, 0.35, 0.55),
    'beach': (0.85, 0.82, 0.65),
    'desert': (0.90, 0.85, 0.60),
    'savanna': (0.75, 0.72, 0.45),
    'grassland': (0.35, 0.65, 0.30),
    'forest': (0.25, 0.55, 0.25),
    'rainforest': (0.20, 0.50, 0.20),
    'taiga': (0.30, 0.45, 0.35),
    'tundra': (0.70, 0.75, 0.72),
    'snow': (0.92, 0.95, 0.98),
    'mountain': (0.50, 0.48, 0.45),
    'volcanic': (0.25, 0.20, 0.20),
    'swamp': (0.35, 0.45, 0.30),
    'marsh': (0.40, 0.50, 0.35),
    # Capitalized keys (from BiomeDNA.get_biome_name)
    'Tundra': (0.70, 0.75, 0.72),
    'Frozen': (0.85, 0.90, 0.95),
    'Taiga': (0.30, 0.45, 0.35),
    'Cold': (0.55, 0.60, 0.58),
    'Rainforest': (0.15, 0.45, 0.18),
    'Temperate': (0.32, 0.58, 0.28),
    'Grassland': (0.40, 0.68, 0.32),
    'Tropical': (0.25, 0.55, 0.30),
    'Desert': (0.92, 0.85, 0.55),
    'Savanna': (0.78, 0.72, 0.42),
}


def get_terrain_color(height: float, biome: str = 'grassland', 
                      water_level: float = 0.0) -> Tuple[float, float, float]:
    """Get terrain color based on height and biome."""
    if height < water_level - 5:
        # Deep water
        return (0.12, 0.28, 0.45)
    elif height < water_level:
        # Shallow water / beach transition
        depth = (water_level - height) / 5.0
        beach = BIOME_COLORS.get('beach', (0.85, 0.82, 0.65))
        water = (0.15, 0.35, 0.55)
        return tuple(b * (1 - depth) + w * depth for b, w in zip(beach, water))
    elif height < water_level + 2:
        # Beach
        return BIOME_COLORS.get('beach', (0.85, 0.82, 0.65))
    elif height > 80:
        # Snow caps
        return BIOME_COLORS.get('snow', (0.92, 0.95, 0.98))
    elif height > 60:
        # Mountain
        t = (height - 60) / 20.0
        mountain = BIOME_COLORS.get('mountain', (0.50, 0.48, 0.45))
        snow = BIOME_COLORS.get('snow', (0.92, 0.95, 0.98))
        return tuple(m * (1 - t) + s * t for m, s in zip(mountain, snow))
    elif height > 40:
        # High altitude - blend to mountain
        t = (height - 40) / 20.0
        base = BIOME_COLORS.get(biome, (0.35, 0.65, 0.30))
        mountain = BIOME_COLORS.get('mountain', (0.50, 0.48, 0.45))
        return tuple(b * (1 - t) + m * t for b, m in zip(base, mountain))
    else:
        # Normal biome color with slight height variation
        base = BIOME_COLORS.get(biome, (0.35, 0.65, 0.30))
        # Darken in valleys, lighten on hills
        height_mod = 1.0 + (height - 20) * 0.005
        return tuple(min(1.0, c * height_mod) for c in base)


# =============================================================================
# TERRAIN RENDERER
# =============================================================================

class ModernTerrainRenderer:
    """
    Modern VBO-based terrain renderer.
    
    Converts heightmap chunks to GPU meshes for efficient rendering.
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Compile shader
        self.program = ctx.program(
            vertex_shader=TERRAIN_VERTEX_SHADER,
            fragment_shader=TERRAIN_FRAGMENT_SHADER
        )
        
        # Chunk meshes: (cx, cz) -> TerrainChunkMesh
        self.chunks: Dict[Tuple[int, int], TerrainChunkMesh] = {}
        
        # Rendering state
        self.projection = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
        self.camera_pos = glm.vec3(0, 50, 0)
        
        # Lighting
        self.light_dir = glm.vec3(0.4, 0.8, 0.3)
        self.ambient = glm.vec3(0.35, 0.38, 0.42)
        
        # Fog
        self.fog_color = glm.vec3(0.65, 0.75, 0.88)
        self.fog_start = 2000.0
        self.fog_end = 6000.0
        
        # Water
        self.water_level = 0.0
        
        # Stats
        self.frame_stats = {
            'chunks_rendered': 0,
            'triangles': 0,
            'draw_calls': 0,
        }
    
    def set_camera(self, x: float, y: float, z: float,
                   yaw: float, pitch: float, fov: float = 60.0,
                   aspect: float = 16/9, near: float = 0.5, far: float = 8000.0):
        """Update camera from waverse camera state."""
        self.camera_pos = glm.vec3(x, y, z)
        
        # Build view matrix
        yaw_rad = glm.radians(yaw)
        pitch_rad = glm.radians(pitch)
        
        # Direction vector
        dir_x = -np.sin(yaw_rad) * np.cos(pitch_rad)
        dir_y = np.sin(pitch_rad)
        dir_z = -np.cos(yaw_rad) * np.cos(pitch_rad)
        
        target = self.camera_pos + glm.vec3(dir_x, dir_y, dir_z)
        self.view = glm.lookAt(self.camera_pos, target, glm.vec3(0, 1, 0))
        
        # Projection
        self.projection = glm.perspective(glm.radians(fov), aspect, near, far)
    
    def create_chunk_mesh(self, cx: int, cz: int, 
                          heightmap: np.ndarray,
                          tile_scale: float,
                          height_scale: float,
                          chunk_world_x: float,
                          chunk_world_z: float,
                          biome: str = 'grassland') -> TerrainChunkMesh:
        """
        Create a VBO mesh from a heightmap chunk.
        OPTIMIZED: Uses numpy vectorization instead of Python loops.
        """
        key = (cx, cz)
        
        # Release old mesh
        if key in self.chunks:
            self.chunks[key].release()
        
        h, w = heightmap.shape
        hz, hx = h - 1, w - 1  # Number of cells
        
        # VECTORIZED: Create coordinate grids
        x_grid = np.arange(hx, dtype='f4')
        z_grid = np.arange(hz, dtype='f4')
        xx, zz = np.meshgrid(x_grid, z_grid)
        
        # World positions (flat arrays for vectorized operations)
        wx0 = (chunk_world_x + xx * tile_scale).flatten()
        wz0 = (chunk_world_z + zz * tile_scale).flatten()
        wx1 = wx0 + tile_scale
        wz1 = wz0 + tile_scale
        
        # Heights at all four corners (vectorized)
        h00 = (heightmap[:-1, :-1] * height_scale).flatten().astype('f4')
        h10 = (heightmap[:-1, 1:] * height_scale).flatten().astype('f4')
        h01 = (heightmap[1:, :-1] * height_scale).flatten().astype('f4')
        h11 = (heightmap[1:, 1:] * height_scale).flatten().astype('f4')
        
        num_cells = len(h00)
        num_verts = num_cells * 6
        
        # VECTORIZED: Build vertices for both triangles
        # Triangle 1: v0=(wx0,h00,wz0), v1=(wx1,h10,wz0), v2=(wx0,h01,wz1)
        # Triangle 2: v3=(wx1,h10,wz0), v4=(wx1,h11,wz1), v5=(wx0,h01,wz1)
        vertices = np.zeros((num_verts, 3), dtype='f4')
        
        # Indices for each triangle vertex
        vertices[0::6, 0] = wx0;  vertices[0::6, 1] = h00; vertices[0::6, 2] = wz0
        vertices[1::6, 0] = wx1;  vertices[1::6, 1] = h10; vertices[1::6, 2] = wz0
        vertices[2::6, 0] = wx0;  vertices[2::6, 1] = h01; vertices[2::6, 2] = wz1
        vertices[3::6, 0] = wx1;  vertices[3::6, 1] = h10; vertices[3::6, 2] = wz0
        vertices[4::6, 0] = wx1;  vertices[4::6, 1] = h11; vertices[4::6, 2] = wz1
        vertices[5::6, 0] = wx0;  vertices[5::6, 1] = h01; vertices[5::6, 2] = wz1
        
        # VECTORIZED: Normals (simplified - pointing roughly up with slight variation)
        # For speed, use approximate normals based on height differences
        normals = np.zeros((num_verts, 3), dtype='f4')
        
        # Triangle 1 normals (cross product of edges)
        dx1 = h10 - h00  # Height change in X
        dz1 = h01 - h00  # Height change in Z
        n1_x = -dx1 / tile_scale
        n1_z = -dz1 / tile_scale
        n1_y = np.ones_like(n1_x)
        n1_len = np.sqrt(n1_x**2 + n1_y**2 + n1_z**2) + 1e-8
        
        normals[0::6, 0] = n1_x / n1_len; normals[0::6, 1] = n1_y / n1_len; normals[0::6, 2] = n1_z / n1_len
        normals[1::6] = normals[0::6]
        normals[2::6] = normals[0::6]
        
        # Triangle 2 normals
        dx2 = h11 - h01
        dz2 = h11 - h10
        n2_x = -dx2 / tile_scale
        n2_z = -dz2 / tile_scale
        n2_y = np.ones_like(n2_x)
        n2_len = np.sqrt(n2_x**2 + n2_y**2 + n2_z**2) + 1e-8
        
        normals[3::6, 0] = n2_x / n2_len; normals[3::6, 1] = n2_y / n2_len; normals[3::6, 2] = n2_z / n2_len
        normals[4::6] = normals[3::6]
        normals[5::6] = normals[3::6]
        
        # FULLY VECTORIZED: Colors based on height (no Python loops)
        colors = np.zeros((num_verts, 3), dtype='f4')
        
        # Get base biome color
        base_color = np.array(BIOME_COLORS.get(biome.capitalize(), BIOME_COLORS.get('Grassland', (0.35, 0.55, 0.28))), dtype='f4')
        water_color = np.array([0.2, 0.25, 0.35], dtype='f4')
        shore_color = np.array([0.76, 0.7, 0.5], dtype='f4')
        snow_color = np.array([0.9, 0.9, 0.95], dtype='f4')
        
        # All 6 heights per cell stacked
        all_heights = np.stack([h00, h10, h01, h10, h11, h01], axis=1).flatten()
        n = len(all_heights)
        result = np.zeros((n, 3), dtype='f4')
        
        # Masks for each terrain type
        underwater = all_heights < self.water_level - 1
        shore = (all_heights >= self.water_level - 1) & (all_heights < self.water_level + 3)
        normal = (all_heights >= self.water_level + 3) & (all_heights < 80)
        high = all_heights >= 80
        
        # Apply colors by mask
        result[underwater] = water_color
        result[shore] = shore_color
        result[normal] = base_color
        
        # High altitude blend (vectorized)
        if np.any(high):
            t = np.clip((all_heights[high] - 80) / 40, 0, 1).reshape(-1, 1)
            result[high] = base_color * (1 - t) + snow_color * t
        
        # Reshape to per-vertex
        colors = result.reshape((num_cells, 6, 3)).reshape((num_verts, 3))
        
        # Interleave data
        data = np.zeros((num_verts, 9), dtype='f4')
        data[:, 0:3] = vertices
        data[:, 3:6] = normals
        data[:, 6:9] = colors
        
        # Create VBO
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')]
        )
        
        # Calculate chunk center for culling
        center_x = chunk_world_x + (w * tile_scale) / 2
        center_z = chunk_world_z + (h * tile_scale) / 2
        
        mesh = TerrainChunkMesh(
            cx=cx, cz=cz,
            vbo=vbo, vao=vao,
            vertex_count=num_verts,
            triangle_count=num_verts // 3,
            center_x=center_x,
            center_z=center_z
        )
        
        self.chunks[key] = mesh
        return mesh
    
    def remove_chunk(self, cx: int, cz: int):
        """Remove a chunk mesh."""
        key = (cx, cz)
        if key in self.chunks:
            self.chunks[key].release()
            del self.chunks[key]
    
    def render(self, visible_chunks: List[Tuple[int, int]] = None):
        """
        Render terrain chunks.
        
        Args:
            visible_chunks: List of chunk coords to render. None = render all.
        """
        self.frame_stats = {'chunks_rendered': 0, 'triangles': 0, 'draw_calls': 0}
        
        # Set uniforms - write glm matrices directly (column-major as OpenGL expects)
        self.program['u_projection'].write(self.projection)
        self.program['u_view'].write(self.view)
        self.program['u_camera_pos'].value = tuple(self.camera_pos)
        self.program['u_light_dir'].value = tuple(self.light_dir)
        self.program['u_ambient'].value = tuple(self.ambient)
        self.program['u_fog_start'].value = self.fog_start
        self.program['u_fog_end'].value = self.fog_end
        self.program['u_fog_color'].value = tuple(self.fog_color)
        self.program['u_water_level'].value = self.water_level
        
        # Render chunks
        keys = visible_chunks if visible_chunks else list(self.chunks.keys())
        
        for key in keys:
            if key not in self.chunks:
                continue
            
            mesh = self.chunks[key]
            if mesh.vao and mesh.vertex_count > 0:
                mesh.vao.render(moderngl.TRIANGLES)
                self.frame_stats['chunks_rendered'] += 1
                self.frame_stats['triangles'] += mesh.triangle_count
                self.frame_stats['draw_calls'] += 1
    
    def cleanup(self):
        """Release all GPU resources."""
        for mesh in self.chunks.values():
            mesh.release()
        self.chunks.clear()


# =============================================================================
# INTEGRATION TEST
# =============================================================================

def test_terrain_renderer():
    """Test with procedural terrain."""
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL
    import time
    
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    
    screen = pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Modern Terrain Test")
    
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.CULL_FACE)
    
    renderer = ModernTerrainRenderer(ctx)
    renderer.water_level = 5.0
    
    print("Generating terrain chunks...")
    
    # Generate 20x20 chunks
    chunk_size = 32
    tile_scale = 4.0
    height_scale = 1.0
    num_chunks = 20
    
    for cx in range(num_chunks):
        for cz in range(num_chunks):
            # Generate heightmap
            heightmap = np.zeros((chunk_size + 1, chunk_size + 1), dtype='f4')
            
            for z in range(chunk_size + 1):
                for x in range(chunk_size + 1):
                    wx = (cx * chunk_size + x) * tile_scale
                    wz = (cz * chunk_size + z) * tile_scale
                    
                    # Multi-octave noise approximation
                    h = np.sin(wx * 0.01) * np.cos(wz * 0.01) * 30
                    h += np.sin(wx * 0.03 + 1.5) * np.cos(wz * 0.03 + 0.7) * 15
                    h += np.sin(wx * 0.07 + 2.1) * np.cos(wz * 0.07 + 1.2) * 7
                    h += 20  # Base height
                    heightmap[z, x] = h
            
            renderer.create_chunk_mesh(
                cx, cz, heightmap,
                tile_scale, height_scale,
                cx * chunk_size * tile_scale,
                cz * chunk_size * tile_scale,
                biome='grassland'
            )
    
    total_chunks = num_chunks * num_chunks
    total_tris = total_chunks * chunk_size * chunk_size * 2
    print(f"Created {total_chunks} chunks, {total_tris:,} triangles")
    
    # Camera state
    cam_x = num_chunks * chunk_size * tile_scale / 2
    cam_y = 80
    cam_z = num_chunks * chunk_size * tile_scale / 2
    yaw = 0
    pitch = -20
    
    clock = pygame.time.Clock()
    running = True
    frame_times = []
    
    # Mouse capture for look
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    print("Controls: WASD=move, Mouse=look, ESC=quit")
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEMOTION:
                yaw += event.rel[0] * 0.2
                pitch -= event.rel[1] * 0.2
                pitch = max(-89, min(89, pitch))
        
        # Movement
        keys = pygame.key.get_pressed()
        speed = 100 * dt
        
        yaw_rad = np.radians(yaw)
        forward_x = -np.sin(yaw_rad)
        forward_z = -np.cos(yaw_rad)
        right_x = np.cos(yaw_rad)
        right_z = -np.sin(yaw_rad)
        
        if keys[pygame.K_w]:
            cam_x += forward_x * speed
            cam_z += forward_z * speed
        if keys[pygame.K_s]:
            cam_x -= forward_x * speed
            cam_z -= forward_z * speed
        if keys[pygame.K_a]:
            cam_x -= right_x * speed
            cam_z -= right_z * speed
        if keys[pygame.K_d]:
            cam_x += right_x * speed
            cam_z += right_z * speed
        if keys[pygame.K_SPACE]:
            cam_y += speed
        if keys[pygame.K_LSHIFT]:
            cam_y -= speed
        
        # Update camera
        renderer.set_camera(cam_x, cam_y, cam_z, yaw, pitch, 
                           fov=70, aspect=1280/720)
        
        # Render
        ctx.clear(0.55, 0.70, 0.85)
        
        start = time.perf_counter()
        renderer.render()
        ctx.finish()
        elapsed = time.perf_counter() - start
        frame_times.append(elapsed)
        
        pygame.display.flip()
        
        # Show stats every 60 frames
        if len(frame_times) >= 60:
            avg_ms = np.mean(frame_times) * 1000
            fps = 1000 / avg_ms
            print(f"FPS: {fps:.0f} | {renderer.frame_stats['triangles']:,} tris | "
                  f"{renderer.frame_stats['draw_calls']} draws")
            frame_times = []
    
    renderer.cleanup()
    pygame.quit()


if __name__ == "__main__":
    test_terrain_renderer()

