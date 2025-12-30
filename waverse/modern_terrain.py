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
    # === SPECIAL EXOTIC BIOMES ===
    # Psychedelic - will be overridden with rainbow cycling, but base is magenta
    'Psychedelic': (0.85, 0.20, 0.85),
    'psychedelic': (0.85, 0.20, 0.85),
    # Hellfire - deep volcanic reds and oranges
    'Hellfire': (0.45, 0.12, 0.08),
    'hellfire': (0.45, 0.12, 0.08),
    # Shadow - dark purples and blacks
    'Shadow': (0.12, 0.08, 0.18),
    'shadow': (0.12, 0.08, 0.18),
    # Crystal - pale cyan/white crystalline
    'Crystal': (0.75, 0.92, 0.95),
    'crystal': (0.75, 0.92, 0.95),
    # Void - near black with hints of deep purple
    'Void': (0.05, 0.02, 0.10),
    'void': (0.05, 0.02, 0.10),
    # Mountain - rocky grays with hints of brown
    'Mountain': (0.45, 0.42, 0.38),
    # Deep Ocean - dark blues
    'Deep_ocean': (0.08, 0.15, 0.35),
}


def get_terrain_color(height: float, biome: str = 'grassland', 
                      water_level: float = 0.0, world_x: float = 0, world_z: float = 0) -> Tuple[float, float, float]:
    """Get terrain color based on height and biome."""
    import math
    
    # === SPECIAL BIOME COLORING ===
    biome_lower = biome.lower() if biome else 'grassland'
    
    if biome_lower == 'psychedelic':
        # TRIPPY MULTI-SCHEME - not just rainbow!
        zone_x = world_x * 0.005
        zone_z = world_z * 0.005
        zone_phase = math.sin(zone_x) * math.cos(zone_z) * 3.0
        swirl = math.sin(zone_x * 3 + zone_z * 2) + math.cos(zone_x * 2 - zone_z * 3)
        height_band = (height * 0.05) % 6.0
        tile_hue = ((world_x * 0.02 + world_z * 0.02 * 1.618) * 2.399) % (2 * math.pi)
        
        phase1 = zone_phase + swirl * 0.5 + height_band
        phase2 = zone_phase * 1.3 - swirl * 0.7 + height_band * 0.5 + 2.1
        phase3 = zone_phase * 0.7 + swirl * 0.3 - height_band * 0.3 + 4.2
        
        scheme_blend = (math.sin(zone_x * 0.7) + 1) * 0.5
        r1 = 0.5 + 0.5 * math.sin(phase1 + tile_hue)
        g1 = 0.3 + 0.4 * math.sin(phase2 + tile_hue + 1.5)
        b1 = 0.5 + 0.5 * math.sin(phase3 + tile_hue + 3.0)
        r2 = 0.4 + 0.4 * math.sin(phase2 - tile_hue + 1.0)
        g2 = 0.5 + 0.5 * math.sin(phase1 + tile_hue * 0.7)
        b2 = 0.6 + 0.4 * math.sin(phase3 - tile_hue * 0.5)
        
        r = max(0.1, min(1.0, r1 * scheme_blend + r2 * (1 - scheme_blend)))
        g = max(0.1, min(1.0, g1 * scheme_blend + g2 * (1 - scheme_blend)))
        b = max(0.1, min(1.0, b1 * scheme_blend + b2 * (1 - scheme_blend)))
        return (r, g, b)
    
    elif biome_lower == 'hellfire':
        # VOLCANIC - reds, oranges, blacks with lava streaks
        # Height creates lava rivers in low areas
        if height < 5:
            # LAVA! Bright orange-red
            glow = 0.7 + 0.3 * math.sin(world_x * 0.1 + world_z * 0.1)
            return (0.95 * glow, 0.35 * glow, 0.05)
        elif height < 15:
            # Cooling lava - dark red/black
            t = (height - 5) / 10.0
            return (0.45 - t * 0.25, 0.15 - t * 0.1, 0.08)
        else:
            # Volcanic rock - dark grays with red tint
            return (0.22 + height * 0.001, 0.12, 0.10)
    
    elif biome_lower == 'shadow':
        # DARK AND SCARY - deep purples, blacks, occasional eerie glow
        darkness = 0.15 + 0.05 * math.sin(world_x * 0.05) * math.sin(world_z * 0.05)
        # Rare glowing spots (like eyes in the dark)
        if abs(math.sin(world_x * 0.3) * math.sin(world_z * 0.3)) > 0.95:
            return (0.4, 0.1, 0.5)  # Eerie purple glow
        return (darkness * 0.6, darkness * 0.4, darkness + 0.08)
    
    elif biome_lower == 'crystal':
        # CRYSTALLINE - pale cyans, whites, with prismatic effects
        prism = abs(math.sin(world_x * 0.15 + world_z * 0.15 + height * 0.2))
        r = 0.7 + prism * 0.25
        g = 0.85 + prism * 0.1
        b = 0.95
        return (min(1.0, r), min(1.0, g), min(1.0, b))
    
    elif biome_lower == 'void':
        # THE VOID - almost entirely black with faint purple nebula
        void_noise = 0.02 + 0.03 * abs(math.sin(world_x * 0.08) * math.cos(world_z * 0.08))
        return (void_noise, void_noise * 0.5, void_noise + 0.05)
    
    # === NORMAL BIOME COLORING ===
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
        
        # For full 3D flight, calculate proper up vector
        # When upside down (|pitch| > 90), flip the up vector
        upside_down = abs(pitch) > 90
        up_flip = -1.0 if upside_down else 1.0
        
        # Up vector perpendicular to forward in the camera's vertical plane
        up_x = np.sin(yaw_rad) * np.sin(pitch_rad) * up_flip
        up_y = np.cos(pitch_rad) * up_flip
        up_z = np.cos(yaw_rad) * np.sin(pitch_rad) * up_flip
        
        target = self.camera_pos + glm.vec3(dir_x, dir_y, dir_z)
        up_vec = glm.vec3(up_x, up_y, up_z)
        self.view = glm.lookAt(self.camera_pos, target, up_vec)
        
        # Projection
        self.projection = glm.perspective(glm.radians(fov), aspect, near, far)
    
    def create_chunk_mesh(self, cx: int, cz: int, 
                          heightmap: np.ndarray,
                          tile_scale: float,
                          height_scale: float,
                          chunk_world_x: float,
                          chunk_world_z: float,
                          biome: str = 'grassland',
                          chaos_factor: float = 0.0) -> TerrainChunkMesh:
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
        
        # NOTE: Chaos terrain disabled for performance
        # Colors still change for exotic biomes, just terrain stays smooth
        
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
        
        # Check for SPECIAL EXOTIC BIOMES
        biome_lower = biome.lower() if biome else 'grassland'
        is_special = biome_lower in ('psychedelic', 'hellfire', 'shadow', 'crystal', 'void')
        
        if is_special:
            # === SPECIAL BIOME COLORING (per-vertex for effects) ===
            import math
            
            # All world X and Z for each vertex
            all_wx = np.stack([wx0, wx1, wx0, wx1, wx1, wx0], axis=1).flatten()
            all_wz = np.stack([wz0, wz0, wz1, wz0, wz1, wz1], axis=1).flatten()
            all_heights = np.stack([h00, h10, h01, h10, h11, h01], axis=1).flatten()
            n = len(all_heights)
            result = np.zeros((n, 3), dtype='f4')
            
            if biome_lower == 'psychedelic':
                # TRIPPY MULTI-SCHEME COLORING - not just rainbow!
                # Use multiple layered patterns for true psychedelic effect
                
                # Base pattern - slower, larger color zones
                zone_x = all_wx * 0.005
                zone_z = all_wz * 0.005
                zone_phase = np.sin(zone_x) * np.cos(zone_z) * 3.0
                
                # Medium detail - swirling patterns
                swirl = np.sin(zone_x * 3 + zone_z * 2) + np.cos(zone_x * 2 - zone_z * 3)
                
                # Fine detail - tile-to-tile variation
                tile_noise = np.sin(all_wx * 0.1) * np.sin(all_wz * 0.1) * 0.5
                
                # Height influence - different colors at different elevations
                height_band = (all_heights * 0.05) % 6.0
                
                # Combine into final phase with multiple harmonics
                phase1 = zone_phase + swirl * 0.5 + height_band
                phase2 = zone_phase * 1.3 - swirl * 0.7 + height_band * 0.5 + 2.1
                phase3 = zone_phase * 0.7 + swirl * 0.3 - height_band * 0.3 + 4.2
                
                # Add tile-level chaos for sharp transitions
                tile_shift = np.floor(all_wx * 0.02) + np.floor(all_wz * 0.02) * 1.618
                tile_hue = (tile_shift * 2.399) % (2 * np.pi)  # Golden angle
                
                # Mix schemes based on position
                scheme_blend = (np.sin(zone_x * 0.7) + 1) * 0.5  # 0-1
                
                # Scheme 1: Warm psychedelic (magentas, oranges, cyans)
                r1 = 0.5 + 0.5 * np.sin(phase1 + tile_hue)
                g1 = 0.3 + 0.4 * np.sin(phase2 + tile_hue + 1.5)
                b1 = 0.5 + 0.5 * np.sin(phase3 + tile_hue + 3.0)
                
                # Scheme 2: Cool psychedelic (teals, purples, greens)
                r2 = 0.4 + 0.4 * np.sin(phase2 - tile_hue + 1.0)
                g2 = 0.5 + 0.5 * np.sin(phase1 + tile_hue * 0.7)
                b2 = 0.6 + 0.4 * np.sin(phase3 - tile_hue * 0.5)
                
                # Blend schemes
                result[:, 0] = np.clip(r1 * scheme_blend + r2 * (1 - scheme_blend) + tile_noise, 0.1, 1.0)
                result[:, 1] = np.clip(g1 * scheme_blend + g2 * (1 - scheme_blend) + tile_noise * 0.5, 0.1, 1.0)
                result[:, 2] = np.clip(b1 * scheme_blend + b2 * (1 - scheme_blend) - tile_noise * 0.3, 0.1, 1.0)
                
                # Occasional color inversion zones for extra trippiness
                invert_zone = np.sin(all_wx * 0.008 + all_wz * 0.012) > 0.7
                result[invert_zone, 0], result[invert_zone, 2] = result[invert_zone, 2].copy(), result[invert_zone, 0].copy()
            
            elif biome_lower == 'hellfire':
                # VOLCANIC HELLSCAPE - molten lava, obsidian, ash
                lava_mask = all_heights < 8
                cooling_mask = (all_heights >= 8) & (all_heights < 20)
                rock_mask = (all_heights >= 20) & (all_heights < 60)
                ash_mask = all_heights >= 60
                
                # Lava with flowing animation effect (position-based shimmer)
                lava_flow = np.sin(all_wx * 0.08 + all_wz * 0.06) * 0.5 + 0.5
                lava_heat = 0.8 + 0.2 * lava_flow
                result[lava_mask, 0] = np.clip(0.95 * lava_heat[lava_mask], 0.7, 1.0)
                result[lava_mask, 1] = np.clip(0.25 + 0.25 * lava_flow[lava_mask], 0.15, 0.5)
                result[lava_mask, 2] = 0.02
                
                # Lava cracks in the cooled rock
                crack_pattern = np.sin(all_wx * 0.2) * np.sin(all_wz * 0.2)
                lava_crack = (np.abs(crack_pattern) < 0.05) & cooling_mask
                
                # Cooling rock - dark with red undertones
                t = (all_heights[cooling_mask] - 8) / 12.0
                result[cooling_mask, 0] = np.clip(0.5 - t * 0.3, 0.15, 0.5)
                result[cooling_mask, 1] = np.clip(0.12 - t * 0.06, 0.05, 0.15)
                result[cooling_mask, 2] = 0.05
                
                # Lava cracks glow through
                result[lava_crack, 0] = 0.9
                result[lava_crack, 1] = 0.4
                result[lava_crack, 2] = 0.05
                
                # Obsidian rock - black glass with purple sheen
                obsidian_sheen = np.sin(all_wx * 0.1 + all_wz * 0.15) * 0.5 + 0.5
                result[rock_mask, 0] = 0.08 + obsidian_sheen[rock_mask] * 0.08
                result[rock_mask, 1] = 0.05
                result[rock_mask, 2] = 0.12 + obsidian_sheen[rock_mask] * 0.06
                
                # Ash-covered peaks - gray with red dust
                result[ash_mask, 0] = 0.35
                result[ash_mask, 1] = 0.28
                result[ash_mask, 2] = 0.25
                
                # Scattered ember spots
                ember_hash = (np.floor(all_wx * 0.3) * 73856093 + np.floor(all_wz * 0.3) * 19349663) % 1000
                ember_mask = (ember_hash < 20) & rock_mask
                result[ember_mask] = [0.95, 0.5, 0.1]
            
            elif biome_lower == 'shadow':
                # DARK AND TERRIFYING - twisted corrupted landscape
                # Base darkness with slow undulating patterns
                darkness = 0.08 + 0.04 * np.sin(all_wx * 0.02) * np.sin(all_wz * 0.02)
                
                # Corruption veins - dark purple tendrils spreading across land
                vein_pattern = np.sin(all_wx * 0.08 + all_wz * 0.05) * np.cos(all_wx * 0.03 - all_wz * 0.07)
                vein_mask = np.abs(vein_pattern) < 0.15
                
                # Base dark purple-gray
                result[:, 0] = darkness * 0.5 + 0.05
                result[:, 1] = darkness * 0.3
                result[:, 2] = darkness * 0.8 + 0.12
                
                # Corruption veins are darker purple
                result[vein_mask, 0] = 0.15
                result[vein_mask, 1] = 0.02
                result[vein_mask, 2] = 0.25
                
                # Glowing eyes scattered across the land (CREEPY!)
                eye_x = np.floor(all_wx * 0.05)
                eye_z = np.floor(all_wz * 0.05)
                eye_hash = (eye_x * 73856093 + eye_z * 19349663) % 1000
                eye_mask = (eye_hash < 15) & (all_heights > 5)  # 1.5% chance, above water
                result[eye_mask] = [0.8, 0.1, 0.2]  # Glowing red eyes
                
                # Occasional eerie green glow spots
                glow_mask = np.abs(np.sin(all_wx * 0.3) * np.sin(all_wz * 0.3)) > 0.92
                result[glow_mask] = [0.1, 0.5, 0.2]
            
            elif biome_lower == 'crystal':
                # CRYSTALLINE WONDERLAND - prismatic rainbow reflections
                # Multiple overlapping wave patterns for iridescence
                wave1 = np.sin(all_wx * 0.12 + all_heights * 0.15)
                wave2 = np.sin(all_wz * 0.14 + all_heights * 0.12 + 2.0)
                wave3 = np.sin((all_wx + all_wz) * 0.08 + all_heights * 0.1 + 4.0)
                
                # Prismatic color from wave interference
                prism = wave1 * wave2 * 0.5 + 0.5
                
                # Base crystal white with rainbow tints
                result[:, 0] = np.clip(0.75 + wave1 * 0.2 + prism * 0.15, 0.5, 1.0)
                result[:, 1] = np.clip(0.85 + wave2 * 0.15, 0.6, 1.0)
                result[:, 2] = np.clip(0.92 + wave3 * 0.1, 0.7, 1.0)
                
                # Crystal facet edges - sharp lines where colors shift
                facet_pattern = np.floor(all_wx * 0.1) + np.floor(all_wz * 0.1)
                facet_hue = (facet_pattern * 0.618) % 1.0  # Golden ratio for variety
                facet_mask = (facet_hue < 0.3)
                result[facet_mask, 0] = np.clip(result[facet_mask, 0] + 0.15, 0, 1)
                result[facet_mask, 1] = np.clip(result[facet_mask, 1] - 0.1, 0, 1)
                
                # Bright reflection spots
                reflect_mask = (wave1 > 0.9) & (wave2 > 0.8)
                result[reflect_mask] = [1.0, 1.0, 1.0]
            
            elif biome_lower == 'void':
                # THE VOID - cosmic horror, stars in the darkness
                # Near-black base with subtle purple undertones
                void_base = 0.02
                result[:, 0] = void_base
                result[:, 1] = void_base * 0.3
                result[:, 2] = void_base + 0.03
                
                # Distant nebula glow - very subtle color patches
                nebula_x = all_wx * 0.003
                nebula_z = all_wz * 0.003
                nebula = np.sin(nebula_x) * np.cos(nebula_z) * 0.5 + 0.5
                result[:, 0] += nebula * 0.04
                result[:, 2] += (1 - nebula) * 0.05
                
                # Stars! Random bright points
                star_hash = (np.floor(all_wx * 0.5) * 73856093 + np.floor(all_wz * 0.5) * 19349663) % 10000
                star_mask = star_hash < 30  # 0.3% chance of star
                result[star_mask] = [0.9, 0.9, 1.0]  # White stars
                
                # Occasional colored stars
                color_star_mask = (star_hash >= 30) & (star_hash < 40)
                result[color_star_mask, 0] = 0.8
                result[color_star_mask, 1] = 0.5
                result[color_star_mask, 2] = 1.0  # Purple stars
                
                # Void rifts - tears in reality with glowing edges
                rift_pattern = np.sin(all_wx * 0.04) + np.sin(all_wz * 0.05)
                rift_mask = np.abs(rift_pattern) < 0.08
                result[rift_mask] = [0.4, 0.0, 0.6]  # Glowing purple rift edges
            
            colors = result.reshape((num_cells, 6, 3)).reshape((num_verts, 3))
        else:
            # === NORMAL BIOME COLORING ===
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

