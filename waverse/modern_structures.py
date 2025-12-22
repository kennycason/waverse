"""
ModernGL Structure Renderer

GPU-accelerated building/structure rendering with:
- Batched geometry for multiple buildings
- Per-structure color from DNA
- Distance-based LOD (full mesh -> simplified box)
- Efficient culling
"""

import numpy as np
import moderngl
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import math

try:
    from pyglm import glm
except ImportError:
    import glm


# =============================================================================
# SHADERS
# =============================================================================

STRUCTURE_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

out vec3 v_normal;
out vec3 v_color;
out vec3 v_world_pos;
out float v_fog_factor;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec3 u_camera_pos;
uniform float u_fog_start;
uniform float u_fog_end;

void main() {
    v_world_pos = in_position;
    v_normal = in_normal;
    v_color = in_color;
    
    // Calculate fog
    float dist = length(in_position - u_camera_pos);
    v_fog_factor = clamp((dist - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);
    
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
}
"""

STRUCTURE_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_normal;
in vec3 v_color;
in vec3 v_world_pos;
in float v_fog_factor;

out vec4 fragColor;

uniform vec3 u_light_dir;
uniform vec3 u_ambient;
uniform vec3 u_fog_color;

void main() {
    // Directional lighting
    vec3 norm = normalize(v_normal);
    float diff = max(dot(norm, normalize(u_light_dir)), 0.0);
    
    // Combine lighting
    vec3 lit_color = v_color * (u_ambient + diff * 0.6);
    
    // Apply fog
    vec3 final_color = mix(lit_color, u_fog_color, v_fog_factor);
    
    fragColor = vec4(final_color, 1.0);
}
"""


# =============================================================================
# GEOMETRY GENERATORS
# =============================================================================

def create_box(x: float, y: float, z: float, 
               w: float, h: float, d: float,
               color: Tuple[float, float, float]) -> Tuple[List, List, List]:
    """
    Create a box (building block) geometry.
    
    Args:
        x, y, z: Center position
        w, h, d: Width, height, depth (half-extents)
        color: RGB color tuple
    
    Returns:
        vertices, normals, colors lists
    """
    vertices = []
    normals = []
    colors = []
    
    r, g, b = color
    
    # Define 6 faces with their normals
    faces = [
        # Front (+Z)
        ([(x-w, y-h, z+d), (x+w, y-h, z+d), (x+w, y+h, z+d), (x-w, y+h, z+d)], (0, 0, 1)),
        # Back (-Z)
        ([(x+w, y-h, z-d), (x-w, y-h, z-d), (x-w, y+h, z-d), (x+w, y+h, z-d)], (0, 0, -1)),
        # Right (+X)
        ([(x+w, y-h, z+d), (x+w, y-h, z-d), (x+w, y+h, z-d), (x+w, y+h, z+d)], (1, 0, 0)),
        # Left (-X)
        ([(x-w, y-h, z-d), (x-w, y-h, z+d), (x-w, y+h, z+d), (x-w, y+h, z-d)], (-1, 0, 0)),
        # Top (+Y)
        ([(x-w, y+h, z+d), (x+w, y+h, z+d), (x+w, y+h, z-d), (x-w, y+h, z-d)], (0, 1, 0)),
        # Bottom (-Y)
        ([(x-w, y-h, z-d), (x+w, y-h, z-d), (x+w, y-h, z+d), (x-w, y-h, z+d)], (0, -1, 0)),
    ]
    
    for face_verts, normal in faces:
        # Two triangles per face
        # Triangle 1: 0, 1, 2
        # Triangle 2: 0, 2, 3
        for idx in [0, 1, 2, 0, 2, 3]:
            vertices.append(face_verts[idx])
            normals.append(normal)
            # Vary color slightly by face for visual interest
            shade = 0.8 + abs(normal[1]) * 0.2  # Top/bottom slightly brighter
            colors.append((r * shade, g * shade, b * shade))
    
    return vertices, normals, colors


def create_ramp(x: float, y_bottom: float, y_top: float, z: float,
                w: float, d: float, direction: Tuple[float, float],
                color: Tuple[float, float, float]) -> Tuple[List, List, List]:
    """
    Create a ramp/stair geometry.
    
    Args:
        x, z: Center position
        y_bottom, y_top: Height range
        w, d: Width and depth
        direction: (dx, dz) direction ramp goes up
        color: RGB color
    """
    vertices = []
    normals = []
    colors = []
    
    r, g, b = color
    
    # Simplified ramp as a wedge
    dx, dz = direction
    h = y_top - y_bottom
    
    if abs(dx) > abs(dz):
        # Ramp along X axis
        if dx > 0:
            # Rising in +X
            verts = [
                (x-w, y_bottom, z-d), (x+w, y_top, z-d), (x+w, y_top, z+d), (x-w, y_bottom, z+d),  # Top slope
                (x-w, y_bottom, z-d), (x-w, y_bottom, z+d), (x+w, y_bottom, z+d), (x+w, y_bottom, z-d),  # Bottom
            ]
        else:
            verts = [
                (x+w, y_bottom, z-d), (x-w, y_top, z-d), (x-w, y_top, z+d), (x+w, y_bottom, z+d),
                (x-w, y_bottom, z-d), (x-w, y_bottom, z+d), (x+w, y_bottom, z+d), (x+w, y_bottom, z-d),
            ]
    else:
        # Ramp along Z axis
        if dz > 0:
            verts = [
                (x-w, y_bottom, z-d), (x+w, y_bottom, z-d), (x+w, y_top, z+d), (x-w, y_top, z+d),
                (x-w, y_bottom, z-d), (x-w, y_bottom, z+d), (x+w, y_bottom, z+d), (x+w, y_bottom, z-d),
            ]
        else:
            verts = [
                (x-w, y_top, z-d), (x+w, y_top, z-d), (x+w, y_bottom, z+d), (x-w, y_bottom, z+d),
                (x-w, y_bottom, z-d), (x-w, y_bottom, z+d), (x+w, y_bottom, z+d), (x+w, y_bottom, z-d),
            ]
    
    # Convert to triangles (2 quads = 4 triangles)
    for i, quad_start in enumerate([0, 4]):
        quad = verts[quad_start:quad_start+4]
        normal = (0, 1, 0) if i == 0 else (0, -1, 0)
        for idx in [0, 1, 2, 0, 2, 3]:
            vertices.append(quad[idx])
            normals.append(normal)
            colors.append((r, g, b))
    
    return vertices, normals, colors


def structure_to_geometry(structure: Any, lod: int = 0) -> Tuple[np.ndarray, int]:
    """
    Convert a waverse Structure to geometry arrays.
    
    Args:
        structure: waverse Structure with walls, floors, etc.
        lod: Level of detail (0=full, 1=simplified, 2=bbox only)
    
    Returns:
        (interleaved vertex data, vertex count)
    """
    all_vertices = []
    all_normals = []
    all_colors = []
    
    # Get structure color from DNA
    dna = getattr(structure, 'dna', None)
    if dna:
        wall_color = getattr(dna, 'wall_color', (0.7, 0.65, 0.6))
        roof_color = getattr(dna, 'roof_color', (0.5, 0.3, 0.2))
        floor_color = getattr(dna, 'floor_color', (0.6, 0.55, 0.5))
    else:
        wall_color = (0.7, 0.65, 0.6)
        roof_color = (0.5, 0.3, 0.2)
        floor_color = (0.6, 0.55, 0.5)
    
    if lod >= 2:
        # Just bounding box
        bbox = getattr(structure, 'bbox', None)
        if bbox:
            min_p, max_p = bbox
            cx = (min_p[0] + max_p[0]) / 2
            cy = (min_p[1] + max_p[1]) / 2
            cz = (min_p[2] + max_p[2]) / 2
            hw = (max_p[0] - min_p[0]) / 2
            hh = (max_p[1] - min_p[1]) / 2
            hd = (max_p[2] - min_p[2]) / 2
            
            v, n, c = create_box(cx, cy, cz, hw, hh, hd, wall_color)
            all_vertices.extend(v)
            all_normals.extend(n)
            all_colors.extend(c)
    else:
        # Full geometry from structure components
        
        # Walls
        for wall in getattr(structure, 'walls', []):
            x, y, z = wall.x, wall.y, wall.z
            w, h, d = wall.width / 2, wall.height / 2, wall.depth / 2
            v, n, c = create_box(x, y + h, z, w, h, d, wall_color)
            all_vertices.extend(v)
            all_normals.extend(n)
            all_colors.extend(c)
        
        # Floors
        for floor in getattr(structure, 'floors', []):
            x, y, z = floor.x, floor.y, floor.z
            w, d = floor.width / 2, floor.depth / 2
            h = 0.2  # Floor thickness
            v, n, c = create_box(x, y, z, w, h, d, floor_color)
            all_vertices.extend(v)
            all_normals.extend(n)
            all_colors.extend(c)
        
        # Ramps/stairs
        for ramp in getattr(structure, 'ramps', []):
            x, z = ramp.x, ramp.z
            y_bot, y_top = ramp.y_bottom, ramp.y_top
            w = ramp.width / 2
            d = ramp.length / 2  # Ramp uses 'length' not 'depth'
            direction = getattr(ramp, 'direction', (1, 0))
            v, n, c = create_ramp(x, y_bot, y_top, z, w, d, direction, floor_color)
            all_vertices.extend(v)
            all_normals.extend(n)
            all_colors.extend(c)
        
        # Pillars
        for pillar in getattr(structure, 'pillars', []):
            x, y, z = pillar.x, pillar.y, pillar.z
            r = getattr(pillar, 'radius', 0.5)
            h = pillar.height / 2
            # Approximate pillar as box
            v, n, c = create_box(x, y + h, z, r, h, r, wall_color)
            all_vertices.extend(v)
            all_normals.extend(n)
            all_colors.extend(c)
        
        # Staircases
        for stair in getattr(structure, 'staircases', []):
            x, z = stair.x, stair.z
            y_bot = getattr(stair, 'y_bottom', stair.y if hasattr(stair, 'y') else 0)
            y_top = getattr(stair, 'y_top', y_bot + 5)
            w = getattr(stair, 'width', 2) / 2
            d = getattr(stair, 'depth', 4) / 2
            direction = getattr(stair, 'direction', (0, 1))
            v, n, c = create_ramp(x, y_bot, y_top, z, w, d, direction, floor_color)
            all_vertices.extend(v)
            all_normals.extend(n)
            all_colors.extend(c)
    
    if not all_vertices:
        return np.array([], dtype='f4'), 0
    
    # Interleave: position (3) + normal (3) + color (3) = 9 floats per vertex
    data = []
    for v, n, c in zip(all_vertices, all_normals, all_colors):
        data.extend([v[0], v[1], v[2], n[0], n[1], n[2], c[0], c[1], c[2]])
    
    return np.array(data, dtype='f4'), len(all_vertices)


# =============================================================================
# RENDERER
# =============================================================================

@dataclass
class StructureMesh:
    """Cached GPU mesh for a structure."""
    vbo: moderngl.Buffer
    vao: moderngl.VertexArray
    vertex_count: int
    center: Tuple[float, float, float]
    lod: int


class ModernStructureRenderer:
    """
    GPU-accelerated structure/building renderer.
    
    Caches structure geometry in VBOs for efficient rendering.
    Uses LOD based on distance from camera.
    """
    
    # LOD thresholds (distance in world units)
    LOD_FULL = 150      # Full detail
    LOD_SIMPLE = 300    # Simplified
    LOD_BBOX = 500      # Bounding box only
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Compile shaders
        self.program = ctx.program(
            vertex_shader=STRUCTURE_VERTEX_SHADER,
            fragment_shader=STRUCTURE_FRAGMENT_SHADER,
        )
        
        # Cached meshes: structure_id -> {lod -> StructureMesh}
        self.meshes: Dict[int, Dict[int, StructureMesh]] = {}
        
        # Uniforms
        self.projection = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
        self.camera_pos = glm.vec3(0, 0, 0)
        self.light_dir = glm.vec3(0.5, 1.0, 0.3)
        self.ambient = glm.vec3(0.4, 0.4, 0.5)
        self.fog_color = glm.vec3(0.7, 0.8, 0.9)
        self.fog_start = 200.0
        self.fog_end = 600.0
        
        # Frame stats
        self.frame_stats = {
            'structures_rendered': 0,
            'triangles': 0,
            'draw_calls': 0,
        }
    
    def set_camera(self, projection: glm.mat4, view: glm.mat4, camera_pos: glm.vec3):
        """Set camera matrices."""
        self.projection = projection
        self.view = view
        self.camera_pos = camera_pos
    
    def _get_or_create_mesh(self, structure: Any, lod: int) -> Optional[StructureMesh]:
        """Get cached mesh or create new one."""
        struct_id = id(structure)
        
        if struct_id not in self.meshes:
            self.meshes[struct_id] = {}
        
        if lod in self.meshes[struct_id]:
            return self.meshes[struct_id][lod]
        
        # Generate geometry
        data, vertex_count = structure_to_geometry(structure, lod)
        
        if vertex_count == 0:
            return None
        
        # Create VBO
        vbo = self.ctx.buffer(data.tobytes())
        
        # Create VAO
        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')],
        )
        
        # Calculate center
        bbox = getattr(structure, 'bbox', None)
        if bbox:
            min_p, max_p = bbox
            center = ((min_p[0] + max_p[0]) / 2, (min_p[1] + max_p[1]) / 2, (min_p[2] + max_p[2]) / 2)
        else:
            center = (getattr(structure, 'x', 0), getattr(structure, 'y', 0), getattr(structure, 'z', 0))
        
        mesh = StructureMesh(vbo, vao, vertex_count, center, lod)
        self.meshes[struct_id][lod] = mesh
        
        return mesh
    
    def _determine_lod(self, center: Tuple[float, float, float]) -> int:
        """Determine LOD based on distance from camera."""
        dx = center[0] - self.camera_pos.x
        dy = center[1] - self.camera_pos.y
        dz = center[2] - self.camera_pos.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        
        if dist < self.LOD_FULL:
            return 0
        elif dist < self.LOD_SIMPLE:
            return 1
        else:
            return 2
    
    def update_structures(self, structure_manager: Any, camera_pos: Tuple[float, float, float]):
        """
        Prepare structures for rendering.
        
        Args:
            structure_manager: waverse StructureManager with structures list
            camera_pos: Camera position for culling
        """
        # For now we just ensure meshes exist - actual rendering happens in render()
        pass
    
    def render(self, structures: List[Any]):
        """
        Render all visible structures.
        
        Args:
            structures: List of waverse Structure objects to render
        """
        self.frame_stats['structures_rendered'] = 0
        self.frame_stats['triangles'] = 0
        self.frame_stats['draw_calls'] = 0
        
        # Set common uniforms
        self.program['u_projection'].write(self.projection)
        self.program['u_view'].write(self.view)
        self.program['u_camera_pos'].write(self.camera_pos)
        self.program['u_light_dir'].write(self.light_dir)
        self.program['u_ambient'].write(self.ambient)
        self.program['u_fog_color'].write(self.fog_color)
        self.program['u_fog_start'].value = self.fog_start
        self.program['u_fog_end'].value = self.fog_end
        
        cam_pos = (self.camera_pos.x, self.camera_pos.y, self.camera_pos.z)
        
        for structure in structures:
            # Get bounding box center
            bbox = getattr(structure, 'bbox', None)
            if bbox:
                min_p, max_p = bbox
                center = ((min_p[0] + max_p[0]) / 2, (min_p[1] + max_p[1]) / 2, (min_p[2] + max_p[2]) / 2)
            else:
                center = (getattr(structure, 'x', 0), getattr(structure, 'y', 0), getattr(structure, 'z', 0))
            
            # Distance culling
            dx = center[0] - cam_pos[0]
            dz = center[2] - cam_pos[2]
            dist_sq = dx * dx + dz * dz
            
            if dist_sq > self.fog_end * self.fog_end:
                continue  # Too far, skip
            
            # Determine LOD
            lod = self._determine_lod(center)
            
            # Get/create mesh
            mesh = self._get_or_create_mesh(structure, lod)
            if mesh is None:
                continue
            
            # Render
            mesh.vao.render(moderngl.TRIANGLES)
            
            self.frame_stats['structures_rendered'] += 1
            self.frame_stats['triangles'] += mesh.vertex_count // 3
            self.frame_stats['draw_calls'] += 1
    
    def invalidate_structure(self, structure: Any):
        """Remove cached meshes for a structure (call when structure changes)."""
        struct_id = id(structure)
        if struct_id in self.meshes:
            for mesh in self.meshes[struct_id].values():
                mesh.vbo.release()
                mesh.vao.release()
            del self.meshes[struct_id]
    
    def cleanup(self):
        """Release all GPU resources."""
        for struct_meshes in self.meshes.values():
            for mesh in struct_meshes.values():
                mesh.vbo.release()
                mesh.vao.release()
        self.meshes.clear()
        self.program.release()

