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
        
        # Walls (with rotation support)
        for wall in getattr(structure, 'walls', []):
            x, y, z = wall.x, wall.y, wall.z
            w, h = wall.width / 2, wall.height / 2
            d = wall.thickness / 2  # Wall uses 'thickness' not 'depth'
            rotation = getattr(wall, 'rotation', 0.0)
            # Use individual wall color if available, else fallback to DNA wall_color
            color = getattr(wall, 'color', wall_color)
            v, n, c = create_box(0, h, 0, w, h, d, color)
            
            # Apply rotation and translation if wall is rotated
            if abs(rotation) > 0.001:
                rad = math.radians(rotation)
                cos_r, sin_r = math.cos(rad), math.sin(rad)
                rotated_v = []
                rotated_n = []
                for vx, vy, vz in v:
                    # Rotate around Y axis, then translate
                    rx = vx * cos_r - vz * sin_r + x
                    rz = vx * sin_r + vz * cos_r + z
                    rotated_v.append((rx, vy + y, rz))
                for nx, ny, nz in n:
                    # Rotate normals too
                    rotated_n.append((nx * cos_r - nz * sin_r, ny, nx * sin_r + nz * cos_r))
                v = rotated_v
                n = rotated_n
            else:
                # Just translate
                v = [(vx + x, vy + y, vz + z) for vx, vy, vz in v]
            
            all_vertices.extend(v)
            all_normals.extend(n)
            all_colors.extend(c)
        
        # Floors
        for floor in getattr(structure, 'floors', []):
            x, y, z = floor.x, floor.y, floor.z
            w, d = floor.width / 2, floor.depth / 2
            t = getattr(floor, 'thickness', 0.2) / 2  # Use floor's thickness
            # Use individual floor color if available
            color = getattr(floor, 'color', floor_color)
            # Floor position - legacy draws from 0 to thickness, so center is at y + t
            v, n, c = create_box(x, y + t, z, w, t, d, color)
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
            # Use individual ramp color if available
            color = getattr(ramp, 'color', floor_color)
            v, n, c = create_ramp(x, y_bot, y_top, z, w, d, direction, color)
            all_vertices.extend(v)
            all_normals.extend(n)
            all_colors.extend(c)
        
        # Pillars
        for pillar in getattr(structure, 'pillars', []):
            x, z = pillar.x, pillar.z
            y_bot = pillar.y_bottom
            y_top = pillar.y_top
            w = pillar.width / 2
            h = (y_top - y_bot) / 2
            y_center = y_bot + h
            # Use individual pillar color if available
            color = getattr(pillar, 'color', wall_color)
            # Approximate pillar as box
            v, n, c = create_box(x, y_center, z, w, h, w, color)
            all_vertices.extend(v)
            all_normals.extend(n)
            all_colors.extend(c)
        
        # Staircases - draw as individual steps like legacy
        for stair in getattr(structure, 'staircases', []):
            stair_x, stair_z = stair.x, stair.z
            y_bot = stair.y_bottom
            height = stair.y_top - stair.y_bottom
            hw = stair.width / 2
            length = stair.length
            stair_color = getattr(stair, 'color', floor_color)
            
            # Rotation
            angle_rad = math.radians(stair.direction)
            cos_r, sin_r = math.cos(angle_rad), math.sin(angle_rad)
            
            # Individual steps like legacy
            num_steps = max(2, int(height / 0.4))
            step_height = height / num_steps
            step_depth = length / num_steps
            
            for step_i in range(num_steps):
                # Local coords (stair goes in local +X from origin)
                local_x = step_i * step_depth + step_depth / 2
                local_y = y_bot + step_i * step_height + step_height / 2
                
                # Rotate local position to world
                world_x = local_x * cos_r + stair_x
                world_z = local_x * sin_r + stair_z
                
                # Create step as a box
                v, n, c = create_box(0, 0, 0, step_depth/2, step_height/2, hw, stair_color)
                
                # Rotate and translate vertices
                for i, (vx, vy, vz) in enumerate(v):
                    # Rotate around Y
                    rx = vx * cos_r - vz * sin_r + world_x
                    rz = vx * sin_r + vz * cos_r + world_z
                    v[i] = (rx, vy + local_y, rz)
                for i, (nx, ny, nz) in enumerate(n):
                    n[i] = (nx * cos_r - nz * sin_r, ny, nx * sin_r + nz * cos_r)
                
                all_vertices.extend(v)
                all_normals.extend(n)
                all_colors.extend(c)
        
        # Doorways (frame posts and lintel)
        for doorway in getattr(structure, 'doorways', []):
            x, y, z = doorway.x, doorway.y, doorway.z
            hw = doorway.width / 2
            h = doorway.height
            frame_w = 0.15  # Half-width of frame post
            rotation = getattr(doorway, 'rotation', 0.0)
            frame_color = getattr(doorway, 'frame_color', (0.35, 0.3, 0.25))
            
            # Left post
            v, n, c = create_box(0, h/2, 0, frame_w, h/2, frame_w, frame_color)
            # Right post
            v2, n2, c2 = create_box(0, h/2, 0, frame_w, h/2, frame_w, frame_color)
            # Top lintel
            v3, n3, c3 = create_box(0, h + frame_w, 0, hw + frame_w*2, frame_w, frame_w, frame_color)
            
            # Apply rotation and translate to positions
            rad = math.radians(rotation)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            
            # Left post at -hw-frame_w
            left_offset = -hw - frame_w
            for i, (vx, vy, vz) in enumerate(v):
                rx = (vx + left_offset) * cos_r - vz * sin_r + x
                rz = (vx + left_offset) * sin_r + vz * cos_r + z
                v[i] = (rx, vy + y, rz)
            for i, (nx, ny, nz) in enumerate(n):
                n[i] = (nx * cos_r - nz * sin_r, ny, nx * sin_r + nz * cos_r)
            
            # Right post at +hw+frame_w
            right_offset = hw + frame_w
            for i, (vx, vy, vz) in enumerate(v2):
                rx = (vx + right_offset) * cos_r - vz * sin_r + x
                rz = (vx + right_offset) * sin_r + vz * cos_r + z
                v2[i] = (rx, vy + y, rz)
            for i, (nx, ny, nz) in enumerate(n2):
                n2[i] = (nx * cos_r - nz * sin_r, ny, nx * sin_r + nz * cos_r)
            
            # Lintel (already centered, just translate)
            for i, (vx, vy, vz) in enumerate(v3):
                rx = vx * cos_r - vz * sin_r + x
                rz = vx * sin_r + vz * cos_r + z
                v3[i] = (rx, vy + y, rz)
            for i, (nx, ny, nz) in enumerate(n3):
                n3[i] = (nx * cos_r - nz * sin_r, ny, nx * sin_r + nz * cos_r)
            
            all_vertices.extend(v)
            all_normals.extend(n)
            all_colors.extend(c)
            all_vertices.extend(v2)
            all_normals.extend(n2)
            all_colors.extend(c2)
            all_vertices.extend(v3)
            all_normals.extend(n3)
            all_colors.extend(c3)
        
        # Arches (simplified as a curved structure)
        for arch in getattr(structure, 'arches', []):
            x, y, z = arch.x, arch.y, arch.z
            hw = arch.width / 2
            h = arch.height
            t = getattr(arch, 'thickness', 0.5) / 2
            rotation = getattr(arch, 'rotation', 0.0)
            arch_color = getattr(arch, 'color', (0.5, 0.45, 0.4))
            
            # Left pillar
            pillar_h = h * 0.7
            v, n, c = create_box(-hw, pillar_h/2, 0, t, pillar_h/2, t, arch_color)
            # Right pillar
            v2, n2, c2 = create_box(hw, pillar_h/2, 0, t, pillar_h/2, t, arch_color)
            # Top beam (simplified - not curved)
            beam_h = h - pillar_h
            v3, n3, c3 = create_box(0, pillar_h + beam_h/2, 0, hw + t, beam_h/2, t, arch_color)
            
            # Apply rotation
            rad = math.radians(rotation)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            
            for geom in [v, v2, v3]:
                for i, (vx, vy, vz) in enumerate(geom):
                    rx = vx * cos_r - vz * sin_r + x
                    rz = vx * sin_r + vz * cos_r + z
                    geom[i] = (rx, vy + y, rz)
            
            for normals in [n, n2, n3]:
                for i, (nx, ny, nz) in enumerate(normals):
                    normals[i] = (nx * cos_r - nz * sin_r, ny, nx * sin_r + nz * cos_r)
            
            all_vertices.extend(v)
            all_normals.extend(n)
            all_colors.extend(c)
            all_vertices.extend(v2)
            all_normals.extend(n2)
            all_colors.extend(c2)
            all_vertices.extend(v3)
            all_normals.extend(n3)
            all_colors.extend(c3)
    
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
        self.fog_start = 2000.0
        self.fog_end = 6000.0
        
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

