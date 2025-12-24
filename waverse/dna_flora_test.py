#!/usr/bin/env python3
"""
DNA Flora Test Grid - Visualize DNA-driven plant variety

Run: python -m waverse.dna_flora_test

Shows a grid of plants with varying DNA parameters to test
the vertex shader deformation system. Goal: match V1 quality!
"""

import pygame
from pygame.locals import *
import moderngl
import numpy as np
import glm
import math
import sys
from typing import Tuple, List, Dict, Any

# Import DNA system
from .dna import PlantDNA, PlantType, ColorGene, Gene, SegmentGene


# ============================================================================
# DNA DEFORMATION VERTEX SHADER
# ============================================================================

DNA_FLORA_VERTEX_SHADER = """
#version 330

// Mesh data
in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

// Per-instance data (standard)
in vec3 in_instance_pos;
in float in_instance_scale;
in float in_instance_rot;
in vec3 in_instance_color;

// Per-instance DNA deformation parameters (NEW!)
in float in_dna_curve;      // Trunk curve amount (-1 to 1)
in float in_dna_twist;      // Twist amount (0 to 1)
in float in_dna_taper;      // Taper ratio (0.3 to 1.0)
in float in_dna_height;     // Height multiplier
in float in_dna_branch_droop;  // Branch droop amount

uniform mat4 u_projection;
uniform mat4 u_view;

out vec3 v_normal;
out vec3 v_color;
out vec3 v_world_pos;
out float v_height;

void main() {
    // Start with original position
    vec3 pos = in_position;
    
    // === DNA DEFORMATION ===
    // Normalize height (0 at base, 1 at top)
    float t = pos.y / max(in_dna_height, 0.1);
    t = clamp(t, 0.0, 1.0);
    
    // 1. TWIST - rotate around Y axis based on height
    float twist_angle = in_dna_twist * t * 3.14159 * 2.0;
    float cos_tw = cos(twist_angle);
    float sin_tw = sin(twist_angle);
    vec3 twisted = vec3(
        pos.x * cos_tw - pos.z * sin_tw,
        pos.y,
        pos.x * sin_tw + pos.z * cos_tw
    );
    pos = twisted;
    
    // 2. CURVE - bend the trunk/stem in X direction
    float curve_offset = in_dna_curve * sin(t * 3.14159) * in_dna_height * 0.3;
    pos.x += curve_offset;
    
    // 3. TAPER - narrow toward the top
    float taper = mix(1.0, in_dna_taper, t);
    pos.x *= taper;
    pos.z *= taper;
    
    // 4. HEIGHT SCALE
    pos.y *= in_dna_height;
    
    // 5. BRANCH DROOP - for foliage, droop outward based on distance from center
    float dist_from_center = length(pos.xz);
    pos.y -= in_dna_branch_droop * dist_from_center * 0.2;
    
    // === STANDARD INSTANCE TRANSFORM ===
    // Apply instance rotation (around Y axis)
    float cos_r = cos(in_instance_rot);
    float sin_r = sin(in_instance_rot);
    vec3 rotated = vec3(
        pos.x * cos_r - pos.z * sin_r,
        pos.y,
        pos.x * sin_r + pos.z * cos_r
    );
    
    // Apply instance scale
    vec3 scaled = rotated * in_instance_scale;
    
    // Apply instance position
    vec3 world_pos = scaled + in_instance_pos;
    
    v_world_pos = world_pos;
    v_height = in_instance_pos.y;
    
    // Transform normal (simplified - no full normal matrix for now)
    v_normal = in_normal;
    
    // Apply colors: mesh color * instance color tint
    v_color = in_color * in_instance_color;
    
    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
}
"""

DNA_FLORA_FRAGMENT_SHADER = """
#version 330

in vec3 v_normal;
in vec3 v_color;
in vec3 v_world_pos;
in float v_height;

out vec4 f_color;

uniform vec3 u_sun_dir;
uniform vec3 u_ambient;

void main() {
    // Simple directional lighting
    vec3 norm = normalize(v_normal);
    float diff = max(dot(norm, u_sun_dir), 0.0);
    
    // Combine ambient and diffuse
    vec3 lighting = u_ambient + vec3(0.7, 0.65, 0.6) * diff;
    
    vec3 color = v_color * lighting;
    
    f_color = vec4(color, 1.0);
}
"""


# ============================================================================
# MESH GENERATION (base templates matching legacy V1 shapes)
# ============================================================================

def create_trunk_mesh(height: float = 1.0, radius: float = 0.08) -> Tuple[List, List, List]:
    """Create a multi-ring trunk cylinder for smooth deformation."""
    vertices = []
    normals = []
    colors = []
    
    trunk_color = (0.45, 0.28, 0.15)
    rings = 12  # More rings = smoother curves when deformed
    segments = 8
    
    for ring in range(rings):
        t = ring / (rings - 1)
        y = t * height
        next_t = min(1.0, (ring + 1) / (rings - 1))
        next_y = next_t * height
        
        for seg in range(segments):
            angle1 = (seg / segments) * 2 * math.pi
            angle2 = ((seg + 1) / segments) * 2 * math.pi
            
            x1 = math.cos(angle1) * radius
            z1 = math.sin(angle1) * radius
            x2 = math.cos(angle2) * radius
            z2 = math.sin(angle2) * radius
            
            n1 = (math.cos(angle1), 0, math.sin(angle1))
            n2 = (math.cos(angle2), 0, math.sin(angle2))
            
            vertices.extend([(x1, y, z1), (x2, y, z2), (x1, next_y, z1)])
            normals.extend([n1, n2, n1])
            colors.extend([trunk_color] * 3)
            
            vertices.extend([(x2, y, z2), (x2, next_y, z2), (x1, next_y, z1)])
            normals.extend([n2, n2, n1])
            colors.extend([trunk_color] * 3)
    
    return vertices, normals, colors


def create_cone_canopy(base_y: float, height: float, radius: float) -> Tuple[List, List, List]:
    """Create a cone-shaped canopy (classic pine tree)."""
    vertices = []
    normals = []
    colors = []
    
    foliage_color = (1.0, 1.0, 1.0)
    segments = 12
    
    # Cone from base to tip
    tip_y = base_y + height
    
    for seg in range(segments):
        angle1 = (seg / segments) * 2 * math.pi
        angle2 = ((seg + 1) / segments) * 2 * math.pi
        
        x1 = math.cos(angle1) * radius
        z1 = math.sin(angle1) * radius
        x2 = math.cos(angle2) * radius
        z2 = math.sin(angle2) * radius
        
        # Triangle from base to tip
        n = (math.cos(angle1 + math.pi/segments), 0.5, math.sin(angle1 + math.pi/segments))
        n_len = math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        n = (n[0]/n_len, n[1]/n_len, n[2]/n_len)
        
        vertices.extend([(x1, base_y, z1), (x2, base_y, z2), (0, tip_y, 0)])
        normals.extend([n, n, (0, 1, 0)])
        colors.extend([foliage_color] * 3)
    
    return vertices, normals, colors


def create_dome_canopy(base_y: float, height: float, radius: float) -> Tuple[List, List, List]:
    """Create a dome-shaped canopy (deciduous tree)."""
    vertices = []
    normals = []
    colors = []
    
    foliage_color = (1.0, 1.0, 1.0)
    rings = 6
    segments = 10
    
    for ring in range(rings):
        phi1 = (ring / rings) * (math.pi / 2)
        phi2 = ((ring + 1) / rings) * (math.pi / 2)
        
        y1 = base_y + math.sin(phi1) * height
        r1 = math.cos(phi1) * radius
        y2 = base_y + math.sin(phi2) * height
        r2 = math.cos(phi2) * radius
        
        for seg in range(segments):
            theta1 = (seg / segments) * 2 * math.pi
            theta2 = ((seg + 1) / segments) * 2 * math.pi
            
            x1a = math.cos(theta1) * r1
            z1a = math.sin(theta1) * r1
            x2a = math.cos(theta2) * r1
            z2a = math.sin(theta2) * r1
            
            x1b = math.cos(theta1) * r2
            z1b = math.sin(theta1) * r2
            x2b = math.cos(theta2) * r2
            z2b = math.sin(theta2) * r2
            
            n1 = (math.cos(theta1), 0.5, math.sin(theta1))
            
            vertices.extend([(x1a, y1, z1a), (x2a, y1, z2a), (x1b, y2, z1b)])
            normals.extend([n1, n1, n1])
            colors.extend([foliage_color] * 3)
            
            vertices.extend([(x2a, y1, z2a), (x2b, y2, z2b), (x1b, y2, z1b)])
            normals.extend([n1, n1, n1])
            colors.extend([foliage_color] * 3)
    
    return vertices, normals, colors


def create_umbrella_canopy(base_y: float, height: float, radius: float) -> Tuple[List, List, List]:
    """Create a flat umbrella canopy (acacia style)."""
    vertices = []
    normals = []
    colors = []
    
    foliage_color = (1.0, 1.0, 1.0)
    segments = 12
    
    # Flat disc with slight center raise
    center_y = base_y + height * 0.3
    edge_y = base_y
    
    for seg in range(segments):
        angle1 = (seg / segments) * 2 * math.pi
        angle2 = ((seg + 1) / segments) * 2 * math.pi
        
        x1 = math.cos(angle1) * radius
        z1 = math.sin(angle1) * radius
        x2 = math.cos(angle2) * radius
        z2 = math.sin(angle2) * radius
        
        # Triangle from center to edge
        vertices.extend([(0, center_y, 0), (x1, edge_y, z1), (x2, edge_y, z2)])
        normals.extend([(0, 1, 0)] * 3)
        colors.extend([foliage_color] * 3)
    
    return vertices, normals, colors


def create_layered_canopy(base_y: float, height: float, radius: float, layers: int = 3) -> Tuple[List, List, List]:
    """Create stacked layer canopy (spruce style)."""
    vertices = []
    normals = []
    colors = []
    
    foliage_color = (1.0, 1.0, 1.0)
    segments = 10
    
    for layer in range(layers):
        layer_t = layer / layers
        layer_y = base_y + layer_t * height
        layer_r = radius * (1 - layer_t * 0.7)  # Narrower at top
        next_layer_y = base_y + ((layer + 0.5) / layers) * height
        
        for seg in range(segments):
            angle1 = (seg / segments) * 2 * math.pi
            angle2 = ((seg + 1) / segments) * 2 * math.pi
            
            x1 = math.cos(angle1) * layer_r
            z1 = math.sin(angle1) * layer_r
            x2 = math.cos(angle2) * layer_r
            z2 = math.sin(angle2) * layer_r
            
            # Triangle pointing up to next layer center
            vertices.extend([(x1, layer_y, z1), (x2, layer_y, z2), (0, next_layer_y, 0)])
            normals.extend([(0, 0.7, 0.3)] * 3)
            colors.extend([foliage_color] * 3)
    
    return vertices, normals, colors


def create_branches(trunk_height: float, count: int = 6, length: float = 0.3) -> Tuple[List, List, List]:
    """Create branches extending from trunk."""
    vertices = []
    normals = []
    colors = []
    
    branch_color = (0.4, 0.25, 0.12)
    
    for b in range(count):
        angle = (b / count) * 2 * math.pi
        branch_y = trunk_height * (0.4 + (b / count) * 0.4)  # Stagger heights
        
        # Branch start at trunk
        start_x = math.cos(angle) * 0.08
        start_z = math.sin(angle) * 0.08
        
        # Branch end outward
        end_x = math.cos(angle) * length
        end_z = math.sin(angle) * length
        end_y = branch_y - 0.05  # Slight droop
        
        # Simple triangular prism
        up = 0.015
        side = 0.015
        
        # Top face
        vertices.extend([
            (start_x, branch_y + up, start_z),
            (end_x, end_y + up, end_z),
            (start_x + side, branch_y, start_z + side)
        ])
        normals.extend([(0, 1, 0)] * 3)
        colors.extend([branch_color] * 3)
        
        vertices.extend([
            (end_x, end_y + up, end_z),
            (end_x + side, end_y, end_z + side),
            (start_x + side, branch_y, start_z + side)
        ])
        normals.extend([(0, 1, 0)] * 3)
        colors.extend([branch_color] * 3)
    
    return vertices, normals, colors


def create_tree_base_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a CONE tree (classic evergreen) with proper structure."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    # Trunk
    v, n, c = create_trunk_mesh(height=0.6, radius=0.06)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Branches
    v, n, c = create_branches(trunk_height=0.6, count=6, length=0.2)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Layered cone canopy (spruce style)
    v, n, c = create_layered_canopy(base_y=0.3, height=0.9, radius=0.4, layers=4)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_dome_tree_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a DOME tree (deciduous oak style)."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    # Trunk
    v, n, c = create_trunk_mesh(height=0.5, radius=0.08)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Branches
    v, n, c = create_branches(trunk_height=0.5, count=8, length=0.3)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Dome canopy
    v, n, c = create_dome_canopy(base_y=0.35, height=0.5, radius=0.5)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_umbrella_tree_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create an UMBRELLA tree (acacia style)."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    # Tall trunk
    v, n, c = create_trunk_mesh(height=0.8, radius=0.05)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Umbrella canopy at top
    v, n, c = create_umbrella_canopy(base_y=0.7, height=0.2, radius=0.6)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_bush_base_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a bush mesh - low dome shape."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    # No trunk, just dome starting at ground
    v, n, c = create_dome_canopy(base_y=0.0, height=0.4, radius=0.4)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_fern_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a fern mesh - fronds radiating outward."""
    vertices = []
    normals = []
    colors = []
    
    foliage_color = (1.0, 1.0, 1.0)
    frond_count = 8
    frond_length = 0.5
    frond_width = 0.08
    
    for f in range(frond_count):
        angle = (f / frond_count) * 2 * math.pi
        
        # Frond curves outward and down
        start_x = 0
        start_z = 0
        start_y = 0.1
        
        mid_x = math.cos(angle) * frond_length * 0.5
        mid_z = math.sin(angle) * frond_length * 0.5
        mid_y = 0.25  # Curves up then down
        
        end_x = math.cos(angle) * frond_length
        end_z = math.sin(angle) * frond_length
        end_y = 0.05  # Droops at end
        
        # Two triangles for frond
        perp_x = -math.sin(angle) * frond_width
        perp_z = math.cos(angle) * frond_width
        
        vertices.extend([
            (start_x, start_y, start_z),
            (mid_x + perp_x, mid_y, mid_z + perp_z),
            (mid_x - perp_x, mid_y, mid_z - perp_z)
        ])
        normals.extend([(0, 1, 0)] * 3)
        colors.extend([foliage_color] * 3)
        
        vertices.extend([
            (mid_x + perp_x, mid_y, mid_z + perp_z),
            (end_x, end_y, end_z),
            (mid_x - perp_x, mid_y, mid_z - perp_z)
        ])
        normals.extend([(0, 1, 0)] * 3)
        colors.extend([foliage_color] * 3)
    
    return (
        np.array(vertices, dtype='f4'),
        np.array(normals, dtype='f4'),
        np.array(colors, dtype='f4')
    )


def create_mushroom_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a mushroom mesh - stem and cap."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    stem_color = (0.9, 0.85, 0.8)
    cap_color = (1.0, 1.0, 1.0)  # Tinted by instance color
    
    # Stem (short cylinder)
    segments = 8
    stem_height = 0.3
    stem_radius = 0.05
    
    for seg in range(segments):
        angle1 = (seg / segments) * 2 * math.pi
        angle2 = ((seg + 1) / segments) * 2 * math.pi
        
        x1 = math.cos(angle1) * stem_radius
        z1 = math.sin(angle1) * stem_radius
        x2 = math.cos(angle2) * stem_radius
        z2 = math.sin(angle2) * stem_radius
        
        all_verts.extend([(x1, 0, z1), (x2, 0, z2), (x1, stem_height, z1)])
        all_norms.extend([(math.cos(angle1), 0, math.sin(angle1))] * 3)
        all_cols.extend([stem_color] * 3)
        
        all_verts.extend([(x2, 0, z2), (x2, stem_height, z2), (x1, stem_height, z1)])
        all_norms.extend([(math.cos(angle2), 0, math.sin(angle2))] * 3)
        all_cols.extend([stem_color] * 3)
    
    # Cap (dome on top)
    cap_base = stem_height - 0.02
    cap_height = 0.15
    cap_radius = 0.2
    
    for seg in range(segments):
        angle1 = (seg / segments) * 2 * math.pi
        angle2 = ((seg + 1) / segments) * 2 * math.pi
        
        x1 = math.cos(angle1) * cap_radius
        z1 = math.sin(angle1) * cap_radius
        x2 = math.cos(angle2) * cap_radius
        z2 = math.sin(angle2) * cap_radius
        
        # Triangle from edge to top
        all_verts.extend([(x1, cap_base, z1), (x2, cap_base, z2), (0, cap_base + cap_height, 0)])
        all_norms.extend([(0, 0.7, 0.3)] * 3)
        all_cols.extend([cap_color] * 3)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


# ============================================================================
# DNA FLORA RENDERER
# ============================================================================

class DNAFloraRenderer:
    """
    Renders flora with DNA-driven vertex deformation.
    Uses instanced rendering with extended per-instance DNA parameters.
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Compile shader
        self.program = ctx.program(
            vertex_shader=DNA_FLORA_VERTEX_SHADER,
            fragment_shader=DNA_FLORA_FRAGMENT_SHADER
        )
        
        # Create base meshes for different plant types
        self.meshes = {
            'tree_cone': create_tree_base_mesh(),
            'tree_dome': create_dome_tree_mesh(),
            'tree_umbrella': create_umbrella_tree_mesh(),
            'bush': create_bush_base_mesh(),
            'fern': create_fern_mesh(),
            'mushroom': create_mushroom_mesh(),
        }
        
        # Create VBOs for each mesh type
        self.mesh_vbos = {}
        for name, mesh in self.meshes.items():
            self.mesh_vbos[name] = self._create_mesh_vbo(mesh)
        
        # Instance data per mesh type
        self.instances = {name: [] for name in self.meshes}
        
        # Camera matrices
        self.projection = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
    
    def _create_mesh_vbo(self, mesh_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> moderngl.Buffer:
        """Create a VBO from mesh data (verts, normals, colors)."""
        verts, norms, cols = mesh_data
        # Interleave: position(3) + normal(3) + color(3) = 9 floats per vertex
        data = np.zeros((len(verts), 9), dtype='f4')
        data[:, 0:3] = verts
        data[:, 3:6] = norms
        data[:, 6:9] = cols
        return self.ctx.buffer(data.tobytes())
    
    def add_instance(self, mesh_type: str, x: float, y: float, z: float, 
                     scale: float, rotation: float,
                     color: Tuple[float, float, float],
                     dna: PlantDNA = None):
        """Add an instance with DNA parameters."""
        if mesh_type not in self.instances:
            mesh_type = 'bush'  # Fallback
        
        # Extract DNA parameters
        curve = 0.0
        twist = 0.0
        taper = 0.8
        height = 1.0
        droop = 0.0
        
        if dna:
            # Get from DNA if available
            if dna.trunk_segments:
                seg = dna.trunk_segments[0]
                curve = getattr(seg, 'curve', 0.0)
                twist = getattr(seg, 'twist', 0.0)
                taper = getattr(seg, 'taper', 0.8)
            
            height_gene = getattr(dna, 'height_gene', None)
            if height_gene:
                height = height_gene.value if hasattr(height_gene, 'value') else 1.0
            
            droop = getattr(dna, 'droop', 0.0)
        
        self.instances[mesh_type].append((
            x, y, z,           # position
            scale,             # scale
            rotation,          # rotation
            *color,            # color RGB
            curve,             # DNA curve
            twist,             # DNA twist
            taper,             # DNA taper
            height,            # DNA height
            droop              # DNA branch droop
        ))
    
    def set_camera(self, projection: glm.mat4, view: glm.mat4):
        self.projection = projection
        self.view = view
    
    def render(self):
        """Render all instances."""
        self.program['u_projection'].write(self.projection)
        self.program['u_view'].write(self.view)
        self.program['u_sun_dir'].value = (0.5, 0.8, 0.3)
        self.program['u_ambient'].value = (0.4, 0.4, 0.45)
        
        # Render each mesh type
        for mesh_type, instances in self.instances.items():
            if instances:
                vbo = self.mesh_vbos[mesh_type]
                vertex_count = len(self.meshes[mesh_type][0])
                self._render_instances(vbo, instances, vertex_count)
    
    def _render_instances(self, mesh_vbo: moderngl.Buffer, instances: list, vertex_count: int):
        """Render instances of a mesh type."""
        if not instances:
            return
        
        # Instance data format: 
        # pos(3) + scale(1) + rot(1) + color(3) + curve(1) + twist(1) + taper(1) + height(1) + droop(1) = 13 floats
        instance_data = np.array(instances, dtype='f4')
        instance_vbo = self.ctx.buffer(instance_data.tobytes())
        
        # Create VAO
        vao = self.ctx.vertex_array(
            self.program,
            [
                # Mesh data
                (mesh_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color'),
                # Instance data (per-instance)
                (instance_vbo, '3f 1f 1f 3f 1f 1f 1f 1f 1f /i', 
                 'in_instance_pos', 'in_instance_scale', 'in_instance_rot', 'in_instance_color',
                 'in_dna_curve', 'in_dna_twist', 'in_dna_taper', 'in_dna_height', 'in_dna_branch_droop'),
            ]
        )
        
        vao.render(moderngl.TRIANGLES, instances=len(instances))
        
        # Cleanup
        vao.release()
        instance_vbo.release()
    
    def clear_instances(self):
        """Clear all instances for next frame."""
        for instances in self.instances.values():
            instances.clear()


# ============================================================================
# TEST GRID VIEWER
# ============================================================================

def run_test_grid():
    """Run the DNA flora test grid viewer."""
    pygame.init()
    
    WIDTH, HEIGHT = 1280, 720
    
    # Request OpenGL 4.1 Core profile (required for macOS)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
    
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("DNA Flora Test Grid - Press Q to quit, WASD to move, Mouse to look")
    
    # Create ModernGL context from existing pygame context
    ctx = moderngl.create_context(require=410)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.CULL_FACE)
    
    # Create renderer
    renderer = DNAFloraRenderer(ctx)
    
    # Camera state
    cam_x, cam_y, cam_z = 0, 5, 15
    cam_yaw, cam_pitch = 0, -20
    
    # Generate test DNA grid with variety of plant types
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    GRID_SIZE = 10  # 10x10 grid
    SPACING = 3.5   # 3.5 units between plants
    
    # Load real DNA from backup files if available
    import os
    import json
    real_dnas = []
    dna_log_dir = os.path.expanduser("~/.waverse/dna_logs")
    if os.path.exists(dna_log_dir):
        for fname in os.listdir(dna_log_dir):
            if fname.startswith("plant_dna_") and fname.endswith(".json"):
                try:
                    with open(os.path.join(dna_log_dir, fname)) as f:
                        data = json.load(f)
                        if 'dna' in data:
                            real_dnas.append(data['dna'])
                except:
                    pass
    print(f"Loaded {len(real_dnas)} real DNA samples from logs")
    
    # Mesh types to cycle through
    mesh_types = ['tree_cone', 'tree_dome', 'tree_umbrella', 'bush', 'fern', 'mushroom']
    
    test_plants = []
    
    for gx in range(GRID_SIZE):
        for gz in range(GRID_SIZE):
            # Cycle through mesh types
            mesh_idx = (gx + gz * 3) % len(mesh_types)
            mesh_type = mesh_types[mesh_idx]
            
            # Create DNA - use real DNA if available, otherwise random
            if real_dnas and rng.random() < 0.3:
                # Use real DNA parameters (30% chance)
                real_dna = real_dnas[int(rng.integers(0, len(real_dnas)))]
                
                # Map to our mesh type based on plant_type
                pt = real_dna.get('plant_type', 'tree').lower()
                if 'fern' in pt:
                    mesh_type = 'fern'
                elif 'mushroom' in pt:
                    mesh_type = 'mushroom'
                elif 'bush' in pt or 'shrub' in pt:
                    mesh_type = 'bush'
                elif 'umbrella' in str(real_dna.get('canopy_shape', '')).lower():
                    mesh_type = 'tree_umbrella'
                elif 'dome' in str(real_dna.get('canopy_shape', '')).lower():
                    mesh_type = 'tree_dome'
                else:
                    mesh_type = 'tree_cone'
                
                # Create DNA from real data
                plant_type = PlantType.TREE
                if mesh_type == 'bush':
                    plant_type = PlantType.BUSH
                elif mesh_type == 'fern':
                    plant_type = PlantType.FERN
                elif mesh_type == 'mushroom':
                    plant_type = PlantType.MUSHROOM
                
                seed = real_dna.get('species_id', int(rng.integers(0, 2**31)))
                dna = PlantDNA.create_random(plant_type, seed)
                
                # Apply real DNA parameters
                if 'trunk_segments' in real_dna and real_dna['trunk_segments']:
                    seg = real_dna['trunk_segments'][0]
                    dna.trunk_segments[0].curve = seg.get('curve', 0)
                    dna.trunk_segments[0].twist = seg.get('twist', 0)
                    dna.trunk_segments[0].taper = seg.get('taper', 0.8)
                
                if 'height_gene' in real_dna:
                    dna.height_gene.value = real_dna['height_gene'].get('value', 1.0)
                
                if 'leaf_color' in real_dna:
                    lc = real_dna['leaf_color']
                    dna.leaf_color.r = lc.get('r', 0.3)
                    dna.leaf_color.g = lc.get('g', 0.6)
                    dna.leaf_color.b = lc.get('b', 0.2)
                
                dna.droop = real_dna.get('droop', 0.0)
            else:
                # Create random DNA
                plant_type = PlantType.TREE
                if mesh_type == 'bush':
                    plant_type = PlantType.BUSH
                elif mesh_type == 'fern':
                    plant_type = PlantType.FERN
                elif mesh_type == 'mushroom':
                    plant_type = PlantType.MUSHROOM
                
                seed = int(rng.integers(0, 2**31))
                dna = PlantDNA.create_random(plant_type, seed)
                
                # Mutate for variety
                dna = dna.mutate(rng, strength=0.6)
            
            # Position in grid
            x = (gx - GRID_SIZE/2) * SPACING
            z = (gz - GRID_SIZE/2) * SPACING
            
            # Color from DNA
            color = (dna.leaf_color.r, dna.leaf_color.g, dna.leaf_color.b)
            
            test_plants.append({
                'x': x, 'y': 0, 'z': z,
                'scale': 2.0 + rng.random() * 1.5,
                'rotation': rng.random() * math.pi * 2,
                'color': color,
                'dna': dna,
                'mesh_type': mesh_type
            })
    
    print(f"Generated {len(test_plants)} test plants")
    print("Controls: WASD to move, Mouse to look, Q to quit")
    
    # Capture mouse
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_q or event.key == K_ESCAPE:
                    running = False
        
        # Mouse look
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        cam_yaw -= mouse_dx * 0.2
        cam_pitch -= mouse_dy * 0.2
        cam_pitch = max(-89, min(89, cam_pitch))
        
        # Keyboard movement
        keys = pygame.key.get_pressed()
        move_speed = 10 * dt
        
        yaw_rad = math.radians(cam_yaw)
        forward_x = -math.sin(yaw_rad)
        forward_z = -math.cos(yaw_rad)
        right_x = math.cos(yaw_rad)
        right_z = -math.sin(yaw_rad)
        
        # Ignore all movement when system modifiers are held (for screenshots: Cmd+Shift+Ctrl+4)
        mods = pygame.key.get_mods()
        system_mod_held = (mods & pygame.KMOD_META) or (mods & pygame.KMOD_CTRL)
        
        if not system_mod_held:
            if keys[K_w]:
                cam_x += forward_x * move_speed
                cam_z += forward_z * move_speed
            if keys[K_s]:
                cam_x -= forward_x * move_speed
                cam_z -= forward_z * move_speed
            if keys[K_a]:
                cam_x -= right_x * move_speed
                cam_z -= right_z * move_speed
            if keys[K_d]:
                cam_x += right_x * move_speed
                cam_z += right_z * move_speed
            if keys[K_SPACE]:
                cam_y += move_speed
            if keys[K_LSHIFT]:
                cam_y -= move_speed
        
        # Build camera matrices
        pitch_rad = math.radians(cam_pitch)
        dir_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
        dir_y = math.sin(pitch_rad)
        dir_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        
        projection = glm.perspective(glm.radians(60), WIDTH/HEIGHT, 0.1, 1000.0)
        cam_pos = glm.vec3(cam_x, cam_y, cam_z)
        target = cam_pos + glm.vec3(dir_x, dir_y, dir_z)
        view = glm.lookAt(cam_pos, target, glm.vec3(0, 1, 0))
        
        renderer.set_camera(projection, view)
        
        # Clear and render
        ctx.clear(0.4, 0.6, 0.8, 1.0)
        
        # Add all plant instances
        renderer.clear_instances()
        for plant in test_plants:
            renderer.add_instance(
                plant['mesh_type'],
                plant['x'], plant['y'], plant['z'],
                plant['scale'], plant['rotation'],
                plant['color'], plant['dna']
            )
        
        renderer.render()
        
        pygame.display.flip()
    
    pygame.quit()


if __name__ == '__main__':
    run_test_grid()

