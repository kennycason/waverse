#!/usr/bin/env python3
"""
DNA Flora Test Grid - Visualize DNA-driven plant variety

Run: python -m waverse.dna_flora_test
     python -m waverse.dna_flora_test --evolve
     python -m waverse.dna_flora_test --evolve --continuous

Shows a grid of plants with varying DNA parameters to test
the vertex shader deformation system. Goal: match V1 quality!

--evolve mode: Watch 4 populations grow from corners and merge with crossover!
--continuous: Plants die after ~15 generations, ecosystem constantly evolves
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


def create_branch_segment(start: Tuple[float, float, float], 
                          end: Tuple[float, float, float],
                          width: float) -> Tuple[List, List, List]:
    """Create a single branch segment as a tapered cylinder."""
    vertices = []
    normals = []
    colors = []
    
    branch_color = (0.45, 0.28, 0.14)
    segments = 4
    
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 0.001:
        return [], [], []
    
    # Create orthonormal basis
    dir_vec = (dx/length, dy/length, dz/length)
    
    # Find perpendicular vectors
    if abs(dir_vec[1]) < 0.9:
        perp1 = (-dir_vec[2], 0, dir_vec[0])
    else:
        perp1 = (1, 0, 0)
    p_len = math.sqrt(perp1[0]**2 + perp1[1]**2 + perp1[2]**2)
    if p_len > 0.001:
        perp1 = (perp1[0]/p_len, perp1[1]/p_len, perp1[2]/p_len)
    
    perp2 = (
        dir_vec[1]*perp1[2] - dir_vec[2]*perp1[1],
        dir_vec[2]*perp1[0] - dir_vec[0]*perp1[2],
        dir_vec[0]*perp1[1] - dir_vec[1]*perp1[0]
    )
    
    taper = 0.5  # End is half as wide
    
    for seg in range(segments):
        a1 = (seg / segments) * 2 * math.pi
        a2 = ((seg + 1) / segments) * 2 * math.pi
        
        # Start ring
        s1_x = start[0] + (math.cos(a1)*perp1[0] + math.sin(a1)*perp2[0]) * width
        s1_y = start[1] + (math.cos(a1)*perp1[1] + math.sin(a1)*perp2[1]) * width
        s1_z = start[2] + (math.cos(a1)*perp1[2] + math.sin(a1)*perp2[2]) * width
        
        s2_x = start[0] + (math.cos(a2)*perp1[0] + math.sin(a2)*perp2[0]) * width
        s2_y = start[1] + (math.cos(a2)*perp1[1] + math.sin(a2)*perp2[1]) * width
        s2_z = start[2] + (math.cos(a2)*perp1[2] + math.sin(a2)*perp2[2]) * width
        
        # End ring (tapered)
        e1_x = end[0] + (math.cos(a1)*perp1[0] + math.sin(a1)*perp2[0]) * width * taper
        e1_y = end[1] + (math.cos(a1)*perp1[1] + math.sin(a1)*perp2[1]) * width * taper
        e1_z = end[2] + (math.cos(a1)*perp1[2] + math.sin(a1)*perp2[2]) * width * taper
        
        e2_x = end[0] + (math.cos(a2)*perp1[0] + math.sin(a2)*perp2[0]) * width * taper
        e2_y = end[1] + (math.cos(a2)*perp1[1] + math.sin(a2)*perp2[1]) * width * taper
        e2_z = end[2] + (math.cos(a2)*perp1[2] + math.sin(a2)*perp2[2]) * width * taper
        
        n = (math.cos(a1)*perp1[0] + math.sin(a1)*perp2[0],
             math.cos(a1)*perp1[1] + math.sin(a1)*perp2[1],
             math.cos(a1)*perp1[2] + math.sin(a1)*perp2[2])
        
        vertices.extend([(s1_x, s1_y, s1_z), (s2_x, s2_y, s2_z), (e1_x, e1_y, e1_z)])
        vertices.extend([(s2_x, s2_y, s2_z), (e2_x, e2_y, e2_z), (e1_x, e1_y, e1_z)])
        normals.extend([n] * 6)
        colors.extend([branch_color] * 6)
    
    return vertices, normals, colors


def create_leaf_cluster(pos: Tuple[float, float, float], size: float, 
                        count: int = 5) -> Tuple[List, List, List]:
    """Create a cluster of leaves at a position."""
    vertices = []
    normals = []
    colors = []
    
    leaf_color = (1.0, 1.0, 1.0)  # Will be tinted by instance color
    
    for i in range(count):
        angle = (i / count) * 2 * math.pi + (i * 0.7)  # Golden angle-ish
        tilt = 0.3 + (i % 3) * 0.2
        
        dx = math.cos(angle) * math.cos(tilt) * size
        dy = math.sin(tilt) * size
        dz = math.sin(angle) * math.cos(tilt) * size
        
        # Diamond-shaped leaf
        vertices.extend([
            pos,
            (pos[0] + dx * 0.3, pos[1] + dy * 0.5, pos[2] + dz * 0.3),
            (pos[0] + dx, pos[1] + dy, pos[2] + dz)
        ])
        vertices.extend([
            pos,
            (pos[0] + dx, pos[1] + dy, pos[2] + dz),
            (pos[0] + dx * 0.3, pos[1] - dy * 0.2, pos[2] + dz * 0.3)
        ])
        
        n = (0, 0.8, 0.2)
        normals.extend([n] * 6)
        colors.extend([leaf_color] * 6)
    
    return vertices, normals, colors


def create_recursive_branches(trunk_height: float, count: int = 6, 
                              max_depth: int = 2, seed: int = 12345) -> Tuple[List, List, List]:
    """Create recursive branching structure like legacy PlantRenderer."""
    vertices = []
    normals = []
    colors = []
    
    rng = np.random.default_rng(seed)
    
    def add_branch(start: Tuple[float, float, float], 
                   direction: Tuple[float, float, float],
                   length: float, width: float, depth: int):
        """Recursively add a branch and its sub-branches."""
        if depth > max_depth or length < 0.02 or width < 0.003:
            return
        
        # Add some natural curve/randomness
        curve = (rng.random() - 0.5) * 0.4 * (1 + depth * 0.3)
        
        # Calculate end point
        end = (
            start[0] + direction[0] * length + curve * length * 0.5,
            start[1] + direction[1] * length - 0.02 * length,  # Slight droop
            start[2] + direction[2] * length
        )
        
        # Draw this branch
        v, n, c = create_branch_segment(start, end, width)
        vertices.extend(v)
        normals.extend(n)
        colors.extend(c)
        
        # Add leaves at branch tips and along branches
        if depth >= max_depth - 1 or rng.random() < 0.4:
            leaf_size = 0.08 * (1 + rng.random() * 0.5)
            v, n, c = create_leaf_cluster(end, leaf_size, count=3 + int(rng.random() * 3))
            vertices.extend(v)
            normals.extend(n)
            colors.extend(c)
        
        # Spawn sub-branches
        if depth < max_depth:
            sub_count = 1 + int(rng.random() * 2)
            for _ in range(sub_count):
                if rng.random() < 0.6:  # Sub-branch probability
                    sub_angle = rng.random() * 2 * math.pi
                    spread = 0.4 + rng.random() * 0.4
                    
                    # New direction
                    new_dir = (
                        direction[0] * 0.5 + math.cos(sub_angle) * spread,
                        direction[1] * 0.6 - 0.1,  # Droop
                        direction[2] * 0.5 + math.sin(sub_angle) * spread
                    )
                    d_len = math.sqrt(new_dir[0]**2 + new_dir[1]**2 + new_dir[2]**2)
                    if d_len > 0.001:
                        new_dir = (new_dir[0]/d_len, new_dir[1]/d_len, new_dir[2]/d_len)
                    
                    add_branch(
                        end,
                        new_dir,
                        length * (0.5 + rng.random() * 0.3),
                        width * 0.6,
                        depth + 1
                    )
    
    # Create main branches from trunk
    for b in range(count):
        angle = (b / count) * 2 * math.pi + rng.random() * 0.3
        branch_y = trunk_height * (0.4 + (b / count) * 0.4)
        
        # Direction outward and slightly up
        spread_angle = 0.3 + rng.random() * 0.4
        direction = (
            math.cos(angle) * math.cos(spread_angle),
            math.sin(spread_angle) * 0.3,
            math.sin(angle) * math.cos(spread_angle)
        )
        
        start = (math.cos(angle) * 0.05, branch_y, math.sin(angle) * 0.05)
        add_branch(start, direction, 0.25 + rng.random() * 0.15, 0.02, 0)
    
    return vertices, normals, colors


def create_branches(trunk_height: float, count: int = 6, length: float = 0.3) -> Tuple[List, List, List]:
    """Create branches extending from trunk (simple version for basic meshes)."""
    return create_recursive_branches(trunk_height, count, max_depth=2)


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


def create_palm_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a palm tree - tall trunk with fronds at top."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    # Tall thin trunk
    v, n, c = create_trunk_mesh(height=0.8, radius=0.04)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Palm fronds at top (like fern but higher)
    foliage_color = (1.0, 1.0, 1.0)
    frond_count = 7
    frond_length = 0.4
    base_y = 0.75
    
    for f in range(frond_count):
        angle = (f / frond_count) * 2 * math.pi
        
        # Frond curves outward and droops
        mid_x = math.cos(angle) * frond_length * 0.4
        mid_z = math.sin(angle) * frond_length * 0.4
        mid_y = base_y + 0.15
        
        end_x = math.cos(angle) * frond_length
        end_z = math.sin(angle) * frond_length
        end_y = base_y - 0.1  # Droops down
        
        perp_x = -math.sin(angle) * 0.06
        perp_z = math.cos(angle) * 0.06
        
        all_verts.extend([
            (0, base_y, 0),
            (mid_x + perp_x, mid_y, mid_z + perp_z),
            (mid_x - perp_x, mid_y, mid_z - perp_z)
        ])
        all_norms.extend([(0, 1, 0)] * 3)
        all_cols.extend([foliage_color] * 3)
        
        all_verts.extend([
            (mid_x, mid_y, mid_z),
            (end_x + perp_x, end_y, end_z + perp_z),
            (end_x - perp_x, end_y, end_z - perp_z)
        ])
        all_norms.extend([(0, 0.5, 0.5)] * 3)
        all_cols.extend([foliage_color] * 3)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_willow_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a weeping willow - drooping branches."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    # Short thick trunk
    v, n, c = create_trunk_mesh(height=0.4, radius=0.08)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Cascading layers
    foliage_color = (1.0, 1.0, 1.0)
    layers = 4
    segments = 12
    
    for layer in range(layers):
        layer_y = 0.35 + layer * 0.12
        layer_r = 0.35 - layer * 0.05
        droop = 0.15 + layer * 0.08
        
        for seg in range(segments):
            angle1 = (seg / segments) * 2 * math.pi
            angle2 = ((seg + 1) / segments) * 2 * math.pi
            
            x1 = math.cos(angle1) * layer_r
            z1 = math.sin(angle1) * layer_r
            x2 = math.cos(angle2) * layer_r
            z2 = math.sin(angle2) * layer_r
            
            # Triangle drooping down
            all_verts.extend([
                (0, layer_y, 0),
                (x1, layer_y - droop, z1),
                (x2, layer_y - droop, z2)
            ])
            all_norms.extend([(0, 0.3, 0.7)] * 3)
            all_cols.extend([foliage_color] * 3)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_spiral_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a spiral plant - twisted upward growth."""
    vertices = []
    normals = []
    colors = []
    
    foliage_color = (1.0, 1.0, 1.0)
    turns = 3
    segments = 24
    height = 0.8
    radius = 0.15
    
    for i in range(segments):
        t = i / segments
        next_t = (i + 1) / segments
        
        # Spiral path
        angle = t * turns * 2 * math.pi
        next_angle = next_t * turns * 2 * math.pi
        
        x1 = math.cos(angle) * radius * (1 - t * 0.5)
        z1 = math.sin(angle) * radius * (1 - t * 0.5)
        y1 = t * height
        
        x2 = math.cos(next_angle) * radius * (1 - next_t * 0.5)
        z2 = math.sin(next_angle) * radius * (1 - next_t * 0.5)
        y2 = next_t * height
        
        # Triangle strip
        vertices.extend([
            (0, y1, 0), (x1, y1, z1), (x2, y2, z2)
        ])
        normals.extend([(math.cos(angle), 0.5, math.sin(angle))] * 3)
        colors.extend([foliage_color] * 3)
    
    return (
        np.array(vertices, dtype='f4'),
        np.array(normals, dtype='f4'),
        np.array(colors, dtype='f4')
    )


def create_grass_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create grass - simple blades."""
    vertices = []
    normals = []
    colors = []
    
    foliage_color = (1.0, 1.0, 1.0)
    blade_count = 5
    
    for b in range(blade_count):
        angle = (b / blade_count) * 2 * math.pi + 0.3
        offset = 0.03
        
        base_x = math.cos(angle) * offset
        base_z = math.sin(angle) * offset
        
        # Blade curves outward
        tip_x = math.cos(angle) * 0.1
        tip_z = math.sin(angle) * 0.1
        tip_y = 0.25 + (b % 3) * 0.05
        
        # Simple triangle blade
        vertices.extend([
            (base_x - 0.01, 0, base_z),
            (base_x + 0.01, 0, base_z),
            (tip_x, tip_y, tip_z)
        ])
        normals.extend([(0, 0.5, 0.5)] * 3)
        colors.extend([foliage_color] * 3)
    
    return (
        np.array(vertices, dtype='f4'),
        np.array(normals, dtype='f4'),
        np.array(colors, dtype='f4')
    )


def create_cactus_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a cactus - thick column with arms."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    cactus_color = (0.3, 0.6, 0.3)  # Green
    
    # Main column
    segments = 8
    height = 0.6
    radius = 0.08
    
    for seg in range(segments):
        angle1 = (seg / segments) * 2 * math.pi
        angle2 = ((seg + 1) / segments) * 2 * math.pi
        
        x1 = math.cos(angle1) * radius
        z1 = math.sin(angle1) * radius
        x2 = math.cos(angle2) * radius
        z2 = math.sin(angle2) * radius
        
        all_verts.extend([(x1, 0, z1), (x2, 0, z2), (x1, height, z1)])
        all_norms.extend([(math.cos(angle1), 0, math.sin(angle1))] * 3)
        all_cols.extend([cactus_color] * 3)
        
        all_verts.extend([(x2, 0, z2), (x2, height, z2), (x1, height, z1)])
        all_norms.extend([(math.cos(angle2), 0, math.sin(angle2))] * 3)
        all_cols.extend([cactus_color] * 3)
    
    # Top cap
    for seg in range(segments):
        angle1 = (seg / segments) * 2 * math.pi
        angle2 = ((seg + 1) / segments) * 2 * math.pi
        
        x1 = math.cos(angle1) * radius
        z1 = math.sin(angle1) * radius
        x2 = math.cos(angle2) * radius
        z2 = math.sin(angle2) * radius
        
        all_verts.extend([(x1, height, z1), (x2, height, z2), (0, height + 0.05, 0)])
        all_norms.extend([(0, 1, 0)] * 3)
        all_cols.extend([cactus_color] * 3)
    
    # Side arm
    arm_base_y = height * 0.4
    arm_length = 0.15
    arm_radius = 0.04
    
    for seg in range(6):
        angle1 = (seg / 6) * 2 * math.pi
        angle2 = ((seg + 1) / 6) * 2 * math.pi
        
        # Arm going right
        x1 = radius + math.sin(angle1) * arm_radius
        z1 = math.cos(angle1) * arm_radius
        x2 = radius + math.sin(angle2) * arm_radius
        z2 = math.cos(angle2) * arm_radius
        
        all_verts.extend([
            (x1, arm_base_y, z1),
            (x2, arm_base_y, z2),
            (radius + arm_length, arm_base_y + 0.15, 0)
        ])
        all_norms.extend([(1, 0.5, 0)] * 3)
        all_cols.extend([cactus_color] * 3)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_flower_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a flower - stem with petals."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    stem_color = (0.2, 0.5, 0.2)
    petal_color = (1.0, 1.0, 1.0)  # Tinted by instance
    center_color = (0.9, 0.7, 0.2)  # Yellow center
    
    # Thin stem
    stem_height = 0.3
    stem_radius = 0.01
    segments = 6
    
    for seg in range(segments):
        angle1 = (seg / segments) * 2 * math.pi
        angle2 = ((seg + 1) / segments) * 2 * math.pi
        
        x1 = math.cos(angle1) * stem_radius
        z1 = math.sin(angle1) * stem_radius
        x2 = math.cos(angle2) * stem_radius
        z2 = math.sin(angle2) * stem_radius
        
        all_verts.extend([(x1, 0, z1), (x2, 0, z2), (x1, stem_height, z1)])
        all_norms.extend([(0, 1, 0)] * 3)
        all_cols.extend([stem_color] * 3)
    
    # Petals
    petal_count = 6
    petal_length = 0.08
    
    for p in range(petal_count):
        angle = (p / petal_count) * 2 * math.pi
        
        px = math.cos(angle) * petal_length
        pz = math.sin(angle) * petal_length
        
        all_verts.extend([
            (0, stem_height, 0),
            (px, stem_height + 0.02, pz),
            (px * 0.5, stem_height + 0.04, pz * 0.5)
        ])
        all_norms.extend([(0, 1, 0)] * 3)
        all_cols.extend([petal_color] * 3)
    
    # Center
    for seg in range(segments):
        angle1 = (seg / segments) * 2 * math.pi
        angle2 = ((seg + 1) / segments) * 2 * math.pi
        
        x1 = math.cos(angle1) * 0.02
        z1 = math.sin(angle1) * 0.02
        x2 = math.cos(angle2) * 0.02
        z2 = math.sin(angle2) * 0.02
        
        all_verts.extend([(x1, stem_height, z1), (x2, stem_height, z2), (0, stem_height + 0.03, 0)])
        all_norms.extend([(0, 1, 0)] * 3)
        all_cols.extend([center_color] * 3)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_oak_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a majestic oak - thick trunk with massive rounded canopy."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    # Thick gnarly trunk
    v, n, c = create_trunk_mesh(height=0.5, radius=0.12)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Many branches with dense leaves
    v, n, c = create_recursive_branches(trunk_height=0.5, count=10, max_depth=3, seed=7777)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Massive dome canopy - multiple overlapping spheres
    foliage_color = (1.0, 1.0, 1.0)
    
    # Main central dome
    v, n, c = create_dome_canopy(base_y=0.4, height=0.6, radius=0.6)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Extra side lobes for that oak look
    for offset_angle in [0, 1.5, 3.0, 4.5]:
        ox = math.cos(offset_angle) * 0.25
        oz = math.sin(offset_angle) * 0.25
        # Create smaller satellite domes
        rings = 4
        segments = 8
        lobe_r = 0.3
        lobe_h = 0.35
        base_y = 0.45
        
        for ring in range(rings):
            phi1 = (ring / rings) * (math.pi / 2)
            phi2 = ((ring + 1) / rings) * (math.pi / 2)
            y1 = base_y + math.sin(phi1) * lobe_h
            r1 = math.cos(phi1) * lobe_r
            y2 = base_y + math.sin(phi2) * lobe_h
            r2 = math.cos(phi2) * lobe_r
            
            for seg in range(segments):
                theta1 = (seg / segments) * 2 * math.pi
                theta2 = ((seg + 1) / segments) * 2 * math.pi
                
                x1a = ox + math.cos(theta1) * r1
                z1a = oz + math.sin(theta1) * r1
                x2a = ox + math.cos(theta2) * r1
                z2a = oz + math.sin(theta2) * r1
                x1b = ox + math.cos(theta1) * r2
                z1b = oz + math.sin(theta1) * r2
                x2b = ox + math.cos(theta2) * r2
                z2b = oz + math.sin(theta2) * r2
                
                n = (math.cos(theta1), 0.5, math.sin(theta1))
                all_verts.extend([(x1a, y1, z1a), (x2a, y1, z2a), (x1b, y2, z1b)])
                all_norms.extend([n] * 3)
                all_cols.extend([foliage_color] * 3)
                all_verts.extend([(x2a, y1, z2a), (x2b, y2, z2b), (x1b, y2, z1b)])
                all_norms.extend([n] * 3)
                all_cols.extend([foliage_color] * 3)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_birch_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a birch - thin white trunk with delicate canopy."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    # Thin white trunk (color will be applied, but we mark it distinctly)
    trunk_color = (0.9, 0.88, 0.8)  # Birch white
    segments = 6
    rings = 8
    height = 0.7
    radius = 0.03  # Very thin
    
    for ring in range(rings):
        t1 = ring / rings
        t2 = (ring + 1) / rings
        y1 = t1 * height
        y2 = t2 * height
        
        for seg in range(segments):
            a1 = (seg / segments) * 2 * math.pi
            a2 = ((seg + 1) / segments) * 2 * math.pi
            
            x1 = math.cos(a1) * radius
            z1 = math.sin(a1) * radius
            x2 = math.cos(a2) * radius
            z2 = math.sin(a2) * radius
            
            # Add characteristic birch "marks" via slight color variation
            mark = 0.85 + 0.1 * math.sin(ring * 3)
            col = (trunk_color[0] * mark, trunk_color[1] * mark, trunk_color[2] * mark)
            
            n = (math.cos(a1), 0, math.sin(a1))
            all_verts.extend([(x1, y1, z1), (x2, y1, z2), (x1, y2, z1)])
            all_norms.extend([n] * 3)
            all_cols.extend([col] * 3)
            all_verts.extend([(x2, y1, z2), (x2, y2, z2), (x1, y2, z1)])
            all_norms.extend([n] * 3)
            all_cols.extend([col] * 3)
    
    # Delicate feathery branches
    v, n, c = create_recursive_branches(trunk_height=0.7, count=6, max_depth=2, seed=2222)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Light, airy canopy
    v, n, c = create_dome_canopy(base_y=0.5, height=0.35, radius=0.35)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_baobab_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a baobab - massive thick trunk with small canopy."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    # MASSIVE trunk - the signature of baobab
    trunk_color = (0.55, 0.4, 0.3)
    segments = 10
    rings = 6
    height = 0.6
    base_radius = 0.25  # Very thick at base
    top_radius = 0.12   # Narrows at top
    
    for ring in range(rings):
        t1 = ring / rings
        t2 = (ring + 1) / rings
        y1 = t1 * height
        y2 = t2 * height
        r1 = base_radius * (1 - t1 * 0.5)  # Tapers
        r2 = base_radius * (1 - t2 * 0.5)
        
        # Add bulge in the middle (characteristic baobab shape)
        bulge1 = 1.0 + 0.15 * math.sin(t1 * math.pi)
        bulge2 = 1.0 + 0.15 * math.sin(t2 * math.pi)
        r1 *= bulge1
        r2 *= bulge2
        
        for seg in range(segments):
            a1 = (seg / segments) * 2 * math.pi
            a2 = ((seg + 1) / segments) * 2 * math.pi
            
            x1a = math.cos(a1) * r1
            z1a = math.sin(a1) * r1
            x2a = math.cos(a2) * r1
            z2a = math.sin(a2) * r1
            x1b = math.cos(a1) * r2
            z1b = math.sin(a1) * r2
            x2b = math.cos(a2) * r2
            z2b = math.sin(a2) * r2
            
            n = (math.cos(a1), 0, math.sin(a1))
            all_verts.extend([(x1a, y1, z1a), (x2a, y1, z2a), (x1b, y2, z1b)])
            all_norms.extend([n] * 3)
            all_cols.extend([trunk_color] * 3)
            all_verts.extend([(x2a, y1, z2a), (x2b, y2, z2b), (x1b, y2, z1b)])
            all_norms.extend([n] * 3)
            all_cols.extend([trunk_color] * 3)
    
    # Stubby branches reaching up
    branch_color = (0.5, 0.35, 0.25)
    for b in range(6):
        angle = (b / 6) * 2 * math.pi
        bx = math.cos(angle) * 0.15
        bz = math.sin(angle) * 0.15
        
        # Short thick branch
        for seg in range(4):
            a = (seg / 4) * 2 * math.pi
            r = 0.03
            all_verts.extend([
                (bx, height, bz),
                (bx + math.cos(a) * r, height + 0.12, bz + math.sin(a) * r),
                (bx + math.cos(angle) * 0.1, height + 0.15, bz + math.sin(angle) * 0.1)
            ])
            all_norms.extend([(0, 1, 0)] * 3)
            all_cols.extend([branch_color] * 3)
    
    # Small canopy (baobabs are sparse on top)
    foliage_color = (1.0, 1.0, 1.0)
    v, n, c = create_dome_canopy(base_y=height + 0.1, height=0.2, radius=0.25)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_pine_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a tall pine - straight trunk with layered cone branches."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    # Tall straight trunk
    v, n, c = create_trunk_mesh(height=0.7, radius=0.05)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Layered cone canopy (classic pine shape)
    foliage_color = (1.0, 1.0, 1.0)
    layers = 5
    segments = 10
    
    for layer in range(layers):
        layer_t = layer / layers
        base_y = 0.25 + layer_t * 0.65
        layer_r = 0.4 * (1 - layer_t * 0.8)  # Shrinks toward top
        tip_y = base_y + 0.15
        
        for seg in range(segments):
            a1 = (seg / segments) * 2 * math.pi
            a2 = ((seg + 1) / segments) * 2 * math.pi
            
            x1 = math.cos(a1) * layer_r
            z1 = math.sin(a1) * layer_r
            x2 = math.cos(a2) * layer_r
            z2 = math.sin(a2) * layer_r
            
            # Cone layer pointing up
            all_verts.extend([(x1, base_y, z1), (x2, base_y, z2), (0, tip_y, 0)])
            n = (0, 0.6, 0.4)
            all_norms.extend([n] * 3)
            all_cols.extend([foliage_color] * 3)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_cypress_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a cypress - tall narrow columnar shape."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    foliage_color = (1.0, 1.0, 1.0)
    height = 1.0
    segments = 10
    rings = 12
    
    # Columnar shape - narrow at base and top, slight bulge in middle
    for ring in range(rings):
        t1 = ring / rings
        t2 = (ring + 1) / rings
        y1 = t1 * height
        y2 = t2 * height
        
        # Slight bulge profile
        r1 = 0.1 + 0.05 * math.sin(t1 * math.pi)
        r2 = 0.1 + 0.05 * math.sin(t2 * math.pi)
        
        for seg in range(segments):
            a1 = (seg / segments) * 2 * math.pi
            a2 = ((seg + 1) / segments) * 2 * math.pi
            
            x1a = math.cos(a1) * r1
            z1a = math.sin(a1) * r1
            x2a = math.cos(a2) * r1
            z2a = math.sin(a2) * r1
            x1b = math.cos(a1) * r2
            z1b = math.sin(a1) * r2
            x2b = math.cos(a2) * r2
            z2b = math.sin(a2) * r2
            
            n = (math.cos(a1), 0, math.sin(a1))
            all_verts.extend([(x1a, y1, z1a), (x2a, y1, z2a), (x1b, y2, z1b)])
            all_norms.extend([n] * 3)
            all_cols.extend([foliage_color] * 3)
            all_verts.extend([(x2a, y1, z2a), (x2b, y2, z2b), (x1b, y2, z1b)])
            all_norms.extend([n] * 3)
            all_cols.extend([foliage_color] * 3)
    
    # Pointed top
    for seg in range(segments):
        a1 = (seg / segments) * 2 * math.pi
        a2 = ((seg + 1) / segments) * 2 * math.pi
        
        x1 = math.cos(a1) * 0.08
        z1 = math.sin(a1) * 0.08
        x2 = math.cos(a2) * 0.08
        z2 = math.sin(a2) * 0.08
        
        all_verts.extend([(x1, height, z1), (x2, height, z2), (0, height + 0.15, 0)])
        all_norms.extend([(0, 1, 0)] * 3)
        all_cols.extend([foliage_color] * 3)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_maple_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a maple - horizontal layered branches with full canopy."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    # Medium trunk
    v, n, c = create_trunk_mesh(height=0.45, radius=0.07)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Horizontal branches spreading out
    v, n, c = create_recursive_branches(trunk_height=0.45, count=8, max_depth=2, seed=5555)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    # Full rounded canopy with horizontal layers
    foliage_color = (1.0, 1.0, 1.0)
    layers = 3
    
    for layer in range(layers):
        layer_y = 0.35 + layer * 0.15
        layer_r = 0.45 - layer * 0.1
        
        # Each layer is a flat disc with scalloped edges
        v, n, c = create_umbrella_canopy(base_y=layer_y, height=0.1, radius=layer_r)
        all_verts.extend(v)
        all_norms.extend(n)
        all_cols.extend(c)
    
    # Top dome
    v, n, c = create_dome_canopy(base_y=0.6, height=0.3, radius=0.35)
    all_verts.extend(v)
    all_norms.extend(n)
    all_cols.extend(c)
    
    return (
        np.array(all_verts, dtype='f4'),
        np.array(all_norms, dtype='f4'),
        np.array(all_cols, dtype='f4')
    )


def create_bonsai_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a bonsai - twisted artistic trunk with cloud-like canopy puffs."""
    all_verts = []
    all_norms = []
    all_cols = []
    
    trunk_color = (0.4, 0.28, 0.18)
    foliage_color = (1.0, 1.0, 1.0)
    
    # Twisted trunk (S-curve)
    segments = 6
    rings = 10
    
    for ring in range(rings):
        t1 = ring / rings
        t2 = (ring + 1) / rings
        y1 = t1 * 0.4
        y2 = t2 * 0.4
        
        # S-curve offset
        curve1 = math.sin(t1 * math.pi * 2) * 0.08
        curve2 = math.sin(t2 * math.pi * 2) * 0.08
        r1 = 0.04 * (1 - t1 * 0.5)
        r2 = 0.04 * (1 - t2 * 0.5)
        
        for seg in range(segments):
            a1 = (seg / segments) * 2 * math.pi
            a2 = ((seg + 1) / segments) * 2 * math.pi
            
            x1a = curve1 + math.cos(a1) * r1
            z1a = math.sin(a1) * r1
            x2a = curve1 + math.cos(a2) * r1
            z2a = math.sin(a2) * r1
            x1b = curve2 + math.cos(a1) * r2
            z1b = math.sin(a1) * r2
            x2b = curve2 + math.cos(a2) * r2
            z2b = math.sin(a2) * r2
            
            n = (math.cos(a1), 0, math.sin(a1))
            all_verts.extend([(x1a, y1, z1a), (x2a, y1, z2a), (x1b, y2, z1b)])
            all_norms.extend([n] * 3)
            all_cols.extend([trunk_color] * 3)
            all_verts.extend([(x2a, y1, z2a), (x2b, y2, z2b), (x1b, y2, z1b)])
            all_norms.extend([n] * 3)
            all_cols.extend([trunk_color] * 3)
    
    # Cloud puffs - small domes at branch tips
    puff_positions = [
        (0.15, 0.45, 0.0),
        (-0.1, 0.5, 0.08),
        (0.0, 0.55, -0.1),
        (0.08, 0.4, 0.1),
    ]
    
    for px, py, pz in puff_positions:
        rings = 4
        segs = 6
        puff_r = 0.08
        puff_h = 0.06
        
        for ring in range(rings):
            phi1 = (ring / rings) * (math.pi / 2)
            phi2 = ((ring + 1) / rings) * (math.pi / 2)
            y1 = py + math.sin(phi1) * puff_h
            r1 = math.cos(phi1) * puff_r
            y2 = py + math.sin(phi2) * puff_h
            r2 = math.cos(phi2) * puff_r
            
            for seg in range(segs):
                theta1 = (seg / segs) * 2 * math.pi
                theta2 = ((seg + 1) / segs) * 2 * math.pi
                
                x1a = px + math.cos(theta1) * r1
                z1a = pz + math.sin(theta1) * r1
                x2a = px + math.cos(theta2) * r1
                z2a = pz + math.sin(theta2) * r1
                x1b = px + math.cos(theta1) * r2
                z1b = pz + math.sin(theta1) * r2
                x2b = px + math.cos(theta2) * r2
                z2b = pz + math.sin(theta2) * r2
                
                n = (math.cos(theta1), 0.5, math.sin(theta1))
                all_verts.extend([(x1a, y1, z1a), (x2a, y1, z2a), (x1b, y2, z1b)])
                all_norms.extend([n] * 3)
                all_cols.extend([foliage_color] * 3)
                all_verts.extend([(x2a, y1, z2a), (x2b, y2, z2b), (x1b, y2, z1b)])
                all_norms.extend([n] * 3)
                all_cols.extend([foliage_color] * 3)
    
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
            # Classic trees
            'tree_cone': create_tree_base_mesh(),
            'tree_dome': create_dome_tree_mesh(),
            'tree_umbrella': create_umbrella_tree_mesh(),
            # Full trees with dense canopies
            'oak': create_oak_mesh(),
            'birch': create_birch_mesh(),
            'maple': create_maple_mesh(),
            'pine': create_pine_mesh(),
            'cypress': create_cypress_mesh(),
            'baobab': create_baobab_mesh(),
            'bonsai': create_bonsai_mesh(),
            # Other plants
            'bush': create_bush_base_mesh(),
            'fern': create_fern_mesh(),
            'mushroom': create_mushroom_mesh(),
            'palm': create_palm_mesh(),
            'willow': create_willow_mesh(),
            'spiral': create_spiral_mesh(),
            'grass': create_grass_mesh(),
            'cactus': create_cactus_mesh(),
            'flower': create_flower_mesh(),
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
    
    # Camera state - start further back and higher to see the whole grid
    cam_x, cam_y, cam_z = 0, 40, 80
    cam_yaw, cam_pitch = 0, -20
    
    # FPS tracking
    fps_history = []
    last_fps_print = 0
    
    # Generate test DNA grid with variety of plant types
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    GRID_SIZE = 32  # 32x32 grid = 1024 plants!
    SPACING = 4.0   # 4 units between plants (more breathing room)
    
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
    
    # Mesh types to cycle through - all 12 types!
    mesh_types = [
        'tree_cone', 'tree_dome', 'tree_umbrella', 
        'bush', 'fern', 'mushroom',
        'palm', 'willow', 'spiral',
        'grass', 'cactus', 'flower'
    ]
    
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
    print("Controls:")
    print("  MOVEMENT (Left Stick):")
    print("    WASD - Move (follows camera direction)")
    print("    Space/H - Fly UP")
    print("    Shift/F - Fly DOWN")
    print("  CAMERA (Right Stick):")
    print("    I/K - Look UP/DOWN (do loops!)")
    print("    J/L - Look LEFT/RIGHT")
    print("  OTHER:")
    print("    P - Save screenshot")
    print("    Q/Escape - Quit")
    
    # Screenshot counter
    screenshot_count = 0
    
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
                elif event.key == K_p:
                    # Take screenshot from OpenGL framebuffer
                    import os
                    from datetime import datetime
                    from PIL import Image
                    
                    screenshot_dir = os.path.expanduser("~/.waverse/screenshots")
                    os.makedirs(screenshot_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{screenshot_dir}/flora_test_{timestamp}.png"
                    
                    # Read pixels from OpenGL framebuffer
                    pixels = ctx.fbo.read(components=3)
                    img = Image.frombytes('RGB', (WIDTH, HEIGHT), pixels)
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL is upside down
                    img.save(filename)
                    
                    screenshot_count += 1
                    print(f"Saved: {filename}")
        
        # Keyboard input
        keys = pygame.key.get_pressed()
        
        # Ignore all input when system modifiers are held (for screenshots: Cmd+Shift+Ctrl+4)
        mods = pygame.key.get_mods()
        system_mod_held = (mods & pygame.KMOD_META) or (mods & pygame.KMOD_CTRL)
        
        if not system_mod_held:
            # === RIGHT STICK: Camera Look (IJKL) ===
            # I = look up, K = look down, J = look left, L = look right
            look_speed = dt * 1.2
            
            if keys[K_i]: cam_pitch += look_speed * 80   # Look UP
            if keys[K_k]: cam_pitch -= look_speed * 80   # Look DOWN
            if keys[K_j]: cam_yaw += look_speed * 100    # Look LEFT
            if keys[K_l]: cam_yaw -= look_speed * 100    # Look RIGHT
            
            # Allow full 360° loops - normalize to -180 to 180
            while cam_pitch > 180:
                cam_pitch -= 360
            while cam_pitch < -180:
                cam_pitch += 360
            
            # === LEFT STICK: Movement (WASD) ===
            move_speed = 15 * dt
            yaw_rad = math.radians(cam_yaw)
            pitch_rad = math.radians(cam_pitch)
            
            # Full 3D movement following camera direction
            forward_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
            forward_y = math.sin(pitch_rad)
            forward_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
            
            # Right vector stays horizontal
            right_x = math.cos(yaw_rad)
            right_z = -math.sin(yaw_rad)
            
            if keys[K_w]:
                cam_x += forward_x * move_speed
                cam_y += forward_y * move_speed
                cam_z += forward_z * move_speed
            if keys[K_s]:
                cam_x -= forward_x * move_speed
                cam_y -= forward_y * move_speed
                cam_z -= forward_z * move_speed
            if keys[K_a]:
                cam_x -= right_x * move_speed
                cam_z -= right_z * move_speed
            if keys[K_d]:
                cam_x += right_x * move_speed
                cam_z += right_z * move_speed
            
            # Vertical flight (H = up, F = down, or Space/Shift)
            if keys[K_SPACE] or keys[K_h]:
                cam_y += move_speed
            if keys[K_LSHIFT] or keys[K_f]:
                cam_y -= move_speed
        
        # Build camera matrices
        yaw_rad = math.radians(cam_yaw)
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
        
        # FPS tracking
        fps = clock.get_fps()
        fps_history.append(fps)
        if len(fps_history) > 60:
            fps_history.pop(0)
        
        import time
        now = time.time()
        if now - last_fps_print > 2.0:  # Print every 2 seconds
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            mesh_count = sum(len(instances) for instances in renderer.instances.values())
            print(f"FPS: {avg_fps:.1f} | Plants: {mesh_count} | Draw calls: {len(renderer.meshes)}")
            last_fps_print = now
    
    pygame.quit()


def run_evolution_mode():
    """
    Evolution mode: Watch FOUR populations grow and merge!
    
    - Four populations start in four corners
    - Each generation, plants spread and mutate
    - When populations meet, crossover occurs creating hybrids
    """
    pygame.init()
    
    WIDTH, HEIGHT = 1280, 720
    
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
    
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("DNA Flora Evolution - 4 Populations Converging!")
    
    ctx = moderngl.create_context(require=410)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.CULL_FACE)
    
    renderer = DNAFloraRenderer(ctx)
    
    # Camera - start higher to see all 4 corners
    cam_x, cam_y, cam_z = 0, 60, 80
    cam_yaw, cam_pitch = 0, -35
    
    # Evolution state
    rng = np.random.default_rng(42)
    
    # Four corners for four populations
    WORLD_SIZE = 60  # World is -60 to +60
    
    # Define 4 distinct populations with unique colors, shapes, and traits
    populations = {
        'A': {
            'center': (-40, -40),  # Bottom-left
            'color': (0.2, 0.7, 0.9),  # Cyan/teal
            'mesh_type': 'oak',
            'height': 1.8,
            'curve': 0.2,
            'twist': 0.0,
            'seed': 1111,
        },
        'B': {
            'center': (40, -40),   # Bottom-right
            'color': (0.9, 0.3, 0.2),  # Red/orange
            'mesh_type': 'pine',
            'height': 2.2,
            'curve': 0.0,
            'twist': 0.3,
            'seed': 2222,
        },
        'C': {
            'center': (-40, 40),   # Top-left
            'color': (0.9, 0.8, 0.2),  # Yellow/gold
            'mesh_type': 'willow',
            'height': 1.5,
            'curve': 0.4,
            'twist': 0.1,
            'seed': 3333,
        },
        'D': {
            'center': (40, 40),    # Top-right
            'color': (0.6, 0.2, 0.8),  # Purple/violet
            'mesh_type': 'baobab',
            'height': 1.2,
            'curve': 0.1,
            'twist': 0.5,
            'seed': 4444,
        },
    }
    
    # Create DNA templates for each population
    pop_dnas = {}
    for pop_name, pop_info in populations.items():
        dna = PlantDNA.create_random(PlantType.TREE, pop_info['seed'])
        dna.leaf_color.r = pop_info['color'][0]
        dna.leaf_color.g = pop_info['color'][1]
        dna.leaf_color.b = pop_info['color'][2]
        dna.height_gene.value = pop_info['height']
        if dna.trunk_segments:
            dna.trunk_segments[0].curve = pop_info['curve']
            dna.trunk_segments[0].twist = pop_info['twist']
        pop_dnas[pop_name] = dna
    
    # Living plants
    plants = []
    
    # Seed initial populations (3x3 tiles each, grid-spaced)
    INIT_SPACING = 5.0
    for pop_name, pop_info in populations.items():
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                x = pop_info['center'][0] + dx * INIT_SPACING
                z = pop_info['center'][1] + dz * INIT_SPACING
                dna = pop_dnas[pop_name].mutate(rng, strength=0.1)
                plants.append({
                    'x': x, 'z': z, 'y': 0,
                    'dna': dna,
                    'mesh_type': pop_info['mesh_type'],
                    'generation': 0,
                    'population': pop_name,
                    'scale': 2.0 + rng.random() * 0.5,
                    'rotation': rng.random() * math.pi * 2,
                })
    
    # Evolution timing
    generation = 0
    time_since_generation = 0.0
    GENERATION_TIME = 1.5  # Faster generations to see convergence
    MAX_PLANTS = 3000
    
    # Continuous mode settings
    continuous_mode = '--continuous' in sys.argv
    PLANT_LIFESPAN = 15  # Generations before a plant can die
    DEATH_CHANCE = 0.15  # Chance of dying after lifespan
    
    print("=" * 60)
    if continuous_mode:
        print("EVOLUTION MODE - 4 POPULATIONS [CONTINUOUS]")
    else:
        print("EVOLUTION MODE - 4 POPULATIONS")
    print("=" * 60)
    print("A (Cyan/Teal):   Bottom-left  - Oak trees")
    print("B (Red/Orange):  Bottom-right - Pine trees")  
    print("C (Yellow/Gold): Top-left     - Willow trees")
    print("D (Purple):      Top-right    - Baobab trees")
    print("")
    if continuous_mode:
        print("CONTINUOUS: Plants die after ~15 generations")
        print("Watch the ecosystem constantly evolve and shift!")
    else:
        print("Watch them grow toward center and CROSSOVER!")
    print("=" * 60)
    print("Controls:")
    print("  WASD - Move (follows camera)   IJKL - Look (do loops!)")
    print("  Space/H - Up   Shift/F - Down   P - Screenshot   Q - Quit")
    print("  LEFT CLICK - Save plant DNA to seed folder!")
    print("=" * 60)
    
    clock = pygame.time.Clock()
    running = True
    fps_history = []
    last_fps_print = 0
    screenshot_count = 0
    selected_plant = None  # For highlighting
    
    # DNA seed folder
    import os
    import json
    seed_dir = os.path.expanduser("~/.waverse/dna_seeds")
    os.makedirs(seed_dir, exist_ok=True)
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click - select and save plant
                    mouse_x, mouse_y = event.pos
                    
                    # Convert mouse position to normalized device coords
                    ndc_x = (2.0 * mouse_x / WIDTH) - 1.0
                    ndc_y = 1.0 - (2.0 * mouse_y / HEIGHT)
                    
                    # Build ray from camera
                    yaw_r = math.radians(cam_yaw)
                    pitch_r = math.radians(cam_pitch)
                    
                    # Camera forward direction
                    cam_fwd = glm.vec3(
                        -math.sin(yaw_r) * math.cos(pitch_r),
                        math.sin(pitch_r),
                        -math.cos(yaw_r) * math.cos(pitch_r)
                    )
                    cam_right = glm.normalize(glm.cross(cam_fwd, glm.vec3(0, 1, 0)))
                    cam_up = glm.cross(cam_right, cam_fwd)
                    
                    # Ray direction (simple approximation)
                    fov_factor = math.tan(math.radians(30))  # Half of 60 degree FOV
                    aspect = WIDTH / HEIGHT
                    ray_dir = glm.normalize(
                        cam_fwd + cam_right * ndc_x * fov_factor * aspect + cam_up * ndc_y * fov_factor
                    )
                    
                    # Find nearest plant to the ray
                    best_plant = None
                    best_dist = float('inf')
                    cam_pos = glm.vec3(cam_x, cam_y, cam_z)
                    
                    for plant in plants:
                        plant_pos = glm.vec3(plant['x'], plant['y'] + plant['scale'], plant['z'])
                        to_plant = plant_pos - cam_pos
                        
                        # Project onto ray
                        t = glm.dot(to_plant, ray_dir)
                        if t > 0:  # Plant is in front
                            closest_on_ray = cam_pos + ray_dir * t
                            dist_to_ray = glm.length(plant_pos - closest_on_ray)
                            dist_from_cam = glm.length(to_plant)
                            
                            # Scale threshold by distance (further = harder to click)
                            threshold = 3.0 + dist_from_cam * 0.05
                            
                            if dist_to_ray < threshold and dist_from_cam < best_dist:
                                best_plant = plant
                                best_dist = dist_from_cam
                    
                    if best_plant:
                        selected_plant = best_plant
                        
                        # Save DNA to seed folder
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        seed_file = f"{seed_dir}/seed_{best_plant['mesh_type']}_{timestamp}.json"
                        
                        # Serialize DNA
                        dna = best_plant['dna']
                        
                        # Helper to convert numpy types to Python native
                        def to_native(val):
                            if hasattr(val, 'item'):  # numpy scalar
                                return val.item()
                            return float(val) if isinstance(val, (int, float)) else val
                        
                        dna_dict = {
                            'mesh_type': best_plant['mesh_type'],
                            'population': best_plant['population'],
                            'generation': int(best_plant['generation']),
                            'scale': to_native(best_plant['scale']),
                            'plant_type': dna.plant_type.name if hasattr(dna.plant_type, 'name') else str(dna.plant_type) if hasattr(dna, 'plant_type') else 'UNKNOWN',
                            'species_id': int(dna.species_id),
                            'height': to_native(dna.height_gene.value) if hasattr(dna.height_gene, 'value') else 1.0,
                            'width': to_native(dna.width_gene.value) if hasattr(dna.width_gene, 'value') else 1.0,
                            'leaf_color': {
                                'r': to_native(dna.leaf_color.r),
                                'g': to_native(dna.leaf_color.g),
                                'b': to_native(dna.leaf_color.b),
                            },
                            'trunk_color': {
                                'r': to_native(dna.trunk_color.r),
                                'g': to_native(dna.trunk_color.g),
                                'b': to_native(dna.trunk_color.b),
                            } if hasattr(dna, 'trunk_color') else None,
                            'trunk_segments': [
                                {
                                    'length': to_native(seg.length),
                                    'width': to_native(seg.width),
                                    'curve': to_native(seg.curve),
                                    'twist': to_native(seg.twist),
                                    'taper': to_native(seg.taper),
                                }
                                for seg in (dna.trunk_segments or [])
                            ],
                            'branch_count': int(getattr(dna, 'branch_count', 0)),
                            'branch_angle': to_native(getattr(dna, 'branch_angle', 0)),
                            'droop': to_native(getattr(dna, 'droop', 0)),
                            'leaf_size': to_native(getattr(dna, 'leaf_size', 1.0)),
                            'canopy_spread': to_native(getattr(dna, 'canopy_spread', 1.0)),
                        }
                        
                        with open(seed_file, 'w') as f:
                            json.dump(dna_dict, f, indent=2)
                        
                        print(f"🌱 SAVED DNA: {seed_file}")
                        print(f"   Type: {best_plant['mesh_type']} | Pop: {best_plant['population']} | Gen: {best_plant['generation']}")
                    else:
                        selected_plant = None
                        print("No plant at click location")
                        
            elif event.type == KEYDOWN:
                if event.key == K_q or event.key == K_ESCAPE:
                    running = False
                elif event.key == K_p:
                    from datetime import datetime
                    from PIL import Image
                    
                    screenshot_dir = os.path.expanduser("~/.waverse/screenshots")
                    os.makedirs(screenshot_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{screenshot_dir}/evolution_{timestamp}.png"
                    
                    pixels = ctx.fbo.read(components=3)
                    img = Image.frombytes('RGB', (WIDTH, HEIGHT), pixels)
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    img.save(filename)
                    screenshot_count += 1
                    print(f"Saved: {filename}")
        
        # Keyboard input
        keys = pygame.key.get_pressed()
        
        # === RIGHT STICK: Camera Look (IJKL) ===
        # I = look up, K = look down, J = look left, L = look right
        look_speed = dt * 1.2  # Smooth control like andrew_maps
        
        if keys[K_i]: cam_pitch += look_speed * 80   # Look UP
        if keys[K_k]: cam_pitch -= look_speed * 80   # Look DOWN
        if keys[K_j]: cam_yaw += look_speed * 100    # Look LEFT
        if keys[K_l]: cam_yaw -= look_speed * 100    # Look RIGHT
        
        # Allow full 360° loops - normalize to -180 to 180
        while cam_pitch > 180:
            cam_pitch -= 360
        while cam_pitch < -180:
            cam_pitch += 360
        
        # === LEFT STICK: Movement (WASD) ===
        move_speed = 25 * dt
        yaw_rad = math.radians(cam_yaw)
        pitch_rad = math.radians(cam_pitch)
        
        # Full 3D movement following camera direction
        forward_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
        forward_y = math.sin(pitch_rad)
        forward_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        
        # Right vector stays horizontal
        right_x = math.cos(yaw_rad)
        right_z = -math.sin(yaw_rad)
        
        if keys[K_w]:
            cam_x += forward_x * move_speed
            cam_y += forward_y * move_speed
            cam_z += forward_z * move_speed
        if keys[K_s]:
            cam_x -= forward_x * move_speed
            cam_y -= forward_y * move_speed
            cam_z -= forward_z * move_speed
        if keys[K_a]:
            cam_x -= right_x * move_speed
            cam_z -= right_z * move_speed
        if keys[K_d]:
            cam_x += right_x * move_speed
            cam_z += right_z * move_speed
        
        # Vertical flight (H = up, F = down like andrew_maps, or Space/Shift)
        if keys[K_SPACE] or keys[K_h]:
            cam_y += move_speed
        if keys[K_LSHIFT] or keys[K_f]:
            cam_y -= move_speed
        
        # === EVOLUTION STEP ===
        MIN_PLANT_SPACING = 4.0  # Minimum distance between plants (grid-like)
        
        def is_space_available(x: float, z: float, existing: list) -> bool:
            """Check if position is far enough from all existing plants."""
            for p in existing:
                dist = math.sqrt((p['x'] - x)**2 + (p['z'] - z)**2)
                if dist < MIN_PLANT_SPACING:
                    return False
            return True
        
        def find_nearest_free_spot(x: float, z: float, existing: list, attempts: int = 8) -> Tuple[float, float]:
            """Try to find a nearby free spot using grid-aligned search."""
            # First try the exact spot
            if is_space_available(x, z, existing):
                return x, z
            
            # Try grid-aligned positions around the target
            for dist_mult in [1.0, 1.5, 2.0]:
                for angle_idx in range(attempts):
                    angle = (angle_idx / attempts) * 2 * math.pi
                    test_x = x + math.cos(angle) * MIN_PLANT_SPACING * dist_mult
                    test_z = z + math.sin(angle) * MIN_PLANT_SPACING * dist_mult
                    if is_space_available(test_x, test_z, existing):
                        return test_x, test_z
            
            return None, None  # No space found
        
        time_since_generation += dt
        if time_since_generation >= GENERATION_TIME and len(plants) < MAX_PLANTS:
            time_since_generation = 0
            generation += 1
            
            new_plants = []
            all_plants = plants + new_plants  # Include newly added for spacing check
            
            for plant in plants:
                # Each plant has a chance to reproduce
                if rng.random() < 0.35:  # 35% reproduction chance
                    # Spread direction (prefer outward expansion)
                    spread_dist = MIN_PLANT_SPACING + rng.random() * 2.0
                    spread_angle = rng.random() * 2 * math.pi
                    
                    # Bias toward center (where populations will meet)
                    bias_x = -plant['x'] * 0.015
                    bias_z = -plant['z'] * 0.015
                    
                    target_x = plant['x'] + math.cos(spread_angle) * spread_dist + bias_x
                    target_z = plant['z'] + math.sin(spread_angle) * spread_dist + bias_z
                    
                    # Keep in bounds
                    target_x = max(-WORLD_SIZE, min(WORLD_SIZE, target_x))
                    target_z = max(-WORLD_SIZE, min(WORLD_SIZE, target_z))
                    
                    # Find a free spot (grid-like spacing)
                    new_x, new_z = find_nearest_free_spot(target_x, target_z, all_plants)
                    if new_x is None:
                        continue  # No space available, skip reproduction
                    
                    # Check for nearby plants from OTHER population (crossover!)
                    nearby_other = None
                    crossover_dist = MIN_PLANT_SPACING * 2.5  # Must be close for crossover
                    for other in plants:
                        if other['population'] != plant['population']:
                            dist = math.sqrt((other['x'] - new_x)**2 + (other['z'] - new_z)**2)
                            if dist < crossover_dist:
                                nearby_other = other
                                break
                    
                    if nearby_other:
                        # CROSSOVER! Blend DNA from both populations
                        child_dna = plant['dna'].crossover(nearby_other['dna'], rng)
                        child_dna = child_dna.mutate(rng, strength=0.2)  # Extra mutation
                        
                        # Hybrid mesh type - interesting combos based on parents
                        hybrid_mesh_types = ['maple', 'birch', 'cypress', 'bonsai', 'spiral', 'palm']
                        mesh_type = hybrid_mesh_types[int(rng.integers(0, len(hybrid_mesh_types)))]
                        
                        # Create hybrid population name (sorted to be consistent)
                        parent_pops = sorted([plant['population'][0], nearby_other['population'][0]])
                        new_pop = ''.join(parent_pops)
                        if len(new_pop) > 2:
                            new_pop = 'X'  # Multi-hybrid
                        print(f"Gen {generation}: {plant['population']}x{nearby_other['population']} CROSSOVER at ({new_x:.0f}, {new_z:.0f})!")
                    else:
                        # Normal reproduction with mutation
                        child_dna = plant['dna'].mutate(rng, strength=0.15)
                        mesh_type = plant['mesh_type']
                        new_pop = plant['population']
                    
                    new_plant = {
                        'x': new_x, 'z': new_z, 'y': 0,
                        'dna': child_dna,
                        'mesh_type': mesh_type,
                        'generation': generation,
                        'population': new_pop,
                        'scale': 1.5 + rng.random() * 1.0,
                        'rotation': rng.random() * math.pi * 2,
                    }
                    new_plants.append(new_plant)
                    all_plants.append(new_plant)  # Update spacing check
            
            plants.extend(new_plants)
            
            # CONTINUOUS MODE: Old plants die to make room for new ones
            if continuous_mode:
                deaths = 0
                surviving_plants = []
                for plant in plants:
                    age = generation - plant['generation']
                    # Plants can die after PLANT_LIFESPAN generations
                    if age > PLANT_LIFESPAN and rng.random() < DEATH_CHANCE:
                        deaths += 1
                    else:
                        surviving_plants.append(plant)
                
                # Also randomly cull if over max to keep performance good
                if len(surviving_plants) > MAX_PLANTS:
                    # Kill oldest plants first
                    surviving_plants.sort(key=lambda p: p['generation'], reverse=True)
                    surviving_plants = surviving_plants[:MAX_PLANTS]
                    deaths += len(plants) - MAX_PLANTS
                
                plants = surviving_plants
                
                if deaths > 0:
                    print(f"  💀 {deaths} plants died of old age")
            
            # Count populations
            pop_counts = {}
            for p in plants:
                pop = p['population']
                pop_counts[pop] = pop_counts.get(pop, 0) + 1
            
            # Format: A=X B=X C=X D=X | Hybrids=X
            pure = ' '.join(f"{k}={v}" for k, v in sorted(pop_counts.items()) if len(k) == 1)
            hybrids = sum(v for k, v in pop_counts.items() if len(k) > 1)
            mode_str = " [CONTINUOUS]" if continuous_mode else ""
            print(f"Gen {generation}: {pure} | Hybrids={hybrids} | Total={len(plants)}{mode_str}")
        
        # Build camera
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
        
        renderer.clear_instances()
        for plant in plants:
            color = (plant['dna'].leaf_color.r, plant['dna'].leaf_color.g, plant['dna'].leaf_color.b)
            renderer.add_instance(
                plant['mesh_type'],
                plant['x'], plant['y'], plant['z'],
                plant['scale'], plant['rotation'],
                color, plant['dna']
            )
        
        renderer.render()
        pygame.display.flip()
        
        # FPS
        fps = clock.get_fps()
        fps_history.append(fps)
        if len(fps_history) > 60:
            fps_history.pop(0)
        
        import time
        now = time.time()
        if now - last_fps_print > 3.0:
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            print(f"FPS: {avg_fps:.1f} | Plants: {len(plants)} | Gen: {generation}")
            last_fps_print = now
    
    pygame.quit()


def run_render_test():
    """
    Render test mode - uses the ACTUAL dna_mesh_generator.py that the main game uses.
    
    This shows exactly what the in-game plants look like, with:
    - Wide spacing for inspection
    - Controller support (like main game)
    - IJKL camera controls
    - Various plant types including ALIEN and exotic
    """
    pygame.init()
    pygame.joystick.init()
    
    WIDTH, HEIGHT = 1280, 720
    
    # OpenGL 4.1 Core profile for macOS
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
    
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("DNA Render Test - ACTUAL in-game mesh generator")
    
    ctx = moderngl.create_context(require=410)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.CULL_FACE)
    
    # Import the ACTUAL mesh generator used by the main game
    from .dna_mesh_generator import generate_plant_mesh, clear_mesh_cache
    
    # Shader for rendering the generated meshes
    RENDER_VERT = """
    #version 330
    in vec3 in_position;
    in vec3 in_normal;
    in vec3 in_color;
    
    uniform mat4 u_mvp;
    uniform mat4 u_model;
    
    out vec3 v_normal;
    out vec3 v_color;
    out vec3 v_pos;
    
    void main() {
        v_pos = (u_model * vec4(in_position, 1.0)).xyz;
        v_normal = mat3(u_model) * in_normal;
        v_color = in_color;
        gl_Position = u_mvp * vec4(in_position, 1.0);
    }
    """
    
    RENDER_FRAG = """
    #version 330
    in vec3 v_normal;
    in vec3 v_color;
    in vec3 v_pos;
    
    out vec4 f_color;
    
    void main() {
        vec3 light_dir = normalize(vec3(0.3, 1.0, 0.4));
        float diff = max(dot(normalize(v_normal), light_dir), 0.0) * 0.7;
        float ambient = 0.4;
        vec3 lit = v_color * (ambient + diff);
        f_color = vec4(lit, 1.0);
    }
    """
    
    prog = ctx.program(vertex_shader=RENDER_VERT, fragment_shader=RENDER_FRAG)
    
    # Generate plants with various types - focus on types that should show meandering trunks
    rng = np.random.default_rng(42)
    
    plant_types_to_test = [
        PlantType.TREE, PlantType.TREE, PlantType.TREE,  # Multiple trees
        PlantType.TALL_TREE, 
        PlantType.ALIEN, PlantType.ALIEN, PlantType.ALIEN,  # Multiple aliens (most curved)
        PlantType.SPIRAL,
        PlantType.WILLOW,
        PlantType.PALM,
        PlantType.MUSHROOM,
        PlantType.BUSH,
        PlantType.FERN,
        PlantType.CACTUS,
        PlantType.OCTOPUS,
        PlantType.TENTACLE,
    ]
    
    GRID_COLS = 8
    SPACING = 12.0  # Wide spacing for inspection
    
    plants = []
    vaos = []
    
    print("Generating plant meshes using dna_mesh_generator (same as main game)...")
    
    for i, plant_type in enumerate(plant_types_to_test):
        # Create random DNA of this type
        seed = int(rng.integers(0, 2**31))
        dna = PlantDNA.create_random(plant_type, seed)
        
        # Apply extra mutation for variety
        dna = dna.mutate(rng, strength=0.5)
        
        # Generate mesh using the ACTUAL generator
        verts, norms, colors = generate_plant_mesh(dna)
        
        if len(verts) == 0:
            print(f"  WARNING: Empty mesh for {plant_type}")
            continue
        
        # Debug: check vertex bounds
        if len(verts) > 0:
            v_min = verts.min(axis=0)
            v_max = verts.max(axis=0)
            print(f"  {plant_type}: {len(verts)} verts, bounds: ({v_min[0]:.1f},{v_min[1]:.1f},{v_min[2]:.1f}) to ({v_max[0]:.1f},{v_max[1]:.1f},{v_max[2]:.1f})")
        
        # Grid position
        col = i % GRID_COLS
        row = i // GRID_COLS
        x = (col - GRID_COLS/2) * SPACING
        z = (row - len(plant_types_to_test)//GRID_COLS/2) * SPACING
        
        # Create VAO - properly interleave position, normal, color per vertex
        interleaved = np.hstack([verts, norms, colors]).astype('f4')
        vbo = ctx.buffer(interleaved.tobytes())
        vao = ctx.simple_vertex_array(prog, vbo, 'in_position', 'in_normal', 'in_color')
        
        plants.append({
            'type': plant_type,
            'dna': dna,
            'x': x, 'z': z,
            'vao': vao,
            'vert_count': len(verts),
            'scale': 1.5,
            'rotation': rng.random() * math.pi * 2,
        })
        vaos.append(vao)
    
    print(f"\nGenerated {len(plants)} plants for inspection")
    print("\nControls:")
    print("  WASD - Move camera")
    print("  IJKL - Look around")
    print("  Space/Shift - Up/Down")
    print("  Controller: Left stick=move, Right stick=look")
    print("  P - Screenshot | Q/Escape - Quit")
    
    # Controller setup
    gamepad = None
    if pygame.joystick.get_count() > 0:
        gamepad = pygame.joystick.Joystick(0)
        gamepad.init()
        print(f"  Gamepad: {gamepad.get_name()}")
    
    # Camera
    cam_x, cam_y, cam_z = 0, 15, 50
    cam_yaw, cam_pitch = 0, -15
    
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
                elif event.key == K_p:
                    # Screenshot
                    from PIL import Image
                    import os
                    from datetime import datetime
                    screenshot_dir = os.path.expanduser("~/.waverse/screenshots")
                    os.makedirs(screenshot_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{screenshot_dir}/render_test_{timestamp}.png"
                    pixels = ctx.fbo.read(components=3)
                    img = Image.frombytes('RGB', (WIDTH, HEIGHT), pixels)
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    img.save(filename)
                    print(f"Saved: {filename}")
        
        # Input handling
        keys = pygame.key.get_pressed()
        mods = pygame.key.get_mods()
        system_mod = (mods & pygame.KMOD_META) or (mods & pygame.KMOD_CTRL)
        
        # Controller input
        gp_move_x, gp_move_y = 0, 0
        gp_look_x, gp_look_y = 0, 0
        
        if gamepad and not system_mod:
            # Left stick: movement
            if gamepad.get_numaxes() >= 2:
                gp_move_x = gamepad.get_axis(0)
                gp_move_y = gamepad.get_axis(1)
                if abs(gp_move_x) < 0.15: gp_move_x = 0
                if abs(gp_move_y) < 0.15: gp_move_y = 0
            
            # Right stick: look
            if gamepad.get_numaxes() >= 4:
                gp_look_x = gamepad.get_axis(3)
                gp_look_y = gamepad.get_axis(4)
                if abs(gp_look_x) < 0.15: gp_look_x = 0
                if abs(gp_look_y) < 0.15: gp_look_y = 0
        
        if not system_mod:
            # Camera look (IJKL or right stick)
            look_speed = 100 * dt
            if keys[K_i]: cam_pitch += look_speed
            if keys[K_k]: cam_pitch -= look_speed
            if keys[K_j]: cam_yaw += look_speed
            if keys[K_l]: cam_yaw -= look_speed
            
            cam_yaw += gp_look_x * look_speed * 1.5
            cam_pitch -= gp_look_y * look_speed * 1.5
            
            # Clamp pitch
            cam_pitch = max(-89, min(89, cam_pitch))
            
            # Movement
            move_speed = 20 * dt
            yaw_rad = math.radians(cam_yaw)
            pitch_rad = math.radians(cam_pitch)
            
            forward_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
            forward_y = math.sin(pitch_rad)
            forward_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
            right_x = math.cos(yaw_rad)
            right_z = -math.sin(yaw_rad)
            
            # Keyboard movement
            if keys[K_w]:
                cam_x += forward_x * move_speed
                cam_y += forward_y * move_speed
                cam_z += forward_z * move_speed
            if keys[K_s]:
                cam_x -= forward_x * move_speed
                cam_y -= forward_y * move_speed
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
            
            # Controller movement (left stick)
            cam_x += right_x * gp_move_x * move_speed
            cam_z += right_z * gp_move_x * move_speed
            cam_x -= forward_x * gp_move_y * move_speed
            cam_y -= forward_y * gp_move_y * move_speed
            cam_z -= forward_z * gp_move_y * move_speed
        
        # Build matrices
        yaw_rad = math.radians(cam_yaw)
        pitch_rad = math.radians(cam_pitch)
        dir_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
        dir_y = math.sin(pitch_rad)
        dir_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        
        proj = glm.perspective(glm.radians(60), WIDTH/HEIGHT, 0.1, 500.0)
        view = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z),
            glm.vec3(cam_x + dir_x, cam_y + dir_y, cam_z + dir_z),
            glm.vec3(0, 1, 0)
        )
        
        # Render
        ctx.clear(0.5, 0.7, 0.9, 1.0)
        
        for plant in plants:
            # Model matrix
            model = glm.mat4(1.0)
            model = glm.translate(model, glm.vec3(plant['x'], 0, plant['z']))
            model = glm.rotate(model, plant['rotation'], glm.vec3(0, 1, 0))
            model = glm.scale(model, glm.vec3(plant['scale']))
            
            mvp = proj * view * model
            
            # Convert glm matrices to numpy arrays properly
            # glm.mat4 can be converted by accessing each column
            def mat4_to_bytes(m):
                # glm matrices are column-major, so we iterate columns
                data = []
                for col in range(4):
                    for row in range(4):
                        data.append(m[col][row])
                return np.array(data, dtype='f4').tobytes()
            
            prog['u_mvp'].write(mat4_to_bytes(mvp))
            prog['u_model'].write(mat4_to_bytes(model))
            
            plant['vao'].render(moderngl.TRIANGLES)
        
        pygame.display.flip()
    
    # Cleanup
    for vao in vaos:
        vao.release()
    
    pygame.quit()
    print("Render test complete!")


if __name__ == '__main__':
    if '--evolve' in sys.argv or '--continuous' in sys.argv:
        run_evolution_mode()
    elif '--render-test' in sys.argv:
        run_render_test()
    else:
        run_test_grid()

