"""
Modern Flora Renderer - Instanced rendering for plants.

Uses GPU instancing to render thousands of plants with minimal draw calls.
Plants with similar shapes are batched together.
"""

import numpy as np
import moderngl
try:
    from pyglm import glm
except ImportError:
    import glm
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
import math

# Import DNA geometry system for DNA-driven mesh generation
from .dna_geometry import GeometryBuilder, plant_dna_to_geometry


# =============================================================================
# SHADERS
# =============================================================================

FLORA_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;

// Per-instance data
in vec3 in_instance_pos;
in float in_instance_scale;
in float in_instance_rot;
in vec3 in_instance_color;

out vec3 v_normal;
out vec3 v_color;
out vec3 v_world_pos;
out float v_height;

uniform mat4 u_projection;
uniform mat4 u_view;

// Wind uniforms
uniform vec2 u_wind_dir;
uniform float u_wind_strength;
uniform float u_wind_time;
uniform float u_has_tornado;
uniform vec2 u_tornado_center;
uniform float u_tornado_radius;
uniform float u_tornado_strength;

// Special biome mode
uniform float u_dance_mode;  // 1.0 = psychedelic dance/bop mode!

mat3 rotateY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(c, 0, s, 0, 1, 0, -s, 0, c);
}

void main() {
    vec3 pos = in_position;
    
    // Apply instance rotation
    pos = rotateY(in_instance_rot) * pos;
    
    // Apply instance transform
    vec3 scaled = pos * in_instance_scale;
    vec3 world_pos = scaled + in_instance_pos;
    
    // === PSYCHEDELIC DANCE MODE ===
    if (u_dance_mode > 0.5) {
        // Plants BOP to the beat! No wind, just rhythmic bouncing
        // Each plant has its own phase based on position
        float plant_phase = in_instance_pos.x * 0.1 + in_instance_pos.z * 0.13;
        
        // Multiple frequency bops (like dancing to a beat)
        float bop1 = sin(u_wind_time * 4.0 + plant_phase) * 0.3;  // Main beat
        float bop2 = sin(u_wind_time * 8.0 + plant_phase * 1.5) * 0.15;  // Double time
        float bop3 = sin(u_wind_time * 2.0 + plant_phase * 0.5) * 0.2;  // Half time sway
        
        float height_factor = max(0.0, in_position.y) / (in_instance_scale * 2.0 + 0.1);
        height_factor = clamp(height_factor, 0.0, 1.0);
        
        // Vertical bop (bounce up and down)
        world_pos.y += (bop1 + bop2) * height_factor * in_instance_scale * 0.5;
        
        // Side sway (like swaying to music)
        world_pos.x += bop3 * height_factor * in_instance_scale * 0.4;
        world_pos.z += bop1 * 0.5 * height_factor * in_instance_scale * 0.3;
        
        // Slight rotation wobble
        float wobble = sin(u_wind_time * 3.0 + plant_phase * 2.0) * 0.1;
        vec3 wobbled = rotateY(wobble * height_factor) * (world_pos - in_instance_pos) + in_instance_pos;
        world_pos = wobbled;
    } else {
        // === NORMAL WIND SWAY ===
        // Wind sway - based on height in local space
        float height_factor = max(0.0, in_position.y) / (in_instance_scale * 2.0 + 0.1);
        height_factor = clamp(height_factor, 0.0, 1.0);
        height_factor = height_factor * height_factor; // Quadratic falloff - base stays still
        
        // Base wind sway
        float sway_phase = u_wind_time * 2.0 + in_instance_pos.x * 0.05 + in_instance_pos.z * 0.07;
        float sway = sin(sway_phase) * u_wind_strength * height_factor * 0.8;
        float sway2 = sin(sway_phase * 0.7 + 1.3) * u_wind_strength * height_factor * 0.3;
        
        world_pos.x += u_wind_dir.x * sway + u_wind_dir.y * sway2;
        world_pos.z += u_wind_dir.y * sway - u_wind_dir.x * sway2;
        
        // Tornado effect (if active) - only in normal mode
        if (u_has_tornado > 0.5) {
            vec2 to_tornado = u_tornado_center - in_instance_pos.xz;
            float dist = length(to_tornado);
            if (dist < u_tornado_radius && dist > 0.1) {
                float falloff = 1.0 - (dist / u_tornado_radius);
                falloff = falloff * falloff; // Stronger near center
                
                // Rotational wind
                vec2 tangent = normalize(vec2(-to_tornado.y, to_tornado.x));
                float tornado_sway = u_tornado_strength * falloff * height_factor * 2.0;
                
                world_pos.x += tangent.x * tornado_sway * sin(u_wind_time * 5.0);
                world_pos.z += tangent.y * tornado_sway * sin(u_wind_time * 5.0);
            }
        }
    }
    
    v_world_pos = world_pos;
    v_height = in_instance_pos.y;
    v_normal = in_normal;
    v_color = in_color * in_instance_color;
    
    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
}
"""

FLORA_FRAGMENT_SHADER = """
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

void main() {
    vec3 normal = normalize(v_normal);
    vec3 light = normalize(u_light_dir);
    
    // Two-sided lighting for foliage
    float diff = abs(dot(normal, light)) * 0.6 + 0.4;
    
    vec3 lit_color = v_color * (u_ambient + diff * 0.5);
    
    // Subsurface scattering approximation for leaves
    float sss = max(0.0, dot(-normal, light)) * 0.2;
    lit_color += v_color * sss * vec3(0.5, 0.8, 0.3);
    
    // Distance fog
    float dist = length(v_world_pos - u_camera_pos);
    float fog_factor = clamp((dist - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);
    fog_factor = fog_factor * fog_factor;
    vec3 final_color = mix(lit_color, u_fog_color, fog_factor);
    
    fragColor = vec4(final_color, 1.0);
}
"""


# =============================================================================
# PLANT MESH TEMPLATES
# =============================================================================

def create_tree_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tree with trunk, VISIBLE BRANCHES, and canopy - matching legacy look.
    The key visual feature is the brown branches extending from trunk.
    ~80 triangles for proper tree silhouette.
    """
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.45, 0.28, 0.15)  # Darker brown for contrast
    branch_color = (0.4, 0.25, 0.12)  # Slightly darker branches
    foliage_color = (1.0, 1.0, 1.0)  # White for tinting
    
    # Main trunk - 4-sided, taller to support branches
    trunk_h = 0.55
    trunk_w_base = 0.06
    trunk_w_top = 0.03
    
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1_b, z1_b = math.cos(angle1) * trunk_w_base, math.sin(angle1) * trunk_w_base
        x2_b, z2_b = math.cos(angle2) * trunk_w_base, math.sin(angle2) * trunk_w_base
        x1_t, z1_t = math.cos(angle1) * trunk_w_top, math.sin(angle1) * trunk_w_top
        x2_t, z2_t = math.cos(angle2) * trunk_w_top, math.sin(angle2) * trunk_w_top
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, trunk_h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, trunk_h, z2_t), (x1_t, trunk_h, z1_t)])
        n = (math.cos(angle1 + math.pi/4), 0.1, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(trunk_color)
    
    # BRANCHES - the key visual feature that was missing!
    # 5-6 branches radiating outward at different heights
    branch_configs = [
        (0.30, 0.0, 0.25, 0.08),    # (start_y, angle_offset, length, end_height_delta)
        (0.35, 1.2, 0.28, 0.12),
        (0.32, 2.5, 0.22, 0.06),
        (0.40, 3.8, 0.30, 0.15),
        (0.38, 5.0, 0.24, 0.10),
        (0.45, 0.8, 0.20, 0.08),
    ]
    
    branch_thickness = 0.015
    for start_y, base_angle, length, height_delta in branch_configs:
        # Branch extends outward and slightly up
        end_x = math.cos(base_angle) * length
        end_z = math.sin(base_angle) * length
        end_y = start_y + height_delta
        
        # Draw branch as a thin quad (2 triangles)
        # Perpendicular direction for width
        perp_x = -math.sin(base_angle) * branch_thickness
        perp_z = math.cos(base_angle) * branch_thickness
        
        # Start point (on trunk)
        s1 = (perp_x, start_y, perp_z)
        s2 = (-perp_x, start_y, -perp_z)
        # End point (tip of branch)
        e1 = (end_x + perp_x * 0.5, end_y, end_z + perp_z * 0.5)
        e2 = (end_x - perp_x * 0.5, end_y, end_z - perp_z * 0.5)
        
        verts.extend([s1, s2, e1])
        verts.extend([s2, e2, e1])
        n = (0, 0.7, 0.3)
        for _ in range(6):
            normals.append(n)
            colors.append(branch_color)
        
        # Sub-branch from tip (smaller)
        sub_angle = base_angle + 0.5
        sub_len = length * 0.4
        sub_end_x = end_x + math.cos(sub_angle) * sub_len
        sub_end_z = end_z + math.sin(sub_angle) * sub_len
        sub_end_y = end_y + height_delta * 0.3
        
        sub_s1 = e1
        sub_s2 = e2
        sub_e = (sub_end_x, sub_end_y, sub_end_z)
        
        verts.extend([sub_s1, sub_s2, sub_e])
        for _ in range(3):
            normals.append(n)
            colors.append(branch_color)
    
    # Canopy - 3 layered cones (on top of branches)
    layers = [(0.45, 0.65, 0.30), (0.58, 0.82, 0.24), (0.72, 0.95, 0.16)]
    for base_y, top_y, radius in layers:
        for i in range(6):
            angle1 = i * math.pi / 3
            angle2 = (i + 1) * math.pi / 3
            x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
            x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
            
            verts.extend([(0, top_y, 0), (x1, base_y, z1), (x2, base_y, z2)])
            n = (0, 0.6, 0.4)
            normals.extend([n, n, n])
            colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_bush_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bush with visible stems and leafy dome - matching legacy style.
    Legacy _draw_bush has multiple visible stem lines.
    """
    verts = []
    normals = []
    colors = []
    
    stem_color = (0.45, 0.32, 0.18)  # Brown stems
    bush_color = (1.0, 1.0, 1.0)  # White for tinting
    
    # Multiple visible stems (like legacy)
    for i in range(5):
        angle = i * 2 * math.pi / 5 + 0.3
        spread = 0.12
        stem_h = 0.35 + i * 0.08
        
        bx = math.cos(angle) * spread
        bz = math.sin(angle) * spread
        end_x = bx * 1.8
        end_z = bz * 1.8
        
        # Stem as thin triangle
        t = 0.015
        verts.extend([(bx - t, 0, bz), (bx + t, 0, bz), (end_x, stem_h, end_z)])
        n = (math.cos(angle), 0.5, math.sin(angle))
        normals.extend([n, n, n])
        colors.extend([stem_color, stem_color, stem_color])
    
    # Leafy dome on top
    top = (0, 0.55, 0)
    radius = 0.4
    
    for i in range(8):
        angle1 = i * math.pi / 4
        angle2 = (i + 1) * math.pi / 4
        x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
        x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
        
        # Upper dome
        mid_y = 0.25
        verts.extend([top, (x1 * 0.8, mid_y, z1 * 0.8), (x2 * 0.8, mid_y, z2 * 0.8)])
        n1 = (math.cos(angle1 + math.pi/8), 0.7, math.sin(angle1 + math.pi/8))
        normals.extend([n1, n1, n1])
        colors.extend([bush_color, bush_color, bush_color])
        
        # Lower section
        verts.extend([(x1 * 0.8, mid_y, z1 * 0.8), (x1, 0.05, z1), (x2, 0.05, z2)])
        verts.extend([(x1 * 0.8, mid_y, z1 * 0.8), (x2, 0.05, z2), (x2 * 0.8, mid_y, z2 * 0.8)])
        n2 = (math.cos(angle1 + math.pi/8), 0.3, math.sin(angle1 + math.pi/8))
        for _ in range(6):
            normals.append(n2)
            colors.append(bush_color)
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_grass_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple grass matching legacy - just 2 triangles (crossed blades).
    Legacy: glVertex3f(-0.08, 0, 0); glVertex3f(0.08, 0, 0); glVertex3f(0, height, 0);
    """
    grass_color = (1.0, 1.0, 1.0)  # White for tinting
    
    verts = [
        # 2 crossed triangles like legacy grass
        (-0.08, 0, 0), (0.08, 0, 0), (0, 1.0, 0),
        (0, 0, -0.08), (0, 0, 0.08), (0, 0.9, 0),
    ]
    normals = [(0, 0.5, 0.866)] * 6
    colors = [grass_color] * 6
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_dome_tree_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dome tree - round layered canopy with branches."""
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.5, 0.32, 0.18)
    branch_color = (0.45, 0.28, 0.14)
    foliage_color = (1.0, 1.0, 1.0)
    
    # Trunk
    trunk_h = 0.45
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1_b, z1_b = math.cos(angle1) * 0.05, math.sin(angle1) * 0.05
        x2_b, z2_b = math.cos(angle2) * 0.05, math.sin(angle2) * 0.05
        x1_t, z1_t = math.cos(angle1) * 0.025, math.sin(angle1) * 0.025
        x2_t, z2_t = math.cos(angle2) * 0.025, math.sin(angle2) * 0.025
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, trunk_h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, trunk_h, z2_t), (x1_t, trunk_h, z1_t)])
        n = (math.cos(angle1 + math.pi/4), 0.1, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(trunk_color)
    
    # Branches spreading outward
    for i in range(5):
        angle = i * 2 * math.pi / 5
        length = 0.22
        start_y = 0.30 + i * 0.03
        end_y = start_y + 0.08
        
        end_x = math.cos(angle) * length
        end_z = math.sin(angle) * length
        
        t = 0.012
        perp_x = -math.sin(angle) * t
        perp_z = math.cos(angle) * t
        
        verts.extend([(perp_x, start_y, perp_z), (-perp_x, start_y, -perp_z), 
                      (end_x, end_y, end_z)])
        n = (0, 0.7, 0.3)
        normals.extend([n, n, n])
        colors.extend([branch_color, branch_color, branch_color])
    
    # Dome canopy - layered semi-spheres
    layers = [
        (0.35, 0.55, 0.35),  # Base layer
        (0.45, 0.70, 0.30),  # Middle
        (0.55, 0.85, 0.22),  # Upper
    ]
    for base_y, top_y, radius in layers:
        for i in range(8):
            angle1 = i * math.pi / 4
            angle2 = (i + 1) * math.pi / 4
            x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
            x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
            
            verts.extend([(0, top_y, 0), (x1, base_y, z1), (x2, base_y, z2)])
            n = (math.cos(angle1 + math.pi/8), 0.5, math.sin(angle1 + math.pi/8))
            normals.extend([n, n, n])
            colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_umbrella_tree_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Umbrella/acacia tree - tall trunk, flat wide canopy with long branches."""
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.42, 0.26, 0.12)
    branch_color = (0.38, 0.22, 0.10)
    foliage_color = (1.0, 1.0, 1.0)
    
    # Tall thin trunk
    trunk_h = 0.65
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1_b, z1_b = math.cos(angle1) * 0.04, math.sin(angle1) * 0.04
        x2_b, z2_b = math.cos(angle2) * 0.04, math.sin(angle2) * 0.04
        x1_t, z1_t = math.cos(angle1) * 0.02, math.sin(angle1) * 0.02
        x2_t, z2_t = math.cos(angle2) * 0.02, math.sin(angle2) * 0.02
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, trunk_h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, trunk_h, z2_t), (x1_t, trunk_h, z1_t)])
        n = (math.cos(angle1 + math.pi/4), 0, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(trunk_color)
    
    # Long horizontal branches (acacia style)
    for i in range(6):
        angle = i * math.pi / 3 + 0.2
        length = 0.35  # Long branches
        start_y = trunk_h * 0.85
        end_y = start_y + 0.02  # Nearly horizontal
        
        end_x = math.cos(angle) * length
        end_z = math.sin(angle) * length
        
        t = 0.01
        perp_x = -math.sin(angle) * t
        perp_z = math.cos(angle) * t
        
        verts.extend([(perp_x, start_y, perp_z), (-perp_x, start_y, -perp_z),
                      (end_x, end_y, end_z)])
        n = (0, 0.7, 0.3)
        normals.extend([n, n, n])
        colors.extend([branch_color, branch_color, branch_color])
    
    # Wide flat canopy
    canopy_y = trunk_h * 0.9
    canopy_top = trunk_h * 0.98
    radius = 0.42
    
    for i in range(8):
        angle1 = i * math.pi / 4
        angle2 = (i + 1) * math.pi / 4
        x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
        x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
        
        verts.extend([(0, canopy_top, 0), (x1, canopy_y - 0.05, z1), (x2, canopy_y - 0.05, z2)])
        n = (0, 0.8, 0.2)
        normals.extend([n, n, n])
        colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_weeping_tree_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weeping willow - drooping branches."""
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.5, 0.35, 0.2)
    branch_color = (0.45, 0.30, 0.15)
    foliage_color = (1.0, 1.0, 1.0)
    
    # Trunk
    trunk_h = 0.5
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1_b, z1_b = math.cos(angle1) * 0.06, math.sin(angle1) * 0.06
        x2_b, z2_b = math.cos(angle2) * 0.06, math.sin(angle2) * 0.06
        x1_t, z1_t = math.cos(angle1) * 0.03, math.sin(angle1) * 0.03
        x2_t, z2_t = math.cos(angle2) * 0.03, math.sin(angle2) * 0.03
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, trunk_h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, trunk_h, z2_t), (x1_t, trunk_h, z1_t)])
        n = (math.cos(angle1 + math.pi/4), 0, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(trunk_color)
    
    # Drooping branches (weeping style)
    for i in range(8):
        angle = i * math.pi / 4
        start_y = trunk_h * 0.8
        mid_y = start_y + 0.1
        end_y = 0.15  # Droop down low
        
        out_dist = 0.2
        mid_x = math.cos(angle) * out_dist
        mid_z = math.sin(angle) * out_dist
        end_x = math.cos(angle) * 0.35
        end_z = math.sin(angle) * 0.35
        
        t = 0.01
        perp_x = -math.sin(angle) * t
        perp_z = math.cos(angle) * t
        
        # Branch from trunk to mid
        verts.extend([(perp_x, start_y, perp_z), (-perp_x, start_y, -perp_z),
                      (mid_x, mid_y, mid_z)])
        # Mid to drooping tip
        verts.extend([(mid_x + perp_x, mid_y, mid_z + perp_z),
                      (mid_x - perp_x, mid_y, mid_z - perp_z),
                      (end_x, end_y, end_z)])
        n = (0, 0.5, 0.5)
        for _ in range(6):
            normals.append(n)
            colors.append(branch_color)
    
    # Small top canopy
    canopy_y = trunk_h * 0.85
    for i in range(6):
        angle1 = i * math.pi / 3
        angle2 = (i + 1) * math.pi / 3
        x1, z1 = math.cos(angle1) * 0.2, math.sin(angle1) * 0.2
        x2, z2 = math.cos(angle2) * 0.2, math.sin(angle2) * 0.2
        
        verts.extend([(0, trunk_h, 0), (x1, canopy_y, z1), (x2, canopy_y, z2)])
        n = (0, 0.7, 0.3)
        normals.extend([n, n, n])
        colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_columnar_tree_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Columnar/cypress tree - tall and narrow."""
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.4, 0.25, 0.12)
    foliage_color = (1.0, 1.0, 1.0)
    
    # Trunk (taller)
    trunk_h = 0.3
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1_b, z1_b = math.cos(angle1) * 0.04, math.sin(angle1) * 0.04
        x2_b, z2_b = math.cos(angle2) * 0.04, math.sin(angle2) * 0.04
        x1_t, z1_t = math.cos(angle1) * 0.025, math.sin(angle1) * 0.025
        x2_t, z2_t = math.cos(angle2) * 0.025, math.sin(angle2) * 0.025
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, trunk_h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, trunk_h, z2_t), (x1_t, trunk_h, z1_t)])
        n = (math.cos(angle1 + math.pi/4), 0, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(trunk_color)
    
    # Tall narrow canopy (multiple stacked rings)
    layers = [
        (0.25, 0.45, 0.12),
        (0.40, 0.60, 0.10),
        (0.55, 0.75, 0.08),
        (0.70, 0.90, 0.06),
        (0.85, 1.0, 0.04),
    ]
    for base_y, top_y, radius in layers:
        for i in range(6):
            angle1 = i * math.pi / 3
            angle2 = (i + 1) * math.pi / 3
            x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
            x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
            
            verts.extend([(0, top_y, 0), (x1, base_y, z1), (x2, base_y, z2)])
            n = (math.cos(angle1 + math.pi/6), 0.4, math.sin(angle1 + math.pi/6))
            normals.extend([n, n, n])
            colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_palm_tree_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Palm tree - tall trunk with fronds at top."""
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.55, 0.4, 0.25)
    frond_color = (1.0, 1.0, 1.0)
    
    # Tall thin trunk
    trunk_h = 0.7
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1_b, z1_b = math.cos(angle1) * 0.05, math.sin(angle1) * 0.05
        x2_b, z2_b = math.cos(angle2) * 0.05, math.sin(angle2) * 0.05
        x1_t, z1_t = math.cos(angle1) * 0.03, math.sin(angle1) * 0.03
        x2_t, z2_t = math.cos(angle2) * 0.03, math.sin(angle2) * 0.03
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, trunk_h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, trunk_h, z2_t), (x1_t, trunk_h, z1_t)])
        n = (math.cos(angle1 + math.pi/4), 0, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(trunk_color)
    
    # Palm fronds radiating from top
    for i in range(7):
        angle = i * 2 * math.pi / 7
        frond_len = 0.35
        
        # Frond arcs outward and slightly down
        mid_x = math.cos(angle) * frond_len * 0.5
        mid_z = math.sin(angle) * frond_len * 0.5
        mid_y = trunk_h + 0.08
        
        end_x = math.cos(angle) * frond_len
        end_z = math.sin(angle) * frond_len
        end_y = trunk_h - 0.1  # Droop at tips
        
        # Frond as triangle
        verts.extend([(0, trunk_h, 0), (mid_x, mid_y, mid_z), (end_x, end_y, end_z)])
        n = (0, 0.6, 0.4)
        normals.extend([n, n, n])
        colors.extend([frond_color, frond_color, frond_color])
        
        # Second triangle for width
        offset = 0.04
        verts.extend([(0, trunk_h, 0),
                      (mid_x + math.cos(angle + 0.3) * offset, mid_y, mid_z + math.sin(angle + 0.3) * offset),
                      (end_x + math.cos(angle + 0.2) * offset * 0.5, end_y, end_z + math.sin(angle + 0.2) * offset * 0.5)])
        normals.extend([n, n, n])
        colors.extend([frond_color, frond_color, frond_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_flower_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flower with stem and petals radiating outward."""
    verts = []
    normals = []
    colors = []
    
    stem_color = (0.2, 0.5, 0.2)  # Green stem
    petal_color = (1.0, 1.0, 1.0)  # White for tinting (DNA color applied)
    center_color = (0.9, 0.8, 0.2)  # Yellow center
    
    # Stem
    verts.extend([(-0.02, 0, 0), (0.02, 0, 0), (0, 0.3, 0)])
    for _ in range(3):
        normals.append((0, 0, 1))
        colors.append(stem_color)
    
    # 5 petals radiating outward
    for i in range(5):
        angle = i * 2 * math.pi / 5
        x_dir = math.cos(angle)
        z_dir = math.sin(angle)
        
        # Petal base at flower center, tip outward
        center = (0, 0.32, 0)
        tip = (x_dir * 0.15, 0.35, z_dir * 0.15)
        side1 = (x_dir * 0.05 - z_dir * 0.04, 0.33, z_dir * 0.05 + x_dir * 0.04)
        side2 = (x_dir * 0.05 + z_dir * 0.04, 0.33, z_dir * 0.05 - x_dir * 0.04)
        
        verts.extend([center, side1, tip])
        verts.extend([center, tip, side2])
        n = (0, 0.9, 0.1)
        for _ in range(6):
            normals.append(n)
            colors.append(petal_color)
    
    # Center dot (2 small triangles)
    verts.extend([(0.03, 0.34, 0.02), (-0.03, 0.34, 0.02), (0, 0.36, 0)])
    verts.extend([(0.03, 0.34, -0.02), (0, 0.36, 0), (-0.03, 0.34, -0.02)])
    for _ in range(6):
        normals.append((0, 1, 0))
        colors.append(center_color)
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_flower_tall_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tall flower like a sunflower or tulip."""
    verts = []
    normals = []
    colors = []
    
    stem_color = (0.25, 0.45, 0.2)
    petal_color = (1.0, 1.0, 1.0)  # White for tinting
    center_color = (0.5, 0.35, 0.15)  # Brown center
    
    # Taller stem
    verts.extend([(-0.025, 0, 0), (0.025, 0, 0), (0.015, 0.55, 0)])
    verts.extend([(-0.025, 0, 0), (0.015, 0.55, 0), (-0.015, 0.55, 0)])
    for _ in range(6):
        normals.append((0, 0.1, 0.99))
        colors.append(stem_color)
    
    # Leaf on stem
    verts.extend([(0, 0.2, 0), (0.12, 0.18, 0.06), (0.08, 0.28, 0.03)])
    for _ in range(3):
        normals.append((0, 0.5, 0.5))
        colors.append(stem_color)
    
    # 8 petals in a ring
    for i in range(8):
        angle = i * math.pi / 4
        x_dir = math.cos(angle)
        z_dir = math.sin(angle)
        
        base_y = 0.55
        center = (0, base_y + 0.02, 0)
        tip = (x_dir * 0.18, base_y + 0.03, z_dir * 0.18)
        side = (x_dir * 0.08, base_y, z_dir * 0.08)
        
        verts.extend([center, side, tip])
        n = (x_dir * 0.3, 0.9, z_dir * 0.3)
        normals.extend([n, n, n])
        colors.extend([petal_color, petal_color, petal_color])
    
    # Brown center dome
    for i in range(6):
        angle1 = i * math.pi / 3
        angle2 = (i + 1) * math.pi / 3
        x1, z1 = math.cos(angle1) * 0.06, math.sin(angle1) * 0.06
        x2, z2 = math.cos(angle2) * 0.06, math.sin(angle2) * 0.06
        
        verts.extend([(0, 0.62, 0), (x1, 0.56, z1), (x2, 0.56, z2)])
        normals.extend([(0, 1, 0), (0, 1, 0), (0, 1, 0)])
        colors.extend([center_color, center_color, center_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_mushroom_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple mushroom matching legacy - stem quad + cap triangle."""
    stem_color = (0.9, 0.88, 0.8)
    cap_color = (1.0, 1.0, 1.0)
    
    verts = [
        # Stem quad (2 tris)
        (-0.05, 0, 0), (0.05, 0, 0), (0.05, 0.7, 0),
        (-0.05, 0, 0), (0.05, 0.7, 0), (-0.05, 0.7, 0),
        # Cap triangle
        (0, 1.0, 0), (-0.3, 0.6, 0), (0.3, 0.6, 0),
    ]
    normals = [(0, 0, 1)] * 6 + [(0, 0.7, 0.3)] * 3
    colors = [stem_color] * 6 + [cap_color] * 3
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_fern_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fern with multiple fronds radiating outward."""
    verts = []
    normals = []
    colors = []
    
    fern_color = (1.0, 1.0, 1.0)  # White for tinting
    
    # 5 fronds radiating outward at various angles
    for i in range(5):
        angle = i * 2 * math.pi / 5 + 0.2
        # Each frond is a triangle leaning outward
        x_dir = math.cos(angle)
        z_dir = math.sin(angle)
        
        # Frond base at center, tips outward and up
        base = (x_dir * 0.02, 0.05, z_dir * 0.02)
        tip = (x_dir * 0.35, 0.25, z_dir * 0.35)
        side = (x_dir * 0.15 - z_dir * 0.08, 0.15, z_dir * 0.15 + x_dir * 0.08)
        
        verts.extend([base, side, tip])
        n = (x_dir * 0.3, 0.9, z_dir * 0.3)
        normals.extend([n, n, n])
        colors.extend([fern_color, fern_color, fern_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_cactus_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cactus with main body and arms."""
    verts = []
    normals = []
    colors = []
    
    cactus_color = (1.0, 1.0, 1.0)  # White for tinting
    
    # Main body - 4-sided prism
    for i in range(4):
        angle1 = i * math.pi / 2 + math.pi / 4
        angle2 = (i + 1) * math.pi / 2 + math.pi / 4
        x1, z1 = math.cos(angle1) * 0.08, math.sin(angle1) * 0.08
        x2, z2 = math.cos(angle2) * 0.08, math.sin(angle2) * 0.08
        
        verts.extend([(x1, 0, z1), (x2, 0, z2), (x1, 0.7, z1)])
        verts.extend([(x2, 0, z2), (x2, 0.7, z2), (x1, 0.7, z1)])
        n = (math.cos(angle1 + math.pi/4), 0.1, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(cactus_color)
    
    # Top cap
    verts.extend([(0, 0.75, 0), (0.08, 0.7, 0.08), (-0.08, 0.7, 0.08)])
    verts.extend([(0, 0.75, 0), (-0.08, 0.7, -0.08), (0.08, 0.7, -0.08)])
    for _ in range(6):
        normals.append((0, 1, 0))
        colors.append(cactus_color)
    
    # Left arm
    verts.extend([(-0.08, 0.35, 0), (-0.22, 0.35, 0), (-0.22, 0.55, 0)])
    verts.extend([(-0.22, 0.55, 0), (-0.15, 0.6, 0), (-0.08, 0.35, 0)])
    for _ in range(6):
        normals.append((0, 0, 1))
        colors.append(cactus_color)
    
    # Right arm (slightly higher)
    verts.extend([(0.08, 0.4, 0), (0.2, 0.4, 0), (0.2, 0.58, 0)])
    verts.extend([(0.2, 0.58, 0), (0.14, 0.63, 0), (0.08, 0.4, 0)])
    for _ in range(6):
        normals.append((0, 0, 1))
        colors.append(cactus_color)
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_tree_pine_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pine tree with pointed cone canopy - classic evergreen silhouette."""
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.4, 0.25, 0.12)
    foliage_color = (1.0, 1.0, 1.0)  # White for tinting
    
    # Thin tall trunk
    trunk_h = 0.4
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1_b, z1_b = math.cos(angle1) * 0.035, math.sin(angle1) * 0.035
        x2_b, z2_b = math.cos(angle2) * 0.035, math.sin(angle2) * 0.035
        x1_t, z1_t = math.cos(angle1) * 0.02, math.sin(angle1) * 0.02
        x2_t, z2_t = math.cos(angle2) * 0.02, math.sin(angle2) * 0.02
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, trunk_h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, trunk_h, z2_t), (x1_t, trunk_h, z1_t)])
        n = (math.cos(angle1 + math.pi/4), 0.1, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(trunk_color)
    
    # 4 cone layers getting smaller toward top (classic pine look)
    layers = [
        (0.20, 0.45, 0.32),  # base_y, top_y, radius
        (0.35, 0.60, 0.26),
        (0.50, 0.78, 0.20),
        (0.68, 0.95, 0.12),
    ]
    
    for base_y, top_y, radius in layers:
        for i in range(6):
            angle1 = i * math.pi / 3
            angle2 = (i + 1) * math.pi / 3
            x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
            x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
            
            # Cone pointing up
            verts.extend([(0, top_y, 0), (x1, base_y, z1), (x2, base_y, z2)])
            n = (math.cos(angle1 + math.pi/6), 0.5, math.sin(angle1 + math.pi/6))
            normals.extend([n, n, n])
            colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_tree_oak_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Oak tree with broad, spreading irregular canopy."""
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.38, 0.25, 0.12)
    branch_color = (0.35, 0.22, 0.10)
    foliage_color = (1.0, 1.0, 1.0)
    
    # Thick trunk
    trunk_h = 0.35
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1_b, z1_b = math.cos(angle1) * 0.08, math.sin(angle1) * 0.08
        x2_b, z2_b = math.cos(angle2) * 0.08, math.sin(angle2) * 0.08
        x1_t, z1_t = math.cos(angle1) * 0.06, math.sin(angle1) * 0.06
        x2_t, z2_t = math.cos(angle2) * 0.06, math.sin(angle2) * 0.06
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, trunk_h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, trunk_h, z2_t), (x1_t, trunk_h, z1_t)])
        n = (math.cos(angle1 + math.pi/4), 0.1, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(trunk_color)
    
    # Major spreading branches
    t = 0.015
    branch_configs = [
        (0.28, 0.0, 0.28, 0.1),
        (0.30, 1.3, 0.25, 0.08),
        (0.32, 2.5, 0.30, 0.12),
        (0.28, 3.8, 0.26, 0.09),
        (0.34, 5.2, 0.22, 0.07),
    ]
    
    for start_y, base_angle, length, height_delta in branch_configs:
        end_x = math.cos(base_angle) * length
        end_z = math.sin(base_angle) * length
        end_y = start_y + height_delta
        perp_x = -math.sin(base_angle) * t
        perp_z = math.cos(base_angle) * t
        
        verts.extend([(perp_x, start_y, perp_z), (-perp_x, start_y, -perp_z), (end_x, end_y, end_z)])
        n = (0, 0.7, 0.3)
        normals.extend([n, n, n])
        colors.extend([branch_color, branch_color, branch_color])
    
    # Wide irregular canopy - multiple overlapping dome sections
    canopy_blobs = [
        (0.0, 0.45, 0.22),      # center
        (0.12, 0.42, 0.18),     # offset 1
        (-0.10, 0.48, 0.16),    # offset 2
        (0.05, 0.52, 0.14),     # offset 3 (higher)
        (-0.08, 0.38, 0.20),    # offset 4
    ]
    
    for cx, base_y, radius in canopy_blobs:
        top_y = base_y + radius * 0.6
        for i in range(6):
            angle1 = i * math.pi / 3
            angle2 = (i + 1) * math.pi / 3
            x1 = cx + math.cos(angle1) * radius
            z1 = math.sin(angle1) * radius
            x2 = cx + math.cos(angle2) * radius
            z2 = math.sin(angle2) * radius
            
            verts.extend([(cx, top_y, 0), (x1, base_y, z1), (x2, base_y, z2)])
            n = (math.cos(angle1 + math.pi/6), 0.6, math.sin(angle1 + math.pi/6))
            normals.extend([n, n, n])
            colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_tree_small_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small/young tree with thin trunk and compact canopy."""
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.45, 0.30, 0.15)
    foliage_color = (1.0, 1.0, 1.0)
    
    # Thin short trunk
    trunk_h = 0.3
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1_b, z1_b = math.cos(angle1) * 0.025, math.sin(angle1) * 0.025
        x2_b, z2_b = math.cos(angle2) * 0.025, math.sin(angle2) * 0.025
        x1_t, z1_t = math.cos(angle1) * 0.012, math.sin(angle1) * 0.012
        x2_t, z2_t = math.cos(angle2) * 0.012, math.sin(angle2) * 0.012
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, trunk_h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, trunk_h, z2_t), (x1_t, trunk_h, z1_t)])
        n = (math.cos(angle1 + math.pi/4), 0.1, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(trunk_color)
    
    # Compact dome canopy
    for i in range(6):
        angle1 = i * math.pi / 3
        angle2 = (i + 1) * math.pi / 3
        x1, z1 = math.cos(angle1) * 0.18, math.sin(angle1) * 0.18
        x2, z2 = math.cos(angle2) * 0.18, math.sin(angle2) * 0.18
        
        verts.extend([(0, 0.55, 0), (x1, 0.28, z1), (x2, 0.28, z2)])
        n = (math.cos(angle1 + math.pi/6), 0.6, math.sin(angle1 + math.pi/6))
        normals.extend([n, n, n])
        colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_vine_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vine with multiple hanging tendrils."""
    verts = []
    normals = []
    colors = []
    
    vine_color = (1.0, 1.0, 1.0)  # White for tinting
    
    # Multiple hanging vine strands
    for i in range(4):
        x_offset = (i - 1.5) * 0.08
        z_offset = ((i % 2) - 0.5) * 0.04
        length = 0.5 + (i % 3) * 0.15
        
        # Each strand is a triangle pointing down
        verts.extend([
            (x_offset - 0.02, 0.1, z_offset),
            (x_offset + 0.02, 0.1, z_offset),
            (x_offset, -length, z_offset)
        ])
        n = (0, 0, 1)
        normals.extend([n, n, n])
        colors.extend([vine_color, vine_color, vine_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_seaweed_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Seaweed with wavy fronds."""
    verts = []
    normals = []
    colors = []
    
    seaweed_color = (1.0, 1.0, 1.0)  # White for tinting
    
    # 3 wavy fronds
    for i in range(3):
        x_offset = (i - 1) * 0.1
        # Each frond sways slightly
        sway = 0.03 * (i - 1)
        
        # Base to mid
        verts.extend([
            (x_offset - 0.03, 0, 0),
            (x_offset + 0.03, 0, 0),
            (x_offset + sway, 0.4, 0.02)
        ])
        # Mid to top
        verts.extend([
            (x_offset - 0.02, 0.35, 0),
            (x_offset + 0.02, 0.35, 0),
            (x_offset + sway * 2, 0.7, 0.03)
        ])
        
        n = (0, 0.3, 0.95)
        for _ in range(6):
            normals.append(n)
            colors.append(seaweed_color)
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_grass_tall_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tall grass (reeds/wheat style) with multiple blades."""
    verts = []
    normals = []
    colors = []
    
    grass_color = (1.0, 1.0, 1.0)  # White for tinting
    
    # 4 tall grass blades
    for i in range(4):
        angle = i * math.pi / 2 + 0.3
        x_base = math.cos(angle) * 0.03
        z_base = math.sin(angle) * 0.03
        x_tip = x_base + math.cos(angle) * 0.08
        z_tip = z_base + math.sin(angle) * 0.08
        height = 0.7 + (i % 2) * 0.15
        
        verts.extend([
            (x_base - 0.015, 0, z_base),
            (x_base + 0.015, 0, z_base),
            (x_tip, height, z_tip)
        ])
        n = (0, 0.3, 0.95)
        normals.extend([n, n, n])
        colors.extend([grass_color, grass_color, grass_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_coral_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple coral - reuse bush mesh."""
    return create_bush_mesh()


def create_lily_pad_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple lily pad - flat triangle."""
    pad_color = (1.0, 1.0, 1.0)
    
    verts = [(-0.3, 0.02, -0.2), (0.3, 0.02, -0.2), (0, 0.02, 0.3)]
    normals = [(0, 1, 0)] * 3
    colors = [pad_color] * 3
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_groundcover_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple groundcover - reuse lily pad (flat)."""
    return create_lily_pad_mesh()


def create_spiral_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple spiral - reuse grass mesh."""
    return create_grass_mesh()


def create_tree_sparse_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tree with sparse branches (2-3 branches) - for low branch_count DNA."""
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.5, 0.32, 0.18)
    branch_color = (0.45, 0.28, 0.14)
    foliage_color = (1.0, 1.0, 1.0)
    
    # Trunk
    trunk_h = 0.6
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1_b, z1_b = math.cos(angle1) * 0.05, math.sin(angle1) * 0.05
        x2_b, z2_b = math.cos(angle2) * 0.05, math.sin(angle2) * 0.05
        x1_t, z1_t = math.cos(angle1) * 0.025, math.sin(angle1) * 0.025
        x2_t, z2_t = math.cos(angle2) * 0.025, math.sin(angle2) * 0.025
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, trunk_h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, trunk_h, z2_t), (x1_t, trunk_h, z1_t)])
        n = (math.cos(angle1 + math.pi/4), 0.1, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(trunk_color)
    
    # Just 3 sparse branches
    branch_configs = [
        (0.35, 0.0, 0.3, 0.1),
        (0.40, 2.1, 0.28, 0.12),
        (0.45, 4.2, 0.25, 0.08),
    ]
    
    t = 0.012
    for start_y, base_angle, length, height_delta in branch_configs:
        end_x = math.cos(base_angle) * length
        end_z = math.sin(base_angle) * length
        end_y = start_y + height_delta
        perp_x = -math.sin(base_angle) * t
        perp_z = math.cos(base_angle) * t
        
        verts.extend([(perp_x, start_y, perp_z), (-perp_x, start_y, -perp_z), (end_x, end_y, end_z)])
        n = (0, 0.7, 0.3)
        normals.extend([n, n, n])
        colors.extend([branch_color, branch_color, branch_color])
    
    # Small canopy
    layers = [(0.50, 0.70, 0.22), (0.62, 0.85, 0.16)]
    for base_y, top_y, radius in layers:
        for i in range(6):
            angle1 = i * math.pi / 3
            angle2 = (i + 1) * math.pi / 3
            x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
            x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
            
            verts.extend([(0, top_y, 0), (x1, base_y, z1), (x2, base_y, z2)])
            n = (0, 0.6, 0.4)
            normals.extend([n, n, n])
            colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_tree_dense_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tree with dense branches (8+ branches) - for high branch_count DNA."""
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.42, 0.26, 0.12)
    branch_color = (0.38, 0.22, 0.10)
    foliage_color = (1.0, 1.0, 1.0)
    
    # Trunk
    trunk_h = 0.5
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1_b, z1_b = math.cos(angle1) * 0.07, math.sin(angle1) * 0.07
        x2_b, z2_b = math.cos(angle2) * 0.07, math.sin(angle2) * 0.07
        x1_t, z1_t = math.cos(angle1) * 0.035, math.sin(angle1) * 0.035
        x2_t, z2_t = math.cos(angle2) * 0.035, math.sin(angle2) * 0.035
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, trunk_h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, trunk_h, z2_t), (x1_t, trunk_h, z1_t)])
        n = (math.cos(angle1 + math.pi/4), 0.1, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(trunk_color)
    
    # Dense branches - 9 of them
    t = 0.01
    for i in range(9):
        base_angle = i * 2 * math.pi / 9 + 0.3
        start_y = 0.25 + (i % 3) * 0.08
        length = 0.2 + (i % 4) * 0.04
        height_delta = 0.05 + (i % 3) * 0.03
        
        end_x = math.cos(base_angle) * length
        end_z = math.sin(base_angle) * length
        end_y = start_y + height_delta
        perp_x = -math.sin(base_angle) * t
        perp_z = math.cos(base_angle) * t
        
        verts.extend([(perp_x, start_y, perp_z), (-perp_x, start_y, -perp_z), (end_x, end_y, end_z)])
        n = (0, 0.7, 0.3)
        normals.extend([n, n, n])
        colors.extend([branch_color, branch_color, branch_color])
        
        # Sub-branch
        sub_angle = base_angle + 0.4
        sub_len = length * 0.5
        sub_end = (end_x + math.cos(sub_angle) * sub_len, end_y + 0.05, end_z + math.sin(sub_angle) * sub_len)
        verts.extend([(end_x, end_y, end_z), (end_x + t, end_y, end_z + t), sub_end])
        normals.extend([n, n, n])
        colors.extend([branch_color, branch_color, branch_color])
    
    # Large dense canopy
    layers = [(0.35, 0.55, 0.38), (0.48, 0.70, 0.32), (0.60, 0.85, 0.25), (0.72, 0.95, 0.18)]
    for base_y, top_y, radius in layers:
        for i in range(8):
            angle1 = i * math.pi / 4
            angle2 = (i + 1) * math.pi / 4
            x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
            x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
            
            verts.extend([(0, top_y, 0), (x1, base_y, z1), (x2, base_y, z2)])
            n = (math.cos(angle1 + math.pi/8), 0.5, math.sin(angle1 + math.pi/8))
            normals.extend([n, n, n])
            colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_tree_tall_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tall tree with high trunk - for tall_tree type."""
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.48, 0.30, 0.16)
    branch_color = (0.42, 0.26, 0.12)
    foliage_color = (1.0, 1.0, 1.0)
    
    # Tall trunk
    trunk_h = 0.75
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1_b, z1_b = math.cos(angle1) * 0.055, math.sin(angle1) * 0.055
        x2_b, z2_b = math.cos(angle2) * 0.055, math.sin(angle2) * 0.055
        x1_t, z1_t = math.cos(angle1) * 0.02, math.sin(angle1) * 0.02
        x2_t, z2_t = math.cos(angle2) * 0.02, math.sin(angle2) * 0.02
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, trunk_h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, trunk_h, z2_t), (x1_t, trunk_h, z1_t)])
        n = (math.cos(angle1 + math.pi/4), 0.1, math.sin(angle1 + math.pi/4))
        for _ in range(6):
            normals.append(n)
            colors.append(trunk_color)
    
    # Branches at top
    t = 0.01
    for i in range(5):
        base_angle = i * 2 * math.pi / 5
        start_y = 0.55 + i * 0.04
        length = 0.18
        
        end_x = math.cos(base_angle) * length
        end_z = math.sin(base_angle) * length
        end_y = start_y + 0.08
        perp_x = -math.sin(base_angle) * t
        perp_z = math.cos(base_angle) * t
        
        verts.extend([(perp_x, start_y, perp_z), (-perp_x, start_y, -perp_z), (end_x, end_y, end_z)])
        n = (0, 0.7, 0.3)
        normals.extend([n, n, n])
        colors.extend([branch_color, branch_color, branch_color])
    
    # Canopy at very top
    layers = [(0.70, 0.85, 0.22), (0.78, 0.95, 0.16)]
    for base_y, top_y, radius in layers:
        for i in range(6):
            angle1 = i * math.pi / 3
            angle2 = (i + 1) * math.pi / 3
            x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
            x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
            
            verts.extend([(0, top_y, 0), (x1, base_y, z1), (x2, base_y, z2)])
            n = (0, 0.6, 0.4)
            normals.extend([n, n, n])
            colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_octopus_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Octopus plant - central bulb with radiating tentacle-like branches."""
    verts = []
    normals = []
    colors = []
    
    bulb_color = (1.0, 1.0, 1.0)  # White for tinting
    
    # Central bulb/body
    for i in range(8):
        angle1 = i * math.pi / 4
        angle2 = (i + 1) * math.pi / 4
        x1, z1 = math.cos(angle1) * 0.15, math.sin(angle1) * 0.15
        x2, z2 = math.cos(angle2) * 0.15, math.sin(angle2) * 0.15
        
        verts.extend([(0, 0.35, 0), (x1, 0.05, z1), (x2, 0.05, z2)])
        n = (math.cos(angle1 + math.pi/8), 0.5, math.sin(angle1 + math.pi/8))
        normals.extend([n, n, n])
        colors.extend([bulb_color, bulb_color, bulb_color])
    
    # Radiating tentacles
    for i in range(6):
        angle = i * math.pi / 3 + 0.2
        # Each tentacle curves outward and down
        mid_dist = 0.25
        end_dist = 0.45
        
        mid_x = math.cos(angle) * mid_dist
        mid_z = math.sin(angle) * mid_dist
        mid_y = 0.25
        
        end_x = math.cos(angle) * end_dist
        end_z = math.sin(angle) * end_dist
        end_y = 0.08  # Droop down
        
        # Tentacle as 2 triangles
        t = 0.03
        verts.extend([(0, 0.2, 0), (mid_x - t, mid_y, mid_z), (mid_x + t, mid_y, mid_z)])
        verts.extend([(mid_x, mid_y, mid_z), (end_x - t * 0.5, end_y, end_z), (end_x + t * 0.5, end_y, end_z)])
        n = (0, 0.6, 0.4)
        for _ in range(6):
            normals.append(n)
            colors.append(bulb_color)
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_tentacle_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tentacle plant - hangs down from attachment point."""
    verts = []
    normals = []
    colors = []
    
    color = (1.0, 1.0, 1.0)  # White for tinting
    
    # Base attachment
    for i in range(6):
        angle1 = i * math.pi / 3
        angle2 = (i + 1) * math.pi / 3
        x1, z1 = math.cos(angle1) * 0.1, math.sin(angle1) * 0.1
        x2, z2 = math.cos(angle2) * 0.1, math.sin(angle2) * 0.1
        
        verts.extend([(0, 1.0, 0), (x1, 0.85, z1), (x2, 0.85, z2)])
        n = (0, 0.8, 0.2)
        normals.extend([n, n, n])
        colors.extend([color, color, color])
    
    # Dangling segments with sway
    segments = [(0.85, 0.65, 0.08, 0.03), (0.65, 0.40, 0.06, 0.06), (0.40, 0.15, 0.04, 0.04), (0.15, 0.0, 0.02, 0.02)]
    for y_top, y_bot, w_top, sway in segments:
        for i in range(4):
            angle1 = i * math.pi / 2
            angle2 = (i + 1) * math.pi / 2
            
            x1_t, z1_t = math.cos(angle1) * w_top + sway, math.sin(angle1) * w_top
            x2_t, z2_t = math.cos(angle2) * w_top + sway, math.sin(angle2) * w_top
            x1_b, z1_b = math.cos(angle1) * w_top * 0.7 - sway, math.sin(angle1) * w_top * 0.7
            x2_b, z2_b = math.cos(angle2) * w_top * 0.7 - sway, math.sin(angle2) * w_top * 0.7
            
            verts.extend([(x1_t, y_top, z1_t), (x2_t, y_top, z2_t), (x1_b, y_bot, z1_b)])
            verts.extend([(x2_t, y_top, z2_t), (x2_b, y_bot, z2_b), (x1_b, y_bot, z1_b)])
            n = (math.cos(angle1 + math.pi/4), 0, math.sin(angle1 + math.pi/4))
            for _ in range(6):
                normals.append(n)
                colors.append(color)
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_crystal_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crystal plant - sharp angular facets, semi-transparent look."""
    verts = []
    normals = []
    colors = []
    
    crystal_color = (0.7, 0.85, 1.0)  # Light blue/cyan tint
    
    # Main crystal spire (hexagonal pyramid)
    h = 1.0
    r = 0.12
    for i in range(6):
        angle1 = i * math.pi / 3
        angle2 = (i + 1) * math.pi / 3
        x1, z1 = math.cos(angle1) * r, math.sin(angle1) * r
        x2, z2 = math.cos(angle2) * r, math.sin(angle2) * r
        
        # Face to tip
        verts.extend([(x1, 0, z1), (x2, 0, z2), (0, h, 0)])
        n = (math.cos(angle1 + math.pi/6), 0.4, math.sin(angle1 + math.pi/6))
        normals.extend([n, n, n])
        colors.extend([crystal_color, crystal_color, crystal_color])
    
    # Secondary smaller crystals at angles
    for i in range(3):
        angle = i * 2 * math.pi / 3 + 0.5
        offset_x = math.cos(angle) * 0.15
        offset_z = math.sin(angle) * 0.15
        h2 = 0.5
        r2 = 0.06
        
        for j in range(4):
            a1 = j * math.pi / 2
            a2 = (j + 1) * math.pi / 2
            x1, z1 = offset_x + math.cos(a1) * r2, offset_z + math.sin(a1) * r2
            x2, z2 = offset_x + math.cos(a2) * r2, offset_z + math.sin(a2) * r2
            
            verts.extend([(x1, 0, z1), (x2, 0, z2), (offset_x, h2, offset_z)])
            n = (math.cos(a1 + math.pi/4), 0.3, math.sin(a1 + math.pi/4))
            normals.extend([n, n, n])
            colors.extend([crystal_color, crystal_color, crystal_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_alien_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alien plant - weird asymmetric organic shape."""
    verts = []
    normals = []
    colors = []
    
    alien_color = (1.0, 1.0, 1.0)  # White for tinting
    
    # Central twisted stalk
    h = 0.8
    for i in range(4):
        angle1 = i * math.pi / 2 + i * 0.3  # Twist
        angle2 = (i + 1) * math.pi / 2 + (i + 1) * 0.3
        r = 0.05
        
        x1_b, z1_b = math.cos(angle1) * r, math.sin(angle1) * r
        x2_b, z2_b = math.cos(angle2) * r, math.sin(angle2) * r
        x1_t, z1_t = math.cos(angle1 + 0.5) * r * 0.5 + 0.03, math.sin(angle1 + 0.5) * r * 0.5
        x2_t, z2_t = math.cos(angle2 + 0.5) * r * 0.5 + 0.03, math.sin(angle2 + 0.5) * r * 0.5
        
        verts.extend([(x1_b, 0, z1_b), (x2_b, 0, z2_b), (x1_t, h, z1_t)])
        verts.extend([(x2_b, 0, z2_b), (x2_t, h, z2_t), (x1_t, h, z1_t)])
        n = (math.cos(angle1), 0.1, math.sin(angle1))
        for _ in range(6):
            normals.append(n)
            colors.append(alien_color)
    
    # Bulbous growths at different heights
    bulbs = [(0.3, 0.08, 0.1), (0.5, -0.05, 0.12), (0.7, 0.06, -0.08)]
    for y, off_x, off_z in bulbs:
        for i in range(5):
            angle1 = i * 2 * math.pi / 5
            angle2 = (i + 1) * 2 * math.pi / 5
            r = 0.08
            
            x1, z1 = off_x + math.cos(angle1) * r, off_z + math.sin(angle1) * r
            x2, z2 = off_x + math.cos(angle2) * r, off_z + math.sin(angle2) * r
            
            verts.extend([(off_x, y + r, off_z), (x1, y, z1), (x2, y, z2)])
            n = (0, 0.8, 0.2)
            normals.extend([n, n, n])
            colors.extend([alien_color, alien_color, alien_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_bioluminescent_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Glowing bioluminescent plant with bulbous pods and tendrils."""
    verts = []
    normals = []
    colors = []
    
    # Bright glowing colors (will be tinted but start bright)
    glow_color = (1.0, 1.0, 1.0)  # White for tinting (DNA glow_color will be used)
    stem_color = (0.3, 0.5, 0.4)
    
    # Central stem
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        r = 0.03
        verts.extend([
            (math.cos(angle1) * r, 0, math.sin(angle1) * r),
            (math.cos(angle2) * r, 0, math.sin(angle2) * r),
            (0, 0.5, 0)
        ])
        n = (math.cos(angle1 + math.pi/4), 0.2, math.sin(angle1 + math.pi/4))
        normals.extend([n, n, n])
        colors.extend([stem_color, stem_color, stem_color])
    
    # Glowing pods at different heights
    pods = [
        (0.15, 0.06, 0.0, 0.08),   # y, x_off, z_off, size
        (0.30, -0.05, 0.07, 0.10),
        (0.45, 0.04, -0.05, 0.07),
        (0.55, 0.0, 0.0, 0.12),    # Top main pod
    ]
    
    for y, off_x, off_z, size in pods:
        # Each pod is a small sphere-ish shape
        for i in range(6):
            angle1 = i * math.pi / 3
            angle2 = (i + 1) * math.pi / 3
            x1 = off_x + math.cos(angle1) * size
            z1 = off_z + math.sin(angle1) * size
            x2 = off_x + math.cos(angle2) * size
            z2 = off_z + math.sin(angle2) * size
            
            verts.extend([(off_x, y + size * 0.7, off_z), (x1, y, z1), (x2, y, z2)])
            n = (math.cos(angle1 + math.pi/6), 0.6, math.sin(angle1 + math.pi/6))
            normals.extend([n, n, n])
            colors.extend([glow_color, glow_color, glow_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_spiral_tree_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tree with spiral/helix trunk and fractal-ish branches."""
    verts = []
    normals = []
    colors = []
    
    trunk_color = (0.5, 0.35, 0.2)
    foliage_color = (1.0, 1.0, 1.0)
    
    # Spiral trunk - helix shape
    segments = 12
    trunk_h = 0.7
    for i in range(segments):
        t1 = i / segments
        t2 = (i + 1) / segments
        angle1 = t1 * math.pi * 3  # 1.5 full rotations
        angle2 = t2 * math.pi * 3
        r = 0.04
        spiral_r = 0.08 * (1 - t1 * 0.5)  # Spiral gets tighter at top
        
        # Position on spiral
        cx1 = math.cos(angle1) * spiral_r
        cz1 = math.sin(angle1) * spiral_r
        cx2 = math.cos(angle2) * spiral_r
        cz2 = math.sin(angle2) * spiral_r
        y1 = t1 * trunk_h
        y2 = t2 * trunk_h
        
        # Quad for trunk segment
        verts.extend([
            (cx1 - r, y1, cz1), (cx1 + r, y1, cz1), (cx2, y2, cz2)
        ])
        n = (math.cos(angle1), 0.2, math.sin(angle1))
        normals.extend([n, n, n])
        colors.extend([trunk_color, trunk_color, trunk_color])
    
    # Spiral foliage clusters along the helix
    for i in range(8):
        t = (i + 0.5) / 8
        angle = t * math.pi * 3
        y = t * trunk_h + 0.1
        cx = math.cos(angle) * 0.08 * (1 - t * 0.5)
        cz = math.sin(angle) * 0.08 * (1 - t * 0.5)
        
        # Small foliage burst
        for j in range(4):
            a1 = j * math.pi / 2 + angle
            a2 = (j + 1) * math.pi / 2 + angle
            r = 0.08 * (1 - t * 0.5)
            
            verts.extend([
                (cx, y + r * 0.5, cz),
                (cx + math.cos(a1) * r, y - r * 0.2, cz + math.sin(a1) * r),
                (cx + math.cos(a2) * r, y - r * 0.2, cz + math.sin(a2) * r)
            ])
            n = (0, 0.8, 0.2)
            normals.extend([n, n, n])
            colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_bulbous_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bulbous plant with multiple swollen growths."""
    verts = []
    normals = []
    colors = []
    
    bulb_color = (1.0, 1.0, 1.0)  # White for tinting
    stem_color = (0.4, 0.5, 0.3)
    
    # Multiple bulbs stacked/clustered
    bulbs = [
        (0.0, 0.0, 0.0, 0.15),     # Base large bulb
        (0.08, 0.18, 0.0, 0.10),   # Upper right
        (-0.06, 0.22, 0.05, 0.08), # Upper left
        (0.0, 0.35, 0.0, 0.12),    # Top
    ]
    
    for bx, by, bz, size in bulbs:
        # Each bulb is a dome
        for i in range(6):
            angle1 = i * math.pi / 3
            angle2 = (i + 1) * math.pi / 3
            x1 = bx + math.cos(angle1) * size
            z1 = bz + math.sin(angle1) * size
            x2 = bx + math.cos(angle2) * size
            z2 = bz + math.sin(angle2) * size
            
            # Top dome
            verts.extend([(bx, by + size * 0.8, bz), (x1, by, z1), (x2, by, z2)])
            n = (math.cos(angle1 + math.pi/6), 0.7, math.sin(angle1 + math.pi/6))
            normals.extend([n, n, n])
            colors.extend([bulb_color, bulb_color, bulb_color])
    
    # Small stem at bottom
    verts.extend([(-0.02, 0, 0), (0.02, 0, 0), (0, -0.1, 0)])
    normals.extend([(0, -1, 0)] * 3)
    colors.extend([stem_color] * 3)
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_spiky_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spiky plant with sharp protrusions in all directions."""
    verts = []
    normals = []
    colors = []
    
    spike_color = (1.0, 1.0, 1.0)  # White for tinting
    core_color = (0.6, 0.5, 0.4)
    
    # Central core
    for i in range(6):
        angle1 = i * math.pi / 3
        angle2 = (i + 1) * math.pi / 3
        r = 0.1
        verts.extend([
            (0, 0.15, 0),
            (math.cos(angle1) * r, 0.05, math.sin(angle1) * r),
            (math.cos(angle2) * r, 0.05, math.sin(angle2) * r)
        ])
        n = (0, 1, 0)
        normals.extend([n, n, n])
        colors.extend([core_color, core_color, core_color])
    
    # Spikes radiating outward at various angles
    spike_dirs = [
        (0, 1, 0, 0.35),      # Up
        (0.7, 0.5, 0, 0.25),  # Up-right
        (-0.6, 0.6, 0.3, 0.28),
        (0.3, 0.3, 0.8, 0.22),
        (-0.4, 0.4, -0.6, 0.26),
        (0.5, 0.2, -0.5, 0.20),
        (-0.8, 0.3, -0.2, 0.24),
        (0.2, 0.8, 0.4, 0.30),
    ]
    
    for dx, dy, dz, length in spike_dirs:
        # Normalize direction
        mag = math.sqrt(dx*dx + dy*dy + dz*dz)
        dx, dy, dz = dx/mag, dy/mag, dz/mag
        
        # Spike base at center, tip outward
        base_y = 0.1
        tip = (dx * length, base_y + dy * length, dz * length)
        
        # Create thin spike (triangle)
        perp_x = -dz * 0.02
        perp_z = dx * 0.02
        
        verts.extend([
            (perp_x, base_y, perp_z),
            (-perp_x, base_y, -perp_z),
            tip
        ])
        n = (dx, dy, dz)
        normals.extend([n, n, n])
        colors.extend([spike_color, spike_color, spike_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_droopy_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Droopy melting plant that hangs downward."""
    verts = []
    normals = []
    colors = []
    
    droop_color = (1.0, 1.0, 1.0)
    
    # Central mass that melts downward
    for i in range(6):
        angle1 = i * math.pi / 3
        angle2 = (i + 1) * math.pi / 3
        
        # Top rim
        top_r = 0.15
        x1_t = math.cos(angle1) * top_r
        z1_t = math.sin(angle1) * top_r
        x2_t = math.cos(angle2) * top_r
        z2_t = math.sin(angle2) * top_r
        
        # Bottom drips down further at some points
        drip = 0.1 + (i % 2) * 0.15  # Alternate longer/shorter
        bottom_r = 0.08
        x1_b = math.cos(angle1) * bottom_r
        z1_b = math.sin(angle1) * bottom_r
        x2_b = math.cos(angle2) * bottom_r
        z2_b = math.sin(angle2) * bottom_r
        
        # Top dome
        verts.extend([(0, 0.4, 0), (x1_t, 0.3, z1_t), (x2_t, 0.3, z2_t)])
        # Side going down
        verts.extend([(x1_t, 0.3, z1_t), (x1_b, -drip, z1_b), (x2_t, 0.3, z2_t)])
        verts.extend([(x2_t, 0.3, z2_t), (x1_b, -drip, z1_b), (x2_b, -drip - 0.05, z2_b)])
        
        n = (math.cos(angle1 + math.pi/6), 0.3, math.sin(angle1 + math.pi/6))
        for _ in range(9):
            normals.append(n)
            colors.append(droop_color)
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_fractal_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fractal-like recursive branching structure."""
    verts = []
    normals = []
    colors = []
    
    branch_color = (1.0, 1.0, 1.0)
    
    def add_branch(x, y, z, angle_h, angle_v, length, depth):
        if depth <= 0 or length < 0.02:
            return
        
        # Calculate end point
        end_x = x + math.cos(angle_h) * math.cos(angle_v) * length
        end_y = y + math.sin(angle_v) * length
        end_z = z + math.sin(angle_h) * math.cos(angle_v) * length
        
        # Create branch triangle
        t = 0.015 * depth
        perp_x = -math.sin(angle_h) * t
        perp_z = math.cos(angle_h) * t
        
        verts.extend([
            (x + perp_x, y, z + perp_z),
            (x - perp_x, y, z - perp_z),
            (end_x, end_y, end_z)
        ])
        n = (math.cos(angle_h), 0.5, math.sin(angle_h))
        normals.extend([n, n, n])
        colors.extend([branch_color, branch_color, branch_color])
        
        # Recurse - 2-3 child branches
        new_length = length * 0.65
        new_depth = depth - 1
        
        # Child branches at different angles
        for i in range(2 + (depth % 2)):
            child_angle_h = angle_h + (i - 1) * 0.8
            child_angle_v = angle_v + 0.3 - i * 0.15
            add_branch(end_x, end_y, end_z, child_angle_h, child_angle_v, new_length, new_depth)
    
    # Start with main trunk going up
    add_branch(0, 0, 0, 0, 0.8, 0.25, 4)
    add_branch(0, 0, 0, math.pi * 0.6, 0.7, 0.22, 4)
    add_branch(0, 0, 0, math.pi * 1.3, 0.75, 0.23, 4)
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_tube_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tubular plant with hollow openings."""
    verts = []
    normals = []
    colors = []
    
    tube_color = (1.0, 1.0, 1.0)
    inner_color = (0.2, 0.15, 0.1)  # Dark inside
    
    # Multiple tubes at different angles
    tubes = [
        (0, 0, 0, 0, 0.5),      # Straight up
        (0.05, 0, 0.02, 0.3, 0.4),  # Leaning
        (-0.03, 0, 0.04, -0.25, 0.35),
    ]
    
    for bx, by, bz, lean, height in tubes:
        # Outer tube wall
        for i in range(6):
            angle1 = i * math.pi / 3
            angle2 = (i + 1) * math.pi / 3
            r_out = 0.06
            r_in = 0.04
            
            x1_b = bx + math.cos(angle1) * r_out
            z1_b = bz + math.sin(angle1) * r_out
            x2_b = bx + math.cos(angle2) * r_out
            z2_b = bz + math.sin(angle2) * r_out
            
            x1_t = bx + lean + math.cos(angle1) * r_out
            z1_t = bz + math.sin(angle1) * r_out
            x2_t = bx + lean + math.cos(angle2) * r_out
            z2_t = bz + math.sin(angle2) * r_out
            
            # Outer wall
            verts.extend([(x1_b, by, z1_b), (x2_b, by, z2_b), (x1_t, by + height, z1_t)])
            verts.extend([(x2_b, by, z2_b), (x2_t, by + height, z2_t), (x1_t, by + height, z1_t)])
            
            n = (math.cos(angle1 + math.pi/6), 0.1, math.sin(angle1 + math.pi/6))
            for _ in range(6):
                normals.append(n)
                colors.append(tube_color)
        
        # Top rim/opening
        for i in range(6):
            angle1 = i * math.pi / 3
            angle2 = (i + 1) * math.pi / 3
            r_out = 0.06
            r_in = 0.035
            
            x1_o = bx + lean + math.cos(angle1) * r_out
            z1_o = bz + math.sin(angle1) * r_out
            x2_o = bx + lean + math.cos(angle2) * r_out
            z2_o = bz + math.sin(angle2) * r_out
            x1_i = bx + lean + math.cos(angle1) * r_in
            z1_i = bz + math.sin(angle1) * r_in
            
            verts.extend([(x1_o, by + height, z1_o), (x2_o, by + height, z2_o), (x1_i, by + height - 0.03, z1_i)])
            normals.extend([(0, 1, 0)] * 3)
            colors.extend([inner_color, inner_color, inner_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_bush_flowering_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bush with visible flowers on top."""
    verts = []
    normals = []
    colors = []
    
    stem_color = (0.4, 0.3, 0.15)
    leaf_color = (1.0, 1.0, 1.0)  # White for tinting
    flower_color = (1.0, 0.85, 0.9)  # Light pink
    
    # Stems
    for i in range(4):
        angle = i * math.pi / 2 + 0.2
        bx = math.cos(angle) * 0.1
        bz = math.sin(angle) * 0.1
        end_x = bx * 2
        end_z = bz * 2
        
        verts.extend([(bx - 0.01, 0, bz), (bx + 0.01, 0, bz), (end_x, 0.35, end_z)])
        n = (0, 0.7, 0.3)
        normals.extend([n, n, n])
        colors.extend([stem_color, stem_color, stem_color])
    
    # Leafy dome
    top = (0, 0.5, 0)
    radius = 0.35
    for i in range(8):
        angle1 = i * math.pi / 4
        angle2 = (i + 1) * math.pi / 4
        x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
        x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
        
        verts.extend([top, (x1, 0.15, z1), (x2, 0.15, z2)])
        n = (math.cos(angle1 + math.pi/8), 0.7, math.sin(angle1 + math.pi/8))
        normals.extend([n, n, n])
        colors.extend([leaf_color, leaf_color, leaf_color])
    
    # Flowers on top
    flower_pos = [(0, 0.52, 0), (0.1, 0.48, 0.08), (-0.08, 0.46, -0.1), (0.05, 0.45, -0.12)]
    for fx, fy, fz in flower_pos:
        for i in range(5):
            angle = i * 2 * math.pi / 5
            px = fx + math.cos(angle) * 0.05
            pz = fz + math.sin(angle) * 0.05
            
            verts.extend([(fx, fy + 0.03, fz), (px, fy, pz), (fx + math.cos(angle + 0.6) * 0.05, fy, fz + math.sin(angle + 0.6) * 0.05)])
            n = (0, 1, 0)
            normals.extend([n, n, n])
            colors.extend([flower_color, flower_color, flower_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


# =============================================================================
# ABSTRACT/WEIRD/TRIPPY MESHES - For exotic biomes!
# =============================================================================

def create_twisted_spire_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A twisting spiral spire that defies gravity."""
    verts = []
    normals = []
    colors = []
    
    spire_color = (1.0, 1.0, 1.0)
    segments = 16
    height = 0.8
    twist = 4 * math.pi  # 2 full rotations
    
    for i in range(segments):
        t1 = i / segments
        t2 = (i + 1) / segments
        y1, y2 = t1 * height, t2 * height
        angle1 = t1 * twist
        angle2 = t2 * twist
        r1 = 0.08 * (1 - t1 * 0.7)  # Tapers
        r2 = 0.08 * (1 - t2 * 0.7)
        
        for j in range(4):
            a1 = j * math.pi / 2 + angle1
            a2 = (j + 1) * math.pi / 2 + angle1
            a3 = j * math.pi / 2 + angle2
            
            verts.extend([
                (math.cos(a1) * r1, y1, math.sin(a1) * r1),
                (math.cos(a2) * r1, y1, math.sin(a2) * r1),
                (math.cos(a3) * r2, y2, math.sin(a3) * r2)
            ])
            n = (math.cos(a1), 0.5, math.sin(a1))
            normals.extend([n, n, n])
            colors.extend([spire_color, spire_color, spire_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_blob_creature_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Amorphous blob with pseudopods."""
    verts = []
    normals = []
    colors = []
    
    blob_color = (1.0, 1.0, 1.0)
    
    # Main blob body (irregular sphere using random offsets)
    for i in range(12):
        for j in range(6):
            theta1 = i * 2 * math.pi / 12
            theta2 = (i + 1) * 2 * math.pi / 12
            phi1 = j * math.pi / 6
            phi2 = (j + 1) * math.pi / 6
            
            r = 0.15 + 0.05 * math.sin(i * 3) * math.cos(j * 2)  # Irregular
            
            x1 = r * math.sin(phi1) * math.cos(theta1)
            y1 = r * math.cos(phi1) + 0.15
            z1 = r * math.sin(phi1) * math.sin(theta1)
            
            x2 = r * math.sin(phi1) * math.cos(theta2)
            y2 = r * math.cos(phi1) + 0.15
            z2 = r * math.sin(phi1) * math.sin(theta2)
            
            x3 = r * math.sin(phi2) * math.cos(theta1)
            y3 = r * math.cos(phi2) + 0.15
            z3 = r * math.sin(phi2) * math.sin(theta1)
            
            verts.extend([(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)])
            n = (x1, y1 - 0.15, z1)
            normals.extend([n, n, n])
            colors.extend([blob_color, blob_color, blob_color])
    
    # Pseudopods extending outward
    for p in range(3):
        angle = p * 2 * math.pi / 3
        px, pz = math.cos(angle) * 0.15, math.sin(angle) * 0.15
        
        for i in range(4):
            t = i / 4
            r = 0.04 * (1 - t)
            px2 = px + math.cos(angle) * t * 0.2
            pz2 = pz + math.sin(angle) * t * 0.2
            py = 0.1 + t * 0.05
            
            verts.extend([
                (px2, py, pz2), (px2 + r, py - r, pz2), (px2, py - r, pz2 + r)
            ])
            normals.extend([(0, 1, 0)] * 3)
            colors.extend([blob_color, blob_color, blob_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_eye_flower_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creepy flower with an eye-like center."""
    verts = []
    normals = []
    colors = []
    
    petal_color = (1.0, 1.0, 1.0)
    eye_white = (0.95, 0.95, 0.95)
    pupil = (0.1, 0.05, 0.15)
    
    # Stem
    for i in range(4):
        angle = i * math.pi / 2
        r = 0.02
        verts.extend([
            (math.cos(angle) * r, 0, math.sin(angle) * r),
            (math.cos(angle + math.pi/2) * r, 0, math.sin(angle + math.pi/2) * r),
            (0, 0.3, 0)
        ])
        normals.extend([(0, 0, 1)] * 3)
        colors.extend([petal_color, petal_color, petal_color])
    
    # Petals
    for i in range(8):
        angle = i * 2 * math.pi / 8
        px1 = math.cos(angle) * 0.03
        pz1 = math.sin(angle) * 0.03
        px2 = math.cos(angle) * 0.15
        pz2 = math.sin(angle) * 0.15
        
        verts.extend([
            (0, 0.3, 0), (px2, 0.28, pz2), (px1, 0.32, pz1)
        ])
        normals.extend([(0, 1, 0)] * 3)
        colors.extend([petal_color, petal_color, petal_color])
    
    # Eye (white part) - dome
    for i in range(6):
        angle1 = i * 2 * math.pi / 6
        angle2 = (i + 1) * 2 * math.pi / 6
        r = 0.05
        
        verts.extend([
            (0, 0.32, 0),
            (math.cos(angle1) * r, 0.3, math.sin(angle1) * r),
            (math.cos(angle2) * r, 0.3, math.sin(angle2) * r)
        ])
        normals.extend([(0, 1, 0)] * 3)
        colors.extend([eye_white, eye_white, eye_white])
    
    # Pupil (small dark center)
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        r = 0.02
        
        verts.extend([
            (0, 0.33, 0),
            (math.cos(angle1) * r, 0.31, math.sin(angle1) * r),
            (math.cos(angle2) * r, 0.31, math.sin(angle2) * r)
        ])
        normals.extend([(0, 1, 0)] * 3)
        colors.extend([pupil, pupil, pupil])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_impossible_geometry_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An impossible/Escherian structure that doesn't make sense."""
    verts = []
    normals = []
    colors = []
    
    color = (1.0, 1.0, 1.0)
    
    # Interlocking rings that couldn't exist
    for ring in range(3):
        angle_offset = ring * 2 * math.pi / 3
        tilt = ring * 0.4
        
        for i in range(8):
            t1 = i * 2 * math.pi / 8
            t2 = (i + 1) * 2 * math.pi / 8
            
            # Ring points (tilted)
            r = 0.12
            y_off = 0.15 + ring * 0.1
            
            x1 = math.cos(t1 + angle_offset) * r
            z1 = math.sin(t1 + angle_offset) * r
            y1 = y_off + math.sin(t1) * tilt * 0.1
            
            x2 = math.cos(t2 + angle_offset) * r
            z2 = math.sin(t2 + angle_offset) * r
            y2 = y_off + math.sin(t2) * tilt * 0.1
            
            # Thin ring segment
            verts.extend([
                (x1, y1, z1), (x2, y2, z2), (x1 * 0.8, y1 + 0.02, z1 * 0.8)
            ])
            normals.extend([(0, 1, 0)] * 3)
            colors.extend([color, color, color])
    
    # Central impossible cube (vertices don't connect properly)
    cube_verts = [
        (0.05, 0.1, 0.05), (0.05, 0.3, 0.05), (-0.05, 0.2, -0.05),
        (-0.05, 0.2, -0.05), (-0.05, 0.4, -0.05), (0.05, 0.3, 0.05),
    ]
    for v in cube_verts:
        verts.append(v)
        normals.append((0, 1, 0))
        colors.append(color)
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_fire_plant_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plant made of flames - for hellfire biome."""
    verts = []
    normals = []
    colors = []
    
    # Multiple flame tongues
    flames = 5
    for f in range(flames):
        base_angle = f * 2 * math.pi / flames
        bx = math.cos(base_angle) * 0.05
        bz = math.sin(base_angle) * 0.05
        
        height = 0.3 + f * 0.08
        flicker = math.sin(f * 1.7) * 0.05
        
        # Each flame is a tapered cone with jagged edges
        for i in range(3):
            angle1 = i * 2 * math.pi / 3 + base_angle
            angle2 = (i + 1) * 2 * math.pi / 3 + base_angle
            
            r_base = 0.06
            r_mid = 0.04 + flicker
            
            # Bottom to middle
            verts.extend([
                (bx + math.cos(angle1) * r_base, 0, bz + math.sin(angle1) * r_base),
                (bx + math.cos(angle2) * r_base, 0, bz + math.sin(angle2) * r_base),
                (bx + math.cos(angle1) * r_mid + flicker, height * 0.5, bz + math.sin(angle1) * r_mid)
            ])
            
            # Middle to tip
            verts.extend([
                (bx + math.cos(angle1) * r_mid + flicker, height * 0.5, bz + math.sin(angle1) * r_mid),
                (bx + math.cos(angle2) * r_mid - flicker, height * 0.5, bz + math.sin(angle2) * r_mid),
                (bx + flicker * 2, height, bz)  # Tip
            ])
            
            for _ in range(6):
                normals.append((0, 0.8, 0.2))
                colors.append((1.0, 1.0, 1.0))  # White, tinted by instance
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_shadow_tendril_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creeping shadow tendril for shadow biome."""
    verts = []
    normals = []
    colors = []
    
    tendril_color = (1.0, 1.0, 1.0)
    
    # Multiple tendrils reaching upward
    tendrils = 4
    for t in range(tendrils):
        base_angle = t * 2 * math.pi / tendrils + t * 0.5
        
        segments = 8
        for i in range(segments):
            s1 = i / segments
            s2 = (i + 1) / segments
            
            # Creepy curving path
            curl = math.sin(s1 * 3) * 0.1
            x1 = math.cos(base_angle) * 0.02 + curl
            x2 = math.cos(base_angle) * 0.02 + math.sin(s2 * 3) * 0.1
            z1 = math.sin(base_angle) * 0.02 + curl * 0.5
            z2 = math.sin(base_angle) * 0.02 + math.sin(s2 * 3) * 0.05
            y1 = s1 * 0.5
            y2 = s2 * 0.5
            r1 = 0.02 * (1 - s1 * 0.6)
            r2 = 0.02 * (1 - s2 * 0.6)
            
            verts.extend([
                (x1 - r1, y1, z1), (x1 + r1, y1, z1), (x2, y2, z2)
            ])
            normals.extend([(0, 0, 1)] * 3)
            colors.extend([tendril_color, tendril_color, tendril_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_void_shard_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sharp angular shard from the void."""
    verts = []
    normals = []
    colors = []
    
    shard_color = (1.0, 1.0, 1.0)
    
    # Multiple jagged shards
    shards = [
        (0, 0.4, 0, 0.05),
        (0.05, 0.25, 0.03, 0.03),
        (-0.04, 0.3, -0.02, 0.04),
    ]
    
    for sx, sy, sz, sr in shards:
        # Each shard is a sharp pyramid
        for i in range(4):
            angle1 = i * math.pi / 2
            angle2 = (i + 1) * math.pi / 2
            
            verts.extend([
                (sx + math.cos(angle1) * sr, 0, sz + math.sin(angle1) * sr),
                (sx + math.cos(angle2) * sr, 0, sz + math.sin(angle2) * sr),
                (sx, sy, sz)  # Sharp tip
            ])
            
            n = (math.cos(angle1 + math.pi/4), 0.5, math.sin(angle1 + math.pi/4))
            normals.extend([n, n, n])
            colors.extend([shard_color, shard_color, shard_color])
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


def create_rainbow_spiral_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A psychedelic spiral of pure rainbow energy."""
    verts = []
    normals = []
    colors = []
    
    # Helix spiral going up
    segments = 24
    for i in range(segments):
        t1 = i / segments
        t2 = (i + 1) / segments
        angle1 = t1 * 6 * math.pi  # 3 full rotations
        angle2 = t2 * 6 * math.pi
        
        r = 0.1 - t1 * 0.05  # Tapers
        y1 = t1 * 0.6
        y2 = t2 * 0.6
        
        x1 = math.cos(angle1) * r
        z1 = math.sin(angle1) * r
        x2 = math.cos(angle2) * r
        z2 = math.sin(angle2) * r
        
        verts.extend([
            (x1, y1, z1), (x2, y2, z2), (x1 * 0.7, y1 + 0.02, z1 * 0.7)
        ])
        
        n = (x1, 0.5, z1)
        normals.extend([n, n, n])
        colors.extend([(1, 1, 1)] * 3)  # White, will be rainbow tinted
    
    return (np.array(verts, dtype='f4'), np.array(normals, dtype='f4'), np.array(colors, dtype='f4'))


# =============================================================================
# INSTANCED MESH
# =============================================================================

@dataclass
class FloraInstanceBatch:
    """A batch of flora instances sharing the same mesh."""
    mesh_type: str
    mesh_vbo: Optional[moderngl.Buffer] = None
    instance_vbo: Optional[moderngl.Buffer] = None
    vao: Optional[moderngl.VertexArray] = None
    vertex_count: int = 0
    instance_count: int = 0
    
    def release(self):
        if self.mesh_vbo:
            self.mesh_vbo.release()
        if self.instance_vbo:
            self.instance_vbo.release()
        if self.vao:
            self.vao.release()


# =============================================================================
# FLORA RENDERER
# =============================================================================

class ModernFloraRenderer:
    """
    Instanced flora renderer.
    
    Groups plants by type and renders each type with a single draw call.
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Compile shader
        self.program = ctx.program(
            vertex_shader=FLORA_VERTEX_SHADER,
            fragment_shader=FLORA_FRAGMENT_SHADER
        )
        
        # Mesh templates: type -> (verts, normals, colors)
        # Multiple tree canopy styles and branch densities for DNA-driven variety
        self.mesh_templates: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
            # Trees - many variants for genetic diversity
            'tree': create_tree_mesh(),                    # Medium branches, cone style
            'tree_sparse': create_tree_sparse_mesh(),      # Few branches (low branch_count)
            'tree_dense': create_tree_dense_mesh(),        # Many branches (high branch_count)
            'tree_tall': create_tree_tall_mesh(),          # Tall tree type
            'tree_small': create_tree_small_mesh(),        # Young/small tree
            'tree_pine': create_tree_pine_mesh(),          # Classic pine/evergreen
            'tree_oak': create_tree_oak_mesh(),            # Broad spreading oak
            'tree_dome': create_dome_tree_mesh(),          # Round layered dome
            'tree_umbrella': create_umbrella_tree_mesh(),  # Flat acacia style
            'tree_weeping': create_weeping_tree_mesh(),    # Willow style
            'tree_columnar': create_columnar_tree_mesh(),  # Tall cypress style
            'tree_palm': create_palm_tree_mesh(),          # Palm with fronds
            # Bushes
            'bush': create_bush_mesh(),
            'bush_flowering': create_bush_flowering_mesh(), # Bush with flowers
            # Ground plants
            'grass': create_grass_mesh(),
            'flower': create_flower_mesh(),
            'flower_tall': create_flower_tall_mesh(),
            'fern': create_fern_mesh(),
            'groundcover': create_groundcover_mesh(),
            'lily_pad': create_lily_pad_mesh(),
            # Special
            'mushroom': create_mushroom_mesh(),
            'cactus': create_cactus_mesh(),
            'vine': create_vine_mesh(),
            'seaweed': create_seaweed_mesh(),
            'grass_tall': create_grass_tall_mesh(),
            'coral': create_coral_mesh(),
            'spiral': create_spiral_mesh(),
            # Exotic life forms
            'octopus': create_octopus_mesh(),
            'tentacle': create_tentacle_mesh(),
            'crystal': create_crystal_mesh(),
            'alien': create_alien_mesh(),
            # Extreme mutations
            'bioluminescent': create_bioluminescent_mesh(),
            'spiral_tree': create_spiral_tree_mesh(),
            'bulbous': create_bulbous_mesh(),
            'spiky': create_spiky_mesh(),
            'droopy': create_droopy_mesh(),
            'fractal': create_fractal_mesh(),
            'tube': create_tube_mesh(),
            # === ABSTRACT/TRIPPY/WEIRD MESHES for exotic biomes ===
            'twisted_spire': create_twisted_spire_mesh(),
            'blob_creature': create_blob_creature_mesh(),
            'eye_flower': create_eye_flower_mesh(),
            'impossible_geometry': create_impossible_geometry_mesh(),
            'fire_plant': create_fire_plant_mesh(),
            'shadow_tendril': create_shadow_tendril_mesh(),
            'void_shard': create_void_shard_mesh(),
            'rainbow_spiral': create_rainbow_spiral_mesh(),
        }
        
        # Instance batches: type -> FloraInstanceBatch
        self.batches: Dict[str, FloraInstanceBatch] = {}
        
        # DNA-generated mesh cache: species_id -> (verts, normals, colors)
        self._dna_mesh_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._dna_mesh_cache_max = 50  # Reduced for memory safety
        
        # Pending instance data (before GPU upload)
        # Initialize with all mesh types from templates
        self.pending_instances: Dict[str, List[Tuple[float, float, float, float, float, float, float, float]]] = {
            key: [] for key in self.mesh_templates.keys()
        }
        
        # Camera
        self.projection = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
        self.camera_pos = glm.vec3(0, 50, 0)
        
        # Lighting
        self.light_dir = glm.vec3(0.4, 0.8, 0.3)
        self.ambient = glm.vec3(0.4, 0.42, 0.45)
        
        # Fog
        self.fog_color = glm.vec3(0.65, 0.75, 0.88)
        self.fog_start = 2000.0
        self.fog_end = 6000.0
        
        # Animation
        self.time = 0.0
        self.wind_strength = 1.0
        
        # Stats
        self.frame_stats = {
            'instances_rendered': 0,
            'draw_calls': 0,
        }
    
    def set_camera(self, projection: glm.mat4, view: glm.mat4, camera_pos: glm.vec3):
        self.projection = projection
        self.view = view
        self.camera_pos = camera_pos
    
    def add_instance(self, mesh_type: str, x: float, y: float, z: float,
                     scale: float, rotation: float,
                     color_r: float = 1.0, color_g: float = 1.0, color_b: float = 1.0):
        """Add a flora instance to be rendered."""
        # Use the actual mesh type if it exists in templates, otherwise fallback
        if mesh_type not in self.mesh_templates:
            # Map to closest available type
            if 'tree' in mesh_type:
                mesh_type = 'tree'  # Fallback for unknown tree types
            else:
                mesh_type = 'bush'  # Fallback for unknown plant types
        
        # Ensure list exists
        if mesh_type not in self.pending_instances:
            self.pending_instances[mesh_type] = []
        
        self.pending_instances[mesh_type].append(
            (x, y, z, scale, rotation, color_r, color_g, color_b)
        )
    
    def add_instance_from_dna(self, dna: Any, x: float, y: float, z: float,
                               scale: float, rotation: float):
        """Add a flora instance using DNA-generated geometry.
        
        This generates (or retrieves cached) mesh from DNA parameters.
        Provides true genetic variation in plant shapes.
        """
        # Get species_id for caching
        if hasattr(dna, 'species_id'):
            species_id = dna.species_id
        elif isinstance(dna, dict):
            species_id = dna.get('species_id', id(dna))
        else:
            species_id = id(dna)
        
        # Generate or retrieve cached mesh
        if species_id not in self._dna_mesh_cache:
            # Generate mesh from DNA
            geometry_root = plant_dna_to_geometry(dna)
            verts, norms, colors = GeometryBuilder.build_mesh(geometry_root)
            
            # Cache it (with LRU eviction if needed)
            if len(self._dna_mesh_cache) >= self._dna_mesh_cache_max:
                # Remove oldest entry
                oldest_key = next(iter(self._dna_mesh_cache))
                del self._dna_mesh_cache[oldest_key]
            
            self._dna_mesh_cache[species_id] = (verts, norms, colors)
        
        # Use species_id as mesh type key
        mesh_key = f"dna_{species_id}"
        
        # Add to mesh templates if not already there
        if mesh_key not in self.mesh_templates:
            self.mesh_templates[mesh_key] = self._dna_mesh_cache[species_id]
        
        # Ensure pending instances list exists
        if mesh_key not in self.pending_instances:
            self.pending_instances[mesh_key] = []
        
        # Get DNA colors
        if hasattr(dna, 'leaf_color'):
            color_r = dna.leaf_color.r
            color_g = dna.leaf_color.g
            color_b = dna.leaf_color.b
        elif isinstance(dna, dict):
            lc = dna.get('leaf_color', {})
            if isinstance(lc, dict):
                color_r = float(lc.get('r', 0.3))
                color_g = float(lc.get('g', 0.6))
                color_b = float(lc.get('b', 0.2))
            else:
                color_r, color_g, color_b = 0.3, 0.6, 0.2
        else:
            color_r, color_g, color_b = 0.3, 0.6, 0.2
        
        self.pending_instances[mesh_key].append(
            (x, y, z, scale, rotation, color_r, color_g, color_b)
        )
    
    def _cleanup_oldest_batches(self, keep_count: int):
        """Remove oldest DNA batches to prevent memory explosion."""
        # Get DNA batches (those starting with 'dna_')
        dna_batch_keys = [k for k in self.batches.keys() if k.startswith('dna_')]
        
        # Remove oldest ones
        to_remove = dna_batch_keys[:len(dna_batch_keys) - keep_count]
        for key in to_remove:
            batch = self.batches.pop(key, None)
            if batch:
                if batch.mesh_vbo:
                    batch.mesh_vbo.release()
                if batch.instance_vbo:
                    batch.instance_vbo.release()
                if batch.vao:
                    batch.vao.release()
            # Also clean from mesh_templates
            self.mesh_templates.pop(key, None)
            self.pending_instances.pop(key, None)
    
    def clear_instances(self):
        """Clear all pending instances."""
        for key in list(self.pending_instances.keys()):
            self.pending_instances[key] = []
    
    def upload_instances(self):
        """Upload pending instances to GPU."""
        # Copy items to avoid dict modification during iteration
        for mesh_type, instances in list(self.pending_instances.items()):
            if not instances:
                continue
            
            # Get or create batch
            if mesh_type not in self.batches:
                verts, norms, cols = self.mesh_templates[mesh_type]
                vertex_count = len(verts)
                
                # Interleave mesh data
                mesh_data = np.zeros((vertex_count, 9), dtype='f4')
                mesh_data[:, 0:3] = verts
                mesh_data[:, 3:6] = norms
                mesh_data[:, 6:9] = cols
                
                mesh_vbo = self.ctx.buffer(mesh_data.tobytes())
                
                batch = FloraInstanceBatch(
                    mesh_type=mesh_type,
                    mesh_vbo=mesh_vbo,
                    vertex_count=vertex_count
                )
                self.batches[mesh_type] = batch
                
                # Limit total batches to prevent memory explosion
                if len(self.batches) > 100:
                    self._cleanup_oldest_batches(50)
            else:
                batch = self.batches[mesh_type]
            
            # Create instance data
            instance_count = len(instances)
            instance_data = np.array(instances, dtype='f4')
            
            # Store old references for cleanup
            old_vbo = batch.instance_vbo
            old_vao = batch.vao
            
            # Create new instance buffer
            batch.instance_vbo = self.ctx.buffer(instance_data.tobytes())
            batch.instance_count = instance_count
            
            # Create new VAO
            batch.vao = self.ctx.vertex_array(
                self.program,
                [
                    (batch.mesh_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color'),
                    (batch.instance_vbo, '3f 1f 1f 3f /i', 'in_instance_pos', 
                     'in_instance_scale', 'in_instance_rot', 'in_instance_color'),
                ]
            )
            
            # Release old resources
            if old_vbo:
                old_vbo.release()
            if old_vao:
                old_vao.release()
    
    def set_wind(self, wind_dir: tuple, wind_strength: float, wind_time: float,
                  has_tornado: bool = False, tornado_center: tuple = (0, 0),
                  tornado_radius: float = 0, tornado_strength: float = 0):
        """Set wind parameters for sway animation."""
        self.wind_dir = wind_dir
        self.wind_strength = wind_strength
        self.wind_time = wind_time
        self.has_tornado = has_tornado
        self.tornado_center = tornado_center
        self.tornado_radius = tornado_radius
        self.tornado_strength = tornado_strength
    
    def render(self, dt: float = 0.016):
        """Render all flora instances."""
        self.time += dt
        self.frame_stats = {'instances_rendered': 0, 'draw_calls': 0}
        
        # Set uniforms - write glm matrices directly (column-major as OpenGL expects)
        self.program['u_projection'].write(self.projection)
        self.program['u_view'].write(self.view)
        self.program['u_camera_pos'].value = tuple(self.camera_pos)
        self.program['u_light_dir'].value = tuple(self.light_dir)
        self.program['u_ambient'].value = tuple(self.ambient)
        self.program['u_fog_start'].value = self.fog_start
        self.program['u_fog_end'].value = self.fog_end
        self.program['u_fog_color'].value = tuple(self.fog_color)
        
        # Wind uniforms
        self.program['u_wind_dir'].value = getattr(self, 'wind_dir', (1.0, 0.0))
        self.program['u_wind_strength'].value = getattr(self, 'wind_strength', 0.3)
        self.program['u_wind_time'].value = getattr(self, 'wind_time', self.time)
        self.program['u_has_tornado'].value = 1.0 if getattr(self, 'has_tornado', False) else 0.0
        self.program['u_tornado_center'].value = getattr(self, 'tornado_center', (0.0, 0.0))
        self.program['u_tornado_radius'].value = getattr(self, 'tornado_radius', 0.0)
        self.program['u_tornado_strength'].value = getattr(self, 'tornado_strength', 0.0)
        
        # Psychedelic dance mode!
        self.program['u_dance_mode'].value = 1.0 if getattr(self, 'dance_mode', False) else 0.0
        
        # Render each batch
        for mesh_type, batch in self.batches.items():
            if batch.vao and batch.instance_count > 0:
                batch.vao.render(moderngl.TRIANGLES, instances=batch.instance_count)
                self.frame_stats['instances_rendered'] += batch.instance_count
                self.frame_stats['draw_calls'] += 1
    
    def cleanup(self):
        """Release GPU resources."""
        for batch in self.batches.values():
            batch.release()
        self.batches.clear()


# =============================================================================
# TEST
# =============================================================================

def test_flora_renderer():
    """Test flora rendering with many instances."""
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL
    import time
    
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    
    screen = pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Modern Flora Test")
    
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.CULL_FACE)
    
    renderer = ModernFloraRenderer(ctx)
    
    print("Generating flora instances...")
    
    # Generate lots of flora
    rng = np.random.default_rng(42)
    world_size = 500
    
    num_trees = 5000
    num_bushes = 8000
    num_grass = 20000
    num_flowers = 3000
    
    # Trees
    for _ in range(num_trees):
        x = rng.random() * world_size
        z = rng.random() * world_size
        y = np.sin(x * 0.02) * np.cos(z * 0.02) * 10 + 10
        scale = 0.8 + rng.random() * 1.5
        rot = rng.random() * math.pi * 2
        # Slight color variation
        g = 0.85 + rng.random() * 0.3
        renderer.add_instance('tree', x, y, z, scale, rot, 1.0, g, 1.0)
    
    # Bushes
    for _ in range(num_bushes):
        x = rng.random() * world_size
        z = rng.random() * world_size
        y = np.sin(x * 0.02) * np.cos(z * 0.02) * 10 + 10
        scale = 0.5 + rng.random() * 1.0
        rot = rng.random() * math.pi * 2
        g = 0.9 + rng.random() * 0.2
        renderer.add_instance('bush', x, y, z, scale, rot, 1.0, g, 1.0)
    
    # Grass
    for _ in range(num_grass):
        x = rng.random() * world_size
        z = rng.random() * world_size
        y = np.sin(x * 0.02) * np.cos(z * 0.02) * 10 + 10
        scale = 0.8 + rng.random() * 0.5
        rot = rng.random() * math.pi * 2
        renderer.add_instance('grass', x, y, z, scale, rot, 1.0, 1.0, 1.0)
    
    # Flowers
    for _ in range(num_flowers):
        x = rng.random() * world_size
        z = rng.random() * world_size
        y = np.sin(x * 0.02) * np.cos(z * 0.02) * 10 + 10
        scale = 0.7 + rng.random() * 0.6
        rot = rng.random() * math.pi * 2
        # Random flower colors
        r = 0.7 + rng.random() * 0.5
        g = 0.5 + rng.random() * 0.5
        b = 0.5 + rng.random() * 0.7
        renderer.add_instance('flower', x, y, z, scale, rot, r, g, b)
    
    renderer.upload_instances()
    
    total_instances = num_trees + num_bushes + num_grass + num_flowers
    print(f"Created {total_instances:,} flora instances")
    print(f"  Trees: {num_trees:,}, Bushes: {num_bushes:,}")
    print(f"  Grass: {num_grass:,}, Flowers: {num_flowers:,}")
    
    # Camera
    cam_x, cam_y, cam_z = world_size / 2, 50, world_size / 2
    yaw, pitch = 0, -15
    
    clock = pygame.time.Clock()
    running = True
    frame_times = []
    
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    print("Controls: WASD=move, Mouse=look, ESC=quit")
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                yaw += event.rel[0] * 0.2
                pitch -= event.rel[1] * 0.2
                pitch = max(-89, min(89, pitch))
        
        keys = pygame.key.get_pressed()
        speed = 80 * dt
        
        yaw_rad = np.radians(yaw)
        fx, fz = -np.sin(yaw_rad), -np.cos(yaw_rad)
        rx, rz = np.cos(yaw_rad), -np.sin(yaw_rad)
        
        if keys[pygame.K_w]: cam_x += fx * speed; cam_z += fz * speed
        if keys[pygame.K_s]: cam_x -= fx * speed; cam_z -= fz * speed
        if keys[pygame.K_a]: cam_x -= rx * speed; cam_z -= rz * speed
        if keys[pygame.K_d]: cam_x += rx * speed; cam_z += rz * speed
        if keys[pygame.K_SPACE]: cam_y += speed
        if keys[pygame.K_LSHIFT]: cam_y -= speed
        
        # Update camera
        pitch_rad = np.radians(pitch)
        dir_x = -np.sin(yaw_rad) * np.cos(pitch_rad)
        dir_y = np.sin(pitch_rad)
        dir_z = -np.cos(yaw_rad) * np.cos(pitch_rad)
        
        projection = glm.perspective(glm.radians(70), 1280/720, 0.1, 1000.0)
        view = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z),
            glm.vec3(cam_x + dir_x, cam_y + dir_y, cam_z + dir_z),
            glm.vec3(0, 1, 0)
        )
        
        renderer.set_camera(projection, view, glm.vec3(cam_x, cam_y, cam_z))
        
        # Render
        ctx.clear(0.55, 0.70, 0.85)
        
        start = time.perf_counter()
        renderer.render(dt)
        ctx.finish()
        elapsed = time.perf_counter() - start
        frame_times.append(elapsed)
        
        pygame.display.flip()
        
        if len(frame_times) >= 60:
            avg_ms = np.mean(frame_times) * 1000
            fps = 1000 / avg_ms
            print(f"FPS: {fps:.0f} | {renderer.frame_stats['instances_rendered']:,} instances | "
                  f"{renderer.frame_stats['draw_calls']} draws")
            frame_times = []
    
    renderer.cleanup()
    pygame.quit()


if __name__ == "__main__":
    test_flora_renderer()

