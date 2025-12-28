"""
ModernGL Animal Renderer

Instanced rendering for animals/fauna with:
- GPU-based animation (walking, flying, swimming)
- Per-instance color tinting from DNA
- DNA-based mesh generation for genetic variation
- Efficient batched drawing
- Distance-based LOD
"""

import numpy as np
import moderngl
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import math

try:
    from pyglm import glm
except ImportError:
    import glm

# Import DNA mesh generator for genetic variation
from .dna_animal_mesh import generate_animal_mesh, clear_animal_mesh_cache


# =============================================================================
# SHADERS
# =============================================================================

ANIMAL_VERTEX_SHADER = """
#version 330 core

// Per-vertex attributes (mesh)
in vec3 in_position;
in vec3 in_normal;

// Per-instance attributes
in vec3 in_instance_pos;      // World position (x, y, z)
in vec4 in_instance_data;     // scale, rotation, animation_phase, type_id
in vec4 in_instance_color;    // r, g, b, alpha

out vec3 v_normal;
out vec3 v_color;
out vec3 v_world_pos;
out float v_fog_factor;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec3 u_camera_pos;
uniform float u_time;
uniform float u_fog_start;
uniform float u_fog_end;

// Animation helpers
vec3 animate_position(vec3 pos, float anim_phase, float type_id, float scale) {
    // Walking bob for ground animals (type 0-2)
    if (type_id < 3.0) {
        float bob = sin(anim_phase * 6.28318 + u_time * 4.0) * 0.1 * scale;
        pos.y += bob;
    }
    // Flying motion for birds (type 3)
    else if (type_id < 4.0) {
        float flap = sin(anim_phase * 6.28318 + u_time * 8.0) * 0.15;
        pos.y += flap * scale;
        // Wing motion - scale X based on height in mesh
        if (pos.y > 0.3) {
            pos.x *= 1.0 + flap * 0.3;
        }
    }
    // Swimming motion for fish (type 4)
    else if (type_id < 5.0) {
        float wiggle = sin(anim_phase * 6.28318 + u_time * 6.0 + pos.z * 2.0) * 0.1;
        pos.x += wiggle * scale;
    }
    return pos;
}

void main() {
    float scale = in_instance_data.x;
    float rotation = in_instance_data.y;
    float anim_phase = in_instance_data.z;
    float type_id = in_instance_data.w;
    
    // Animate the vertex position
    vec3 animated_pos = animate_position(in_position, anim_phase, type_id, scale);
    
    // Apply scale
    vec3 scaled = animated_pos * scale;
    
    // Apply rotation (around Y axis)
    float c = cos(rotation);
    float s = sin(rotation);
    vec3 rotated = vec3(
        scaled.x * c - scaled.z * s,
        scaled.y,
        scaled.x * s + scaled.z * c
    );
    
    // Translate to world position
    vec3 world_pos = rotated + in_instance_pos;
    v_world_pos = world_pos;
    
    // Transform normal
    vec3 rotated_normal = vec3(
        in_normal.x * c - in_normal.z * s,
        in_normal.y,
        in_normal.x * s + in_normal.z * c
    );
    v_normal = rotated_normal;
    
    // Pass color
    v_color = in_instance_color.rgb;
    
    // Calculate fog
    float dist = length(world_pos - u_camera_pos);
    v_fog_factor = clamp((dist - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);
    
    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
}
"""

ANIMAL_FRAGMENT_SHADER = """
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
    // Simple directional lighting
    vec3 norm = normalize(v_normal);
    float diff = max(dot(norm, normalize(u_light_dir)), 0.0);
    
    // Combine lighting with color
    vec3 lit_color = v_color * (u_ambient + diff * 0.7);
    
    // Apply fog
    vec3 final_color = mix(lit_color, u_fog_color, v_fog_factor);
    
    fragColor = vec4(final_color, 1.0);
}
"""


# =============================================================================
# MESH GENERATION
# =============================================================================

def create_worm_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a segmented worm/snake mesh with proper triangles."""
    vertices = []
    normals = []
    
    segments = 6
    sides = 6
    length = 1.5
    radius = 0.12
    
    for i in range(segments):
        t0 = i / segments
        t1 = (i + 1) / segments
        z0 = t0 * length - length / 2
        z1 = t1 * length - length / 2
        
        # Taper at ends
        r0 = radius * (1.0 - abs(t0 - 0.5) * 0.6)
        r1 = radius * (1.0 - abs(t1 - 0.5) * 0.6)
        
        for j in range(sides):
            a0 = j * 2 * math.pi / sides
            a1 = (j + 1) * 2 * math.pi / sides
            
            # Four corners of this quad section
            x00 = math.cos(a0) * r0
            y00 = math.sin(a0) * r0 + r0
            x10 = math.cos(a1) * r0
            y10 = math.sin(a1) * r0 + r0
            x01 = math.cos(a0) * r1
            y01 = math.sin(a0) * r1 + r1
            x11 = math.cos(a1) * r1
            y11 = math.sin(a1) * r1 + r1
            
            # Two triangles per quad
            vertices.extend([
                [x00, y00, z0], [x10, y10, z0], [x01, y01, z1],
                [x10, y10, z0], [x11, y11, z1], [x01, y01, z1],
            ])
            
            # Normals pointing outward
            n0 = [math.cos(a0), math.sin(a0), 0]
            n1 = [math.cos(a1), math.sin(a1), 0]
            normals.extend([n0, n1, n0, n1, n1, n0])
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_bird_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple bird mesh with wings using proper triangles."""
    vertices = []
    normals = []
    
    # Body as a simple pyramid/tetrahedron
    # Beak (front), tail (back), top, left wing, right wing
    beak = [0, 0.3, 0.4]
    tail = [0, 0.3, -0.4]
    top = [0, 0.5, 0]
    left = [-0.15, 0.25, 0]
    right = [0.15, 0.25, 0]
    
    # Body triangles (6 for diamond shape)
    body_tris = [
        [beak, top, right],      # Front-top-right
        [beak, left, top],       # Front-top-left
        [beak, right, left],     # Front-bottom (belly)
        [tail, right, top],      # Back-top-right
        [tail, top, left],       # Back-top-left
        [tail, left, right],     # Back-bottom (belly)
    ]
    
    for tri in body_tris:
        vertices.extend(tri)
        normals.extend([[0, 1, 0]] * 3)
    
    # Wings as flat triangles
    wing_left = [
        [-0.15, 0.32, 0.1], [-0.5, 0.35, 0], [-0.15, 0.32, -0.15]
    ]
    wing_right = [
        [0.15, 0.32, 0.1], [0.15, 0.32, -0.15], [0.5, 0.35, 0]
    ]
    
    vertices.extend(wing_left)
    normals.extend([[0, 1, 0]] * 3)
    vertices.extend(wing_right)
    normals.extend([[0, 1, 0]] * 3)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_mammal_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple quadruped mesh with proper triangles."""
    vertices = []
    normals = []
    
    def add_box(cx, cy, cz, hw, hh, hd):
        """Add a box (12 triangles) centered at cx, cy, cz with half-sizes."""
        # 8 corners
        c = [
            [cx-hw, cy-hh, cz-hd], [cx+hw, cy-hh, cz-hd],
            [cx+hw, cy+hh, cz-hd], [cx-hw, cy+hh, cz-hd],
            [cx-hw, cy-hh, cz+hd], [cx+hw, cy-hh, cz+hd],
            [cx+hw, cy+hh, cz+hd], [cx-hw, cy+hh, cz+hd],
        ]
        # 6 faces as 12 triangles (indices)
        faces = [
            (0,1,2), (0,2,3),  # Back
            (5,4,7), (5,7,6),  # Front
            (4,0,3), (4,3,7),  # Left
            (1,5,6), (1,6,2),  # Right
            (3,2,6), (3,6,7),  # Top
            (4,5,1), (4,1,0),  # Bottom
        ]
        face_normals = [
            [0,0,-1], [0,0,-1], [0,0,1], [0,0,1],
            [-1,0,0], [-1,0,0], [1,0,0], [1,0,0],
            [0,1,0], [0,1,0], [0,-1,0], [0,-1,0],
        ]
        for i, (a,b,cc) in enumerate(faces):
            vertices.extend([c[a], c[b], c[cc]])
            normals.extend([face_normals[i]] * 3)
    
    # Body
    add_box(0, 0.45, 0, 0.25, 0.15, 0.4)
    
    # Head
    add_box(0, 0.55, 0.5, 0.12, 0.12, 0.15)
    
    # 4 legs
    add_box(-0.18, 0.15, 0.3, 0.05, 0.15, 0.05)   # Front left
    add_box(0.18, 0.15, 0.3, 0.05, 0.15, 0.05)    # Front right
    add_box(-0.18, 0.15, -0.3, 0.05, 0.15, 0.05)  # Back left
    add_box(0.18, 0.15, -0.3, 0.05, 0.15, 0.05)   # Back right
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_fish_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a fish-shaped mesh with proper triangles."""
    vertices = []
    normals = []
    
    # Fish body as a diamond (8 triangles)
    nose = [0, 0.1, 0.4]
    tail = [0, 0.1, -0.4]
    top = [0, 0.25, 0]
    bottom = [0, 0, 0]
    left = [-0.15, 0.1, 0]
    right = [0.15, 0.1, 0]
    
    body_tris = [
        # Front half
        [nose, top, right], [nose, left, top],
        [nose, right, bottom], [nose, bottom, left],
        # Back half
        [tail, right, top], [tail, top, left],
        [tail, bottom, right], [tail, left, bottom],
    ]
    
    for tri in body_tris:
        vertices.extend(tri)
        normals.extend([[0, 1, 0]] * 3)
    
    # Tail fin (2 triangles forming V)
    tail_top = [[0, 0.1, -0.4], [0.15, 0.2, -0.55], [0, 0.1, -0.5]]
    tail_bot = [[0, 0.1, -0.4], [0, 0.1, -0.5], [0.15, 0, -0.55]]
    vertices.extend(tail_top)
    vertices.extend(tail_bot)
    normals.extend([[1, 0, 0]] * 6)
    
    # Dorsal fin (1 triangle)
    dorsal = [[0, 0.25, 0.1], [0, 0.35, 0], [0, 0.25, -0.1]]
    vertices.extend(dorsal)
    normals.extend([[1, 0, 0]] * 3)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_insect_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple insect/bug mesh with proper triangles."""
    vertices = []
    normals = []
    
    # Three body segments as simple ellipsoids (8 triangles each)
    def add_segment(cz, r, h):
        # Simple 8-sided "sphere" approximation
        top = [0, h + 0.05, cz]
        bottom = [0, 0.05, cz]
        for i in range(8):
            a0 = i * math.pi / 4
            a1 = (i + 1) * math.pi / 4
            mid0 = [math.cos(a0) * r, h/2 + 0.05, cz + math.sin(a0) * r * 0.3]
            mid1 = [math.cos(a1) * r, h/2 + 0.05, cz + math.sin(a1) * r * 0.3]
            
            # Top cone triangle
            vertices.extend([top, mid0, mid1])
            normals.extend([[0, 1, 0]] * 3)
            # Bottom cone triangle
            vertices.extend([bottom, mid1, mid0])
            normals.extend([[0, -1, 0]] * 3)
    
    # Head (small), thorax (medium), abdomen (larger)
    add_segment(0.15, 0.06, 0.08)   # Head
    add_segment(0, 0.08, 0.1)        # Thorax
    add_segment(-0.18, 0.1, 0.12)    # Abdomen
    
    # 6 legs as thin triangles
    for side in [-1, 1]:
        for zoff in [-0.05, 0, 0.05]:
            vertices.extend([
                [side * 0.08, 0.08, zoff],
                [side * 0.2, 0.02, zoff],
                [side * 0.08, 0.06, zoff],
            ])
            normals.extend([[0, 1, 0]] * 3)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_reptile_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a reptile/lizard mesh with long body and tail."""
    vertices = []
    normals = []
    
    # Long body - 3 connected segments
    segments = [(0.25, 0.06, 0.04), (0, 0.08, 0.06), (-0.25, 0.06, 0.04)]
    for z, r, h in segments:
        for i in range(6):
            a0 = i * math.pi / 3
            a1 = (i + 1) * math.pi / 3
            top = [0, h + 0.02, z]
            mid0 = [math.cos(a0) * r, h/2 + 0.02, z + math.sin(a0) * r * 0.5]
            mid1 = [math.cos(a1) * r, h/2 + 0.02, z + math.sin(a1) * r * 0.5]
            vertices.extend([top, mid0, mid1])
            normals.extend([[0, 1, 0]] * 3)
    
    # Long tail
    for i in range(4):
        z = -0.35 - i * 0.12
        r = 0.03 * (1 - i * 0.2)
        vertices.extend([
            [0, 0.03, z], [r, 0.02, z - 0.1], [-r, 0.02, z - 0.1]
        ])
        normals.extend([[0, 1, 0]] * 3)
    
    # 4 stubby legs
    for side in [-1, 1]:
        for zoff in [0.1, -0.15]:
            vertices.extend([
                [side * 0.08, 0.04, zoff],
                [side * 0.15, 0.0, zoff + 0.03],
                [side * 0.15, 0.0, zoff - 0.03],
            ])
            normals.extend([[side, 0, 0]] * 3)
    
    # Head
    vertices.extend([
        [0, 0.06, 0.35], [-0.04, 0.02, 0.28], [0.04, 0.02, 0.28]
    ])
    normals.extend([[0, 0.5, 0.866]] * 3)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_amphibian_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a frog/toad mesh with big eyes and powerful hind legs."""
    vertices = []
    normals = []
    
    # Rounded body (dome)
    for i in range(8):
        a0 = i * math.pi / 4
        a1 = (i + 1) * math.pi / 4
        vertices.extend([
            [0, 0.15, 0],
            [math.cos(a0) * 0.12, 0.02, math.sin(a0) * 0.1],
            [math.cos(a1) * 0.12, 0.02, math.sin(a1) * 0.1]
        ])
        normals.extend([[0, 1, 0]] * 3)
    
    # Big bulging eyes
    for side in [-1, 1]:
        eye_x = side * 0.06
        for i in range(4):
            a0 = i * math.pi / 2
            a1 = (i + 1) * math.pi / 2
            vertices.extend([
                [eye_x, 0.2, 0.08],
                [eye_x + math.cos(a0) * 0.03, 0.15, 0.08 + math.sin(a0) * 0.03],
                [eye_x + math.cos(a1) * 0.03, 0.15, 0.08 + math.sin(a1) * 0.03]
            ])
            normals.extend([[0, 1, 0]] * 3)
    
    # Front legs (small)
    for side in [-1, 1]:
        vertices.extend([
            [side * 0.1, 0.03, 0.06],
            [side * 0.18, 0.0, 0.1],
            [side * 0.18, 0.0, 0.02],
        ])
        normals.extend([[0, 0, 1]] * 3)
    
    # Back legs (big and powerful)
    for side in [-1, 1]:
        # Thigh
        vertices.extend([
            [side * 0.1, 0.05, -0.08],
            [side * 0.2, 0.02, -0.12],
            [side * 0.15, 0.0, -0.04],
        ])
        # Lower leg
        vertices.extend([
            [side * 0.2, 0.02, -0.12],
            [side * 0.28, 0.0, -0.05],
            [side * 0.22, 0.0, -0.15],
        ])
        normals.extend([[0, 1, 0]] * 6)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_crab_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a crab mesh with wide body and claws."""
    vertices = []
    normals = []
    
    # Wide flat body
    for i in range(6):
        a0 = i * math.pi / 3
        a1 = (i + 1) * math.pi / 3
        vertices.extend([
            [0, 0.08, 0],
            [math.cos(a0) * 0.15, 0.02, math.sin(a0) * 0.1],
            [math.cos(a1) * 0.15, 0.02, math.sin(a1) * 0.1]
        ])
        normals.extend([[0, 1, 0]] * 3)
    
    # 8 legs (4 on each side)
    for side in [-1, 1]:
        for i, zoff in enumerate([-0.06, -0.02, 0.02, 0.06]):
            leg_len = 0.12 + (abs(i - 1.5)) * 0.03
            vertices.extend([
                [side * 0.12, 0.04, zoff],
                [side * (0.12 + leg_len), 0.0, zoff + 0.02],
                [side * (0.12 + leg_len), 0.0, zoff - 0.02],
            ])
            normals.extend([[side, 0, 0]] * 3)
    
    # Claws (front)
    for side in [-1, 1]:
        # Arm
        vertices.extend([
            [side * 0.1, 0.05, 0.1],
            [side * 0.18, 0.04, 0.15],
            [side * 0.15, 0.02, 0.08],
        ])
        # Claw pincer
        vertices.extend([
            [side * 0.18, 0.06, 0.15],
            [side * 0.25, 0.04, 0.2],
            [side * 0.22, 0.02, 0.13],
        ])
        vertices.extend([
            [side * 0.18, 0.03, 0.15],
            [side * 0.22, 0.02, 0.2],
            [side * 0.25, 0.04, 0.15],
        ])
        normals.extend([[0, 1, 0]] * 9)
    
    # Eyes on stalks
    for side in [-1, 1]:
        vertices.extend([
            [side * 0.04, 0.08, 0.1],
            [side * 0.05, 0.14, 0.12],
            [side * 0.03, 0.08, 0.1],
        ])
        normals.extend([[0, 1, 0]] * 3)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_jellyfish_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a jellyfish mesh with dome and tentacles."""
    vertices = []
    normals = []
    
    # Bell/dome top
    for i in range(8):
        a0 = i * math.pi / 4
        a1 = (i + 1) * math.pi / 4
        r = 0.15
        vertices.extend([
            [0, 0.2, 0],
            [math.cos(a0) * r, 0.05, math.sin(a0) * r],
            [math.cos(a1) * r, 0.05, math.sin(a1) * r]
        ])
        normals.extend([[0, 1, 0]] * 3)
    
    # Inner bell (darker)
    for i in range(8):
        a0 = i * math.pi / 4
        a1 = (i + 1) * math.pi / 4
        r = 0.12
        vertices.extend([
            [0, 0.08, 0],
            [math.cos(a1) * r, 0.05, math.sin(a1) * r],
            [math.cos(a0) * r, 0.05, math.sin(a0) * r]
        ])
        normals.extend([[0, -1, 0]] * 3)
    
    # Tentacles (8 hanging down)
    for i in range(8):
        angle = i * math.pi / 4
        x = math.cos(angle) * 0.1
        z = math.sin(angle) * 0.1
        length = 0.2 + (i % 3) * 0.1  # Varying lengths
        
        vertices.extend([
            [x - 0.01, 0.05, z],
            [x + 0.01, 0.05, z],
            [x + (i % 2) * 0.03 - 0.015, -length, z]
        ])
        normals.extend([[0, 0, 1]] * 3)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_snake_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a snake mesh - long sinuous body."""
    vertices = []
    normals = []
    
    # Body as series of connected segments with wave pattern
    num_segments = 10
    for i in range(num_segments):
        t = i / num_segments
        # Sinuous wave
        x = math.sin(t * math.pi * 2) * 0.1
        z = t * 0.6 - 0.3  # Length
        r = 0.04 * (1 - t * 0.5)  # Tapers toward tail
        
        for j in range(4):
            a0 = j * math.pi / 2
            a1 = (j + 1) * math.pi / 2
            vertices.extend([
                [x, 0.06, z],
                [x + math.cos(a0) * r, 0.04 + math.sin(a0) * r * 0.5, z],
                [x + math.cos(a1) * r, 0.04 + math.sin(a1) * r * 0.5, z]
            ])
            normals.extend([[0, 1, 0]] * 3)
    
    # Head (triangular)
    vertices.extend([
        [0, 0.07, -0.35],
        [-0.03, 0.04, -0.28],
        [0.03, 0.04, -0.28]
    ])
    normals.extend([[0, 0.5, -0.866]] * 3)
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


# =============================================================================
# RENDERER
# =============================================================================

@dataclass
class AnimalInstance:
    """Data for a single animal instance."""
    x: float
    y: float
    z: float
    scale: float
    rotation: float
    anim_phase: float
    type_id: int  # 0=worm, 1=mammal, 2=insect, 3=bird, 4=fish
    r: float
    g: float
    b: float


class ModernAnimalRenderer:
    """
    GPU-instanced animal renderer using ModernGL.
    """
    
    # Animal type IDs
    TYPE_WORM = 0
    TYPE_MAMMAL = 1
    TYPE_INSECT = 2
    TYPE_BIRD = 3
    TYPE_FISH = 4
    TYPE_REPTILE = 5
    TYPE_AMPHIBIAN = 6
    TYPE_CRAB = 7
    TYPE_JELLYFISH = 8
    TYPE_SNAKE = 9
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Compile shaders
        self.program = ctx.program(
            vertex_shader=ANIMAL_VERTEX_SHADER,
            fragment_shader=ANIMAL_FRAGMENT_SHADER,
        )
        
        # Create mesh VBOs for each animal type
        self.meshes: Dict[int, Tuple[moderngl.Buffer, int]] = {}
        self._create_meshes()
        
        # Instance data buffer (dynamic)
        self.max_instances = 10000
        self.instance_buffer = ctx.buffer(reserve=self.max_instances * 11 * 4)  # 11 floats per instance
        
        # Create VAOs for each mesh type
        self.vaos: Dict[int, moderngl.VertexArray] = {}
        self._create_vaos()
        
        # Instance data storage
        self.instances: Dict[int, List[AnimalInstance]] = {
            self.TYPE_WORM: [],
            self.TYPE_MAMMAL: [],
            self.TYPE_INSECT: [],
            self.TYPE_BIRD: [],
            self.TYPE_FISH: [],
            self.TYPE_REPTILE: [],
            self.TYPE_AMPHIBIAN: [],
            self.TYPE_CRAB: [],
            self.TYPE_JELLYFISH: [],
            self.TYPE_SNAKE: [],
        }
        
        # Uniforms
        self.projection = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
        self.camera_pos = glm.vec3(0, 0, 0)
        self.light_dir = glm.vec3(0.5, 1.0, 0.3)
        self.ambient = glm.vec3(0.4, 0.4, 0.5)
        self.fog_color = glm.vec3(0.7, 0.8, 0.9)
        self.fog_start = 2000.0
        self.fog_end = 6000.0
        self.time = 0.0
        
        # Stats
        self.frame_stats = {
            'instances_rendered': 0,
            'draw_calls': 0,
        }
        
        # DNA-based mesh batches (species_id -> (vao, vbo, instance_list))
        self.dna_batches: Dict[int, Tuple[moderngl.VertexArray, moderngl.Buffer, moderngl.Buffer, int, List]] = {}
        self._dna_mesh_cache_max = 100
        
        # Create DNA program (with color vertex attribute)
        self.dna_program = self._create_dna_program()
    
    def _create_dna_program(self):
        """Create shader program for DNA-based meshes (with per-vertex color)."""
        vert = """
        #version 330 core
        
        in vec3 in_position;
        in vec3 in_normal;
        in vec3 in_color;
        
        // Per-instance data
        in vec3 in_instance_pos;
        in vec4 in_instance_data;  // scale, rotation, anim_phase, _
        
        out vec3 v_normal;
        out vec3 v_color;
        out vec3 v_world_pos;
        out float v_fog_factor;
        
        uniform mat4 u_projection;
        uniform mat4 u_view;
        uniform vec3 u_camera_pos;
        uniform float u_time;
        uniform float u_fog_start;
        uniform float u_fog_end;
        
        mat3 rotateY(float angle) {
            float c = cos(angle);
            float s = sin(angle);
            return mat3(c, 0, s, 0, 1, 0, -s, 0, c);
        }
        
        void main() {
            float scale = in_instance_data.x;
            float rotation = in_instance_data.y;
            float anim_phase = in_instance_data.z;
            
            // Apply rotation
            vec3 pos = rotateY(rotation) * in_position;
            
            // Scale and translate
            vec3 world_pos = pos * scale + in_instance_pos;
            
            // Simple bounce animation based on phase
            float bounce = sin(u_time * 3.0 + anim_phase * 6.28) * 0.05 * scale;
            world_pos.y += bounce;
            
            gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
            
            // Transform normal
            v_normal = rotateY(rotation) * in_normal;
            v_color = in_color;
            v_world_pos = world_pos;
            
            // Fog
            float dist = distance(u_camera_pos, world_pos);
            v_fog_factor = clamp((dist - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);
        }
        """
        
        frag = """
        #version 330 core
        
        in vec3 v_normal;
        in vec3 v_color;
        in vec3 v_world_pos;
        in float v_fog_factor;
        
        out vec4 f_color;
        
        uniform vec3 u_light_dir;
        uniform vec3 u_ambient;
        uniform vec3 u_fog_color;
        
        void main() {
            vec3 normal = normalize(v_normal);
            vec3 light = normalize(u_light_dir);
            
            float diff = max(dot(normal, light), 0.0);
            vec3 color = v_color * (u_ambient + diff * 0.6);
            
            // Apply fog
            color = mix(color, u_fog_color, v_fog_factor);
            
            f_color = vec4(color, 1.0);
        }
        """
        
        return self.ctx.program(vertex_shader=vert, fragment_shader=frag)
    
    def add_dna_instance(self, dna: Any, x: float, y: float, z: float,
                         scale: float, rotation: float, anim_phase: float):
        """Add an animal instance using DNA-generated mesh."""
        # Get species ID for batching
        if hasattr(dna, 'species_id'):
            species_id = dna.species_id
        elif isinstance(dna, dict):
            species_id = dna.get('species_id', id(dna))
        else:
            species_id = id(dna)
        
        # Create batch if needed
        if species_id not in self.dna_batches:
            self._create_dna_batch(species_id, dna)
        
        # Add instance to batch
        if species_id in self.dna_batches:
            _, _, _, _, instances = self.dna_batches[species_id]
            instances.append((x, y, z, scale, rotation, anim_phase))
    
    def _create_dna_batch(self, species_id: int, dna: Any):
        """Create a rendering batch for a DNA species."""
        # Limit cache size
        if len(self.dna_batches) >= self._dna_mesh_cache_max:
            # Remove oldest
            oldest = next(iter(self.dna_batches))
            vao, mesh_vbo, inst_vbo, _, _ = self.dna_batches[oldest]
            vao.release()
            mesh_vbo.release()
            inst_vbo.release()
            del self.dna_batches[oldest]
        
        try:
            # Generate mesh from DNA
            verts, norms, colors = generate_animal_mesh(dna)
            
            if len(verts) == 0:
                return
            
            # Create mesh VBO (position + normal + color)
            mesh_data = np.zeros((len(verts), 9), dtype='f4')
            mesh_data[:, 0:3] = verts
            mesh_data[:, 3:6] = norms
            mesh_data[:, 6:9] = colors
            mesh_vbo = self.ctx.buffer(mesh_data.tobytes())
            
            # Create instance buffer
            inst_vbo = self.ctx.buffer(reserve=1000 * 7 * 4)  # 7 floats per instance (pos, data)
            
            # Create VAO
            vao = self.ctx.vertex_array(
                self.dna_program,
                [
                    (mesh_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color'),
                    (inst_vbo, '3f 4f /i', 'in_instance_pos', 'in_instance_data'),
                ],
            )
            
            self.dna_batches[species_id] = (vao, mesh_vbo, inst_vbo, len(verts), [])
            
        except Exception as e:
            print(f"Error creating DNA animal batch: {e}")
    
    def clear_dna_instances(self):
        """Clear all DNA-based instances (keep batches for reuse)."""
        for species_id in self.dna_batches:
            _, _, _, _, instances = self.dna_batches[species_id]
            instances.clear()
    
    def _create_meshes(self):
        """Create VBOs for each animal mesh type."""
        mesh_creators = {
            self.TYPE_WORM: create_worm_mesh,
            self.TYPE_MAMMAL: create_mammal_mesh,
            self.TYPE_INSECT: create_insect_mesh,
            self.TYPE_BIRD: create_bird_mesh,
            self.TYPE_FISH: create_fish_mesh,
            self.TYPE_REPTILE: create_reptile_mesh,
            self.TYPE_AMPHIBIAN: create_amphibian_mesh,
            self.TYPE_CRAB: create_crab_mesh,
            self.TYPE_JELLYFISH: create_jellyfish_mesh,
            self.TYPE_SNAKE: create_snake_mesh,
        }
        
        for type_id, creator in mesh_creators.items():
            verts, norms = creator()
            
            # Interleave vertex data
            data = np.hstack([verts, norms]).astype('f4')
            vbo = self.ctx.buffer(data.tobytes())
            
            self.meshes[type_id] = (vbo, len(verts))
    
    def _create_vaos(self):
        """Create VAOs that combine mesh VBOs with instance buffer."""
        for type_id, (mesh_vbo, _) in self.meshes.items():
            vao = self.ctx.vertex_array(
                self.program,
                [
                    (mesh_vbo, '3f 3f', 'in_position', 'in_normal'),
                    (self.instance_buffer, '3f 4f 4f /i', 
                     'in_instance_pos', 'in_instance_data', 'in_instance_color'),
                ],
            )
            self.vaos[type_id] = vao
    
    def clear_instances(self):
        """Clear all animal instances."""
        for type_id in self.instances:
            self.instances[type_id].clear()
    
    def add_instance(self, type_id: int, x: float, y: float, z: float,
                    scale: float, rotation: float, anim_phase: float,
                    r: float, g: float, b: float):
        """Add an animal instance."""
        if type_id not in self.instances:
            type_id = self.TYPE_WORM  # Default
        
        self.instances[type_id].append(AnimalInstance(
            x, y, z, scale, rotation, anim_phase, type_id, r, g, b
        ))
    
    def set_camera(self, projection: glm.mat4, view: glm.mat4, camera_pos: glm.vec3):
        """Set camera matrices."""
        self.projection = projection
        self.view = view
        self.camera_pos = camera_pos
    
    def render(self, dt: float = 0.016):
        """Render all animal instances."""
        self.time += dt
        self.frame_stats['instances_rendered'] = 0
        self.frame_stats['draw_calls'] = 0
        
        # Set common uniforms
        self.program['u_projection'].write(self.projection)
        self.program['u_view'].write(self.view)
        self.program['u_camera_pos'].write(self.camera_pos)
        self.program['u_time'].value = self.time
        self.program['u_light_dir'].write(self.light_dir)
        self.program['u_ambient'].write(self.ambient)
        self.program['u_fog_color'].write(self.fog_color)
        self.program['u_fog_start'].value = self.fog_start
        self.program['u_fog_end'].value = self.fog_end
        
        # Render each type
        for type_id, instances in self.instances.items():
            if not instances:
                continue
            
            # Build instance data array
            instance_data = []
            for inst in instances:
                instance_data.extend([
                    inst.x, inst.y, inst.z,  # position
                    inst.scale, inst.rotation, inst.anim_phase, float(inst.type_id),  # data
                    inst.r, inst.g, inst.b, 1.0,  # color
                ])
            
            # Upload instance data
            data = np.array(instance_data, dtype='f4')
            self.instance_buffer.write(data.tobytes())
            
            # Render with instancing
            _, vertex_count = self.meshes[type_id]
            vao = self.vaos[type_id]
            vao.render(moderngl.TRIANGLES, vertices=vertex_count, instances=len(instances))
            
            self.frame_stats['instances_rendered'] += len(instances)
            self.frame_stats['draw_calls'] += 1
        
        # Render DNA-based animals
        self._render_dna_batches()
    
    def _render_dna_batches(self):
        """Render all DNA-based animal batches."""
        if not self.dna_batches:
            return
        
        # Set uniforms for DNA program
        self.dna_program['u_projection'].write(self.projection)
        self.dna_program['u_view'].write(self.view)
        self.dna_program['u_camera_pos'].write(self.camera_pos)
        self.dna_program['u_time'].value = self.time
        self.dna_program['u_light_dir'].write(self.light_dir)
        self.dna_program['u_ambient'].write(self.ambient)
        self.dna_program['u_fog_color'].write(self.fog_color)
        self.dna_program['u_fog_start'].value = self.fog_start
        self.dna_program['u_fog_end'].value = self.fog_end
        
        for species_id, (vao, mesh_vbo, inst_vbo, vertex_count, instances) in self.dna_batches.items():
            if not instances:
                continue
            
            # Build instance data: x, y, z, scale, rotation, anim_phase, 0 (padding)
            instance_data = []
            for x, y, z, scale, rotation, anim_phase in instances:
                instance_data.extend([x, y, z, scale, rotation, anim_phase, 0])
            
            # Upload instance data
            data = np.array(instance_data, dtype='f4')
            inst_vbo.write(data.tobytes())
            
            # Render
            vao.render(moderngl.TRIANGLES, vertices=vertex_count, instances=len(instances))
            
            self.frame_stats['instances_rendered'] += len(instances)
            self.frame_stats['draw_calls'] += 1
    
    def cleanup(self):
        """Release GPU resources."""
        for vbo, _ in self.meshes.values():
            vbo.release()
        self.instance_buffer.release()
        for vao in self.vaos.values():
            vao.release()
        self.program.release()
        
        # Clean up DNA batches
        for species_id, (vao, mesh_vbo, inst_vbo, _, _) in self.dna_batches.items():
            vao.release()
            mesh_vbo.release()
            inst_vbo.release()
        self.dna_batches.clear()
        
        if hasattr(self, 'dna_program'):
            self.dna_program.release()


# =============================================================================
# HELPER: Map waverse animal types to renderer types
# =============================================================================

def get_animal_type_id(animal_type: str) -> int:
    """Map waverse animal type string to renderer type ID."""
    type_map = {
        # Worms
        'worm': ModernAnimalRenderer.TYPE_WORM,
        'centipede': ModernAnimalRenderer.TYPE_WORM,
        
        # Snakes (now separate)
        'snake': ModernAnimalRenderer.TYPE_SNAKE,
        'serpent': ModernAnimalRenderer.TYPE_SNAKE,
        
        # Mammals/large creatures
        'mammal': ModernAnimalRenderer.TYPE_MAMMAL,
        'dinosaur': ModernAnimalRenderer.TYPE_MAMMAL,
        
        # Reptiles
        'reptile': ModernAnimalRenderer.TYPE_REPTILE,
        'lizard': ModernAnimalRenderer.TYPE_REPTILE,
        'croc': ModernAnimalRenderer.TYPE_REPTILE,
        'crocodile': ModernAnimalRenderer.TYPE_REPTILE,
        'alligator': ModernAnimalRenderer.TYPE_REPTILE,
        'gecko': ModernAnimalRenderer.TYPE_REPTILE,
        
        # Amphibians
        'amphibian': ModernAnimalRenderer.TYPE_AMPHIBIAN,
        'frog': ModernAnimalRenderer.TYPE_AMPHIBIAN,
        'toad': ModernAnimalRenderer.TYPE_AMPHIBIAN,
        'salamander': ModernAnimalRenderer.TYPE_AMPHIBIAN,
        'newt': ModernAnimalRenderer.TYPE_AMPHIBIAN,
        
        # Crabs/crustaceans
        'crab': ModernAnimalRenderer.TYPE_CRAB,
        'lobster': ModernAnimalRenderer.TYPE_CRAB,
        'shrimp': ModernAnimalRenderer.TYPE_CRAB,
        'crustacean': ModernAnimalRenderer.TYPE_CRAB,
        
        # Jellyfish/sea creatures
        'jellyfish': ModernAnimalRenderer.TYPE_JELLYFISH,
        'octopus': ModernAnimalRenderer.TYPE_JELLYFISH,
        'squid': ModernAnimalRenderer.TYPE_JELLYFISH,
        
        # Insects/small creatures  
        'insect': ModernAnimalRenderer.TYPE_INSECT,
        'spider': ModernAnimalRenderer.TYPE_INSECT,
        'trilobite': ModernAnimalRenderer.TYPE_INSECT,
        'snail': ModernAnimalRenderer.TYPE_INSECT,
        'beetle': ModernAnimalRenderer.TYPE_INSECT,
        'ant': ModernAnimalRenderer.TYPE_INSECT,
        
        # Birds
        'bird': ModernAnimalRenderer.TYPE_BIRD,
        'screamer': ModernAnimalRenderer.TYPE_BIRD,
        'floater': ModernAnimalRenderer.TYPE_BIRD,
        
        # Fish
        'fish': ModernAnimalRenderer.TYPE_FISH,
        'eel': ModernAnimalRenderer.TYPE_FISH,
        'shark': ModernAnimalRenderer.TYPE_FISH,
        'manta': ModernAnimalRenderer.TYPE_FISH,
        
        # More exotic types - map to closest visual
        'alien': ModernAnimalRenderer.TYPE_JELLYFISH,
        'metroid': ModernAnimalRenderer.TYPE_JELLYFISH,
        'amoeba': ModernAnimalRenderer.TYPE_JELLYFISH,
        'blob': ModernAnimalRenderer.TYPE_JELLYFISH,
        'polyp': ModernAnimalRenderer.TYPE_JELLYFISH,
        'hydra': ModernAnimalRenderer.TYPE_JELLYFISH,
        'nudibranch': ModernAnimalRenderer.TYPE_JELLYFISH,
        'coral': ModernAnimalRenderer.TYPE_JELLYFISH,
        'anemone': ModernAnimalRenderer.TYPE_JELLYFISH,
        'seastar': ModernAnimalRenderer.TYPE_CRAB,
        'urchin': ModernAnimalRenderer.TYPE_CRAB,
        'barnacle': ModernAnimalRenderer.TYPE_CRAB,
        'nautilus': ModernAnimalRenderer.TYPE_FISH,
        
        # Sci-fi creatures
        'biomech_spider': ModernAnimalRenderer.TYPE_INSECT,
        'mech_crawler': ModernAnimalRenderer.TYPE_CRAB,
        'drone': ModernAnimalRenderer.TYPE_BIRD,
        'hive_creature': ModernAnimalRenderer.TYPE_INSECT,
        'symbiote': ModernAnimalRenderer.TYPE_JELLYFISH,
        'xenomorph': ModernAnimalRenderer.TYPE_REPTILE,
        'facehugger': ModernAnimalRenderer.TYPE_CRAB,
        'headcrab': ModernAnimalRenderer.TYPE_CRAB,
        'stalker': ModernAnimalRenderer.TYPE_REPTILE,
        'brute': ModernAnimalRenderer.TYPE_MAMMAL,
        
        # More standard types
        'hopper': ModernAnimalRenderer.TYPE_AMPHIBIAN,
    }
    return type_map.get(animal_type, ModernAnimalRenderer.TYPE_WORM)

