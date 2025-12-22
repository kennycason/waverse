"""
ModernGL Animal Renderer

Instanced rendering for animals/fauna with:
- GPU-based animation (walking, flying, swimming)
- Per-instance color tinting from DNA
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
        }
        
        # Uniforms
        self.projection = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
        self.camera_pos = glm.vec3(0, 0, 0)
        self.light_dir = glm.vec3(0.5, 1.0, 0.3)
        self.ambient = glm.vec3(0.4, 0.4, 0.5)
        self.fog_color = glm.vec3(0.7, 0.8, 0.9)
        self.fog_start = 100.0
        self.fog_end = 400.0
        self.time = 0.0
        
        # Stats
        self.frame_stats = {
            'instances_rendered': 0,
            'draw_calls': 0,
        }
    
    def _create_meshes(self):
        """Create VBOs for each animal mesh type."""
        mesh_creators = {
            self.TYPE_WORM: create_worm_mesh,
            self.TYPE_MAMMAL: create_mammal_mesh,
            self.TYPE_INSECT: create_insect_mesh,
            self.TYPE_BIRD: create_bird_mesh,
            self.TYPE_FISH: create_fish_mesh,
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
    
    def cleanup(self):
        """Release GPU resources."""
        for vbo, _ in self.meshes.values():
            vbo.release()
        self.instance_buffer.release()
        for vao in self.vaos.values():
            vao.release()
        self.program.release()


# =============================================================================
# HELPER: Map waverse animal types to renderer types
# =============================================================================

def get_animal_type_id(animal_type: str) -> int:
    """Map waverse animal type string to renderer type ID."""
    type_map = {
        # Worms/snakes
        'worm': ModernAnimalRenderer.TYPE_WORM,
        'snake': ModernAnimalRenderer.TYPE_WORM,
        'centipede': ModernAnimalRenderer.TYPE_WORM,
        
        # Mammals/large creatures
        'mammal': ModernAnimalRenderer.TYPE_MAMMAL,
        'dinosaur': ModernAnimalRenderer.TYPE_MAMMAL,
        'croc': ModernAnimalRenderer.TYPE_MAMMAL,
        
        # Insects/small creatures  
        'insect': ModernAnimalRenderer.TYPE_INSECT,
        'spider': ModernAnimalRenderer.TYPE_INSECT,
        'trilobite': ModernAnimalRenderer.TYPE_INSECT,
        'snail': ModernAnimalRenderer.TYPE_INSECT,
        
        # Birds
        'bird': ModernAnimalRenderer.TYPE_BIRD,
        
        # Fish
        'fish': ModernAnimalRenderer.TYPE_FISH,
    }
    return type_map.get(animal_type, ModernAnimalRenderer.TYPE_WORM)

