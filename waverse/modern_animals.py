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
    """Create a segmented worm/snake mesh."""
    vertices = []
    normals = []
    
    segments = 8
    radius = 0.15
    length = 2.0
    
    for i in range(segments):
        t = i / (segments - 1)
        z = t * length - length / 2
        
        # Taper at ends
        r = radius * (1.0 - abs(t - 0.5) * 0.5)
        
        for j in range(8):
            angle = j * math.pi / 4
            x = math.cos(angle) * r
            y = math.sin(angle) * r + r  # Offset up from ground
            
            vertices.append([x, y, z])
            
            # Normal pointing outward
            nx = math.cos(angle)
            ny = math.sin(angle)
            normals.append([nx, ny, 0])
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_bird_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple bird mesh with wings."""
    vertices = []
    normals = []
    
    # Body (ellipsoid-ish)
    body_verts = [
        # Front
        [0, 0.3, 0.4], [0.15, 0.25, 0], [-0.15, 0.25, 0],
        # Back
        [0, 0.35, -0.3], [0.1, 0.3, 0], [-0.1, 0.3, 0],
        # Top
        [0, 0.45, 0], [0.1, 0.4, 0.1], [-0.1, 0.4, 0.1],
        # Wings (triangles)
        [0.15, 0.3, 0], [0.6, 0.35, -0.1], [0.2, 0.3, -0.2],
        [-0.15, 0.3, 0], [-0.6, 0.35, -0.1], [-0.2, 0.3, -0.2],
    ]
    
    for v in body_verts:
        vertices.append(v)
        # Simple upward-ish normal
        normals.append([0, 1, 0])
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_mammal_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple quadruped mesh."""
    vertices = []
    normals = []
    
    # Body box
    body = [
        # Top
        [-0.3, 0.6, -0.5], [0.3, 0.6, -0.5], [0.3, 0.6, 0.5], [-0.3, 0.6, 0.5],
        # Bottom
        [-0.3, 0.3, -0.5], [0.3, 0.3, -0.5], [0.3, 0.3, 0.5], [-0.3, 0.3, 0.5],
        # Legs (simple cylinders approximated as boxes)
        [-0.25, 0.3, 0.4], [-0.2, 0, 0.4], [-0.15, 0.3, 0.4],  # Front left
        [0.25, 0.3, 0.4], [0.2, 0, 0.4], [0.15, 0.3, 0.4],     # Front right
        [-0.25, 0.3, -0.4], [-0.2, 0, -0.4], [-0.15, 0.3, -0.4],  # Back left
        [0.25, 0.3, -0.4], [0.2, 0, -0.4], [0.15, 0.3, -0.4],     # Back right
        # Head
        [0, 0.7, 0.7], [-0.15, 0.55, 0.5], [0.15, 0.55, 0.5],
    ]
    
    for v in body:
        vertices.append(v)
        normals.append([0, 1, 0])  # Simplified
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_fish_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a fish-shaped mesh."""
    vertices = []
    normals = []
    
    # Simple fish shape
    fish = [
        # Body diamond
        [0, 0, 0.5],   # Nose
        [0.2, 0.1, 0], [0, 0.15, 0], [-0.2, 0.1, 0],  # Top
        [0.2, -0.1, 0], [0, -0.1, 0], [-0.2, -0.1, 0],  # Bottom
        [0, 0, -0.4],  # Tail base
        # Tail fin
        [0, 0.2, -0.6], [0, 0, -0.4], [0, -0.2, -0.6],
        # Dorsal fin
        [0, 0.3, 0], [0, 0.15, 0.1], [0, 0.15, -0.1],
    ]
    
    for v in fish:
        vertices.append(v)
        normals.append([0, 1, 0])
    
    return np.array(vertices, dtype='f4'), np.array(normals, dtype='f4')


def create_insect_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple insect/bug mesh."""
    vertices = []
    normals = []
    
    # Three body segments
    for i, z in enumerate([-0.2, 0, 0.2]):
        r = 0.1 if i == 1 else 0.07
        for j in range(6):
            angle = j * math.pi / 3
            x = math.cos(angle) * r
            y = math.sin(angle) * r + 0.1
            vertices.append([x, y, z])
            normals.append([math.cos(angle), math.sin(angle), 0])
    
    # Legs (6 simple lines represented as thin triangles)
    for side in [-1, 1]:
        for z in [-0.15, 0, 0.15]:
            vertices.extend([
                [side * 0.08, 0.1, z],
                [side * 0.25, 0, z],
                [side * 0.08, 0.08, z],
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

