"""
Modern Flora Renderer - Instanced rendering for plants.

Uses GPU instancing to render thousands of plants with minimal draw calls.
Plants with similar shapes are batched together.
"""

import numpy as np
import moderngl
import glm
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
import math


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
uniform float u_time;
uniform float u_wind_strength;

mat3 rotateY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(c, 0, s, 0, 1, 0, -s, 0, c);
}

void main() {
    // Wind animation - sway based on height and time
    float sway = sin(u_time * 2.0 + in_instance_pos.x * 0.1 + in_instance_pos.z * 0.15);
    sway *= in_position.y * u_wind_strength * 0.02;  // More sway at top
    
    vec3 pos = in_position;
    pos.x += sway;
    pos.z += sway * 0.5;
    
    // Apply instance transform
    mat3 rot = rotateY(in_instance_rot);
    vec3 scaled = pos * in_instance_scale;
    vec3 rotated = rot * scaled;
    vec3 world_pos = rotated + in_instance_pos;
    
    v_world_pos = world_pos;
    v_height = in_instance_pos.y;
    v_normal = rot * in_normal;
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
    """Create a simple stylized tree mesh."""
    verts = []
    normals = []
    colors = []
    
    # Trunk (hexagonal prism for efficiency)
    trunk_h = 2.0
    trunk_r = 0.15
    trunk_color = (0.35, 0.22, 0.12)
    
    for i in range(6):
        angle1 = i * math.pi / 3
        angle2 = (i + 1) * math.pi / 3
        
        x1, z1 = math.cos(angle1) * trunk_r, math.sin(angle1) * trunk_r
        x2, z2 = math.cos(angle2) * trunk_r, math.sin(angle2) * trunk_r
        
        # Side face (2 triangles)
        nx = (math.cos(angle1) + math.cos(angle2)) / 2
        nz = (math.sin(angle1) + math.sin(angle2)) / 2
        n = (nx, 0, nz)
        
        for p, c in [((x1, 0, z1), trunk_color), ((x2, 0, z2), trunk_color),
                     ((x1, trunk_h, z1), trunk_color)]:
            verts.append(p)
            normals.append(n)
            colors.append(c)
        for p, c in [((x2, 0, z2), trunk_color), ((x2, trunk_h, z2), trunk_color),
                     ((x1, trunk_h, z1), trunk_color)]:
            verts.append(p)
            normals.append(n)
            colors.append(c)
    
    # Foliage (cone/pyramid for simplicity)
    foliage_h = 3.0
    foliage_r = 1.2
    foliage_base = trunk_h * 0.7
    foliage_color = (0.15, 0.45, 0.18)
    
    for i in range(8):
        angle1 = i * math.pi / 4
        angle2 = (i + 1) * math.pi / 4
        
        x1, z1 = math.cos(angle1) * foliage_r, math.sin(angle1) * foliage_r
        x2, z2 = math.cos(angle2) * foliage_r, math.sin(angle2) * foliage_r
        
        # Calculate normal for cone face
        mid_angle = (angle1 + angle2) / 2
        slope = foliage_r / foliage_h
        ny = slope / math.sqrt(1 + slope * slope)
        nx = math.cos(mid_angle) * (1 - ny * ny) ** 0.5
        nz = math.sin(mid_angle) * (1 - ny * ny) ** 0.5
        n = (nx, ny, nz)
        
        # Cone face
        verts.append((0, foliage_base + foliage_h, 0))
        normals.append(n)
        colors.append((foliage_color[0] * 0.9, foliage_color[1] * 1.1, foliage_color[2] * 0.9))
        
        verts.append((x1, foliage_base, z1))
        normals.append(n)
        colors.append(foliage_color)
        
        verts.append((x2, foliage_base, z2))
        normals.append(n)
        colors.append(foliage_color)
    
    return (np.array(verts, dtype='f4'),
            np.array(normals, dtype='f4'),
            np.array(colors, dtype='f4'))


def create_bush_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a simple bush mesh (sphere-ish)."""
    verts = []
    normals = []
    colors = []
    
    bush_color = (0.2, 0.5, 0.22)
    radius = 0.6
    
    # Simple octahedron for bush shape
    top = (0, radius * 1.2, 0)
    bottom = (0, 0, 0)
    
    equator = [
        (radius, radius * 0.6, 0),
        (0, radius * 0.6, radius),
        (-radius, radius * 0.6, 0),
        (0, radius * 0.6, -radius),
    ]
    
    # Top faces
    for i in range(4):
        p1 = equator[i]
        p2 = equator[(i + 1) % 4]
        
        # Calculate normal
        v1 = np.array(p1) - np.array(top)
        v2 = np.array(p2) - np.array(top)
        n = tuple(np.cross(v2, v1))
        n = tuple(x / (np.linalg.norm(n) + 1e-8) for x in n)
        
        verts.extend([top, p1, p2])
        normals.extend([n, n, n])
        c = (bush_color[0] * (0.9 + i * 0.05), 
             bush_color[1] * (0.95 + i * 0.02),
             bush_color[2] * (0.9 + i * 0.05))
        colors.extend([c, c, c])
    
    # Bottom faces
    for i in range(4):
        p1 = equator[i]
        p2 = equator[(i + 1) % 4]
        
        v1 = np.array(p2) - np.array(bottom)
        v2 = np.array(p1) - np.array(bottom)
        n = tuple(np.cross(v2, v1))
        n = tuple(x / (np.linalg.norm(n) + 1e-8) for x in n)
        
        verts.extend([bottom, p2, p1])
        normals.extend([n, n, n])
        c = (bush_color[0] * 0.7, bush_color[1] * 0.8, bush_color[2] * 0.7)
        colors.extend([c, c, c])
    
    return (np.array(verts, dtype='f4'),
            np.array(normals, dtype='f4'),
            np.array(colors, dtype='f4'))


def create_grass_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create grass blade mesh."""
    verts = []
    normals = []
    colors = []
    
    # Multiple blades
    for blade in range(5):
        angle = blade * math.pi * 2 / 5
        offset_x = math.cos(angle) * 0.1
        offset_z = math.sin(angle) * 0.1
        
        # Blade is a thin quad
        height = 0.4 + (blade % 3) * 0.15
        width = 0.05
        
        base_color = (0.25, 0.55, 0.2)
        tip_color = (0.35, 0.65, 0.25)
        
        # Front face
        verts.extend([
            (offset_x - width, 0, offset_z),
            (offset_x + width, 0, offset_z),
            (offset_x, height, offset_z + 0.05),
        ])
        n = (0, 0.3, 0.95)
        normals.extend([n, n, n])
        colors.extend([base_color, base_color, tip_color])
        
        # Back face
        verts.extend([
            (offset_x + width, 0, offset_z),
            (offset_x - width, 0, offset_z),
            (offset_x, height, offset_z + 0.05),
        ])
        n = (0, 0.3, -0.95)
        normals.extend([n, n, n])
        colors.extend([base_color, base_color, tip_color])
    
    return (np.array(verts, dtype='f4'),
            np.array(normals, dtype='f4'),
            np.array(colors, dtype='f4'))


def create_flower_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a simple flower mesh."""
    verts = []
    normals = []
    colors = []
    
    # Stem
    stem_h = 0.5
    stem_r = 0.02
    stem_color = (0.2, 0.45, 0.15)
    
    for i in range(4):
        angle1 = i * math.pi / 2
        angle2 = (i + 1) * math.pi / 2
        x1, z1 = math.cos(angle1) * stem_r, math.sin(angle1) * stem_r
        x2, z2 = math.cos(angle2) * stem_r, math.sin(angle2) * stem_r
        
        verts.extend([(x1, 0, z1), (x2, 0, z2), (x1, stem_h, z1)])
        verts.extend([(x2, 0, z2), (x2, stem_h, z2), (x1, stem_h, z1)])
        for _ in range(6):
            normals.append((math.cos(angle1), 0, math.sin(angle1)))
            colors.append(stem_color)
    
    # Petals (5 triangles radiating out)
    petal_colors = [
        (0.9, 0.3, 0.35),  # Red
        (0.95, 0.85, 0.3),  # Yellow
        (0.85, 0.4, 0.85),  # Purple
        (0.95, 0.6, 0.3),   # Orange
        (0.9, 0.5, 0.6),    # Pink
    ]
    petal_color = petal_colors[0]  # Will be tinted by instance color
    
    for i in range(5):
        angle = i * math.pi * 2 / 5
        px = math.cos(angle) * 0.2
        pz = math.sin(angle) * 0.2
        
        verts.extend([
            (0, stem_h, 0),
            (px, stem_h + 0.05, pz),
            (px * 0.5, stem_h + 0.15, pz * 0.5),
        ])
        n = (0, 1, 0)
        normals.extend([n, n, n])
        colors.extend([petal_color, petal_color, 
                       (petal_color[0] * 1.2, petal_color[1] * 1.2, petal_color[2] * 1.2)])
    
    return (np.array(verts, dtype='f4'),
            np.array(normals, dtype='f4'),
            np.array(colors, dtype='f4'))


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
        self.mesh_templates: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {
            'tree': create_tree_mesh(),
            'bush': create_bush_mesh(),
            'grass': create_grass_mesh(),
            'flower': create_flower_mesh(),
        }
        
        # Instance batches: type -> FloraInstanceBatch
        self.batches: Dict[str, FloraInstanceBatch] = {}
        
        # Pending instance data (before GPU upload)
        self.pending_instances: Dict[str, List[Tuple[float, float, float, float, float, float, float, float]]] = {
            'tree': [], 'bush': [], 'grass': [], 'flower': []
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
        self.fog_start = 100.0
        self.fog_end = 350.0
        
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
        if mesh_type not in self.pending_instances:
            mesh_type = 'bush'  # Fallback
        
        self.pending_instances[mesh_type].append(
            (x, y, z, scale, rotation, color_r, color_g, color_b)
        )
    
    def clear_instances(self):
        """Clear all pending instances."""
        for key in self.pending_instances:
            self.pending_instances[key] = []
    
    def upload_instances(self):
        """Upload pending instances to GPU."""
        for mesh_type, instances in self.pending_instances.items():
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
            else:
                batch = self.batches[mesh_type]
            
            # Create instance data
            instance_count = len(instances)
            instance_data = np.array(instances, dtype='f4')
            
            # Release old instance VBO
            if batch.instance_vbo:
                batch.instance_vbo.release()
            if batch.vao:
                batch.vao.release()
            
            batch.instance_vbo = self.ctx.buffer(instance_data.tobytes())
            batch.instance_count = instance_count
            
            # Create VAO
            batch.vao = self.ctx.vertex_array(
                self.program,
                [
                    (batch.mesh_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color'),
                    (batch.instance_vbo, '3f 1f 1f 3f /i', 'in_instance_pos', 
                     'in_instance_scale', 'in_instance_rot', 'in_instance_color'),
                ]
            )
    
    def render(self, dt: float = 0.016):
        """Render all flora instances."""
        self.time += dt
        self.frame_stats = {'instances_rendered': 0, 'draw_calls': 0}
        
        # Set uniforms
        proj_bytes = np.array(self.projection.to_list(), dtype='f4').tobytes()
        view_bytes = np.array(self.view.to_list(), dtype='f4').tobytes()
        
        self.program['u_projection'].write(proj_bytes)
        self.program['u_view'].write(view_bytes)
        self.program['u_camera_pos'].value = tuple(self.camera_pos)
        self.program['u_light_dir'].value = tuple(self.light_dir)
        self.program['u_ambient'].value = tuple(self.ambient)
        self.program['u_fog_start'].value = self.fog_start
        self.program['u_fog_end'].value = self.fog_end
        self.program['u_fog_color'].value = tuple(self.fog_color)
        self.program['u_time'].value = self.time
        self.program['u_wind_strength'].value = self.wind_strength
        
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

