"""
ModernGL Renderer Prototype

This is a proof-of-concept to validate ModernGL for waverse.
Key features to test:
1. VBO-based terrain rendering (vs immediate mode)
2. Instanced rendering for flora/fauna
3. Dynamic geometry updates (terrain editing)
4. Integration with existing pygame window
5. Shader-based effects

Performance targets:
- 60+ FPS with 100k+ triangles
- Efficient chunk updates
- GPU-based LOD
"""

import numpy as np
import moderngl
try:
    from pyglm import glm
except ImportError:
    import glm
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
import time


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
out vec3 v_position;

uniform mat4 u_projection;
uniform mat4 u_view;

void main() {
    v_position = in_position;
    v_normal = in_normal;
    v_color = in_color;
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
}
"""

TERRAIN_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_normal;
in vec3 v_color;
in vec3 v_position;

out vec4 fragColor;

uniform vec3 u_light_dir;
uniform vec3 u_ambient;
uniform float u_fog_start;
uniform float u_fog_end;
uniform vec3 u_fog_color;

void main() {
    // Diffuse lighting
    float diff = max(dot(normalize(v_normal), normalize(u_light_dir)), 0.0);
    vec3 lit_color = v_color * (u_ambient + diff * 0.7);
    
    // Distance fog
    float dist = length(v_position);
    float fog_factor = clamp((dist - u_fog_start) / (u_fog_end - u_fog_start), 0.0, 1.0);
    vec3 final_color = mix(lit_color, u_fog_color, fog_factor);
    
    fragColor = vec4(final_color, 1.0);
}
"""

# Instanced shader for flora/fauna - renders many copies with one draw call
INSTANCED_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;   // Local mesh vertex
in vec3 in_normal;
in vec3 in_color;

// Per-instance data
in vec3 in_instance_pos;    // World position
in float in_instance_scale; // Scale factor
in float in_instance_rot;   // Y rotation (radians)
in vec3 in_instance_color;  // Color tint

out vec3 v_normal;
out vec3 v_color;
out vec3 v_position;

uniform mat4 u_projection;
uniform mat4 u_view;

mat3 rotateY(float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat3(
        c, 0, s,
        0, 1, 0,
        -s, 0, c
    );
}

void main() {
    // Apply instance transform
    mat3 rot = rotateY(in_instance_rot);
    vec3 scaled = in_position * in_instance_scale;
    vec3 rotated = rot * scaled;
    vec3 world_pos = rotated + in_instance_pos;
    
    v_position = world_pos;
    v_normal = rot * in_normal;
    v_color = in_color * in_instance_color;  // Multiply base color by tint
    
    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
}
"""

INSTANCED_FRAGMENT_SHADER = TERRAIN_FRAGMENT_SHADER  # Same lighting


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ChunkMesh:
    """GPU mesh for a terrain chunk."""
    vbo: Optional[moderngl.Buffer] = None
    vao: Optional[moderngl.VertexArray] = None
    vertex_count: int = 0
    needs_update: bool = True
    
    def release(self):
        if self.vbo:
            self.vbo.release()
        if self.vao:
            self.vao.release()


@dataclass
class InstancedMesh:
    """A mesh that can be rendered many times with different transforms."""
    # Base mesh (shared geometry)
    mesh_vbo: Optional[moderngl.Buffer] = None
    vertex_count: int = 0
    
    # Instance data (per-copy transforms)
    instance_vbo: Optional[moderngl.Buffer] = None
    instance_count: int = 0
    
    vao: Optional[moderngl.VertexArray] = None
    needs_update: bool = True


# =============================================================================
# MODERN RENDERER
# =============================================================================

class ModernRenderer:
    """
    Modern OpenGL renderer using VBOs and instancing.
    
    This replaces the legacy immediate-mode rendering with:
    - Vertex Buffer Objects for terrain chunks
    - Instanced rendering for repeated objects (flora, fauna)
    - Shader-based lighting and fog
    - Efficient batch updates
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Compile shaders
        self.terrain_prog = ctx.program(
            vertex_shader=TERRAIN_VERTEX_SHADER,
            fragment_shader=TERRAIN_FRAGMENT_SHADER
        )
        self.instanced_prog = ctx.program(
            vertex_shader=INSTANCED_VERTEX_SHADER,
            fragment_shader=INSTANCED_FRAGMENT_SHADER
        )
        
        # Chunk meshes: (cx, cz) -> ChunkMesh
        self.chunk_meshes: Dict[Tuple[int, int], ChunkMesh] = {}
        
        # Instanced meshes: mesh_id -> InstancedMesh
        self.instanced_meshes: Dict[str, InstancedMesh] = {}
        
        # Camera matrices
        self.projection = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
        
        # Lighting
        self.light_dir = glm.vec3(0.5, 1.0, 0.3)
        self.ambient = glm.vec3(0.3, 0.35, 0.4)
        self.fog_color = glm.vec3(0.7, 0.8, 0.9)
        self.fog_start = 200.0
        self.fog_end = 500.0
        
        # Stats
        self.stats = {
            'draw_calls': 0,
            'triangles': 0,
            'chunks_rendered': 0,
            'instances_rendered': 0,
        }
    
    def set_camera(self, projection: glm.mat4, view: glm.mat4):
        """Update camera matrices."""
        self.projection = projection
        self.view = view
    
    def create_chunk_mesh(self, cx: int, cz: int, 
                          vertices: np.ndarray, 
                          normals: np.ndarray,
                          colors: np.ndarray) -> ChunkMesh:
        """
        Create or update a terrain chunk mesh.
        
        Args:
            cx, cz: Chunk coordinates
            vertices: Nx3 float32 array of positions
            normals: Nx3 float32 array of normals
            colors: Nx3 float32 array of colors (0-1 range)
        """
        key = (cx, cz)
        
        # Release old mesh if exists
        if key in self.chunk_meshes:
            self.chunk_meshes[key].release()
        
        # Interleave vertex data: [pos, normal, color] per vertex
        # 9 floats per vertex (3+3+3)
        vertex_count = len(vertices)
        data = np.zeros((vertex_count, 9), dtype='f4')
        data[:, 0:3] = vertices
        data[:, 3:6] = normals
        data[:, 6:9] = colors
        
        vbo = self.ctx.buffer(data.tobytes())
        vao = self.ctx.vertex_array(
            self.terrain_prog,
            [(vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color')]
        )
        
        mesh = ChunkMesh(vbo=vbo, vao=vao, vertex_count=vertex_count, needs_update=False)
        self.chunk_meshes[key] = mesh
        return mesh
    
    def create_instanced_mesh(self, mesh_id: str,
                              base_vertices: np.ndarray,
                              base_normals: np.ndarray, 
                              base_colors: np.ndarray,
                              instance_positions: np.ndarray,
                              instance_scales: np.ndarray,
                              instance_rotations: np.ndarray,
                              instance_colors: np.ndarray) -> InstancedMesh:
        """
        Create an instanced mesh for rendering many copies.
        
        Args:
            mesh_id: Unique identifier for this mesh type
            base_vertices: Mx3 base mesh vertices
            base_normals: Mx3 base mesh normals
            base_colors: Mx3 base mesh colors
            instance_positions: Nx3 world positions for each instance
            instance_scales: N floats for scale
            instance_rotations: N floats for Y rotation (radians)
            instance_colors: Nx3 color tints for each instance
        """
        if mesh_id in self.instanced_meshes:
            old = self.instanced_meshes[mesh_id]
            if old.mesh_vbo:
                old.mesh_vbo.release()
            if old.instance_vbo:
                old.instance_vbo.release()
            if old.vao:
                old.vao.release()
        
        # Base mesh data (interleaved)
        vertex_count = len(base_vertices)
        mesh_data = np.zeros((vertex_count, 9), dtype='f4')
        mesh_data[:, 0:3] = base_vertices
        mesh_data[:, 3:6] = base_normals
        mesh_data[:, 6:9] = base_colors
        mesh_vbo = self.ctx.buffer(mesh_data.tobytes())
        
        # Instance data: pos(3) + scale(1) + rot(1) + color(3) = 8 floats
        instance_count = len(instance_positions)
        instance_data = np.zeros((instance_count, 8), dtype='f4')
        instance_data[:, 0:3] = instance_positions
        instance_data[:, 3] = instance_scales
        instance_data[:, 4] = instance_rotations
        instance_data[:, 5:8] = instance_colors
        instance_vbo = self.ctx.buffer(instance_data.tobytes())
        
        vao = self.ctx.vertex_array(
            self.instanced_prog,
            [
                (mesh_vbo, '3f 3f 3f', 'in_position', 'in_normal', 'in_color'),
                (instance_vbo, '3f 1f 1f 3f /i', 'in_instance_pos', 'in_instance_scale', 
                 'in_instance_rot', 'in_instance_color'),
            ]
        )
        
        mesh = InstancedMesh(
            mesh_vbo=mesh_vbo,
            vertex_count=vertex_count,
            instance_vbo=instance_vbo,
            instance_count=instance_count,
            vao=vao,
            needs_update=False
        )
        self.instanced_meshes[mesh_id] = mesh
        return mesh
    
    def update_instances(self, mesh_id: str,
                         instance_positions: np.ndarray,
                         instance_scales: np.ndarray,
                         instance_rotations: np.ndarray,
                         instance_colors: np.ndarray):
        """Update instance data without recreating mesh."""
        if mesh_id not in self.instanced_meshes:
            return
        
        mesh = self.instanced_meshes[mesh_id]
        instance_count = len(instance_positions)
        
        instance_data = np.zeros((instance_count, 8), dtype='f4')
        instance_data[:, 0:3] = instance_positions
        instance_data[:, 3] = instance_scales
        instance_data[:, 4] = instance_rotations
        instance_data[:, 5:8] = instance_colors
        
        # Orphan the old buffer and write new data
        mesh.instance_vbo.orphan(size=instance_data.nbytes)
        mesh.instance_vbo.write(instance_data.tobytes())
        mesh.instance_count = instance_count
    
    def render(self, chunk_keys: List[Tuple[int, int]] = None,
               instanced_mesh_ids: List[str] = None):
        """
        Render all geometry.
        
        Args:
            chunk_keys: List of chunk coordinates to render (None = all)
            instanced_mesh_ids: List of instanced meshes to render (None = all)
        """
        self.stats = {'draw_calls': 0, 'triangles': 0, 
                      'chunks_rendered': 0, 'instances_rendered': 0}
        
        # Set up common uniforms - write glm matrices directly
        # Render terrain chunks
        self.terrain_prog['u_projection'].write(self.projection)
        self.terrain_prog['u_view'].write(self.view)
        self.terrain_prog['u_light_dir'].value = tuple(self.light_dir)
        self.terrain_prog['u_ambient'].value = tuple(self.ambient)
        self.terrain_prog['u_fog_start'].value = self.fog_start
        self.terrain_prog['u_fog_end'].value = self.fog_end
        self.terrain_prog['u_fog_color'].value = tuple(self.fog_color)
        
        keys = chunk_keys if chunk_keys else list(self.chunk_meshes.keys())
        for key in keys:
            if key in self.chunk_meshes:
                mesh = self.chunk_meshes[key]
                if mesh.vao and mesh.vertex_count > 0:
                    mesh.vao.render(moderngl.TRIANGLES)
                    self.stats['draw_calls'] += 1
                    self.stats['triangles'] += mesh.vertex_count // 3
                    self.stats['chunks_rendered'] += 1
        
        # Render instanced meshes
        self.instanced_prog['u_projection'].write(self.projection)
        self.instanced_prog['u_view'].write(self.view)
        self.instanced_prog['u_light_dir'].value = tuple(self.light_dir)
        self.instanced_prog['u_ambient'].value = tuple(self.ambient)
        self.instanced_prog['u_fog_start'].value = self.fog_start
        self.instanced_prog['u_fog_end'].value = self.fog_end
        self.instanced_prog['u_fog_color'].value = tuple(self.fog_color)
        
        mesh_ids = instanced_mesh_ids if instanced_mesh_ids else list(self.instanced_meshes.keys())
        for mesh_id in mesh_ids:
            if mesh_id in self.instanced_meshes:
                mesh = self.instanced_meshes[mesh_id]
                if mesh.vao and mesh.instance_count > 0:
                    mesh.vao.render(moderngl.TRIANGLES, instances=mesh.instance_count)
                    self.stats['draw_calls'] += 1
                    self.stats['triangles'] += (mesh.vertex_count // 3) * mesh.instance_count
                    self.stats['instances_rendered'] += mesh.instance_count
    
    def cleanup(self):
        """Release all GPU resources."""
        for mesh in self.chunk_meshes.values():
            mesh.release()
        self.chunk_meshes.clear()
        
        for mesh in self.instanced_meshes.values():
            if mesh.mesh_vbo:
                mesh.mesh_vbo.release()
            if mesh.instance_vbo:
                mesh.instance_vbo.release()
            if mesh.vao:
                mesh.vao.release()
        self.instanced_meshes.clear()


# =============================================================================
# BENCHMARK TEST
# =============================================================================

def benchmark_modern_renderer():
    """
    Benchmark the modern renderer to validate performance.
    
    Creates a test scene with:
    - 100 terrain chunks (each 32x32 = 2048 triangles)
    - 10,000 instanced flora objects
    """
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL
    
    pygame.init()
    
    # Request OpenGL 3.3 core profile
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    
    pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("ModernGL Benchmark")
    
    # Create ModernGL context from existing pygame/OpenGL context
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.CULL_FACE)
    
    renderer = ModernRenderer(ctx)
    
    print("=" * 60)
    print("ModernGL Renderer Benchmark")
    print("=" * 60)
    
    # Generate test terrain chunks
    print("\nGenerating terrain chunks...")
    chunk_size = 32
    num_chunks = 10  # 10x10 = 100 chunks
    
    for cx in range(num_chunks):
        for cz in range(num_chunks):
            # Generate a simple heightmap
            vertices = []
            normals = []
            colors = []
            
            for x in range(chunk_size):
                for z in range(chunk_size):
                    wx = cx * chunk_size + x
                    wz = cz * chunk_size + z
                    
                    # Simple sine wave terrain
                    h = np.sin(wx * 0.1) * np.cos(wz * 0.1) * 5
                    
                    # Two triangles per grid cell
                    for tri in range(2):
                        if tri == 0:
                            offsets = [(0, 0), (1, 0), (0, 1)]
                        else:
                            offsets = [(1, 0), (1, 1), (0, 1)]
                        
                        for dx, dz in offsets:
                            px = wx + dx
                            pz = wz + dz
                            py = np.sin(px * 0.1) * np.cos(pz * 0.1) * 5
                            vertices.append([px, py, pz])
                            normals.append([0, 1, 0])  # Simplified
                            colors.append([0.3, 0.6 + py * 0.02, 0.2])
            
            renderer.create_chunk_mesh(
                cx, cz,
                np.array(vertices, dtype='f4'),
                np.array(normals, dtype='f4'),
                np.array(colors, dtype='f4')
            )
    
    total_terrain_tris = num_chunks * num_chunks * chunk_size * chunk_size * 2
    print(f"  Created {num_chunks * num_chunks} chunks")
    print(f"  Total terrain triangles: {total_terrain_tris:,}")
    
    # Generate instanced flora
    print("\nGenerating instanced flora...")
    num_flora = 10000
    
    # Simple tree mesh (6 triangles for trunk + 12 for foliage = 18 tris)
    tree_verts = []
    tree_normals = []
    tree_colors = []
    
    # Trunk (box)
    trunk_h = 1.0
    trunk_w = 0.15
    for face in range(4):
        angle = face * np.pi / 2
        n = [np.sin(angle), 0, np.cos(angle)]
        for tri in range(2):
            if tri == 0:
                corners = [(0, 0), (1, 0), (0, 1)]
            else:
                corners = [(1, 0), (1, 1), (0, 1)]
            for cx, cy in corners:
                x = np.sin(angle + cx * np.pi / 2) * trunk_w
                z = np.cos(angle + cx * np.pi / 2) * trunk_w
                y = cy * trunk_h
                tree_verts.append([x, y, z])
                tree_normals.append(n)
                tree_colors.append([0.4, 0.25, 0.1])
    
    # Foliage (cone approximation)
    foliage_h = 1.5
    foliage_r = 0.5
    for i in range(8):
        angle1 = i * np.pi / 4
        angle2 = (i + 1) * np.pi / 4
        
        tree_verts.append([0, trunk_h + foliage_h, 0])
        tree_verts.append([np.sin(angle1) * foliage_r, trunk_h, np.cos(angle1) * foliage_r])
        tree_verts.append([np.sin(angle2) * foliage_r, trunk_h, np.cos(angle2) * foliage_r])
        
        for _ in range(3):
            tree_normals.append([0, 0.5, 0.5])
            tree_colors.append([0.1, 0.5, 0.15])
    
    # Instance data
    rng = np.random.default_rng(42)
    world_size = num_chunks * chunk_size
    positions = rng.random((num_flora, 3)).astype('f4')
    positions[:, 0] *= world_size
    positions[:, 1] = 0  # Will be on ground
    positions[:, 2] *= world_size
    
    scales = 0.5 + rng.random(num_flora).astype('f4') * 1.5
    rotations = rng.random(num_flora).astype('f4') * np.pi * 2
    colors = 0.7 + rng.random((num_flora, 3)).astype('f4') * 0.3
    
    renderer.create_instanced_mesh(
        "trees",
        np.array(tree_verts, dtype='f4'),
        np.array(tree_normals, dtype='f4'),
        np.array(tree_colors, dtype='f4'),
        positions, scales, rotations, colors
    )
    
    tree_tris = len(tree_verts) // 3
    print(f"  Created {num_flora:,} tree instances")
    print(f"  Triangles per tree: {tree_tris}")
    print(f"  Total flora triangles: {tree_tris * num_flora:,}")
    
    total_tris = total_terrain_tris + tree_tris * num_flora
    print(f"\n  TOTAL TRIANGLES: {total_tris:,}")
    
    # Benchmark rendering
    print("\nBenchmarking render performance...")
    
    # Set up camera
    projection = glm.perspective(glm.radians(60), 1280/720, 0.1, 1000.0)
    
    frame_times = []
    clock = pygame.time.Clock()
    
    for frame in range(300):  # 5 seconds at 60 FPS
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break
        
        # Orbit camera
        t = frame * 0.02
        cam_x = np.cos(t) * 200 + 160
        cam_z = np.sin(t) * 200 + 160
        cam_y = 50
        
        view = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z),
            glm.vec3(160, 0, 160),
            glm.vec3(0, 1, 0)
        )
        
        renderer.set_camera(projection, view)
        
        # Clear and render
        ctx.clear(0.5, 0.6, 0.8)
        
        start = time.perf_counter()
        renderer.render()
        ctx.finish()  # Wait for GPU
        elapsed = time.perf_counter() - start
        frame_times.append(elapsed)
        
        pygame.display.flip()
        clock.tick(60)
    
    # Report results
    avg_time = np.mean(frame_times) * 1000
    min_time = np.min(frame_times) * 1000
    max_time = np.max(frame_times) * 1000
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Triangles rendered: {total_tris:,}")
    print(f"  Draw calls: {renderer.stats['draw_calls']}")
    print(f"  Avg frame time: {avg_time:.2f} ms ({1000/avg_time:.0f} FPS)")
    print(f"  Min frame time: {min_time:.2f} ms")
    print(f"  Max frame time: {max_time:.2f} ms")
    print(f"\n  Triangles per millisecond: {total_tris / avg_time:,.0f}")
    
    # Cleanup
    renderer.cleanup()
    pygame.quit()
    
    return avg_time, total_tris


if __name__ == "__main__":
    benchmark_modern_renderer()

