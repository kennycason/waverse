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
# MESH GENERATION (base templates)
# ============================================================================

def create_tree_base_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a base tree mesh that will be deformed by DNA parameters.
    Returns vertices, normals, colors.
    
    The mesh has:
    - Trunk: cylinder with rings at different heights (for smooth bending)
    - Foliage: cone/dome shape above trunk
    - Branches: simple extensions
    """
    vertices = []
    normals = []
    colors = []
    
    trunk_color = (0.45, 0.28, 0.15)
    foliage_color = (1.0, 1.0, 1.0)  # White for tinting
    
    # === TRUNK (multi-ring cylinder for smooth deformation) ===
    trunk_height = 1.0  # Normalized, will be scaled by DNA
    trunk_radius = 0.1
    rings = 8  # More rings = smoother curves
    segments = 8
    
    for ring in range(rings):
        t = ring / (rings - 1)
        y = t * trunk_height
        next_t = min(1.0, (ring + 1) / (rings - 1))
        next_y = next_t * trunk_height
        
        for seg in range(segments):
            angle1 = (seg / segments) * 2 * math.pi
            angle2 = ((seg + 1) / segments) * 2 * math.pi
            
            x1 = math.cos(angle1) * trunk_radius
            z1 = math.sin(angle1) * trunk_radius
            x2 = math.cos(angle2) * trunk_radius
            z2 = math.sin(angle2) * trunk_radius
            
            # Two triangles per quad
            # Normal points outward
            n1 = (math.cos(angle1), 0, math.sin(angle1))
            n2 = (math.cos(angle2), 0, math.sin(angle2))
            
            # Triangle 1
            vertices.extend([(x1, y, z1), (x2, y, z2), (x1, next_y, z1)])
            normals.extend([n1, n2, n1])
            colors.extend([trunk_color, trunk_color, trunk_color])
            
            # Triangle 2
            vertices.extend([(x2, y, z2), (x2, next_y, z2), (x1, next_y, z1)])
            normals.extend([n2, n2, n1])
            colors.extend([trunk_color, trunk_color, trunk_color])
    
    # === FOLIAGE (layered cone/dome) ===
    foliage_start = trunk_height * 0.6
    foliage_height = trunk_height * 0.6
    foliage_layers = 4
    
    for layer in range(foliage_layers):
        layer_t = layer / foliage_layers
        next_layer_t = (layer + 1) / foliage_layers
        
        # Radius decreases as we go up
        r1 = 0.4 * (1 - layer_t * 0.7)
        r2 = 0.4 * (1 - next_layer_t * 0.7)
        
        y1 = foliage_start + layer_t * foliage_height
        y2 = foliage_start + next_layer_t * foliage_height
        
        for seg in range(segments):
            angle1 = (seg / segments) * 2 * math.pi
            angle2 = ((seg + 1) / segments) * 2 * math.pi
            
            x1a, z1a = math.cos(angle1) * r1, math.sin(angle1) * r1
            x2a, z2a = math.cos(angle2) * r1, math.sin(angle2) * r1
            x1b, z1b = math.cos(angle1) * r2, math.sin(angle1) * r2
            x2b, z2b = math.cos(angle2) * r2, math.sin(angle2) * r2
            
            # Normal pointing outward and up
            n = (math.cos(angle1 + math.pi/segments), 0.5, math.sin(angle1 + math.pi/segments))
            n_len = math.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
            n = (n[0]/n_len, n[1]/n_len, n[2]/n_len)
            
            # Two triangles per quad
            vertices.extend([(x1a, y1, z1a), (x2a, y1, z2a), (x1b, y2, z1b)])
            normals.extend([n, n, n])
            colors.extend([foliage_color, foliage_color, foliage_color])
            
            vertices.extend([(x2a, y1, z2a), (x2b, y2, z2b), (x1b, y2, z1b)])
            normals.extend([n, n, n])
            colors.extend([foliage_color, foliage_color, foliage_color])
    
    # === BRANCHES (simple horizontal extensions) ===
    branch_height = trunk_height * 0.5
    branch_count = 4
    branch_length = 0.25
    branch_radius = 0.03
    
    for b in range(branch_count):
        angle = (b / branch_count) * 2 * math.pi + 0.3  # Offset for variety
        
        bx = math.cos(angle) * trunk_radius
        bz = math.sin(angle) * trunk_radius
        by = branch_height + (b * 0.05)  # Stagger heights
        
        # Branch direction
        dir_x = math.cos(angle)
        dir_z = math.sin(angle)
        
        # Simple elongated box for branch
        end_x = bx + dir_x * branch_length
        end_z = bz + dir_z * branch_length
        end_y = by - 0.05  # Slight droop
        
        # Just a line with some thickness (6 triangles for a simple prism)
        # Top triangle
        vertices.extend([
            (bx, by + branch_radius, bz),
            (end_x, end_y + branch_radius, end_z),
            (bx + 0.02, by, bz + 0.02)
        ])
        normals.extend([(0, 1, 0), (0, 1, 0), (0, 1, 0)])
        colors.extend([trunk_color, trunk_color, trunk_color])
        
        vertices.extend([
            (end_x, end_y + branch_radius, end_z),
            (end_x + 0.02, end_y, end_z + 0.02),
            (bx + 0.02, by, bz + 0.02)
        ])
        normals.extend([(0, 1, 0), (0, 1, 0), (0, 1, 0)])
        colors.extend([trunk_color, trunk_color, trunk_color])
    
    return (
        np.array(vertices, dtype='f4'),
        np.array(normals, dtype='f4'),
        np.array(colors, dtype='f4')
    )


def create_bush_base_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a base bush mesh - dome shape."""
    vertices = []
    normals = []
    colors = []
    
    foliage_color = (1.0, 1.0, 1.0)  # White for tinting
    
    # Hemisphere dome
    rings = 6
    segments = 8
    radius = 0.4
    
    for ring in range(rings):
        phi1 = (ring / rings) * (math.pi / 2)
        phi2 = ((ring + 1) / rings) * (math.pi / 2)
        
        y1 = math.sin(phi1) * radius
        r1 = math.cos(phi1) * radius
        y2 = math.sin(phi2) * radius
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
            
            # Normal = position normalized
            n1 = (x1a, y1, z1a)
            n2 = (x2a, y1, z2a)
            
            vertices.extend([(x1a, y1, z1a), (x2a, y1, z2a), (x1b, y2, z1b)])
            normals.extend([n1, n2, n1])
            colors.extend([foliage_color, foliage_color, foliage_color])
            
            vertices.extend([(x2a, y1, z2a), (x2b, y2, z2b), (x1b, y2, z1b)])
            normals.extend([n2, n2, n1])
            colors.extend([foliage_color, foliage_color, foliage_color])
    
    return (
        np.array(vertices, dtype='f4'),
        np.array(normals, dtype='f4'),
        np.array(colors, dtype='f4')
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
        
        # Create base meshes
        self.tree_mesh = create_tree_base_mesh()
        self.bush_mesh = create_bush_base_mesh()
        
        # Create VBOs for meshes
        self.tree_vbo = self._create_mesh_vbo(self.tree_mesh)
        self.bush_vbo = self._create_mesh_vbo(self.bush_mesh)
        
        # Instance data: will be created per render
        self.tree_instances = []
        self.bush_instances = []
        
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
    
    def add_tree_instance(self, x: float, y: float, z: float, 
                          scale: float, rotation: float,
                          color: Tuple[float, float, float],
                          dna: PlantDNA):
        """Add a tree instance with DNA parameters."""
        # Extract DNA parameters
        curve = 0.0
        twist = 0.0
        taper = 0.8
        height = 1.0
        droop = 0.0
        
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
        
        self.tree_instances.append((
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
    
    def add_bush_instance(self, x: float, y: float, z: float,
                          scale: float, rotation: float,
                          color: Tuple[float, float, float],
                          dna: PlantDNA):
        """Add a bush instance with DNA parameters."""
        curve = 0.0
        twist = 0.0
        taper = 1.0
        height = 0.5
        droop = 0.0
        
        if dna.trunk_segments:
            seg = dna.trunk_segments[0]
            curve = getattr(seg, 'curve', 0.0) * 0.3  # Less curve for bushes
            twist = getattr(seg, 'twist', 0.0) * 0.5
        
        height_gene = getattr(dna, 'height_gene', None)
        if height_gene:
            height = height_gene.value * 0.5 if hasattr(height_gene, 'value') else 0.5
        
        self.bush_instances.append((
            x, y, z,
            scale,
            rotation,
            *color,
            curve, twist, taper, height, droop
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
        
        # Render trees
        if self.tree_instances:
            self._render_instances(self.tree_vbo, self.tree_instances, len(self.tree_mesh[0]))
        
        # Render bushes
        if self.bush_instances:
            self._render_instances(self.bush_vbo, self.bush_instances, len(self.bush_mesh[0]))
    
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
        self.tree_instances.clear()
        self.bush_instances.clear()


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
    
    # Generate test DNA grid
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    
    GRID_SIZE = 8  # 8x8 grid
    SPACING = 3.0  # 3 units between plants
    
    test_plants = []
    
    for gx in range(GRID_SIZE):
        for gz in range(GRID_SIZE):
            # Alternate between trees and bushes
            is_tree = (gx + gz) % 2 == 0
            
            # Create random DNA
            seed = int(rng.integers(0, 2**31))
            if is_tree:
                dna = PlantDNA.create_random(PlantType.TREE, seed)
            else:
                dna = PlantDNA.create_random(PlantType.BUSH, seed)
            
            # Mutate for variety
            dna = dna.mutate(rng, strength=0.5)
            
            # Position in grid
            x = (gx - GRID_SIZE/2) * SPACING
            z = (gz - GRID_SIZE/2) * SPACING
            
            # Color from DNA
            color = (dna.leaf_color.r, dna.leaf_color.g, dna.leaf_color.b)
            
            test_plants.append({
                'x': x, 'y': 0, 'z': z,
                'scale': 2.0 + rng.random(),
                'rotation': rng.random() * math.pi * 2,
                'color': color,
                'dna': dna,
                'is_tree': is_tree
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
            if plant['is_tree']:
                renderer.add_tree_instance(
                    plant['x'], plant['y'], plant['z'],
                    plant['scale'], plant['rotation'],
                    plant['color'], plant['dna']
                )
            else:
                renderer.add_bush_instance(
                    plant['x'], plant['y'], plant['z'],
                    plant['scale'], plant['rotation'],
                    plant['color'], plant['dna']
                )
        
        renderer.render()
        
        pygame.display.flip()
    
    pygame.quit()


if __name__ == '__main__':
    run_test_grid()

