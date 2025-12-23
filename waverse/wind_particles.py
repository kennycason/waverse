"""
Wind Particle System - Debris and leaves following wind currents.

Features:
- Small polygon particles (triangles/leaves)
- Follow wind direction with noise
- Spiral toward tornado center when active
- Low particle count for efficiency (50-100)
"""

import numpy as np
import moderngl
from typing import Tuple, Optional
import math

try:
    from pyglm import glm
except ImportError:
    import glm


# =============================================================================
# SHADERS
# =============================================================================

WIND_PARTICLE_VERTEX_SHADER = """
#version 330 core

// Only the attributes we actually use (others would be optimized out)
in vec3 in_position;     // Particle center position
in vec3 in_color;        // Particle color
in float in_size;        // Particle size
in float in_life;        // 0.0 = dead, 1.0 = full life

out vec3 v_color;
out float v_alpha;

uniform mat4 u_projection;
uniform mat4 u_view;

void main() {
    // Billboard particle - always faces camera
    vec4 view_pos = u_view * vec4(in_position, 1.0);
    
    // Offset for particle corners (we'll render as point sprites)
    float scale = in_size * (0.5 + in_life * 0.5);  // Shrink as life decreases
    
    gl_Position = u_projection * view_pos;
    gl_PointSize = scale * 50.0 / (-view_pos.z);  // Size in screen space
    
    v_color = in_color;
    v_alpha = in_life * 0.7;  // Fade out
}
"""

WIND_PARTICLE_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_color;
in float v_alpha;

out vec4 fragColor;

void main() {
    // Point sprite - create triangle-ish shape
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    // Triangle shape (approximate)
    float shape = step(coord.y, 0.3 - abs(coord.x) * 0.8);
    
    if (shape < 0.5 || dist > 0.5) {
        discard;
    }
    
    fragColor = vec4(v_color, v_alpha * shape);
}
"""


# =============================================================================
# WIND PARTICLE RENDERER
# =============================================================================

class WindParticleRenderer:
    """
    GPU-based wind particle system for debris/leaves.
    
    Particles follow wind currents and spiral in tornadoes.
    """
    
    MAX_PARTICLES = 100
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Compile shaders
        self.program = ctx.program(
            vertex_shader=WIND_PARTICLE_VERTEX_SHADER,
            fragment_shader=WIND_PARTICLE_FRAGMENT_SHADER,
        )
        
        # Random generator (MUST be initialized first for _init_particles)
        self.rng = np.random.default_rng(42)
        
        # Camera matrices
        self.projection = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
        self.camera_pos = glm.vec3(0, 50, 0)
        
        # Wind state (updated from WindManager)
        self.wind_dir = (1.0, 0.0)
        self.wind_strength = 0.3
        self.has_tornado = False
        self.tornado_center = (0.0, 0.0)
        self.tornado_radius = 50.0
        self.tornado_strength = 1.0
        
        # Spawn area (around camera)
        self.spawn_radius = 150.0
        self.spawn_height_min = 1.0
        self.spawn_height_max = 30.0
        
        # Time
        self.time = 0.0
        
        # Particle data (CPU-side for updates)
        self.positions = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float32)
        self.velocities = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float32)
        self.colors = np.zeros((self.MAX_PARTICLES, 3), dtype=np.float32)
        self.sizes = np.zeros(self.MAX_PARTICLES, dtype=np.float32)
        self.rotations = np.zeros(self.MAX_PARTICLES, dtype=np.float32)
        self.lives = np.zeros(self.MAX_PARTICLES, dtype=np.float32)
        
        # Initialize with random positions and colors
        self._init_particles()
        
        # Create VBO (will be updated each frame)
        self._create_buffers()
    
    def _init_particles(self):
        """Initialize particles with random state."""
        for i in range(self.MAX_PARTICLES):
            self._respawn_particle(i, initial=True)
    
    def _respawn_particle(self, idx: int, initial: bool = False):
        """Respawn a particle at a random position near camera."""
        # Random position around camera
        angle = self.rng.random() * math.pi * 2
        dist = self.rng.random() * self.spawn_radius
        
        cx, cy, cz = self.camera_pos.x, self.camera_pos.y, self.camera_pos.z
        self.positions[idx] = [
            cx + math.cos(angle) * dist,
            cy + self.spawn_height_min + self.rng.random() * (self.spawn_height_max - self.spawn_height_min),
            cz + math.sin(angle) * dist,
        ]
        
        # Initial velocity (follow wind + random)
        self.velocities[idx] = [
            self.wind_dir[0] * self.wind_strength * 10 + self.rng.uniform(-2, 2),
            self.rng.uniform(-0.5, 0.5),
            self.wind_dir[1] * self.wind_strength * 10 + self.rng.uniform(-2, 2),
        ]
        
        # Color - earthy tones (leaves, dust, debris)
        color_type = self.rng.integers(0, 4)
        if color_type == 0:  # Green leaf
            self.colors[idx] = [0.3 + self.rng.random() * 0.2, 0.5 + self.rng.random() * 0.2, 0.2]
        elif color_type == 1:  # Brown leaf
            self.colors[idx] = [0.5 + self.rng.random() * 0.2, 0.35 + self.rng.random() * 0.15, 0.2]
        elif color_type == 2:  # Yellow leaf
            self.colors[idx] = [0.7 + self.rng.random() * 0.2, 0.6 + self.rng.random() * 0.2, 0.2]
        else:  # Dust/debris
            self.colors[idx] = [0.55, 0.5, 0.45]
        
        # Size and rotation
        self.sizes[idx] = 0.2 + self.rng.random() * 0.4
        self.rotations[idx] = self.rng.random() * math.pi * 2
        
        # Life (stagger initial spawns)
        if initial:
            self.lives[idx] = self.rng.random()
        else:
            self.lives[idx] = 1.0
    
    def _create_buffers(self):
        """Create GPU buffers for particle data."""
        # Pack only GPU-needed data (position, color, size, life)
        # Velocity is CPU-only for physics
        data = np.zeros(self.MAX_PARTICLES, dtype=[
            ('position', np.float32, 3),
            ('color', np.float32, 3),
            ('size', np.float32),
            ('life', np.float32),
        ])
        
        data['position'] = self.positions
        data['color'] = self.colors
        data['size'] = self.sizes
        data['life'] = self.lives
        
        self.vbo = self.ctx.buffer(data.tobytes())
        
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vbo, '3f 3f 1f 1f', 'in_position', 'in_color', 'in_size', 'in_life'),
            ],
        )
    
    def set_camera(self, projection, view, camera_pos):
        """Set camera matrices."""
        self.projection = projection
        self.view = view
        if isinstance(camera_pos, glm.vec3):
            self.camera_pos = camera_pos
        else:
            self.camera_pos = glm.vec3(*camera_pos)
    
    def set_wind(self, wind_dir: tuple, wind_strength: float,
                  has_tornado: bool = False, tornado_center: tuple = (0, 0),
                  tornado_radius: float = 50, tornado_strength: float = 1.0):
        """Set wind parameters."""
        self.wind_dir = wind_dir
        self.wind_strength = wind_strength
        self.has_tornado = has_tornado
        self.tornado_center = tornado_center
        self.tornado_radius = tornado_radius
        self.tornado_strength = tornado_strength
    
    def update(self, dt: float):
        """Update particle positions and states."""
        self.time += dt
        
        for i in range(self.MAX_PARTICLES):
            if self.lives[i] <= 0:
                self._respawn_particle(i)
                continue
            
            # Decay life
            self.lives[i] -= dt * 0.1  # ~10 seconds lifetime
            
            # Get position
            px, py, pz = self.positions[i]
            vx, vy, vz = self.velocities[i]
            
            # Base wind force
            wind_force_x = self.wind_dir[0] * self.wind_strength * 5
            wind_force_z = self.wind_dir[1] * self.wind_strength * 5
            
            # Tornado effect
            if self.has_tornado:
                dx = px - (self.camera_pos.x + self.tornado_center[0])
                dz = pz - (self.camera_pos.z + self.tornado_center[1])
                dist = math.sqrt(dx * dx + dz * dz)
                
                if dist < self.tornado_radius and dist > 0.1:
                    # Rotational force
                    falloff = 1.0 - (dist / self.tornado_radius)
                    tangent_x = -dz / dist
                    tangent_z = dx / dist
                    
                    # Inward pull
                    inward_x = -dx / dist
                    inward_z = -dz / dist
                    
                    tornado_force = self.tornado_strength * falloff * 15
                    wind_force_x += (tangent_x * 0.7 + inward_x * 0.3) * tornado_force
                    wind_force_z += (tangent_z * 0.7 + inward_z * 0.3) * tornado_force
                    
                    # Upward lift near tornado center
                    vy += falloff * self.tornado_strength * 3 * dt
            
            # Apply wind force to velocity
            vx += (wind_force_x - vx * 0.5) * dt  # Drag
            vz += (wind_force_z - vz * 0.5) * dt
            vy -= 1.0 * dt  # Gravity
            
            # Add turbulence
            vx += math.sin(self.time * 3 + i * 0.1) * 2 * dt
            vz += math.cos(self.time * 2.5 + i * 0.15) * 2 * dt
            
            # Update position
            px += vx * dt
            py += vy * dt
            pz += vz * dt
            
            # Respawn if too far from camera or hit ground
            cam_dist = math.sqrt(
                (px - self.camera_pos.x)**2 + 
                (pz - self.camera_pos.z)**2
            )
            if cam_dist > self.spawn_radius * 1.5 or py < 0:
                self._respawn_particle(i)
                continue
            
            # Store updated values
            self.positions[i] = [px, py, pz]
            self.velocities[i] = [vx, vy, vz]
            self.rotations[i] += (self.wind_strength * 2 + 1) * dt  # Spin
        
        # Update GPU buffer
        self._update_buffer()
    
    def _update_buffer(self):
        """Update GPU buffer with current particle state."""
        # Only GPU-needed data (matches _create_buffers format)
        data = np.zeros(self.MAX_PARTICLES, dtype=[
            ('position', np.float32, 3),
            ('color', np.float32, 3),
            ('size', np.float32),
            ('life', np.float32),
        ])
        
        data['position'] = self.positions
        data['color'] = self.colors
        data['size'] = self.sizes
        data['life'] = self.lives
        
        self.vbo.write(data.tobytes())
    
    def render(self):
        """Render wind particles."""
        # Set uniforms (only set if they exist - shader may optimize them out)
        self.program['u_projection'].write(self.projection)
        self.program['u_view'].write(self.view)
        # u_time is declared but not used in shader, so it gets optimized out
        # Skip setting it to avoid KeyError
        
        # Enable point sprites and blending
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        # Render as points (point sprites)
        self.vao.render(moderngl.POINTS, vertices=self.MAX_PARTICLES)
        
        # Disable
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.PROGRAM_POINT_SIZE)
    
    def cleanup(self):
        """Release GPU resources."""
        self.vbo.release()
        self.vao.release()
        self.program.release()

