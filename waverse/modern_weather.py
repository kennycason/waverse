"""
ModernGL Weather Renderer

GPU-accelerated weather particle effects:
- Rain with splash effects
- Snow with accumulation
- Fog volume
- Lightning flashes
"""

import numpy as np
import moderngl
from typing import Tuple, Optional
import math
import random

try:
    from pyglm import glm
except ImportError:
    import glm


# =============================================================================
# SHADERS
# =============================================================================

PARTICLE_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;      // Base position
in vec3 in_velocity;      // Velocity for animation
in float in_life;         // Life/phase (0-1)
in float in_size;         // Particle size

out float v_alpha;
out float v_life;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform float u_time;
uniform vec3 u_camera_pos;
uniform float u_particle_type;  // 0=rain, 1=snow

void main() {
    // Animate position based on velocity and time
    // Particles loop in a volume around camera
    vec3 pos = in_position;
    
    // Apply velocity with time-based offset
    float phase = fract(in_life + u_time * 0.1);  // Loop phase
    pos += in_velocity * phase * 10.0;
    
    // Wrap particles around camera (create infinite rain/snow effect)
    float range = 100.0;
    pos.x = mod(pos.x - u_camera_pos.x + range, range * 2.0) - range + u_camera_pos.x;
    pos.z = mod(pos.z - u_camera_pos.z + range, range * 2.0) - range + u_camera_pos.z;
    
    // Height wrapping
    float height_range = 50.0;
    pos.y = mod(pos.y - u_camera_pos.y + height_range, height_range * 2.0) - height_range + u_camera_pos.y;
    
    v_life = phase;
    
    // Alpha based on distance from camera
    float dist = length(pos - u_camera_pos);
    v_alpha = 1.0 - smoothstep(50.0, 100.0, dist);
    
    gl_Position = u_projection * u_view * vec4(pos, 1.0);
    
    // Point size based on distance and particle size
    // Cap to prevent massive particles when very close
    float point_size = in_size * 200.0 / max(5.0, dist);
    point_size = clamp(point_size, 1.0, 20.0);  // Limit size
    
    // Snow is larger, rain is streaky
    if (u_particle_type > 0.5) {
        gl_PointSize = min(point_size * 2.0, 30.0);
    } else {
        gl_PointSize = point_size;
    }
}
"""

PARTICLE_FRAGMENT_SHADER = """
#version 330 core

in float v_alpha;
in float v_life;

out vec4 fragColor;

uniform float u_particle_type;  // 0=rain, 1=snow
uniform vec3 u_particle_color;

void main() {
    // Point sprite coordinates
    vec2 coord = gl_PointCoord - vec2(0.5);
    
    float alpha = v_alpha;
    vec3 color = u_particle_color;
    
    if (u_particle_type > 0.5) {
        // Snow: soft circular
        float dist = length(coord);
        alpha *= smoothstep(0.5, 0.3, dist);
        color = vec3(1.0, 1.0, 1.0);  // White snow
    } else {
        // Rain: vertical streak
        float stretch = abs(coord.y) * 2.0;
        float width = abs(coord.x) * 4.0;
        alpha *= smoothstep(1.0, 0.0, stretch) * smoothstep(0.5, 0.0, width);
        color = vec3(0.7, 0.75, 0.85);  // Bluish rain
    }
    
    if (alpha < 0.01) discard;
    
    fragColor = vec4(color, alpha * 0.6);
}
"""

# Lightning flash (fullscreen)
FLASH_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;

void main() {
    gl_Position = vec4(in_position.xy, 0.99, 1.0);
}
"""

FLASH_FRAGMENT_SHADER = """
#version 330 core

out vec4 fragColor;

uniform float u_intensity;  // 0-1 flash intensity

void main() {
    vec3 color = vec3(0.9, 0.9, 1.0);  // Bluish white
    fragColor = vec4(color, u_intensity);
}
"""


# =============================================================================
# RENDERER
# =============================================================================

class ModernWeatherRenderer:
    """
    GPU-accelerated weather particle system.
    """
    
    def __init__(self, ctx: moderngl.Context, max_particles: int = 10000):
        self.ctx = ctx
        self.max_particles = max_particles
        
        # Particle shader
        self.particle_program = ctx.program(
            vertex_shader=PARTICLE_VERTEX_SHADER,
            fragment_shader=PARTICLE_FRAGMENT_SHADER,
        )
        
        # Flash shader
        self.flash_program = ctx.program(
            vertex_shader=FLASH_VERTEX_SHADER,
            fragment_shader=FLASH_FRAGMENT_SHADER,
        )
        
        # Generate particle data
        self._init_particles()
        
        # Create flash quad
        flash_verts = np.array([-1, -1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0], dtype='f4')
        self.flash_vbo = ctx.buffer(flash_verts.tobytes())
        self.flash_vao = ctx.vertex_array(
            self.flash_program,
            [(self.flash_vbo, '3f', 'in_position')]
        )
        
        # State
        self.time = 0.0
        self.weather_type = 'clear'  # 'clear', 'rain', 'heavy_rain', 'snow', 'storm'
        self.intensity = 0.0  # 0-1 precipitation intensity
        self.lightning_flash = 0.0  # 0-1 flash intensity
        self.lightning_timer = 0.0
        
        # Camera
        self.projection = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
        self.camera_pos = glm.vec3(0, 0, 0)
        
        # Stats
        self.frame_stats = {
            'particles_rendered': 0,
            'draw_calls': 0,
        }
    
    def _init_particles(self):
        """Initialize particle buffer with random positions/velocities."""
        # Each particle: position (3) + velocity (3) + life (1) + size (1) = 8 floats
        data = []
        
        for _ in range(self.max_particles):
            # Random position in a large volume
            x = random.uniform(-100, 100)
            y = random.uniform(0, 50)
            z = random.uniform(-100, 100)
            
            # Rain falls down, slight wind
            vx = random.uniform(-0.5, 0.5)
            vy = random.uniform(-8, -5)  # Falling
            vz = random.uniform(-0.5, 0.5)
            
            life = random.random()
            size = random.uniform(0.5, 1.5)
            
            data.extend([x, y, z, vx, vy, vz, life, size])
        
        self.particle_data = np.array(data, dtype='f4')
        self.particle_vbo = self.ctx.buffer(self.particle_data.tobytes())
        
        self.particle_vao = self.ctx.vertex_array(
            self.particle_program,
            [(self.particle_vbo, '3f 3f 1f 1f', 
              'in_position', 'in_velocity', 'in_life', 'in_size')]
        )
    
    def set_camera(self, projection: glm.mat4, view: glm.mat4, camera_pos: glm.vec3):
        """Set camera matrices."""
        self.projection = projection
        self.view = view
        self.camera_pos = camera_pos
    
    def set_weather(self, weather_type: str, intensity: float = 1.0, biome: str = None):
        """
        Set weather conditions.
        
        Args:
            weather_type: 'clear', 'rain', 'heavy_rain', 'snow', 'storm', or special biome types
            intensity: 0-1 precipitation intensity
            biome: Optional biome for special effects
        """
        self.weather_type = weather_type
        self.intensity = intensity
        
        # === SPECIAL EXOTIC BIOME WEATHER ===
        if biome == 'psychedelic' or weather_type == 'rainbow':
            # RAINBOW SPARKLES - slow floating, all directions!
            for i in range(0, len(self.particle_data), 8):
                self.particle_data[i+3] = random.uniform(-2, 2)  # vx - swirl
                self.particle_data[i+4] = random.uniform(-1, 2)  # vy - some go UP!
                self.particle_data[i+5] = random.uniform(-2, 2)  # vz
                self.particle_data[i+7] = random.uniform(1.5, 3.0)  # BIG sparkles
            self.particle_vbo.write(self.particle_data.tobytes())
            self.particle_color = (0.9, 0.5, 0.9, 0.8)  # Magenta base (shader adds rainbow)
            return
        
        if biome == 'hellfire' or weather_type == 'ember':
            # EMBER/ASH - particles rise from below with embers
            for i in range(0, len(self.particle_data), 8):
                self.particle_data[i+3] = random.uniform(-1, 1)  # vx
                self.particle_data[i+4] = random.uniform(1, 4)  # vy - RISES!
                self.particle_data[i+5] = random.uniform(-1, 1)  # vz
                self.particle_data[i+7] = random.uniform(0.5, 2.0)
            self.particle_vbo.write(self.particle_data.tobytes())
            self.particle_color = (1.0, 0.4, 0.1, 0.9)  # Orange embers
            return
        
        if biome == 'shadow' or weather_type == 'dark_fog':
            # DARK FOG - slow drifting darkness
            for i in range(0, len(self.particle_data), 8):
                self.particle_data[i+3] = random.uniform(-0.3, 0.3)  # vx - slow
                self.particle_data[i+4] = random.uniform(-0.2, 0.2)  # vy - barely moves
                self.particle_data[i+5] = random.uniform(-0.3, 0.3)  # vz
                self.particle_data[i+7] = random.uniform(3.0, 6.0)  # LARGE fog patches
            self.particle_vbo.write(self.particle_data.tobytes())
            self.particle_color = (0.1, 0.05, 0.15, 0.5)  # Dark purple fog
            return
        
        if biome == 'crystal' or weather_type == 'shimmer':
            # CRYSTAL SHIMMER - slow falling sparkles
            for i in range(0, len(self.particle_data), 8):
                self.particle_data[i+3] = random.uniform(-0.5, 0.5)
                self.particle_data[i+4] = random.uniform(-1, -0.3)  # Gentle fall
                self.particle_data[i+5] = random.uniform(-0.5, 0.5)
                self.particle_data[i+7] = random.uniform(1.0, 2.5)
            self.particle_vbo.write(self.particle_data.tobytes())
            self.particle_color = (0.8, 0.95, 1.0, 0.7)  # Pale cyan sparkle
            return
        
        if biome == 'void':
            # VOID - barely any particles, very faint
            self.intensity = 0.1  # Almost nothing
            for i in range(0, len(self.particle_data), 8):
                self.particle_data[i+3] = random.uniform(-0.1, 0.1)
                self.particle_data[i+4] = random.uniform(-0.1, 0.1)
                self.particle_data[i+5] = random.uniform(-0.1, 0.1)
                self.particle_data[i+7] = random.uniform(0.5, 1.0)
            self.particle_vbo.write(self.particle_data.tobytes())
            self.particle_color = (0.05, 0.02, 0.08, 0.3)  # Barely visible
            return
        
        # === NORMAL WEATHER ===
        # Adjust particle velocities for weather type
        if weather_type == 'snow':
            # Snow falls slower, more horizontal drift
            for i in range(0, len(self.particle_data), 8):
                self.particle_data[i+3] = random.uniform(-1, 1)  # vx
                self.particle_data[i+4] = random.uniform(-2, -1)  # vy (slower)
                self.particle_data[i+5] = random.uniform(-1, 1)  # vz
                self.particle_data[i+7] = random.uniform(1.0, 2.0)  # larger
            self.particle_vbo.write(self.particle_data.tobytes())
            self.particle_color = (1.0, 1.0, 1.0, 0.8)  # White
        elif weather_type in ('rain', 'heavy_rain', 'storm'):
            # Rain falls fast
            for i in range(0, len(self.particle_data), 8):
                self.particle_data[i+3] = random.uniform(-0.5, 0.5)  # vx
                self.particle_data[i+4] = random.uniform(-10, -6)  # vy (fast)
                self.particle_data[i+5] = random.uniform(-0.5, 0.5)  # vz
                self.particle_data[i+7] = random.uniform(0.3, 1.0)  # smaller
            self.particle_vbo.write(self.particle_data.tobytes())
            self.particle_color = (0.6, 0.7, 0.8, 0.6)  # Blue-gray
    
    def trigger_lightning(self):
        """Trigger a lightning flash."""
        self.lightning_flash = 1.0
        self.lightning_timer = 0.15  # Flash duration in seconds
    
    def render(self, dt: float = 0.016):
        """Render weather effects."""
        self.time += dt
        
        # Handle lightning timer
        if self.lightning_timer > 0:
            self.lightning_timer -= dt
            if self.lightning_timer <= 0:
                self.lightning_flash = 0.0
        
        # Random lightning in storms
        if self.weather_type == 'storm' and random.random() < 0.005:
            self.trigger_lightning()
        
        self.frame_stats['particles_rendered'] = 0
        self.frame_stats['draw_calls'] = 0
        
        # Skip rendering if clear weather or no precipitation
        if self.weather_type in ('clear', 'none', '') or self.intensity < 0.01:
            return
        
        # Determine particle count based on intensity
        # Cap particle count to prevent overdraw issues
        max_render = min(self.max_particles, 3000)  # Cap for performance
        particle_count = int(max_render * self.intensity)
        if self.weather_type == 'heavy_rain':
            particle_count = min(int(particle_count * 1.5), max_render)
        elif self.weather_type == 'storm':
            particle_count = max_render
        
        # Enable blending and disable depth write (particles shouldn't occlude)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.depth_func = '<='  # Allow particles at same depth
        
        # Disable depth write to prevent particles from blocking scene
        # Particles should be see-through
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        
        # Set uniforms
        self.particle_program['u_projection'].write(self.projection)
        self.particle_program['u_view'].write(self.view)
        self.particle_program['u_time'].value = self.time
        self.particle_program['u_camera_pos'].write(self.camera_pos)
        
        # Particle type: 0=rain, 1=snow
        particle_type = 1.0 if self.weather_type == 'snow' else 0.0
        self.particle_program['u_particle_type'].value = particle_type
        
        # Set color based on weather type
        if self.weather_type == 'snow':
            color = glm.vec3(0.95, 0.97, 1.0)  # White-ish snow
        else:
            color = glm.vec3(0.6, 0.7, 0.85)  # Blue-ish rain
        self.particle_program['u_particle_color'].write(color)
        
        # Render particles as points
        self.particle_vao.render(moderngl.POINTS, vertices=particle_count)
        
        self.frame_stats['particles_rendered'] = particle_count
        self.frame_stats['draw_calls'] += 1
        
        # Render lightning flash
        if self.lightning_flash > 0.01:
            self.flash_program['u_intensity'].value = self.lightning_flash * 0.3
            self.flash_vao.render(moderngl.TRIANGLE_STRIP)
            self.frame_stats['draw_calls'] += 1
        
        # Disable blending
        self.ctx.disable(moderngl.BLEND)
        self.ctx.disable(moderngl.PROGRAM_POINT_SIZE)
    
    def cleanup(self):
        """Release GPU resources."""
        self.particle_vbo.release()
        self.particle_vao.release()
        self.particle_program.release()
        self.flash_vbo.release()
        self.flash_vao.release()
        self.flash_program.release()

