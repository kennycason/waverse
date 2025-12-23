"""
ModernGL Volumetric Cloud Renderer

GPU-accelerated volumetric clouds with:
- Ray marching through noise-based density field
- Light scattering and silver lining effect
- Wind animation
- Day/night color transitions
- Optimized for real-time performance
"""

import numpy as np
import moderngl
from typing import Tuple
import math

try:
    from pyglm import glm
except ImportError:
    import glm


# =============================================================================
# SHADERS
# =============================================================================

CLOUD_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;

out vec3 v_ray_dir;
out vec2 v_uv;

uniform mat4 u_inv_projection;
uniform mat4 u_inv_view;

void main() {
    v_uv = in_position.xy * 0.5 + 0.5;
    
    // Create ray direction from screen position
    vec4 clip = vec4(in_position.xy, 1.0, 1.0);
    vec4 view = u_inv_projection * clip;
    view = vec4(view.xy, -1.0, 0.0);
    vec3 world_dir = (u_inv_view * view).xyz;
    
    v_ray_dir = normalize(world_dir);
    gl_Position = vec4(in_position.xy, 0.999, 1.0);
}
"""

CLOUD_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_ray_dir;
in vec2 v_uv;

out vec4 fragColor;

uniform vec3 u_camera_pos;
uniform vec3 u_sun_dir;
uniform float u_time;
uniform float u_time_of_day;  // 0-1, 0.5 = noon
uniform float u_cloud_coverage;  // 0-1, how cloudy
uniform float u_cloud_speed;  // Wind speed

// =============================================================================
// STYLIZED LOW-POLY CLOUDS
// Chunky, geometric shapes that match the game's aesthetic
// =============================================================================

// Cloud layer settings
const float CLOUD_MIN_HEIGHT = 600.0;
const float CLOUD_MAX_HEIGHT = 1000.0;
const float CLOUD_THICKNESS = 400.0;

// Fewer steps for chunkier look
const int MAX_STEPS = 16;
const float STEP_SIZE = 80.0;

// Hash for randomness
float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

// CHUNKY cell-based noise - creates blocky cloud shapes
float chunkyNoise(vec3 p) {
    // Quantize position to create blocky cells
    vec3 cell = floor(p);
    
    // Get hash value for this cell
    float h = hash(cell);
    
    // Sharp falloff from cell center (creates puffy chunks)
    vec3 f = fract(p);
    vec3 center_dist = abs(f - 0.5) * 2.0;
    float dist_from_center = max(max(center_dist.x, center_dist.y), center_dist.z);
    
    // Hard edge with slight softening
    float edge = 1.0 - step(0.7, dist_from_center);
    
    return h * edge;
}

// Multi-scale chunky clouds - SPARSE version
float cloudDensity(vec3 p) {
    float density = 0.0;
    
    // Large chunks only (fewer, bigger cloud shapes)
    density += chunkyNoise(p * 0.15) * 0.7;
    
    // Occasional medium bumps
    density += chunkyNoise(p * 0.4 + 7.3) * 0.2;
    
    // Very sparse - subtract some noise to create gaps
    density -= chunkyNoise(p * 0.25 + 23.1) * 0.3;
    
    return max(0.0, density);
}

// Sample cloud at position
float sampleCloud(vec3 pos) {
    // Height within cloud layer
    float height_factor = (pos.y - CLOUD_MIN_HEIGHT) / CLOUD_THICKNESS;
    if (height_factor < 0.0 || height_factor > 1.0) return 0.0;
    
    // Vertical profile - flatter on bottom, rounded on top (cumulus style)
    float height_shape = 1.0;
    if (height_factor < 0.3) {
        height_shape = step(0.1, height_factor);  // Flat bottom
    } else {
        height_shape = 1.0 - pow((height_factor - 0.3) / 0.7, 2.0);  // Rounded top
    }
    
    // Wind animation (slower, more chunky movement)
    vec3 wind = vec3(u_time * u_cloud_speed * 8.0, 0.0, u_time * u_cloud_speed * 2.0);
    vec3 sample_pos = pos * 0.003 + wind * 0.003;
    
    // Get chunky density
    float density = cloudDensity(sample_pos);
    
    // Coverage threshold with HARD cutoff (stylized) - HIGHER threshold = sparser clouds
    float threshold = 0.55 - u_cloud_coverage * 0.3;
    density = step(threshold, density) * (density - threshold) / (1.0 - threshold);
    
    // Additional sparsity - only keep strongest clouds
    density = pow(density, 1.5);
    
    return density * height_shape;
}

void main() {
    vec3 ray_dir = normalize(v_ray_dir);
    
    // Only render above horizon
    if (ray_dir.y < 0.02) {
        fragColor = vec4(0.0);
        return;
    }
    
    // Ray-cloud layer intersection
    float t_min = (CLOUD_MIN_HEIGHT - u_camera_pos.y) / ray_dir.y;
    float t_max = (CLOUD_MAX_HEIGHT - u_camera_pos.y) / ray_dir.y;
    
    if (t_min > t_max) {
        float temp = t_min;
        t_min = t_max;
        t_max = temp;
    }
    t_min = max(t_min, 0.0);
    
    if (t_max < 0.0) {
        fragColor = vec4(0.0);
        return;
    }
    
    // Ray march
    vec3 pos = u_camera_pos + ray_dir * t_min;
    float step_size = min(STEP_SIZE, (t_max - t_min) / float(MAX_STEPS));
    
    float total_density = 0.0;
    float max_density = 0.0;
    
    for (int i = 0; i < MAX_STEPS; i++) {
        float d = sampleCloud(pos);
        total_density += d * step_size * 0.005;
        max_density = max(max_density, d);
        pos += ray_dir * step_size;
    }
    
    // Day/night colors (bold, saturated)
    float day = smoothstep(0.2, 0.35, u_time_of_day) * smoothstep(0.8, 0.65, u_time_of_day);
    float sunset = smoothstep(0.15, 0.25, u_time_of_day) * smoothstep(0.35, 0.25, u_time_of_day)
                 + smoothstep(0.65, 0.75, u_time_of_day) * smoothstep(0.85, 0.75, u_time_of_day);
    float night = 1.0 - day - sunset;
    
    // Flat, stylized colors (no complex lighting)
    vec3 cloud_bright = vec3(1.0, 1.0, 1.0);      // Day: pure white
    vec3 cloud_shadow = vec3(0.7, 0.75, 0.85);    // Day shadow: slight blue-gray
    vec3 cloud_sunset = vec3(1.0, 0.65, 0.45);    // Sunset: coral orange
    vec3 cloud_night = vec3(0.25, 0.25, 0.35);    // Night: dark blue-gray
    
    // Simple flat shading based on sun direction
    vec3 sun_dir = normalize(u_sun_dir);
    float sun_facing = dot(ray_dir, sun_dir) * 0.5 + 0.5;
    
    // Mix bright/shadow based on ray direction (simple stylized lighting)
    vec3 day_color = mix(cloud_shadow, cloud_bright, sun_facing * 0.5 + 0.5);
    
    vec3 cloud_color = day_color * day 
                     + cloud_sunset * sunset 
                     + cloud_night * night;
    
    // Alpha with hard edges - sparser clouds
    float alpha = clamp(total_density * 1.5, 0.0, 0.9);
    
    // Even sharper alpha cutoff for fewer, distinct clouds
    alpha = smoothstep(0.15, 0.4, alpha);
    
    // Horizon fade
    float horizon_fade = smoothstep(0.02, 0.12, ray_dir.y);
    alpha *= horizon_fade;
    
    fragColor = vec4(cloud_color, alpha);
}
"""


# =============================================================================
# RENDERER
# =============================================================================

class ModernCloudRenderer:
    """
    GPU-accelerated volumetric cloud renderer.
    
    Uses ray marching with 3D noise to create realistic volumetric clouds.
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Compile shaders
        self.program = ctx.program(
            vertex_shader=CLOUD_VERTEX_SHADER,
            fragment_shader=CLOUD_FRAGMENT_SHADER,
        )
        
        # Create fullscreen quad
        vertices = np.array([
            -1.0, -1.0, 0.0,
             1.0, -1.0, 0.0,
            -1.0,  1.0, 0.0,
             1.0,  1.0, 0.0,
        ], dtype='f4')
        
        self.vbo = ctx.buffer(vertices.tobytes())
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, '3f', 'in_position')],
        )
        
        # State
        self.time = 0.0
        self.time_of_day = 0.5  # Noon
        self.cloud_coverage = 0.5  # 50% cloudy
        self.cloud_speed = 1.0  # Normal wind
        
        self.camera_pos = glm.vec3(0, 0, 0)
        self.sun_dir = glm.vec3(0.5, 1.0, 0.3)
        
        # Matrices
        self.inv_projection = glm.mat4(1.0)
        self.inv_view = glm.mat4(1.0)
        
        # Stats
        self.frame_stats = {
            'draw_calls': 1,
        }
    
    def set_camera(self, projection: glm.mat4, view: glm.mat4, camera_pos: glm.vec3):
        """Set camera matrices (we need inverses for ray generation)."""
        self.inv_projection = glm.inverse(projection)
        self.inv_view = glm.inverse(view)
        self.camera_pos = camera_pos
    
    def set_time(self, time_of_day: float):
        """Set time of day and calculate sun direction."""
        self.time_of_day = time_of_day
        
        # Calculate sun position
        sun_angle = (time_of_day - 0.25) * math.pi * 2
        self.sun_dir = glm.vec3(
            math.cos(sun_angle) * 0.5,
            max(0.1, math.sin(sun_angle)),
            0.3
        )
    
    def set_weather(self, cloud_coverage: float, wind_speed: float = 1.0):
        """Set cloud coverage and wind speed."""
        self.cloud_coverage = max(0.0, min(1.0, cloud_coverage))
        self.cloud_speed = wind_speed
    
    def render(self, dt: float = 0.016):
        """Render volumetric clouds."""
        self.time += dt
        
        # Skip if no clouds
        if self.cloud_coverage < 0.05:
            return
        
        # Enable blending (clouds are semi-transparent)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        # Don't write to depth (clouds are background elements)
        # Note: depth_mask doesn't exist on ctx, so we handle via rendering order
        
        # Set uniforms
        self.program['u_inv_projection'].write(self.inv_projection)
        self.program['u_inv_view'].write(self.inv_view)
        self.program['u_camera_pos'].write(self.camera_pos)
        self.program['u_sun_dir'].write(self.sun_dir)
        self.program['u_time'].value = self.time
        self.program['u_time_of_day'].value = self.time_of_day
        self.program['u_cloud_coverage'].value = self.cloud_coverage
        self.program['u_cloud_speed'].value = self.cloud_speed
        
        # Render fullscreen
        self.vao.render(moderngl.TRIANGLE_STRIP)
        
        # Restore state
        self.ctx.disable(moderngl.BLEND)
    
    def cleanup(self):
        """Release GPU resources."""
        self.vbo.release()
        self.vao.release()
        self.program.release()

