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

// Cloud layer settings
const float CLOUD_MIN_HEIGHT = 800.0;
const float CLOUD_MAX_HEIGHT = 1500.0;
const float CLOUD_THICKNESS = 700.0;

// Ray march settings
const int MAX_STEPS = 32;
const float STEP_SIZE = 50.0;

// Noise functions
float hash(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise3D(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    return mix(
        mix(mix(hash(i + vec3(0,0,0)), hash(i + vec3(1,0,0)), f.x),
            mix(hash(i + vec3(0,1,0)), hash(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash(i + vec3(0,0,1)), hash(i + vec3(1,0,1)), f.x),
            mix(hash(i + vec3(0,1,1)), hash(i + vec3(1,1,1)), f.x), f.y),
        f.z
    );
}

// Fractal Brownian Motion for cloud density
float fbm(vec3 p, int octaves) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < octaves; i++) {
        value += amplitude * noise3D(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}

// Sample cloud density at a point
float sampleCloud(vec3 pos) {
    // Normalize height within cloud layer
    float height_factor = (pos.y - CLOUD_MIN_HEIGHT) / CLOUD_THICKNESS;
    if (height_factor < 0.0 || height_factor > 1.0) return 0.0;
    
    // Vertical density profile (thicker in middle)
    float height_density = 1.0 - abs(height_factor - 0.5) * 2.0;
    height_density = smoothstep(0.0, 0.3, height_density);
    
    // Wind animation
    vec3 wind_offset = vec3(u_time * u_cloud_speed * 20.0, 0.0, u_time * u_cloud_speed * 5.0);
    vec3 sample_pos = pos * 0.001 + wind_offset * 0.001;
    
    // Multi-octave noise for cloud shape
    float density = fbm(sample_pos * 2.0, 4);
    
    // Add detail noise
    density += fbm(sample_pos * 8.0, 3) * 0.25;
    
    // Coverage threshold
    float coverage_threshold = 1.0 - u_cloud_coverage;
    density = smoothstep(coverage_threshold, coverage_threshold + 0.3, density);
    
    return density * height_density;
}

// Light scattering approximation
float lightMarch(vec3 pos) {
    vec3 light_dir = normalize(u_sun_dir);
    float light_step = 30.0;
    float total_density = 0.0;
    
    // March toward sun
    for (int i = 0; i < 6; i++) {
        pos += light_dir * light_step;
        total_density += sampleCloud(pos) * 0.5;
    }
    
    // Beer-Lambert law for light absorption
    return exp(-total_density * 0.5);
}

void main() {
    vec3 ray_dir = normalize(v_ray_dir);
    
    // Only render clouds above horizon
    if (ray_dir.y < 0.01) {
        fragColor = vec4(0.0);
        return;
    }
    
    // Calculate ray intersection with cloud layer
    float t_min = (CLOUD_MIN_HEIGHT - u_camera_pos.y) / ray_dir.y;
    float t_max = (CLOUD_MAX_HEIGHT - u_camera_pos.y) / ray_dir.y;
    
    if (t_min > t_max) {
        float temp = t_min;
        t_min = t_max;
        t_max = temp;
    }
    
    t_min = max(t_min, 0.0);
    
    // Skip if cloud layer is behind us
    if (t_max < 0.0) {
        fragColor = vec4(0.0);
        return;
    }
    
    // Ray march through cloud layer
    vec3 pos = u_camera_pos + ray_dir * t_min;
    float step = min(STEP_SIZE, (t_max - t_min) / float(MAX_STEPS));
    
    float transmittance = 1.0;
    vec3 accumulated_color = vec3(0.0);
    
    // Day/night cloud colors
    float day_factor = smoothstep(0.2, 0.35, u_time_of_day) * smoothstep(0.8, 0.65, u_time_of_day);
    float sunset_factor = smoothstep(0.15, 0.25, u_time_of_day) * smoothstep(0.35, 0.25, u_time_of_day)
                        + smoothstep(0.65, 0.75, u_time_of_day) * smoothstep(0.85, 0.75, u_time_of_day);
    
    vec3 cloud_color_day = vec3(1.0, 1.0, 1.0);
    vec3 cloud_color_sunset = vec3(1.0, 0.7, 0.5);
    vec3 cloud_color_night = vec3(0.15, 0.15, 0.2);
    
    vec3 base_cloud_color = cloud_color_day * day_factor 
                          + cloud_color_sunset * sunset_factor 
                          + cloud_color_night * (1.0 - day_factor - sunset_factor);
    
    // Sun direction for lighting
    vec3 sun_dir = normalize(u_sun_dir);
    
    for (int i = 0; i < MAX_STEPS; i++) {
        if (transmittance < 0.01) break;
        
        float density = sampleCloud(pos);
        
        if (density > 0.01) {
            // Light contribution at this point
            float light = lightMarch(pos);
            
            // Silver lining effect (bright edges toward sun)
            float sun_dot = dot(ray_dir, sun_dir);
            float silver_lining = pow(max(sun_dot, 0.0), 8.0) * 0.5;
            
            // Combine lighting
            vec3 cloud_color = base_cloud_color * (0.3 + light * 0.7 + silver_lining);
            
            // Add ambient light from sky
            cloud_color += vec3(0.1, 0.12, 0.15) * day_factor;
            
            // Accumulate
            float alpha = density * step * 0.02;
            accumulated_color += cloud_color * alpha * transmittance;
            transmittance *= 1.0 - alpha;
        }
        
        pos += ray_dir * step;
    }
    
    // Output with alpha for blending
    float alpha = 1.0 - transmittance;
    
    // Fade clouds at horizon to blend with sky
    float horizon_fade = smoothstep(0.0, 0.15, ray_dir.y);
    alpha *= horizon_fade;
    
    fragColor = vec4(accumulated_color, alpha);
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

