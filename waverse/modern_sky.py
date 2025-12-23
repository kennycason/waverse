"""
ModernGL Sky Renderer

GPU-accelerated sky rendering with:
- Procedural atmospheric scattering
- Day/night cycle with sun and moon
- Stars at night
- Cloud layers
- Horizon blending
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

SKY_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;

out vec3 v_ray_dir;

uniform mat4 u_projection;
uniform mat4 u_view;

void main() {
    // Create view-space ray direction
    vec4 clip_pos = vec4(in_position.xy, 1.0, 1.0);
    vec4 view_pos = inverse(u_projection) * clip_pos;
    view_pos = vec4(view_pos.xy, -1.0, 0.0);
    vec3 world_dir = (inverse(u_view) * view_pos).xyz;
    
    v_ray_dir = normalize(world_dir);
    gl_Position = vec4(in_position.xy, 0.9999, 1.0);  // Behind everything
}
"""

SKY_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_ray_dir;

out vec4 fragColor;

uniform float u_time_of_day;  // 0.0 = midnight, 0.5 = noon, 1.0 = midnight
uniform vec3 u_sun_dir;
uniform vec3 u_camera_pos;

// =============================================================================
// STYLIZED/PLAYFUL SKY - Bold colors, simple gradients, geometric feel
// =============================================================================

// Night: Deep indigo/purple with a touch of warmth
const vec3 NIGHT_ZENITH = vec3(0.08, 0.05, 0.18);    // Deep purple-blue
const vec3 NIGHT_HORIZON = vec3(0.12, 0.08, 0.22);   // Slightly lighter purple

// Dawn/Dusk: Vibrant oranges/pinks/corals
const vec3 SUNRISE_ZENITH = vec3(0.35, 0.25, 0.55);   // Purple-pink
const vec3 SUNRISE_HORIZON = vec3(1.0, 0.55, 0.35);   // Warm coral-orange

// Day: Bright, playful sky blue with warmth
const vec3 DAY_ZENITH = vec3(0.35, 0.65, 0.95);       // Bright saturated sky blue
const vec3 DAY_HORIZON = vec3(0.75, 0.88, 0.95);      // Light peachy horizon

// Sunset: Rich warm tones  
const vec3 SUNSET_ZENITH = vec3(0.45, 0.30, 0.60);    // Purple
const vec3 SUNSET_HORIZON = vec3(1.0, 0.45, 0.25);    // Vivid orange-red

// Sun/moon - stylized, slightly larger
const vec3 SUN_COLOR = vec3(1.0, 0.95, 0.7);   // Warm yellow
const vec3 MOON_COLOR = vec3(0.95, 0.93, 1.0);  // Cool white-blue

// Star hash
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec3 getSkyColor(float y, float time) {
    // Calculate time phases (sharper transitions for stylized look)
    float night = smoothstep(0.8, 0.95, time) + smoothstep(0.2, 0.05, time);
    float sunrise = smoothstep(0.18, 0.25, time) * smoothstep(0.38, 0.28, time);
    float day = smoothstep(0.32, 0.45, time) * smoothstep(0.68, 0.55, time);
    float sunset = smoothstep(0.62, 0.72, time) * smoothstep(0.88, 0.78, time);
    
    // Normalize
    float total = night + sunrise + day + sunset + 0.001;
    night /= total; sunrise /= total; day /= total; sunset /= total;
    
    // Horizon blend - steeper for more graphic look
    float horizon = smoothstep(-0.05, 0.35, y);
    
    // Blend zenith colors
    vec3 zenith = NIGHT_ZENITH * night + SUNRISE_ZENITH * sunrise 
                + DAY_ZENITH * day + SUNSET_ZENITH * sunset;
    
    // Blend horizon colors  
    vec3 hor = NIGHT_HORIZON * night + SUNRISE_HORIZON * sunrise 
             + DAY_HORIZON * day + SUNSET_HORIZON * sunset;
    
    // Simple two-band gradient
    return mix(hor, zenith, horizon);
}

float starField(vec3 dir) {
    if (dir.y < 0.0) return 0.0;
    
    vec2 uv = dir.xz / (dir.y + 1.0) * 80.0;
    vec2 grid = floor(uv);
    float star = 0.0;
    
    // Fewer, brighter stars for stylized look
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            vec2 cell = grid + vec2(dx, dy);
            float h = hash(cell);
            
            if (h > 0.96) {
                vec2 star_pos = cell + vec2(hash(cell + 0.1), hash(cell + 0.2));
                float d = length(uv - star_pos);
                // Sharper stars (more geometric)
                float brightness = step(d, 0.08 + h * 0.05);
                star += brightness * (0.5 + h * 0.5);
            }
        }
    }
    
    return star;
}

void main() {
    vec3 dir = normalize(v_ray_dir);
    
    // Base sky gradient
    vec3 color = getSkyColor(dir.y, u_time_of_day);
    
    // Sun - larger, flatter disc for stylized look
    float sun_dot = dot(dir, normalize(u_sun_dir));
    float sun_disk = smoothstep(0.995, 0.998, sun_dot);  // Larger disc
    float sun_halo = smoothstep(0.97, 0.995, sun_dot) * 0.3;  // Soft halo
    
    float day_factor = smoothstep(0.2, 0.32, u_time_of_day) * smoothstep(0.8, 0.68, u_time_of_day);
    color = mix(color, SUN_COLOR, (sun_disk + sun_halo) * day_factor);
    
    // Moon - simple circle
    vec3 moon_dir = -u_sun_dir;
    float moon_dot = dot(dir, normalize(moon_dir));
    float moon_disk = smoothstep(0.993, 0.997, moon_dot);
    float night_factor = smoothstep(0.28, 0.15, u_time_of_day) + smoothstep(0.72, 0.85, u_time_of_day);
    night_factor = clamp(night_factor, 0.0, 1.0);
    color = mix(color, MOON_COLOR, moon_disk * night_factor * 0.95);
    
    // Stars - bright simple points
    float stars = starField(dir) * night_factor;
    color = mix(color, vec3(1.0, 0.98, 0.9), stars);
    
    fragColor = vec4(color, 1.0);
}
"""


# =============================================================================
# RENDERER
# =============================================================================

class ModernSkyRenderer:
    """
    GPU-accelerated procedural sky renderer.
    
    Renders a fullscreen quad with ray-marched atmospheric effects.
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Compile shaders
        self.program = ctx.program(
            vertex_shader=SKY_VERTEX_SHADER,
            fragment_shader=SKY_FRAGMENT_SHADER,
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
        self.time_of_day = 0.5  # Noon
        self.sun_dir = glm.vec3(0.5, 1.0, 0.3)
        self.camera_pos = glm.vec3(0, 0, 0)
        
        # Camera matrices
        self.projection = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
        
        # Stats
        self.frame_stats = {
            'draw_calls': 1,
        }
    
    def set_camera(self, projection: glm.mat4, view: glm.mat4, camera_pos: glm.vec3):
        """Set camera matrices."""
        self.projection = projection
        self.view = view
        self.camera_pos = camera_pos
    
    def set_time(self, time_of_day: float):
        """
        Set time of day (0.0 = midnight, 0.5 = noon, 1.0 = midnight).
        Also calculates sun direction based on time.
        """
        self.time_of_day = time_of_day
        
        # Calculate sun position (rises in east at 0.25, sets in west at 0.75)
        sun_angle = (time_of_day - 0.25) * math.pi * 2
        self.sun_dir = glm.vec3(
            math.cos(sun_angle) * 0.5,
            math.sin(sun_angle),
            0.3
        )
    
    def get_sky_color(self) -> Tuple[float, float, float]:
        """Get current sky color for fog matching."""
        # Simplified version of shader logic
        time = self.time_of_day
        
        night = max(0, min(1, (time - 0.75) / 0.25) if time > 0.75 else (0.25 - time) / 0.25 if time < 0.25 else 0)
        day = max(0, min(1, 1 - abs(time - 0.5) * 4))
        
        if day > 0.5:
            return (0.5, 0.7, 0.9)  # Day blue
        elif night > 0.5:
            return (0.02, 0.02, 0.08)  # Night dark
        else:
            return (0.7, 0.5, 0.4)  # Sunrise/sunset
    
    def render(self):
        """Render the sky."""
        # Disable depth write so sky is always behind
        self.ctx.depth_func = '<='
        
        # Set uniforms
        self.program['u_projection'].write(self.projection)
        self.program['u_view'].write(self.view)
        self.program['u_time_of_day'].value = self.time_of_day
        self.program['u_sun_dir'].write(self.sun_dir)
        # Note: u_camera_pos is unused in shader (optimized out), skip it
        
        # Render fullscreen quad as triangle strip
        self.vao.render(moderngl.TRIANGLE_STRIP)
        
        # Restore depth func
        self.ctx.depth_func = '<'
    
    def cleanup(self):
        """Release GPU resources."""
        self.vbo.release()
        self.vao.release()
        self.program.release()

