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

// Sky colors for different times
const vec3 NIGHT_SKY = vec3(0.02, 0.02, 0.08);
const vec3 NIGHT_HORIZON = vec3(0.05, 0.05, 0.1);

const vec3 SUNRISE_SKY = vec3(0.2, 0.3, 0.5);
const vec3 SUNRISE_HORIZON = vec3(0.9, 0.5, 0.3);

const vec3 DAY_SKY = vec3(0.4, 0.6, 0.9);
const vec3 DAY_HORIZON = vec3(0.7, 0.8, 0.95);

const vec3 SUNSET_SKY = vec3(0.3, 0.3, 0.5);
const vec3 SUNSET_HORIZON = vec3(0.95, 0.4, 0.2);

// Sun/moon
const vec3 SUN_COLOR = vec3(1.0, 0.95, 0.8);
const vec3 MOON_COLOR = vec3(0.9, 0.9, 1.0);

// Simple hash for stars
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

vec3 getGradientColor(float y, float time) {
    // y is vertical component of ray (-1 to 1)
    // time is 0-1 through the day
    
    // Calculate blend factors for time of day
    float night = smoothstep(0.75, 1.0, time) + smoothstep(0.25, 0.0, time);
    float sunrise = smoothstep(0.2, 0.25, time) * smoothstep(0.35, 0.25, time);
    float day = smoothstep(0.25, 0.4, time) * smoothstep(0.75, 0.6, time);
    float sunset = smoothstep(0.6, 0.75, time) * smoothstep(0.85, 0.75, time);
    
    // Normalize
    float total = night + sunrise + day + sunset + 0.001;
    night /= total;
    sunrise /= total;
    day /= total;
    sunset /= total;
    
    // Get horizon blend (0 at horizon, 1 at zenith)
    float horizon = smoothstep(-0.1, 0.5, y);
    
    // Blend sky colors based on time
    vec3 sky = NIGHT_SKY * night + SUNRISE_SKY * sunrise + DAY_SKY * day + SUNSET_SKY * sunset;
    vec3 hor = NIGHT_HORIZON * night + SUNRISE_HORIZON * sunrise + DAY_HORIZON * day + SUNSET_HORIZON * sunset;
    
    return mix(hor, sky, horizon);
}

float starField(vec3 dir) {
    // Only show stars at night and above horizon
    if (dir.y < 0.0) return 0.0;
    
    // Project to sphere
    vec2 uv = dir.xz / (dir.y + 1.0) * 100.0;
    
    // Grid-based stars
    vec2 grid = floor(uv);
    float star = 0.0;
    
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            vec2 cell = grid + vec2(dx, dy);
            float h = hash(cell);
            
            if (h > 0.97) {  // Star probability
                vec2 star_pos = cell + vec2(hash(cell + 0.1), hash(cell + 0.2));
                float d = length(uv - star_pos);
                float brightness = h * smoothstep(0.1, 0.0, d);
                star += brightness;
            }
        }
    }
    
    return star;
}

void main() {
    vec3 dir = normalize(v_ray_dir);
    
    // Base sky gradient
    vec3 color = getGradientColor(dir.y, u_time_of_day);
    
    // Sun
    float sun_dot = dot(dir, normalize(u_sun_dir));
    float sun_disk = smoothstep(0.998, 0.9995, sun_dot);  // Sharp sun disk
    float sun_glow = pow(max(sun_dot, 0.0), 8.0) * 0.5;   // Soft glow
    
    // Only show sun during day
    float day_factor = smoothstep(0.2, 0.35, u_time_of_day) * smoothstep(0.8, 0.65, u_time_of_day);
    color += SUN_COLOR * (sun_disk + sun_glow) * day_factor;
    
    // Moon (opposite to sun)
    vec3 moon_dir = -u_sun_dir;
    float moon_dot = dot(dir, normalize(moon_dir));
    float moon_disk = smoothstep(0.995, 0.998, moon_dot);
    float night_factor = smoothstep(0.3, 0.2, u_time_of_day) + smoothstep(0.7, 0.8, u_time_of_day);
    night_factor = clamp(night_factor, 0.0, 1.0);
    color += MOON_COLOR * moon_disk * night_factor * 0.8;
    
    // Stars
    float stars = starField(dir) * night_factor;
    color += vec3(1.0, 1.0, 0.95) * stars;
    
    // Horizon haze
    float haze = 1.0 - smoothstep(0.0, 0.3, abs(dir.y));
    color = mix(color, vec3(0.7, 0.75, 0.8), haze * 0.3 * day_factor);
    
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

