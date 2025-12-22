"""
ModernGL Water Renderer

GPU-accelerated water rendering with:
- Animated wave displacement
- Simple reflections (sky color)
- Depth-based transparency
- Foam at shorelines
- Caustics effect
"""

import numpy as np
import moderngl
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

try:
    from pyglm import glm
except ImportError:
    import glm


# =============================================================================
# SHADERS
# =============================================================================

WATER_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;

out vec3 v_world_pos;
out vec2 v_uv;
out float v_depth;
out float v_foam;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform float u_time;
uniform float u_water_level;
uniform vec3 u_camera_pos;

// Wave parameters
const float WAVE_SPEED = 0.3;
const float WAVE_HEIGHT = 0.5;
const float WAVE_LENGTH = 20.0;

float wave(vec2 pos, float time) {
    // Multiple overlapping waves for more natural look
    float w1 = sin(pos.x / WAVE_LENGTH + time * WAVE_SPEED) * WAVE_HEIGHT;
    float w2 = sin(pos.y / (WAVE_LENGTH * 0.7) + time * WAVE_SPEED * 1.3) * WAVE_HEIGHT * 0.5;
    float w3 = sin((pos.x + pos.y) / (WAVE_LENGTH * 1.5) + time * WAVE_SPEED * 0.7) * WAVE_HEIGHT * 0.3;
    return w1 + w2 + w3;
}

void main() {
    vec3 pos = in_position;
    
    // Apply wave displacement
    pos.y = u_water_level + wave(pos.xz, u_time);
    
    v_world_pos = pos;
    v_uv = pos.xz * 0.01;  // UV for texture/effects
    
    // Calculate depth (distance from shore - approximated by distance from camera)
    float dist = length(pos.xz - u_camera_pos.xz);
    v_depth = clamp(dist / 200.0, 0.0, 1.0);
    
    // Foam factor (higher near shore / where waves peak)
    float wave_peak = abs(wave(pos.xz, u_time)) / WAVE_HEIGHT;
    v_foam = wave_peak * 0.5;
    
    gl_Position = u_projection * u_view * vec4(pos, 1.0);
}
"""

WATER_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_world_pos;
in vec2 v_uv;
in float v_depth;
in float v_foam;

out vec4 fragColor;

uniform float u_time;
uniform vec3 u_sky_color;
uniform vec3 u_camera_pos;
uniform vec3 u_light_dir;

// Water colors
const vec3 WATER_SHALLOW = vec3(0.2, 0.5, 0.6);
const vec3 WATER_DEEP = vec3(0.05, 0.2, 0.4);
const vec3 FOAM_COLOR = vec3(0.9, 0.95, 1.0);

// Fresnel effect - more reflection at glancing angles
float fresnel(vec3 normal, vec3 view_dir) {
    float f0 = 0.02;
    return f0 + (1.0 - f0) * pow(1.0 - max(dot(normal, view_dir), 0.0), 5.0);
}

void main() {
    // Calculate normal from wave derivatives (approximated)
    vec3 normal = normalize(vec3(
        sin(v_uv.x * 100.0 + u_time) * 0.1,
        1.0,
        cos(v_uv.y * 100.0 + u_time * 0.7) * 0.1
    ));
    
    // View direction
    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
    
    // Base water color - blend shallow to deep based on depth
    vec3 water_color = mix(WATER_SHALLOW, WATER_DEEP, v_depth);
    
    // Simple lighting
    float diff = max(dot(normal, normalize(u_light_dir)), 0.0);
    water_color *= (0.6 + 0.4 * diff);
    
    // Fresnel reflection of sky
    float fres = fresnel(normal, view_dir);
    water_color = mix(water_color, u_sky_color, fres * 0.4);
    
    // Specular highlight
    vec3 half_vec = normalize(normalize(u_light_dir) + view_dir);
    float spec = pow(max(dot(normal, half_vec), 0.0), 64.0);
    water_color += vec3(1.0, 0.95, 0.8) * spec * 0.5;
    
    // Add foam
    water_color = mix(water_color, FOAM_COLOR, v_foam * 0.3);
    
    // Caustics effect (subtle rippling light patterns)
    float caustics = sin(v_uv.x * 50.0 + u_time * 2.0) * sin(v_uv.y * 50.0 + u_time * 1.5);
    caustics = caustics * 0.5 + 0.5;  // Normalize to 0-1
    water_color += vec3(0.1, 0.15, 0.2) * caustics * (1.0 - v_depth) * 0.2;
    
    // Alpha - more opaque when deeper, slightly transparent at edges
    float alpha = mix(0.7, 0.95, v_depth);
    
    fragColor = vec4(water_color, alpha);
}
"""


# =============================================================================
# RENDERER
# =============================================================================

class ModernWaterRenderer:
    """
    GPU-accelerated water surface renderer.
    
    Creates a large water plane that covers the world at a given water level,
    with animated waves and visual effects.
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Compile shaders
        self.program = ctx.program(
            vertex_shader=WATER_VERTEX_SHADER,
            fragment_shader=WATER_FRAGMENT_SHADER,
        )
        
        # Create water plane mesh (large quad grid for wave detail)
        self.vbo, self.vertex_count = self._create_water_mesh()
        
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, '3f', 'in_position')],
        )
        
        # Uniforms
        self.water_level = 20.0
        self.time = 0.0
        self.sky_color = glm.vec3(0.6, 0.7, 0.9)
        self.light_dir = glm.vec3(0.5, 1.0, 0.3)
        
        # Camera matrices (set by parent)
        self.projection = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
        self.camera_pos = glm.vec3(0, 50, 0)
        
        # Frame stats
        self.frame_stats = {
            'triangles': 0,
            'draw_calls': 0,
        }
    
    def _create_water_mesh(self, size: float = 2000.0, resolution: int = 64) -> Tuple[moderngl.Buffer, int]:
        """
        Create a water plane mesh.
        
        Args:
            size: Total size of water plane (centered on origin)
            resolution: Grid resolution (vertices per side)
        """
        vertices = []
        half = size / 2
        step = size / resolution
        
        for i in range(resolution):
            for j in range(resolution):
                x0 = -half + i * step
                z0 = -half + j * step
                x1 = x0 + step
                z1 = z0 + step
                
                # Two triangles per grid cell
                # Triangle 1
                vertices.extend([x0, 0, z0])
                vertices.extend([x1, 0, z0])
                vertices.extend([x0, 0, z1])
                
                # Triangle 2
                vertices.extend([x1, 0, z0])
                vertices.extend([x1, 0, z1])
                vertices.extend([x0, 0, z1])
        
        data = np.array(vertices, dtype='f4')
        vbo = self.ctx.buffer(data.tobytes())
        vertex_count = len(vertices) // 3
        
        return vbo, vertex_count
    
    def set_camera(self, projection: glm.mat4, view: glm.mat4, camera_pos: glm.vec3):
        """Set camera matrices."""
        self.projection = projection
        self.view = view
        self.camera_pos = camera_pos
    
    def update_position(self, camera_x: float, camera_z: float):
        """
        Update water plane position to follow camera.
        
        We regenerate the mesh centered on the camera position
        to ensure water is always visible nearby.
        """
        # For now, we use a static large plane
        # Could optimize by creating a dynamic mesh that follows camera
        pass
    
    def render(self, dt: float = 0.016):
        """Render the water surface."""
        self.time += dt
        
        # Set uniforms
        self.program['u_projection'].write(self.projection)
        self.program['u_view'].write(self.view)
        self.program['u_time'].value = self.time
        self.program['u_water_level'].value = self.water_level
        self.program['u_camera_pos'].write(self.camera_pos)
        self.program['u_sky_color'].write(self.sky_color)
        self.program['u_light_dir'].write(self.light_dir)
        
        # Enable blending for water transparency
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        
        # Render
        self.vao.render(moderngl.TRIANGLES)
        
        # Disable blending
        self.ctx.disable(moderngl.BLEND)
        
        # Update stats
        self.frame_stats['triangles'] = self.vertex_count // 3
        self.frame_stats['draw_calls'] = 1
    
    def cleanup(self):
        """Release GPU resources."""
        self.vbo.release()
        self.vao.release()
        self.program.release()

