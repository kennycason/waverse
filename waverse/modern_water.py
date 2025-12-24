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

# Simple flat water - no waves, very fast
FLAT_WATER_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;

out vec3 v_world_pos;
out vec2 v_uv;
out float v_depth;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform float u_water_level;
uniform vec3 u_camera_pos;

void main() {
    vec3 pos = in_position;
    pos.y = u_water_level;  // Flat water at water level
    
    v_world_pos = pos;
    v_uv = pos.xz * 0.01;
    
    // Depth (distance from camera)
    float dist = length(pos.xz - u_camera_pos.xz);
    v_depth = clamp(dist / 300.0, 0.0, 1.0);
    
    gl_Position = u_projection * u_view * vec4(pos, 1.0);
}
"""

FLAT_WATER_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_world_pos;
in vec2 v_uv;
in float v_depth;

out vec4 fragColor;

uniform vec3 u_water_shallow;
uniform vec3 u_water_deep;
uniform vec3 u_sky_color;
uniform vec3 u_camera_pos;

void main() {
    // Simple gradient from shallow to deep
    vec3 water_color = mix(u_water_shallow, u_water_deep, v_depth);
    
    // Subtle sky reflection at distance
    water_color = mix(water_color, u_sky_color * 0.8, v_depth * 0.2);
    
    // Solid alpha for clean look
    fragColor = vec4(water_color, 0.9);
}
"""

# Animated wave water - more complex
WATER_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;

out vec3 v_world_pos;
out vec2 v_uv;
out float v_depth;
out float v_foam;
out float v_wave_height;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform float u_time;
uniform float u_water_level;
uniform float u_tide_level;
uniform vec3 u_camera_pos;

// Wind uniforms
uniform vec2 u_wind_dir;
uniform float u_wind_strength;

// Wave parameters - traveling waves with wind direction
const float BASE_WAVE_HEIGHT = 0.8;   // Wave height
const float WAVE_SPEED = 0.4;         // Visible wave travel speed

// Stepped/quantized function for chunky waves
float step_wave(float x, float steps) {
    return floor(x * steps + 0.5) / steps;
}

// Gerstner-style traveling wave (actually propagates!)
float traveling_wave(vec2 pos, float time, vec2 dir, float freq, float speed, float amp) {
    // Wave travels in direction 'dir'
    float phase = dot(pos, dir) * freq - time * speed;
    return sin(phase) * amp;
}

float chunky_wave(vec2 pos, float time, vec2 wind_dir, float wind_str) {
    // Use RELATIVE position (mod to avoid huge numbers at far coords)
    vec2 local_pos = mod(pos, 500.0);
    
    // === TRAVELING WAVES (move in wind direction!) ===
    // Primary swell - large waves traveling with wind
    float swell = traveling_wave(local_pos, time, wind_dir, 0.015, WAVE_SPEED, 1.0);
    swell = step_wave(swell, 4.0);  // Chunky
    
    // Secondary swell - slightly offset angle
    vec2 cross_dir = vec2(wind_dir.y * 0.7 - wind_dir.x * 0.3, -wind_dir.x * 0.7 - wind_dir.y * 0.3);
    float cross = traveling_wave(local_pos, time, cross_dir, 0.022, WAVE_SPEED * 0.7, 0.5);
    cross = step_wave(cross, 5.0);
    
    // Interference pattern - creates ripple intersection effects
    float interference = traveling_wave(local_pos, time, -wind_dir, 0.03, WAVE_SPEED * 0.5, 0.3);
    
    // Choppy detail waves - smaller, faster
    float chop1 = traveling_wave(local_pos, time, wind_dir, 0.06, WAVE_SPEED * 1.5, 0.15);
    float chop2 = traveling_wave(local_pos, time, cross_dir, 0.08, WAVE_SPEED * 1.2, 0.1);
    
    // Random-looking ripples (circular interference simulation)
    float ripple1 = sin(length(local_pos - vec2(100, 100)) * 0.1 - time * 0.3) * 0.08;
    float ripple2 = sin(length(local_pos - vec2(250, 180)) * 0.12 - time * 0.25) * 0.06;
    
    // Combine - wind strength affects height
    float height_mult = BASE_WAVE_HEIGHT * (0.4 + wind_str * 0.6);
    float total = swell * height_mult + cross * height_mult * 0.4;
    total += interference * height_mult * 0.3;
    total += (chop1 + chop2) * (0.5 + wind_str * 0.5);
    total += (ripple1 + ripple2) * (0.3 + wind_str * 0.3);
    
    return total;
}

void main() {
    vec3 pos = in_position;
    
    // Normalize wind direction (default to +X if no wind)
    vec2 wind_dir = length(u_wind_dir) > 0.01 ? normalize(u_wind_dir) : vec2(1.0, 0.0);
    float wind_str = clamp(u_wind_strength, 0.0, 2.0);
    
    // Calculate wave height
    float wave_h = chunky_wave(pos.xz, u_time, wind_dir, wind_str);
    
    // Apply water level + gentle tide + wave
    // Tide is scaled down for subtlety
    pos.y = u_water_level + u_tide_level * 0.3 + wave_h;
    
    v_world_pos = pos;
    v_wave_height = wave_h;
    v_uv = pos.xz * 0.01;
    
    // Depth (distance from camera)
    float dist = length(pos.xz - u_camera_pos.xz);
    v_depth = clamp(dist / 200.0, 0.0, 1.0);
    
    // Foam on wave peaks
    float peak_factor = clamp((wave_h / BASE_WAVE_HEIGHT + 0.5), 0.0, 1.0);
    v_foam = peak_factor * peak_factor * (0.3 + wind_str * 0.4);
    
    gl_Position = u_projection * u_view * vec4(pos, 1.0);
}
"""

WATER_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_world_pos;
in vec2 v_uv;
in float v_depth;
in float v_foam;
in float v_wave_height;

out vec4 fragColor;

uniform float u_time;
uniform vec3 u_sky_color;
uniform vec3 u_camera_pos;
uniform vec3 u_light_dir;
uniform float u_wind_strength;

// Wind Waker style - bright, cheerful cartoon water!
uniform vec3 u_water_shallow;   // Bright turquoise at surface
uniform vec3 u_water_deep;      // Deeper blue below
uniform vec3 u_foam_color;      // Foam/crest color

const vec3 WAVE_CREST_BOOST = vec3(0.15, 0.18, 0.12);  // Lighter at peaks

// Fresnel effect - more reflection at glancing angles
float fresnel(vec3 normal, vec3 view_dir) {
    float f0 = 0.02;
    return f0 + (1.0 - f0) * pow(1.0 - max(dot(normal, view_dir), 0.0), 5.0);
}

void main() {
    // Calculate normal from wave height (chunky faceted look)
    // This gives hard edges on the stepped waves
    vec3 normal = normalize(vec3(
        v_wave_height * 0.3,  // Wave tilt
        1.0,
        v_wave_height * 0.2
    ));
    
    // View direction
    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
    
    // Wave height factor for color variation
    float wave_factor = clamp(v_wave_height * 0.5 + 0.5, 0.0, 1.0);
    
    // Base water color - blend shallow to deep based on depth
    vec3 water_color = mix(u_water_shallow, u_water_deep, v_depth);
    
    // Stylized: lighter color at wave crests
    water_color += WAVE_CREST_BOOST * wave_factor;
    
    // Simple flat lighting (more stylized)
    float diff = max(dot(normal, normalize(u_light_dir)), 0.0);
    // Quantize lighting for cel-shaded look
    diff = floor(diff * 3.0 + 0.5) / 3.0;
    water_color *= (0.65 + 0.35 * diff);
    
    // Fresnel reflection of sky (subtle)
    float fres = fresnel(normal, view_dir);
    water_color = mix(water_color, u_sky_color, fres * 0.3);
    
    // Specular highlight - larger, softer for stylized look
    vec3 half_vec = normalize(normalize(u_light_dir) + view_dir);
    float spec = pow(max(dot(normal, half_vec), 0.0), 16.0);
    water_color += vec3(1.0, 0.98, 0.9) * spec * 0.4;
    
    // Add foam at wave peaks
    water_color = mix(water_color, u_foam_color, v_foam * 0.5);
    
    // Wind-based shimmer (more active in windy conditions)
    float shimmer = sin(v_uv.x * 30.0 + u_time * 3.0) * sin(v_uv.y * 25.0 + u_time * 2.0);
    shimmer = shimmer * 0.5 + 0.5;
    water_color += vec3(0.08, 0.12, 0.15) * shimmer * u_wind_strength * 0.5;
    
    // Alpha - solid for stylized look, slight variation with waves
    float alpha = 0.88 + wave_factor * 0.07;
    
    fragColor = vec4(water_color, alpha);
}
"""


# =============================================================================
# RENDERER
# =============================================================================

class ModernWaterRenderer:
    """
    GPU-accelerated water surface renderer.
    
    Creates a large water plane that covers the world at a given water level.
    Supports two modes:
    - Flat (default): Simple flat water, very fast
    - Waves (--enable-waves): Animated chunky waves
    """
    
    def __init__(self, ctx: moderngl.Context, enable_waves: bool = False):
        self.ctx = ctx
        self.enable_waves = enable_waves
        
        # Compile shaders based on mode
        if enable_waves:
            self.program = ctx.program(
                vertex_shader=WATER_VERTEX_SHADER,
                fragment_shader=WATER_FRAGMENT_SHADER,
            )
        else:
            # Simple flat water - much faster
            self.program = ctx.program(
                vertex_shader=FLAT_WATER_VERTEX_SHADER,
                fragment_shader=FLAT_WATER_FRAGMENT_SHADER,
            )
        
        # Create water plane mesh
        # Use lower resolution for flat water since we don't need detail
        resolution = 60 if enable_waves else 30
        self.vbo, self.vertex_count = self._create_water_mesh(resolution=resolution)
        
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, '3f', 'in_position')],
        )
        
        # Uniforms
        self.water_level = 20.0
        self.time = 0.0
        self.sky_color = glm.vec3(0.6, 0.7, 0.9)
        self.light_dir = glm.vec3(0.5, 1.0, 0.3)
        
        # Wind Waker style water colors - bright cheerful turquoise!
        self.water_shallow = glm.vec3(0.2, 0.75, 0.85)   # Bright turquoise
        self.water_deep = glm.vec3(0.1, 0.45, 0.7)       # Deeper sky blue
        self.foam_color = glm.vec3(0.95, 0.98, 1.0)      # White foam
        
        # Wind/tide parameters
        self.wind_dir = (1.0, 0.0)
        self.wind_strength = 0.3
        self.tide_level = 0.0
        
        # Camera matrices (set by parent)
        self.projection = glm.mat4(1.0)
        self.view = glm.mat4(1.0)
        self.camera_pos = glm.vec3(0, 50, 0)
        
        # Frame stats
        self.frame_stats = {
            'triangles': 0,
            'draw_calls': 0,
        }
    
    def _create_water_mesh(self, size: float = 4000.0, resolution: int = 80) -> Tuple[moderngl.Buffer, int]:
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
        
        # Store mesh info for dynamic repositioning
        self.mesh_size = size
        self.mesh_resolution = resolution
        self.mesh_center_x = 0.0
        self.mesh_center_z = 0.0
        self.needs_initial_position = True  # Flag to force first update
        
        return vbo, vertex_count
    
    def set_camera(self, projection: glm.mat4, view: glm.mat4, camera_pos: glm.vec3):
        """Set camera matrices."""
        self.projection = projection
        self.view = view
        self.camera_pos = camera_pos
    
    def update_position(self, camera_x: float, camera_z: float):
        """
        Update water plane position to follow camera.
        
        Regenerates the mesh centered on the camera position
        to ensure water is always visible nearby.
        """
        # Force rebuild on first frame
        if self.needs_initial_position:
            self._rebuild_mesh_at(camera_x, camera_z)
            self.needs_initial_position = False
            return
        
        # Check if camera has moved far enough to need new mesh
        dx = camera_x - self.mesh_center_x
        dz = camera_z - self.mesh_center_z
        dist_sq = dx * dx + dz * dz
        
        # Rebuild mesh when camera moves more than 1/4 of the mesh size
        threshold = self.mesh_size * 0.25
        if dist_sq > threshold * threshold:
            self._rebuild_mesh_at(camera_x, camera_z)
    
    def _rebuild_mesh_at(self, center_x: float, center_z: float):
        """Rebuild the water mesh centered at new position."""
        vertices = []
        half = self.mesh_size / 2
        step = self.mesh_size / self.mesh_resolution
        
        for i in range(self.mesh_resolution):
            for j in range(self.mesh_resolution):
                x0 = center_x - half + i * step
                z0 = center_z - half + j * step
                x1 = x0 + step
                z1 = z0 + step
                
                # Two triangles per grid cell
                vertices.extend([x0, 0, z0])
                vertices.extend([x1, 0, z0])
                vertices.extend([x0, 0, z1])
                
                vertices.extend([x1, 0, z0])
                vertices.extend([x1, 0, z1])
                vertices.extend([x0, 0, z1])
        
        data = np.array(vertices, dtype='f4')
        
        # Orphan the old buffer and write new data
        self.vbo.orphan(data.nbytes)
        self.vbo.write(data.tobytes())
        
        self.mesh_center_x = center_x
        self.mesh_center_z = center_z
    
    def set_wind(self, wind_dir: tuple, wind_strength: float, tide_level: float = 0.0):
        """Set wind parameters for wave animation."""
        self.wind_dir = wind_dir
        self.wind_strength = wind_strength
        self.tide_level = tide_level
    
    def set_water_colors(self, shallow: tuple = None, deep: tuple = None, foam: tuple = None):
        """
        Set water colors for different environments.
        
        Presets:
        - Tropical: bright turquoise (0.2, 0.75, 0.85), deep blue (0.1, 0.45, 0.7)
        - Cold: gray-blue (0.3, 0.5, 0.6), dark blue (0.15, 0.3, 0.5)
        - Swamp: murky green (0.25, 0.5, 0.35), dark (0.1, 0.25, 0.2)
        - Volcanic: orange-red (0.7, 0.35, 0.2), dark red (0.4, 0.15, 0.1)
        """
        if shallow:
            self.water_shallow = glm.vec3(*shallow)
        if deep:
            self.water_deep = glm.vec3(*deep)
        if foam:
            self.foam_color = glm.vec3(*foam)
    
    def render(self, dt: float = 0.016):
        """Render the water surface."""
        # Position water on first frame (before any rendering)
        if self.needs_initial_position:
            self._rebuild_mesh_at(self.camera_pos.x, self.camera_pos.z)
            self.needs_initial_position = False
            self.time = 1000.0  # Start at stable time
        
        # Update water position to follow camera
        self.update_position(self.camera_pos.x, self.camera_pos.z)
        
        # Set common uniforms
        self.program['u_projection'].write(self.projection)
        self.program['u_view'].write(self.view)
        self.program['u_water_level'].value = self.water_level
        self.program['u_camera_pos'].write(self.camera_pos)
        
        # Water color uniforms (both modes)
        self.program['u_water_shallow'].write(self.water_shallow)
        self.program['u_water_deep'].write(self.water_deep)
        self.program['u_sky_color'].write(self.sky_color)
        
        # Wave-only uniforms
        if self.enable_waves:
            self.time += dt * 0.5  # Slow time for calmer waves
            self.program['u_time'].value = self.time
            self.program['u_tide_level'].value = self.tide_level
            self.program['u_light_dir'].write(self.light_dir)
            self.program['u_wind_dir'].value = self.wind_dir
            self.program['u_wind_strength'].value = self.wind_strength
            self.program['u_foam_color'].write(self.foam_color)
        
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

