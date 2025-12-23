"""
ModernGL HUD Renderer

GPU-accelerated HUD elements:
- Crosshair
- Coordinate display (bitmap font)
- Minimap
- Performance stats
"""

import numpy as np
import moderngl
from typing import Dict, Tuple, Optional, List
import math

try:
    from pyglm import glm
except ImportError:
    import glm


# =============================================================================
# SHADERS
# =============================================================================

HUD_VERTEX_SHADER = """
#version 330 core

in vec2 in_position;
in vec2 in_uv;
in vec4 in_color;

out vec2 v_uv;
out vec4 v_color;

uniform vec2 u_screen_size;
uniform vec2 u_offset;

void main() {
    // Convert to normalized device coordinates (-1 to 1)
    vec2 pos = (in_position + u_offset) / u_screen_size * 2.0 - 1.0;
    pos.y = -pos.y;  // Flip Y for screen coords
    
    gl_Position = vec4(pos, 0.0, 1.0);
    v_uv = in_uv;
    v_color = in_color;
}
"""

HUD_FRAGMENT_SHADER = """
#version 330 core

in vec2 v_uv;
in vec4 v_color;

out vec4 fragColor;

uniform sampler2D u_texture;
uniform int u_use_texture;

void main() {
    if (u_use_texture == 1) {
        float alpha = texture(u_texture, v_uv).r;  // Mono font texture
        fragColor = vec4(v_color.rgb, v_color.a * alpha);
    } else {
        fragColor = v_color;
    }
}
"""


# Simple 8x8 bitmap font (subset of ASCII printable chars)
# Each char is 8 rows of 8 bits
BITMAP_FONT = {
    ' ': [0x00]*8,
    '0': [0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00],
    '1': [0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00],
    '2': [0x3C, 0x66, 0x06, 0x0C, 0x18, 0x30, 0x7E, 0x00],
    '3': [0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00],
    '4': [0x0C, 0x1C, 0x3C, 0x6C, 0x7E, 0x0C, 0x0C, 0x00],
    '5': [0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00],
    '6': [0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00],
    '7': [0x7E, 0x06, 0x0C, 0x18, 0x18, 0x18, 0x18, 0x00],
    '8': [0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00],
    '9': [0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00],
    '-': [0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00],
    '.': [0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00],
    ',': [0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x30, 0x00],
    ':': [0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00],
    'X': [0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00],
    'Y': [0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00],
    'Z': [0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00],
    'F': [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00],
    'P': [0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00],
    'S': [0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00],
    'T': [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00],
    'R': [0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x00],
    'I': [0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00],
    '%': [0x62, 0x64, 0x08, 0x10, 0x26, 0x46, 0x00, 0x00],
    '/': [0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x00, 0x00],
    '(': [0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00],
    ')': [0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00],
}

# Add uppercase letters
for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    if letter not in BITMAP_FONT:
        # Simple placeholder - vertical lines
        BITMAP_FONT[letter] = [0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00]


class ModernHUDRenderer:
    """
    GPU-accelerated HUD rendering for Core profile OpenGL.
    """
    
    def __init__(self, ctx: moderngl.Context, screen_width: int = 1920, screen_height: int = 1080):
        self.ctx = ctx
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Compile shaders
        self.program = ctx.program(
            vertex_shader=HUD_VERTEX_SHADER,
            fragment_shader=HUD_FRAGMENT_SHADER,
        )
        
        # Create font texture atlas
        self._create_font_texture()
        
        # Create dynamic vertex buffer for HUD elements
        # Format: x, y, u, v, r, g, b, a
        self.max_quads = 1000
        self.vertex_buffer = np.zeros(self.max_quads * 6 * 8, dtype='f4')
        self.vbo = ctx.buffer(reserve=self.vertex_buffer.nbytes, dynamic=True)
        
        # VAO
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, '2f 2f 4f', 'in_position', 'in_uv', 'in_color')],
        )
        
        # Current HUD state
        self.crosshair_enabled = True
        self.coords_enabled = True
        self.minimap_enabled = True
        self.stats_enabled = False
        
        # Frame stats
        self.frame_stats = {}
        
    def _create_font_texture(self):
        """Create a texture atlas from the bitmap font."""
        # Create a 128x128 texture (16x16 chars, each 8x8)
        texture_size = 128
        char_size = 8
        chars_per_row = 16
        
        texture_data = np.zeros((texture_size, texture_size), dtype=np.uint8)
        
        # Map printable ASCII to positions
        self.char_uvs = {}
        for i, char in enumerate(' 0123456789-.,:%/()XYZFPSTRI'):
            if char in BITMAP_FONT:
                row = i // chars_per_row
                col = i % chars_per_row
                
                # Write bitmap to texture
                for y, byte in enumerate(BITMAP_FONT[char]):
                    for x in range(8):
                        if byte & (0x80 >> x):
                            ty = row * char_size + y
                            tx = col * char_size + x
                            texture_data[ty, tx] = 255
                
                # Store UV coordinates
                u0 = col * char_size / texture_size
                v0 = row * char_size / texture_size
                u1 = (col + 1) * char_size / texture_size
                v1 = (row + 1) * char_size / texture_size
                self.char_uvs[char] = (u0, v0, u1, v1)
        
        # Create texture
        self.font_texture = self.ctx.texture(
            (texture_size, texture_size), 1, 
            texture_data.tobytes()
        )
        self.font_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
    
    def resize(self, width: int, height: int):
        """Update screen size."""
        self.screen_width = width
        self.screen_height = height
    
    def _add_quad(self, vertices: List, x: float, y: float, w: float, h: float,
                  u0: float = 0, v0: float = 0, u1: float = 1, v1: float = 1,
                  r: float = 1, g: float = 1, b: float = 1, a: float = 1):
        """Add a textured quad to the vertex list."""
        # Two triangles for a quad
        # Triangle 1
        vertices.extend([x, y, u0, v0, r, g, b, a])
        vertices.extend([x+w, y, u1, v0, r, g, b, a])
        vertices.extend([x, y+h, u0, v1, r, g, b, a])
        # Triangle 2
        vertices.extend([x+w, y, u1, v0, r, g, b, a])
        vertices.extend([x+w, y+h, u1, v1, r, g, b, a])
        vertices.extend([x, y+h, u0, v1, r, g, b, a])
    
    def _draw_text(self, vertices: List, text: str, x: float, y: float, 
                   scale: float = 2.0, r: float = 1, g: float = 1, b: float = 1, a: float = 1):
        """Add text characters to vertex list."""
        char_width = 8 * scale
        char_height = 8 * scale
        
        for i, char in enumerate(text.upper()):
            if char in self.char_uvs:
                u0, v0, u1, v1 = self.char_uvs[char]
                self._add_quad(vertices, 
                              x + i * char_width, y, 
                              char_width, char_height,
                              u0, v0, u1, v1,
                              r, g, b, a)
    
    def render(self, camera_x: float = 0, camera_y: float = 0, camera_z: float = 0,
               fps: float = 60, time_of_day: float = 0.5,
               chunk_x: int = 0, chunk_z: int = 0,
               terrain_chunks: int = 0, flora_instances: int = 0,
               biome: str = ""):
        """Render HUD elements."""
        vertices = []
        
        # Crosshair
        if self.crosshair_enabled:
            cx = self.screen_width / 2
            cy = self.screen_height / 2
            size = 12
            thickness = 2
            
            # Horizontal line
            self._add_quad(vertices, cx - size, cy - thickness/2, 
                          size*2, thickness, 0, 0, 0, 0, 1, 1, 1, 0.8)
            # Vertical line
            self._add_quad(vertices, cx - thickness/2, cy - size, 
                          thickness, size*2, 0, 0, 0, 0, 1, 1, 1, 0.8)
        
        # Coordinates - draw in top-left
        if self.coords_enabled:
            text = f"X:{int(camera_x):,} Y:{int(camera_y):,} Z:{int(camera_z):,}"
            # Simplified: just show numbers
            coord_text = f"{int(camera_x)} {int(camera_y)} {int(camera_z)}"
            self._draw_text(vertices, coord_text, 10, 10, 2, 1, 1, 1, 0.9)
            
            # FPS
            fps_text = f"FPS:{int(fps)}"
            self._draw_text(vertices, fps_text, 10, 30, 2, 0.7, 1, 0.7, 0.9)
        
        # Upload and render
        if vertices:
            vertex_data = np.array(vertices, dtype='f4')
            self.vbo.write(vertex_data.tobytes())
            
            # Set uniforms
            self.program['u_screen_size'].value = (self.screen_width, self.screen_height)
            self.program['u_offset'].value = (0, 0)
            self.program['u_use_texture'].value = 1
            
            # Bind font texture
            self.font_texture.use(0)
            
            # Enable blending
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            
            # Disable depth test for HUD
            self.ctx.disable(moderngl.DEPTH_TEST)
            
            # Render
            self.vao.render(moderngl.TRIANGLES, vertices=len(vertices) // 8)
            
            # Re-enable depth test
            self.ctx.enable(moderngl.DEPTH_TEST)
    
    def cleanup(self):
        """Release GPU resources."""
        self.vbo.release()
        self.vao.release()
        self.program.release()
        self.font_texture.release()

