"""
ModernGL HUD Renderer

GPU-accelerated HUD elements:
- Crosshair (centered)
- Compass (centered, below crosshair)
- Coordinate display (top left)
- Minecraft-style item bar (bottom center)
- Status text (bottom, with black border for visibility)
- Performance stats (optional)
- DNA log PNG previews
"""

import numpy as np
import moderngl
from typing import Dict, Tuple, Optional, List
import math
import os
import time
from pathlib import Path

try:
    from pyglm import glm
except ImportError:
    import glm

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


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
uniform int u_use_texture;  // 0 = solid, 1 = mono font, 2 = RGBA image

void main() {
    // If UV is essentially (0,0) - this is a solid color quad, not textured
    // We detect this by checking if UV is very small (non-textured quads use 0,0,0,0)
    bool is_solid = (v_uv.x < 0.001 && v_uv.y < 0.001);
    
    if (u_use_texture == 2 && !is_solid) {
        // Full RGBA texture (PNG images)
        vec4 texColor = texture(u_texture, v_uv);
        fragColor = texColor * v_color;  // Multiply with vertex color for tinting
    } else if (u_use_texture == 1 && !is_solid) {
        float alpha = texture(u_texture, v_uv).r;  // Mono font texture
        fragColor = vec4(v_color.rgb, v_color.a * alpha);
    } else {
        // Solid color - use vertex color directly
        fragColor = v_color;
    }
}
"""


# Simple 8x8 bitmap font (extended character set)
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
    '/': [0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x00, 0x00],
    '(': [0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00],
    ')': [0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00],
    '%': [0x62, 0x64, 0x08, 0x10, 0x26, 0x46, 0x00, 0x00],
    '!': [0x18, 0x18, 0x18, 0x18, 0x00, 0x00, 0x18, 0x00],
    '?': [0x3C, 0x66, 0x06, 0x0C, 0x18, 0x00, 0x18, 0x00],
    '[': [0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00],
    ']': [0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00],
    '+': [0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00],
    '=': [0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00],
    '_': [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7E, 0x00],
    '>': [0x30, 0x18, 0x0C, 0x06, 0x0C, 0x18, 0x30, 0x00],
    '<': [0x0C, 0x18, 0x30, 0x60, 0x30, 0x18, 0x0C, 0x00],
    '*': [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00],
    # Uppercase letters
    'A': [0x18, 0x3C, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00],
    'B': [0x7C, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x7C, 0x00],
    'C': [0x3C, 0x66, 0x60, 0x60, 0x60, 0x66, 0x3C, 0x00],
    'D': [0x78, 0x6C, 0x66, 0x66, 0x66, 0x6C, 0x78, 0x00],
    'E': [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x7E, 0x00],
    'F': [0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00],
    'G': [0x3C, 0x66, 0x60, 0x6E, 0x66, 0x66, 0x3C, 0x00],
    'H': [0x66, 0x66, 0x66, 0x7E, 0x66, 0x66, 0x66, 0x00],
    'I': [0x3C, 0x18, 0x18, 0x18, 0x18, 0x18, 0x3C, 0x00],
    'J': [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x6C, 0x38, 0x00],
    'K': [0x66, 0x6C, 0x78, 0x70, 0x78, 0x6C, 0x66, 0x00],
    'L': [0x60, 0x60, 0x60, 0x60, 0x60, 0x60, 0x7E, 0x00],
    'M': [0x63, 0x77, 0x7F, 0x6B, 0x63, 0x63, 0x63, 0x00],
    'N': [0x66, 0x76, 0x7E, 0x7E, 0x6E, 0x66, 0x66, 0x00],
    'O': [0x3C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00],
    'P': [0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00],
    'Q': [0x3C, 0x66, 0x66, 0x66, 0x6A, 0x6C, 0x36, 0x00],
    'R': [0x7C, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0x00],
    'S': [0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00],
    'T': [0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00],
    'U': [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x00],
    'V': [0x66, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x00],
    'W': [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00],
    'X': [0x66, 0x66, 0x3C, 0x18, 0x3C, 0x66, 0x66, 0x00],
    'Y': [0x66, 0x66, 0x66, 0x3C, 0x18, 0x18, 0x18, 0x00],
    'Z': [0x7E, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x7E, 0x00],
}


# Tool icon definitions (8x8 bitmaps)
TOOL_ICONS = {
    'pickaxe': [
        0b00001111,
        0b00001111,
        0b00000111,
        0b00001110,
        0b00011100,
        0b00111000,
        0b01110000,
        0b11100000,
    ],
    'shovel': [
        0b00111100,
        0b01111110,
        0b01111110,
        0b00111100,
        0b00011000,
        0b00011000,
        0b00011000,
        0b00011000,
    ],
    'camera': [
        0b00110000,
        0b01111100,
        0b11111110,
        0b11011010,
        0b11011010,
        0b11111110,
        0b01111100,
        0b00000000,
    ],
    'axe': [
        0b00001111,
        0b00011111,
        0b00111111,
        0b00011110,
        0b00011100,
        0b00111000,
        0b01110000,
        0b11100000,
    ],
    'magnifier': [
        0b00111100,
        0b01000010,
        0b10000001,
        0b10000001,
        0b01000010,
        0b00111100,
        0b00001100,
        0b00000110,
    ],
    'microscope': [
        0b00011100,
        0b00111110,
        0b00011100,
        0b00011000,
        0b00011000,
        0b00111100,
        0b11111111,
        0b11111111,
    ],
}


# Compass directions
COMPASS_POINTS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


class ModernHUDRenderer:
    """
    GPU-accelerated HUD rendering for Core profile OpenGL.
    
    Features:
    - Centered crosshair
    - Compass bar (below crosshair)
    - Coordinate display (top left)
    - Minecraft-style item bar (bottom center)
    - Status text with black border for visibility
    """
    
    # Available tools with descriptions
    # First 3 match ToolType.ALL_TOOLS order: SCAN, MINE, FILL
    TOOLS = [
        ('camera', 'SCAN', 'SCAN TOOL - ANALYZE AND LOG DNA OF PLANTS/ANIMALS'),
        ('pickaxe', 'MINE', 'MINE TOOL - DIG INTO TERRAIN AND LOWER GROUND'),
        ('shovel', 'FILL', 'FILL TOOL - RAISE TERRAIN AND BUILD UP GROUND'),
        ('axe', 'CUT', 'CUT TOOL - HARVEST PLANTS AND TREES [COMING SOON]'),
        ('magnifier', 'ZOOM', 'ZOOM TOOL - INSPECT OBJECTS UP CLOSE [COMING SOON]'),
        ('microscope', 'MICRO', 'MICRO TOOL - VIEW CELLULAR DETAILS [COMING SOON]'),
    ]
    
    INVENTORY_COLS = 2  # 2 columns for inventory grid
    
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
        self.max_quads = 2000
        self.vertex_buffer = np.zeros(self.max_quads * 6 * 8, dtype='f4')
        self.vbo = ctx.buffer(reserve=self.vertex_buffer.nbytes, dynamic=True)
        
        # VAO
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, '2f 2f 4f', 'in_position', 'in_uv', 'in_color')],
        )
        
        # HUD visibility flags
        self.crosshair_enabled = True
        self.coords_enabled = True
        self.compass_enabled = True
        self.item_bar_enabled = True
        self.status_enabled = True
        self.stats_enabled = False  # Performance overlay
        
        # Current state
        self.selected_tool = 0  # Index into TOOLS
        self.status_text = ""
        self.status_timer = 0.0  # Remaining time to display (synced from camera)
        self.yaw = 0.0  # Camera yaw for compass
        
        # Frame stats
        self.frame_stats = {}
        
        # Menu state
        self.menu_open = False
        self.menu_tab = 1  # 0=Inventory, 1=Log, 2=Controls
        self.menu_tab_names = ["INVENTORY", "LOG", "CONTROLS"]
        self.log_items = []  # List of DNA log filenames
        self.log_index = 0   # Currently selected log item
        self.log_filter = 0  # 0=ALL, 1=PLANTS, 2=ANIMALS
        self.log_filter_names = ["ALL", "PLANTS", "ANIMALS"]
        self.favorites = set()  # Set of favorited log filenames
        self.inventory_index = 0  # Selected tool in inventory tab
        
        # PNG image cache for DNA log previews
        self._image_cache: Dict[str, moderngl.Texture] = {}
        self._cache_max_size = 20  # Limit cached images
        self._dna_logs_path = Path.home() / ".waverse" / "dna_logs"
        self._pending_preview: Optional[Dict] = None  # Store pending preview info
        self._dna_cache: Dict[str, Dict] = {}  # Cache for loaded DNA data
        self._preview_rotation = 0.0  # Animation rotation for preview
        
    def _load_png_texture(self, png_path: str) -> Optional[moderngl.Texture]:
        """Load a PNG image as a ModernGL texture, with caching."""
        if not PIL_AVAILABLE:
            return None
        
        # Check cache first
        if png_path in self._image_cache:
            return self._image_cache[png_path]
        
        try:
            if not os.path.exists(png_path):
                return None
            
            # Load with PIL
            img = Image.open(png_path)
            img = img.convert('RGBA')
            
            # Resize if too large (for performance)
            max_size = 256
            if img.width > max_size or img.height > max_size:
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Flip vertically for OpenGL
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            
            # Create texture
            texture = self.ctx.texture(img.size, 4, img.tobytes())
            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
            # Cache it (with LRU-style eviction)
            if len(self._image_cache) >= self._cache_max_size:
                # Remove oldest entry
                oldest = next(iter(self._image_cache))
                self._image_cache[oldest].release()
                del self._image_cache[oldest]
            
            self._image_cache[png_path] = texture
            return texture
            
        except Exception as e:
            print(f"Error loading PNG {png_path}: {e}")
            return None
    
    def _draw_image(self, x: float, y: float, w: float, h: float, 
                   texture: moderngl.Texture, alpha: float = 1.0):
        """Draw a textured quad with the given image texture."""
        # Build vertices for the image quad
        vertices = []
        
        # Full UV coords for image
        u0, v0, u1, v1 = 0.0, 0.0, 1.0, 1.0
        
        # Quad vertices (2 triangles)
        # UV is flipped because we already flipped the image
        v0_flipped, v1_flipped = 1.0, 0.0
        
        color = (1.0, 1.0, 1.0, alpha)  # White tint = original colors
        
        # Triangle 1
        vertices.extend([x, y, u0, v0_flipped, *color])
        vertices.extend([x + w, y, u1, v0_flipped, *color])
        vertices.extend([x, y + h, u0, v1_flipped, *color])
        # Triangle 2
        vertices.extend([x + w, y, u1, v0_flipped, *color])
        vertices.extend([x + w, y + h, u1, v1_flipped, *color])
        vertices.extend([x, y + h, u0, v1_flipped, *color])
        
        # Update VBO
        vertex_data = np.array(vertices, dtype='f4')
        self.vbo.write(vertex_data.tobytes())
        
        # Render with image texture
        texture.use(0)
        self.program['u_texture'].value = 0
        self.program['u_use_texture'].value = 2  # RGBA image mode
        self.program['u_screen_size'].value = (self.screen_width, self.screen_height)
        self.program['u_offset'].value = (0, 0)
        
        self.vao.render(moderngl.TRIANGLES, vertices=6)
        
    def _create_font_texture(self):
        """Create a texture atlas from the bitmap font and tool icons."""
        # Create a 256x256 texture
        texture_size = 256
        char_size = 8
        chars_per_row = texture_size // char_size
        
        texture_data = np.zeros((texture_size, texture_size), dtype=np.uint8)
        
        # Write font characters (rows 0-3)
        self.char_uvs = {}
        chars = ' 0123456789-.,:%/()!?[]+=_><*ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for i, char in enumerate(chars):
            if char in BITMAP_FONT:
                row = i // chars_per_row
                col = i % chars_per_row
                
                # Write bitmap to texture
                for y, byte in enumerate(BITMAP_FONT[char]):
                    for x in range(8):
                        if byte & (0x80 >> x):
                            ty = row * char_size + y
                            tx = col * char_size + x
                            if ty < texture_size and tx < texture_size:
                                texture_data[ty, tx] = 255
                
                # Store UV coordinates
                u0 = col * char_size / texture_size
                v0 = row * char_size / texture_size
                u1 = (col + 1) * char_size / texture_size
                v1 = (row + 1) * char_size / texture_size
                self.char_uvs[char] = (u0, v0, u1, v1)
        
        # Write tool icons (row 6)
        self.tool_uvs = {}
        for i, (tool_name, icon_data) in enumerate(TOOL_ICONS.items()):
            row = 6
            col = i
            
            for y, byte in enumerate(icon_data):
                for x in range(8):
                    if byte & (0x80 >> x):
                        ty = row * char_size + y
                        tx = col * char_size + x
                        if ty < texture_size and tx < texture_size:
                            texture_data[ty, tx] = 255
            
            u0 = col * char_size / texture_size
            v0 = row * char_size / texture_size
            u1 = (col + 1) * char_size / texture_size
            v1 = (row + 1) * char_size / texture_size
            self.tool_uvs[tool_name] = (u0, v0, u1, v1)
        
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
    
    def set_yaw(self, yaw: float):
        """Set camera yaw for compass."""
        self.yaw = yaw
    
    def select_tool(self, index: int):
        """Select a tool by index."""
        if 0 <= index < len(self.TOOLS):
            self.selected_tool = index
    
    def next_tool(self):
        """Select next tool."""
        self.selected_tool = (self.selected_tool + 1) % len(self.TOOLS)
    
    def prev_tool(self):
        """Select previous tool."""
        self.selected_tool = (self.selected_tool - 1) % len(self.TOOLS)
    
    def set_status(self, text: str, duration: float = 3.0):
        """Set status text with timer."""
        self.status_text = text
        self.status_timer = duration
    
    def get_current_tool(self) -> str:
        """Get current tool name."""
        return self.TOOLS[self.selected_tool][0]
    
    def set_tool_by_name(self, tool_name: str):
        """Set selected tool by name (SCAN, MINE, FILL, etc.)."""
        # Map from camera tool names to HUD tool indices
        tool_map = {
            'SCAN': 2,   # camera icon
            'MINE': 0,   # pickaxe icon
            'FILL': 1,   # shovel icon
            'CUT': 3,    # axe icon
            'ZOOM': 4,   # magnifier icon
            'MICRO': 5,  # microscope icon
        }
        if tool_name.upper() in tool_map:
            self.selected_tool = tool_map[tool_name.upper()]
    
    def _add_quad(self, vertices: List, x: float, y: float, w: float, h: float,
                  u0: float = 0, v0: float = 0, u1: float = 1, v1: float = 1,
                  r: float = 1, g: float = 1, b: float = 1, a: float = 1):
        """Add a textured quad to the vertex list."""
        # Two triangles for a quad
        vertices.extend([x, y, u0, v0, r, g, b, a])
        vertices.extend([x+w, y, u1, v0, r, g, b, a])
        vertices.extend([x, y+h, u0, v1, r, g, b, a])
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
    
    def _draw_text_with_shadow(self, vertices: List, text: str, x: float, y: float,
                                scale: float = 2.0, r: float = 1, g: float = 1, b: float = 1, a: float = 1,
                                shadow_offset: float = 2.0):
        """Draw text with black shadow/border for visibility."""
        # Draw shadow (black, offset)
        self._draw_text(vertices, text, x + shadow_offset, y + shadow_offset, scale, 0, 0, 0, a * 0.8)
        # Draw main text
        self._draw_text(vertices, text, x, y, scale, r, g, b, a)
    
    def _draw_crosshair(self, vertices: List):
        """Draw centered crosshair."""
        cx = self.screen_width / 2
        cy = self.screen_height / 2
        size = 14
        thickness = 2
        gap = 4  # Gap in center
        
        # Horizontal lines (with gap)
        self._add_quad(vertices, cx - size, cy - thickness/2, 
                      size - gap, thickness, 0, 0, 0, 0, 1, 1, 1, 0.9)
        self._add_quad(vertices, cx + gap, cy - thickness/2, 
                      size - gap, thickness, 0, 0, 0, 0, 1, 1, 1, 0.9)
        
        # Vertical lines (with gap)
        self._add_quad(vertices, cx - thickness/2, cy - size, 
                      thickness, size - gap, 0, 0, 0, 0, 1, 1, 1, 0.9)
        self._add_quad(vertices, cx - thickness/2, cy + gap, 
                      thickness, size - gap, 0, 0, 0, 0, 1, 1, 1, 0.9)
        
        # Center dot
        self._add_quad(vertices, cx - 1, cy - 1, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1.0)
    
    def _draw_compass(self, vertices: List):
        """Draw compass bar at top of screen."""
        cx = self.screen_width / 2
        cy = 8  # Top of screen
        
        bar_width = 550  # Wide for directions + future markers
        bar_height = 32
        
        # Background bar (semi-transparent dark for visibility)
        self._add_quad(vertices, cx - bar_width/2, cy, 
                      bar_width, bar_height, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.55)
        
        # Calculate compass direction from yaw
        # yaw = 0 is -Z (North), 90 is -X (West), etc.
        # Normalize yaw to 0-360
        yaw_norm = (-self.yaw + 360) % 360
        
        # Draw direction markers
        # Each marker is placed based on its angle relative to current yaw
        directions = [
            (0, 'N', (1.0, 0.4, 0.4)),      # Red for North
            (45, 'NE', (0.8, 0.8, 0.8)),
            (90, 'E', (1.0, 1.0, 1.0)),
            (135, 'SE', (0.8, 0.8, 0.8)),
            (180, 'S', (0.4, 0.4, 1.0)),    # Blue for South
            (225, 'SW', (0.8, 0.8, 0.8)),
            (270, 'W', (1.0, 1.0, 1.0)),
            (315, 'NW', (0.8, 0.8, 0.8)),
        ]
        
        bar_left = cx - bar_width / 2
        bar_right = cx + bar_width / 2
        
        for angle, label, color in directions:
            # Calculate relative angle
            rel_angle = (angle - yaw_norm + 180) % 360 - 180  # -180 to 180
            
            # Only show if within view range
            if abs(rel_angle) < 90:
                # Map to bar position
                x_offset = (rel_angle / 90) * (bar_width / 2)
                
                # Calculate text position and width
                text_scale = 2.0
                char_width = 8 * text_scale
                text_width = len(label) * char_width
                text_x = cx + x_offset - text_width / 2  # Center text
                
                # Skip if text would extend past bar edges (with padding)
                padding = 10
                if text_x < bar_left + padding or text_x + text_width > bar_right - padding:
                    continue
                
                self._draw_text(vertices, label, text_x, cy + 6, text_scale, 
                               color[0], color[1], color[2], 0.9)
        
        # Draw center indicator (small triangle/line at bottom of compass)
        self._add_quad(vertices, cx - 1, cy + bar_height - 4, 2, 4, 0, 0, 0, 0, 1, 0.8, 0.2, 1.0)
    
    def _draw_item_bar(self, vertices: List):
        """Draw Minecraft-style item bar at bottom - minimal, just icons and selection."""
        num_slots = len(self.TOOLS)
        slot_size = 52
        slot_gap = 16  # More spacing between items
        
        cx = self.screen_width / 2
        total_width = num_slots * slot_size + (num_slots - 1) * slot_gap
        bar_x = cx - total_width / 2
        bar_y = self.screen_height - slot_size - 30  # Near bottom
        
        # No background bar - just the slots
        
        # Draw slots
        for i, (tool_name, tool_label, _) in enumerate(self.TOOLS):
            slot_x = bar_x + i * (slot_size + slot_gap)
            slot_y = bar_y
            
            is_selected = (i == self.selected_tool)
            
            # Selected slot gets a bright yellow/gold square border
            if is_selected:
                border_size = 3
                # Draw gold selection square
                self._add_quad(vertices, slot_x - border_size, slot_y - border_size, 
                              slot_size + border_size * 2, slot_size + border_size * 2, 
                              0, 0, 0, 0, 1.0, 0.85, 0.1, 1.0)  # Bright gold
                # Inner dark background
                self._add_quad(vertices, slot_x, slot_y, slot_size, slot_size, 
                              0, 0, 0, 0, 0.15, 0.15, 0.18, 0.9)
            
            # Tool icon (scaled up)
            if tool_name in self.tool_uvs:
                icon_scale = 4
                icon_size = 8 * icon_scale
                icon_x = slot_x + (slot_size - icon_size) / 2
                icon_y = slot_y + (slot_size - icon_size) / 2 - 4
                
                u0, v0, u1, v1 = self.tool_uvs[tool_name]
                
                # Icon color based on tool
                if tool_name == 'pickaxe':
                    color = (0.7, 0.5, 0.3)  # Brown
                elif tool_name == 'shovel':
                    color = (0.6, 0.6, 0.6)  # Gray
                elif tool_name == 'camera':
                    color = (0.3, 0.7, 0.9)  # Cyan
                elif tool_name == 'axe':
                    color = (0.5, 0.3, 0.2)  # Dark brown
                elif tool_name == 'magnifier':
                    color = (0.4, 0.8, 0.4)  # Green
                elif tool_name == 'microscope':
                    color = (0.8, 0.4, 0.8)  # Purple
                else:
                    color = (1.0, 1.0, 1.0)
                
                self._add_quad(vertices, icon_x, icon_y, icon_size, icon_size,
                              u0, v0, u1, v1, color[0], color[1], color[2], 1.0)
            
            # Tool label (centered text below icon)
            char_width = 8 * 1.5  # scale = 1.5
            text_width = len(tool_label) * char_width
            label_x = slot_x + (slot_size - text_width) / 2
            label_y = slot_y + slot_size + 4  # Below the slot
            label_alpha = 1.0 if is_selected else 0.6
            self._draw_text_with_shadow(vertices, tool_label, label_x, label_y, 1.5, 
                                        0.9, 0.9, 0.9, label_alpha, shadow_offset=1)
    
    def _draw_status_text(self, vertices: List):
        """Draw status text in top-right area (below compass) with auto-fade."""
        if not self.status_text or self.status_timer <= 0:
            return
        
        # Fade out during last 0.5 seconds
        if self.status_timer < 0.5:
            alpha = self.status_timer / 0.5
        else:
            alpha = 1.0
        
        scale = 1.6
        char_width = 8 * scale
        line_height = int(8 * scale + 4)
        
        # Max width for text (right side of screen, leave margin)
        max_width = 350
        max_chars = int(max_width / char_width)
        
        # Position: top-right, aligned with top HUD elements
        margin_right = 15
        start_y = 10  # Same level as coordinates/compass
        
        # Word wrap the text
        words = self.status_text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            if len(test_line) <= max_chars:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                # Handle very long words
                if len(word) > max_chars:
                    while len(word) > max_chars:
                        lines.append(word[:max_chars])
                        word = word[max_chars:]
                    current_line = word
                else:
                    current_line = word
        if current_line:
            lines.append(current_line)
        
        # Calculate box dimensions
        total_height = len(lines) * line_height
        max_line_width = max(len(line) for line in lines) * char_width if lines else 0
        
        # Position from right edge
        box_x = self.screen_width - max_line_width - margin_right - 12
        box_y = start_y
        
        # Background box
        padding = 6
        self._add_quad(vertices, box_x - padding, box_y - padding/2, 
                      max_line_width + padding * 2, total_height + padding,
                      0, 0, 0, 0, 0.0, 0.0, 0.0, 0.7 * alpha)
        
        # Draw each line
        for i, line in enumerate(lines):
            line_y = box_y + i * line_height
            # Right-align each line
            line_width = len(line) * char_width
            line_x = self.screen_width - line_width - margin_right - 6
            self._draw_text_with_shadow(vertices, line, line_x, line_y, scale, 
                                        1.0, 1.0, 0.5, alpha, shadow_offset=1)
    
    def _draw_menu(self, vertices: List):
        """Draw the menu overlay."""
        if not self.menu_open:
            return
        
        # Menu dimensions
        menu_w = min(800, self.screen_width - 50)
        menu_h = min(500, self.screen_height - 50)
        menu_x = (self.screen_width - menu_w) / 2
        menu_y = (self.screen_height - menu_h) / 2
        
        # Semi-transparent fullscreen overlay
        self._add_quad(vertices, 0, 0, self.screen_width, self.screen_height,
                      0, 0, 0, 0, 0.0, 0.0, 0.0, 0.7)
        
        # Menu background
        self._add_quad(vertices, menu_x, menu_y, menu_w, menu_h,
                      0, 0, 0, 0, 0.12, 0.12, 0.18, 0.95)
        
        # Menu border (4 lines as thin quads)
        border_color = (0.4, 0.6, 0.8, 1.0)
        border_w = 2
        # Top
        self._add_quad(vertices, menu_x, menu_y, menu_w, border_w,
                      0, 0, 0, 0, *border_color)
        # Bottom
        self._add_quad(vertices, menu_x, menu_y + menu_h - border_w, menu_w, border_w,
                      0, 0, 0, 0, *border_color)
        # Left
        self._add_quad(vertices, menu_x, menu_y, border_w, menu_h,
                      0, 0, 0, 0, *border_color)
        # Right
        self._add_quad(vertices, menu_x + menu_w - border_w, menu_y, border_w, menu_h,
                      0, 0, 0, 0, *border_color)
        
        # Tab bar
        tab_h = 35
        tab_w = menu_w / len(self.menu_tab_names)
        
        for i, name in enumerate(self.menu_tab_names):
            tab_x = menu_x + i * tab_w
            tab_y = menu_y
            
            # Tab background (selected = brighter)
            if i == self.menu_tab:
                self._add_quad(vertices, tab_x + 2, tab_y + 2, tab_w - 4, tab_h - 2,
                              0, 0, 0, 0, 0.25, 0.35, 0.5, 1.0)
            else:
                self._add_quad(vertices, tab_x + 2, tab_y + 2, tab_w - 4, tab_h - 2,
                              0, 0, 0, 0, 0.1, 0.1, 0.15, 1.0)
            
            # Tab text
            text_scale = 1.8
            text_w = len(name) * 8 * text_scale
            text_x = tab_x + (tab_w - text_w) / 2
            text_y = tab_y + 10
            color = (1.0, 1.0, 1.0) if i == self.menu_tab else (0.6, 0.6, 0.6)
            self._draw_text(vertices, name, text_x, text_y, text_scale, *color, 1.0)
        
        # Content area
        content_y = menu_y + tab_h + 10
        content_h = menu_h - tab_h - 45
        
        # Draw content based on selected tab
        if self.menu_tab == 0:  # Inventory
            self._draw_inventory_tab(vertices, menu_x, content_y, menu_w, content_h)
        
        elif self.menu_tab == 1:  # Log
            self._draw_log_tab(vertices, menu_x, content_y, menu_w, content_h)
        
        elif self.menu_tab == 2:  # Controls
            self._draw_controls_tab(vertices, menu_x, content_y, menu_w, content_h)
        
        # Help text at bottom
        help_y = menu_y + menu_h - 28
        if self.menu_tab == 0:  # Inventory
            help_text = "DPAD:SELECT  A:EQUIP  L1/R1:TABS  START:CLOSE"
        elif self.menu_tab == 1:  # Log
            help_text = "UP/DN:SCROLL  L/R:FILTER  A:FAV  L1/R1:TABS  START:CLOSE"
        else:
            help_text = "L1/R1:TABS  START:CLOSE"
        
        text_scale = 1.5
        text_w = len(help_text) * 8 * text_scale
        help_x = menu_x + (menu_w - text_w) / 2
        self._draw_text(vertices, help_text, help_x, help_y, text_scale, 0.5, 0.5, 0.5, 1.0)
    
    def _draw_log_tab(self, vertices: List, menu_x: float, content_y: float, 
                      menu_w: float, content_h: float):
        """Draw the LOG tab content."""
        # Filter header
        filter_text = f"FILTER:[{self.log_filter_names[self.log_filter]}]"
        self._draw_text(vertices, filter_text, menu_x + 20, content_y, 1.8, 0.5, 0.8, 1.0, 1.0)
        content_y += 25
        content_h -= 25
        
        if len(self.log_items) == 0:
            if self.log_filter == 0:  # ALL filter but no items
                self._draw_text(vertices, "NO DNA LOGS YET", menu_x + 20, content_y, 2.0, 0.7, 0.7, 0.7, 1.0)
                self._draw_text(vertices, "USE SCAN TOOL R1 TO LOG CREATURES", 
                               menu_x + 20, content_y + 25, 1.8, 0.5, 0.5, 0.5, 1.0)
            else:  # Filter active but no matching items
                filter_name = self.log_filter_names[self.log_filter]
                self._draw_text(vertices, f"NO {filter_name} FOUND", menu_x + 20, content_y, 2.0, 0.7, 0.7, 0.7, 1.0)
                self._draw_text(vertices, "TRY SCANNING SOME OR CHANGE FILTER", 
                               menu_x + 20, content_y + 25, 1.8, 0.5, 0.5, 0.5, 1.0)
            return
        
        # Split: list on left, preview hint on right
        list_width = menu_w * 0.6
        
        # List items
        item_h = 24
        visible_items = int(content_h / item_h)
        start_idx = max(0, self.log_index - visible_items // 2)
        
        for i, item in enumerate(self.log_items[start_idx:start_idx + visible_items]):
            iy = content_y + i * item_h
            actual_idx = start_idx + i
            
            # Highlight selected item
            if actual_idx == self.log_index:
                self._add_quad(vertices, menu_x + 10, iy - 2, list_width - 20, item_h - 2,
                              0, 0, 0, 0, 0.25, 0.4, 0.6, 0.8)
            
            # Parse filename for display
            parts = item.replace('.json', '').split('_')
            if len(parts) >= 3:
                entity_type = parts[0].upper()
                date = parts[2] if len(parts) > 2 else ""
                time_str = parts[3] if len(parts) > 3 else ""
                display_text = f"{entity_type} - {date} {time_str}"
            else:
                display_text = item[:30]
            
            # Add star for favorites
            is_favorite = item in self.favorites
            if is_favorite:
                display_text = "* " + display_text
            
            # Item color
            if is_favorite:
                color = (1.0, 0.85, 0.2)  # Gold
            elif actual_idx == self.log_index:
                color = (0.4, 0.8, 1.0)   # Cyan
            else:
                color = (0.7, 0.7, 0.7)   # Gray
            
            self._draw_text(vertices, display_text[:35], menu_x + 20, iy, 1.6, *color, 1.0)
        
        # Preview area (right side)
        preview_x = menu_x + list_width + 20
        preview_w = menu_w - list_width - 40
        preview_h = content_h - 20
        
        # Preview background (dark)
        self._add_quad(vertices, preview_x, content_y, preview_w, preview_h,
                      0, 0, 0, 0, 0.02, 0.03, 0.05, 1.0)
        
        # Preview border
        self._add_quad(vertices, preview_x, content_y, preview_w, 2,
                      0, 0, 0, 0, 0.3, 0.4, 0.5, 0.8)
        self._add_quad(vertices, preview_x, content_y + preview_h - 2, preview_w, 2,
                      0, 0, 0, 0, 0.3, 0.4, 0.5, 0.8)
        self._add_quad(vertices, preview_x, content_y, 2, preview_h,
                      0, 0, 0, 0, 0.3, 0.4, 0.5, 0.8)
        self._add_quad(vertices, preview_x + preview_w - 2, content_y, 2, preview_h,
                      0, 0, 0, 0, 0.3, 0.4, 0.5, 0.8)
        
        # Show entity info in preview area (works on all platforms)
        if 0 <= self.log_index < len(self.log_items):
            selected = self.log_items[self.log_index]
            parts = selected.replace('.json', '').split('_')
            
            # Entity type header
            if len(parts) >= 1:
                entity_type = parts[0].upper()
                type_color = (0.4, 0.8, 0.4) if entity_type == "PLANT" else (0.8, 0.6, 0.3)
                self._draw_text(vertices, entity_type, preview_x + 10, content_y + 10, 2.0, *type_color, 1.0)
            
            # Date/time
            if len(parts) >= 4:
                date_str = parts[2]
                time_str = parts[3]
                # Format: YYYYMMDD -> YYYY-MM-DD
                if len(date_str) == 8:
                    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                else:
                    date_fmt = date_str
                # Format: HHMMSS -> HH:MM:SS
                if len(time_str) == 6:
                    time_fmt = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}"
                else:
                    time_fmt = time_str
                self._draw_text(vertices, f"DATE: {date_fmt}", preview_x + 10, content_y + 35, 1.5, 0.6, 0.6, 0.6, 1.0)
                self._draw_text(vertices, f"TIME: {time_fmt}", preview_x + 10, content_y + 52, 1.5, 0.6, 0.6, 0.6, 1.0)
            
            # Favorite status
            if selected in self.favorites:
                self._draw_text(vertices, "* FAVORITE *", preview_x + 10, content_y + 75, 1.5, 1.0, 0.85, 0.2, 1.0)
            
            # Store preview info for later PNG rendering (after main vertices are drawn)
            self._pending_preview = {
                'filename': selected,
                'x': preview_x + 5,
                'y': content_y + 95,
                'w': preview_w - 10,
                'h': preview_h - 110
            }
    
    def _draw_inventory_tab(self, vertices: List, menu_x: float, content_y: float,
                             menu_w: float, content_h: float):
        """Draw the INVENTORY tab content - shows available tools."""
        # Header
        self._draw_text(vertices, "TOOLS", menu_x + 20, content_y, 2.0, 0.4, 0.8, 1.0, 1.0)
        content_y += 30
        
        # Tool grid - 2 columns
        cols = 2
        slot_w = (menu_w - 60) / cols
        slot_h = 60
        
        # First 3 tools are implemented
        num_implemented = 3
        
        for i, (icon_name, label, description) in enumerate(self.TOOLS):
            col = i % cols
            row = i // cols
            
            slot_x = menu_x + 20 + col * slot_w
            slot_y = content_y + row * slot_h
            
            is_selected = (i == self.inventory_index)
            is_equipped = (i == self.selected_tool)
            is_available = (i < num_implemented)
            
            # Slot background
            if is_selected:
                # Selected item - bright highlight
                if is_available:
                    self._add_quad(vertices, slot_x, slot_y, slot_w - 10, slot_h - 8,
                                  0, 0, 0, 0, 0.25, 0.4, 0.6, 0.9)
                else:
                    # Unavailable - dimmer selection
                    self._add_quad(vertices, slot_x, slot_y, slot_w - 10, slot_h - 8,
                                  0, 0, 0, 0, 0.2, 0.2, 0.25, 0.8)
            elif is_equipped:
                # Currently equipped - subtle highlight
                self._add_quad(vertices, slot_x, slot_y, slot_w - 10, slot_h - 8,
                              0, 0, 0, 0, 0.15, 0.25, 0.15, 0.7)
            
            # Tool name - dim if unavailable
            if is_available:
                name_color = (1.0, 1.0, 1.0) if is_selected else (0.7, 0.7, 0.7)
            else:
                name_color = (0.5, 0.5, 0.5) if is_selected else (0.35, 0.35, 0.35)
            self._draw_text(vertices, label, slot_x + 10, slot_y + 10, 2.2, *name_color, 1.0)
            
            # Status indicator
            if is_equipped:
                self._draw_text(vertices, "[EQUIPPED]", slot_x + 10, slot_y + 35, 1.3, 0.4, 0.9, 0.4, 1.0)
            elif not is_available:
                self._draw_text(vertices, "[COMING SOON]", slot_x + 10, slot_y + 35, 1.2, 0.5, 0.4, 0.3, 0.8)
            else:
                self._draw_text(vertices, "PRESS A TO EQUIP", slot_x + 10, slot_y + 35, 1.2, 0.5, 0.5, 0.5, 0.7)
        
        # Description area at bottom
        desc_y = menu_y = content_y + ((len(self.TOOLS) + cols - 1) // cols) * slot_h + 20
        
        # Description background
        self._add_quad(vertices, menu_x + 15, desc_y, menu_w - 30, 50,
                      0, 0, 0, 0, 0.08, 0.08, 0.12, 0.9)
        
        # Description border
        self._add_quad(vertices, menu_x + 15, desc_y, menu_w - 30, 2,
                      0, 0, 0, 0, 0.3, 0.4, 0.5, 0.8)
        
        # Selected tool description
        if 0 <= self.inventory_index < len(self.TOOLS):
            _, label, description = self.TOOLS[self.inventory_index]
            self._draw_text(vertices, description, menu_x + 25, desc_y + 18, 1.6, 0.9, 0.9, 0.7, 1.0)
    
    def _draw_controls_tab(self, vertices: List, menu_x: float, content_y: float,
                           menu_w: float, content_h: float):
        """Draw the CONTROLS tab content."""
        line_h = 18
        left_col = menu_x + 20
        right_col = menu_x + menu_w / 2 + 20
        y = content_y
        
        # Keyboard column
        self._draw_text(vertices, "=== KEYBOARD ===", left_col, y, 1.8, 0.4, 0.8, 1.0, 1.0)
        y += line_h + 5
        
        kb_controls = [
            "WASD/ARROWS: MOVE",
            "IJKL: LOOK",
            "H/SPACE: UP/JUMP",
            "F/SHIFT: DOWN",
            "[/-: SLOWER",
            "]/=: FASTER",
            "B HOLD: RUN 2X",
            "R: USE TOOL",
            ",/.: SWITCH TOOL",
            "T: TOOL SIZE",
            "P: SCREENSHOT",
            "TAB: MENU",
            "X: TOUR MODE",
            "N: WARP",
        ]
        
        for ctrl in kb_controls:
            self._draw_text(vertices, ctrl, left_col, y, 1.5, 0.7, 0.7, 0.7, 1.0)
            y += line_h
        
        # Gamepad column
        y = content_y
        self._draw_text(vertices, "=== GAMEPAD ===", right_col, y, 1.8, 0.4, 0.8, 1.0, 1.0)
        y += line_h + 5
        
        gp_controls = [
            "L STICK: MOVE",
            "R STICK: LOOK",
            "A: ACTION/TALK",
            "B: JUMP",
            "X: FLY/WALK",
            "Y: SCREENSHOT",
            "L2/R2: SPEED",
            "R1: USE TOOL",
            "L3 HOLD: RUN 2X",
            "DPAD L/R: TOOLS",
            "DPAD U/D: RADIUS",
            "START: MENU",
            "SELECT: WARP",
        ]
        
        for ctrl in gp_controls:
            self._draw_text(vertices, ctrl, right_col, y, 1.5, 0.7, 0.7, 0.7, 1.0)
            y += line_h
    
    def sync_from_camera(self, camera):
        """Sync menu state from Camera object."""
        self.menu_open = camera.menu_open
        self.menu_tab = camera.menu_tab
        self.log_items = list(camera.log_items)  # Copy list to get current state
        self.log_index = camera.log_index
        self.log_filter = camera.log_filter
        self.favorites = set(camera.favorites)  # Copy set to get current state
        
        # Sync status text and timer from camera
        if hasattr(camera, 'status_message'):
            self.status_text = camera.status_message
        if hasattr(camera, 'status_timer'):
            self.status_timer = camera.status_timer
        
        # Sync selected tool from camera
        if hasattr(camera, 'current_tool_index'):
            self.selected_tool = camera.current_tool_index
        
        # Sync inventory selection
        self.inventory_index = camera.inventory_index if hasattr(camera, 'inventory_index') else 0
    
    def get_preview_info(self):
        """Get info needed for entity preview rendering.
        
        Returns: (should_render, x, y, size, json_path) or (False, 0, 0, 0, None)
        """
        if not self.menu_open or self.menu_tab != 1:
            return (False, 0, 0, 0, None)
        
        if len(self.log_items) == 0 or self.log_index >= len(self.log_items):
            return (False, 0, 0, 0, None)
        
        # Calculate preview area (must match _draw_log_tab layout)
        menu_w = min(800, self.screen_width - 50)
        menu_h = min(500, self.screen_height - 50)
        menu_x = (self.screen_width - menu_w) / 2
        menu_y = (self.screen_height - menu_h) / 2
        
        tab_h = 35
        content_y = menu_y + tab_h + 10 + 25  # +25 for filter header
        content_h = menu_h - tab_h - 45 - 25
        
        list_width = menu_w * 0.6
        preview_x = menu_x + list_width + 20
        preview_w = menu_w - list_width - 40
        preview_h = content_h - 20
        preview_size = min(preview_w, preview_h) - 20  # Leave some padding
        
        # Center the preview in the preview area
        preview_render_x = preview_x + (preview_w - preview_size) / 2
        preview_render_y = content_y + (preview_h - preview_size) / 2
        
        selected_item = self.log_items[self.log_index]
        
        return (True, preview_render_x, preview_render_y, preview_size, selected_item)
    
    def render(self, camera_x: float = 0, camera_y: float = 0, camera_z: float = 0,
               fps: float = 60, time_of_day: float = 0.5,
               chunk_x: int = 0, chunk_z: int = 0,
               terrain_chunks: int = 0, flora_instances: int = 0,
               biome: str = ""):
        """Render HUD elements."""
        vertices = []
        
        # Crosshair
        if self.crosshair_enabled:
            self._draw_crosshair(vertices)
        
        # Compass
        if self.compass_enabled:
            self._draw_compass(vertices)
        
        # Coordinates & FPS (top left)
        if self.coords_enabled:
            y_offset = 10
            
            # Coordinates with shadow (X=horizontal, Z=height, Y=depth for top-down view)
            coord_text = f"X:{int(camera_x)} Z:{int(camera_y)} Y:{int(camera_z)}"
            self._draw_text_with_shadow(vertices, coord_text, 10, y_offset, 2.0, 1, 1, 1, 0.9)
            
            # FPS
            fps_text = f"FPS:{int(fps)}"
            self._draw_text_with_shadow(vertices, fps_text, 10, y_offset + 22, 1.8, 0.5, 1, 0.5, 0.8)
            
            # Biome (if provided)
            # if biome:
            #     self._draw_text_with_shadow(vertices, biome.upper(), 10, y_offset + 42, 1.8, 
            #                                0.8, 0.8, 1.0, 0.8)
        
        # Item bar
        if self.item_bar_enabled:
            self._draw_item_bar(vertices)
        
        # Status text
        if self.status_enabled:
            self._draw_status_text(vertices)
        
        # Performance stats (optional)
        if self.stats_enabled and self.frame_stats:
            y = 80
            for key, value in self.frame_stats.items():
                text = f"{key}:{value}"
                self._draw_text(vertices, text, 10, y, 1.5, 0.8, 0.8, 0.8, 0.7)
                y += 16
        
        # Clear pending preview before menu draw (will be set if needed)
        self._pending_preview = None
        
        # Menu overlay (drawn last, on top of everything)
        self._draw_menu(vertices)
        
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
        
        # After main HUD, draw preview if pending (on Log tab)
        if self._pending_preview:
            self._render_png_preview()
        
        # Re-enable depth test
        self.ctx.enable(moderngl.DEPTH_TEST)
    
    def _render_png_preview(self):
        """Render the PNG preview image for the selected log entry."""
        if not self._pending_preview:
            return
            
        filename = self._pending_preview['filename']
        x = self._pending_preview['x']
        y = self._pending_preview['y']
        w = self._pending_preview['w']
        h = self._pending_preview['h']
        
        # Get PNG path from JSON filename
        png_filename = filename.replace('.json', '.png')
        png_path = str(self._dna_logs_path / png_filename)
        
        # Try to load the texture
        texture = self._load_png_texture(png_path)
        
        if texture:
            # Calculate aspect-correct size
            img_aspect = texture.width / texture.height
            area_aspect = w / h
            
            if img_aspect > area_aspect:
                # Image is wider - fit to width
                draw_w = w
                draw_h = w / img_aspect
            else:
                # Image is taller - fit to height
                draw_h = h
                draw_w = h * img_aspect
            
            # Center in the preview area
            draw_x = x + (w - draw_w) / 2
            draw_y = y + (h - draw_h) / 2
            
            # Draw the image
            self._draw_image(draw_x, draw_y, draw_w, draw_h, texture)
        else:
            # No PNG - render a 2D stylized preview from DNA
            self._render_dna_preview(filename, x, y, w, h)
    
    def _load_dna_from_json(self, filename: str) -> Optional[Dict]:
        """Load DNA data from a JSON log file."""
        if filename in self._dna_cache:
            return self._dna_cache[filename]
        
        try:
            json_path = self._dna_logs_path / filename
            if not json_path.exists():
                return None
            
            import json
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Cache it
            if len(self._dna_cache) >= self._cache_max_size:
                # Remove oldest
                oldest = next(iter(self._dna_cache))
                del self._dna_cache[oldest]
            
            self._dna_cache[filename] = data
            return data
            
        except Exception as e:
            print(f"Error loading DNA: {e}")
            return None
    
    def _render_dna_preview(self, filename: str, x: float, y: float, w: float, h: float):
        """Render a stylized 2D preview of the entity from its DNA."""
        import json
        
        # Update rotation for animation
        self._preview_rotation += 0.02
        
        # Load DNA
        dna_data = self._load_dna_from_json(filename)
        if not dna_data:
            return
        
        vertices = []
        center_x = x + w / 2
        center_y = y + h / 2
        
        # Determine entity type
        is_plant = 'plant' in filename.lower()
        
        if is_plant:
            # Plant preview - draw stylized tree/plant shape
            self._draw_plant_preview(vertices, dna_data, center_x, center_y, min(w, h) * 0.4)
        else:
            # Animal preview - draw stylized animal shape
            self._draw_animal_preview(vertices, dna_data, center_x, center_y, min(w, h) * 0.4)
        
        # Render the preview vertices
        if vertices:
            vertex_data = np.array(vertices, dtype='f4')
            self.vbo.write(vertex_data.tobytes())
            self.font_texture.use(0)
            self.program['u_use_texture'].value = 0  # Solid colors
            self.program['u_screen_size'].value = (self.screen_width, self.screen_height)
            self.program['u_offset'].value = (0, 0)
            self.vao.render(moderngl.TRIANGLES, vertices=len(vertices) // 8)
    
    def _draw_plant_preview(self, vertices: List, dna: Dict, cx: float, cy: float, size: float):
        """Draw a high-detail 2D plant from DNA with actual parameters."""
        # Get the actual DNA dict (might be nested)
        plant_dna = dna.get('dna', dna)
        
        # Extract colors from DNA - plant colors are {r, g, b} dicts
        trunk_color = self._extract_rgb_dict(plant_dna.get('trunk_color'), (0.4, 0.25, 0.15))
        leaf_color = self._extract_rgb_dict(plant_dna.get('leaf_color'), (0.2, 0.6, 0.2))
        flower_color = self._extract_rgb_dict(plant_dna.get('flower_color'), (0.8, 0.4, 0.6))
        
        # Get actual DNA parameters for sizing
        height_gene = plant_dna.get('height_gene', {})
        width_gene = plant_dna.get('width_gene', {})
        plant_height = height_gene.get('value', 1.0) if isinstance(height_gene, dict) else 1.0
        plant_width = width_gene.get('value', 1.0) if isinstance(width_gene, dict) else 1.0
        
        leaf_density = plant_dna.get('leaf_density', 0.5)
        leaf_size_val = plant_dna.get('leaf_size', 0.5)
        branch_count = plant_dna.get('branch_count', 3)
        branch_angle = plant_dna.get('branch_angle', 0.5)
        
        # Get plant type for shape variation
        plant_type = plant_dna.get('plant_type', 'tree')
        
        # Animation for gentle sway
        anim_t = self._preview_rotation
        sway = math.sin(anim_t * 1.5) * size * 0.02
        
        # Scale based on DNA height/width
        trunk_h = size * 0.5 * min(plant_height, 2.0)
        trunk_w = size * 0.08 * min(plant_width, 2.0)
        
        # Base of trunk
        trunk_base_y = cy + size * 0.35
        trunk_top_y = trunk_base_y - trunk_h
        
        if plant_type in ('tree', 'conifer', 'palm'):
            # Draw detailed trunk with taper
            for i in range(5):
                seg_y = trunk_base_y - (trunk_h * i / 5)
                seg_w = trunk_w * (1 - i * 0.12)  # Taper
                seg_h = trunk_h / 5 + 2
                seg_sway = sway * (i / 5)
                self._add_quad(vertices, cx - seg_w/2 + seg_sway, seg_y - seg_h, seg_w, seg_h,
                              0, 0, 0, 0, 
                              trunk_color[0] * (0.9 + i * 0.02), 
                              trunk_color[1] * (0.9 + i * 0.02), 
                              trunk_color[2] * (0.9 + i * 0.02), 1.0)
            
            if plant_type == 'conifer':
                # Draw layered triangle canopy
                layers = int(3 + branch_count / 2)
                for layer in range(layers):
                    layer_y = trunk_top_y - layer * size * 0.12
                    layer_w = size * (0.15 + layer * 0.18) * plant_width
                    layer_h = size * 0.2
                    layer_sway = sway * (1 + layer * 0.2)
                    # Triangle as 3 vertices
                    shade = 0.85 + layer * 0.05
                    self._draw_triangle(vertices, 
                                       cx + layer_sway, layer_y - layer_h,  # top
                                       cx - layer_w/2 + layer_sway, layer_y,  # bottom left
                                       cx + layer_w/2 + layer_sway, layer_y,  # bottom right
                                       (leaf_color[0] * shade, leaf_color[1] * shade, leaf_color[2] * shade))
            elif plant_type == 'palm':
                # Draw fronds radiating from top
                frond_count = max(5, int(branch_count * 1.5))
                for i in range(frond_count):
                    angle = (i / frond_count) * math.pi + math.pi
                    frond_len = size * 0.5 * (0.8 + leaf_size_val * 0.4)
                    frond_sway = math.sin(anim_t * 2 + i * 0.5) * size * 0.05
                    frond_end_x = cx + math.cos(angle) * frond_len + frond_sway
                    frond_end_y = trunk_top_y + math.sin(angle) * frond_len * 0.6
                    self._draw_line(vertices, cx + sway, trunk_top_y, 
                                   frond_end_x, frond_end_y, size * 0.03, leaf_color)
            else:
                # Regular tree - layered circular canopy (high detail)
                canopy_layers = int(2 + leaf_density * 3)
                for layer in range(canopy_layers):
                    layer_size = size * (0.4 + layer * 0.15) * plant_width
                    layer_y = trunk_top_y + layer * size * 0.08
                    layer_sway = sway * (1.2 - layer * 0.1)
                    shade = 0.7 + layer * 0.1
                    self._draw_circle_shape(vertices, cx + layer_sway, layer_y, layer_size, 
                                           (leaf_color[0] * shade, leaf_color[1] * shade, leaf_color[2] * shade), 16)
                
                # Add some highlight dots for leaves
                for i in range(int(leaf_density * 8)):
                    angle = i * 0.8 + anim_t * 0.5
                    dist = size * 0.2 + (i % 3) * size * 0.1
                    lx = cx + math.cos(angle) * dist + sway
                    ly = trunk_top_y + math.sin(angle) * dist * 0.5
                    self._draw_circle_shape(vertices, lx, ly, size * 0.06, 
                                           (leaf_color[0] * 1.2, leaf_color[1] * 1.2, leaf_color[2]), 6)
        
        elif plant_type in ('bush', 'shrub'):
            # Low spreading bush
            trunk_h = size * 0.15
            self._add_quad(vertices, cx - trunk_w/2, trunk_base_y - trunk_h, trunk_w, trunk_h,
                          0, 0, 0, 0, *trunk_color, 1.0)
            # Multiple overlapping circles for foliage
            foliage_count = int(3 + leaf_density * 4)
            for i in range(foliage_count):
                angle = (i / foliage_count) * math.pi * 2
                dist = size * 0.15
                fx = cx + math.cos(angle) * dist + sway * (0.5 + i * 0.1)
                fy = trunk_base_y - trunk_h - size * 0.1 + math.sin(angle) * size * 0.1
                fsize = size * (0.25 + leaf_size_val * 0.15)
                shade = 0.8 + (i % 3) * 0.1
                self._draw_circle_shape(vertices, fx, fy, fsize, 
                                       (leaf_color[0] * shade, leaf_color[1] * shade, leaf_color[2] * shade), 12)
        
        elif plant_type in ('flower', 'succulent'):
            # Draw stem
            stem_h = size * 0.3
            self._add_quad(vertices, cx - trunk_w * 0.3 + sway, trunk_base_y - stem_h, trunk_w * 0.6, stem_h,
                          0, 0, 0, 0, *trunk_color, 1.0)
            # Draw petals radiating from center
            petal_count = max(5, int(branch_count * 1.5))
            flower_y = trunk_base_y - stem_h - size * 0.1
            for i in range(petal_count):
                angle = (i / petal_count) * math.pi * 2 + anim_t * 0.3
                petal_len = size * 0.2 * (1 + leaf_size_val * 0.5)
                px = cx + math.cos(angle) * petal_len + sway
                py = flower_y + math.sin(angle) * petal_len * 0.5
                petal_size = size * 0.12
                self._draw_circle_shape(vertices, px, py, petal_size, flower_color, 8)
            # Center
            self._draw_circle_shape(vertices, cx + sway, flower_y, size * 0.1, 
                                   (leaf_color[0] * 0.8, leaf_color[1] * 1.2, leaf_color[2] * 0.5), 10)
        
        elif plant_type in ('grass', 'fern'):
            # Multiple blades with curve
            blade_count = int(5 + leaf_density * 6)
            for i in range(blade_count):
                blade_x = cx + (i - blade_count/2) * size * 0.07
                blade_h = size * (0.4 + plant_height * 0.2) * (0.8 + (i % 3) * 0.1)
                blade_w = size * 0.025
                # Draw curved blade using multiple segments
                for j in range(4):
                    seg_y = trunk_base_y - blade_h * j / 4
                    seg_h = blade_h / 4 + 2
                    curve = math.sin(j * 0.5 + i * 0.3) * size * 0.03 + sway * (j / 4)
                    seg_w = blade_w * (1 - j * 0.15)
                    shade = 0.7 + j * 0.1
                    self._add_quad(vertices, blade_x - seg_w/2 + curve, seg_y - seg_h, seg_w, seg_h,
                                  0, 0, 0, 0, 
                                  leaf_color[0] * shade, leaf_color[1] * shade, leaf_color[2] * shade, 0.95)
        
        elif plant_type == 'mushroom':
            # Detailed mushroom with spots
            cap_color = self._extract_rgb_dict(plant_dna.get('cap_color'), flower_color)
            stem_h = size * 0.35
            stem_w = size * 0.1
            # Draw stem with slight bulge at base
            for i in range(3):
                seg_y = trunk_base_y - stem_h * i / 3
                bulge = 1 + (2 - i) * 0.1
                seg_w = stem_w * bulge
                self._add_quad(vertices, cx - seg_w/2, seg_y - stem_h/3, seg_w, stem_h/3 + 2,
                              0, 0, 0, 0, 0.95, 0.9, 0.85, 1.0)
            # Cap (dome shape using circles)
            cap_y = trunk_base_y - stem_h
            cap_size = size * 0.45 * plant_width
            self._draw_circle_shape(vertices, cx, cap_y - cap_size * 0.2, cap_size, cap_color, 16)
            # Spots on cap
            spot_count = int(leaf_density * 5)
            for i in range(spot_count):
                angle = i * 1.2 + 0.5
                dist = cap_size * 0.25
                sx = cx + math.cos(angle) * dist
                sy = cap_y - cap_size * 0.2 + math.sin(angle) * dist * 0.3
                spot_size = size * 0.04
                self._draw_circle_shape(vertices, sx, sy, spot_size, (0.95, 0.95, 0.9), 6)
        
        elif plant_type == 'cactus':
            # Tall cactus with arms
            cactus_h = size * 0.6 * plant_height
            cactus_w = size * 0.12 * plant_width
            # Main body
            self._add_quad(vertices, cx - cactus_w/2, trunk_base_y - cactus_h, cactus_w, cactus_h,
                          0, 0, 0, 0, *trunk_color, 1.0)
            # Arms
            if branch_count > 1:
                arm_h = cactus_h * 0.3
                arm_w = cactus_w * 0.8
                arm_y = trunk_base_y - cactus_h * 0.5
                # Left arm
                self._add_quad(vertices, cx - cactus_w/2 - arm_w, arm_y, arm_w, arm_h * 0.3,
                              0, 0, 0, 0, *trunk_color, 1.0)
                self._add_quad(vertices, cx - cactus_w/2 - arm_w, arm_y - arm_h, arm_w * 0.3, arm_h,
                              0, 0, 0, 0, *trunk_color, 1.0)
                # Right arm
                self._add_quad(vertices, cx + cactus_w/2, arm_y - arm_h * 0.5, arm_w, arm_h * 0.3,
                              0, 0, 0, 0, *trunk_color, 1.0)
                self._add_quad(vertices, cx + cactus_w/2 + arm_w * 0.7, arm_y - arm_h * 0.5 - arm_h, arm_w * 0.3, arm_h,
                              0, 0, 0, 0, *trunk_color, 1.0)
            # Flower on top
            if leaf_density > 0.3:
                self._draw_circle_shape(vertices, cx, trunk_base_y - cactus_h - size * 0.05, size * 0.08, flower_color, 8)
        
        else:
            # Default - medium detail tree
            self._add_quad(vertices, cx - trunk_w/2, trunk_base_y - trunk_h, trunk_w, trunk_h,
                          0, 0, 0, 0, *trunk_color, 1.0)
            self._draw_circle_shape(vertices, cx + sway, trunk_base_y - trunk_h - size * 0.15, size * 0.5, leaf_color, 12)
    
    def _draw_triangle(self, vertices: List, x1: float, y1: float, x2: float, y2: float, 
                       x3: float, y3: float, color: Tuple[float, float, float]):
        """Draw a triangle with the given vertices."""
        vertices.extend([x1, y1, 0, 0, *color, 1.0])
        vertices.extend([x2, y2, 0, 0, *color, 1.0])
        vertices.extend([x3, y3, 0, 0, *color, 1.0])
    
    def _draw_animal_preview(self, vertices: List, dna: Dict, cx: float, cy: float, size: float):
        """Draw a high-detail 2D animal from DNA using actual body segments."""
        # Get the actual DNA dict (might be nested)
        animal_dna = dna.get('dna', dna)
        
        # Get actual body segments and their colors/sizes
        body_segments = animal_dna.get('body_segments', [])
        limbs = animal_dna.get('limbs', [])
        features = animal_dna.get('features', [])
        base_scale = animal_dna.get('base_scale', 1.0)
        
        # Calculate scale factor for the preview area
        scale = size / max(base_scale, 0.5)
        
        # Get animal type for shape variations
        animal_type = animal_dna.get('animal_type', 'mammal')
        
        # Animation phase
        anim_t = self._preview_rotation
        anim_y = math.sin(anim_t * 2) * size * 0.03
        
        # Default colors if no segments
        primary = (0.5, 0.4, 0.3)
        secondary = (0.4, 0.3, 0.2)
        
        # --- DRAW BODY SEGMENTS (HIGH DETAIL) ---
        num_segs = len(body_segments)
        total_len = 1.0  # Default
        body_scale = scale * 0.4  # Default
        
        if num_segs > 0:
            # Calculate total body length for positioning
            total_len = 0.0
            for seg in body_segments:
                seg_size = seg.get('size', [0.3, 0.3, 0.3])
                total_len += seg_size[0] if isinstance(seg_size, (list, tuple)) else 0.3
            
            # Scale to fit preview
            body_scale = min(scale * 0.4, size * 0.9 / max(total_len, 1.0))
            
            # Starting position (left side of body)
            seg_x = cx - total_len * body_scale * 0.4
            
            for i, seg in enumerate(body_segments):
                seg_color = self._extract_array_color(seg.get('color'), primary)
                seg_size = seg.get('size', [0.3, 0.3, 0.3])
                if isinstance(seg_size, (list, tuple)) and len(seg_size) >= 3:
                    seg_w = seg_size[0] * body_scale * 1.5
                    seg_h = seg_size[1] * body_scale * 1.5
                else:
                    seg_w = 0.3 * body_scale
                    seg_h = 0.3 * body_scale
                
                # Animate segments with wave motion for snake/serpent types
                if animal_type in ('worm', 'serpent', 'snake', 'centipede', 'millipede'):
                    wave = math.sin(anim_t * 3 + i * 0.6) * size * 0.08
                    seg_y_offset = wave
                else:
                    seg_y_offset = anim_y
                
                # Draw segment as circle (higher detail)
                self._draw_circle_shape(vertices, seg_x + seg_w/2, cy + seg_y_offset, 
                                       max(seg_w, seg_h), seg_color, 16)
                
                # Store primary/secondary for features
                if i == 0:
                    primary = seg_color
                elif i == 1:
                    secondary = seg_color
                
                seg_x += seg_w * 0.7  # Overlap segments slightly
        else:
            # Fallback: single body circle
            self._draw_circle_shape(vertices, cx, cy + anim_y, size * 0.6, primary, 16)
        
        # --- DRAW LIMBS (HIGH DETAIL) ---
        if limbs:
            for limb_data in limbs:
                limb_color = self._extract_array_color(limb_data.get('color'), secondary)
                limb_segs = limb_data.get('segments', [])
                limb_type = limb_data.get('limb_type', 'leg')
                
                # Calculate limb position based on attachment
                attach = limb_data.get('attachment_point', [0, 0, 0])
                if isinstance(attach, (list, tuple)) and len(attach) >= 2:
                    limb_x = cx + attach[0] * scale * 0.3
                    limb_y = cy + attach[1] * scale * 0.3 + anim_y
                else:
                    limb_x = cx
                    limb_y = cy + anim_y
                
                # Draw limb segments
                for j, lseg in enumerate(limb_segs):
                    lseg_len = lseg.get('length', 0.2) * scale * 0.5
                    lseg_thick = lseg.get('thickness', 0.05) * scale * 0.4
                    
                    # Animate limbs
                    limb_anim = math.sin(anim_t * 6 + j * 1.5) * size * 0.04
                    
                    # Draw as rectangle (leg segment)
                    self._add_quad(vertices, limb_x - lseg_thick/2, limb_y + limb_anim, 
                                  lseg_thick, lseg_len, 0, 0, 0, 0, *limb_color, 0.9)
                    limb_y += lseg_len * 0.8
        else:
            # Draw type-based limbs if no explicit limb data
            limb_pairs = animal_dna.get('limb_pairs', 0)
            if animal_type in ('bird', 'bat', 'pterosaur', 'moth'):
                # Wings
                wing_w = size * 0.6
                wing_h = size * 0.15
                wing_anim = math.sin(anim_t * 8) * size * 0.15
                # Left wing
                self._add_quad(vertices, cx - size * 0.3 - wing_w, cy - wing_h/2 + wing_anim, 
                              wing_w, wing_h, 0, 0, 0, 0, *secondary, 0.85)
                # Right wing
                self._add_quad(vertices, cx + size * 0.3, cy - wing_h/2 + wing_anim, 
                              wing_w, wing_h, 0, 0, 0, 0, *secondary, 0.85)
            elif animal_type in ('spider', 'scorpion', 'beetle', 'mantis'):
                # 8 legs for arachnids
                leg_count = 8 if animal_type == 'spider' else 6
                for i in range(leg_count):
                    angle = (i / leg_count) * math.pi - math.pi/2
                    leg_len = size * 0.4
                    leg_w = size * 0.04
                    leg_anim = math.sin(anim_t * 6 + i * 0.8) * size * 0.06
                    leg_x = cx + math.cos(angle) * size * 0.2
                    leg_y = cy + math.sin(angle) * size * 0.1 + leg_anim
                    leg_end_x = leg_x + math.cos(angle) * leg_len
                    leg_end_y = leg_y + math.sin(angle) * leg_len * 0.5 + size * 0.2
                    # Draw as line (two triangles)
                    self._draw_line(vertices, leg_x, leg_y, leg_end_x, leg_end_y, leg_w, secondary)
            elif limb_pairs > 0 and animal_type not in ('worm', 'serpent', 'snake', 'fish', 'jellyfish'):
                # Default legs for quadrupeds etc
                leg_w = size * 0.08
                leg_h = size * 0.25
                positions = [(-0.25, 0), (0.2, 0)] if limb_pairs <= 2 else [(-0.3, 0), (-0.1, 0), (0.1, 0), (0.25, 0)]
                for i, (lx, ly) in enumerate(positions[:limb_pairs * 2]):
                    leg_anim = math.sin(anim_t * 5 + i * 1.5) * size * 0.06
                    self._add_quad(vertices, cx + lx * size - leg_w/2, cy + size * 0.15 + leg_anim, 
                                  leg_w, leg_h, 0, 0, 0, 0, *secondary, 0.9)
        
        # --- DRAW FEATURES (EYES, HORNS, ETC) ---
        for feat in features:
            feat_type = feat.get('feature_type', '')
            feat_color = self._extract_array_color(feat.get('color'), (0.1, 0.1, 0.1))
            feat_size = feat.get('size', 0.1) * scale * 0.2
            feat_count = feat.get('count', 1)
            feat_pos = feat.get('position', [0, 0, 0])
            
            if feat_type == 'eye':
                # Draw eyes on the head (first segment)
                eye_spacing = size * 0.08
                for e in range(min(feat_count, 8)):
                    eye_x = cx - total_len * body_scale * 0.35 + (e - feat_count/2) * eye_spacing * 0.5
                    eye_y = cy - size * 0.05 + anim_y
                    # White of eye
                    self._draw_circle_shape(vertices, eye_x, eye_y, feat_size * 1.5, (0.95, 0.95, 0.95), 12)
                    # Pupil
                    self._draw_circle_shape(vertices, eye_x, eye_y, feat_size * 0.8, feat_color, 8)
            elif feat_type == 'horn' or animal_dna.get('has_horns', False):
                horn_len = animal_dna.get('horn_length', 0.3) * scale * 0.4
                horn_w = size * 0.04
                horn_x = cx - total_len * body_scale * 0.4
                self._add_quad(vertices, horn_x - horn_w, cy - size * 0.1 - horn_len + anim_y, 
                              horn_w, horn_len, 0, 0, 0, 0, 0.8, 0.75, 0.6, 1.0)
            elif feat_type == 'antenna':
                ant_len = size * 0.2
                ant_w = size * 0.02
                ant_x = cx - total_len * body_scale * 0.35
                ant_anim = math.sin(anim_t * 4) * size * 0.03
                self._add_quad(vertices, ant_x - ant_w/2 - size * 0.05, cy - size * 0.1 - ant_len + ant_anim + anim_y, 
                              ant_w, ant_len, 0, 0, 0, 0, *feat_color, 0.9)
                self._add_quad(vertices, ant_x - ant_w/2 + size * 0.05, cy - size * 0.1 - ant_len - ant_anim + anim_y, 
                              ant_w, ant_len, 0, 0, 0, 0, *feat_color, 0.9)
        
        # Draw default eyes if no eye feature found
        if not any(f.get('feature_type') == 'eye' for f in features):
            eye_size = size * 0.05
            head_x = cx - (total_len * body_scale * 0.35 if num_segs > 0 else size * 0.25)
            self._draw_circle_shape(vertices, head_x, cy - size * 0.03 + anim_y, eye_size, (0.1, 0.1, 0.1), 8)
    
    def _draw_line(self, vertices: List, x1: float, y1: float, x2: float, y2: float, 
                   width: float, color: Tuple[float, float, float]):
        """Draw a line as a quad between two points."""
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length < 0.001:
            return
        # Perpendicular unit vector
        px = -dy / length * width / 2
        py = dx / length * width / 2
        # Four corners
        corners = [
            (x1 + px, y1 + py),
            (x1 - px, y1 - py),
            (x2 - px, y2 - py),
            (x2 + px, y2 + py),
        ]
        # Two triangles
        for i in [0, 1, 2, 0, 2, 3]:
            vertices.extend([corners[i][0], corners[i][1], 0, 0, *color, 0.9])
    
    def _draw_circle_shape(self, vertices: List, cx: float, cy: float, size: float, 
                          color: Tuple[float, float, float], segments: int = 8):
        """Draw an approximate circle using triangles."""
        for i in range(segments):
            angle1 = (i / segments) * 2 * math.pi
            angle2 = ((i + 1) / segments) * 2 * math.pi
            
            x1 = cx + math.cos(angle1) * size * 0.5
            y1 = cy + math.sin(angle1) * size * 0.5
            x2 = cx + math.cos(angle2) * size * 0.5
            y2 = cy + math.sin(angle2) * size * 0.5
            
            # Triangle from center to edge
            vertices.extend([cx, cy, 0, 0, *color, 1.0])
            vertices.extend([x1, y1, 0, 0, *color, 1.0])
            vertices.extend([x2, y2, 0, 0, *color, 1.0])
    
    def _extract_rgb_dict(self, color_dict: Optional[Dict], 
                         default: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Extract color from {r, g, b} dict format (used by plants)."""
        if color_dict and isinstance(color_dict, dict):
            try:
                r = float(color_dict.get('r', default[0]))
                g = float(color_dict.get('g', default[1]))
                b = float(color_dict.get('b', default[2]))
                return (r, g, b)
            except:
                pass
        return default
    
    def _extract_array_color(self, color_array: Optional[List], 
                            default: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Extract color from [r, g, b] array format (used by animals)."""
        if color_array and isinstance(color_array, (list, tuple)) and len(color_array) >= 3:
            try:
                return (float(color_array[0]), float(color_array[1]), float(color_array[2]))
            except:
                pass
        return default
    
    def cleanup(self):
        """Release GPU resources."""
        # Release cached image textures
        for texture in self._image_cache.values():
            texture.release()
        self._image_cache.clear()
        
        self.vbo.release()
        self.vao.release()
        self.program.release()
        self.font_texture.release()
