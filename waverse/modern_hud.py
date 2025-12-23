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
        self._pending_preview: Optional[Dict] = None  # Store pending PNG preview info
        
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
    
    def set_status(self, text: str):
        """Set status text."""
        self.status_text = text
    
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
        """Draw status text at bottom-left with shadow."""
        if not self.status_text:
            return
        
        scale = 2.0
        text_width = len(self.status_text) * 8 * scale
        x = 15  # Bottom-left
        y = self.screen_height - 40  # Near bottom
        
        # Background bar
        padding = 8
        self._add_quad(vertices, x - padding, y - padding/2, 
                      text_width + padding * 2, 8 * scale + padding,
                      0, 0, 0, 0, 0.0, 0.0, 0.0, 0.6)
        
        # Text with shadow
        self._draw_text_with_shadow(vertices, self.status_text, x, y, scale, 
                                    1.0, 1.0, 0.5, 1.0, shadow_offset=2)
    
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
        
        # Sync status text
        if hasattr(camera, 'status_message') and camera.status_message:
            self.status_text = camera.status_message
        
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
        
        # After main HUD, draw PNG preview if pending (on Log tab)
        if self._pending_preview and PIL_AVAILABLE:
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
            # No PNG available - draw placeholder text
            # Build vertices for a small text message
            vertices = []
            center_x = x + w / 2
            center_y = y + h / 2
            self._draw_text(vertices, "NO PREVIEW", center_x - 40, center_y - 10, 1.5, 0.4, 0.4, 0.4, 1.0)
            self._draw_text(vertices, "PNG NOT FOUND", center_x - 55, center_y + 8, 1.3, 0.3, 0.3, 0.3, 1.0)
            
            if vertices:
                vertex_data = np.array(vertices, dtype='f4')
                self.vbo.write(vertex_data.tobytes())
                self.font_texture.use(0)
                self.program['u_use_texture'].value = 1
                self.vao.render(moderngl.TRIANGLES, vertices=len(vertices) // 8)
    
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
