"""
Microscope View Renderer

Full-screen 2D view into the microscopic world.
Renders cells, organelles, and the environment.
"""

import math
import numpy as np
import moderngl
from typing import List, Tuple, Optional, Dict
import time

from .micro_life import (
    MicroscopeWorld, Microorganism, MicroDNA, OrganelleType,
    MICRO_TEMPLATES
)


# =============================================================================
# SHADERS
# =============================================================================

MICRO_VERTEX_SHADER = """
#version 330

uniform mat4 u_projection;
uniform vec2 u_offset;
uniform float u_scale;

in vec2 in_position;
in vec4 in_color;

out vec4 v_color;

void main() {
    vec2 world_pos = in_position * u_scale + u_offset;
    gl_Position = u_projection * vec4(world_pos, 0.0, 1.0);
    v_color = in_color;
}
"""

MICRO_FRAGMENT_SHADER = """
#version 330

in vec4 v_color;
out vec4 fragColor;

void main() {
    fragColor = v_color;
}
"""

# Glow shader for bioluminescent organisms
GLOW_FRAGMENT_SHADER = """
#version 330

in vec4 v_color;
out vec4 fragColor;

void main() {
    // Simple glow effect
    float alpha = v_color.a;
    vec3 glow = v_color.rgb * 1.5;
    fragColor = vec4(glow, alpha * 0.6);
}
"""


# =============================================================================
# MICROSCOPE VIEW RENDERER
# =============================================================================

class MicroscopeView:
    """
    Full-screen 2D microscope view renderer.
    
    Shows the microscopic world with cells, organelles, and environment.
    """
    
    def __init__(self, ctx: moderngl.Context, width: int = 1280, height: int = 720):
        self.ctx = ctx
        self.width = width
        self.height = height
        
        # World simulation
        self.world = MicroscopeWorld(width=50.0, height=50.0)
        
        # View state
        self.view_x = 25.0  # Center of view
        self.view_y = 25.0
        self.zoom = 0.5     # Zoom level (higher = more zoomed in) - start zoomed out
        
        # Animation
        self.time = 0.0
        
        # Selected organism for DNA sampling
        self.selected: Optional[Microorganism] = None
        
        # Create shader program
        self.program = ctx.program(
            vertex_shader=MICRO_VERTEX_SHADER,
            fragment_shader=MICRO_FRAGMENT_SHADER,
        )
        
        # Dynamic vertex buffer - start with 4MB, will grow if needed
        self._vbo_size = 4 * 1024 * 1024
        self.vbo = ctx.buffer(reserve=self._vbo_size)
        self.vao = ctx.vertex_array(
            self.program,
            [(self.vbo, '2f 4f', 'in_position', 'in_color')],
        )
        
        # Background color (dark blue-ish for water)
        self.bg_color = (0.02, 0.04, 0.08)
        
        # Set up projection matrix (2D orthographic)
        self._update_projection()
    
    def _update_projection(self):
        """Update the projection matrix based on view."""
        # Calculate view bounds
        aspect = self.width / self.height
        view_width = 10.0 / self.zoom
        view_height = view_width / aspect
        
        left = -view_width / 2
        right = view_width / 2
        bottom = -view_height / 2
        top = view_height / 2
        
        # Orthographic projection matrix
        proj = np.array([
            [2.0 / (right - left), 0, 0, -(right + left) / (right - left)],
            [0, 2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ], dtype='f4')
        
        self.program['u_projection'].write(proj.tobytes())
    
    def resize(self, width: int, height: int):
        """Handle window resize."""
        self.width = width
        self.height = height
        self._update_projection()
    
    def update(self, dt: float):
        """Update the microscope simulation."""
        self.time += dt
        self.world.update(dt)
    
    def pan(self, dx: float, dy: float):
        """Pan the view."""
        self.view_x += dx / self.zoom
        self.view_y += dy / self.zoom
    
    def zoom_in(self, factor: float = 1.2):
        """Zoom in."""
        self.zoom *= factor
        self._update_projection()
    
    def zoom_out(self, factor: float = 1.2):
        """Zoom out."""
        self.zoom /= factor
        self.zoom = max(0.5, self.zoom)  # Min zoom
        self._update_projection()
    
    def screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        # Normalize to -1..1
        nx = (screen_x / self.width) * 2 - 1
        ny = (screen_y / self.height) * 2 - 1
        ny = -ny  # Flip Y
        
        # Scale by view
        aspect = self.width / self.height
        view_width = 10.0 / self.zoom
        view_height = view_width / aspect
        
        world_x = self.view_x + nx * view_width / 2
        world_y = self.view_y + ny * view_height / 2
        
        return world_x, world_y
    
    def select_at(self, screen_x: float, screen_y: float) -> Optional[Microorganism]:
        """Select organism at screen position."""
        world_x, world_y = self.screen_to_world(screen_x, screen_y)
        
        # Find closest organism
        closest = None
        closest_dist = float('inf')
        
        for org in self.world.organisms:
            dx = org.x - world_x
            dy = org.y - world_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < org.dna.base_size and dist < closest_dist:
                closest = org
                closest_dist = dist
        
        self.selected = closest
        return closest
    
    def render(self):
        """Render the microscope view."""
        # Clear background
        self.ctx.clear(*self.bg_color, 1.0)
        
        # Update uniforms
        self.program['u_offset'].value = (-self.view_x, -self.view_y)
        self.program['u_scale'].value = 1.0
        self._update_projection()
        
        # Build vertex data
        vertices = []
        
        # Render background particles/debris (visual only)
        self._add_background_particles(vertices)
        
        # Render organisms
        for org in self.world.organisms:
            self._add_organism_vertices(vertices, org)
        
        # Render selection highlight
        if self.selected:
            self._add_selection_highlight(vertices, self.selected)
        
        # Upload and render
        if vertices:
            data = np.array(vertices, dtype='f4').tobytes()
            
            # Grow buffer if needed
            if len(data) > self._vbo_size:
                self.vbo.release()
                self._vbo_size = len(data) * 2  # Double for headroom
                self.vbo = self.ctx.buffer(reserve=self._vbo_size)
                self.vao.release()
                self.vao = self.ctx.vertex_array(
                    self.program,
                    [(self.vbo, '2f 4f', 'in_position', 'in_color')],
                )
            
            self.vbo.write(data)
            self.vao.render(moderngl.TRIANGLES, vertices=len(vertices) // 6)
    
    def _add_background_particles(self, vertices: List):
        """Add floating particles in the background."""
        # Small dots floating around
        num_particles = 50
        for i in range(num_particles):
            # Deterministic but animated positions
            px = (math.sin(i * 1.23 + self.time * 0.1) * 0.5 + 0.5) * self.world.width
            py = (math.cos(i * 2.34 + self.time * 0.08) * 0.5 + 0.5) * self.world.height
            
            # Small triangle for each particle
            size = 0.05 + (i % 5) * 0.01
            alpha = 0.1 + (i % 3) * 0.05
            color = (0.3, 0.4, 0.5, alpha)
            
            self._add_circle(vertices, px, py, size, 6, color)
    
    def _add_organism_vertices(self, vertices: List, org: Microorganism):
        """Add vertices for an organism."""
        x, y = org.x, org.y
        size = org.dna.base_size
        membrane = org.dna.membrane
        
        # Cell membrane
        membrane_color = (*membrane.color, 1.0 - membrane.transparency)
        
        # Draw membrane as polygon
        self._add_polygon(vertices, x, y, org.membrane_points, membrane_color)
        
        # Cell wall if present
        if membrane.has_wall:
            wall_color = (*membrane.wall_color, 0.8)
            self._add_ring(vertices, x, y, org.membrane_points, 
                          membrane.thickness * size * 2, wall_color)
        
        # Organelles
        for ox, oy, org_gene in org.organelle_positions:
            self._add_organelle(vertices, x + ox, y + oy, org_gene, org)
        
        # Glow effect for bioluminescent organisms
        if org.dna.glow > 0:
            glow_color = (*org.dna.glow_color, org.dna.glow * 0.3)
            self._add_circle(vertices, x, y, size * 1.5, 16, glow_color)
    
    def _add_organelle(self, vertices: List, x: float, y: float, 
                       org_gene, parent_org: Microorganism):
        """Add vertices for a single organelle."""
        size = org_gene.size * parent_org.dna.base_size
        color = (*org_gene.color, 0.9)
        org_type = org_gene.organelle_type
        
        if org_type == OrganelleType.NUCLEUS:
            # Large central circle with inner detail
            self._add_circle(vertices, x, y, size, 16, color)
            # Nucleolus
            inner_color = (color[0] * 0.7, color[1] * 0.7, color[2] * 0.8, 0.9)
            self._add_circle(vertices, x, y, size * 0.4, 8, inner_color)
        
        elif org_type == OrganelleType.MITOCHONDRIA:
            # Elongated oval with inner folds
            self._add_oval(vertices, x, y, size * 1.5, size * 0.8, 12, color)
            # Inner cristae (simplified)
            inner_color = (color[0] * 0.6, color[1] * 0.6, color[2] * 0.6, 0.7)
            self._add_oval(vertices, x, y, size * 0.8, size * 0.4, 8, inner_color)
        
        elif org_type == OrganelleType.CHLOROPLAST:
            # Green oval
            self._add_oval(vertices, x, y, size * 1.3, size * 0.9, 12, color)
        
        elif org_type == OrganelleType.RIBOSOME:
            # Tiny dot
            self._add_circle(vertices, x, y, size, 6, color)
        
        elif org_type == OrganelleType.VACUOLE:
            # Large transparent circle
            vac_color = (*org_gene.color, 0.4)
            self._add_circle(vertices, x, y, size, 16, vac_color)
        
        elif org_type == OrganelleType.FLAGELLUM:
            # Wavy line extending from cell
            self._add_flagellum(vertices, x, y, size, parent_org, color)
        
        elif org_type == OrganelleType.CILIA:
            # Short hair
            angle = math.atan2(y, x)  # Direction from center
            self._add_line(vertices, x, y, x + math.cos(angle) * size,
                          y + math.sin(angle) * size, size * 0.1, color)
        
        elif org_type == OrganelleType.PSEUDOPOD:
            # Blob extension (already part of membrane for amoeba)
            pass
        
        elif org_type == OrganelleType.GOLGI:
            # Stack of curves
            for i in range(3):
                offset = (i - 1) * size * 0.4
                self._add_oval(vertices, x, y + offset, size * 1.2, size * 0.3, 10, color)
        
        elif org_type == OrganelleType.ER:
            # Network of tubes (simplified as blob)
            self._add_circle(vertices, x, y, size, 8, (*org_gene.color, 0.5))
        
        elif org_type == OrganelleType.LYSOSOME:
            # Small circle with darker interior
            self._add_circle(vertices, x, y, size, 10, color)
        
        else:
            # Generic circle
            self._add_circle(vertices, x, y, size, 10, color)
    
    def _add_flagellum(self, vertices: List, x: float, y: float, 
                       length: float, org: Microorganism, color: Tuple):
        """Add a wavy flagellum."""
        segments = 8
        angle = math.atan2(y, x)  # Direction from center outward
        
        wave_freq = 3.0
        wave_amp = length * 0.3
        phase = org.phase
        
        prev_x, prev_y = x, y
        for i in range(1, segments + 1):
            t = i / segments
            dist = length * t * 2
            wave = math.sin(t * wave_freq * math.pi + phase) * wave_amp * t
            
            px = x + math.cos(angle) * dist + math.cos(angle + math.pi/2) * wave
            py = y + math.sin(angle) * dist + math.sin(angle + math.pi/2) * wave
            
            thickness = length * 0.1 * (1 - t * 0.5)
            self._add_line(vertices, prev_x, prev_y, px, py, thickness, color)
            prev_x, prev_y = px, py
    
    def _add_selection_highlight(self, vertices: List, org: Microorganism):
        """Add highlight ring around selected organism."""
        x, y = org.x, org.y
        size = org.dna.base_size * 1.3
        
        # Pulsing highlight
        pulse = 0.5 + 0.5 * math.sin(self.time * 4)
        color = (1.0, 1.0, 0.5, 0.3 + pulse * 0.3)
        
        # Ring
        self._add_ring_simple(vertices, x, y, size, size * 1.1, 24, color)
    
    # === PRIMITIVE HELPERS ===
    
    def _add_circle(self, vertices: List, cx: float, cy: float, 
                    radius: float, segments: int, color: Tuple):
        """Add a filled circle as triangles."""
        for i in range(segments):
            angle1 = (i / segments) * 2 * math.pi
            angle2 = ((i + 1) / segments) * 2 * math.pi
            
            # Triangle from center to edge
            vertices.extend([cx, cy, *color])
            vertices.extend([cx + math.cos(angle1) * radius, 
                           cy + math.sin(angle1) * radius, *color])
            vertices.extend([cx + math.cos(angle2) * radius,
                           cy + math.sin(angle2) * radius, *color])
    
    def _add_oval(self, vertices: List, cx: float, cy: float,
                  rx: float, ry: float, segments: int, color: Tuple):
        """Add a filled oval."""
        for i in range(segments):
            angle1 = (i / segments) * 2 * math.pi
            angle2 = ((i + 1) / segments) * 2 * math.pi
            
            vertices.extend([cx, cy, *color])
            vertices.extend([cx + math.cos(angle1) * rx,
                           cy + math.sin(angle1) * ry, *color])
            vertices.extend([cx + math.cos(angle2) * rx,
                           cy + math.sin(angle2) * ry, *color])
    
    def _add_polygon(self, vertices: List, cx: float, cy: float,
                     points: List[Tuple[float, float]], color: Tuple):
        """Add a filled polygon from points."""
        if len(points) < 3:
            return
        
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            
            # Triangle from center
            vertices.extend([cx, cy, *color])
            vertices.extend([cx + p1[0], cy + p1[1], *color])
            vertices.extend([cx + p2[0], cy + p2[1], *color])
    
    def _add_ring(self, vertices: List, cx: float, cy: float,
                  points: List[Tuple[float, float]], 
                  thickness: float, color: Tuple):
        """Add a ring around points."""
        if len(points) < 3:
            return
        
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            
            # Outer edge
            angle1 = math.atan2(p1[1], p1[0])
            angle2 = math.atan2(p2[1], p2[0])
            
            r1 = math.sqrt(p1[0]**2 + p1[1]**2)
            r2 = math.sqrt(p2[0]**2 + p2[1]**2)
            
            # Inner and outer points
            inner1 = (cx + p1[0], cy + p1[1])
            inner2 = (cx + p2[0], cy + p2[1])
            outer1 = (cx + math.cos(angle1) * (r1 + thickness),
                     cy + math.sin(angle1) * (r1 + thickness))
            outer2 = (cx + math.cos(angle2) * (r2 + thickness),
                     cy + math.sin(angle2) * (r2 + thickness))
            
            # Two triangles for quad
            vertices.extend([*inner1, *color])
            vertices.extend([*outer1, *color])
            vertices.extend([*inner2, *color])
            
            vertices.extend([*inner2, *color])
            vertices.extend([*outer1, *color])
            vertices.extend([*outer2, *color])
    
    def _add_ring_simple(self, vertices: List, cx: float, cy: float,
                        inner_r: float, outer_r: float, 
                        segments: int, color: Tuple):
        """Add a simple circular ring."""
        for i in range(segments):
            angle1 = (i / segments) * 2 * math.pi
            angle2 = ((i + 1) / segments) * 2 * math.pi
            
            inner1 = (cx + math.cos(angle1) * inner_r, cy + math.sin(angle1) * inner_r)
            inner2 = (cx + math.cos(angle2) * inner_r, cy + math.sin(angle2) * inner_r)
            outer1 = (cx + math.cos(angle1) * outer_r, cy + math.sin(angle1) * outer_r)
            outer2 = (cx + math.cos(angle2) * outer_r, cy + math.sin(angle2) * outer_r)
            
            # Two triangles
            vertices.extend([*inner1, *color])
            vertices.extend([*outer1, *color])
            vertices.extend([*inner2, *color])
            
            vertices.extend([*inner2, *color])
            vertices.extend([*outer1, *color])
            vertices.extend([*outer2, *color])
    
    def _add_line(self, vertices: List, x1: float, y1: float,
                  x2: float, y2: float, thickness: float, color: Tuple):
        """Add a line as a quad."""
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        if length < 0.001:
            return
        
        # Perpendicular direction
        nx = -dy / length * thickness / 2
        ny = dx / length * thickness / 2
        
        # Four corners
        p1 = (x1 + nx, y1 + ny)
        p2 = (x1 - nx, y1 - ny)
        p3 = (x2 - nx, y2 - ny)
        p4 = (x2 + nx, y2 + ny)
        
        # Two triangles
        vertices.extend([*p1, *color])
        vertices.extend([*p2, *color])
        vertices.extend([*p3, *color])
        
        vertices.extend([*p1, *color])
        vertices.extend([*p3, *color])
        vertices.extend([*p4, *color])
    
    def cleanup(self):
        """Release GPU resources."""
        self.vbo.release()
        self.vao.release()
        self.program.release()


# =============================================================================
# STANDALONE TEST
# =============================================================================

def run_microscope_demo():
    """Run standalone microscope demo."""
    import pygame
    
    pygame.init()
    pygame.joystick.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    
    screen = pygame.display.set_mode((1280, 720), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Microscope View - waverse")
    
    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    
    view = MicroscopeView(ctx, 1280, 720)
    
    # Add some specific organism types
    for template_name in ["bacteria", "amoeba", "algae", "paramecium"]:
        for _ in range(5):
            dna = MICRO_TEMPLATES[template_name]()
            x = view.world.rng.random() * view.world.width
            y = view.world.rng.random() * view.world.height
            view.world.add_organism(x, y, dna)
    
    clock = pygame.time.Clock()
    running = True
    paused = False
    
    # Initialize joysticks
    joysticks = []
    for i in range(pygame.joystick.get_count()):
        js = pygame.joystick.Joystick(i)
        js.init()
        joysticks.append(js)
        print(f"Controller {i}: {js.get_name()}")
    
    print("Controls:")
    print("  WASD / Arrow keys / Left stick: Pan view")
    print("  Mouse wheel / L2/R2 triggers: Zoom")
    print("  Space / Start button: Pause/unpause")
    print("  Click / A button: Select organism")
    print("  R / Y button: Add random organism")
    print("  ESC / B button: Exit")
    
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Paused" if paused else "Running")
                elif event.key == pygame.K_r:
                    # Add random organism at center
                    dna = MicroDNA.create_random()
                    view.world.add_organism(view.view_x, view.view_y, dna)
                    print(f"Added organism. Total: {len(view.world.organisms)}")
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    org = view.select_at(*event.pos)
                    if org:
                        print(f"Selected: species={org.dna.species_id}, energy={org.energy:.1f}")
                elif event.button == 4:  # Scroll up
                    view.zoom_in()
                elif event.button == 5:  # Scroll down
                    view.zoom_out()
            elif event.type == pygame.JOYBUTTONDOWN:
                # Controller button handling
                if event.button == 0:  # A button - select at center
                    org = view.select_at(640, 360)  # Center of screen
                    if org:
                        print(f"Selected: species={org.dna.species_id}, energy={org.energy:.1f}")
                elif event.button == 1:  # B button - exit
                    running = False
                elif event.button == 3:  # Y button - add random
                    dna = MicroDNA.create_random()
                    view.world.add_organism(view.view_x, view.view_y, dna)
                    print(f"Added organism. Total: {len(view.world.organisms)}")
                elif event.button == 9:  # Start button - pause
                    paused = not paused
                    print("Paused" if paused else "Running")
        
        # Pan with keys
        keys = pygame.key.get_pressed()
        pan_speed = 5.0 / view.zoom
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            view.pan(-pan_speed, 0)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            view.pan(pan_speed, 0)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            view.pan(0, pan_speed)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            view.pan(0, -pan_speed)
        
        # Controller input
        for js in joysticks:
            # Left stick for panning
            DEADZONE = 0.15
            lx = js.get_axis(0)  # Left stick X
            ly = js.get_axis(1)  # Left stick Y
            
            if abs(lx) > DEADZONE:
                view.pan(lx * pan_speed, 0)
            if abs(ly) > DEADZONE:
                view.pan(0, -ly * pan_speed)  # Inverted Y
            
            # D-pad for panning
            try:
                hat = js.get_hat(0)
                if hat[0] != 0:
                    view.pan(hat[0] * pan_speed * 0.5, 0)
                if hat[1] != 0:
                    view.pan(0, hat[1] * pan_speed * 0.5)
            except:
                pass
            
            # Triggers for zoom (L2 = axis 4, R2 = axis 5 on most controllers)
            try:
                l2 = js.get_axis(4)  # L2 trigger (-1 to 1)
                r2 = js.get_axis(5)  # R2 trigger (-1 to 1)
                
                # Normalize triggers (often -1 = released, 1 = pressed)
                l2_pressed = (l2 + 1) / 2  # 0 to 1
                r2_pressed = (r2 + 1) / 2  # 0 to 1
                
                if r2_pressed > 0.1:
                    view.zoom_in(1.0 + r2_pressed * 0.05)
                if l2_pressed > 0.1:
                    view.zoom_out(1.0 + l2_pressed * 0.05)
            except:
                pass
            
            # Shoulder buttons for zoom (L1/R1)
            try:
                if js.get_button(4):  # L1
                    view.zoom_out(1.02)
                if js.get_button(5):  # R1
                    view.zoom_in(1.02)
            except:
                pass
        
        # Update
        if not paused:
            view.update(dt)
        
        # Render
        view.render()
        
        pygame.display.flip()
    
    view.cleanup()
    pygame.quit()


if __name__ == "__main__":
    run_microscope_demo()

