"""
Flora Rendering System - Renders plants from DNA with LOD support.

This module handles:
- Rendering plants from PlantDNA
- Level of Detail (LOD) for distant plants
- Display list caching for performance
- Multi-segment trunk rendering
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict
from OpenGL.GL import *

from .dna import PlantDNA, PlantType, DNAPool, SegmentGene


@dataclass
class PlantInstance:
    """A specific plant instance at a world location."""
    x: float
    y: float  # Ground height (already scaled)
    z: float
    dna: PlantDNA
    scale: float = 1.0
    rotation: float = 0.0  # Y-axis rotation in degrees


class PlantRenderer:
    """Renders individual plants from DNA."""
    
    @staticmethod
    def draw_full(plant: PlantInstance):
        """Draw plant at full detail using its DNA."""
        glPushMatrix()
        glTranslatef(plant.x, plant.y, plant.z)
        glRotatef(plant.rotation, 0, 1, 0)
        glScalef(plant.scale, plant.scale, plant.scale)
        
        dna = plant.dna
        
        if dna.plant_type == PlantType.GRASS:
            PlantRenderer._draw_grass(dna)
        elif dna.plant_type == PlantType.FERN:
            PlantRenderer._draw_fern(dna)
        elif dna.plant_type in (PlantType.BUSH, PlantType.SHRUB):
            PlantRenderer._draw_bush(dna)
        elif dna.plant_type == PlantType.MUSHROOM:
            PlantRenderer._draw_mushroom(dna)
        elif dna.plant_type == PlantType.CACTUS:
            PlantRenderer._draw_cactus(dna)
        else:
            # Trees (TREE, TALL_TREE, PALM, ALIEN)
            PlantRenderer._draw_tree(dna)
        
        glPopMatrix()
    
    @staticmethod
    def draw_simple(plant: PlantInstance):
        """Draw simplified plant for medium distance."""
        glPushMatrix()
        glTranslatef(plant.x, plant.y, plant.z)
        glScalef(plant.scale, plant.scale, plant.scale)
        
        dna = plant.dna
        height = dna.height_gene.value
        
        if dna.plant_type == PlantType.GRASS:
            # Simple grass - crossed quads
            glColor3f(*dna.leaf_color.rgb)
            glBegin(GL_TRIANGLES)
            glVertex3f(-0.08, 0, 0)
            glVertex3f(0.08, 0, 0)
            glVertex3f(0, height, 0)
            glVertex3f(0, 0, -0.08)
            glVertex3f(0, 0, 0.08)
            glVertex3f(0, height * 0.9, 0)
            glEnd()
        elif dna.plant_type in (PlantType.FERN, PlantType.BUSH, PlantType.SHRUB):
            # Simple bush - triangle
            glColor3f(*dna.leaf_color.rgb)
            spread = height * 0.5
            glBegin(GL_TRIANGLES)
            glVertex3f(0, height, 0)
            glVertex3f(-spread, 0, -spread * 0.5)
            glVertex3f(spread, 0, spread * 0.5)
            glVertex3f(0, height * 0.9, 0)
            glVertex3f(-spread * 0.5, 0, spread)
            glVertex3f(spread * 0.5, 0, -spread)
            glEnd()
        elif dna.plant_type == PlantType.MUSHROOM:
            # Simple mushroom - stem + cap
            glColor3f(*dna.trunk_color.rgb)
            w = dna.width_gene.value * 0.5
            glBegin(GL_QUADS)
            glVertex3f(-w, 0, 0)
            glVertex3f(w, 0, 0)
            glVertex3f(w, height * 0.7, 0)
            glVertex3f(-w, height * 0.7, 0)
            glEnd()
            glColor3f(*dna.leaf_color.rgb)
            cap = height * dna.canopy_spread
            glBegin(GL_TRIANGLES)
            glVertex3f(0, height, 0)
            glVertex3f(-cap, height * 0.6, 0)
            glVertex3f(cap, height * 0.6, 0)
            glEnd()
        else:
            # Trees - trunk + layered cone canopy (looks better than flat triangle)
            glColor3f(*dna.trunk_color.rgb)
            w = dna.width_gene.value
            glBegin(GL_QUADS)
            glVertex3f(-w, 0, 0)
            glVertex3f(w, 0, 0)
            glVertex3f(w * 0.5, height * 0.55, 0)
            glVertex3f(-w * 0.5, height * 0.55, 0)
            glEnd()
            
            # Multi-layer cone canopy (3 overlapping cones)
            glColor3f(*dna.leaf_color.rgb)
            base_spread = height * dna.canopy_spread * 0.45
            
            # Bottom layer - widest
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, height * 0.75, 0)
            for i in range(7):
                angle = (i / 6) * 2 * 3.14159
                x = math.cos(angle) * base_spread
                z = math.sin(angle) * base_spread
                glVertex3f(x, height * 0.35, z)
            glEnd()
            
            # Middle layer
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, height * 0.95, 0)
            for i in range(7):
                angle = (i / 6) * 2 * 3.14159
                x = math.cos(angle) * base_spread * 0.7
                z = math.sin(angle) * base_spread * 0.7
                glVertex3f(x, height * 0.55, z)
            glEnd()
            
            # Top layer - narrowest
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, height * 1.1, 0)
            for i in range(7):
                angle = (i / 6) * 2 * 3.14159
                x = math.cos(angle) * base_spread * 0.4
                z = math.sin(angle) * base_spread * 0.4
                glVertex3f(x, height * 0.75, z)
            glEnd()
        
        glPopMatrix()
    
    @staticmethod
    def draw_point(plant: PlantInstance):
        """Draw plant as a single colored point for far distance."""
        glColor3f(*plant.dna.leaf_color.rgb)
        height = plant.dna.height_gene.value * plant.scale
        glVertex3f(plant.x, plant.y + height * 0.5, plant.z)
    
    # === Detailed Drawing Methods ===
    
    @staticmethod
    def _draw_grass(dna: PlantDNA):
        """Draw grass tuft with multiple blades."""
        glColor3f(*dna.leaf_color.rgb)
        height = dna.height_gene.value
        blade_count = max(5, dna.branch_count)
        
        glBegin(GL_TRIANGLES)
        for i in range(blade_count):
            angle = (i / blade_count) * 2 * math.pi + i * 0.4
            spread = 0.12 + (i % 3) * 0.03
            
            bx = math.cos(angle) * spread
            bz = math.sin(angle) * spread
            
            # Tip curves outward
            tip_x = bx * 1.8
            tip_z = bz * 1.8
            blade_height = height * (0.6 + (i % 4) * 0.12)
            
            # Blade as triangle
            glVertex3f(bx - 0.015, 0, bz)
            glVertex3f(bx + 0.015, 0, bz)
            glVertex3f(tip_x, blade_height, tip_z)
        glEnd()
    
    @staticmethod
    def _draw_fern(dna: PlantDNA):
        """Draw fern with multiple fronds."""
        height = dna.height_gene.value
        frond_count = max(4, dna.branch_count)
        
        for i in range(frond_count):
            angle = (i / frond_count) * 2 * math.pi
            
            glPushMatrix()
            glRotatef(math.degrees(angle), 0, 1, 0)
            glRotatef(25 + dna.droop * 35, 1, 0, 0)
            
            # Frond stem
            glColor3f(*dna.trunk_color.rgb)
            frond_len = height * 0.85
            
            glBegin(GL_LINES)
            glVertex3f(0, 0.05, 0)
            glVertex3f(0, 0.05, frond_len)
            glEnd()
            
            # Leaflets along frond
            glColor3f(*dna.leaf_color.rgb)
            glBegin(GL_TRIANGLES)
            leaflets = 8
            for j in range(leaflets):
                t = (j + 1) / (leaflets + 1)
                z = frond_len * t
                size = 0.18 * (1 - t * 0.4) * dna.leaf_size
                
                # Left leaflet
                glVertex3f(0, 0.05, z)
                glVertex3f(-size, 0.05 + size * 0.25, z + size * 0.4)
                glVertex3f(0, 0.05, z + size * 0.6)
                
                # Right leaflet
                glVertex3f(0, 0.05, z)
                glVertex3f(size, 0.05 + size * 0.25, z + size * 0.4)
                glVertex3f(0, 0.05, z + size * 0.6)
            glEnd()
            
            glPopMatrix()
    
    @staticmethod
    def _draw_bush(dna: PlantDNA):
        """Draw bush with multiple stems and dome canopy."""
        height = dna.height_gene.value
        
        # Multiple stems
        glColor3f(*dna.trunk_color.rgb)
        stem_count = max(3, dna.branch_count)
        
        for i in range(stem_count):
            angle = (i / stem_count) * 2 * math.pi
            spread = 0.15 + dna.asymmetry * 0.1
            
            bx = math.cos(angle) * spread
            bz = math.sin(angle) * spread
            stem_height = height * (0.5 + (i % 3) * 0.15)
            
            glBegin(GL_LINES)
            glVertex3f(bx, 0, bz)
            glVertex3f(bx * 1.3, stem_height, bz * 1.3)
            glEnd()
        
        # Leafy dome
        glColor3f(*dna.leaf_color.rgb)
        radius = height * dna.canopy_spread * 0.6
        
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, height, 0)
        segments = 12
        for i in range(segments + 1):
            angle = (i / segments) * 2 * math.pi
            x = math.cos(angle) * radius
            z = math.sin(angle) * radius
            glVertex3f(x, height * 0.25, z)
        glEnd()
        
        # Flowers if present
        if dna.has_flowers and dna.flower_size > 0:
            glColor3f(*dna.flower_color.rgb)
            for i in range(3):
                angle = i * 2.1
                fx = math.cos(angle) * radius * 0.7
                fz = math.sin(angle) * radius * 0.7
                fy = height * 0.6
                
                size = dna.flower_size * 0.3
                glBegin(GL_TRIANGLES)
                glVertex3f(fx, fy + size, fz)
                glVertex3f(fx - size, fy, fz)
                glVertex3f(fx + size, fy, fz)
                glEnd()
    
    @staticmethod
    def _draw_mushroom(dna: PlantDNA):
        """Draw mushroom with stem and cap."""
        height = dna.height_gene.value
        width = dna.width_gene.value
        
        # Stem
        glColor3f(*dna.trunk_color.rgb)
        segments = 8
        
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = (i / segments) * 2 * math.pi
            x = math.cos(angle) * width * 0.4
            z = math.sin(angle) * width * 0.4
            
            glVertex3f(x, 0, z)
            glVertex3f(x * 0.8, height * 0.7, z * 0.8)
        glEnd()
        
        # Cap
        glColor3f(*dna.leaf_color.rgb)
        cap_radius = height * dna.canopy_spread * 0.8
        
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, height, 0)
        for i in range(segments + 1):
            angle = (i / segments) * 2 * math.pi
            x = math.cos(angle) * cap_radius
            z = math.sin(angle) * cap_radius
            glVertex3f(x, height * 0.65, z)
        glEnd()
        
        # Glow effect
        if dna.has_glow:
            glColor4f(*dna.glow_color.rgb, dna.glow_intensity * 0.5)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, height * 0.75, 0)
            for i in range(segments + 1):
                angle = (i / segments) * 2 * math.pi
                x = math.cos(angle) * cap_radius * 0.7
                z = math.sin(angle) * cap_radius * 0.7
                glVertex3f(x, height * 0.68, z)
            glEnd()
    
    @staticmethod
    def _draw_cactus(dna: PlantDNA):
        """Draw cactus with optional arms."""
        height = dna.height_gene.value
        width = dna.width_gene.value
        
        glColor3f(*dna.trunk_color.rgb)
        
        # Main body
        segments = 8
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = (i / segments) * 2 * math.pi
            x = math.cos(angle) * width * 0.5
            z = math.sin(angle) * width * 0.5
            
            glVertex3f(x, 0, z)
            glVertex3f(x * 0.9, height, z * 0.9)
        glEnd()
        
        # Top cap
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, height + width * 0.3, 0)
        for i in range(segments + 1):
            angle = (i / segments) * 2 * math.pi
            x = math.cos(angle) * width * 0.45
            z = math.sin(angle) * width * 0.45
            glVertex3f(x, height, z)
        glEnd()
        
        # Arms
        for i in range(dna.branch_count):
            arm_angle = (i / max(1, dna.branch_count)) * 2 * math.pi + 0.5
            arm_height = height * (0.4 + i * 0.15)
            
            glPushMatrix()
            glTranslatef(0, arm_height, 0)
            glRotatef(math.degrees(arm_angle), 0, 1, 0)
            glRotatef(70, 0, 0, 1)
            
            arm_len = height * 0.4
            arm_w = width * 0.3
            
            glBegin(GL_QUAD_STRIP)
            for j in range(segments + 1):
                a = (j / segments) * 2 * math.pi
                x = math.cos(a) * arm_w * 0.5
                z = math.sin(a) * arm_w * 0.5
                glVertex3f(x, 0, z)
                glVertex3f(x * 0.8, arm_len, z * 0.8)
            glEnd()
            
            glPopMatrix()
        
        # Flower on top
        if dna.has_flowers:
            glColor3f(*dna.flower_color.rgb)
            flower_y = height + width * 0.3
            size = dna.flower_size * 0.4
            
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, flower_y + size * 0.5, 0)
            for i in range(7):
                angle = (i / 6) * 2 * math.pi
                x = math.cos(angle) * size
                z = math.sin(angle) * size
                glVertex3f(x, flower_y, z)
            glEnd()
    
    @staticmethod
    def _draw_tree(dna: PlantDNA):
        """Draw tree with multi-segment trunk, branches, and canopy."""
        # Draw multi-segment trunk
        glColor3f(*dna.trunk_color.rgb)
        
        current_y = 0
        current_width = dna.width_gene.value
        total_height = dna.height_gene.value
        
        for seg in dna.trunk_segments:
            seg_height = seg.length * total_height / len(dna.trunk_segments)
            PlantRenderer._draw_trunk_segment(
                current_y, seg_height, current_width,
                seg.taper, seg.curve, seg.twist
            )
            current_y += seg_height
            current_width *= seg.taper
        
        # Draw branches
        if dna.branch_count > 0:
            branch_start_y = total_height * dna.branch_height
            PlantRenderer._draw_branches(dna, branch_start_y)
        
        # Draw canopy
        glColor3f(*dna.leaf_color.rgb)
        PlantRenderer._draw_canopy(dna, total_height)
        
        # Draw flowers if present
        if dna.has_flowers and dna.flower_size > 0:
            PlantRenderer._draw_flowers(dna, total_height)
        
        # Draw glow if present
        if dna.has_glow and dna.glow_intensity > 0:
            PlantRenderer._draw_glow(dna, total_height)
    
    @staticmethod
    def _draw_trunk_segment(start_y: float, height: float, width: float,
                            taper: float, curve: float, twist: float):
        """Draw a single trunk segment with optional curve and twist."""
        segments = 6
        rings = 4
        
        glBegin(GL_QUAD_STRIP)
        for ring in range(rings + 1):
            t = ring / rings
            y = start_y + t * height
            
            # Apply curve
            offset_x = curve * math.sin(t * math.pi) * height * 0.15
            
            # Current width (with taper)
            w = width * (1 - t * (1 - taper))
            
            # Twist angle
            twist_angle = twist * t * math.pi
            
            for seg in range(segments + 1):
                angle = (seg / segments) * 2 * math.pi + twist_angle
                x = math.cos(angle) * w + offset_x
                z = math.sin(angle) * w
                
                if ring < rings:
                    glVertex3f(x, y, z)
                else:
                    glVertex3f(x, y, z)
        glEnd()
    
    @staticmethod
    def _draw_branches(dna: PlantDNA, start_height: float):
        """Draw branches from DNA specification."""
        for i in range(dna.branch_count):
            angle = (i / dna.branch_count) * 360 * dna.branch_spread
            angle += dna.asymmetry * 30 * math.sin(i * 2.5)
            
            glPushMatrix()
            glTranslatef(0, start_height + i * 0.3, 0)
            glRotatef(angle, 0, 1, 0)
            glRotatef(dna.branch_angle * 75 + dna.droop * 20, 1, 0, 0)
            
            # Draw branch segments
            branch_len = dna.height_gene.value * 0.25
            branch_w = dna.width_gene.value * 0.25
            
            glColor3f(*dna.trunk_color.rgb)
            for seg in dna.branch_segments:
                seg_len = seg.length * branch_len
                glBegin(GL_QUADS)
                glVertex3f(-branch_w, 0, 0)
                glVertex3f(branch_w, 0, 0)
                glVertex3f(branch_w * seg.taper, seg_len, 0)
                glVertex3f(-branch_w * seg.taper, seg_len, 0)
                glEnd()
                branch_w *= seg.taper
            
            glPopMatrix()
    
    @staticmethod
    def _draw_canopy(dna: PlantDNA, height: float):
        """Draw tree canopy based on shape."""
        spread = height * dna.canopy_spread * 0.5
        canopy_base = height * 0.5
        
        if dna.canopy_shape == "cone":
            # Cone shape
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, height * 1.1, 0)
            segments = 12
            for i in range(segments + 1):
                angle = (i / segments) * 2 * math.pi
                x = math.cos(angle) * spread
                z = math.sin(angle) * spread
                glVertex3f(x, canopy_base, z)
            glEnd()
            
        elif dna.canopy_shape == "umbrella":
            # Flat umbrella shape
            segments = 12
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, height * 0.95, 0)
            for i in range(segments + 1):
                angle = (i / segments) * 2 * math.pi
                x = math.cos(angle) * spread * 1.2
                z = math.sin(angle) * spread * 1.2
                glVertex3f(x, height * 0.85 - dna.droop * spread * 0.3, z)
            glEnd()
            
        elif dna.canopy_shape == "weeping":
            # Weeping willow style
            segments = 16
            for layer in range(3):
                layer_y = height * (0.9 - layer * 0.15)
                layer_spread = spread * (0.6 + layer * 0.25)
                droop_amount = dna.droop * layer * 0.2
                
                glBegin(GL_TRIANGLE_FAN)
                glVertex3f(0, layer_y, 0)
                for i in range(segments + 1):
                    angle = (i / segments) * 2 * math.pi
                    x = math.cos(angle) * layer_spread
                    z = math.sin(angle) * layer_spread
                    y = layer_y - droop_amount - abs(math.sin(angle * 3)) * 0.2
                    glVertex3f(x, y, z)
                glEnd()
                
        elif dna.canopy_shape == "columnar":
            # Tall narrow shape
            segments = 8
            layers = 4
            for layer in range(layers):
                layer_y = canopy_base + (height - canopy_base) * layer / layers
                next_y = canopy_base + (height - canopy_base) * (layer + 1) / layers
                layer_spread = spread * 0.4 * (1 - layer * 0.15)
                
                glBegin(GL_QUAD_STRIP)
                for i in range(segments + 1):
                    angle = (i / segments) * 2 * math.pi
                    x = math.cos(angle) * layer_spread
                    z = math.sin(angle) * layer_spread
                    glVertex3f(x, layer_y, z)
                    glVertex3f(x * 0.9, next_y, z * 0.9)
                glEnd()
                
        else:  # "dome" (default)
            # Layered dome
            layers = 3
            for layer in range(layers):
                layer_y = height * (0.55 + layer * 0.12)
                layer_spread = spread * (1 - layer * 0.15)
                
                glBegin(GL_TRIANGLE_FAN)
                glVertex3f(0, layer_y + height * 0.15, 0)
                segments = 10
                for i in range(segments + 1):
                    angle = (i / segments) * 2 * math.pi
                    x = math.cos(angle) * layer_spread
                    z = math.sin(angle) * layer_spread
                    y = layer_y - dna.droop * layer_spread * 0.2
                    glVertex3f(x, y, z)
                glEnd()
    
    @staticmethod
    def _draw_flowers(dna: PlantDNA, height: float):
        """Draw flowers on the plant."""
        glColor3f(*dna.flower_color.rgb)
        
        spread = height * dna.canopy_spread * 0.4
        size = dna.flower_size * 0.25
        
        positions = [
            (0, height * 0.85, 0),
            (spread * 0.6, height * 0.7, spread * 0.3),
            (-spread * 0.5, height * 0.75, -spread * 0.4),
            (spread * 0.3, height * 0.65, -spread * 0.5),
        ]
        
        for px, py, pz in positions:
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(px, py + size * 0.5, pz)
            petals = 6
            for i in range(petals + 1):
                angle = (i / petals) * 2 * math.pi
                fx = px + math.cos(angle) * size
                fz = pz + math.sin(angle) * size
                glVertex3f(fx, py, fz)
            glEnd()
    
    @staticmethod
    def _draw_glow(dna: PlantDNA, height: float):
        """Draw bioluminescent glow effect."""
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        glColor4f(*dna.glow_color.rgb, dna.glow_intensity * 0.4)
        
        spread = height * dna.canopy_spread * 0.5
        
        # Glow sphere
        segments = 8
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, height * 0.7, 0)
        for i in range(segments + 1):
            angle = (i / segments) * 2 * math.pi
            x = math.cos(angle) * spread * 0.8
            z = math.sin(angle) * spread * 0.8
            glVertex3f(x, height * 0.5, z)
        glEnd()
        
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


class FloraManager:
    """Manages plant generation and rendering with LOD and DNA pooling."""
    
    # LOD distances (in world units)
    LOD_FULL = 120     # Full 3D geometry
    LOD_SIMPLE = 300   # Simplified geometry
    LOD_BILLBOARD = 600  # Just colored points
    
    def __init__(self, world_seed: int = 42):
        self.world_seed = world_seed
        self.dna_pool = DNAPool(world_seed)
        self.chunk_plants: Dict[Tuple[int, int], List[PlantInstance]] = {}
        self.display_lists: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    
    def get_plants_for_chunk(self, cx: int, cz: int, heightmap, 
                             chunk_world_x: float, chunk_world_z: float,
                             tile_scale: float, height_scale: float) -> List[PlantInstance]:
        """Generate or retrieve plants for a chunk using DNA from the pool."""
        key = (cx, cz)
        if key in self.chunk_plants:
            return self.chunk_plants[key]
        
        # Get DNA species for this chunk (with neighbor crossover)
        chunk_dna_list = self.dna_pool.get_dna_for_chunk(cx, cz)
        
        # Deterministic RNG for plant placement
        chunk_seed = abs(hash((self.world_seed, cx, cz, "plants"))) % (2**31)
        rng = np.random.default_rng(chunk_seed)
        
        plants = []
        h, w = heightmap.shape
        
        # Place plants using chunk's DNA species (reduced ~10% for perf)
        num_plants = rng.integers(18, 40)
        
        for _ in range(num_plants):
            local_x = rng.integers(2, w - 2)
            local_z = rng.integers(2, h - 2)
            
            ground_h = heightmap[local_z, local_x]
            
            # Placement constraints
            if ground_h < 1:  # Underwater
                continue
            if ground_h > 45:  # Too high
                continue
            
            # World position
            world_x = chunk_world_x + local_x * tile_scale
            world_z = chunk_world_z + local_z * tile_scale
            
            # Pick a species from this chunk's DNA pool
            dna = rng.choice(chunk_dna_list)
            
            # Apply slight per-plant mutation for variety
            if rng.random() < 0.3:
                dna = dna.mutate(rng, strength=0.15)
            
            # Random variation
            scale = 0.5 + rng.random() * 1.0
            rotation = rng.random() * 360
            
            plants.append(PlantInstance(
                x=world_x,
                y=ground_h * height_scale,
                z=world_z,
                dna=dna,
                scale=scale,
                rotation=rotation
            ))
        
        self.chunk_plants[key] = plants
        return plants
    
    def create_display_lists(self, cx: int, cz: int, 
                             plants: List[PlantInstance]) -> Tuple[int, int, int]:
        """Create display lists for full, simple, and point LODs."""
        key = (cx, cz)
        if key in self.display_lists:
            return self.display_lists[key]
        
        # Full detail
        full_list = glGenLists(1)
        glNewList(full_list, GL_COMPILE)
        for plant in plants:
            PlantRenderer.draw_full(plant)
        glEndList()
        
        # Simple detail
        simple_list = glGenLists(1)
        glNewList(simple_list, GL_COMPILE)
        for plant in plants:
            PlantRenderer.draw_simple(plant)
        glEndList()
        
        # Point sprites
        point_list = glGenLists(1)
        glNewList(point_list, GL_COMPILE)
        glPointSize(4)
        glBegin(GL_POINTS)
        for plant in plants:
            PlantRenderer.draw_point(plant)
        glEnd()
        glEndList()
        
        self.display_lists[key] = (full_list, simple_list, point_list)
        return self.display_lists[key]
    
    def render_chunk_flora(self, cx: int, cz: int, camera_x: float, camera_z: float,
                           heightmap, chunk_world_x: float, chunk_world_z: float,
                           tile_scale: float, height_scale: float = 3.5):
        """Render plants for a chunk with appropriate LOD."""
        plants = self.get_plants_for_chunk(
            cx, cz, heightmap, chunk_world_x, chunk_world_z, tile_scale, height_scale
        )
        
        if not plants:
            return
        
        full_list, simple_list, point_list = self.create_display_lists(cx, cz, plants)
        
        # Distance to chunk center
        chunk_center_x = chunk_world_x + 16 * tile_scale
        chunk_center_z = chunk_world_z + 16 * tile_scale
        dist = math.sqrt((camera_x - chunk_center_x)**2 + (camera_z - chunk_center_z)**2)
        
        glDisable(GL_LIGHTING)
        
        if dist < self.LOD_FULL:
            glCallList(full_list)
        elif dist < self.LOD_SIMPLE:
            glCallList(simple_list)
        elif dist < self.LOD_BILLBOARD:
            glCallList(point_list)
        
        glEnable(GL_LIGHTING)
    
    def cleanup_chunk(self, cx: int, cz: int):
        """Remove cached data for a chunk."""
        key = (cx, cz)
        if key in self.display_lists:
            for dl in self.display_lists[key]:
                glDeleteLists(dl, 1)
            del self.display_lists[key]
        if key in self.chunk_plants:
            del self.chunk_plants[key]
    
    def update_camera_position(self, camera_cx: int, camera_cz: int):
        """Called when camera moves to potentially crossover DNA from neighbors."""
        self.dna_pool.cleanup_distant_chunks(camera_cx, camera_cz, max_distance=25)
