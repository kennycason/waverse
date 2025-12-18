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
    render_fraction: float = 1.0  # 0.0-1.0, how much of plant to render (energy-based)


class PlantRenderer:
    """Renders individual plants from DNA."""
    
    # Class variable for passing render_fraction to static methods
    _current_render_fraction: float = 1.0
    
    @staticmethod
    def draw_full(plant: PlantInstance):
        """Draw plant at full detail using its DNA."""
        glPushMatrix()
        glTranslatef(plant.x, plant.y, plant.z)
        glRotatef(plant.rotation, 0, 1, 0)
        
        # render_fraction controls how much of the plant to draw (1.0 = full, 0.5 = half)
        # This is used to show damage/eating - branches disappear from top
        render_frac = getattr(plant, 'render_fraction', 1.0)
        
        # Keep trunk full size - only affect branch rendering
        glScalef(plant.scale, plant.scale, plant.scale)
        
        # Store render_fraction for branch methods to use
        PlantRenderer._current_render_fraction = render_frac
        
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
        elif dna.plant_type in (PlantType.VINE, PlantType.SPINY_VINE, PlantType.SEAWEED):
            PlantRenderer._draw_vine(dna)
        elif dna.plant_type == PlantType.OCTOPUS:
            PlantRenderer._draw_octopus(dna)
        elif dna.plant_type == PlantType.TENTACLE:
            PlantRenderer._draw_tentacle(dna)
        elif dna.plant_type == PlantType.SPIRAL:
            PlantRenderer._draw_spiral(dna)
        elif dna.plant_type in (PlantType.GROUNDCOVER, PlantType.CREEPER, PlantType.LICHEN, PlantType.MOSS_PAD):
            PlantRenderer._draw_groundcover(dna)
        elif dna.plant_type == PlantType.LILY_PAD:
            PlantRenderer._draw_lily_pad(dna)
        elif dna.plant_type == PlantType.CORAL:
            PlantRenderer._draw_coral(dna)
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
        elif dna.plant_type in (PlantType.VINE, PlantType.SPINY_VINE, PlantType.SEAWEED):
            # Simple vine - thin quad strip
            glColor3f(*dna.trunk_color.rgb)
            w = 0.05
            glBegin(GL_QUAD_STRIP)
            for i in range(6):
                t = i / 5
                sway = math.sin(t * math.pi * 2) * 0.2
                glVertex3f(sway - w, t * height, 0)
                glVertex3f(sway + w, t * height, 0)
            glEnd()
        elif dna.plant_type == PlantType.OCTOPUS:
            # Simple octopus - triangles radiating out
            glColor3f(*dna.trunk_color.rgb)
            glBegin(GL_TRIANGLES)
            for i in range(max(1, dna.branch_count)):
                angle = (i / max(1, dna.branch_count)) * 2 * math.pi
                a2 = angle + 0.1
                glVertex3f(0, height * 0.3, 0)
                glVertex3f(math.cos(angle) * height * 0.5, 0, math.sin(angle) * height * 0.5)
                glVertex3f(math.cos(a2) * height * 0.4, 0, math.sin(a2) * height * 0.4)
            glEnd()
        elif dna.plant_type == PlantType.TENTACLE:
            # Simple tentacle - drooping triangles
            glColor3f(*dna.trunk_color.rgb)
            glBegin(GL_TRIANGLES)
            glVertex3f(0, height, 0)
            glVertex3f(-0.1, height * 0.5, 0)
            glVertex3f(0.1, height * 0.5, 0)
            glVertex3f(0, height * 0.5, 0)
            glVertex3f(-0.1, 0, 0)
            glVertex3f(0.1, 0, 0)
            glEnd()
        elif dna.plant_type == PlantType.SPIRAL:
            # Simple spiral - quad strip helix
            glColor3f(*dna.trunk_color.rgb)
            w = 0.05
            glBegin(GL_QUAD_STRIP)
            for i in range(12):
                t = i / 11
                angle = t * 4 * math.pi
                r = 0.2 * (1 - t * 0.3)
                x, z = math.cos(angle) * r, math.sin(angle) * r
                glVertex3f(x - w, t * height, z)
                glVertex3f(x + w, t * height, z)
            glEnd()
        elif dna.plant_type in (PlantType.GROUNDCOVER, PlantType.CREEPER, PlantType.LICHEN, PlantType.MOSS_PAD):
            # Simple ground cover - flat circle
            glColor3f(*dna.leaf_color.rgb)
            width = dna.width_gene.value
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, height, 0)
            for i in range(9):
                angle = (i / 8) * 2 * math.pi
                glVertex3f(math.cos(angle) * width * 0.5, 0, math.sin(angle) * width * 0.5)
            glEnd()
        elif dna.plant_type == PlantType.LILY_PAD:
            # Simple lily pad - flat circle
            glColor3f(*dna.leaf_color.rgb)
            width = dna.width_gene.value
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, 0.02, 0)
            for i in range(9):
                angle = (i / 8) * 2 * math.pi
                glVertex3f(math.cos(angle) * width * 0.5, 0.02, math.sin(angle) * width * 0.5)
            glEnd()
        elif dna.plant_type == PlantType.CORAL:
            # Simple coral - a few branches
            glColor3f(*dna.trunk_color.rgb)
            glBegin(GL_TRIANGLES)
            for i in range(max(2, dna.branch_count // 2)):
                angle = (i / max(1, dna.branch_count // 2)) * 2 * math.pi
                w = dna.width_gene.value * 0.3
                glVertex3f(0, 0, 0)
                glVertex3f(math.cos(angle) * w, height * 0.7, math.sin(angle) * w)
                glVertex3f(math.cos(angle + 0.2) * w * 0.5, height * 0.5, math.sin(angle + 0.2) * w * 0.5)
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
    def _draw_vine(dna: PlantDNA):
        """Draw vine with many curving segments."""
        height = dna.height_gene.value
        width = dna.width_gene.value
        
        glColor3f(*dna.trunk_color.rgb)
        
        # Draw main vine as connected quad strip (thicker than lines)
        current_x, current_y, current_z = 0, 0, 0
        direction = 0  # Yaw angle
        vine_width = width * 0.1
        
        glBegin(GL_QUAD_STRIP)
        glVertex3f(-vine_width, 0, 0)
        glVertex3f(vine_width, 0, 0)
        
        for seg in dna.trunk_segments:
            seg_len = seg.length * height / len(dna.trunk_segments)
            direction += seg.curve * 0.5
            
            current_x += math.sin(direction) * seg_len * 0.3
            current_y += seg_len
            current_z += math.cos(direction) * seg_len * 0.3
            
            glVertex3f(current_x - vine_width, current_y, current_z)
            glVertex3f(current_x + vine_width, current_y, current_z)
        glEnd()
        
        # Draw leaves along vine
        glColor3f(*dna.leaf_color.rgb)
        leaf_y = 0
        for seg in dna.trunk_segments:
            seg_len = seg.length * height / len(dna.trunk_segments)
            leaf_y += seg_len
            
            size = 0.2 + dna.leaf_density * 0.3
            glBegin(GL_TRIANGLES)
            glVertex3f(0, leaf_y, 0)
            glVertex3f(-size, leaf_y - size * 0.5, -size * 0.3)
            glVertex3f(size, leaf_y - size * 0.3, size * 0.3)
            glEnd()
        
        # Spines for spiny vines
        if dna.plant_type == PlantType.SPINY_VINE:
            glColor3f(0.3, 0.25, 0.2)
            spine_y = 0
            glBegin(GL_TRIANGLES)
            for i, seg in enumerate(dna.trunk_segments):
                seg_len = seg.length * height / len(dna.trunk_segments)
                spine_y += seg_len * 0.5
                
                for j in range(3):
                    angle = (j / 3 + i * 0.33) * 2 * math.pi
                    sx = math.cos(angle) * 0.3
                    sz = math.sin(angle) * 0.3
                    # Draw spine as thin triangle
                    glVertex3f(0, spine_y, 0)
                    glVertex3f(sx, spine_y + 0.05, sz)
                    glVertex3f(sx * 0.9, spine_y - 0.05, sz * 0.9)
                spine_y += seg_len * 0.5
            glEnd()
    
    @staticmethod
    def _draw_octopus(dna: PlantDNA):
        """Draw octopus-like recursive branching plant."""
        height = dna.height_gene.value
        width = dna.width_gene.value
        
        glColor3f(*dna.trunk_color.rgb)
        
        # Central body/bulb
        segments = 8
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, height * 0.3, 0)
        for i in range(segments + 1):
            angle = (i / segments) * 2 * math.pi
            x = math.cos(angle) * width * 0.6
            z = math.sin(angle) * width * 0.6
            glVertex3f(x, 0, z)
        glEnd()
        
        # Tentacles
        num_tentacles = dna.branch_count
        for t in range(num_tentacles):
            base_angle = (t / num_tentacles) * 2 * math.pi
            
            glPushMatrix()
            glRotatef(math.degrees(base_angle), 0, 1, 0)
            glTranslatef(width * 0.4, height * 0.2, 0)
            glRotatef(60 + dna.branch_angle * 30, 0, 0, 1)
            
            # Draw tentacle as tapering curve
            tent_len = height * 0.8
            tent_width = width * 0.15
            
            glBegin(GL_QUAD_STRIP)
            for i in range(8):
                t = i / 7
                # Curve outward then down
                curve = math.sin(t * math.pi * 0.8) * 0.5
                y = t * tent_len * 0.7
                x = curve * tent_len * 0.5
                w = tent_width * (1 - t * 0.8)
                
                glVertex3f(x - w, y, 0)
                glVertex3f(x + w, y, 0)
            glEnd()
            
            glPopMatrix()
        
        # Glow effect
        if dna.has_glow:
            glColor4f(*dna.glow_color.rgb, 0.3)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, height * 0.3, 0)
            glow_r = width * 1.2
            for i in range(segments + 1):
                angle = (i / segments) * 2 * math.pi
                glVertex3f(math.cos(angle) * glow_r, 0, math.sin(angle) * glow_r)
            glEnd()
    
    @staticmethod
    def _draw_tentacle(dna: PlantDNA):
        """Draw upside-down dangling tentacle growth."""
        height = dna.height_gene.value
        width = dna.width_gene.value
        
        glColor3f(*dna.trunk_color.rgb)
        
        # Base attachment point (like it's hanging from something)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, height, 0)
        for i in range(7):
            angle = (i / 6) * 2 * math.pi
            glVertex3f(math.cos(angle) * width * 0.4, height * 0.9, math.sin(angle) * width * 0.4)
        glEnd()
        
        # Dangling segments
        current_y = height * 0.9
        for i, seg in enumerate(dna.trunk_segments):
            seg_len = seg.length * height / len(dna.trunk_segments)
            next_y = current_y - seg_len
            
            # Curve adds sway
            sway = seg.curve * 0.3
            
            w = width * (0.3 - i * 0.03)
            glBegin(GL_QUAD_STRIP)
            for j in range(5):
                t = j / 4
                y = current_y - t * seg_len
                x_off = sway * math.sin(t * math.pi)
                w_t = w * (1 - t * 0.2)
                
                glVertex3f(x_off - w_t, y, 0)
                glVertex3f(x_off + w_t, y, 0)
            glEnd()
            
            current_y = next_y
        
        # Tip
        glBegin(GL_TRIANGLES)
        glVertex3f(0, current_y - width * 0.5, 0)
        glVertex3f(-width * 0.2, current_y, 0)
        glVertex3f(width * 0.2, current_y, 0)
        glEnd()
        
        # Glow
        if dna.has_glow:
            glColor4f(*dna.glow_color.rgb, 0.4)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, current_y - width * 0.3, 0)
            for i in range(7):
                angle = (i / 6) * 2 * math.pi
                glVertex3f(math.cos(angle) * width * 0.6, current_y, math.sin(angle) * width * 0.6)
            glEnd()
    
    @staticmethod
    def _draw_spiral(dna: PlantDNA):
        """Draw spiraling growth pattern."""
        height = dna.height_gene.value
        width = dna.width_gene.value
        
        glColor3f(*dna.trunk_color.rgb)
        
        # Spiral up
        total_rotation = len(dna.trunk_segments) * 2  # Number of full rotations
        current_y = 0
        radius = width * 0.8
        
        glBegin(GL_QUAD_STRIP)
        steps = len(dna.trunk_segments) * 8
        for i in range(steps + 1):
            t = i / steps
            angle = t * total_rotation * 2 * math.pi
            y = t * height
            r = radius * (1 - t * 0.3)  # Taper
            
            x = math.cos(angle) * r
            z = math.sin(angle) * r
            
            # Inner and outer edge of spiral ribbon
            inner_r = r * 0.7
            glVertex3f(math.cos(angle) * inner_r, y, math.sin(angle) * inner_r)
            glVertex3f(x, y, z)
        glEnd()
        
        # Tip
        glColor3f(*dna.leaf_color.rgb)
        glBegin(GL_TRIANGLES)
        glVertex3f(0, height + width * 0.5, 0)
        tip_w = width * 0.3
        for i in range(6):
            a1 = (i / 6) * 2 * math.pi
            a2 = ((i + 1) / 6) * 2 * math.pi
            glVertex3f(math.cos(a1) * tip_w, height, math.sin(a1) * tip_w)
            glVertex3f(math.cos(a2) * tip_w, height, math.sin(a2) * tip_w)
        glEnd()
    
    @staticmethod
    def _draw_groundcover(dna: PlantDNA):
        """Draw sprawling ground cover - wide, flat, spreading plants."""
        width = dna.width_gene.value
        height = dna.height_gene.value
        
        glColor3f(*dna.leaf_color.rgb)
        
        # Draw radiating flat patches/leaves
        num_patches = max(4, dna.branch_count)
        for i in range(num_patches):
            angle = (i / num_patches) * 2 * math.pi + (hash(i) % 100) / 200.0
            dist = width * (0.3 + (hash(i + 100) % 70) / 100.0)
            
            x = math.cos(angle) * dist
            z = math.sin(angle) * dist
            
            # Each patch is a small dome or flat circle
            patch_size = width * 0.2 * (0.5 + (hash(i + 200) % 50) / 100.0)
            patch_height = height * (0.5 + (hash(i + 300) % 50) / 100.0)
            
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(x, patch_height, z)  # Center top
            for j in range(9):
                a = (j / 8) * 2 * math.pi
                glVertex3f(x + math.cos(a) * patch_size, 0, z + math.sin(a) * patch_size)
            glEnd()
        
        # Central mound
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, height * 1.2, 0)
        for j in range(9):
            a = (j / 8) * 2 * math.pi
            glVertex3f(math.cos(a) * width * 0.3, 0, math.sin(a) * width * 0.3)
        glEnd()
    
    @staticmethod
    def _draw_lily_pad(dna: PlantDNA):
        """Draw floating lily pad with optional flower."""
        width = dna.width_gene.value
        
        # Main pad - circular with notch
        glColor3f(*dna.leaf_color.rgb)
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, 0.02, 0)  # Center, slightly above water
        segments = 16
        for i in range(segments + 1):
            angle = (i / segments) * 2 * math.pi
            # Skip a small wedge for the classic lily pad notch
            if 0.4 < (i / segments) < 0.5:
                continue
            r = width * 0.5
            glVertex3f(math.cos(angle) * r, 0.02, math.sin(angle) * r)
        glEnd()
        
        # Add slight rim/edge coloring
        glColor3f(dna.leaf_color.r * 0.7, dna.leaf_color.g * 0.9, dna.leaf_color.b * 0.7)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = (i / segments) * 2 * math.pi
            if 0.4 < (i / segments) < 0.5:
                continue
            r = width * 0.5
            glVertex3f(math.cos(angle) * r, 0.03, math.sin(angle) * r)
        glEnd()
        
        # Optional flower
        if hasattr(dna, 'has_flower') and dna.has_flower:
            glColor3f(*dna.flower_color.rgb)
            # Flower petals
            petal_count = 6
            for i in range(petal_count):
                angle = (i / petal_count) * 2 * math.pi
                glBegin(GL_TRIANGLES)
                glVertex3f(0, 0.3, 0)  # Center
                pa = 0.15
                glVertex3f(math.cos(angle - pa) * 0.1, 0.15, math.sin(angle - pa) * 0.1)
                glVertex3f(math.cos(angle + pa) * 0.1, 0.15, math.sin(angle + pa) * 0.1)
                glEnd()
            # Yellow center
            glColor3f(1.0, 0.9, 0.3)
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, 0.32, 0)
            for i in range(7):
                a = (i / 6) * 2 * math.pi
                glVertex3f(math.cos(a) * 0.05, 0.28, math.sin(a) * 0.05)
            glEnd()
    
    @staticmethod
    def _draw_coral(dna: PlantDNA):
        """Draw coral - branching underwater structure."""
        height = dna.height_gene.value
        width = dna.width_gene.value
        
        glColor3f(*dna.trunk_color.rgb)
        
        # Main branches growing up
        num_branches = max(3, dna.branch_count)
        for i in range(num_branches):
            angle = (i / num_branches) * 2 * math.pi + (hash(i) % 100) / 100.0
            lean = dna.branch_angle
            
            # Branch parameters
            branch_height = height * (0.5 + (hash(i + 10) % 50) / 100.0)
            branch_width = width * (0.3 + (hash(i + 20) % 40) / 100.0)
            
            # Draw as tapered column
            glBegin(GL_QUAD_STRIP)
            segments = 6
            for j in range(segments + 1):
                t = j / segments
                y = t * branch_height
                current_w = branch_width * (1 - t * 0.6)
                
                # Offset outward as it grows
                x_offset = math.cos(angle) * lean * y
                z_offset = math.sin(angle) * lean * y
                
                for k in [0, 1]:
                    a = (k * 0.5) * 2 * math.pi
                    glVertex3f(
                        x_offset + math.cos(a + angle) * current_w,
                        y,
                        z_offset + math.sin(a + angle) * current_w
                    )
            glEnd()
            
            # Ball tip
            tip_x = math.cos(angle) * lean * branch_height
            tip_z = math.sin(angle) * lean * branch_height
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(tip_x, branch_height + branch_width * 0.3, tip_z)
            for k in range(7):
                a = (k / 6) * 2 * math.pi
                glVertex3f(
                    tip_x + math.cos(a) * branch_width * 0.5,
                    branch_height,
                    tip_z + math.sin(a) * branch_width * 0.5
                )
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
        """Draw branches with recursive sub-branching from DNA specification."""
        # Use a seeded RNG for consistent randomness per plant
        branch_rng = np.random.default_rng(int(dna.species_id) & 0xFFFFFFFF)
        
        # Get render_fraction - skip upper branches when damaged
        render_frac = getattr(PlantRenderer, '_current_render_fraction', 1.0)
        
        # Calculate how many branches to draw based on render_fraction
        # At 100% = all branches, at 50% = half branches (from bottom), at 0% = none
        branches_to_draw = max(1, int(dna.branch_count * render_frac))
        
        for i in range(branches_to_draw):
            angle = (i / max(1, dna.branch_count)) * 360 * dna.branch_spread
            angle += dna.asymmetry * 30 * math.sin(i * 2.5)
            
            glPushMatrix()
            # Vary branch height along trunk
            branch_y = start_height + (i / max(1, dna.branch_count)) * (dna.height_gene.value * 0.4)
            glTranslatef(0, branch_y, 0)
            glRotatef(angle, 0, 1, 0)
            glRotatef(dna.branch_angle * 75 + dna.droop * 20, 1, 0, 0)
            
            # Draw this branch with recursive sub-branches
            branch_len = dna.height_gene.value * 0.3
            branch_w = dna.width_gene.value * 0.3
            PlantRenderer._draw_branch_recursive(
                dna, branch_len, branch_w, 
                depth=0, max_depth=dna.recursive_depth, 
                rng=branch_rng
            )
            
            glPopMatrix()
    
    @staticmethod
    def _draw_branch_recursive(dna: PlantDNA, length: float, width: float, 
                                depth: int, max_depth: int, rng: np.random.Generator):
        """Recursively draw a branch with potential sub-branches."""
        if width < 0.01 or length < 0.05:
            return
        
        # Draw this branch segment
        glColor3f(*dna.trunk_color.rgb)
        
        # Add some natural variation
        curve = (rng.random() - 0.5) * 0.3 * (1 + depth * 0.5)
        
        # Draw the branch as a tapered cylinder approximation
        segments = 4
        taper = 0.7 if depth < max_depth else 0.5
        
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            t = i / segments
            # Current position along branch
            y = t * length
            w = width * (1.0 - t * (1 - taper))
            # Add curve
            x_offset = curve * t * t * length
            
            glVertex3f(-w + x_offset, y, 0)
            glVertex3f(w + x_offset, y, 0)
        glEnd()
        
        # Move to end of this segment for sub-branches
        glTranslatef(curve * length, length, 0)
        
        # Maybe spawn sub-branches if we haven't hit max depth
        if depth < max_depth:
            # Determine number of sub-branches based on DNA
            sub_branch_count = 0
            for _ in range(3):  # Up to 3 potential sub-branches per branch
                if rng.random() < dna.sub_branch_chance:
                    sub_branch_count += 1
            
            for j in range(sub_branch_count):
                glPushMatrix()
                
                # Rotate around the branch axis
                sub_angle = (j / max(1, sub_branch_count)) * 360 + rng.random() * 60 - 30
                glRotatef(sub_angle, 0, 1, 0)
                
                # Angle away from parent branch
                spread_angle = 25 + rng.random() * 35 + dna.droop * 15
                glRotatef(spread_angle, 1, 0, 0)
                
                # Sub-branches are smaller
                sub_length = length * (0.5 + rng.random() * 0.3)
                sub_width = width * taper * (0.6 + rng.random() * 0.2)
                
                # Recurse
                PlantRenderer._draw_branch_recursive(
                    dna, sub_length, sub_width,
                    depth + 1, max_depth, rng
                )
                
                glPopMatrix()
        
        # Draw leaves at branch tips
        if depth >= max_depth - 1 or (depth > 0 and rng.random() < 0.3):
            glColor3f(*dna.leaf_color.rgb)
            leaf_size = dna.leaf_size * 0.15 * (1.0 + rng.random() * 0.5)
            
            # Draw a few leaves
            num_leaves = 2 + int(rng.random() * 3)
            for k in range(num_leaves):
                glPushMatrix()
                glRotatef(k * 120 + rng.random() * 30, 0, 1, 0)
                glRotatef(30 + rng.random() * 40, 1, 0, 0)
                
                # Simple leaf shape
                glBegin(GL_TRIANGLES)
                glVertex3f(0, 0, 0)
                glVertex3f(leaf_size * 0.3, leaf_size, 0)
                glVertex3f(-leaf_size * 0.3, leaf_size, 0)
                glEnd()
                
                glPopMatrix()
    
    @staticmethod
    def _draw_canopy(dna: PlantDNA, height: float):
        """Draw tree canopy based on shape."""
        # Get render_fraction - canopy shrinks when damaged
        render_frac = getattr(PlantRenderer, '_current_render_fraction', 1.0)
        
        # Skip canopy entirely if very damaged
        if render_frac < 0.2:
            return
        
        # Scale canopy by render_fraction
        spread = height * dna.canopy_spread * 0.5 * render_frac
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
    
    # LOD distances (in world units) - more aggressive for performance
    LOD_FULL = 60      # Full 3D geometry (was 120)
    LOD_SIMPLE = 150   # Simplified geometry (was 300)
    LOD_BILLBOARD = 350  # Just colored points (was 600)
    
    def __init__(self, world_seed: int = 42):
        self.world_seed = world_seed
        self.dna_pool = DNAPool(world_seed)
        self.chunk_plants: Dict[Tuple[int, int], List[PlantInstance]] = {}
        self.display_lists: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    
    # Max plants - higher cap since rendering is optimized
    MAX_TOTAL_PLANTS = 16000
    
    def cleanup_distant_chunks(self, center_cx: int, center_cz: int, max_distance: int = 15):
        """Remove plants from distant chunks to prevent memory bloat."""
        to_remove = []
        for (cx, cz) in list(self.chunk_plants.keys()):
            if abs(cx - center_cx) > max_distance or abs(cz - center_cz) > max_distance:
                to_remove.append((cx, cz))
        
        for key in to_remove:
            if key in self.chunk_plants:
                del self.chunk_plants[key]
            if key in self.display_lists:
                # Delete OpenGL display lists
                for dl in self.display_lists[key]:
                    try:
                        glDeleteLists(dl, 1)
                    except:
                        pass
                del self.display_lists[key]
        
        # Emergency cleanup if still over limit
        total_plants = sum(len(p) for p in self.chunk_plants.values())
        if total_plants > self.MAX_TOTAL_PLANTS:
            # Remove furthest chunks until under limit
            chunks_by_dist = sorted(
                self.chunk_plants.keys(),
                key=lambda c: max(abs(c[0] - center_cx), abs(c[1] - center_cz)),
                reverse=True
            )
            for key in chunks_by_dist:
                if total_plants <= self.MAX_TOTAL_PLANTS:
                    break
                total_plants -= len(self.chunk_plants[key])
                del self.chunk_plants[key]
                if key in self.display_lists:
                    for dl in self.display_lists[key]:
                        try:
                            glDeleteLists(dl, 1)
                        except:
                            pass
                    del self.display_lists[key]
    
    def get_plants_for_chunk(self, cx: int, cz: int, heightmap, 
                             chunk_world_x: float, chunk_world_z: float,
                             tile_scale: float, height_scale: float) -> List[PlantInstance]:
        """Generate or retrieve plants for a chunk using DNA from the pool."""
        key = (cx, cz)
        if key in self.chunk_plants:
            return self.chunk_plants[key]
        
        # Don't spawn if at capacity - return empty
        total = sum(len(p) for p in self.chunk_plants.values())
        if total >= self.MAX_TOTAL_PLANTS:
            self.chunk_plants[key] = []  # Mark as processed but empty
            return []
        
        # Get DNA species for this chunk (with neighbor crossover)
        chunk_dna_list = self.dna_pool.get_dna_for_chunk(cx, cz)
        
        # Deterministic RNG for plant placement
        chunk_seed = abs(hash((self.world_seed, cx, cz, "plants"))) % (2**31)
        rng = np.random.default_rng(chunk_seed)
        
        plants = []
        h, w = heightmap.shape
        
        # Place plants - increased density since rendering is optimized
        # ~625 nearby chunks (radius 12) * 15-25 plants = 9,375-15,625 rendered
        num_plants = rng.integers(15, 25)
        
        # Underwater plants  
        num_underwater = rng.integers(5, 12)
        
        for _ in range(num_plants + num_underwater):
            local_x = rng.integers(2, w - 2)
            local_z = rng.integers(2, h - 2)
            
            ground_h = heightmap[local_z, local_x]
            
            # World position
            world_x = chunk_world_x + local_x * tile_scale
            world_z = chunk_world_z + local_z * tile_scale
            
            # Pick species based on environment
            if ground_h < -2:  # Deep underwater - spawn underwater plants
                # Underwater plants: seaweed, coral
                underwater_types = [PlantType.SEAWEED, PlantType.CORAL]
                underwater_dnas = [d for d in chunk_dna_list if d.plant_type in underwater_types]
                if not underwater_dnas:
                    # Generate underwater DNA on the fly
                        dna = PlantDNA.create_random(PlantType.SEAWEED if rng.random() < 0.7 else PlantType.CORAL, int(rng.integers(0, 2**31)))
                else:
                    dna = rng.choice(underwater_dnas)
            elif ground_h < 1:  # Shallow water / shoreline
                # Mix of lily pads and shoreline plants
                shore_types = [PlantType.LILY_PAD, PlantType.SEAWEED, PlantType.FERN, PlantType.GRASS]
                shore_dnas = [d for d in chunk_dna_list if d.plant_type in shore_types]
                if not shore_dnas:
                    dna = PlantDNA.create_random(rng.choice([PlantType.LILY_PAD, PlantType.SEAWEED]), int(rng.integers(0, 2**31)))
                else:
                    dna = rng.choice(shore_dnas)
            elif ground_h > 40:  # High elevation - alpine zone
                # Prefer pines, crystals, grass, some hardy bushes
                alpine_types = [PlantType.GRASS, PlantType.PINE, PlantType.CRYSTAL, 
                               PlantType.BUSH, PlantType.SHRUB, PlantType.CACTUS,
                               PlantType.LICHEN, PlantType.MOSS_PAD]
                alpine_dnas = [d for d in chunk_dna_list if d.plant_type in alpine_types]
                if alpine_dnas:
                    dna = rng.choice(alpine_dnas)
                else:
                    dna = rng.choice(chunk_dna_list)
            elif ground_h > 30:  # Mountain zone
                # Mixed - some trees, mostly smaller plants
                mountain_types = [PlantType.PINE, PlantType.TREE, PlantType.BUSH, 
                                 PlantType.SHRUB, PlantType.GRASS, PlantType.FERN,
                                 PlantType.MOSS_PAD, PlantType.LICHEN]
                mountain_dnas = [d for d in chunk_dna_list if d.plant_type in mountain_types]
                if mountain_dnas:
                    dna = rng.choice(mountain_dnas)
                else:
                    dna = rng.choice(chunk_dna_list)
            else:
                # Normal terrain - occasionally spawn ground cover
                if rng.random() < 0.15:  # 15% chance of ground cover
                    ground_types = [PlantType.GROUNDCOVER, PlantType.CREEPER, 
                                   PlantType.LICHEN, PlantType.MOSS_PAD]
                    ground_dnas = [d for d in chunk_dna_list if d.plant_type in ground_types]
                    if ground_dnas:
                        dna = rng.choice(ground_dnas)
                    else:
                        dna = PlantDNA.create_random(rng.choice(ground_types), int(rng.integers(0, 2**31)))
                else:
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
