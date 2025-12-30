"""
Animal Rendering and AI System.

Handles:
- Rendering animals from AnimalDNA
- Articulated joint animation
- Simple AI behaviors (wander, flock, swarm, etc.)
- LOD rendering for distance
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from OpenGL.GL import *

from .animal_dna import (
    AnimalDNA, AnimalType, MovementType, AIBehavior,
    BodySegmentGene, LimbGene, FeatureGene
)


@dataclass
class AnimalInstance:
    """A specific animal at a world location."""
    x: float
    y: float
    z: float
    dna: AnimalDNA
    
    # Movement state
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    rotation: float = 0.0  # Y-axis rotation (facing direction)
    
    # Animation state
    anim_time: float = 0.0
    anim_phase: float = 0.0
    
    # AI state
    target_x: float = 0.0
    target_z: float = 0.0
    state: str = "idle"  # idle, moving, fleeing
    state_timer: float = 0.0
    
    # Visual scale (for baby animals that grow)
    scale: float = 1.0
    
    def update(self, dt: float, neighbors: List["AnimalInstance"] = None, 
               player_pos: Tuple[float, float, float] = None,
               get_ground_height=None):
        """Update animal position and AI."""
        # Normalize dt - dt is frame time in arbitrary units, convert to ~seconds
        # At 60fps, dt is usually around 1.0 (frame count), so divide by 60
        dt_seconds = dt / 60.0
        
        self.anim_time += dt_seconds * self.dna.animation_speed
        
        # Update AI behavior
        self._update_ai(dt_seconds, neighbors, player_pos)
        
        # Apply movement - speed is units per second
        speed = self.dna.movement_speed * 3.0  # Base speed multiplier
        
        if self.dna.movement_type == MovementType.FLY:
            # Flying - move in 3D with swooping, circling paths
            fly_speed = speed * 2.0  # Flying is fast!
            
            # Add swooping/circling behavior
            swoop = math.sin(self.anim_time * 2.0) * 0.4
            circle = math.cos(self.anim_time * 1.5) * 0.3
            
            self.x += (self.vx + circle) * fly_speed * dt_seconds
            self.y += (self.vy + swoop * 0.3) * fly_speed * dt_seconds
            self.z += (self.vz - circle) * fly_speed * dt_seconds
            
            # Keep above ground with dynamic flight height
            if get_ground_height:
                ground = get_ground_height(self.x, self.z)
                base_height = 8 + self.dna.base_scale * 4
                height_variation = math.sin(self.anim_time * 0.8) * 5
                min_height = ground + base_height + height_variation
                self.y = max(self.y, min_height)
                self.y = min(self.y, ground + 60)
        elif self.dna.movement_type == MovementType.SWIM:
            # Swimming - move in water
            swim_speed = speed * 1.5
            self.x += self.vx * swim_speed * dt_seconds
            self.z += self.vz * swim_speed * dt_seconds
            # Undulate up/down while swimming
            self.y = 0.5 + math.sin(self.anim_time * 3) * 0.3
        elif self.dna.movement_type == MovementType.FLOAT:
            # Floating - gentle drift (metroids, jellyfish)
            float_speed = speed * 0.5
            self.x += self.vx * float_speed * dt_seconds
            self.z += self.vz * float_speed * dt_seconds
            # Gentle bobbing
            self.y += math.sin(self.anim_time * 1.5) * 0.02
            if get_ground_height:
                ground = get_ground_height(self.x, self.z)
                self.y = max(self.y, ground + 3)
        elif self.dna.movement_type == MovementType.HOP:
            # Hopping movement - parabolic jumps
            hop_speed = speed * 1.2
            self.x += self.vx * hop_speed * dt_seconds
            self.z += self.vz * hop_speed * dt_seconds
            
            if get_ground_height:
                ground = get_ground_height(self.x, self.z)
                # Create hopping arc - sin wave offset from ground
                hop_phase = (self.anim_time * self.dna.animation_speed * 3) % (2 * math.pi)
                hop_height = max(0, math.sin(hop_phase)) * (0.5 + self.dna.base_scale * 0.5)
                self.y = ground + hop_height
        else:
            # Ground movement (walking, crawling) - increased speed!
            walk_speed = speed * 2.5  # Much faster ground movement
            self.x += self.vx * walk_speed * dt_seconds
            self.z += self.vz * walk_speed * dt_seconds
            if get_ground_height:
                self.y = get_ground_height(self.x, self.z)
        
        # Update rotation to face movement direction - smooth interpolation
        if abs(self.vx) > 0.01 or abs(self.vz) > 0.01:
            target_rot = math.degrees(math.atan2(self.vx, self.vz))
            # Very smooth rotation using lerp
            diff = target_rot - self.rotation
            while diff > 180: diff -= 360
            while diff < -180: diff += 360
            # Smooth lerp factor - faster for big differences, slower for small
            lerp_speed = min(0.15, 0.05 + abs(diff) * 0.002)
            self.rotation += diff * lerp_speed
    
    def _update_ai(self, dt: float, neighbors: List["AnimalInstance"], 
                   player_pos: Tuple[float, float, float]):
        """Update AI behavior."""
        self.state_timer -= dt
        
        behavior = self.dna.ai_behavior
        
        if behavior == AIBehavior.WANDER:
            self._ai_wander(dt)
        elif behavior == AIBehavior.GRAZE:
            self._ai_graze(dt)
        elif behavior == AIBehavior.FLOCK:
            self._ai_flock(dt, neighbors)
        elif behavior == AIBehavior.SWARM:
            self._ai_swarm(dt, neighbors)
        elif behavior == AIBehavior.SCHOOL:
            self._ai_school(dt, neighbors)
        elif behavior == AIBehavior.FLEE:
            self._ai_flee(dt, player_pos)
        else:
            self._ai_wander(dt)
    
    def _ai_wander(self, dt: float):
        """Random wandering behavior."""
        # Flying creatures are more active
        is_flying = self.dna.movement_type == MovementType.FLY
        
        if self.state_timer <= 0:
            if self.state == "idle":
                # Start moving to new position
                angle = np.random.random() * 2 * math.pi
                # Flying creatures travel further
                dist = (10 + np.random.random() * 30) if is_flying else (5 + np.random.random() * 15)
                self.target_x = self.x + math.cos(angle) * dist
                self.target_z = self.z + math.sin(angle) * dist
                self.state = "moving"
                # Flying creatures don't stop as often
                self.state_timer = (5 + np.random.random() * 10) if is_flying else (3 + np.random.random() * 5)
                
                # Flying creatures also adjust altitude
                if is_flying:
                    self.vy = (np.random.random() - 0.5) * 0.5
            else:
                # Stop and rest - flying creatures rest less
                self.state = "idle"
                self.state_timer = (0.5 + np.random.random() * 1) if is_flying else (1 + np.random.random() * 3)
                if not is_flying:
                    self.vx = 0
                    self.vz = 0
                else:
                    # Flying creatures keep drifting slightly
                    self.vx *= 0.3
                    self.vz *= 0.3
                    self.vy = 0
        
        if self.state == "moving":
            # Move toward target
            dx = self.target_x - self.x
            dz = self.target_z - self.z
            dist = math.sqrt(dx*dx + dz*dz)
            if dist > 0.5:
                self.vx = dx / dist
                self.vz = dz / dist
            else:
                self.state = "idle"
                self.state_timer = 0.5 if is_flying else 1
    
    def _ai_graze(self, dt: float):
        """Grazing behavior - occasional movement with grazing pauses."""
        if self.state_timer <= 0:
            if np.random.random() < 0.6:  # 60% chance to move (was 30%)
                # Move a short to medium distance
                angle = np.random.random() * 2 * math.pi
                dist = 3 + np.random.random() * 8  # Move further
                self.target_x = self.x + math.cos(angle) * dist
                self.target_z = self.z + math.sin(angle) * dist
                self.state = "moving"
                self.state_timer = 3 + np.random.random() * 4
            else:
                self.state = "idle"
                self.state_timer = 1 + np.random.random() * 2  # Shorter idle (was 3-8s)
                self.vx = 0
                self.vz = 0
        
        if self.state == "moving":
            dx = self.target_x - self.x
            dz = self.target_z - self.z
            dist = math.sqrt(dx*dx + dz*dz)
            if dist > 0.3:
                self.vx = dx / dist  # Full speed (was * 0.5)
                self.vz = dz / dist
            else:
                self.state = "idle"
    
    def _ai_flock(self, dt: float, neighbors: List["AnimalInstance"]):
        """Bird-like flocking behavior - more active movement."""
        if not neighbors:
            self._ai_wander(dt)
            return
        
        # Get nearby same-species neighbors
        flock = [n for n in neighbors 
                 if n.dna.animal_type == self.dna.animal_type
                 and self._distance_to(n) < self.dna.awareness_radius]
        
        if len(flock) < 2:
            self._ai_wander(dt)
            return
        
        # Separation - avoid crowding
        sep_x, sep_z = 0, 0
        for n in flock:
            dx = self.x - n.x
            dz = self.z - n.z
            dist = max(0.1, math.sqrt(dx*dx + dz*dz))
            if dist < 3:  # Increased separation distance
                sep_x += dx / dist
                sep_z += dz / dist
        
        # Alignment - match velocity of neighbors
        align_x = sum(n.vx for n in flock) / len(flock)
        align_z = sum(n.vz for n in flock) / len(flock)
        
        # Cohesion - move toward center of flock
        center_x = sum(n.x for n in flock) / len(flock)
        center_z = sum(n.z for n in flock) / len(flock)
        coh_x = center_x - self.x
        coh_z = center_z - self.z
        
        # Add some forward momentum so flock keeps moving
        forward_bias = 0.5
        
        # Combine forces - stronger alignment for more coordinated movement
        self.vx = sep_x * 0.25 + align_x * 0.5 + coh_x * 0.15 + forward_bias * math.cos(self.rotation * math.pi / 180)
        self.vz = sep_z * 0.25 + align_z * 0.5 + coh_z * 0.15 + forward_bias * math.sin(self.rotation * math.pi / 180)
        
        # Normalize but keep minimum speed
        speed = math.sqrt(self.vx**2 + self.vz**2)
        if speed > 0.1:
            self.vx /= speed
            self.vz /= speed
        else:
            # If too slow, pick a random direction
            angle = np.random.random() * 2 * math.pi
            self.vx = math.cos(angle)
            self.vz = math.sin(angle)
        
        # Flying animals also adjust Y with some variation
        if self.dna.movement_type == MovementType.FLY:
            avg_y = sum(n.y for n in flock) / len(flock)
            self.vy = (avg_y - self.y) * 0.15 + (np.random.random() - 0.5) * 0.1
    
    def _ai_swarm(self, dt: float, neighbors: List["AnimalInstance"]):
        """Insect-like swarming - more chaotic than flocking."""
        self._ai_flock(dt, neighbors)
        
        # Add randomness
        self.vx += (np.random.random() - 0.5) * 0.5
        self.vz += (np.random.random() - 0.5) * 0.5
        
        # Normalize
        speed = math.sqrt(self.vx**2 + self.vz**2)
        if speed > 0.1:
            self.vx /= speed
            self.vz /= speed
    
    def _ai_school(self, dt: float, neighbors: List["AnimalInstance"]):
        """Fish schooling - like flocking but in water."""
        self._ai_flock(dt, neighbors)
        self.vy = 0  # Stay at water level
    
    def _ai_flee(self, dt: float, player_pos: Tuple[float, float, float]):
        """Flee from player if too close."""
        if player_pos is None:
            self._ai_graze(dt)
            return
        
        dx = self.x - player_pos[0]
        dz = self.z - player_pos[2]
        dist = math.sqrt(dx*dx + dz*dz)
        
        if dist < self.dna.flee_radius:
            # Run away!
            self.state = "fleeing"
            if dist > 0.1:
                self.vx = dx / dist * 1.5
                self.vz = dz / dist * 1.5
        else:
            if self.state == "fleeing":
                self.state = "idle"
                self.state_timer = 2
            self._ai_graze(dt)
    
    def _distance_to(self, other: "AnimalInstance") -> float:
        """Calculate distance to another animal."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)


class AnimalRenderer:
    """Renders animals from DNA with articulated joints."""
    
    @staticmethod
    def draw_full(animal: AnimalInstance, at_origin: bool = False):
        """Draw animal at full detail with animation.
        
        Args:
            animal: The animal instance to render
            at_origin: If True, render at origin (0,0,0) instead of animal's position
        """
        glPushMatrix()
        if not at_origin:
            glTranslatef(animal.x, animal.y, animal.z)
            glRotatef(animal.rotation, 0, 1, 0)
        
        scale = animal.dna.base_scale
        glScalef(scale, scale, scale)
        
        dna = animal.dna
        anim_t = animal.anim_time
        
        # Draw body segments
        offset = 0
        for i, seg in enumerate(dna.body_segments):
            glPushMatrix()
            glTranslatef(-offset, 0, 0)
            
            # Body segment animation (breathing, undulation)
            if dna.movement_type == MovementType.CRAWL:
                # Strong side-to-side undulation for snakes/worms
                wave_freq = 4.0  # Speed of wave
                wave_amp = 0.15 * (1 + i * 0.1)  # Amplitude increases toward tail
                phase_offset = i * 1.2  # Phase difference between segments
                
                side_wave = math.sin(anim_t * wave_freq + phase_offset) * wave_amp
                # Slight vertical following ground contour
                vert_wave = abs(math.sin(anim_t * wave_freq * 0.5 + phase_offset)) * 0.05
                glTranslatef(0, vert_wave, side_wave)
            
            AnimalRenderer._draw_segment(seg)
            offset += seg.size[0] * 0.7
            glPopMatrix()
        
        # Draw limbs
        if dna.limbs:
            AnimalRenderer._draw_limbs(dna, anim_t)
        
        # Draw features
        for feat in dna.features:
            AnimalRenderer._draw_feature(feat, dna.body_segments[0].size if dna.body_segments else (0.5, 0.3, 0.3))
        
        glPopMatrix()
    
    @staticmethod
    def draw_simple(animal: AnimalInstance):
        """Draw simplified animal for medium distance."""
        glPushMatrix()
        glTranslatef(animal.x, animal.y, animal.z)
        glRotatef(animal.rotation, 0, 1, 0)
        
        scale = animal.dna.base_scale
        glScalef(scale, scale, scale)
        
        dna = animal.dna
        
        # Just draw main body as ellipsoid
        if dna.body_segments:
            seg = dna.body_segments[0]
            glColor3f(*seg.color[:3] if len(seg.color) >= 3 else seg.color)
            
            # Simple box approximation
            w, h, d = seg.size
            glBegin(GL_QUADS)
            # Top
            glVertex3f(-w/2, h/2, -d/2)
            glVertex3f(w/2, h/2, -d/2)
            glVertex3f(w/2, h/2, d/2)
            glVertex3f(-w/2, h/2, d/2)
            # Front
            glVertex3f(-w/2, -h/2, d/2)
            glVertex3f(w/2, -h/2, d/2)
            glVertex3f(w/2, h/2, d/2)
            glVertex3f(-w/2, h/2, d/2)
            # Sides
            glVertex3f(w/2, -h/2, -d/2)
            glVertex3f(w/2, -h/2, d/2)
            glVertex3f(w/2, h/2, d/2)
            glVertex3f(w/2, h/2, -d/2)
            glEnd()
        
        glPopMatrix()
    
    @staticmethod
    def draw_point(animal: AnimalInstance):
        """Draw animal as point for far distance."""
        color = animal.dna.primary_color
        glColor3f(*color)
        glVertex3f(animal.x, animal.y + animal.dna.base_scale * 0.3, animal.z)
    
    @staticmethod
    def draw_billboard(animal: AnimalInstance, scale: float = 1.0):
        """Draw animal as a simple 2D shape that matches its silhouette.
        
        Different animal types get different shapes:
        - Birds/flyers: Diamond/wing shape
        - Fish/swimmers: Horizontal ellipse
        - Spiders/bugs: Wide low shape
        - Mammals/walkers: Rounded rectangle
        - Snakes/worms: Horizontal line
        """
        atype = animal.dna.animal_type
        # Scale down to match actual visual size (0.5 factor)
        base_size = animal.dna.base_scale * 0.5 * scale
        
        # Position
        cx, cy, cz = animal.x, animal.y + animal.dna.base_scale * 0.15, animal.z
        
        # Color
        color = animal.dna.primary_color
        glColor3f(*color)
        
        if atype in ('bird', 'moth', 'butterfly', 'bat'):
            # Diamond/wing shape (wide horizontal)
            hw = base_size * 1.2
            hh = base_size * 0.4
            glBegin(GL_QUADS)
            glVertex3f(cx, cy + hh, cz)  # Top
            glVertex3f(cx - hw, cy, cz)  # Left
            glVertex3f(cx, cy - hh, cz)  # Bottom  
            glVertex3f(cx + hw, cy, cz)  # Right
            glEnd()
            
        elif atype in ('fish', 'shark', 'whale', 'manta'):
            # Fish shape - horizontal diamond with tail
            hw = base_size * 1.0
            hh = base_size * 0.35
            glBegin(GL_TRIANGLES)
            # Body
            glVertex3f(cx + hw, cy, cz)  # Nose
            glVertex3f(cx - hw * 0.3, cy + hh, cz)
            glVertex3f(cx - hw * 0.3, cy - hh, cz)
            # Tail
            glVertex3f(cx - hw * 0.3, cy, cz)
            glVertex3f(cx - hw, cy + hh * 0.8, cz)
            glVertex3f(cx - hw, cy - hh * 0.8, cz)
            glEnd()
            
        elif atype in ('spider', 'crab', 'scorpion'):
            # Wide low shape
            hw = base_size * 0.8
            hh = base_size * 0.25
            glBegin(GL_QUADS)
            glVertex3f(cx - hw, cy + hh, cz)
            glVertex3f(cx + hw, cy + hh, cz)
            glVertex3f(cx + hw, cy - hh, cz)
            glVertex3f(cx - hw, cy - hh, cz)
            glEnd()
            
        elif atype in ('snake', 'worm', 'eel'):
            # Long horizontal line
            hw = base_size * 1.5
            hh = base_size * 0.15
            glBegin(GL_QUADS)
            glVertex3f(cx - hw, cy + hh, cz)
            glVertex3f(cx + hw, cy + hh, cz)
            glVertex3f(cx + hw, cy - hh, cz)
            glVertex3f(cx - hw, cy - hh, cz)
            glEnd()
            
        elif atype in ('jellyfish', 'octopus', 'squid'):
            # Dome with tentacles
            hw = base_size * 0.5
            hh = base_size * 0.4
            # Dome
            glBegin(GL_TRIANGLES)
            glVertex3f(cx, cy + hh, cz)
            glVertex3f(cx - hw, cy, cz)
            glVertex3f(cx + hw, cy, cz)
            glEnd()
            # Tentacles (small triangle below)
            glBegin(GL_TRIANGLES)
            glVertex3f(cx, cy - hh * 1.5, cz)
            glVertex3f(cx - hw * 0.8, cy, cz)
            glVertex3f(cx + hw * 0.8, cy, cz)
            glEnd()
            
        else:
            # Default: rounded rectangle (generic quadruped)
            hw = base_size * 0.6
            hh = base_size * 0.4
            glBegin(GL_QUADS)
            glVertex3f(cx - hw, cy + hh, cz)
            glVertex3f(cx + hw, cy + hh, cz)
            glVertex3f(cx + hw, cy - hh, cz)
            glVertex3f(cx - hw, cy - hh, cz)
            glEnd()
    
    @staticmethod
    def _draw_segment(seg: BodySegmentGene):
        """Draw a body segment."""
        color = seg.color[:3] if len(seg.color) >= 3 else seg.color
        glColor3f(*color)
        
        w, h, d = seg.size
        
        if seg.shape == "ellipsoid":
            AnimalRenderer._draw_ellipsoid(w/2, h/2, d/2, 8)
        elif seg.shape == "box":
            AnimalRenderer._draw_box(w, h, d)
        elif seg.shape == "cylinder":
            AnimalRenderer._draw_cylinder(w/2, h, 8)
        elif seg.shape == "cone":
            AnimalRenderer._draw_cone(w/2, h, 8)
        else:
            AnimalRenderer._draw_ellipsoid(w/2, h/2, d/2, 8)
    
    @staticmethod
    def _draw_limbs(dna: AnimalDNA, anim_t: float):
        """Draw all limbs with animation."""
        if not dna.limbs:
            return
        
        body_size = dna.body_segments[0].size if dna.body_segments else (0.5, 0.3, 0.3)
        
        for limb_idx, limb in enumerate(dna.limbs):
            for pair in range(dna.limb_pairs):
                for side in [-1, 1]:  # Left and right
                    glPushMatrix()
                    
                    # Position limb on body
                    limb_x = -body_size[0] * 0.2 * (pair - dna.limb_pairs / 2)
                    limb_y = -body_size[1] * 0.3
                    limb_z = side * body_size[2] * 0.5
                    
                    glTranslatef(limb_x, limb_y, limb_z)
                    
                    # Animate based on limb type
                    if limb.limb_type == "leg":
                        # Walking animation
                        phase = pair * 0.5 + (0.5 if side < 0 else 0)
                        swing = math.sin(anim_t * 4 + phase * math.pi) * limb.joint_range
                        glRotatef(swing, 0, 0, 1)
                    elif limb.limb_type == "wing":
                        # Flapping animation
                        flap = math.sin(anim_t * 8) * 45
                        glRotatef(flap * side, 1, 0, 0)
                    elif limb.limb_type == "fin":
                        # Swimming animation
                        wave = math.sin(anim_t * 5 + pair) * 20
                        glRotatef(wave, 0, 1, 0)
                    elif limb.limb_type == "tentacle":
                        # Wavy tentacle animation
                        wave = math.sin(anim_t * 2 + pair * 0.5) * 30
                        glRotatef(wave, 0, 0, 1)
                    
                    # Draw limb segments
                    AnimalRenderer._draw_limb_segments(limb, anim_t, pair)
                    
                    glPopMatrix()
    
    @staticmethod
    def _draw_limb_segments(limb: LimbGene, anim_t: float, pair_idx: int):
        """Draw segments of a limb."""
        glColor3f(*limb.color)
        
        for i, (length, width) in enumerate(zip(limb.segment_lengths, limb.segment_widths)):
            # Joint rotation for animation
            if i > 0:
                joint_anim = math.sin(anim_t * 3 + i * 0.5 + pair_idx) * limb.joint_range * 0.5
                glRotatef(joint_anim, 0, 0, 1)
            
            # Draw segment as tapered cylinder
            glBegin(GL_QUAD_STRIP)
            segments = 6
            for j in range(segments + 1):
                angle = (j / segments) * 2 * math.pi
                x0 = math.cos(angle) * width
                z0 = math.sin(angle) * width
                x1 = math.cos(angle) * width * 0.7
                z1 = math.sin(angle) * width * 0.7
                
                glVertex3f(x0, 0, z0)
                glVertex3f(x1, -length, z1)
            glEnd()
            
            # Move to end of segment
            glTranslatef(0, -length, 0)
        
        # Draw membrane for wings
        if limb.membrane and limb.limb_type == "wing":
            glColor4f(*limb.membrane_color[:3], 0.5 if len(limb.membrane_color) < 4 else limb.membrane_color[3])
            glEnable(GL_BLEND)
            
            total_len = sum(limb.segment_lengths)
            glBegin(GL_TRIANGLES)
            glVertex3f(0, 0, 0)
            glVertex3f(0, total_len * 0.8, total_len * 0.5)
            glVertex3f(0, total_len, 0)
            glEnd()
            
            glDisable(GL_BLEND)
    
    @staticmethod
    def _draw_feature(feat: FeatureGene, body_size: Tuple[float, float, float]):
        """Draw a feature (eyes, mouth, etc.)."""
        glColor3f(*feat.color)
        
        for i in range(feat.count):
            glPushMatrix()
            
            # Position based on feature position and count
            if feat.count > 1:
                # Spread features (e.g., two eyes)
                offset = (i - (feat.count - 1) / 2) * feat.size * 2.5
                glTranslatef(
                    feat.position[0] * body_size[0],
                    feat.position[1] * body_size[1],
                    feat.position[2] * body_size[2] + offset
                )
            else:
                glTranslatef(
                    feat.position[0] * body_size[0],
                    feat.position[1] * body_size[1],
                    feat.position[2] * body_size[2]
                )
            
            if feat.feature_type == "eye":
                # Draw eye as sphere
                AnimalRenderer._draw_ellipsoid(feat.size, feat.size, feat.size * 0.8, 6)
                # Pupil
                glColor3f(0.02, 0.02, 0.02)
                glTranslatef(feat.size * 0.5, 0, 0)
                AnimalRenderer._draw_ellipsoid(feat.size * 0.4, feat.size * 0.4, feat.size * 0.3, 4)
            elif feat.feature_type == "mouth":
                # Draw mouth as curved line
                glBegin(GL_LINE_STRIP)
                for j in range(5):
                    t = j / 4
                    x = t * feat.size
                    y = math.sin(t * math.pi) * feat.size * 0.3
                    glVertex3f(x, y, 0)
                glEnd()
            elif feat.feature_type == "antenna":
                # Draw antenna as thin cone
                AnimalRenderer._draw_cone(feat.size * 0.1, feat.size, 4)
            elif feat.feature_type == "horn":
                # Draw horn as cone
                glRotatef(-30, 0, 0, 1)
                AnimalRenderer._draw_cone(feat.size * 0.2, feat.size, 6)
            elif feat.feature_type == "tail":
                # Draw tail as tapered cylinder
                glRotatef(90, 0, 0, 1)
                for j in range(3):
                    seg_len = feat.size / 3
                    seg_width = feat.size * 0.15 * (1 - j * 0.25)
                    AnimalRenderer._draw_cylinder(seg_width, seg_len, 4)
                    glTranslatef(0, seg_len, 0)
            
            glPopMatrix()
    
    @staticmethod
    def _draw_ellipsoid(rx: float, ry: float, rz: float, slices: int):
        """Draw an ellipsoid."""
        for i in range(slices):
            lat0 = math.pi * (-0.5 + float(i) / slices)
            lat1 = math.pi * (-0.5 + float(i + 1) / slices)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * float(j) / slices
                
                x0 = math.cos(lat0) * math.cos(lng) * rx
                y0 = math.sin(lat0) * ry
                z0 = math.cos(lat0) * math.sin(lng) * rz
                
                x1 = math.cos(lat1) * math.cos(lng) * rx
                y1 = math.sin(lat1) * ry
                z1 = math.cos(lat1) * math.sin(lng) * rz
                
                glVertex3f(x0, y0, z0)
                glVertex3f(x1, y1, z1)
            glEnd()
    
    @staticmethod
    def _draw_box(w: float, h: float, d: float):
        """Draw a box."""
        w, h, d = w/2, h/2, d/2
        glBegin(GL_QUADS)
        # Top
        glVertex3f(-w, h, -d); glVertex3f(w, h, -d)
        glVertex3f(w, h, d); glVertex3f(-w, h, d)
        # Bottom
        glVertex3f(-w, -h, -d); glVertex3f(-w, -h, d)
        glVertex3f(w, -h, d); glVertex3f(w, -h, -d)
        # Front
        glVertex3f(-w, -h, d); glVertex3f(-w, h, d)
        glVertex3f(w, h, d); glVertex3f(w, -h, d)
        # Back
        glVertex3f(-w, -h, -d); glVertex3f(w, -h, -d)
        glVertex3f(w, h, -d); glVertex3f(-w, h, -d)
        # Left
        glVertex3f(-w, -h, -d); glVertex3f(-w, h, -d)
        glVertex3f(-w, h, d); glVertex3f(-w, -h, d)
        # Right
        glVertex3f(w, -h, -d); glVertex3f(w, -h, d)
        glVertex3f(w, h, d); glVertex3f(w, h, -d)
        glEnd()
    
    @staticmethod
    def _draw_cylinder(radius: float, height: float, slices: int):
        """Draw a cylinder."""
        glBegin(GL_QUAD_STRIP)
        for i in range(slices + 1):
            angle = (i / slices) * 2 * math.pi
            x = math.cos(angle) * radius
            z = math.sin(angle) * radius
            glVertex3f(x, 0, z)
            glVertex3f(x, height, z)
        glEnd()
    
    @staticmethod
    def _draw_cone(radius: float, height: float, slices: int):
        """Draw a cone."""
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, height, 0)
        for i in range(slices + 1):
            angle = (i / slices) * 2 * math.pi
            x = math.cos(angle) * radius
            z = math.sin(angle) * radius
            glVertex3f(x, 0, z)
        glEnd()


class AnimalManager:
    """Manages animal spawning, AI updates, and rendering."""
    
    # Increased LOD distances for better visibility
    LOD_FULL = 80      # Full detail
    LOD_SIMPLE = 180   # Simplified geometry  
    LOD_POINT = 350    # Point sprites (visible from far)
    
    # Spatial indexing for O(1) neighbor lookups
    USE_SPATIAL_INDEX = True
    SPATIAL_CELL_SIZE = 30.0  # Cell size for spatial hashing
    
    def __init__(self, world_seed: int = 42):
        self.world_seed = world_seed
        self.animals: List[AnimalInstance] = []
        self.chunk_animals: Dict[Tuple[int, int], List[AnimalInstance]] = {}
        
        # Spatial index for O(1) neighbor lookups
        self._spatial_index = None
        if self.USE_SPATIAL_INDEX:
            from .spatial import SpatialIndex
            self._spatial_index = SpatialIndex(cell_size=self.SPATIAL_CELL_SIZE)
        
        # Pre-generate species templates
        self.species_templates = self._generate_species()
    
    def _generate_species(self) -> Dict[str, List[AnimalDNA]]:
        """Generate base species templates - HIGH DIVERSITY!"""
        templates = {}
        for i, animal_type in enumerate(AnimalType.ALL):
            templates[animal_type] = [
                AnimalDNA.create_random(animal_type, self.world_seed + i * 100 + j)
                for j in range(5)  # 5 variations per type (was 2)
            ]
        return templates
    
    # Animal cap - balanced for performance
    MAX_TOTAL_ANIMALS = 200  # Reduced from 350 for stable FPS
    
    def spawn_animals_for_chunk(self, cx: int, cz: int, heightmap, 
                                 chunk_world_x: float, chunk_world_z: float,
                                 tile_scale: float, height_scale: float,
                                 biome_name: str = None):
        """Spawn animals for a chunk.
        
        Args:
            biome_name: Optional biome for adjusting animal density/types.
                       Rainforest/Tropical = many animals
                       Desert = few, hardy animals
                       Tundra = sparse arctic creatures
        """
        key = (cx, cz)
        if key in self.chunk_animals:
            return
        
        # Don't spawn if already at capacity
        if len(self.animals) >= self.MAX_TOTAL_ANIMALS:
            return
        
        chunk_seed = abs(hash((self.world_seed, cx, cz, "animals"))) % (2**31)
        rng = np.random.default_rng(chunk_seed)
        
        h, w = heightmap.shape
        animals = []
        
        # BIOME-BASED animal density!
        biome = (biome_name or '').lower()
        
        # === SPECIAL EXOTIC BIOMES - counts halved for performance ===
        if biome == 'psychedelic':
            num_animals = rng.integers(4, 8)
        elif biome == 'hellfire':
            num_animals = rng.integers(2, 4)
        elif biome == 'shadow':
            num_animals = rng.integers(1, 3)
        elif biome == 'crystal':
            num_animals = rng.integers(1, 2)
        elif biome == 'void':
            num_animals = rng.integers(0, 1)
        # === NORMAL BIOMES - halved ===
        elif biome in ('rainforest', 'tropical'):
            num_animals = rng.integers(2, 5)
        elif biome == 'savanna':
            num_animals = rng.integers(2, 4)
        elif biome in ('temperate', 'taiga'):
            num_animals = rng.integers(1, 3)
        elif biome == 'desert':
            num_animals = rng.integers(0, 2)
        elif biome in ('tundra', 'frozen'):
            num_animals = rng.integers(0, 2)
        elif biome == 'swamp':
            num_animals = rng.integers(2, 4)
        elif biome in ('ocean', 'underwater', 'coral_reef'):
            num_animals = rng.integers(3, 6)
        elif biome == 'meadow':
            num_animals = rng.integers(2, 4)
        elif biome in ('forest', 'deciduous'):
            num_animals = rng.integers(2, 4)
        elif biome == 'volcanic':
            num_animals = rng.integers(0, 2)
        else:
            num_animals = rng.integers(1, 3)
        
        for _ in range(num_animals):
            local_x = rng.integers(5, w - 5)
            local_z = rng.integers(5, h - 5)
            
            ground_h = heightmap[local_z, local_x]
            
            # Determine what can spawn here - favor GROUND animals heavily
            if ground_h < 2:
                # Near/in water - aquatic and amphibious creatures!
                animal_type = rng.choice([
                    # Amphibious
                    AnimalType.CROC, AnimalType.CROC,
                    AnimalType.HOPPER,  # Frogs
                    AnimalType.AMPHIBIAN,
                    AnimalType.TURTLE, AnimalType.TURTLE,  # Sea turtles!
                    # Aquatic - diverse sea life
                    AnimalType.OCTOPUS, AnimalType.OCTOPUS,
                    AnimalType.SQUID, AnimalType.CEPHALOPOD,
                    AnimalType.JELLYFISH, AnimalType.JELLYFISH,
                    AnimalType.FISH, AnimalType.FISH, AnimalType.FISH,
                    AnimalType.CRUSTACEAN, AnimalType.CRUSTACEAN,
                    AnimalType.ANEMONE,
                    AnimalType.SEASTAR,
                    AnimalType.NUDIBRANCH,
                    AnimalType.NAUTILUS,
                    AnimalType.MANTA,
                    AnimalType.HYDRA,
                    AnimalType.SERPENT,  # Sea serpents!
                    AnimalType.PENGUIN,  # Penguins near water
                    # Amoebas love water
                    AnimalType.AMOEBA, AnimalType.AMOEBA,
                    AnimalType.BLOB,
                ])
            elif ground_h < 15:
                # Lowlands - TONS of ground variety including new types!
                animal_type = rng.choice([
                    # Big quadrupeds
                    AnimalType.MAMMAL, AnimalType.MAMMAL, AnimalType.MAMMAL,
                    AnimalType.DINOSAUR, AnimalType.DINOSAUR,
                    AnimalType.CROC, AnimalType.CROC,
                    AnimalType.TURTLE,  # Tortoises
                    # Bipeds!
                    AnimalType.RAPTOR, AnimalType.RAPTOR,  # Fast predator dinosaurs!
                    AnimalType.GORILLA, AnimalType.GORILLA,  # Apes!
                    # Medium ground animals
                    AnimalType.REPTILE, AnimalType.REPTILE, AnimalType.REPTILE,
                    AnimalType.HOPPER, AnimalType.HOPPER,  # Rabbits, frogs
                    AnimalType.SERPENT, AnimalType.SERPENT,  # Big snakes!
                    AnimalType.WORM, AnimalType.WORM, AnimalType.WORM,  # Snakes/worms
                    # Crawlers - spiders, scorpions, centipedes, beetles, mantises!
                    AnimalType.SPIDER, AnimalType.SPIDER, AnimalType.SPIDER,
                    AnimalType.SCORPION, AnimalType.SCORPION,  # Scorpions!
                    AnimalType.CENTIPEDE, AnimalType.CENTIPEDE,
                    AnimalType.MILLIPEDE,  # Millipedes!
                    AnimalType.BEETLE, AnimalType.BEETLE,  # Beetles!
                    AnimalType.MANTIS,  # Praying mantis!
                    AnimalType.TRILOBITE, AnimalType.TRILOBITE,
                    AnimalType.INSECT, AnimalType.INSECT,
                    AnimalType.SNAIL, AnimalType.SNAIL,
                    # Flyers - birds, bats, moths, pterosaurs!
                    AnimalType.BIRD, AnimalType.BIRD, AnimalType.BIRD,
                    AnimalType.BAT, AnimalType.BAT,  # Bats!
                    AnimalType.MOTH,  # Moths!
                    AnimalType.PTEROSAUR,  # Flying dinosaurs!
                    AnimalType.METROID,  # Rare floating horror
                    # Weird creatures
                    AnimalType.AMOEBA,
                    AnimalType.BLOB,
                    AnimalType.ALIEN,
                ])
            elif ground_h < 30:
                # Hills - diverse ground animals with more variety
                animal_type = rng.choice([
                    # Quadrupeds
                    AnimalType.MAMMAL, AnimalType.MAMMAL,
                    AnimalType.DINOSAUR, AnimalType.DINOSAUR,
                    AnimalType.REPTILE, AnimalType.REPTILE,
                    AnimalType.TURTLE,  # Mountain tortoises
                    # Bipeds
                    AnimalType.RAPTOR, AnimalType.RAPTOR,
                    AnimalType.GORILLA,
                    AnimalType.HOPPER, AnimalType.HOPPER,
                    # Crawlers
                    AnimalType.SPIDER, AnimalType.SPIDER,
                    AnimalType.SCORPION,  # Hill scorpions
                    AnimalType.CENTIPEDE, AnimalType.CENTIPEDE,
                    AnimalType.BEETLE,  # Mountain beetles
                    AnimalType.SERPENT, AnimalType.SERPENT,  # Hill snakes
                    AnimalType.WORM, AnimalType.WORM,
                    # Flyers
                    AnimalType.BIRD, AnimalType.BIRD, AnimalType.BIRD,
                    AnimalType.BAT,
                    AnimalType.PTEROSAUR,
                    AnimalType.INSECT, AnimalType.INSECT,
                    AnimalType.MOTH,
                    AnimalType.TRILOBITE,
                    AnimalType.SNAIL,
                ])
            else:
                # High ground - hardy mammals, birds, bats, some hoppers, mountain creatures
                animal_type = rng.choice([
                    AnimalType.MAMMAL, AnimalType.MAMMAL, AnimalType.MAMMAL,
                    AnimalType.GORILLA,  # Mountain gorillas!
                    AnimalType.HOPPER,  # Mountain goat-like
                    AnimalType.BIRD, AnimalType.BIRD, AnimalType.BIRD,
                    AnimalType.BAT, AnimalType.BAT,  # Mountain bats
                    AnimalType.PTEROSAUR,  # Soaring pterosaurs
                    AnimalType.GOLEM,  # Mountain golems!
                    AnimalType.METROID,  # They float up high
                    AnimalType.ALIEN,    # Aliens too
                ])
            
            # BIOME-SPECIFIC animal type overrides!
            # This adds biome flavor on top of height-based selection
            
            # === SPECIAL EXOTIC BIOMES ===
            if biome == 'psychedelic':
                # PSYCHEDELIC: Weird floaty creatures, aliens, blobs, metroids
                animal_type = rng.choice([
                    AnimalType.ALIEN, AnimalType.ALIEN, AnimalType.ALIEN,
                    AnimalType.METROID, AnimalType.METROID,
                    AnimalType.BLOB, AnimalType.BLOB,
                    AnimalType.AMOEBA, AnimalType.AMOEBA,
                    AnimalType.JELLYFISH, AnimalType.JELLYFISH,
                    AnimalType.OCTOPUS, AnimalType.CEPHALOPOD,
                    AnimalType.HYDRA,
                    AnimalType.NUDIBRANCH, AnimalType.NUDIBRANCH,
                    AnimalType.MOTH,  # Psychedelic moths!
                    AnimalType.GOLEM,  # Crystal golems
                ])
            elif biome == 'hellfire':
                # HELLFIRE: Fire creatures, demons, lava things
                animal_type = rng.choice([
                    AnimalType.DINOSAUR, AnimalType.DINOSAUR, AnimalType.DINOSAUR,  # Fire dragons!
                    AnimalType.RAPTOR, AnimalType.RAPTOR,  # Fire raptors!
                    AnimalType.REPTILE, AnimalType.REPTILE, AnimalType.REPTILE,
                    AnimalType.SERPENT, AnimalType.SERPENT,  # Fire serpents!
                    AnimalType.WORM, AnimalType.WORM,  # Fire worms
                    AnimalType.SPIDER, AnimalType.SPIDER,  # Hell spiders (giant!)
                    AnimalType.SCORPION, AnimalType.SCORPION,  # Fire scorpions!
                    AnimalType.ALIEN,  # Demons
                    AnimalType.CENTIPEDE,  # Fire centipedes
                    AnimalType.GOLEM,  # Lava golems!
                    AnimalType.TRILOBITE,
                ])
            elif biome == 'shadow':
                # SHADOW: Creepy crawlers, stalkers, dark things
                animal_type = rng.choice([
                    AnimalType.SPIDER, AnimalType.SPIDER, AnimalType.SPIDER,  # Giant shadow spiders!
                    AnimalType.SCORPION, AnimalType.SCORPION,  # Shadow scorpions
                    AnimalType.SERPENT, AnimalType.SERPENT,  # Shadow snakes
                    AnimalType.WORM, AnimalType.WORM,
                    AnimalType.CENTIPEDE, AnimalType.CENTIPEDE,
                    AnimalType.MILLIPEDE,  # Dark millipedes
                    AnimalType.BAT, AnimalType.BAT, AnimalType.BAT,  # Shadow bats!
                    AnimalType.ALIEN, AnimalType.ALIEN,  # Shadow demons
                    AnimalType.METROID,  # Shadow metroids
                    AnimalType.STALKER,  # Stalkers!
                    AnimalType.BLOB,  # Dark blobs
                ])
            elif biome == 'crystal':
                # CRYSTAL: Crystalline creatures, delicate things
                animal_type = rng.choice([
                    AnimalType.INSECT, AnimalType.INSECT, AnimalType.INSECT,
                    AnimalType.BEETLE, AnimalType.BEETLE,  # Crystal beetles
                    AnimalType.MOTH, AnimalType.MOTH,  # Crystal moths
                    AnimalType.JELLYFISH, AnimalType.JELLYFISH,
                    AnimalType.SNAIL, AnimalType.SNAIL,
                    AnimalType.AMOEBA,
                    AnimalType.NUDIBRANCH,
                    AnimalType.BIRD,  # Crystal birds
                    AnimalType.GOLEM,  # Crystal golems!
                ])
            elif biome == 'void':
                # VOID: Almost nothing, occasional horrors
                if rng.random() < 0.7:  # 70% chance - skip, nothing lives here
                    continue
                animal_type = rng.choice([
                    AnimalType.METROID, AnimalType.METROID,
                    AnimalType.ALIEN,
                    AnimalType.BLOB,
                    AnimalType.AMOEBA,
                    AnimalType.GOLEM,  # Void golems
                ])
            
            # === NORMAL BIOMES ===
            elif biome in ('rainforest', 'tropical'):
                # Jungle: birds, insects, reptiles, exotic creatures - VERY diverse!
                if rng.random() < 0.5:  # 50% chance to override with jungle creature
                    animal_type = rng.choice([
                        # Birds - lots of birds!
                        AnimalType.BIRD, AnimalType.BIRD, AnimalType.BIRD, AnimalType.BIRD,
                        # Insects - beetles, mantises, moths
                        AnimalType.INSECT, AnimalType.INSECT,
                        AnimalType.BEETLE, AnimalType.BEETLE,
                        AnimalType.MANTIS, AnimalType.MANTIS,  # Jungle mantises!
                        AnimalType.MOTH,  # Jungle moths
                        # Crawlers - spiders, scorpions, centipedes
                        AnimalType.SPIDER, AnimalType.SPIDER,
                        AnimalType.SCORPION,  # Jungle scorpions
                        AnimalType.CENTIPEDE,
                        AnimalType.MILLIPEDE,  # Jungle millipedes
                        # Reptiles and snakes
                        AnimalType.REPTILE, AnimalType.REPTILE,
                        AnimalType.SERPENT, AnimalType.SERPENT,  # Big snakes!
                        AnimalType.WORM,  # Snakes!
                        # Primates!
                        AnimalType.GORILLA, AnimalType.GORILLA,
                        AnimalType.HOPPER,  # Tree frogs
                        AnimalType.BAT,  # Fruit bats
                        AnimalType.ALIEN,  # Exotic creatures
                        AnimalType.OCTOPUS,  # Weird jungle things
                    ])
            elif biome == 'desert':
                # Desert: reptiles, insects, scorpions, desert creatures
                if rng.random() < 0.5:
                    animal_type = rng.choice([
                        AnimalType.REPTILE, AnimalType.REPTILE, AnimalType.REPTILE,
                        AnimalType.SERPENT,  # Big desert snakes
                        AnimalType.WORM,  # Sand snakes
                        AnimalType.INSECT, AnimalType.INSECT,
                        AnimalType.BEETLE, AnimalType.BEETLE,  # Desert beetles
                        AnimalType.SPIDER, AnimalType.SPIDER,
                        AnimalType.SCORPION, AnimalType.SCORPION, AnimalType.SCORPION,  # Lots of scorpions!
                        AnimalType.CENTIPEDE,
                        AnimalType.TURTLE,  # Desert tortoises
                    ])
            elif biome in ('tundra', 'frozen'):
                # Arctic: hardy mammals, birds, penguins
                if rng.random() < 0.5:
                    animal_type = rng.choice([
                        AnimalType.MAMMAL, AnimalType.MAMMAL, AnimalType.MAMMAL,
                        AnimalType.BIRD, AnimalType.BIRD,
                        AnimalType.PENGUIN, AnimalType.PENGUIN, AnimalType.PENGUIN,  # Penguins!
                        AnimalType.HOPPER,  # Arctic hares
                        AnimalType.BAT,  # Arctic bats (rare)
                    ])
            elif biome == 'savanna':
                # Savanna: herds of mammals, birds, predators
                if rng.random() < 0.5:
                    animal_type = rng.choice([
                        # Herds
                        AnimalType.MAMMAL, AnimalType.MAMMAL, AnimalType.MAMMAL, AnimalType.MAMMAL,
                        # Big predators
                        AnimalType.DINOSAUR, AnimalType.DINOSAUR,
                        AnimalType.RAPTOR, AnimalType.RAPTOR,  # Savanna raptors!
                        # Birds
                        AnimalType.BIRD, AnimalType.BIRD, AnimalType.BIRD,
                        # Others
                        AnimalType.REPTILE,
                        AnimalType.SERPENT,  # Savanna snakes
                        AnimalType.INSECT,
                        AnimalType.BEETLE,
                        AnimalType.SCORPION,  # Savanna scorpions
                        AnimalType.TURTLE,  # Tortoises
                    ])
            elif biome == 'swamp':
                # Swamp: amphibians, crocs, insects, lots of life!
                if rng.random() < 0.5:
                    animal_type = rng.choice([
                        AnimalType.CROC, AnimalType.CROC, AnimalType.CROC,
                        AnimalType.TURTLE, AnimalType.TURTLE,  # Swamp turtles!
                        AnimalType.HOPPER, AnimalType.HOPPER,  # Frogs
                        AnimalType.AMPHIBIAN, AnimalType.AMPHIBIAN,
                        AnimalType.INSECT, AnimalType.INSECT,
                        AnimalType.MOTH,  # Swamp moths
                        AnimalType.SERPENT,  # Swamp snakes
                        AnimalType.WORM,  # Water snakes
                        AnimalType.FISH,
                        AnimalType.SNAIL, AnimalType.SNAIL,  # Swamp snails
                        AnimalType.MILLIPEDE,  # Swamp millipedes
                    ])
            elif biome in ('ocean', 'underwater', 'coral_reef'):
                # Ocean: fish, cephalopods, crustaceans, marine life
                animal_type = rng.choice([
                    # Fish - lots of fish!
                    AnimalType.FISH, AnimalType.FISH, AnimalType.FISH, AnimalType.FISH,
                    # Cephalopods
                    AnimalType.OCTOPUS, AnimalType.OCTOPUS,
                    AnimalType.SQUID, AnimalType.SQUID,
                    AnimalType.CEPHALOPOD,
                    AnimalType.NAUTILUS,  # Nautilus!
                    # Rays and big swimmers
                    AnimalType.MANTA, AnimalType.MANTA,
                    # Jellies and weird things
                    AnimalType.JELLYFISH, AnimalType.JELLYFISH, AnimalType.JELLYFISH,
                    AnimalType.ANEMONE,
                    AnimalType.SEASTAR,
                    AnimalType.NUDIBRANCH, AnimalType.NUDIBRANCH,
                    # Crustaceans
                    AnimalType.CRUSTACEAN, AnimalType.CRUSTACEAN,
                    # Turtles
                    AnimalType.TURTLE, AnimalType.TURTLE,
                    # Dangerous
                    AnimalType.SERPENT,  # Sea serpents!
                    AnimalType.HYDRA,  # Deep sea horror
                ])
            elif biome == 'meadow':
                # Meadow: butterflies, bees, small mammals, birds
                if rng.random() < 0.5:
                    animal_type = rng.choice([
                        # Lots of flying insects!
                        AnimalType.INSECT, AnimalType.INSECT, AnimalType.INSECT, AnimalType.INSECT,
                        AnimalType.MOTH, AnimalType.MOTH, AnimalType.MOTH,  # Butterflies/moths
                        AnimalType.BEETLE,
                        # Birds
                        AnimalType.BIRD, AnimalType.BIRD, AnimalType.BIRD,
                        # Small mammals - rabbits, mice
                        AnimalType.HOPPER, AnimalType.HOPPER, AnimalType.HOPPER,  # Rabbits
                        AnimalType.MAMMAL,
                        # Others
                        AnimalType.SERPENT,  # Grass snakes
                        AnimalType.SNAIL,  # Garden snails
                    ])
            elif biome in ('forest', 'deciduous'):
                # Deciduous forest: deer, birds, foxes, woodland creatures
                if rng.random() < 0.5:
                    animal_type = rng.choice([
                        # Mammals - deer, bears, foxes
                        AnimalType.MAMMAL, AnimalType.MAMMAL, AnimalType.MAMMAL, AnimalType.MAMMAL,
                        AnimalType.GORILLA,  # Bears
                        # Birds - owls, woodpeckers
                        AnimalType.BIRD, AnimalType.BIRD, AnimalType.BIRD,
                        AnimalType.BAT,  # Forest bats
                        # Forest floor creatures
                        AnimalType.HOPPER, AnimalType.HOPPER,  # Rabbits, squirrels
                        AnimalType.REPTILE,  # Lizards
                        AnimalType.SERPENT,  # Forest snakes
                        AnimalType.INSECT, AnimalType.INSECT,
                        AnimalType.BEETLE,  # Forest beetles
                        AnimalType.SNAIL,  # Forest snails
                        AnimalType.CENTIPEDE,  # Under logs
                    ])
            elif biome == 'taiga':
                # Taiga: wolves, bears, moose, hardy birds
                if rng.random() < 0.5:
                    animal_type = rng.choice([
                        # Big mammals
                        AnimalType.MAMMAL, AnimalType.MAMMAL, AnimalType.MAMMAL,
                        AnimalType.GORILLA,  # Bears
                        # Birds
                        AnimalType.BIRD, AnimalType.BIRD,
                        # Forest creatures
                        AnimalType.HOPPER,  # Rabbits
                        AnimalType.BAT,  # Bats
                        AnimalType.SERPENT,  # Cold snakes
                    ])
            elif biome == 'volcanic':
                # Volcanic: fire-adapted creatures, similar to hellfire but less extreme
                if rng.random() < 0.5:
                    animal_type = rng.choice([
                        AnimalType.REPTILE, AnimalType.REPTILE, AnimalType.REPTILE,
                        AnimalType.INSECT, AnimalType.INSECT,
                        AnimalType.BEETLE,
                        AnimalType.SCORPION,
                        AnimalType.CENTIPEDE,
                        AnimalType.SPIDER,
                        AnimalType.BIRD,  # Heat-tolerant birds
                    ])
            
            # Get template and mutate
            templates = self.species_templates.get(animal_type, self.species_templates[AnimalType.MAMMAL])
            base_dna = rng.choice(templates)
            dna = base_dna.mutate(rng, strength=0.4)
            
            # === SPECIAL BIOME COLORING ===
            if biome == 'psychedelic':
                # Rainbow colors!
                phase = rng.random() * 6.28
                dna.primary_color = (
                    0.5 + 0.5 * np.sin(phase),
                    0.5 + 0.5 * np.sin(phase + 2.09),
                    0.5 + 0.5 * np.sin(phase + 4.19)
                )
                dna.has_glow = rng.random() < 0.3  # Some glow!
            elif biome == 'hellfire':
                # Fiery reds and oranges
                dna.primary_color = (
                    0.7 + rng.random() * 0.3,  # Lots of red
                    0.1 + rng.random() * 0.4,  # Some orange
                    0.0 + rng.random() * 0.1   # Almost no blue
                )
                dna.has_glow = rng.random() < 0.2  # Ember glow
            elif biome == 'shadow':
                # Dark purples and blacks
                dna.primary_color = (
                    0.1 + rng.random() * 0.2,
                    0.05 + rng.random() * 0.1,
                    0.15 + rng.random() * 0.2
                )
                # Occasional glowing eyes
                if rng.random() < 0.2:
                    dna.has_glow = True
                    dna.glow_color = (0.5, 0.1, 0.6)
            elif biome == 'crystal':
                # Pale, crystalline
                dna.primary_color = (
                    0.7 + rng.random() * 0.25,
                    0.8 + rng.random() * 0.15,
                    0.85 + rng.random() * 0.15
                )
            elif biome == 'void':
                # Nearly invisible, faint glow
                dna.primary_color = (0.05, 0.02, 0.08)
                dna.has_glow = True
                dna.glow_color = (0.2, 0.05, 0.25)
                dna.glow_intensity = 0.2
            elif biome in ('ocean', 'underwater', 'coral_reef'):
                # COLORFUL tropical fish! Vibrant reef colors
                if rng.random() < 0.7:  # 70% colorful
                    reef_colors = [
                        (0.95, 0.5, 0.1),   # Orange clownfish
                        (0.2, 0.6, 0.95),   # Blue tang
                        (0.95, 0.95, 0.2),  # Yellow tang
                        (0.9, 0.3, 0.5),    # Pink reef fish
                        (0.4, 0.9, 0.5),    # Green reef fish
                        (0.7, 0.2, 0.9),    # Purple reef fish
                        (0.95, 0.1, 0.1),   # Red reef fish
                        (0.1, 0.9, 0.9),    # Cyan reef fish
                    ]
                    c = reef_colors[int(rng.integers(0, len(reef_colors)))]
                    dna.primary_color = c
                    # Add random pattern stripes
                    if rng.random() < 0.4:
                        stripe = reef_colors[int(rng.integers(0, len(reef_colors)))]
                        dna.secondary_color = stripe
            elif biome == 'meadow':
                # Natural colors - browns, greens, some colorful butterflies
                if animal_type == AnimalType.MOTH or animal_type == AnimalType.INSECT:
                    # Colorful butterflies!
                    phase = rng.random() * 6.28
                    dna.primary_color = (
                        0.6 + 0.4 * np.sin(phase),
                        0.5 + 0.5 * np.sin(phase + 1.5),
                        0.5 + 0.5 * np.sin(phase + 3.0)
                    )
            elif biome == 'swamp':
                # Muddy greens and browns
                dna.primary_color = (
                    0.3 + rng.random() * 0.2,
                    0.35 + rng.random() * 0.2,
                    0.15 + rng.random() * 0.1
                )
            elif biome in ('tundra', 'frozen'):
                # White/pale arctic colors
                dna.primary_color = (
                    0.85 + rng.random() * 0.15,
                    0.85 + rng.random() * 0.15,
                    0.9 + rng.random() * 0.1
                )
            elif biome == 'volcanic':
                # Dark grays and reds
                dna.primary_color = (
                    0.3 + rng.random() * 0.25,
                    0.2 + rng.random() * 0.15,
                    0.15 + rng.random() * 0.1
                )
            
            # World position
            world_x = chunk_world_x + local_x * tile_scale
            world_z = chunk_world_z + local_z * tile_scale
            world_y = ground_h * height_scale
            
            # Adjust Y for flying/swimming
            if dna.movement_type == MovementType.FLY:
                world_y += 5 + rng.random() * 10
            elif dna.movement_type in (MovementType.SWIM, MovementType.FLOAT):
                world_y = 0.5
            
            animal = AnimalInstance(
                x=world_x, y=world_y, z=world_z,
                dna=dna,
                rotation=rng.random() * 360,
                anim_phase=rng.random()
            )
            
            animals.append(animal)
            self.animals.append(animal)
            
            # Add to spatial index for O(1) neighbor lookups
            if self._spatial_index:
                from .spatial import EntityType as SpatialType
                self._spatial_index.insert(animal, animal.x, animal.z, SpatialType.ANIMAL, id(animal))
        
        self.chunk_animals[key] = animals
    
    def spawn_baby_animal(self, x: float, y: float, z: float, parent_dna: dict):
        """Spawn a baby animal at the location where an egg hatched."""
        # Debug logging (disabled by default)
        # print(f"[HATCH] Spawning baby animal at ({x:.1f}, {y:.1f}, {z:.1f}) from parent: {parent_dna.get('animal_type', 'unknown')}")
        
        # Determine chunk
        chunk_size = 128  # Approximate chunk size in world units
        cx = int(x // chunk_size)
        cz = int(z // chunk_size)
        key = (cx, cz)
        
        # Determine animal type from parent DNA
        animal_type = parent_dna.get('animal_type', AnimalType.MAMMAL)
        if animal_type not in AnimalType.ALL:
            animal_type = AnimalType.MAMMAL
        
        # Create DNA based on parent (with some mutation)
        templates = self.species_templates.get(animal_type, self.species_templates.get(AnimalType.MAMMAL, []))
        if templates:
            rng = np.random.default_rng(abs(hash((x, z, animal_type))) % (2**31))
            base_dna = rng.choice(templates)
            dna = base_dna.mutate(rng, strength=0.2)  # Light mutation from parent
        else:
            # Fallback - create random DNA
            dna = AnimalDNA.create_random(animal_type, int(x * 1000 + z))
        
        # Start on ground, then adjust for movement type
        world_y = y
        if dna.movement_type == MovementType.FLY:
            world_y = y + 2  # Start just above ground, will fly up
        elif dna.movement_type in (MovementType.SWIM, MovementType.FLOAT):
            world_y = max(0.5, y)  # In water
        
        # Create baby animal (smaller but visible scale)
        animal = AnimalInstance(
            x=x, y=world_y, z=z,
            dna=dna,
            rotation=np.random.random() * 360,
            anim_phase=np.random.random(),
            scale=0.5  # Baby starts at 50% (was 30%, too hard to see)
        )
        # Debug logging (disabled by default)
        # print(f"[HATCH] Created {dna.animal_type} baby at chunk {key}, total animals: {len(self.animals) + 1}")
        
        # Add to collections
        if key not in self.chunk_animals:
            self.chunk_animals[key] = []
        self.chunk_animals[key].append(animal)
        self.animals.append(animal)
        
        # Add to spatial index
        if self._spatial_index:
            from .spatial import EntityType as SpatialType
            self._spatial_index.insert(animal, animal.x, animal.z, SpatialType.ANIMAL, id(animal))
    
    def update(self, dt: float, player_pos: Tuple[float, float, float], 
               get_ground_height=None):
        """Update animals - smart updates: expensive AI rarely, cheap movement always."""
        if not player_pos:
            return
        
        px, py, pz = player_pos
        
        for animal in self.animals:
            # Distance to player
            dist_sq = (animal.x - px)**2 + (animal.z - pz)**2
            
            if dist_sq > 160000:  # 400^2 - too far, skip entirely
                continue
            
            # Always apply cached velocity (cheap!) 
            # This keeps animals moving smoothly between AI updates
            if hasattr(animal, '_cached_vx'):
                old_x, old_z = animal.x, animal.z
                animal.x += animal._cached_vx * dt * 0.02
                animal.z += animal._cached_vz * dt * 0.02
                # Update spatial index if position changed significantly
                if self._spatial_index and (abs(animal.x - old_x) > 1 or abs(animal.z - old_z) > 1):
                    self._spatial_index.update_position(id(animal), animal.x, animal.z)
            
            # Full AI update only for VERY close animals (expensive - neighbor checks)
            if dist_sq < 10000:  # 100^2
                neighbors = None
                if dist_sq < 3600 and animal.dna.group_tendency > 0.3:  # 60^2
                    # Use spatial index for O(1) neighbor lookup instead of O(n) loop!
                    if self._spatial_index:
                        # query_radius is a generator, take first few results
                        nearby = list(self._spatial_index.query_radius(animal.x, animal.z, 25.0))[:5]
                        neighbors = [n.entity for n in nearby if n.entity is not animal][:3]
                    else:
                        # Fallback O(n) loop
                        neighbors = [
                            other for other in self.animals
                            if other is not animal 
                            and (animal.x - other.x)**2 + (animal.z - other.z)**2 < 625
                        ][:3]
                
                animal.update(dt, neighbors, player_pos, get_ground_height)
                # Cache velocity for interpolation
                animal._cached_vx = getattr(animal, 'vx', 0)
                animal._cached_vz = getattr(animal, 'vz', 0)
            elif dist_sq < 40000:  # 200^2 - medium distance: simple update, no neighbors
                animal.update(dt, None, None, get_ground_height)
                animal._cached_vx = getattr(animal, 'vx', 0)
                animal._cached_vz = getattr(animal, 'vz', 0)
    
    def render(self, camera_x: float, camera_y: float, camera_z: float):
        """Render nearby animals with LOD - optimized with squared distances and shape-aware billboards."""
        glDisable(GL_LIGHTING)
        
        # Pre-compute squared thresholds (avoid sqrt per animal!)
        full_sq = self.LOD_FULL * self.LOD_FULL
        simple_sq = self.LOD_SIMPLE * self.LOD_SIMPLE
        point_sq = self.LOD_POINT * self.LOD_POINT
        
        # Extended range for billboards
        billboard_far_sq = (self.LOD_POINT * 1.5) ** 2  # 525 units
        billboard_horizon_sq = (self.LOD_POINT * 2.0) ** 2  # 700 units
        
        full_animals = []
        simple_animals = []
        billboard_animals = []  # (animal, scale) tuples
        
        for animal in self.animals:
            dist_sq = (animal.x - camera_x)**2 + (animal.z - camera_z)**2
            
            if dist_sq > billboard_horizon_sq:
                continue
            
            # Use squared distances - no sqrt needed!
            if dist_sq < full_sq:
                full_animals.append(animal)
            elif dist_sq < simple_sq:
                simple_animals.append(animal)
            elif dist_sq < point_sq:
                # Full-size billboard
                billboard_animals.append((animal, 1.0))
            elif dist_sq < billboard_far_sq:
                # Half-size billboard
                billboard_animals.append((animal, 0.5))
            else:
                # Quarter-size billboard (horizon)
                billboard_animals.append((animal, 0.25))
        
        # Render each LOD level
        for animal in full_animals:
            AnimalRenderer.draw_full(animal)
        
        for animal in simple_animals:
            AnimalRenderer.draw_simple(animal)
        
        # Render shape-aware billboards at appropriate scales
        for animal, scale in billboard_animals:
            AnimalRenderer.draw_billboard(animal, scale)
        
        glEnable(GL_LIGHTING)
    
    def cleanup_distant_chunks(self, center_cx: int, center_cz: int, max_distance: int = 15):
        """Remove animals from distant chunks."""
        to_remove = []
        for (cx, cz), animals in list(self.chunk_animals.items()):
            if abs(cx - center_cx) > max_distance or abs(cz - center_cz) > max_distance:
                to_remove.append((cx, cz))
                for animal in animals:
                    if animal in self.animals:
                        self.animals.remove(animal)
                        # Remove from spatial index
                        if self._spatial_index:
                            self._spatial_index.remove(id(animal))
        
        for key in to_remove:
            del self.chunk_animals[key]
        
        # Hard cap enforcement - if still over limit, aggressively cull
        if len(self.animals) > self.MAX_TOTAL_ANIMALS:
            # Keep only the most recent animals
            excess = len(self.animals) - self.MAX_TOTAL_ANIMALS
            removed = self.animals[:excess]
            self.animals = self.animals[excess:]
            
            # Remove from spatial index
            if self._spatial_index:
                for animal in removed:
                    self._spatial_index.remove(id(animal))
            
            # Also clean up chunk_animals dict
            for (cx, cz), chunk_list in list(self.chunk_animals.items()):
                self.chunk_animals[(cx, cz)] = [a for a in chunk_list if a not in removed]
                if not self.chunk_animals[(cx, cz)]:
                    del self.chunk_animals[(cx, cz)]

