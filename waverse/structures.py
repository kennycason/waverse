"""
Structure System - Buildings, walls, floors with collision.

Structures are composed of simple primitives:
- Wall: Vertical rectangular surface
- Floor: Horizontal rectangular surface  
- Ramp/Stairs: Angled surface for vertical movement
- Ladder: Climbable vertical element

Features:
- Collision detection only when player is nearby (performance)
- LOD rendering for distant structures
- Procedural building generation
- Future: player-built structures
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from OpenGL.GL import *


@dataclass
class Wall:
    """A vertical rectangular wall segment."""
    x: float          # Center X
    y: float          # Base Y (bottom of wall)
    z: float          # Center Z
    width: float      # Width (along facing direction)
    height: float     # Height
    thickness: float = 0.3
    rotation: float = 0.0   # Rotation in degrees around Y axis
    color: Tuple[float, float, float] = (0.6, 0.55, 0.5)
    
    def get_corners(self) -> List[Tuple[float, float, float]]:
        """Get world-space corners for collision."""
        rad = math.radians(self.rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        hw = self.width / 2
        ht = self.thickness / 2
        
        # Local corners, then rotate
        corners = []
        for dx in [-hw, hw]:
            for dz in [-ht, ht]:
                wx = self.x + dx * cos_r - dz * sin_r
                wz = self.z + dx * sin_r + dz * cos_r
                corners.append((wx, wz))
        return corners
    
    def check_collision(self, px: float, py: float, pz: float, radius: float = 0.5) -> bool:
        """Check if a point collides with this wall."""
        # Quick height check first
        if py < self.y or py > self.y + self.height:
            return False
        
        # Transform player position to wall's local space
        rad = math.radians(self.rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        
        # Translate to wall center
        dx = px - self.x
        dz = pz - self.z
        
        # Rotate to wall's local coords
        local_x = dx * cos_r + dz * sin_r
        local_z = -dx * sin_r + dz * cos_r
        
        # Check against box with radius
        hw = self.width / 2 + radius
        ht = self.thickness / 2 + radius
        
        return abs(local_x) < hw and abs(local_z) < ht


@dataclass
class Floor:
    """A horizontal floor/ceiling surface."""
    x: float          # Center X
    y: float          # Y height
    z: float          # Center Z
    width: float      # X dimension
    depth: float      # Z dimension
    thickness: float = 0.2
    color: Tuple[float, float, float] = (0.5, 0.45, 0.4)
    
    def check_collision_above(self, px: float, py: float, pz: float, 
                              player_height: float = 2.0) -> Optional[float]:
        """Check if player is standing on this floor. Returns floor Y if standing."""
        # Check if player is within floor bounds
        if abs(px - self.x) > self.width / 2:
            return None
        if abs(pz - self.z) > self.depth / 2:
            return None
        
        # Check if player is at floor level (within step-up range)
        floor_top = self.y + self.thickness
        if py >= floor_top - 0.5 and py <= floor_top + 1.0:
            return floor_top
        return None
    
    def check_collision_below(self, px: float, py: float, pz: float,
                              player_height: float = 2.0) -> bool:
        """Check if player's head hits this ceiling."""
        if abs(px - self.x) > self.width / 2:
            return False
        if abs(pz - self.z) > self.depth / 2:
            return False
        
        # Head hits ceiling
        head_y = py + player_height
        return head_y > self.y and py < self.y


@dataclass
class Ramp:
    """An angled surface for walking up/down (stairs/ramp)."""
    x: float          # Start X (bottom)
    y_bottom: float   # Bottom Y
    z: float          # Center Z
    length: float     # Horizontal length
    height: float     # Vertical rise
    width: float      # Width
    rotation: float = 0.0   # Direction facing (0 = +X direction)
    color: Tuple[float, float, float] = (0.55, 0.5, 0.45)
    is_ladder: bool = False  # If true, very steep (climbable)
    
    @property
    def y_top(self) -> float:
        return self.y_bottom + self.height
    
    @property  
    def angle(self) -> float:
        """Angle in degrees."""
        return math.degrees(math.atan2(self.height, self.length))
    
    def get_height_at(self, px: float, pz: float) -> Optional[float]:
        """Get the ramp height at a world position, or None if not on ramp."""
        # Transform to local coords
        rad = math.radians(self.rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        
        dx = px - self.x
        dz = pz - self.z
        
        local_x = dx * cos_r + dz * sin_r  # Along ramp direction
        local_z = -dx * sin_r + dz * cos_r  # Perpendicular
        
        # Check if on ramp
        if local_x < 0 or local_x > self.length:
            return None
        if abs(local_z) > self.width / 2:
            return None
        
        # Interpolate height
        t = local_x / self.length
        return self.y_bottom + t * self.height


@dataclass
class Structure:
    """A complete structure made of walls, floors, and ramps."""
    x: float          # World X center
    y: float          # World Y base
    z: float          # World Z center
    walls: List[Wall] = field(default_factory=list)
    floors: List[Floor] = field(default_factory=list)
    ramps: List[Ramp] = field(default_factory=list)
    
    # Bounding box for quick culling
    bbox_min: Tuple[float, float, float] = (0, 0, 0)
    bbox_max: Tuple[float, float, float] = (0, 0, 0)
    
    display_list: int = 0
    lod_display_list: int = 0
    
    def compute_bounds(self):
        """Compute bounding box from all elements."""
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for wall in self.walls:
            min_x = min(min_x, wall.x - wall.width/2)
            max_x = max(max_x, wall.x + wall.width/2)
            min_y = min(min_y, wall.y)
            max_y = max(max_y, wall.y + wall.height)
            min_z = min(min_z, wall.z - wall.thickness)
            max_z = max(max_z, wall.z + wall.thickness)
        
        for floor in self.floors:
            min_x = min(min_x, floor.x - floor.width/2)
            max_x = max(max_x, floor.x + floor.width/2)
            min_y = min(min_y, floor.y)
            max_y = max(max_y, floor.y + floor.thickness)
            min_z = min(min_z, floor.z - floor.depth/2)
            max_z = max(max_z, floor.z + floor.depth/2)
        
        for ramp in self.ramps:
            min_x = min(min_x, ramp.x)
            max_x = max(max_x, ramp.x + ramp.length)
            min_y = min(min_y, ramp.y_bottom)
            max_y = max(max_y, ramp.y_top)
            min_z = min(min_z, ramp.z - ramp.width/2)
            max_z = max(max_z, ramp.z + ramp.width/2)
        
        self.bbox_min = (min_x, min_y, min_z)
        self.bbox_max = (max_x, max_y, max_z)
    
    def is_nearby(self, px: float, pz: float, radius: float = 50) -> bool:
        """Quick check if player is near this structure."""
        cx = (self.bbox_min[0] + self.bbox_max[0]) / 2
        cz = (self.bbox_min[2] + self.bbox_max[2]) / 2
        dx = px - cx
        dz = pz - cz
        return dx*dx + dz*dz < radius * radius


def generate_tile_building(x: float, y: float, z: float, 
                           tiles_x: int = 3, tiles_z: int = 3,
                           floors: int = 2, 
                           tile_size: float = 4.0,
                           floor_height: float = 4.5,
                           seed: int = 42) -> Structure:
    """
    Generate a tile-based building like NMS.
    - Grid of tiles (e.g. 3x3)
    - Each tile has floor/ceiling panels
    - Stair openings cut through floor tiles
    - Walls around perimeter with door opening
    - Various roof styles, window patterns, and wall decorations
    """
    rng = np.random.default_rng(seed)
    
    structure = Structure(x=x, y=y, z=z)
    
    width = tiles_x * tile_size
    depth = tiles_z * tile_size
    
    # Building style variations
    building_style = rng.choice(["rustic", "modern", "alien", "ancient", "industrial"])
    roof_style = rng.choice(["flat", "peaked", "dome", "terraced"])
    has_windows = rng.random() < 0.7  # 70% have windows
    window_style = rng.choice(["square", "tall", "round", "slit"])
    has_balcony = floors > 1 and rng.random() < 0.3  # 30% chance per multi-story
    has_pillars = rng.random() < 0.2  # 20% have exterior pillars
    
    # Colors with style-based variation
    if building_style == "rustic":
        wall_color = (0.5 + rng.random() * 0.2, 
                      0.4 + rng.random() * 0.15, 
                      0.3 + rng.random() * 0.1)
        roof_color = (0.35, 0.28, 0.2)
    elif building_style == "modern":
        gray = 0.5 + rng.random() * 0.3
        wall_color = (gray, gray, gray + 0.05)
        roof_color = (0.3, 0.3, 0.32)
    elif building_style == "alien":
        wall_color = (0.3 + rng.random() * 0.3, 
                      0.4 + rng.random() * 0.4, 
                      0.5 + rng.random() * 0.3)
        roof_color = (0.2, 0.3, 0.4)
    elif building_style == "ancient":
        wall_color = (0.65 + rng.random() * 0.15, 
                      0.6 + rng.random() * 0.1, 
                      0.5 + rng.random() * 0.1)
        roof_color = (0.5, 0.45, 0.35)
    else:  # industrial
        wall_color = (0.4 + rng.random() * 0.1, 
                      0.35 + rng.random() * 0.1, 
                      0.3 + rng.random() * 0.1)
        roof_color = (0.25, 0.25, 0.28)
    
    floor_color = (wall_color[0] * 0.7, wall_color[1] * 0.7, wall_color[2] * 0.7)
    stair_color = (0.5, 0.45, 0.4)
    window_color = (0.3, 0.4, 0.6)  # Bluish glass
    
    wall_thickness = 0.35
    hw = width / 2
    hd = depth / 2
    
    # Pick door wall (0=+Z, 1=-Z, 2=-X, 3=+X) and door tile
    door_wall = rng.integers(0, 4)
    if door_wall in [0, 1]:
        door_tile = rng.integers(0, tiles_x)
    else:
        door_tile = rng.integers(0, tiles_z)
    
    # Pick stair tile (corner, not door tile on ground)
    stair_tile_x = 0 if rng.random() < 0.5 else tiles_x - 1
    stair_tile_z = 0 if rng.random() < 0.5 else tiles_z - 1
    
    for floor_idx in range(floors):
        floor_y = y + floor_idx * floor_height
        
        # Floor tiles (except ground floor, and skip stair opening from below)
        if floor_idx > 0:
            for tx in range(tiles_x):
                for tz in range(tiles_z):
                    # Skip stair opening (stair comes up from floor below)
                    if tx == stair_tile_x and tz == stair_tile_z:
                        continue
                    
                    tile_x = x - hw + (tx + 0.5) * tile_size
                    tile_z = z - hd + (tz + 0.5) * tile_size
                    
                    structure.floors.append(Floor(
                        x=tile_x, y=floor_y, z=tile_z,
                        width=tile_size - 0.1,
                        depth=tile_size - 0.1,
                        thickness=0.25,
                        color=floor_color
                    ))
        
        # Perimeter walls as tile segments
        door_height = 3.5  # Taller doors to fit player comfortably
        wall_height = floor_height
        
        # Front wall (+Z side)
        for tx in range(tiles_x):
            wx = x - hw + (tx + 0.5) * tile_size
            wz = z + hd
            is_door = (door_wall == 0 and tx == door_tile and floor_idx == 0)
            
            if is_door:
                # Wall above door
                structure.walls.append(Wall(
                    x=wx, y=floor_y + door_height, z=wz,
                    width=tile_size, height=wall_height - door_height,
                    thickness=wall_thickness, rotation=0, color=wall_color
                ))
            else:
                structure.walls.append(Wall(
                    x=wx, y=floor_y, z=wz,
                    width=tile_size, height=wall_height,
                    thickness=wall_thickness, rotation=0, color=wall_color
                ))
        
        # Back wall (-Z side)
        for tx in range(tiles_x):
            wx = x - hw + (tx + 0.5) * tile_size
            wz = z - hd
            is_door = (door_wall == 1 and tx == door_tile and floor_idx == 0)
            
            if is_door:
                structure.walls.append(Wall(
                    x=wx, y=floor_y + door_height, z=wz,
                    width=tile_size, height=wall_height - door_height,
                    thickness=wall_thickness, rotation=180, color=wall_color
                ))
            else:
                structure.walls.append(Wall(
                    x=wx, y=floor_y, z=wz,
                    width=tile_size, height=wall_height,
                    thickness=wall_thickness, rotation=180, color=wall_color
                ))
        
        # Left wall (-X side)
        for tz in range(tiles_z):
            wx = x - hw
            wz = z - hd + (tz + 0.5) * tile_size
            is_door = (door_wall == 2 and tz == door_tile and floor_idx == 0)
            
            if is_door:
                structure.walls.append(Wall(
                    x=wx, y=floor_y + door_height, z=wz,
                    width=tile_size, height=wall_height - door_height,
                    thickness=wall_thickness, rotation=90, color=wall_color
                ))
            else:
                structure.walls.append(Wall(
                    x=wx, y=floor_y, z=wz,
                    width=tile_size, height=wall_height,
                    thickness=wall_thickness, rotation=90, color=wall_color
                ))
        
        # Right wall (+X side)
        for tz in range(tiles_z):
            wx = x + hw
            wz = z - hd + (tz + 0.5) * tile_size
            is_door = (door_wall == 3 and tz == door_tile and floor_idx == 0)
            
            if is_door:
                structure.walls.append(Wall(
                    x=wx, y=floor_y + door_height, z=wz,
                    width=tile_size, height=wall_height - door_height,
                    thickness=wall_thickness, rotation=-90, color=wall_color
                ))
            else:
                structure.walls.append(Wall(
                    x=wx, y=floor_y, z=wz,
                    width=tile_size, height=wall_height,
                    thickness=wall_thickness, rotation=-90, color=wall_color
                ))
        
        # Stairs to next floor (if not top floor)
        if floor_idx < floors - 1:
            stair_x = x - hw + (stair_tile_x + 0.5) * tile_size
            stair_z = z - hd + (stair_tile_z + 0.5) * tile_size
            
            # Determine stair direction (toward center)
            if stair_tile_x == 0:
                stair_rot = 0  # Stairs go +X
                stair_start_x = stair_x - tile_size * 0.35
            else:
                stair_rot = 180  # Stairs go -X
                stair_start_x = stair_x + tile_size * 0.35
            
            structure.ramps.append(Ramp(
                x=stair_start_x,
                y_bottom=floor_y,
                z=stair_z,
                length=tile_size * 0.9,
                height=floor_height,
                width=tile_size * 0.7,
                rotation=stair_rot,
                color=stair_color
            ))
    
    # Roof tiles (solid ceiling on top)
    roof_y = y + floors * floor_height
    for tx in range(tiles_x):
        for tz in range(tiles_z):
            tile_x = x - hw + (tx + 0.5) * tile_size
            tile_z = z - hd + (tz + 0.5) * tile_size
            
            structure.floors.append(Floor(
                x=tile_x, y=roof_y, z=tile_z,
                width=tile_size,
                depth=tile_size,
                thickness=0.35,
                color=roof_color
            ))
    
    structure.compute_bounds()
    return structure


class StructureRenderer:
    """Renders structures efficiently."""
    
    LOD_FULL = 80
    LOD_SIMPLE = 200
    
    @staticmethod
    def _draw_wall(wall: Wall):
        """Draw a single wall."""
        glPushMatrix()
        glTranslatef(wall.x, wall.y, wall.z)
        glRotatef(wall.rotation, 0, 1, 0)
        
        hw = wall.width / 2
        ht = wall.thickness / 2
        h = wall.height
        
        glColor3f(*wall.color)
        
        glBegin(GL_QUADS)
        # Front face
        glNormal3f(0, 0, 1)
        glVertex3f(-hw, 0, ht)
        glVertex3f(hw, 0, ht)
        glVertex3f(hw, h, ht)
        glVertex3f(-hw, h, ht)
        
        # Back face
        glNormal3f(0, 0, -1)
        glVertex3f(hw, 0, -ht)
        glVertex3f(-hw, 0, -ht)
        glVertex3f(-hw, h, -ht)
        glVertex3f(hw, h, -ht)
        
        # Top
        glNormal3f(0, 1, 0)
        glVertex3f(-hw, h, -ht)
        glVertex3f(-hw, h, ht)
        glVertex3f(hw, h, ht)
        glVertex3f(hw, h, -ht)
        
        # Left
        glNormal3f(-1, 0, 0)
        glVertex3f(-hw, 0, -ht)
        glVertex3f(-hw, 0, ht)
        glVertex3f(-hw, h, ht)
        glVertex3f(-hw, h, -ht)
        
        # Right
        glNormal3f(1, 0, 0)
        glVertex3f(hw, 0, ht)
        glVertex3f(hw, 0, -ht)
        glVertex3f(hw, h, -ht)
        glVertex3f(hw, h, ht)
        glEnd()
        
        glPopMatrix()
    
    @staticmethod
    def _draw_floor(floor: Floor):
        """Draw a floor/ceiling."""
        glPushMatrix()
        glTranslatef(floor.x, floor.y, floor.z)
        
        hw = floor.width / 2
        hd = floor.depth / 2
        t = floor.thickness
        
        glColor3f(*floor.color)
        
        glBegin(GL_QUADS)
        # Top surface
        glNormal3f(0, 1, 0)
        glVertex3f(-hw, t, -hd)
        glVertex3f(-hw, t, hd)
        glVertex3f(hw, t, hd)
        glVertex3f(hw, t, -hd)
        
        # Bottom surface
        glNormal3f(0, -1, 0)
        glVertex3f(-hw, 0, hd)
        glVertex3f(-hw, 0, -hd)
        glVertex3f(hw, 0, -hd)
        glVertex3f(hw, 0, hd)
        
        # Edges
        glNormal3f(0, 0, 1)
        glVertex3f(-hw, 0, hd)
        glVertex3f(hw, 0, hd)
        glVertex3f(hw, t, hd)
        glVertex3f(-hw, t, hd)
        
        glNormal3f(0, 0, -1)
        glVertex3f(hw, 0, -hd)
        glVertex3f(-hw, 0, -hd)
        glVertex3f(-hw, t, -hd)
        glVertex3f(hw, t, -hd)
        glEnd()
        
        glPopMatrix()
    
    @staticmethod
    def _draw_ramp(ramp: Ramp):
        """Draw a ramp/stairs."""
        glPushMatrix()
        glTranslatef(ramp.x, ramp.y_bottom, ramp.z)
        glRotatef(ramp.rotation, 0, 1, 0)
        
        hw = ramp.width / 2
        length = ramp.length
        height = ramp.height
        
        glColor3f(*ramp.color)
        
        glBegin(GL_QUADS)
        # Ramp surface
        glNormal3f(-height, length, 0)  # Perpendicular to slope
        glVertex3f(0, 0, -hw)
        glVertex3f(0, 0, hw)
        glVertex3f(length, height, hw)
        glVertex3f(length, height, -hw)
        
        # Bottom
        glNormal3f(0, -1, 0)
        glVertex3f(0, 0, hw)
        glVertex3f(0, 0, -hw)
        glVertex3f(length, 0, -hw)
        glVertex3f(length, 0, hw)
        
        # Sides
        glNormal3f(0, 0, 1)
        glVertex3f(0, 0, hw)
        glVertex3f(length, 0, hw)
        glVertex3f(length, height, hw)
        
        glNormal3f(0, 0, -1)
        glVertex3f(length, 0, -hw)
        glVertex3f(0, 0, -hw)
        glVertex3f(length, height, -hw)
        glEnd()
        
        # Back triangle
        glBegin(GL_TRIANGLES)
        glNormal3f(1, 0, 0)
        glVertex3f(length, 0, -hw)
        glVertex3f(length, 0, hw)
        glVertex3f(length, height, 0)
        glEnd()
        
        glPopMatrix()
    
    @staticmethod
    def render_structure(structure: Structure, cam_x: float, cam_z: float):
        """Render a structure with LOD."""
        # Distance check
        cx = (structure.bbox_min[0] + structure.bbox_max[0]) / 2
        cz = (structure.bbox_min[2] + structure.bbox_max[2]) / 2
        dist_sq = (cam_x - cx)**2 + (cam_z - cz)**2
        
        if dist_sq > StructureRenderer.LOD_SIMPLE ** 2:
            return  # Too far, don't render
        
        for wall in structure.walls:
            StructureRenderer._draw_wall(wall)
        
        for floor in structure.floors:
            StructureRenderer._draw_floor(floor)
        
        for ramp in structure.ramps:
            StructureRenderer._draw_ramp(ramp)


class StructureManager:
    """Manages all structures in the world."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.structures: List[Structure] = []
        self.spawned_chunks: set = set()  # Track which chunks we've processed
    
    def add_structure(self, structure: Structure):
        """Add a structure to the world."""
        self.structures.append(structure)
    
    def spawn_random_buildings(self, cx: int, cz: int, chunk_world_x: float, 
                               chunk_world_z: float, heightmap, height_scale: float,
                               tile_scale: float = 2.0):
        """Maybe spawn a building in this chunk."""
        # Skip if we've already processed this chunk
        chunk_key = (cx, cz)
        if chunk_key in self.spawned_chunks:
            return
        self.spawned_chunks.add(chunk_key)
        
        chunk_seed = abs(hash((self.seed, cx, cz, "structures"))) % (2**31)
        rng = np.random.default_rng(chunk_seed)
        
        # Low chance of building per chunk
        if rng.random() > 0.02:  # 2% chance
            return
        
        # Find a flat spot
        h, w = heightmap.shape
        local_x = rng.integers(8, w - 8)
        local_z = rng.integers(8, h - 8)
        
        # Check if reasonably flat
        center_h = heightmap[local_z, local_x]
        if center_h < 5 or center_h > 40:  # Not in water or on mountains
            return
        
        # Check flatness
        heights = [
            heightmap[local_z-2, local_x-2],
            heightmap[local_z-2, local_x+2],
            heightmap[local_z+2, local_x-2],
            heightmap[local_z+2, local_x+2],
        ]
        if max(heights) - min(heights) > 3:  # Too hilly
            return
        
        # IMPORTANT: Scale local coordinates by tile_scale to get world coords
        world_x = chunk_world_x + local_x * tile_scale
        world_z = chunk_world_z + local_z * tile_scale
        world_y = center_h * height_scale
        
        floors = 1 + rng.integers(0, 4)  # 1-4 floors
        tiles_x = 2 + rng.integers(0, 4)  # 2-5 tiles wide
        tiles_z = 2 + rng.integers(0, 4)  # 2-5 tiles deep
        tile_size = 3.5 + rng.random() * 1.5  # 3.5-5.0 per tile
        
        building = generate_tile_building(
            world_x, world_y, world_z,
            tiles_x=tiles_x,
            tiles_z=tiles_z,
            floors=floors,
            tile_size=tile_size,
            seed=chunk_seed
        )
        self.add_structure(building)
    
    def check_collision(self, px: float, py: float, pz: float, 
                        radius: float = 0.5, player_height: float = 3.5) -> Tuple[bool, Optional[float]]:
        """
        Check collision with all nearby structures.
        Returns (blocked, floor_height).
        - blocked: True if player would hit a wall or ceiling
        - floor_height: If standing on a floor/ramp, returns that height (highest floor BELOW player)
        """
        floor_height = None
        blocked = False
        
        for structure in self.structures:
            if not structure.is_nearby(px, pz, 60):
                continue
            
            # Check walls
            for wall in structure.walls:
                if wall.check_collision(px, py, pz, radius):
                    blocked = True
            
            # Check floors (walking on) and ceilings (head hits)
            for floor in structure.floors:
                # Is player within this floor's X/Z bounds?
                if abs(px - floor.x) > floor.width / 2:
                    continue
                if abs(pz - floor.z) > floor.depth / 2:
                    continue
                
                floor_top = floor.y + floor.thickness
                
                # Floor is below player (can land on it)
                # We land on floors that are below our feet
                if floor_top <= py + 0.5:  # Floor is at or below our feet
                    if floor_height is None or floor_top > floor_height:
                        floor_height = floor_top
                
                # Ceiling is above player (head hits) - blocks upward movement
                # If the floor is above our head level
                head_y = py + player_height
                if floor.y > py and floor.y < head_y + 1.0:
                    blocked = True
            
            # Check ramps (walking on)
            for ramp in structure.ramps:
                h = ramp.get_height_at(px, pz)
                if h is not None and h <= py + 1.0:  # Ramp below us
                    if floor_height is None or h > floor_height:
                        floor_height = h
        
        return blocked, floor_height
    
    def render(self, cam_x: float, cam_y: float, cam_z: float):
        """Render all visible structures."""
        for structure in self.structures:
            StructureRenderer.render_structure(structure, cam_x, cam_z)
    
    def cleanup_distant(self, cx: int, cz: int, chunk_size: float = 32, max_chunks: int = 40):
        """Remove structures far from player."""
        world_x = cx * chunk_size
        world_z = cz * chunk_size
        max_dist = max_chunks * chunk_size
        
        self.structures = [
            s for s in self.structures
            if abs(s.x - world_x) < max_dist and abs(s.z - world_z) < max_dist
        ]
        
        # Also cleanup spawned_chunks set for distant chunks
        max_chunk_dist = max_chunks
        self.spawned_chunks = {
            (scx, scz) for (scx, scz) in self.spawned_chunks
            if abs(scx - cx) < max_chunk_dist and abs(scz - cz) < max_chunk_dist
        }

