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
    
    def check_collision(self, px: float, py: float, pz: float, radius: float = 0.5, 
                        player_height: float = 3.0) -> bool:
        """Check if player body collides with this wall.
        
        Args:
            px, py, pz: Player position (py is HEAD/camera height)
            radius: Player collision radius
            player_height: Height from feet to head
        """
        # Height overlap check: player body (feet to head) vs wall (base to top)
        player_feet = py - player_height
        player_head = py
        wall_base = self.y
        wall_top = self.y + self.height
        
        # Two ranges overlap if: start1 < end2 AND start2 < end1
        if not (player_feet < wall_top and wall_base < player_head):
            return False  # No vertical overlap
        
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
class Pillar:
    """A vertical support pillar/leg for buildings on slopes."""
    x: float          # Center X
    y_bottom: float   # Bottom Y (ground level)
    y_top: float      # Top Y (connects to floor)
    z: float          # Center Z
    width: float = 0.6  # Square pillar width
    color: Tuple[float, float, float] = (0.4, 0.38, 0.35)


@dataclass
class Structure:
    """A complete structure made of walls, floors, ramps, and pillars."""
    x: float          # World X center
    y: float          # World Y base
    z: float          # World Z center
    walls: List[Wall] = field(default_factory=list)
    floors: List[Floor] = field(default_factory=list)
    ramps: List[Ramp] = field(default_factory=list)
    pillars: List[Pillar] = field(default_factory=list)
    
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
        
        for pillar in self.pillars:
            hw = pillar.width / 2
            min_x = min(min_x, pillar.x - hw)
            max_x = max(max_x, pillar.x + hw)
            min_y = min(min_y, pillar.y_bottom)
            max_y = max(max_y, pillar.y_top)
            min_z = min(min_z, pillar.z - hw)
            max_z = max(max_z, pillar.z + hw)
        
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
                           tile_size: float = 10.0,  # Large tiles for player scale
                           floor_height: float = 14.0,  # Very tall floors - lots of headroom
                           seed: int = 42,
                           terrain_heights: List[float] = None) -> Structure:
    """
    Generate a tile-based building with proper doors, ramps, and varied designs.
    
    Features:
    - Full-height door openings (truly empty, no header)
    - Simple 45° ramps with landing space
    - Pillars at tile joints extending to ground
    - Windows, open walls, balconies for variety
    - Multi-floor structures with proper stair openings
    """
    rng = np.random.default_rng(seed)
    
    structure = Structure(x=x, y=y, z=z)
    
    width = tiles_x * tile_size
    depth = tiles_z * tile_size
    hw = width / 2
    hd = depth / 2
    
    # Structure DNA - controls variety
    building_style = rng.choice(["rustic", "modern", "alien", "ancient", "industrial", "tower", "warehouse"])
    has_windows = rng.random() < 0.6  # 60% have windows
    window_chance = 0.3 if has_windows else 0.0  # Per-wall segment
    open_wall_chance = 0.15  # 15% chance of fully open wall segment (balcony/porch)
    has_interior_pillars = floors > 1 and rng.random() < 0.4  # 40% multi-story have interior pillars
    
    # Colors based on style
    style_colors = {
        "rustic": ((0.55, 0.45, 0.35), (0.4, 0.32, 0.25)),
        "modern": ((0.7, 0.7, 0.72), (0.35, 0.35, 0.38)),
        "alien": ((0.35, 0.55, 0.65), (0.25, 0.4, 0.5)),
        "ancient": ((0.7, 0.65, 0.55), (0.55, 0.5, 0.4)),
        "industrial": ((0.45, 0.42, 0.38), (0.3, 0.28, 0.25)),
        "tower": ((0.5, 0.5, 0.55), (0.4, 0.4, 0.45)),
        "warehouse": ((0.6, 0.55, 0.5), (0.45, 0.4, 0.35)),
    }
    base_wall, base_roof = style_colors.get(building_style, style_colors["rustic"])
    
    # Add some random variation to colors
    wall_color = tuple(max(0, min(1, c + rng.uniform(-0.1, 0.1))) for c in base_wall)
    roof_color = tuple(max(0, min(1, c + rng.uniform(-0.05, 0.05))) for c in base_roof)
    floor_color = tuple(c * 0.75 for c in wall_color)
    pillar_color = tuple(c * 0.65 for c in wall_color)
    ramp_color = tuple(c * 0.85 for c in wall_color)
    
    wall_thickness = 1.5  # Thick walls for reliable collision
    pillar_width = 0.5 + rng.random() * 0.3  # 0.5-0.8 width variation
    
    # =========================================================================
    # SUPPORT PILLARS - At tile corners, extending to ground
    # =========================================================================
    min_ground = y  # Default to building base
    if terrain_heights is not None:
        min_ground = min(terrain_heights)
    
    # Pillars at each tile corner (joint positions)
    for tx in range(tiles_x + 1):
        for tz in range(tiles_z + 1):
            # Skip some interior pillars for variety (keep corners always)
            is_corner = (tx in [0, tiles_x]) and (tz in [0, tiles_z])
            is_edge = (tx in [0, tiles_x]) or (tz in [0, tiles_z])
            
            if not is_corner and not is_edge:
                # Interior pillar - only if has_interior_pillars
                if not has_interior_pillars:
                    continue
                if rng.random() > 0.5:  # Skip some interior pillars
                    continue
            
            px = x - hw + tx * tile_size
            pz = z - hd + tz * tile_size
            
            # Calculate ground height at this position
            if terrain_heights is not None:
                # Interpolate from corner heights
                tx_frac = tx / tiles_x
                tz_frac = tz / tiles_z
                h00, h10, h01, h11 = terrain_heights
                ground_h = (h00 * (1-tx_frac) * (1-tz_frac) +
                           h10 * tx_frac * (1-tz_frac) +
                           h01 * (1-tx_frac) * tz_frac +
                           h11 * tx_frac * tz_frac)
            else:
                ground_h = y
            
            pillar_height = y - ground_h
            
            # Add pillar at edges/corners - always extend to ground (with extra below for safety)
            if is_edge or is_corner:
                # Extend 2 units below ground to prevent visual gaps
                safe_bottom = ground_h - 2.0
                structure.pillars.append(Pillar(
                    x=px, y_bottom=safe_bottom, y_top=y, z=pz,
                    width=pillar_width if is_corner else pillar_width * 0.7,
                    color=pillar_color
                ))
    
    # =========================================================================
    # DOOR SELECTION - Full height opening, truly empty
    # =========================================================================
    door_wall = rng.integers(0, 4)  # 0=+Z, 1=-Z, 2=-X, 3=+X
    if door_wall in [0, 1]:
        door_tile = rng.integers(0, tiles_x)
    else:
        door_tile = rng.integers(0, tiles_z)
    
    # =========================================================================
    # RAMP SELECTION - 45° ramp with landing space
    # =========================================================================
    # Ramp needs 2 tiles: one for ramp, one for landing at top
    # Place ramp along one wall, not in corner
    if floors > 1:
        if tiles_x >= 2:
            ramp_tile_x = rng.integers(0, tiles_x - 1)  # Leave space for landing
            ramp_tile_z = 0 if rng.random() < 0.5 else tiles_z - 1
            ramp_direction = "x"  # Ramp goes along X axis
        else:
            ramp_tile_x = 0 if rng.random() < 0.5 else tiles_x - 1
            ramp_tile_z = rng.integers(0, max(1, tiles_z - 1))
            ramp_direction = "z"
    else:
        ramp_tile_x = ramp_tile_z = -1  # No ramp needed
        ramp_direction = None
    
    # =========================================================================
    # GENERATE FLOORS
    # =========================================================================
    # Check if building is on a slope - needs ground floor if so
    has_slope = False
    if terrain_heights is not None:
        slope = max(terrain_heights) - min(terrain_heights)
        has_slope = slope > 2.0
    
    for floor_idx in range(floors):
        floor_y = y + floor_idx * floor_height
        
        # --- Floor tiles ---
        # Add floor for upper floors OR ground floor on slopes (to close the gap)
        if floor_idx > 0 or has_slope:
            for tx in range(tiles_x):
                for tz in range(tiles_z):
                    # Skip ramp opening from below (upper floors only)
                    is_ramp_opening = False
                    if floor_idx > 0:
                        if ramp_direction == "x" and tx in [ramp_tile_x, ramp_tile_x + 1] and tz == ramp_tile_z:
                            is_ramp_opening = True
                        elif ramp_direction == "z" and tz in [ramp_tile_z, ramp_tile_z + 1] and tx == ramp_tile_x:
                            is_ramp_opening = True
                    
                    if is_ramp_opening:
                        continue
                    
                    tile_x = x - hw + (tx + 0.5) * tile_size
                    tile_z = z - hd + (tz + 0.5) * tile_size
                    
                    structure.floors.append(Floor(
                        x=tile_x, y=floor_y, z=tile_z,
                        width=tile_size + 0.5,  # Overlap to prevent seam gaps
                        depth=tile_size + 0.5,  # Overlap to prevent seam gaps
                        thickness=0.5,  # Thicker for better collision
                        color=floor_color
                    ))
        
        # --- Walls ---
        wall_height_base = floor_height
        
        # For ground floor on slopes, walls extend down to min terrain
        if floor_idx == 0 and has_slope and terrain_heights is not None:
            min_terrain = min(terrain_heights)
            wall_base = min_terrain - 1.0  # Extend below for safety
            extended_wall_height = (floor_y + wall_height_base) - wall_base
        else:
            wall_base = floor_y
            extended_wall_height = wall_height_base
        
        def add_wall_segment(wx, wz, rotation, is_door_pos):
            """Add a wall segment, handling doors, windows, and open walls."""
            # DOOR: Completely skip this wall segment (full height opening)
            if is_door_pos and floor_idx == 0:
                return  # No wall at all = empty door
            
            # OPEN WALL (balcony/porch): Skip on upper floors sometimes
            if floor_idx > 0 and rng.random() < open_wall_chance:
                # Add a railing instead (half-height wall)
                structure.walls.append(Wall(
                    x=wx, y=floor_y, z=wz,
                    width=tile_size - 0.2, height=1.2,  # Railing height
                    thickness=0.15, rotation=rotation, 
                    color=(wall_color[0]*0.8, wall_color[1]*0.8, wall_color[2]*0.8)
                ))
                return
            
            # WINDOW: Add wall with gap in middle (only if not ground floor on slope)
            if rng.random() < window_chance and floor_idx > 0:
                window_bottom = 1.5
                window_height = 2.0
                # Wall below window
                structure.walls.append(Wall(
                    x=wx, y=floor_y, z=wz,
                    width=tile_size - 0.2, height=window_bottom,
                    thickness=wall_thickness, rotation=rotation, color=wall_color
                ))
                # Wall above window
                structure.walls.append(Wall(
                    x=wx, y=floor_y + window_bottom + window_height, z=wz,
                    width=tile_size - 0.2, height=wall_height_base - window_bottom - window_height,
                    thickness=wall_thickness, rotation=rotation, color=wall_color
                ))
                return
            
            # SOLID WALL - extend to ground on slopes for floor 0
            structure.walls.append(Wall(
                x=wx, y=wall_base, z=wz,
                width=tile_size - 0.2, height=extended_wall_height,
                thickness=wall_thickness, rotation=rotation, color=wall_color
            ))
        
        # Front wall (+Z side)
        for tx in range(tiles_x):
            wx = x - hw + (tx + 0.5) * tile_size
            wz = z + hd
            is_door = (door_wall == 0 and tx == door_tile)
            add_wall_segment(wx, wz, 0, is_door)
        
        # Back wall (-Z side)
        for tx in range(tiles_x):
            wx = x - hw + (tx + 0.5) * tile_size
            wz = z - hd
            is_door = (door_wall == 1 and tx == door_tile)
            add_wall_segment(wx, wz, 180, is_door)
        
        # Left wall (-X side)
        for tz in range(tiles_z):
            wx = x - hw
            wz = z - hd + (tz + 0.5) * tile_size
            is_door = (door_wall == 2 and tz == door_tile)
            add_wall_segment(wx, wz, 90, is_door)
        
        # Right wall (+X side)
        for tz in range(tiles_z):
            wx = x + hw
            wz = z - hd + (tz + 0.5) * tile_size
            is_door = (door_wall == 3 and tz == door_tile)
            add_wall_segment(wx, wz, -90, is_door)
        
        # --- Ramp to next floor (if not top floor) ---
        if floor_idx < floors - 1 and ramp_direction is not None:
            if ramp_direction == "x":
                ramp_start_x = x - hw + ramp_tile_x * tile_size
                ramp_z = z - hd + (ramp_tile_z + 0.5) * tile_size
                ramp_length = tile_size * 2 - 0.5  # Spans 2 tiles with margin
                ramp_rot = 0 if ramp_tile_z == 0 else 0  # Along X axis
                
                structure.ramps.append(Ramp(
                    x=ramp_start_x,
                    y_bottom=floor_y,  # Start at floor level
                    z=ramp_z,
                    length=ramp_length + 1.0,  # Extend past floor edge
                    height=floor_height + 0.5,  # Extend above floor for overlap
                    width=tile_size,  # Full tile width
                    rotation=ramp_rot,
                    color=ramp_color
                ))
            else:  # ramp_direction == "z"
                ramp_x = x - hw + (ramp_tile_x + 0.5) * tile_size
                ramp_start_z = z - hd + ramp_tile_z * tile_size
                ramp_length = tile_size * 2 - 0.5
                
                structure.ramps.append(Ramp(
                    x=ramp_x,
                    y_bottom=floor_y,  # Start at floor level
                    z=ramp_start_z,
                    length=ramp_length + 1.0,  # Extend past floor edge
                    height=floor_height + 0.5,  # Extend above floor for overlap
                    width=tile_size,  # Full tile width
                    rotation=90,  # Along Z axis
                    color=ramp_color
                ))
    
    # =========================================================================
    # ROOF
    # =========================================================================
    roof_y = y + floors * floor_height
    for tx in range(tiles_x):
        for tz in range(tiles_z):
            tile_x = x - hw + (tx + 0.5) * tile_size
            tile_z = z - hd + (tz + 0.5) * tile_size
            
            structure.floors.append(Floor(
                x=tile_x, y=roof_y, z=tile_z,
                width=tile_size + 0.5,  # Overlap
                depth=tile_size + 0.5,  # Overlap
                thickness=0.5,  # Thicker
                color=roof_color
            ))
    
    structure.compute_bounds()
    return structure


def generate_maze(x: float, y: float, z: float,
                  width: int = 10, depth: int = 10,
                  floors: int = 1,
                  cell_size: float = 6.0,
                  wall_height: float = 10.0,
                  seed: int = 42,
                  terrain_heights: List[float] = None) -> Structure:
    """Generate a maze structure using recursive backtracking.
    
    Args:
        x, y, z: World position (center)
        width, depth: Number of cells in X and Z
        floors: Number of vertical floors (1 = 2D maze, >1 = 3D maze)
        cell_size: Size of each cell in world units
        wall_height: Height of maze walls
        seed: Random seed for reproducible mazes
        terrain_heights: Corner heights for pillar generation
    
    Returns:
        Structure with maze walls, floors, and entrance/exit
    """
    rng = np.random.default_rng(seed)
    structure = Structure(x=x, y=y, z=z)
    
    # Colors for maze
    wall_colors = [
        (0.5, 0.4, 0.3),   # Brown stone
        (0.55, 0.55, 0.5), # Gray stone
        (0.4, 0.45, 0.35), # Mossy
        (0.6, 0.5, 0.4),   # Sandy
    ]
    wall_color = wall_colors[rng.integers(0, len(wall_colors))]
    floor_color = (wall_color[0] * 0.8, wall_color[1] * 0.8, wall_color[2] * 0.8)
    
    wall_thickness = 0.5
    
    # Maze dimensions in world units
    maze_width = width * cell_size
    maze_depth = depth * cell_size
    half_width = maze_width / 2
    half_depth = maze_depth / 2
    
    # Generate maze using recursive backtracking
    # Grid: True = wall, False = passage
    # We use a grid where odd cells are passages and even cells are walls
    grid_w = width * 2 + 1
    grid_h = depth * 2 + 1
    
    def generate_maze_grid(floor_num: int) -> np.ndarray:
        """Generate a maze grid for one floor."""
        floor_seed = seed + floor_num * 1000
        floor_rng = np.random.default_rng(floor_seed)
        
        grid = np.ones((grid_h, grid_w), dtype=bool)  # All walls
        
        # Start from random cell
        start_x = floor_rng.integers(0, width) * 2 + 1
        start_z = floor_rng.integers(0, depth) * 2 + 1
        grid[start_z, start_x] = False
        
        # Stack for backtracking
        stack = [(start_x, start_z)]
        
        while stack:
            cx, cz = stack[-1]
            
            # Find unvisited neighbors (2 cells away)
            neighbors = []
            for dx, dz in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, nz = cx + dx, cz + dz
                if 0 < nx < grid_w - 1 and 0 < nz < grid_h - 1:
                    if grid[nz, nx]:  # Still a wall (unvisited)
                        neighbors.append((nx, nz, dx // 2, dz // 2))
            
            if neighbors:
                # Choose random neighbor
                nx, nz, dx, dz = neighbors[floor_rng.integers(0, len(neighbors))]
                # Carve path
                grid[cz + dz, cx + dx] = False  # Wall between
                grid[nz, nx] = False  # Cell
                stack.append((nx, nz))
            else:
                stack.pop()
        
        return grid
    
    # Generate each floor
    for floor_num in range(floors):
        floor_y = y + floor_num * wall_height
        grid = generate_maze_grid(floor_num)
        
        # Add entrance on first floor (south wall)
        if floor_num == 0:
            entrance_x = rng.integers(1, width) * 2
            grid[0, entrance_x] = False
            grid[1, entrance_x] = False
        
        # Add exit on first floor (north wall)
        if floor_num == 0:
            exit_x = rng.integers(1, width) * 2
            grid[grid_h - 1, exit_x] = False
            grid[grid_h - 2, exit_x] = False
        
        # Add stairs between floors (for 3D maze)
        if floor_num < floors - 1:
            # Find a few open cells to add stairs
            stair_count = max(1, min(3, (width * depth) // 20))
            stair_cells = []
            for _ in range(stair_count):
                for attempt in range(50):
                    sx = rng.integers(0, width) * 2 + 1
                    sz = rng.integers(0, depth) * 2 + 1
                    if not grid[sz, sx] and (sx, sz) not in stair_cells:
                        stair_cells.append((sx, sz))
                        break
            
            # Add ramps at stair positions
            for sx, sz in stair_cells:
                cell_world_x = x - half_width + sx * cell_size / 2
                cell_world_z = z - half_depth + sz * cell_size / 2
                
                structure.ramps.append(Ramp(
                    x=cell_world_x,
                    y_bottom=floor_y,
                    z=cell_world_z,
                    width=cell_size * 0.8,
                    length=cell_size * 0.8,
                    height=wall_height,
                    rotation=rng.choice([0, 90, 180, 270]),
                    color=(0.4, 0.35, 0.3)
                ))
        
        # Calculate wall base height - walls should extend to lowest terrain
        min_terrain = y
        if terrain_heights and len(terrain_heights) >= 4:
            min_terrain = min(terrain_heights)
        
        # For ground floor, walls extend from min terrain to floor + wall_height
        if floor_num == 0:
            wall_base_y = min_terrain - 1.0  # Extend slightly below for safety
            total_wall_height = (floor_y + wall_height) - wall_base_y
        else:
            wall_base_y = floor_y
            total_wall_height = wall_height
        
        # Generate walls from grid - create individual wall segments for each wall cell
        # This ensures proper maze structure with walls in both directions
        cell_world_size = cell_size / 2  # Each grid cell is half the cell_size
        
        for gz in range(grid_h):
            for gx in range(grid_w):
                if not grid[gz, gx]:  # Skip passages
                    continue
                    
                # World position for this wall cell
                wx = x - half_width + gx * cell_world_size
                wz = z - half_depth + gz * cell_world_size
                
                # Check which neighbors are also walls to determine wall orientation
                left_wall = gx > 0 and grid[gz, gx - 1]
                right_wall = gx < grid_w - 1 and grid[gz, gx + 1]
                up_wall = gz > 0 and grid[gz - 1, gx]
                down_wall = gz < grid_h - 1 and grid[gz + 1, gx]
                
                # Create wall segment based on connectivity
                horiz = left_wall or right_wall
                vert = up_wall or down_wall
                
                if horiz and not vert:
                    # Horizontal wall segment
                    structure.walls.append(Wall(
                        x=wx, y=wall_base_y, z=wz,
                        width=cell_world_size + wall_thickness,
                        height=total_wall_height,
                        thickness=wall_thickness,
                        rotation=0,
                        color=wall_color
                    ))
                elif vert and not horiz:
                    # Vertical wall segment (rotated 90 degrees)
                    structure.walls.append(Wall(
                        x=wx, y=wall_base_y, z=wz,
                        width=cell_world_size + wall_thickness,
                        height=total_wall_height,
                        thickness=wall_thickness,
                        rotation=90,
                        color=wall_color
                    ))
                else:
                    # Junction or corner - create a small pillar/post
                    structure.walls.append(Wall(
                        x=wx, y=wall_base_y, z=wz,
                        width=wall_thickness * 1.5,
                        height=total_wall_height,
                        thickness=wall_thickness * 1.5,
                        rotation=0,
                        color=wall_color
                    ))
        
        # ALWAYS add floor for mazes (they need bounded walkspace)
        structure.floors.append(Floor(
            x=x, y=floor_y, z=z,
            width=maze_width + 2,
            depth=maze_depth + 2,
            thickness=0.5,  # Thicker for better collision
            color=floor_color
        ))
        
        # Add ceiling for top floor
        if floor_num == floors - 1:
            structure.floors.append(Floor(
                x=x, y=floor_y + wall_height, z=z,
                width=maze_width + 2,
                depth=maze_depth + 2,
                thickness=0.3,
                color=floor_color
            ))
    
    # Add support pillars at corners if on slope
    if terrain_heights and len(terrain_heights) >= 4:
        min_terrain = min(terrain_heights)
        if y - min_terrain > 2:
            corners = [
                (x - half_width, z - half_depth, terrain_heights[0]),  # -X -Z
                (x + half_width, z - half_depth, terrain_heights[1]),  # +X -Z
                (x - half_width, z + half_depth, terrain_heights[2]),  # -X +Z
                (x + half_width, z + half_depth, terrain_heights[3]),  # +X +Z
            ]
            for px, pz, local_ground in corners:
                if y - local_ground > 1:
                    structure.pillars.append(Pillar(
                        x=px, y_bottom=local_ground - 1.0, y_top=y + 0.5, z=pz,
                        width=1.0,
                        color=(wall_color[0] * 0.7, wall_color[1] * 0.7, wall_color[2] * 0.7)
                    ))
        
        # Add entry ramp from lowest terrain point to maze floor
        # Find which edge has lowest terrain
        edge_heights = [
            (terrain_heights[0] + terrain_heights[2]) / 2,  # -X edge (west)
            (terrain_heights[1] + terrain_heights[3]) / 2,  # +X edge (east)
            (terrain_heights[0] + terrain_heights[1]) / 2,  # -Z edge (south)
            (terrain_heights[2] + terrain_heights[3]) / 2,  # +Z edge (north)
        ]
        lowest_edge = edge_heights.index(min(edge_heights))
        lowest_height = min(edge_heights)
        ramp_height_diff = y - lowest_height
        
        # Only add ramp if there's a significant height difference
        if ramp_height_diff > 1.5:
            ramp_length = max(ramp_height_diff * 2, cell_size * 2)  # 45° or gentler
            ramp_width = cell_size * 0.8
            
            if lowest_edge == 0:  # West edge (-X)
                rx = x - half_width - ramp_length / 2
                rz = z
                ramp_rot = 90
            elif lowest_edge == 1:  # East edge (+X)
                rx = x + half_width + ramp_length / 2
                rz = z
                ramp_rot = 270
            elif lowest_edge == 2:  # South edge (-Z)
                rx = x
                rz = z - half_depth - ramp_length / 2
                ramp_rot = 0
            else:  # North edge (+Z)
                rx = x
                rz = z + half_depth + ramp_length / 2
                ramp_rot = 180
            
            structure.ramps.append(Ramp(
                x=rx,
                y_bottom=lowest_height,
                z=rz,
                width=ramp_width,
                length=ramp_length,
                height=ramp_height_diff,
                rotation=ramp_rot,
                color=(floor_color[0] * 0.9, floor_color[1] * 0.9, floor_color[2] * 0.9)
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
        """Draw a simple flat ramp/slope - just the walking surface, no side walls."""
        glPushMatrix()
        glTranslatef(ramp.x, ramp.y_bottom, ramp.z)
        glRotatef(ramp.rotation, 0, 1, 0)
        
        hw = ramp.width / 2
        length = ramp.length
        height = ramp.height
        thickness = 0.3  # Thin platform
        
        glColor3f(*ramp.color)
        
        # Calculate normal for the slope
        # Slope goes from (0, 0) to (length, height)
        slope_len = math.sqrt(length*length + height*height)
        nx = -height / slope_len
        ny = length / slope_len
        
        glBegin(GL_QUADS)
        # Top surface (the walking surface)
        glNormal3f(nx, ny, 0)
        glVertex3f(0, 0, -hw)
        glVertex3f(0, 0, hw)
        glVertex3f(length, height, hw)
        glVertex3f(length, height, -hw)
        
        # Bottom surface (underneath the ramp)
        glNormal3f(-nx, -ny, 0)
        glVertex3f(0, -thickness, hw)
        glVertex3f(0, -thickness, -hw)
        glVertex3f(length, height - thickness, -hw)
        glVertex3f(length, height - thickness, hw)
        
        # Front edge (at bottom of ramp)
        glNormal3f(-1, 0, 0)
        glVertex3f(0, -thickness, -hw)
        glVertex3f(0, -thickness, hw)
        glVertex3f(0, 0, hw)
        glVertex3f(0, 0, -hw)
        
        # Back edge (at top of ramp)
        glNormal3f(1, 0, 0)
        glVertex3f(length, height - thickness, hw)
        glVertex3f(length, height - thickness, -hw)
        glVertex3f(length, height, -hw)
        glVertex3f(length, height, hw)
        
        # Side edges
        glNormal3f(0, 0, 1)
        glVertex3f(0, 0, hw)
        glVertex3f(0, -thickness, hw)
        glVertex3f(length, height - thickness, hw)
        glVertex3f(length, height, hw)
        
        glNormal3f(0, 0, -1)
        glVertex3f(0, -thickness, -hw)
        glVertex3f(0, 0, -hw)
        glVertex3f(length, height, -hw)
        glVertex3f(length, height - thickness, -hw)
        glEnd()
        
        glPopMatrix()
    
    @staticmethod
    def _draw_pillar(pillar: Pillar):
        """Draw a vertical support pillar."""
        glPushMatrix()
        glTranslatef(pillar.x, pillar.y_bottom, pillar.z)
        
        hw = pillar.width / 2
        h = pillar.y_top - pillar.y_bottom
        
        glColor3f(*pillar.color)
        
        glBegin(GL_QUADS)
        # Front
        glNormal3f(0, 0, 1)
        glVertex3f(-hw, 0, hw)
        glVertex3f(hw, 0, hw)
        glVertex3f(hw, h, hw)
        glVertex3f(-hw, h, hw)
        # Back
        glNormal3f(0, 0, -1)
        glVertex3f(hw, 0, -hw)
        glVertex3f(-hw, 0, -hw)
        glVertex3f(-hw, h, -hw)
        glVertex3f(hw, h, -hw)
        # Left
        glNormal3f(-1, 0, 0)
        glVertex3f(-hw, 0, -hw)
        glVertex3f(-hw, 0, hw)
        glVertex3f(-hw, h, hw)
        glVertex3f(-hw, h, -hw)
        # Right
        glNormal3f(1, 0, 0)
        glVertex3f(hw, 0, hw)
        glVertex3f(hw, 0, -hw)
        glVertex3f(hw, h, -hw)
        glVertex3f(hw, h, hw)
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
        
        for pillar in structure.pillars:
            StructureRenderer._draw_pillar(pillar)


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
        if rng.random() > 0.03:  # 3% chance (slightly more buildings)
            return
        
        # Find a spot
        h, w = heightmap.shape
        local_x = rng.integers(8, w - 8)
        local_z = rng.integers(8, h - 8)
        
        # Check height range
        center_h = heightmap[local_z, local_x]
        if center_h < 3 or center_h > 50:  # Not underwater or extreme peaks
            return
        
        # Pre-calculate building size to sample correct terrain positions
        tiles_x_temp = 2 + rng.integers(0, 3)  # Same as below
        tiles_z_temp = 2 + rng.integers(0, 3)
        tile_size_temp = 10.0 + rng.random() * 4.0
        
        # Sample terrain at actual building corners (in heightmap coords)
        building_half_width = int((tiles_x_temp * tile_size_temp / 2) / tile_scale)
        building_half_depth = int((tiles_z_temp * tile_size_temp / 2) / tile_scale)
        
        # Clamp to valid heightmap range
        h, w = heightmap.shape
        x_min = max(0, local_x - building_half_width)
        x_max = min(w - 1, local_x + building_half_width)
        z_min = max(0, local_z - building_half_depth)
        z_max = min(h - 1, local_z + building_half_depth)
        
        corner_heights = [
            heightmap[z_min, x_min] * height_scale,  # -X -Z corner
            heightmap[z_min, x_max] * height_scale,  # +X -Z corner
            heightmap[z_max, x_min] * height_scale,  # -X +Z corner
            heightmap[z_max, x_max] * height_scale,  # +X +Z corner
        ]
        max_slope = max(corner_heights) - min(corner_heights)
        
        # Allow buildings on steeper slopes (pillars will handle it)
        if max_slope > 20:  # But not too extreme
            return
        
        # IMPORTANT: Scale local coordinates by tile_scale to get world coords
        world_x = chunk_world_x + local_x * tile_scale
        world_z = chunk_world_z + local_z * tile_scale
        # Use the maximum height as the floor level (building sits on top)
        world_y = max(corner_heights)
        
        # Decide building type: regular building or maze
        building_type_roll = rng.random()
        
        if building_type_roll < 0.15:  # 15% chance of maze
            # Generate maze
            maze_roll = rng.random()
            if maze_roll < 0.7:  # 70% of mazes are 2D (single floor)
                maze_floors = 1
                maze_width = 6 + rng.integers(0, 6)  # 6-11 cells
                maze_depth = 6 + rng.integers(0, 6)
            else:  # 30% of mazes are 3D (multi-floor)
                maze_floors = 2 + rng.integers(0, 2)  # 2-3 floors
                maze_width = 5 + rng.integers(0, 4)  # 5-8 cells (smaller for 3D)
                maze_depth = 5 + rng.integers(0, 4)
            
            cell_size = 5.0 + rng.random() * 3.0  # 5-8 units per cell
            
            building = generate_maze(
                world_x, world_y, world_z,
                width=maze_width,
                depth=maze_depth,
                floors=maze_floors,
                cell_size=cell_size,
                wall_height=10.0 + rng.random() * 5.0,  # 10-15 units tall walls
                seed=chunk_seed,
                terrain_heights=corner_heights
            )
        else:
            # Regular building
            floor_roll = rng.random()
            if floor_roll < 0.2:
                floors = 1  # 20% single floor
            elif floor_roll < 0.5:
                floors = 2  # 30% two floors
            elif floor_roll < 0.8:
                floors = 3  # 30% three floors
            else:
                floors = 4 + rng.integers(0, 3)  # 20% tall (4-6 floors)
            
            # Use pre-calculated values from terrain sampling above
            tiles_x = tiles_x_temp
            tiles_z = tiles_z_temp
            tile_size = tile_size_temp
            
            building = generate_tile_building(
                world_x, world_y, world_z,
                tiles_x=tiles_x,
                tiles_z=tiles_z,
                floors=floors,
                tile_size=tile_size,
                seed=chunk_seed,
                terrain_heights=corner_heights
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
            
            # Check walls - use player body height for proper collision
            for wall in structure.walls:
                if wall.check_collision(px, py, pz, radius, player_height):
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

