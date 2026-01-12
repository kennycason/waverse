"""
Building Generator - Creates building geometry from BuildingDNA.

This module generates the actual primitives (walls, floors, roofs, etc.)
from a BuildingDNA definition, then caches them in display lists for performance.
"""

import math
from typing import List, Tuple, Optional
import numpy as np
from OpenGL.GL import *

from .building_dna import BuildingDNA, BuildingType, RoofType, WallStyle, ColorPalette
from .structures import Structure, Wall, Floor, Ramp, Pillar, Doorway, Arch, Staircase


def generate_building_from_dna(
    x: float, y: float, z: float,
    dna: BuildingDNA,
    terrain_heights: Optional[List[float]] = None
) -> Structure:
    """
    Generate a complete building structure from DNA.
    
    Args:
        x, y, z: World position (center bottom of building)
        dna: BuildingDNA defining the building's characteristics
        terrain_heights: [h00, h10, h01, h11] corner heights for sloped terrain
    
    Returns:
        Structure with all primitives ready for rendering
    """
    # Special case: Parkour gets its own generator
    if dna.building_type == BuildingType.PARKOUR:
        return _generate_parkour(x, y, z, dna, terrain_heights)
    
    rng = np.random.default_rng(dna.seed)
    structure = Structure(x=x, y=y, z=z)
    
    hw = dna.width / 2
    hd = dna.depth / 2
    
    # Determine base height (highest terrain point)
    if terrain_heights and len(terrain_heights) >= 4:
        base_y = max(terrain_heights)
        min_terrain = min(terrain_heights)
    else:
        base_y = y
        min_terrain = y
    
    # =========================================================================
    # SUPPORT PILLARS (for sloped terrain)
    # =========================================================================
    if terrain_heights and base_y - min_terrain > 1.5:
        _add_support_pillars(structure, x, base_y, z, dna, terrain_heights, rng)
    
    # =========================================================================
    # FLOORS AND WALLS
    # =========================================================================
    for floor_idx in range(dna.floors):
        floor_y = base_y + floor_idx * dna.floor_height
        
        # Calculate inset for this floor (towers can taper)
        inset = dna.wall_inset * floor_idx / max(1, dna.floors - 1) if dna.wall_inset > 0 else 0
        floor_hw = hw - inset
        floor_hd = hd - inset
        
        # Stair configuration (consistent across floors)
        stair_width = 4.5   # Wide enough to walk on (1.5x for world scale)
        stair_length = dna.floor_height * 0.8  # Matches Staircase.length property (steeper to fit)
        stair_margin = 2.5  # Gap from wall (increased for safety)
        has_stairs = dna.floors > 1 and dna.has_stairs
        
        # Determine stair position (consistent for all floors)
        # entrance_wall: 0=+Z front, 1=-Z back, 2=-X left, 3=+X right
        # Put stairs in back corner, opposite from entrance
        # Stairs go along the back (-Z) wall, running in +X direction
        # This keeps them away from the front entrance
        
        # Position stairs in back-left corner, running toward back-right
        # Stair origin is at bottom step, stairs extend in +X direction
        stair_dir = 0  # Stairs go in +X direction
        stair_x = x - floor_hw + stair_margin  # Start near left wall (inside)
        stair_z = z - floor_hd + stair_margin + stair_width / 2  # Near back wall (inside)
        
        # Opening is where stairs arrive at top (at end of stair length)
        opening_x = stair_x + stair_length / 2  # Center of opening
        opening_z = stair_z
        
        # If entrance is at back, flip stairs to front
        if dna.entrance_wall == 1:  # Entrance at -Z (back)
            stair_z = z + floor_hd - stair_margin - stair_width / 2  # Near front wall
            opening_z = stair_z
        
        floor_color = tuple(c * 0.85 for c in dna.colors.primary)
        
        # Determine if stairs are at front or back
        stairs_at_back = dna.entrance_wall != 1  # True unless entrance is at back
        
        # Add floor slab(s) - with stair opening on upper floors
        if floor_idx > 0 or (terrain_heights and base_y > min_terrain + 1):
            if has_stairs and floor_idx > 0:
                # Upper floor with stair opening - create floor pieces around the opening
                # Opening area: stair_length x stair_width centered at (opening_x, opening_z)
                _add_floor_with_opening(
                    structure, x, floor_y, z, floor_hw, floor_hd,
                    opening_x, opening_z, stair_length, stair_width,
                    stairs_at_back, floor_color
                )
            else:
                # Ground floor or no stairs - full floor
                structure.floors.append(Floor(
                    x=x, y=floor_y, z=z,
                    width=floor_hw * 2 + 0.5,
                    depth=floor_hd * 2 + 0.5,
                    thickness=0.4,
                    color=floor_color
                ))
        
        # Add walls
        _add_floor_walls(structure, x, floor_y, z, floor_hw, floor_hd, 
                        dna, floor_idx, rng, min_terrain if terrain_heights else base_y)
        
        # Add balconies (upper floors only)
        if dna.has_balconies and floor_idx > 0:
            _add_balconies(structure, x, floor_y, z, floor_hw, floor_hd, dna, rng)
        
        # Add interior stairs to next floor (not on top floor)
        if has_stairs and floor_idx < dna.floors - 1:
            structure.staircases.append(Staircase(
                x=stair_x,
                y_bottom=floor_y,
                y_top=floor_y + dna.floor_height,
                z=stair_z,
                width=stair_width,
                direction=stair_dir,
                color=tuple(c * 0.7 for c in dna.colors.primary)
            ))
    
    # =========================================================================
    # ROOF
    # =========================================================================
    roof_y = base_y + dna.floors * dna.floor_height
    _add_roof(structure, x, roof_y, z, hw, hd, dna, rng)
    
    # =========================================================================
    # ENTRANCE / AWNING / ENTRANCE STAIRS
    # =========================================================================
    if dna.has_awning:
        _add_awning(structure, x, base_y, z, hw, hd, dna, rng)
    
    # Add entrance stairs/ramp if ground floor is elevated
    if terrain_heights:
        min_terrain = min(terrain_heights)
        entrance_height = base_y - min_terrain
        if entrance_height > 0.5:  # Door is elevated, need stairs to reach it
            _add_entrance_stairs(structure, x, base_y, z, hw, hd, dna, min_terrain)
    
    # =========================================================================
    # DECORATIVE ARCHES (temples, monuments, villas)
    # =========================================================================
    if dna.building_type in (BuildingType.TEMPLE, BuildingType.MONUMENT, BuildingType.VILLA):
        _add_decorative_arches(structure, x, base_y, z, hw, hd, dna, rng)
    
    # =========================================================================
    # ROOFTOP FEATURES
    # =========================================================================
    if dna.has_rooftop_features:
        _add_rooftop_features(structure, x, roof_y, z, hw, hd, dna, rng)
    
    # =========================================================================
    # CHIMNEY
    # =========================================================================
    if dna.has_chimney:
        _add_chimney(structure, x, roof_y, z, hw, hd, dna, rng)
    
    structure.compute_bounds()
    return structure


def _add_floor_with_opening(structure: Structure, x: float, floor_y: float, z: float,
                            hw: float, hd: float,
                            opening_x: float, opening_z: float,
                            opening_length: float, opening_width: float,
                            stairs_at_back: bool, color: tuple):
    """
    Create floor rectangles around a stair opening.
    Stairs run in X direction, opening is at stair top.
    
    Layout (top view, stairs at back -Z):
    |-----------------------|
    |  [stair area/opening] |   <- back (stair side)
    |-----------------------|
    |                       |
    |    main floor (r2)    |   <- rest of floor
    |                       |
    |-----------------------|
    """
    thickness = 0.4
    gap = 0.2  # Small gap around opening
    
    # Opening dimensions (stairs run in X, opening is stair_length x stair_width)
    opening_half_x = opening_length / 2
    opening_half_z = opening_width / 2
    
    # Main floor section - the large area away from the stair opening
    if stairs_at_back:
        # Opening near -Z, main floor toward +Z
        r2_z1 = opening_z + opening_half_z + gap
        r2_z2 = z + hd
    else:
        # Opening near +Z, main floor toward -Z
        r2_z1 = z - hd
        r2_z2 = opening_z - opening_half_z - gap
    
    if r2_z2 > r2_z1 + 1.0:
        structure.floors.append(Floor(
            x=x, y=floor_y, z=(r2_z1 + r2_z2) / 2,
            width=hw * 2 + 0.5,
            depth=r2_z2 - r2_z1,
            thickness=thickness, color=color
        ))
    
    # Side sections next to opening (left and right of the opening)
    # Left section: from left wall to left edge of opening
    left_x1 = x - hw
    left_x2 = opening_x - opening_half_x - gap
    if left_x2 > left_x1 + 0.5:
        stair_row_z1 = opening_z - opening_half_z
        stair_row_z2 = opening_z + opening_half_z
        structure.floors.append(Floor(
            x=(left_x1 + left_x2) / 2, y=floor_y, z=(stair_row_z1 + stair_row_z2) / 2,
            width=left_x2 - left_x1,
            depth=stair_row_z2 - stair_row_z1,
            thickness=thickness, color=color
        ))
    
    # Right section: from right edge of opening to right wall
    right_x1 = opening_x + opening_half_x + gap
    right_x2 = x + hw
    if right_x2 > right_x1 + 0.5:
        stair_row_z1 = opening_z - opening_half_z
        stair_row_z2 = opening_z + opening_half_z
        structure.floors.append(Floor(
            x=(right_x1 + right_x2) / 2, y=floor_y, z=(stair_row_z1 + stair_row_z2) / 2,
            width=right_x2 - right_x1,
            depth=stair_row_z2 - stair_row_z1,
            thickness=thickness, color=color
        ))


def _add_support_pillars(structure: Structure, x: float, base_y: float, z: float,
                         dna: BuildingDNA, terrain_heights: List[float], 
                         rng: np.random.Generator):
    """Add pillars at corners for buildings on slopes."""
    hw = dna.width / 2
    hd = dna.depth / 2
    
    corners = [
        (x - hw, z - hd, terrain_heights[0]),  # -X -Z
        (x + hw, z - hd, terrain_heights[1]),  # +X -Z
        (x - hw, z + hd, terrain_heights[2]),  # -X +Z
        (x + hw, z + hd, terrain_heights[3]),  # +X +Z
    ]
    
    pillar_color = tuple(c * 0.7 for c in dna.colors.primary)
    pillar_width = 0.8 + rng.random() * 0.4
    
    for px, pz, ground_h in corners:
        if base_y - ground_h > 1.0:
            structure.pillars.append(Pillar(
                x=px, y_bottom=ground_h - 1.0, y_top=base_y + 0.1, z=pz,
                width=pillar_width,
                color=pillar_color
            ))


def _add_floor_walls(structure: Structure, x: float, floor_y: float, z: float,
                     hw: float, hd: float, dna: BuildingDNA, floor_idx: int,
                     rng: np.random.Generator, min_terrain: float):
    """Add walls for one floor, with windows and door openings."""
    wall_height = dna.floor_height
    wall_thickness = 0.5
    
    # Ground floor walls extend to min terrain
    if floor_idx == 0:
        actual_base = min_terrain - 0.5
        actual_height = (floor_y + wall_height) - actual_base
    else:
        actual_base = floor_y
        actual_height = wall_height
    
    # Determine door position
    door_wall = dna.entrance_wall
    
    # Windows per wall
    window_spacing = dna.width / (dna.window_cols + 1)
    
    # Create walls for each side
    sides = [
        (0, z + hd, 0, hw * 2, True),    # +Z front
        (0, z - hd, 180, hw * 2, True),  # -Z back
        (x - hw, 0, 90, hd * 2, False),  # -X left
        (x + hw, 0, -90, hd * 2, False), # +X right
    ]
    
    for side_idx, (ox, oz, rot, length, is_xz) in enumerate(sides):
        wx = x + ox if is_xz else ox
        wz = oz if is_xz else z + oz
        
        # Is this the door wall on ground floor?
        is_door_wall = (side_idx == door_wall and floor_idx == 0)
        
        if is_door_wall:
            # Door opening - split wall into two segments
            # Player needs ~4 units height and ~3 units width to fit comfortably
            door_width = 4.0
            door_height = min(5.5, actual_height * 0.9)  # Taller door for player
            left_width = (length - door_width) / 2 - 0.5
            right_width = left_width
            
            # Left segment
            if left_width > 1:
                lx = wx - (length/4 + door_width/4) if is_xz else wx
                lz = wz if is_xz else wz - (length/4 + door_width/4)
                structure.walls.append(Wall(
                    x=lx, y=actual_base, z=lz,
                    width=left_width, height=actual_height,
                    thickness=wall_thickness, rotation=rot,
                    color=dna.colors.primary
                ))
            
            # Right segment
            if right_width > 1:
                rx = wx + (length/4 + door_width/4) if is_xz else wx
                rz = wz if is_xz else wz + (length/4 + door_width/4)
                structure.walls.append(Wall(
                    x=rx, y=actual_base, z=rz,
                    width=right_width, height=actual_height,
                    thickness=wall_thickness, rotation=rot,
                    color=dna.colors.primary
                ))
            
            # Add doorway frame with optional arch
            has_arch = rng.random() < 0.3  # 30% chance of arched doorway
            structure.doorways.append(Doorway(
                x=wx, y=floor_y, z=wz,
                width=door_width, height=door_height,
                rotation=rot,
                frame_color=tuple(c * 0.6 for c in dna.colors.primary),
                has_arch=has_arch
            ))
        else:
            # Regular wall - may have windows
            if dna.has_windows and floor_idx > 0:
                # Wall with window openings (simplified: just different colored sections)
                # For now, full wall - windows are visual only
                structure.walls.append(Wall(
                    x=wx, y=actual_base, z=wz,
                    width=length - 0.2, height=actual_height,
                    thickness=wall_thickness, rotation=rot,
                    color=dna.colors.primary
                ))
                
                # Add window "frames" as thin protrusions
                _add_window_frames(structure, wx, floor_y, wz, length, dna, rot, is_xz, rng)
            else:
                # Solid wall
                structure.walls.append(Wall(
                    x=wx, y=actual_base, z=wz,
                    width=length - 0.2, height=actual_height,
                    thickness=wall_thickness, rotation=rot,
                    color=dna.colors.primary
                ))


def _add_window_frames(structure: Structure, wx: float, floor_y: float, wz: float,
                       wall_length: float, dna: BuildingDNA, rotation: float,
                       is_xz: bool, rng: np.random.Generator):
    """Add decorative window frames to a wall."""
    # Window parameters
    window_height = dna.floor_height * 0.4
    window_width = 1.5 + rng.random()
    window_bottom = floor_y + dna.floor_height * 0.3
    
    num_windows = min(dna.window_cols, int(wall_length / 4))
    if num_windows < 1:
        return
    
    spacing = wall_length / (num_windows + 1)
    
    rad = math.radians(rotation)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    
    for i in range(num_windows):
        offset = -wall_length/2 + spacing * (i + 1)
        
        # Calculate window center position
        if abs(rotation) < 1 or abs(rotation - 180) < 1:
            # Wall along X axis
            win_x = wx + offset
            win_z = wz + (0.3 if rotation < 90 else -0.3)
        else:
            # Wall along Z axis
            win_x = wx + (0.3 if rotation > 0 else -0.3)
            win_z = wz + offset
        
        # Add window as small floor (horizontal element representing window sill)
        structure.floors.append(Floor(
            x=win_x, y=window_bottom, z=win_z,
            width=window_width, depth=0.15,
            thickness=0.1,
            color=dna.colors.window
        ))


def _add_balconies(structure: Structure, x: float, floor_y: float, z: float,
                   hw: float, hd: float, dna: BuildingDNA, rng: np.random.Generator):
    """Add balconies to upper floors."""
    balcony_depth = 2.0 + rng.random()
    balcony_width = dna.width * 0.4
    railing_height = 1.2
    
    # Random side for balcony
    sides = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    side = sides[rng.integers(0, 4)]
    
    if rng.random() < dna.balcony_chance:
        bx = x + side[0] * (hw + balcony_depth/2)
        bz = z + side[1] * (hd + balcony_depth/2)
        
        # Balcony floor
        if side[0] != 0:  # Left or right
            bw, bd = balcony_depth, balcony_width
        else:  # Front or back
            bw, bd = balcony_width, balcony_depth
        
        structure.floors.append(Floor(
            x=bx, y=floor_y, z=bz,
            width=bw, depth=bd,
            thickness=0.2,
            color=dna.colors.secondary
        ))
        
        # Railing (simplified as low walls)
        railing_color = tuple(c * 0.8 for c in dna.colors.secondary)
        
        # Front railing
        if side[0] != 0:
            structure.walls.append(Wall(
                x=bx + side[0] * balcony_depth/2, y=floor_y, z=bz,
                width=balcony_width - 0.2, height=railing_height,
                thickness=0.1, rotation=90 if side[0] > 0 else -90,
                color=railing_color
            ))
        else:
            structure.walls.append(Wall(
                x=bx, y=floor_y, z=bz + side[1] * balcony_depth/2,
                width=balcony_width - 0.2, height=railing_height,
                thickness=0.1, rotation=0 if side[1] > 0 else 180,
                color=railing_color
            ))


def _add_roof(structure: Structure, x: float, roof_y: float, z: float,
              hw: float, hd: float, dna: BuildingDNA, rng: np.random.Generator):
    """Add roof based on roof type."""
    
    if dna.roof_type == RoofType.FLAT:
        # Flat roof with parapet
        structure.floors.append(Floor(
            x=x, y=roof_y, z=z,
            width=hw * 2 + 0.5, depth=hd * 2 + 0.5,
            thickness=0.5,
            color=dna.colors.roof
        ))
        
        # Optional parapet walls
        if rng.random() < 0.5:
            parapet_height = 0.8 + rng.random() * 0.5
            parapet_color = tuple(c * 0.9 for c in dna.colors.roof)
            for dx, dz, rot in [(0, hd, 0), (0, -hd, 180), (-hw, 0, 90), (hw, 0, -90)]:
                structure.walls.append(Wall(
                    x=x + dx, y=roof_y + 0.5, z=z + dz,
                    width=hw * 2 if dx == 0 else hd * 2,
                    height=parapet_height, thickness=0.15, rotation=rot,
                    color=parapet_color
                ))
    
    elif dna.roof_type == RoofType.GABLED:
        # Gabled roof (triangular)
        peak_height = min(hw, hd) * 0.5
        _add_gabled_roof(structure, x, roof_y, z, hw, hd, peak_height, dna.colors.roof, rng)
    
    elif dna.roof_type == RoofType.HIPPED:
        # Hipped roof (all sides slope)
        peak_height = min(hw, hd) * 0.4
        _add_hipped_roof(structure, x, roof_y, z, hw, hd, peak_height, dna.colors.roof, rng)
    
    elif dna.roof_type == RoofType.PYRAMID:
        # Pyramid roof (four-sided)
        peak_height = min(hw, hd) * 0.6
        _add_pyramid_roof(structure, x, roof_y, z, hw, hd, peak_height, dna.colors.roof, rng)
    
    elif dna.roof_type == RoofType.DOME:
        # Dome roof (approximated with steps)
        dome_height = min(hw, hd) * 0.5
        _add_dome_roof(structure, x, roof_y, z, min(hw, hd), dome_height, dna.colors.roof, rng)
    
    elif dna.roof_type == RoofType.SHED:
        # Shed roof (single slope)
        slope_height = hd * 0.3
        _add_shed_roof(structure, x, roof_y, z, hw, hd, slope_height, dna.colors.roof, rng)
    
    elif dna.roof_type == RoofType.STEPPED:
        # Stepped/ziggurat roof
        _add_stepped_roof(structure, x, roof_y, z, hw, hd, dna.colors.roof, rng)
    
    elif dna.roof_type == RoofType.CONICAL:
        # Conical/pointed roof (lighthouse, tower)
        cone_height = min(hw, hd) * 0.8
        _add_conical_roof(structure, x, roof_y, z, min(hw, hd), cone_height, dna.colors.roof, rng)
    
    elif dna.roof_type == RoofType.PAGODA:
        # Multi-tiered Asian style
        _add_pagoda_roof(structure, x, roof_y, z, hw, hd, dna.colors.roof, rng)
    
    elif dna.roof_type == RoofType.MANSARD:
        # French style with two slopes
        _add_mansard_roof(structure, x, roof_y, z, hw, hd, dna.colors.roof, rng)
    
    elif dna.roof_type == RoofType.BUTTERFLY:
        # Modern V-shaped roof
        _add_butterfly_roof(structure, x, roof_y, z, hw, hd, dna.colors.roof, rng)
    
    elif dna.roof_type == RoofType.GEODESIC:
        # Triangulated dome
        dome_height = min(hw, hd) * 0.5
        _add_geodesic_roof(structure, x, roof_y, z, min(hw, hd), dome_height, dna.colors.roof, rng)
    
    elif dna.roof_type == RoofType.SAWTOOTH:
        # Industrial zigzag
        _add_sawtooth_roof(structure, x, roof_y, z, hw, hd, dna.colors.roof, rng)
    
    elif dna.roof_type == RoofType.BARREL:
        # Curved/cylindrical roof
        barrel_height = min(hw, hd) * 0.35
        _add_barrel_roof(structure, x, roof_y, z, hw, hd, barrel_height, dna.colors.roof, rng)


def _add_gabled_roof(structure: Structure, x: float, y: float, z: float,
                     hw: float, hd: float, peak_height: float,
                     color: Tuple[float, float, float], rng: np.random.Generator):
    """Add a gabled (triangular) roof using ramps."""
    # Two sloped surfaces meeting at a ridge
    # Front slope
    structure.ramps.append(Ramp(
        x=x - hw, y_bottom=y, z=z,
        length=hw, height=peak_height,
        width=hd * 2,
        rotation=0,
        color=color
    ))
    # Back slope  
    structure.ramps.append(Ramp(
        x=x + hw, y_bottom=y, z=z,
        length=hw, height=peak_height,
        width=hd * 2,
        rotation=180,
        color=color
    ))


def _add_hipped_roof(structure: Structure, x: float, y: float, z: float,
                     hw: float, hd: float, peak_height: float,
                     color: Tuple[float, float, float], rng: np.random.Generator):
    """Add a hipped roof (all sides slope toward center)."""
    # Simplified: flat top section + sloped edges
    inner_hw = hw * 0.5
    inner_hd = hd * 0.5
    
    # Top platform
    structure.floors.append(Floor(
        x=x, y=y + peak_height, z=z,
        width=inner_hw * 2, depth=inner_hd * 2,
        thickness=0.3,
        color=color
    ))
    
    # Slopes on each side
    slope_width = hw - inner_hw
    for dx, dz, rot in [(1, 0, -90), (-1, 0, 90), (0, 1, 0), (0, -1, 180)]:
        if dx != 0:
            rx = x + dx * (hw - slope_width/2)
            rz = z
            rw = hd * 2
        else:
            rx = x
            rz = z + dz * (hd - slope_width/2)
            rw = hw * 2
        
        structure.ramps.append(Ramp(
            x=rx if dx == 0 else rx - slope_width/2,
            y_bottom=y, z=rz if dz == 0 else rz - slope_width/2,
            length=slope_width, height=peak_height,
            width=rw,
            rotation=rot,
            color=color
        ))


def _add_pyramid_roof(structure: Structure, x: float, y: float, z: float,
                      hw: float, hd: float, peak_height: float,
                      color: Tuple[float, float, float], rng: np.random.Generator):
    """Add a four-sided pyramid roof."""
    # Similar to hipped but comes to a point
    steps = 3
    for i in range(steps):
        t = i / steps
        level_y = y + peak_height * t
        scale = 1.0 - t * 0.9
        
        structure.floors.append(Floor(
            x=x, y=level_y, z=z,
            width=hw * 2 * scale, depth=hd * 2 * scale,
            thickness=peak_height / steps,
            color=tuple(c * (0.9 + 0.1 * t) for c in color)
        ))


def _add_dome_roof(structure: Structure, x: float, y: float, z: float,
                   radius: float, height: float,
                   color: Tuple[float, float, float], rng: np.random.Generator):
    """Add a dome roof approximated with stacked cylinders/rings."""
    segments = 5
    for i in range(segments):
        t = i / segments
        level_y = y + height * math.sin(t * math.pi / 2)
        scale = math.cos(t * math.pi / 2)
        
        if scale > 0.1:
            structure.floors.append(Floor(
                x=x, y=level_y, z=z,
                width=radius * 2 * scale, depth=radius * 2 * scale,
                thickness=height / segments * 1.5,
                color=tuple(c * (0.85 + 0.15 * t) for c in color)
            ))


def _add_shed_roof(structure: Structure, x: float, y: float, z: float,
                   hw: float, hd: float, slope_height: float,
                   color: Tuple[float, float, float], rng: np.random.Generator):
    """Add a shed roof (single slope)."""
    structure.ramps.append(Ramp(
        x=x, y_bottom=y, z=z - hd,
        length=hd * 2, height=slope_height,
        width=hw * 2,
        rotation=0,
        color=color
    ))


def _add_stepped_roof(structure: Structure, x: float, y: float, z: float,
                      hw: float, hd: float,
                      color: Tuple[float, float, float], rng: np.random.Generator):
    """Add a stepped/ziggurat roof."""
    steps = 3 + rng.integers(0, 3)
    step_height = 1.5
    
    for i in range(steps):
        scale = 1.0 - (i + 1) * 0.15
        level_y = y + i * step_height
        
        structure.floors.append(Floor(
            x=x, y=level_y, z=z,
            width=hw * 2 * scale, depth=hd * 2 * scale,
            thickness=step_height,
            color=tuple(c * (0.9 + 0.03 * i) for c in color)
        ))


def _add_conical_roof(structure: Structure, x: float, y: float, z: float,
                      radius: float, height: float,
                      color: Tuple[float, float, float], rng: np.random.Generator):
    """Add a conical/pointed roof (lighthouse, tower)."""
    segments = 6
    for i in range(segments):
        t = i / segments
        level_y = y + height * t
        scale = 1.0 - t * 0.85  # Taper to point
        
        if scale > 0.05:
            structure.floors.append(Floor(
                x=x, y=level_y, z=z,
                width=radius * 2 * scale, depth=radius * 2 * scale,
                thickness=height / segments * 1.2,
                color=tuple(c * (0.9 + 0.1 * t) for c in color)
            ))


def _add_pagoda_roof(structure: Structure, x: float, y: float, z: float,
                     hw: float, hd: float,
                     color: Tuple[float, float, float], rng: np.random.Generator):
    """Add a multi-tiered pagoda-style roof."""
    tiers = 2 + rng.integers(0, 2)
    tier_height = min(hw, hd) * 0.25
    
    for tier in range(tiers):
        tier_y = y + tier * tier_height * 0.8
        # Each tier is slightly smaller and has curved-looking edges (approximated)
        scale = 1.0 - tier * 0.25
        overhang = 1.0 + (tiers - tier) * 0.3  # Lower tiers extend further
        
        # Main tier platform with overhang
        structure.floors.append(Floor(
            x=x, y=tier_y + tier_height * 0.7, z=z,
            width=hw * 2 * scale * overhang, depth=hd * 2 * scale * overhang,
            thickness=0.3,
            color=color
        ))
        
        # Upturned corners (approximated with small ramps)
        if tier < tiers - 1:
            for dx, dz in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                cx = x + dx * hw * scale * overhang * 0.9
                cz = z + dz * hd * scale * overhang * 0.9
                structure.floors.append(Floor(
                    x=cx, y=tier_y + tier_height * 0.85, z=cz,
                    width=hw * 0.3, depth=hd * 0.3,
                    thickness=0.2,
                    color=tuple(c * 1.1 for c in color)
                ))


def _add_mansard_roof(structure: Structure, x: float, y: float, z: float,
                      hw: float, hd: float,
                      color: Tuple[float, float, float], rng: np.random.Generator):
    """Add a mansard (French) roof with two slopes on each side."""
    lower_height = min(hw, hd) * 0.25
    upper_height = min(hw, hd) * 0.15
    lower_inset = 0.3  # How much the lower slope goes in
    
    # Lower steep section (approximated as walls)
    for dx, dz, rot, length in [(0, hd, 0, hw * 2), (0, -hd, 180, hw * 2), 
                                  (-hw, 0, 90, hd * 2), (hw, 0, -90, hd * 2)]:
        structure.walls.append(Wall(
            x=x + dx, y=y, z=z + dz,
            width=length, height=lower_height,
            thickness=0.3, rotation=rot,
            color=color
        ))
    
    # Upper flat-ish section
    structure.floors.append(Floor(
        x=x, y=y + lower_height, z=z,
        width=hw * 2 * (1 - lower_inset), depth=hd * 2 * (1 - lower_inset),
        thickness=upper_height,
        color=tuple(c * 0.95 for c in color)
    ))


def _add_butterfly_roof(structure: Structure, x: float, y: float, z: float,
                        hw: float, hd: float,
                        color: Tuple[float, float, float], rng: np.random.Generator):
    """Add a butterfly (V-shaped) roof."""
    wing_height = min(hw, hd) * 0.2
    
    # Two sloped sections going DOWN to center
    structure.ramps.append(Ramp(
        x=x, y_bottom=y + wing_height, z=z - hd,
        length=hd, height=wing_height,
        width=hw * 2,
        rotation=180,  # Slope down toward center
        color=color
    ))
    structure.ramps.append(Ramp(
        x=x, y_bottom=y + wing_height, z=z + hd,
        length=hd, height=wing_height,
        width=hw * 2,
        rotation=0,  # Slope down toward center
        color=color
    ))


def _add_geodesic_roof(structure: Structure, x: float, y: float, z: float,
                       radius: float, height: float,
                       color: Tuple[float, float, float], rng: np.random.Generator):
    """Add a geodesic dome roof (triangulated appearance)."""
    # Approximate with stacked hexagonal-ish rings
    rings = 4
    for i in range(rings):
        t = i / rings
        level_y = y + height * (1 - math.cos(t * math.pi / 2))
        scale = math.cos(t * math.pi / 2.2)
        
        if scale > 0.15:
            # Slightly irregular sizing for triangulated look
            w_scale = scale * (0.95 + rng.random() * 0.1)
            d_scale = scale * (0.95 + rng.random() * 0.1)
            structure.floors.append(Floor(
                x=x, y=level_y, z=z,
                width=radius * 2 * w_scale, depth=radius * 2 * d_scale,
                thickness=height / rings * 1.3,
                color=tuple(min(1.0, c * (0.85 + 0.15 * t)) for c in color)
            ))


def _add_sawtooth_roof(structure: Structure, x: float, y: float, z: float,
                       hw: float, hd: float,
                       color: Tuple[float, float, float], rng: np.random.Generator):
    """Add an industrial sawtooth roof (zigzag profile)."""
    teeth = 2 + rng.integers(0, 3)
    tooth_width = hw * 2 / teeth
    tooth_height = min(hw, hd) * 0.25
    
    for i in range(teeth):
        tooth_x = x - hw + tooth_width * (i + 0.5)
        
        # Vertical face (window side)
        structure.walls.append(Wall(
            x=tooth_x - tooth_width * 0.4, y=y, z=z,
            width=hd * 2, height=tooth_height,
            thickness=0.2, rotation=90,
            color=tuple(c * 0.9 for c in color)
        ))
        
        # Sloped face
        structure.ramps.append(Ramp(
            x=tooth_x + tooth_width * 0.1, y_bottom=y, z=z,
            length=tooth_width * 0.8, height=tooth_height,
            width=hd * 2,
            rotation=-90,
            color=color
        ))


def _add_barrel_roof(structure: Structure, x: float, y: float, z: float,
                     hw: float, hd: float, height: float,
                     color: Tuple[float, float, float], rng: np.random.Generator):
    """Add a barrel (curved/cylindrical) roof."""
    segments = 5
    for i in range(segments):
        t = (i + 0.5) / segments
        # Arc across width (X axis)
        arc_x = x - hw + hw * 2 * t
        arc_y = y + height * math.sin(t * math.pi)
        seg_width = hw * 2 / segments * 1.1
        
        structure.floors.append(Floor(
            x=arc_x, y=arc_y, z=z,
            width=seg_width, depth=hd * 2,
            thickness=height / 3,
            color=tuple(c * (0.9 + 0.1 * math.sin(t * math.pi)) for c in color)
        ))


def _add_awning(structure: Structure, x: float, base_y: float, z: float,
                hw: float, hd: float, dna: BuildingDNA, rng: np.random.Generator):
    """Add a shop awning over the entrance."""
    entrance_wall = dna.entrance_wall
    awning_depth = dna.awning_depth
    awning_height = dna.floor_height * 0.6
    awning_width = dna.width * 0.7
    
    # Position based on entrance wall
    if entrance_wall == 0:  # +Z
        ax, az = x, z + hd + awning_depth/2
    elif entrance_wall == 1:  # -Z
        ax, az = x, z - hd - awning_depth/2
    elif entrance_wall == 2:  # -X
        ax, az = x - hw - awning_depth/2, z
        awning_width = dna.depth * 0.7
    else:  # +X
        ax, az = x + hw + awning_depth/2, z
        awning_width = dna.depth * 0.7
    
    # Awning as a sloped floor
    awning_color = dna.colors.secondary
    
    if entrance_wall in [0, 1]:
        aw, ad = awning_width, awning_depth
    else:
        aw, ad = awning_depth, awning_width
    
    structure.floors.append(Floor(
        x=ax, y=base_y + awning_height, z=az,
        width=aw, depth=ad,
        thickness=0.2,
        color=awning_color
    ))


def _add_entrance_stairs(structure: Structure, x: float, base_y: float, z: float,
                         hw: float, hd: float, dna: BuildingDNA, ground_y: float):
    """Add external stairs to reach an elevated entrance."""
    entrance_wall = dna.entrance_wall
    entrance_height = base_y - ground_y
    stair_width = 4.0  # Wide entrance stairs
    stair_length = entrance_height * 1.5  # Horizontal run (comfortable slope)
    
    # Staircase class draws from origin (step 0, bottom) toward direction (step N, top)
    # Origin must be at ground level, away from building
    # Top step must end up at the door (at wall)
    # Direction: 0=+X, 90=+Z, 180=-X, 270=-Z
    
    # entrance_wall: 0=+Z front, 1=-Z back, 2=-X left, 3=+X right
    # Origin AT door, stairs extend OUTWARD (away from building)
    # Swapping y_bottom/y_top to flip the stair geometry
    if entrance_wall == 0:  # Door at +Z front wall
        stair_x = x
        stair_z = z + hd
        stair_dir = 90  # Outward (+Z)
    elif entrance_wall == 1:  # Door at -Z back wall  
        stair_x = x
        stair_z = z - hd
        stair_dir = 270  # Outward (-Z)
    elif entrance_wall == 2:  # Door at -X left wall
        stair_x = x - hw
        stair_z = z
        stair_dir = 180  # Outward (-X)
    else:  # Door at +X right wall
        stair_x = x + hw
        stair_z = z
        stair_dir = 0  # Outward (+X)
    
    # Swap y_bottom/y_top: origin is at door (high), stairs descend outward
    structure.staircases.append(Staircase(
        x=stair_x,
        y_bottom=base_y,  # Swapped: door level at origin
        y_top=ground_y,   # Swapped: ground level at far end
        z=stair_z,
        width=stair_width,
        direction=stair_dir,
        color=tuple(c * 0.8 for c in dna.colors.primary)
    ))


def _add_decorative_arches(structure: Structure, x: float, base_y: float, z: float,
                           hw: float, hd: float, dna: BuildingDNA, 
                           rng: np.random.Generator):
    """Add decorative arches at entrance or around the building."""
    entrance_wall = dna.entrance_wall
    arch_height = dna.floor_height * 1.2
    arch_width = dna.width * 0.4
    
    # Main entrance arch
    if entrance_wall == 0:  # +Z
        ax, az = x, z + hd + 1.5
        rot = 0
    elif entrance_wall == 1:  # -Z
        ax, az = x, z - hd - 1.5
        rot = 180
    elif entrance_wall == 2:  # -X
        ax, az = x - hw - 1.5, z
        arch_width = dna.depth * 0.4
        rot = 90
    else:  # +X
        ax, az = x + hw + 1.5, z
        arch_width = dna.depth * 0.4
        rot = -90
    
    # Add entrance arch
    structure.arches.append(Arch(
        x=ax, y=base_y, z=az,
        width=arch_width, height=arch_height,
        thickness=0.6, rotation=rot,
        color=tuple(c * 0.85 for c in dna.colors.primary)
    ))
    
    # For temples, add side arches
    if dna.building_type == BuildingType.TEMPLE:
        # Colonnade of arches along the sides
        num_arches = max(2, int(max(hw, hd) / 4))
        spacing = (hw * 2 - arch_width) / max(1, num_arches)
        
        for i in range(num_arches):
            offset = -hw + arch_width/2 + i * spacing + spacing/2
            
            # Side arches (perpendicular to entrance)
            if entrance_wall in [0, 1]:  # Entrance on Z, arches on X
                structure.arches.append(Arch(
                    x=x + offset, y=base_y, z=z - hd - 1.0,
                    width=arch_width * 0.8, height=arch_height * 0.9,
                    thickness=0.4, rotation=180,
                    color=tuple(c * 0.8 for c in dna.colors.primary)
                ))


def _add_rooftop_features(structure: Structure, x: float, roof_y: float, z: float,
                          hw: float, hd: float, dna: BuildingDNA, 
                          rng: np.random.Generator):
    """Add AC units, antennas, or helipad to roof."""
    feature_type = rng.choice(["ac_units", "antenna", "helipad"])
    
    if feature_type == "ac_units":
        # Small boxes on roof
        num_units = 2 + rng.integers(0, 4)
        for _ in range(num_units):
            ux = x + rng.uniform(-hw * 0.7, hw * 0.7)
            uz = z + rng.uniform(-hd * 0.7, hd * 0.7)
            
            structure.floors.append(Floor(
                x=ux, y=roof_y + 0.5, z=uz,
                width=1.5 + rng.random(), depth=1.0 + rng.random() * 0.5,
                thickness=1.2 + rng.random() * 0.5,
                color=(0.6, 0.62, 0.65)
            ))
    
    elif feature_type == "antenna":
        # Tall thin pillar
        structure.pillars.append(Pillar(
            x=x + rng.uniform(-hw * 0.3, hw * 0.3),
            y_bottom=roof_y,
            y_top=roof_y + 8 + rng.random() * 6,
            z=z + rng.uniform(-hd * 0.3, hd * 0.3),
            width=0.3,
            color=(0.5, 0.5, 0.55)
        ))
    
    elif feature_type == "helipad":
        # Circular platform (approximated as square)
        pad_size = min(hw, hd) * 1.2
        structure.floors.append(Floor(
            x=x, y=roof_y + 0.6, z=z,
            width=pad_size * 2, depth=pad_size * 2,
            thickness=0.3,
            color=(0.35, 0.38, 0.4)
        ))


def _add_chimney(structure: Structure, x: float, roof_y: float, z: float,
                 hw: float, hd: float, dna: BuildingDNA, rng: np.random.Generator):
    """Add a chimney to the roof."""
    # Position near edge of roof
    cx = x + rng.choice([-1, 1]) * hw * 0.6
    cz = z + rng.choice([-1, 1]) * hd * 0.6
    
    chimney_height = 3 + rng.random() * 2
    chimney_width = 1.0 + rng.random() * 0.5
    
    structure.pillars.append(Pillar(
        x=cx, y_bottom=roof_y - 1, y_top=roof_y + chimney_height,
        z=cz, width=chimney_width,
        color=(0.45, 0.4, 0.38)
    ))


def _generate_parkour(x: float, y: float, z: float, dna: BuildingDNA,
                      terrain_heights: Optional[List[float]] = None) -> Structure:
    """
    Generate a parkour playground - a sequence of platforms, ramps, and poles
    designed for jumping and traversing.
    """
    rng = np.random.default_rng(dna.seed)
    structure = Structure(x=x, y=y, z=z)
    
    hw = dna.width / 2
    hd = dna.depth / 2
    
    # Base height
    if terrain_heights and len(terrain_heights) >= 4:
        base_y = max(terrain_heights)
    else:
        base_y = y
    
    # Color palette for parkour - bright, varied
    colors = [
        dna.colors.primary,
        dna.colors.secondary,
        tuple(c * 1.2 for c in dna.colors.primary),  # Brighter version
        (0.8, 0.3, 0.2),  # Red accent
        (0.2, 0.6, 0.8),  # Blue accent
        (0.3, 0.7, 0.3),  # Green accent
        (0.9, 0.7, 0.2),  # Yellow/gold
    ]
    
    # Generate a grid of potential platform positions - larger grid for bigger area
    grid_size = 8  # 8x8 grid of potential positions
    cell_w = dna.width / grid_size
    cell_d = dna.depth / grid_size
    
    # Track placed platforms for connectivity
    platforms = []
    current_height = base_y + 1.5  # Start slightly above ground
    
    # =========================================================================
    # MULTIPLE ENTRY POINTS - from ground level (corners)
    # =========================================================================
    entry_positions = [
        (x - hw + cell_w * 1.5, z - hd + cell_d * 1.5, 0),    # Near corner 1
        (x + hw - cell_w * 1.5, z - hd + cell_d * 1.5, 90),   # Near corner 2
        (x - hw + cell_w * 1.5, z + hd - cell_d * 1.5, 270),  # Near corner 3
        (x + hw - cell_w * 1.5, z + hd - cell_d * 1.5, 180),  # Near corner 4
    ]
    
    for entry_x, entry_z, ramp_dir in entry_positions:
        # Entry platform - large starting area
        structure.floors.append(Floor(
            x=entry_x, y=base_y + 0.3, z=entry_z,
            width=cell_w * 1.8, depth=cell_d * 1.8, thickness=0.6,
            color=colors[0]
        ))
        platforms.append((entry_x, base_y + 0.3, entry_z, cell_w * 1.8))
        
        # Entry ramp going up - wide and gentle
        structure.ramps.append(Ramp(
            x=entry_x, y_bottom=base_y + 0.3, z=entry_z,
            width=cell_w * 1.2, length=cell_w * 2.0, height=4.0,
            rotation=ramp_dir, color=colors[1]
        ))
    
    # =========================================================================
    # MAIN PLATFORMS - scattered at various heights (MANY more for big area)
    # =========================================================================
    n_platforms = 25 + rng.integers(0, 15)  # 25-40 platforms
    
    for i in range(n_platforms):
        # Pick a random grid position
        gx = rng.integers(0, grid_size)
        gz = rng.integers(0, grid_size)
        
        px = x - hw + (gx + 0.5) * cell_w + rng.uniform(-cell_w * 0.3, cell_w * 0.3)
        pz = z - hd + (gz + 0.5) * cell_d + rng.uniform(-cell_d * 0.3, cell_d * 0.3)
        
        # Height follows a general upward trend with variation
        base_progression = (i / n_platforms) * dna.floors * dna.floor_height
        height_var = rng.uniform(-dna.floor_height * 1.5, dna.floor_height * 1.5)
        py = base_y + 2.0 + base_progression + height_var
        py = max(base_y + 0.5, py)  # Don't go below ground
        
        # Platform type selection - more variety
        platform_type = rng.choice([
            "large", "large", "medium", "medium", "rectangle", "long", 
            "pole", "pole", "ramp_up", "ramp_down", "bridge"
        ])
        
        color = colors[rng.integers(0, len(colors))]
        
        if platform_type == "large":
            # LARGE square platforms - very easy to land on (3x bigger)
            size = 12.0 + rng.random() * 8.0  # 12-20 units
            structure.floors.append(Floor(
                x=px, y=py, z=pz,
                width=size, depth=size, thickness=0.6,
                color=color
            ))
            platforms.append((px, py, pz, size))
            
        elif platform_type == "medium":
            # Medium square platforms - still generous
            size = 8.0 + rng.random() * 4.0  # 8-12 units
            structure.floors.append(Floor(
                x=px, y=py, z=pz,
                width=size, depth=size, thickness=0.5,
                color=color
            ))
            platforms.append((px, py, pz, size))
            
        elif platform_type == "rectangle":
            # Long rectangular platforms - good for running
            w = 14.0 + rng.random() * 8.0  # 14-22 units long
            d = 6.0 + rng.random() * 4.0   # 6-10 units wide
            if rng.random() < 0.5:
                w, d = d, w  # Swap for variety
            structure.floors.append(Floor(
                x=px, y=py, z=pz,
                width=w, depth=d, thickness=0.5,
                color=color
            ))
            platforms.append((px, py, pz, max(w, d)))
            
        elif platform_type == "long":
            # Super long runways
            w = 20.0 + rng.random() * 15.0  # 20-35 units!
            d = 5.0 + rng.random() * 3.0
            if rng.random() < 0.5:
                w, d = d, w
            structure.floors.append(Floor(
                x=px, y=py, z=pz,
                width=w, depth=d, thickness=0.4,
                color=color
            ))
            platforms.append((px, py, pz, max(w, d)))
            
        elif platform_type == "pole":
            # Platform on a tall pole - generous landing area
            pole_height = 5 + rng.random() * 10
            pillar_y = base_y - 1
            structure.pillars.append(Pillar(
                x=px, y_bottom=pillar_y, y_top=py,
                z=pz, width=1.2 + rng.random() * 0.6,
                color=tuple(c * 0.6 for c in color)
            ))
            size = 8.0 + rng.random() * 5.0  # 8-13 units
            structure.floors.append(Floor(
                x=px, y=py, z=pz,
                width=size, depth=size, thickness=0.5,
                color=color
            ))
            platforms.append((px, py, pz, size))
            
        elif platform_type == "ramp_up":
            # Ramp going up - wide and long for easy running
            direction = rng.choice([0, 90, 180, 270])
            ramp_len = 12 + rng.random() * 8  # 12-20 units
            ramp_height = 3 + rng.random() * 4
            structure.ramps.append(Ramp(
                x=px, y_bottom=py, z=pz,
                width=6.0 + rng.random() * 3.0, length=ramp_len, height=ramp_height,
                rotation=direction, color=color
            ))
            platforms.append((px, py + ramp_height, pz, ramp_len))
            
        elif platform_type == "bridge":
            # Long bridge connecting areas
            direction = rng.choice([0, 90])
            bridge_len = 15 + rng.random() * 10  # 15-25 units
            structure.floors.append(Floor(
                x=px, y=py, z=pz,
                width=bridge_len if direction == 0 else 4.0,
                depth=4.0 if direction == 0 else bridge_len,
                thickness=0.3,
                color=color
            ))
            platforms.append((px, py, pz, bridge_len))
            
        elif platform_type == "ramp_down":
            # Ramp going down - with larger start platform
            direction = rng.choice([0, 90, 180, 270])
            ramp_len = 10 + rng.random() * 6  # Longer ramps
            ramp_height = 3 + rng.random() * 4
            # Start platform - generous landing area
            structure.floors.append(Floor(
                x=px, y=py, z=pz,
                width=10.0, depth=10.0, thickness=0.5,
                color=color
            ))
            # Ramp going down
            structure.ramps.append(Ramp(
                x=px, y_bottom=py - ramp_height, z=pz,
                width=6.0 + rng.random() * 2.0, length=ramp_len, height=ramp_height,
                rotation=direction, color=tuple(c * 0.9 for c in color)
            ))
            platforms.append((px, py, pz, 10.0))
    
    # =========================================================================
    # CONNECTING BRIDGES (walkways between nearby platforms)
    # =========================================================================
    for i, (px1, py1, pz1, size1) in enumerate(platforms):
        for px2, py2, pz2, size2 in platforms[i+1:]:
            dist = math.sqrt((px2 - px1)**2 + (pz2 - pz1)**2)
            height_diff = abs(py2 - py1)
            
            # Connect platforms that are reasonably close (wider range for bigger area)
            if 10 < dist < 35 and height_diff < 6 and rng.random() < 0.15:
                # Walkable bridge - wider and easier to cross
                mid_x = (px1 + px2) / 2
                mid_z = (pz1 + pz2) / 2
                mid_y = max(py1, py2) + 0.1
                
                angle = math.atan2(pz2 - pz1, px2 - px1)
                
                # Bridge as walkable floor - much wider for comfortable crossing
                structure.floors.append(Floor(
                    x=mid_x, y=mid_y, z=mid_z,
                    width=dist * 0.85, depth=5.0 + rng.random() * 2.0,
                    thickness=0.4,
                    color=colors[rng.integers(0, len(colors))]
                ))
    
    # =========================================================================
    # CLIMBING TOWERS (vertical pillars with platforms - like cat towers!)
    # =========================================================================
    n_climb_poles = 5 + rng.integers(0, 5)  # More towers
    for _ in range(n_climb_poles):
        pole_x = x + rng.uniform(-hw * 0.8, hw * 0.8)
        pole_z = z + rng.uniform(-hd * 0.8, hd * 0.8)
        pole_height = 15 + rng.random() * 20  # Much taller towers
        
        # Multiple platforms along the pole - generous landing areas
        n_pole_platforms = 3 + rng.integers(0, 4)
        for p in range(n_pole_platforms):
            plat_y = base_y + (p + 1) * pole_height / (n_pole_platforms + 1)
            size = 8.0 + rng.random() * 4.0  # Much larger platforms (8-12 units)
            structure.floors.append(Floor(
                x=pole_x, y=plat_y, z=pole_z,
                width=size, depth=size, thickness=0.5,
                color=colors[rng.integers(0, len(colors))]
            ))
        
        # The pole itself - thick and sturdy
        structure.pillars.append(Pillar(
            x=pole_x, y_bottom=base_y - 1, y_top=base_y + pole_height,
            z=pole_z, width=1.5,
            color=(0.4, 0.4, 0.45)
        ))
    
    # =========================================================================
    # CENTRAL HUB PLATFORM (massive gathering area in the middle)
    # =========================================================================
    hub_y = base_y + dna.floors * dna.floor_height * 0.4
    
    # Central hub - huge platform to rest and plan
    structure.floors.append(Floor(
        x=x, y=hub_y, z=z,
        width=25.0, depth=25.0, thickness=0.8,
        color=colors[0]
    ))
    
    # Hub support pillars (4 corners)
    for dx, dz in [(-10, -10), (10, -10), (-10, 10), (10, 10)]:
        structure.pillars.append(Pillar(
            x=x + dx, y_bottom=base_y - 1, y_top=hub_y,
            z=z + dz, width=2.0,
            color=(0.35, 0.35, 0.4)
        ))
    
    # =========================================================================
    # FINAL GOAL PLATFORM (high and flashy - massive victory area)
    # =========================================================================
    goal_x = x + hw * 0.4
    goal_z = z + hd * 0.4
    goal_y = base_y + dna.floors * dna.floor_height + 5
    
    # Tall support tower
    structure.pillars.append(Pillar(
        x=goal_x, y_bottom=base_y - 1, y_top=goal_y,
        z=goal_z, width=2.5,
        color=(0.3, 0.3, 0.35)
    ))
    
    # Goal platform (gold/yellow) - massive victory area!
    structure.floors.append(Floor(
        x=goal_x, y=goal_y, z=goal_z,
        width=18.0, depth=18.0, thickness=0.8,
        color=(0.9, 0.75, 0.2)  # Gold
    ))
    
    # Victory pillar/trophy - tall beacon
    structure.pillars.append(Pillar(
        x=goal_x, y_bottom=goal_y, y_top=goal_y + 8,
        z=goal_z, width=1.0,
        color=(0.95, 0.85, 0.3)
    ))
    
    # Ramps leading to goal from hub
    structure.ramps.append(Ramp(
        x=goal_x - 12, y_bottom=hub_y, z=goal_z,
        width=6.0, length=15.0, height=goal_y - hub_y,
        rotation=0, color=(0.85, 0.7, 0.2)
    ))
    
    structure.compute_bounds()
    return structure

