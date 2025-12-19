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
from .structures import Structure, Wall, Floor, Ramp, Pillar


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
        
        # Add floor slab (except ground floor unless on slope)
        if floor_idx > 0 or (terrain_heights and base_y > min_terrain + 1):
            structure.floors.append(Floor(
                x=x, y=floor_y, z=z,
                width=floor_hw * 2 + 0.5,
                depth=floor_hd * 2 + 0.5,
                thickness=0.4,
                color=tuple(c * 0.85 for c in dna.colors.primary)
            ))
        
        # Add walls
        _add_floor_walls(structure, x, floor_y, z, floor_hw, floor_hd, 
                        dna, floor_idx, rng, min_terrain if terrain_heights else base_y)
        
        # Add balconies (upper floors only)
        if dna.has_balconies and floor_idx > 0:
            _add_balconies(structure, x, floor_y, z, floor_hw, floor_hd, dna, rng)
    
    # =========================================================================
    # ROOF
    # =========================================================================
    roof_y = base_y + dna.floors * dna.floor_height
    _add_roof(structure, x, roof_y, z, hw, hd, dna, rng)
    
    # =========================================================================
    # ENTRANCE / AWNING
    # =========================================================================
    if dna.has_awning:
        _add_awning(structure, x, base_y, z, hw, hd, dna, rng)
    
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
            door_width = 3.0
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

