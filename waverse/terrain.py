"""
Terrain Data Structures - Columns, Segments, Tiles

This module defines the core data structures for terrain:
- Segment: A vertical slice of solid material
- TerrainColumn: A stack of segments at one (x, z) position  
- TerrainTile: A quad with 4 corner columns
- Segment merging for efficient rendering
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set, Any
from enum import Enum, auto
import numpy as np


class Material(Enum):
    """Material types for terrain segments."""
    AIR = auto()
    STONE = auto()
    DIRT = auto()
    GRASS = auto()
    SAND = auto()
    WATER = auto()
    BEDROCK = auto()
    
    def is_solid(self) -> bool:
        return self not in (Material.AIR, Material.WATER)


@dataclass
class Segment:
    """
    A vertical segment of solid material in a terrain column.
    
    Represents a contiguous vertical slice of terrain from y_bottom to y_top.
    Multiple segments in a column create caves/overhangs.
    """
    y_bottom: float
    y_top: float
    material: Material = Material.STONE
    
    # Optional: track modifications (for saving player changes)
    modified: bool = False
    
    @property
    def height(self) -> float:
        """Height of this segment."""
        return self.y_top - self.y_bottom
    
    @property
    def center_y(self) -> float:
        """Vertical center of segment."""
        return (self.y_top + self.y_bottom) / 2
    
    def contains_y(self, y: float) -> bool:
        """Check if this segment contains the given y coordinate."""
        return self.y_bottom <= y <= self.y_top
    
    def overlaps(self, other: "Segment") -> bool:
        """Check if this segment overlaps with another."""
        return not (self.y_top < other.y_bottom or self.y_bottom > other.y_top)
    
    def can_merge_with(self, other: "Segment", tolerance: float = 0.01) -> bool:
        """
        Check if this segment can be merged with another.
        Segments can merge if they:
        - Have the same material
        - Are adjacent (touching or overlapping)
        """
        if self.material != other.material:
            return False
        # Check if adjacent (with tolerance for floating point)
        return (abs(self.y_top - other.y_bottom) < tolerance or
                abs(self.y_bottom - other.y_top) < tolerance or
                self.overlaps(other))
    
    def merge_with(self, other: "Segment") -> "Segment":
        """Merge this segment with another, returning the combined segment."""
        return Segment(
            y_bottom=min(self.y_bottom, other.y_bottom),
            y_top=max(self.y_top, other.y_top),
            material=self.material,
            modified=self.modified or other.modified,
        )
    
    def split_at(self, y: float, gap_height: float) -> Tuple[Optional["Segment"], Optional["Segment"]]:
        """
        Split this segment at y coordinate, creating a gap.
        
        Returns (lower_segment, upper_segment), either can be None if gap consumes it.
        """
        gap_bottom = y - gap_height / 2
        gap_top = y + gap_height / 2
        
        lower = None
        upper = None
        
        if self.y_bottom < gap_bottom:
            lower = Segment(self.y_bottom, gap_bottom, self.material, modified=True)
        
        if self.y_top > gap_top:
            upper = Segment(gap_top, self.y_top, self.material, modified=True)
        
        return (lower, upper)


@dataclass
class TerrainColumn:
    """
    A vertical column of terrain at one (x, z) grid position.
    
    Contains a list of segments representing solid material.
    Gaps between segments are air (caves, sky, etc).
    """
    x: int
    z: int
    segments: List[Segment] = field(default_factory=list)
    
    # Base height from wave generation (before any modifications)
    base_surface_height: float = 0.0
    
    # Track if this column has been modified by player
    dirty: bool = False
    
    def __post_init__(self):
        """Ensure segments are sorted by y_bottom."""
        self.segments.sort(key=lambda s: s.y_bottom)
    
    @property
    def surface_height(self) -> float:
        """Height of the topmost surface."""
        if not self.segments:
            return self.base_surface_height
        return self.segments[-1].y_top
    
    @property
    def floor_height(self) -> float:
        """Height of the lowest point."""
        if not self.segments:
            return self.base_surface_height
        return self.segments[0].y_bottom
    
    def is_solid_at(self, y: float) -> bool:
        """Check if there's solid material at height y."""
        for seg in self.segments:
            if seg.contains_y(y):
                return seg.material.is_solid()
        return False
    
    def get_segment_at(self, y: float) -> Optional[Segment]:
        """Get the segment containing height y, if any."""
        for seg in self.segments:
            if seg.contains_y(y):
                return seg
        return None
    
    def dig_top(self, amount: float) -> float:
        """
        Dig down from the surface by amount.
        
        Returns the actual amount dug (may be less if hit a cave).
        """
        if not self.segments:
            return 0.0
        
        top_seg = self.segments[-1]
        actual_dig = min(amount, top_seg.height)
        
        top_seg.y_top -= actual_dig
        top_seg.modified = True
        self.dirty = True
        
        # Remove segment if it's been completely dug away
        if top_seg.height <= 0:
            self.segments.pop()
        
        return actual_dig
    
    def dig_at(self, y: float, height: float) -> bool:
        """
        Dig a horizontal tunnel at height y with given height.
        Creates a gap in any segment containing y.
        
        Returns True if any digging occurred.
        """
        affected_segments = []
        new_segments = []
        
        for i, seg in enumerate(self.segments):
            if seg.contains_y(y):
                lower, upper = seg.split_at(y, height)
                if lower:
                    new_segments.append(lower)
                if upper:
                    new_segments.append(upper)
                affected_segments.append(i)
            else:
                new_segments.append(seg)
        
        if affected_segments:
            self.segments = sorted(new_segments, key=lambda s: s.y_bottom)
            self.dirty = True
            return True
        return False
    
    def add_segment(self, segment: Segment):
        """Add a segment and merge with any overlapping segments."""
        # Find segments to merge with
        to_merge = [segment]
        remaining = []
        
        for existing in self.segments:
            if segment.can_merge_with(existing):
                to_merge.append(existing)
            else:
                remaining.append(existing)
        
        # Merge all overlapping segments
        merged = to_merge[0]
        for seg in to_merge[1:]:
            merged = merged.merge_with(seg)
        
        remaining.append(merged)
        self.segments = sorted(remaining, key=lambda s: s.y_bottom)
        self.dirty = True
    
    def optimize_segments(self):
        """Merge any adjacent segments with same material."""
        if len(self.segments) < 2:
            return
        
        optimized = [self.segments[0]]
        
        for seg in self.segments[1:]:
            if optimized[-1].can_merge_with(seg):
                optimized[-1] = optimized[-1].merge_with(seg)
            else:
                optimized.append(seg)
        
        self.segments = optimized
    
    @classmethod
    def from_surface_height(cls, x: int, z: int, surface_height: float,
                           bedrock_level: float = -100.0) -> "TerrainColumn":
        """Create a simple column from surface height (no caves)."""
        return cls(
            x=x,
            z=z,
            segments=[Segment(bedrock_level, surface_height, Material.STONE)],
            base_surface_height=surface_height,
        )
    
    @classmethod
    def from_density_column(cls, x: int, z: int, densities: np.ndarray,
                           y_start: float, y_step: float,
                           threshold: float = 0.0) -> "TerrainColumn":
        """
        Create a column from a 1D density array.
        Solid where density > threshold.
        """
        segments = []
        in_solid = False
        seg_start = 0.0
        
        for i, density in enumerate(densities):
            y = y_start + i * y_step
            is_solid = density > threshold
            
            if is_solid and not in_solid:
                # Starting a new solid segment
                seg_start = y
                in_solid = True
            elif not is_solid and in_solid:
                # Ending a solid segment
                segments.append(Segment(seg_start, y, Material.STONE))
                in_solid = False
        
        # Close final segment if still in solid
        if in_solid:
            final_y = y_start + len(densities) * y_step
            segments.append(Segment(seg_start, final_y, Material.STONE))
        
        surface_height = segments[-1].y_top if segments else y_start
        
        return cls(
            x=x,
            z=z,
            segments=segments,
            base_surface_height=surface_height,
        )


@dataclass
class TunnelFace:
    """
    An exposed face on a tile edge (entrance to horizontal tunnel).
    """
    direction: str  # 'north', 'south', 'east', 'west'
    y_bottom: float
    y_top: float
    width: float = 1.0  # 0.0 to 1.0, how much of edge is open
    
    @property
    def height(self) -> float:
        return self.y_top - self.y_bottom


@dataclass
class TerrainTile:
    """
    A terrain tile (quad) defined by 4 corner columns.
    
    Corner layout:
        NW (0) ─── NE (1)
          │         │
          │  tile   │
          │         │
        SW (2) ─── SE (3)
    
    The tile's surface is a quad connecting the 4 corner surface heights.
    For caves/tunnels, each corner has its own column with segments.
    """
    # Grid position
    x: int
    z: int
    
    # The 4 corner columns (NW, NE, SW, SE)
    corners: Tuple[TerrainColumn, TerrainColumn, TerrainColumn, TerrainColumn] = None
    
    # Quick access to surface heights (computed from corners)
    _surface_heights: Optional[Tuple[float, float, float, float]] = None
    
    # Dig depth modification (for simple heightmap-style digging)
    dig_depth: float = 0.0
    
    # Horizontal tunnel openings
    tunnel_faces: Dict[str, List[TunnelFace]] = field(default_factory=dict)
    
    # Track if mesh needs regeneration
    mesh_dirty: bool = True
    
    @property
    def surface_heights(self) -> Tuple[float, float, float, float]:
        """Get the 4 corner surface heights."""
        if self._surface_heights is None and self.corners:
            self._surface_heights = tuple(
                c.surface_height - self.dig_depth for c in self.corners
            )
        return self._surface_heights or (0.0, 0.0, 0.0, 0.0)
    
    @property
    def center_height(self) -> float:
        """Average height of the tile."""
        return sum(self.surface_heights) / 4
    
    @property
    def nw_height(self) -> float:
        return self.surface_heights[0]
    
    @property
    def ne_height(self) -> float:
        return self.surface_heights[1]
    
    @property
    def sw_height(self) -> float:
        return self.surface_heights[2]
    
    @property
    def se_height(self) -> float:
        return self.surface_heights[3]
    
    def invalidate_cache(self):
        """Mark cached values as needing recalculation."""
        self._surface_heights = None
        self.mesh_dirty = True
    
    def dig(self, amount: float):
        """
        Dig this tile down (lowers all 4 corners equally).
        """
        self.dig_depth += amount
        self.invalidate_cache()
        
        # Also dig each corner column
        if self.corners:
            for corner in self.corners:
                corner.dig_top(amount)
    
    def dig_horizontal(self, direction: str, y: float, height: float):
        """
        Dig horizontally into neighboring tile at height y.
        """
        # Record the tunnel face
        if direction not in self.tunnel_faces:
            self.tunnel_faces[direction] = []
        
        face = TunnelFace(direction, y - height/2, y + height/2)
        self.tunnel_faces[direction].append(face)
        
        # Carve the appropriate corner columns
        if self.corners:
            # Map direction to affected corners
            dir_corners = {
                'north': [0, 1],  # NW, NE
                'south': [2, 3],  # SW, SE
                'west': [0, 2],   # NW, SW
                'east': [1, 3],   # NE, SE
            }
            for idx in dir_corners.get(direction, []):
                self.corners[idx].dig_at(y, height)
        
        self.invalidate_cache()


# =============================================================================
# Segment Merging for Efficient Rendering
# =============================================================================

@dataclass
class MergedSegmentGroup:
    """
    A group of horizontally adjacent segments that can be rendered as one shape.
    
    When multiple adjacent tiles have segments at the same y-range with the
    same material, we can merge them into a single larger quad for rendering.
    """
    # World coordinates of the merged region
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    
    # Vertical extent (averaged/interpolated from corners)
    y_bottom: float
    y_top: float
    
    # Material
    material: Material = Material.STONE
    
    # Contributing tiles (for tracking updates)
    tile_coords: Set[Tuple[int, int]] = field(default_factory=set)
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def depth(self) -> float:
        return self.z_max - self.z_min
    
    @property 
    def height(self) -> float:
        return self.y_top - self.y_bottom


def find_mergeable_segments(tiles: Dict[Tuple[int, int], TerrainTile],
                            y_tolerance: float = 0.5,
                            tile_size: float = 1.0) -> List[MergedSegmentGroup]:
    """
    Find groups of tiles whose segments can be merged for efficient rendering.
    
    This is the "mega bonus" optimization - merging horizontal neighbors
    to reduce draw calls.
    
    Algorithm:
    1. For each tile, get its segment ranges
    2. Find adjacent tiles with overlapping segment ranges
    3. Merge into larger rectangular groups where possible
    
    Args:
        tiles: Dictionary of tiles by (x, z) coordinates
        y_tolerance: How close y values need to be to merge
        tile_size: Size of each tile in world units
        
    Returns:
        List of merged segment groups ready for rendering
    """
    if not tiles:
        return []
    
    merged_groups = []
    processed = set()
    
    # Sort tiles by position for consistent processing
    tile_list = sorted(tiles.items(), key=lambda t: (t[0][0], t[0][1]))
    
    for (tx, tz), tile in tile_list:
        if (tx, tz) in processed:
            continue
        
        if not tile.corners or not tile.corners[0].segments:
            continue
        
        # Try to start a merge group from this tile's top segment
        top_seg = tile.corners[0].segments[-1] if tile.corners[0].segments else None
        if not top_seg:
            continue
        
        # Find all adjacent tiles that can merge with this one
        group_tiles = {(tx, tz)}
        y_bottom = min(c.segments[-1].y_bottom for c in tile.corners if c.segments)
        y_top = max(c.segments[-1].y_top for c in tile.corners if c.segments)
        material = top_seg.material
        
        # Expand in +x direction
        check_x = tx + 1
        while (check_x, tz) in tiles and (check_x, tz) not in processed:
            neighbor = tiles[(check_x, tz)]
            if _can_merge_tiles_horizontally(tile, neighbor, y_tolerance, material):
                group_tiles.add((check_x, tz))
                check_x += 1
            else:
                break
        
        # Expand in +z direction (for all x in current strip)
        x_coords = sorted(set(t[0] for t in group_tiles))
        check_z = tz + 1
        while True:
            # Check if entire row can merge
            row_can_merge = True
            for x in x_coords:
                if (x, check_z) not in tiles or (x, check_z) in processed:
                    row_can_merge = False
                    break
                neighbor = tiles[(x, check_z)]
                ref_tile = tiles[(x, tz)]
                if not _can_merge_tiles_horizontally(ref_tile, neighbor, y_tolerance, material):
                    row_can_merge = False
                    break
            
            if row_can_merge:
                for x in x_coords:
                    group_tiles.add((x, check_z))
                check_z += 1
            else:
                break
        
        # Create merged group
        if len(group_tiles) > 0:
            x_coords = [t[0] for t in group_tiles]
            z_coords = [t[1] for t in group_tiles]
            
            merged = MergedSegmentGroup(
                x_min=min(x_coords) * tile_size,
                x_max=(max(x_coords) + 1) * tile_size,
                z_min=min(z_coords) * tile_size,
                z_max=(max(z_coords) + 1) * tile_size,
                y_bottom=y_bottom,
                y_top=y_top,
                material=material,
                tile_coords=group_tiles,
            )
            merged_groups.append(merged)
            processed.update(group_tiles)
    
    return merged_groups


def _can_merge_tiles_horizontally(tile1: TerrainTile, tile2: TerrainTile,
                                   y_tolerance: float,
                                   material: Material) -> bool:
    """Check if two adjacent tiles can be merged for rendering."""
    if not tile1.corners or not tile2.corners:
        return False
    
    for i in range(4):
        segs1 = tile1.corners[i].segments
        segs2 = tile2.corners[i].segments
        
        if not segs1 or not segs2:
            return False
        
        top1 = segs1[-1]
        top2 = segs2[-1]
        
        # Check material matches
        if top1.material != material or top2.material != material:
            return False
        
        # Check y ranges are close enough
        if abs(top1.y_top - top2.y_top) > y_tolerance:
            return False
        if abs(top1.y_bottom - top2.y_bottom) > y_tolerance:
            return False
    
    return True


# =============================================================================
# Utility Functions
# =============================================================================

def interpolate_height(tile: TerrainTile, local_x: float, local_z: float) -> float:
    """
    Bilinear interpolation of height within a tile.
    
    local_x, local_z should be in [0, 1] range within the tile.
    """
    nw, ne, sw, se = tile.surface_heights
    
    # Interpolate along x at top and bottom
    top = nw * (1 - local_x) + ne * local_x
    bottom = sw * (1 - local_x) + se * local_x
    
    # Interpolate along z
    return top * (1 - local_z) + bottom * local_z


def get_tile_normal(tile: TerrainTile, tile_size: float = 1.0) -> Tuple[float, float, float]:
    """
    Calculate the surface normal for a tile.
    """
    nw, ne, sw, se = tile.surface_heights
    
    # Calculate gradients
    dx = ((ne - nw) + (se - sw)) / 2 / tile_size
    dz = ((sw - nw) + (se - ne)) / 2 / tile_size
    
    # Normal is perpendicular to surface
    nx, ny, nz = -dx, 1.0, -dz
    
    # Normalize
    length = np.sqrt(nx*nx + ny*ny + nz*nz)
    if length > 0:
        nx, ny, nz = nx/length, ny/length, nz/length
    
    return (nx, ny, nz)

