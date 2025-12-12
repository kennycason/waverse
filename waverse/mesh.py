"""
Mesh Generation - Convert terrain data to renderable geometry.

Features:
- Generates vertex/normal/color data for OpenGL
- Supports merged segments for efficient rendering
- Creates geometry for caves, tunnels, walls
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import math

from .terrain import (
    TerrainTile, TerrainColumn, Segment, Material,
    MergedSegmentGroup, find_mergeable_segments, get_tile_normal,
)
from .chunk import Chunk


@dataclass
class MeshData:
    """
    Renderable mesh data ready for OpenGL.
    """
    # Vertex data: [(x, y, z), ...]
    vertices: np.ndarray = None
    
    # Normal data: [(nx, ny, nz), ...]
    normals: np.ndarray = None
    
    # Color data: [(r, g, b), ...] or [(r, g, b, a), ...]
    colors: np.ndarray = None
    
    # Optional: indices for indexed rendering
    indices: Optional[np.ndarray] = None
    
    # Primitive type ('quads', 'triangles', 'lines')
    primitive: str = 'quads'
    
    # Count of primitives
    @property
    def vertex_count(self) -> int:
        return len(self.vertices) if self.vertices is not None else 0
    
    @property
    def primitive_count(self) -> int:
        if self.vertices is None:
            return 0
        if self.primitive == 'quads':
            return len(self.vertices) // 4
        elif self.primitive == 'triangles':
            return len(self.vertices) // 3
        return len(self.vertices) // 2


def height_to_color(height: float, water_level: float = 0.0) -> Tuple[float, float, float]:
    """
    Convert terrain height to a color.
    
    Returns RGB values in 0-1 range.
    """
    # Normalize height relative to water level
    h = height - water_level
    
    # Deep water (dark blue)
    if h < -10:
        return (0.05, 0.15, 0.4)
    # Water (blue)
    elif h < -2:
        t = (h + 10) / 8
        return (0.05 + t * 0.1, 0.15 + t * 0.2, 0.4 + t * 0.2)
    # Shallow water (light blue)
    elif h < 0:
        t = (h + 2) / 2
        return (0.15 + t * 0.1, 0.35 + t * 0.1, 0.6)
    # Beach (sand)
    elif h < 1:
        return (0.76, 0.7, 0.5)
    # Grass (green)
    elif h < 5:
        t = (h - 1) / 4
        return (0.25 - t * 0.05, 0.55 - t * 0.1, 0.2)
    # Forest (darker green)
    elif h < 15:
        t = (h - 5) / 10
        return (0.2 - t * 0.05, 0.45 - t * 0.1, 0.15)
    # Rock (brown/gray)
    elif h < 30:
        t = (h - 15) / 15
        return (0.4 + t * 0.15, 0.35 + t * 0.1, 0.25 + t * 0.1)
    # Mountain (gray)
    elif h < 50:
        t = (h - 30) / 20
        return (0.55 + t * 0.15, 0.45 + t * 0.2, 0.35 + t * 0.25)
    # Snow (white)
    else:
        t = min(1, (h - 50) / 20)
        return (0.7 + t * 0.25, 0.7 + t * 0.25, 0.75 + t * 0.2)


def material_to_color(material: Material, height: float = 0) -> Tuple[float, float, float]:
    """Convert material type to a color."""
    colors = {
        Material.AIR: (0.7, 0.85, 1.0),  # Sky blue
        Material.STONE: (0.5, 0.5, 0.52),
        Material.DIRT: (0.45, 0.32, 0.2),
        Material.GRASS: (0.3, 0.55, 0.25),
        Material.SAND: (0.76, 0.7, 0.5),
        Material.WATER: (0.2, 0.4, 0.7),
        Material.BEDROCK: (0.15, 0.15, 0.18),
    }
    return colors.get(material, (0.5, 0.5, 0.5))


def generate_tile_mesh(tile: TerrainTile, tile_size: float = 1.0,
                       water_level: float = 0.0) -> MeshData:
    """
    Generate mesh data for a single tile.
    
    Creates a quad for the tile surface.
    """
    nw, ne, sw, se = tile.surface_heights
    
    # World coordinates of tile corners
    x0, z0 = tile.x, tile.z
    x1, z1 = tile.x + tile_size, tile.z + tile_size
    
    # Vertices: NW, SW, SE, NE (counter-clockwise when viewed from above +Y)
    # This ensures the face normal points UP
    vertices = np.array([
        [x0, nw, z0],  # NW
        [x0, sw, z1],  # SW
        [x1, se, z1],  # SE
        [x1, ne, z0],  # NE
    ], dtype=np.float32)
    
    # Calculate normal
    nx, ny, nz = get_tile_normal(tile, tile_size)
    normals = np.array([
        [nx, ny, nz],
        [nx, ny, nz],
        [nx, ny, nz],
        [nx, ny, nz],
    ], dtype=np.float32)
    
    # Color based on average height
    avg_height = (nw + ne + sw + se) / 4
    r, g, b = height_to_color(avg_height, water_level)
    colors = np.array([
        [r, g, b],
        [r, g, b],
        [r, g, b],
        [r, g, b],
    ], dtype=np.float32)
    
    return MeshData(
        vertices=vertices,
        normals=normals,
        colors=colors,
        primitive='quads',
    )


def generate_chunk_mesh(chunk: Chunk, water_level: float = 0.0,
                        use_merging: bool = True) -> MeshData:
    """
    Generate mesh data for an entire chunk.
    
    Args:
        chunk: The chunk to mesh
        water_level: Water level for coloring
        use_merging: Whether to use segment merging optimization
    """
    all_vertices = []
    all_normals = []
    all_colors = []
    
    if use_merging and len(chunk.tiles) > 10:
        # Use segment merging for large chunks
        mesh = _generate_merged_chunk_mesh(chunk, water_level)
        if mesh.vertex_count > 0:
            return mesh
    
    # Fallback: generate mesh tile by tile
    for (lx, lz), tile in chunk.tiles.items():
        tile_mesh = generate_tile_mesh(tile, chunk.tile_size, water_level)
        
        all_vertices.append(tile_mesh.vertices)
        all_normals.append(tile_mesh.normals)
        all_colors.append(tile_mesh.colors)
    
    if not all_vertices:
        return MeshData()
    
    return MeshData(
        vertices=np.vstack(all_vertices),
        normals=np.vstack(all_normals),
        colors=np.vstack(all_colors),
        primitive='quads',
    )


def _generate_merged_chunk_mesh(chunk: Chunk, water_level: float) -> MeshData:
    """
    Generate mesh using segment merging optimization.
    
    Groups adjacent tiles with similar heights into larger quads.
    """
    merged_groups = find_mergeable_segments(
        chunk.tiles, 
        y_tolerance=0.5,
        tile_size=chunk.tile_size
    )
    
    if not merged_groups:
        return MeshData()
    
    all_vertices = []
    all_normals = []
    all_colors = []
    
    for group in merged_groups:
        # Create a single quad for the merged group
        x0, z0 = group.x_min, group.z_min
        x1, z1 = group.x_max, group.z_max
        y = (group.y_bottom + group.y_top) / 2  # Use average height
        
        # Counter-clockwise when viewed from above: NW, SW, SE, NE
        vertices = np.array([
            [x0, y, z0],  # NW
            [x0, y, z1],  # SW
            [x1, y, z1],  # SE
            [x1, y, z0],  # NE
        ], dtype=np.float32)
        
        # Flat normal (up)
        normals = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        
        # Color based on height
        r, g, b = height_to_color(y, water_level)
        colors = np.array([
            [r, g, b],
            [r, g, b],
            [r, g, b],
            [r, g, b],
        ], dtype=np.float32)
        
        all_vertices.append(vertices)
        all_normals.append(normals)
        all_colors.append(colors)
    
    # Also generate tiles that weren't merged
    merged_coords = set()
    for group in merged_groups:
        merged_coords.update(group.tile_coords)
    
    for (lx, lz), tile in chunk.tiles.items():
        if (lx, lz) not in merged_coords:
            tile_mesh = generate_tile_mesh(tile, chunk.tile_size, water_level)
            all_vertices.append(tile_mesh.vertices)
            all_normals.append(tile_mesh.normals)
            all_colors.append(tile_mesh.colors)
    
    return MeshData(
        vertices=np.vstack(all_vertices) if all_vertices else None,
        normals=np.vstack(all_normals) if all_normals else None,
        colors=np.vstack(all_colors) if all_colors else None,
        primitive='quads',
    )


def generate_wall_mesh(tile: TerrainTile, neighbor: Optional[TerrainTile],
                       direction: str, tile_size: float = 1.0,
                       water_level: float = 0.0) -> Optional[MeshData]:
    """
    Generate wall mesh between a tile and its neighbor (for cliffs/pits).
    
    Walls are needed when there's a significant height difference.
    """
    if neighbor is None:
        return None
    
    # Get heights along the shared edge
    if direction == 'north':
        our_heights = (tile.nw_height, tile.ne_height)
        their_heights = (neighbor.sw_height, neighbor.se_height)
        x0, x1 = tile.x, tile.x + tile_size
        z = tile.z
    elif direction == 'south':
        our_heights = (tile.sw_height, tile.se_height)
        their_heights = (neighbor.nw_height, neighbor.ne_height)
        x0, x1 = tile.x, tile.x + tile_size
        z = tile.z + tile_size
    elif direction == 'west':
        our_heights = (tile.nw_height, tile.sw_height)
        their_heights = (neighbor.ne_height, neighbor.se_height)
        x = tile.x
        z0, z1 = tile.z, tile.z + tile_size
    elif direction == 'east':
        our_heights = (tile.ne_height, tile.se_height)
        their_heights = (neighbor.nw_height, neighbor.sw_height)
        x = tile.x + tile_size
        z0, z1 = tile.z, tile.z + tile_size
    else:
        return None
    
    # Check if wall is needed
    height_diff = max(
        abs(our_heights[0] - their_heights[0]),
        abs(our_heights[1] - their_heights[1])
    )
    
    if height_diff < 0.1:
        return None
    
    # Create wall quad
    if direction in ('north', 'south'):
        if our_heights[0] > their_heights[0] or our_heights[1] > their_heights[1]:
            # We're higher - wall faces their direction
            vertices = np.array([
                [x0, our_heights[0], z],
                [x1, our_heights[1], z],
                [x1, their_heights[1], z],
                [x0, their_heights[0], z],
            ], dtype=np.float32)
            nz = -1 if direction == 'north' else 1
            normals = np.full((4, 3), [0, 0, nz], dtype=np.float32)
        else:
            return None
    else:
        if our_heights[0] > their_heights[0] or our_heights[1] > their_heights[1]:
            vertices = np.array([
                [x, our_heights[0], z0],
                [x, our_heights[1], z1],
                [x, their_heights[1], z1],
                [x, their_heights[0], z0],
            ], dtype=np.float32)
            nx = -1 if direction == 'west' else 1
            normals = np.full((4, 3), [nx, 0, 0], dtype=np.float32)
        else:
            return None
    
    # Wall color (darker, rocky)
    avg_height = sum(our_heights + their_heights) / 4
    r, g, b = height_to_color(avg_height, water_level)
    colors = np.full((4, 3), [r * 0.7, g * 0.7, b * 0.7], dtype=np.float32)
    
    return MeshData(
        vertices=vertices,
        normals=normals,
        colors=colors,
        primitive='quads',
    )


def generate_cave_mesh(column: TerrainColumn, tile_size: float = 1.0) -> List[MeshData]:
    """
    Generate mesh for cave walls in a column.
    
    Creates geometry for the interior surfaces of caves.
    """
    meshes = []
    
    if len(column.segments) < 2:
        return meshes
    
    # Generate ceiling/floor surfaces for each gap between segments
    for i in range(len(column.segments) - 1):
        lower_seg = column.segments[i]
        upper_seg = column.segments[i + 1]
        
        gap_bottom = lower_seg.y_top
        gap_top = upper_seg.y_bottom
        
        if gap_top - gap_bottom < 0.1:
            continue
        
        x, z = column.x, column.z
        
        # Cave ceiling (underside of upper segment) - normal points DOWN
        # Clockwise when viewed from below (so CCW from above, but we want it visible from below)
        ceiling_verts = np.array([
            [x, gap_top, z],
            [x + tile_size, gap_top, z],
            [x + tile_size, gap_top, z + tile_size],
            [x, gap_top, z + tile_size],
        ], dtype=np.float32)
        
        meshes.append(MeshData(
            vertices=ceiling_verts,
            normals=np.full((4, 3), [0, -1, 0], dtype=np.float32),
            colors=np.full((4, 3), [0.35, 0.35, 0.38], dtype=np.float32),
            primitive='quads',
        ))
        
        # Cave floor (top of lower segment) - normal points UP
        # Counter-clockwise when viewed from above
        floor_verts = np.array([
            [x, gap_bottom, z],
            [x, gap_bottom, z + tile_size],
            [x + tile_size, gap_bottom, z + tile_size],
            [x + tile_size, gap_bottom, z],
        ], dtype=np.float32)
        
        meshes.append(MeshData(
            vertices=floor_verts,
            normals=np.full((4, 3), [0, 1, 0], dtype=np.float32),
            colors=np.full((4, 3), [0.4, 0.38, 0.35], dtype=np.float32),
            primitive='quads',
        ))
    
    return meshes


def generate_water_mesh(x_min: float, z_min: float, x_max: float, z_max: float,
                        water_level: float = 0.0) -> MeshData:
    """Generate a water plane mesh."""
    # Counter-clockwise when viewed from above
    vertices = np.array([
        [x_min, water_level, z_min],  # NW
        [x_min, water_level, z_max],  # SW
        [x_max, water_level, z_max],  # SE
        [x_max, water_level, z_min],  # NE
    ], dtype=np.float32)
    
    normals = np.full((4, 3), [0, 1, 0], dtype=np.float32)
    
    # Semi-transparent blue
    colors = np.full((4, 4), [0.2, 0.4, 0.7, 0.6], dtype=np.float32)
    
    return MeshData(
        vertices=vertices,
        normals=normals,
        colors=colors,
        primitive='quads',
    )


def combine_meshes(meshes: List[MeshData]) -> MeshData:
    """Combine multiple meshes into one."""
    if not meshes:
        return MeshData()
    
    all_vertices = [m.vertices for m in meshes if m.vertices is not None]
    all_normals = [m.normals for m in meshes if m.normals is not None]
    all_colors = [m.colors for m in meshes if m.colors is not None]
    
    if not all_vertices:
        return MeshData()
    
    return MeshData(
        vertices=np.vstack(all_vertices),
        normals=np.vstack(all_normals) if all_normals else None,
        colors=np.vstack(all_colors) if all_colors else None,
        primitive=meshes[0].primitive,
    )

