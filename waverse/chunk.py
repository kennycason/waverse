"""
Chunk System - Infinite world management through chunks.

Chunks are fixed-size regions of the world that are:
- Generated on demand as the player explores
- Cached for performance
- Unloaded when far from the player
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Set, Any, TYPE_CHECKING
import numpy as np
from collections import OrderedDict
import time

from .terrain import TerrainColumn, TerrainTile, Segment, Material
from .waves import evaluate_waves, evaluate_density, generate_height_grid

if TYPE_CHECKING:
    from .dna import WaveDNA


# Default configuration
DEFAULT_CHUNK_SIZE = 32  # Tiles per chunk edge
DEFAULT_TILE_SIZE = 1.0  # World units per tile
DEFAULT_CACHE_SIZE = 64  # Max chunks to keep in memory


@dataclass
class Chunk:
    """
    A chunk is a fixed-size region of the world.
    
    Chunks contain a grid of tiles and manage their generation/caching.
    """
    # Chunk grid position (not world coordinates)
    cx: int
    cz: int
    
    # Chunk size in tiles
    size: int = DEFAULT_CHUNK_SIZE
    
    # Tile size in world units
    tile_size: float = DEFAULT_TILE_SIZE
    
    # The tiles in this chunk (indexed by local coordinates)
    tiles: Dict[Tuple[int, int], TerrainTile] = field(default_factory=dict)
    
    # Shared columns between tiles (to avoid duplication)
    # Key: (local_x, local_z) of column grid point
    columns: Dict[Tuple[int, int], TerrainColumn] = field(default_factory=dict)
    
    # Generation state
    generated: bool = False
    generation_time: float = 0.0
    
    # Modification tracking
    modified: bool = False
    last_access: float = field(default_factory=time.time)
    
    # Cached mesh data (for rendering)
    mesh_dirty: bool = True
    mesh_data: Optional[Any] = None
    
    @property
    def world_x(self) -> float:
        """World X coordinate of chunk's southwest corner."""
        return self.cx * self.size * self.tile_size
    
    @property
    def world_z(self) -> float:
        """World Z coordinate of chunk's southwest corner."""
        return self.cz * self.size * self.tile_size
    
    @property
    def world_bounds(self) -> Tuple[float, float, float, float]:
        """Returns (x_min, z_min, x_max, z_max) in world coordinates."""
        size_world = self.size * self.tile_size
        return (
            self.world_x,
            self.world_z,
            self.world_x + size_world,
            self.world_z + size_world,
        )
    
    def world_to_local(self, world_x: float, world_z: float) -> Tuple[int, int]:
        """Convert world coordinates to local tile coordinates."""
        local_x = int((world_x - self.world_x) / self.tile_size)
        local_z = int((world_z - self.world_z) / self.tile_size)
        return (local_x, local_z)
    
    def local_to_world(self, local_x: int, local_z: int) -> Tuple[float, float]:
        """Convert local tile coordinates to world coordinates."""
        return (
            self.world_x + local_x * self.tile_size,
            self.world_z + local_z * self.tile_size,
        )
    
    def contains_world_point(self, world_x: float, world_z: float) -> bool:
        """Check if a world coordinate is within this chunk."""
        x_min, z_min, x_max, z_max = self.world_bounds
        return x_min <= world_x < x_max and z_min <= world_z < z_max
    
    def get_tile(self, local_x: int, local_z: int) -> Optional[TerrainTile]:
        """Get tile at local coordinates."""
        return self.tiles.get((local_x, local_z))
    
    def get_tile_at_world(self, world_x: float, world_z: float) -> Optional[TerrainTile]:
        """Get tile at world coordinates."""
        local_x, local_z = self.world_to_local(world_x, world_z)
        return self.get_tile(local_x, local_z)
    
    def get_height_at(self, world_x: float, world_z: float) -> float:
        """
        Get interpolated terrain height at world coordinates.
        """
        local_x, local_z = self.world_to_local(world_x, world_z)
        tile = self.get_tile(local_x, local_z)
        
        if tile is None:
            return 0.0
        
        # Calculate position within tile (0-1 range)
        tile_world_x, tile_world_z = self.local_to_world(local_x, local_z)
        frac_x = (world_x - tile_world_x) / self.tile_size
        frac_z = (world_z - tile_world_z) / self.tile_size
        
        # Bilinear interpolation of corner heights
        from .terrain import interpolate_height
        return interpolate_height(tile, frac_x, frac_z)
    
    def mark_accessed(self):
        """Update last access time (for LRU caching)."""
        self.last_access = time.time()
    
    def mark_modified(self):
        """Mark chunk as modified (needs saving)."""
        self.modified = True
        self.mesh_dirty = True


def generate_chunk(dna: "WaveDNA", cx: int, cz: int, 
                   use_caves: bool = True) -> Chunk:
    """
    Generate a new chunk from DNA.
    
    This is the main chunk generation function that:
    1. Creates the chunk structure
    2. Generates height values using wave functions
    3. Optionally generates cave structures
    4. Creates tiles and columns
    """
    start_time = time.time()
    
    chunk = Chunk(
        cx=cx,
        cz=cz,
        size=dna.chunk_size,
        tile_size=dna.tile_size,
    )
    
    # Generate height grid for this chunk
    # We need size+1 points to have 4 corners for each tile
    grid_size = chunk.size + 1
    
    heights = generate_height_grid(
        dna,
        chunk.world_x,
        chunk.world_z,
        grid_size,
        grid_size,
        chunk.tile_size,
    )
    
    # Create columns at each grid point
    for lz in range(grid_size):
        for lx in range(grid_size):
            world_x = chunk.world_x + lx * chunk.tile_size
            world_z = chunk.world_z + lz * chunk.tile_size
            height = heights[lz, lx]
            
            # Create column with single segment (no caves for now)
            column = TerrainColumn.from_surface_height(
                x=int(world_x),
                z=int(world_z),
                surface_height=height,
                bedrock_level=dna.bedrock_level,
            )
            chunk.columns[(lx, lz)] = column
    
    # Create tiles using shared columns
    for lz in range(chunk.size):
        for lx in range(chunk.size):
            # Get the 4 corner columns
            nw = chunk.columns[(lx, lz)]
            ne = chunk.columns[(lx + 1, lz)]
            sw = chunk.columns[(lx, lz + 1)]
            se = chunk.columns[(lx + 1, lz + 1)]
            
            tile = TerrainTile(
                x=int(chunk.world_x + lx * chunk.tile_size),
                z=int(chunk.world_z + lz * chunk.tile_size),
                corners=(nw, ne, sw, se),
            )
            chunk.tiles[(lx, lz)] = tile
    
    # Generate caves if enabled and DNA has cave layers
    if use_caves and dna.cave_layers:
        _generate_chunk_caves(chunk, dna)
    
    chunk.generated = True
    chunk.generation_time = time.time() - start_time
    
    return chunk


def _generate_chunk_caves(chunk: Chunk, dna: "WaveDNA"):
    """
    Generate cave structures within a chunk using 3D density evaluation.
    """
    # Sample density at each column position at various heights
    y_min = dna.bedrock_level
    y_max = max(col.surface_height for col in chunk.columns.values())
    y_step = 2.0  # Sample every 2 units vertically
    
    y_samples = np.arange(y_min, y_max, y_step)
    if len(y_samples) == 0:
        return
    
    for (lx, lz), column in chunk.columns.items():
        world_x = chunk.world_x + lx * chunk.tile_size
        world_z = chunk.world_z + lz * chunk.tile_size
        
        # Evaluate density at all y positions for this column
        x_arr = np.full(len(y_samples), world_x)
        z_arr = np.full(len(y_samples), world_z)
        
        densities = evaluate_density(dna, x_arr, y_samples, z_arr)
        
        # Rebuild column segments based on density
        new_column = TerrainColumn.from_density_column(
            x=int(world_x),
            z=int(world_z),
            densities=densities,
            y_start=y_min,
            y_step=y_step,
            threshold=0.0,
        )
        
        # Copy over the new segments
        column.segments = new_column.segments
        column.optimize_segments()
    
    # Mark all tiles as needing mesh update
    for tile in chunk.tiles.values():
        tile.invalidate_cache()


class ChunkManager:
    """
    Manages the infinite world through chunk loading/unloading.
    
    Features:
    - Generates chunks on demand
    - LRU cache for chunk memory management
    - Neighbor awareness for seamless terrain
    - Dirty tracking for saving modifications
    """
    
    def __init__(self, dna: "WaveDNA", 
                 cache_size: int = DEFAULT_CACHE_SIZE,
                 generate_caves: bool = True):
        """
        Initialize the chunk manager.
        
        Args:
            dna: The world DNA for generation
            cache_size: Maximum number of chunks to keep in memory
            generate_caves: Whether to generate cave structures
        """
        self.dna = dna
        self.cache_size = cache_size
        self.generate_caves = generate_caves
        
        # Chunk cache (OrderedDict for LRU behavior)
        self.chunks: OrderedDict[Tuple[int, int], Chunk] = OrderedDict()
        
        # Track which chunks have been modified
        self.modified_chunks: Set[Tuple[int, int]] = set()
        
        # Statistics
        self.chunks_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0
    
    def world_to_chunk(self, world_x: float, world_z: float) -> Tuple[int, int]:
        """Convert world coordinates to chunk coordinates."""
        chunk_size_world = self.dna.chunk_size * self.dna.tile_size
        cx = int(np.floor(world_x / chunk_size_world))
        cz = int(np.floor(world_z / chunk_size_world))
        return (cx, cz)
    
    def get_chunk(self, cx: int, cz: int) -> Chunk:
        """
        Get a chunk, generating it if necessary.
        
        This is the main access point for chunks.
        """
        key = (cx, cz)
        
        if key in self.chunks:
            # Cache hit - move to end (most recently used)
            self.chunks.move_to_end(key)
            self.chunks[key].mark_accessed()
            self.cache_hits += 1
            return self.chunks[key]
        
        # Cache miss - generate new chunk
        self.cache_misses += 1
        chunk = generate_chunk(self.dna, cx, cz, self.generate_caves)
        self.chunks_generated += 1
        
        # Add to cache
        self.chunks[key] = chunk
        
        # Evict old chunks if over capacity
        while len(self.chunks) > self.cache_size:
            self._evict_oldest()
        
        return chunk
    
    def get_chunk_at_world(self, world_x: float, world_z: float) -> Chunk:
        """Get the chunk containing the given world coordinates."""
        cx, cz = self.world_to_chunk(world_x, world_z)
        return self.get_chunk(cx, cz)
    
    def get_chunks_in_radius(self, cx: int, cz: int, 
                              radius: int) -> List[Chunk]:
        """
        Get all chunks within a radius of the given chunk position.
        
        Useful for preloading chunks around the player.
        """
        chunks = []
        for dz in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                chunks.append(self.get_chunk(cx + dx, cz + dz))
        return chunks
    
    def preload_around(self, world_x: float, world_z: float, radius: int = 2):
        """
        Preload chunks around a world position.
        
        Call this when the player moves to ensure nearby chunks are ready.
        """
        cx, cz = self.world_to_chunk(world_x, world_z)
        self.get_chunks_in_radius(cx, cz, radius)
    
    def get_height_at(self, world_x: float, world_z: float) -> float:
        """Get terrain height at world coordinates."""
        chunk = self.get_chunk_at_world(world_x, world_z)
        return chunk.get_height_at(world_x, world_z)
    
    def get_tile_at(self, world_x: float, world_z: float) -> Optional[TerrainTile]:
        """Get the tile at world coordinates."""
        chunk = self.get_chunk_at_world(world_x, world_z)
        return chunk.get_tile_at_world(world_x, world_z)
    
    def dig_at(self, world_x: float, world_z: float, amount: float):
        """
        Dig down at a world position.
        """
        tile = self.get_tile_at(world_x, world_z)
        if tile:
            tile.dig(amount)
            cx, cz = self.world_to_chunk(world_x, world_z)
            self.modified_chunks.add((cx, cz))
            self.chunks[(cx, cz)].mark_modified()
    
    def dig_horizontal_at(self, world_x: float, world_z: float, 
                          y: float, direction: str, height: float = 2.0):
        """
        Dig horizontally at a world position.
        """
        tile = self.get_tile_at(world_x, world_z)
        if tile:
            tile.dig_horizontal(direction, y, height)
            cx, cz = self.world_to_chunk(world_x, world_z)
            self.modified_chunks.add((cx, cz))
            self.chunks[(cx, cz)].mark_modified()
    
    def _evict_oldest(self):
        """Evict the least recently used chunk (if not modified)."""
        # Try to evict unmodified chunks first
        for key in list(self.chunks.keys()):
            if key not in self.modified_chunks:
                del self.chunks[key]
                return
        
        # If all chunks are modified, evict the oldest anyway
        # (in practice, you'd want to save it first)
        if self.chunks:
            oldest_key = next(iter(self.chunks))
            del self.chunks[oldest_key]
    
    def unload_chunk(self, cx: int, cz: int):
        """Explicitly unload a chunk (after saving if needed)."""
        key = (cx, cz)
        if key in self.chunks:
            del self.chunks[key]
            self.modified_chunks.discard(key)
    
    def get_loaded_chunks(self) -> List[Tuple[int, int]]:
        """Get list of currently loaded chunk coordinates."""
        return list(self.chunks.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "loaded_chunks": len(self.chunks),
            "cache_size": self.cache_size,
            "chunks_generated": self.chunks_generated,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "modified_chunks": len(self.modified_chunks),
        }
    
    def clear_cache(self):
        """Clear all cached chunks (loses unsaved modifications!)."""
        self.chunks.clear()
        self.modified_chunks.clear()


# =============================================================================
# Chunk Serialization (for saving/loading)
# =============================================================================

def chunk_to_dict(chunk: Chunk) -> Dict[str, Any]:
    """
    Serialize a chunk to a dictionary (for saving).
    
    Only saves modifications - the base terrain can be regenerated from DNA.
    """
    # Only save modified columns
    modified_columns = {}
    for (lx, lz), column in chunk.columns.items():
        if column.dirty:
            modified_columns[f"{lx},{lz}"] = {
                "segments": [
                    {"y_bottom": s.y_bottom, "y_top": s.y_top, 
                     "material": s.material.name}
                    for s in column.segments
                ]
            }
    
    # Save tunnel faces
    tunnel_data = {}
    for (lx, lz), tile in chunk.tiles.items():
        if tile.tunnel_faces:
            tunnel_data[f"{lx},{lz}"] = {
                direction: [
                    {"y_bottom": f.y_bottom, "y_top": f.y_top, "width": f.width}
                    for f in faces
                ]
                for direction, faces in tile.tunnel_faces.items()
            }
    
    return {
        "cx": chunk.cx,
        "cz": chunk.cz,
        "size": chunk.size,
        "tile_size": chunk.tile_size,
        "modified_columns": modified_columns,
        "tunnel_data": tunnel_data,
    }


def apply_chunk_modifications(chunk: Chunk, data: Dict[str, Any]):
    """
    Apply saved modifications to a regenerated chunk.
    """
    # Restore modified columns
    for key, col_data in data.get("modified_columns", {}).items():
        lx, lz = map(int, key.split(","))
        if (lx, lz) in chunk.columns:
            column = chunk.columns[(lx, lz)]
            column.segments = [
                Segment(
                    y_bottom=s["y_bottom"],
                    y_top=s["y_top"],
                    material=Material[s["material"]],
                    modified=True,
                )
                for s in col_data["segments"]
            ]
            column.dirty = True
    
    # Restore tunnel faces
    for key, tile_data in data.get("tunnel_data", {}).items():
        lx, lz = map(int, key.split(","))
        if (lx, lz) in chunk.tiles:
            tile = chunk.tiles[(lx, lz)]
            from .terrain import TunnelFace
            for direction, faces in tile_data.items():
                tile.tunnel_faces[direction] = [
                    TunnelFace(
                        direction=direction,
                        y_bottom=f["y_bottom"],
                        y_top=f["y_top"],
                        width=f.get("width", 1.0),
                    )
                    for f in faces
                ]
            tile.invalidate_cache()
    
    chunk.modified = True

