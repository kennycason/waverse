"""
Spatial Index for efficient entity lookups.

Uses grid-based spatial hashing for O(1) cell lookups and O(K) neighbor queries
where K is the number of entities in nearby cells (not total entities).

This is critical for performance when checking entity interactions like:
- Animals eating nearby plants
- Carnivores hunting nearby prey
- Animals finding mates
- Collision detection
"""

from collections import defaultdict
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum, auto
import math


class EntityType(Enum):
    """Types of entities that can be indexed."""
    PLANT = auto()
    ANIMAL = auto()
    STRUCTURE = auto()
    EGG = auto()
    ANY = auto()  # For queries that don't filter by type


@dataclass
class SpatialEntity:
    """Wrapper for entities in the spatial index."""
    entity: Any           # The actual entity object
    entity_id: int        # Unique ID (usually id(entity))
    entity_type: EntityType
    x: float
    z: float
    # Optional cached data to avoid repeated getattr calls
    cached_data: Dict[str, Any] = field(default_factory=dict)


class SpatialIndex:
    """
    Grid-based spatial index for fast neighbor queries.
    
    The world is divided into cells of `cell_size` units.
    Entities are stored in the cell(s) they occupy.
    Neighbor queries only check relevant cells.
    
    Time Complexity:
    - insert: O(1)
    - remove: O(1) 
    - update_position: O(1)
    - query_radius: O(K) where K = entities in nearby cells
    - query_nearest: O(K log K) due to sorting
    
    Space Complexity: O(N) where N = number of entities
    """
    
    def __init__(self, cell_size: float = 10.0):
        """
        Initialize spatial index.
        
        Args:
            cell_size: Size of each grid cell. Smaller = more precise but more cells.
                      Should be roughly the size of typical query radius.
        """
        self.cell_size = cell_size
        
        # Main storage: cell coords -> list of SpatialEntity
        self._cells: Dict[Tuple[int, int], List[SpatialEntity]] = defaultdict(list)
        
        # Reverse lookup: entity_id -> (cell_x, cell_z, SpatialEntity)
        self._entity_locations: Dict[int, Tuple[int, int, SpatialEntity]] = {}
        
        # Type indexes for fast type-filtered queries
        self._by_type: Dict[EntityType, Set[int]] = defaultdict(set)
        
        # Stats for debugging/profiling
        self.stats = {
            'inserts': 0,
            'removes': 0,
            'updates': 0,
            'queries': 0,
        }
    
    def _get_cell(self, x: float, z: float) -> Tuple[int, int]:
        """Convert world coordinates to cell coordinates."""
        return (int(math.floor(x / self.cell_size)), 
                int(math.floor(z / self.cell_size)))
    
    def insert(self, entity: Any, x: float, z: float, 
               entity_type: EntityType = EntityType.ANY,
               entity_id: Optional[int] = None,
               cached_data: Optional[Dict[str, Any]] = None) -> int:
        """
        Insert an entity into the spatial index.
        
        Args:
            entity: The entity object
            x, z: World position
            entity_type: Type for filtered queries
            entity_id: Unique ID (defaults to id(entity))
            cached_data: Optional pre-computed data to cache
            
        Returns:
            The entity_id used
        """
        if entity_id is None:
            entity_id = id(entity)
        
        # Remove if already exists (update case)
        if entity_id in self._entity_locations:
            self.remove(entity_id)
        
        cell = self._get_cell(x, z)
        spatial_entity = SpatialEntity(
            entity=entity,
            entity_id=entity_id,
            entity_type=entity_type,
            x=x,
            z=z,
            cached_data=cached_data or {}
        )
        
        self._cells[cell].append(spatial_entity)
        self._entity_locations[entity_id] = (cell[0], cell[1], spatial_entity)
        self._by_type[entity_type].add(entity_id)
        
        self.stats['inserts'] += 1
        return entity_id
    
    def remove(self, entity_id: int) -> bool:
        """
        Remove an entity from the index.
        
        Args:
            entity_id: The entity's unique ID
            
        Returns:
            True if entity was found and removed
        """
        if entity_id not in self._entity_locations:
            return False
        
        cx, cz, spatial_entity = self._entity_locations[entity_id]
        cell = (cx, cz)
        
        # Remove from cell
        self._cells[cell] = [e for e in self._cells[cell] if e.entity_id != entity_id]
        
        # Clean up empty cells
        if not self._cells[cell]:
            del self._cells[cell]
        
        # Remove from type index
        self._by_type[spatial_entity.entity_type].discard(entity_id)
        
        # Remove from location lookup
        del self._entity_locations[entity_id]
        
        self.stats['removes'] += 1
        return True
    
    def update_position(self, entity_id: int, new_x: float, new_z: float) -> bool:
        """
        Update an entity's position efficiently.
        
        Only moves between cells if necessary.
        
        Args:
            entity_id: The entity's unique ID
            new_x, new_z: New world position
            
        Returns:
            True if entity was found and updated
        """
        if entity_id not in self._entity_locations:
            return False
        
        old_cx, old_cz, spatial_entity = self._entity_locations[entity_id]
        new_cell = self._get_cell(new_x, new_z)
        
        # Update position
        spatial_entity.x = new_x
        spatial_entity.z = new_z
        
        # Check if cell changed
        if (old_cx, old_cz) != new_cell:
            # Remove from old cell
            old_cell = (old_cx, old_cz)
            self._cells[old_cell] = [e for e in self._cells[old_cell] if e.entity_id != entity_id]
            if not self._cells[old_cell]:
                del self._cells[old_cell]
            
            # Add to new cell
            self._cells[new_cell].append(spatial_entity)
            self._entity_locations[entity_id] = (new_cell[0], new_cell[1], spatial_entity)
        
        self.stats['updates'] += 1
        return True
    
    def get(self, entity_id: int) -> Optional[SpatialEntity]:
        """Get a spatial entity by ID."""
        if entity_id not in self._entity_locations:
            return None
        return self._entity_locations[entity_id][2]
    
    def query_radius(self, x: float, z: float, radius: float,
                     entity_type: Optional[EntityType] = None,
                     exclude_ids: Optional[Set[int]] = None) -> Generator[SpatialEntity, None, None]:
        """
        Find all entities within radius of a point.
        
        Args:
            x, z: Center point
            radius: Search radius
            entity_type: Optional type filter (None = all types)
            exclude_ids: Optional set of entity IDs to skip
            
        Yields:
            SpatialEntity objects within radius
        """
        self.stats['queries'] += 1
        
        radius_sq = radius * radius
        cells_to_check = int(math.ceil(radius / self.cell_size)) + 1
        center_cell = self._get_cell(x, z)
        
        exclude_ids = exclude_ids or set()
        type_filter = self._by_type.get(entity_type) if entity_type else None
        
        for dx in range(-cells_to_check, cells_to_check + 1):
            for dz in range(-cells_to_check, cells_to_check + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dz)
                
                for spatial_entity in self._cells.get(cell, []):
                    # Skip excluded
                    if spatial_entity.entity_id in exclude_ids:
                        continue
                    
                    # Type filter
                    if type_filter is not None and spatial_entity.entity_id not in type_filter:
                        continue
                    
                    # Distance check (squared to avoid sqrt)
                    ex, ez = spatial_entity.x, spatial_entity.z
                    dist_sq = (x - ex) ** 2 + (z - ez) ** 2
                    
                    if dist_sq <= radius_sq:
                        yield spatial_entity
    
    def query_radius_sorted(self, x: float, z: float, radius: float,
                            entity_type: Optional[EntityType] = None,
                            exclude_ids: Optional[Set[int]] = None,
                            max_results: Optional[int] = None) -> List[Tuple[SpatialEntity, float]]:
        """
        Find entities within radius, sorted by distance.
        
        Args:
            x, z: Center point
            radius: Search radius
            entity_type: Optional type filter
            exclude_ids: Optional set of entity IDs to skip
            max_results: Optional limit on results
            
        Returns:
            List of (SpatialEntity, distance) tuples, sorted nearest first
        """
        results = []
        for entity in self.query_radius(x, z, radius, entity_type, exclude_ids):
            dist = math.sqrt((x - entity.x) ** 2 + (z - entity.z) ** 2)
            results.append((entity, dist))
        
        results.sort(key=lambda x: x[1])
        
        if max_results:
            return results[:max_results]
        return results
    
    def query_nearest(self, x: float, z: float, 
                      entity_type: Optional[EntityType] = None,
                      exclude_ids: Optional[Set[int]] = None,
                      max_radius: float = 100.0) -> Optional[Tuple[SpatialEntity, float]]:
        """
        Find the nearest entity to a point.
        
        Args:
            x, z: Center point
            entity_type: Optional type filter
            exclude_ids: Optional set of entity IDs to skip
            max_radius: Maximum search radius
            
        Returns:
            (SpatialEntity, distance) or None if nothing found
        """
        results = self.query_radius_sorted(x, z, max_radius, entity_type, exclude_ids, max_results=1)
        return results[0] if results else None
    
    def query_cell(self, cell_x: int, cell_z: int,
                   entity_type: Optional[EntityType] = None) -> List[SpatialEntity]:
        """
        Get all entities in a specific cell.
        
        Args:
            cell_x, cell_z: Cell coordinates
            entity_type: Optional type filter
            
        Returns:
            List of SpatialEntity in that cell
        """
        cell = (cell_x, cell_z)
        entities = self._cells.get(cell, [])
        
        if entity_type is None:
            return list(entities)
        
        return [e for e in entities if e.entity_type == entity_type]
    
    def query_chunk(self, chunk_x: int, chunk_z: int, chunk_size: float = 128.0,
                    entity_type: Optional[EntityType] = None) -> List[SpatialEntity]:
        """
        Get all entities in a game chunk.
        
        Args:
            chunk_x, chunk_z: Chunk coordinates
            chunk_size: Size of game chunks in world units
            entity_type: Optional type filter
            
        Returns:
            List of SpatialEntity in that chunk
        """
        # Calculate cell range for this chunk
        min_x = chunk_x * chunk_size
        max_x = min_x + chunk_size
        min_z = chunk_z * chunk_size
        max_z = min_z + chunk_size
        
        min_cell = self._get_cell(min_x, min_z)
        max_cell = self._get_cell(max_x - 0.01, max_z - 0.01)  # -0.01 to stay in chunk
        
        results = []
        type_filter = self._by_type.get(entity_type) if entity_type else None
        
        for cx in range(min_cell[0], max_cell[0] + 1):
            for cz in range(min_cell[1], max_cell[1] + 1):
                for entity in self._cells.get((cx, cz), []):
                    if type_filter is not None and entity.entity_id not in type_filter:
                        continue
                    # Verify entity is actually in chunk bounds
                    if min_x <= entity.x < max_x and min_z <= entity.z < max_z:
                        results.append(entity)
        
        return results
    
    def get_all_of_type(self, entity_type: EntityType) -> Generator[SpatialEntity, None, None]:
        """
        Get all entities of a specific type.
        
        Args:
            entity_type: The type to filter by
            
        Yields:
            All SpatialEntity of that type
        """
        for entity_id in self._by_type.get(entity_type, set()):
            if entity_id in self._entity_locations:
                yield self._entity_locations[entity_id][2]
    
    def count(self, entity_type: Optional[EntityType] = None) -> int:
        """
        Count entities in the index.
        
        Args:
            entity_type: Optional type filter
            
        Returns:
            Number of entities
        """
        if entity_type is None:
            return len(self._entity_locations)
        return len(self._by_type.get(entity_type, set()))
    
    def clear(self):
        """Remove all entities from the index."""
        self._cells.clear()
        self._entity_locations.clear()
        self._by_type.clear()
    
    def clear_type(self, entity_type: EntityType):
        """Remove all entities of a specific type."""
        ids_to_remove = list(self._by_type.get(entity_type, set()))
        for entity_id in ids_to_remove:
            self.remove(entity_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics for debugging/profiling."""
        return {
            **self.stats,
            'total_entities': len(self._entity_locations),
            'total_cells': len(self._cells),
            'entities_by_type': {t.name: len(ids) for t, ids in self._by_type.items()},
            'avg_entities_per_cell': (
                sum(len(c) for c in self._cells.values()) / len(self._cells)
                if self._cells else 0
            ),
        }
    
    def rebuild(self):
        """
        Rebuild the index from scratch.
        
        Useful after many updates to compact storage.
        """
        # Collect all entities - _entity_locations stores (cx, cz, SpatialEntity)
        entities = [(loc[2].entity, loc[2].x, loc[2].z, loc[2].entity_type, 
                     loc[2].entity_id, loc[2].cached_data)
                    for loc in self._entity_locations.values()]
        
        # Clear and re-insert
        self.clear()
        for entity, x, z, etype, eid, cached in entities:
            self.insert(entity, x, z, etype, eid, cached)


# Convenience function for creating typed indexes
def create_entity_index(cell_size: float = 10.0) -> SpatialIndex:
    """Create a new spatial index with default settings."""
    return SpatialIndex(cell_size=cell_size)

