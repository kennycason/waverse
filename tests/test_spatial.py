"""
Comprehensive unit tests for the SpatialIndex.

Tests cover:
- Basic CRUD operations (insert, remove, update)
- Radius queries with/without type filters
- Edge cases (empty index, boundary conditions)
- Performance characteristics
- Chunk-based queries
"""

import pytest
import math
import random
from dataclasses import dataclass
from typing import List

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from waverse.spatial import SpatialIndex, SpatialEntity, EntityType, create_entity_index


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockPlant:
    """Mock plant entity for testing."""
    x: float
    z: float
    name: str = "plant"


@dataclass  
class MockAnimal:
    """Mock animal entity for testing."""
    x: float
    z: float
    name: str = "animal"


@pytest.fixture
def empty_index():
    """Create an empty spatial index."""
    return SpatialIndex(cell_size=10.0)


@pytest.fixture
def populated_index():
    """Create an index with some test entities."""
    index = SpatialIndex(cell_size=10.0)
    
    # Add plants in a grid pattern
    for i in range(5):
        for j in range(5):
            plant = MockPlant(x=i * 5, z=j * 5, name=f"plant_{i}_{j}")
            index.insert(plant, plant.x, plant.z, EntityType.PLANT)
    
    # Add some animals
    for i in range(3):
        animal = MockAnimal(x=i * 20 + 2, z=i * 20 + 2, name=f"animal_{i}")
        index.insert(animal, animal.x, animal.z, EntityType.ANIMAL)
    
    return index


# =============================================================================
# Basic Insert/Remove Tests
# =============================================================================

class TestInsertRemove:
    """Test basic insert and remove operations."""
    
    def test_insert_single_entity(self, empty_index):
        """Test inserting a single entity."""
        plant = MockPlant(x=5, z=5)
        entity_id = empty_index.insert(plant, plant.x, plant.z, EntityType.PLANT)
        
        assert empty_index.count() == 1
        assert empty_index.count(EntityType.PLANT) == 1
        assert empty_index.count(EntityType.ANIMAL) == 0
        
        # Verify we can retrieve it
        spatial = empty_index.get(entity_id)
        assert spatial is not None
        assert spatial.entity is plant
        assert spatial.x == 5
        assert spatial.z == 5
    
    def test_insert_multiple_entities(self, empty_index):
        """Test inserting multiple entities."""
        for i in range(100):
            plant = MockPlant(x=i, z=i)
            empty_index.insert(plant, plant.x, plant.z, EntityType.PLANT)
        
        assert empty_index.count() == 100
        assert empty_index.count(EntityType.PLANT) == 100
    
    def test_insert_with_custom_id(self, empty_index):
        """Test inserting with a custom entity ID."""
        plant = MockPlant(x=5, z=5)
        custom_id = 12345
        
        returned_id = empty_index.insert(plant, plant.x, plant.z, EntityType.PLANT, entity_id=custom_id)
        
        assert returned_id == custom_id
        assert empty_index.get(custom_id) is not None
    
    def test_insert_overwrites_existing(self, empty_index):
        """Test that inserting with same ID updates the entity."""
        plant1 = MockPlant(x=5, z=5, name="first")
        plant2 = MockPlant(x=10, z=10, name="second")
        custom_id = 12345
        
        empty_index.insert(plant1, plant1.x, plant1.z, EntityType.PLANT, entity_id=custom_id)
        empty_index.insert(plant2, plant2.x, plant2.z, EntityType.PLANT, entity_id=custom_id)
        
        assert empty_index.count() == 1
        spatial = empty_index.get(custom_id)
        assert spatial.entity.name == "second"
        assert spatial.x == 10
        assert spatial.z == 10
    
    def test_remove_existing(self, empty_index):
        """Test removing an existing entity."""
        plant = MockPlant(x=5, z=5)
        entity_id = empty_index.insert(plant, plant.x, plant.z, EntityType.PLANT)
        
        result = empty_index.remove(entity_id)
        
        assert result is True
        assert empty_index.count() == 0
        assert empty_index.get(entity_id) is None
    
    def test_remove_nonexistent(self, empty_index):
        """Test removing a non-existent entity."""
        result = empty_index.remove(99999)
        assert result is False
    
    def test_clear(self, populated_index):
        """Test clearing the index."""
        assert populated_index.count() > 0
        
        populated_index.clear()
        
        assert populated_index.count() == 0
        assert populated_index.count(EntityType.PLANT) == 0
        assert populated_index.count(EntityType.ANIMAL) == 0
    
    def test_clear_type(self, populated_index):
        """Test clearing only one entity type."""
        plant_count = populated_index.count(EntityType.PLANT)
        animal_count = populated_index.count(EntityType.ANIMAL)
        
        populated_index.clear_type(EntityType.PLANT)
        
        assert populated_index.count(EntityType.PLANT) == 0
        assert populated_index.count(EntityType.ANIMAL) == animal_count


# =============================================================================
# Position Update Tests
# =============================================================================

class TestPositionUpdate:
    """Test position update operations."""
    
    def test_update_position_same_cell(self, empty_index):
        """Test updating position within the same cell."""
        plant = MockPlant(x=5, z=5)
        entity_id = empty_index.insert(plant, plant.x, plant.z, EntityType.PLANT)
        
        # Move slightly (should stay in same cell with cell_size=10)
        result = empty_index.update_position(entity_id, 7, 7)
        
        assert result is True
        spatial = empty_index.get(entity_id)
        assert spatial.x == 7
        assert spatial.z == 7
    
    def test_update_position_different_cell(self, empty_index):
        """Test updating position to a different cell."""
        plant = MockPlant(x=5, z=5)
        entity_id = empty_index.insert(plant, plant.x, plant.z, EntityType.PLANT)
        
        # Move to different cell
        result = empty_index.update_position(entity_id, 25, 25)
        
        assert result is True
        spatial = empty_index.get(entity_id)
        assert spatial.x == 25
        assert spatial.z == 25
        
        # Should be findable at new position
        found = list(empty_index.query_radius(25, 25, 5))
        assert len(found) == 1
        assert found[0].entity_id == entity_id
        
        # Should NOT be found at old position
        found_old = list(empty_index.query_radius(5, 5, 5))
        assert len(found_old) == 0
    
    def test_update_position_nonexistent(self, empty_index):
        """Test updating position of non-existent entity."""
        result = empty_index.update_position(99999, 10, 10)
        assert result is False


# =============================================================================
# Radius Query Tests
# =============================================================================

class TestRadiusQueries:
    """Test radius-based queries."""
    
    def test_query_radius_empty(self, empty_index):
        """Test radius query on empty index."""
        results = list(empty_index.query_radius(0, 0, 100))
        assert len(results) == 0
    
    def test_query_radius_finds_nearby(self, empty_index):
        """Test that radius query finds entities within range."""
        # Place entities at known positions
        p1 = MockPlant(x=0, z=0)
        p2 = MockPlant(x=5, z=0)  # Distance 5
        p3 = MockPlant(x=10, z=0)  # Distance 10
        p4 = MockPlant(x=20, z=0)  # Distance 20
        
        empty_index.insert(p1, 0, 0, EntityType.PLANT)
        empty_index.insert(p2, 5, 0, EntityType.PLANT)
        empty_index.insert(p3, 10, 0, EntityType.PLANT)
        empty_index.insert(p4, 20, 0, EntityType.PLANT)
        
        # Query with radius 7 from origin
        results = list(empty_index.query_radius(0, 0, 7))
        assert len(results) == 2  # p1 and p2
        
        # Query with radius 12
        results = list(empty_index.query_radius(0, 0, 12))
        assert len(results) == 3  # p1, p2, p3
        
        # Query with radius 25
        results = list(empty_index.query_radius(0, 0, 25))
        assert len(results) == 4  # All
    
    def test_query_radius_excludes_far(self, empty_index):
        """Test that radius query excludes entities out of range."""
        p_near = MockPlant(x=3, z=4)  # Distance 5 from origin
        p_far = MockPlant(x=100, z=100)  # Far away
        
        empty_index.insert(p_near, 3, 4, EntityType.PLANT)
        empty_index.insert(p_far, 100, 100, EntityType.PLANT)
        
        results = list(empty_index.query_radius(0, 0, 10))
        assert len(results) == 1
        assert results[0].entity is p_near
    
    def test_query_radius_with_type_filter(self, populated_index):
        """Test radius query with entity type filter."""
        # Query for plants only
        plants = list(populated_index.query_radius(0, 0, 100, entity_type=EntityType.PLANT))
        animals = list(populated_index.query_radius(0, 0, 100, entity_type=EntityType.ANIMAL))
        all_entities = list(populated_index.query_radius(0, 0, 100))
        
        assert len(plants) == populated_index.count(EntityType.PLANT)
        assert len(animals) == populated_index.count(EntityType.ANIMAL)
        assert len(all_entities) == populated_index.count()
    
    def test_query_radius_with_exclude(self, empty_index):
        """Test radius query with exclusion set."""
        p1 = MockPlant(x=0, z=0)
        p2 = MockPlant(x=1, z=1)
        p3 = MockPlant(x=2, z=2)
        
        id1 = empty_index.insert(p1, 0, 0, EntityType.PLANT)
        id2 = empty_index.insert(p2, 1, 1, EntityType.PLANT)
        id3 = empty_index.insert(p3, 2, 2, EntityType.PLANT)
        
        # Exclude p2
        results = list(empty_index.query_radius(1, 1, 10, exclude_ids={id2}))
        assert len(results) == 2
        result_ids = {r.entity_id for r in results}
        assert id1 in result_ids
        assert id3 in result_ids
        assert id2 not in result_ids
    
    def test_query_radius_sorted(self, empty_index):
        """Test sorted radius query returns nearest first."""
        p1 = MockPlant(x=5, z=0)   # Distance 5
        p2 = MockPlant(x=10, z=0)  # Distance 10
        p3 = MockPlant(x=3, z=0)   # Distance 3
        
        empty_index.insert(p1, 5, 0, EntityType.PLANT)
        empty_index.insert(p2, 10, 0, EntityType.PLANT)
        empty_index.insert(p3, 3, 0, EntityType.PLANT)
        
        results = empty_index.query_radius_sorted(0, 0, 20)
        
        assert len(results) == 3
        assert results[0][1] == 3.0  # Nearest
        assert results[1][1] == 5.0
        assert results[2][1] == 10.0  # Farthest
    
    def test_query_radius_sorted_max_results(self, empty_index):
        """Test sorted radius query respects max_results limit."""
        # Insert 10 plants at increasing distances
        for i in range(1, 11):
            p = MockPlant(x=i * 5, z=0, name=f"plant_{i}")
            empty_index.insert(p, i * 5, 0, EntityType.PLANT)
        
        # Query with max_results=3
        results = empty_index.query_radius_sorted(0, 0, 100, max_results=3)
        
        assert len(results) == 3
        # Should be the 3 nearest
        assert results[0][1] == 5.0   # Distance 5
        assert results[1][1] == 10.0  # Distance 10
        assert results[2][1] == 15.0  # Distance 15
    
    def test_query_radius_sorted_max_results_exceeds_total(self, empty_index):
        """Test max_results larger than available entities."""
        p1 = MockPlant(x=5, z=0)
        p2 = MockPlant(x=10, z=0)
        
        empty_index.insert(p1, 5, 0, EntityType.PLANT)
        empty_index.insert(p2, 10, 0, EntityType.PLANT)
        
        # Request more than available
        results = empty_index.query_radius_sorted(0, 0, 20, max_results=10)
        
        assert len(results) == 2  # Only 2 available
    
    def test_query_radius_sorted_with_type_filter_and_max_results(self, empty_index):
        """Test max_results works correctly with type filter."""
        for i in range(5):
            p = MockPlant(x=i * 3 + 1, z=0)
            empty_index.insert(p, p.x, 0, EntityType.PLANT)
        for i in range(5):
            a = MockAnimal(x=i * 3 + 2, z=0)
            empty_index.insert(a, a.x, 0, EntityType.ANIMAL)
        
        # Get 2 nearest plants only
        results = empty_index.query_radius_sorted(
            0, 0, 50, entity_type=EntityType.PLANT, max_results=2
        )
        
        assert len(results) == 2
        for entity, dist in results:
            assert entity.entity_type == EntityType.PLANT
    
    def test_query_nearest(self, empty_index):
        """Test finding the nearest entity."""
        p1 = MockPlant(x=5, z=0)
        p2 = MockPlant(x=10, z=0)
        p3 = MockPlant(x=3, z=0)
        
        empty_index.insert(p1, 5, 0, EntityType.PLANT)
        empty_index.insert(p2, 10, 0, EntityType.PLANT)
        empty_index.insert(p3, 3, 0, EntityType.PLANT)
        
        result = empty_index.query_nearest(0, 0)
        
        assert result is not None
        assert result[0].entity is p3
        assert result[1] == 3.0
    
    def test_query_nearest_empty(self, empty_index):
        """Test query_nearest on empty index."""
        result = empty_index.query_nearest(0, 0)
        assert result is None


# =============================================================================
# Cell and Chunk Query Tests
# =============================================================================

class TestCellChunkQueries:
    """Test cell and chunk-based queries."""
    
    def test_query_cell(self, empty_index):
        """Test querying a specific cell."""
        # Cell size is 10, so (0,0) to (10,10) is cell (0,0)
        p1 = MockPlant(x=5, z=5)   # Cell (0,0)
        p2 = MockPlant(x=15, z=5)  # Cell (1,0)
        p3 = MockPlant(x=5, z=15)  # Cell (0,1)
        
        empty_index.insert(p1, 5, 5, EntityType.PLANT)
        empty_index.insert(p2, 15, 5, EntityType.PLANT)
        empty_index.insert(p3, 5, 15, EntityType.PLANT)
        
        cell_00 = empty_index.query_cell(0, 0)
        cell_10 = empty_index.query_cell(1, 0)
        cell_01 = empty_index.query_cell(0, 1)
        cell_11 = empty_index.query_cell(1, 1)
        
        assert len(cell_00) == 1
        assert len(cell_10) == 1
        assert len(cell_01) == 1
        assert len(cell_11) == 0
    
    def test_query_chunk(self, empty_index):
        """Test querying a game chunk."""
        # Add entities across multiple cells but same chunk
        for i in range(10):
            p = MockPlant(x=i * 10 + 5, z=i * 10 + 5)
            empty_index.insert(p, p.x, p.z, EntityType.PLANT)
        
        # Query chunk (0,0) which is 0-128 in world coords
        chunk_entities = empty_index.query_chunk(0, 0, chunk_size=128)
        
        # All entities should be in chunk 0,0
        assert len(chunk_entities) == 10
    
    def test_query_chunk_with_type(self, populated_index):
        """Test chunk query with type filter."""
        plants = populated_index.query_chunk(0, 0, chunk_size=128, entity_type=EntityType.PLANT)
        animals = populated_index.query_chunk(0, 0, chunk_size=128, entity_type=EntityType.ANIMAL)
        
        # Verify all returned entities are of correct type
        for p in plants:
            assert p.entity_type == EntityType.PLANT
        for a in animals:
            assert a.entity_type == EntityType.ANIMAL


# =============================================================================
# Edge Cases and Boundary Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_negative_coordinates(self, empty_index):
        """Test entities at negative coordinates."""
        p1 = MockPlant(x=-50, z=-50)
        p2 = MockPlant(x=-45, z=-45)  # Close to p1
        
        id1 = empty_index.insert(p1, -50, -50, EntityType.PLANT)
        id2 = empty_index.insert(p2, -45, -45, EntityType.PLANT)
        
        # Should be able to find them - query at (-47, -47) with radius 10
        # p1 distance: sqrt(9 + 9) = ~4.24
        # p2 distance: sqrt(4 + 4) = ~2.83
        results = list(empty_index.query_radius(-47, -47, 10))
        assert len(results) == 2
    
    def test_large_coordinates(self, empty_index):
        """Test entities at very large coordinates."""
        p1 = MockPlant(x=100000, z=100000)
        p2 = MockPlant(x=100005, z=100005)
        
        empty_index.insert(p1, 100000, 100000, EntityType.PLANT)
        empty_index.insert(p2, 100005, 100005, EntityType.PLANT)
        
        results = list(empty_index.query_radius(100002, 100002, 10))
        assert len(results) == 2
    
    def test_zero_radius_query(self, empty_index):
        """Test query with zero radius."""
        p1 = MockPlant(x=0, z=0)
        empty_index.insert(p1, 0, 0, EntityType.PLANT)
        
        # Zero radius should still find entity at exact position
        results = list(empty_index.query_radius(0, 0, 0))
        assert len(results) == 1
    
    def test_entity_on_cell_boundary(self, empty_index):
        """Test entity exactly on cell boundary."""
        # Cell boundary at x=10 (cell_size=10)
        p = MockPlant(x=10.0, z=5.0)
        entity_id = empty_index.insert(p, 10.0, 5.0, EntityType.PLANT)
        
        # Should be findable
        spatial = empty_index.get(entity_id)
        assert spatial is not None
        
        # Should be in cell (1, 0) not (0, 0)
        cell = empty_index._get_cell(10.0, 5.0)
        assert cell == (1, 0)
    
    def test_many_entities_same_position(self, empty_index):
        """Test many entities at the same position."""
        for i in range(100):
            p = MockPlant(x=5, z=5, name=f"plant_{i}")
            empty_index.insert(p, 5, 5, EntityType.PLANT)
        
        assert empty_index.count() == 100
        
        results = list(empty_index.query_radius(5, 5, 1))
        assert len(results) == 100


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_insert_many(self, empty_index):
        """Test inserting many entities is fast."""
        import time
        
        start = time.time()
        for i in range(10000):
            x = random.uniform(-1000, 1000)
            z = random.uniform(-1000, 1000)
            p = MockPlant(x=x, z=z)
            empty_index.insert(p, x, z, EntityType.PLANT)
        elapsed = time.time() - start
        
        assert empty_index.count() == 10000
        assert elapsed < 2.0  # Should complete in under 2 seconds
    
    def test_query_radius_performance(self, empty_index):
        """Test that radius queries scale with nearby entities, not total."""
        import time
        
        # Add 10000 entities spread across large area
        for i in range(10000):
            x = random.uniform(-10000, 10000)
            z = random.uniform(-10000, 10000)
            p = MockPlant(x=x, z=z)
            empty_index.insert(p, x, z, EntityType.PLANT)
        
        # Query a small radius - should be fast regardless of total count
        start = time.time()
        for _ in range(1000):
            list(empty_index.query_radius(0, 0, 50))
        elapsed = time.time() - start
        
        # 1000 queries should complete in under 1 second
        assert elapsed < 1.0
    
    def test_get_all_of_type_performance(self, empty_index):
        """Test iterating all entities of a type."""
        import time
        
        # Add mixed entities
        for i in range(5000):
            p = MockPlant(x=i, z=0)
            empty_index.insert(p, i, 0, EntityType.PLANT)
        for i in range(5000):
            a = MockAnimal(x=i, z=100)
            empty_index.insert(a, i, 100, EntityType.ANIMAL)
        
        start = time.time()
        plants = list(empty_index.get_all_of_type(EntityType.PLANT))
        elapsed = time.time() - start
        
        assert len(plants) == 5000
        assert elapsed < 0.5


# =============================================================================
# Statistics and Utilities Tests  
# =============================================================================

class TestUtilities:
    """Test utility methods."""
    
    def test_get_stats(self, populated_index):
        """Test getting index statistics."""
        stats = populated_index.get_stats()
        
        assert 'total_entities' in stats
        assert 'total_cells' in stats
        assert 'entities_by_type' in stats
        assert stats['total_entities'] == populated_index.count()
    
    def test_rebuild(self, empty_index):
        """Test rebuilding the index."""
        # Add entities
        for i in range(100):
            p = MockPlant(x=i, z=i)
            empty_index.insert(p, i, i, EntityType.PLANT)
        
        # Remove some
        for i in range(50):
            empty_index.remove(id(MockPlant(x=i, z=i)))  # Won't find by new id
        
        original_count = empty_index.count()
        
        # Rebuild
        empty_index.rebuild()
        
        # Count should be same
        assert empty_index.count() == original_count
    
    def test_cached_data(self, empty_index):
        """Test storing cached data with entities."""
        p = MockPlant(x=5, z=5)
        cached = {'energy': 100, 'growth': 0.5}
        
        entity_id = empty_index.insert(p, 5, 5, EntityType.PLANT, cached_data=cached)
        
        spatial = empty_index.get(entity_id)
        assert spatial.cached_data['energy'] == 100
        assert spatial.cached_data['growth'] == 0.5


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

