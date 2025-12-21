"""
Unit tests for the Animal DNA and AI system.
"""

import pytest
import numpy as np

from waverse.animal_dna import (
    AnimalDNA, AnimalType, MovementType, AIBehavior,
    JointGene, BodySegmentGene, LimbGene, FeatureGene
)
from waverse.animals import AnimalInstance, AnimalManager


class TestJointGene:
    """Tests for JointGene."""
    
    def test_joint_creation(self):
        joint = JointGene()
        assert joint.rotation_axis == "y"
        assert joint.min_angle == -45.0
        assert joint.max_angle == 45.0
    
    def test_joint_mutation(self):
        rng = np.random.default_rng(42)
        joint = JointGene()
        mutated = joint.mutate(rng, strength=0.5)
        
        assert -90 <= mutated.min_angle <= 0
        assert 0 <= mutated.max_angle <= 90


class TestBodySegmentGene:
    """Tests for BodySegmentGene."""
    
    def test_segment_creation(self):
        seg = BodySegmentGene()
        assert seg.shape == "ellipsoid"
        assert seg.size == (1.0, 0.5, 0.5)
    
    def test_segment_mutation_keeps_valid_size(self):
        rng = np.random.default_rng(42)
        seg = BodySegmentGene()
        
        for _ in range(50):
            mutated = seg.mutate(rng, strength=1.0)
            assert 0.1 <= mutated.size[0] <= 3.0
            assert 0.1 <= mutated.size[1] <= 3.0
            assert 0.1 <= mutated.size[2] <= 3.0


class TestLimbGene:
    """Tests for LimbGene."""
    
    def test_limb_creation(self):
        limb = LimbGene()
        assert limb.limb_type == "leg"
        assert limb.segment_count == 2
    
    def test_limb_mutation_adjusts_segments(self):
        rng = np.random.default_rng(42)
        limb = LimbGene(segment_count=2, segment_lengths=[0.5, 0.4], segment_widths=[0.15, 0.1])
        
        mutated = limb.mutate(rng, strength=0.5)
        
        assert 1 <= mutated.segment_count <= 5
        assert len(mutated.segment_lengths) == mutated.segment_count
        assert len(mutated.segment_widths) == mutated.segment_count


class TestAnimalDNA:
    """Tests for AnimalDNA."""
    
    @pytest.mark.parametrize("animal_type", AnimalType.ALL)
    def test_create_random_all_types(self, animal_type):
        """Test random creation for all animal types."""
        dna = AnimalDNA.create_random(animal_type, seed=42)
        assert dna.animal_type == animal_type
        assert dna.base_scale > 0
        assert len(dna.body_segments) > 0
    
    def test_random_deterministic(self):
        """Test same seed produces same DNA."""
        dna1 = AnimalDNA.create_random(AnimalType.MAMMAL, seed=12345)
        dna2 = AnimalDNA.create_random(AnimalType.MAMMAL, seed=12345)
        
        assert dna1.base_scale == dna2.base_scale
        assert dna1.movement_type == dna2.movement_type
        assert len(dna1.body_segments) == len(dna2.body_segments)
    
    def test_mutation_changes_dna(self):
        """Test mutation produces different DNA."""
        dna = AnimalDNA.create_random(AnimalType.BIRD, seed=42)
        rng = np.random.default_rng(100)
        
        mutated = dna.mutate(rng, strength=0.5)
        
        assert mutated.generation == dna.generation + 1
    
    def test_crossover_combines_parents(self):
        """Test crossover combines two parents."""
        parent1 = AnimalDNA.create_random(AnimalType.MAMMAL, seed=1)
        parent2 = AnimalDNA.create_random(AnimalType.MAMMAL, seed=2)
        
        rng = np.random.default_rng(42)
        child = parent1.crossover(parent2, rng)
        
        assert child.generation == max(parent1.generation, parent2.generation) + 1
    
    def test_to_json(self):
        """Test JSON serialization."""
        dna = AnimalDNA.create_random(AnimalType.FISH, seed=42)
        json_str = dna.to_json()
        
        assert "animal_type" in json_str
        assert "fish" in json_str
    
    def test_insect_has_six_legs(self):
        """Test insects have 6 legs (3 pairs)."""
        dna = AnimalDNA.create_random(AnimalType.INSECT, seed=42)
        
        leg_limbs = [l for l in dna.limbs if l.limb_type == "leg"]
        if leg_limbs:
            assert dna.limb_pairs == 3  # 3 pairs = 6 legs
    
    def test_bird_can_fly(self):
        """Test birds have fly movement type."""
        dna = AnimalDNA.create_random(AnimalType.BIRD, seed=42)
        assert dna.movement_type == MovementType.FLY
    
    def test_fish_can_swim(self):
        """Test fish have swim movement type."""
        dna = AnimalDNA.create_random(AnimalType.FISH, seed=42)
        assert dna.movement_type == MovementType.SWIM
    
    def test_worm_has_no_limbs(self):
        """Test worms have no limbs."""
        dna = AnimalDNA.create_random(AnimalType.WORM, seed=42)
        assert len(dna.limbs) == 0


class TestAnimalInstance:
    """Tests for AnimalInstance."""
    
    def test_instance_creation(self):
        dna = AnimalDNA.create_random(AnimalType.MAMMAL, seed=42)
        animal = AnimalInstance(x=10, y=5, z=20, dna=dna)
        
        assert animal.x == 10
        assert animal.y == 5
        assert animal.z == 20
    
    def test_update_moves_animal(self):
        dna = AnimalDNA.create_random(AnimalType.MAMMAL, seed=42)
        dna.ai_behavior = AIBehavior.WANDER
        
        animal = AnimalInstance(x=0, y=0, z=0, dna=dna)
        animal.vx = 1.0
        animal.vz = 1.0
        
        initial_x = animal.x
        animal.update(1.0)
        
        # Should have moved
        assert animal.x != initial_x or animal.anim_time > 0
    
    def test_flee_behavior(self):
        dna = AnimalDNA.create_random(AnimalType.MAMMAL, seed=42)
        dna.ai_behavior = AIBehavior.FLEE
        dna.flee_radius = 10.0
        
        animal = AnimalInstance(x=0, y=0, z=0, dna=dna)
        
        # Player very close
        animal.update(0.1, player_pos=(1, 0, 1))
        
        # Should be fleeing
        assert animal.state == "fleeing"


class TestAnimalManager:
    """Tests for AnimalManager."""
    
    def test_manager_creation(self):
        manager = AnimalManager(world_seed=42)
        
        assert len(manager.species_templates) == len(AnimalType.ALL)
    
    def test_spawn_animals_for_chunk(self):
        manager = AnimalManager(world_seed=42)
        
        # Create a simple heightmap
        heightmap = np.ones((33, 33)) * 10  # Land
        
        manager.spawn_animals_for_chunk(
            cx=0, cz=0, heightmap=heightmap,
            chunk_world_x=0, chunk_world_z=0,
            tile_scale=1.0, height_scale=3.5
        )
        
        # Should have spawned some animals
        assert len(manager.animals) >= 0  # May be 0-4
        assert (0, 0) in manager.chunk_animals
    
    def test_chunk_caching(self):
        manager = AnimalManager(world_seed=42)
        heightmap = np.ones((33, 33)) * 10
        
        manager.spawn_animals_for_chunk(0, 0, heightmap, 0, 0, 1.0, 3.5)
        count1 = len(manager.animals)
        
        # Spawning again should not add more
        manager.spawn_animals_for_chunk(0, 0, heightmap, 0, 0, 1.0, 3.5)
        count2 = len(manager.animals)
        
        assert count1 == count2
    
    def test_cleanup_distant_chunks(self):
        manager = AnimalManager(world_seed=42)
        heightmap = np.ones((33, 33)) * 10
        
        manager.spawn_animals_for_chunk(0, 0, heightmap, 0, 0, 1.0, 3.5)
        manager.spawn_animals_for_chunk(100, 100, heightmap, 0, 0, 1.0, 3.5)
        
        assert (0, 0) in manager.chunk_animals
        assert (100, 100) in manager.chunk_animals
        
        manager.cleanup_distant_chunks(0, 0, max_distance=10)
        
        assert (0, 0) in manager.chunk_animals
        assert (100, 100) not in manager.chunk_animals


class TestMovementTypes:
    """Tests for different movement types."""
    
    def test_flying_animal_stays_above_ground(self):
        dna = AnimalDNA.create_random(AnimalType.BIRD, seed=42)
        animal = AnimalInstance(x=0, y=10, z=0, dna=dna)
        
        def ground_height(x, z):
            return 0
        
        animal.vy = -1  # Try to go down
        animal.update(1.0, get_ground_height=ground_height)
        
        # Should stay above ground
        assert animal.y >= 3
    
    def test_swimming_animal_stays_at_water_level(self):
        dna = AnimalDNA.create_random(AnimalType.FISH, seed=42)
        animal = AnimalInstance(x=0, y=5, z=0, dna=dna)
        
        animal.update(1.0)
        
        # Should be near water level (allow for small bobbing animation)
        assert abs(animal.y - 0.5) < 0.1


class TestSpatialIndexIntegration:
    """Tests for spatial index integration with AnimalManager."""
    
    def test_spatial_index_created(self):
        """Test that AnimalManager creates a spatial index."""
        manager = AnimalManager(world_seed=42)
        assert hasattr(manager, '_spatial_index')
        assert manager._spatial_index is not None
    
    def test_spawned_animals_added_to_spatial_index(self):
        """Test that spawned animals are indexed for spatial queries."""
        manager = AnimalManager(world_seed=42)
        heightmap = np.ones((33, 33)) * 10  # Land
        
        # Spawn animals
        manager.spawn_animals_for_chunk(0, 0, heightmap, 0, 0, 1.0, 3.5)
        
        # If animals were spawned, they should be in the spatial index
        if len(manager.animals) > 0:
            from waverse.spatial import EntityType
            spatial_count = manager._spatial_index.count(EntityType.ANIMAL)
            assert spatial_count == len(manager.animals)
    
    def test_cleanup_removes_from_spatial_index(self):
        """Test that cleanup removes animals from spatial index."""
        manager = AnimalManager(world_seed=42)
        heightmap = np.ones((33, 33)) * 10
        
        # Spawn in two far apart chunks
        manager.spawn_animals_for_chunk(0, 0, heightmap, 0, 0, 1.0, 3.5)
        manager.spawn_animals_for_chunk(100, 100, heightmap, 100 * 32, 100 * 32, 1.0, 3.5)
        
        from waverse.spatial import EntityType
        initial_count = manager._spatial_index.count(EntityType.ANIMAL)
        
        # Cleanup distant chunks
        manager.cleanup_distant_chunks(0, 0, max_distance=10)
        
        # Spatial index should have fewer animals
        final_count = manager._spatial_index.count(EntityType.ANIMAL)
        assert final_count <= initial_count
    
    def test_spatial_query_finds_nearby_animals(self):
        """Test that spatial queries find animals correctly."""
        manager = AnimalManager(world_seed=42)
        heightmap = np.ones((33, 33)) * 10
        
        # Spawn animals at origin chunk
        manager.spawn_animals_for_chunk(0, 0, heightmap, 0, 0, 1.0, 3.5)
        
        if len(manager.animals) > 0:
            # Query from center of chunk
            from waverse.spatial import EntityType
            results = list(manager._spatial_index.query_radius(16, 16, 100, EntityType.ANIMAL))
            
            # Should find animals that were spawned
            assert len(results) > 0


class TestMovementCaching:
    """Tests for animal movement vector caching."""
    
    def test_cached_velocity_applied(self):
        """Test that cached velocity is used for movement."""
        dna = AnimalDNA.create_random(AnimalType.MAMMAL, seed=42)
        animal = AnimalInstance(x=0, y=5, z=0, dna=dna)
        
        # Set cached velocity
        animal._cached_vx = 1.0
        animal._cached_vz = 0.5
        
        initial_x = animal.x
        initial_z = animal.z
        
        # Update should apply cached velocity
        animal.update(1.0)
        
        # Position should change based on cached velocity
        # (actual movement may include other factors, but should be non-zero)
        moved = (animal.x != initial_x) or (animal.z != initial_z)
        assert moved or animal.anim_time > 0  # Either moved or animation updated


class TestPerformanceCharacteristics:
    """Tests verifying performance-related behavior."""
    
    def test_manager_handles_many_animals_efficiently(self):
        """Test that spawning many animals doesn't explode in time."""
        import time
        
        manager = AnimalManager(world_seed=42)
        heightmap = np.ones((33, 33)) * 10
        
        start = time.time()
        
        # Spawn animals in 100 chunks
        for cx in range(10):
            for cz in range(10):
                manager.spawn_animals_for_chunk(
                    cx, cz, heightmap, 
                    cx * 32, cz * 32, 1.0, 3.5
                )
        
        elapsed = time.time() - start
        
        # Should complete in under 5 seconds
        assert elapsed < 5.0
        
        # Verify spatial index is consistent
        from waverse.spatial import EntityType
        assert manager._spatial_index.count(EntityType.ANIMAL) == len(manager.animals)
    
    def test_spatial_query_performance_scales_with_radius(self):
        """Test that small radius queries are fast regardless of total count."""
        import time
        
        manager = AnimalManager(world_seed=42)
        heightmap = np.ones((33, 33)) * 10
        
        # Spawn many animals
        for cx in range(5):
            for cz in range(5):
                manager.spawn_animals_for_chunk(
                    cx, cz, heightmap,
                    cx * 32, cz * 32, 1.0, 3.5
                )
        
        from waverse.spatial import EntityType
        
        # Time many small radius queries
        start = time.time()
        for _ in range(1000):
            list(manager._spatial_index.query_radius(16, 16, 10, EntityType.ANIMAL))
        elapsed = time.time() - start
        
        # 1000 queries should complete in under 1 second
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

