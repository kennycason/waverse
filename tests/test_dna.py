"""
Unit tests for the DNA system.

Tests cover:
- Gene mutation and crossover
- PlantDNA creation and manipulation
- DNA serialization
- DNAPool neighbor crossover
"""

import pytest
import numpy as np
import json

from waverse.dna import (
    Gene, ColorGene, SegmentGene, PlantDNA, PlantType, DNAPool
)


class TestGene:
    """Tests for the Gene class."""
    
    def test_gene_creation(self):
        """Test basic gene creation."""
        gene = Gene(0.5, 0.0, 1.0, 0.1)
        assert gene.value == 0.5
        assert gene.min_val == 0.0
        assert gene.max_val == 1.0
        assert gene.mutation_rate == 0.1
    
    def test_gene_mutation_stays_in_bounds(self):
        """Test that mutation keeps values within bounds."""
        rng = np.random.default_rng(42)
        gene = Gene(0.9, 0.0, 1.0, 0.5)
        
        # Mutate many times
        for _ in range(100):
            mutated = gene.mutate(rng, strength=1.0)
            assert mutated.min_val <= mutated.value <= mutated.max_val
    
    def test_gene_crossover(self):
        """Test gene crossover produces valid offspring."""
        rng = np.random.default_rng(42)
        gene1 = Gene(0.2, 0.0, 1.0)
        gene2 = Gene(0.8, 0.0, 1.0)
        
        offspring = gene1.crossover(gene2, rng)
        
        assert offspring.min_val == gene1.min_val
        assert offspring.max_val == gene1.max_val
        assert 0.0 <= offspring.value <= 1.0
    
    def test_gene_deterministic_with_seed(self):
        """Test that same seed produces same mutation."""
        gene = Gene(0.5, 0.0, 1.0, 0.2)
        
        rng1 = np.random.default_rng(12345)
        mutated1 = gene.mutate(rng1, strength=0.5)
        
        rng2 = np.random.default_rng(12345)
        mutated2 = gene.mutate(rng2, strength=0.5)
        
        assert mutated1.value == mutated2.value


class TestColorGene:
    """Tests for the ColorGene class."""
    
    def test_color_creation(self):
        """Test color gene creation."""
        color = ColorGene(0.5, 0.3, 0.8)
        assert color.rgb == (0.5, 0.3, 0.8)
    
    def test_color_from_hsv(self):
        """Test HSV to RGB conversion."""
        # Red
        red = ColorGene.from_hsv(0.0, 1.0, 1.0)
        assert abs(red.r - 1.0) < 0.01
        assert abs(red.g - 0.0) < 0.01
        
        # Green  
        green = ColorGene.from_hsv(0.333, 1.0, 1.0)
        assert abs(green.g - 1.0) < 0.01
        
        # Blue
        blue = ColorGene.from_hsv(0.666, 1.0, 1.0)
        assert abs(blue.b - 1.0) < 0.01
    
    def test_color_mutation_stays_valid(self):
        """Test color mutation keeps RGB in 0-1 range."""
        rng = np.random.default_rng(42)
        color = ColorGene(0.9, 0.1, 0.5)
        
        for _ in range(100):
            mutated = color.mutate(rng, strength=1.0)
            assert 0.0 <= mutated.r <= 1.0
            assert 0.0 <= mutated.g <= 1.0
            assert 0.0 <= mutated.b <= 1.0
    
    def test_color_crossover(self):
        """Test color crossover blends properly."""
        rng = np.random.default_rng(42)
        c1 = ColorGene(0.0, 0.0, 0.0)
        c2 = ColorGene(1.0, 1.0, 1.0)
        
        # Crossover should produce intermediate values
        offspring = c1.crossover(c2, rng)
        assert 0.0 <= offspring.r <= 1.0
        assert 0.0 <= offspring.g <= 1.0
        assert 0.0 <= offspring.b <= 1.0


class TestSegmentGene:
    """Tests for the SegmentGene class."""
    
    def test_segment_creation(self):
        """Test segment creation with defaults."""
        seg = SegmentGene()
        assert seg.length == 1.0
        assert seg.width == 0.2
        assert seg.taper == 0.8
    
    def test_segment_mutation(self):
        """Test segment mutation keeps values valid."""
        rng = np.random.default_rng(42)
        seg = SegmentGene(1.0, 0.2, 0.8, 0.0, 0.0)
        
        for _ in range(50):
            mutated = seg.mutate(rng, strength=1.0)
            assert 0.1 <= mutated.length <= 5.0
            assert 0.05 <= mutated.width <= 1.0
            assert 0.3 <= mutated.taper <= 1.0
            assert -1 <= mutated.curve <= 1
            assert 0 <= mutated.twist <= 1


class TestPlantDNA:
    """Tests for the PlantDNA class."""
    
    def test_create_random_all_types(self):
        """Test random creation for all plant types."""
        for plant_type in PlantType.ALL_TYPES:
            dna = PlantDNA.create_random(plant_type, seed=42)
            assert dna.plant_type == plant_type
            assert dna.height_gene.value > 0
            assert dna.species_id is not None
    
    def test_random_deterministic(self):
        """Test that same seed produces same DNA."""
        dna1 = PlantDNA.create_random(PlantType.TREE, seed=12345)
        dna2 = PlantDNA.create_random(PlantType.TREE, seed=12345)
        
        assert dna1.height_gene.value == dna2.height_gene.value
        assert dna1.branch_count == dna2.branch_count
        assert dna1.leaf_color.rgb == dna2.leaf_color.rgb
    
    def test_mutation_changes_dna(self):
        """Test that mutation produces different DNA."""
        dna = PlantDNA.create_random(PlantType.TREE, seed=42)
        rng = np.random.default_rng(100)
        
        mutated = dna.mutate(rng, strength=0.5)
        
        # Should be different
        assert mutated.generation == dna.generation + 1
        # At least some values should change
        changes = (
            mutated.height_gene.value != dna.height_gene.value or
            mutated.branch_angle != dna.branch_angle or
            mutated.leaf_color.rgb != dna.leaf_color.rgb
        )
        assert changes
    
    def test_crossover_combines_parents(self):
        """Test that crossover combines traits from both parents."""
        parent1 = PlantDNA.create_random(PlantType.TREE, seed=1)
        parent2 = PlantDNA.create_random(PlantType.TREE, seed=2)
        
        rng = np.random.default_rng(42)
        child = parent1.crossover(parent2, rng)
        
        # Child should have incremented generation
        assert child.generation == max(parent1.generation, parent2.generation) + 1
        
        # Child values should be influenced by both parents
        # (statistically, over many crossovers, values should be between parents)
    
    def test_total_height_calculation(self):
        """Test total height is calculated from segments."""
        dna = PlantDNA()
        dna.height_gene = Gene(10.0, 1.0, 20.0)
        dna.trunk_segments = [
            SegmentGene(length=0.5),
            SegmentGene(length=0.5),
        ]
        
        # Total height = sum(segment lengths) * height_gene.value
        expected = 1.0 * 10.0  # 0.5 + 0.5 = 1.0, * 10 = 10
        assert dna.total_height == expected
    
    def test_to_dict_serialization(self):
        """Test DNA can be serialized to dict."""
        dna = PlantDNA.create_random(PlantType.ALIEN, seed=42)
        data = dna.to_dict()
        
        assert "plant_type" in data
        assert data["plant_type"] == PlantType.ALIEN
        assert "height" in data
        assert "trunk_color" in data
        assert len(data["trunk_color"]) == 3
    
    def test_to_json_serialization(self):
        """Test DNA can be serialized to JSON."""
        dna = PlantDNA.create_random(PlantType.TREE, seed=42)
        json_str = dna.to_json()
        
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["plant_type"] == PlantType.TREE


class TestDNAPool:
    """Tests for the DNAPool class."""
    
    def test_pool_creation(self):
        """Test DNA pool initialization."""
        pool = DNAPool(seed=42)
        
        assert pool.seed == 42
        assert len(pool.templates) == len(PlantType.ALL_TYPES)
    
    def test_chunk_dna_deterministic(self):
        """Test same chunk always gets same DNA."""
        pool1 = DNAPool(seed=42)
        pool2 = DNAPool(seed=42)
        
        dna1 = pool1.get_dna_for_chunk(5, 10)
        dna2 = pool2.get_dna_for_chunk(5, 10)
        
        # Should have same number of species
        assert len(dna1) == len(dna2)
        
        # Same types
        types1 = [d.plant_type for d in dna1]
        types2 = [d.plant_type for d in dna2]
        assert types1 == types2
    
    def test_chunk_dna_caching(self):
        """Test that chunk DNA is cached."""
        pool = DNAPool(seed=42)
        
        dna1 = pool.get_dna_for_chunk(0, 0)
        dna2 = pool.get_dna_for_chunk(0, 0)
        
        # Should be same object (cached)
        assert dna1 is dna2
    
    def test_neighbor_crossover(self):
        """Test that neighbors influence chunk DNA."""
        pool = DNAPool(seed=42)
        
        # Generate some neighbors first
        pool.get_dna_for_chunk(0, 0)
        pool.get_dna_for_chunk(2, 0)
        
        # Get center chunk (should be influenced by neighbors)
        center_dna = pool.get_dna_for_chunk(1, 0)
        
        assert len(center_dna) > 0
        # The DNA should exist and be valid
        for dna in center_dna:
            assert dna.height_gene.value > 0
    
    def test_cleanup_distant_chunks(self):
        """Test cleanup removes far chunks."""
        pool = DNAPool(seed=42)
        
        # Generate chunks
        pool.get_dna_for_chunk(0, 0)
        pool.get_dna_for_chunk(50, 50)
        
        assert (0, 0) in pool.chunk_dna
        assert (50, 50) in pool.chunk_dna
        
        # Cleanup around origin
        pool.cleanup_distant_chunks(0, 0, max_distance=10)
        
        # Far chunk should be removed
        assert (0, 0) in pool.chunk_dna
        assert (50, 50) not in pool.chunk_dna


class TestPlantTypeVariety:
    """Tests for different plant type configurations."""
    
    @pytest.mark.parametrize("plant_type", PlantType.ALL_TYPES)
    def test_plant_type_has_valid_config(self, plant_type):
        """Test each plant type produces valid DNA."""
        dna = PlantDNA.create_random(plant_type, seed=42)
        
        assert dna.plant_type == plant_type
        assert dna.height_gene.value >= dna.height_gene.min_val
        assert dna.height_gene.value <= dna.height_gene.max_val
        assert dna.width_gene.value > 0
        assert 0 <= dna.leaf_color.r <= 1
        assert 0 <= dna.leaf_color.g <= 1
        assert 0 <= dna.leaf_color.b <= 1
    
    def test_grass_is_short(self):
        """Test grass plants are short."""
        dna = PlantDNA.create_random(PlantType.GRASS, seed=42)
        assert dna.height_gene.value < 2.0
    
    def test_tall_tree_is_tall(self):
        """Test tall trees are actually tall."""
        dna = PlantDNA.create_random(PlantType.TALL_TREE, seed=42)
        assert dna.height_gene.value >= 8.0
    
    def test_cactus_has_no_leaves(self):
        """Test cactus has no leaf density."""
        dna = PlantDNA.create_random(PlantType.CACTUS, seed=42)
        assert dna.leaf_density == 0
    
    def test_mushroom_may_glow(self):
        """Test some mushrooms have glow (probabilistic)."""
        glow_count = 0
        for seed in range(100):
            dna = PlantDNA.create_random(PlantType.MUSHROOM, seed=seed)
            if dna.has_glow:
                glow_count += 1
        
        # Should have some glowing mushrooms (roughly 40%)
        assert 20 < glow_count < 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

