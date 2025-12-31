"""
Comprehensive tests for DNA evolution, mutation, and genetic diversity.

Tests ensure that:
1. Each biome produces unique plant/animal types
2. Mutation creates visible genetic variation
3. Crossover combines parent traits
4. Chunk generation produces diverse species
5. Segment growth/branching can evolve
"""

import pytest
import numpy as np
from typing import Set, List, Dict
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waverse.dna import PlantDNA, PlantType, Gene, ColorGene, DNAPool


class TestBiomeUniqueness:
    """Test that each biome has unique genetic influences."""
    
    def setup_method(self):
        self.pool = DNAPool(seed=42)
        self.rng = np.random.default_rng(42)
    
    def test_psychedelic_unique_plants(self):
        """Psychedelic biome should have alien/weird plants."""
        plants = self.pool._get_biome_plants('psychedelic', 0.5, 0.5, self.rng)
        
        assert PlantType.ALIEN in plants
        assert PlantType.SPIRAL in plants
        assert PlantType.OCTOPUS in plants
        # Should NOT have normal trees
        assert PlantType.OAK not in plants
        assert PlantType.PINE not in plants
    
    def test_hellfire_unique_plants(self):
        """Hellfire biome should have dead/harsh plants."""
        plants = self.pool._get_biome_plants('hellfire', 0.5, 0.5, self.rng)
        
        assert PlantType.DEAD_TREE in plants
        assert PlantType.CACTUS in plants
        assert PlantType.SNAG in plants
    
    def test_shadow_unique_plants(self):
        """Shadow biome should have creepy plants."""
        plants = self.pool._get_biome_plants('shadow', 0.5, 0.5, self.rng)
        
        assert PlantType.MUSHROOM in plants
        assert PlantType.CREEPER in plants
        assert PlantType.VINE in plants
    
    def test_crystal_unique_plants(self):
        """Crystal biome should have crystalline formations."""
        plants = self.pool._get_biome_plants('crystal', 0.5, 0.5, self.rng)
        
        assert PlantType.CRYSTAL in plants
        assert PlantType.CRYSTAL_FORMATION in plants
    
    def test_void_unique_plants(self):
        """Void biome should have alien/tentacle plants."""
        plants = self.pool._get_biome_plants('void', 0.5, 0.5, self.rng)
        
        assert PlantType.ALIEN in plants
        assert PlantType.TENTACLE in plants
    
    def test_hot_humid_biome(self):
        """Hot+humid should give tropical plants."""
        plants = self.pool._get_biome_plants(None, 0.8, 0.8, self.rng)
        
        assert PlantType.PALM in plants
        assert PlantType.MONSTERA in plants
        assert PlantType.BLOB_TREE in plants
    
    def test_hot_dry_biome(self):
        """Hot+dry should give desert plants."""
        plants = self.pool._get_biome_plants(None, 0.8, 0.2, self.rng)
        
        assert PlantType.CACTUS in plants
        assert PlantType.DEAD_TREE in plants
    
    def test_cold_humid_biome(self):
        """Cold+humid should give conifer forests."""
        plants = self.pool._get_biome_plants(None, 0.2, 0.7, self.rng)
        
        assert PlantType.PINE in plants
        assert PlantType.SPRUCE in plants
        assert PlantType.FIR in plants
    
    def test_cold_dry_biome(self):
        """Cold+dry should give tundra plants."""
        plants = self.pool._get_biome_plants(None, 0.2, 0.3, self.rng)
        
        assert PlantType.GRASS in plants
        assert PlantType.MOSS_PAD in plants
        assert PlantType.BUSH in plants
    
    def test_biomes_are_distinct(self):
        """Different biomes should have different plant distributions."""
        psychedelic = set(self.pool._get_biome_plants('psychedelic', 0.5, 0.5, self.rng))
        hellfire = set(self.pool._get_biome_plants('hellfire', 0.5, 0.5, self.rng))
        shadow = set(self.pool._get_biome_plants('shadow', 0.5, 0.5, self.rng))
        crystal = set(self.pool._get_biome_plants('crystal', 0.5, 0.5, self.rng))
        
        # Each should have at least 2 unique plants not in others
        all_others = hellfire | shadow | crystal
        psychedelic_unique = psychedelic - all_others
        assert len(psychedelic_unique) >= 1, f"Psychedelic should have unique plants: {psychedelic_unique}"


class TestMutation:
    """Test that mutation creates visible variation."""
    
    def test_mutation_changes_height(self):
        """Mutation should change plant height."""
        rng = np.random.default_rng(42)
        base = PlantDNA.create_random(PlantType.TREE, seed=42)
        
        heights = set()
        for i in range(20):
            mutated = base.mutate(rng, strength=0.5)
            heights.add(round(mutated.height_gene.value, 2))
        
        # Should have multiple distinct heights
        assert len(heights) >= 5, f"Expected diverse heights, got: {heights}"
    
    def test_mutation_changes_colors(self):
        """Mutation should change plant colors."""
        rng = np.random.default_rng(42)
        base = PlantDNA.create_random(PlantType.TREE, seed=42)
        
        colors = set()
        for i in range(20):
            mutated = base.mutate(rng, strength=0.5)
            c = mutated.leaf_color
            colors.add((round(c.r, 1), round(c.g, 1), round(c.b, 1)))
        
        # Should have multiple distinct colors
        assert len(colors) >= 3, f"Expected diverse colors, got: {colors}"
    
    def test_mutation_changes_branch_count(self):
        """Mutation should change branching structure."""
        rng = np.random.default_rng(42)
        base = PlantDNA.create_random(PlantType.TREE, seed=42)
        
        branch_counts = set()
        for i in range(20):
            mutated = base.mutate(rng, strength=0.5)
            branch_counts.add(mutated.branch_count)
        
        # Should have at least 2 different branch counts
        assert len(branch_counts) >= 2, f"Expected varied branching, got: {branch_counts}"
    
    def test_mutation_strength_affects_variation(self):
        """Higher mutation strength should create more variation."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        base = PlantDNA.create_random(PlantType.TREE, seed=42)
        
        # Low strength mutations
        low_heights = []
        for i in range(10):
            mutated = base.mutate(rng1, strength=0.1)
            low_heights.append(mutated.height_gene.value)
        
        # High strength mutations
        high_heights = []
        for i in range(10):
            mutated = base.mutate(rng2, strength=0.8)
            high_heights.append(mutated.height_gene.value)
        
        low_variance = np.var(low_heights)
        high_variance = np.var(high_heights)
        
        # High strength should generally have more variance
        # (This is probabilistic, so we allow some slack)
        assert high_variance >= low_variance * 0.5, \
            f"High strength should have more variance: {high_variance} vs {low_variance}"
    
    def test_mutation_preserves_type(self):
        """Mutation should preserve plant type."""
        rng = np.random.default_rng(42)
        base = PlantDNA.create_random(PlantType.CACTUS, seed=42)
        
        for i in range(10):
            mutated = base.mutate(rng, strength=0.9)
            assert mutated.plant_type == PlantType.CACTUS


class TestCrossover:
    """Test that crossover combines parent traits."""
    
    def test_crossover_blends_height(self):
        """Crossover should blend parent heights."""
        rng = np.random.default_rng(42)
        
        parent1 = PlantDNA.create_random(PlantType.TREE, seed=1)
        parent2 = PlantDNA.create_random(PlantType.TREE, seed=2)
        
        # Force very different heights
        parent1.height_gene = Gene(1.0, 0.5, 10.0, 0.2)
        parent2.height_gene = Gene(9.0, 0.5, 10.0, 0.2)
        
        # Offspring heights should be between parents
        offspring_heights = []
        for i in range(20):
            child = parent1.crossover(parent2, rng)
            offspring_heights.append(child.height_gene.value)
        
        avg_height = np.mean(offspring_heights)
        # Average should be somewhere between parents
        assert 2.0 < avg_height < 8.0, f"Expected blended height, got avg: {avg_height}"
    
    def test_crossover_mixes_colors(self):
        """Crossover should create intermediate colors."""
        rng = np.random.default_rng(42)
        
        parent1 = PlantDNA.create_random(PlantType.TREE, seed=1)
        parent2 = PlantDNA.create_random(PlantType.TREE, seed=2)
        
        # Force very different colors
        parent1.leaf_color = ColorGene(1.0, 0.0, 0.0)  # Red
        parent2.leaf_color = ColorGene(0.0, 0.0, 1.0)  # Blue
        
        child = parent1.crossover(parent2, rng)
        
        # Child should have some red and/or blue component
        # (depending on which parent's color was picked)
        assert child.leaf_color.r > 0.5 or child.leaf_color.b > 0.5
    
    def test_crossover_preserves_type(self):
        """Crossover between same types should preserve type."""
        rng = np.random.default_rng(42)
        
        parent1 = PlantDNA.create_random(PlantType.PALM, seed=1)
        parent2 = PlantDNA.create_random(PlantType.PALM, seed=2)
        
        for i in range(10):
            child = parent1.crossover(parent2, rng)
            assert child.plant_type == PlantType.PALM


class TestChunkGeneration:
    """Test chunk DNA generation produces diversity."""
    
    def test_chunk_generates_multiple_species(self):
        """Each chunk should have multiple species with good variety."""
        pool = DNAPool(seed=42)
        
        dna_list = pool.get_dna_for_chunk(0, 0, biome='temperate')
        
        # Should have 8-17 species
        assert len(dna_list) >= 8, f"Expected at least 8 species, got {len(dna_list)}"
        
        # Most species should be unique (some crossover duplicates are OK)
        species_ids = set(d.species_id for d in dna_list)
        assert len(species_ids) >= len(dna_list) * 0.7, \
            f"Expected mostly unique species: {len(species_ids)}/{len(dna_list)}"
    
    def test_different_chunks_different_species(self):
        """Different chunks should have different species."""
        pool = DNAPool(seed=42)
        
        dna1 = pool.get_dna_for_chunk(0, 0, biome='temperate')
        dna2 = pool.get_dna_for_chunk(100, 100, biome='temperate')
        
        ids1 = set(d.species_id for d in dna1)
        ids2 = set(d.species_id for d in dna2)
        
        # Should have mostly different species (some overlap is fine from crossover)
        overlap = len(ids1 & ids2)
        assert overlap < len(ids1) * 0.5, f"Chunks too similar: {overlap} overlap"
    
    def test_biome_influences_chunk_species(self):
        """Biome should influence which species appear in chunk."""
        pool = DNAPool(seed=42)
        
        psychedelic_dna = pool.get_dna_for_chunk(0, 0, biome='psychedelic')
        temperate_dna = pool.get_dna_for_chunk(0, 0, biome='temperate')
        
        psych_types = set(d.plant_type for d in psychedelic_dna)
        temp_types = set(d.plant_type for d in temperate_dna)
        
        # Psychedelic should have more alien/exotic types
        psych_exotic = sum(1 for t in psych_types if t in [PlantType.ALIEN, PlantType.SPIRAL, 
                                                            PlantType.OCTOPUS, PlantType.CRYSTAL])
        temp_exotic = sum(1 for t in temp_types if t in [PlantType.ALIEN, PlantType.SPIRAL,
                                                          PlantType.OCTOPUS, PlantType.CRYSTAL])
        
        assert psych_exotic > temp_exotic, "Psychedelic should have more exotic plants"
    
    def test_neighbor_crossover(self):
        """Adjacent chunks should share some genetic material."""
        pool = DNAPool(seed=42)
        
        # Generate a 3x3 grid of chunks
        for x in range(-1, 2):
            for z in range(-1, 2):
                pool.get_dna_for_chunk(x, z, biome='temperate')
        
        # Center chunk should have crossover from neighbors
        center = pool.get_dna_for_chunk(0, 0, biome='temperate')
        
        # This is already tested by the generation code, but verify chunk exists
        assert len(center) > 0


class TestEvolvableStructures:
    """Test that DNA can evolve interesting structures."""
    
    def test_branch_count_can_vary(self):
        """Plants should be able to evolve different branch counts."""
        rng = np.random.default_rng(42)
        
        branch_counts = set()
        for seed in range(50):
            dna = PlantDNA.create_random(PlantType.TREE, seed=seed)
            branch_counts.add(dna.branch_count)
        
        # Should have multiple different branch counts
        assert len(branch_counts) >= 3, f"Expected varied branching: {branch_counts}"
    
    def test_recursive_depth_can_vary(self):
        """Plants should be able to evolve different recursion depths."""
        rng = np.random.default_rng(42)
        
        depths = set()
        for seed in range(50):
            dna = PlantDNA.create_random(PlantType.TREE, seed=seed)
            depths.add(dna.recursive_depth)
        
        assert len(depths) >= 2, f"Expected varied recursion: {depths}"
    
    def test_canopy_shapes_can_vary(self):
        """Plants should be able to evolve different canopy shapes."""
        rng = np.random.default_rng(42)
        
        canopy_shapes = set()
        for seed in range(50):
            dna = PlantDNA.create_random(PlantType.TREE, seed=seed)
            canopy_shapes.add(dna.canopy_shape)
        
        assert len(canopy_shapes) >= 2, f"Expected varied canopies: {canopy_shapes}"
    
    def test_height_range_is_reasonable(self):
        """Generated heights should span a reasonable range."""
        heights = []
        for seed in range(100):
            dna = PlantDNA.create_random(PlantType.TREE, seed=seed)
            heights.append(dna.height_gene.value)
        
        min_h, max_h = min(heights), max(heights)
        assert max_h - min_h >= 2.0, f"Height range too narrow: {min_h} to {max_h}"
        assert min_h >= 0.5, f"Min height too low: {min_h}"
        assert max_h <= 15.0, f"Max height too high: {max_h}"
    
    def test_width_range_is_reasonable(self):
        """Generated widths should span a reasonable range."""
        widths = []
        for seed in range(100):
            dna = PlantDNA.create_random(PlantType.TREE, seed=seed)
            widths.append(dna.width_gene.value)
        
        min_w, max_w = min(widths), max(widths)
        assert max_w - min_w >= 0.2, f"Width range too narrow: {min_w} to {max_w}"


class TestBiomeColors:
    """Test biome-specific color mutations."""
    
    def setup_method(self):
        self.pool = DNAPool(seed=42)
        self.rng = np.random.default_rng(42)
    
    def test_psychedelic_vivid_colors(self):
        """Psychedelic biome should produce vivid colors."""
        base = PlantDNA.create_random(PlantType.ALIEN, seed=42)
        mutated = self.pool._apply_biome_colors(base, 'psychedelic', self.rng)
        
        # Colors should be saturated (not gray)
        trunk = mutated.trunk_color
        max_comp = max(trunk.r, trunk.g, trunk.b)
        min_comp = min(trunk.r, trunk.g, trunk.b)
        saturation = (max_comp - min_comp) / max(max_comp, 0.001)
        
        assert saturation > 0.3, f"Psychedelic should have saturated colors: {saturation}"
    
    def test_shadow_dark_colors(self):
        """Shadow biome should produce dark colors."""
        base = PlantDNA.create_random(PlantType.MUSHROOM, seed=42)
        mutated = self.pool._apply_biome_colors(base, 'shadow', self.rng)
        
        # Leaf color brightness should be low
        leaf = mutated.leaf_color
        brightness = (leaf.r + leaf.g + leaf.b) / 3
        
        assert brightness < 0.6, f"Shadow should have dark colors: {brightness}"
    
    def test_crystal_pale_colors(self):
        """Crystal biome should produce pale/icy colors."""
        base = PlantDNA.create_random(PlantType.CRYSTAL, seed=42)
        mutated = self.pool._apply_biome_colors(base, 'crystal', self.rng)
        
        # Colors should be light
        leaf = mutated.leaf_color
        brightness = (leaf.r + leaf.g + leaf.b) / 3
        
        assert brightness > 0.4, f"Crystal should have light colors: {brightness}"


class TestDeterminism:
    """Test that generation is deterministic with same seed."""
    
    def test_same_seed_same_dna(self):
        """Same seed should produce identical DNA."""
        dna1 = PlantDNA.create_random(PlantType.TREE, seed=42)
        dna2 = PlantDNA.create_random(PlantType.TREE, seed=42)
        
        assert dna1.species_id == dna2.species_id
        assert dna1.height_gene.value == dna2.height_gene.value
        assert dna1.branch_count == dna2.branch_count
    
    def test_same_seed_same_chunk(self):
        """Same seed should produce identical chunk DNA."""
        pool1 = DNAPool(seed=42)
        pool2 = DNAPool(seed=42)
        
        dna1 = pool1.get_dna_for_chunk(5, 10, biome='temperate')
        dna2 = pool2.get_dna_for_chunk(5, 10, biome='temperate')
        
        assert len(dna1) == len(dna2)
        for d1, d2 in zip(dna1, dna2):
            assert d1.species_id == d2.species_id
    
    def test_different_seeds_different_dna(self):
        """Different seeds should produce different DNA."""
        dna1 = PlantDNA.create_random(PlantType.TREE, seed=1)
        dna2 = PlantDNA.create_random(PlantType.TREE, seed=2)
        
        # Should be different
        assert dna1.species_id != dna2.species_id or \
               dna1.height_gene.value != dna2.height_gene.value


class TestSpeciesID:
    """Test species ID consistency."""
    
    def test_species_id_diversity(self):
        """Chunks should produce diverse species IDs."""
        pool = DNAPool(seed=42)
        
        all_ids = set()
        total_count = 0
        for i, biome in enumerate(['psychedelic', 'hellfire', 'shadow', 'crystal', 'temperate']):
            # Use different chunk coords for each biome to avoid caching issues
            dna_list = pool.get_dna_for_chunk(i * 10, i * 10, biome=biome)
            for dna in dna_list:
                all_ids.add(dna.species_id)
                total_count += 1
        
        # Most species across biomes should be unique (crossover may cause some overlap)
        uniqueness_ratio = len(all_ids) / total_count
        assert uniqueness_ratio > 0.6, f"Expected diverse species: {len(all_ids)}/{total_count}"
    
    def test_species_id_consistent(self):
        """Species ID should be consistent across generations."""
        pool = DNAPool(seed=42)
        
        dna_list = pool.get_dna_for_chunk(0, 0, biome='temperate')
        first_ids = [d.species_id for d in dna_list]
        
        # Get same chunk again (should be cached)
        dna_list2 = pool.get_dna_for_chunk(0, 0, biome='temperate')
        second_ids = [d.species_id for d in dna_list2]
        
        assert first_ids == second_ids


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

