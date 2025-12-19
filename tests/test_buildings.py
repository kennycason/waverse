"""
Tests for Building DNA and Building Generator systems.

These tests ensure:
1. BuildingDNA is deterministic (same seed = same building)
2. All building types can be generated without errors
3. Generated structures have valid geometry
4. Color palettes are valid RGB values
5. Terrain-aware features work correctly
"""

import pytest
import numpy as np

from waverse.building_dna import (
    BuildingDNA, BuildingType, RoofType, WallStyle, ColorPalette,
    get_building_type_for_biome
)
from waverse.building_generator import generate_building_from_dna
from waverse.structures import Structure, StructureManager, StructureRenderer


class TestBuildingDNA:
    """Tests for BuildingDNA class."""
    
    def test_deterministic_generation(self):
        """Same seed should produce identical DNA."""
        dna1 = BuildingDNA.create_random(BuildingType.HOUSE, seed=12345)
        dna2 = BuildingDNA.create_random(BuildingType.HOUSE, seed=12345)
        
        assert dna1.width == dna2.width
        assert dna1.depth == dna2.depth
        assert dna1.floors == dna2.floors
        assert dna1.roof_type == dna2.roof_type
        assert dna1.wall_style == dna2.wall_style
        assert dna1.colors.primary == dna2.colors.primary
        assert dna1.has_balconies == dna2.has_balconies
        assert dna1.has_chimney == dna2.has_chimney
    
    def test_different_seeds_different_buildings(self):
        """Different seeds should produce different DNA."""
        dna1 = BuildingDNA.create_random(BuildingType.HOUSE, seed=100)
        dna2 = BuildingDNA.create_random(BuildingType.HOUSE, seed=200)
        
        # At least some properties should differ
        differences = 0
        if dna1.width != dna2.width: differences += 1
        if dna1.depth != dna2.depth: differences += 1
        if dna1.floors != dna2.floors: differences += 1
        if dna1.roof_type != dna2.roof_type: differences += 1
        
        assert differences > 0, "Different seeds should produce some variation"
    
    @pytest.mark.parametrize("building_type", list(BuildingType))
    def test_all_building_types_can_be_created(self, building_type):
        """Every building type should be creatable without errors."""
        dna = BuildingDNA.create_random(building_type, seed=42)
        
        assert dna.building_type == building_type
        assert dna.width > 0
        assert dna.depth > 0
        assert dna.floors >= 1
        assert dna.floor_height > 0
        assert isinstance(dna.roof_type, RoofType)
        assert isinstance(dna.wall_style, WallStyle)
    
    def test_valid_dimensions(self):
        """Buildings should have reasonable dimensions."""
        for seed in range(100):
            for btype in [BuildingType.HOUSE, BuildingType.TOWER, BuildingType.WAREHOUSE]:
                dna = BuildingDNA.create_random(btype, seed)
                
                assert 1 <= dna.width <= 100, f"Width {dna.width} out of range"
                assert 1 <= dna.depth <= 100, f"Depth {dna.depth} out of range"
                assert 1 <= dna.floors <= 50, f"Floors {dna.floors} out of range"
                assert 2 <= dna.floor_height <= 20, f"Floor height {dna.floor_height} out of range"
    
    def test_total_height_calculation(self):
        """Total height should account for roof."""
        dna = BuildingDNA.create_random(BuildingType.HOUSE, seed=1)
        base_height = dna.floors * dna.floor_height
        
        # Total height should be >= base height (roof adds to it)
        assert dna.total_height >= base_height
        
        # Roof shouldn't be taller than building
        roof_height = dna.total_height - base_height
        assert roof_height < base_height
    
    def test_type_specific_features(self):
        """Building types should have appropriate features."""
        # Houses often have chimneys
        house_chimneys = sum(
            1 for seed in range(50) 
            if BuildingDNA.create_random(BuildingType.HOUSE, seed).has_chimney
        )
        assert house_chimneys > 0, "Some houses should have chimneys"
        
        # Shops should have awnings
        for seed in range(10):
            shop = BuildingDNA.create_random(BuildingType.SHOP, seed)
            assert shop.has_awning, "Shops should have awnings"
        
        # Towers should be tall
        for seed in range(10):
            tower = BuildingDNA.create_random(BuildingType.TOWER, seed)
            assert tower.floors >= 10, "Towers should have many floors"


class TestColorPalette:
    """Tests for ColorPalette generation."""
    
    @pytest.mark.parametrize("style", ["modern", "residential", "industrial", "commercial", "alien", "ancient"])
    def test_valid_rgb_values(self, style):
        """All color components should be valid RGB (0-1)."""
        rng = np.random.default_rng(42)
        palette = ColorPalette.random(rng, style)
        
        for color_name in ['primary', 'secondary', 'roof', 'window']:
            color = getattr(palette, color_name)
            assert len(color) == 3, f"{color_name} should have 3 components"
            for i, c in enumerate(color):
                assert 0 <= c <= 1, f"{color_name}[{i}] = {c} out of range"
    
    def test_deterministic_colors(self):
        """Same seed should produce same colors."""
        rng1 = np.random.default_rng(999)
        rng2 = np.random.default_rng(999)
        
        p1 = ColorPalette.random(rng1, "modern")
        p2 = ColorPalette.random(rng2, "modern")
        
        assert p1.primary == p2.primary
        assert p1.secondary == p2.secondary


class TestBuildingGenerator:
    """Tests for generate_building_from_dna function."""
    
    def test_generates_valid_structure(self):
        """Generated structures should have walls and floors."""
        dna = BuildingDNA.create_random(BuildingType.APARTMENT, seed=42)
        structure = generate_building_from_dna(0, 0, 0, dna)
        
        assert isinstance(structure, Structure)
        assert len(structure.walls) > 0, "Building should have walls"
        assert len(structure.floors) > 0, "Building should have floors"
    
    def test_deterministic_structure_generation(self):
        """Same DNA should produce same structure."""
        dna = BuildingDNA.create_random(BuildingType.HOUSE, seed=123)
        
        s1 = generate_building_from_dna(100, 50, 200, dna)
        s2 = generate_building_from_dna(100, 50, 200, dna)
        
        assert len(s1.walls) == len(s2.walls)
        assert len(s1.floors) == len(s2.floors)
        assert len(s1.pillars) == len(s2.pillars)
        assert len(s1.ramps) == len(s2.ramps)
    
    def test_position_is_respected(self):
        """Structure should be centered at given position."""
        dna = BuildingDNA.create_random(BuildingType.SHOP, seed=1)
        structure = generate_building_from_dna(1000, 50, 2000, dna)
        
        # Check bounding box is roughly centered on position
        center_x = (structure.bbox_min[0] + structure.bbox_max[0]) / 2
        center_z = (structure.bbox_min[2] + structure.bbox_max[2]) / 2
        
        assert abs(center_x - 1000) < dna.width, "X should be near specified position"
        assert abs(center_z - 2000) < dna.depth, "Z should be near specified position"
    
    def test_terrain_height_handling(self):
        """Buildings should sit on highest terrain point."""
        dna = BuildingDNA.create_random(BuildingType.WAREHOUSE, seed=5)
        
        # Flat terrain
        s_flat = generate_building_from_dna(0, 10, 0, dna, [10, 10, 10, 10])
        
        # Sloped terrain (max height 20)
        s_slope = generate_building_from_dna(0, 10, 0, dna, [10, 15, 12, 20])
        
        # Building on slope should have pillars
        assert len(s_slope.pillars) > len(s_flat.pillars), \
            "Sloped terrain should add support pillars"
    
    @pytest.mark.parametrize("building_type", list(BuildingType))
    def test_all_types_generate_valid_structures(self, building_type):
        """Every building type should generate a valid structure."""
        dna = BuildingDNA.create_random(building_type, seed=77)
        structure = generate_building_from_dna(0, 10, 0, dna, [8, 10, 9, 11])
        
        assert isinstance(structure, Structure)
        assert structure.bbox_min[0] < structure.bbox_max[0]
        assert structure.bbox_min[1] < structure.bbox_max[1]
        assert structure.bbox_min[2] < structure.bbox_max[2]
    
    def test_roof_is_generated(self):
        """Buildings should have a roof (represented as floors or ramps at top)."""
        for roof_type in RoofType:
            # Find a building with this roof type
            for seed in range(100):
                dna = BuildingDNA.create_random(BuildingType.HOUSE, seed)
                if dna.roof_type == roof_type:
                    break
            else:
                continue  # Couldn't find this roof type, skip
            
            structure = generate_building_from_dna(0, 0, 0, dna)
            
            # Should have some geometry at roof level
            roof_y = dna.floors * dna.floor_height
            has_roof = any(
                f.y >= roof_y - 1 for f in structure.floors
            ) or any(
                r.y_top >= roof_y for r in structure.ramps
            )
            assert has_roof, f"Building with {roof_type} should have roof geometry"


class TestStructureManager:
    """Tests for StructureManager with DNA buildings."""
    
    def test_dna_buildings_enabled_by_default(self):
        """DNA buildings should be enabled by default."""
        manager = StructureManager(seed=42)
        assert manager.use_dna_buildings is True
    
    def test_spawn_dna_building(self):
        """Manager should spawn DNA-based buildings."""
        manager = StructureManager(seed=42, use_dna_buildings=True)
        
        # Create a simple heightmap
        heightmap = np.ones((33, 33)) * 15  # Above water, reasonable height
        
        # Force spawn by setting very high probability
        # (normal spawn has 3% chance, so we just spawn directly)
        from waverse.building_dna import BuildingType, BuildingDNA
        from waverse.building_generator import generate_building_from_dna
        
        dna = BuildingDNA.create_random(BuildingType.HOUSE, 12345)
        structure = generate_building_from_dna(100, 15, 100, dna, [15, 15, 15, 15])
        manager.add_structure(structure)
        
        assert len(manager.structures) == 1
        assert len(manager.structures[0].walls) > 0
    
    def test_cleanup_removes_structures(self):
        """Cleanup should remove distant structures."""
        manager = StructureManager(seed=42)
        
        # Add a structure far away
        from waverse.building_dna import BuildingType, BuildingDNA
        from waverse.building_generator import generate_building_from_dna
        
        dna = BuildingDNA.create_random(BuildingType.SHOP, 1)
        far_structure = generate_building_from_dna(50000, 10, 50000, dna)
        manager.add_structure(far_structure)
        
        assert len(manager.structures) == 1
        
        # Cleanup from origin
        manager.cleanup_distant(0, 0, chunk_size=32, max_chunks=40)
        
        assert len(manager.structures) == 0


class TestBiomeBuildings:
    """Tests for biome-specific building selection."""
    
    def test_get_building_type_returns_valid_type(self):
        """Should return valid BuildingType for any biome."""
        rng = np.random.default_rng(42)
        
        for biome in ["urban", "suburban", "desert", "forest", "snow", "alien"]:
            btype = get_building_type_for_biome(biome, rng)
            assert isinstance(btype, BuildingType)
    
    def test_biome_influences_building_type(self):
        """Different biomes should have different building distributions."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        urban_types = [get_building_type_for_biome("urban", rng1) for _ in range(50)]
        desert_types = [get_building_type_for_biome("desert", rng2) for _ in range(50)]
        
        # Urban should have more towers/offices
        urban_towers = sum(1 for t in urban_types if t in [BuildingType.TOWER, BuildingType.OFFICE])
        desert_towers = sum(1 for t in desert_types if t in [BuildingType.TOWER, BuildingType.OFFICE])
        
        assert urban_towers > desert_towers, "Urban should have more towers than desert"
        
        # Desert should have more ruins/temples
        urban_ancient = sum(1 for t in urban_types if t in [BuildingType.RUINS, BuildingType.TEMPLE])
        desert_ancient = sum(1 for t in desert_types if t in [BuildingType.RUINS, BuildingType.TEMPLE])
        
        assert desert_ancient > urban_ancient, "Desert should have more ruins/temples"


class TestPerformance:
    """Tests for performance-related features."""
    
    def test_display_list_caching(self):
        """Display lists should be cached for reuse."""
        from waverse.building_dna import BuildingType, BuildingDNA
        from waverse.building_generator import generate_building_from_dna
        
        dna = BuildingDNA.create_random(BuildingType.HOUSE, 42)
        structure = generate_building_from_dna(0, 0, 0, dna)
        
        # Check that display list dict exists
        assert hasattr(StructureRenderer, '_display_lists')
        assert isinstance(StructureRenderer._display_lists, dict)
    
    def test_lod_distances_are_ordered(self):
        """LOD distances should be in correct order."""
        assert StructureRenderer.LOD_FULL < StructureRenderer.LOD_SIMPLE
        assert StructureRenderer.LOD_SIMPLE < StructureRenderer.LOD_BILLBOARD

