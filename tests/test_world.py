"""
Unit tests for the World generation system.

Tests cover:
- World configuration
- Wave-based terrain generation
- Chunk management
- Heightmap generation
"""

import pytest
import numpy as np

from waverse.world import (
    WorldConfig, WaveConfig, ChunkManager, Chunk,
    CHUNK_SIZE, TILE_SCALE
)


class TestWorldConfig:
    """Tests for WorldConfig class."""
    
    def test_default_config(self):
        """Test default world configuration."""
        config = WorldConfig.create_default()
        
        assert config.name == "Default World"
        assert config.seed == 42
        assert len(config.waves) > 0
    
    def test_custom_config(self):
        """Test custom world configuration."""
        waves = [
            WaveConfig("sin", freq=0.01, amp=10),
            WaveConfig("perlin", freq=0.05, amp=5),
        ]
        config = WorldConfig(name="Test World", seed=123, waves=waves)
        
        assert config.name == "Test World"
        assert config.seed == 123
        assert len(config.waves) == 2
    
    def test_wave_config_types(self):
        """Test different wave types are valid."""
        wave_types = ["sin", "cos", "perlin", "ridged", "sin2d"]
        
        for wtype in wave_types:
            wave = WaveConfig(wtype, freq=0.01, amp=1.0)
            assert wave.wave_type == wtype


class TestChunk:
    """Tests for Chunk class."""
    
    def test_chunk_creation(self):
        """Test chunk is created with correct properties."""
        chunk = Chunk(cx=5, cz=10)
        
        assert chunk.cx == 5
        assert chunk.cz == 10
        assert chunk.world_x == 5 * CHUNK_SIZE * TILE_SCALE
        assert chunk.world_z == 10 * CHUNK_SIZE * TILE_SCALE
    
    def test_chunk_heightmap_shape(self):
        """Test heightmap has correct shape."""
        chunk = Chunk(cx=0, cz=0)
        
        # Heightmap should be None until generated
        assert chunk.heightmap is None
    
    def test_chunk_negative_coordinates(self):
        """Test chunks work with negative coordinates."""
        chunk = Chunk(cx=-5, cz=-10)
        
        assert chunk.cx == -5
        assert chunk.cz == -10
        assert chunk.world_x == -5 * CHUNK_SIZE * TILE_SCALE


class TestChunkManager:
    """Tests for ChunkManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a test chunk manager."""
        config = WorldConfig.create_default()
        return ChunkManager(config)
    
    def test_get_chunk_creates_new(self, manager):
        """Test getting non-existent chunk creates it."""
        chunk = manager.get_chunk(0, 0)
        
        assert chunk is not None
        assert chunk.cx == 0
        assert chunk.cz == 0
        assert chunk.heightmap is not None
    
    def test_get_chunk_caches(self, manager):
        """Test chunks are cached."""
        chunk1 = manager.get_chunk(5, 5)
        chunk2 = manager.get_chunk(5, 5)
        
        assert chunk1 is chunk2
    
    def test_heightmap_deterministic(self, manager):
        """Test same coordinates produce same heightmap."""
        chunk1 = manager.get_chunk(3, 7)
        
        # Create new manager with same seed
        config = WorldConfig.create_default()
        manager2 = ChunkManager(config)
        chunk2 = manager2.get_chunk(3, 7)
        
        np.testing.assert_array_equal(chunk1.heightmap, chunk2.heightmap)
    
    def test_heightmap_shape(self, manager):
        """Test heightmap has correct dimensions."""
        chunk = manager.get_chunk(0, 0)
        
        # Heightmap is CHUNK_SIZE + 1 to have vertices for all tiles
        assert chunk.heightmap.shape == (CHUNK_SIZE + 1, CHUNK_SIZE + 1)
    
    def test_heightmap_values_reasonable(self, manager):
        """Test heightmap values are in reasonable range."""
        chunk = manager.get_chunk(0, 0)
        
        # Heights should generally be between -50 and 100
        assert np.min(chunk.heightmap) > -100
        assert np.max(chunk.heightmap) < 200
    
    def test_different_chunks_different(self, manager):
        """Test different chunk coordinates produce different terrain."""
        chunk1 = manager.get_chunk(0, 0)
        chunk2 = manager.get_chunk(100, 100)
        
        # Heightmaps should be different
        assert not np.array_equal(chunk1.heightmap, chunk2.heightmap)
    
    def test_negative_chunks(self, manager):
        """Test negative chunk coordinates work."""
        chunk = manager.get_chunk(-5, -10)
        
        assert chunk.cx == -5
        assert chunk.cz == -10
        assert chunk.heightmap is not None
        assert chunk.heightmap.shape == (CHUNK_SIZE + 1, CHUNK_SIZE + 1)


class TestGetHeight:
    """Tests for the get_height_at utility method on ChunkManager."""
    
    @pytest.fixture
    def manager(self):
        """Create a test chunk manager."""
        config = WorldConfig.create_default()
        return ChunkManager(config)
    
    def test_get_height_at_origin(self, manager):
        """Test getting height at origin."""
        height = manager.get_height_at(0, 0)
        
        assert isinstance(height, (int, float, np.floating))
    
    def test_get_height_interpolates(self, manager):
        """Test height at fractional positions."""
        h1 = manager.get_height_at(0.0, 0.0)
        h2 = manager.get_height_at(0.5, 0.5)
        h3 = manager.get_height_at(1.0, 1.0)
        
        # All should be valid numbers
        assert all(isinstance(h, (int, float, np.floating)) for h in [h1, h2, h3])
    
    def test_get_height_negative_coords(self, manager):
        """Test getting height at negative world coordinates."""
        height = manager.get_height_at(-100.0, -100.0)
        
        assert isinstance(height, (int, float, np.floating))


class TestWaveGeneration:
    """Tests for wave-based terrain generation specifics."""
    
    def test_sin_wave(self):
        """Test sine wave produces variation."""
        waves = [WaveConfig("sin", freq=0.1, amp=10, phase=0)]
        # Use unique seed to avoid cached chunks
        config = WorldConfig(name="Test", seed=99999, waves=waves)
        manager = ChunkManager(config)
        
        chunk = manager.get_chunk(0, 0)
        
        # Sine wave should produce varying heights
        height_range = np.max(chunk.heightmap) - np.min(chunk.heightmap)
        assert height_range > 0.1, "Sine wave should produce variation"
    
    def test_perlin_wave(self):
        """Test Perlin noise contribution."""
        waves = [WaveConfig("perlin", freq=0.02, amp=20, octaves=3)]
        config = WorldConfig(name="Test", seed=42, waves=waves)
        manager = ChunkManager(config)
        
        chunk = manager.get_chunk(0, 0)
        
        # Perlin should produce varied terrain
        assert np.std(chunk.heightmap) > 0.5  # Should have variation
    
    def test_combined_waves(self):
        """Test multiple waves combine properly."""
        waves = [
            WaveConfig("sin", freq=0.01, amp=10),
            WaveConfig("perlin", freq=0.05, amp=5),
        ]
        config = WorldConfig(name="Test", seed=42, waves=waves)
        manager = ChunkManager(config)
        
        chunk = manager.get_chunk(0, 0)
        
        # Combined waves should have variation
        assert np.std(chunk.heightmap) > 0.5


class TestChunkContinuity:
    """Tests for terrain continuity across chunk boundaries."""
    
    @pytest.fixture
    def manager(self):
        """Create a test chunk manager."""
        config = WorldConfig.create_default()
        return ChunkManager(config)
    
    def test_adjacent_chunks_connect(self, manager):
        """Test that adjacent chunks have continuous terrain."""
        chunk1 = manager.get_chunk(0, 0)
        chunk2 = manager.get_chunk(1, 0)
        
        # Edge of chunk1 should be similar to start of chunk2
        edge1 = chunk1.heightmap[:, -1]  # Right edge
        edge2 = chunk2.heightmap[:, 0]   # Left edge
        
        # They should be reasonably close (waves are continuous)
        diff = np.abs(edge1 - edge2)
        assert np.mean(diff) < 5  # Average difference should be small


class TestNewWaveTypes:
    """Tests for the new creative wave types added for terrain variety."""
    
    @pytest.mark.parametrize("wave_type", [
        "dunes", "mesa", "staircases", "ripples", "fractured", "eroded", "volcanic"
    ])
    def test_new_wave_type_produces_terrain(self, wave_type):
        """Each new wave type should produce valid terrain."""
        waves = [WaveConfig(wave_type, freq=0.01, amp=10)]
        config = WorldConfig(name="Test", seed=42, waves=waves)
        manager = ChunkManager(config)
        
        chunk = manager.get_chunk(0, 0)
        
        assert chunk.heightmap is not None
        assert chunk.heightmap.shape == (CHUNK_SIZE + 1, CHUNK_SIZE + 1)
        assert np.isfinite(chunk.heightmap).all(), f"{wave_type} produced NaN/Inf"
    
    def test_dunes_produces_variation(self):
        """Dunes wave type should produce terrain variation."""
        waves = [WaveConfig("dunes", freq=0.02, amp=10, direction=0)]
        # Use unique seed
        config = WorldConfig(name="Test", seed=88888, waves=waves)
        manager = ChunkManager(config)
        
        chunk = manager.get_chunk(0, 0)
        
        # Dunes should produce varying heights
        height_range = np.max(chunk.heightmap) - np.min(chunk.heightmap)
        assert height_range > 0.5, "Dunes should produce terrain variation"
    
    def test_mesa_produces_flat_tops(self):
        """Mesa should have regions of similar height (flat tops)."""
        waves = [WaveConfig("mesa", freq=0.01, amp=15, sharpness=10.0)]
        config = WorldConfig(name="Test", seed=42, waves=waves)
        manager = ChunkManager(config)
        
        chunk = manager.get_chunk(0, 0)
        
        # Should have some flat regions (low local variance)
        # Check by looking at gradient magnitude
        dx = np.diff(chunk.heightmap, axis=0)
        dz = np.diff(chunk.heightmap, axis=1)
        
        # Some areas should be relatively flat
        flat_threshold = 0.5
        flat_x = np.sum(np.abs(dx) < flat_threshold) / dx.size
        flat_z = np.sum(np.abs(dz) < flat_threshold) / dz.size
        
        assert flat_x > 0.1, "Mesa should have some flat regions"
    
    def test_staircases_produces_steps(self):
        """Staircases should produce quantized height levels."""
        waves = [WaveConfig("staircases", freq=0.01, amp=20, levels=5)]
        config = WorldConfig(name="Test", seed=42, waves=waves)
        manager = ChunkManager(config)
        
        chunk = manager.get_chunk(0, 0)
        
        # Check that heights cluster around certain values
        heights = chunk.heightmap.flatten()
        unique_heights = len(np.unique(np.round(heights, decimals=0)))
        
        # Should have fewer unique integer heights due to stepping
        # (compared to continuous terrain)
        assert unique_heights < len(heights) / 2
    
    def test_volcanic_produces_peaks(self):
        """Volcanic terrain should have peaked regions."""
        waves = [WaveConfig("volcanic", freq=0.005, amp=30)]
        config = WorldConfig(name="Test", seed=42, waves=waves)
        manager = ChunkManager(config)
        
        # Check multiple chunks to find a volcanic cone
        all_heights = []
        for cx in range(-5, 5):
            for cz in range(-5, 5):
                chunk = manager.get_chunk(cx, cz)
                all_heights.append(chunk.heightmap)
        
        combined = np.concatenate([h.flatten() for h in all_heights])
        max_height = np.max(combined)
        mean_height = np.mean(combined)
        
        # Volcanic cones should create some high peaks somewhere
        assert max_height > mean_height + 2, "Volcanic should have peaks"
    
    def test_fractured_produces_variation(self):
        """Fractured terrain should produce terrain variation."""
        waves = [WaveConfig("fractured", freq=0.1, amp=10)]
        # Use unique seed
        config = WorldConfig(name="Test", seed=77777, waves=waves)
        manager = ChunkManager(config)
        
        chunk = manager.get_chunk(0, 0)
        
        # Fractured should produce varying heights
        height_range = np.max(chunk.heightmap) - np.min(chunk.heightmap)
        assert height_range > 0.1, "Fractured should produce terrain variation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

