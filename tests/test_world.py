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
        """Test sine wave contribution."""
        waves = [WaveConfig("sin", freq=0.1, amp=10, phase=0)]
        config = WorldConfig(name="Test", seed=42, waves=waves)
        manager = ChunkManager(config)
        
        chunk = manager.get_chunk(0, 0)
        
        # Sine wave should produce heights within reasonable bounds
        assert np.min(chunk.heightmap) >= -20  # Some margin
        assert np.max(chunk.heightmap) <= 20
    
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

