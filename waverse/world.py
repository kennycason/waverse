"""
Simplified World Generation - Heightmap-based with stacked waves.

This is a streamlined version that just works, similar to the original
terrain explorer but with wave-based DNA generation.
"""

import numpy as np
import math
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field

# Try to import noise library
try:
    import noise
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False


# =============================================================================
# Wave Functions - Stack these for complex terrain
# =============================================================================

def wave_sin(x: np.ndarray, z: np.ndarray, freq: float, amp: float, 
             phase: float = 0, direction: float = 0) -> np.ndarray:
    """Sine wave."""
    if direction != 0:
        cos_d, sin_d = np.cos(direction), np.sin(direction)
        t = (x * cos_d + z * sin_d) * freq
    else:
        t = (x + z) * freq
    return np.sin(t * 2 * np.pi + phase) * amp


def wave_cos(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
             phase: float = 0, direction: float = 0) -> np.ndarray:
    """Cosine wave."""
    if direction != 0:
        cos_d, sin_d = np.cos(direction), np.sin(direction)
        t = (x * cos_d + z * sin_d) * freq
    else:
        t = (x + z) * freq
    return np.cos(t * 2 * np.pi + phase) * amp


def wave_sin2d(x: np.ndarray, z: np.ndarray, freq_x: float, freq_z: float,
               amp: float, phase: float = 0) -> np.ndarray:
    """2D sine - independent frequencies for x and z."""
    return (np.sin(x * freq_x * 2 * np.pi + phase) * 
            np.sin(z * freq_z * 2 * np.pi) * amp)


def wave_radial(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                cx: float = 0, cz: float = 0) -> np.ndarray:
    """Radial wave emanating from center point."""
    dist = np.sqrt((x - cx)**2 + (z - cz)**2)
    return np.sin(dist * freq * 2 * np.pi) * amp


def _smooth_noise(x: np.ndarray, z: np.ndarray, seed: int = 0) -> np.ndarray:
    """Generate smooth interpolated value noise."""
    # Grid cell coordinates
    x0 = np.floor(x).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    x1 = x0 + 1
    z1 = z0 + 1
    
    # Interpolation weights (smoothstep for smooth results)
    sx = x - x0
    sz = z - z0
    sx = sx * sx * (3 - 2 * sx)  # Smoothstep
    sz = sz * sz * (3 - 2 * sz)
    
    # Hash function for random values at grid points
    def hash_2d(ix, iz):
        n = ix * 374761393 + iz * 668265263 + seed
        n = (n ^ (n >> 13)) * 1274126177
        n = n ^ (n >> 16)
        return (n & 0x7fffffff) / 0x7fffffff * 2 - 1  # -1 to 1
    
    # Get values at 4 corners
    v00 = hash_2d(x0, z0)
    v10 = hash_2d(x1, z0)
    v01 = hash_2d(x0, z1)
    v11 = hash_2d(x1, z1)
    
    # Bilinear interpolation
    v0 = v00 * (1 - sx) + v10 * sx
    v1 = v01 * (1 - sx) + v11 * sx
    return v0 * (1 - sz) + v1 * sz


def wave_perlin(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                octaves: int = 4, seed: int = 0) -> np.ndarray:
    """Perlin-like noise using smooth interpolated value noise."""
    result = np.zeros_like(x, dtype=np.float64)
    amplitude = 1.0
    frequency = freq
    max_amp = 0.0
    
    for i in range(min(octaves, 4)):
        result += _smooth_noise(x * frequency, z * frequency, seed + i * 1000) * amplitude
        max_amp += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    
    return (result / max_amp) * amp


def wave_ridged(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                octaves: int = 4, seed: int = 0) -> np.ndarray:
    """Ridged noise - creates mountain ridge patterns."""
    # Get base noise
    noise_val = wave_perlin(x, z, freq, 1.0, octaves, seed)
    # Create ridges by taking abs and inverting
    ridged = 1.0 - np.abs(noise_val)
    # Square for sharper ridges
    return (ridged * ridged) * amp


def wave_terraces(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                  levels: int = 5) -> np.ndarray:
    """Terrace/step wave pattern."""
    t = (x + z) * freq
    continuous = np.sin(t * 2 * np.pi)
    terraced = np.floor(continuous * levels) / levels
    return terraced * amp


def wave_voronoi(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                 seed: int = 0) -> np.ndarray:
    """Voronoi-like cellular pattern."""
    # Simple cellular approximation
    np.random.seed(seed)
    cell_x = np.floor(x * freq)
    cell_z = np.floor(z * freq)
    
    # Distance to nearest cell center (simplified)
    frac_x = (x * freq) - cell_x
    frac_z = (z * freq) - cell_z
    
    dist = np.sqrt(frac_x**2 + frac_z**2)
    return (1.0 - dist * 1.4) * amp


# =============================================================================
# Simple DNA - Just a list of wave configs
# =============================================================================

@dataclass
class WaveConfig:
    """Configuration for a single wave."""
    wave_type: str = "sin"  # sin, cos, sin2d, radial, perlin, ridged, terraces, voronoi
    freq: float = 0.01      # Frequency (lower = larger features)
    freq_z: float = None    # Optional separate Z frequency (for sin2d)
    amp: float = 10.0       # Amplitude (height contribution)
    phase: float = 0.0      # Phase offset
    direction: float = 0.0  # Direction angle in radians
    cx: float = 0.0         # Center X (for radial)
    cz: float = 0.0         # Center Z (for radial)
    octaves: int = 4        # Octaves (for noise types)
    levels: int = 5         # Levels (for terraces)


@dataclass 
class WorldConfig:
    """Complete world configuration."""
    name: str = "Waverse World"
    seed: int = 42
    water_level: float = 0.0
    
    # List of waves to stack (low freq to high freq typically)
    waves: List[WaveConfig] = field(default_factory=list)
    
    @classmethod
    def create_default(cls) -> "WorldConfig":
        """Create a world with large flat areas and spaced-out features."""
        return cls(
            name="Default World",
            seed=42,
            waves=[
                # Base - flat with gentle variation
                WaveConfig("sin", freq=0.0008, amp=5, phase=0),
                WaveConfig("cos", freq=0.0012, amp=4, phase=0.3, direction=0.4),
                
                # Very gentle rolling plains
                WaveConfig("perlin", freq=0.003, amp=4, octaves=2),
                
                # Sparse hills - low frequency means far apart
                WaveConfig("sin2d", freq=0.004, freq_z=0.003, amp=6),
                WaveConfig("perlin", freq=0.008, amp=5, octaves=2),
                
                # Occasional mountains - very sparse
                WaveConfig("ridged", freq=0.006, amp=18, octaves=3),
                
                # Subtle ground texture
                WaveConfig("perlin", freq=0.03, amp=1.5, octaves=2),
                WaveConfig("perlin", freq=0.08, amp=0.5, octaves=1),
            ]
        )
    
    @classmethod
    def create_islands(cls) -> "WorldConfig":
        """Island archipelago with lots of water."""
        return cls(
            name="Island World",
            seed=123,
            water_level=8.0,  # Higher water level
            waves=[
                # Base - mostly underwater
                WaveConfig("sin", freq=0.001, amp=5),
                
                # Ocean floor variation
                WaveConfig("perlin", freq=0.004, amp=6, octaves=3),
                
                # Island bumps - creates isolated landmasses
                WaveConfig("perlin", freq=0.008, amp=18, octaves=4),
                WaveConfig("ridged", freq=0.01, amp=12, octaves=3),
                
                # Some larger land masses
                WaveConfig("sin2d", freq=0.003, freq_z=0.004, amp=10),
                
                # Detail
                WaveConfig("perlin", freq=0.04, amp=2, octaves=2),
            ]
        )
    
    @classmethod
    def create_mountains(cls) -> "WorldConfig":
        """Mountain world with valleys and lakes."""
        return cls(
            name="Mountain World",
            seed=999,
            water_level=0.0,
            waves=[
                # Base - creates valley floors
                WaveConfig("sin", freq=0.001, amp=5),
                WaveConfig("perlin", freq=0.003, amp=8, octaves=2),
                
                # Valley/plateau structure
                WaveConfig("sin2d", freq=0.004, freq_z=0.003, amp=10),
                
                # Mountain ridges - the main attraction
                WaveConfig("ridged", freq=0.006, amp=25, octaves=5),
                WaveConfig("ridged", freq=0.012, amp=15, octaves=4),
                
                # Foothills
                WaveConfig("perlin", freq=0.015, amp=6, octaves=3),
                
                # Rocky detail
                WaveConfig("perlin", freq=0.04, amp=3, octaves=2),
                WaveConfig("ridged", freq=0.06, amp=2, octaves=2),
            ]
        )
    
    @classmethod
    def create_psychedelic(cls) -> "WorldConfig":
        """Trippy interference patterns - still explorable."""
        return cls(
            name="Psychedelic Realm",
            seed=420,
            water_level=0.0,
            waves=[
                # Base
                WaveConfig("sin", freq=0.002, amp=8),
                
                # Interference pattern - multiple overlapping waves
                WaveConfig("sin", freq=0.015, amp=8, direction=0),
                WaveConfig("sin", freq=0.015, amp=8, direction=0.5),
                WaveConfig("sin", freq=0.015, amp=8, direction=1.0),
                WaveConfig("sin", freq=0.015, amp=8, direction=1.5),
                
                # Radial ripples
                WaveConfig("radial", freq=0.02, amp=6, cx=100, cz=100),
                WaveConfig("radial", freq=0.015, amp=5, cx=-80, cz=50),
                
                # High frequency ripples
                WaveConfig("sin2d", freq=0.06, freq_z=0.05, amp=3),
                
                # Some noise for organic feel
                WaveConfig("perlin", freq=0.03, amp=4, octaves=3),
            ]
        )


# =============================================================================
# Height Evaluation
# =============================================================================

WAVE_FUNCS = {
    "sin": wave_sin,
    "cos": wave_cos,
    "sin2d": wave_sin2d,
    "radial": wave_radial,
    "perlin": wave_perlin,
    "ridged": wave_ridged,
    "terraces": wave_terraces,
    "voronoi": wave_voronoi,
}


def get_height(config: WorldConfig, x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Get terrain height at coordinates by stacking all waves.
    
    This is the core function - just add up all the wave contributions.
    """
    height = np.zeros_like(x, dtype=np.float64)
    
    for wave in config.waves:
        func = WAVE_FUNCS.get(wave.wave_type, wave_sin)
        
        if wave.wave_type == "sin2d":
            freq_z = wave.freq_z if wave.freq_z else wave.freq
            height += wave_sin2d(x, z, wave.freq, freq_z, wave.amp, wave.phase)
        elif wave.wave_type == "radial":
            height += wave_radial(x, z, wave.freq, wave.amp, wave.cx, wave.cz)
        elif wave.wave_type in ("perlin", "ridged"):
            height += func(x, z, wave.freq, wave.amp, wave.octaves, config.seed)
        elif wave.wave_type == "terraces":
            height += wave_terraces(x, z, wave.freq, wave.amp, wave.levels)
        elif wave.wave_type == "voronoi":
            height += wave_voronoi(x, z, wave.freq, wave.amp, config.seed)
        else:
            height += func(x, z, wave.freq, wave.amp, wave.phase, wave.direction)
    
    return height


# =============================================================================
# Chunk System - Simple heightmap per chunk
# =============================================================================

CHUNK_SIZE = 32  # Tiles per chunk (smaller = faster generation)
TILE_SCALE = 1.0  # World units per tile

# Disk cache for chunks
import os
CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '.chunk_cache')


@dataclass
class Chunk:
    """A chunk of terrain - just a heightmap grid."""
    cx: int  # Chunk X coordinate
    cz: int  # Chunk Z coordinate
    heightmap: np.ndarray = None  # 2D array of heights
    
    @property
    def world_x(self) -> float:
        """World X of chunk origin."""
        return self.cx * CHUNK_SIZE * TILE_SCALE
    
    @property
    def world_z(self) -> float:
        """World Z of chunk origin."""
        return self.cz * CHUNK_SIZE * TILE_SCALE


def _get_cache_path(seed: int, cx: int, cz: int) -> str:
    """Get cache file path for a chunk."""
    cache_dir = os.path.join(CACHE_DIR, f"seed_{seed}")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"chunk_{cx}_{cz}.npy")


def generate_chunk(config: WorldConfig, cx: int, cz: int, use_cache: bool = True) -> Chunk:
    """Generate a chunk's heightmap from wave config, with disk caching."""
    chunk = Chunk(cx=cx, cz=cz)
    
    # Try to load from cache first
    if use_cache:
        cache_path = _get_cache_path(config.seed, cx, cz)
        if os.path.exists(cache_path):
            try:
                chunk.heightmap = np.load(cache_path)
                return chunk
            except:
                pass  # Cache corrupted, regenerate
    
    # Create coordinate grids - need CHUNK_SIZE+1 points for CHUNK_SIZE tiles
    size = CHUNK_SIZE + 1
    
    # Local coordinates within chunk
    local_x = np.arange(size) * TILE_SCALE
    local_z = np.arange(size) * TILE_SCALE
    
    # Convert to world coordinates
    world_x = local_x + chunk.world_x
    world_z = local_z + chunk.world_z
    
    # Create meshgrid
    xx, zz = np.meshgrid(world_x, world_z)
    
    # Generate heights
    chunk.heightmap = get_height(config, xx, zz)
    
    # Save to cache
    if use_cache:
        try:
            cache_path = _get_cache_path(config.seed, cx, cz)
            np.save(cache_path, chunk.heightmap)
        except:
            pass  # Cache write failed, ignore
    
    return chunk


class ChunkManager:
    """Manages chunk loading/caching."""
    
    def __init__(self, config: WorldConfig, cache_size: int = 128):
        self.config = config
        self.cache_size = cache_size
        self.chunks: Dict[Tuple[int, int], Chunk] = {}
        self.worker = None  # Optional background worker
    
    def set_worker(self, worker):
        """Set the background chunk worker."""
        self.worker = worker
    
    def world_to_chunk_coords(self, world_x: float, world_z: float) -> Tuple[int, int]:
        """Convert world position to chunk coordinates."""
        chunk_world_size = CHUNK_SIZE * TILE_SCALE
        cx = int(np.floor(world_x / chunk_world_size))
        cz = int(np.floor(world_z / chunk_world_size))
        return (cx, cz)
    
    def get_chunk(self, cx: int, cz: int) -> Chunk:
        """Get or generate a chunk, using worker if available."""
        key = (cx, cz)
        
        # Check local cache first
        if key in self.chunks:
            return self.chunks[key]
        
        # Try to get from worker (pre-generated)
        if self.worker:
            chunk = self.worker.get_chunk(cx, cz)
            if chunk:
                self.chunks[key] = chunk
                # Evict old chunks if cache full
                while len(self.chunks) > self.cache_size:
                    oldest = next(iter(self.chunks))
                    del self.chunks[oldest]
                return chunk
        
        # Generate synchronously as fallback
        self.chunks[key] = generate_chunk(self.config, cx, cz)
        
        # Evict old chunks if cache full
        while len(self.chunks) > self.cache_size:
            oldest = next(iter(self.chunks))
            del self.chunks[oldest]
        
        return self.chunks[key]
    
    def get_height_at(self, world_x: float, world_z: float) -> float:
        """Get interpolated height at world position."""
        cx, cz = self.world_to_chunk_coords(world_x, world_z)
        chunk = self.get_chunk(cx, cz)
        
        # Position within chunk
        chunk_world_size = CHUNK_SIZE * TILE_SCALE
        local_x = world_x - cx * chunk_world_size
        local_z = world_z - cz * chunk_world_size
        
        # Grid indices
        gx = local_x / TILE_SCALE
        gz = local_z / TILE_SCALE
        
        x0, z0 = int(gx), int(gz)
        x1, z1 = min(x0 + 1, CHUNK_SIZE), min(z0 + 1, CHUNK_SIZE)
        
        # Bilinear interpolation
        tx = gx - x0
        tz = gz - z0
        
        h00 = chunk.heightmap[z0, x0]
        h10 = chunk.heightmap[z0, x1]
        h01 = chunk.heightmap[z1, x0]
        h11 = chunk.heightmap[z1, x1]
        
        h = (h00 * (1-tx) * (1-tz) +
             h10 * tx * (1-tz) +
             h01 * (1-tx) * tz +
             h11 * tx * tz)
        
        return float(h)

