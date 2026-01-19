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


def wave_cliff(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
               direction: float = 0, sharpness: float = 8.0, seed: int = 0) -> np.ndarray:
    """Sharp cliff/step terrain - creates sudden elevation changes."""
    # Directional wave with sharp sigmoid transition
    cos_d, sin_d = np.cos(direction), np.sin(direction)
    t = (x * cos_d + z * sin_d) * freq
    
    # Add some noise to break up the perfectly straight line
    noise_offset = _smooth_noise(x * freq * 2, z * freq * 2, seed) * 0.3
    t = t + noise_offset
    
    # Sigmoid for sharp transition (tanh gives smooth step)
    stepped = np.tanh(np.sin(t * 2 * np.pi) * sharpness)
    return stepped * amp


def wave_plateau(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                 seed: int = 0, flatness: float = 0.6) -> np.ndarray:
    """Flat-topped plateaus with steep edges."""
    # Use noise to define plateau regions
    noise_val = _smooth_noise(x * freq, z * freq, seed)
    
    # Create flat tops by clamping high values
    plateau = np.where(noise_val > flatness, 1.0,
                       np.where(noise_val < -flatness, -1.0,
                               noise_val / flatness))
    
    # Sharpen the edges with a power function
    plateau = np.sign(plateau) * np.abs(plateau) ** 0.5
    return plateau * amp


def wave_canyon(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                direction: float = 0, width: float = 0.15, seed: int = 0) -> np.ndarray:
    """Linear canyon/trench - cuts deep into terrain."""
    cos_d, sin_d = np.cos(direction), np.sin(direction)
    
    # Position along canyon direction (determines where canyons are)
    t = (x * cos_d + z * sin_d) * freq
    
    # Perpendicular distance (determines canyon walls)
    perp = (-x * sin_d + z * cos_d) * freq * 3
    
    # Canyon positions (using sine to create multiple canyons)
    canyon_dist = np.abs(np.sin(t * 2 * np.pi))
    
    # Add noise to make canyons wind
    wind = _smooth_noise(x * freq * 0.5, z * freq * 0.5, seed) * 0.2
    canyon_dist = np.abs(np.sin((t + wind) * 2 * np.pi))
    
    # Create canyon profile (deep in center, steep walls)
    in_canyon = canyon_dist < width
    depth = np.where(in_canyon, 
                     -1.0 * (1.0 - canyon_dist / width) ** 2,  # Parabolic floor
                     0.0)
    return depth * amp


def wave_crater(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                seed: int = 0) -> np.ndarray:
    """Crater/lake basin pattern - circular depressions."""
    # Use Voronoi-like cells for crater positions
    cell_x = np.floor(x * freq)
    cell_z = np.floor(z * freq)
    frac_x = (x * freq) - cell_x
    frac_z = (z * freq) - cell_z
    
    # Randomize center within cell
    rng_hash = (cell_x * 374761393 + cell_z * 668265263 + seed).astype(np.int64)
    cx_offset = ((rng_hash >> 8) & 0xFF) / 255.0 - 0.5
    cz_offset = ((rng_hash >> 16) & 0xFF) / 255.0 - 0.5
    
    # Distance to crater center
    dx = frac_x - 0.5 - cx_offset * 0.3
    dz = frac_z - 0.5 - cz_offset * 0.3
    dist = np.sqrt(dx**2 + dz**2)
    
    # Crater profile: raised rim, depressed center
    crater_size = 0.3 + ((rng_hash & 0xFF) / 255.0) * 0.2
    in_crater = dist < crater_size
    
    profile = np.where(in_crater,
                       np.where(dist < crater_size * 0.7,
                               -1.0 + (dist / (crater_size * 0.7)) * 0.3,  # Floor
                               0.3 * (1.0 - (dist - crater_size * 0.7) / (crater_size * 0.3))),  # Rim
                       0.0)
    return profile * amp


def wave_mountain_range(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                        direction: float = 0, seed: int = 0) -> np.ndarray:
    """Sharp mountain range along a direction."""
    cos_d, sin_d = np.cos(direction), np.sin(direction)
    
    # Distance from range center line
    perp = np.abs(-x * sin_d + z * cos_d) * freq * 2
    
    # Position along range (for peak variation)
    along = (x * cos_d + z * sin_d) * freq
    
    # Height along range (ridged noise for peaks)
    peak_noise = 1.0 - np.abs(_smooth_noise(along * 3, z * freq * 0.1, seed))
    peak_noise = peak_noise ** 2  # Sharpen peaks
    
    # Mountain profile (high in center, falls off to sides)
    range_width = 0.4 + _smooth_noise(along, along * 0.5, seed + 1000) * 0.15
    height = np.maximum(0, 1.0 - perp / range_width)
    height = height ** 1.5  # Steeper sides
    
    return height * peak_noise * amp


def wave_dunes(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
               direction: float = 0, seed: int = 0) -> np.ndarray:
    """Rolling sand dune formations with asymmetric profiles."""
    cos_d, sin_d = np.cos(direction), np.sin(direction)
    
    # Rotate coordinates to align with dune direction
    aligned = x * cos_d + z * sin_d
    perp = -x * sin_d + z * cos_d
    
    # Dune wave with asymmetric profile (gentle windward, steep leeward)
    dune_phase = aligned * freq + _smooth_noise(perp * freq * 0.3, aligned * freq * 0.1, seed) * 2
    
    # Sawtooth-like profile: slow rise, quick drop
    dune = np.mod(dune_phase, 2 * np.pi) / (2 * np.pi)
    dune = dune ** 0.7  # Asymmetric
    
    # Add some height variation between dunes
    height_var = 0.7 + 0.3 * _smooth_noise(perp * freq * 0.5, aligned * freq * 0.2, seed + 1)
    
    return dune * height_var * amp


def wave_mesa(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
              seed: int = 0, steepness: float = 8.0) -> np.ndarray:
    """Flat-topped elevated terrain with steep cliff sides."""
    # Use noise to define mesa regions
    noise = _smooth_noise(x * freq, z * freq, seed)
    
    # Apply sigmoid to create flat tops and steep transitions
    mesa = 1.0 / (1.0 + np.exp(-steepness * (noise - 0.3)))
    
    # Add slight surface variation on top
    top_detail = _smooth_noise(x * freq * 4, z * freq * 4, seed + 500) * 0.1
    mesa = mesa + mesa * top_detail
    
    return mesa * amp


def wave_staircases(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                    steps: int = 6, seed: int = 0) -> np.ndarray:
    """Stepped terrain like rice terraces or geological layers."""
    # Base slope
    slope = _smooth_noise(x * freq, z * freq, seed)
    
    # Quantize to steps
    terraced = np.floor(slope * steps) / steps
    
    # Add slight slope within each terrace
    within_step = np.mod(slope * steps, 1.0) * 0.15
    
    return (terraced + within_step) * amp


def wave_ripples(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                 cx: float = 0, cz: float = 0, seed: int = 0) -> np.ndarray:
    """Concentric ripple patterns with decay."""
    # Multiple ripple centers from noise
    n_centers = 3
    height = np.zeros_like(x)
    
    for i in range(n_centers):
        # Offset center based on seed
        offset_x = _smooth_noise(i * 100 + seed, 0, seed) * 500
        offset_z = _smooth_noise(0, i * 100 + seed, seed + 1) * 500
        
        dist = np.sqrt((x - cx - offset_x)**2 + (z - cz - offset_z)**2)
        
        # Ripple with decay
        decay = np.exp(-dist * freq * 0.15)
        ripple = np.sin(dist * freq) * decay
        height += ripple
    
    return height * amp / n_centers


def wave_fractured(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                   seed: int = 0) -> np.ndarray:
    """Broken plate-like terrain with visible fracture lines."""
    # Create voronoi-like regions
    # Use noise to create cell centers
    cell_x = np.floor(x * freq + 0.5)
    cell_z = np.floor(z * freq + 0.5)
    
    # Height per cell (constant within cell)
    cell_height = _smooth_noise(cell_x, cell_z, seed)
    
    # Create fracture lines at cell edges using distance to cell center
    frac_x = np.abs(np.mod(x * freq + 0.5, 1.0) - 0.5) * 2
    frac_z = np.abs(np.mod(z * freq + 0.5, 1.0) - 0.5) * 2
    
    # Combine: lower near edges (fractures)
    edge_factor = np.minimum(frac_x, frac_z)
    edge_factor = np.clip(edge_factor * 3, 0, 1)  # Narrow the edge depression
    
    return cell_height * edge_factor * amp


def wave_eroded(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                seed: int = 0) -> np.ndarray:
    """Simulates eroded terrain with smooth valleys and weathered peaks."""
    # Base terrain
    base = _smooth_noise(x * freq, z * freq, seed)
    
    # Simulate erosion: smooth the lows, sharpen the highs
    eroded = np.where(base < 0.5,
                      base * 0.8,  # Flatten valleys
                      0.4 + (base - 0.5) * 1.4)  # Preserve peaks
    
    # Add drainage patterns (subtle channels)
    channel_noise = _smooth_noise(x * freq * 3, z * freq * 0.5, seed + 100)
    channels = np.abs(channel_noise) ** 2
    
    return (eroded - channels * 0.1) * amp


def wave_volcanic(x: np.ndarray, z: np.ndarray, freq: float, amp: float,
                  seed: int = 0) -> np.ndarray:
    """Volcanic terrain with cones and calderas."""
    height = np.zeros_like(x)
    
    # Create several volcanic cones
    n_volcanoes = 4
    for i in range(n_volcanoes):
        # Random volcano position
        vx = _smooth_noise(i * 1000, seed, seed) * 300 / freq
        vz = _smooth_noise(seed, i * 1000, seed + 1) * 300 / freq
        
        dist = np.sqrt((x - vx)**2 + (z - vz)**2) * freq
        
        # Cone shape with caldera (depression at top)
        cone = np.maximum(0, 1.0 - dist * 1.5)
        cone = cone ** 1.5  # Steeper sides
        
        # Add caldera for larger cones
        if i == 0:  # Main volcano has caldera
            caldera = np.exp(-(dist * 8)**2) * 0.4
            cone = cone - caldera
        
        height = np.maximum(height, cone)
    
    return height * amp


# =============================================================================
# Simple DNA - Just a list of wave configs
# =============================================================================

@dataclass
class WaveConfig:
    """Configuration for a single wave."""
    # Wave types: sin, cos, sin2d, radial, perlin, ridged, terraces, voronoi,
    #             cliff, plateau, canyon, crater, mountain_range,
    #             dunes, mesa, staircases, ripples, fractured, eroded, volcanic
    wave_type: str = "sin"
    freq: float = 0.01      # Frequency (lower = larger features)
    freq_z: float = None    # Optional separate Z frequency (for sin2d)
    amp: float = 10.0       # Amplitude (height contribution)
    phase: float = 0.0      # Phase offset
    direction: float = 0.0  # Direction angle in radians
    cx: float = 0.0         # Center X (for radial)
    cz: float = 0.0         # Center Z (for radial)
    octaves: int = 4        # Octaves (for noise types)
    levels: int = 5         # Levels (for terraces)
    sharpness: float = 8.0  # Sharpness (for cliffs)
    width: float = 0.15     # Width (for canyons)
    flatness: float = 0.6   # Flatness threshold (for plateaus)


@dataclass 
class WorldConfig:
    """Complete world configuration."""
    name: str = "Waverse World"
    seed: int = 42
    water_level: float = 0.0
    # Global post-scale applied after all wave contributions are summed.
    # Lets us tune overall elevation without editing every WaveConfig.
    height_scale: float = 1.0
    
    # List of waves to stack (low freq to high freq typically)
    waves: List[WaveConfig] = field(default_factory=list)
    
    @classmethod
    def create_default(cls) -> "WorldConfig":
        """Create a dramatic world with varied terrain features.
        
        NOTE: Frequencies are scaled for TILE_SCALE=3.0 (3x world scale).
        This creates larger, smoother terrain with fewer chunks rendered.
        """
        return cls(
            name="Default World",
            seed=42,
            # Keep default terrain within unit-test expected bounds (< 200 max height)
            height_scale=0.75,
            waves=[
                # === MASSIVE TERRAIN FEATURES ===
                # Frequencies scaled for TILE_SCALE=3.0, amplitudes BOOSTED for drama!
                
                # CONTINENTAL BASE - huge rolling landmasses with flat areas
                WaveConfig("sin", freq=0.000015, amp=40, phase=0),  # Gentle continental roll
                WaveConfig("cos", freq=0.00002, amp=30, phase=0.5, direction=0.7),
                WaveConfig("perlin", freq=0.00003, amp=35, octaves=2),  # Continental noise
                
                # LARGE PLATEAUS - flat elevated regions (creates flat zones!)
                WaveConfig("plateau", freq=0.00008, amp=50, flatness=0.6),  # Big flat areas!
                
                # MAJOR MOUNTAIN RANGES - dramatic peaks!
                WaveConfig("mountain_range", freq=0.00015, amp=80, direction=0.3),  # Huge range!
                WaveConfig("mountain_range", freq=0.0002, amp=60, direction=1.8),   # Secondary range
                
                # MESAS - dramatic flat-topped buttes
                WaveConfig("mesa", freq=0.00012, amp=55, sharpness=12.0),
                
                # VOLCANIC REGIONS - cones and calderas
                WaveConfig("volcanic", freq=0.00015, amp=70),  # Dramatic volcanoes!
                
                # OCEAN TRENCHES / DEEP LAKES - deep depressions
                WaveConfig("crater", freq=0.0001, amp=-35),  # Deeper trenches
                
                # === MEDIUM SCALE FEATURES ===
                
                # Regional hills and valleys
                WaveConfig("sin", freq=0.00012, amp=20, phase=0.2),
                WaveConfig("perlin", freq=0.0002, amp=25, octaves=2),
                
                # DUNES - rolling sand dune fields
                WaveConfig("dunes", freq=0.0007, amp=18, direction=0.4),
                
                # CLIFF LINES - sharp elevation changes  
                WaveConfig("cliff", freq=0.0003, amp=25, direction=0.9, sharpness=8.0),
                WaveConfig("cliff", freq=0.00035, amp=18, direction=2.2, sharpness=6.0),
                
                # CANYONS - deep linear cuts
                WaveConfig("canyon", freq=0.00025, amp=30, direction=0.5, width=0.15),
                
                # FRACTURED TERRAIN - tectonic breaks
                WaveConfig("fractured", freq=0.0005, amp=15),
                
                # === LOCAL DETAIL ===
                
                # Local hills - smaller undulations
                WaveConfig("sin2d", freq=0.0005, freq_z=0.00045, amp=12),
                WaveConfig("perlin", freq=0.0008, amp=15, octaves=2),
                
                # RIDGED PEAKS - sharp mountain peaks
                WaveConfig("ridged", freq=0.001, amp=25, octaves=3),  # Sharper peaks!
                
                # ERODED TERRAIN - weathered valleys
                WaveConfig("eroded", freq=0.0012, amp=10),
                
                # STEPPED TERRAIN - natural terraces
                WaveConfig("staircases", freq=0.002, amp=8, levels=6),
                
                # Surface detail - small bumps
                WaveConfig("perlin", freq=0.003, amp=5, octaves=2),
                WaveConfig("perlin", freq=0.008, amp=2, octaves=1),
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
    "cliff": wave_cliff,
    "plateau": wave_plateau,
    "canyon": wave_canyon,
    "crater": wave_crater,
    "mountain_range": wave_mountain_range,
    # New creative wave types
    "dunes": wave_dunes,
    "mesa": wave_mesa,
    "staircases": wave_staircases,
    "ripples": wave_ripples,
    "fractured": wave_fractured,
    "eroded": wave_eroded,
    "volcanic": wave_volcanic,
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
        elif wave.wave_type == "cliff":
            height += wave_cliff(x, z, wave.freq, wave.amp, wave.direction, wave.sharpness, config.seed)
        elif wave.wave_type == "plateau":
            height += wave_plateau(x, z, wave.freq, wave.amp, config.seed, wave.flatness)
        elif wave.wave_type == "canyon":
            height += wave_canyon(x, z, wave.freq, wave.amp, wave.direction, wave.width, config.seed)
        elif wave.wave_type == "crater":
            height += wave_crater(x, z, wave.freq, wave.amp, config.seed)
        elif wave.wave_type == "mountain_range":
            height += wave_mountain_range(x, z, wave.freq, wave.amp, wave.direction, config.seed)
        # New creative wave types
        elif wave.wave_type == "dunes":
            height += wave_dunes(x, z, wave.freq, wave.amp, wave.direction, config.seed)
        elif wave.wave_type == "mesa":
            height += wave_mesa(x, z, wave.freq, wave.amp, config.seed, wave.sharpness)
        elif wave.wave_type == "staircases":
            height += wave_staircases(x, z, wave.freq, wave.amp, wave.levels, config.seed)
        elif wave.wave_type == "ripples":
            height += wave_ripples(x, z, wave.freq, wave.amp, wave.cx, wave.cz, config.seed)
        elif wave.wave_type == "fractured":
            height += wave_fractured(x, z, wave.freq, wave.amp, config.seed)
        elif wave.wave_type == "eroded":
            height += wave_eroded(x, z, wave.freq, wave.amp, config.seed)
        elif wave.wave_type == "volcanic":
            height += wave_volcanic(x, z, wave.freq, wave.amp, config.seed)
        else:
            height += func(x, z, wave.freq, wave.amp, wave.phase, wave.direction)
    
    return height * float(getattr(config, "height_scale", 1.0))


# =============================================================================
# Chunk System - Simple heightmap per chunk
# =============================================================================

CHUNK_SIZE = 32  # Tiles per chunk (smaller = faster generation)
TILE_SCALE = 3.0  # World units per tile (3x = larger chunks, smoother terrain, better perf)

# Disk cache for chunks - stored in ~/.waverse/chunks/
import os
WAVERSE_DIR = os.path.expanduser("~/.waverse")
CACHE_DIR = os.path.join(WAVERSE_DIR, "chunks")
os.makedirs(CACHE_DIR, exist_ok=True)


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


def _get_cache_path(config: WorldConfig, cx: int, cz: int) -> str:
    """Get cache file path for a chunk.

    NOTE: The cache key includes parameters that materially affect height output.
    This avoids stale-cache issues when tuning world generation.
    """
    hs = int(round(float(getattr(config, "height_scale", 1.0)) * 1000))
    cache_dir = os.path.join(CACHE_DIR, f"seed_{config.seed}_hs_{hs}")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"chunk_{cx}_{cz}.npy")


def generate_chunk(config: WorldConfig, cx: int, cz: int, use_cache: bool = True) -> Chunk:
    """Generate a chunk's heightmap from wave config, with disk caching."""
    chunk = Chunk(cx=cx, cz=cz)
    
    # Try to load from cache first
    if use_cache:
        cache_path = _get_cache_path(config, cx, cz)
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
            cache_path = _get_cache_path(config, cx, cz)
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

