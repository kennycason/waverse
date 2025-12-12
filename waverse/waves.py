"""
Wave Functions - The mathematical heart of terrain generation.

Implements various wave functions that can be stacked Fourier-style
to create complex terrain patterns.
"""

import numpy as np
import math
from typing import Callable, Dict, Optional, Tuple, Union, TYPE_CHECKING
from functools import lru_cache

if TYPE_CHECKING:
    from .dna import Wave, WaveLayer, Feature, WaveDNA


# Try to import noise library, fall back to numpy-based noise if not available
try:
    import noise
    NOISE_AVAILABLE = True
except ImportError:
    NOISE_AVAILABLE = False
    print("Warning: 'noise' library not found. Using numpy-based noise fallback.")


# =============================================================================
# Core Wave Functions
# =============================================================================

def wave_sin(t: np.ndarray) -> np.ndarray:
    """Classic sine wave."""
    return np.sin(t)


def wave_cos(t: np.ndarray) -> np.ndarray:
    """Classic cosine wave."""
    return np.cos(t)


def wave_triangle(t: np.ndarray) -> np.ndarray:
    """Triangle wave - linear ramps up and down."""
    return 2 * np.abs(2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))) - 1


def wave_sawtooth(t: np.ndarray) -> np.ndarray:
    """Sawtooth wave - linear ramp with sharp drop."""
    return 2 * (t / (2 * np.pi) - np.floor(t / (2 * np.pi) + 0.5))


def wave_square(t: np.ndarray) -> np.ndarray:
    """Square wave - binary high/low."""
    return np.sign(np.sin(t))


def _numpy_noise_2d(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Simple numpy-based pseudo-noise for when noise library isn't available.
    Not as good as Perlin, but works as a fallback.
    """
    # Hash-based pseudo-random
    n = x * 374761393 + y * 668265263 + seed
    n = (n.astype(np.int64) ^ (n.astype(np.int64) >> 13)) * 1274126177
    n = n.astype(np.int64) ^ (n.astype(np.int64) >> 16)
    return (n.astype(np.float64) / 2147483648.0) % 1.0 * 2 - 1


def _numpy_noise_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, seed: int = 0) -> np.ndarray:
    """3D version of numpy fallback noise."""
    n = x * 374761393 + y * 668265263 + z * 1440670337 + seed
    n = (n.astype(np.int64) ^ (n.astype(np.int64) >> 13)) * 1274126177
    n = n.astype(np.int64) ^ (n.astype(np.int64) >> 16)
    return (n.astype(np.float64) / 2147483648.0) % 1.0 * 2 - 1


def wave_perlin_2d(x: np.ndarray, z: np.ndarray, octaves: int = 4, seed: int = 0) -> np.ndarray:
    """
    Perlin noise - organic, natural-looking variation.
    Works on 2D coordinates (x, z plane).
    """
    if NOISE_AVAILABLE:
        # Vectorized perlin using the noise library
        result = np.zeros_like(x, dtype=np.float64)
        it = np.nditer([x, z], flags=['multi_index'])
        for xi, zi in it:
            result[it.multi_index] = noise.pnoise2(
                float(xi), float(zi),
                octaves=octaves,
                persistence=0.5,
                lacunarity=2.0,
                base=seed
            )
        return result
    else:
        # Fallback: layered numpy noise
        result = np.zeros_like(x, dtype=np.float64)
        amplitude = 1.0
        frequency = 1.0
        for _ in range(octaves):
            result += _numpy_noise_2d(x * frequency, z * frequency, seed) * amplitude
            amplitude *= 0.5
            frequency *= 2.0
        return result / 2.0  # Normalize roughly to [-1, 1]


def wave_perlin_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                   octaves: int = 4, seed: int = 0) -> np.ndarray:
    """
    3D Perlin noise for cave generation.
    """
    if NOISE_AVAILABLE:
        result = np.zeros_like(x, dtype=np.float64)
        it = np.nditer([x, y, z], flags=['multi_index'])
        for xi, yi, zi in it:
            result[it.multi_index] = noise.pnoise3(
                float(xi), float(yi), float(zi),
                octaves=octaves,
                persistence=0.5,
                lacunarity=2.0,
                base=seed
            )
        return result
    else:
        result = np.zeros_like(x, dtype=np.float64)
        amplitude = 1.0
        frequency = 1.0
        for _ in range(octaves):
            result += _numpy_noise_3d(
                x * frequency, y * frequency, z * frequency, seed
            ) * amplitude
            amplitude *= 0.5
            frequency *= 2.0
        return result / 2.0


def wave_simplex_2d(x: np.ndarray, z: np.ndarray, octaves: int = 4, seed: int = 0) -> np.ndarray:
    """
    Simplex noise - faster variant of Perlin with fewer directional artifacts.
    """
    if NOISE_AVAILABLE:
        result = np.zeros_like(x, dtype=np.float64)
        it = np.nditer([x, z], flags=['multi_index'])
        for xi, zi in it:
            result[it.multi_index] = noise.snoise2(
                float(xi), float(zi),
                octaves=octaves,
                persistence=0.5,
                lacunarity=2.0,
                base=seed
            )
        return result
    else:
        # Fall back to our numpy noise with slight variation
        return wave_perlin_2d(x + 1000, z + 1000, octaves, seed + 12345)


def wave_ridged_2d(x: np.ndarray, z: np.ndarray, octaves: int = 4, seed: int = 0) -> np.ndarray:
    """
    Ridged noise - creates sharp ridges and valleys, great for mountains.
    """
    noise_val = wave_perlin_2d(x, z, octaves, seed)
    # Take absolute value and invert to create ridges
    ridged = 1.0 - np.abs(noise_val)
    # Square it to sharpen the ridges
    return ridged * ridged * 2 - 1


# =============================================================================
# Wave Function Registry
# =============================================================================

WAVE_FUNCTIONS: Dict[str, Callable] = {
    "sin": wave_sin,
    "cos": wave_cos,
    "triangle": wave_triangle,
    "sawtooth": wave_sawtooth,
    "square": wave_square,
    # Note: perlin, simplex, ridged are handled specially (need 2D/3D coords)
}

NOISE_FUNCTIONS_2D = {
    "perlin": wave_perlin_2d,
    "simplex": wave_simplex_2d,
    "ridged": wave_ridged_2d,
}

NOISE_FUNCTIONS_3D = {
    "perlin": wave_perlin_3d,
    # simplex 3d and ridged 3d can be added similarly
}


# =============================================================================
# Falloff Functions (for localized features)
# =============================================================================

def falloff_linear(distance: np.ndarray, radius: float) -> np.ndarray:
    """Linear falloff from 1 at center to 0 at radius."""
    return np.clip(1.0 - distance / radius, 0.0, 1.0)


def falloff_gaussian(distance: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian falloff - smooth bell curve."""
    sigma = radius / 3.0  # 3 sigma = ~99.7% of influence within radius
    return np.exp(-(distance ** 2) / (2 * sigma ** 2))


def falloff_cosine(distance: np.ndarray, radius: float) -> np.ndarray:
    """Cosine falloff - smooth S-curve."""
    t = np.clip(distance / radius, 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(t * np.pi))


def falloff_smooth(distance: np.ndarray, radius: float) -> np.ndarray:
    """Smoothstep falloff - very smooth at edges."""
    t = np.clip(distance / radius, 0.0, 1.0)
    return 1.0 - (t * t * (3.0 - 2.0 * t))


def falloff_sharp(distance: np.ndarray, radius: float) -> np.ndarray:
    """Sharp falloff - maintains strength until close to edge."""
    t = np.clip(distance / radius, 0.0, 1.0)
    return 1.0 - t ** 4


FALLOFF_FUNCTIONS: Dict[str, Callable] = {
    "linear": falloff_linear,
    "gaussian": falloff_gaussian,
    "cosine": falloff_cosine,
    "smooth": falloff_smooth,
    "sharp": falloff_sharp,
}


# =============================================================================
# Wave Evaluation
# =============================================================================

def evaluate_single_wave(wave: "Wave", x: np.ndarray, z: np.ndarray, 
                         seed: int = 0) -> np.ndarray:
    """
    Evaluate a single Wave at the given x, z coordinates.
    
    Args:
        wave: The Wave definition
        x: X coordinates (can be scalar or array)
        z: Z coordinates (can be scalar or array)
        seed: World seed for noise functions
        
    Returns:
        Height values at each coordinate
    """
    # Convert to numpy arrays if needed
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    
    # Apply direction rotation if specified
    if wave.direction != 0:
        cos_d = np.cos(wave.direction)
        sin_d = np.sin(wave.direction)
        x_rot = x * cos_d - z * sin_d
        z_rot = x * sin_d + z * cos_d
        x, z = x_rot, z_rot
    
    wave_type = wave.wave_type
    
    # Handle noise-based functions specially
    if wave_type in NOISE_FUNCTIONS_2D:
        noise_func = NOISE_FUNCTIONS_2D[wave_type]
        # Noise functions use frequency directly as coordinate scaling
        result = noise_func(
            x * wave.freq_x + wave.phase,
            z * wave.freq_z,
            octaves=max(1, wave.harmonic),
            seed=seed
        )
    elif wave_type in WAVE_FUNCTIONS:
        # Standard periodic waves
        wave_func = WAVE_FUNCTIONS[wave_type]
        # Create the input "t" from x and z frequencies
        t = (x * wave.freq_x + z * wave.freq_z) * 2 * np.pi + wave.phase
        result = wave_func(t)
    else:
        raise ValueError(f"Unknown wave type: {wave_type}")
    
    return result * wave.amplitude


def evaluate_single_wave_3d(wave: "Wave", x: np.ndarray, y: np.ndarray, 
                            z: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Evaluate a 3D wave (for caves/overhangs).
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    
    freq_y = wave.freq_y if wave.freq_y is not None else wave.freq_x
    
    if wave.wave_type in NOISE_FUNCTIONS_3D:
        noise_func = NOISE_FUNCTIONS_3D[wave.wave_type]
        result = noise_func(
            x * wave.freq_x + wave.phase,
            y * freq_y,
            z * wave.freq_z,
            octaves=max(1, wave.harmonic),
            seed=seed
        )
    else:
        # For non-noise 3D waves, combine all three dimensions
        if wave.wave_type in WAVE_FUNCTIONS:
            wave_func = WAVE_FUNCTIONS[wave.wave_type]
            t = (x * wave.freq_x + y * freq_y + z * wave.freq_z) * 2 * np.pi + wave.phase
            result = wave_func(t)
        else:
            # Default to perlin for unknown 3D types
            result = wave_perlin_3d(
                x * wave.freq_x, y * freq_y, z * wave.freq_z,
                octaves=4, seed=seed
            )
    
    return result * wave.amplitude


def evaluate_layer(layer: "WaveLayer", x: np.ndarray, z: np.ndarray,
                   seed: int = 0) -> np.ndarray:
    """
    Evaluate all waves in a layer and combine them.
    """
    if not layer.waves:
        return np.zeros_like(x, dtype=np.float64)
    
    result = np.zeros_like(x, dtype=np.float64)
    
    for wave in layer.waves:
        result += evaluate_single_wave(wave, x, z, seed)
    
    return result * layer.weight


def evaluate_feature(feature: "Feature", x: np.ndarray, z: np.ndarray,
                     seed: int = 0) -> np.ndarray:
    """
    Evaluate a localized feature with falloff.
    """
    # Calculate distance from feature center
    dx = x - feature.center_x
    dz = z - feature.center_z
    distance = np.sqrt(dx ** 2 + dz ** 2)
    
    # Get falloff weights
    falloff_func = FALLOFF_FUNCTIONS.get(feature.falloff, falloff_gaussian)
    weights = falloff_func(distance, feature.radius)
    
    # Only evaluate where we have influence (optimization)
    mask = weights > 0.001
    if not np.any(mask):
        return np.zeros_like(x, dtype=np.float64)
    
    # Evaluate feature waves
    result = np.zeros_like(x, dtype=np.float64)
    
    # Use feature's local seed if specified
    local_seed = feature.local_seed if feature.local_seed is not None else seed
    
    for wave in feature.waves:
        result += evaluate_single_wave(wave, x, z, local_seed)
    
    # Add height offset and apply falloff
    result = (result + feature.height_offset) * weights
    
    return result


def evaluate_waves(dna: "WaveDNA", x: Union[float, np.ndarray], 
                   z: Union[float, np.ndarray]) -> np.ndarray:
    """
    Evaluate complete terrain height at given coordinates using the full DNA.
    
    This is the main entry point for terrain generation.
    
    Args:
        dna: The world DNA
        x: X coordinates (world space)
        z: Z coordinates (world space)
        
    Returns:
        Height values at each coordinate
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    
    # Start with zero height
    result = np.zeros_like(x, dtype=np.float64)
    
    # Apply each global layer
    for layer in dna.layers:
        layer_result = evaluate_layer(layer, x, z, dna.seed)
        
        # Apply blend mode
        if layer.blend_mode == "add":
            result += layer_result
        elif layer.blend_mode == "multiply":
            result *= layer_result
        elif layer.blend_mode == "max":
            result = np.maximum(result, layer_result)
        elif layer.blend_mode == "min":
            result = np.minimum(result, layer_result)
        elif layer.blend_mode == "average":
            result = (result + layer_result) / 2
    
    # Apply localized features
    for feature in dna.features:
        result += evaluate_feature(feature, x, z, dna.seed)
    
    return result


def evaluate_density(dna: "WaveDNA", x: np.ndarray, y: np.ndarray, 
                     z: np.ndarray) -> np.ndarray:
    """
    Evaluate terrain density at 3D coordinates (for caves/overhangs).
    
    Returns:
        Density values: > 0 means solid, < 0 means air
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    z = np.atleast_1d(np.asarray(z, dtype=np.float64))
    
    # Get surface height at this x, z
    surface_height = evaluate_waves(dna, x, z)
    
    # Base density: above surface = air (-), below surface = solid (+)
    density = surface_height - y
    
    # Apply cave layers (carve out where density would otherwise be positive)
    for cave_layer in dna.cave_layers:
        # Only affect within cave depth range
        in_range = (y >= cave_layer.y_min) & (y <= cave_layer.y_max)
        
        if np.any(in_range):
            cave_value = np.zeros_like(density)
            for wave in cave_layer.waves:
                cave_value += evaluate_single_wave_3d(wave, x, y, z, dna.seed)
            
            # Where cave value exceeds threshold, carve out
            cave_mask = in_range & (cave_value > cave_layer.threshold)
            
            # Reduce density in cave regions
            cave_strength = (cave_value - cave_layer.threshold) * cave_layer.density
            density = np.where(cave_mask, density - cave_strength * 10, density)
    
    return density


# =============================================================================
# Utility Functions
# =============================================================================

def generate_height_grid(dna: "WaveDNA", x_start: float, z_start: float,
                         width: int, height: int, 
                         scale: float = 1.0) -> np.ndarray:
    """
    Generate a 2D grid of heights for a region.
    
    Args:
        dna: World DNA
        x_start: Starting X coordinate (world space)
        z_start: Starting Z coordinate (world space)
        width: Grid width in tiles
        height: Grid height in tiles
        scale: Tile size in world units
        
    Returns:
        2D numpy array of heights, shape (height, width)
    """
    x = np.arange(width) * scale + x_start
    z = np.arange(height) * scale + z_start
    
    xx, zz = np.meshgrid(x, z)
    
    return evaluate_waves(dna, xx.flatten(), zz.flatten()).reshape(height, width)


def generate_density_volume(dna: "WaveDNA", x_start: float, y_start: float,
                            z_start: float, size_x: int, size_y: int, 
                            size_z: int, scale: float = 1.0) -> np.ndarray:
    """
    Generate a 3D volume of density values (for cave generation).
    
    Returns:
        3D numpy array of densities, shape (size_z, size_y, size_x)
    """
    x = np.arange(size_x) * scale + x_start
    y = np.arange(size_y) * scale + y_start
    z = np.arange(size_z) * scale + z_start
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    densities = evaluate_density(
        dna, 
        xx.flatten(), 
        yy.flatten(), 
        zz.flatten()
    )
    
    return densities.reshape(size_x, size_y, size_z)

