"""
Chunk DNA System - Genetic parameters for terrain colors and sky.

Each chunk has DNA that controls:
- Terrain color palette (what colors map to what heights)
- Sky appearance (sun, moon, stars)

DNA evolves very slowly between chunks for gradual environmental shifts.
Sky/astral parameters change even more slowly than terrain colors.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
import numpy as np
import copy


Color = Tuple[float, float, float]


# =============================================================================
# MUTATION CONFIG - Easy to tweak! Future: these could evolve too! 
# =============================================================================
@dataclass
class MutationConfig:
    """Centralized mutation rates - tweak these to control evolution speed."""
    # How often mutations happen (0.0 = never, 1.0 = always)
    terrain_mutation_chance: float = 0.05   # 5% chance per chunk
    sky_mutation_chance: float = 0.05       # 5% chance per chunk
    
    # How strong mutations are (higher = bigger color shifts)
    terrain_mutation_strength: float = 0.06  # Noticeable but gradual
    sky_mutation_strength: float = 0.05      # Slightly gentler for sky
    
    # Chance of "burst" mutations (sudden big changes)
    burst_mutation_chance: float = 0.15      # 15% during a mutation
    burst_mutation_strength: float = 0.25    # How big the burst is
    
    # Color range limits (wider = more psychedelic potential)
    saturation_range: Tuple[float, float] = (0.3, 2.5)
    brightness_range: Tuple[float, float] = (0.4, 1.8)
    hue_shift_range: Tuple[float, float] = (-1.0, 1.0)  # Full rainbow
    
    # Sun/Moon size limits
    sun_size_range: Tuple[float, float] = (0.5, 2.0)
    moon_size_range: Tuple[float, float] = (0.5, 2.0)


# Global config instance - import and modify to change behavior!
MUTATION_CONFIG = MutationConfig()


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _mutate_color(color: Color, rng: np.random.Generator, strength: float) -> Color:
    """Mutate a color slightly."""
    return tuple(_clamp(c + rng.normal(0, strength)) for c in color)


def _blend_colors(c1: Color, c2: Color, t: float) -> Color:
    """Blend two colors."""
    return tuple(c1[i] * (1 - t) + c2[i] * t for i in range(3))


@dataclass
class TerrainPalette:
    """Color palette for terrain heights."""
    # Each tuple is (height_threshold, color)
    # Heights below first threshold use first color
    deep_water: Color = (0.08, 0.15, 0.35)      # < -5
    shallow_water: Color = (0.15, 0.30, 0.50)   # -5 to 2
    beach: Color = (0.85, 0.78, 0.55)           # 2 to 4
    grass: Color = (0.30, 0.55, 0.25)           # 4 to 15
    forest: Color = (0.18, 0.40, 0.15)          # 15 to 25
    hills: Color = (0.45, 0.40, 0.30)           # 25 to 35
    mountain: Color = (0.50, 0.48, 0.45)        # 35 to 50
    snow: Color = (0.92, 0.93, 0.95)            # > 50
    
    # Color variation/saturation
    saturation_mult: float = 1.0
    brightness_mult: float = 1.0
    hue_shift: float = 0.0  # -0.1 to 0.1
    
    def mutate(self, rng: np.random.Generator, strength: float = None) -> "TerrainPalette":
        """Create mutated copy using global MUTATION_CONFIG."""
        cfg = MUTATION_CONFIG
        strength = strength or cfg.terrain_mutation_strength
        
        # Check mutation chance
        if rng.random() > cfg.terrain_mutation_chance:
            return copy.deepcopy(self)
        
        new_palette = copy.deepcopy(self)
        
        # Apply color mutations
        new_palette.deep_water = _mutate_color(self.deep_water, rng, strength)
        new_palette.shallow_water = _mutate_color(self.shallow_water, rng, strength)
        new_palette.beach = _mutate_color(self.beach, rng, strength)
        new_palette.grass = _mutate_color(self.grass, rng, strength)
        new_palette.forest = _mutate_color(self.forest, rng, strength)
        new_palette.hills = _mutate_color(self.hills, rng, strength)
        new_palette.mountain = _mutate_color(self.mountain, rng, strength)
        new_palette.snow = _mutate_color(self.snow, rng, strength)
        
        # Overall adjustments - use config ranges
        new_palette.saturation_mult = _clamp(
            self.saturation_mult + rng.normal(0, strength), 
            cfg.saturation_range[0], cfg.saturation_range[1])
        new_palette.brightness_mult = _clamp(
            self.brightness_mult + rng.normal(0, strength * 0.8), 
            cfg.brightness_range[0], cfg.brightness_range[1])
        new_palette.hue_shift = _clamp(
            self.hue_shift + rng.normal(0, strength * 0.5), 
            cfg.hue_shift_range[0], cfg.hue_shift_range[1])
        
        # Burst mutation - occasional bigger color jump
        if rng.random() < cfg.burst_mutation_chance:
            burst_color = rng.choice(["grass", "forest", "hills", "mountain", "water"])
            burst_shift = rng.normal(0, cfg.burst_mutation_strength)
            if burst_color == "grass":
                new_palette.grass = tuple(_clamp(c + burst_shift) for c in new_palette.grass)
            elif burst_color == "forest":
                new_palette.forest = tuple(_clamp(c + burst_shift) for c in new_palette.forest)
            elif burst_color == "hills":
                new_palette.hills = tuple(_clamp(c + burst_shift) for c in new_palette.hills)
            elif burst_color == "mountain":
                new_palette.mountain = tuple(_clamp(c + burst_shift) for c in new_palette.mountain)
            elif burst_color == "water":
                new_palette.deep_water = tuple(_clamp(c + burst_shift) for c in new_palette.deep_water)
        
        return new_palette
    
    def crossover(self, other: "TerrainPalette", rng: np.random.Generator) -> "TerrainPalette":
        """Blend two palettes."""
        blend = 0.3 + rng.random() * 0.4  # 30-70% blend
        
        new_palette = TerrainPalette()
        new_palette.deep_water = _blend_colors(self.deep_water, other.deep_water, blend)
        new_palette.shallow_water = _blend_colors(self.shallow_water, other.shallow_water, blend)
        new_palette.beach = _blend_colors(self.beach, other.beach, blend)
        new_palette.grass = _blend_colors(self.grass, other.grass, blend)
        new_palette.forest = _blend_colors(self.forest, other.forest, blend)
        new_palette.hills = _blend_colors(self.hills, other.hills, blend)
        new_palette.mountain = _blend_colors(self.mountain, other.mountain, blend)
        new_palette.snow = _blend_colors(self.snow, other.snow, blend)
        
        new_palette.saturation_mult = self.saturation_mult * (1 - blend) + other.saturation_mult * blend
        new_palette.brightness_mult = self.brightness_mult * (1 - blend) + other.brightness_mult * blend
        new_palette.hue_shift = self.hue_shift * (1 - blend) + other.hue_shift * blend
        
        return new_palette
    
    def get_color(self, height: float) -> Color:
        """Get terrain color for a height value."""
        # Determine base color from height
        if height < -5:
            base = self.deep_water
        elif height < 2:
            t = (height + 5) / 7
            base = _blend_colors(self.deep_water, self.shallow_water, t)
        elif height < 4:
            t = (height - 2) / 2
            base = _blend_colors(self.shallow_water, self.beach, t)
        elif height < 15:
            t = (height - 4) / 11
            base = _blend_colors(self.beach, self.grass, t) if t < 0.3 else _blend_colors(self.grass, self.forest, (t - 0.3) / 0.7)
        elif height < 25:
            t = (height - 15) / 10
            base = _blend_colors(self.forest, self.hills, t)
        elif height < 35:
            t = (height - 25) / 10
            base = _blend_colors(self.hills, self.mountain, t)
        elif height < 50:
            t = (height - 35) / 15
            base = _blend_colors(self.mountain, self.snow, t)
        else:
            base = self.snow
        
        # Apply overall adjustments
        r, g, b = base
        
        # Hue shift (rotate RGB)
        if abs(self.hue_shift) > 0.001:
            # Simple hue rotation approximation
            shift = self.hue_shift
            r2 = r + shift * (g - b)
            g2 = g + shift * (b - r)
            b2 = b + shift * (r - g)
            r, g, b = _clamp(r2), _clamp(g2), _clamp(b2)
        
        # Saturation adjustment
        gray = (r + g + b) / 3
        r = gray + (r - gray) * self.saturation_mult
        g = gray + (g - gray) * self.saturation_mult
        b = gray + (b - gray) * self.saturation_mult
        
        # Brightness adjustment
        r *= self.brightness_mult
        g *= self.brightness_mult
        b *= self.brightness_mult
        
        return (_clamp(r), _clamp(g), _clamp(b))


@dataclass
class SkyDNA:
    """DNA for sky appearance - sun, moon, stars."""
    
    # Sun properties
    sun_size: float = 1.0  # 0.5 to 2.0
    sun_color: Color = (1.0, 0.95, 0.8)  # Core color
    sun_glow_color: Color = (1.0, 0.6, 0.2)  # Glow color
    sun_intensity: float = 1.0  # 0.7 to 1.3
    
    # Moon properties
    moon_size: float = 1.0  # 0.5 to 2.0
    moon_color: Color = (0.95, 0.95, 1.0)
    moon_glow: float = 0.3  # 0.1 to 0.5
    
    # Stars
    star_count_mult: float = 1.0  # 0.5 to 2.0 (multiplier on base count)
    star_brightness: float = 1.0  # 0.5 to 1.5
    star_color_variance: float = 0.3  # How much star colors vary
    star_twinkle_speed: float = 1.0  # 0.5 to 2.0
    
    # Sky colors (these affect the overall sky gradient)
    sky_day_tint: Color = (0.0, 0.0, 0.0)  # Added to day sky
    sky_night_tint: Color = (0.0, 0.0, 0.0)  # Added to night sky
    
    # Fog/atmosphere
    fog_density: float = 1.0  # 0.7 to 1.3
    fog_color_shift: Color = (0.0, 0.0, 0.0)
    
    def mutate(self, rng: np.random.Generator, strength: float = None) -> "SkyDNA":
        """Create mutated copy using global MUTATION_CONFIG."""
        cfg = MUTATION_CONFIG
        strength = strength or cfg.sky_mutation_strength
        
        # Check mutation chance
        if rng.random() > cfg.sky_mutation_chance:
            return copy.deepcopy(self)
        
        new_dna = copy.deepcopy(self)
        
        # Pick what to mutate - usually just one thing
        mutation_type = rng.choice(["sun", "moon", "stars", "sky", "fog"], 
                                    p=[0.2, 0.2, 0.3, 0.2, 0.1])
        
        if mutation_type == "sun":
            new_dna.sun_size = _clamp(self.sun_size + rng.normal(0, strength), 
                                       cfg.sun_size_range[0], cfg.sun_size_range[1])
            new_dna.sun_color = _mutate_color(self.sun_color, rng, strength)
            new_dna.sun_glow_color = _mutate_color(self.sun_glow_color, rng, strength)
            new_dna.sun_intensity = _clamp(self.sun_intensity + rng.normal(0, strength), 0.7, 1.3)
        
        elif mutation_type == "moon":
            new_dna.moon_size = _clamp(self.moon_size + rng.normal(0, strength), 
                                        cfg.moon_size_range[0], cfg.moon_size_range[1])
            new_dna.moon_color = _mutate_color(self.moon_color, rng, strength)
            new_dna.moon_glow = _clamp(self.moon_glow + rng.normal(0, strength), 0.1, 0.6)
        
        elif mutation_type == "stars":
            new_dna.star_count_mult = _clamp(self.star_count_mult + rng.normal(0, strength * 2), 0.5, 2.5)
            new_dna.star_brightness = _clamp(self.star_brightness + rng.normal(0, strength), 0.5, 1.5)
            new_dna.star_color_variance = _clamp(self.star_color_variance + rng.normal(0, strength), 0.1, 0.6)
            new_dna.star_twinkle_speed = _clamp(self.star_twinkle_speed + rng.normal(0, strength), 0.3, 2.0)
        
        elif mutation_type == "sky":
            new_dna.sky_day_tint = _mutate_color(self.sky_day_tint, rng, strength)
            new_dna.sky_night_tint = _mutate_color(self.sky_night_tint, rng, strength)
        
        elif mutation_type == "fog":
            new_dna.fog_density = _clamp(self.fog_density + rng.normal(0, strength), 0.6, 1.5)
            new_dna.fog_color_shift = _mutate_color(self.fog_color_shift, rng, strength)
        
        return new_dna
    
    def crossover(self, other: "SkyDNA", rng: np.random.Generator) -> "SkyDNA":
        """Blend two sky DNAs."""
        blend = 0.3 + rng.random() * 0.4
        
        new_dna = SkyDNA()
        
        # Blend all properties
        new_dna.sun_size = self.sun_size * (1 - blend) + other.sun_size * blend
        new_dna.sun_color = _blend_colors(self.sun_color, other.sun_color, blend)
        new_dna.sun_glow_color = _blend_colors(self.sun_glow_color, other.sun_glow_color, blend)
        new_dna.sun_intensity = self.sun_intensity * (1 - blend) + other.sun_intensity * blend
        
        new_dna.moon_size = self.moon_size * (1 - blend) + other.moon_size * blend
        new_dna.moon_color = _blend_colors(self.moon_color, other.moon_color, blend)
        new_dna.moon_glow = self.moon_glow * (1 - blend) + other.moon_glow * blend
        
        new_dna.star_count_mult = self.star_count_mult * (1 - blend) + other.star_count_mult * blend
        new_dna.star_brightness = self.star_brightness * (1 - blend) + other.star_brightness * blend
        new_dna.star_color_variance = self.star_color_variance * (1 - blend) + other.star_color_variance * blend
        new_dna.star_twinkle_speed = self.star_twinkle_speed * (1 - blend) + other.star_twinkle_speed * blend
        
        new_dna.sky_day_tint = _blend_colors(self.sky_day_tint, other.sky_day_tint, blend)
        new_dna.sky_night_tint = _blend_colors(self.sky_night_tint, other.sky_night_tint, blend)
        
        new_dna.fog_density = self.fog_density * (1 - blend) + other.fog_density * blend
        new_dna.fog_color_shift = _blend_colors(self.fog_color_shift, other.fog_color_shift, blend)
        
        return new_dna


@dataclass
class ChunkDNA:
    """Complete DNA for a chunk's visual appearance."""
    palette: TerrainPalette = field(default_factory=TerrainPalette)
    sky: SkyDNA = field(default_factory=SkyDNA)
    generation: int = 0
    
    def mutate(self, rng: np.random.Generator) -> "ChunkDNA":
        """Create mutated copy."""
        new_dna = ChunkDNA()
        new_dna.palette = self.palette.mutate(rng)
        new_dna.sky = self.sky.mutate(rng)
        new_dna.generation = self.generation + 1
        return new_dna
    
    def crossover(self, other: "ChunkDNA", rng: np.random.Generator) -> "ChunkDNA":
        """Create offspring from two parent DNAs."""
        new_dna = ChunkDNA()
        new_dna.palette = self.palette.crossover(other.palette, rng)
        new_dna.sky = self.sky.crossover(other.sky, rng)
        new_dna.generation = max(self.generation, other.generation) + 1
        return new_dna


class ChunkDNAManager:
    """Manages chunk DNA inheritance and evolution."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.chunk_dna: Dict[Tuple[int, int], ChunkDNA] = {}
        self.default_dna = ChunkDNA()
    
    def get_dna(self, cx: int, cz: int) -> ChunkDNA:
        """Get or create DNA for a chunk."""
        key = (cx, cz)
        
        if key in self.chunk_dna:
            return self.chunk_dna[key]
        
        # Create DNA based on neighbors
        chunk_seed = abs(hash((self.seed, cx, cz, "chunk_dna"))) % (2**31)
        rng = np.random.default_rng(chunk_seed)
        
        # Find neighboring chunks with DNA
        neighbors = []
        for dx in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dz == 0:
                    continue
                neighbor_key = (cx + dx, cz + dz)
                if neighbor_key in self.chunk_dna:
                    neighbors.append(self.chunk_dna[neighbor_key])
        
        if not neighbors:
            # First chunk - use default with tiny mutation
            new_dna = self.default_dna.mutate(rng)
        elif len(neighbors) == 1:
            # One neighbor - inherit and mutate
            new_dna = neighbors[0].mutate(rng)
        else:
            # Multiple neighbors - crossover two random ones, then mutate
            parent1, parent2 = rng.choice(neighbors, size=2, replace=False) if len(neighbors) >= 2 else (neighbors[0], neighbors[0])
            new_dna = parent1.crossover(parent2, rng).mutate(rng)
        
        self.chunk_dna[key] = new_dna
        return new_dna
    
    def cleanup_distant(self, center_cx: int, center_cz: int, max_distance: int = 30):
        """Remove DNA for distant chunks to save memory."""
        to_remove = [
            key for key in self.chunk_dna
            if abs(key[0] - center_cx) > max_distance or abs(key[1] - center_cz) > max_distance
        ]
        for key in to_remove:
            del self.chunk_dna[key]


# =============================================================================
# WIND DNA - Wind vectors per chunk with smooth gradients
# =============================================================================
@dataclass
class WindDNA:
    """Wind pattern for a chunk - grid of wind vectors with smooth gradients."""
    
    # Grid of wind vectors (direction in radians, magnitude 0-1)
    # Shape: (grid_size, grid_size, 2) where [..., 0] = direction, [..., 1] = magnitude
    grid_size: int = 4  # 4x4 = 16 wind sample points per chunk
    
    # Base wind parameters
    base_direction: float = 0.0  # Radians (0 = north, pi/2 = east)
    base_magnitude: float = 0.3  # Average wind strength (0-1)
    magnitude_variation: float = 0.2  # How much magnitude varies across chunk
    direction_variation: float = 0.3  # How much direction varies (radians)
    
    # Cycle parameters (rotating winds)
    has_cycle: bool = False
    cycle_speed: float = 0.0  # Radians per second (0 = no cycle)
    cycle_center: Tuple[float, float] = (0.5, 0.5)  # Center of rotation (0-1, 0-1)
    
    # Gust parameters
    gust_chance: float = 0.1  # Chance of gusts
    gust_strength: float = 0.3  # Additional magnitude during gusts
    
    # The actual wind grid (computed from parameters)
    wind_grid: np.ndarray = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.wind_grid is None:
            self._generate_grid()
    
    def _generate_grid(self):
        """Generate wind vector grid from parameters."""
        self.wind_grid = np.zeros((self.grid_size, self.grid_size, 2), dtype=np.float32)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Normalized position (0-1)
                px = i / (self.grid_size - 1) if self.grid_size > 1 else 0.5
                py = j / (self.grid_size - 1) if self.grid_size > 1 else 0.5
                
                # Direction with smooth variation
                dir_offset = np.sin(px * np.pi * 2) * np.cos(py * np.pi) * self.direction_variation
                direction = self.base_direction + dir_offset
                
                # Add rotation for cyclic wind
                if self.has_cycle:
                    dx = px - self.cycle_center[0]
                    dy = py - self.cycle_center[1]
                    # Tangent direction for circular motion
                    tangent = np.arctan2(-dx, dy)
                    direction = direction * 0.5 + tangent * 0.5
                
                # Magnitude with smooth variation
                mag_offset = np.sin(px * np.pi * 1.5) * np.sin(py * np.pi * 1.5) * self.magnitude_variation
                magnitude = np.clip(self.base_magnitude + mag_offset, 0.05, 1.0)
                
                self.wind_grid[i, j, 0] = direction
                self.wind_grid[i, j, 1] = magnitude
    
    def get_wind_at(self, local_x: float, local_z: float, time: float = 0.0) -> Tuple[float, float, float]:
        """Get interpolated wind at a local position (0-1, 0-1). Returns (dir, mag, gust)."""
        # Clamp to valid range
        local_x = np.clip(local_x, 0.0, 1.0)
        local_z = np.clip(local_z, 0.0, 1.0)
        
        # Grid coordinates
        gx = local_x * (self.grid_size - 1)
        gz = local_z * (self.grid_size - 1)
        
        # Bilinear interpolation
        x0, x1 = int(gx), min(int(gx) + 1, self.grid_size - 1)
        z0, z1 = int(gz), min(int(gz) + 1, self.grid_size - 1)
        
        fx = gx - x0
        fz = gz - z0
        
        # Interpolate direction (needs special handling for angle wraparound)
        d00 = self.wind_grid[x0, z0, 0]
        d01 = self.wind_grid[x0, z1, 0]
        d10 = self.wind_grid[x1, z0, 0]
        d11 = self.wind_grid[x1, z1, 0]
        
        # Average angles properly
        direction = self._interpolate_angles(d00, d01, d10, d11, fx, fz)
        
        # Add time-based cycle rotation
        if self.has_cycle and self.cycle_speed != 0:
            direction += time * self.cycle_speed
        
        # Interpolate magnitude
        m00 = self.wind_grid[x0, z0, 1]
        m01 = self.wind_grid[x0, z1, 1]
        m10 = self.wind_grid[x1, z0, 1]
        m11 = self.wind_grid[x1, z1, 1]
        
        magnitude = (m00 * (1-fx) * (1-fz) + m01 * (1-fx) * fz + 
                    m10 * fx * (1-fz) + m11 * fx * fz)
        
        # Gust calculation (time-based noise)
        gust = 0.0
        if self.gust_chance > 0:
            gust_noise = np.sin(time * 2.0 + local_x * 10 + local_z * 10) * 0.5 + 0.5
            if gust_noise > (1.0 - self.gust_chance):
                gust = self.gust_strength * gust_noise
        
        return direction, magnitude, gust
    
    def _interpolate_angles(self, a00, a01, a10, a11, fx, fz):
        """Bilinear interpolation of angles (handles wraparound)."""
        # Convert to unit vectors, interpolate, convert back
        def angle_to_vec(a):
            return np.cos(a), np.sin(a)
        
        v00 = angle_to_vec(a00)
        v01 = angle_to_vec(a01)
        v10 = angle_to_vec(a10)
        v11 = angle_to_vec(a11)
        
        vx = (v00[0] * (1-fx) * (1-fz) + v01[0] * (1-fx) * fz + 
              v10[0] * fx * (1-fz) + v11[0] * fx * fz)
        vy = (v00[1] * (1-fx) * (1-fz) + v01[1] * (1-fx) * fz + 
              v10[1] * fx * (1-fz) + v11[1] * fx * fz)
        
        return np.arctan2(vy, vx)
    
    def mutate(self, rng: np.random.Generator) -> "WindDNA":
        """Create mutated copy."""
        new_dna = WindDNA(grid_size=self.grid_size)
        
        # 20% chance to mutate each parameter
        mutation_rate = 0.2
        
        # Direction drifts slowly
        if rng.random() < mutation_rate:
            new_dna.base_direction = self.base_direction + rng.normal(0, 0.2)
        else:
            new_dna.base_direction = self.base_direction
        
        # Magnitude changes gradually
        if rng.random() < mutation_rate:
            new_dna.base_magnitude = np.clip(self.base_magnitude + rng.normal(0, 0.1), 0.05, 0.8)
        else:
            new_dna.base_magnitude = self.base_magnitude
        
        # Variation parameters
        new_dna.magnitude_variation = np.clip(self.magnitude_variation + rng.normal(0, 0.05), 0.0, 0.4)
        new_dna.direction_variation = np.clip(self.direction_variation + rng.normal(0, 0.05), 0.0, 0.6)
        
        # Cycle - small chance to gain/lose
        if rng.random() < 0.05:  # 5% chance to toggle cycle
            new_dna.has_cycle = not self.has_cycle
            if new_dna.has_cycle:
                new_dna.cycle_speed = rng.uniform(0.1, 0.5) * rng.choice([-1, 1])
                new_dna.cycle_center = (rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7))
        else:
            new_dna.has_cycle = self.has_cycle
            new_dna.cycle_speed = self.cycle_speed
            new_dna.cycle_center = self.cycle_center
        
        # Gust parameters
        new_dna.gust_chance = np.clip(self.gust_chance + rng.normal(0, 0.02), 0.0, 0.3)
        new_dna.gust_strength = np.clip(self.gust_strength + rng.normal(0, 0.05), 0.0, 0.5)
        
        # Regenerate grid
        new_dna._generate_grid()
        
        return new_dna
    
    def crossover(self, other: "WindDNA", rng: np.random.Generator) -> "WindDNA":
        """Create offspring from two parent DNAs."""
        new_dna = WindDNA(grid_size=self.grid_size)
        blend = rng.uniform(0.3, 0.7)
        
        # Blend parameters
        new_dna.base_direction = self._blend_angles(self.base_direction, other.base_direction, blend)
        new_dna.base_magnitude = self.base_magnitude * (1 - blend) + other.base_magnitude * blend
        new_dna.magnitude_variation = self.magnitude_variation * (1 - blend) + other.magnitude_variation * blend
        new_dna.direction_variation = self.direction_variation * (1 - blend) + other.direction_variation * blend
        
        # Cycle from one parent
        if rng.random() < blend:
            new_dna.has_cycle = other.has_cycle
            new_dna.cycle_speed = other.cycle_speed
            new_dna.cycle_center = other.cycle_center
        else:
            new_dna.has_cycle = self.has_cycle
            new_dna.cycle_speed = self.cycle_speed
            new_dna.cycle_center = self.cycle_center
        
        # Gust parameters
        new_dna.gust_chance = self.gust_chance * (1 - blend) + other.gust_chance * blend
        new_dna.gust_strength = self.gust_strength * (1 - blend) + other.gust_strength * blend
        
        # Regenerate grid
        new_dna._generate_grid()
        
        return new_dna
    
    def _blend_angles(self, a1: float, a2: float, blend: float) -> float:
        """Blend two angles properly."""
        v1 = (np.cos(a1), np.sin(a1))
        v2 = (np.cos(a2), np.sin(a2))
        vx = v1[0] * (1 - blend) + v2[0] * blend
        vy = v1[1] * (1 - blend) + v2[1] * blend
        return np.arctan2(vy, vx)
    
    @staticmethod
    def create_random(rng: np.random.Generator) -> "WindDNA":
        """Create a random wind pattern."""
        dna = WindDNA()
        dna.base_direction = rng.uniform(0, 2 * np.pi)
        dna.base_magnitude = rng.uniform(0.1, 0.6)
        dna.magnitude_variation = rng.uniform(0.05, 0.25)
        dna.direction_variation = rng.uniform(0.1, 0.4)
        dna.has_cycle = rng.random() < 0.15  # 15% have cycles
        if dna.has_cycle:
            dna.cycle_speed = rng.uniform(0.1, 0.4) * rng.choice([-1, 1])
            dna.cycle_center = (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))
        dna.gust_chance = rng.uniform(0.0, 0.2)
        dna.gust_strength = rng.uniform(0.1, 0.3)
        dna._generate_grid()
        return dna


class WindManager:
    """Manages wind DNA per chunk and provides wind queries."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.chunk_wind: Dict[Tuple[int, int], WindDNA] = {}
        self.default_wind = WindDNA()
        self.time = 0.0
    
    def update(self, dt: float):
        """Update wind time."""
        self.time += dt
    
    def get_wind_dna(self, cx: int, cz: int) -> WindDNA:
        """Get or create wind DNA for a chunk."""
        key = (cx, cz)
        
        if key in self.chunk_wind:
            return self.chunk_wind[key]
        
        # Create from neighbors
        chunk_seed = abs(hash((self.seed, cx, cz, "wind"))) % (2**31)
        rng = np.random.default_rng(chunk_seed)
        
        neighbors = []
        for dx in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dz == 0:
                    continue
                neighbor_key = (cx + dx, cz + dz)
                if neighbor_key in self.chunk_wind:
                    neighbors.append(self.chunk_wind[neighbor_key])
        
        if not neighbors:
            new_wind = WindDNA.create_random(rng)
        elif len(neighbors) == 1:
            new_wind = neighbors[0].mutate(rng)
        else:
            parents = rng.choice(neighbors, size=min(2, len(neighbors)), replace=False)
            if len(parents) == 2:
                new_wind = parents[0].crossover(parents[1], rng).mutate(rng)
            else:
                new_wind = parents[0].mutate(rng)
        
        self.chunk_wind[key] = new_wind
        return new_wind
    
    def get_wind_at_world(self, world_x: float, world_z: float, chunk_size: float) -> Tuple[float, float, float]:
        """Get wind at world coordinates. Returns (direction, magnitude, gust)."""
        # Calculate chunk coordinates
        cx = int(world_x // chunk_size)
        cz = int(world_z // chunk_size)
        
        # Local position within chunk (0-1)
        local_x = (world_x % chunk_size) / chunk_size
        local_z = (world_z % chunk_size) / chunk_size
        
        # Get wind from this chunk
        wind_dna = self.get_wind_dna(cx, cz)
        return wind_dna.get_wind_at(local_x, local_z, self.time)
    
    def get_wind_vector(self, world_x: float, world_z: float, chunk_size: float) -> Tuple[float, float]:
        """Get wind as (vx, vz) vector."""
        direction, magnitude, gust = self.get_wind_at_world(world_x, world_z, chunk_size)
        total_mag = magnitude + gust
        return (np.cos(direction) * total_mag, np.sin(direction) * total_mag)
    
    def cleanup_distant(self, center_cx: int, center_cz: int, max_distance: int = 30):
        """Remove wind data for distant chunks."""
        to_remove = [
            key for key in self.chunk_wind
            if abs(key[0] - center_cx) > max_distance or abs(key[1] - center_cz) > max_distance
        ]
        for key in to_remove:
            del self.chunk_wind[key]

