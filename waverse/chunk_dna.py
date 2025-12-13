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
    
    def mutate(self, rng: np.random.Generator, strength: float = 0.02) -> "TerrainPalette":
        """Create mutated copy. Very low strength for gradual change."""
        # Only mutate sometimes
        if rng.random() > 0.15:  # 85% chance of NO mutation
            return copy.deepcopy(self)
        
        new_palette = copy.deepcopy(self)
        
        # Very subtle color shifts
        rate = strength * 0.5
        new_palette.deep_water = _mutate_color(self.deep_water, rng, rate)
        new_palette.shallow_water = _mutate_color(self.shallow_water, rng, rate)
        new_palette.beach = _mutate_color(self.beach, rng, rate)
        new_palette.grass = _mutate_color(self.grass, rng, rate)
        new_palette.forest = _mutate_color(self.forest, rng, rate)
        new_palette.hills = _mutate_color(self.hills, rng, rate)
        new_palette.mountain = _mutate_color(self.mountain, rng, rate)
        new_palette.snow = _mutate_color(self.snow, rng, rate)
        
        # Overall adjustments - allow wider ranges for psychedelic areas
        new_palette.saturation_mult = _clamp(self.saturation_mult + rng.normal(0, 0.03), 0.5, 1.8)
        new_palette.brightness_mult = _clamp(self.brightness_mult + rng.normal(0, 0.02), 0.6, 1.4)
        # Hue shift can go full range for truly psychedelic colors
        new_palette.hue_shift = _clamp(self.hue_shift + rng.normal(0, 0.015), -0.5, 0.5)
        
        # Rare "burst" mutation - occasionally make a bigger color jump
        if rng.random() < 0.05:  # 5% chance
            burst_color = rng.choice(["grass", "forest", "hills", "mountain"])
            burst_shift = rng.normal(0, 0.15)  # Larger shift
            if burst_color == "grass":
                new_palette.grass = tuple(_clamp(c + burst_shift) for c in new_palette.grass)
            elif burst_color == "forest":
                new_palette.forest = tuple(_clamp(c + burst_shift) for c in new_palette.forest)
            elif burst_color == "hills":
                new_palette.hills = tuple(_clamp(c + burst_shift) for c in new_palette.hills)
            elif burst_color == "mountain":
                new_palette.mountain = tuple(_clamp(c + burst_shift) for c in new_palette.mountain)
        
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
    
    def mutate(self, rng: np.random.Generator, strength: float = 0.01) -> "SkyDNA":
        """
        Create mutated copy. 
        VERY low mutation rate - changes should span dozens of chunks.
        """
        # Only mutate 5% of the time for ultra-slow evolution
        if rng.random() > 0.05:
            return copy.deepcopy(self)
        
        new_dna = copy.deepcopy(self)
        
        # Pick what to mutate - usually just one thing
        mutation_type = rng.choice(["sun", "moon", "stars", "sky", "fog"], 
                                    p=[0.2, 0.2, 0.3, 0.2, 0.1])
        
        tiny = strength * 0.3  # Even smaller for astral bodies
        
        if mutation_type == "sun":
            new_dna.sun_size = _clamp(self.sun_size + rng.normal(0, tiny), 0.5, 2.0)
            new_dna.sun_color = _mutate_color(self.sun_color, rng, tiny)
            new_dna.sun_glow_color = _mutate_color(self.sun_glow_color, rng, tiny)
            new_dna.sun_intensity = _clamp(self.sun_intensity + rng.normal(0, tiny), 0.7, 1.3)
        
        elif mutation_type == "moon":
            new_dna.moon_size = _clamp(self.moon_size + rng.normal(0, tiny), 0.5, 2.0)
            new_dna.moon_color = _mutate_color(self.moon_color, rng, tiny)
            new_dna.moon_glow = _clamp(self.moon_glow + rng.normal(0, tiny), 0.1, 0.6)
        
        elif mutation_type == "stars":
            new_dna.star_count_mult = _clamp(self.star_count_mult + rng.normal(0, tiny * 2), 0.5, 2.5)
            new_dna.star_brightness = _clamp(self.star_brightness + rng.normal(0, tiny), 0.5, 1.5)
            new_dna.star_color_variance = _clamp(self.star_color_variance + rng.normal(0, tiny), 0.1, 0.6)
            new_dna.star_twinkle_speed = _clamp(self.star_twinkle_speed + rng.normal(0, tiny), 0.3, 2.0)
        
        elif mutation_type == "sky":
            new_dna.sky_day_tint = _mutate_color(self.sky_day_tint, rng, tiny)
            new_dna.sky_night_tint = _mutate_color(self.sky_night_tint, rng, tiny)
        
        elif mutation_type == "fog":
            new_dna.fog_density = _clamp(self.fog_density + rng.normal(0, tiny), 0.6, 1.5)
            new_dna.fog_color_shift = _mutate_color(self.fog_color_shift, rng, tiny)
        
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

