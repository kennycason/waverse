"""
Climate and Weather System.

Two layers:
1. Biome DNA - Very slow evolution, determines base climate (snowy, tropical, temperate, arid)
2. Weather State - Faster changes, current conditions (rain, snow, clear, cloudy)

Visual effects are kept performant with particle pooling and distance culling.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import numpy as np
from OpenGL.GL import *


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class BiomeDNA:
    """
    Biome characteristics - very slow evolution across chunks.
    These define the "permanent" climate of an area.
    """
    # Core biome parameters (0-1 scale)
    temperature: float = 0.5      # 0=frozen, 0.5=temperate, 1=hot
    humidity: float = 0.6         # 0=arid, 0.5=normal, 1=tropical (higher default)
    elevation_factor: float = 0.5 # Affects temperature at height
    
    # Weather tendencies (how likely certain weather is) - higher defaults
    rain_tendency: float = 0.5    # Base chance of rain (was 0.3)
    snow_tendency: float = 0.2    # Base chance of snow (was 0.1)
    storm_tendency: float = 0.25  # Base chance of storms (was 0.1)
    cloud_base: float = 0.4       # Base cloud cover (was 0.3)
    wind_base: float = 0.3        # Base wind strength
    
    # SPECIAL BIOMES - rare exotic zones!
    # None = normal biome, otherwise overrides everything
    # Options: "psychedelic", "hellfire", "shadow", "crystal", "void", "mountain", "deep_ocean"
    special_biome: str = None
    chaos_factor: float = 0.0  # 0=normal terrain, 1=extremely chaotic/jagged
    
    # Height modifiers for special terrain biomes
    height_multiplier: float = 1.0  # 1=normal, 3+=mountains, 0.3=deep ocean
    height_offset: float = 0.0      # Added to all heights (negative for oceans)
    
    def mutate(self, rng: np.random.Generator, strength: float = 0.02) -> "BiomeDNA":
        """Very slow mutation - biomes should span many chunks."""
        # Special biomes are sticky - they persist across the zone
        new_special = self.special_biome
        new_chaos = self.chaos_factor
        
        # Check for debug mode (more frequent biome changes for testing)
        import os
        debug_biomes = os.environ.get('WAVERSE_DEBUG_BIOMES', '0') == '1'
        demo_mode = os.environ.get('WAVERSE_DEMO_BIOMES', '0') == '1'
        
        # Chance to exit or enter a special biome
        # Demo mode: Larger, more contiguous exotic biomes
        # Debug mode: 10% chance, Normal mode: 0.5% chance
        if demo_mode:
            change_chance = 0.02  # 2% chance - biomes ~50 chunks wide (stable areas!)
            enter_chance = 0.85   # 85% chance the new biome is exotic
        elif debug_biomes:
            change_chance = 0.10
            enter_chance = 0.50
        else:
            change_chance = 0.005
            enter_chance = 0.02
        
        if rng.random() < change_chance:
            if self.special_biome:
                # Exiting special biome
                new_special = None
                new_chaos = 0.0
            else:
                # Chance to enter a special biome
                if rng.random() < enter_chance:
                    # Exotic biomes only - mountain/ocean removed (should be natural terrain)
                    new_special = rng.choice(['psychedelic', 'psychedelic', 'psychedelic',
                                             'hellfire', 'shadow', 'crystal', 'void'])
                    new_chaos = 0.3 + rng.random() * 0.4  # Moderate chaos
        
        # If in special biome, chaos can vary
        if new_special:
            new_chaos = _clamp(new_chaos + rng.normal(0, 0.1))
        
        # Height modifiers removed - mountains/oceans are natural terrain features now
        new_height_mult = 1.0
        new_height_offset = 0.0
        
        # Only 8% chance of any mutation for normal params
        if rng.random() > 0.08:
            return BiomeDNA(
                temperature=self.temperature,
                humidity=self.humidity,
                elevation_factor=self.elevation_factor,
                rain_tendency=self.rain_tendency,
                snow_tendency=self.snow_tendency,
                storm_tendency=self.storm_tendency,
                cloud_base=self.cloud_base,
                wind_base=self.wind_base,
                special_biome=new_special,
                chaos_factor=new_chaos,
                height_multiplier=new_height_mult,
                height_offset=new_height_offset,
            )
        
        # Tiny mutations
        s = strength * 0.5
        return BiomeDNA(
            temperature=_clamp(self.temperature + rng.normal(0, s)),
            humidity=_clamp(self.humidity + rng.normal(0, s)),
            elevation_factor=_clamp(self.elevation_factor + rng.normal(0, s * 0.5)),
            rain_tendency=_clamp(self.rain_tendency + rng.normal(0, s)),
            snow_tendency=_clamp(self.snow_tendency + rng.normal(0, s)),
            storm_tendency=_clamp(self.storm_tendency + rng.normal(0, s * 0.5)),
            cloud_base=_clamp(self.cloud_base + rng.normal(0, s)),
            wind_base=_clamp(self.wind_base + rng.normal(0, s)),
            special_biome=new_special,
            chaos_factor=new_chaos,
            height_multiplier=new_height_mult,
            height_offset=new_height_offset,
        )
    
    def crossover(self, other: "BiomeDNA", rng: np.random.Generator) -> "BiomeDNA":
        """Blend two biome DNAs."""
        t = 0.3 + rng.random() * 0.4
        
        # Special biomes: prefer the one that exists, or pick randomly if both exist
        if self.special_biome and other.special_biome:
            new_special = rng.choice([self.special_biome, other.special_biome])
        else:
            new_special = self.special_biome or other.special_biome
        
        # Blend chaos and height modifiers
        new_chaos = self.chaos_factor * (1-t) + other.chaos_factor * t
        new_height_mult = self.height_multiplier * (1-t) + other.height_multiplier * t
        new_height_offset = self.height_offset * (1-t) + other.height_offset * t
        
        return BiomeDNA(
            temperature=self.temperature * (1-t) + other.temperature * t,
            humidity=self.humidity * (1-t) + other.humidity * t,
            elevation_factor=self.elevation_factor * (1-t) + other.elevation_factor * t,
            rain_tendency=self.rain_tendency * (1-t) + other.rain_tendency * t,
            snow_tendency=self.snow_tendency * (1-t) + other.snow_tendency * t,
            storm_tendency=self.storm_tendency * (1-t) + other.storm_tendency * t,
            cloud_base=self.cloud_base * (1-t) + other.cloud_base * t,
            wind_base=self.wind_base * (1-t) + other.wind_base * t,
            special_biome=new_special,
            chaos_factor=new_chaos,
            height_multiplier=new_height_mult,
            height_offset=new_height_offset,
        )
    
    def get_biome_name(self) -> str:
        """Get human-readable biome name."""
        # Special exotic biomes override normal temperature/humidity logic
        if self.special_biome:
            return self.special_biome.capitalize()
        
        if self.temperature < 0.25:
            if self.humidity > 0.5:
                return "Tundra"
            return "Frozen"
        elif self.temperature < 0.45:
            if self.humidity > 0.6:
                return "Taiga"
            return "Cold"
        elif self.temperature < 0.65:
            if self.humidity > 0.7:
                return "Rainforest"
            elif self.humidity > 0.4:
                return "Temperate"
            return "Grassland"
        else:
            if self.humidity > 0.6:
                return "Tropical"
            elif self.humidity < 0.3:
                return "Desert"
            return "Savanna"


@dataclass
class WeatherVisualDNA:
    """
    Visual DNA for weather effects - evolves across regions.
    Defines colors, shapes, and patterns for rain/snow/clouds.
    """
    # Rain visuals
    rain_color: Tuple[float, float, float] = (0.6, 0.7, 0.9)  # RGB
    rain_alpha: float = 0.6
    rain_waviness: float = 0.3      # 0=straight, 1=very wavy
    rain_streak_length: float = 1.5  # Length of rain streaks
    
    # Snow visuals
    snow_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    snow_size: float = 3.0          # Base snowflake size
    snow_drift: float = 0.5         # How much snow drifts sideways
    
    # Cloud visuals
    cloud_color: Tuple[float, float, float] = (0.9, 0.9, 0.92)
    cloud_speed: float = 0.01       # How fast clouds move
    cloud_size_base: float = 80.0   # Base cloud size
    cloud_size_variance: float = 30.0  # Size variation
    cloud_count: int = 8            # Number of cloud patches
    cloud_height_offset: float = 150.0  # Height above camera
    
    def mutate(self, rng: np.random.Generator, strength: float = 0.05) -> "WeatherVisualDNA":
        """Mutate weather visuals."""
        if rng.random() > 0.15:  # 15% chance of mutation
            return WeatherVisualDNA(
                rain_color=self.rain_color,
                rain_alpha=self.rain_alpha,
                rain_waviness=self.rain_waviness,
                rain_streak_length=self.rain_streak_length,
                snow_color=self.snow_color,
                snow_size=self.snow_size,
                snow_drift=self.snow_drift,
                cloud_color=self.cloud_color,
                cloud_speed=self.cloud_speed,
                cloud_size_base=self.cloud_size_base,
                cloud_size_variance=self.cloud_size_variance,
                cloud_count=self.cloud_count,
                cloud_height_offset=self.cloud_height_offset,
            )
        
        s = strength
        return WeatherVisualDNA(
            rain_color=tuple(_clamp(c + rng.normal(0, s * 0.3)) for c in self.rain_color),
            rain_alpha=_clamp(self.rain_alpha + rng.normal(0, s * 0.2), 0.3, 0.9),
            rain_waviness=_clamp(self.rain_waviness + rng.normal(0, s), 0.0, 0.8),
            rain_streak_length=_clamp(self.rain_streak_length + rng.normal(0, s * 0.5), 0.8, 3.0),
            snow_color=tuple(_clamp(c + rng.normal(0, s * 0.1), 0.8, 1.0) for c in self.snow_color),
            snow_size=_clamp(self.snow_size + rng.normal(0, s * 2), 2.0, 6.0),
            snow_drift=_clamp(self.snow_drift + rng.normal(0, s), 0.2, 1.0),
            cloud_color=tuple(_clamp(c + rng.normal(0, s * 0.15), 0.5, 1.0) for c in self.cloud_color),
            cloud_speed=_clamp(self.cloud_speed + rng.normal(0, s * 0.01), 0.005, 0.03),
            cloud_size_base=_clamp(self.cloud_size_base + rng.normal(0, s * 20), 50.0, 120.0),
            cloud_size_variance=_clamp(self.cloud_size_variance + rng.normal(0, s * 10), 10.0, 50.0),
            cloud_count=int(_clamp(self.cloud_count + rng.normal(0, s * 2), 4, 12)),
            cloud_height_offset=_clamp(self.cloud_height_offset + rng.normal(0, s * 30), 100.0, 250.0),
        )
    
    def crossover(self, other: "WeatherVisualDNA", rng: np.random.Generator) -> "WeatherVisualDNA":
        """Blend two weather visual DNAs."""
        t = 0.3 + rng.random() * 0.4
        return WeatherVisualDNA(
            rain_color=tuple(self.rain_color[i] * (1-t) + other.rain_color[i] * t for i in range(3)),
            rain_alpha=self.rain_alpha * (1-t) + other.rain_alpha * t,
            rain_waviness=self.rain_waviness * (1-t) + other.rain_waviness * t,
            rain_streak_length=self.rain_streak_length * (1-t) + other.rain_streak_length * t,
            snow_color=tuple(self.snow_color[i] * (1-t) + other.snow_color[i] * t for i in range(3)),
            snow_size=self.snow_size * (1-t) + other.snow_size * t,
            snow_drift=self.snow_drift * (1-t) + other.snow_drift * t,
            cloud_color=tuple(self.cloud_color[i] * (1-t) + other.cloud_color[i] * t for i in range(3)),
            cloud_speed=self.cloud_speed * (1-t) + other.cloud_speed * t,
            cloud_size_base=self.cloud_size_base * (1-t) + other.cloud_size_base * t,
            cloud_size_variance=self.cloud_size_variance * (1-t) + other.cloud_size_variance * t,
            cloud_count=int(self.cloud_count * (1-t) + other.cloud_count * t),
            cloud_height_offset=self.cloud_height_offset * (1-t) + other.cloud_height_offset * t,
        )


@dataclass
class WeatherState:
    """Current weather conditions - changes over time."""
    precipitation: float = 0.0     # 0=none, 0.5=light, 1=heavy
    precipitation_type: str = "none"  # none, rain, snow, sleet
    cloud_cover: float = 0.3       # 0=clear, 1=overcast
    wind_strength: float = 0.2     # 0=calm, 1=strong
    wind_direction: float = 0.0    # Radians
    lightning_active: bool = False
    fog_density: float = 0.0       # 0=clear, 1=thick fog
    
    # Visual DNA for this weather region
    visual_dna: WeatherVisualDNA = field(default_factory=WeatherVisualDNA)
    
    # Transition tracking
    target_precipitation: float = 0.0
    target_clouds: float = 0.3
    weather_timer: float = 0.0     # Time until next weather change


class ClimateManager:
    """Manages biome DNA and weather per region (groups of chunks)."""
    
    # Weather is regional - covers 4x4 chunk areas for consistency
    WEATHER_REGION_SIZE = 4
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.biomes: Dict[Tuple[int, int], BiomeDNA] = {}
        self.regional_weather: Dict[Tuple[int, int], WeatherState] = {}  # Per region, not per chunk
        self.current_chunk = (0, 0)
        self.current_biome = BiomeDNA()
        self.current_weather = WeatherState()
        self.time = 0.0
        
        # Lightning flash state
        self.lightning_flash = 0.0
        self.lightning_timer = 0.0
    
    def get_biome(self, cx: int, cz: int) -> BiomeDNA:
        """Get or create biome DNA for a chunk."""
        key = (cx, cz)
        if key in self.biomes:
            return self.biomes[key]
        
        # Create from neighbors
        chunk_seed = abs(hash((self.seed, cx, cz, "biome"))) % (2**31)
        rng = np.random.default_rng(chunk_seed)
        
        neighbors = []
        for dx in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dz == 0:
                    continue
                nkey = (cx + dx, cz + dz)
                if nkey in self.biomes:
                    neighbors.append(self.biomes[nkey])
        
        if not neighbors:
            # Create initial biome based on world position
            # Use large-scale noise for biome distribution
            world_x = cx * 32
            world_z = cz * 32
            temp_noise = math.sin(world_x * 0.0003) * 0.3 + math.cos(world_z * 0.0004) * 0.2
            humid_noise = math.cos(world_x * 0.0002 + 1) * 0.3 + math.sin(world_z * 0.0003) * 0.2
            
            new_biome = BiomeDNA(
                temperature=_clamp(0.5 + temp_noise + rng.normal(0, 0.1)),
                humidity=_clamp(0.5 + humid_noise + rng.normal(0, 0.1)),
                rain_tendency=_clamp(0.3 + humid_noise * 0.3),
                snow_tendency=_clamp(0.1 - temp_noise * 0.2),
            )
        elif len(neighbors) == 1:
            new_biome = neighbors[0].mutate(rng)
        else:
            p1, p2 = rng.choice(neighbors, 2, replace=len(neighbors) < 2)
            new_biome = p1.crossover(p2, rng).mutate(rng)
        
        self.biomes[key] = new_biome
        return new_biome
    
    def get_blended_biome(self, world_x: float, world_z: float, blend_radius: int = 2) -> BiomeDNA:
        """
        Get a BLENDED biome at a specific world position.
        Samples nearby chunks and interpolates based on distance.
        This creates smooth transitions between biomes!
        
        Args:
            world_x, world_z: World position
            blend_radius: How many chunks to sample in each direction
        
        Returns:
            A blended BiomeDNA that may mix properties from multiple biomes
        """
        chunk_size = 32  # Standard chunk size
        cx = int(world_x // chunk_size)
        cz = int(world_z // chunk_size)
        
        # Position within chunk (0-1)
        local_x = (world_x % chunk_size) / chunk_size
        local_z = (world_z % chunk_size) / chunk_size
        
        # Sample surrounding biomes with distance-based weights
        total_weight = 0.0
        blended = {
            'temperature': 0.0, 'humidity': 0.0, 'elevation_factor': 0.0,
            'rain_tendency': 0.0, 'snow_tendency': 0.0, 'storm_tendency': 0.0,
            'cloud_base': 0.0, 'wind_base': 0.0, 'chaos_factor': 0.0
        }
        special_biomes = {}  # Track special biomes and their weights
        
        for dx in range(-blend_radius, blend_radius + 1):
            for dz in range(-blend_radius, blend_radius + 1):
                biome = self.get_biome(cx + dx, cz + dz)
                
                # Distance from position to chunk center
                chunk_center_x = (cx + dx + 0.5)
                chunk_center_z = (cz + dz + 0.5)
                pos_x = cx + local_x
                pos_z = cz + local_z
                dist = math.sqrt((chunk_center_x - pos_x)**2 + (chunk_center_z - pos_z)**2)
                
                # Weight: closer = stronger influence (inverse square falloff)
                weight = 1.0 / (1.0 + dist * dist)
                total_weight += weight
                
                # Accumulate blended values
                blended['temperature'] += biome.temperature * weight
                blended['humidity'] += biome.humidity * weight
                blended['elevation_factor'] += biome.elevation_factor * weight
                blended['rain_tendency'] += biome.rain_tendency * weight
                blended['snow_tendency'] += biome.snow_tendency * weight
                blended['storm_tendency'] += biome.storm_tendency * weight
                blended['cloud_base'] += biome.cloud_base * weight
                blended['wind_base'] += biome.wind_base * weight
                blended['chaos_factor'] += getattr(biome, 'chaos_factor', 0.0) * weight
                
                # Track special biomes
                if biome.special_biome:
                    if biome.special_biome not in special_biomes:
                        special_biomes[biome.special_biome] = 0.0
                    special_biomes[biome.special_biome] += weight
        
        # Normalize
        if total_weight > 0:
            for key in blended:
                blended[key] /= total_weight
        
        # Determine special biome: use the one with highest weight, if any
        final_special = None
        if special_biomes:
            dominant = max(special_biomes.items(), key=lambda x: x[1])
            # Only set if it has significant presence (>30% of weight)
            if dominant[1] / total_weight > 0.3:
                final_special = dominant[0]
        
        return BiomeDNA(
            temperature=blended['temperature'],
            humidity=blended['humidity'],
            elevation_factor=blended['elevation_factor'],
            rain_tendency=blended['rain_tendency'],
            snow_tendency=blended['snow_tendency'],
            storm_tendency=blended['storm_tendency'],
            cloud_base=blended['cloud_base'],
            wind_base=blended['wind_base'],
            special_biome=final_special,
            chaos_factor=blended['chaos_factor']
        )
    
    def _chunk_to_region(self, cx: int, cz: int) -> Tuple[int, int]:
        """Convert chunk coords to weather region coords."""
        return (cx // self.WEATHER_REGION_SIZE, cz // self.WEATHER_REGION_SIZE)
    
    def get_weather(self, cx: int, cz: int) -> WeatherState:
        """Get or create weather state for a region (group of chunks)."""
        region_key = self._chunk_to_region(cx, cz)
        
        if region_key not in self.regional_weather:
            # Get average biome for this region
            biome = self.get_biome(cx, cz)
            new_weather = WeatherState(
                cloud_cover=biome.cloud_base + random.random() * 0.3,
                wind_strength=biome.wind_base + random.random() * 0.2,
                wind_direction=random.random() * math.pi * 2,
                weather_timer=5 + random.random() * 15,  # Quick first roll (5-20 seconds)
            )
            # Immediately roll weather so regions start with weather
            self._roll_new_weather(new_weather, biome)
            self.regional_weather[region_key] = new_weather
            
            # Debug: show new region weather
            if new_weather.precipitation_type != "none":
                print(f"  Weather: {new_weather.precipitation_type} ({new_weather.target_precipitation:.1%}) in region {region_key}")
        return self.regional_weather[region_key]
    
    def update(self, dt: float, cx: int, cz: int):
        """Update weather for current chunk."""
        self.time += dt / 60.0  # Convert to seconds
        self.current_chunk = (cx, cz)
        self.current_biome = self.get_biome(cx, cz)
        weather = self.get_weather(cx, cz)
        self.current_weather = weather
        
        # Update weather timer
        weather.weather_timer -= dt / 60.0
        
        if weather.weather_timer <= 0:
            # Time for weather change
            self._roll_new_weather(weather, self.current_biome)
            weather.weather_timer = 30 + random.random() * 60  # 30-90 seconds (shorter cycles)
        
        # Faster transitions for more noticeable weather changes
        transition_speed = 0.08 * dt
        weather.precipitation += (weather.target_precipitation - weather.precipitation) * transition_speed
        weather.cloud_cover += (weather.target_clouds - weather.cloud_cover) * transition_speed
        
        # Update lightning
        if weather.lightning_active and weather.precipitation > 0.5:
            self.lightning_timer -= dt / 60.0
            if self.lightning_timer <= 0:
                self.lightning_flash = 1.0
                self.lightning_timer = 2 + random.random() * 8  # 2-10 seconds between strikes
        
        # Fade lightning flash
        self.lightning_flash *= 0.85
        
        # Slowly shift wind direction
        weather.wind_direction += math.sin(self.time * 0.1) * 0.01
    
    def _roll_new_weather(self, weather: WeatherState, biome: BiomeDNA):
        """Determine new target weather based on biome."""
        roll = random.random()
        
        # Higher precipitation chance - about 50-70% in humid areas
        precip_chance = biome.rain_tendency * 1.5 + biome.humidity * 0.4
        precip_chance = min(0.75, precip_chance)  # Cap at 75%
        
        if roll < precip_chance:
            # Precipitation!
            intensity = 0.4 + random.random() * 0.6  # 0.4-1.0 (heavier)
            weather.target_precipitation = intensity
            weather.target_clouds = 0.7 + intensity * 0.25  # Heavier clouds
            
            # Rain or snow based on temperature
            if biome.temperature < 0.3 or (biome.temperature < 0.45 and random.random() < biome.snow_tendency * 2):
                weather.precipitation_type = "snow"
            else:
                weather.precipitation_type = "rain"
            
            # Storm chance during heavy rain - more frequent storms!
            storm_chance = biome.storm_tendency * 1.5 + (intensity - 0.5) * 0.3
            if intensity > 0.5 and random.random() < storm_chance:
                weather.lightning_active = True
                weather.wind_strength = 0.5 + random.random() * 0.4
            else:
                weather.lightning_active = False
        else:
            # Clear/cloudy
            weather.target_precipitation = 0
            weather.precipitation_type = "none"
            weather.lightning_active = False
            weather.target_clouds = biome.cloud_base + random.random() * 0.4
            weather.wind_strength = biome.wind_base + random.random() * 0.2
        
        # Evolve visual DNA based on biome characteristics
        rng = np.random.default_rng(abs(hash((self.seed, self.time, roll))))
        weather.visual_dna = weather.visual_dna.mutate(rng)
        
        # Tint rain/cloud colors based on biome
        if biome.temperature > 0.7:  # Hot/tropical - warmer rain tints
            weather.visual_dna = WeatherVisualDNA(
                rain_color=(_clamp(0.5 + rng.random() * 0.2), 
                            _clamp(0.6 + rng.random() * 0.2), 
                            _clamp(0.7 + rng.random() * 0.2)),
                rain_alpha=weather.visual_dna.rain_alpha,
                rain_waviness=0.2 + rng.random() * 0.3,  # More tropical waviness
                rain_streak_length=weather.visual_dna.rain_streak_length,
                snow_color=weather.visual_dna.snow_color,
                snow_size=weather.visual_dna.snow_size,
                snow_drift=weather.visual_dna.snow_drift,
                cloud_color=(_clamp(0.85 + rng.random() * 0.1),
                             _clamp(0.85 + rng.random() * 0.1),
                             _clamp(0.88 + rng.random() * 0.1)),
                cloud_speed=0.015 + rng.random() * 0.01,
                cloud_size_base=weather.visual_dna.cloud_size_base,
                cloud_size_variance=weather.visual_dna.cloud_size_variance,
                cloud_count=weather.visual_dna.cloud_count,
                cloud_height_offset=weather.visual_dna.cloud_height_offset,
            )
        elif biome.temperature < 0.3:  # Cold - icy blue tints
            weather.visual_dna = WeatherVisualDNA(
                rain_color=weather.visual_dna.rain_color,
                rain_alpha=weather.visual_dna.rain_alpha,
                rain_waviness=0.1,  # Straighter in cold
                rain_streak_length=weather.visual_dna.rain_streak_length,
                snow_color=(_clamp(0.95 + rng.random() * 0.05),
                            _clamp(0.97 + rng.random() * 0.03),
                            1.0),
                snow_size=3.5 + rng.random() * 2.0,  # Bigger flakes in cold
                snow_drift=0.3 + rng.random() * 0.4,
                cloud_color=(_clamp(0.8 + rng.random() * 0.15),
                             _clamp(0.85 + rng.random() * 0.1),
                             _clamp(0.95 + rng.random() * 0.05)),
                cloud_speed=0.005 + rng.random() * 0.01,  # Slower in cold
                cloud_size_base=weather.visual_dna.cloud_size_base,
                cloud_size_variance=weather.visual_dna.cloud_size_variance,
                cloud_count=weather.visual_dna.cloud_count,
                cloud_height_offset=120 + rng.random() * 50,  # Lower clouds in cold
            )
    
    def cleanup_distant(self, cx: int, cz: int, max_dist: int = 30):
        """Remove data for distant chunks/regions."""
        to_remove = [k for k in self.biomes if abs(k[0]-cx) > max_dist or abs(k[1]-cz) > max_dist]
        for k in to_remove:
            del self.biomes[k]
        
        # Clean up regional weather
        region_dist = max_dist // self.WEATHER_REGION_SIZE + 1
        current_region = self._chunk_to_region(cx, cz)
        to_remove_regions = [
            k for k in self.regional_weather 
            if abs(k[0] - current_region[0]) > region_dist or abs(k[1] - current_region[1]) > region_dist
        ]
        for k in to_remove_regions:
            del self.regional_weather[k]


class WeatherRenderer:
    """Renders weather effects - rain, snow, clouds, lightning."""
    
    # Particle pool sizes - increased for more dramatic weather
    MAX_RAIN_PARTICLES = 800
    MAX_SNOW_PARTICLES = 500
    
    def __init__(self):
        self.rain_particles: List[List[float]] = []  # [x, y, z, speed]
        self.snow_particles: List[List[float]] = []  # [x, y, z, drift, size]
        self.initialized = False
    
    def _init_particles(self, camera_x: float, camera_y: float, camera_z: float):
        """Initialize particle pools."""
        # Rain particles
        self.rain_particles = []
        for _ in range(self.MAX_RAIN_PARTICLES):
            self.rain_particles.append([
                camera_x + (random.random() - 0.5) * 100,
                camera_y + random.random() * 60,
                camera_z + (random.random() - 0.5) * 100,
                15 + random.random() * 10,  # Fall speed
            ])
        
        # Snow particles
        self.snow_particles = []
        for _ in range(self.MAX_SNOW_PARTICLES):
            self.snow_particles.append([
                camera_x + (random.random() - 0.5) * 80,
                camera_y + random.random() * 50,
                camera_z + (random.random() - 0.5) * 80,
                random.random() * math.pi * 2,  # Drift phase
                0.1 + random.random() * 0.15,   # Size
            ])
        
        self.initialized = True
    
    def update(self, dt: float, camera_x: float, camera_y: float, camera_z: float,
               weather: WeatherState, biome: BiomeDNA):
        """Update particle positions."""
        if not self.initialized:
            self._init_particles(camera_x, camera_y, camera_z)
        
        time_scale = dt / 60.0  # Normalize to ~seconds
        
        # Wind effect
        wind_x = math.cos(weather.wind_direction) * weather.wind_strength * 2
        wind_z = math.sin(weather.wind_direction) * weather.wind_strength * 2
        
        # Update rain
        if weather.precipitation > 0.05 and weather.precipitation_type == "rain":
            active_count = int(self.MAX_RAIN_PARTICLES * weather.precipitation)
            for i, p in enumerate(self.rain_particles[:active_count]):
                p[1] -= p[3] * time_scale  # Fall
                p[0] += wind_x * time_scale
                p[2] += wind_z * time_scale
                
                # Reset if below camera or too far
                if p[1] < camera_y - 20 or abs(p[0] - camera_x) > 60 or abs(p[2] - camera_z) > 60:
                    p[0] = camera_x + (random.random() - 0.5) * 100
                    p[1] = camera_y + 30 + random.random() * 30
                    p[2] = camera_z + (random.random() - 0.5) * 100
        
        # Update snow
        if weather.precipitation > 0.05 and weather.precipitation_type == "snow":
            active_count = int(self.MAX_SNOW_PARTICLES * weather.precipitation)
            for i, p in enumerate(self.snow_particles[:active_count]):
                p[3] += 0.05 * time_scale  # Drift phase
                
                # Slower fall with drift
                p[1] -= (2 + random.random()) * time_scale
                p[0] += (wind_x + math.sin(p[3]) * 0.5) * time_scale
                p[2] += (wind_z + math.cos(p[3]) * 0.5) * time_scale
                
                # Reset
                if p[1] < camera_y - 15 or abs(p[0] - camera_x) > 50 or abs(p[2] - camera_z) > 50:
                    p[0] = camera_x + (random.random() - 0.5) * 80
                    p[1] = camera_y + 25 + random.random() * 25
                    p[2] = camera_z + (random.random() - 0.5) * 80
    
    def render(self, camera_x: float, camera_y: float, camera_z: float,
               weather: WeatherState, lightning_flash: float):
        """Render weather effects."""
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Lightning flash - brief white overlay
        if lightning_flash > 0.1:
            glDisable(GL_DEPTH_TEST)
            glColor4f(1, 1, 1, lightning_flash * 0.3)
            glBegin(GL_QUADS)
            glVertex3f(camera_x - 1000, camera_y - 100, camera_z - 1000)
            glVertex3f(camera_x + 1000, camera_y - 100, camera_z - 1000)
            glVertex3f(camera_x + 1000, camera_y + 500, camera_z + 1000)
            glVertex3f(camera_x - 1000, camera_y + 500, camera_z + 1000)
            glEnd()
            glEnable(GL_DEPTH_TEST)
        
        # Rain with DNA-based visuals
        if weather.precipitation > 0.05 and weather.precipitation_type == "rain":
            active_count = int(self.MAX_RAIN_PARTICLES * weather.precipitation)
            vdna = weather.visual_dna
            
            # Rain color from DNA
            alpha = vdna.rain_alpha + weather.precipitation * 0.3
            glColor4f(vdna.rain_color[0], vdna.rain_color[1], vdna.rain_color[2], alpha)
            glLineWidth(2)
            
            # Regular rain streaks
            streak_len = vdna.rain_streak_length
            glBegin(GL_LINES)
            for p in self.rain_particles[:active_count]:
                glVertex3f(p[0], p[1], p[2])
                glVertex3f(p[0], p[1] + streak_len, p[2])
            glEnd()
            
            # Wavy rain streams (a few polygon ribbons for visual interest)
            if vdna.rain_waviness > 0.1 and weather.precipitation > 0.3:
                wavy_count = min(15, int(active_count * vdna.rain_waviness * 0.05))
                glColor4f(vdna.rain_color[0] * 0.9, vdna.rain_color[1] * 0.95, 
                          vdna.rain_color[2], alpha * 0.5)
                
                for i in range(wavy_count):
                    p = self.rain_particles[i * 5 % active_count]
                    # Draw wavy polygon stream
                    glBegin(GL_LINE_STRIP)
                    wave_amp = vdna.rain_waviness * 0.5
                    for j in range(6):
                        t = j / 5.0
                        wave_x = math.sin((p[1] + j) * 0.5 + lightning_flash * 10) * wave_amp
                        wave_z = math.cos((p[1] + j) * 0.4) * wave_amp * 0.5
                        glVertex3f(p[0] + wave_x, p[1] + t * 4, p[2] + wave_z)
                    glEnd()
        
        # Snow with DNA-based visuals
        if weather.precipitation > 0.05 and weather.precipitation_type == "snow":
            active_count = int(self.MAX_SNOW_PARTICLES * weather.precipitation)
            vdna = weather.visual_dna
            
            # Snow color and size from DNA
            glColor4f(vdna.snow_color[0], vdna.snow_color[1], vdna.snow_color[2], 0.9)
            glPointSize(vdna.snow_size)
            
            glBegin(GL_POINTS)
            for p in self.snow_particles[:active_count]:
                glVertex3f(p[0], p[1], p[2])
            glEnd()
            
            # Larger flakes for variety
            glPointSize(vdna.snow_size * 1.3)
            glColor4f(vdna.snow_color[0], vdna.snow_color[1], vdna.snow_color[2], 0.6)
            glBegin(GL_POINTS)
            for p in self.snow_particles[:active_count:3]:
                glVertex3f(p[0] + 0.1, p[1] + 0.2, p[2])
            glEnd()
        
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
    
    def render_clouds(self, camera_x: float, camera_y: float, camera_z: float,
                      cloud_cover: float, time: float, visual_dna: WeatherVisualDNA = None):
        """Render cloud layer with DNA-based visuals."""
        if cloud_cover < 0.1:
            return
        
        # Use defaults if no DNA provided
        if visual_dna is None:
            visual_dna = WeatherVisualDNA()
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        
        cloud_height = camera_y + visual_dna.cloud_height_offset
        cloud_alpha = cloud_cover * 0.4
        
        # Cloud color from DNA
        glColor4f(visual_dna.cloud_color[0], visual_dna.cloud_color[1], 
                  visual_dna.cloud_color[2], cloud_alpha)
        
        # Multiple cloud patches with DNA-based properties
        cloud_count = visual_dna.cloud_count
        cloud_speed = visual_dna.cloud_speed
        
        for i in range(cloud_count):
            angle = (i / cloud_count) * math.pi * 2 + time * cloud_speed
            dist = 200 + math.sin(angle * 3 + i) * 60
            cx = camera_x + math.cos(angle) * dist
            cz = camera_z + math.sin(angle) * dist
            
            # Size from DNA with variance
            base_size = visual_dna.cloud_size_base
            variance = visual_dna.cloud_size_variance
            size = base_size + math.sin(i + time * 0.02) * variance
            
            # Vary alpha slightly per cloud for depth
            local_alpha = cloud_alpha * (0.8 + math.sin(i * 1.5) * 0.2)
            glColor4f(visual_dna.cloud_color[0], visual_dna.cloud_color[1],
                      visual_dna.cloud_color[2], local_alpha)
            
            glBegin(GL_QUADS)
            glVertex3f(cx - size, cloud_height, cz - size)
            glVertex3f(cx + size, cloud_height, cz - size)
            glVertex3f(cx + size, cloud_height, cz + size)
            glVertex3f(cx - size, cloud_height, cz + size)
            glEnd()
            
            # Add smaller secondary cloud puff nearby for more organic shape
            if i % 2 == 0:
                offset_x = math.cos(angle + 0.5) * size * 0.6
                offset_z = math.sin(angle + 0.5) * size * 0.6
                small_size = size * 0.5
                glColor4f(visual_dna.cloud_color[0], visual_dna.cloud_color[1],
                          visual_dna.cloud_color[2], local_alpha * 0.7)
                glBegin(GL_QUADS)
                glVertex3f(cx + offset_x - small_size, cloud_height + 5, cz + offset_z - small_size)
                glVertex3f(cx + offset_x + small_size, cloud_height + 5, cz + offset_z - small_size)
                glVertex3f(cx + offset_x + small_size, cloud_height + 5, cz + offset_z + small_size)
                glVertex3f(cx + offset_x - small_size, cloud_height + 5, cz + offset_z + small_size)
                glEnd()
        
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

