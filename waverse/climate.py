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
    humidity: float = 0.5         # 0=arid, 0.5=normal, 1=tropical
    elevation_factor: float = 0.5 # Affects temperature at height
    
    # Weather tendencies (how likely certain weather is)
    rain_tendency: float = 0.3    # Base chance of rain
    snow_tendency: float = 0.1    # Base chance of snow (modified by temp)
    storm_tendency: float = 0.1   # Base chance of storms
    cloud_base: float = 0.3       # Base cloud cover
    wind_base: float = 0.3        # Base wind strength
    
    def mutate(self, rng: np.random.Generator, strength: float = 0.02) -> "BiomeDNA":
        """Very slow mutation - biomes should span many chunks."""
        # Only 8% chance of any mutation
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
        )
    
    def crossover(self, other: "BiomeDNA", rng: np.random.Generator) -> "BiomeDNA":
        """Blend two biome DNAs."""
        t = 0.3 + rng.random() * 0.4
        return BiomeDNA(
            temperature=self.temperature * (1-t) + other.temperature * t,
            humidity=self.humidity * (1-t) + other.humidity * t,
            elevation_factor=self.elevation_factor * (1-t) + other.elevation_factor * t,
            rain_tendency=self.rain_tendency * (1-t) + other.rain_tendency * t,
            snow_tendency=self.snow_tendency * (1-t) + other.snow_tendency * t,
            storm_tendency=self.storm_tendency * (1-t) + other.storm_tendency * t,
            cloud_base=self.cloud_base * (1-t) + other.cloud_base * t,
            wind_base=self.wind_base * (1-t) + other.wind_base * t,
        )
    
    def get_biome_name(self) -> str:
        """Get human-readable biome name."""
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
class WeatherState:
    """Current weather conditions - changes over time."""
    precipitation: float = 0.0     # 0=none, 0.5=light, 1=heavy
    precipitation_type: str = "none"  # none, rain, snow, sleet
    cloud_cover: float = 0.3       # 0=clear, 1=overcast
    wind_strength: float = 0.2     # 0=calm, 1=strong
    wind_direction: float = 0.0    # Radians
    lightning_active: bool = False
    fog_density: float = 0.0       # 0=clear, 1=thick fog
    
    # Transition tracking
    target_precipitation: float = 0.0
    target_clouds: float = 0.3
    weather_timer: float = 0.0     # Time until next weather change


class ClimateManager:
    """Manages biome DNA and weather per chunk."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.biomes: Dict[Tuple[int, int], BiomeDNA] = {}
        self.weather: Dict[Tuple[int, int], WeatherState] = {}
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
    
    def get_weather(self, cx: int, cz: int) -> WeatherState:
        """Get or create weather state for a chunk."""
        key = (cx, cz)
        if key not in self.weather:
            biome = self.get_biome(cx, cz)
            self.weather[key] = WeatherState(
                cloud_cover=biome.cloud_base + random.random() * 0.2,
                wind_strength=biome.wind_base + random.random() * 0.2,
                wind_direction=random.random() * math.pi * 2,
            )
        return self.weather[key]
    
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
            weather.weather_timer = 30 + random.random() * 60  # 30-90 seconds
        
        # Smooth transitions
        transition_speed = 0.02 * dt
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
        
        # Determine if precipitation
        precip_chance = biome.rain_tendency + biome.humidity * 0.2
        if roll < precip_chance:
            # Precipitation!
            intensity = 0.3 + random.random() * 0.7
            weather.target_precipitation = intensity
            weather.target_clouds = 0.6 + intensity * 0.3
            
            # Rain or snow based on temperature
            if biome.temperature < 0.3 or (biome.temperature < 0.45 and random.random() < biome.snow_tendency * 2):
                weather.precipitation_type = "snow"
            else:
                weather.precipitation_type = "rain"
            
            # Storm chance during heavy rain
            if intensity > 0.6 and random.random() < biome.storm_tendency:
                weather.lightning_active = True
                weather.wind_strength = 0.5 + random.random() * 0.4
            else:
                weather.lightning_active = False
        else:
            # Clear/cloudy
            weather.target_precipitation = 0
            weather.precipitation_type = "none"
            weather.lightning_active = False
            weather.target_clouds = biome.cloud_base + random.random() * 0.3
            weather.wind_strength = biome.wind_base + random.random() * 0.2
    
    def cleanup_distant(self, cx: int, cz: int, max_dist: int = 30):
        """Remove data for distant chunks."""
        to_remove = [k for k in self.biomes if abs(k[0]-cx) > max_dist or abs(k[1]-cz) > max_dist]
        for k in to_remove:
            del self.biomes[k]
            if k in self.weather:
                del self.weather[k]


class WeatherRenderer:
    """Renders weather effects - rain, snow, clouds, lightning."""
    
    # Particle pool sizes (performance tuned)
    MAX_RAIN_PARTICLES = 400
    MAX_SNOW_PARTICLES = 300
    
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
        if weather.precipitation > 0.1 and weather.precipitation_type == "rain":
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
        if weather.precipitation > 0.1 and weather.precipitation_type == "snow":
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
        
        # Rain
        if weather.precipitation > 0.1 and weather.precipitation_type == "rain":
            active_count = int(self.MAX_RAIN_PARTICLES * weather.precipitation)
            
            # Rain color - slightly blue, semi-transparent
            alpha = 0.3 + weather.precipitation * 0.3
            glColor4f(0.7, 0.75, 0.85, alpha)
            glLineWidth(1)
            
            glBegin(GL_LINES)
            for p in self.rain_particles[:active_count]:
                # Rain streak
                glVertex3f(p[0], p[1], p[2])
                glVertex3f(p[0], p[1] + 0.8, p[2])  # Short streak
            glEnd()
        
        # Snow
        if weather.precipitation > 0.1 and weather.precipitation_type == "snow":
            active_count = int(self.MAX_SNOW_PARTICLES * weather.precipitation)
            
            # Snow - white points
            glColor4f(1, 1, 1, 0.8)
            glPointSize(3)
            
            glBegin(GL_POINTS)
            for p in self.snow_particles[:active_count]:
                glVertex3f(p[0], p[1], p[2])
            glEnd()
            
            # Larger flakes for variety
            glPointSize(4)
            glColor4f(1, 1, 1, 0.6)
            glBegin(GL_POINTS)
            for p in self.snow_particles[:active_count:3]:  # Every 3rd
                glVertex3f(p[0] + 0.1, p[1] + 0.2, p[2])
            glEnd()
        
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
    
    def render_clouds(self, camera_x: float, camera_y: float, camera_z: float,
                      cloud_cover: float, time: float):
        """Render simple cloud layer."""
        if cloud_cover < 0.1:
            return
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_DEPTH_TEST)
        
        cloud_height = camera_y + 150
        cloud_alpha = cloud_cover * 0.4
        
        # Simple cloud quads at height
        glColor4f(0.9, 0.9, 0.92, cloud_alpha)
        
        # Multiple cloud patches
        for i in range(8):
            angle = (i / 8) * math.pi * 2 + time * 0.01
            dist = 200 + math.sin(angle * 3) * 50
            cx = camera_x + math.cos(angle) * dist
            cz = camera_z + math.sin(angle) * dist
            size = 80 + math.sin(i + time * 0.02) * 30
            
            glBegin(GL_QUADS)
            glVertex3f(cx - size, cloud_height, cz - size)
            glVertex3f(cx + size, cloud_height, cz - size)
            glVertex3f(cx + size, cloud_height, cz + size)
            glVertex3f(cx - size, cloud_height, cz + size)
            glEnd()
        
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)

