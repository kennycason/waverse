"""
Wind System - Global wind state management for plants, water, and particles.

Features:
- Base wind direction and strength (slowly drifting)
- Gusts (periodic strength variations)
- Tornado events (localized extreme wind)
- Provides uniforms for shaders
"""

import math
import time
from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np


@dataclass
class TornadoState:
    """Active tornado event."""
    center_x: float
    center_z: float
    radius: float = 50.0          # Affected area
    strength: float = 5.0         # Wind multiplier
    rotation_speed: float = 2.0   # Radians per second
    lifetime: float = 30.0        # Seconds until dissipation
    spawn_time: float = 0.0       # When it spawned
    
    @property
    def age(self) -> float:
        return time.time() - self.spawn_time
    
    @property
    def is_expired(self) -> bool:
        return self.age > self.lifetime
    
    @property
    def intensity(self) -> float:
        """Intensity curve: ramp up, sustain, ramp down."""
        age = self.age
        if age < 3.0:  # Ramp up
            return age / 3.0
        elif age > self.lifetime - 5.0:  # Ramp down
            return (self.lifetime - age) / 5.0
        return 1.0


class WindManager:
    """
    Manages global wind state for the world.
    
    Wind affects:
    - Flora sway animation
    - Water wave direction and amplitude
    - Weather particles
    - Debris/wind particles
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        
        # Base wind state
        self._base_angle = self.rng.random() * math.pi * 2  # Initial direction
        self._base_strength = 0.3 + self.rng.random() * 0.3  # 0.3-0.6
        
        # Wind drift (slowly changing direction)
        self._drift_speed = 0.02  # Radians per second
        self._drift_direction = 1 if self.rng.random() < 0.5 else -1
        
        # Gust state
        self._gust_phase = 0.0
        self._gust_frequency = 0.5  # Oscillations per second
        self._gust_amplitude = 0.2  # Max gust addition
        
        # Tornado
        self.tornado: Optional[TornadoState] = None
        self._tornado_check_interval = 3.0  # Faster checks
        self._last_tornado_check = 0.0
        self._tornado_chance = 0.5  # 50% per check during storms - more tornadoes!
        
        # Also spawn tornadoes in windy non-storm weather occasionally
        self._calm_tornado_chance = 0.05  # 5% chance even without storm
        
        # Tide
        self._tide_phase = 0.0
        self._tide_period = 300.0  # 5 minute cycle
        self._tide_amplitude = 2.0  # Max water level change
        
        # Time tracking
        self._last_update = time.time()
        self._total_time = 0.0
    
    def update(self, dt: float, is_stormy: bool = False):
        """Update wind state each frame."""
        self._total_time += dt
        
        # Drift base direction
        self._base_angle += self._drift_speed * self._drift_direction * dt
        
        # Occasionally reverse drift
        if self.rng.random() < 0.001:  # ~1% per second at 60fps
            self._drift_direction *= -1
        
        # Update gust phase
        self._gust_phase += self._gust_frequency * dt * math.pi * 2
        
        # Update tide
        self._tide_phase += dt / self._tide_period * math.pi * 2
        
        # Check for tornado spawn
        self._last_tornado_check += dt
        if self._last_tornado_check >= self._tornado_check_interval:
            self._last_tornado_check = 0.0
            if self.tornado is None:
                # Higher chance during storms, lower chance otherwise
                chance = self._tornado_chance if is_stormy else self._calm_tornado_chance
                if self.rng.random() < chance:
                    self._spawn_tornado()
        
        # Update/expire tornado
        if self.tornado and self.tornado.is_expired:
            self.tornado = None
    
    def _spawn_tornado(self):
        """Spawn a new tornado at a random nearby location."""
        # Spawn within ~200 units (will be positioned relative to camera)
        angle = self.rng.random() * math.pi * 2
        dist = 50 + self.rng.random() * 150
        self.tornado = TornadoState(
            center_x=math.cos(angle) * dist,
            center_z=math.sin(angle) * dist,
            radius=30 + self.rng.random() * 40,
            strength=3.0 + self.rng.random() * 4.0,
            rotation_speed=1.5 + self.rng.random() * 2.0,
            lifetime=15 + self.rng.random() * 30,
            spawn_time=time.time()
        )
        print(f"[WIND] Tornado spawned! Radius={self.tornado.radius:.0f}, Strength={self.tornado.strength:.1f}")
    
    def force_tornado(self, x: float = 0, z: float = 50):
        """Force spawn a tornado for testing."""
        self.tornado = TornadoState(
            center_x=x,
            center_z=z,
            radius=40,
            strength=5.0,
            rotation_speed=2.0,
            lifetime=60.0,  # Long duration for testing
            spawn_time=time.time()
        )
        print(f"[WIND] Forced tornado at ({x}, {z})")
    
    @property
    def direction(self) -> Tuple[float, float]:
        """Current wind direction as unit vector (x, z)."""
        return (math.cos(self._base_angle), math.sin(self._base_angle))
    
    @property
    def angle(self) -> float:
        """Current wind angle in radians."""
        return self._base_angle
    
    @property
    def strength(self) -> float:
        """Current wind strength (0.0 to ~1.5)."""
        gust = (math.sin(self._gust_phase) * 0.5 + 0.5) * self._gust_amplitude
        return self._base_strength + gust
    
    @property
    def gust_factor(self) -> float:
        """Current gust factor (0.0 to 1.0)."""
        return (math.sin(self._gust_phase) * 0.5 + 0.5)
    
    @property
    def tide_level(self) -> float:
        """Current tide offset."""
        return math.sin(self._tide_phase) * self._tide_amplitude
    
    @property 
    def time(self) -> float:
        """Total elapsed time for shader animations."""
        return self._total_time
    
    def get_wind_at(self, x: float, z: float) -> Tuple[float, float, float]:
        """
        Get wind vector at a specific world position.
        Returns (wind_x, wind_z, strength).
        
        Accounts for tornado if active.
        """
        base_dx, base_dz = self.direction
        base_strength = self.strength
        
        if self.tornado and not self.tornado.is_expired:
            # Distance to tornado center
            dx = x - self.tornado.center_x
            dz = z - self.tornado.center_z
            dist = math.sqrt(dx * dx + dz * dz)
            
            if dist < self.tornado.radius:
                # Inside tornado - rotational wind
                intensity = self.tornado.intensity
                falloff = 1.0 - (dist / self.tornado.radius)  # Stronger toward center
                
                # Perpendicular direction (rotational)
                if dist > 0.1:
                    # Rotate 90 degrees for tangential wind
                    rot_dx = -dz / dist
                    rot_dz = dx / dist
                    
                    # Blend with inward pull
                    inward_dx = -dx / dist
                    inward_dz = -dz / dist
                    
                    blend = 0.7  # 70% rotation, 30% inward
                    final_dx = rot_dx * blend + inward_dx * (1 - blend)
                    final_dz = rot_dz * blend + inward_dz * (1 - blend)
                    
                    tornado_strength = self.tornado.strength * intensity * falloff
                    
                    # Blend base wind with tornado wind
                    total_strength = base_strength + tornado_strength
                    wind_dx = (base_dx * base_strength + final_dx * tornado_strength) / total_strength
                    wind_dz = (base_dz * base_strength + final_dz * tornado_strength) / total_strength
                    
                    return (wind_dx, wind_dz, total_strength)
        
        return (base_dx, base_dz, base_strength)
    
    def get_shader_uniforms(self) -> dict:
        """Get all wind-related uniforms for shaders."""
        dx, dz = self.direction
        return {
            'u_wind_dir': (dx, dz),
            'u_wind_strength': self.strength,
            'u_wind_time': self._total_time,
            'u_gust_factor': self.gust_factor,
            'u_tide_level': self.tide_level,
            'u_has_tornado': 1.0 if self.tornado else 0.0,
            'u_tornado_center': (self.tornado.center_x, self.tornado.center_z) if self.tornado else (0.0, 0.0),
            'u_tornado_radius': self.tornado.radius if self.tornado else 0.0,
            'u_tornado_strength': self.tornado.strength * self.tornado.intensity if self.tornado else 0.0,
        }
    
    def set_base_strength(self, strength: float):
        """Set base wind strength (0.0 to 1.0)."""
        self._base_strength = max(0.0, min(1.0, strength))
    
    def set_stormy(self, is_stormy: bool):
        """Increase wind during storms."""
        if is_stormy:
            self._base_strength = min(1.0, self._base_strength + 0.3)
            self._gust_amplitude = 0.4
        else:
            self._gust_amplitude = 0.2

