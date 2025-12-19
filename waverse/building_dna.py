"""
Building DNA System - Procedural building generation with rich variety.

Inspired by low-poly city packs, this system generates diverse buildings:
- Residential: houses, apartments, villas
- Commercial: shops, offices, towers
- Industrial: warehouses, factories
- Special: temples, monuments, observatories

Each building has DNA that determines its shape, style, and features.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Literal
from enum import Enum
import numpy as np


class BuildingType(Enum):
    """Categories of buildings."""
    # Residential
    HOUSE = "house"              # Small 1-2 floor home with pitched roof
    APARTMENT = "apartment"      # Medium 3-6 floor flat roof building
    VILLA = "villa"              # Larger upscale home with complex roof
    
    # Commercial  
    SHOP = "shop"                # 1-2 floor with awning and large windows
    OFFICE = "office"            # Medium office building
    TOWER = "tower"              # Tall skyscraper
    HOTEL = "hotel"              # Multi-floor with balconies
    
    # Industrial
    WAREHOUSE = "warehouse"      # Large footprint, low height
    FACTORY = "factory"          # Industrial with smokestacks
    HANGAR = "hangar"            # Curved/arched roof
    
    # Special
    TEMPLE = "temple"            # Religious/spiritual structure
    MONUMENT = "monument"        # Statue/obelisk/memorial
    OBSERVATORY = "observatory"  # Dome-topped
    RUINS = "ruins"              # Partially destroyed ancient structure


class RoofType(Enum):
    """Types of roof shapes."""
    FLAT = "flat"                # Modern flat roof (can have features)
    GABLED = "gabled"            # Classic triangular pitched roof
    HIPPED = "hipped"            # All sides slope down
    PYRAMID = "pyramid"          # Four-sided pyramid
    SHED = "shed"                # Single slope
    DOME = "dome"                # Hemispherical
    BARREL = "barrel"            # Cylindrical/curved
    STEPPED = "stepped"          # Tiered/stepped like ziggurats


class WallStyle(Enum):
    """Visual style of walls."""
    SOLID = "solid"              # Plain solid color
    BRICK = "brick"              # Brick pattern (simulated with color bands)
    GLASS = "glass"              # Reflective/transparent look
    CONCRETE = "concrete"        # Industrial gray
    WOOD = "wood"                # Natural wood tones
    STONE = "stone"              # Stone blocks
    METAL = "metal"              # Corrugated/industrial


@dataclass
class ColorPalette:
    """Color scheme for a building."""
    primary: Tuple[float, float, float]      # Main wall color
    secondary: Tuple[float, float, float]    # Accent color (trim, details)
    roof: Tuple[float, float, float]         # Roof color
    window: Tuple[float, float, float]       # Window tint
    
    @classmethod
    def random(cls, rng: np.random.Generator, style: str = "modern") -> "ColorPalette":
        """Generate a random color palette for a style."""
        if style == "modern":
            primaries = [
                (0.85, 0.85, 0.88),  # White/light gray
                (0.45, 0.55, 0.65),  # Steel blue
                (0.9, 0.88, 0.82),   # Cream
                (0.35, 0.38, 0.42),  # Dark gray
            ]
            primary = primaries[rng.integers(0, len(primaries))]
            secondary = (primary[0] * 0.7, primary[1] * 0.7, primary[2] * 0.75)
            roof = (0.25, 0.28, 0.32)  # Dark roofs
            window = (0.3, 0.5, 0.7)  # Blue tint
            
        elif style == "residential":
            primaries = [
                (0.92, 0.9, 0.85),   # Cream
                (0.88, 0.82, 0.75),  # Beige
                (0.75, 0.82, 0.85),  # Light blue
                (0.85, 0.8, 0.75),   # Tan
                (0.82, 0.85, 0.78),  # Sage
            ]
            primary = primaries[rng.integers(0, len(primaries))]
            secondary = (0.45, 0.38, 0.32)  # Brown trim
            roofs = [
                (0.55, 0.35, 0.28),  # Terracotta
                (0.35, 0.35, 0.38),  # Slate
                (0.42, 0.32, 0.25),  # Brown
                (0.28, 0.32, 0.35),  # Dark gray
            ]
            roof = roofs[rng.integers(0, len(roofs))]
            window = (0.6, 0.75, 0.9)
            
        elif style == "industrial":
            primaries = [
                (0.55, 0.52, 0.48),  # Weathered gray
                (0.5, 0.45, 0.38),   # Rust brown
                (0.48, 0.5, 0.52),   # Blue-gray
                (0.6, 0.55, 0.45),   # Tan
            ]
            primary = primaries[rng.integers(0, len(primaries))]
            secondary = (primary[0] * 0.8, primary[1] * 0.75, primary[2] * 0.7)
            roof = (0.4, 0.42, 0.45)
            window = (0.4, 0.45, 0.5)
            
        elif style == "commercial":
            primaries = [
                (0.25, 0.45, 0.6),   # Corporate blue
                (0.5, 0.55, 0.58),   # Silver
                (0.7, 0.68, 0.65),   # Warm gray
                (0.4, 0.5, 0.45),    # Teal
            ]
            primary = primaries[rng.integers(0, len(primaries))]
            accents = [
                (0.85, 0.55, 0.25),  # Orange
                (0.25, 0.65, 0.45),  # Green
                (0.75, 0.25, 0.3),   # Red
                (0.9, 0.75, 0.2),    # Yellow
            ]
            secondary = accents[rng.integers(0, len(accents))]
            roof = (0.22, 0.24, 0.28)
            window = (0.35, 0.55, 0.75)
            
        elif style == "alien":
            # Otherworldly colors - ensure all values are clamped to 0-1
            hue = rng.random()
            primary = (
                max(0, min(1, 0.4 + 0.35 * math.sin(hue * 6.28))),
                max(0, min(1, 0.45 + 0.3 * math.sin(hue * 6.28 + 2.1))),
                max(0, min(1, 0.5 + 0.35 * math.sin(hue * 6.28 + 4.2)))
            )
            secondary = tuple(max(0, min(1, c * 1.2)) for c in primary)
            roof = tuple(max(0, min(1, c * 0.7)) for c in primary)
            window = (0.6, 0.8, 0.9)
            
        else:  # ancient/ruins
            primary = (0.65 + rng.random() * 0.15, 
                      0.58 + rng.random() * 0.12,
                      0.45 + rng.random() * 0.1)
            secondary = (primary[0] * 0.85, primary[1] * 0.82, primary[2] * 0.8)
            roof = (primary[0] * 0.75, primary[1] * 0.72, primary[2] * 0.68)
            window = (0.2, 0.22, 0.25)
        
        return cls(primary=primary, secondary=secondary, roof=roof, window=window)


@dataclass 
class BuildingDNA:
    """DNA that defines a building's appearance and structure."""
    building_type: BuildingType
    seed: int
    
    # Dimensions (in world units)
    width: float              # X dimension
    depth: float              # Z dimension  
    floors: int               # Number of floors
    floor_height: float       # Height per floor
    
    # Style
    roof_type: RoofType
    wall_style: WallStyle
    colors: ColorPalette
    
    # Features (probabilities/flags)
    has_balconies: bool = False
    balcony_chance: float = 0.3
    has_awning: bool = False
    awning_depth: float = 2.0
    has_windows: bool = True
    window_rows: int = 2          # Windows per floor
    window_cols: int = 3          # Windows per wall
    has_entrance: bool = True
    entrance_wall: int = 0        # 0=+Z, 1=-Z, 2=-X, 3=+X
    has_rooftop_features: bool = False  # AC units, antenna, helipad
    has_chimney: bool = False
    
    # Variation
    wall_inset: float = 0.0       # Upper floors can be inset
    corner_style: str = "square"  # "square", "rounded", "chamfered"
    
    @classmethod
    def create_random(cls, building_type: BuildingType, seed: int) -> "BuildingDNA":
        """Create random DNA for a building type."""
        # Ensure seed is non-negative for numpy
        rng = np.random.default_rng(abs(seed) % (2**31))
        
        # Type-specific defaults
        if building_type == BuildingType.HOUSE:
            width = 8 + rng.random() * 8       # 8-16
            depth = 8 + rng.random() * 8       # 8-16
            floors = 1 if rng.random() < 0.4 else 2
            floor_height = 6.0 + rng.random() * 2.0  # Taller ceilings for walkability
            roof_type = rng.choice([RoofType.GABLED, RoofType.HIPPED, RoofType.PYRAMID])
            wall_style = rng.choice([WallStyle.BRICK, WallStyle.WOOD, WallStyle.SOLID])
            colors = ColorPalette.random(rng, "residential")
            has_balconies = floors > 1 and rng.random() < 0.3
            has_chimney = rng.random() < 0.4
            
        elif building_type == BuildingType.APARTMENT:
            width = 15 + rng.random() * 15     # 15-30
            depth = 12 + rng.random() * 10     # 12-22
            floors = 3 + rng.integers(0, 5)    # 3-7
            floor_height = 5.5 + rng.random() * 1.5  # Taller ceilings
            roof_type = RoofType.FLAT
            wall_style = rng.choice([WallStyle.CONCRETE, WallStyle.BRICK, WallStyle.SOLID])
            colors = ColorPalette.random(rng, "modern")
            has_balconies = rng.random() < 0.7
            has_chimney = False
            
        elif building_type == BuildingType.VILLA:
            width = 18 + rng.random() * 12     # 18-30
            depth = 15 + rng.random() * 10     # 15-25
            floors = 2 + rng.integers(0, 2)    # 2-3
            floor_height = 6.5 + rng.random() * 1.5  # Taller ceilings
            roof_type = rng.choice([RoofType.HIPPED, RoofType.GABLED])
            wall_style = rng.choice([WallStyle.STONE, WallStyle.SOLID])
            colors = ColorPalette.random(rng, "residential")
            has_balconies = True
            has_chimney = rng.random() < 0.6
            
        elif building_type == BuildingType.SHOP:
            width = 10 + rng.random() * 10     # 10-20
            depth = 8 + rng.random() * 8       # 8-16
            floors = 1 if rng.random() < 0.5 else 2
            floor_height = 5.5 + rng.random() * 1.5  # Taller ceilings
            roof_type = rng.choice([RoofType.FLAT, RoofType.SHED])
            wall_style = rng.choice([WallStyle.BRICK, WallStyle.CONCRETE])
            colors = ColorPalette.random(rng, "commercial")
            has_balconies = False
            has_awning = True
            awning_depth = 2.0 + rng.random() * 2.0
            has_chimney = False
            
        elif building_type == BuildingType.OFFICE:
            width = 20 + rng.random() * 15     # 20-35
            depth = 15 + rng.random() * 12     # 15-27
            floors = 4 + rng.integers(0, 6)    # 4-9
            floor_height = 5.5 + rng.random()  # Taller ceilings
            roof_type = RoofType.FLAT
            wall_style = rng.choice([WallStyle.GLASS, WallStyle.CONCRETE])
            colors = ColorPalette.random(rng, "commercial")
            has_balconies = False
            has_rooftop_features = rng.random() < 0.6
            has_chimney = False
            
        elif building_type == BuildingType.TOWER:
            width = 15 + rng.random() * 10     # 15-25
            depth = 15 + rng.random() * 10     # 15-25
            floors = 10 + rng.integers(0, 20)  # 10-29
            floor_height = 5.0 + rng.random()  # Taller ceilings
            roof_type = rng.choice([RoofType.FLAT, RoofType.PYRAMID, RoofType.DOME])
            wall_style = WallStyle.GLASS
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_rooftop_features = True
            wall_inset = rng.random() * 0.5  # Upper floors can taper
            has_chimney = False
            
        elif building_type == BuildingType.WAREHOUSE:
            width = 30 + rng.random() * 30     # 30-60
            depth = 25 + rng.random() * 25     # 25-50
            floors = 1 if rng.random() < 0.7 else 2
            floor_height = 8.0 + rng.random() * 4.0
            roof_type = rng.choice([RoofType.FLAT, RoofType.SHED, RoofType.BARREL])
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "industrial")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.FACTORY:
            width = 35 + rng.random() * 25
            depth = 30 + rng.random() * 20
            floors = 2 + rng.integers(0, 2)
            floor_height = 6.0 + rng.random() * 3.0
            roof_type = rng.choice([RoofType.FLAT, RoofType.SHED])
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "industrial")
            has_balconies = False
            has_chimney = True  # Smokestacks
            has_chimney = True
            
        elif building_type == BuildingType.TEMPLE:
            width = 20 + rng.random() * 15
            depth = 25 + rng.random() * 15
            floors = 1 + rng.integers(0, 3)
            floor_height = 6.0 + rng.random() * 3.0
            roof_type = rng.choice([RoofType.PYRAMID, RoofType.DOME, RoofType.STEPPED])
            wall_style = WallStyle.STONE
            colors = ColorPalette.random(rng, "ancient")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.OBSERVATORY:
            width = 12 + rng.random() * 8
            depth = 12 + rng.random() * 8
            floors = 2 + rng.integers(0, 2)
            floor_height = 5.0 + rng.random() * 2.0
            roof_type = RoofType.DOME
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.RUINS:
            width = 15 + rng.random() * 20
            depth = 12 + rng.random() * 18
            floors = 1 + rng.integers(0, 2)
            floor_height = 4.0 + rng.random() * 2.0
            roof_type = RoofType.FLAT  # Ruins have no roof
            wall_style = WallStyle.STONE
            colors = ColorPalette.random(rng, "ancient")
            has_balconies = False
            has_chimney = False
            
        else:  # Generic fallback
            width = 12 + rng.random() * 12
            depth = 10 + rng.random() * 10
            floors = 2 + rng.integers(0, 3)
            floor_height = 4.0 + rng.random()
            roof_type = RoofType.FLAT
            wall_style = WallStyle.SOLID
            colors = ColorPalette.random(rng, "modern")
            has_balconies = rng.random() < 0.3
            has_chimney = False
        
        return cls(
            building_type=building_type,
            seed=seed,
            width=width,
            depth=depth,
            floors=floors,
            floor_height=floor_height,
            roof_type=roof_type,
            wall_style=wall_style,
            colors=colors,
            has_balconies=has_balconies,
            balcony_chance=0.3 + rng.random() * 0.4,
            has_awning=has_awning if building_type == BuildingType.SHOP else False,
            awning_depth=awning_depth if building_type == BuildingType.SHOP else 2.0,
            has_windows=True,
            window_rows=1 + rng.integers(0, 3),
            window_cols=2 + rng.integers(0, 4),
            has_entrance=True,
            entrance_wall=rng.integers(0, 4),
            has_rooftop_features=has_rooftop_features if building_type in [BuildingType.OFFICE, BuildingType.TOWER] else False,
            has_chimney=has_chimney,
            wall_inset=wall_inset if building_type == BuildingType.TOWER else 0.0,
            corner_style=rng.choice(["square", "rounded", "chamfered"])
        )
    
    @property
    def total_height(self) -> float:
        """Total building height including roof."""
        base = self.floors * self.floor_height
        # Add roof height
        if self.roof_type == RoofType.GABLED:
            return base + min(self.width, self.depth) * 0.4
        elif self.roof_type == RoofType.PYRAMID:
            return base + min(self.width, self.depth) * 0.5
        elif self.roof_type == RoofType.DOME:
            return base + min(self.width, self.depth) * 0.3
        elif self.roof_type == RoofType.HIPPED:
            return base + min(self.width, self.depth) * 0.35
        else:
            return base + 0.5  # Flat roof parapet


# ============================================================================
# BUILDING ARCHETYPES - Pre-defined building configurations
# ============================================================================

def get_building_type_for_biome(biome: str, rng: np.random.Generator) -> BuildingType:
    """Choose an appropriate building type based on biome."""
    if biome in ["urban", "city"]:
        weights = {
            BuildingType.APARTMENT: 0.25,
            BuildingType.OFFICE: 0.2,
            BuildingType.TOWER: 0.1,
            BuildingType.SHOP: 0.25,
            BuildingType.WAREHOUSE: 0.1,
            BuildingType.HOUSE: 0.1,
        }
    elif biome in ["suburban", "plains"]:
        weights = {
            BuildingType.HOUSE: 0.5,
            BuildingType.SHOP: 0.15,
            BuildingType.APARTMENT: 0.15,
            BuildingType.WAREHOUSE: 0.1,
            BuildingType.VILLA: 0.1,
        }
    elif biome in ["desert", "canyon"]:
        weights = {
            BuildingType.TEMPLE: 0.2,
            BuildingType.RUINS: 0.25,
            BuildingType.HOUSE: 0.3,
            BuildingType.WAREHOUSE: 0.15,
            BuildingType.MONUMENT: 0.1,
        }
    elif biome in ["forest", "jungle"]:
        weights = {
            BuildingType.HOUSE: 0.4,
            BuildingType.TEMPLE: 0.2,
            BuildingType.RUINS: 0.2,
            BuildingType.OBSERVATORY: 0.1,
            BuildingType.VILLA: 0.1,
        }
    elif biome in ["snow", "tundra"]:
        weights = {
            BuildingType.HOUSE: 0.4,
            BuildingType.FACTORY: 0.2,
            BuildingType.WAREHOUSE: 0.2,
            BuildingType.OBSERVATORY: 0.1,
            BuildingType.APARTMENT: 0.1,
        }
    else:  # Default/alien
        weights = {
            BuildingType.TOWER: 0.15,
            BuildingType.TEMPLE: 0.15,
            BuildingType.OBSERVATORY: 0.15,
            BuildingType.HOUSE: 0.2,
            BuildingType.RUINS: 0.15,
            BuildingType.MONUMENT: 0.1,
            BuildingType.APARTMENT: 0.1,
        }
    
    types = list(weights.keys())
    probs = np.array(list(weights.values()))
    probs /= probs.sum()  # Normalize
    
    return rng.choice(types, p=probs)

