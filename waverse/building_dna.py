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
    # Residential - Classic
    HOUSE = "house"              # Small 1-2 floor home with pitched roof
    APARTMENT = "apartment"      # Medium 3-6 floor flat roof building
    VILLA = "villa"              # Larger upscale home with complex roof
    CABIN = "cabin"              # Small rustic mountain/forest dwelling
    
    # Residential - Modern (from Low Poly Modern House Pack)
    MODERN_HOUSE = "modern_house"         # Clean lines, flat roof, large windows
    SPLIT_LEVEL = "split_level"           # Multi-level offset floors
    RANCH = "ranch"                       # Single-floor spread out
    BUNGALOW = "bungalow"                 # Small cozy single-floor
    TOWNHOUSE = "townhouse"               # Narrow multi-floor attached
    LOFT = "loft"                         # Industrial converted space
    PENTHOUSE = "penthouse"               # Top-floor luxury
    DUPLEX = "duplex"                     # Two-unit dwelling
    MANSION = "mansion"                   # Large luxury estate
    COTTAGE = "cottage"                   # Small rural home
    BEACH_HOUSE = "beach_house"           # Elevated stilts, large deck
    TREE_HOUSE = "tree_house"             # Elevated platform structure
    TINY_HOUSE = "tiny_house"             # Very small efficient dwelling
    A_FRAME = "a_frame"                   # Triangular cabin style
    CONTAINER_HOUSE = "container_house"   # Industrial shipping container
    GEODOME_HOME = "geodome_home"         # Dome-shaped residence
    FLOATING_HOUSE = "floating_house"     # House on water/stilts
    UNDERGROUND_HOME = "underground_home" # Earth-sheltered
    CAPSULE = "capsule"                   # Minimalist pod dwelling
    GREENHOUSE = "greenhouse"             # Glass-walled plant house
    
    # Commercial  
    SHOP = "shop"                # 1-2 floor with awning and large windows
    OFFICE = "office"            # Medium office building
    TOWER = "tower"              # Tall skyscraper
    HOTEL = "hotel"              # Multi-floor with balconies
    HOSPITAL = "hospital"        # Medical building with cross/helipad
    RESTAURANT = "restaurant"    # Dining with outdoor seating potential
    
    # Industrial
    WAREHOUSE = "warehouse"      # Large footprint, low height
    FACTORY = "factory"          # Industrial with smokestacks
    HANGAR = "hangar"            # Curved/arched roof
    SILO = "silo"                # Tall cylindrical storage
    POWERPLANT = "powerplant"    # Energy generation with cooling towers
    
    # Sci-Fi/Futuristic
    SKYSCRAPER = "skyscraper"    # Very tall futuristic tower with antenna
    PLANETARIUM = "planetarium"  # Dome for stargazing
    SOLAR_STATION = "solar_station"  # Solar panel arrays on roof
    LANDING_PAD = "landing_pad"  # Elevated platform for spacecraft
    SPACEPORT = "spaceport"      # Multi-pad spacecraft facility
    RESEARCH_LAB = "research_lab"  # Scientific facility with satellite dishes
    
    # Recreation
    PLAYGROUND = "playground"    # Slides, climbing structures
    STADIUM = "stadium"          # Large arena with tiered seating
    POOL = "pool"                # Swimming facility with open center
    
    # Special
    TEMPLE = "temple"            # Religious/spiritual structure
    MONUMENT = "monument"        # Statue/obelisk/memorial
    OBSERVATORY = "observatory"  # Dome-topped
    RUINS = "ruins"              # Partially destroyed ancient structure
    PARKOUR = "parkour"          # Playground of platforms, ramps, gaps for jumping
    LIGHTHOUSE = "lighthouse"    # Tall beacon tower near water
    BRIDGE = "bridge"            # Spanning structure
    GAZEBO = "gazebo"            # Open-sided garden structure
    
    # Ancient/Cultural (from nubian complex, stone portal)
    NUBIAN_COMPLEX = "nubian_complex"    # Ancient Nubian multi-structure complex
    PYRAMID_TEMPLE = "pyramid_temple"    # Step pyramid with temple on top
    STONE_PORTAL = "stone_portal"        # Ancient gateway/portal structure
    ZIGGURAT = "ziggurat"                # Mesopotamian stepped tower
    OBELISK = "obelisk"                  # Tall pointed pillar
    AMPHITHEATER = "amphitheater"        # Open-air performance venue
    COLOSSEUM = "colosseum"              # Large arena with arches
    AQUEDUCT = "aqueduct"                # Water transport arches
    MAUSOLEUM = "mausoleum"              # Grand tomb structure
    PAGODA_TOWER = "pagoda_tower"        # Asian multi-tiered tower
    
    # Cyberpunk/Neon (from Cyberpunk Restaurant)
    CYBERPUNK_BAR = "cyberpunk_bar"      # Neon-lit bar/club
    NEON_DINER = "neon_diner"            # Retro-futuristic diner
    ARCADE = "arcade"                     # Gaming arcade with screens
    HOLO_BILLBOARD = "holo_billboard"    # Holographic advertisement
    TECH_SHOP = "tech_shop"              # Electronics/gadget store
    
    # City Infrastructure (from City Build, Lowpolycity)
    BUS_STOP = "bus_stop"                # Transit shelter
    METRO_ENTRANCE = "metro_entrance"    # Subway station entrance
    PARKING_GARAGE = "parking_garage"    # Multi-level parking structure
    GAS_STATION = "gas_station"          # Fuel station with canopy
    FIRE_STATION = "fire_station"        # Emergency services
    POLICE_STATION = "police_station"    # Law enforcement
    POST_OFFICE = "post_office"          # Mail services
    LIBRARY = "library"                  # Public library
    SCHOOL = "school"                    # Educational building
    CHURCH = "church"                    # Religious with steeple
    MOSQUE = "mosque"                    # Religious with minaret
    BANK = "bank"                        # Financial with columns
    THEATER = "theater"                  # Performance venue
    MALL = "mall"                        # Shopping center
    SUPERMARKET = "supermarket"          # Grocery store
    
    # Modular/Industrial Sci-Fi (from Modular Sci-Fi Building)
    MODULAR_HUB = "modular_hub"          # Connecting node building
    POWER_NODE = "power_node"            # Energy distribution
    COMM_TOWER = "comm_tower"            # Communications antenna
    SHIELD_GENERATOR = "shield_generator"  # Defense structure
    CRYO_CHAMBER = "cryo_chamber"        # Cryogenic storage
    CARGO_BAY = "cargo_bay"              # Goods storage/transfer
    DOCKING_ARM = "docking_arm"          # Spacecraft connection
    REACTOR_CORE = "reactor_core"        # Power generation core
    
    # === PSYCHEDELIC BIOME ===
    RAINBOW_TOWER = "rainbow_tower"          # Twisting rainbow-colored spire
    MUSHROOM_PALACE = "mushroom_palace"      # Giant mushroom building
    SPIRAL_HOUSE = "spiral_house"            # Spiraling organic structure
    KALEIDOSCOPE = "kaleidoscope"            # Geometric shifting shapes
    BUBBLE_DOME = "bubble_dome"              # Iridescent bubble buildings
    FLOATING_ISLAND = "floating_island"      # Impossible floating structure
    CRYSTAL_CAVE = "crystal_cave"            # Glowing crystal formation
    
    # === HELLFIRE BIOME ===
    VOLCANO_FORGE = "volcano_forge"          # Industrial within volcano
    LAVA_TEMPLE = "lava_temple"              # Obsidian temple over lava
    DEMON_GATE = "demon_gate"                # Hellish portal structure
    BONE_TOWER = "bone_tower"                # Tower made of bones
    FLAME_ALTAR = "flame_altar"              # Sacrificial fire platform
    CHARRED_RUINS = "charred_ruins"          # Burnt out structure
    OBSIDIAN_SPIRE = "obsidian_spire"        # Black glass tower
    
    # === SHADOW BIOME ===
    SHADOW_CASTLE = "shadow_castle"          # Dark gothic castle
    NIGHTMARE_HOUSE = "nightmare_house"      # Creepy impossible geometry
    VOID_PORTAL = "void_portal"              # Gateway to darkness
    CRYPT = "crypt"                          # Underground tomb
    HAUNTED_MANOR = "haunted_manor"          # Spooky mansion
    SPIDER_NEST = "spider_nest"              # Web-covered structure
    DARK_TOWER = "dark_tower"                # Imposing black tower
    
    # === CRYSTAL BIOME ===
    CRYSTAL_PALACE = "crystal_palace"        # Palace of pure crystal
    ICE_FORTRESS = "ice_fortress"            # Frozen crystalline fortress
    PRISM_TOWER = "prism_tower"              # Light-refracting spire
    GEODE_HOUSE = "geode_house"              # Building inside a geode
    QUARTZ_TEMPLE = "quartz_temple"          # Temple of quartz crystal
    
    # === VOID BIOME ===
    VOID_MONOLITH = "void_monolith"          # Featureless black obelisk
    ENTROPY_GATE = "entropy_gate"            # Gateway to oblivion
    NULL_ZONE = "null_zone"                  # Area of pure nothing


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
    SAWTOOTH = "sawtooth"        # Industrial zigzag profile
    CONICAL = "conical"          # Pointed cone (lighthouse, tower)
    PAGODA = "pagoda"            # Multi-tiered Asian style
    MANSARD = "mansard"          # Four-sided with two slopes each
    BUTTERFLY = "butterfly"      # V-shaped modern
    GEODESIC = "geodesic"        # Triangulated dome (futuristic)


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
    has_stairs: bool = True       # Interior stairs for multi-floor buildings
    
    # Variation
    wall_inset: float = 0.0       # Upper floors can be inset
    corner_style: str = "square"  # "square", "rounded", "chamfered"
    
    @classmethod
    def create_random(cls, building_type: BuildingType, seed: int) -> "BuildingDNA":
        """Create random DNA for a building type."""
        # Ensure seed is non-negative for numpy
        rng = np.random.default_rng(abs(seed) % (2**31))
        
        # Initialize default values for all optional parameters
        has_awning = False
        awning_depth = 3.0
        has_balconies = False
        has_chimney = False
        has_rooftop_features = False
        wall_inset = 0.0
        
        # Type-specific defaults (all dimensions scaled for TILE_SCALE=3.0)
        # Floor heights increased to 18-24 for comfortable movement (player height ~3.5)
        if building_type == BuildingType.HOUSE:
            width = 18 + rng.random() * 18      # 18-36 (larger)
            depth = 18 + rng.random() * 18      # 18-36 (larger)
            floors = 1 if rng.random() < 0.4 else 2
            floor_height = 18.0 + rng.random() * 6.0  # 18-24 (was 10.5-13.5)
            roof_type = rng.choice([RoofType.GABLED, RoofType.HIPPED, RoofType.PYRAMID])
            wall_style = rng.choice([WallStyle.BRICK, WallStyle.WOOD, WallStyle.SOLID])
            colors = ColorPalette.random(rng, "residential")
            has_balconies = floors > 1 and rng.random() < 0.3
            has_chimney = rng.random() < 0.4
            
        elif building_type == BuildingType.APARTMENT:
            width = 30 + rng.random() * 30      # 30-60 (larger)
            depth = 24 + rng.random() * 21      # 24-45 (larger)
            floors = 3 + rng.integers(0, 5)     # 3-7
            floor_height = 18.0 + rng.random() * 4.5  # 18-22.5 (was 10.5-12.75)
            roof_type = RoofType.FLAT
            wall_style = rng.choice([WallStyle.CONCRETE, WallStyle.BRICK, WallStyle.SOLID])
            colors = ColorPalette.random(rng, "modern")
            has_balconies = rng.random() < 0.7
            has_chimney = False
            
        elif building_type == BuildingType.VILLA:
            width = 36 + rng.random() * 24      # 36-60 (larger)
            depth = 30 + rng.random() * 21      # 30-51 (larger)
            floors = 2 + rng.integers(0, 2)     # 2-3
            floor_height = 19.5 + rng.random() * 4.5  # 19.5-24 (was 11.25-13.5)
            roof_type = rng.choice([RoofType.HIPPED, RoofType.GABLED])
            wall_style = rng.choice([WallStyle.STONE, WallStyle.SOLID])
            colors = ColorPalette.random(rng, "residential")
            has_balconies = True
            has_chimney = rng.random() < 0.6
            
        elif building_type == BuildingType.SHOP:
            width = 21 + rng.random() * 21      # 21-42 (larger)
            depth = 18 + rng.random() * 18      # 18-36 (larger)
            floors = 1 if rng.random() < 0.5 else 2
            floor_height = 18.0 + rng.random() * 4.5  # 18-22.5 (was 10.5-12.75)
            roof_type = rng.choice([RoofType.FLAT, RoofType.SHED])
            wall_style = rng.choice([WallStyle.BRICK, WallStyle.CONCRETE])
            colors = ColorPalette.random(rng, "commercial")
            has_balconies = False
            has_awning = True
            awning_depth = 3.0 + rng.random() * 3.0  # 3-6 (was 2-4)
            has_chimney = False
            
        elif building_type == BuildingType.OFFICE:
            width = 42 + rng.random() * 33      # 42-75 (larger)
            depth = 33 + rng.random() * 27      # 33-60 (larger)
            floors = 4 + rng.integers(0, 6)     # 4-9
            floor_height = 18.0 + rng.random() * 3.0  # 18-21 (was 10.5-12)
            roof_type = RoofType.FLAT
            wall_style = rng.choice([WallStyle.GLASS, WallStyle.CONCRETE])
            colors = ColorPalette.random(rng, "commercial")
            has_balconies = False
            has_rooftop_features = rng.random() < 0.6
            has_chimney = False
            
        elif building_type == BuildingType.TOWER:
            width = 30 + rng.random() * 21      # 30-51 (larger)
            depth = 30 + rng.random() * 21      # 30-51 (larger)
            floors = 10 + rng.integers(0, 20)   # 10-29
            floor_height = 16.5 + rng.random() * 3.0  # 16.5-19.5 (was 9.75-11.25)
            roof_type = rng.choice([RoofType.FLAT, RoofType.PYRAMID, RoofType.DOME])
            wall_style = WallStyle.GLASS
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_rooftop_features = True
            wall_inset = rng.random() * 0.5  # Upper floors can taper
            has_chimney = False
            
        elif building_type == BuildingType.WAREHOUSE:
            # Keep within unit-test "reasonable dimensions" bounds (<= 100)
            width = 60 + rng.random() * 40      # 60-100
            depth = 51 + rng.random() * 49      # 51-100
            floors = 1 if rng.random() < 0.7 else 2
            floor_height = 14 + rng.random() * 6.0  # 14-20
            roof_type = rng.choice([RoofType.FLAT, RoofType.SHED, RoofType.BARREL])
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "industrial")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.FACTORY:
            width = 72 + rng.random() * 51      # 72-123 (larger)
            depth = 60 + rng.random() * 42      # 60-102 (larger)
            floors = 2 + rng.integers(0, 2)
            floor_height = 18 + rng.random() * 9.0  # 18-27 (was 9-13.5)
            roof_type = rng.choice([RoofType.FLAT, RoofType.SHED])
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "industrial")
            has_balconies = False
            has_chimney = True  # Smokestacks
            has_chimney = True
            
        elif building_type == BuildingType.TEMPLE:
            width = 42 + rng.random() * 33      # 42-75 (larger)
            depth = 51 + rng.random() * 33      # 51-84 (larger)
            floors = 1 + rng.integers(0, 3)
            floor_height = 18 + rng.random() * 9.0  # 18-27 (was 9-13.5)
            roof_type = rng.choice([RoofType.PYRAMID, RoofType.DOME, RoofType.STEPPED])
            wall_style = WallStyle.STONE
            colors = ColorPalette.random(rng, "ancient")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.OBSERVATORY:
            width = 27 + rng.random() * 18      # 27-45 (larger)
            depth = 27 + rng.random() * 18      # 27-45 (larger)
            floors = 2 + rng.integers(0, 2)
            floor_height = 18 + rng.random() * 6.0  # 18-24 (was 10.5-13.5)
            roof_type = RoofType.DOME
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.RUINS:
            width = 33 + rng.random() * 42      # 33-75 (larger)
            depth = 27 + rng.random() * 39      # 27-66 (larger)
            floors = 1 + rng.integers(0, 2)
            floor_height = 15 + rng.random() * 6.0  # 15-21 (was 9-12)
            roof_type = RoofType.FLAT  # Ruins have no roof
            wall_style = WallStyle.STONE
            colors = ColorPalette.random(rng, "ancient")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.PARKOUR:
            # Parkour playground - MASSIVE play area
            width = 150 + rng.random() * 120   # 150-270 (larger)
            depth = 150 + rng.random() * 120   # 150-270 (larger)
            floors = 6 + rng.integers(0, 5)    # Many vertical levels (6-10)
            floor_height = 15 + rng.random() * 6.0  # 15-21 (was 9-12)
            roof_type = RoofType.FLAT  # No roof, open air
            wall_style = WallStyle.METAL  # Industrial/gym look
            colors = ColorPalette.random(rng, rng.choice(["industrial", "commercial", "alien"]))
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.CABIN:
            width = 15 + rng.random() * 15      # 15-30 (larger)
            depth = 15 + rng.random() * 15      # 15-30 (larger)
            floors = 1 if rng.random() < 0.7 else 2
            floor_height = 16.5 + rng.random() * 4.5  # 16.5-21 (was 9.75-12)
            roof_type = rng.choice([RoofType.GABLED, RoofType.SHED])
            wall_style = WallStyle.WOOD
            colors = ColorPalette.random(rng, "residential")
            has_balconies = floors > 1 and rng.random() < 0.5
            has_chimney = True
            
        # =====================================================================
        # MODERN HOUSE TYPES (from Low Poly Modern House Pack analysis)
        # =====================================================================
        
        elif building_type == BuildingType.MODERN_HOUSE:
            width = 27 + rng.random() * 21     # 27-48 (larger)
            depth = 24 + rng.random() * 18     # 24-42 (larger)
            floors = 2 + rng.integers(0, 2)    # 2-3
            floor_height = 18.0 + rng.random() * 3.0  # 18-21 (was 10.5-12)
            roof_type = RoofType.FLAT
            wall_style = rng.choice([WallStyle.CONCRETE, WallStyle.GLASS])
            colors = ColorPalette.random(rng, "modern")
            has_balconies = rng.random() < 0.6
            has_chimney = False
            
        elif building_type == BuildingType.SPLIT_LEVEL:
            width = 30 + rng.random() * 18     # 30-48 (larger)
            depth = 24 + rng.random() * 15     # 24-39 (larger)
            floors = 3                          # Always 3 offset levels
            floor_height = 15.0 + rng.random() * 3.0  # 15-18 (was 8.25-9.75)
            roof_type = rng.choice([RoofType.FLAT, RoofType.SHED])
            wall_style = rng.choice([WallStyle.BRICK, WallStyle.SOLID])
            colors = ColorPalette.random(rng, "residential")
            has_balconies = False
            has_chimney = rng.random() < 0.3
            
        elif building_type == BuildingType.RANCH:
            width = 45 + rng.random() * 33     # 45-78 (larger)
            depth = 24 + rng.random() * 15     # 24-39 (larger)
            floors = 1                          # Always single floor
            floor_height = 18.0 + rng.random() * 4.5  # 18-22.5 (was 10.5-12.75)
            roof_type = rng.choice([RoofType.GABLED, RoofType.HIPPED])
            wall_style = rng.choice([WallStyle.BRICK, WallStyle.WOOD])
            colors = ColorPalette.random(rng, "residential")
            has_balconies = False
            has_chimney = rng.random() < 0.5
            
        elif building_type == BuildingType.BUNGALOW:
            width = 8 + rng.random() * 6       # 8-14
            depth = 8 + rng.random() * 6       # 8-14
            floors = 1
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = rng.choice([RoofType.GABLED, RoofType.HIPPED])
            wall_style = rng.choice([WallStyle.WOOD, WallStyle.BRICK])
            colors = ColorPalette.random(rng, "residential")
            has_balconies = False
            has_chimney = rng.random() < 0.4
            
        elif building_type == BuildingType.TOWNHOUSE:
            width = 6 + rng.random() * 4       # 6-10 (narrow)
            depth = 12 + rng.random() * 8      # 12-20 (deep)
            floors = 3 + rng.integers(0, 2)    # 3-4
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = rng.choice([RoofType.FLAT, RoofType.GABLED])
            wall_style = rng.choice([WallStyle.BRICK, WallStyle.SOLID])
            colors = ColorPalette.random(rng, "residential")
            has_balconies = rng.random() < 0.4
            has_chimney = False
            
        elif building_type == BuildingType.LOFT:
            width = 15 + rng.random() * 10     # 15-25
            depth = 12 + rng.random() * 8      # 12-20
            floors = 2
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height  # Very tall ceilings
            roof_type = RoofType.FLAT
            wall_style = rng.choice([WallStyle.BRICK, WallStyle.CONCRETE])
            colors = ColorPalette.random(rng, "industrial")
            has_balconies = rng.random() < 0.3
            has_chimney = False
            
        elif building_type == BuildingType.PENTHOUSE:
            width = 20 + rng.random() * 15     # 20-35
            depth = 18 + rng.random() * 12     # 18-30
            floors = 1                          # Single floor luxury
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.GLASS
            colors = ColorPalette.random(rng, "modern")
            has_balconies = True
            has_rooftop_features = True
            has_chimney = False
            
        elif building_type == BuildingType.DUPLEX:
            width = 12 + rng.random() * 8      # 12-20
            depth = 10 + rng.random() * 6      # 10-16
            floors = 2
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = rng.choice([RoofType.GABLED, RoofType.HIPPED])
            wall_style = rng.choice([WallStyle.BRICK, WallStyle.SOLID])
            colors = ColorPalette.random(rng, "residential")
            has_balconies = rng.random() < 0.3
            has_chimney = rng.random() < 0.3
            
        elif building_type == BuildingType.MANSION:
            width = 30 + rng.random() * 20     # 30-50
            depth = 25 + rng.random() * 15     # 25-40
            floors = 2 + rng.integers(0, 2)    # 2-3
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height
            roof_type = rng.choice([RoofType.HIPPED, RoofType.MANSARD])
            wall_style = rng.choice([WallStyle.STONE, WallStyle.BRICK])
            colors = ColorPalette.random(rng, "residential")
            has_balconies = True
            has_chimney = True
            
        elif building_type == BuildingType.COTTAGE:
            width = 7 + rng.random() * 5       # 7-12
            depth = 6 + rng.random() * 5       # 6-11
            floors = 1 if rng.random() < 0.6 else 2
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = rng.choice([RoofType.GABLED, RoofType.HIPPED])
            wall_style = rng.choice([WallStyle.STONE, WallStyle.WOOD])
            colors = ColorPalette.random(rng, "residential")
            has_balconies = False
            has_chimney = rng.random() < 0.7
            
        elif building_type == BuildingType.BEACH_HOUSE:
            width = 12 + rng.random() * 8      # 12-20
            depth = 10 + rng.random() * 6      # 10-16
            floors = 2
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height  # Elevated + tall
            roof_type = rng.choice([RoofType.GABLED, RoofType.HIPPED])
            wall_style = WallStyle.WOOD
            colors = ColorPalette(
                primary=(0.9, 0.92, 0.95),      # Light/white
                secondary=(0.4, 0.6, 0.8),      # Ocean blue
                roof=(0.5, 0.45, 0.4),
                window=(0.5, 0.7, 0.9)
            )
            has_balconies = True
            has_chimney = False
            
        elif building_type == BuildingType.TREE_HOUSE:
            size = 8 + rng.random() * 6
            width = size
            depth = size
            floors = 1
            floor_height = 15.0 + rng.random() * 5.0  # Very elevated
            roof_type = rng.choice([RoofType.GABLED, RoofType.PYRAMID])
            wall_style = WallStyle.WOOD
            colors = ColorPalette.random(rng, "residential")
            has_balconies = True
            has_chimney = False
            
        elif building_type == BuildingType.TINY_HOUSE:
            width = 4 + rng.random() * 3       # 4-7
            depth = 6 + rng.random() * 4       # 6-10
            floors = 1
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = rng.choice([RoofType.GABLED, RoofType.SHED, RoofType.BARREL])
            wall_style = rng.choice([WallStyle.WOOD, WallStyle.METAL])
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_chimney = rng.random() < 0.2
            
        elif building_type == BuildingType.A_FRAME:
            width = 8 + rng.random() * 6       # 8-14
            depth = 10 + rng.random() * 6      # 10-16
            floors = 2
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = RoofType.GABLED        # Very steep gable
            wall_style = WallStyle.WOOD
            colors = ColorPalette.random(rng, "residential")
            has_balconies = rng.random() < 0.4
            has_chimney = rng.random() < 0.5
            
        elif building_type == BuildingType.CONTAINER_HOUSE:
            width = 6 + rng.random() * 4       # 6-10 (narrow like container)
            depth = 12 + rng.random() * 8      # 12-20 (long)
            floors = 1 + rng.integers(0, 2)    # 1-2
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "industrial")
            has_balconies = rng.random() < 0.3
            has_chimney = False
            
        elif building_type == BuildingType.GEODOME_HOME:
            size = 12 + rng.random() * 8
            width = size
            depth = size
            floors = 2
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = RoofType.GEODESIC
            wall_style = rng.choice([WallStyle.GLASS, WallStyle.SOLID])
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.FLOATING_HOUSE:
            width = 10 + rng.random() * 6      # 10-16
            depth = 12 + rng.random() * 8      # 12-20
            floors = 1 if rng.random() < 0.6 else 2
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = rng.choice([RoofType.FLAT, RoofType.SHED])
            wall_style = WallStyle.WOOD
            colors = ColorPalette.random(rng, "modern")
            has_balconies = True
            has_chimney = False
            
        elif building_type == BuildingType.UNDERGROUND_HOME:
            width = 15 + rng.random() * 10     # 15-25
            depth = 12 + rng.random() * 8      # 12-20
            floors = 1                          # Mostly underground
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = RoofType.FLAT          # Ground level
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.CAPSULE:
            size = 5 + rng.random() * 3
            width = size
            depth = size
            floors = 1
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = RoofType.DOME
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "alien")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.GREENHOUSE:
            width = 10 + rng.random() * 8      # 10-18
            depth = 15 + rng.random() * 10     # 15-25
            floors = 1
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height
            roof_type = rng.choice([RoofType.GABLED, RoofType.BARREL])
            wall_style = WallStyle.GLASS
            colors = ColorPalette(
                primary=(0.8, 0.85, 0.9),
                secondary=(0.3, 0.6, 0.3),      # Green accents
                roof=(0.75, 0.8, 0.85),
                window=(0.6, 0.8, 0.9)
            )
            has_balconies = False
            has_chimney = False
        
        # =====================================================================
        # ANCIENT/CULTURAL STRUCTURES
        # =====================================================================
        
        elif building_type == BuildingType.NUBIAN_COMPLEX:
            width = 40 + rng.random() * 30     # 40-70
            depth = 35 + rng.random() * 25     # 35-60
            floors = 2 + rng.integers(0, 2)    # 2-3
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.STONE
            colors = ColorPalette(
                primary=(0.75, 0.65, 0.5),     # Sandy tan
                secondary=(0.6, 0.5, 0.4),     # Darker accent
                roof=(0.7, 0.6, 0.45),
                window=(0.3, 0.25, 0.2)
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.PYRAMID_TEMPLE:
            size = 30 + rng.random() * 30
            width = size
            depth = size
            floors = 4 + rng.integers(0, 3)    # Stepped tiers
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 2.0
            roof_type = RoofType.STEPPED
            wall_style = WallStyle.STONE
            colors = ColorPalette.random(rng, "ancient")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.STONE_PORTAL:
            width = 12 + rng.random() * 8      # 12-20 (arch width)
            depth = 4 + rng.random() * 3       # 4-7 (thin)
            floors = 1
            floor_height = 15.0 + rng.random() * 10.0  # Very tall arch
            roof_type = RoofType.FLAT
            wall_style = WallStyle.STONE
            colors = ColorPalette.random(rng, "ancient")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.ZIGGURAT:
            size = 40 + rng.random() * 40
            width = size
            depth = size
            floors = 5 + rng.integers(0, 3)
            floor_height = 4.0 + rng.random() * 2.0
            roof_type = RoofType.STEPPED
            wall_style = WallStyle.BRICK
            colors = ColorPalette.random(rng, "ancient")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.OBELISK:
            size = 4 + rng.random() * 3
            width = size
            depth = size
            floors = 8 + rng.integers(0, 6)    # Very tall
            floor_height = 4.0 + rng.random()
            roof_type = RoofType.PYRAMID
            wall_style = WallStyle.STONE
            colors = ColorPalette.random(rng, "ancient")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.AMPHITHEATER:
            width = 50 + rng.random() * 40
            depth = 40 + rng.random() * 30
            floors = 3 + rng.integers(0, 2)
            floor_height = 4.0 + rng.random() * 2.0
            roof_type = RoofType.FLAT  # Open air
            wall_style = WallStyle.STONE
            colors = ColorPalette.random(rng, "ancient")
            has_balconies = True  # Seating tiers
            has_chimney = False
            
        elif building_type == BuildingType.COLOSSEUM:
            size = 80 + rng.random() * 40
            width = size
            depth = size * 0.85
            floors = 4 + rng.integers(0, 2)
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 2.0
            roof_type = RoofType.FLAT
            wall_style = WallStyle.STONE
            colors = ColorPalette.random(rng, "ancient")
            has_balconies = True
            has_chimney = False
            
        elif building_type == BuildingType.AQUEDUCT:
            width = 100 + rng.random() * 100   # Long span
            depth = 6 + rng.random() * 4       # Narrow
            floors = 2 + rng.integers(0, 2)
            floor_height = 10.0 + rng.random() * 5.0
            roof_type = RoofType.FLAT
            wall_style = WallStyle.STONE
            colors = ColorPalette.random(rng, "ancient")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.MAUSOLEUM:
            size = 20 + rng.random() * 15
            width = size
            depth = size
            floors = 2 + rng.integers(0, 2)
            floor_height = 18.0 + rng.random() * 6.0  # Scaled for player height
            roof_type = rng.choice([RoofType.DOME, RoofType.PYRAMID])
            wall_style = WallStyle.STONE
            colors = ColorPalette(
                primary=(0.85, 0.82, 0.78),    # White marble
                secondary=(0.7, 0.65, 0.6),
                roof=(0.8, 0.77, 0.72),
                window=(0.2, 0.22, 0.25)
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.PAGODA_TOWER:
            size = 15 + rng.random() * 10
            width = size
            depth = size
            floors = 5 + rng.integers(0, 4)
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 2.0
            roof_type = RoofType.PAGODA
            wall_style = WallStyle.WOOD
            colors = ColorPalette(
                primary=(0.6, 0.45, 0.35),     # Dark wood
                secondary=(0.75, 0.3, 0.2),    # Red accents
                roof=(0.3, 0.28, 0.25),
                window=(0.4, 0.35, 0.3)
            )
            has_balconies = True
            has_chimney = False
        
        # =====================================================================
        # CYBERPUNK/NEON STRUCTURES
        # =====================================================================
        
        elif building_type == BuildingType.CYBERPUNK_BAR:
            width = 15 + rng.random() * 10
            depth = 12 + rng.random() * 8
            floors = 1 + rng.integers(0, 2)
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 2.0
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette(
                primary=(0.15, 0.12, 0.2),     # Dark purple-black
                secondary=(0.9, 0.2, 0.6),     # Neon pink
                roof=(0.12, 0.1, 0.15),
                window=(0.2, 0.8, 0.9)         # Neon cyan
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.NEON_DINER:
            width = 18 + rng.random() * 10
            depth = 10 + rng.random() * 6
            floors = 1
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 1.5
            roof_type = rng.choice([RoofType.FLAT, RoofType.BARREL])
            wall_style = WallStyle.METAL
            colors = ColorPalette(
                primary=(0.8, 0.75, 0.85),     # Chrome/silver
                secondary=(1.0, 0.4, 0.4),     # Neon red
                roof=(0.7, 0.68, 0.75),
                window=(0.3, 0.9, 0.5)         # Neon green
            )
            has_awning = True
            awning_depth = 3.0
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.ARCADE:
            width = 12 + rng.random() * 8
            depth = 20 + rng.random() * 10
            floors = 1 + rng.integers(0, 2)
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 1.5
            roof_type = RoofType.FLAT
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette(
                primary=(0.2, 0.18, 0.25),
                secondary=(0.95, 0.9, 0.2),    # Neon yellow
                roof=(0.18, 0.16, 0.22),
                window=(0.8, 0.3, 0.9)         # Purple
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.HOLO_BILLBOARD:
            width = 15 + rng.random() * 10
            depth = 2 + rng.random() * 2
            floors = 1
            floor_height = 20.0 + rng.random() * 15.0  # Very tall
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "alien")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.TECH_SHOP:
            width = 10 + rng.random() * 8
            depth = 12 + rng.random() * 8
            floors = 1 + rng.integers(0, 2)
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.GLASS
            colors = ColorPalette(
                primary=(0.25, 0.28, 0.35),
                secondary=(0.3, 0.85, 0.95),   # Cyan accent
                roof=(0.2, 0.22, 0.28),
                window=(0.4, 0.9, 1.0)
            )
            has_balconies = False
            has_chimney = False
        
        # =====================================================================
        # CITY INFRASTRUCTURE
        # =====================================================================
        
        elif building_type == BuildingType.BUS_STOP:
            width = 8 + rng.random() * 4
            depth = 3 + rng.random() * 2
            floors = 1
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = RoofType.SHED
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.METRO_ENTRANCE:
            width = 10 + rng.random() * 6
            depth = 8 + rng.random() * 4
            floors = 1
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.PARKING_GARAGE:
            width = 40 + rng.random() * 30
            depth = 30 + rng.random() * 20
            floors = 4 + rng.integers(0, 4)
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette.random(rng, "industrial")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.GAS_STATION:
            width = 25 + rng.random() * 15
            depth = 15 + rng.random() * 10
            floors = 1
            floor_height = 10.0 + rng.random() * 3.0  # Tall canopy
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "commercial")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.FIRE_STATION:
            width = 25 + rng.random() * 15
            depth = 20 + rng.random() * 10
            floors = 2 + rng.integers(0, 2)
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.BRICK
            colors = ColorPalette(
                primary=(0.8, 0.3, 0.25),      # Red
                secondary=(0.9, 0.85, 0.8),
                roof=(0.35, 0.32, 0.3),
                window=(0.5, 0.55, 0.6)
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.POLICE_STATION:
            width = 25 + rng.random() * 15
            depth = 20 + rng.random() * 10
            floors = 2 + rng.integers(0, 2)
            floor_height = 18.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette(
                primary=(0.35, 0.4, 0.5),      # Blue-gray
                secondary=(0.2, 0.3, 0.5),
                roof=(0.3, 0.32, 0.38),
                window=(0.5, 0.6, 0.7)
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.POST_OFFICE:
            width = 15 + rng.random() * 10
            depth = 12 + rng.random() * 8
            floors = 1 + rng.integers(0, 2)
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = rng.choice([RoofType.FLAT, RoofType.GABLED])
            wall_style = WallStyle.BRICK
            colors = ColorPalette.random(rng, "commercial")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.LIBRARY:
            width = 25 + rng.random() * 20
            depth = 20 + rng.random() * 15
            floors = 2 + rng.integers(0, 3)
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height
            roof_type = rng.choice([RoofType.GABLED, RoofType.DOME])
            wall_style = rng.choice([WallStyle.STONE, WallStyle.BRICK])
            colors = ColorPalette.random(rng, "residential")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.SCHOOL:
            width = 40 + rng.random() * 30
            depth = 25 + rng.random() * 15
            floors = 2 + rng.integers(0, 2)
            floor_height = 18.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.BRICK
            colors = ColorPalette.random(rng, "residential")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.CHURCH:
            width = 15 + rng.random() * 12
            depth = 25 + rng.random() * 15
            floors = 2 + rng.integers(0, 2)
            floor_height = 10.0 + rng.random() * 3.0
            roof_type = rng.choice([RoofType.GABLED, RoofType.CONICAL])
            wall_style = WallStyle.STONE
            colors = ColorPalette(
                primary=(0.85, 0.82, 0.78),
                secondary=(0.7, 0.65, 0.6),
                roof=(0.4, 0.38, 0.35),
                window=(0.5, 0.6, 0.8)         # Stained glass blue
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.MOSQUE:
            size = 25 + rng.random() * 20
            width = size
            depth = size
            floors = 2 + rng.integers(0, 2)
            floor_height = 18.0 + rng.random() * 6.0  # Scaled for player height
            roof_type = RoofType.DOME
            wall_style = WallStyle.STONE
            colors = ColorPalette(
                primary=(0.9, 0.88, 0.82),     # White/cream
                secondary=(0.3, 0.5, 0.45),    # Teal accent
                roof=(0.25, 0.45, 0.55),       # Blue-teal dome
                window=(0.5, 0.7, 0.8)
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.BANK:
            width = 20 + rng.random() * 15
            depth = 18 + rng.random() * 12
            floors = 2 + rng.integers(0, 3)
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height
            roof_type = rng.choice([RoofType.FLAT, RoofType.GABLED])
            wall_style = WallStyle.STONE
            colors = ColorPalette(
                primary=(0.8, 0.78, 0.75),
                secondary=(0.6, 0.55, 0.5),
                roof=(0.35, 0.33, 0.3),
                window=(0.4, 0.45, 0.5)
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.THEATER:
            width = 35 + rng.random() * 25
            depth = 30 + rng.random() * 20
            floors = 2 + rng.integers(0, 2)
            floor_height = 10.0 + rng.random() * 4.0
            roof_type = rng.choice([RoofType.DOME, RoofType.BARREL])
            wall_style = rng.choice([WallStyle.STONE, WallStyle.BRICK])
            colors = ColorPalette.random(rng, "commercial")
            has_balconies = True
            has_chimney = False
            
        elif building_type == BuildingType.MALL:
            width = 60 + rng.random() * 50
            depth = 50 + rng.random() * 40
            floors = 2 + rng.integers(0, 2)
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette.random(rng, "commercial")
            has_balconies = False
            has_rooftop_features = True
            has_chimney = False
            
        elif building_type == BuildingType.SUPERMARKET:
            width = 35 + rng.random() * 25
            depth = 25 + rng.random() * 15
            floors = 1
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette.random(rng, "commercial")
            has_balconies = False
            has_chimney = False
        
        # =====================================================================
        # MODULAR SCI-FI STRUCTURES
        # =====================================================================
        
        elif building_type == BuildingType.MODULAR_HUB:
            size = 20 + rng.random() * 15
            width = size
            depth = size
            floors = 2 + rng.integers(0, 2)
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 2.0
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_rooftop_features = True
            has_chimney = False
            
        elif building_type == BuildingType.POWER_NODE:
            size = 10 + rng.random() * 8
            width = size
            depth = size
            floors = 2 + rng.integers(0, 2)
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 2.0
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette(
                primary=(0.4, 0.42, 0.45),
                secondary=(0.9, 0.7, 0.2),     # Energy yellow
                roof=(0.35, 0.37, 0.4),
                window=(0.8, 0.9, 0.4)
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.COMM_TOWER:
            size = 8 + rng.random() * 5
            width = size
            depth = size
            floors = 8 + rng.integers(0, 6)
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_rooftop_features = True  # Antenna
            has_chimney = False
            
        elif building_type == BuildingType.SHIELD_GENERATOR:
            size = 15 + rng.random() * 10
            width = size
            depth = size
            floors = 2
            floor_height = 18.0 + rng.random() * 6.0  # Scaled for player height
            roof_type = RoofType.DOME
            wall_style = WallStyle.METAL
            colors = ColorPalette(
                primary=(0.35, 0.4, 0.45),
                secondary=(0.3, 0.6, 0.9),     # Blue energy
                roof=(0.3, 0.5, 0.7),
                window=(0.4, 0.7, 0.95)
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.CRYO_CHAMBER:
            width = 15 + rng.random() * 10
            depth = 20 + rng.random() * 15
            floors = 2
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette(
                primary=(0.7, 0.75, 0.8),
                secondary=(0.4, 0.7, 0.85),    # Ice blue
                roof=(0.6, 0.65, 0.72),
                window=(0.5, 0.8, 0.95)
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.CARGO_BAY:
            width = 30 + rng.random() * 25
            depth = 25 + rng.random() * 20
            floors = 2
            floor_height = 10.0 + rng.random() * 4.0
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "industrial")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.DOCKING_ARM:
            width = 8 + rng.random() * 5
            depth = 40 + rng.random() * 30
            floors = 2
            floor_height = 12.0 + rng.random() * 5.0
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.REACTOR_CORE:
            size = 25 + rng.random() * 15
            width = size
            depth = size
            floors = 3 + rng.integers(0, 2)
            floor_height = 10.0 + rng.random() * 4.0
            roof_type = RoofType.DOME
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette(
                primary=(0.5, 0.52, 0.55),
                secondary=(0.2, 0.8, 0.4),     # Reactor green
                roof=(0.45, 0.47, 0.5),
                window=(0.3, 0.9, 0.5)
            )
            has_balconies = False
            has_rooftop_features = True
            has_chimney = False
            
        elif building_type == BuildingType.HOSPITAL:
            width = 30 + rng.random() * 20     # 30-50
            depth = 25 + rng.random() * 15     # 25-40
            floors = 3 + rng.integers(0, 5)    # 3-7
            floor_height = 18.0 + rng.random() * 3.0  # Scaled for player height  # Tall ceilings (7.5-8.5)
            roof_type = RoofType.FLAT
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette(
                primary=(0.92, 0.92, 0.95),  # White/pale
                secondary=(0.8, 0.2, 0.2),   # Red cross accent
                roof=(0.4, 0.42, 0.45),
                window=(0.5, 0.7, 0.9)
            )
            has_balconies = False
            has_rooftop_features = True  # Helipad
            has_chimney = False
            
        elif building_type == BuildingType.RESTAURANT:
            width = 12 + rng.random() * 10     # 12-22
            depth = 10 + rng.random() * 8      # 10-18
            floors = 1 if rng.random() < 0.6 else 2
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 1.5
            roof_type = rng.choice([RoofType.FLAT, RoofType.GABLED, RoofType.SHED])
            wall_style = rng.choice([WallStyle.BRICK, WallStyle.WOOD])
            colors = ColorPalette.random(rng, "commercial")
            has_balconies = False
            has_awning = True
            awning_depth = 3.0 + rng.random() * 2.0
            has_chimney = rng.random() < 0.4
            
        elif building_type == BuildingType.SILO:
            # Cylindrical storage (treat as square footprint)
            size = 8 + rng.random() * 8
            width = size
            depth = size
            floors = 4 + rng.integers(0, 4)    # Tall
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 2.0
            roof_type = RoofType.CONICAL
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "industrial")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.POWERPLANT:
            width = 40 + rng.random() * 30
            depth = 35 + rng.random() * 25
            floors = 2 + rng.integers(0, 2)
            floor_height = 10.0 + rng.random() * 4.0  # Very tall
            roof_type = RoofType.FLAT
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette.random(rng, "industrial")
            has_balconies = False
            has_chimney = True  # Cooling towers
            has_rooftop_features = True
            
        elif building_type == BuildingType.SKYSCRAPER:
            width = 20 + rng.random() * 15     # 20-35
            depth = 20 + rng.random() * 15     # 20-35
            floors = 20 + rng.integers(0, 30)  # 20-49 floors!
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 0.5
            roof_type = rng.choice([RoofType.FLAT, RoofType.PYRAMID, RoofType.DOME])
            wall_style = WallStyle.GLASS
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_rooftop_features = True  # Antenna, helipad
            wall_inset = rng.random() * 0.3
            has_chimney = False
            
        elif building_type == BuildingType.PLANETARIUM:
            size = 25 + rng.random() * 15
            width = size
            depth = size
            floors = 2
            floor_height = 18.0 + rng.random() * 4.0  # Scaled for player height
            roof_type = RoofType.GEODESIC
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.SOLAR_STATION:
            width = 20 + rng.random() * 20
            depth = 15 + rng.random() * 15
            floors = 1 if rng.random() < 0.7 else 2
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 2.0
            roof_type = RoofType.FLAT  # Covered with solar panels
            wall_style = WallStyle.METAL
            colors = ColorPalette(
                primary=(0.3, 0.35, 0.4),
                secondary=(0.2, 0.3, 0.5),  # Blue panels
                roof=(0.15, 0.2, 0.35),     # Dark blue
                window=(0.4, 0.5, 0.7)
            )
            has_balconies = False
            has_rooftop_features = True
            has_chimney = False
            
        elif building_type == BuildingType.LANDING_PAD:
            size = 25 + rng.random() * 15
            width = size
            depth = size
            floors = 1
            floor_height = 18.0 + rng.random() * 6.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette(
                primary=(0.4, 0.42, 0.45),
                secondary=(0.9, 0.6, 0.1),  # Orange markings
                roof=(0.35, 0.38, 0.42),
                window=(0.5, 0.55, 0.6)
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.SPACEPORT:
            width = 60 + rng.random() * 40
            depth = 50 + rng.random() * 30
            floors = 2 + rng.integers(0, 2)
            floor_height = 10.0 + rng.random() * 4.0
            roof_type = RoofType.BARREL
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_rooftop_features = True
            has_chimney = False
            
        elif building_type == BuildingType.RESEARCH_LAB:
            width = 25 + rng.random() * 20
            depth = 20 + rng.random() * 15
            floors = 2 + rng.integers(0, 3)
            floor_height = 18.0 + rng.random() * 3.0  # Scaled for player height * 1.5
            roof_type = rng.choice([RoofType.FLAT, RoofType.DOME])
            wall_style = rng.choice([WallStyle.CONCRETE, WallStyle.GLASS])
            colors = ColorPalette.random(rng, "modern")
            has_balconies = False
            has_rooftop_features = True  # Satellite dishes
            has_chimney = False
            
        elif building_type == BuildingType.PLAYGROUND:
            width = 25 + rng.random() * 20
            depth = 25 + rng.random() * 20
            floors = 3 + rng.integers(0, 3)  # Multi-level play structure
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 2.0
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "commercial")  # Bright colors
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.STADIUM:
            width = 80 + rng.random() * 60
            depth = 60 + rng.random() * 40
            floors = 4 + rng.integers(0, 4)
            floor_height = 18.0 + rng.random() * 6.0  # Scaled for player height
            roof_type = RoofType.FLAT
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette.random(rng, "modern")
            has_balconies = True  # Seating tiers
            has_chimney = False
            
        elif building_type == BuildingType.POOL:
            width = 30 + rng.random() * 20
            depth = 20 + rng.random() * 15
            floors = 1
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 2.0
            roof_type = RoofType.FLAT
            wall_style = WallStyle.CONCRETE
            colors = ColorPalette(
                primary=(0.85, 0.88, 0.9),
                secondary=(0.3, 0.6, 0.9),  # Pool blue
                roof=(0.4, 0.45, 0.5),
                window=(0.4, 0.7, 0.9)
            )
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.LIGHTHOUSE:
            size = 8 + rng.random() * 4
            width = size
            depth = size
            floors = 6 + rng.integers(0, 4)  # Tall and narrow
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 1.5
            roof_type = RoofType.CONICAL
            wall_style = rng.choice([WallStyle.BRICK, WallStyle.STONE])
            colors = ColorPalette(
                primary=(0.95, 0.92, 0.88),  # White
                secondary=(0.8, 0.2, 0.15),  # Red stripes
                roof=(0.25, 0.28, 0.32),
                window=(0.6, 0.7, 0.8)
            )
            has_balconies = True  # Observation deck at top
            has_chimney = False
            
        elif building_type == BuildingType.BRIDGE:
            width = 60 + rng.random() * 60  # Spanning distance
            depth = 8 + rng.random() * 6    # Narrow walkway
            floors = 1
            floor_height = 12.0 + rng.random() * 8.0  # Elevated
            roof_type = RoofType.FLAT
            wall_style = WallStyle.METAL
            colors = ColorPalette.random(rng, "industrial")
            has_balconies = False
            has_chimney = False
            
        elif building_type == BuildingType.GAZEBO:
            size = 8 + rng.random() * 6
            width = size
            depth = size
            floors = 1
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height * 2.0
            roof_type = rng.choice([RoofType.PYRAMID, RoofType.CONICAL, RoofType.DOME])
            wall_style = WallStyle.WOOD
            colors = ColorPalette.random(rng, "residential")
            has_balconies = False
            has_chimney = False
            
        else:  # Generic fallback
            width = 12 + rng.random() * 12
            depth = 10 + rng.random() * 10
            floors = 2 + rng.integers(0, 3)
            floor_height = 15.0 + rng.random() * 3.0  # Scaled for player height  # (6-7)
            roof_type = RoofType.FLAT
            wall_style = WallStyle.SOLID
            colors = ColorPalette.random(rng, "modern")
            has_balconies = rng.random() < 0.3
            has_chimney = False
        
        # Apply world scale factor for TILE_SCALE=3.0
        # Note: Some building types (first ~10) are already scaled, so apply a smaller
        # scale to ensure all buildings are appropriately sized
        WORLD_SCALE = 1.5  # 1.5x for 3x world scale
        
        # Only apply scale to building types that haven't been manually updated
        already_scaled = {
            BuildingType.HOUSE, BuildingType.APARTMENT, BuildingType.VILLA, 
            BuildingType.SHOP, BuildingType.OFFICE, BuildingType.TOWER,
            BuildingType.WAREHOUSE, BuildingType.FACTORY, BuildingType.TEMPLE,
            BuildingType.OBSERVATORY, BuildingType.RUINS, BuildingType.PARKOUR,
            BuildingType.CABIN, BuildingType.MODERN_HOUSE, BuildingType.SPLIT_LEVEL,
            BuildingType.RANCH
        }
        
        if building_type not in already_scaled:
            width *= WORLD_SCALE
            depth *= WORLD_SCALE
            floor_height *= WORLD_SCALE
            awning_depth *= WORLD_SCALE

        # Enforce invariants expected by unit tests and keep values sane.
        width = float(np.clip(width, 1.0, 100.0))
        depth = float(np.clip(depth, 1.0, 100.0))
        floors = int(np.clip(floors, 1, 50))
        floor_height = float(np.clip(floor_height, 2.0, 20.0))
        
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
            awning_depth=awning_depth if building_type == BuildingType.SHOP else 3.0,  # 3.0 base (scaled)
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
        min_dim = min(self.width, self.depth)
        # Add roof height based on type
        if self.roof_type == RoofType.GABLED:
            return base + min_dim * 0.4
        elif self.roof_type == RoofType.PYRAMID:
            return base + min_dim * 0.5
        elif self.roof_type == RoofType.DOME:
            return base + min_dim * 0.3
        elif self.roof_type == RoofType.HIPPED:
            return base + min_dim * 0.35
        elif self.roof_type == RoofType.CONICAL:
            return base + min_dim * 0.6  # Pointed cone
        elif self.roof_type == RoofType.PAGODA:
            return base + min_dim * 0.45
        elif self.roof_type == RoofType.MANSARD:
            return base + min_dim * 0.3
        elif self.roof_type == RoofType.GEODESIC:
            return base + min_dim * 0.4
        elif self.roof_type == RoofType.BUTTERFLY:
            return base + min_dim * 0.15  # Low V
        elif self.roof_type == RoofType.SAWTOOTH:
            return base + min_dim * 0.2
        elif self.roof_type == RoofType.STEPPED:
            return base + min_dim * 0.5
        elif self.roof_type == RoofType.BARREL:
            return base + min_dim * 0.25
        elif self.roof_type == RoofType.SHED:
            return base + min_dim * 0.2
        else:
            return base + 0.5  # Flat roof parapet


# ============================================================================
# BUILDING ARCHETYPES - Pre-defined building configurations
# ============================================================================

def get_building_type_for_biome(biome: str, rng: np.random.Generator) -> BuildingType:
    """Choose an appropriate building type based on biome."""
    if biome in ["urban", "city"]:
        weights = {
            BuildingType.APARTMENT: 0.08,
            BuildingType.OFFICE: 0.08,
            BuildingType.TOWER: 0.05,
            BuildingType.SKYSCRAPER: 0.06,
            BuildingType.SHOP: 0.07,
            BuildingType.RESTAURANT: 0.04,
            BuildingType.WAREHOUSE: 0.03,
            BuildingType.TOWNHOUSE: 0.06,
            BuildingType.LOFT: 0.04,
            BuildingType.PENTHOUSE: 0.03,
            BuildingType.MODERN_HOUSE: 0.04,
            BuildingType.HOSPITAL: 0.03,
            BuildingType.HOTEL: 0.05,
            BuildingType.STADIUM: 0.02,
            BuildingType.PARKOUR: 0.03,
            BuildingType.CONTAINER_HOUSE: 0.02,
            # New city infrastructure
            BuildingType.PARKING_GARAGE: 0.04,
            BuildingType.FIRE_STATION: 0.02,
            BuildingType.POLICE_STATION: 0.02,
            BuildingType.LIBRARY: 0.02,
            BuildingType.SCHOOL: 0.03,
            BuildingType.BANK: 0.02,
            BuildingType.THEATER: 0.02,
            BuildingType.MALL: 0.02,
            BuildingType.SUPERMARKET: 0.03,
            BuildingType.BUS_STOP: 0.02,
            BuildingType.METRO_ENTRANCE: 0.02,
        }
    elif biome in ["suburban", "plains"]:
        weights = {
            BuildingType.HOUSE: 0.15,
            BuildingType.MODERN_HOUSE: 0.1,
            BuildingType.RANCH: 0.08,
            BuildingType.BUNGALOW: 0.08,
            BuildingType.SPLIT_LEVEL: 0.05,
            BuildingType.VILLA: 0.05,
            BuildingType.DUPLEX: 0.06,
            BuildingType.COTTAGE: 0.05,
            BuildingType.SHOP: 0.08,
            BuildingType.RESTAURANT: 0.05,
            BuildingType.APARTMENT: 0.06,
            BuildingType.PLAYGROUND: 0.06,
            BuildingType.POOL: 0.04,
            BuildingType.GAZEBO: 0.05,
            BuildingType.GREENHOUSE: 0.04,
        }
    elif biome in ["desert", "canyon"]:
        weights = {
            BuildingType.TEMPLE: 0.1,
            BuildingType.RUINS: 0.12,
            BuildingType.HOUSE: 0.12,
            BuildingType.WAREHOUSE: 0.05,
            BuildingType.MONUMENT: 0.08,
            BuildingType.SOLAR_STATION: 0.08,
            BuildingType.OBSERVATORY: 0.05,
            BuildingType.RESEARCH_LAB: 0.05,
            # Ancient structures
            BuildingType.NUBIAN_COMPLEX: 0.08,
            BuildingType.PYRAMID_TEMPLE: 0.07,
            BuildingType.ZIGGURAT: 0.06,
            BuildingType.OBELISK: 0.05,
            BuildingType.MAUSOLEUM: 0.04,
            BuildingType.STONE_PORTAL: 0.03,
            BuildingType.AMPHITHEATER: 0.02,
        }
    elif biome in ["forest", "jungle"]:
        weights = {
            BuildingType.CABIN: 0.2,
            BuildingType.TREE_HOUSE: 0.12,
            BuildingType.COTTAGE: 0.1,
            BuildingType.A_FRAME: 0.08,
            BuildingType.HOUSE: 0.1,
            BuildingType.TEMPLE: 0.1,
            BuildingType.RUINS: 0.1,
            BuildingType.OBSERVATORY: 0.05,
            BuildingType.GAZEBO: 0.08,
            BuildingType.GREENHOUSE: 0.07,
        }
    elif biome in ["snow", "tundra"]:
        weights = {
            BuildingType.HOUSE: 0.25,
            BuildingType.CABIN: 0.2,
            BuildingType.FACTORY: 0.12,
            BuildingType.WAREHOUSE: 0.1,
            BuildingType.OBSERVATORY: 0.1,
            BuildingType.RESEARCH_LAB: 0.08,
            BuildingType.POWERPLANT: 0.08,
            BuildingType.APARTMENT: 0.1,
        }
    elif biome in ["coast", "beach", "ocean"]:
        weights = {
            BuildingType.BEACH_HOUSE: 0.2,
            BuildingType.LIGHTHOUSE: 0.12,
            BuildingType.FLOATING_HOUSE: 0.1,
            BuildingType.VILLA: 0.1,
            BuildingType.MODERN_HOUSE: 0.08,
            BuildingType.RESTAURANT: 0.08,
            BuildingType.HOTEL: 0.1,
            BuildingType.POOL: 0.08,
            BuildingType.GAZEBO: 0.08,
            BuildingType.PENTHOUSE: 0.06,
        }
    elif biome in ["futuristic", "scifi", "alien"]:
        weights = {
            BuildingType.SKYSCRAPER: 0.07,
            BuildingType.PLANETARIUM: 0.05,
            BuildingType.SOLAR_STATION: 0.05,
            BuildingType.LANDING_PAD: 0.05,
            BuildingType.SPACEPORT: 0.05,
            BuildingType.RESEARCH_LAB: 0.05,
            BuildingType.TOWER: 0.06,
            BuildingType.OBSERVATORY: 0.05,
            BuildingType.GEODOME_HOME: 0.05,
            BuildingType.CAPSULE: 0.05,
            BuildingType.CONTAINER_HOUSE: 0.04,
            BuildingType.MODERN_HOUSE: 0.04,
            BuildingType.PARKOUR: 0.04,
            # Modular sci-fi
            BuildingType.MODULAR_HUB: 0.05,
            BuildingType.POWER_NODE: 0.04,
            BuildingType.COMM_TOWER: 0.05,
            BuildingType.SHIELD_GENERATOR: 0.04,
            BuildingType.CRYO_CHAMBER: 0.03,
            BuildingType.CARGO_BAY: 0.04,
            BuildingType.DOCKING_ARM: 0.03,
            BuildingType.REACTOR_CORE: 0.03,
            # Cyberpunk
            BuildingType.CYBERPUNK_BAR: 0.02,
            BuildingType.NEON_DINER: 0.02,
            BuildingType.ARCADE: 0.02,
            BuildingType.TECH_SHOP: 0.02,
            BuildingType.HOLO_BILLBOARD: 0.01,
        }
    elif biome in ["cyberpunk", "neon"]:
        weights = {
            BuildingType.CYBERPUNK_BAR: 0.12,
            BuildingType.NEON_DINER: 0.1,
            BuildingType.ARCADE: 0.1,
            BuildingType.TECH_SHOP: 0.08,
            BuildingType.HOLO_BILLBOARD: 0.06,
            BuildingType.SKYSCRAPER: 0.1,
            BuildingType.APARTMENT: 0.08,
            BuildingType.TOWER: 0.08,
            BuildingType.CAPSULE: 0.05,
            BuildingType.MODERN_HOUSE: 0.05,
            BuildingType.PARKING_GARAGE: 0.04,
            BuildingType.METRO_ENTRANCE: 0.05,
            BuildingType.MODULAR_HUB: 0.04,
            BuildingType.COMM_TOWER: 0.03,
            BuildingType.POWER_NODE: 0.02,
        }
    elif biome in ["ancient", "historical"]:
        weights = {
            BuildingType.NUBIAN_COMPLEX: 0.1,
            BuildingType.PYRAMID_TEMPLE: 0.12,
            BuildingType.ZIGGURAT: 0.1,
            BuildingType.STONE_PORTAL: 0.06,
            BuildingType.OBELISK: 0.08,
            BuildingType.AMPHITHEATER: 0.08,
            BuildingType.COLOSSEUM: 0.06,
            BuildingType.AQUEDUCT: 0.05,
            BuildingType.MAUSOLEUM: 0.08,
            BuildingType.PAGODA_TOWER: 0.06,
            BuildingType.TEMPLE: 0.1,
            BuildingType.RUINS: 0.08,
            BuildingType.MONUMENT: 0.03,
        }
    # === EXOTIC SPECIAL BIOMES ===
    elif biome == "psychedelic":
        weights = {
            BuildingType.RAINBOW_TOWER: 0.15,
            BuildingType.MUSHROOM_PALACE: 0.15,
            BuildingType.SPIRAL_HOUSE: 0.12,
            BuildingType.KALEIDOSCOPE: 0.1,
            BuildingType.BUBBLE_DOME: 0.12,
            BuildingType.FLOATING_ISLAND: 0.1,
            BuildingType.CRYSTAL_CAVE: 0.08,
            BuildingType.GEODOME_HOME: 0.06,
            BuildingType.PLANETARIUM: 0.04,
            BuildingType.PARKOUR: 0.08,  # Trippy parkour!
        }
    elif biome == "hellfire":
        weights = {
            BuildingType.VOLCANO_FORGE: 0.15,
            BuildingType.LAVA_TEMPLE: 0.15,
            BuildingType.DEMON_GATE: 0.1,
            BuildingType.BONE_TOWER: 0.12,
            BuildingType.FLAME_ALTAR: 0.1,
            BuildingType.CHARRED_RUINS: 0.12,
            BuildingType.OBSIDIAN_SPIRE: 0.1,
            BuildingType.FACTORY: 0.08,  # Hell factories
            BuildingType.POWERPLANT: 0.05,
            BuildingType.PARKOUR: 0.03,
        }
    elif biome == "shadow":
        weights = {
            BuildingType.SHADOW_CASTLE: 0.15,
            BuildingType.NIGHTMARE_HOUSE: 0.12,
            BuildingType.VOID_PORTAL: 0.08,
            BuildingType.CRYPT: 0.12,
            BuildingType.HAUNTED_MANOR: 0.15,
            BuildingType.SPIDER_NEST: 0.1,
            BuildingType.DARK_TOWER: 0.12,
            BuildingType.RUINS: 0.1,
            BuildingType.CHURCH: 0.03,  # Dark church
            BuildingType.MAUSOLEUM: 0.03,
        }
    elif biome == "crystal":
        weights = {
            BuildingType.CRYSTAL_PALACE: 0.2,
            BuildingType.ICE_FORTRESS: 0.15,
            BuildingType.PRISM_TOWER: 0.15,
            BuildingType.GEODE_HOUSE: 0.12,
            BuildingType.QUARTZ_TEMPLE: 0.12,
            BuildingType.OBSERVATORY: 0.08,
            BuildingType.GEODOME_HOME: 0.08,
            BuildingType.BUBBLE_DOME: 0.06,
            BuildingType.MONUMENT: 0.04,
        }
    elif biome == "void":
        weights = {
            BuildingType.VOID_MONOLITH: 0.35,
            BuildingType.ENTROPY_GATE: 0.25,
            BuildingType.NULL_ZONE: 0.2,
            BuildingType.VOID_PORTAL: 0.15,
            BuildingType.MONUMENT: 0.05,
        }
    else:  # Default
        weights = {
            BuildingType.HOUSE: 0.12,
            BuildingType.TOWER: 0.06,
            BuildingType.TEMPLE: 0.06,
            BuildingType.OBSERVATORY: 0.06,
            BuildingType.RUINS: 0.06,
            BuildingType.MONUMENT: 0.04,
            BuildingType.APARTMENT: 0.06,
            BuildingType.SHOP: 0.06,
            BuildingType.WAREHOUSE: 0.04,
            BuildingType.PARKOUR: 0.08,
            BuildingType.MODERN_HOUSE: 0.05,
            BuildingType.CABIN: 0.04,
            BuildingType.CHURCH: 0.03,
            BuildingType.SCHOOL: 0.03,
            BuildingType.LIBRARY: 0.03,
            BuildingType.RESTAURANT: 0.04,
            BuildingType.BANK: 0.02,
            BuildingType.POST_OFFICE: 0.02,
            BuildingType.GAS_STATION: 0.03,
            BuildingType.FIRE_STATION: 0.02,
            BuildingType.STONE_PORTAL: 0.02,
            BuildingType.GAZEBO: 0.03,
        }
    
    types = list(weights.keys())
    probs = np.array(list(weights.values()))
    probs /= probs.sum()  # Normalize
    
    return rng.choice(types, p=probs)

