"""
DNA System for Procedural Generation.

This module provides a genetic system for plants, creatures, and other
procedurally generated entities. DNA can be:
- Randomly generated
- Mutated (small random changes)
- Crossed over (combined from two parents)
- Serialized to/from JSON

The DNA uses a gene-based system where each gene controls a specific trait.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import json
import copy


# Type aliases for clarity
Color = Tuple[float, float, float]
Range = Tuple[float, float]


@dataclass
class Gene:
    """A single gene with a value, valid range, and mutation rate."""
    value: float
    min_val: float = 0.0
    max_val: float = 1.0
    mutation_rate: float = 0.1  # How much this gene can mutate (0-1)
    
    def mutate(self, rng: np.random.Generator, strength: float = 1.0) -> "Gene":
        """Return a mutated copy of this gene."""
        mutation = rng.normal(0, self.mutation_rate * strength)
        new_value = np.clip(self.value + mutation, self.min_val, self.max_val)
        return Gene(new_value, self.min_val, self.max_val, self.mutation_rate)
    
    def crossover(self, other: "Gene", rng: np.random.Generator) -> "Gene":
        """Combine with another gene (blend or pick one)."""
        if rng.random() < 0.5:
            # Blend
            blend = rng.random()
            new_value = self.value * blend + other.value * (1 - blend)
        else:
            # Pick one
            new_value = self.value if rng.random() < 0.5 else other.value
        return Gene(new_value, self.min_val, self.max_val, self.mutation_rate)


@dataclass
class ColorGene:
    """A gene that represents an RGB color."""
    r: float = 0.5
    g: float = 0.5
    b: float = 0.5
    mutation_rate: float = 0.15
    
    @property
    def rgb(self) -> Color:
        return (self.r, self.g, self.b)
    
    def mutate(self, rng: np.random.Generator, strength: float = 1.0) -> "ColorGene":
        """Return a mutated copy with potential for larger color shifts."""
        def mutate_channel(val):
            # Regular small mutation
            mutation = rng.normal(0, self.mutation_rate * strength)
            # Rare larger mutation for color variety
            if rng.random() < 0.08:  # 8% chance of bigger shift
                mutation += rng.normal(0, 0.2 * strength)
            return float(np.clip(val + mutation, 0, 1))
        
        return ColorGene(
            mutate_channel(self.r),
            mutate_channel(self.g),
            mutate_channel(self.b),
            self.mutation_rate
        )
    
    def crossover(self, other: "ColorGene", rng: np.random.Generator) -> "ColorGene":
        """Combine with another color gene."""
        blend = rng.random()
        return ColorGene(
            self.r * blend + other.r * (1 - blend),
            self.g * blend + other.g * (1 - blend),
            self.b * blend + other.b * (1 - blend),
            self.mutation_rate
        )
    
    @classmethod
    def from_hsv(cls, h: float, s: float, v: float) -> "ColorGene":
        """Create from HSV values (0-1 range)."""
        # HSV to RGB conversion
        h = h * 6.0
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        
        if i == 0: r, g, b = v, t, p
        elif i == 1: r, g, b = q, v, p
        elif i == 2: r, g, b = p, v, t
        elif i == 3: r, g, b = p, q, v
        elif i == 4: r, g, b = t, p, v
        else: r, g, b = v, p, q
        
        return cls(r, g, b)


@dataclass 
class SegmentGene:
    """A gene describing a plant segment (trunk section, branch, etc.)."""
    length: float = 1.0       # Segment length (0.1 - 5.0)
    width: float = 0.2        # Segment width (0.05 - 1.0)
    taper: float = 0.8        # Width reduction toward end (0.3 - 1.0)
    curve: float = 0.0        # Curvature (-1 to 1)
    twist: float = 0.0        # Twist/spiral (0 to 1)
    
    def mutate(self, rng: np.random.Generator, strength: float = 1.0) -> "SegmentGene":
        """Return mutated copy."""
        rate = 0.1 * strength
        return SegmentGene(
            float(np.clip(self.length + rng.normal(0, rate), 0.1, 5.0)),
            float(np.clip(self.width + rng.normal(0, rate * 0.5), 0.05, 1.0)),
            float(np.clip(self.taper + rng.normal(0, rate * 0.3), 0.3, 1.0)),
            float(np.clip(self.curve + rng.normal(0, rate * 0.5), -1, 1)),
            float(np.clip(self.twist + rng.normal(0, rate * 0.3), 0, 1)),
        )
    
    def crossover(self, other: "SegmentGene", rng: np.random.Generator) -> "SegmentGene":
        """Combine with another segment gene."""
        blend = rng.random()
        return SegmentGene(
            self.length * blend + other.length * (1 - blend),
            self.width * blend + other.width * (1 - blend),
            self.taper * blend + other.taper * (1 - blend),
            self.curve * blend + other.curve * (1 - blend),
            self.twist * blend + other.twist * (1 - blend),
        )


class PlantType:
    """Plant type categories."""
    GRASS = "grass"
    FLOWER = "flower"
    FERN = "fern"
    BUSH = "bush"
    SHRUB = "shrub"
    TREE = "tree"
    TALL_TREE = "tall_tree"
    PINE = "pine"
    PALM = "palm"
    WILLOW = "willow"
    CACTUS = "cactus"
    MUSHROOM = "mushroom"
    CORAL = "coral"
    CRYSTAL = "crystal"
    ALIEN = "alien"
    # New exotic types
    VINE = "vine"                 # Climbing/trailing vines
    SPINY_VINE = "spiny_vine"     # Thorny vines
    OCTOPUS = "octopus"           # Recursive branching like octopus
    TENTACLE = "tentacle"         # Upside-down dangling growths
    SPIRAL = "spiral"             # Spiraling growth patterns
    SEAWEED = "seaweed"           # Underwater swaying plants
    # Sprawling ground-cover plants
    GROUNDCOVER = "groundcover"   # Low spreading mats
    CREEPER = "creeper"           # Spreading vines along ground
    LICHEN = "lichen"             # Crusty spreading growth
    MOSS_PAD = "moss_pad"         # Thick moss cushions
    LILY_PAD = "lily_pad"         # Floating water plants
    
    ALL_TYPES = [GRASS, FLOWER, FERN, BUSH, SHRUB, TREE, TALL_TREE, PINE, PALM, WILLOW, 
                 CACTUS, MUSHROOM, CORAL, CRYSTAL, ALIEN, VINE, SPINY_VINE, OCTOPUS, 
                 TENTACLE, SPIRAL, SEAWEED, GROUNDCOVER, CREEPER, LICHEN, MOSS_PAD, LILY_PAD]
    
    # Underwater-specific types
    UNDERWATER_TYPES = [SEAWEED, CORAL]
    
    # Sprawling ground types
    GROUND_TYPES = [GROUNDCOVER, CREEPER, LICHEN, MOSS_PAD]


@dataclass
class PlantDNA:
    """
    Complete DNA for a plant, controlling all aspects of growth and appearance.
    
    The DNA is hierarchical:
    - Base traits (type, overall size)
    - Trunk/stem segments (multi-jointed growth)
    - Branch patterns
    - Leaf/canopy traits
    - Special features (flowers, fruits, glow)
    - Colors
    """
    # Identity
    plant_type: str = PlantType.TREE
    species_id: int = 0  # For tracking lineage
    generation: int = 0  # Mutation generation
    
    # Base traits
    height_gene: Gene = field(default_factory=lambda: Gene(5.0, 0.2, 25.0, 0.2))
    width_gene: Gene = field(default_factory=lambda: Gene(0.3, 0.05, 1.5, 0.15))
    
    # Multi-segment trunk (allows for jointed/curved growth)
    trunk_segments: List[SegmentGene] = field(default_factory=lambda: [
        SegmentGene(1.0, 0.3, 0.85, 0.0, 0.0)
    ])
    
    # Branching
    branch_count: int = 4
    branch_angle: float = 0.4      # 0 = up, 1 = horizontal
    branch_spread: float = 1.0     # How evenly spread around trunk
    branch_height: float = 0.6     # Where branches start (0-1 of height)
    branch_segments: List[SegmentGene] = field(default_factory=lambda: [
        SegmentGene(0.5, 0.1, 0.7, 0.1, 0.0)
    ])
    sub_branch_chance: float = 0.3  # Chance of sub-branches
    
    # Leaves/Canopy
    leaf_density: float = 0.7      # 0 = sparse, 1 = dense
    leaf_size: float = 0.5         # Relative leaf size
    leaf_shape: str = "round"      # round, pointed, frond, needle, blade
    canopy_shape: str = "dome"     # dome, cone, umbrella, weeping, columnar
    canopy_spread: float = 0.5     # How wide the canopy spreads
    
    # Special features
    has_flowers: bool = False
    flower_size: float = 0.0
    has_fruit: bool = False
    fruit_size: float = 0.0
    has_glow: bool = False
    glow_intensity: float = 0.0
    has_thorns: bool = False
    
    # Colors
    trunk_color: ColorGene = field(default_factory=lambda: ColorGene(0.35, 0.25, 0.15))
    leaf_color: ColorGene = field(default_factory=lambda: ColorGene(0.2, 0.55, 0.2))
    flower_color: ColorGene = field(default_factory=lambda: ColorGene(0.9, 0.3, 0.5))
    glow_color: ColorGene = field(default_factory=lambda: ColorGene(0.5, 0.8, 0.5))
    
    # Variation genes (for procedural detail)
    asymmetry: float = 0.1         # How asymmetric the growth is
    droop: float = 0.0             # How much branches droop
    wind_sway: float = 0.3         # Animation responsiveness (future)
    
    # Advanced growth patterns
    recursive_depth: int = 1       # Levels of recursive branching (1-4)
    growth_direction: float = 0.0  # -1 = droop down, 0 = up, 1 = spread horizontal
    spiral_factor: float = 0.0     # Amount of spiral twist in growth
    bulb_count: int = 0            # Number of bulbous growths
    bulb_size: float = 0.0         # Size of bulbous growths
    
    # Surface details
    bark_texture: str = "smooth"   # smooth, rough, scaly, peeling, ridged
    surface_bumps: float = 0.0     # 0 = smooth, 1 = very bumpy
    has_moss: bool = False         # Moss/lichen growth
    moss_density: float = 0.0      # How much moss coverage
    
    # Additional colors
    secondary_trunk_color: ColorGene = field(default_factory=lambda: ColorGene(0.3, 0.22, 0.12))
    tip_color: ColorGene = field(default_factory=lambda: ColorGene(0.4, 0.6, 0.3))  # Tips of branches/leaves
    fruit_color: ColorGene = field(default_factory=lambda: ColorGene(0.8, 0.2, 0.2))
    moss_color: ColorGene = field(default_factory=lambda: ColorGene(0.2, 0.4, 0.15))
    
    # Special features
    bioluminescent: bool = False   # Different from glow - more subtle
    crystal_growth: bool = False   # Crystal formations on plant
    spore_pods: bool = False       # Has spore pods
    tendrils: int = 0              # Number of tendrils/vines
    root_exposure: float = 0.0     # Visible above-ground roots (0-1)
    
    # Orientation
    upside_down: bool = False      # Plant grows downward (hanging)
    lean_angle: float = 0.0        # How much the plant leans (-1 to 1)
    
    def mutate(self, rng: np.random.Generator = None, strength: float = 0.5) -> "PlantDNA":
        """
        Create a mutated copy of this DNA.
        
        Args:
            rng: Random generator (creates one if None)
            strength: Mutation strength (0 = no change, 1 = strong mutation)
        
        Returns:
            New PlantDNA with mutations applied
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Deep copy first
        new_dna = copy.deepcopy(self)
        new_dna.generation += 1
        
        # Mutate genes
        new_dna.height_gene = self.height_gene.mutate(rng, strength)
        new_dna.width_gene = self.width_gene.mutate(rng, strength)
        
        # Mutate segments
        new_dna.trunk_segments = [s.mutate(rng, strength) for s in self.trunk_segments]
        new_dna.branch_segments = [s.mutate(rng, strength) for s in self.branch_segments]
        
        # Maybe add/remove a trunk segment
        if rng.random() < 0.1 * strength:
            if len(new_dna.trunk_segments) < 5 and rng.random() < 0.5:
                # Add segment
                new_dna.trunk_segments.append(SegmentGene(
                    0.5 + rng.random(), 0.15, 0.8, rng.random() * 0.3 - 0.15, 0
                ))
            elif len(new_dna.trunk_segments) > 1:
                # Remove segment
                new_dna.trunk_segments.pop()
        
        # Mutate scalar values
        rate = 0.1 * strength
        new_dna.branch_count = max(0, min(12, self.branch_count + int(rng.normal(0, 1))))
        new_dna.branch_angle = float(np.clip(self.branch_angle + rng.normal(0, rate), 0, 1))
        new_dna.branch_spread = float(np.clip(self.branch_spread + rng.normal(0, rate), 0.2, 1.5))
        new_dna.branch_height = float(np.clip(self.branch_height + rng.normal(0, rate), 0.2, 0.9))
        new_dna.sub_branch_chance = float(np.clip(self.sub_branch_chance + rng.normal(0, rate), 0, 0.8))
        
        new_dna.leaf_density = float(np.clip(self.leaf_density + rng.normal(0, rate), 0.1, 1))
        new_dna.leaf_size = float(np.clip(self.leaf_size + rng.normal(0, rate), 0.1, 1.5))
        new_dna.canopy_spread = float(np.clip(self.canopy_spread + rng.normal(0, rate), 0.2, 1.5))
        
        new_dna.asymmetry = float(np.clip(self.asymmetry + rng.normal(0, rate * 0.5), 0, 0.5))
        new_dna.droop = float(np.clip(self.droop + rng.normal(0, rate), 0, 1))
        
        # Mutate colors
        new_dna.trunk_color = self.trunk_color.mutate(rng, strength)
        new_dna.leaf_color = self.leaf_color.mutate(rng, strength)
        new_dna.flower_color = self.flower_color.mutate(rng, strength)
        new_dna.glow_color = self.glow_color.mutate(rng, strength)
        
        # Small chance to gain/lose features
        if rng.random() < 0.05 * strength:
            new_dna.has_flowers = not self.has_flowers
            if new_dna.has_flowers:
                new_dna.flower_size = 0.1 + rng.random() * 0.4
        if rng.random() < 0.03 * strength:
            new_dna.has_glow = not self.has_glow
            if new_dna.has_glow:
                new_dna.glow_intensity = 0.3 + rng.random() * 0.5
        
        # Mutate new advanced parameters
        new_dna.recursive_depth = max(1, min(4, self.recursive_depth + int(rng.normal(0, 0.5) * strength)))
        new_dna.growth_direction = float(np.clip(self.growth_direction + rng.normal(0, rate), -1, 1))
        new_dna.spiral_factor = float(np.clip(self.spiral_factor + rng.normal(0, rate), 0, 1))
        new_dna.bulb_count = max(0, min(8, self.bulb_count + int(rng.normal(0, 0.5) * strength)))
        new_dna.bulb_size = float(np.clip(self.bulb_size + rng.normal(0, rate), 0, 1))
        
        new_dna.surface_bumps = float(np.clip(self.surface_bumps + rng.normal(0, rate), 0, 1))
        new_dna.moss_density = float(np.clip(self.moss_density + rng.normal(0, rate), 0, 1))
        new_dna.root_exposure = float(np.clip(self.root_exposure + rng.normal(0, rate), 0, 1))
        new_dna.lean_angle = float(np.clip(self.lean_angle + rng.normal(0, rate), -1, 1))
        new_dna.tendrils = max(0, min(6, self.tendrils + int(rng.normal(0, 0.3) * strength)))
        
        # Mutate additional colors
        new_dna.secondary_trunk_color = self.secondary_trunk_color.mutate(rng, strength)
        new_dna.tip_color = self.tip_color.mutate(rng, strength)
        new_dna.fruit_color = self.fruit_color.mutate(rng, strength)
        new_dna.moss_color = self.moss_color.mutate(rng, strength)
        
        # Small chance to gain/lose new features
        if rng.random() < 0.02 * strength:
            new_dna.has_moss = not self.has_moss
        if rng.random() < 0.02 * strength:
            new_dna.bioluminescent = not self.bioluminescent
        if rng.random() < 0.02 * strength:
            new_dna.crystal_growth = not self.crystal_growth
        if rng.random() < 0.02 * strength:
            new_dna.spore_pods = not self.spore_pods
        if rng.random() < 0.01 * strength:
            new_dna.upside_down = not self.upside_down
        
        # Bark texture can mutate
        if rng.random() < 0.03 * strength:
            new_dna.bark_texture = rng.choice(["smooth", "rough", "scaly", "peeling", "ridged"])
        
        return new_dna
    
    def crossover(self, other: "PlantDNA", rng: np.random.Generator = None) -> "PlantDNA":
        """
        Create offspring DNA by combining two parent DNAs.
        
        Args:
            other: The other parent DNA
            rng: Random generator
        
        Returns:
            New PlantDNA combining traits from both parents
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Start with copy of self
        child = copy.deepcopy(self)
        child.generation = max(self.generation, other.generation) + 1
        
        # Crossover genes
        child.height_gene = self.height_gene.crossover(other.height_gene, rng)
        child.width_gene = self.width_gene.crossover(other.width_gene, rng)
        
        # For segments, blend or pick
        if rng.random() < 0.5:
            child.trunk_segments = [
                s1.crossover(s2, rng) 
                for s1, s2 in zip(self.trunk_segments, other.trunk_segments)
            ] if len(self.trunk_segments) == len(other.trunk_segments) else (
                self.trunk_segments if rng.random() < 0.5 else other.trunk_segments
            )
        else:
            child.trunk_segments = other.trunk_segments
        
        # Crossover scalars (pick or blend)
        def cross_scalar(a, b):
            if rng.random() < 0.5:
                return a * rng.random() + b * (1 - rng.random())
            return a if rng.random() < 0.5 else b
        
        def cross_int(a, b):
            return a if rng.random() < 0.5 else b
        
        child.branch_count = cross_int(self.branch_count, other.branch_count)
        child.branch_angle = cross_scalar(self.branch_angle, other.branch_angle)
        child.branch_spread = cross_scalar(self.branch_spread, other.branch_spread)
        child.branch_height = cross_scalar(self.branch_height, other.branch_height)
        child.leaf_density = cross_scalar(self.leaf_density, other.leaf_density)
        child.leaf_size = cross_scalar(self.leaf_size, other.leaf_size)
        child.canopy_spread = cross_scalar(self.canopy_spread, other.canopy_spread)
        child.asymmetry = cross_scalar(self.asymmetry, other.asymmetry)
        child.droop = cross_scalar(self.droop, other.droop)
        
        # Crossover shapes (pick one)
        child.leaf_shape = self.leaf_shape if rng.random() < 0.5 else other.leaf_shape
        child.canopy_shape = self.canopy_shape if rng.random() < 0.5 else other.canopy_shape
        
        # Crossover colors
        child.trunk_color = self.trunk_color.crossover(other.trunk_color, rng)
        child.leaf_color = self.leaf_color.crossover(other.leaf_color, rng)
        child.flower_color = self.flower_color.crossover(other.flower_color, rng)
        
        # Inherit features from either parent
        child.has_flowers = self.has_flowers or other.has_flowers if rng.random() < 0.5 else (
            self.has_flowers and other.has_flowers)
        child.has_glow = self.has_glow if rng.random() < 0.5 else other.has_glow
        
        return child
    
    @property
    def total_height(self) -> float:
        """Calculate total plant height from segments."""
        return sum(s.length for s in self.trunk_segments) * self.height_gene.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize DNA to dictionary for JSON storage."""
        return {
            "plant_type": self.plant_type,
            "species_id": int(self.species_id) if self.species_id else 0,
            "generation": self.generation,
            "height": float(self.height_gene.value),
            "width": float(self.width_gene.value),
            "trunk_segments": [
                {"length": float(s.length), "width": float(s.width), "taper": float(s.taper), 
                 "curve": float(s.curve), "twist": float(s.twist)}
                for s in self.trunk_segments
            ],
            "branch_count": int(self.branch_count),
            "branch_angle": float(self.branch_angle),
            "leaf_density": float(self.leaf_density),
            "trunk_color": [float(c) for c in self.trunk_color.rgb],
            "leaf_color": [float(c) for c in self.leaf_color.rgb],
            "has_flowers": bool(self.has_flowers),
            "has_glow": bool(self.has_glow),
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def create_random(cls, plant_type: str = None, seed: int = None) -> "PlantDNA":
        """
        Create a random plant DNA of a given type.
        
        Args:
            plant_type: Type of plant (uses random if None)
            seed: Random seed for reproducibility
        
        Returns:
            New random PlantDNA
        """
        rng = np.random.default_rng(seed)
        
        if plant_type is None:
            # Weighted random type
            weights = [0.25, 0.12, 0.12, 0.08, 0.18, 0.08, 0.05, 0.04, 0.03, 0.05]
            plant_type = rng.choice(PlantType.ALL_TYPES, p=weights)
        
        dna = cls(plant_type=plant_type, species_id=seed or rng.integers(0, 100000))
        
        # Configure based on type
        if plant_type == PlantType.GRASS:
            dna.height_gene = Gene(0.2 + rng.random() * 0.5, 0.1, 1.0, 0.1)
            dna.width_gene = Gene(0.02, 0.01, 0.05, 0.05)
            dna.trunk_segments = []
            dna.branch_count = 5 + rng.integers(0, 10)
            dna.leaf_shape = "blade"
            dna.canopy_shape = "columnar"
            dna.leaf_color = ColorGene(0.2 + rng.random() * 0.1, 0.4 + rng.random() * 0.3, 0.15)
            
        elif plant_type == PlantType.FERN:
            dna.height_gene = Gene(0.6 + rng.random() * 1.5, 0.3, 2.5, 0.15)
            dna.width_gene = Gene(0.05, 0.02, 0.1, 0.05)
            dna.trunk_segments = [SegmentGene(0.1, 0.05, 0.9, 0.1, 0)]
            dna.branch_count = 4 + rng.integers(0, 8)
            dna.branch_angle = 0.6 + rng.random() * 0.3
            dna.leaf_shape = "frond"
            dna.droop = 0.3 + rng.random() * 0.4
            dna.leaf_color = ColorGene(0.15, 0.45 + rng.random() * 0.25, 0.2)
            
        elif plant_type in (PlantType.BUSH, PlantType.SHRUB):
            dna.height_gene = Gene(0.8 + rng.random() * 2.0, 0.5, 3.5, 0.2)
            dna.width_gene = Gene(0.1 + rng.random() * 0.15, 0.05, 0.3, 0.1)
            dna.trunk_segments = [
                SegmentGene(0.3, 0.15, 0.8, rng.random() * 0.2 - 0.1, 0)
                for _ in range(1 + rng.integers(0, 2))
            ]
            dna.branch_count = 5 + rng.integers(0, 8)
            dna.branch_angle = 0.4 + rng.random() * 0.4
            dna.branch_height = 0.1 + rng.random() * 0.3
            dna.canopy_shape = "dome"
            dna.leaf_density = 0.7 + rng.random() * 0.3
            dna.leaf_color = ColorGene(0.2 + rng.random() * 0.1, 0.4 + rng.random() * 0.3, 0.15)
            dna.has_flowers = rng.random() < 0.3
            if dna.has_flowers:
                dna.flower_size = 0.1 + rng.random() * 0.2
                dna.flower_color = ColorGene.from_hsv(rng.random(), 0.6 + rng.random() * 0.4, 0.8)
            
        elif plant_type == PlantType.TREE:
            dna.height_gene = Gene(4 + rng.random() * 8, 2, 15, 0.25)
            dna.width_gene = Gene(0.2 + rng.random() * 0.3, 0.1, 0.6, 0.15)
            dna.trunk_segments = [
                SegmentGene(
                    0.6 + rng.random() * 0.6,
                    0.3 - i * 0.08,
                    0.85,
                    rng.random() * 0.15 - 0.075,
                    0
                )
                for i in range(2 + rng.integers(0, 2))
            ]
            dna.branch_count = 5 + rng.integers(0, 7)
            dna.branch_angle = 0.25 + rng.random() * 0.4
            dna.branch_height = 0.4 + rng.random() * 0.35
            dna.sub_branch_chance = 0.4 + rng.random() * 0.4  # 40-80% chance
            dna.recursive_depth = 2 + rng.integers(0, 2)  # 2-3 levels of branching
            dna.canopy_shape = rng.choice(["dome", "cone", "umbrella", "weeping"])
            dna.canopy_spread = 0.4 + rng.random() * 0.5
            dna.droop = rng.random() * 0.4  # Some droop variation
            dna.asymmetry = 0.1 + rng.random() * 0.3  # Natural asymmetry
            dna.trunk_color = ColorGene(0.25 + rng.random() * 0.2, 0.15 + rng.random() * 0.12, 0.08 + rng.random() * 0.05)
            dna.leaf_color = ColorGene(0.1 + rng.random() * 0.15, 0.3 + rng.random() * 0.4, 0.08 + rng.random() * 0.1)
            
        elif plant_type == PlantType.TALL_TREE:
            dna.height_gene = Gene(12 + rng.random() * 12, 8, 30, 0.3)
            dna.width_gene = Gene(0.4 + rng.random() * 0.5, 0.2, 1.0, 0.2)
            dna.trunk_segments = [
                SegmentGene(
                    0.8 + rng.random() * 0.5,
                    0.5 - i * 0.1,
                    0.9,
                    rng.random() * 0.1 - 0.05,
                    0
                )
                for i in range(3 + rng.integers(0, 3))
            ]
            dna.branch_count = 6 + rng.integers(0, 8)
            dna.branch_angle = 0.2 + rng.random() * 0.35
            dna.branch_height = 0.5 + rng.random() * 0.25
            dna.sub_branch_chance = 0.5 + rng.random() * 0.35  # 50-85% chance
            dna.recursive_depth = 2 + rng.integers(0, 3)  # 2-4 levels of branching
            dna.canopy_shape = rng.choice(["dome", "cone", "weeping"])
            dna.canopy_spread = 0.5 + rng.random() * 0.6
            dna.droop = rng.random() * 0.35
            dna.asymmetry = 0.1 + rng.random() * 0.25
            dna.trunk_color = ColorGene(0.2 + rng.random() * 0.18, 0.12 + rng.random() * 0.1, 0.06 + rng.random() * 0.05)
            dna.leaf_color = ColorGene(0.08 + rng.random() * 0.12, 0.25 + rng.random() * 0.35, 0.05 + rng.random() * 0.1)
            
        elif plant_type == PlantType.PALM:
            dna.height_gene = Gene(6 + rng.random() * 10, 4, 18, 0.25)
            dna.width_gene = Gene(0.25 + rng.random() * 0.2, 0.15, 0.5, 0.1)
            dna.trunk_segments = [
                SegmentGene(1.2 + rng.random() * 0.5, 0.3, 0.95, rng.random() * 0.3 - 0.15, 0)
                for _ in range(2 + rng.integers(0, 2))
            ]
            dna.branch_count = 6 + rng.integers(0, 8)
            dna.branch_angle = 0.5 + rng.random() * 0.4
            dna.branch_height = 0.95  # Branches only at top
            dna.leaf_shape = "frond"
            dna.canopy_shape = "umbrella"
            dna.droop = 0.4 + rng.random() * 0.3
            dna.trunk_color = ColorGene(0.4, 0.35, 0.25)
            dna.leaf_color = ColorGene(0.2, 0.5 + rng.random() * 0.2, 0.15)
            
        elif plant_type == PlantType.CACTUS:
            dna.height_gene = Gene(1 + rng.random() * 4, 0.5, 6, 0.2)
            dna.width_gene = Gene(0.3 + rng.random() * 0.4, 0.2, 0.8, 0.15)
            dna.trunk_segments = [
                SegmentGene(1.0 + rng.random() * 0.5, 0.5, 0.95, 0, 0)
            ]
            dna.branch_count = rng.integers(0, 4)
            dna.branch_angle = 0.1 + rng.random() * 0.2
            dna.leaf_density = 0  # No leaves
            dna.has_thorns = True
            dna.trunk_color = ColorGene(0.2, 0.45 + rng.random() * 0.2, 0.15)
            dna.has_flowers = rng.random() < 0.3
            if dna.has_flowers:
                dna.flower_color = ColorGene.from_hsv(rng.random(), 0.8, 0.9)
                dna.flower_size = 0.15 + rng.random() * 0.2
            
        elif plant_type == PlantType.MUSHROOM:
            dna.height_gene = Gene(0.3 + rng.random() * 1.5, 0.1, 2.5, 0.2)
            dna.width_gene = Gene(0.1 + rng.random() * 0.2, 0.05, 0.4, 0.1)
            dna.trunk_segments = [SegmentGene(0.8, 0.2, 1.1, 0, 0)]  # Taper outward
            dna.branch_count = 0
            dna.canopy_shape = "dome"
            dna.canopy_spread = 0.8 + rng.random() * 0.6
            dna.trunk_color = ColorGene(0.85, 0.8, 0.7)
            dna.leaf_color = ColorGene.from_hsv(rng.random(), 0.5 + rng.random() * 0.4, 0.6 + rng.random() * 0.3)
            dna.has_glow = rng.random() < 0.4
            if dna.has_glow:
                dna.glow_intensity = 0.3 + rng.random() * 0.5
                dna.glow_color = ColorGene(
                    dna.leaf_color.r * 0.5 + 0.5,
                    dna.leaf_color.g * 0.5 + 0.5,
                    dna.leaf_color.b * 0.5 + 0.5
                )
            
        elif plant_type == PlantType.FLOWER:
            # Colorful flowers
            dna.height_gene = Gene(0.3 + rng.random() * 0.6, 0.1, 1.2, 0.15)
            dna.width_gene = Gene(0.02, 0.01, 0.05, 0.05)
            dna.trunk_segments = [SegmentGene(0.8, 0.03, 0.9, rng.random() * 0.3, 0)]
            dna.branch_count = 0
            dna.has_flowers = True
            dna.flower_size = 0.15 + rng.random() * 0.25
            dna.flower_color = ColorGene.from_hsv(rng.random(), 0.7 + rng.random() * 0.3, 0.8 + rng.random() * 0.2)
            dna.trunk_color = ColorGene(0.2, 0.4 + rng.random() * 0.2, 0.15)
            dna.leaf_color = ColorGene(0.2, 0.5, 0.2)
            dna.canopy_shape = "dome"
            
        elif plant_type == PlantType.PINE:
            # Coniferous pine tree
            dna.height_gene = Gene(6 + rng.random() * 12, 4, 20, 0.25)
            dna.width_gene = Gene(0.3 + rng.random() * 0.3, 0.15, 0.7, 0.15)
            dna.trunk_segments = [
                SegmentGene(1.0, 0.35, 0.92, 0, 0),
                SegmentGene(0.8, 0.25, 0.9, 0, 0),
            ]
            dna.branch_count = 6 + rng.integers(0, 5)
            dna.branch_angle = 0.35 + rng.random() * 0.2
            dna.branch_height = 0.3 + rng.random() * 0.2
            dna.canopy_shape = "cone"
            dna.canopy_spread = 0.35 + rng.random() * 0.2
            dna.trunk_color = ColorGene(0.35, 0.22, 0.12)
            dna.leaf_color = ColorGene(0.08 + rng.random() * 0.08, 0.25 + rng.random() * 0.15, 0.08)
            
        elif plant_type == PlantType.WILLOW:
            # Weeping willow style
            dna.height_gene = Gene(8 + rng.random() * 8, 5, 18, 0.25)
            dna.width_gene = Gene(0.4 + rng.random() * 0.3, 0.2, 0.8, 0.15)
            dna.trunk_segments = [
                SegmentGene(0.7, 0.4, 0.85, rng.random() * 0.2 - 0.1, 0),
                SegmentGene(0.5, 0.3, 0.85, rng.random() * 0.15, 0),
            ]
            dna.branch_count = 8 + rng.integers(0, 6)
            dna.branch_angle = 0.5 + rng.random() * 0.3
            dna.droop = 0.6 + rng.random() * 0.3
            dna.canopy_shape = "weeping"
            dna.canopy_spread = 0.6 + rng.random() * 0.3
            dna.trunk_color = ColorGene(0.3, 0.25, 0.15)
            dna.leaf_color = ColorGene(0.25, 0.5 + rng.random() * 0.2, 0.2)
            
        elif plant_type == PlantType.CORAL:
            # Coral-like branching structure
            dna.height_gene = Gene(0.5 + rng.random() * 1.5, 0.3, 2.5, 0.2)
            dna.width_gene = Gene(0.08 + rng.random() * 0.15, 0.05, 0.3, 0.1)
            dna.trunk_segments = [SegmentGene(0.4, 0.15, 0.7, rng.random() * 0.4, 0)]
            dna.branch_count = 5 + rng.integers(0, 8)
            dna.branch_angle = 0.3 + rng.random() * 0.4
            dna.sub_branch_chance = 0.6 + rng.random() * 0.3
            dna.leaf_density = 0
            dna.trunk_color = ColorGene.from_hsv(rng.random(), 0.5 + rng.random() * 0.4, 0.6 + rng.random() * 0.3)
            dna.leaf_color = dna.trunk_color  # Same color throughout
            
        elif plant_type == PlantType.CRYSTAL:
            # Crystalline/geometric growth
            dna.height_gene = Gene(0.8 + rng.random() * 2.5, 0.4, 4.0, 0.25)
            dna.width_gene = Gene(0.15 + rng.random() * 0.25, 0.08, 0.5, 0.15)
            dna.trunk_segments = [SegmentGene(1.0, 0.25, 0.6, 0, 0)]  # Sharp taper
            dna.branch_count = 3 + rng.integers(0, 5)
            dna.branch_angle = 0.2 + rng.random() * 0.3
            dna.asymmetry = 0.3 + rng.random() * 0.2
            dna.leaf_density = 0
            hue = rng.choice([0.5, 0.55, 0.75, 0.85, 0.95])  # Blues, purples, pinks
            dna.trunk_color = ColorGene.from_hsv(hue, 0.3 + rng.random() * 0.3, 0.7 + rng.random() * 0.3)
            dna.leaf_color = dna.trunk_color
            dna.has_glow = rng.random() < 0.5
            if dna.has_glow:
                dna.glow_intensity = 0.4 + rng.random() * 0.4
                dna.glow_color = ColorGene(
                    min(1, dna.trunk_color.r + 0.3),
                    min(1, dna.trunk_color.g + 0.3),
                    min(1, dna.trunk_color.b + 0.3)
                )
            
        elif plant_type == PlantType.ALIEN:
            # Truly random/weird with high potential for complex branching
            dna.height_gene = Gene(1 + rng.random() * 15, 0.5, 20, 0.35)
            dna.width_gene = Gene(0.1 + rng.random() * 0.5, 0.05, 1.0, 0.25)
            dna.trunk_segments = [
                SegmentGene(
                    0.3 + rng.random() * 1.0,
                    0.1 + rng.random() * 0.4,
                    0.5 + rng.random() * 0.5,
                    rng.random() * 0.8 - 0.4,
                    rng.random() * 0.5
                )
                for _ in range(1 + rng.integers(0, 5))
            ]
            dna.branch_count = 3 + rng.integers(0, 12)
            dna.branch_angle = rng.random()
            dna.sub_branch_chance = 0.3 + rng.random() * 0.6  # High variance
            dna.recursive_depth = 1 + rng.integers(0, 4)  # 1-4 levels - can be very complex
            dna.droop = rng.random() * 0.8 - 0.3  # Can droop or reach up
            dna.leaf_shape = rng.choice(["round", "pointed", "frond", "needle", "blade"])
            dna.canopy_shape = rng.choice(["dome", "cone", "umbrella", "weeping", "columnar"])
            dna.asymmetry = 0.2 + rng.random() * 0.4
            dna.spiral_factor = rng.random() * 0.5
            
            # Alien colors
            dna.trunk_color = ColorGene.from_hsv(rng.random(), 0.3 + rng.random() * 0.5, 0.3 + rng.random() * 0.4)
            dna.leaf_color = ColorGene.from_hsv(rng.random(), 0.5 + rng.random() * 0.5, 0.5 + rng.random() * 0.5)
            
            # High chance of special features
            dna.has_flowers = rng.random() < 0.5
            dna.has_glow = rng.random() < 0.4
            dna.has_fruit = rng.random() < 0.3
            
            if dna.has_flowers:
                dna.flower_color = ColorGene.from_hsv(rng.random(), 0.7, 0.9)
                dna.flower_size = 0.1 + rng.random() * 0.4
            if dna.has_glow:
                dna.glow_intensity = 0.4 + rng.random() * 0.6
                dna.glow_color = ColorGene(
                    0.5 + rng.random() * 0.5,
                    0.5 + rng.random() * 0.5,
                    0.5 + rng.random() * 0.5
                )
            if dna.has_fruit:
                dna.fruit_size = 0.1 + rng.random() * 0.3
        
        elif plant_type == PlantType.VINE:
            # Climbing/trailing vines with many segments
            dna.height_gene = Gene(3 + rng.random() * 8, 0.3, 15, 0.25)
            dna.width_gene = Gene(0.05 + rng.random() * 0.1, 0.02, 0.2, 0.2)
            # Many curved segments for vine growth
            num_segments = 5 + rng.integers(0, 8)
            dna.trunk_segments = [
                SegmentGene(
                    0.5 + rng.random() * 0.8,  # Length
                    0.8 + rng.random() * 0.2,  # Width scale (thin)
                    0.95,  # Minor taper
                    (rng.random() - 0.5) * 1.0,  # Curve
                    rng.random() * 0.4  # Twist
                )
                for _ in range(num_segments)
            ]
            dna.branch_count = 2 + rng.integers(0, 5)
            dna.branch_angle = 0.3 + rng.random() * 0.5
            dna.leaf_shape = "round"
            dna.leaf_density = 0.4 + rng.random() * 0.4
            dna.trunk_color = ColorGene.from_hsv(0.25 + rng.random() * 0.15, 0.4 + rng.random() * 0.3, 0.3 + rng.random() * 0.3)
            dna.leaf_color = ColorGene.from_hsv(0.3 + rng.random() * 0.1, 0.5 + rng.random() * 0.3, 0.4 + rng.random() * 0.3)
            
        elif plant_type == PlantType.SPINY_VINE:
            # Thorny vines - similar to vine but with spine features
            dna.height_gene = Gene(2 + rng.random() * 6, 0.3, 12, 0.25)
            dna.width_gene = Gene(0.08 + rng.random() * 0.12, 0.03, 0.25, 0.2)
            num_segments = 4 + rng.integers(0, 6)
            dna.trunk_segments = [
                SegmentGene(
                    0.4 + rng.random() * 0.6,
                    0.85 + rng.random() * 0.15,
                    0.9,
                    (rng.random() - 0.5) * 0.8,
                    rng.random() * 0.3
                )
                for _ in range(num_segments)
            ]
            dna.branch_count = 8 + rng.integers(0, 12)  # Many small spines
            dna.branch_angle = 0.7 + rng.random() * 0.3  # Nearly perpendicular
            dna.leaf_density = 0.1 + rng.random() * 0.2  # Few leaves
            dna.trunk_color = ColorGene.from_hsv(0.08 + rng.random() * 0.1, 0.3 + rng.random() * 0.3, 0.25 + rng.random() * 0.25)
            dna.leaf_color = ColorGene.from_hsv(0.1 + rng.random() * 0.05, 0.4 + rng.random() * 0.2, 0.3 + rng.random() * 0.2)
            dna.has_flowers = rng.random() < 0.4
            if dna.has_flowers:
                dna.flower_color = ColorGene.from_hsv(rng.random(), 0.7 + rng.random() * 0.3, 0.8 + rng.random() * 0.2)
            
        elif plant_type == PlantType.OCTOPUS:
            # Recursive branching like octopus tentacles
            dna.height_gene = Gene(2 + rng.random() * 5, 0.4, 10, 0.3)
            dna.width_gene = Gene(0.3 + rng.random() * 0.4, 0.1, 1.0, 0.25)
            # Short central body, then branches
            dna.trunk_segments = [
                SegmentGene(0.3 + rng.random() * 0.3, 0.8, 0.7, 0, 0)
            ]
            dna.branch_count = 5 + rng.integers(0, 5)  # Tentacle count
            dna.branch_angle = 0.4 + rng.random() * 0.3  # Spread out
            dna.canopy_shape = "dome"
            dna.asymmetry = 0.4 + rng.random() * 0.3  # Organic asymmetry
            dna.leaf_density = 0
            # Subdued ocean colors
            hue = rng.choice([0.0, 0.05, 0.5, 0.55, 0.75])  # Reds, oranges, teals, purples
            dna.trunk_color = ColorGene.from_hsv(hue, 0.4 + rng.random() * 0.4, 0.4 + rng.random() * 0.4)
            dna.leaf_color = dna.trunk_color
            dna.has_glow = rng.random() < 0.3
            if dna.has_glow:
                dna.glow_intensity = 0.3 + rng.random() * 0.4
                dna.glow_color = ColorGene.from_hsv(hue, 0.5, 0.8)
            
        elif plant_type == PlantType.TENTACLE:
            # Upside-down dangling growths (like hanging from ceiling)
            dna.height_gene = Gene(3 + rng.random() * 7, 0.4, 12, 0.3)
            dna.width_gene = Gene(0.1 + rng.random() * 0.15, 0.03, 0.3, 0.2)
            # Many drooping segments
            num_segments = 4 + rng.integers(0, 6)
            dna.trunk_segments = [
                SegmentGene(
                    0.5 + rng.random() * 0.5,
                    0.9 + rng.random() * 0.1,
                    0.85,
                    -0.2 - rng.random() * 0.3,  # Droop downward
                    rng.random() * 0.2
                )
                for _ in range(num_segments)
            ]
            dna.branch_count = 3 + rng.integers(0, 4)
            dna.branch_angle = 0.1 + rng.random() * 0.2  # Slight angle
            dna.asymmetry = 0.2 + rng.random() * 0.2
            # Pale/translucent colors
            dna.trunk_color = ColorGene.from_hsv(rng.random(), 0.2 + rng.random() * 0.3, 0.6 + rng.random() * 0.3)
            dna.leaf_color = dna.trunk_color
            dna.has_glow = rng.random() < 0.5
            if dna.has_glow:
                dna.glow_intensity = 0.5 + rng.random() * 0.4
                
        elif plant_type == PlantType.SPIRAL:
            # Spiraling growth patterns
            dna.height_gene = Gene(2 + rng.random() * 6, 0.3, 10, 0.25)
            dna.width_gene = Gene(0.15 + rng.random() * 0.2, 0.05, 0.5, 0.2)
            # Many segments with consistent twist
            num_segments = 6 + rng.integers(0, 8)
            twist_dir = 1 if rng.random() > 0.5 else -1
            dna.trunk_segments = [
                SegmentGene(
                    0.4 + rng.random() * 0.3,
                    0.95,
                    0.95,
                    0.05 * i * twist_dir,  # Increasing curve
                    0.15 * twist_dir  # Consistent twist
                )
                for i in range(num_segments)
            ]
            dna.branch_count = 0  # No branches, just spiral
            dna.leaf_density = 0.3 + rng.random() * 0.3
            dna.leaf_shape = "round"
            hue = rng.random()
            dna.trunk_color = ColorGene.from_hsv(hue, 0.5 + rng.random() * 0.3, 0.4 + rng.random() * 0.3)
            dna.leaf_color = ColorGene.from_hsv((hue + 0.1) % 1.0, 0.6, 0.5)
            
        elif plant_type == PlantType.SEAWEED:
            # Underwater swaying plants
            dna.height_gene = Gene(2 + rng.random() * 8, 0.3, 12, 0.25)
            dna.width_gene = Gene(0.05 + rng.random() * 0.1, 0.02, 0.2, 0.15)
            # Long wavy segments
            num_segments = 8 + rng.integers(0, 10)
            dna.trunk_segments = [
                SegmentGene(
                    0.3 + rng.random() * 0.3,
                    0.95,
                    0.98,
                    (rng.random() - 0.5) * 0.4,  # Gentle waves
                    0
                )
                for _ in range(num_segments)
            ]
            dna.branch_count = 1 + rng.integers(0, 4)
            dna.branch_angle = 0.1 + rng.random() * 0.2
            dna.leaf_shape = "blade"
            dna.leaf_density = 0.2 + rng.random() * 0.3
            # Greens, browns, reds for seaweed
            hue = rng.choice([0.25, 0.3, 0.35, 0.05, 0.95])
            dna.trunk_color = ColorGene.from_hsv(hue, 0.3 + rng.random() * 0.4, 0.3 + rng.random() * 0.3)
            dna.leaf_color = ColorGene.from_hsv(hue, 0.4 + rng.random() * 0.3, 0.4 + rng.random() * 0.3)
            
        elif plant_type == PlantType.GROUNDCOVER:
            # Low spreading ground cover - wide and flat
            dna.height_gene = Gene(0.1 + rng.random() * 0.3, 0.05, 0.6, 0.15)
            dna.width_gene = Gene(1.5 + rng.random() * 3.0, 0.5, 6.0, 0.25)  # Wide spread
            dna.trunk_segments = [SegmentGene(0.1, 0.8, 0.95, 0, 0)]
            dna.branch_count = 8 + rng.integers(0, 12)  # Many spreading branches
            dna.branch_angle = 0.8 + rng.random() * 0.15  # Nearly horizontal
            dna.sub_branch_chance = 0.5 + rng.random() * 0.3
            dna.leaf_density = 0.8 + rng.random() * 0.2
            dna.leaf_size = 0.3 + rng.random() * 0.4
            hue = rng.choice([0.25, 0.3, 0.35, 0.15, 0.4])  # Greens, yellows
            dna.trunk_color = ColorGene.from_hsv(hue, 0.3 + rng.random() * 0.3, 0.3 + rng.random() * 0.2)
            dna.leaf_color = ColorGene.from_hsv(hue, 0.4 + rng.random() * 0.4, 0.4 + rng.random() * 0.4)
            
        elif plant_type == PlantType.CREEPER:
            # Ground-hugging vines that spread outward
            dna.height_gene = Gene(0.05 + rng.random() * 0.15, 0.02, 0.3, 0.1)
            dna.width_gene = Gene(2.0 + rng.random() * 4.0, 0.8, 8.0, 0.3)  # Long spread
            num_segments = 5 + rng.integers(0, 8)
            dna.trunk_segments = [
                SegmentGene(
                    0.5 + rng.random() * 0.5,
                    0.95,
                    0.98,
                    (rng.random() - 0.5) * 0.3,  # Winding path
                    0
                )
                for _ in range(num_segments)
            ]
            dna.branch_count = 3 + rng.integers(0, 5)
            dna.branch_angle = 0.3 + rng.random() * 0.4
            dna.sub_branch_chance = 0.4
            dna.leaf_density = 0.4 + rng.random() * 0.4
            dna.tendrils = True
            hue = rng.choice([0.3, 0.35, 0.4, 0.1])
            dna.trunk_color = ColorGene.from_hsv(hue, 0.3 + rng.random() * 0.3, 0.25 + rng.random() * 0.2)
            dna.leaf_color = ColorGene.from_hsv(hue, 0.5 + rng.random() * 0.3, 0.4 + rng.random() * 0.3)
            
        elif plant_type == PlantType.LICHEN:
            # Crusty, spreading, irregular patches
            dna.height_gene = Gene(0.02 + rng.random() * 0.08, 0.01, 0.15, 0.1)
            dna.width_gene = Gene(1.0 + rng.random() * 2.5, 0.3, 5.0, 0.25)
            dna.trunk_segments = [SegmentGene(0.05, 0.9, 0.95, 0, 0)]
            dna.branch_count = 12 + rng.integers(0, 15)  # Many irregular patches
            dna.branch_angle = 0.85 + rng.random() * 0.1  # Nearly flat
            dna.asymmetry = 0.5 + rng.random() * 0.3
            dna.leaf_density = 0
            dna.surface_bumps = 0.6 + rng.random() * 0.3
            # Lichens: grays, yellows, oranges, pale greens
            hue = rng.choice([0.1, 0.15, 0.25, 0.0, 0.05])
            dna.trunk_color = ColorGene.from_hsv(hue, 0.2 + rng.random() * 0.4, 0.4 + rng.random() * 0.4)
            dna.leaf_color = dna.trunk_color
            
        elif plant_type == PlantType.MOSS_PAD:
            # Thick, cushiony moss mounds
            dna.height_gene = Gene(0.15 + rng.random() * 0.35, 0.05, 0.6, 0.15)
            dna.width_gene = Gene(0.8 + rng.random() * 2.0, 0.3, 4.0, 0.2)
            dna.trunk_segments = [SegmentGene(0.3, 0.6, 0.8, 0, 0)]  # Domed shape
            dna.branch_count = 0  # Solid mound
            dna.leaf_density = 1.0  # Fully covered
            dna.leaf_size = 0.05 + rng.random() * 0.1  # Tiny leaves
            dna.has_moss = True
            dna.moss_density = 1.0
            hue = rng.choice([0.25, 0.3, 0.35, 0.4])  # Rich greens
            dna.trunk_color = ColorGene.from_hsv(hue, 0.4 + rng.random() * 0.3, 0.3 + rng.random() * 0.25)
            dna.leaf_color = ColorGene.from_hsv(hue, 0.5 + rng.random() * 0.3, 0.35 + rng.random() * 0.3)
            dna.moss_color = ColorGene.from_hsv(hue + 0.05, 0.5, 0.4)
            
        elif plant_type == PlantType.LILY_PAD:
            # Floating water surface plants
            dna.height_gene = Gene(0.02 + rng.random() * 0.05, 0.01, 0.1, 0.1)
            dna.width_gene = Gene(0.5 + rng.random() * 1.5, 0.2, 3.0, 0.2)
            dna.trunk_segments = [SegmentGene(0.02, 0.9, 0.95, 0, 0)]
            dna.branch_count = 1 + rng.integers(0, 4)  # Multiple pads
            dna.branch_angle = 0.9  # Flat on water
            dna.leaf_shape = "round"
            dna.leaf_size = 0.8 + rng.random() * 0.4
            dna.leaf_density = 1.0
            hue = 0.3 + rng.random() * 0.1  # Green
            dna.trunk_color = ColorGene.from_hsv(hue, 0.4, 0.35)
            dna.leaf_color = ColorGene.from_hsv(hue, 0.5 + rng.random() * 0.3, 0.4 + rng.random() * 0.3)
            # Sometimes has a flower
            if rng.random() < 0.4:
                dna.has_flower = True
                dna.flower_color = ColorGene.from_hsv(rng.choice([0.0, 0.1, 0.85, 0.95]), 0.6, 0.9)
        
        return dna


class DNAPool:
    """
    Manages a population of DNA for a region, handling mutation and crossover
    as the player moves through the world.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.chunk_dna: Dict[Tuple[int, int], List[PlantDNA]] = {}
        
        # Template DNA for each biome (base species)
        self.templates = self._generate_templates()
    
    def _generate_templates(self) -> Dict[str, List[PlantDNA]]:
        """Generate base template DNA for each plant type."""
        templates = {}
        for i, plant_type in enumerate(PlantType.ALL_TYPES):
            templates[plant_type] = [
                PlantDNA.create_random(plant_type, self.seed + i * 100 + j)
                for j in range(3)  # 3 variations per type
            ]
        return templates
    
    def get_dna_for_chunk(self, cx: int, cz: int) -> List[PlantDNA]:
        """
        Get or generate DNA for plants in a chunk.
        Uses neighboring chunk DNA for crossover to create gradual variation.
        """
        key = (cx, cz)
        if key in self.chunk_dna:
            return self.chunk_dna[key]
        
        # Deterministic RNG for this chunk
        chunk_seed = abs(hash((self.seed, cx, cz))) % (2**31)
        chunk_rng = np.random.default_rng(chunk_seed)
        
        # Get neighbor DNA for crossover (if available)
        neighbors = []
        for dx, dz in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor_key = (cx + dx, cz + dz)
            if neighbor_key in self.chunk_dna:
                neighbors.extend(self.chunk_dna[neighbor_key])
        
        # Generate DNA for this chunk
        num_species = 3 + chunk_rng.integers(0, 5)
        chunk_dna = []
        
        for i in range(num_species):
            # Pick a plant type based on position (pseudo-biome)
            biome_factor = np.sin(cx * 0.1) * np.cos(cz * 0.1)
            if biome_factor > 0.3:
                plant_type = chunk_rng.choice([PlantType.TREE, PlantType.TALL_TREE, PlantType.BUSH])
            elif biome_factor < -0.3:
                plant_type = chunk_rng.choice([PlantType.GRASS, PlantType.FERN, PlantType.CACTUS])
            else:
                plant_type = chunk_rng.choice(PlantType.ALL_TYPES)
            
            # Get base template
            base_templates = self.templates.get(plant_type, self.templates[PlantType.TREE])
            base = chunk_rng.choice(base_templates)
            
            # If we have neighbors, crossover with them
            if neighbors and chunk_rng.random() < 0.6:
                # Find similar neighbor
                similar = [n for n in neighbors if n.plant_type == base.plant_type]
                if similar:
                    parent2 = chunk_rng.choice(similar)
                    offspring = base.crossover(parent2, chunk_rng)
                    # Also mutate slightly
                    offspring = offspring.mutate(chunk_rng, strength=0.3)
                    chunk_dna.append(offspring)
                    continue
            
            # Otherwise just mutate the template
            mutated = base.mutate(chunk_rng, strength=0.4)
            chunk_dna.append(mutated)
        
        self.chunk_dna[key] = chunk_dna
        return chunk_dna
    
    def cleanup_distant_chunks(self, center_cx: int, center_cz: int, max_distance: int = 30):
        """Remove DNA for chunks that are too far away to save memory."""
        to_remove = []
        for (cx, cz) in self.chunk_dna:
            if abs(cx - center_cx) > max_distance or abs(cz - center_cz) > max_distance:
                to_remove.append((cx, cz))
        
        for key in to_remove:
            del self.chunk_dna[key]
