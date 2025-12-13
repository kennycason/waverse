"""
Animal DNA System - Procedural creature generation.

Animals are built from:
- Body segments (connected polygons)
- Joints (rotation points between segments)
- Limbs (legs, wings, fins, tentacles)
- Features (eyes, mouths, antennae, tails)

Each animal has:
- Movement type (walk, hop, swim, crawl, fly)
- AI behavior (wander, flock, swarm, graze)
- Animation parameters
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
import copy
import json


Color = Tuple[float, float, float]


@dataclass
class JointGene:
    """A joint connecting two body parts with rotation limits."""
    position: Tuple[float, float, float] = (0, 0, 0)  # Relative position on parent
    rotation_axis: str = "y"  # x, y, z, or "ball" for all axes
    min_angle: float = -45.0  # Degrees
    max_angle: float = 45.0
    speed: float = 1.0  # Animation speed multiplier
    phase: float = 0.0  # Animation phase offset (0-1)
    
    def mutate(self, rng: np.random.Generator, strength: float = 0.5) -> "JointGene":
        """Return mutated copy."""
        rate = 0.15 * strength
        return JointGene(
            position=(
                float(np.clip(self.position[0] + rng.normal(0, rate), -1, 1)),
                float(np.clip(self.position[1] + rng.normal(0, rate), -1, 1)),
                float(np.clip(self.position[2] + rng.normal(0, rate), -1, 1)),
            ),
            rotation_axis=self.rotation_axis,
            min_angle=float(np.clip(self.min_angle + rng.normal(0, 10 * strength), -90, 0)),
            max_angle=float(np.clip(self.max_angle + rng.normal(0, 10 * strength), 0, 90)),
            speed=float(np.clip(self.speed + rng.normal(0, 0.2 * strength), 0.2, 3.0)),
            phase=float((self.phase + rng.normal(0, 0.1 * strength)) % 1.0),
        )


@dataclass
class BodySegmentGene:
    """A body segment - a 3D shape with optional child segments."""
    shape: str = "ellipsoid"  # ellipsoid, box, cone, cylinder
    size: Tuple[float, float, float] = (1.0, 0.5, 0.5)  # width, height, depth
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    pattern: str = "solid"  # solid, striped, spotted, gradient
    pattern_color: Tuple[float, float, float] = (0.3, 0.3, 0.3)
    
    def mutate(self, rng: np.random.Generator, strength: float = 0.5) -> "BodySegmentGene":
        """Return mutated copy."""
        rate = 0.1 * strength
        
        def mutate_color(c):
            return tuple(float(np.clip(v + rng.normal(0, rate), 0, 1)) for v in c)
        
        return BodySegmentGene(
            shape=self.shape if rng.random() > 0.05 * strength else rng.choice(
                ["ellipsoid", "box", "cone", "cylinder"]),
            size=(
                float(np.clip(self.size[0] + rng.normal(0, rate), 0.1, 3.0)),
                float(np.clip(self.size[1] + rng.normal(0, rate), 0.1, 3.0)),
                float(np.clip(self.size[2] + rng.normal(0, rate), 0.1, 3.0)),
            ),
            color=mutate_color(self.color),
            pattern=self.pattern,
            pattern_color=mutate_color(self.pattern_color),
        )


@dataclass
class LimbGene:
    """A limb (leg, wing, fin, tentacle) with segments."""
    limb_type: str = "leg"  # leg, wing, fin, tentacle, arm
    segment_count: int = 2  # Number of segments (1-5)
    segment_lengths: List[float] = field(default_factory=lambda: [0.5, 0.4])
    segment_widths: List[float] = field(default_factory=lambda: [0.15, 0.1])
    color: Tuple[float, float, float] = (0.4, 0.4, 0.4)
    
    # Joint properties for each segment
    joint_speed: float = 1.0
    joint_range: float = 45.0  # Degrees
    
    # For wings
    membrane: bool = False
    membrane_color: Tuple[float, float, float] = (0.6, 0.6, 0.8)
    
    def mutate(self, rng: np.random.Generator, strength: float = 0.5) -> "LimbGene":
        """Return mutated copy."""
        rate = 0.1 * strength
        
        # Possibly change segment count
        new_count = self.segment_count
        if rng.random() < 0.1 * strength:
            new_count = max(1, min(5, self.segment_count + rng.choice([-1, 1])))
        
        # Mutate segment properties
        new_lengths = [
            float(np.clip(l + rng.normal(0, rate), 0.1, 1.5))
            for l in self.segment_lengths
        ]
        new_widths = [
            float(np.clip(w + rng.normal(0, rate * 0.5), 0.05, 0.5))
            for w in self.segment_widths
        ]
        
        # Adjust list lengths if segment count changed
        while len(new_lengths) < new_count:
            new_lengths.append(0.3 + rng.random() * 0.3)
        while len(new_widths) < new_count:
            new_widths.append(0.1 + rng.random() * 0.1)
        new_lengths = new_lengths[:new_count]
        new_widths = new_widths[:new_count]
        
        return LimbGene(
            limb_type=self.limb_type,
            segment_count=new_count,
            segment_lengths=new_lengths,
            segment_widths=new_widths,
            color=tuple(float(np.clip(c + rng.normal(0, rate), 0, 1)) for c in self.color),
            joint_speed=float(np.clip(self.joint_speed + rng.normal(0, 0.2 * strength), 0.3, 3.0)),
            joint_range=float(np.clip(self.joint_range + rng.normal(0, 10 * strength), 15, 90)),
            membrane=self.membrane,
            membrane_color=self.membrane_color,
        )


@dataclass
class FeatureGene:
    """A feature like eyes, mouth, antennae."""
    feature_type: str = "eye"  # eye, mouth, antenna, horn, spike, tail
    count: int = 2  # Number of this feature (e.g., 2 eyes)
    size: float = 0.1
    color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    position: Tuple[float, float, float] = (0.4, 0.3, 0.2)  # Relative to body
    glow: bool = False
    
    def mutate(self, rng: np.random.Generator, strength: float = 0.5) -> "FeatureGene":
        """Return mutated copy."""
        rate = 0.1 * strength
        return FeatureGene(
            feature_type=self.feature_type,
            count=max(1, min(8, self.count + int(rng.normal(0, 0.5 * strength)))),
            size=float(np.clip(self.size + rng.normal(0, rate), 0.02, 0.5)),
            color=tuple(float(np.clip(c + rng.normal(0, rate), 0, 1)) for c in self.color),
            position=tuple(float(np.clip(p + rng.normal(0, rate), -1, 1)) for p in self.position),
            glow=self.glow if rng.random() > 0.05 * strength else not self.glow,
        )


class MovementType:
    """Movement types for animals."""
    WALK = "walk"       # Quadruped/biped walking
    HOP = "hop"         # Hopping (rabbits, frogs)
    CRAWL = "crawl"     # Crawling (snakes, worms)
    SWIM = "swim"       # Swimming (fish, whales)
    FLY = "fly"         # Flying (birds, insects)
    FLOAT = "float"     # Floating/drifting (jellyfish)
    
    ALL = [WALK, HOP, CRAWL, SWIM, FLY, FLOAT]


class AIBehavior:
    """AI behavior patterns."""
    WANDER = "wander"     # Random wandering
    GRAZE = "graze"       # Stay in area, occasional movement
    FLOCK = "flock"       # Bird-like flocking
    SWARM = "swarm"       # Insect-like swarming
    PREDATOR = "predator" # Hunt other animals
    FLEE = "flee"         # Run from player/predators
    SCHOOL = "school"     # Fish schooling
    
    ALL = [WANDER, GRAZE, FLOCK, SWARM, PREDATOR, FLEE, SCHOOL]


class AnimalType:
    """High-level animal categories."""
    INSECT = "insect"
    BIRD = "bird"
    FISH = "fish"
    MAMMAL = "mammal"
    REPTILE = "reptile"
    AMPHIBIAN = "amphibian"
    JELLYFISH = "jellyfish"
    WORM = "worm"
    ALIEN = "alien"
    METROID = "metroid"  # Floating brain-like creature
    SPIDER = "spider"    # 8-legged crawlers
    DINOSAUR = "dinosaur"  # Big bipeds and quadrupeds
    CROC = "croc"        # Alligators/crocodiles
    HOPPER = "hopper"    # Rabbits, frogs, kangaroos
    
    ALL = [INSECT, BIRD, FISH, MAMMAL, REPTILE, AMPHIBIAN, JELLYFISH, WORM, ALIEN, METROID, SPIDER, DINOSAUR, CROC, HOPPER]
    
    # Ground animals for weighted spawning
    GROUND = [MAMMAL, REPTILE, WORM, INSECT, SPIDER, DINOSAUR, CROC, HOPPER]


@dataclass
class AnimalDNA:
    """
    Complete DNA for a procedural animal.
    
    Structure:
    - Body: Main body segment(s)
    - Limbs: Attached at joints
    - Features: Eyes, mouth, etc.
    - Movement: How it moves
    - AI: How it behaves
    """
    # Identity
    animal_type: str = AnimalType.MAMMAL
    species_id: int = 0
    generation: int = 0
    
    # Body structure
    body_segments: List[BodySegmentGene] = field(default_factory=lambda: [
        BodySegmentGene(shape="ellipsoid", size=(1.0, 0.6, 0.5))
    ])
    body_joints: List[JointGene] = field(default_factory=list)
    
    # Limbs
    limbs: List[LimbGene] = field(default_factory=list)
    limb_pairs: int = 2  # Number of limb pairs (bilateral symmetry)
    
    # Features
    features: List[FeatureGene] = field(default_factory=lambda: [
        FeatureGene(feature_type="eye", count=2, size=0.08, color=(0.1, 0.1, 0.1)),
    ])
    
    # Movement
    movement_type: str = MovementType.WALK
    movement_speed: float = 1.0  # Base speed multiplier
    animation_speed: float = 1.0
    
    # AI Behavior
    ai_behavior: str = AIBehavior.WANDER
    awareness_radius: float = 10.0  # How far it can "see"
    flee_radius: float = 5.0  # Distance to start fleeing
    group_tendency: float = 0.5  # 0 = solitary, 1 = always in groups
    
    # Size
    base_scale: float = 1.0
    
    # Colors
    primary_color: Tuple[float, float, float] = (0.5, 0.4, 0.3)
    secondary_color: Tuple[float, float, float] = (0.6, 0.5, 0.4)
    
    def mutate(self, rng: np.random.Generator = None, strength: float = 0.5) -> "AnimalDNA":
        """Create mutated copy of this DNA."""
        if rng is None:
            rng = np.random.default_rng()
        
        new_dna = copy.deepcopy(self)
        new_dna.generation += 1
        
        # Mutate body segments
        new_dna.body_segments = [seg.mutate(rng, strength) for seg in self.body_segments]
        
        # Maybe add/remove body segment
        if rng.random() < 0.05 * strength and len(new_dna.body_segments) < 5:
            new_dna.body_segments.append(BodySegmentGene(
                shape=rng.choice(["ellipsoid", "box", "cylinder"]),
                size=(0.5 + rng.random() * 0.5, 0.3 + rng.random() * 0.3, 0.3 + rng.random() * 0.3),
                color=new_dna.primary_color
            ))
        
        # Mutate limbs
        new_dna.limbs = [limb.mutate(rng, strength) for limb in self.limbs]
        
        # Mutate features
        new_dna.features = [feat.mutate(rng, strength) for feat in self.features]
        
        # Mutate scalars
        rate = 0.1 * strength
        new_dna.movement_speed = float(np.clip(self.movement_speed + rng.normal(0, 0.2 * strength), 0.3, 3.0))
        new_dna.animation_speed = float(np.clip(self.animation_speed + rng.normal(0, 0.2 * strength), 0.3, 3.0))
        new_dna.awareness_radius = float(np.clip(self.awareness_radius + rng.normal(0, 2 * strength), 3, 30))
        new_dna.group_tendency = float(np.clip(self.group_tendency + rng.normal(0, rate), 0, 1))
        new_dna.base_scale = float(np.clip(self.base_scale + rng.normal(0, 0.15 * strength), 0.2, 3.0))
        
        # Mutate colors - with occasional larger shifts for variety
        def mutate_color(c):
            result = list(c)
            for i in range(3):
                mutation = rng.normal(0, rate)
                # Rare larger color mutation for psychedelic creatures
                if rng.random() < 0.1:  # 10% chance per channel
                    mutation += rng.normal(0, 0.15)
                result[i] = float(np.clip(c[i] + mutation, 0, 1))
            return tuple(result)
        
        new_dna.primary_color = mutate_color(self.primary_color)
        new_dna.secondary_color = mutate_color(self.secondary_color)
        
        return new_dna
    
    def crossover(self, other: "AnimalDNA", rng: np.random.Generator = None) -> "AnimalDNA":
        """Create offspring by combining two parent DNAs."""
        if rng is None:
            rng = np.random.default_rng()
        
        child = copy.deepcopy(self if rng.random() < 0.5 else other)
        child.generation = max(self.generation, other.generation) + 1
        
        # Pick traits from either parent
        if rng.random() < 0.5:
            child.body_segments = copy.deepcopy(other.body_segments)
        if rng.random() < 0.5:
            child.limbs = copy.deepcopy(other.limbs)
        if rng.random() < 0.5:
            child.features = copy.deepcopy(other.features)
        
        # Blend scalars
        blend = rng.random()
        child.movement_speed = self.movement_speed * blend + other.movement_speed * (1 - blend)
        child.base_scale = self.base_scale * blend + other.base_scale * (1 - blend)
        child.group_tendency = self.group_tendency * blend + other.group_tendency * (1 - blend)
        
        # Blend colors
        for i in range(3):
            child.primary_color = tuple(
                self.primary_color[j] * blend + other.primary_color[j] * (1 - blend)
                for j in range(3)
            )
        
        return child
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "animal_type": self.animal_type,
            "species_id": int(self.species_id),
            "generation": self.generation,
            "movement_type": self.movement_type,
            "ai_behavior": self.ai_behavior,
            "base_scale": float(self.base_scale),
            "primary_color": [float(c) for c in self.primary_color],
            "body_segment_count": len(self.body_segments),
            "limb_count": len(self.limbs),
            "feature_count": len(self.features),
        }
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def create_random(cls, animal_type: str = None, seed: int = None) -> "AnimalDNA":
        """Create a random animal of a given type."""
        rng = np.random.default_rng(seed)
        
        if animal_type is None:
            animal_type = rng.choice(AnimalType.ALL)
        
        dna = cls(animal_type=animal_type, species_id=seed or rng.integers(0, 100000))
        
        # Configure based on type
        if animal_type == AnimalType.INSECT:
            dna = cls._create_insect(rng)
        elif animal_type == AnimalType.BIRD:
            dna = cls._create_bird(rng)
        elif animal_type == AnimalType.FISH:
            dna = cls._create_fish(rng)
        elif animal_type == AnimalType.MAMMAL:
            dna = cls._create_mammal(rng)
        elif animal_type == AnimalType.REPTILE:
            dna = cls._create_reptile(rng)
        elif animal_type == AnimalType.JELLYFISH:
            dna = cls._create_jellyfish(rng)
        elif animal_type == AnimalType.WORM:
            dna = cls._create_worm(rng)
        elif animal_type == AnimalType.ALIEN:
            dna = cls._create_alien(rng)
        elif animal_type == AnimalType.METROID:
            dna = cls._create_metroid(rng)
        elif animal_type == AnimalType.SPIDER:
            dna = cls._create_spider(rng)
        elif animal_type == AnimalType.DINOSAUR:
            dna = cls._create_dinosaur(rng)
        elif animal_type == AnimalType.CROC:
            dna = cls._create_croc(rng)
        elif animal_type == AnimalType.HOPPER:
            dna = cls._create_hopper(rng)
        else:
            dna = cls._create_mammal(rng)
        
        dna.animal_type = animal_type
        dna.species_id = seed or rng.integers(0, 100000)
        return dna
    
    @classmethod
    def _create_insect(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create insect DNA - mostly crawling beetles and ants, some flying."""
        # Random insect color
        hue = rng.random()
        color = _hsv_to_rgb(hue, 0.5 + rng.random() * 0.4, 0.3 + rng.random() * 0.4)
        
        # Size multiplier for variety
        size_mult = 0.5 + rng.random() * 1.5  # 0.5x to 2x base size
        
        dna = cls()
        dna.body_segments = [
            BodySegmentGene(shape="ellipsoid", size=(0.3 * size_mult, 0.2 * size_mult, 0.15 * size_mult), color=color),
            BodySegmentGene(shape="ellipsoid", size=(0.4 * size_mult, 0.25 * size_mult, 0.2 * size_mult), color=color),
            BodySegmentGene(shape="ellipsoid", size=(0.5 * size_mult, 0.3 * size_mult, 0.25 * size_mult), color=color),
        ]
        
        dna.limbs = [
            LimbGene(limb_type="leg", segment_count=3, 
                    segment_lengths=[0.2 * size_mult, 0.25 * size_mult, 0.15 * size_mult], 
                    segment_widths=[0.03 * size_mult, 0.02 * size_mult, 0.01 * size_mult],
                    color=(0.2, 0.2, 0.2)),
        ]
        dna.limb_pairs = 3  # 6 legs
        
        # Only 30% have wings - most are ground crawlers
        if rng.random() < 0.3:
            dna.limbs.append(LimbGene(
                limb_type="wing", segment_count=1,
                segment_lengths=[0.6 * size_mult], segment_widths=[0.02 * size_mult],
                membrane=True, membrane_color=(0.8, 0.8, 0.9, 0.5)
            ))
            dna.movement_type = MovementType.FLY
        else:
            dna.movement_type = MovementType.CRAWL
        
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=0.08 * size_mult, color=(0.1, 0.0, 0.0), position=(0.35, 0.15, 0.1)),
            FeatureGene(feature_type="antenna", count=2, size=0.15 * size_mult, color=(0.2, 0.2, 0.2), position=(0.4, 0.2, 0.05)),
        ]
        
        dna.ai_behavior = AIBehavior.WANDER
        dna.base_scale = 0.2 + rng.random() * 0.5  # Bigger insects
        dna.primary_color = color
        dna.movement_speed = 0.5 + rng.random() * 1.0
        dna.animation_speed = 2.0 + rng.random() * 2.0
        
        return dna
    
    @classmethod
    def _create_bird(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create bird DNA - some flying, some ground birds."""
        hue = rng.random()
        body_color = _hsv_to_rgb(hue, 0.4 + rng.random() * 0.4, 0.4 + rng.random() * 0.4)
        wing_color = _hsv_to_rgb((hue + 0.1) % 1, 0.5, 0.5)
        
        # Size variety - small songbirds to large ostriches
        is_ground_bird = rng.random() < 0.35  # 35% are ground birds
        size_mult = 0.4 + rng.random() * 0.8 if not is_ground_bird else 1.0 + rng.random() * 1.5
        
        dna = cls()
        dna.body_segments = [
            BodySegmentGene(shape="ellipsoid", size=(0.25 * size_mult, 0.2 * size_mult, 0.2 * size_mult), color=body_color),
            BodySegmentGene(shape="ellipsoid", size=(0.6 * size_mult, 0.4 * size_mult, 0.35 * size_mult), color=body_color),
        ]
        
        if is_ground_bird:
            # Ground bird - stronger legs, smaller wings
            dna.limbs = [
                LimbGene(limb_type="wing", segment_count=1,
                        segment_lengths=[0.3 * size_mult], segment_widths=[0.05 * size_mult],
                        color=wing_color, membrane=True, membrane_color=wing_color),
                LimbGene(limb_type="leg", segment_count=2,
                        segment_lengths=[0.35 * size_mult, 0.25 * size_mult], segment_widths=[0.06 * size_mult, 0.04 * size_mult],
                        color=(0.6, 0.5, 0.2)),
            ]
            dna.movement_type = MovementType.WALK
            dna.ai_behavior = rng.choice([AIBehavior.GRAZE, AIBehavior.WANDER])
            dna.movement_speed = 0.8 + rng.random() * 0.6
        else:
            # Flying bird
            dna.limbs = [
                LimbGene(limb_type="wing", segment_count=2,
                        segment_lengths=[0.5 * size_mult, 0.4 * size_mult], segment_widths=[0.08 * size_mult, 0.05 * size_mult],
                        color=wing_color, membrane=True, membrane_color=wing_color),
                LimbGene(limb_type="leg", segment_count=2,
                        segment_lengths=[0.2 * size_mult, 0.15 * size_mult], segment_widths=[0.03 * size_mult, 0.02 * size_mult],
                        color=(0.6, 0.5, 0.2)),
            ]
            dna.movement_type = MovementType.FLY
            dna.ai_behavior = AIBehavior.FLOCK
            dna.movement_speed = 1.5 + rng.random() * 1.5
        
        dna.limb_pairs = 1
        
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=0.05 * size_mult, color=(0.05, 0.05, 0.05), position=(0.35, 0.1, 0.12)),
            FeatureGene(feature_type="mouth", count=1, size=0.12 * size_mult, color=(0.7, 0.5, 0.1), position=(0.45, 0.0, 0.0)),
        ]
        
        dna.base_scale = size_mult
        dna.primary_color = body_color
        dna.secondary_color = wing_color
        dna.group_tendency = 0.4 + rng.random() * 0.4
        
        return dna
    
    @classmethod
    def _create_fish(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create fish DNA."""
        hue = rng.random()
        color = _hsv_to_rgb(hue, 0.5 + rng.random() * 0.4, 0.5 + rng.random() * 0.4)
        fin_color = _hsv_to_rgb((hue + 0.05) % 1, 0.6, 0.6)
        
        dna = cls()
        dna.body_segments = [
            BodySegmentGene(shape="ellipsoid", size=(0.8, 0.35, 0.2), color=color, 
                          pattern="gradient" if rng.random() < 0.5 else "solid"),
        ]
        
        dna.limbs = [
            # Side fins
            LimbGene(limb_type="fin", segment_count=1,
                    segment_lengths=[0.2], segment_widths=[0.15],
                    color=fin_color, membrane=True),
            # Tail fin
            LimbGene(limb_type="fin", segment_count=1,
                    segment_lengths=[0.25], segment_widths=[0.2],
                    color=fin_color, membrane=True),
        ]
        dna.limb_pairs = 1
        
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=0.06, color=(0.1, 0.1, 0.1), position=(0.35, 0.05, 0.12)),
            FeatureGene(feature_type="mouth", count=1, size=0.05, color=(0.4, 0.2, 0.2), position=(0.45, -0.05, 0.0)),
        ]
        
        dna.movement_type = MovementType.SWIM
        dna.ai_behavior = AIBehavior.SCHOOL
        dna.base_scale = 0.2 + rng.random() * 0.6
        dna.primary_color = color
        dna.movement_speed = 1.0 + rng.random() * 1.0
        dna.group_tendency = 0.7 + rng.random() * 0.25
        
        return dna
    
    @classmethod
    def _create_mammal(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create mammal DNA - wide variety from small critters to large beasts."""
        # Earth tones
        hue = 0.03 + rng.random() * 0.12  # Browns/tans/grays
        color = _hsv_to_rgb(hue, 0.2 + rng.random() * 0.4, 0.3 + rng.random() * 0.5)
        
        # Size category: 0=small, 1=medium, 2=large, 3=huge
        # Weighted toward bigger animals
        size_category = rng.choice([0, 1, 1, 2, 2, 2, 3, 3])
        size_scales = [0.5, 1.0, 2.0, 4.0]  # Bigger overall
        size_mult = size_scales[size_category] * (0.8 + rng.random() * 0.4)
        
        dna = cls()
        
        # Body proportions vary by size
        body_length = 0.8 + size_category * 0.3  # Bigger = longer body
        body_height = 0.4 + size_category * 0.15
        
        dna.body_segments = [
            BodySegmentGene(shape="ellipsoid", size=(0.3 * size_mult, 0.25 * size_mult, 0.22 * size_mult), color=color),
            BodySegmentGene(shape="ellipsoid", size=(body_length * size_mult, body_height * size_mult, 0.4 * size_mult), color=color),
        ]
        
        # Leg proportions scale with body
        leg_length = 0.2 + size_category * 0.1
        dna.limbs = [
            LimbGene(limb_type="leg", segment_count=2,
                    segment_lengths=[leg_length * size_mult, leg_length * 0.8 * size_mult], 
                    segment_widths=[0.08 * size_mult, 0.06 * size_mult],
                    color=color, joint_range=40),
        ]
        dna.limb_pairs = 2  # 4 legs
        
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=0.04 * size_mult, color=(0.1, 0.05, 0.0), position=(0.3, 0.1, 0.1)),
            FeatureGene(feature_type="mouth", count=1, size=0.06 * size_mult, color=(0.3, 0.15, 0.15), position=(0.4, -0.08, 0.0)),
            FeatureGene(feature_type="tail", count=1, size=0.3 * size_mult, color=color, position=(-0.45, 0.1, 0.0)),
        ]
        
        # Ears for most mammals
        if rng.random() < 0.8:
            ear_size = 0.08 + size_category * 0.04
            dna.features.append(FeatureGene(
                feature_type="antenna", count=2, size=ear_size * size_mult, color=color, position=(0.2, 0.22, 0.08)
            ))
        
        # Big mammals might have horns
        if size_category >= 2 and rng.random() < 0.4:
            dna.features.append(FeatureGene(
                feature_type="horn", count=2, size=0.15 * size_mult, color=(0.4, 0.35, 0.3), position=(0.15, 0.25, 0.1)
            ))
        
        dna.movement_type = MovementType.WALK
        # Small ones hop sometimes, big ones always walk
        if size_category == 0 and rng.random() < 0.4:
            dna.movement_type = MovementType.HOP
        
        dna.ai_behavior = rng.choice([AIBehavior.GRAZE, AIBehavior.GRAZE, AIBehavior.WANDER, AIBehavior.FLEE])
        dna.base_scale = size_mult
        dna.primary_color = color
        dna.movement_speed = 1.2 - size_category * 0.2 + rng.random() * 0.5  # Big = slower
        
        return dna
    
    @classmethod
    def _create_reptile(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create reptile DNA - from small lizards to large crocodilians."""
        hue = 0.15 + rng.random() * 0.25  # Greens/browns/olive
        color = _hsv_to_rgb(hue, 0.35 + rng.random() * 0.4, 0.25 + rng.random() * 0.4)
        
        # Size variety - small gecko to large monitor/croc
        size_category = rng.choice([0, 0, 1, 1, 2, 3])  # Most are small/medium
        size_scales = [0.4, 0.8, 1.5, 2.5]
        size_mult = size_scales[size_category] * (0.8 + rng.random() * 0.4)
        
        # Larger reptiles have longer bodies and tails
        body_length = 0.5 + size_category * 0.2
        tail_length = 0.4 + size_category * 0.3
        
        dna = cls()
        dna.body_segments = [
            BodySegmentGene(shape="ellipsoid", size=(0.22 * size_mult, 0.12 * size_mult, 0.16 * size_mult), color=color),
            BodySegmentGene(shape="ellipsoid", size=(body_length * size_mult, 0.2 * size_mult, 0.28 * size_mult), color=color),
            BodySegmentGene(shape="cone", size=(tail_length * size_mult, 0.1 * size_mult, 0.12 * size_mult), color=color),
        ]
        
        # Legs splay out more for crawling
        leg_length = 0.12 + size_category * 0.06
        dna.limbs = [
            LimbGene(limb_type="leg", segment_count=2,
                    segment_lengths=[leg_length * size_mult, leg_length * 0.8 * size_mult], 
                    segment_widths=[0.05 * size_mult, 0.04 * size_mult],
                    color=color, joint_range=35),
        ]
        dna.limb_pairs = 2
        
        # Reptile eyes with slit pupils
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=0.04 * size_mult, 
                       color=(0.9, 0.7, 0.0), position=(0.38, 0.06, 0.09)),
        ]
        
        # Big reptiles might have spines or crests
        if size_category >= 2 and rng.random() < 0.5:
            dna.features.append(FeatureGene(
                feature_type="spike", count=4 + rng.integers(0, 4), 
                size=0.06 * size_mult, color=(0.3, 0.3, 0.25), 
                position=(0.0, 0.15, 0.0)
            ))
        
        dna.movement_type = MovementType.CRAWL
        dna.ai_behavior = rng.choice([AIBehavior.WANDER, AIBehavior.GRAZE])
        dna.base_scale = size_mult
        dna.primary_color = color
        dna.movement_speed = 0.6 - size_category * 0.1 + rng.random() * 0.5  # Big = slower
        
        return dna
    
    @classmethod
    def _create_jellyfish(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create jellyfish DNA."""
        hue = rng.random()
        color = _hsv_to_rgb(hue, 0.3 + rng.random() * 0.4, 0.6 + rng.random() * 0.3)
        
        dna = cls()
        dna.body_segments = [
            BodySegmentGene(shape="ellipsoid", size=(0.4, 0.25, 0.4), color=(*color, 0.6)),  # Bell
        ]
        
        dna.limbs = [
            LimbGene(limb_type="tentacle", segment_count=4,
                    segment_lengths=[0.3, 0.25, 0.2, 0.15], segment_widths=[0.02, 0.015, 0.01, 0.008],
                    color=color),
        ]
        dna.limb_pairs = 4  # 8 tentacles
        
        dna.features = []  # Jellyfish don't have obvious features
        
        dna.movement_type = MovementType.FLOAT
        dna.ai_behavior = AIBehavior.WANDER
        dna.base_scale = 0.3 + rng.random() * 0.7
        dna.primary_color = color
        dna.movement_speed = 0.2 + rng.random() * 0.4
        dna.animation_speed = 0.5 + rng.random() * 0.5
        
        # Jellyfish often glow
        if rng.random() < 0.5:
            dna.features.append(FeatureGene(
                feature_type="eye", count=0, size=0, glow=True, color=color
            ))
        
        return dna
    
    @classmethod
    def _create_worm(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create worm/snake DNA - from small worms to large snakes."""
        # Choose between worm (small, pink) and snake (larger, patterned)
        is_snake = rng.random() < 0.6
        
        if is_snake:
            # Snakes - greens, browns, with patterns
            hue = rng.choice([0.08, 0.15, 0.25, 0.35])  # Brown, tan, green, olive
            color = _hsv_to_rgb(hue, 0.4 + rng.random() * 0.3, 0.3 + rng.random() * 0.4)
            pattern_color = _hsv_to_rgb((hue + 0.05) % 1, 0.5, 0.5)
            segment_count = 8 + rng.integers(0, 8)  # 8-15 segments
            size_mult = 0.8 + rng.random() * 2.0  # Can be quite large
            segment_width = 0.12 + rng.random() * 0.08
        else:
            # Worms - smaller, pink/red
            hue = 0.0 + rng.random() * 0.08
            color = _hsv_to_rgb(hue, 0.3 + rng.random() * 0.2, 0.5 + rng.random() * 0.3)
            pattern_color = color
            segment_count = 4 + rng.integers(0, 4)
            size_mult = 0.3 + rng.random() * 0.4
            segment_width = 0.08
        
        dna = cls()
        
        # Create tapering body - head to tail
        head_width = segment_width * 1.2
        tail_width = segment_width * 0.4
        
        dna.body_segments = []
        for i in range(segment_count):
            t = i / max(1, segment_count - 1)  # 0 at head, 1 at tail
            # Taper from head to tail
            width = head_width * (1 - t * 0.7)
            height = width * 0.8
            length = 0.15 + (0.1 if is_snake else 0.05)
            
            # Alternate pattern for snakes
            seg_color = color if (i % 2 == 0) or not is_snake else pattern_color
            
            dna.body_segments.append(BodySegmentGene(
                shape="ellipsoid", 
                size=(length * size_mult, height * size_mult, width * size_mult), 
                color=seg_color,
                pattern="striped" if is_snake and rng.random() < 0.3 else "solid"
            ))
        
        dna.limbs = []  # No limbs
        dna.limb_pairs = 0
        
        # Snakes have more prominent eyes, forked tongue
        eye_size = 0.04 if is_snake else 0.02
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=eye_size * size_mult, 
                       color=(0.1, 0.1, 0.0) if is_snake else (0.05, 0.05, 0.05), 
                       position=(0.4, 0.05, 0.06)),
        ]
        
        if is_snake:
            # Forked tongue
            dna.features.append(FeatureGene(
                feature_type="antenna", count=1, size=0.08 * size_mult, 
                color=(0.8, 0.2, 0.2), position=(0.5, -0.02, 0.0)
            ))
        
        dna.movement_type = MovementType.CRAWL
        dna.ai_behavior = AIBehavior.WANDER if is_snake else AIBehavior.GRAZE
        dna.base_scale = size_mult
        dna.primary_color = color
        dna.secondary_color = pattern_color
        dna.movement_speed = 0.4 + rng.random() * 0.6 if is_snake else 0.2 + rng.random() * 0.3
        dna.animation_speed = 1.5 + rng.random() * 1.0  # Undulation speed
        
        return dna
    
    @classmethod
    def _create_alien(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create weird alien creature DNA."""
        hue = rng.random()
        color = _hsv_to_rgb(hue, 0.5 + rng.random() * 0.5, 0.4 + rng.random() * 0.5)
        secondary = _hsv_to_rgb((hue + 0.3 + rng.random() * 0.4) % 1, 0.6, 0.6)
        
        dna = cls()
        
        # Random number of body segments
        seg_count = 1 + rng.integers(0, 4)
        dna.body_segments = [
            BodySegmentGene(
                shape=rng.choice(["ellipsoid", "box", "cone", "cylinder"]),
                size=(0.2 + rng.random() * 0.6, 0.15 + rng.random() * 0.4, 0.15 + rng.random() * 0.4),
                color=color if i % 2 == 0 else secondary,
                pattern=rng.choice(["solid", "striped", "spotted"])
            )
            for i in range(seg_count)
        ]
        
        # Random limbs
        limb_type = rng.choice(["leg", "tentacle", "wing", "fin", "arm"])
        dna.limbs = [
            LimbGene(
                limb_type=limb_type,
                segment_count=1 + rng.integers(0, 4),
                segment_lengths=[0.2 + rng.random() * 0.3 for _ in range(1 + rng.integers(0, 4))],
                segment_widths=[0.05 + rng.random() * 0.1 for _ in range(1 + rng.integers(0, 4))],
                color=secondary,
                membrane=limb_type == "wing",
            )
        ]
        dna.limb_pairs = 1 + rng.integers(0, 5)
        
        # Random features
        dna.features = [
            FeatureGene(
                feature_type="eye",
                count=1 + rng.integers(0, 7),
                size=0.03 + rng.random() * 0.1,
                color=_hsv_to_rgb(rng.random(), 0.8, 0.9),
                glow=rng.random() < 0.3,
                position=(0.3 + rng.random() * 0.2, rng.random() * 0.2, rng.random() * 0.15)
            )
        ]
        
        # Maybe add more features
        if rng.random() < 0.5:
            dna.features.append(FeatureGene(
                feature_type=rng.choice(["antenna", "horn", "spike"]),
                count=1 + rng.integers(0, 4),
                size=0.1 + rng.random() * 0.2,
                color=secondary
            ))
        
        dna.movement_type = rng.choice(MovementType.ALL)
        dna.ai_behavior = rng.choice(AIBehavior.ALL)
        dna.base_scale = 0.2 + rng.random() * 1.5
        dna.primary_color = color
        dna.secondary_color = secondary
        dna.movement_speed = 0.5 + rng.random() * 2.0
        dna.animation_speed = 0.5 + rng.random() * 2.0
        
        return dna
    
    @classmethod
    def _create_metroid(cls, rng: np.random.Generator) -> "AnimalDNA":
        """
        Create Metroid-like creature - floating brain/jellyfish parasite.
        
        Translucent dome body with visible internal structures,
        dangling tentacles/fangs, ominous glow.
        """
        # Creepy colors - greens, teals, translucent
        hue = 0.4 + rng.random() * 0.2  # Green to cyan
        body_color = _hsv_to_rgb(hue, 0.4 + rng.random() * 0.3, 0.5 + rng.random() * 0.3)
        inner_color = _hsv_to_rgb(hue - 0.1, 0.6, 0.7)  # Inner glow
        tentacle_color = _hsv_to_rgb(hue + 0.05, 0.5, 0.4)
        
        # Size - they can be small parasites or large hunters
        size_mult = 0.5 + rng.random() * 1.5
        
        dna = cls()
        
        # Main body - dome/bell shape (like a jellyfish top)
        dna.body_segments = [
            # Outer membrane (translucent dome)
            BodySegmentGene(
                shape="ellipsoid", 
                size=(0.5 * size_mult, 0.35 * size_mult, 0.5 * size_mult), 
                color=(*body_color, 0.6),  # Semi-transparent
                pattern="gradient"
            ),
            # Inner "brain" structure
            BodySegmentGene(
                shape="ellipsoid", 
                size=(0.3 * size_mult, 0.2 * size_mult, 0.3 * size_mult), 
                color=inner_color,
                pattern="spotted"  # Brain-like texture
            ),
        ]
        
        # Dangling tentacles/fangs
        num_tentacles = 3 + rng.integers(0, 4)  # 3-6 tentacles
        tentacle_length = 0.4 + rng.random() * 0.4
        
        dna.limbs = [
            LimbGene(
                limb_type="tentacle", 
                segment_count=4,
                segment_lengths=[tentacle_length * size_mult * (0.9 ** i) for i in range(4)],
                segment_widths=[0.04 * size_mult * (0.8 ** i) for i in range(4)],
                color=tentacle_color,
                joint_speed=0.5,
                joint_range=40
            ),
        ]
        dna.limb_pairs = num_tentacles  # Multiple dangling appendages
        
        # Features - multiple small eyes, maybe mandibles
        eye_count = 2 + rng.integers(0, 4)  # 2-5 eyes
        dna.features = [
            # Creepy clustered eyes
            FeatureGene(
                feature_type="eye", 
                count=eye_count, 
                size=0.06 * size_mult, 
                color=(0.9, 0.1, 0.1),  # Red eyes
                glow=True,  # Glowing!
                position=(0.3, 0.1, 0.1)
            ),
        ]
        
        # Maybe add fang-like protrusions
        if rng.random() < 0.7:
            dna.features.append(FeatureGene(
                feature_type="spike", 
                count=2 + rng.integers(0, 3),
                size=0.12 * size_mult,
                color=(0.8, 0.7, 0.6),
                position=(0.0, -0.3, 0.0)  # Bottom, like fangs
            ))
        
        # Metroids float menacingly
        dna.movement_type = MovementType.FLOAT
        dna.ai_behavior = rng.choice([AIBehavior.WANDER, AIBehavior.SWARM])
        dna.base_scale = size_mult
        dna.primary_color = body_color
        dna.secondary_color = inner_color
        
        # Slow, drifting movement
        dna.movement_speed = 0.4 + rng.random() * 0.4
        dna.animation_speed = 0.6 + rng.random() * 0.4  # Slow pulsing
        dna.group_tendency = 0.3 + rng.random() * 0.3  # Sometimes swarm
        
        return dna
    
    @classmethod
    def _create_spider(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create spider/arachnid DNA - 8 legs, various sizes."""
        # Dark colors - blacks, browns, some colorful
        if rng.random() < 0.7:
            hue = 0.05 + rng.random() * 0.1  # Browns/blacks
            color = _hsv_to_rgb(hue, 0.3 + rng.random() * 0.3, 0.15 + rng.random() * 0.25)
        else:
            hue = rng.random()  # Colorful spider
            color = _hsv_to_rgb(hue, 0.6, 0.4)
        
        # Size - from tiny to tarantula-sized
        size_mult = 0.3 + rng.random() * 1.2
        
        dna = cls()
        
        # Cephalothorax (head+body) and abdomen
        dna.body_segments = [
            BodySegmentGene(shape="ellipsoid", size=(0.25 * size_mult, 0.15 * size_mult, 0.2 * size_mult), color=color),
            BodySegmentGene(shape="ellipsoid", size=(0.4 * size_mult, 0.3 * size_mult, 0.35 * size_mult), color=color),
        ]
        
        # 8 long spindly legs
        leg_length = 0.4 + rng.random() * 0.3
        dna.limbs = [
            LimbGene(limb_type="leg", segment_count=3,
                    segment_lengths=[leg_length * size_mult * 0.4, leg_length * size_mult * 0.35, leg_length * size_mult * 0.25],
                    segment_widths=[0.025 * size_mult, 0.018 * size_mult, 0.01 * size_mult],
                    color=color, joint_range=50),
        ]
        dna.limb_pairs = 4  # 8 legs total
        
        # Multiple eyes
        eye_count = rng.choice([4, 6, 8])
        dna.features = [
            FeatureGene(feature_type="eye", count=eye_count, size=0.03 * size_mult, 
                       color=(0.1, 0.1, 0.1), position=(0.4, 0.08, 0.08)),
        ]
        
        # Maybe fangs
        if rng.random() < 0.6:
            dna.features.append(FeatureGene(
                feature_type="spike", count=2, size=0.08 * size_mult,
                color=(0.2, 0.15, 0.1), position=(0.45, -0.05, 0.06)
            ))
        
        dna.movement_type = MovementType.CRAWL
        dna.ai_behavior = rng.choice([AIBehavior.WANDER, AIBehavior.FLEE])
        dna.base_scale = size_mult
        dna.primary_color = color
        dna.movement_speed = 0.8 + rng.random() * 0.8
        dna.animation_speed = 1.5 + rng.random() * 1.0
        
        return dna
    
    @classmethod
    def _create_dinosaur(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create dinosaur DNA - big bipeds or quadrupeds."""
        # Greens, browns, grays
        hue = rng.choice([0.1, 0.15, 0.25, 0.3])
        color = _hsv_to_rgb(hue, 0.35 + rng.random() * 0.3, 0.3 + rng.random() * 0.35)
        accent = _hsv_to_rgb((hue + 0.1) % 1, 0.5, 0.5)
        
        # Always big!
        size_mult = 2.0 + rng.random() * 3.0  # 2x to 5x scale
        
        # Biped (T-rex style) or quadruped (brontosaurus style)
        is_biped = rng.random() < 0.4
        
        dna = cls()
        
        if is_biped:
            # T-rex style - big head, small arms, powerful legs
            dna.body_segments = [
                BodySegmentGene(shape="ellipsoid", size=(0.5 * size_mult, 0.35 * size_mult, 0.3 * size_mult), color=color),  # Head
                BodySegmentGene(shape="ellipsoid", size=(0.9 * size_mult, 0.6 * size_mult, 0.5 * size_mult), color=color),  # Body
                BodySegmentGene(shape="cone", size=(0.8 * size_mult, 0.3 * size_mult, 0.25 * size_mult), color=color),  # Tail
            ]
            
            dna.limbs = [
                # Tiny arms
                LimbGene(limb_type="arm", segment_count=2,
                        segment_lengths=[0.15 * size_mult, 0.1 * size_mult],
                        segment_widths=[0.06 * size_mult, 0.04 * size_mult],
                        color=color),
                # Powerful legs
                LimbGene(limb_type="leg", segment_count=2,
                        segment_lengths=[0.5 * size_mult, 0.4 * size_mult],
                        segment_widths=[0.15 * size_mult, 0.1 * size_mult],
                        color=color, joint_range=50),
            ]
            dna.limb_pairs = 1
        else:
            # Brontosaurus style - long neck, big body, 4 legs
            neck_length = 0.6 + rng.random() * 0.4
            dna.body_segments = [
                BodySegmentGene(shape="ellipsoid", size=(0.25 * size_mult, 0.2 * size_mult, 0.2 * size_mult), color=color),  # Head
                BodySegmentGene(shape="cylinder", size=(neck_length * size_mult, 0.15 * size_mult, 0.15 * size_mult), color=color),  # Neck
                BodySegmentGene(shape="ellipsoid", size=(1.2 * size_mult, 0.7 * size_mult, 0.6 * size_mult), color=color),  # Body
                BodySegmentGene(shape="cone", size=(0.7 * size_mult, 0.2 * size_mult, 0.2 * size_mult), color=color),  # Tail
            ]
            
            dna.limbs = [
                LimbGene(limb_type="leg", segment_count=2,
                        segment_lengths=[0.45 * size_mult, 0.35 * size_mult],
                        segment_widths=[0.12 * size_mult, 0.1 * size_mult],
                        color=color, joint_range=30),
            ]
            dna.limb_pairs = 2
        
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=0.06 * size_mult, 
                       color=(0.8, 0.6, 0.1), position=(0.35, 0.12, 0.12)),
            FeatureGene(feature_type="mouth", count=1, size=0.15 * size_mult,
                       color=(0.4, 0.2, 0.2), position=(0.45, -0.05, 0.0)),
        ]
        
        # Spines/plates on back
        if rng.random() < 0.5:
            dna.features.append(FeatureGene(
                feature_type="spike", count=4 + rng.integers(0, 6),
                size=0.12 * size_mult, color=accent, position=(0.0, 0.35, 0.0)
            ))
        
        dna.movement_type = MovementType.WALK
        dna.ai_behavior = rng.choice([AIBehavior.GRAZE, AIBehavior.WANDER])
        dna.base_scale = size_mult
        dna.primary_color = color
        dna.secondary_color = accent
        dna.movement_speed = 0.4 + rng.random() * 0.4  # Slow lumbering
        dna.animation_speed = 0.6 + rng.random() * 0.4
        
        return dna
    
    @classmethod
    def _create_croc(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create crocodile/alligator DNA - long body, powerful jaws."""
        # Dark greens and browns
        hue = 0.2 + rng.random() * 0.15
        color = _hsv_to_rgb(hue, 0.35 + rng.random() * 0.25, 0.2 + rng.random() * 0.25)
        belly_color = _hsv_to_rgb(0.1, 0.2, 0.5)  # Lighter belly
        
        # Medium to large
        size_mult = 1.0 + rng.random() * 2.0
        
        # Long body with segments
        body_length = 0.9 + rng.random() * 0.4
        
        dna = cls()
        dna.body_segments = [
            # Long snout
            BodySegmentGene(shape="box", size=(0.4 * size_mult, 0.1 * size_mult, 0.12 * size_mult), color=color),
            # Head
            BodySegmentGene(shape="ellipsoid", size=(0.25 * size_mult, 0.15 * size_mult, 0.2 * size_mult), color=color),
            # Body
            BodySegmentGene(shape="ellipsoid", size=(body_length * size_mult, 0.2 * size_mult, 0.35 * size_mult), color=color),
            # Tail - long and powerful
            BodySegmentGene(shape="cone", size=(0.7 * size_mult, 0.15 * size_mult, 0.18 * size_mult), color=color),
        ]
        
        # Short but strong legs
        dna.limbs = [
            LimbGene(limb_type="leg", segment_count=2,
                    segment_lengths=[0.12 * size_mult, 0.1 * size_mult],
                    segment_widths=[0.06 * size_mult, 0.05 * size_mult],
                    color=color, joint_range=35),
        ]
        dna.limb_pairs = 2
        
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=0.04 * size_mult,
                       color=(0.9, 0.7, 0.1), position=(0.3, 0.1, 0.1)),
            # Teeth visible
            FeatureGene(feature_type="spike", count=6 + rng.integers(0, 4),
                       size=0.03 * size_mult, color=(0.9, 0.9, 0.85), position=(0.5, -0.03, 0.05)),
        ]
        
        # Scutes/ridges on back
        dna.features.append(FeatureGene(
            feature_type="spike", count=8 + rng.integers(0, 6),
            size=0.04 * size_mult, color=color, position=(0.0, 0.12, 0.0)
        ))
        
        dna.movement_type = MovementType.CRAWL
        dna.ai_behavior = rng.choice([AIBehavior.GRAZE, AIBehavior.WANDER])
        dna.base_scale = size_mult
        dna.primary_color = color
        dna.secondary_color = belly_color
        dna.movement_speed = 0.3 + rng.random() * 0.4
        dna.animation_speed = 0.8 + rng.random() * 0.4
        
        return dna
    
    @classmethod
    def _create_hopper(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create hopping creature DNA - rabbits, frogs, kangaroos."""
        # Variety of colors
        hopper_type = rng.choice(["rabbit", "frog", "kangaroo"])
        
        if hopper_type == "rabbit":
            hue = 0.08 + rng.random() * 0.08
            color = _hsv_to_rgb(hue, 0.25 + rng.random() * 0.2, 0.5 + rng.random() * 0.3)
            size_mult = 0.5 + rng.random() * 0.5
        elif hopper_type == "frog":
            hue = 0.25 + rng.random() * 0.15  # Greens
            if rng.random() < 0.3:  # Poison dart frog
                hue = rng.random()
                color = _hsv_to_rgb(hue, 0.8, 0.7)
            else:
                color = _hsv_to_rgb(hue, 0.5 + rng.random() * 0.3, 0.4 + rng.random() * 0.3)
            size_mult = 0.3 + rng.random() * 0.5
        else:  # kangaroo
            hue = 0.06 + rng.random() * 0.06
            color = _hsv_to_rgb(hue, 0.35 + rng.random() * 0.2, 0.4 + rng.random() * 0.3)
            size_mult = 1.5 + rng.random() * 1.5  # Big!
        
        dna = cls()
        
        if hopper_type == "frog":
            # Squat body, big eyes
            dna.body_segments = [
                BodySegmentGene(shape="ellipsoid", size=(0.35 * size_mult, 0.25 * size_mult, 0.35 * size_mult), color=color),
            ]
            dna.limbs = [
                # Front legs (short)
                LimbGene(limb_type="leg", segment_count=2,
                        segment_lengths=[0.1 * size_mult, 0.08 * size_mult],
                        segment_widths=[0.04 * size_mult, 0.03 * size_mult],
                        color=color),
                # Back legs (long, powerful)
                LimbGene(limb_type="leg", segment_count=2,
                        segment_lengths=[0.25 * size_mult, 0.2 * size_mult],
                        segment_widths=[0.06 * size_mult, 0.04 * size_mult],
                        color=color, joint_range=60),
            ]
            dna.limb_pairs = 1
            dna.features = [
                FeatureGene(feature_type="eye", count=2, size=0.08 * size_mult,
                           color=(0.1, 0.1, 0.1), glow=False, position=(0.25, 0.15, 0.15)),
            ]
        elif hopper_type == "rabbit":
            # Fluffy body, long ears
            dna.body_segments = [
                BodySegmentGene(shape="ellipsoid", size=(0.2 * size_mult, 0.15 * size_mult, 0.15 * size_mult), color=color),
                BodySegmentGene(shape="ellipsoid", size=(0.35 * size_mult, 0.25 * size_mult, 0.25 * size_mult), color=color),
            ]
            dna.limbs = [
                LimbGene(limb_type="leg", segment_count=2,
                        segment_lengths=[0.12 * size_mult, 0.1 * size_mult],
                        segment_widths=[0.04 * size_mult, 0.03 * size_mult],
                        color=color, joint_range=45),
            ]
            dna.limb_pairs = 2
            dna.features = [
                FeatureGene(feature_type="eye", count=2, size=0.04 * size_mult,
                           color=(0.1, 0.05, 0.0), position=(0.35, 0.08, 0.08)),
                # Long ears
                FeatureGene(feature_type="antenna", count=2, size=0.2 * size_mult,
                           color=color, position=(0.1, 0.2, 0.06)),
                # Fluffy tail
                FeatureGene(feature_type="tail", count=1, size=0.08 * size_mult,
                           color=(1.0, 1.0, 1.0), position=(-0.4, 0.1, 0.0)),
            ]
        else:  # kangaroo
            # Upright stance, big back legs, tail for balance
            dna.body_segments = [
                BodySegmentGene(shape="ellipsoid", size=(0.25 * size_mult, 0.2 * size_mult, 0.2 * size_mult), color=color),
                BodySegmentGene(shape="ellipsoid", size=(0.5 * size_mult, 0.6 * size_mult, 0.4 * size_mult), color=color),
            ]
            dna.limbs = [
                # Small front arms
                LimbGene(limb_type="arm", segment_count=2,
                        segment_lengths=[0.12 * size_mult, 0.1 * size_mult],
                        segment_widths=[0.04 * size_mult, 0.03 * size_mult],
                        color=color),
                # Powerful back legs
                LimbGene(limb_type="leg", segment_count=2,
                        segment_lengths=[0.35 * size_mult, 0.3 * size_mult],
                        segment_widths=[0.1 * size_mult, 0.07 * size_mult],
                        color=color, joint_range=55),
            ]
            dna.limb_pairs = 1
            dna.features = [
                FeatureGene(feature_type="eye", count=2, size=0.04 * size_mult,
                           color=(0.1, 0.05, 0.0), position=(0.35, 0.1, 0.1)),
                FeatureGene(feature_type="antenna", count=2, size=0.12 * size_mult,
                           color=color, position=(0.15, 0.2, 0.07)),
                # Big tail
                FeatureGene(feature_type="tail", count=1, size=0.5 * size_mult,
                           color=color, position=(-0.4, -0.1, 0.0)),
            ]
        
        dna.movement_type = MovementType.HOP
        dna.ai_behavior = rng.choice([AIBehavior.GRAZE, AIBehavior.FLEE, AIBehavior.WANDER])
        dna.base_scale = size_mult
        dna.primary_color = color
        dna.movement_speed = 1.0 + rng.random() * 1.0
        dna.animation_speed = 1.5 + rng.random() * 1.0
        
        return dna


def _hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Convert HSV to RGB (all values 0-1)."""
    h = h * 6.0
    i = int(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    
    if i == 0: return (v, t, p)
    elif i == 1: return (q, v, p)
    elif i == 2: return (p, v, t)
    elif i == 3: return (p, q, v)
    elif i == 4: return (t, p, v)
    else: return (v, p, q)

