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
    
    ALL = [INSECT, BIRD, FISH, MAMMAL, REPTILE, AMPHIBIAN, JELLYFISH, WORM, ALIEN]


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
        
        # Mutate colors
        def mutate_color(c):
            return tuple(float(np.clip(v + rng.normal(0, rate), 0, 1)) for v in c)
        
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
        else:
            dna = cls._create_mammal(rng)
        
        dna.animal_type = animal_type
        dna.species_id = seed or rng.integers(0, 100000)
        return dna
    
    @classmethod
    def _create_insect(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create insect DNA."""
        # Random insect color
        hue = rng.random()
        color = _hsv_to_rgb(hue, 0.5 + rng.random() * 0.4, 0.3 + rng.random() * 0.4)
        
        dna = cls()
        dna.body_segments = [
            BodySegmentGene(shape="ellipsoid", size=(0.3, 0.2, 0.15), color=color),  # Head
            BodySegmentGene(shape="ellipsoid", size=(0.4, 0.25, 0.2), color=color),  # Thorax
            BodySegmentGene(shape="ellipsoid", size=(0.5, 0.3, 0.25), color=color),  # Abdomen
        ]
        
        dna.limbs = [
            LimbGene(limb_type="leg", segment_count=3, 
                    segment_lengths=[0.2, 0.25, 0.15], segment_widths=[0.03, 0.02, 0.01],
                    color=(0.2, 0.2, 0.2)),
        ]
        dna.limb_pairs = 3  # 6 legs
        
        # Maybe add wings
        if rng.random() < 0.6:
            dna.limbs.append(LimbGene(
                limb_type="wing", segment_count=1,
                segment_lengths=[0.6], segment_widths=[0.02],
                membrane=True, membrane_color=(0.8, 0.8, 0.9, 0.5)
            ))
        
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=0.08, color=(0.1, 0.0, 0.0), position=(0.35, 0.15, 0.1)),
            FeatureGene(feature_type="antenna", count=2, size=0.15, color=(0.2, 0.2, 0.2), position=(0.4, 0.2, 0.05)),
        ]
        
        dna.movement_type = MovementType.FLY if dna.limbs[-1].limb_type == "wing" else MovementType.CRAWL
        dna.ai_behavior = AIBehavior.SWARM if rng.random() < 0.5 else AIBehavior.WANDER
        dna.base_scale = 0.1 + rng.random() * 0.3
        dna.primary_color = color
        dna.movement_speed = 1.0 + rng.random() * 1.5
        dna.animation_speed = 2.0 + rng.random() * 2.0
        
        return dna
    
    @classmethod
    def _create_bird(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create bird DNA."""
        hue = rng.random()
        body_color = _hsv_to_rgb(hue, 0.4 + rng.random() * 0.4, 0.4 + rng.random() * 0.4)
        wing_color = _hsv_to_rgb((hue + 0.1) % 1, 0.5, 0.5)
        
        dna = cls()
        dna.body_segments = [
            BodySegmentGene(shape="ellipsoid", size=(0.25, 0.2, 0.2), color=body_color),  # Head
            BodySegmentGene(shape="ellipsoid", size=(0.6, 0.4, 0.35), color=body_color),  # Body
        ]
        
        dna.limbs = [
            # Wings
            LimbGene(limb_type="wing", segment_count=2,
                    segment_lengths=[0.5, 0.4], segment_widths=[0.08, 0.05],
                    color=wing_color, membrane=True, membrane_color=wing_color),
            # Legs
            LimbGene(limb_type="leg", segment_count=2,
                    segment_lengths=[0.2, 0.15], segment_widths=[0.03, 0.02],
                    color=(0.6, 0.5, 0.2)),
        ]
        dna.limb_pairs = 1  # 2 wings, 2 legs
        
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=0.05, color=(0.05, 0.05, 0.05), position=(0.35, 0.1, 0.12)),
            FeatureGene(feature_type="mouth", count=1, size=0.12, color=(0.7, 0.5, 0.1), position=(0.45, 0.0, 0.0)),  # Beak
        ]
        
        dna.movement_type = MovementType.FLY
        dna.ai_behavior = AIBehavior.FLOCK
        dna.base_scale = 0.3 + rng.random() * 0.5
        dna.primary_color = body_color
        dna.secondary_color = wing_color
        dna.movement_speed = 1.5 + rng.random() * 1.5
        dna.group_tendency = 0.6 + rng.random() * 0.3
        
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
        """Create mammal DNA."""
        # Earth tones
        hue = 0.05 + rng.random() * 0.1  # Browns/tans
        color = _hsv_to_rgb(hue, 0.3 + rng.random() * 0.4, 0.3 + rng.random() * 0.5)
        
        dna = cls()
        dna.body_segments = [
            BodySegmentGene(shape="ellipsoid", size=(0.3, 0.25, 0.22), color=color),  # Head
            BodySegmentGene(shape="ellipsoid", size=(0.8, 0.45, 0.4), color=color),   # Body
        ]
        
        dna.limbs = [
            LimbGene(limb_type="leg", segment_count=2,
                    segment_lengths=[0.25, 0.2], segment_widths=[0.08, 0.06],
                    color=color, joint_range=40),
        ]
        dna.limb_pairs = 2  # 4 legs
        
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=0.04, color=(0.1, 0.05, 0.0), position=(0.3, 0.1, 0.1)),
            FeatureGene(feature_type="mouth", count=1, size=0.06, color=(0.3, 0.15, 0.15), position=(0.4, -0.08, 0.0)),
            FeatureGene(feature_type="tail", count=1, size=0.3, color=color, position=(-0.45, 0.1, 0.0)),
        ]
        
        # Maybe add ears
        if rng.random() < 0.7:
            dna.features.append(FeatureGene(
                feature_type="antenna", count=2, size=0.1, color=color, position=(0.2, 0.2, 0.08)
            ))
        
        dna.movement_type = MovementType.WALK if rng.random() > 0.3 else MovementType.HOP
        dna.ai_behavior = rng.choice([AIBehavior.GRAZE, AIBehavior.WANDER, AIBehavior.FLEE])
        dna.base_scale = 0.3 + rng.random() * 1.0
        dna.primary_color = color
        dna.movement_speed = 0.8 + rng.random() * 1.2
        
        return dna
    
    @classmethod
    def _create_reptile(cls, rng: np.random.Generator) -> "AnimalDNA":
        """Create reptile DNA."""
        hue = 0.2 + rng.random() * 0.2  # Greens/browns
        color = _hsv_to_rgb(hue, 0.4 + rng.random() * 0.4, 0.3 + rng.random() * 0.4)
        
        dna = cls()
        dna.body_segments = [
            BodySegmentGene(shape="ellipsoid", size=(0.2, 0.12, 0.15), color=color),  # Head
            BodySegmentGene(shape="ellipsoid", size=(0.6, 0.2, 0.25), color=color),   # Body
            BodySegmentGene(shape="cone", size=(0.5, 0.1, 0.1), color=color),         # Tail
        ]
        
        dna.limbs = [
            LimbGene(limb_type="leg", segment_count=2,
                    segment_lengths=[0.15, 0.12], segment_widths=[0.04, 0.03],
                    color=color),
        ]
        dna.limb_pairs = 2
        
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=0.035, color=(0.8, 0.6, 0.0), position=(0.35, 0.05, 0.08)),
        ]
        
        dna.movement_type = MovementType.CRAWL
        dna.ai_behavior = AIBehavior.WANDER
        dna.base_scale = 0.2 + rng.random() * 0.6
        dna.primary_color = color
        dna.movement_speed = 0.5 + rng.random() * 0.8
        
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
        """Create worm/snake DNA."""
        hue = 0.0 + rng.random() * 0.15  # Pinks/reds/browns
        color = _hsv_to_rgb(hue, 0.3 + rng.random() * 0.3, 0.4 + rng.random() * 0.4)
        
        segment_count = 4 + rng.integers(0, 5)
        
        dna = cls()
        dna.body_segments = [
            BodySegmentGene(
                shape="ellipsoid", 
                size=(0.15 + 0.05 * (segment_count - i) / segment_count, 0.1, 0.1), 
                color=color
            )
            for i in range(segment_count)
        ]
        
        dna.limbs = []  # No limbs
        dna.limb_pairs = 0
        
        dna.features = [
            FeatureGene(feature_type="eye", count=2, size=0.02, color=(0.05, 0.05, 0.05), position=(0.4, 0.03, 0.04)),
        ]
        
        dna.movement_type = MovementType.CRAWL
        dna.ai_behavior = AIBehavior.WANDER
        dna.base_scale = 0.3 + rng.random() * 0.5
        dna.primary_color = color
        dna.movement_speed = 0.3 + rng.random() * 0.5
        
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

