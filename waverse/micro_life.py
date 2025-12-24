"""
Microscopic Life Simulation

DNA-driven microorganisms with:
- Cell membranes that wrap organelles
- Evolvable organelles (nucleus, mitochondria, ribosomes, etc.)
- Math-based movement patterns
- Eating/interaction behaviors
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum, auto
import random


# =============================================================================
# MICRO DNA SYSTEM
# =============================================================================

class OrganelleType(Enum):
    """Types of organelles that can exist in cells."""
    NUCLEUS = auto()          # DNA storage, controls cell
    MITOCHONDRIA = auto()     # Energy production
    RIBOSOME = auto()         # Protein synthesis
    CHLOROPLAST = auto()      # Photosynthesis (plant-like)
    VACUOLE = auto()          # Storage/waste
    FLAGELLUM = auto()        # Movement appendage
    CILIA = auto()            # Hair-like movement
    PSEUDOPOD = auto()        # Blob-like extension for movement/eating
    GOLGI = auto()            # Protein packaging
    ER = auto()               # Endoplasmic reticulum
    LYSOSOME = auto()         # Digestion
    CELL_WALL = auto()        # Rigid outer layer (bacteria/plants)


class MovementPattern(Enum):
    """Movement behavior patterns."""
    DRIFT = auto()            # Random slow drift
    SPIRAL = auto()           # Spiral motion
    WIGGLE = auto()           # Side-to-side wiggle
    PULSE = auto()            # Pulsing/contracting movement
    CHASE = auto()            # Move toward food/prey
    FLEE = auto()             # Move away from predators
    STATIC = auto()           # Stationary


@dataclass
class OrganelleGene:
    """DNA encoding for a single organelle."""
    organelle_type: OrganelleType
    count: int = 1                    # Number of this organelle (1-20)
    size: float = 0.1                 # Relative size (0.05-0.3)
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    efficiency: float = 1.0           # How well it performs its function
    position_bias: float = 0.5        # 0=center, 1=edge of cell


@dataclass 
class MembranGene:
    """DNA encoding for the cell membrane."""
    shape: str = "circle"             # circle, oval, blob, amoeba
    thickness: float = 0.02           # Membrane thickness
    color: Tuple[float, float, float] = (0.8, 0.9, 0.8)
    transparency: float = 0.3         # 0=opaque, 1=transparent
    flexibility: float = 0.5          # How much the shape can deform
    has_wall: bool = False            # Rigid cell wall (bacteria/plants)
    wall_color: Tuple[float, float, float] = (0.4, 0.6, 0.4)


@dataclass
class MovementGene:
    """DNA encoding for movement behavior."""
    pattern: MovementPattern = MovementPattern.DRIFT
    speed: float = 0.5                # Base movement speed
    frequency: float = 1.0            # Oscillation frequency for patterns
    amplitude: float = 0.3            # Movement amplitude
    turn_rate: float = 0.1            # How fast it can turn


@dataclass
class MetabolismGene:
    """DNA encoding for metabolism and eating."""
    energy_capacity: float = 100.0    # Max energy storage
    energy_consumption: float = 1.0   # Energy used per second
    food_types: List[str] = field(default_factory=lambda: ["organic"])
    is_predator: bool = False         # Can eat other cells
    is_photosynthetic: bool = False   # Can use light for energy
    division_threshold: float = 80.0  # Energy level to divide


@dataclass
class MicroDNA:
    """Complete DNA for a microorganism."""
    # Identity
    species_id: int = 0
    generation: int = 0
    
    # Cell structure
    base_size: float = 1.0            # Base cell radius
    membrane: MembranGene = field(default_factory=MembranGene)
    organelles: List[OrganelleGene] = field(default_factory=list)
    
    # Behavior
    movement: MovementGene = field(default_factory=MovementGene)
    metabolism: MetabolismGene = field(default_factory=MetabolismGene)
    
    # Aesthetic
    glow: float = 0.0                 # Bioluminescence (0-1)
    glow_color: Tuple[float, float, float] = (0.5, 1.0, 0.8)
    
    @classmethod
    def create_random(cls, seed: int = None) -> 'MicroDNA':
        """Generate a random microorganism DNA."""
        rng = random.Random(seed)
        
        # Random membrane
        membrane = MembranGene(
            shape=rng.choice(["circle", "oval", "blob", "amoeba"]),
            thickness=0.01 + rng.random() * 0.03,
            color=(0.6 + rng.random() * 0.4, 
                   0.7 + rng.random() * 0.3, 
                   0.6 + rng.random() * 0.4),
            transparency=0.2 + rng.random() * 0.5,
            flexibility=rng.random(),
            has_wall=rng.random() < 0.2,
        )
        
        # Random organelles
        organelles = []
        
        # Always have a nucleus
        organelles.append(OrganelleGene(
            organelle_type=OrganelleType.NUCLEUS,
            count=1,
            size=0.15 + rng.random() * 0.1,
            color=(0.3, 0.3, 0.6),
            position_bias=0.2,
        ))
        
        # Random mitochondria
        if rng.random() < 0.8:
            organelles.append(OrganelleGene(
                organelle_type=OrganelleType.MITOCHONDRIA,
                count=rng.randint(2, 8),
                size=0.05 + rng.random() * 0.05,
                color=(0.9, 0.4, 0.3),
                position_bias=0.6,
            ))
        
        # Random other organelles
        for org_type in [OrganelleType.RIBOSOME, OrganelleType.VACUOLE, 
                         OrganelleType.GOLGI, OrganelleType.ER]:
            if rng.random() < 0.4:
                organelles.append(OrganelleGene(
                    organelle_type=org_type,
                    count=rng.randint(1, 5),
                    size=0.03 + rng.random() * 0.07,
                    color=(rng.random(), rng.random(), rng.random()),
                    position_bias=0.3 + rng.random() * 0.5,
                ))
        
        # Movement appendages
        if rng.random() < 0.3:
            organelles.append(OrganelleGene(
                organelle_type=rng.choice([OrganelleType.FLAGELLUM, 
                                           OrganelleType.CILIA, 
                                           OrganelleType.PSEUDOPOD]),
                count=rng.randint(1, 4),
                size=0.1 + rng.random() * 0.2,
                color=membrane.color,
                position_bias=1.0,  # At edge
            ))
        
        # Photosynthetic?
        is_photo = rng.random() < 0.2
        if is_photo:
            organelles.append(OrganelleGene(
                organelle_type=OrganelleType.CHLOROPLAST,
                count=rng.randint(3, 12),
                size=0.06 + rng.random() * 0.04,
                color=(0.2, 0.7, 0.3),
                position_bias=0.5,
            ))
        
        # Movement
        movement = MovementGene(
            pattern=rng.choice(list(MovementPattern)),
            speed=0.2 + rng.random() * 0.8,
            frequency=0.5 + rng.random() * 2.0,
            amplitude=0.1 + rng.random() * 0.4,
            turn_rate=0.05 + rng.random() * 0.2,
        )
        
        # Metabolism
        metabolism = MetabolismGene(
            energy_capacity=50 + rng.random() * 100,
            energy_consumption=0.5 + rng.random() * 1.5,
            is_predator=rng.random() < 0.15,
            is_photosynthetic=is_photo,
            division_threshold=60 + rng.random() * 40,
        )
        
        return cls(
            species_id=rng.randint(0, 999999),
            base_size=0.5 + rng.random() * 1.5,
            membrane=membrane,
            organelles=organelles,
            movement=movement,
            metabolism=metabolism,
            glow=rng.random() * 0.3 if rng.random() < 0.1 else 0.0,
            glow_color=(rng.random(), rng.random(), rng.random()),
        )
    
    def mutate(self, mutation_rate: float = 0.1) -> 'MicroDNA':
        """Create a mutated copy of this DNA."""
        rng = random.Random()
        
        # Copy self
        new_dna = MicroDNA(
            species_id=self.species_id,
            generation=self.generation + 1,
            base_size=self.base_size,
            membrane=MembranGene(**vars(self.membrane)),
            organelles=[OrganelleGene(**vars(o)) for o in self.organelles],
            movement=MovementGene(**vars(self.movement)),
            metabolism=MetabolismGene(
                energy_capacity=self.metabolism.energy_capacity,
                energy_consumption=self.metabolism.energy_consumption,
                food_types=list(self.metabolism.food_types),
                is_predator=self.metabolism.is_predator,
                is_photosynthetic=self.metabolism.is_photosynthetic,
                division_threshold=self.metabolism.division_threshold,
            ),
            glow=self.glow,
            glow_color=self.glow_color,
        )
        
        # Apply mutations
        if rng.random() < mutation_rate:
            new_dna.base_size *= (0.9 + rng.random() * 0.2)
        
        if rng.random() < mutation_rate:
            new_dna.membrane.flexibility *= (0.8 + rng.random() * 0.4)
        
        if rng.random() < mutation_rate:
            new_dna.movement.speed *= (0.8 + rng.random() * 0.4)
        
        # Mutate organelle counts
        for org in new_dna.organelles:
            if rng.random() < mutation_rate:
                org.count = max(1, org.count + rng.randint(-1, 1))
            if rng.random() < mutation_rate:
                org.size *= (0.9 + rng.random() * 0.2)
        
        return new_dna


# =============================================================================
# MICROORGANISM ENTITY
# =============================================================================

class Microorganism:
    """A single microorganism in the simulation."""
    
    def __init__(self, x: float, y: float, dna: MicroDNA):
        self.x = x
        self.y = y
        self.dna = dna
        
        # State
        self.energy = dna.metabolism.energy_capacity * 0.7
        self.age = 0.0
        self.angle = random.random() * 2 * math.pi
        self.phase = random.random() * 2 * math.pi  # For oscillation
        
        # Computed positions for organelles (generated once, updated on deform)
        self.organelle_positions: List[Tuple[float, float]] = []
        self._generate_organelle_positions()
        
        # Visual state
        self.membrane_points: List[Tuple[float, float]] = []
        self._generate_membrane()
    
    def _generate_organelle_positions(self):
        """Generate positions for all organelles inside the cell."""
        self.organelle_positions = []
        rng = random.Random(self.dna.species_id)
        
        for org in self.dna.organelles:
            for i in range(org.count):
                # Position based on bias (0=center, 1=edge)
                r = org.position_bias * self.dna.base_size * 0.8
                r *= (0.5 + rng.random() * 0.5)  # Some variation
                theta = rng.random() * 2 * math.pi
                
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                self.organelle_positions.append((x, y, org))
    
    def _generate_membrane(self):
        """Generate membrane boundary points."""
        num_points = 32
        self.membrane_points = []
        
        shape = self.dna.membrane.shape
        size = self.dna.base_size
        
        for i in range(num_points):
            theta = (i / num_points) * 2 * math.pi
            
            if shape == "circle":
                r = size
            elif shape == "oval":
                r = size * (1.0 + 0.3 * math.cos(2 * theta))
            elif shape == "blob":
                # Perlin-like bumpy shape
                r = size * (1.0 + 0.2 * math.sin(3 * theta) + 0.1 * math.sin(5 * theta))
            elif shape == "amoeba":
                # Very irregular
                r = size * (1.0 + 0.3 * math.sin(2 * theta + self.phase) 
                           + 0.2 * math.sin(4 * theta - self.phase * 0.5)
                           + 0.1 * math.sin(7 * theta + self.phase * 2))
            else:
                r = size
            
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            self.membrane_points.append((x, y))
    
    def update(self, dt: float, world_width: float, world_height: float):
        """Update microorganism state."""
        self.age += dt
        self.phase += dt * self.dna.movement.frequency
        
        # Consume energy
        self.energy -= self.dna.metabolism.energy_consumption * dt
        
        # Photosynthesis - gain energy from "light"
        if self.dna.metabolism.is_photosynthetic:
            self.energy += 0.5 * dt  # Simplified light energy
        
        # Movement based on pattern
        self._apply_movement(dt)
        
        # Wrap around world
        self.x = self.x % world_width
        self.y = self.y % world_height
        
        # Update membrane for amoeba shapes
        if self.dna.membrane.shape == "amoeba":
            self._generate_membrane()
    
    def _apply_movement(self, dt: float):
        """Apply movement based on DNA pattern."""
        pattern = self.dna.movement.pattern
        speed = self.dna.movement.speed
        freq = self.dna.movement.frequency
        amp = self.dna.movement.amplitude
        
        if pattern == MovementPattern.DRIFT:
            # Slow random drift with occasional direction changes
            self.angle += (random.random() - 0.5) * self.dna.movement.turn_rate
            self.x += math.cos(self.angle) * speed * dt * 0.3
            self.y += math.sin(self.angle) * speed * dt * 0.3
        
        elif pattern == MovementPattern.SPIRAL:
            # Spiraling motion
            self.angle += freq * dt
            self.x += math.cos(self.angle) * speed * dt
            self.y += math.sin(self.angle) * speed * dt
        
        elif pattern == MovementPattern.WIGGLE:
            # Side-to-side wiggle while moving forward
            wiggle = math.sin(self.phase * freq) * amp
            self.x += (math.cos(self.angle) + wiggle * math.cos(self.angle + math.pi/2)) * speed * dt
            self.y += (math.sin(self.angle) + wiggle * math.sin(self.angle + math.pi/2)) * speed * dt
        
        elif pattern == MovementPattern.PULSE:
            # Pulsing forward motion
            pulse = 0.5 + 0.5 * math.sin(self.phase * freq)
            self.x += math.cos(self.angle) * speed * pulse * dt
            self.y += math.sin(self.angle) * speed * pulse * dt
        
        elif pattern == MovementPattern.STATIC:
            # No movement, just slight drift
            self.x += (random.random() - 0.5) * 0.01 * dt
            self.y += (random.random() - 0.5) * 0.01 * dt
    
    def can_divide(self) -> bool:
        """Check if cell has enough energy to divide."""
        return self.energy >= self.dna.metabolism.division_threshold
    
    def divide(self) -> 'Microorganism':
        """Create a new cell through division."""
        # Split energy
        self.energy *= 0.5
        
        # Create offspring with mutated DNA
        child_dna = self.dna.mutate(mutation_rate=0.05)
        
        # Position slightly offset
        offset_angle = random.random() * 2 * math.pi
        offset_dist = self.dna.base_size * 2
        child = Microorganism(
            self.x + math.cos(offset_angle) * offset_dist,
            self.y + math.sin(offset_angle) * offset_dist,
            child_dna
        )
        child.energy = self.energy
        
        return child


# =============================================================================
# MICROSCOPE WORLD
# =============================================================================

class MicroscopeWorld:
    """
    The microscopic world simulation.
    Contains microorganisms and handles their interactions.
    """
    
    def __init__(self, width: float = 100.0, height: float = 100.0, seed: int = None):
        self.width = width
        self.height = height
        self.seed = seed or random.randint(0, 999999)
        self.rng = random.Random(self.seed)
        
        self.organisms: List[Microorganism] = []
        self.time = 0.0
        
        # Performance limits
        self.max_organisms = 200
        
        # Spawn initial population
        self._spawn_initial_population()
    
    def _spawn_initial_population(self, count: int = 30):
        """Spawn initial diverse population."""
        for i in range(count):
            dna = MicroDNA.create_random(seed=self.seed + i)
            x = self.rng.random() * self.width
            y = self.rng.random() * self.height
            self.organisms.append(Microorganism(x, y, dna))
    
    def clear(self):
        """Remove all organisms."""
        self.organisms.clear()
    
    def update(self, dt: float):
        """Update all organisms."""
        self.time += dt
        
        # Update each organism
        new_organisms = []
        dead = []
        
        for org in self.organisms:
            org.update(dt, self.width, self.height)
            
            # Death from starvation
            if org.energy <= 0:
                dead.append(org)
                continue
            
            # Division
            if org.can_divide() and len(self.organisms) + len(new_organisms) < self.max_organisms:
                child = org.divide()
                new_organisms.append(child)
        
        # Remove dead
        for org in dead:
            self.organisms.remove(org)
        
        # Add new
        self.organisms.extend(new_organisms)
    
    def get_organisms_in_view(self, center_x: float, center_y: float, 
                               view_radius: float) -> List[Microorganism]:
        """Get organisms visible in the current view."""
        visible = []
        for org in self.organisms:
            dx = org.x - center_x
            dy = org.y - center_y
            if dx*dx + dy*dy < view_radius * view_radius:
                visible.append(org)
        return visible
    
    def add_organism(self, x: float, y: float, dna: MicroDNA = None):
        """Add a new organism at position."""
        if dna is None:
            dna = MicroDNA.create_random()
        self.organisms.append(Microorganism(x, y, dna))


# =============================================================================
# DNA TEMPLATES FOR DIFFERENT CELL TYPES
# =============================================================================

def create_bacteria_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a simple bacteria."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.3 + rng.random() * 0.3,
        membrane=MembranGene(
            shape="oval",
            has_wall=True,
            color=(0.7, 0.8, 0.7),
            wall_color=(0.5, 0.6, 0.5),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.2, 
                         color=(0.4, 0.4, 0.6), position_bias=0.0),
            OrganelleGene(OrganelleType.RIBOSOME, count=rng.randint(5, 15), size=0.02,
                         color=(0.3, 0.3, 0.3), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=0.8 + rng.random() * 0.4,
            frequency=2.0,
        ),
    )


def create_amoeba_dna(seed: int = None) -> MicroDNA:
    """Create DNA for an amoeba."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=1.5 + rng.random() * 1.0,
        membrane=MembranGene(
            shape="amoeba",
            flexibility=0.9,
            color=(0.8, 0.85, 0.9),
            transparency=0.4,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.2,
                         color=(0.3, 0.3, 0.5), position_bias=0.1),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(5, 12), size=0.06,
                         color=(0.9, 0.4, 0.3), position_bias=0.5),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(2, 5), size=0.1,
                         color=(0.7, 0.8, 0.9), position_bias=0.4),
            OrganelleGene(OrganelleType.PSEUDOPOD, count=3, size=0.3,
                         color=(0.8, 0.85, 0.9), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.3,
        ),
        metabolism=MetabolismGene(
            is_predator=True,
        ),
    )


def create_algae_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a photosynthetic algae cell."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.8 + rng.random() * 0.5,
        membrane=MembranGene(
            shape="circle",
            color=(0.6, 0.9, 0.6),
            has_wall=True,
            wall_color=(0.4, 0.7, 0.4),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.15,
                         color=(0.3, 0.4, 0.3), position_bias=0.0),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(8, 20), size=0.07,
                         color=(0.2, 0.7, 0.3), position_bias=0.6),
            OrganelleGene(OrganelleType.VACUOLE, count=1, size=0.25,
                         color=(0.8, 0.9, 0.8), position_bias=0.3),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.1,
        ),
        metabolism=MetabolismGene(
            is_photosynthetic=True,
        ),
        glow=0.1,
        glow_color=(0.5, 1.0, 0.5),
    )


def create_paramecium_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a paramecium (ciliated protozoan)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=1.2 + rng.random() * 0.5,
        membrane=MembranGene(
            shape="oval",
            color=(0.85, 0.9, 0.85),
            transparency=0.3,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=2, size=0.12,  # Macro + micro nucleus
                         color=(0.4, 0.4, 0.6), position_bias=0.2),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(8, 15), size=0.04,
                         color=(0.9, 0.5, 0.4), position_bias=0.5),
            OrganelleGene(OrganelleType.VACUOLE, count=2, size=0.08,
                         color=(0.7, 0.8, 1.0), position_bias=0.6),
            OrganelleGene(OrganelleType.CILIA, count=30, size=0.05,
                         color=(0.8, 0.85, 0.8), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.SPIRAL,
            speed=1.0 + rng.random() * 0.5,
            frequency=1.5,
        ),
    )


# Template registry
MICRO_TEMPLATES = {
    "bacteria": create_bacteria_dna,
    "amoeba": create_amoeba_dna,
    "algae": create_algae_dna,
    "paramecium": create_paramecium_dna,
    "random": MicroDNA.create_random,
}

