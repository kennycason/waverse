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
    # Shapes: circle, oval, blob, amoeba, star, spiral, rod, comma,
    #         diamond, heart, crescent, kidney, spiky, horned, trilobed,
    #         chain, filament, branched, lobed, pennate (diatom)
    shape: str = "circle"
    thickness: float = 0.02           # Membrane thickness
    color: Tuple[float, float, float] = (0.8, 0.9, 0.8)
    transparency: float = 0.3         # 0=opaque, 1=transparent
    flexibility: float = 0.5          # How much the shape can deform
    has_wall: bool = False            # Rigid cell wall (bacteria/plants)
    wall_color: Tuple[float, float, float] = (0.4, 0.6, 0.4)
    # Extra shape parameters
    points: int = 5                   # Number of points for star/spiky
    elongation: float = 2.0           # Length/width ratio for rod/oval
    spike_length: float = 0.3         # Relative spike length for spiky


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
        num_points = 48  # More points for detailed shapes
        self.membrane_points = []
        
        shape = self.dna.membrane.shape
        size = self.dna.base_size
        mem = self.dna.membrane
        
        for i in range(num_points):
            theta = (i / num_points) * 2 * math.pi
            
            if shape == "circle":
                r = size
                
            elif shape == "oval":
                elong = getattr(mem, 'elongation', 2.0)
                r = size / math.sqrt((math.cos(theta)**2 / elong) + (math.sin(theta)**2 * elong))
                
            elif shape == "blob":
                # Perlin-like bumpy shape
                r = size * (1.0 + 0.2 * math.sin(3 * theta) + 0.1 * math.sin(5 * theta))
                
            elif shape == "amoeba":
                # Very irregular, animated
                r = size * (1.0 + 0.3 * math.sin(2 * theta + self.phase) 
                           + 0.2 * math.sin(4 * theta - self.phase * 0.5)
                           + 0.1 * math.sin(7 * theta + self.phase * 2))
                           
            elif shape == "star":
                # Star shape with n points
                points = getattr(mem, 'points', 5)
                spike = getattr(mem, 'spike_length', 0.4)
                r = size * (1.0 - spike + spike * abs(math.cos(points * theta / 2)))
                
            elif shape == "spiky":
                # Many short spikes (like a virus/corona)
                points = getattr(mem, 'points', 12)
                spike = getattr(mem, 'spike_length', 0.25)
                r = size * (1.0 + spike * (0.5 + 0.5 * math.cos(points * theta)))
                
            elif shape == "rod":
                # Elongated rod/bacillus shape
                elong = getattr(mem, 'elongation', 3.0)
                if abs(math.sin(theta)) < 0.3:
                    r = size * elong
                else:
                    r = size / abs(math.sin(theta)) if abs(math.sin(theta)) > 0.1 else size * 3
                r = min(r, size * elong)
                
            elif shape == "comma":
                # Comma/vibrio shape - curved rod
                base_r = size * (1.0 + 0.6 * math.cos(theta))
                curve = 0.3 * math.sin(theta) * math.cos(theta * 0.5)
                r = base_r * (1.0 + curve)
                
            elif shape == "spiral":
                # Spirillum/spirochete - spiral bacteria
                winds = getattr(mem, 'points', 3)
                r = size * 0.3 * (1.0 + 0.5 * math.sin(winds * theta + self.phase))
                
            elif shape == "diamond":
                # Diamond/rhombus shape
                r = size / (abs(math.cos(theta)) + abs(math.sin(theta)))
                
            elif shape == "crescent":
                # Crescent moon shape
                r = size * (0.8 + 0.5 * math.cos(theta) - 0.3 * math.cos(2 * theta))
                r = max(r, size * 0.2)
                
            elif shape == "kidney":
                # Kidney/bean shape
                r = size * (1.0 + 0.4 * math.cos(theta) - 0.2 * math.cos(2 * theta))
                
            elif shape == "heart":
                # Heart-like shape
                r = size * (1.0 - 0.5 * abs(math.sin(theta)) + 0.3 * math.cos(theta))
                
            elif shape == "horned":
                # Dinoflagellate with horns (like Ceratium)
                horns = getattr(mem, 'points', 3)
                base_r = size * 0.6
                horn_angles = [0, 2*math.pi/3, 4*math.pi/3][:horns]
                for horn_angle in horn_angles:
                    diff = abs(theta - horn_angle)
                    if diff > math.pi:
                        diff = 2*math.pi - diff
                    if diff < 0.3:
                        base_r = max(base_r, size * 1.5 * (1.0 - diff/0.3))
                r = base_r
                
            elif shape == "trilobed":
                # Three-lobed shape
                r = size * (0.6 + 0.4 * abs(math.cos(1.5 * theta)))
                
            elif shape == "lobed":
                # Multiple lobes (desmid-like)
                lobes = getattr(mem, 'points', 4)
                r = size * (0.7 + 0.3 * abs(math.cos(lobes * theta / 2)))
                
            elif shape == "pennate":
                # Pennate diatom (elongated with pointed ends)
                elong = getattr(mem, 'elongation', 4.0)
                t = (theta + math.pi) % (2 * math.pi)
                if t < math.pi:
                    r = size * elong * math.sin(t)
                else:
                    r = size * elong * math.sin(2*math.pi - t)
                r = max(r, size * 0.15)
                
            elif shape == "radiating":
                # Radiolarian with radiating spines
                spines = getattr(mem, 'points', 8)
                spike = getattr(mem, 'spike_length', 0.6)
                spine_val = abs(math.cos(spines * theta / 2))
                if spine_val > 0.9:
                    r = size * (1.0 + spike)
                else:
                    r = size * (0.7 + 0.3 * spine_val)
                    
            elif shape == "segmented":
                # Segmented worm-like (for filamentous organisms)
                segs = getattr(mem, 'points', 5)
                base = size * 2.5
                seg_wave = 0.15 * math.sin(segs * theta)
                r = base * (0.3 + 0.1 * math.cos(2 * theta)) * (1.0 + seg_wave)
                
            elif shape == "tentacled":
                # Blob with tentacle-like extensions
                tents = getattr(mem, 'points', 6)
                spike = getattr(mem, 'spike_length', 0.5)
                base = size * 0.8
                for tent_i in range(tents):
                    tent_angle = tent_i * 2 * math.pi / tents
                    diff = abs(theta - tent_angle)
                    if diff > math.pi:
                        diff = 2*math.pi - diff
                    if diff < 0.2:
                        base = max(base, size * (1.0 + spike * (1.0 - diff/0.2)))
                r = base
                
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


def create_diatom_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a diatom (silica-walled algae with geometric patterns)."""
    rng = random.Random(seed)
    
    # Diatoms come in various shapes - pennate (elongated) or centric (circular)
    is_centric = rng.random() < 0.5
    
    # Beautiful golden-brown to green colors
    colors = [
        (0.8, 0.7, 0.3),   # Golden brown
        (0.7, 0.75, 0.4),  # Olive
        (0.6, 0.8, 0.5),   # Light green
        (0.9, 0.8, 0.4),   # Yellow-brown
    ]
    base_color = rng.choice(colors)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.6 + rng.random() * 0.8,
        membrane=MembranGene(
            shape="circle" if is_centric else "oval",
            has_wall=True,
            color=base_color,
            wall_color=(0.9, 0.95, 0.85),  # Silica wall - glassy
            transparency=0.2,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.1,
                         color=(0.4, 0.3, 0.3), position_bias=0.0),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(4, 10), size=0.08,
                         color=(0.6, 0.7, 0.3), position_bias=0.5),
            # Lipid droplets (appear as vacuoles)
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(1, 3), size=0.06,
                         color=(0.95, 0.9, 0.7), position_bias=0.4),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.05,
        ),
        metabolism=MetabolismGene(
            is_photosynthetic=True,
        ),
    )


def create_dinoflagellate_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a dinoflagellate (bioluminescent, spinning motion)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.7 + rng.random() * 0.5,
        membrane=MembranGene(
            shape="blob",
            color=(0.7, 0.8, 0.9),
            transparency=0.4,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.15,
                         color=(0.3, 0.3, 0.5), position_bias=0.0),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(3, 8), size=0.06,
                         color=(0.4, 0.6, 0.4), position_bias=0.5),
            OrganelleGene(OrganelleType.FLAGELLUM, count=2, size=0.2,
                         color=(0.8, 0.85, 0.9), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.SPIRAL,
            speed=0.6 + rng.random() * 0.4,
            frequency=2.5,
        ),
        metabolism=MetabolismGene(
            is_photosynthetic=rng.random() < 0.5,
        ),
        glow=0.3 + rng.random() * 0.4,  # Bioluminescent!
        glow_color=(0.4, 0.8, 1.0),  # Blue glow
    )


def create_euglena_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a euglena (flagellated photosynthetic)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.8 + rng.random() * 0.4,
        membrane=MembranGene(
            shape="oval",
            color=(0.5, 0.8, 0.5),
            flexibility=0.7,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.12,
                         color=(0.3, 0.35, 0.4), position_bias=0.2),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(10, 20), size=0.05,
                         color=(0.2, 0.7, 0.3), position_bias=0.5),
            OrganelleGene(OrganelleType.FLAGELLUM, count=1, size=0.4,
                         color=(0.6, 0.75, 0.6), position_bias=1.0),
            # Eyespot (stigma) - rendered as small vacuole
            OrganelleGene(OrganelleType.VACUOLE, count=1, size=0.04,
                         color=(1.0, 0.3, 0.2), position_bias=0.8),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=0.7 + rng.random() * 0.3,
            frequency=1.8,
            amplitude=0.2,
        ),
        metabolism=MetabolismGene(
            is_photosynthetic=True,
        ),
    )


def create_vorticella_dna(seed: int = None) -> MicroDNA:
    """Create DNA for vorticella (stalked ciliate that contracts)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.9 + rng.random() * 0.4,
        membrane=MembranGene(
            shape="blob",
            color=(0.75, 0.85, 0.8),
            transparency=0.35,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.15,
                         color=(0.4, 0.4, 0.55), position_bias=0.3),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(5, 10), size=0.04,
                         color=(0.85, 0.45, 0.35), position_bias=0.5),
            OrganelleGene(OrganelleType.CILIA, count=20, size=0.06,
                         color=(0.8, 0.85, 0.8), position_bias=1.0),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(2, 4), size=0.08,
                         color=(0.65, 0.75, 0.9), position_bias=0.4),
        ],
        movement=MovementGene(
            pattern=MovementPattern.PULSE,  # Contracts on stalk
            speed=0.2,
            frequency=0.5,
        ),
    )


def create_rotifer_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a rotifer-like creature (wheel animalcule)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=1.8 + rng.random() * 0.8,
        membrane=MembranGene(
            shape="blob",
            color=(0.9, 0.88, 0.85),
            transparency=0.25,
            flexibility=0.6,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.12,
                         color=(0.35, 0.35, 0.5), position_bias=0.3),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(10, 18), size=0.03,
                         color=(0.9, 0.45, 0.35), position_bias=0.5),
            OrganelleGene(OrganelleType.CILIA, count=40, size=0.04,  # Corona (wheel organ)
                         color=(0.85, 0.88, 0.85), position_bias=1.0),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(3, 6), size=0.07,
                         color=(0.7, 0.78, 0.88), position_bias=0.4),
            # Internal organs as golgi
            OrganelleGene(OrganelleType.GOLGI, count=2, size=0.06,
                         color=(0.8, 0.7, 0.6), position_bias=0.6),
        ],
        movement=MovementGene(
            pattern=MovementPattern.SPIRAL,
            speed=0.8 + rng.random() * 0.4,
            frequency=2.0,
        ),
        metabolism=MetabolismGene(
            is_predator=True,  # Filter feeder
        ),
    )


def create_coccus_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a coccus (spherical bacteria)."""
    rng = random.Random(seed)
    
    colors = [
        (0.9, 0.85, 0.7),   # Staphylococcus (golden)
        (0.7, 0.8, 0.9),    # Light blue
        (0.85, 0.75, 0.85), # Lavender
    ]
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.2 + rng.random() * 0.2,
        membrane=MembranGene(
            shape="circle",
            has_wall=True,
            color=rng.choice(colors),
            wall_color=(0.6, 0.65, 0.6),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.25,
                         color=(0.4, 0.45, 0.5), position_bias=0.0),
            OrganelleGene(OrganelleType.RIBOSOME, count=rng.randint(3, 8), size=0.03,
                         color=(0.35, 0.35, 0.35), position_bias=0.4),
        ],
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.3 + rng.random() * 0.2,
        ),
    )


def create_spirillum_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a spirillum (spiral-shaped bacteria)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.25 + rng.random() * 0.15,
        membrane=MembranGene(
            shape="oval",  # Will appear spiral-ish when small
            has_wall=True,
            color=(0.75, 0.8, 0.75),
            wall_color=(0.55, 0.6, 0.55),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.3,
                         color=(0.4, 0.45, 0.5), position_bias=0.0),
            OrganelleGene(OrganelleType.FLAGELLUM, count=2, size=0.15,
                         color=(0.7, 0.75, 0.7), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.SPIRAL,
            speed=1.0 + rng.random() * 0.5,
            frequency=3.0,
        ),
    )


def create_desmid_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a desmid (ornate green algae)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=1.0 + rng.random() * 0.6,
        membrane=MembranGene(
            shape="blob",
            has_wall=True,
            color=(0.4, 0.75, 0.45),
            wall_color=(0.35, 0.6, 0.4),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.1,
                         color=(0.3, 0.4, 0.35), position_bias=0.0),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(6, 14), size=0.08,
                         color=(0.25, 0.65, 0.3), position_bias=0.5),
            OrganelleGene(OrganelleType.VACUOLE, count=2, size=0.12,
                         color=(0.7, 0.85, 0.75), position_bias=0.3),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.02,
        ),
        metabolism=MetabolismGene(
            is_photosynthetic=True,
        ),
        glow=0.05,
        glow_color=(0.5, 0.9, 0.5),
    )


def create_volvox_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a volvox (colonial green algae - large spherical colony)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=2.5 + rng.random() * 1.5,
        membrane=MembranGene(
            shape="circle",
            color=(0.5, 0.85, 0.55),
            transparency=0.5,  # Very transparent - see-through colony
        ),
        organelles=[
            # Many small cells around the edge
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(30, 60), size=0.03,
                         color=(0.3, 0.7, 0.35), position_bias=0.9),
            # Interior is mostly hollow
            OrganelleGene(OrganelleType.VACUOLE, count=1, size=0.5,
                         color=(0.6, 0.85, 0.65), position_bias=0.0),
            # Flagella around edge
            OrganelleGene(OrganelleType.FLAGELLUM, count=16, size=0.08,
                         color=(0.55, 0.8, 0.55), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.SPIRAL,
            speed=0.4 + rng.random() * 0.2,
            frequency=0.8,
        ),
        metabolism=MetabolismGene(
            is_photosynthetic=True,
        ),
        glow=0.15,
        glow_color=(0.4, 0.95, 0.5),
    )


def create_spirogyra_dna(seed: int = None) -> MicroDNA:
    """Create DNA for spirogyra (filamentous algae with spiral chloroplasts)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.5 + rng.random() * 0.3,
        membrane=MembranGene(
            shape="oval",  # Elongated cells
            has_wall=True,
            color=(0.55, 0.85, 0.55),
            wall_color=(0.45, 0.7, 0.45),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.08,
                         color=(0.35, 0.4, 0.35), position_bias=0.0),
            # Spiral ribbon-like chloroplasts
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(2, 4), size=0.2,
                         color=(0.25, 0.7, 0.3), position_bias=0.6),
            OrganelleGene(OrganelleType.VACUOLE, count=1, size=0.15,
                         color=(0.75, 0.9, 0.75), position_bias=0.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.01,
        ),
        metabolism=MetabolismGene(
            is_photosynthetic=True,
        ),
    )


def create_stentor_dna(seed: int = None) -> MicroDNA:
    """Create DNA for stentor (trumpet-shaped ciliate)."""
    rng = random.Random(seed)
    
    colors = [
        (0.3, 0.6, 0.8),   # Blue (Stentor coeruleus)
        (0.5, 0.8, 0.5),   # Green (with symbiotic algae)
        (0.8, 0.75, 0.7),  # Brown
    ]
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=2.0 + rng.random() * 1.0,
        membrane=MembranGene(
            shape="blob",
            color=rng.choice(colors),
            transparency=0.3,
            flexibility=0.8,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.2,  # Beaded nucleus
                         color=(0.4, 0.4, 0.55), position_bias=0.4),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(12, 25), size=0.03,
                         color=(0.9, 0.5, 0.4), position_bias=0.5),
            OrganelleGene(OrganelleType.CILIA, count=50, size=0.04,
                         color=(0.85, 0.88, 0.85), position_bias=1.0),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(4, 8), size=0.06,
                         color=(0.65, 0.75, 0.85), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.PULSE,  # Contracting
            speed=0.3,
            frequency=0.3,
        ),
        metabolism=MetabolismGene(
            is_predator=True,  # Filter feeder
        ),
    )


def create_tardigrade_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a tardigrade (water bear) - tough little animal."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=2.2 + rng.random() * 0.8,
        membrane=MembranGene(
            shape="blob",
            color=(0.85, 0.8, 0.75),
            transparency=0.2,
            flexibility=0.4,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.1,
                         color=(0.4, 0.35, 0.45), position_bias=0.3),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(15, 25), size=0.02,
                         color=(0.9, 0.5, 0.4), position_bias=0.5),
            # Legs (represented as pseudopods at edges)
            OrganelleGene(OrganelleType.PSEUDOPOD, count=8, size=0.15,
                         color=(0.8, 0.75, 0.7), position_bias=1.0),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(3, 6), size=0.06,
                         color=(0.7, 0.75, 0.8), position_bias=0.4),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=0.2 + rng.random() * 0.15,
            amplitude=0.15,
        ),
    )


def create_hydra_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a hydra (freshwater cnidarian with tentacles)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=3.0 + rng.random() * 1.5,
        membrane=MembranGene(
            shape="blob",
            color=(0.7, 0.85, 0.75),
            transparency=0.35,
            flexibility=0.7,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.08,
                         color=(0.35, 0.4, 0.4), position_bias=0.2),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(10, 18), size=0.025,
                         color=(0.85, 0.45, 0.4), position_bias=0.5),
            # Tentacles
            OrganelleGene(OrganelleType.FLAGELLUM, count=rng.randint(5, 8), size=0.5,
                         color=(0.75, 0.85, 0.78), position_bias=1.0),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(2, 4), size=0.1,
                         color=(0.65, 0.8, 0.7), position_bias=0.3),
        ],
        movement=MovementGene(
            pattern=MovementPattern.PULSE,
            speed=0.15,
            frequency=0.4,
        ),
        metabolism=MetabolismGene(
            is_predator=True,
        ),
    )


def create_nematode_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a nematode (roundworm) - elongated worm-like creature."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=1.8 + rng.random() * 1.0,
        membrane=MembranGene(
            shape="oval",
            color=(0.9, 0.88, 0.85),
            transparency=0.25,
            flexibility=0.8,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.06,
                         color=(0.4, 0.38, 0.45), position_bias=0.2),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(20, 35), size=0.015,
                         color=(0.88, 0.48, 0.38), position_bias=0.5),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(4, 8), size=0.04,
                         color=(0.75, 0.78, 0.85), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=0.8 + rng.random() * 0.4,
            frequency=2.5,
            amplitude=0.35,
        ),
    )


def create_copepod_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a copepod (small crustacean zooplankton)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=1.5 + rng.random() * 0.8,
        membrane=MembranGene(
            shape="oval",
            color=(0.85, 0.88, 0.9),
            transparency=0.3,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.1,
                         color=(0.35, 0.35, 0.45), position_bias=0.2),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(12, 20), size=0.02,
                         color=(0.9, 0.5, 0.4), position_bias=0.5),
            # Antennae
            OrganelleGene(OrganelleType.FLAGELLUM, count=2, size=0.6,
                         color=(0.8, 0.83, 0.85), position_bias=1.0),
            # Legs
            OrganelleGene(OrganelleType.CILIA, count=10, size=0.12,
                         color=(0.82, 0.85, 0.87), position_bias=0.9),
            # Eye
            OrganelleGene(OrganelleType.VACUOLE, count=1, size=0.08,
                         color=(0.9, 0.2, 0.2), position_bias=0.8),
        ],
        movement=MovementGene(
            pattern=MovementPattern.PULSE,
            speed=1.2 + rng.random() * 0.6,
            frequency=4.0,
        ),
    )


def create_daphnia_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a daphnia (water flea) - transparent crustacean."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=2.0 + rng.random() * 1.0,
        membrane=MembranGene(
            shape="blob",
            color=(0.88, 0.9, 0.92),
            transparency=0.55,  # Very transparent - see organs inside
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.06,
                         color=(0.4, 0.35, 0.45), position_bias=0.3),
            # Visible heart and gut
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=1, size=0.15,
                         color=(0.95, 0.3, 0.3), position_bias=0.4),  # Heart
            OrganelleGene(OrganelleType.GOLGI, count=1, size=0.2,
                         color=(0.6, 0.5, 0.4), position_bias=0.5),  # Gut
            # Large compound eye
            OrganelleGene(OrganelleType.VACUOLE, count=1, size=0.12,
                         color=(0.1, 0.1, 0.1), position_bias=0.85),
            # Swimming antennae
            OrganelleGene(OrganelleType.FLAGELLUM, count=4, size=0.4,
                         color=(0.85, 0.87, 0.9), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.PULSE,  # Jerky swimming
            speed=0.9 + rng.random() * 0.5,
            frequency=3.0,
        ),
    )


def create_radiolarian_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a radiolarian (beautiful silica skeleton protozoan)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=1.2 + rng.random() * 0.8,
        membrane=MembranGene(
            shape="circle",
            has_wall=True,
            color=(0.9, 0.92, 0.95),
            wall_color=(0.95, 0.97, 1.0),  # Glassy silica
            transparency=0.4,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.2,
                         color=(0.5, 0.45, 0.55), position_bias=0.0),
            # Radial spines (as flagella)
            OrganelleGene(OrganelleType.FLAGELLUM, count=rng.randint(12, 24), size=0.25,
                         color=(0.92, 0.94, 0.97), position_bias=1.0),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(4, 8), size=0.05,
                         color=(0.85, 0.88, 0.92), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.1,
        ),
        glow=0.15,
        glow_color=(0.8, 0.9, 1.0),
    )


def create_foraminifera_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a foraminifera (shelled amoeba-like protozoan)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.8 + rng.random() * 0.5,
        membrane=MembranGene(
            shape="circle",
            has_wall=True,
            color=(0.95, 0.9, 0.8),  # Calcium carbonate shell
            wall_color=(0.9, 0.85, 0.75),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.15,
                         color=(0.4, 0.4, 0.5), position_bias=0.0),
            # Pseudopods extending through shell pores
            OrganelleGene(OrganelleType.PSEUDOPOD, count=rng.randint(8, 16), size=0.2,
                         color=(0.85, 0.8, 0.75), position_bias=1.0),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(2, 4), size=0.06,
                         color=(0.8, 0.75, 0.7), position_bias=0.4),
        ],
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.08,
        ),
    )


def create_heliozoan_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a heliozoan (sun animalcule) - spiky radial protozoan."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=1.0 + rng.random() * 0.6,
        membrane=MembranGene(
            shape="circle",
            color=(0.92, 0.94, 0.9),
            transparency=0.35,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.25,
                         color=(0.45, 0.45, 0.55), position_bias=0.0),
            # Axopodia - stiff radiating spines
            OrganelleGene(OrganelleType.FLAGELLUM, count=rng.randint(20, 40), size=0.35,
                         color=(0.9, 0.92, 0.88), position_bias=1.0),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(5, 10), size=0.04,
                         color=(0.85, 0.88, 0.85), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.03,
        ),
        metabolism=MetabolismGene(
            is_predator=True,  # Catches prey on spines
        ),
    )


def create_chlamydomonas_dna(seed: int = None) -> MicroDNA:
    """Create DNA for chlamydomonas (green algae with two flagella)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.5 + rng.random() * 0.3,
        membrane=MembranGene(
            shape="circle",
            color=(0.5, 0.8, 0.5),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.15,
                         color=(0.35, 0.4, 0.35), position_bias=0.0),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=1, size=0.4,
                         color=(0.3, 0.7, 0.35), position_bias=0.3),
            # Two flagella
            OrganelleGene(OrganelleType.FLAGELLUM, count=2, size=0.5,
                         color=(0.55, 0.75, 0.55), position_bias=1.0),
            # Eyespot
            OrganelleGene(OrganelleType.VACUOLE, count=1, size=0.05,
                         color=(1.0, 0.4, 0.2), position_bias=0.7),
        ],
        movement=MovementGene(
            pattern=MovementPattern.SPIRAL,
            speed=0.7 + rng.random() * 0.3,
            frequency=1.5,
        ),
        metabolism=MetabolismGene(
            is_photosynthetic=True,
        ),
    )


def create_oscillatoria_dna(seed: int = None) -> MicroDNA:
    """Create DNA for oscillatoria (filamentous cyanobacteria that oscillates)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.4 + rng.random() * 0.2,
        membrane=MembranGene(
            shape="oval",
            has_wall=True,
            color=(0.3, 0.6, 0.5),
            wall_color=(0.25, 0.5, 0.45),
        ),
        organelles=[
            # No true nucleus (prokaryote)
            OrganelleGene(OrganelleType.RIBOSOME, count=rng.randint(8, 15), size=0.03,
                         color=(0.35, 0.4, 0.4), position_bias=0.5),
            # Thylakoids (photosynthetic membranes)
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(3, 6), size=0.12,
                         color=(0.25, 0.55, 0.45), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=0.3,
            frequency=1.0,
            amplitude=0.25,
        ),
        metabolism=MetabolismGene(
            is_photosynthetic=True,
        ),
    )


def create_blepharisma_dna(seed: int = None) -> MicroDNA:
    """Create DNA for blepharisma (pink/red ciliate)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=1.4 + rng.random() * 0.6,
        membrane=MembranGene(
            shape="oval",
            color=(0.95, 0.6, 0.65),  # Distinctive pink color
            transparency=0.25,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.15,
                         color=(0.5, 0.35, 0.4), position_bias=0.2),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(8, 14), size=0.03,
                         color=(0.85, 0.45, 0.5), position_bias=0.5),
            OrganelleGene(OrganelleType.CILIA, count=40, size=0.04,
                         color=(0.9, 0.65, 0.7), position_bias=1.0),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(2, 4), size=0.07,
                         color=(0.85, 0.55, 0.6), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.SPIRAL,
            speed=0.8 + rng.random() * 0.4,
            frequency=1.2,
        ),
    )


def create_colpoda_dna(seed: int = None) -> MicroDNA:
    """Create DNA for colpoda (kidney-shaped ciliate common in soil)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.6 + rng.random() * 0.3,
        membrane=MembranGene(
            shape="blob",
            color=(0.8, 0.82, 0.78),
            flexibility=0.4,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.18,
                         color=(0.4, 0.4, 0.48), position_bias=0.2),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(5, 10), size=0.03,
                         color=(0.85, 0.5, 0.45), position_bias=0.5),
            OrganelleGene(OrganelleType.CILIA, count=25, size=0.05,
                         color=(0.78, 0.8, 0.76), position_bias=1.0),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(1, 3), size=0.1,
                         color=(0.7, 0.75, 0.72), position_bias=0.4),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=0.5 + rng.random() * 0.3,
            frequency=1.8,
        ),
    )


def create_didinium_dna(seed: int = None) -> MicroDNA:
    """Create DNA for didinium (barrel-shaped predator that eats paramecia)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.7 + rng.random() * 0.3,
        membrane=MembranGene(
            shape="oval",
            color=(0.85, 0.87, 0.82),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.2,
                         color=(0.4, 0.42, 0.5), position_bias=0.3),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(6, 12), size=0.03,
                         color=(0.88, 0.5, 0.42), position_bias=0.5),
            # Two rings of cilia
            OrganelleGene(OrganelleType.CILIA, count=30, size=0.06,
                         color=(0.82, 0.84, 0.8), position_bias=0.95),
            # Proboscis for feeding
            OrganelleGene(OrganelleType.PSEUDOPOD, count=1, size=0.3,
                         color=(0.8, 0.82, 0.78), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.CHASE,
            speed=1.5 + rng.random() * 0.5,
            frequency=2.0,
        ),
        metabolism=MetabolismGene(
            is_predator=True,
        ),
    )


# =============================================================================
# NEW ORGANISMS FROM REFERENCE IMAGES
# =============================================================================

def create_coronavirus_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a coronavirus-like particle (spiky sphere)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.4 + rng.random() * 0.2,
        membrane=MembranGene(
            shape="spiky",
            points=rng.randint(12, 20),  # Many spikes
            spike_length=0.3 + rng.random() * 0.2,
            color=(0.85, 0.3, 0.3),  # Red/pink
            has_wall=True,
            wall_color=(0.6, 0.2, 0.2),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.35,
                         color=(0.95, 0.5, 0.5), position_bias=0.0),
            OrganelleGene(OrganelleType.RIBOSOME, count=rng.randint(6, 12), size=0.06,
                         color=(0.75, 0.25, 0.25), position_bias=0.4),
        ],
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.1,
        ),
    )


def create_bacteriophage_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a bacteriophage (virus with geometric head and legs)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.35 + rng.random() * 0.15,
        membrane=MembranGene(
            shape="diamond",
            color=(0.3, 0.4, 0.7),  # Blue
            has_wall=True,
            wall_color=(0.2, 0.3, 0.5),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.4,
                         color=(0.4, 0.5, 0.8), position_bias=0.0),
            OrganelleGene(OrganelleType.FLAGELLUM, count=rng.randint(4, 6), size=0.4,
                         color=(0.25, 0.35, 0.55), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.05,
        ),
    )


def create_streptococcus_dna(seed: int = None) -> MicroDNA:
    """Create DNA for streptococcus (chains of spherical bacteria)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.25 + rng.random() * 0.1,
        membrane=MembranGene(
            shape="circle",
            color=(0.9, 0.85, 0.7),  # Tan/cream
            has_wall=True,
            wall_color=(0.7, 0.65, 0.5),
        ),
        organelles=[
            OrganelleGene(OrganelleType.RIBOSOME, count=rng.randint(4, 8), size=0.04,
                         color=(0.8, 0.75, 0.6), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.08,
        ),
    )


def create_staphylococcus_dna(seed: int = None) -> MicroDNA:
    """Create DNA for staphylococcus (grape-like clusters of bacteria)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.2 + rng.random() * 0.1,
        membrane=MembranGene(
            shape="circle",
            color=(0.95, 0.9, 0.4),  # Yellow/gold
            has_wall=True,
            wall_color=(0.75, 0.7, 0.3),
        ),
        organelles=[
            OrganelleGene(OrganelleType.RIBOSOME, count=rng.randint(3, 6), size=0.05,
                         color=(0.85, 0.8, 0.35), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.02,
        ),
    )


def create_vibrio_dna(seed: int = None) -> MicroDNA:
    """Create DNA for vibrio (comma-shaped bacterium)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.5 + rng.random() * 0.2,
        membrane=MembranGene(
            shape="comma",
            color=(0.5, 0.75, 0.85),  # Light blue
            has_wall=True,
            wall_color=(0.4, 0.6, 0.7),
        ),
        organelles=[
            OrganelleGene(OrganelleType.RIBOSOME, count=rng.randint(6, 10), size=0.03,
                         color=(0.45, 0.65, 0.75), position_bias=0.5),
            OrganelleGene(OrganelleType.FLAGELLUM, count=1, size=0.6,
                         color=(0.4, 0.6, 0.7), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=0.7,
            frequency=2.0,
        ),
    )


def create_bacillus_dna(seed: int = None) -> MicroDNA:
    """Create DNA for bacillus (rod-shaped bacterium, may have spores)."""
    rng = random.Random(seed)
    has_spore = rng.random() < 0.3
    
    organelles = [
        OrganelleGene(OrganelleType.RIBOSOME, count=rng.randint(8, 15), size=0.025,
                     color=(0.75, 0.75, 0.65), position_bias=0.5),
    ]
    if has_spore:
        organelles.append(
            OrganelleGene(OrganelleType.VACUOLE, count=1, size=0.25,
                         color=(0.9, 0.85, 0.7), position_bias=0.7),
        )
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.3 + rng.random() * 0.15,
        membrane=MembranGene(
            shape="rod",
            elongation=3.0 + rng.random(),
            color=(0.85, 0.85, 0.75),
            has_wall=True,
            wall_color=(0.65, 0.65, 0.55),
        ),
        organelles=organelles,
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.1,
        ),
    )


def create_ceratium_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Ceratium (dinoflagellate with distinctive horns)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.8 + rng.random() * 0.4,
        membrane=MembranGene(
            shape="horned",
            points=3,
            color=(0.4, 0.6, 0.5),  # Brown-green
            has_wall=True,
            wall_color=(0.35, 0.5, 0.4),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.2,
                         color=(0.5, 0.55, 0.45), position_bias=0.2),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(4, 8), size=0.08,
                         color=(0.35, 0.55, 0.4), position_bias=0.5),
            OrganelleGene(OrganelleType.FLAGELLUM, count=2, size=0.3,
                         color=(0.3, 0.5, 0.4), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.SPIRAL,
            speed=0.4,
            frequency=0.8,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_peridinium_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Peridinium (armored dinoflagellate)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.6 + rng.random() * 0.3,
        membrane=MembranGene(
            shape="blob",
            color=(0.5, 0.55, 0.4),
            has_wall=True,
            wall_color=(0.4, 0.45, 0.3),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.18,
                         color=(0.45, 0.5, 0.38), position_bias=0.2),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(5, 10), size=0.06,
                         color=(0.4, 0.55, 0.35), position_bias=0.5),
            OrganelleGene(OrganelleType.FLAGELLUM, count=2, size=0.35,
                         color=(0.45, 0.5, 0.38), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.SPIRAL,
            speed=0.35,
            frequency=1.0,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_navicula_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Navicula (boat-shaped/pennate diatom)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.45 + rng.random() * 0.2,
        membrane=MembranGene(
            shape="pennate",
            elongation=4.0 + rng.random() * 2.0,
            color=(0.7, 0.75, 0.5),  # Golden-brown
            has_wall=True,
            wall_color=(0.6, 0.65, 0.4),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.12,
                         color=(0.6, 0.65, 0.45), position_bias=0.1),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(2, 4), size=0.15,
                         color=(0.55, 0.65, 0.35), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.1,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_pinnularia_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Pinnularia (elongated pennate diatom with striations)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.5 + rng.random() * 0.25,
        membrane=MembranGene(
            shape="pennate",
            elongation=5.0 + rng.random() * 2.0,
            color=(0.6, 0.7, 0.45),
            has_wall=True,
            wall_color=(0.5, 0.6, 0.35),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.1,
                         color=(0.5, 0.6, 0.4), position_bias=0.0),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=2, size=0.2,
                         color=(0.45, 0.6, 0.3), position_bias=0.4),
            OrganelleGene(OrganelleType.VACUOLE, count=2, size=0.08,
                         color=(0.65, 0.7, 0.5), position_bias=0.7),
        ],
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.12,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_asterionella_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Asterionella (star-forming colonial diatom)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.35 + rng.random() * 0.15,
        membrane=MembranGene(
            shape="star",
            points=rng.randint(6, 10),
            spike_length=0.5,
            color=(0.75, 0.8, 0.55),
            has_wall=True,
            wall_color=(0.6, 0.65, 0.4),
        ),
        organelles=[
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(1, 2), size=0.18,
                         color=(0.5, 0.65, 0.35), position_bias=0.3),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.02,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_pediastrum_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Pediastrum (green algae forming star-like colonies)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.7 + rng.random() * 0.4,
        membrane=MembranGene(
            shape="star",
            points=rng.randint(8, 16),
            spike_length=0.35,
            color=(0.4, 0.7, 0.35),  # Green
            has_wall=True,
            wall_color=(0.3, 0.55, 0.25),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.12,
                         color=(0.35, 0.55, 0.3), position_bias=0.1),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(4, 8), size=0.08,
                         color=(0.35, 0.65, 0.3), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.03,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_micrasterias_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Micrasterias (ornate lobed green algae/desmid)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.8 + rng.random() * 0.4,
        membrane=MembranGene(
            shape="lobed",
            points=rng.randint(6, 10),
            color=(0.35, 0.65, 0.4),  # Green
            has_wall=True,
            wall_color=(0.25, 0.5, 0.3),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.1,
                         color=(0.3, 0.5, 0.35), position_bias=0.0),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(6, 12), size=0.07,
                         color=(0.3, 0.6, 0.35), position_bias=0.5),
            OrganelleGene(OrganelleType.VACUOLE, count=2, size=0.1,
                         color=(0.45, 0.7, 0.5), position_bias=0.4),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.02,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_closterium_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Closterium (crescent-shaped desmid)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.6 + rng.random() * 0.3,
        membrane=MembranGene(
            shape="crescent",
            color=(0.4, 0.7, 0.45),
            has_wall=True,
            wall_color=(0.3, 0.55, 0.35),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.1,
                         color=(0.35, 0.55, 0.4), position_bias=0.0),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=2, size=0.15,
                         color=(0.35, 0.6, 0.4), position_bias=0.4),
            OrganelleGene(OrganelleType.VACUOLE, count=2, size=0.08,
                         color=(0.5, 0.7, 0.55), position_bias=0.8),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.03,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_cosmarium_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Cosmarium (pinched desmid, like two half-cells joined)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.5 + rng.random() * 0.25,
        membrane=MembranGene(
            shape="kidney",
            color=(0.45, 0.7, 0.5),
            has_wall=True,
            wall_color=(0.35, 0.55, 0.4),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.12,
                         color=(0.4, 0.55, 0.45), position_bias=0.0),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=2, size=0.12,
                         color=(0.4, 0.65, 0.45), position_bias=0.4),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.02,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_scenedesmus_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Scenedesmus (green algae in linear colonies)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.35 + rng.random() * 0.15,
        membrane=MembranGene(
            shape="oval",
            elongation=2.5,
            color=(0.5, 0.75, 0.45),
            has_wall=True,
            wall_color=(0.4, 0.6, 0.35),
        ),
        organelles=[
            OrganelleGene(OrganelleType.CHLOROPLAST, count=1, size=0.2,
                         color=(0.45, 0.7, 0.4), position_bias=0.3),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.02,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_anabaena_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Anabaena (filamentous cyanobacteria with heterocysts)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.25 + rng.random() * 0.1,
        membrane=MembranGene(
            shape="circle",
            color=(0.3, 0.55, 0.5),  # Blue-green
            has_wall=True,
            wall_color=(0.25, 0.45, 0.4),
        ),
        organelles=[
            OrganelleGene(OrganelleType.RIBOSOME, count=rng.randint(4, 8), size=0.04,
                         color=(0.25, 0.45, 0.4), position_bias=0.5),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(2, 4), size=0.1,
                         color=(0.2, 0.5, 0.45), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.05,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_nostoc_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Nostoc (colonial cyanobacteria, forms gelatinous masses)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.3 + rng.random() * 0.15,
        membrane=MembranGene(
            shape="blob",
            color=(0.35, 0.5, 0.45),
            flexibility=0.6,
            has_wall=True,
            wall_color=(0.28, 0.42, 0.38),
        ),
        organelles=[
            OrganelleGene(OrganelleType.RIBOSOME, count=rng.randint(5, 10), size=0.03,
                         color=(0.3, 0.45, 0.4), position_bias=0.5),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(2, 4), size=0.12,
                         color=(0.25, 0.5, 0.42), position_bias=0.4),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.02,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_spirulina_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Spirulina (spiral cyanobacteria, edible superfood)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.4 + rng.random() * 0.2,
        membrane=MembranGene(
            shape="spiral",
            points=rng.randint(4, 7),  # Number of turns
            color=(0.25, 0.55, 0.5),  # Blue-green
            has_wall=True,
            wall_color=(0.2, 0.45, 0.4),
        ),
        organelles=[
            OrganelleGene(OrganelleType.RIBOSOME, count=rng.randint(6, 12), size=0.025,
                         color=(0.2, 0.45, 0.4), position_bias=0.5),
            OrganelleGene(OrganelleType.CHLOROPLAST, count=rng.randint(3, 6), size=0.08,
                         color=(0.22, 0.52, 0.45), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=0.25,
            frequency=0.8,
        ),
        metabolism=MetabolismGene(is_photosynthetic=True),
    )


def create_trypanosoma_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Trypanosoma (blood parasite with undulating membrane)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.6 + rng.random() * 0.3,
        membrane=MembranGene(
            shape="pennate",
            elongation=4.0,
            color=(0.85, 0.75, 0.85),
            flexibility=0.7,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.12,
                         color=(0.6, 0.5, 0.65), position_bias=0.3),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=1, size=0.15,
                         color=(0.8, 0.5, 0.5), position_bias=0.5),
            OrganelleGene(OrganelleType.FLAGELLUM, count=1, size=0.5,
                         color=(0.75, 0.65, 0.75), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=1.2,
            frequency=3.0,
            amplitude=0.4,
        ),
        metabolism=MetabolismGene(is_predator=True),
    )


def create_giardia_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Giardia (pear-shaped flagellate with two nuclei)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.5 + rng.random() * 0.2,
        membrane=MembranGene(
            shape="heart",
            color=(0.8, 0.85, 0.75),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=2, size=0.1,
                         color=(0.5, 0.55, 0.5), position_bias=0.3),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(2, 4), size=0.06,
                         color=(0.7, 0.75, 0.65), position_bias=0.5),
            OrganelleGene(OrganelleType.FLAGELLUM, count=8, size=0.35,
                         color=(0.7, 0.75, 0.65), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.SPIRAL,
            speed=0.6,
            frequency=1.5,
        ),
    )


def create_trichomonas_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Trichomonas (flagellate with undulating membrane)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.55 + rng.random() * 0.2,
        membrane=MembranGene(
            shape="blob",
            color=(0.85, 0.82, 0.78),
            flexibility=0.5,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.15,
                         color=(0.55, 0.5, 0.55), position_bias=0.2),
            OrganelleGene(OrganelleType.GOLGI, count=1, size=0.1,
                         color=(0.7, 0.65, 0.6), position_bias=0.3),
            OrganelleGene(OrganelleType.FLAGELLUM, count=4, size=0.4,
                         color=(0.75, 0.72, 0.68), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=0.7,
            frequency=2.0,
        ),
    )


def create_plasmodium_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Plasmodium (malaria parasite, various life stages)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.4 + rng.random() * 0.2,
        membrane=MembranGene(
            shape="amoeba",
            color=(0.9, 0.75, 0.8),
            flexibility=0.6,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.2,
                         color=(0.65, 0.5, 0.6), position_bias=0.2),
            OrganelleGene(OrganelleType.RIBOSOME, count=rng.randint(6, 12), size=0.03,
                         color=(0.8, 0.65, 0.7), position_bias=0.5),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(1, 3), size=0.08,
                         color=(0.85, 0.7, 0.75), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.PULSE,
            speed=0.3,
            frequency=0.8,
        ),
        metabolism=MetabolismGene(is_predator=True),
    )


def create_euplotes_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Euplotes (crawling ciliate with cirri 'legs')."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.9 + rng.random() * 0.4,
        membrane=MembranGene(
            shape="oval",
            elongation=1.5,
            color=(0.8, 0.85, 0.78),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=2, size=0.1,  # Macro + micro nucleus
                         color=(0.5, 0.52, 0.48), position_bias=0.3),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(8, 15), size=0.025,
                         color=(0.85, 0.5, 0.45), position_bias=0.5),
            OrganelleGene(OrganelleType.CILIA, count=20, size=0.08,  # Cirri (fused cilia)
                         color=(0.7, 0.75, 0.68), position_bias=1.0),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(2, 4), size=0.06,
                         color=(0.75, 0.8, 0.72), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=0.5,
            frequency=1.5,
        ),
    )


def create_stylonychia_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Stylonychia (similar to Euplotes, with long bristles)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.85 + rng.random() * 0.35,
        membrane=MembranGene(
            shape="oval",
            elongation=1.8,
            color=(0.78, 0.82, 0.75),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=2, size=0.08,
                         color=(0.48, 0.5, 0.46), position_bias=0.3),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(10, 18), size=0.02,
                         color=(0.82, 0.48, 0.42), position_bias=0.5),
            OrganelleGene(OrganelleType.CILIA, count=25, size=0.1,
                         color=(0.68, 0.72, 0.65), position_bias=1.0),
            OrganelleGene(OrganelleType.FLAGELLUM, count=rng.randint(4, 8), size=0.25,  # Long bristles
                         color=(0.65, 0.7, 0.62), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=0.55,
            frequency=1.8,
        ),
    )


def create_lacrymaria_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Lacrymaria (ciliate with very long extensible 'neck')."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.7 + rng.random() * 0.3,
        membrane=MembranGene(
            shape="tentacled",
            points=1,
            spike_length=1.5,
            color=(0.82, 0.85, 0.78),
            flexibility=0.7,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.15,
                         color=(0.52, 0.55, 0.48), position_bias=0.5),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(6, 12), size=0.03,
                         color=(0.82, 0.48, 0.42), position_bias=0.5),
            OrganelleGene(OrganelleType.CILIA, count=30, size=0.05,
                         color=(0.75, 0.78, 0.72), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.PULSE,
            speed=0.4,
            frequency=0.6,
        ),
        metabolism=MetabolismGene(is_predator=True),
    )


def create_oxytricha_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Oxytricha (oval ciliate with stiff cilia 'styles')."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.65 + rng.random() * 0.25,
        membrane=MembranGene(
            shape="oval",
            elongation=1.4,
            color=(0.75, 0.78, 0.72),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=2, size=0.09,
                         color=(0.45, 0.48, 0.43), position_bias=0.25),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(8, 14), size=0.025,
                         color=(0.8, 0.46, 0.4), position_bias=0.5),
            OrganelleGene(OrganelleType.CILIA, count=18, size=0.06,
                         color=(0.65, 0.68, 0.62), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.WIGGLE,
            speed=0.5,
            frequency=2.0,
        ),
    )


def create_actinophrys_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Actinophrys (sun animalcule with radiating spines)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.8 + rng.random() * 0.4,
        membrane=MembranGene(
            shape="radiating",
            points=rng.randint(15, 25),
            spike_length=0.8,
            color=(0.92, 0.94, 0.88),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.2,
                         color=(0.55, 0.58, 0.52), position_bias=0.0),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(4, 8), size=0.08,
                         color=(0.85, 0.88, 0.8), position_bias=0.4),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(6, 12), size=0.03,
                         color=(0.82, 0.5, 0.45), position_bias=0.5),
        ],
        movement=MovementGene(
            pattern=MovementPattern.STATIC,
            speed=0.05,
        ),
        metabolism=MetabolismGene(is_predator=True),
    )


def create_difflugia_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Difflugia (amoeba with a shell made of debris)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.7 + rng.random() * 0.3,
        membrane=MembranGene(
            shape="blob",
            color=(0.6, 0.55, 0.45),  # Brown, debris-like
            has_wall=True,
            wall_color=(0.5, 0.45, 0.35),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.15,
                         color=(0.4, 0.38, 0.32), position_bias=0.2),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(2, 4), size=0.08,
                         color=(0.55, 0.52, 0.42), position_bias=0.5),
            OrganelleGene(OrganelleType.PSEUDOPOD, count=rng.randint(2, 4), size=0.15,
                         color=(0.65, 0.6, 0.5), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.PULSE,
            speed=0.15,
            frequency=0.5,
        ),
    )


def create_arcella_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Arcella (disc-shaped shelled amoeba)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.6 + rng.random() * 0.25,
        membrane=MembranGene(
            shape="circle",
            color=(0.7, 0.55, 0.35),  # Amber/brown
            has_wall=True,
            wall_color=(0.6, 0.45, 0.25),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=2, size=0.1,
                         color=(0.45, 0.35, 0.25), position_bias=0.3),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(3, 6), size=0.06,
                         color=(0.65, 0.52, 0.32), position_bias=0.5),
            OrganelleGene(OrganelleType.PSEUDOPOD, count=rng.randint(2, 4), size=0.12,
                         color=(0.72, 0.58, 0.38), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.1,
        ),
    )


def create_pelomyxa_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Pelomyxa (giant amoeba with many nuclei)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=1.5 + rng.random() * 0.8,
        membrane=MembranGene(
            shape="amoeba",
            color=(0.75, 0.72, 0.65),
            flexibility=0.8,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=rng.randint(4, 10), size=0.08,
                         color=(0.45, 0.42, 0.38), position_bias=0.5),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(5, 10), size=0.1,
                         color=(0.68, 0.65, 0.58), position_bias=0.5),
            OrganelleGene(OrganelleType.PSEUDOPOD, count=rng.randint(4, 8), size=0.2,
                         color=(0.78, 0.75, 0.68), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.PULSE,
            speed=0.2,
            frequency=0.4,
        ),
    )


def create_vampyrella_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Vampyrella (vampiric amoeba that feeds on algae)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.5 + rng.random() * 0.2,
        membrane=MembranGene(
            shape="amoeba",
            color=(0.9, 0.6, 0.5),  # Orange-red (due to carotenoids from prey)
            flexibility=0.7,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.18,
                         color=(0.6, 0.35, 0.3), position_bias=0.2),
            OrganelleGene(OrganelleType.VACUOLE, count=rng.randint(2, 5), size=0.1,
                         color=(0.85, 0.55, 0.45), position_bias=0.5),
            OrganelleGene(OrganelleType.PSEUDOPOD, count=rng.randint(3, 6), size=0.15,
                         color=(0.92, 0.65, 0.55), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.CHASE,
            speed=0.4,
            frequency=1.0,
        ),
        metabolism=MetabolismGene(is_predator=True),
    )


def create_noctiluca_dna(seed: int = None) -> MicroDNA:
    """Create DNA for Noctiluca (bioluminescent dinoflagellate 'sea sparkle')."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=1.2 + rng.random() * 0.5,
        membrane=MembranGene(
            shape="circle",
            color=(0.9, 0.92, 0.85),
            transparency=0.5,
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.12,
                         color=(0.6, 0.62, 0.55), position_bias=0.2),
            OrganelleGene(OrganelleType.VACUOLE, count=1, size=0.4,  # Large central vacuole
                         color=(0.85, 0.88, 0.8), position_bias=0.0),
            OrganelleGene(OrganelleType.FLAGELLUM, count=1, size=0.3,
                         color=(0.8, 0.82, 0.75), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.DRIFT,
            speed=0.1,
        ),
        glow=0.6,
        glow_color=(0.4, 0.9, 0.6),  # Blue-green bioluminescence
    )


def create_tintinnid_dna(seed: int = None) -> MicroDNA:
    """Create DNA for a Tintinnid (ciliate living in a vase-shaped shell/lorica)."""
    rng = random.Random(seed)
    
    return MicroDNA(
        species_id=rng.randint(0, 999999),
        base_size=0.6 + rng.random() * 0.3,
        membrane=MembranGene(
            shape="blob",
            color=(0.65, 0.6, 0.5),
            has_wall=True,
            wall_color=(0.55, 0.5, 0.4),
        ),
        organelles=[
            OrganelleGene(OrganelleType.NUCLEUS, count=1, size=0.15,
                         color=(0.4, 0.38, 0.32), position_bias=0.3),
            OrganelleGene(OrganelleType.MITOCHONDRIA, count=rng.randint(4, 8), size=0.03,
                         color=(0.78, 0.45, 0.4), position_bias=0.5),
            OrganelleGene(OrganelleType.CILIA, count=25, size=0.08,
                         color=(0.7, 0.65, 0.55), position_bias=1.0),
        ],
        movement=MovementGene(
            pattern=MovementPattern.SPIRAL,
            speed=0.4,
            frequency=1.2,
        ),
    )


# Template registry - organized by type
MICRO_TEMPLATES = {
    # ===================
    # BACTERIA
    # ===================
    "bacteria": create_bacteria_dna,
    "coccus": create_coccus_dna,
    "spirillum": create_spirillum_dna,
    "oscillatoria": create_oscillatoria_dna,
    "bacillus": create_bacillus_dna,
    "vibrio": create_vibrio_dna,
    "streptococcus": create_streptococcus_dna,
    "staphylococcus": create_staphylococcus_dna,
    "spirulina": create_spirulina_dna,
    "anabaena": create_anabaena_dna,
    "nostoc": create_nostoc_dna,
    
    # ===================
    # VIRUSES/PHAGES
    # ===================
    "coronavirus": create_coronavirus_dna,
    "bacteriophage": create_bacteriophage_dna,
    
    # ===================
    # PROTOZOA (Ciliates)
    # ===================
    "paramecium": create_paramecium_dna,
    "vorticella": create_vorticella_dna,
    "stentor": create_stentor_dna,
    "blepharisma": create_blepharisma_dna,
    "colpoda": create_colpoda_dna,
    "didinium": create_didinium_dna,
    "euplotes": create_euplotes_dna,
    "stylonychia": create_stylonychia_dna,
    "lacrymaria": create_lacrymaria_dna,
    "oxytricha": create_oxytricha_dna,
    "tintinnid": create_tintinnid_dna,
    
    # ===================
    # PROTOZOA (Amoebae)
    # ===================
    "amoeba": create_amoeba_dna,
    "difflugia": create_difflugia_dna,
    "arcella": create_arcella_dna,
    "pelomyxa": create_pelomyxa_dna,
    "vampyrella": create_vampyrella_dna,
    
    # ===================
    # PROTOZOA (Flagellates)
    # ===================
    "euglena": create_euglena_dna,
    "chlamydomonas": create_chlamydomonas_dna,
    "giardia": create_giardia_dna,
    "trichomonas": create_trichomonas_dna,
    "trypanosoma": create_trypanosoma_dna,
    "plasmodium": create_plasmodium_dna,
    
    # ===================
    # PROTOZOA (Heliozoa/Radiolaria)
    # ===================
    "heliozoan": create_heliozoan_dna,
    "radiolarian": create_radiolarian_dna,
    "foraminifera": create_foraminifera_dna,
    "actinophrys": create_actinophrys_dna,
    
    # ===================
    # DINOFLAGELLATES
    # ===================
    "dinoflagellate": create_dinoflagellate_dna,
    "ceratium": create_ceratium_dna,
    "peridinium": create_peridinium_dna,
    "noctiluca": create_noctiluca_dna,
    
    # ===================
    # DIATOMS (Pennate & Centric)
    # ===================
    "diatom": create_diatom_dna,
    "navicula": create_navicula_dna,
    "pinnularia": create_pinnularia_dna,
    "asterionella": create_asterionella_dna,
    
    # ===================
    # GREEN ALGAE / DESMIDS
    # ===================
    "algae": create_algae_dna,
    "desmid": create_desmid_dna,
    "volvox": create_volvox_dna,
    "spirogyra": create_spirogyra_dna,
    "pediastrum": create_pediastrum_dna,
    "micrasterias": create_micrasterias_dna,
    "closterium": create_closterium_dna,
    "cosmarium": create_cosmarium_dna,
    "scenedesmus": create_scenedesmus_dna,
    
    # ===================
    # MICRO-ANIMALS
    # ===================
    "tardigrade": create_tardigrade_dna,
    "hydra": create_hydra_dna,
    "nematode": create_nematode_dna,
    "copepod": create_copepod_dna,
    "daphnia": create_daphnia_dna,
    "rotifer": create_rotifer_dna,
    
    # ===================
    # RANDOM
    # ===================
    "random": MicroDNA.create_random,
}


# Terrain-specific organism populations
TERRAIN_POPULATIONS = {
    "water": {
        # Open water - diverse aquatic life
        # Bacteria
        "bacteria": 6,
        "coccus": 3,
        "vibrio": 3,
        "spirillum": 2,
        "spirulina": 2,
        # Diatoms - super common in water!
        "diatom": 6,
        "navicula": 4,
        "pinnularia": 3,
        "asterionella": 2,
        # Dinoflagellates
        "dinoflagellate": 4,
        "ceratium": 2,
        "peridinium": 3,
        "noctiluca": 1,  # Bioluminescent!
        # Green algae
        "algae": 3,
        "chlamydomonas": 4,
        "volvox": 2,
        "pediastrum": 2,
        # Ciliates
        "paramecium": 4,
        "stentor": 1,
        "blepharisma": 1,
        "vorticella": 2,
        "euplotes": 1,
        "lacrymaria": 1,
        "tintinnid": 2,
        # Flagellates
        "euglena": 4,
        # Radiolaria/Heliozoa
        "radiolarian": 3,
        "heliozoan": 2,
        "actinophrys": 1,
        # Micro-animals
        "rotifer": 3,
        "copepod": 3,
        "daphnia": 2,
        "hydra": 1,
    },
    "plant": {
        # Plant surface - bacteria, algae, some protozoa
        # Bacteria
        "bacteria": 10,
        "coccus": 5,
        "bacillus": 4,
        "streptococcus": 2,
        # Cyanobacteria
        "oscillatoria": 4,
        "anabaena": 3,
        "nostoc": 2,
        "spirulina": 2,
        # Algae and desmids
        "algae": 5,
        "spirogyra": 4,
        "desmid": 3,
        "micrasterias": 2,
        "closterium": 2,
        "cosmarium": 2,
        "scenedesmus": 3,
        "chlamydomonas": 3,
        # Diatoms on plant surfaces
        "navicula": 2,
        "pinnularia": 2,
        # Ciliates
        "vorticella": 3,  # Often attached to plants
        "colpoda": 2,
        "stylonychia": 1,
        "oxytricha": 1,
        # Amoebae
        "amoeba": 3,
        "arcella": 1,
        # Flagellates
        "euglena": 2,
        # Micro-animals
        "tardigrade": 2,  # Water bears love moss!
        "nematode": 3,
        "rotifer": 2,
    },
    "ground": {
        # Soil - bacteria, amoebae, ciliates, nematodes
        # Bacteria - very abundant in soil!
        "bacteria": 10,
        "coccus": 6,
        "bacillus": 5,
        "streptococcus": 3,
        "staphylococcus": 3,
        "spirillum": 3,
        # Cyanobacteria
        "oscillatoria": 2,
        "nostoc": 3,
        "anabaena": 2,
        # Amoebae - common in soil!
        "amoeba": 5,
        "arcella": 3,
        "difflugia": 3,
        "pelomyxa": 1,
        "vampyrella": 1,
        # Ciliates
        "colpoda": 4,  # Very common soil ciliate
        "paramecium": 2,
        "didinium": 1,
        "oxytricha": 2,
        "stylonychia": 1,
        # Flagellates
        "giardia": 1,
        "trichomonas": 1,
        # Soil diatoms
        "diatom": 2,
        "navicula": 2,
        # Foraminifera (in marine-derived soils)
        "foraminifera": 2,
        # Micro-animals
        "nematode": 5,  # Very common in soil!
        "tardigrade": 2,
        "rotifer": 2,
    },
}

