"""
Life Simulation System for Waverse

Handles:
- Plant growth/death based on sun/rain
- Animal growth, hunger, death
- Animal reproduction (male/female, eggs)
- Food chain (herbivores eat plants, carnivores eat animals)

Designed for efficiency - updates are batched and infrequent.
"""
from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from enum import Enum

if TYPE_CHECKING:
    from .flora import PlantInstance
    from .animals import AnimalInstance

# Type aliases for clarity
ChunkKey = tuple[int, int]
PlantsByChunk = dict[ChunkKey, list[Any]]  # PlantInstance
AnimalsByChunk = dict[ChunkKey, list[Any]]  # AnimalInstance
DNADict = dict[str, Any]


# =============================================================================
# GLOBAL CONFIGURATION - Tweak these for testing!
# =============================================================================
class LifeConfig:
    """Global configuration for life simulation rates."""
    
    # Time scale: how many game-hours pass per real second
    # 0.05 = 3 game-minutes per real second (slower, more natural)
    TIME_SCALE = 0.05
    
    # Plant growth rate multiplier (slower = more natural)
    PLANT_GROWTH_RATE = 1.0
    
    # Plant visual scale range (min% to max% based on growth)
    PLANT_MIN_SCALE = 0.3   # 30% at growth=0
    PLANT_MAX_SCALE = 1.0   # 100% at growth=1
    
    # Growth variance - random variance in growth rate per plant
    GROWTH_VARIANCE = 0.5   # 0-50% variance
    
    # Animal growth rate multiplier  
    ANIMAL_GROWTH_RATE = 1.0
    
    # Plant spawn chance per update when conditions are good
    PLANT_SPAWN_CHANCE = 0.03  # 3% chance
    
    # Pregnancy duration in game-hours
    PREGNANCY_DURATION = 2.0  # 2 hours
    
    # Egg hatch time range in game-hours
    EGG_HATCH_MIN = 1.0
    EGG_HATCH_MAX = 3.0
    
    # Performance caps
    MAX_PLANTS_PER_CHUNK = 40
    MAX_TOTAL_ANIMALS = 150
    MAX_EGGS = 30
    
    # === PERFORMANCE TUNING ===
    # Life simulation runs every UPDATE_INTERVAL seconds (NOT every frame!)
    # At 60fps, 0.5 = only 2 updates/sec instead of 60 = 30x savings
    UPDATE_INTERVAL = 0.4       # Seconds between life updates
    UPDATE_RADIUS = 12          # Update chunks within this radius (bigger = more life!)
    MAX_PLANTS_PER_UPDATE = 80  # Max plants to update per chunk per cycle
    MAX_ANIMALS_PER_UPDATE = 50 # Max animals to update per cycle
    
    # Logging (set to False to disable console spam)
    LOG_ENABLED = False
    LOG_INTERVAL = 15.0  # Log stats every 15 seconds


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"


class Diet(Enum):
    HERBIVORE = "herbivore"  # Eats plants
    CARNIVORE = "carnivore"  # Eats animals
    OMNIVORE = "omnivore"    # Eats both


@dataclass
class LifeState:
    """Life state for any living entity (plant or animal)."""
    growth: float = 0.5  # 0.0 = seed/baby, 1.0 = fully grown
    health: float = 1.0  # 0.0 = dead, 1.0 = healthy
    age: float = 0.0     # Age in game-hours
    
    def is_alive(self) -> bool:
        return self.health > 0.0
    
    def is_mature(self) -> bool:
        return self.growth >= 0.6  # Balanced - not too hard, not too easy


@dataclass
class PlantLife(LifeState):
    """Extended life state for plants - energy-based system."""
    max_energy: float = 100.0  # Based on plant size
    energy: float = 50.0       # Current energy (0 = dead)
    
    def update(self, dt_hours: float, is_raining: bool, is_day: bool, sun_intensity: float = 1.0,
                growth_rate_mod: float = 1.0):
        """Update plant life state."""
        self.age += dt_hours
        
        # Growth (with variance from growth_rate_mod)
        if self.energy > self.max_energy * 0.3:
            base_rate = 0.05 * LifeConfig.PLANT_GROWTH_RATE
            growth_rate = base_rate * growth_rate_mod  # Apply per-plant variance
            self.growth = min(1.0, self.growth + dt_hours * growth_rate)
            # Max energy increases as plant grows
            self.max_energy = 50 + self.growth * 150  # 50-200 energy capacity
        
        # Energy regeneration from sun/water
        # Bigger plants absorb more energy
        if is_day and sun_intensity > 0:
            absorption_rate = 5.0 * self.growth * sun_intensity  # Bigger = more absorption
            if is_raining:
                absorption_rate *= 1.5  # Bonus with water
            self.energy = min(self.max_energy, self.energy + dt_hours * absorption_rate)
        
        # Slow energy drain at night
        if not is_day:
            self.energy = max(0, self.energy - dt_hours * 0.5)
        
        # Health tied to energy ratio
        self.health = self.energy / self.max_energy
        
        # Die if no energy
        if self.energy <= 0:
            self.health = 0.0
    
    def get_render_fraction(self) -> float:
        """Get fraction of plant to render (0.0 to 1.0) based on energy."""
        if self.max_energy <= 0:
            return 1.0
        # Render fraction based on current energy vs max
        fraction = self.energy / self.max_energy
        # Always render at least 10% if alive
        return max(0.1, fraction) if self.health > 0 else 0.0
    
    def take_bite(self, animal_size: float = 1.0) -> float:
        """Animal takes a bite. Returns energy gained by animal."""
        # Bite size based on animal size (bigger animals take bigger bites)
        bite_energy = 5.0 + animal_size * 5.0  # 5-15 energy per bite
        
        # Can't take more than plant has
        actual_bite = min(bite_energy, self.energy)
        self.energy -= actual_bite
        
        return actual_bite


@dataclass 
class AnimalLife(LifeState):
    """Extended life state for animals - energy-based system."""
    gender: Gender = Gender.MALE
    diet: Diet = Diet.HERBIVORE
    max_energy: float = 100.0  # Based on animal size
    energy: float = 50.0       # Current energy
    pregnant: bool = False
    pregnancy_timer: float = 0.0
    eat_cooldown: float = 0.0  # Time until can eat again
    
    def __post_init__(self):
        # Randomize gender if not set
        if random.random() < 0.5:
            self.gender = Gender.FEMALE
    
    def update(self, dt_hours: float, is_day: bool, growth_rate_mod: float = 1.0):
        """Update animal life state.
        
        Args:
            dt_hours: Time delta in game-hours
            is_day: Whether it's daytime
            growth_rate_mod: Growth rate modifier from DNA (0.5-2.0)
        """
        self.age += dt_hours
        
        # Reduce eat cooldown
        if self.eat_cooldown > 0:
            self.eat_cooldown = max(0, self.eat_cooldown - dt_hours)
        
        # Energy drain over time (metabolism) - MUCH slower
        # Bigger animals burn more but still slow
        burn_rate = 0.5 + self.growth * 1.0  # 0.5-1.5 energy/hour (was 2-5!)
        if is_day:
            burn_rate *= 1.1  # Slightly more active during day
        self.energy = max(0, self.energy - dt_hours * burn_rate)
        
        # Growth when well-fed - uses DNA growth rate!
        if self.energy > self.max_energy * 0.5 and self.growth < 1.0:
            base_rate = 0.03 * LifeConfig.ANIMAL_GROWTH_RATE
            growth_rate = base_rate * growth_rate_mod  # Apply DNA-evolved rate
            self.growth = min(1.0, self.growth + dt_hours * growth_rate)
            # Max energy increases as animal grows
            self.max_energy = 50 + self.growth * 150  # 50-200 capacity
        
        # Health tied to energy ratio
        energy_ratio = self.energy / self.max_energy
        if energy_ratio > 0.3:
            self.health = min(1.0, self.health + dt_hours * 0.05)
        elif energy_ratio < 0.1:
            # Starving
            self.health = max(0.0, self.health - dt_hours * 0.1)
        
        # Pregnancy progress (costs energy)
        if self.pregnant:
            self.pregnancy_timer += dt_hours
            self.energy = max(0, self.energy - dt_hours * 2.0)  # Extra energy cost
        
        # Old age
        max_age = 500 + self.growth * 200
        if self.age > max_age:
            self.health = max(0.0, self.health - dt_hours * 0.01)
    
    def can_reproduce(self) -> bool:
        """Check if animal can reproduce."""
        return (self.is_mature() and 
                self.energy > self.max_energy * 0.6 and  # Need 60% energy
                not self.pregnant and
                self.health > 0.5 and
                self.eat_cooldown <= 0)
    
    def can_eat(self) -> bool:
        """Check if animal can eat (not on cooldown)."""
        return self.eat_cooldown <= 0 and self.energy < self.max_energy * 0.9
    
    def eat(self, energy_gained: float):
        """Animal eats something, gains energy."""
        # Eating is very rewarding - gain 2x what you ate
        self.energy = min(self.max_energy, self.energy + energy_gained * 2.0)
        # Short cooldown so they can eat again soon
        self.eat_cooldown = 0.1 + energy_gained / 50.0  # 0.1-0.4 hours
    
    def get_hunger_level(self) -> float:
        """Get hunger as 0-1 (for compatibility). 0 = starving, 1 = full."""
        return self.energy / self.max_energy


@dataclass
class Egg:
    """An egg that will hatch into a new animal."""
    x: float
    y: float
    z: float
    parent1_dna: DNADict  # DNA from parent 1
    parent2_dna: DNADict  # DNA from parent 2
    hatch_timer: float = 0.0
    hatch_time: float = field(default_factory=lambda: random.uniform(
        LifeConfig.EGG_HATCH_MIN, LifeConfig.EGG_HATCH_MAX))
    size: float = 0.3  # Visual size
    
    # DNA-derived appearance (set in __post_init__)
    base_color: tuple[float, float, float] = (0.9, 0.85, 0.7)  # Default cream
    spot_color: tuple[float, float, float] = (0.6, 0.5, 0.4)   # Default brown spots
    has_spots: bool = False
    has_stripes: bool = False
    elongation: float = 1.0  # 1.0 = round, 1.5 = elongated
    
    def __post_init__(self):
        """Derive egg appearance from parent DNA."""
        if self.parent1_dna:
            # Use parent DNA to determine egg appearance
            dna = self.parent1_dna
            
            # Get parent color if available
            if 'body_color' in dna:
                bc = dna['body_color']
                if isinstance(bc, (list, tuple)) and len(bc) >= 3:
                    # Lighter, more pastel version of parent color
                    self.base_color = (
                        min(1.0, bc[0] * 0.5 + 0.5),
                        min(1.0, bc[1] * 0.5 + 0.5),
                        min(1.0, bc[2] * 0.5 + 0.5)
                    )
            
            # Spot color from accent if available
            if 'accent_color' in dna:
                ac = dna['accent_color']
                if isinstance(ac, (list, tuple)) and len(ac) >= 3:
                    self.spot_color = (ac[0] * 0.7, ac[1] * 0.7, ac[2] * 0.7)
            
            # Pattern based on animal type
            animal_type = dna.get('animal_type', '')
            if animal_type in ('bird', 'reptile', 'dinosaur', 'croc'):
                self.has_spots = True
            elif animal_type in ('fish', 'amphibian', 'snake'):
                self.has_stripes = True
            elif animal_type in ('insect', 'spider', 'alien'):
                self.has_spots = random.random() < 0.5
            
            # Size = 10% of parent size (minimum 0.6 for visibility)
            if 'base_scale' in dna:
                self.size = max(0.6, dna['base_scale'] * 0.15)
            elif 'body_scale' in dna:
                self.size = max(0.6, dna['body_scale'] * 0.15)
            else:
                self.size = 0.6
            
            # Elongation based on animal type
            if animal_type in ('snake', 'worm', 'fish'):
                self.elongation = 1.4
            elif animal_type in ('bird', 'dinosaur'):
                self.elongation = 1.2
            else:
                self.elongation = 1.0 + random.random() * 0.2
    
    def update(self, dt_hours: float, is_warm: bool = True) -> bool:
        """Update egg. Returns True if ready to hatch."""
        if is_warm:
            self.hatch_timer += dt_hours
        return self.hatch_timer >= self.hatch_time


class LifeSimulator:
    """
    Manages life simulation for the world.
    
    Updates are batched and run at low frequency for performance.
    Only entities near the player are actively simulated.
    """
    
    def __init__(self, seed: int = 42) -> None:
        self.seed: int = seed
        self.rng: random.Random = random.Random(seed)
        
        # Life states keyed by entity id
        self.plant_life: dict[int, PlantLife] = {}
        self.animal_life: dict[int, AnimalLife] = {}
        
        # Active eggs in the world
        self.eggs: list[Egg] = []
        
        # Pending births (chunk_key -> list of DNA dicts)
        self.pending_births: dict[ChunkKey, list[DNADict]] = {}
        
        # Pending plant spawns
        self.pending_plants: dict[ChunkKey, int] = {}  # chunk -> count to spawn
        
        # Dead entities to remove
        self.dead_plants: list[int] = []
        self.dead_animals: list[int] = []
        
        # Chunks that need display list refresh (plants changed)
        self.chunks_needing_refresh: set[ChunkKey] = set()
        
        # Update timing - use config for performance tuning
        self.update_timer: float = 0.0
        self.update_interval: float = LifeConfig.UPDATE_INTERVAL
        
        # Logging
        self.log_timer: float = 0.0
        self.last_log_stats: dict[str, Any] = {}
        
        # Stats
        self.births: int = 0
        self.deaths: int = 0
        self.plants_grown: int = 0
        self.animals_ate: int = 0
        self.matings: int = 0
        self.eggs_hatched: int = 0
    
    def get_or_create_plant_life(self, plant_id: int, plant_size: float = None) -> PlantLife:
        """Get or create life state for a plant."""
        if plant_id not in self.plant_life:
            # Initial growth based on plant size if provided
            growth = self.rng.uniform(0.4, 0.8)
            # Max energy scales with plant size
            size = plant_size if plant_size else 2.0
            max_energy = 50 + size * 30  # Bigger plants = more energy
            
            self.plant_life[plant_id] = PlantLife(
                growth=growth,
                health=self.rng.uniform(0.7, 1.0),
                max_energy=max_energy,
                energy=max_energy * self.rng.uniform(0.5, 0.9)  # Start partially full
            )
        return self.plant_life[plant_id]
    
    def get_or_create_animal_life(self, animal_id: int, animal_type: str = None) -> AnimalLife:
        """Get or create life state for an animal."""
        if animal_id not in self.animal_life:
            # Determine diet based on animal type
            diet = Diet.HERBIVORE
            if animal_type:
                type_lower = animal_type.lower()
                if any(x in type_lower for x in ['spider', 'snake', 'worm', 'metroid', 'alligator', 'dinosaur', 'squid']):
                    diet = Diet.CARNIVORE
                elif any(x in type_lower for x in ['bird', 'fish', 'crab', 'snail']):
                    diet = Diet.OMNIVORE
            
            # Initial energy based on random size
            initial_growth = self.rng.uniform(0.5, 0.95)  # Start more mature
            max_energy = 80 + initial_growth * 120  # 80-200 (higher base)
            
            self.animal_life[animal_id] = AnimalLife(
                growth=initial_growth,
                health=self.rng.uniform(0.9, 1.0),
                max_energy=max_energy,
                energy=max_energy * self.rng.uniform(0.7, 1.0),  # Start 70-100% full!
                diet=diet,
                gender=Gender.FEMALE if self.rng.random() < 0.5 else Gender.MALE
            )
        return self.animal_life[animal_id]
    
    def update(self, dt: float, camera_x: float, camera_z: float,
               is_raining: bool, is_day: bool, sun_intensity: float,
               plants_by_chunk: dict, animals_by_chunk: dict,
               chunk_size: float = 128.0):
        """
        Main update loop. Call every frame but only does work periodically.
        
        Args:
            dt: Delta time in seconds
            camera_x, camera_z: Player position
            is_raining: Current weather
            is_day: Day/night
            sun_intensity: 0-1 sun strength
            plants_by_chunk: Dict of (cx, cz) -> list of plants
            animals_by_chunk: Dict of (cx, cz) -> list of animals
            chunk_size: World units per chunk
        """
        self.update_timer += dt
        self.log_timer += dt
        
        if self.update_timer < self.update_interval:
            return
        
        self.update_timer = 0.0
        dt_hours = self.update_interval * LifeConfig.TIME_SCALE
        
        # Clear pending removals
        self.dead_plants.clear()
        self.dead_animals.clear()
        
        # Get chunks near player (only simulate nearby for performance)
        cam_cx = int(camera_x // chunk_size)
        cam_cz = int(camera_z // chunk_size)
        
        # Use configurable radius for performance
        sim_radius = LifeConfig.UPDATE_RADIUS
        
        # Track how many entities we've updated this cycle
        plants_updated = 0
        animals_updated = 0
        max_plants = LifeConfig.MAX_PLANTS_PER_UPDATE * (sim_radius * 2 + 1) ** 2
        max_animals = LifeConfig.MAX_ANIMALS_PER_UPDATE * (sim_radius * 2 + 1)
        
        for dx in range(-sim_radius, sim_radius + 1):
            for dz in range(-sim_radius, sim_radius + 1):
                cx, cz = cam_cx + dx, cam_cz + dz
                chunk_key = (cx, cz)
                
                # Update plants in this chunk (with limit)
                if chunk_key in plants_by_chunk and plants_updated < max_plants:
                    chunk_plants = plants_by_chunk[chunk_key]
                    # Limit plants per chunk
                    plants_to_update = chunk_plants[:LifeConfig.MAX_PLANTS_PER_UPDATE]
                    self._update_plants(plants_to_update, dt_hours, 
                                       is_raining, is_day, sun_intensity, chunk_key)
                    plants_updated += len(plants_to_update)
                
                # Update animals in this chunk (with limit)
                if chunk_key in animals_by_chunk and animals_updated < max_animals:
                    chunk_animals = animals_by_chunk[chunk_key]
                    animals_to_update = chunk_animals[:LifeConfig.MAX_ANIMALS_PER_UPDATE]
                    self._update_animals(animals_to_update, dt_hours,
                                        is_day, plants_by_chunk.get(chunk_key, []),
                                        chunk_key)
                    animals_updated += len(animals_to_update)
        
        # Update eggs
        self._update_eggs(dt_hours, is_day)
        
        # Periodic logging
        if LifeConfig.LOG_ENABLED and self.log_timer >= LifeConfig.LOG_INTERVAL:
            self._log_status(is_raining, is_day, sun_intensity)
            self.log_timer = 0.0
    
    def _update_plants(self, plants: list, dt_hours: float, 
                       is_raining: bool, is_day: bool, sun_intensity: float,
                       chunk_key: tuple):
        """Update plants in a chunk."""
        plants_to_remove = []
        any_visual_change = False
        
        for plant in plants:
            plant_id = id(plant)
            
            # Get plant size for energy scaling
            plant_height = getattr(plant.dna, 'height_gene', None)
            plant_size = plant_height.value if plant_height else 2.0
            
            life = self.get_or_create_plant_life(plant_id, plant_size)
            
            old_fraction = life.get_render_fraction()
            old_growth = life.growth
            
            # Get growth rate from plant's DNA (evolved trait!)
            # Falls back to random variance if DNA doesn't have growth_rate
            if hasattr(plant, 'dna') and hasattr(plant.dna, 'growth_rate'):
                growth_mod = plant.dna.growth_rate
            else:
                growth_mod = getattr(plant, 'growth_rate_mod', 1.0)
            life.update(dt_hours, is_raining, is_day, sun_intensity, growth_mod)
            new_fraction = life.get_render_fraction()
            
            # Store render fraction on plant for rendering
            plant.render_fraction = new_fraction
            
            # Also update plant scale based on growth (THIS IS KEY FOR VISIBLE GROWTH!)
            base_scale = getattr(plant, 'base_scale', plant.scale)
            if not hasattr(plant, 'base_scale'):
                plant.base_scale = plant.scale  # Store original scale
                # Add growth variance - each plant has a random growth rate modifier
                plant.growth_rate_mod = 0.5 + self.rng.random() * LifeConfig.GROWTH_VARIANCE
            
            # Apply parameterized scale range
            min_scale = LifeConfig.PLANT_MIN_SCALE
            max_scale = LifeConfig.PLANT_MAX_SCALE
            plant.scale = base_scale * (min_scale + life.growth * (max_scale - min_scale))
            
            # Check for visual change (energy, growth, or scale changed)
            # Be more aggressive - any change triggers refresh
            if abs(old_fraction - new_fraction) > 0.01 or abs(old_growth - life.growth) > 0.005:
                any_visual_change = True
                # Verbose growth logging (commented out for performance)
                # if LifeConfig.LOG_ENABLED and abs(old_growth - life.growth) > 0.02:
                #     print(f"  [LIFE] Plant grew: {old_growth*100:.0f}% -> {life.growth*100:.0f}%")
            
            # Check for death - mark for removal
            if not life.is_alive():
                plants_to_remove.append(plant)
                self.dead_plants.append(plant_id)
                self.deaths += 1
                if LifeConfig.LOG_ENABLED:
                    print(f"  [LIFE] Plant died (no energy) at age {life.age:.1f}h")
        
        # Actually remove dead plants from the list
        for dead_plant in plants_to_remove:
            if dead_plant in plants:
                plants.remove(dead_plant)
                any_visual_change = True
        
        # Mark chunk for refresh if anything changed visually
        # (only refresh occasionally to save performance)
        if any_visual_change:
            self.chunks_needing_refresh.add(chunk_key)
        
        # Chance to spawn new plant if conditions good (cap at 40 per chunk)
        if (is_raining or sun_intensity > 0.5) and len(plants) < 40:
            if self.rng.random() < LifeConfig.PLANT_SPAWN_CHANCE:
                self.pending_plants[chunk_key] = self.pending_plants.get(chunk_key, 0) + 1
                self.plants_grown += 1
                if LifeConfig.LOG_ENABLED and self.rng.random() < 0.3:
                    print(f"  [LIFE] New plant sprouting!")
    
    def _update_animals(self, animals: list, dt_hours: float, is_day: bool,
                        plants: list, chunk_key: tuple):
        """Update animals in a chunk."""
        # Build spatial index for nearby checks
        animal_positions = [(a, a.x, a.z, id(a)) for a in animals if hasattr(a, 'x')]
        
        for animal in animals:
            animal_id = id(animal)
            animal_type = getattr(animal, 'animal_type', None) or getattr(animal.dna, 'animal_type', 'walker')
            life = self.get_or_create_animal_life(animal_id, animal_type)
            
            # Get growth rate from animal's DNA (evolved trait!)
            if hasattr(animal, 'dna') and hasattr(animal.dna, 'growth_rate'):
                growth_mod = animal.dna.growth_rate
            else:
                growth_mod = 1.0  # Default if no DNA growth rate
            life.update(dt_hours, is_day, growth_mod)
            
            # Apply growth to visual scale
            if hasattr(animal, 'scale'):
                animal.scale = 0.4 + life.growth * 0.6
            
            # Eating behavior (only if can eat - not on cooldown)
            if life.can_eat() and hasattr(animal, 'x'):
                if life.diet in (Diet.HERBIVORE, Diet.OMNIVORE) and plants:
                    # Try to eat nearby plant
                    for plant in plants[:10]:
                        if hasattr(plant, 'x'):
                            # Size check: animal can only eat plants smaller than itself
                            plant_height = getattr(plant.dna, 'height_gene', None)
                            plant_size = plant_height.value if plant_height else 2.0
                            animal_size = life.growth * 3.0
                            
                            # Can't eat trees if you're small
                            if plant_size > animal_size * 2:
                                continue
                            
                            # Skip if plant has no energy left
                            plant_life = self.get_or_create_plant_life(id(plant))
                            if plant_life.energy < 5:
                                continue  # Not enough to eat
                            
                            # Must be reasonably close (5 units - easier to find food)
                            dist = math.sqrt((animal.x - plant.x)**2 + (animal.z - plant.z)**2)
                            if dist < 5.0:
                                # Take a bite! Energy transfer
                                energy_gained = plant_life.take_bite(life.growth)
                                life.eat(energy_gained)
                                self.animals_ate += 1
                                
                                # Mark chunk for visual refresh
                                cx = int(plant.x // 128)
                                cz = int(plant.z // 128)
                                self.chunks_needing_refresh.add((cx, cz))
                                
                                if LifeConfig.LOG_ENABLED and self.rng.random() < 0.3:
                                    pct = plant_life.get_render_fraction() * 100
                                    print(f"  [LIFE] {animal_type} ate {energy_gained:.0f} energy, plant at {pct:.0f}%")
                                break
                
                if life.diet in (Diet.CARNIVORE, Diet.OMNIVORE):
                    # Try to eat smaller animals (must be close!)
                    for other, ox, oz, other_id in animal_positions:
                        if other_id == animal_id:
                            continue
                        other_life = self.animal_life.get(other_id)
                        if other_life and other_life.growth < life.growth * 0.7:
                            dist = math.sqrt((animal.x - ox)**2 + (animal.z - oz)**2)
                            if dist < 2.0:  # Must be very close
                                # Gain energy from prey (half their energy)
                                energy_gained = other_life.energy * 0.5
                                life.eat(energy_gained)
                                other_life.health = 0  # Kill prey
                                self.animals_ate += 1
                                if LifeConfig.LOG_ENABLED:
                                    print(f"  [LIFE] {animal_type} hunted! +{energy_gained:.0f} energy")
                                break
            
            # Reproduction (costs energy for both)
            if life.can_reproduce() and life.gender == Gender.FEMALE:
                # Look for nearby male
                for other, ox, oz, other_id in animal_positions:
                    if other_id == animal_id:
                        continue
                    other_life = self.animal_life.get(other_id)
                    if (other_life and 
                        other_life.gender == Gender.MALE and 
                        other_life.can_reproduce()):
                        dist = math.sqrt((animal.x - ox)**2 + (animal.z - oz)**2)
                        if dist < 4.0:  # Must be close to mate
                            # Mate! Both lose energy
                            life.pregnant = True
                            life.pregnancy_timer = 0.0
                            life.energy -= 20  # Female energy cost
                            other_life.energy -= 15  # Male energy cost
                            self.matings += 1
                            if LifeConfig.LOG_ENABLED:
                                print(f"  [LIFE] {animal_type} mated! (-20 energy)")
                            break
            
            # Give birth (lay egg) - with cap check
            if life.pregnant and life.pregnancy_timer >= LifeConfig.PREGNANCY_DURATION:
                life.pregnant = False
                # Only create egg if under cap
                if len(self.eggs) < LifeConfig.MAX_EGGS and hasattr(animal, 'x') and hasattr(animal, 'dna'):
                    egg = Egg(
                        x=animal.x + self.rng.uniform(-1, 1),
                        y=getattr(animal, 'y', 0),
                        z=animal.z + self.rng.uniform(-1, 1),
                        parent1_dna=animal.dna.to_dict() if hasattr(animal.dna, 'to_dict') else {},
                        parent2_dna={}
                    )
                    self.eggs.append(egg)
                    self.births += 1
                    if LifeConfig.LOG_ENABLED:
                        print(f"  [LIFE] {animal_type} laid an egg at ({animal.x:.0f}, {animal.z:.0f})!")
            
            # Check for death
            if not life.is_alive():
                self.dead_animals.append(animal_id)
                self.deaths += 1
                if LifeConfig.LOG_ENABLED:
                    energy_ratio = life.energy / life.max_energy if life.max_energy > 0 else 0
                    cause = "starvation" if energy_ratio < 0.1 else "old age" if life.age > 300 else "unknown"
                    print(f"  [LIFE] {animal_type} died from {cause} at age {life.age:.1f}h")
    
    def _update_eggs(self, dt_hours: float, is_warm: bool):
        """Update all eggs, hatching those that are ready."""
        hatched = []
        remaining = []
        
        for egg in self.eggs:
            if egg.update(dt_hours, is_warm):
                hatched.append(egg)
            else:
                remaining.append(egg)
        
        self.eggs = remaining
        
        # Count total animals for cap check
        total_animals = len(self.animal_life)
        
        # Queue hatched eggs for animal spawning (with cap)
        for egg in hatched:
            if total_animals >= LifeConfig.MAX_TOTAL_ANIMALS:
                if LifeConfig.LOG_ENABLED:
                    print(f"  [LIFE] Egg hatched but animal cap reached!")
                break  # Stop hatching if at cap
            
            cx = int(egg.x // 128)  # Approximate chunk
            cz = int(egg.z // 128)
            chunk_key = (cx, cz)
            if chunk_key not in self.pending_births:
                self.pending_births[chunk_key] = []
            self.pending_births[chunk_key].append({
                'x': egg.x,
                'y': egg.y, 
                'z': egg.z,
                'parent_dna': egg.parent1_dna
            })
            self.eggs_hatched += 1
            total_animals += 1  # Count this one
            if LifeConfig.LOG_ENABLED:
                print(f"  [LIFE] Egg hatched at ({egg.x:.0f}, {egg.z:.0f})! New animal born!")
    
    def _log_status(self, is_raining: bool, is_day: bool, sun_intensity: float):
        """Log periodic status update."""
        # Count living entities
        living_plants = sum(1 for p in self.plant_life.values() if p.is_alive())
        mature_plants = sum(1 for p in self.plant_life.values() if p.is_alive() and p.is_mature())
        living_animals = sum(1 for a in self.animal_life.values() if a.is_alive())
        mature_animals = sum(1 for a in self.animal_life.values() if a.is_alive() and a.is_mature())
        pregnant = sum(1 for a in self.animal_life.values() if a.pregnant)
        
        # Energy stats
        avg_plant_energy = sum(p.energy / p.max_energy for p in self.plant_life.values() if p.max_energy > 0) / max(1, len(self.plant_life))
        avg_animal_energy = sum(a.energy / a.max_energy for a in self.animal_life.values() if a.max_energy > 0) / max(1, len(self.animal_life))
        hungry_animals = sum(1 for a in self.animal_life.values() if a.energy < a.max_energy * 0.3)
        
        # Growth stats
        avg_plant_growth = sum(p.growth for p in self.plant_life.values()) / max(1, len(self.plant_life))
        avg_animal_growth = sum(a.growth for a in self.animal_life.values()) / max(1, len(self.animal_life))
        
        weather = "🌧️ Rain" if is_raining else ("☀️ Day" if is_day else "🌙 Night")
        
        print(f"\n{'='*55}")
        print(f"  [LIFE SIMULATION] {weather} (Sun: {sun_intensity:.1f})")
        print(f"{'='*55}")
        print(f"  Plants: {living_plants} ({mature_plants} mature)")
        print(f"    Growth: {avg_plant_growth:.0%} | Energy: {avg_plant_energy:.0%}")
        print(f"  Animals: {living_animals} ({mature_animals} mature) | {pregnant} pregnant")
        print(f"    Growth: {avg_animal_growth:.0%} | Energy: {avg_animal_energy:.0%} | {hungry_animals} hungry")
        print(f"  Eggs: {len(self.eggs)}")
        print(f"  Stats: {self.births} births, {self.deaths} deaths, {self.matings} matings")
        print(f"         {self.animals_ate} meals, {self.eggs_hatched} hatched, {self.plants_grown} sprouted")
        print(f"{'='*55}\n")
    
    def get_growth_scale(self, entity_id: int, is_plant: bool = True) -> float:
        """Get the growth scale for rendering."""
        if is_plant:
            life = self.plant_life.get(entity_id)
            if life:
                return 0.3 + life.growth * 0.7
        else:
            life = self.animal_life.get(entity_id)
            if life:
                return 0.4 + life.growth * 0.6
        return 1.0
    
    def render_eggs(self, camera_x: float, camera_z: float):
        """Render eggs near the camera with DNA-based appearance."""
        from OpenGL.GL import (glPushMatrix, glPopMatrix, glTranslatef, 
                               glScalef, glColor3f, glColor4f, glBegin, glEnd, glVertex3f,
                               GL_TRIANGLE_FAN, GL_TRIANGLE_STRIP, GL_POINTS, GL_LINES, GL_QUADS,
                               glDisable, glEnable, GL_LIGHTING, glPointSize, glLineWidth)
        
        glDisable(GL_LIGHTING)
        
        for egg in self.eggs:
            dist = math.sqrt((egg.x - camera_x)**2 + (egg.z - camera_z)**2)
            if dist > 200:  # Increased visibility range
                continue
            
            glPushMatrix()
            glTranslatef(egg.x, egg.y + egg.size * 0.6, egg.z)
            
            # GIANT WHITE MARKER ABOVE EGG (debug visibility!)
            glColor3f(1.0, 1.0, 1.0)  # Bright white
            glLineWidth(4.0)
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 20.0, 0)  # 20 units tall white line!
            glEnd()
            
            # White diamond at top
            glBegin(GL_QUADS)
            glVertex3f(-1, 20, 0)
            glVertex3f(0, 22, 0)
            glVertex3f(1, 20, 0)
            glVertex3f(0, 18, 0)
            glEnd()
            glLineWidth(1.0)
            
            # Progress affects color saturation (more vibrant near hatching)
            progress = min(1.0, egg.hatch_timer / egg.hatch_time)
            
            # Base egg color from DNA, slightly shifting as it develops
            r = egg.base_color[0] * (1.0 - progress * 0.1)
            g = egg.base_color[1] * (1.0 - progress * 0.05)
            b = egg.base_color[2] * (1.0 + progress * 0.1)
            
            # Scale for elongation
            glScalef(egg.size * 0.7, egg.size * egg.elongation, egg.size * 0.7)
            
            # Draw main egg body
            segments = 12
            rings = 6
            
            glColor3f(r, g, b)
            
            # Draw egg as latitude rings
            for ring in range(rings):
                t1 = ring / rings
                t2 = (ring + 1) / rings
                y1 = math.cos(t1 * math.pi) * 0.5
                y2 = math.cos(t2 * math.pi) * 0.5
                r1 = math.sin(t1 * math.pi) * 0.5 * (1.0 - t1 * 0.2)  # Taper at top
                r2 = math.sin(t2 * math.pi) * 0.5 * (1.0 - t2 * 0.2)
                
                glBegin(GL_TRIANGLE_STRIP)
                for i in range(segments + 1):
                    angle = i * 2 * math.pi / segments
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    
                    # Alternate color for spots pattern
                    if egg.has_spots and (i + ring) % 3 == 0:
                        glColor3f(egg.spot_color[0], egg.spot_color[1], egg.spot_color[2])
                    elif egg.has_stripes and ring % 2 == 0:
                        glColor3f(egg.spot_color[0], egg.spot_color[1], egg.spot_color[2])
                    else:
                        glColor3f(r, g, b)
                    
                    glVertex3f(cos_a * r1, y1, sin_a * r1)
                    glVertex3f(cos_a * r2, y2, sin_a * r2)
                glEnd()
            
            # Draw wobble/pulse effect when close to hatching
            if progress > 0.8:
                wobble = math.sin(egg.hatch_timer * 10) * 0.05 * (progress - 0.8) / 0.2
                glScalef(1.0 + wobble, 1.0 - wobble * 0.5, 1.0 + wobble)
            
            glPopMatrix()
        
        glEnable(GL_LIGHTING)
    
    def cleanup_entity(self, entity_id: int, is_plant: bool = True):
        """Remove life state for an entity."""
        if is_plant:
            self.plant_life.pop(entity_id, None)
        else:
            self.animal_life.pop(entity_id, None)
    
    def get_stats(self) -> dict:
        """Get simulation statistics."""
        return {
            'total_plants': len(self.plant_life),
            'total_animals': len(self.animal_life),
            'eggs': len(self.eggs),
            'births': self.births,
            'deaths': self.deaths,
            'plants_grown': self.plants_grown
        }

