"""
Life Simulation System for Waverse

Handles:
- Plant growth/death based on sun/rain
- Animal growth, hunger, death
- Animal reproduction (male/female, eggs)
- Food chain (herbivores eat plants, carnivores eat animals)

Designed for efficiency - updates are batched and infrequent.
"""

import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum


# =============================================================================
# GLOBAL CONFIGURATION - Tweak these for testing!
# =============================================================================
class LifeConfig:
    """Global configuration for life simulation rates."""
    
    # Time scale: how many game-hours pass per real second
    # Normal: 0.2 (1 real second = 0.2 game hours = 12 game minutes)
    TIME_SCALE = 0.5  # Moderate speed
    
    # Plant growth rate multiplier (1.0 = normal)
    PLANT_GROWTH_RATE = 2.0  # 2x growth
    
    # Plant regrowth rate (how fast eaten parts regrow)
    PLANT_REGROWTH_RATE = 0.05  # 5% regrowth per game-hour
    
    # Plant spawn chance per update when conditions are good
    PLANT_SPAWN_CHANCE = 0.03  # 3% chance
    
    # Animal hunger rate multiplier (higher = get hungry faster)
    ANIMAL_HUNGER_RATE = 0.5  # Slower hunger - eat less often
    
    # Animal growth rate multiplier
    ANIMAL_GROWTH_RATE = 1.5  # 1.5x growth
    
    # How much of a plant an animal eats per bite (0-1)
    PLANT_BITE_SIZE = 0.15  # 15% of plant per bite
    
    # Reproduction thresholds
    REPRODUCE_HUNGER_THRESHOLD = 0.6  # Need to be fairly full
    REPRODUCE_ENERGY_THRESHOLD = 0.5
    
    # Pregnancy duration in game-hours
    PREGNANCY_DURATION = 3.0
    
    # Egg hatch time range in game-hours
    EGG_HATCH_MIN = 4.0
    EGG_HATCH_MAX = 8.0
    
    # Eating distance
    PLANT_EAT_DISTANCE = 4.0
    ANIMAL_EAT_DISTANCE = 3.0
    MATE_DISTANCE = 6.0
    
    # Logging
    LOG_ENABLED = True
    LOG_INTERVAL = 10.0  # Log stats every 10 seconds


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
    """Extended life state for plants."""
    water_level: float = 0.5  # How hydrated
    sun_exposure: float = 0.5  # How much sun received recently
    eaten_amount: float = 0.0  # 0.0 = intact, 1.0 = completely eaten
    
    def update(self, dt_hours: float, is_raining: bool, is_day: bool, sun_intensity: float = 1.0):
        """Update plant life state."""
        self.age += dt_hours
        
        # Water from rain
        if is_raining:
            self.water_level = min(1.0, self.water_level + dt_hours * 0.5)
        else:
            self.water_level = max(0.0, self.water_level - dt_hours * 0.03)
        
        # Sun exposure
        if is_day:
            self.sun_exposure = min(1.0, self.sun_exposure + dt_hours * sun_intensity * 0.3)
        else:
            self.sun_exposure = max(0.0, self.sun_exposure - dt_hours * 0.05)
        
        # Growth and regrowth requires water and sun
        if self.water_level > 0.2 and self.sun_exposure > 0.1:
            growth_rate = 0.02 * self.water_level * self.sun_exposure * LifeConfig.PLANT_GROWTH_RATE
            self.growth = min(1.0, self.growth + dt_hours * growth_rate)
            self.health = min(1.0, self.health + dt_hours * 0.03)
            
            # Regrow eaten parts
            if self.eaten_amount > 0:
                regrow_rate = LifeConfig.PLANT_REGROWTH_RATE * self.water_level * self.sun_exposure
                self.eaten_amount = max(0.0, self.eaten_amount - dt_hours * regrow_rate)
        else:
            # Slowly decline without resources
            self.health = max(0.0, self.health - dt_hours * 0.005)
        
        # Old age (very slow decline)
        if self.age > 1000:
            self.health = max(0.0, self.health - dt_hours * 0.002)
        
        # Die if completely eaten
        if self.eaten_amount >= 1.0:
            self.health = 0.0
    
    def get_visual_scale(self) -> float:
        """Get the visual scale factor based on growth and eaten amount."""
        # Base scale from growth (0.3 to 1.0)
        base_scale = 0.3 + self.growth * 0.7
        # Reduce by eaten amount (keep at least 10% if still alive)
        eaten_factor = max(0.1, 1.0 - self.eaten_amount * 0.9)
        return base_scale * eaten_factor
    
    def take_bite(self, bite_size: float = None) -> float:
        """Animal takes a bite. Returns nutrition gained."""
        if bite_size is None:
            bite_size = LifeConfig.PLANT_BITE_SIZE
        
        # Can't eat more than what's left
        available = 1.0 - self.eaten_amount
        actual_bite = min(bite_size, available)
        
        self.eaten_amount += actual_bite
        
        # Nutrition proportional to bite and plant health
        nutrition = actual_bite * 0.3 * self.health
        return nutrition


@dataclass 
class AnimalLife(LifeState):
    """Extended life state for animals."""
    gender: Gender = Gender.MALE
    diet: Diet = Diet.HERBIVORE
    hunger: float = 0.5  # 0.0 = starving, 1.0 = full
    energy: float = 0.8  # For movement/reproduction
    pregnant: bool = False
    pregnancy_timer: float = 0.0
    
    def __post_init__(self):
        # Randomize gender if not set
        if random.random() < 0.5:
            self.gender = Gender.FEMALE
    
    def update(self, dt_hours: float, is_day: bool):
        """Update animal life state."""
        self.age += dt_hours
        
        # Hunger increases over time
        base_hunger_rate = 0.03 if is_day else 0.015  # Less hungry at night
        hunger_rate = base_hunger_rate * LifeConfig.ANIMAL_HUNGER_RATE
        self.hunger = max(0.0, self.hunger - dt_hours * hunger_rate)
        
        # Energy from food
        if self.hunger > 0.4:
            self.energy = min(1.0, self.energy + dt_hours * 0.08)
        else:
            self.energy = max(0.0, self.energy - dt_hours * 0.03)
        
        # Growth when fed
        if self.hunger > 0.25 and self.growth < 1.0:
            growth_rate = 0.01 * LifeConfig.ANIMAL_GROWTH_RATE
            self.growth = min(1.0, self.growth + dt_hours * growth_rate)
        
        # Health from being fed
        if self.hunger > 0.3:
            self.health = min(1.0, self.health + dt_hours * 0.02)
        elif self.hunger < 0.15:
            # Starving
            self.health = max(0.0, self.health - dt_hours * 0.02)
        
        # Pregnancy progress
        if self.pregnant:
            self.pregnancy_timer += dt_hours
            self.hunger = max(0.0, self.hunger - dt_hours * 0.02)  # Extra hunger
        
        # Old age (varies by size - smaller animals live shorter)
        max_age = 500 + self.growth * 200
        if self.age > max_age:
            self.health = max(0.0, self.health - dt_hours * 0.005)
    
    def can_reproduce(self) -> bool:
        """Check if animal can reproduce."""
        return (self.is_mature() and 
                self.hunger > LifeConfig.REPRODUCE_HUNGER_THRESHOLD and 
                self.energy > LifeConfig.REPRODUCE_ENERGY_THRESHOLD and 
                not self.pregnant and
                self.health > 0.4)
    
    def eat(self, nutrition: float = 0.3):
        """Animal eats something."""
        self.hunger = min(1.0, self.hunger + nutrition)
        self.energy = min(1.0, self.energy + nutrition * 0.5)


@dataclass
class Egg:
    """An egg that will hatch into a new animal."""
    x: float
    y: float
    z: float
    parent1_dna: dict  # DNA from parent 1
    parent2_dna: dict  # DNA from parent 2
    hatch_timer: float = 0.0
    hatch_time: float = field(default_factory=lambda: random.uniform(
        LifeConfig.EGG_HATCH_MIN, LifeConfig.EGG_HATCH_MAX))
    size: float = 0.3  # Visual size
    
    # DNA-derived appearance (set in __post_init__)
    base_color: tuple = (0.9, 0.85, 0.7)  # Default cream
    spot_color: tuple = (0.6, 0.5, 0.4)   # Default brown spots
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
            
            # Size based on parent size
            if 'body_scale' in dna:
                self.size = 0.15 + dna['body_scale'] * 0.3
            
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
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Life states keyed by entity id
        self.plant_life: Dict[int, PlantLife] = {}
        self.animal_life: Dict[int, AnimalLife] = {}
        
        # Active eggs in the world
        self.eggs: List[Egg] = []
        
        # Pending births (chunk_key -> list of DNA dicts)
        self.pending_births: Dict[Tuple[int, int], List[dict]] = {}
        
        # Pending plant spawns
        self.pending_plants: Dict[Tuple[int, int], int] = {}  # chunk -> count to spawn
        
        # Dead entities to remove
        self.dead_plants: List[int] = []
        self.dead_animals: List[int] = []
        
        # Chunks that need display list refresh (plants changed)
        self.chunks_needing_refresh: set = set()
        
        # Update timing
        self.update_timer = 0.0
        self.update_interval = 0.3  # Seconds between updates (faster)
        
        # Logging
        self.log_timer = 0.0
        self.last_log_stats = {}
        
        # Stats
        self.births = 0
        self.deaths = 0
        self.plants_grown = 0
        self.animals_ate = 0
        self.matings = 0
        self.eggs_hatched = 0
    
    def get_or_create_plant_life(self, plant_id: int, initial_growth: float = None) -> PlantLife:
        """Get or create life state for a plant."""
        if plant_id not in self.plant_life:
            growth = initial_growth if initial_growth is not None else self.rng.uniform(0.3, 1.0)
            self.plant_life[plant_id] = PlantLife(
                growth=growth,
                health=self.rng.uniform(0.7, 1.0),
                water_level=self.rng.uniform(0.3, 0.7),
                sun_exposure=self.rng.uniform(0.3, 0.7)
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
            
            self.animal_life[animal_id] = AnimalLife(
                growth=self.rng.uniform(0.4, 0.9),  # Start more mature
                health=self.rng.uniform(0.8, 1.0),
                hunger=self.rng.uniform(0.5, 0.9),  # Start less hungry
                energy=self.rng.uniform(0.6, 1.0),
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
        
        # Get chunks near player (only simulate nearby)
        cam_cx = int(camera_x // chunk_size)
        cam_cz = int(camera_z // chunk_size)
        
        # Simulate in a small radius
        sim_radius = 3
        
        for dx in range(-sim_radius, sim_radius + 1):
            for dz in range(-sim_radius, sim_radius + 1):
                cx, cz = cam_cx + dx, cam_cz + dz
                chunk_key = (cx, cz)
                
                # Update plants in this chunk
                if chunk_key in plants_by_chunk:
                    self._update_plants(plants_by_chunk[chunk_key], dt_hours, 
                                       is_raining, is_day, sun_intensity, chunk_key)
                
                # Update animals in this chunk
                if chunk_key in animals_by_chunk:
                    self._update_animals(animals_by_chunk[chunk_key], dt_hours,
                                        is_day, plants_by_chunk.get(chunk_key, []),
                                        chunk_key)
        
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
            life = self.get_or_create_plant_life(plant_id)
            
            old_eaten = life.eaten_amount
            life.update(dt_hours, is_raining, is_day, sun_intensity)
            
            # Apply visual scale to plant
            if hasattr(plant, 'scale'):
                new_scale = life.get_visual_scale()
                if abs(plant.scale - new_scale) > 0.05:
                    plant.scale = new_scale
                    any_visual_change = True
            
            # Check for regrowth (visual change)
            if abs(old_eaten - life.eaten_amount) > 0.05:
                any_visual_change = True
            
            # Check for death - mark for removal
            if not life.is_alive():
                plants_to_remove.append(plant)
                self.dead_plants.append(plant_id)
                self.deaths += 1
                if LifeConfig.LOG_ENABLED:
                    cause = "eaten" if life.eaten_amount >= 1.0 else "withered"
                    print(f"  [LIFE] Plant {cause} at age {life.age:.1f}h")
        
        # Actually remove dead plants from the list
        for dead_plant in plants_to_remove:
            if dead_plant in plants:
                plants.remove(dead_plant)
                any_visual_change = True
        
        # Mark chunk for refresh if anything changed visually
        if any_visual_change:
            self.chunks_needing_refresh.add(chunk_key)
        
        # Chance to spawn new plant if conditions good
        if (is_raining or sun_intensity > 0.5) and len(plants) < 60:
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
            
            life.update(dt_hours, is_day)
            
            # Apply growth to visual scale
            if hasattr(animal, 'scale'):
                animal.scale = 0.4 + life.growth * 0.6
            
            # Eating behavior
            if life.hunger < 0.7 and hasattr(animal, 'x'):
                if life.diet in (Diet.HERBIVORE, Diet.OMNIVORE) and plants:
                    # Try to eat nearby plant (only if small enough to eat!)
                    for plant in plants[:10]:  # Check more plants
                        if hasattr(plant, 'x'):
                            # Size check: animal can only eat plants smaller than itself
                            plant_height = getattr(plant.dna, 'height_gene', None)
                            plant_size = plant_height.value if plant_height else 2.0
                            animal_size = life.growth * 3.0  # Rough animal size
                            
                            # Can't eat trees if you're small (grass/fern ok)
                            if plant_size > animal_size * 2:
                                continue  # Plant too big to eat
                            
                            # Skip if plant already mostly eaten
                            plant_life = self.get_or_create_plant_life(id(plant))
                            if plant_life.eaten_amount > 0.8:
                                continue  # Not much left to eat
                            
                            dist = math.sqrt((animal.x - plant.x)**2 + (animal.z - plant.z)**2)
                            if dist < LifeConfig.PLANT_EAT_DISTANCE:
                                # Take a bite! Nutrition based on plant health
                                nutrition = plant_life.take_bite()
                                life.eat(nutrition)
                                self.animals_ate += 1
                                
                                # Mark chunk for visual refresh
                                cx = int(plant.x // 128)
                                cz = int(plant.z // 128)
                                self.chunks_needing_refresh.add((cx, cz))
                                
                                if LifeConfig.LOG_ENABLED and self.rng.random() < 0.2:
                                    eaten_pct = plant_life.eaten_amount * 100
                                    print(f"  [LIFE] {animal_type} nibbled plant - {eaten_pct:.0f}% eaten")
                                break
                
                if life.diet in (Diet.CARNIVORE, Diet.OMNIVORE):
                    # Try to eat smaller animals
                    for other, ox, oz, other_id in animal_positions:
                        if other_id == animal_id:
                            continue
                        other_life = self.animal_life.get(other_id)
                        if other_life and other_life.growth < life.growth * 0.7:
                            dist = math.sqrt((animal.x - ox)**2 + (animal.z - oz)**2)
                            if dist < LifeConfig.ANIMAL_EAT_DISTANCE:
                                life.eat(0.5)
                                other_life.health = 0  # Kill prey
                                self.animals_ate += 1
                                if LifeConfig.LOG_ENABLED:
                                    print(f"  [LIFE] {animal_type} ate another animal!")
                                break
            
            # Reproduction
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
                        if dist < LifeConfig.MATE_DISTANCE:
                            # Mate!
                            life.pregnant = True
                            life.pregnancy_timer = 0.0
                            other_life.energy -= 0.2
                            self.matings += 1
                            if LifeConfig.LOG_ENABLED:
                                print(f"  [LIFE] {animal_type} mated! Now pregnant.")
                            break
            
            # Give birth (lay egg)
            if life.pregnant and life.pregnancy_timer >= LifeConfig.PREGNANCY_DURATION:
                life.pregnant = False
                # Create egg
                if hasattr(animal, 'x') and hasattr(animal, 'dna'):
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
                    cause = "starvation" if life.hunger < 0.1 else "old age" if life.age > 300 else "unknown"
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
        
        # Queue hatched eggs for animal spawning
        for egg in hatched:
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
        hungry = sum(1 for a in self.animal_life.values() if a.hunger < 0.3)
        
        # Average stats
        avg_plant_growth = sum(p.growth for p in self.plant_life.values()) / max(1, len(self.plant_life))
        avg_animal_hunger = sum(a.hunger for a in self.animal_life.values()) / max(1, len(self.animal_life))
        
        weather = "🌧️ Rain" if is_raining else ("☀️ Day" if is_day else "🌙 Night")
        
        print(f"\n{'='*50}")
        print(f"  [LIFE SIMULATION] {weather} (Sun: {sun_intensity:.1f})")
        print(f"{'='*50}")
        print(f"  Plants: {living_plants} ({mature_plants} mature) | Avg growth: {avg_plant_growth:.1%}")
        print(f"  Animals: {living_animals} ({mature_animals} mature) | {pregnant} pregnant | {hungry} hungry")
        print(f"  Avg hunger: {avg_animal_hunger:.1%}")
        print(f"  Eggs: {len(self.eggs)}")
        print(f"  Stats: {self.births} births, {self.deaths} deaths, {self.matings} matings")
        print(f"         {self.animals_ate} meals eaten, {self.eggs_hatched} eggs hatched")
        print(f"         {self.plants_grown} plants sprouted")
        print(f"{'='*50}\n")
    
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
                               GL_TRIANGLE_FAN, GL_TRIANGLE_STRIP, GL_POINTS,
                               glDisable, glEnable, GL_LIGHTING, glPointSize)
        
        glDisable(GL_LIGHTING)
        
        for egg in self.eggs:
            dist = math.sqrt((egg.x - camera_x)**2 + (egg.z - camera_z)**2)
            if dist > 100:
                continue
            
            glPushMatrix()
            glTranslatef(egg.x, egg.y + egg.size * 0.6, egg.z)
            
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

