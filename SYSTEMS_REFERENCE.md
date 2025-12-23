# Waverse Systems Reference

## Structure Types

### Buildings (Procedurally Generated)
- **Floors** - Horizontal platforms at various heights
- **Walls** - Vertical surfaces with thickness
- **Pillars** - Vertical columns supporting floors (y_bottom to y_top)
- **Ramps/Staircases** - Angled surfaces connecting floor levels
- **Doors** - Openings in walls
- **Roofs** - Angled top surfaces

### Parkour Elements
- **Parkour Tiles** - Intentionally floating platforms for jumping challenges
- These are meant to be disconnected/floating - NOT a bug!

### Special Structures
- Procedurally placed based on terrain and biome

### Building Types (from building_dna.py)

| Category | Types |
|----------|-------|
| **Residential** | house, apartment, villa, cabin, modern_house, split_level, ranch, bungalow, townhouse, loft, penthouse, duplex, mansion, cottage, beach_house, tree_house, tiny_house, a_frame, container_house, geodome_home, floating_house, underground_home, capsule, greenhouse |
| **Commercial** | shop, office, tower, hotel, hospital, restaurant |
| **Industrial** | warehouse, factory, hangar, silo, powerplant |
| **Sci-Fi** | skyscraper, planetarium, solar_station, landing_pad, spaceport, research_lab, modular_hub, power_node, comm_tower, shield_generator, cryo_chamber, cargo_bay, docking_arm, reactor_core |
| **Recreation** | playground, stadium, pool |
| **Special** | temple, monument, observatory, ruins, parkour, lighthouse, bridge, gazebo |
| **Ancient/Cultural** | nubian_complex, pyramid_temple, stone_portal, ziggurat, obelisk, amphitheater, colosseum, aqueduct, mausoleum, pagoda_tower |
| **Cyberpunk** | cyberpunk_bar, neon_diner, arcade, holo_billboard, tech_shop |
| **City Infrastructure** | bus_stop, metro_entrance, parking_garage, gas_station, fire_station, police_station, post_office, library, school, church, mosque, bank, theater, mall, supermarket |

### Roof Types
| Type | Description |
|------|-------------|
| `flat` | Modern flat with parapet |
| `gabled` | Classic triangular |
| `hipped` | All sides slope |
| `pyramid` | Four-sided pyramid |
| `shed` | Single slope |
| `dome` | Hemispherical |
| `barrel` | Cylindrical/curved |
| `stepped` | Ziggurat style |
| `sawtooth` | Industrial zigzag |
| `conical` | Pointed tower style |
| `pagoda` | Multi-tiered Asian |
| `mansard` | French two-slope |
| `butterfly` | V-shaped modern |
| `geodesic` | Triangulated dome |

---

## Life Types (Animals)

### Animal Categories (from DNA)
| Type | Description |
|------|-------------|
| `worm` | Simple ground-dwelling creature |
| `mammal` | Four-legged land animal (various sizes) |
| `bird` | Flying creature with wings |
| `fish` | Water-dwelling creature |
| `insect` | Small multi-legged creature |

### Animal DNA Attributes
- `animal_type` - Category (worm, mammal, bird, fish, insect)
- `size` - Scale gene
- `primary_color` - Main body color (ColorGene)
- `secondary_color` - Pattern/accent color
- `pattern_type` - solid, spotted, striped, patched

---

## Flora Types (Plants)

### Plant Categories (from DNA)
| Type | Description |
|------|-------------|
| `tree` | Large woody plant with trunk |
| `pine` | Coniferous tree |
| `oak` | Deciduous tree with dome canopy |
| `willow` | Tree with weeping branches |
| `maple` | Tree with dome canopy |
| `palm` | Tropical tree with fronds |
| `bush` | Low shrubby plant |
| `shrub` | Similar to bush |
| `grass` | Ground cover |
| `fern` | Leafy ground plant |
| `flower` | Flowering plant |
| `mushroom` | Fungal growth |

### Canopy Shapes (DNA attribute)
| Shape | Description | Modern Mesh |
|-------|-------------|-------------|
| `cone` | Triangular pine shape | `tree` |
| `dome` | Round layered | `tree_dome` |
| `sphere` | Round | `tree_dome` (fallback) |
| `layered` | Stacked | `tree_dome` (fallback) |
| `umbrella` | Flat acacia style | `tree_umbrella` |
| `weeping` | Droopy willow | `tree_weeping` |
| `cascading` | Droopy | `tree_weeping` (fallback) |
| `columnar` | Tall narrow cypress | `tree_columnar` |
| `explosion` | Burst pattern | `tree` (fallback) |

### Plant DNA Attributes
- `plant_type` - Category
- `canopy_shape` - Shape of foliage
- `height_gene` - Size (value property)
- `width_gene` - Spread
- `leaf_color` - Foliage color (ColorGene)
- `trunk_color` - Bark color
- `flower_color` - Bloom color
- `has_flowers` - Boolean
- `has_glow` - Boolean (bioluminescent)
- `leaf_density` - 0-1
- `droop` - How much branches hang

---

## Biome Types (from ClimateManager)

| Biome | Temperature | Humidity | Building Styles |
|-------|-------------|----------|-----------------|
| `Tundra` | < 0.25 | > 0.5 | cabin, warehouse, research_lab |
| `Frozen` | < 0.25 | ≤ 0.5 | cabin, factory, observatory |
| `Taiga` | 0.25-0.45 | > 0.6 | cabin, a_frame, cottage |
| `Cold` | 0.25-0.45 | ≤ 0.6 | house, apartment, factory |
| `Rainforest` | 0.45-0.65 | > 0.7 | tree_house, temple, ruins |
| `Temperate` | 0.45-0.65 | 0.4-0.7 | house, modern_house, shop |
| `Grassland` | 0.45-0.65 | < 0.4 | ranch, bungalow, gazebo |
| `Tropical` | > 0.65 | > 0.6 | beach_house, villa, floating_house |
| `Desert` | > 0.65 | < 0.3 | nubian_complex, pyramid_temple, ziggurat |
| `Savanna` | > 0.65 | 0.3-0.6 | house, temple, monument |
| `Urban/City` | - | - | skyscraper, parking_garage, mall |
| `Cyberpunk/Neon` | - | - | cyberpunk_bar, neon_diner, arcade |
| `Ancient/Historical` | - | - | colosseum, amphitheater, aqueduct |
| `Futuristic/SciFi` | - | - | modular_hub, reactor_core, spaceport |

---

## Weather Types

| Type | Description |
|------|-------------|
| `none` | Clear weather |
| `rain` | Standard precipitation |
| `heavy_rain` | Intense rain |
| `storm` | Rain with lightning |
| `snow` | Cold precipitation |

### Weather DNA Attributes
- `precipitation` - 0-1 intensity
- `precipitation_type` - none, rain, snow
- `cloud_cover` - 0-1
- `wind_strength` - 0-1
- `lightning_active` - Boolean

---

## Rendering Systems

### Modern Renderer Components
| Component | File | Purpose |
|-----------|------|---------|
| Terrain | `modern_terrain.py` | VBO-based terrain chunks |
| Flora | `modern_flora.py` | Instanced plants |
| Animals | `modern_animals.py` | Instanced creatures |
| Structures | `modern_structures.py` | Buildings & parkour |
| Water | `modern_water.py` | Animated water surface |
| Sky | `modern_sky.py` | Day/night cycle |
| Clouds | `modern_clouds.py` | Volumetric ray-marched |
| Weather | `modern_weather.py` | Rain/snow particles |
| HUD | `modern_hud.py` | Crosshair, coords, FPS |
| Integration | `modern_integration.py` | Bridges legacy managers |

### Render Distances (Current)
- Terrain: 18 chunks
- Flora: 12 chunks
- Animals: 10 chunks
- Fog: 400-900 units

---

## Key Files

| File | Purpose |
|------|---------|
| `explorer.py` | Main game loop, camera, controls |
| `chunk.py` | Terrain chunk management |
| `flora.py` | Legacy plant manager |
| `life.py` | Animal simulation |
| `dna.py` | PlantDNA, AnimalDNA definitions |
| `climate.py` | BiomeDNA, WeatherState |
| `structure.py` | Building generation |

---

## Controls

### Keyboard
- WASD/Arrows: Movement
- Space/H: Up | Shift/F: Down
- IJKL: Camera rotation
- B (hold): Run
- R: Action/Use tool
- Tab: Menu
- F3: Perf stats | F4: Overlay

### Gamepad
- Left stick: Move
- Right stick: Camera
- A: Jump (variable height)
- B: Action
- R2/L2: Speed up/down
- D-pad: Tool switching

