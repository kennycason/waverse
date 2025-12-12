# Waverse

An infinite procedural world built from stacked wave functions. Terrain, plants, and creatures are all generated from DNA-like structures that mutate and crossover as you explore.

![Waverse World](screenshots/screenshot01.png)

## What is this?

Waverse generates terrain by stacking sine waves, perlin noise, and other wave functions on top of each other. Think of it like a Fourier transform for landscapes. The result is an infinite world you can walk or fly through in real-time.

Plants and animals have their own DNA that controls their shape, color, and behavior. When new chunks of the world are generated, they inherit and mutate DNA from neighboring chunks. This creates gradual variation as you travel - forests slowly shift in character, creatures change form.

![Day and Night](screenshots/screenshot02.png)

## Key Features

**Terrain Generation**
- Waves stacked on waves: sin, cos, perlin, ridged noise
- Continuous infinite world via chunk system
- Background worker pre-generates chunks ahead of you

**Flora DNA**
- 15 plant types: grass, flowers, ferns, bushes, trees, pines, palms, willows, cacti, mushrooms, coral, crystals, alien forms
- Multi-segment trunks, branches, canopies
- Neighboring chunks crossover DNA to create gradual biome transitions

**Fauna DNA**
- 9 animal types: insects, birds, fish, mammals, reptiles, amphibians, jellyfish, worms, alien creatures
- Articulated bodies with animated joints
- Simple AI: wander, graze, flock, swarm, flee

**Day/Night Cycle**
- Sun and moon follow shifting orbital paths
- Sky colors transition through dawn, day, dusk, night
- Stars visible at night

![Flora Variety](screenshots/screenshot03.png)

## Running It

```bash
pip install -r requirements.txt
python main.py
```

To clear cached chunks and regenerate:
```bash
python main.py --clear-cache
```

## Controls

| Key | Action |
|-----|--------|
| WASD / Arrows | Move |
| H / Space | Fly up |
| F / Shift | Fly down / Land |
| IJKL | Look around |
| Right-click + Mouse | Mouse look |
| 1-5 | Movement speed (1=slow, 5=fast) |
| S | Save position |
| ESC | Exit |

## How DNA Crossover Works

Each chunk has a pool of plant and animal DNA templates. When a new chunk generates:

1. It looks at what DNA exists in neighboring chunks
2. It picks parents from those neighbors
3. It creates offspring via crossover (blending traits) and mutation (random changes)
4. The offspring become the species for the new chunk

This means if you walk in one direction, you'll see gradual shifts in the flora and fauna. Walk far enough and the world looks completely different.

## Project Structure

```
waverse/
  dna.py          - Plant DNA with genes for growth, color, features
  animal_dna.py   - Animal DNA with body segments, limbs, AI behavior
  flora.py        - Plant rendering with LOD
  animals.py      - Animal rendering and AI updates
  world.py        - Terrain generation from wave configs
  chunk_worker.py - Background thread for pre-generating chunks
  explorer.py     - OpenGL renderer and game loop
  sky.py          - Day/night cycle, sun, moon, stars
```

## License

MIT
