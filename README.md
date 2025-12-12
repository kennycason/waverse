# Waverse 🌊

**Infinite Wave-Based World Generator**

A procedural world generation system using stacked wave functions (Fourier-style) with DNA-based deterministic generation. Build crazy terrains from waves on waves, with support for caves, tunnels, and more.

## Features

- **Wave-Based Terrain**: Stack sine, cosine, triangle, perlin, and other wave functions to create complex terrain patterns
- **DNA System**: JSON-serializable world definitions for saving, loading, and sharing worlds
- **Infinite World**: Chunk-based generation with automatic loading/unloading
- **Beyond Heightmaps**: Vertical segment system supports caves, overhangs, and tunnels
- **Segment Merging**: Automatic optimization to reduce draw calls for flat areas
- **Digging**: Modify terrain in real-time

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default world
python main.py

# Run with psychedelic preset
python main.py --psychedelic

# 2D preview (no OpenGL needed)
python main.py --preview

# Custom seed
python main.py --seed 12345

# Load custom DNA
python main.py --dna my_world.json

# Save generated DNA
python main.py --save-dna my_world.json
```

## Controls (3D Explorer)

| Key | Action |
|-----|--------|
| WASD / Arrows | Move |
| H / Space | Fly up |
| F / Shift | Fly down |
| Right-Click + Mouse | Look around |
| IJKL | Keyboard look |
| Left-Click | Dig |
| Alt (hold) | Speed boost |
| ESC | Exit |

## DNA Structure

The world is defined by a JSON DNA structure:

```json
{
  "name": "My World",
  "seed": 42,
  "water_level": 0.0,
  "chunk_size": 32,
  "tile_size": 1.0,
  "layers": [
    {
      "name": "continental",
      "waves": [
        {"freq_x": 0.002, "freq_z": 0.002, "amplitude": 40, "wave_type": "sin"}
      ]
    }
  ],
  "features": [
    {
      "feature_type": "mountain",
      "center_x": 100,
      "center_z": 100,
      "radius": 80,
      "falloff": "gaussian",
      "height_offset": 30
    }
  ]
}
```

### Wave Types

| Type | Description |
|------|-------------|
| `sin` | Classic sine wave |
| `cos` | Classic cosine wave |
| `triangle` | Linear ramps up and down |
| `sawtooth` | Linear ramp with sharp drop |
| `square` | Binary high/low |
| `perlin` | Organic noise (requires `noise` library) |
| `simplex` | Faster variant of perlin |
| `ridged` | Sharp ridges, great for mountains |

### Falloff Types (for localized features)

| Type | Description |
|------|-------------|
| `linear` | Linear decay from center |
| `gaussian` | Smooth bell curve |
| `cosine` | Smooth S-curve |
| `smooth` | Very smooth at edges |
| `sharp` | Maintains strength until near edge |

## Architecture

```
waverse/
├── dna.py        # WaveDNA, WaveLayer, Wave, Feature classes
├── waves.py      # Wave function implementations
├── terrain.py    # TerrainColumn, TerrainTile, segment merging
├── chunk.py      # Chunk, ChunkManager for infinite world
├── mesh.py       # Mesh generation from terrain
└── renderer.py   # OpenGL rendering
```

### Key Concepts

1. **WaveDNA**: The complete genetic code of a world. Deterministic - same DNA = same world.

2. **WaveLayer**: A group of waves summed together. Layers can be:
   - Continental (low frequency, large features)
   - Regional (medium frequency, hills/valleys)
   - Detail (high frequency, texture)

3. **Feature**: A localized modification with center, radius, and falloff.

4. **TerrainColumn**: Vertical stack of segments at one (x, z) position. Supports caves via multiple segments.

5. **Chunk**: A fixed-size region of tiles. Generated on demand, cached, unloaded when far.

6. **Segment Merging**: Adjacent tiles with similar heights are merged into larger quads for rendering efficiency.

## Examples

### Create a Mountain World

```python
from waverse.dna import WaveDNA, WaveLayer, Wave, Feature

dna = WaveDNA(
    name="Mountain World",
    seed=999,
    layers=[
        WaveLayer(
            name="base",
            waves=[
                Wave(freq_x=0.005, freq_z=0.005, amplitude=30, wave_type="perlin"),
            ]
        ),
    ],
    features=[
        Feature(
            feature_type="peak",
            center_x=0, center_z=0,
            radius=200,
            falloff="gaussian",
            waves=[Wave(freq_x=0.01, freq_z=0.01, amplitude=80, wave_type="ridged")],
            height_offset=50,
        ),
    ],
)

dna.save("mountain_world.json")
```

### Generate Terrain Programmatically

```python
from waverse.dna import WaveDNA
from waverse.waves import evaluate_waves, generate_height_grid
import numpy as np

dna = WaveDNA.create_default()

# Get height at a single point
height = evaluate_waves(dna, 100.0, 50.0)[0]
print(f"Height at (100, 50): {height}")

# Generate a grid
heights = generate_height_grid(dna, x_start=-100, z_start=-100, 
                                width=200, height=200, scale=1.0)
print(f"Grid shape: {heights.shape}")
```

## Roadmap

- [ ] Cave generation with 3D wave carving
- [ ] Biome system based on height/moisture
- [ ] Fractal vegetation and creatures
- [ ] River carving algorithms
- [ ] Multiplayer chunk synchronization
- [ ] GPU-accelerated wave evaluation

## License

MIT

