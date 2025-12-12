#!/usr/bin/env python3
"""
Render all cached chunks as a giant PNG world map.

Reads .chunk_cache/ and produces a bird's-eye view of the generated world.
Black areas indicate chunks that haven't been generated yet.

Usage:
    python tools/render_world_map.py
    python tools/render_world_map.py --output world_map.png
    python tools/render_world_map.py --seed 42  # specific seed folder
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_cache_dirs(base_path: Path) -> list:
    """Find all seed_* directories in cache."""
    if not base_path.exists():
        return []
    return sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("seed_")])


def load_chunks(cache_dir: Path) -> dict:
    """Load all chunk heightmaps from a cache directory."""
    chunks = {}
    
    for chunk_file in cache_dir.glob("chunk_*.npy"):
        # Parse chunk coordinates from filename: chunk_X_Z.npy
        name = chunk_file.stem  # chunk_X_Z
        parts = name.split("_")
        if len(parts) == 3:
            try:
                cx = int(parts[1])
                cz = int(parts[2])
                heightmap = np.load(chunk_file)
                chunks[(cx, cz)] = heightmap
            except (ValueError, Exception) as e:
                print(f"  Skipping {chunk_file.name}: {e}")
    
    return chunks


def height_to_color(h: float) -> tuple:
    """Convert height to RGB color (0-255)."""
    if h < 0:
        # Deep water - dark blue
        t = max(0, min(1, (h + 20) / 20))
        return (int(20 + t * 30), int(40 + t * 60), int(120 + t * 40))
    elif h < 2:
        # Shallow water - lighter blue
        return (60, 100, 180)
    elif h < 4:
        # Beach - tan/sand
        return (210, 190, 140)
    elif h < 15:
        # Grass - green
        t = (h - 4) / 11
        return (int(80 - t * 20), int(160 - t * 30), int(80 - t * 20))
    elif h < 25:
        # Forest - dark green
        t = (h - 15) / 10
        return (int(60 - t * 10), int(130 - t * 30), int(60 - t * 10))
    elif h < 35:
        # Hills - brown/tan
        t = (h - 25) / 10
        return (int(120 + t * 30), int(100 + t * 20), int(70 + t * 10))
    elif h < 50:
        # Mountain - gray
        t = (h - 35) / 15
        return (int(150 - t * 30), int(140 - t * 30), int(130 - t * 20))
    else:
        # Snow - white
        t = min(1, (h - 50) / 20)
        return (int(200 + t * 55), int(200 + t * 55), int(210 + t * 45))


def render_chunk(heightmap: np.ndarray, scale: int = 1) -> np.ndarray:
    """Render a single chunk as an RGB image."""
    h, w = heightmap.shape
    
    if scale > 1:
        # Downsample by taking every Nth point
        heightmap = heightmap[::scale, ::scale]
        h, w = heightmap.shape
    
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for z in range(h):
        for x in range(w):
            color = height_to_color(heightmap[z, x])
            img[z, x] = color
    
    return img


def render_world_map(chunks: dict, scale: int = 2) -> np.ndarray:
    """Render all chunks into a single world map image."""
    if not chunks:
        print("No chunks to render!")
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Find bounds
    min_cx = min(cx for cx, cz in chunks.keys())
    max_cx = max(cx for cx, cz in chunks.keys())
    min_cz = min(cz for cx, cz in chunks.keys())
    max_cz = max(cz for cx, cz in chunks.keys())
    
    print(f"  Chunk range: X=[{min_cx}, {max_cx}], Z=[{min_cz}, {max_cz}]")
    
    # Get chunk dimensions from first chunk
    sample_chunk = next(iter(chunks.values()))
    chunk_h, chunk_w = sample_chunk.shape
    chunk_h //= scale
    chunk_w //= scale
    
    # Calculate image size
    num_chunks_x = max_cx - min_cx + 1
    num_chunks_z = max_cz - min_cz + 1
    img_w = num_chunks_x * chunk_w
    img_h = num_chunks_z * chunk_h
    
    print(f"  Image size: {img_w} x {img_h} pixels")
    print(f"  Total chunks: {len(chunks)} / {num_chunks_x * num_chunks_z} possible")
    
    # Create black background
    world_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    
    # Render each chunk
    for (cx, cz), heightmap in chunks.items():
        chunk_img = render_chunk(heightmap, scale)
        
        # Position in image (flip Z for top-down view)
        img_x = (cx - min_cx) * chunk_w
        img_z = (cz - min_cz) * chunk_h
        
        # Handle edge cases where chunk might be slightly different size
        actual_h = min(chunk_img.shape[0], img_h - img_z)
        actual_w = min(chunk_img.shape[1], img_w - img_x)
        
        # Copy into world image
        world_img[img_z:img_z + actual_h, 
                  img_x:img_x + actual_w] = chunk_img[:actual_h, :actual_w]
    
    return world_img


def main():
    parser = argparse.ArgumentParser(description="Render cached chunks as a world map PNG")
    parser.add_argument("--output", "-o", default="world_map.png", help="Output file path")
    parser.add_argument("--seed", type=int, help="Specific seed folder to use")
    parser.add_argument("--scale", type=int, default=2, help="Downscale factor (1=full, 2=half, etc.)")
    parser.add_argument("--cache-dir", default=".chunk_cache", help="Cache directory path")
    args = parser.parse_args()
    
    # Find cache directory
    cache_base = Path(args.cache_dir)
    if not cache_base.exists():
        cache_base = Path(__file__).parent.parent / ".chunk_cache"
    
    if not cache_base.exists():
        print(f"Error: Cache directory not found: {cache_base}")
        print("Run the game first to generate some chunks!")
        return 1
    
    # Find seed directories
    seed_dirs = find_cache_dirs(cache_base)
    if not seed_dirs:
        print(f"Error: No seed folders found in {cache_base}")
        return 1
    
    # Select seed directory
    if args.seed:
        cache_dir = cache_base / f"seed_{args.seed}"
        if not cache_dir.exists():
            print(f"Error: Seed folder not found: {cache_dir}")
            print(f"Available: {[d.name for d in seed_dirs]}")
            return 1
    else:
        cache_dir = seed_dirs[-1]  # Use most recent
        print(f"Using cache: {cache_dir.name}")
    
    # Load chunks
    print(f"Loading chunks from {cache_dir}...")
    chunks = load_chunks(cache_dir)
    print(f"  Loaded {len(chunks)} chunks")
    
    if not chunks:
        print("No chunks found!")
        return 1
    
    # Render
    print("Rendering world map...")
    world_img = render_world_map(chunks, scale=args.scale)
    
    # Save
    try:
        from PIL import Image
        img = Image.fromarray(world_img)
        img.save(args.output)
        print(f"Saved to: {args.output}")
    except ImportError:
        # Fallback to matplotlib
        import matplotlib.pyplot as plt
        plt.imsave(args.output, world_img)
        print(f"Saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

