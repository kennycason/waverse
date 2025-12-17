#!/usr/bin/env python3
"""
Waverse - Infinite Wave-Based World Explorer

Stack waves on waves to create terrain using Fourier-style composition.

Usage:
    python main.py                # Default world
    python main.py --islands      # Island archipelago
    python main.py --mountains    # Dramatic mountains
    python main.py --psychedelic  # Trippy interference patterns
    python main.py --seed 12345   # Custom seed

Controls:
    WASD/Arrows: Move
    H: Fly up
    F: Fly down  
    Right-Click + Mouse: Look around
    IJKL: Keyboard look
    Alt: Speed boost
    ESC: Exit
"""

import argparse
import sys
import os
import json
import shutil
import random
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waverse.world import WorldConfig, WaveConfig, CACHE_DIR
from waverse.explorer import run_explorer


def clear_cache():
    """Clear the chunk cache."""
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        print(f"Cleared cache: {CACHE_DIR}")
    else:
        print("No cache to clear.")


def main():
    parser = argparse.ArgumentParser(
        description="Waverse - Wave-Based World Explorer",
    )
    
    parser.add_argument('--seed', '-s', type=int, default=None, help='World seed (random if not specified)')
    parser.add_argument('--islands', action='store_true', help='Island world')
    parser.add_argument('--mountains', action='store_true', help='Mountain world')
    parser.add_argument('--psychedelic', '-p', action='store_true', help='Psychedelic world')
    parser.add_argument('--clear-cache', action='store_true', help='Clear chunk cache and exit')
    
    args = parser.parse_args()
    
    if args.clear_cache:
        clear_cache()
        return
    
    # Determine seed: CLI arg > saved seed > random
    if args.seed is not None:
        seed = args.seed
        print(f"  World seed: {seed} (from --seed)")
    else:
        # Try to load seed from save file first
        save_file = os.path.expanduser("~/.waverse/config.json")
        saved_seed = None
        if os.path.exists(save_file):
            try:
                with open(save_file, "r") as f:
                    saved_data = json.load(f)
                    saved_seed = saved_data.get("seed")
            except:
                pass
        
        if saved_seed is not None:
            seed = saved_seed
            print(f"  World seed: {seed} (from save file)")
        else:
            seed = int(time.time() * 1000) % (2**31)
            print(f"  World seed: {seed} (new random)")
    
    # Choose world type
    if args.islands:
        config = WorldConfig.create_islands()
    elif args.mountains:
        config = WorldConfig.create_mountains()
    elif args.psychedelic:
        config = WorldConfig.create_psychedelic()
    else:
        config = WorldConfig.create_default()
    
    # Apply seed
    config.seed = seed
    
    # Run!
    run_explorer(config)


if __name__ == '__main__':
    main()
