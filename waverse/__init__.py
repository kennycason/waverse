"""
Waverse - Infinite Wave-Based World Generator

Stack waves on waves (Fourier-style) for procedural terrain generation.
"""

__version__ = "0.2.0"

from .world import (
    WorldConfig,
    WaveConfig,
    ChunkManager,
    Chunk,
    get_height,
    CHUNK_SIZE,
    TILE_SCALE,
)

from .explorer import run_explorer

__all__ = [
    "WorldConfig",
    "WaveConfig", 
    "ChunkManager",
    "Chunk",
    "get_height",
    "run_explorer",
    "CHUNK_SIZE",
    "TILE_SCALE",
]
