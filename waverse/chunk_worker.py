"""
Background Chunk Generation Worker.

Runs in a separate thread/process to pre-generate chunks ahead of the player.
This ensures smooth gameplay without loading hitches.
"""

import threading
import queue
import time
from typing import Set, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np

from .world import WorldConfig, generate_chunk, Chunk, CHUNK_SIZE, TILE_SCALE


@dataclass(order=True)
class ChunkRequest:
    """Request to generate a chunk."""
    priority: float  # Lower = higher priority (must be first for ordering!)
    cx: int = 0
    cz: int = 0


class ChunkWorker:
    """
    Background worker that pre-generates chunks around the player.
    
    Features:
    - Runs in background thread
    - Prioritizes chunks by distance to player
    - Keeps N chunks radius always ready
    - Never blocks the main thread
    """
    
    def __init__(self, config: WorldConfig, preload_radius: int = 25):
        self.config = config
        self.preload_radius = preload_radius
        
        # Generated chunks cache
        self.chunks: Dict[Tuple[int, int], Chunk] = {}
        self.chunks_lock = threading.Lock()
        
        # Generation queue
        self.request_queue = queue.PriorityQueue()
        self.pending: Set[Tuple[int, int]] = set()
        self.pending_lock = threading.Lock()
        
        # Player position (updated from main thread)
        self.player_chunk = (0, 0)
        self.player_lock = threading.Lock()
        
        # Worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.scanner_thread = threading.Thread(target=self._scanner_loop, daemon=True)
        
        # Stats
        self.chunks_generated = 0
        self.chunks_cached = 0
    
    def start(self):
        """Start the background workers."""
        self.worker_thread.start()
        self.scanner_thread.start()
        print(f"  [ChunkWorker] Started with radius {self.preload_radius}")
    
    def stop(self):
        """Stop the workers."""
        self.running = False
        # Add poison pill to unblock queue
        self.request_queue.put((float('inf'), ChunkRequest(0, 0, float('inf'))))
    
    def update_player_position(self, cx: int, cz: int):
        """Update player chunk position (called from main thread)."""
        with self.player_lock:
            self.player_chunk = (cx, cz)
    
    def get_chunk(self, cx: int, cz: int) -> Optional[Chunk]:
        """
        Get a chunk if it's ready, None otherwise.
        Main thread should fall back to sync generation if None.
        """
        key = (cx, cz)
        with self.chunks_lock:
            if key in self.chunks:
                return self.chunks[key]
        
        # Request it if not pending
        self._request_chunk(cx, cz, urgent=True)
        return None
    
    def get_chunk_or_generate(self, cx: int, cz: int) -> Chunk:
        """Get a chunk, generating synchronously if not cached."""
        key = (cx, cz)
        
        # Check cache first
        with self.chunks_lock:
            if key in self.chunks:
                return self.chunks[key]
        
        # Generate synchronously
        chunk = generate_chunk(self.config, cx, cz)
        
        with self.chunks_lock:
            self.chunks[key] = chunk
        
        return chunk
    
    def _request_chunk(self, cx: int, cz: int, urgent: bool = False):
        """Add a chunk to the generation queue."""
        key = (cx, cz)
        
        with self.pending_lock:
            if key in self.pending:
                return
            self.pending.add(key)
        
        # Calculate priority (distance to player)
        with self.player_lock:
            pcx, pcz = self.player_chunk
        
        dist = abs(cx - pcx) + abs(cz - pcz)
        priority = 0 if urgent else dist
        
        self.request_queue.put((priority, ChunkRequest(priority, cx, cz)))
    
    def _worker_loop(self):
        """Background worker that generates chunks."""
        while self.running:
            try:
                # Get next request (blocks if empty)
                priority, request = self.request_queue.get(timeout=0.5)
                
                if not self.running:
                    break
                
                key = (request.cx, request.cz)
                
                # Skip if already generated
                with self.chunks_lock:
                    if key in self.chunks:
                        with self.pending_lock:
                            self.pending.discard(key)
                        continue
                
                # Generate the chunk
                chunk = generate_chunk(self.config, request.cx, request.cz)
                
                # Store it
                with self.chunks_lock:
                    self.chunks[key] = chunk
                    self.chunks_generated += 1
                
                with self.pending_lock:
                    self.pending.discard(key)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"  [ChunkWorker] Error: {e}")
    
    def _scanner_loop(self):
        """Scans for chunks that need generation around the player."""
        while self.running:
            try:
                with self.player_lock:
                    pcx, pcz = self.player_chunk
                
                # Spiral outward from player
                for radius in range(1, self.preload_radius + 1):
                    if not self.running:
                        break
                    
                    for dx in range(-radius, radius + 1):
                        for dz in range(-radius, radius + 1):
                            if abs(dx) != radius and abs(dz) != radius:
                                continue  # Only perimeter
                            
                            cx, cz = pcx + dx, pcz + dz
                            key = (cx, cz)
                            
                            with self.chunks_lock:
                                if key in self.chunks:
                                    continue
                            
                            self._request_chunk(cx, cz)
                
                # Cleanup distant chunks
                self._cleanup_distant()
                
                # Sleep a bit before next scan
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  [ChunkWorker] Scanner error: {e}")
    
    def _cleanup_distant(self):
        """Remove chunks that are too far from player."""
        with self.player_lock:
            pcx, pcz = self.player_chunk
        
        max_dist = self.preload_radius + 5
        to_remove = []
        
        with self.chunks_lock:
            for (cx, cz) in self.chunks:
                if abs(cx - pcx) > max_dist or abs(cz - pcz) > max_dist:
                    to_remove.append((cx, cz))
            
            for key in to_remove:
                del self.chunks[key]
                self.chunks_cached -= 1
    
    def get_stats(self) -> Dict:
        """Get worker statistics."""
        with self.chunks_lock:
            cached = len(self.chunks)
        with self.pending_lock:
            pending = len(self.pending)
        
        return {
            "cached": cached,
            "generated": self.chunks_generated,
            "pending": pending,
            "queue_size": self.request_queue.qsize()
        }

