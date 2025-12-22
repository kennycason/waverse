"""
ModernGL Integration Layer

Bridges the existing waverse systems (chunks, flora, fauna) with
the new modern renderers. Allows incremental migration.
"""

import numpy as np
import moderngl
try:
    from pyglm import glm
except ImportError:
    import glm
import pygame
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import math
import time

from .modern_terrain import ModernTerrainRenderer
from .modern_flora import ModernFloraRenderer
from .modern_animals import ModernAnimalRenderer, get_animal_type_id
from .modern_water import ModernWaterRenderer
from .modern_sky import ModernSkyRenderer
from .modern_structures import ModernStructureRenderer
from .modern_weather import ModernWeatherRenderer


# =============================================================================
# COMBINED RENDERER
# =============================================================================

class ModernWorldRenderer:
    """
    Combined modern renderer for terrain + flora + fauna.
    
    Integrates with existing waverse systems:
    - ChunkManager for heightmaps
    - FloraManager for plant data
    - AnimalManager for fauna data (TODO)
    - Camera for view state
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # Sub-renderers
        self.sky = ModernSkyRenderer(ctx)  # Rendered first (background)
        self.terrain = ModernTerrainRenderer(ctx)
        self.structures = ModernStructureRenderer(ctx)  # Buildings, before flora
        self.flora = ModernFloraRenderer(ctx)
        self.animals = ModernAnimalRenderer(ctx)
        self.water = ModernWaterRenderer(ctx)
        self.weather = ModernWeatherRenderer(ctx)  # Rendered last (particles)
        
        # Integration state
        self.loaded_terrain_chunks: set = set()
        self.loaded_flora_chunks: set = set()
        
        # Performance tracking
        self.frame_stats = {
            'terrain_chunks': 0,
            'terrain_tris': 0,
            'structure_count': 0,
            'structure_tris': 0,
            'flora_instances': 0,
            'animal_instances': 0,
            'water_tris': 0,
            'weather_particles': 0,
            'total_draw_calls': 0,
            'frame_time_ms': 0,
        }
        
        # Current structure list (updated each frame)
        self._visible_structures: List[Any] = []
        
        # Render settings
        self.render_distance = 12  # Chunks
        self.flora_render_distance = 8  # Chunks
        self.animal_render_distance = 10  # Chunks
        
        # Chunk size info (set by waverse)
        self.chunk_size = 64  # Grid cells per chunk
        self.tile_scale = 2.0  # World units per cell
        self.height_scale = 1.0
    
    def set_chunk_params(self, chunk_size: int, tile_scale: float, height_scale: float):
        """Set chunk generation parameters."""
        self.chunk_size = chunk_size
        self.tile_scale = tile_scale
        self.height_scale = height_scale
    
    def set_camera_from_waverse(self, camera: Any, aspect: float = 16/9):
        """
        Update renderer camera from waverse Camera object.
        
        Args:
            camera: waverse Camera instance with x, y, z, yaw, pitch attributes
            aspect: Screen aspect ratio
        """
        # Get camera values
        x, y, z = camera.x, camera.y, camera.z
        yaw, pitch = camera.yaw, camera.pitch
        fov = getattr(camera, 'fov', 60.0)
        
        # Update terrain renderer
        self.terrain.set_camera(x, y, z, yaw, pitch, fov, aspect)
        
        # Update flora renderer  
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        
        dir_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
        dir_y = math.sin(pitch_rad)
        dir_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        
        projection = glm.perspective(glm.radians(fov), aspect, 0.1, 1000.0)
        cam_pos = glm.vec3(x, y, z)
        target = cam_pos + glm.vec3(dir_x, dir_y, dir_z)
        view = glm.lookAt(cam_pos, target, glm.vec3(0, 1, 0))
        
        self.sky.set_camera(projection, view, cam_pos)
        self.structures.set_camera(projection, view, cam_pos)
        self.flora.set_camera(projection, view, cam_pos)
        self.animals.set_camera(projection, view, cam_pos)
        self.water.set_camera(projection, view, cam_pos)
        self.weather.set_camera(projection, view, cam_pos)
    
    def load_terrain_chunk(self, cx: int, cz: int, chunk_manager: Any,
                           biome: str = 'grassland'):
        """
        Load a terrain chunk from the ChunkManager.
        
        Args:
            cx, cz: Chunk coordinates
            chunk_manager: waverse ChunkManager with get_chunk() method
        """
        key = (cx, cz)
        if key in self.loaded_terrain_chunks:
            return
        
        # Get chunk from ChunkManager
        chunk = chunk_manager.get_chunk(cx, cz)
        if chunk is None:
            return
        
        # Get heightmap
        heightmap = chunk.heightmap  # This is a numpy array
        
        # Calculate world position
        chunk_world_x = cx * self.chunk_size * self.tile_scale
        chunk_world_z = cz * self.chunk_size * self.tile_scale
        
        # Create GPU mesh
        self.terrain.create_chunk_mesh(
            cx, cz, heightmap,
            self.tile_scale, self.height_scale,
            chunk_world_x, chunk_world_z,
            biome
        )
        
        self.loaded_terrain_chunks.add(key)
    
    def unload_terrain_chunk(self, cx: int, cz: int):
        """Unload a terrain chunk from GPU."""
        key = (cx, cz)
        if key in self.loaded_terrain_chunks:
            self.terrain.remove_chunk(cx, cz)
            self.loaded_terrain_chunks.discard(key)
    
    def load_flora_for_chunk(self, cx: int, cz: int, flora_manager: Any):
        """
        Load flora instances from a chunk.
        
        Args:
            cx, cz: Chunk coordinates
            flora_manager: waverse FloraManager with chunk_plants dict
        """
        key = (cx, cz)
        if key in self.loaded_flora_chunks:
            return
        
        # Get plants from flora manager
        plants = flora_manager.chunk_plants.get(key, [])
        
        for plant in plants:
            # Extract plant data
            x = plant.x
            y = plant.y
            z = plant.z
            scale = getattr(plant, 'scale', 1.0)
            rotation = getattr(plant, 'rotation', 0.0)
            
            # Determine mesh type from plant DNA
            dna = getattr(plant, 'dna', None)
            if dna:
                plant_type = getattr(dna, 'plant_type', 'bush')
                # Map waverse plant types to our mesh types
                if plant_type in ('tree', 'pine', 'oak', 'palm'):
                    mesh_type = 'tree'
                elif plant_type in ('bush', 'shrub'):
                    mesh_type = 'bush'
                elif plant_type in ('grass', 'fern', 'reed'):
                    mesh_type = 'grass'
                elif plant_type in ('flower', 'mushroom'):
                    mesh_type = 'flower'
                else:
                    mesh_type = 'bush'
                
                # Get color from DNA
                color = getattr(dna, 'leaf_color', (0.3, 0.6, 0.3))
                r, g, b = color[:3] if len(color) >= 3 else (1, 1, 1)
            else:
                mesh_type = 'bush'
                r, g, b = 1.0, 1.0, 1.0
            
            self.flora.add_instance(mesh_type, x, y, z, scale, rotation, r, g, b)
        
        self.loaded_flora_chunks.add(key)
    
    def update_chunks_around_camera(self, camera: Any, chunk_manager: Any,
                                    flora_manager: Any = None,
                                    animal_manager: Any = None,
                                    structure_manager: Any = None):
        """
        Load/unload chunks based on camera position.
        
        Args:
            camera: waverse Camera
            chunk_manager: waverse ChunkManager
            flora_manager: waverse FloraManager (optional)
            animal_manager: waverse AnimalManager (optional)
        """
        # Calculate camera chunk
        chunk_world_size = self.chunk_size * self.tile_scale
        cam_cx = int(camera.x // chunk_world_size)
        cam_cz = int(camera.z // chunk_world_size)
        
        # Determine which chunks should be loaded
        needed_terrain = set()
        needed_flora = set()
        
        for dx in range(-self.render_distance, self.render_distance + 1):
            for dz in range(-self.render_distance, self.render_distance + 1):
                dist = math.sqrt(dx * dx + dz * dz)
                
                cx, cz = cam_cx + dx, cam_cz + dz
                
                if dist <= self.render_distance:
                    needed_terrain.add((cx, cz))
                
                if dist <= self.flora_render_distance:
                    needed_flora.add((cx, cz))
        
        # Load new terrain chunks
        for key in needed_terrain - self.loaded_terrain_chunks:
            self.load_terrain_chunk(key[0], key[1], chunk_manager)
        
        # Unload distant terrain chunks
        for key in self.loaded_terrain_chunks - needed_terrain:
            self.unload_terrain_chunk(key[0], key[1])
        
        # Handle flora if manager provided
        if flora_manager:
            # Clear and rebuild flora (simpler than tracking deltas)
            if needed_flora != self.loaded_flora_chunks:
                self.flora.clear_instances()
                self.loaded_flora_chunks.clear()
                
                for key in needed_flora:
                    self.load_flora_for_chunk(key[0], key[1], flora_manager)
                
                self.flora.upload_instances()
        
        # Handle animals if manager provided
        if animal_manager:
            self.update_animals(camera, animal_manager)
        
        # Handle structures if manager provided
        if structure_manager:
            self.update_structures(camera, structure_manager)

    def update_structures(self, camera: Any, structure_manager: Any):
        """Update visible structures list."""
        self._visible_structures.clear()
        
        cam_x, cam_z = camera.x, camera.z
        render_dist = self.structures.fog_end * 1.2  # Render slightly past fog
        render_dist_sq = render_dist * render_dist
        
        for structure in getattr(structure_manager, 'structures', []):
            # Get structure center
            bbox = getattr(structure, 'bbox', None)
            if bbox:
                min_p, max_p = bbox
                cx = (min_p[0] + max_p[0]) / 2
                cz = (min_p[2] + max_p[2]) / 2
            else:
                cx = getattr(structure, 'x', 0)
                cz = getattr(structure, 'z', 0)
            
            # Distance culling
            dx = cx - cam_x
            dz = cz - cam_z
            if dx * dx + dz * dz < render_dist_sq:
                self._visible_structures.append(structure)

    def update_animals(self, camera: Any, animal_manager: Any):
        """
        Update animal instances from the animal manager.
        
        Animals are more dynamic than flora - they move, so we rebuild every frame.
        """
        self.animals.clear_instances()
        
        cam_x, cam_z = camera.x, camera.z
        render_dist_sq = (self.animal_render_distance * self.chunk_size * self.tile_scale) ** 2
        
        # Get all animals and filter by distance
        for animal in animal_manager.animals:
            dx = animal.x - cam_x
            dz = animal.z - cam_z
            dist_sq = dx * dx + dz * dz
            
            if dist_sq > render_dist_sq:
                continue
            
            # Get animal properties
            x, y, z = animal.x, animal.y, animal.z
            scale = getattr(animal, 'scale', 1.0)
            
            # Calculate facing direction from velocity or random
            vx = getattr(animal, 'vx', 0)
            vz = getattr(animal, 'vz', 0)
            if vx != 0 or vz != 0:
                rotation = math.atan2(vx, vz)
            else:
                rotation = getattr(animal, 'rotation', 0)
            
            # Animation phase based on movement/time
            anim_phase = getattr(animal, 'anim_time', 0)
            
            # Get type and color from DNA
            dna = getattr(animal, 'dna', None)
            if dna:
                animal_type = getattr(dna, 'animal_type', 'worm')
                type_id = get_animal_type_id(animal_type)
                
                # Get color
                color = getattr(dna, 'primary_color', (0.6, 0.5, 0.4))
                r, g, b = color[:3] if len(color) >= 3 else (0.6, 0.5, 0.4)
            else:
                type_id = 0  # Default to worm
                r, g, b = 0.6, 0.5, 0.4
            
            self.animals.add_instance(type_id, x, y, z, scale, rotation, anim_phase, r, g, b)
    
    def set_lighting(self, sun_dir: Tuple[float, float, float] = None,
                     ambient: Tuple[float, float, float] = None,
                     fog_color: Tuple[float, float, float] = None,
                     fog_start: float = None, fog_end: float = None):
        """Update lighting parameters."""
        if sun_dir:
            self.terrain.light_dir = glm.vec3(*sun_dir)
            self.structures.light_dir = glm.vec3(*sun_dir)
            self.flora.light_dir = glm.vec3(*sun_dir)
            self.animals.light_dir = glm.vec3(*sun_dir)
            self.water.light_dir = glm.vec3(*sun_dir)
        if ambient:
            self.terrain.ambient = glm.vec3(*ambient)
            self.structures.ambient = glm.vec3(*ambient)
            self.flora.ambient = glm.vec3(*ambient)
            self.animals.ambient = glm.vec3(*ambient)
        if fog_color:
            self.terrain.fog_color = glm.vec3(*fog_color)
            self.structures.fog_color = glm.vec3(*fog_color)
            self.flora.fog_color = glm.vec3(*fog_color)
            self.animals.fog_color = glm.vec3(*fog_color)
            self.water.sky_color = glm.vec3(*fog_color)  # Sky color for reflections
        if fog_start is not None:
            self.terrain.fog_start = fog_start
            self.structures.fog_start = fog_start
            self.flora.fog_start = fog_start
            self.animals.fog_start = fog_start
        if fog_end is not None:
            self.terrain.fog_end = fog_end
            self.structures.fog_end = fog_end
            self.flora.fog_end = fog_end
            self.animals.fog_end = fog_end
    
    def set_water_level(self, level: float):
        """Set water level for terrain coloring and water surface."""
        self.terrain.water_level = level
        self.water.water_level = level
    
    def set_weather(self, weather_type: str, intensity: float = 1.0):
        """Set weather conditions for particle effects."""
        self.weather.set_weather(weather_type, intensity)
    
    def render(self, dt: float = 0.016):
        """
        Render the world.
        
        Args:
            dt: Delta time for animations
        """
        start = time.perf_counter()
        
        # Render sky first (background)
        self.sky.render()
        
        # Render terrain (opaque)
        self.terrain.render()
        
        # Render structures (buildings, opaque)
        self.structures.render(self._visible_structures)
        
        # Render flora
        self.flora.render(dt)
        
        # Render animals
        self.animals.render(dt)
        
        # Render water (transparent, needs blending)
        self.water.render(dt)
        
        # Render weather particles last (in front of everything)
        self.weather.render(dt)
        
        elapsed = time.perf_counter() - start
        
        # Update stats
        self.frame_stats = {
            'terrain_chunks': self.terrain.frame_stats['chunks_rendered'],
            'terrain_tris': self.terrain.frame_stats['triangles'],
            'structure_count': self.structures.frame_stats['structures_rendered'],
            'structure_tris': self.structures.frame_stats['triangles'],
            'flora_instances': self.flora.frame_stats['instances_rendered'],
            'animal_instances': self.animals.frame_stats['instances_rendered'],
            'water_tris': self.water.frame_stats['triangles'],
            'weather_particles': self.weather.frame_stats['particles_rendered'],
            'total_draw_calls': (self.terrain.frame_stats['draw_calls'] + 
                                self.structures.frame_stats['draw_calls'] +
                                self.flora.frame_stats['draw_calls'] +
                                self.animals.frame_stats['draw_calls'] +
                                self.water.frame_stats['draw_calls'] +
                                self.weather.frame_stats['draw_calls']),
            'frame_time_ms': elapsed * 1000,
        }
    
    def set_time_of_day(self, time: float):
        """Set time of day (0.0 = midnight, 0.5 = noon, 1.0 = midnight)."""
        self.sky.set_time(time)
        
        # Sync sun direction and sky color with other renderers
        sun_dir = (self.sky.sun_dir.x, self.sky.sun_dir.y, self.sky.sun_dir.z)
        sky_color = self.sky.get_sky_color()
        
        self.set_lighting(sun_dir=sun_dir, fog_color=sky_color)
    
    def cleanup(self):
        """Release all GPU resources."""
        self.sky.cleanup()
        self.terrain.cleanup()
        self.structures.cleanup()
        self.flora.cleanup()
        self.animals.cleanup()
        self.water.cleanup()
        self.weather.cleanup()
        self.loaded_terrain_chunks.clear()
        self.loaded_flora_chunks.clear()
        self._visible_structures.clear()


# =============================================================================
# STANDALONE DEMO
# =============================================================================

def demo_modern_world():
    """
    Standalone demo of the modern renderer with procedural world.
    """
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    
    screen = pygame.display.set_mode((1280, 720), pygame.DOUBLEBUF | pygame.OPENGL)
    pygame.display.set_caption("Modern World Demo")
    
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.CULL_FACE)
    
    renderer = ModernWorldRenderer(ctx)
    renderer.set_chunk_params(chunk_size=32, tile_scale=4.0, height_scale=1.0)
    renderer.set_water_level(5.0)
    
    # Generate procedural terrain
    print("Generating world...")
    
    chunk_size = 32
    tile_scale = 4.0
    num_chunks = 16
    
    for cx in range(num_chunks):
        for cz in range(num_chunks):
            # Generate heightmap
            heightmap = np.zeros((chunk_size + 1, chunk_size + 1), dtype='f4')
            
            for z in range(chunk_size + 1):
                for x in range(chunk_size + 1):
                    wx = (cx * chunk_size + x) * tile_scale
                    wz = (cz * chunk_size + z) * tile_scale
                    
                    h = np.sin(wx * 0.008) * np.cos(wz * 0.008) * 40
                    h += np.sin(wx * 0.025 + 1.5) * np.cos(wz * 0.025 + 0.7) * 15
                    h += np.sin(wx * 0.06 + 2.1) * np.cos(wz * 0.06 + 1.2) * 5
                    h += 25
                    heightmap[z, x] = h
            
            renderer.terrain.create_chunk_mesh(
                cx, cz, heightmap,
                tile_scale, 1.0,
                cx * chunk_size * tile_scale,
                cz * chunk_size * tile_scale,
                'grassland'
            )
            renderer.loaded_terrain_chunks.add((cx, cz))
    
    # Generate flora
    print("Generating flora...")
    rng = np.random.default_rng(42)
    world_size = num_chunks * chunk_size * tile_scale
    
    for _ in range(8000):
        x = rng.random() * world_size
        z = rng.random() * world_size
        y = np.sin(x * 0.008) * np.cos(z * 0.008) * 40
        y += np.sin(x * 0.025 + 1.5) * np.cos(z * 0.025 + 0.7) * 15
        y += np.sin(x * 0.06 + 2.1) * np.cos(z * 0.06 + 1.2) * 5
        y += 25
        
        if y < 5:  # Skip underwater
            continue
        
        mesh_type = rng.choice(['tree', 'bush', 'grass', 'flower'], 
                               p=[0.15, 0.25, 0.5, 0.1])
        scale = 0.5 + rng.random() * 1.5 if mesh_type != 'grass' else 0.8 + rng.random() * 0.4
        rot = rng.random() * math.pi * 2
        g = 0.85 + rng.random() * 0.3
        r = 0.9 + rng.random() * 0.2 if mesh_type == 'flower' else 1.0
        b = 0.9 + rng.random() * 0.2 if mesh_type == 'flower' else 1.0
        
        renderer.flora.add_instance(mesh_type, x, y, z, scale, rot, r, g, b)
    
    renderer.flora.upload_instances()
    
    total_tris = len(renderer.loaded_terrain_chunks) * chunk_size * chunk_size * 2
    print(f"World ready: {len(renderer.loaded_terrain_chunks)} chunks, ~{total_tris:,} terrain tris")
    print(f"Flora: {sum(len(v) for v in renderer.flora.pending_instances.values()):,} instances")
    
    # Camera
    cam_x, cam_y, cam_z = world_size / 2, 80, world_size / 2
    yaw, pitch = 0, -20
    
    # Simple camera class for demo
    class DemoCamera:
        def __init__(self):
            self.x = cam_x
            self.y = cam_y
            self.z = cam_z
            self.yaw = yaw
            self.pitch = pitch
            self.fov = 70.0
    
    camera = DemoCamera()
    
    clock = pygame.time.Clock()
    running = True
    frame_count = 0
    total_time = 0
    
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    
    print("Controls: WASD=move, Mouse=look, ESC=quit")
    
    while running:
        dt = clock.tick(60) / 1000.0
        total_time += dt
        frame_count += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEMOTION:
                camera.yaw += event.rel[0] * 0.2
                camera.pitch -= event.rel[1] * 0.2
                camera.pitch = max(-89, min(89, camera.pitch))
        
        keys = pygame.key.get_pressed()
        speed = 100 * dt
        
        yaw_rad = math.radians(camera.yaw)
        fx, fz = -math.sin(yaw_rad), -math.cos(yaw_rad)
        rx, rz = math.cos(yaw_rad), -math.sin(yaw_rad)
        
        if keys[pygame.K_w]: camera.x += fx * speed; camera.z += fz * speed
        if keys[pygame.K_s]: camera.x -= fx * speed; camera.z -= fz * speed
        if keys[pygame.K_a]: camera.x -= rx * speed; camera.z -= rz * speed
        if keys[pygame.K_d]: camera.x += rx * speed; camera.z += rz * speed
        if keys[pygame.K_SPACE]: camera.y += speed
        if keys[pygame.K_LSHIFT]: camera.y -= speed
        
        # Update renderer
        renderer.set_camera_from_waverse(camera, 1280/720)
        
        # Animate sun
        sun_angle = total_time * 0.1
        renderer.set_lighting(
            sun_dir=(math.cos(sun_angle) * 0.5, 0.8, math.sin(sun_angle) * 0.3)
        )
        
        # Render
        ctx.clear(0.55, 0.70, 0.85)
        renderer.render(dt)
        
        pygame.display.flip()
        
        # Print stats every 2 seconds
        if frame_count % 120 == 0:
            stats = renderer.frame_stats
            fps = 1.0 / (stats['frame_time_ms'] / 1000) if stats['frame_time_ms'] > 0 else 0
            print(f"FPS: {fps:.0f} | Terrain: {stats['terrain_tris']:,} tris | "
                  f"Flora: {stats['flora_instances']:,} | Draws: {stats['total_draw_calls']}")
    
    renderer.cleanup()
    pygame.quit()
    
    print(f"\nSession: {frame_count} frames in {total_time:.1f}s = {frame_count/total_time:.0f} avg FPS")


if __name__ == "__main__":
    demo_modern_world()

