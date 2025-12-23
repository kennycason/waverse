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
from .modern_clouds import ModernCloudRenderer
from .modern_structures import ModernStructureRenderer
from .modern_weather import ModernWeatherRenderer
from .modern_hud import ModernHUDRenderer
from .wind_system import WindManager
from .wind_particles import WindParticleRenderer


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
    
    def __init__(self, ctx: moderngl.Context, enable_waves: bool = False):
        self.ctx = ctx
        self.enable_waves = enable_waves
        
        # Sub-renderers
        self.sky = ModernSkyRenderer(ctx)  # Rendered first (background)
        self.clouds = ModernCloudRenderer(ctx)  # After sky, volumetric
        self.terrain = ModernTerrainRenderer(ctx)
        self.structures = ModernStructureRenderer(ctx)  # Buildings, before flora
        self.flora = ModernFloraRenderer(ctx)
        self.animals = ModernAnimalRenderer(ctx)
        self.water = ModernWaterRenderer(ctx, enable_waves=enable_waves)  # Flat or waves
        self.weather = ModernWeatherRenderer(ctx)  # Rendered last (particles)
        self.wind_particles = WindParticleRenderer(ctx)  # Debris following wind
        self.hud = ModernHUDRenderer(ctx)  # HUD overlay
        
        # Wind system
        self.wind = WindManager(seed=42)
        
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
        # Render distances - extended for better view (can be overridden by set_render_distances)
        self.render_distance = 50  # Chunks (terrain) - ~6km view distance
        self.flora_render_distance = 45  # Chunks (plants - GPU instanced, efficient)
        self.animal_render_distance = 25  # Chunks (animals)
    
    def set_render_distances(self, terrain: int = None, flora: int = None, animals: int = None):
        """Update render distances dynamically."""
        if terrain is not None:
            self.render_distance = terrain
        if flora is not None:
            self.flora_render_distance = flora
        if animals is not None:
            self.animal_render_distance = animals
        
        # Chunk size info (set by waverse)
        self.chunk_size = 64  # Grid cells per chunk
        self.tile_scale = 2.0  # World units per cell
        self.height_scale = 1.0
    
    def set_chunk_params(self, chunk_size: int, tile_scale: float, height_scale: float):
        """Set chunk generation parameters."""
        self.chunk_size = chunk_size
        self.tile_scale = tile_scale
        self.height_scale = height_scale
    
    def set_screen_size(self, width: int, height: int):
        """Update HUD for new screen size."""
        self.hud.resize(width, height)
    
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
        
        # Store for HUD
        self._camera_pos = (x, y, z)
        
        # Update terrain renderer (use same near/far as other renderers!)
        self.terrain.set_camera(x, y, z, yaw, pitch, fov, aspect, near=0.5, far=8000.0)
        
        # Update flora renderer  
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        
        dir_x = -math.sin(yaw_rad) * math.cos(pitch_rad)
        dir_y = math.sin(pitch_rad)
        dir_z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        
        projection = glm.perspective(glm.radians(fov), aspect, 0.5, 8000.0)
        cam_pos = glm.vec3(x, y, z)
        target = cam_pos + glm.vec3(dir_x, dir_y, dir_z)
        view = glm.lookAt(cam_pos, target, glm.vec3(0, 1, 0))
        
        self.sky.set_camera(projection, view, cam_pos)
        self.clouds.set_camera(projection, view, cam_pos)
        self.structures.set_camera(projection, view, cam_pos)
        self.flora.set_camera(projection, view, cam_pos)
        self.animals.set_camera(projection, view, cam_pos)
        self.water.set_camera(projection, view, cam_pos)
        self.weather.set_camera(projection, view, cam_pos)
        self.wind_particles.set_camera(projection, view, cam_pos)
    
    def load_terrain_chunk(self, cx: int, cz: int, chunk_manager: Any,
                           climate_manager: Any = None):
        """
        Load a terrain chunk from the ChunkManager.
        
        Args:
            cx, cz: Chunk coordinates
            chunk_manager: waverse ChunkManager with get_chunk() method
            climate_manager: waverse ClimateManager for biome data
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
        
        # Get biome from ClimateManager
        biome = 'grassland'  # Default
        chaos_factor = 0.0
        if climate_manager:
            biome_dna = climate_manager.get_biome(cx, cz)
            if biome_dna:
                biome = biome_dna.get_biome_name()
                # Track current biome for special effects (dance mode, etc.)
                self._current_biome = biome.lower()
                # Get chaos factor for exotic biomes (jagged terrain)
                chaos_factor = getattr(biome_dna, 'chaos_factor', 0.0)
        
        # Calculate world position
        chunk_world_x = cx * self.chunk_size * self.tile_scale
        chunk_world_z = cz * self.chunk_size * self.tile_scale
        
        # Create GPU mesh
        self.terrain.create_chunk_mesh(
            cx, cz, heightmap,
            self.tile_scale, self.height_scale,
            chunk_world_x, chunk_world_z,
            biome,
            chaos_factor
        )
        
        self.loaded_terrain_chunks.add(key)
    
    def unload_terrain_chunk(self, cx: int, cz: int):
        """Unload a terrain chunk from GPU."""
        key = (cx, cz)
        if key in self.loaded_terrain_chunks:
            self.terrain.remove_chunk(cx, cz)
            self.loaded_terrain_chunks.discard(key)
    
    def load_flora_for_chunk(self, cx: int, cz: int, flora_manager: Any, 
                            chunk_manager: Any = None, cam_x: float = 0, cam_z: float = 0):
        """
        Load flora instances from a chunk with distance-based density falloff.
        
        Args:
            cx, cz: Chunk coordinates
            flora_manager: waverse FloraManager with chunk_plants dict
            chunk_manager: waverse ChunkManager to get heightmap data
            cam_x, cam_z: Camera position for distance-based density
        """
        key = (cx, cz)
        if key in self.loaded_flora_chunks:
            return
        
        # Get or generate plants - need heightmap from chunk_manager
        plants = flora_manager.chunk_plants.get(key, None)
        
        if plants is None and chunk_manager:
            chunk = chunk_manager.get_chunk(cx, cz)
            if chunk and hasattr(chunk, 'heightmap'):
                # Generate plants for this chunk
                from .world import TILE_SCALE
                from .explorer import HEIGHT_SCALE
                
                # Get biome name for this chunk if climate_manager available
                biome_name = None
                if hasattr(self, '_climate_manager') and self._climate_manager:
                    biome_dna = self._climate_manager.get_biome(cx, cz)
                    if biome_dna:
                        biome_name = biome_dna.get_biome_name()
                
                plants = flora_manager.get_plants_for_chunk(
                    cx, cz, chunk.heightmap, 
                    cx * self.chunk_size * self.tile_scale,  # chunk_world_x
                    cz * self.chunk_size * self.tile_scale,  # chunk_world_z
                    TILE_SCALE, HEIGHT_SCALE,
                    biome_name=biome_name
                )
            else:
                plants = []
        elif plants is None:
            plants = []
        
        # Calculate chunk center for distance-based density
        chunk_world_size = self.chunk_size * self.tile_scale
        chunk_center_x = (cx + 0.5) * chunk_world_size
        chunk_center_z = (cz + 0.5) * chunk_world_size
        chunk_dist = math.sqrt((chunk_center_x - cam_x)**2 + (chunk_center_z - cam_z)**2)
        
        # Distance-based density falloff:
        # - Close chunks (< 500 units): render all plants
        # - Medium chunks (500-1500): render every other plant
        # - Far chunks (1500-2500): render 1 in 4 plants
        # - Very far (> 2500): render 1 in 8 plants
        skip_rate = 1  # render all by default
        if chunk_dist > 2500:
            skip_rate = 8
        elif chunk_dist > 1500:
            skip_rate = 4
        elif chunk_dist > 500:
            skip_rate = 2
        
        for i, plant in enumerate(plants):
            # Use plant index + position hash for consistent culling
            if skip_rate > 1:
                plant_hash = hash((int(plant.x * 100), int(plant.z * 100)))
                if (i + plant_hash) % skip_rate != 0:
                    continue
            self._add_plant_instance(plant)
        
        self.loaded_flora_chunks.add(key)
    
    def _add_plant_instance(self, plant):
        """Add a single plant to the flora renderer using full DNA data."""
        x = plant.x
        y = plant.y
        z = plant.z
        base_scale = getattr(plant, 'scale', 1.0)
        rotation = getattr(plant, 'rotation', 0.0)
        
        # Determine mesh type and get visual properties from DNA
        dna = getattr(plant, 'dna', None)
        if dna:
            plant_type = getattr(dna, 'plant_type', 'bush')
            canopy_shape = getattr(dna, 'canopy_shape', 'dome')
            
            # Map waverse plant types and canopy shapes to our mesh types
            # Convert enum to string if needed
            type_str = str(plant_type).lower().replace('planttype.', '')
            shape_str = str(canopy_shape).lower() if canopy_shape else 'dome'
            
            # === EXOTIC MESH OVERRIDES (check first!) ===
            # These exotic shapes override normal plant type selection
            exotic_meshes = {
                'twisted_spire', 'eye_flower', 'impossible_geometry',
                'fire_plant', 'shadow_tendril', 'void_shard', 
                'rainbow_spiral', 'blob_creature'
            }
            if shape_str in exotic_meshes:
                mesh_type = shape_str
            elif type_str in ('tree', 'pine', 'oak', 'willow', 'maple', 'birch', 'cedar', 'spruce', 'fir', 'elm', 'beech'):
                # Use canopy_shape, leaf_shape, branch_count for maximum variety
                shape_str = str(canopy_shape).lower() if canopy_shape else 'dome'
                leaf_shape = str(getattr(dna, 'leaf_shape', 'round')).lower()
                branch_count = getattr(dna, 'branch_count', 5)
                height_val = getattr(dna, 'height_gene', None)
                height = height_val.value if height_val and hasattr(height_val, 'value') else 5.0
                droop = getattr(dna, 'droop', 0.0)
                
                # Special canopy shapes take priority
                canopy_map = {
                    'umbrella': 'tree_umbrella',
                    'weeping': 'tree_weeping',
                    'cascading': 'tree_weeping',
                    'columnar': 'tree_columnar',
                    # === EXOTIC BIOME MESHES ===
                    'twisted_spire': 'twisted_spire',
                    'eye_flower': 'eye_flower',
                    'impossible_geometry': 'impossible_geometry',
                    'fire_plant': 'fire_plant',
                    'shadow_tendril': 'shadow_tendril',
                    'void_shard': 'void_shard',
                    'rainbow_spiral': 'rainbow_spiral',
                    'blob_creature': 'blob_creature',
                }
                
                # Species-specific overrides
                if type_str in ('pine', 'spruce', 'fir', 'cedar'):
                    mesh_type = 'tree_pine'  # Use pine mesh for conifers
                elif type_str == 'oak':
                    mesh_type = 'tree_oak'  # Broad spreading oak
                elif type_str == 'willow' or droop > 0.5:
                    mesh_type = 'tree_weeping'  # Weeping style for willows
                elif leaf_shape == 'needle':
                    mesh_type = 'tree_pine'  # Needle leaves = conifer
                elif shape_str in canopy_map:
                    mesh_type = canopy_map[shape_str]
                elif height < 3.0:
                    mesh_type = 'tree_small'  # Small/young trees
                elif branch_count <= 3:
                    mesh_type = 'tree_sparse'
                elif branch_count >= 7:
                    mesh_type = 'tree_dense'
                elif shape_str in ('cone', 'explosion'):
                    mesh_type = 'tree'
                elif shape_str == 'sphere' or shape_str == 'layered':
                    mesh_type = 'tree_oak'  # Round broad canopy
                else:
                    mesh_type = 'tree_dome'
            elif type_str == 'tall_tree':
                mesh_type = 'tree_tall'
            elif type_str == 'palm':
                mesh_type = 'tree_palm'
            elif type_str in ('bush', 'shrub', 'hedge'):
                # Use flowering variant if plant has flowers
                has_flowers = getattr(dna, 'has_flowers', False)
                if has_flowers:
                    mesh_type = 'bush_flowering'
                else:
                    mesh_type = 'bush'
            elif type_str == 'grass':
                mesh_type = 'grass'
            elif type_str == 'fern':
                mesh_type = 'fern'
            elif type_str in ('reed', 'bamboo', 'wheat'):
                mesh_type = 'grass'
            elif type_str == 'mushroom':
                mesh_type = 'mushroom'
            elif type_str == 'cactus':
                mesh_type = 'cactus'
            elif type_str == 'succulent':
                mesh_type = 'cactus'
            elif type_str == 'flower':
                # Tall flowers vs short flowers based on height
                height_val = getattr(dna, 'height_gene', None)
                height = height_val.value if height_val and hasattr(height_val, 'value') else 0.5
                if height > 1.0:
                    mesh_type = 'flower_tall'
                else:
                    mesh_type = 'flower'
            elif type_str in ('vine', 'spiny_vine', 'seaweed'):
                mesh_type = 'vine'
            elif type_str == 'octopus':
                mesh_type = 'octopus'
            elif type_str == 'tentacle':
                mesh_type = 'tentacle'
            elif type_str == 'spiral':
                mesh_type = 'spiral'
            elif type_str == 'crystal':
                mesh_type = 'crystal'
            elif type_str == 'alien':
                mesh_type = 'alien'
            elif type_str in ('groundcover', 'creeper', 'lichen', 'moss_pad'):
                mesh_type = 'groundcover'
            elif type_str == 'lily_pad':
                mesh_type = 'lily_pad'
            elif type_str == 'coral':
                mesh_type = 'coral'
            # Tropical plants
            elif type_str in ('banana', 'monstera', 'heliconia', 'ficus'):
                mesh_type = 'tree_palm'  # Use palm for tropical trees
            elif type_str == 'baobab':
                mesh_type = 'tree_dome'  # Thick trunk, sparse top
            elif type_str == 'mangrove':
                mesh_type = 'tree_weeping'  # Exposed roots style
            # Dead plants
            elif type_str in ('dead_tree', 'snag'):
                mesh_type = 'tree_sparse'  # Bare branches
            elif type_str in ('stump', 'fallen_log'):
                mesh_type = 'mushroom'  # Low stump shape
            # Rocks/stones (environmental)
            elif type_str in ('stone', 'boulder', 'mossy_rock', 'rock_cluster', 'flat_rock'):
                mesh_type = 'crystal'  # Use crystal for rock-like shapes
            elif type_str == 'crystal_formation':
                mesh_type = 'crystal'
            # Exotic tree variants
            elif type_str in ('blob_tree', 'layered_tree', 'clump_tree'):
                mesh_type = 'tree_dome'
            else:
                mesh_type = 'bush'
            
            # SPECIAL DNA PROPERTIES can override mesh type for extreme mutations!
            # Check for exotic/mutant properties
            has_glow = getattr(dna, 'has_glow', False) or getattr(dna, 'bioluminescent', False)
            spiral_factor = getattr(dna, 'spiral_factor', 0.0)
            bulb_count = getattr(dna, 'bulb_count', 0)
            has_thorns = getattr(dna, 'has_thorns', False)
            droop_val = getattr(dna, 'droop', 0.0)
            recursive_depth = getattr(dna, 'recursive_depth', 1)
            tendrils = getattr(dna, 'tendrils', 0)
            crystal_growth = getattr(dna, 'crystal_growth', False)
            
            # Override mesh type based on extreme mutations
            if crystal_growth:
                mesh_type = 'crystal'
            elif has_glow and mesh_type not in ('flower', 'flower_tall'):
                mesh_type = 'bioluminescent'
            elif spiral_factor > 0.5 and 'tree' in mesh_type:
                mesh_type = 'spiral_tree'
            elif bulb_count >= 3:
                mesh_type = 'bulbous'
            elif has_thorns and mesh_type not in ('cactus',):
                mesh_type = 'spiky'
            elif droop_val > 0.7 and mesh_type not in ('tree_weeping', 'vine'):
                mesh_type = 'droopy'
            elif recursive_depth >= 3 and 'tree' in mesh_type:
                mesh_type = 'fractal'
            elif tendrils >= 3:
                mesh_type = 'tube'
            
            # Get height from DNA for scale variation
            # Legacy renders at height_gene.value * plant.scale (e.g. 5.0 * 1.0 = 5 units)
            # Our meshes are ~1 unit tall, so scale should be height_gene.value * base_scale
            height_gene = getattr(dna, 'height_gene', None)
            if height_gene and hasattr(height_gene, 'value'):
                # Direct height as scale - meshes are 1 unit, final height = height_gene * base_scale
                scale = height_gene.value * base_scale
                scale = max(0.5, min(15.0, scale))  # Reasonable bounds
            else:
                scale = base_scale * 3.0  # Default ~3 units tall
            
            # Get color from DNA based on plant type
            # Flowers use flower_color, bioluminescent use glow_color, others use leaf_color
            if mesh_type == 'flower' or mesh_type == 'flower_tall':
                color = getattr(dna, 'flower_color', None)
                if color is None:
                    color = getattr(dna, 'leaf_color', None)
            elif mesh_type == 'bioluminescent':
                color = getattr(dna, 'glow_color', None)
                if color is None:
                    color = getattr(dna, 'leaf_color', None)
            elif mesh_type == 'crystal':
                # Crystals use a brighter version of leaf color
                color = getattr(dna, 'tip_color', None)
                if color is None:
                    color = getattr(dna, 'leaf_color', None)
            else:
                color = getattr(dna, 'leaf_color', None)
            
            if color is None:
                r, g, b = 0.3, 0.6, 0.3
            elif hasattr(color, 'r'):
                # ColorGene object - add slight variation for natural look
                var = 0.08
                r = max(0.1, min(1.0, color.r + (hash((int(x), int(z))) % 100 - 50) * var * 0.01))
                g = max(0.1, min(1.0, color.g + (hash((int(z), int(x))) % 100 - 50) * var * 0.01))
                b = max(0.1, min(1.0, color.b + (hash((int(x + z),)) % 100 - 50) * var * 0.01))
            elif hasattr(color, '__getitem__'):
                r, g, b = color[0], color[1], color[2]
            else:
                r, g, b = 0.3, 0.6, 0.3
        else:
            mesh_type = 'bush'
            scale = base_scale * 3.0  # Default ~3 units tall
            r, g, b = 0.4, 0.6, 0.35
        
        self.flora.add_instance(mesh_type, x, y, z, scale, rotation, r, g, b)
    
    def update_chunks_around_camera(self, camera: Any, chunk_manager: Any,
                                    flora_manager: Any = None,
                                    animal_manager: Any = None,
                                    structure_manager: Any = None,
                                    climate_manager: Any = None):
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
        
        # Get camera forward direction for prioritization
        # Chunks in front of camera load FIRST! (strong priority)
        cam_yaw = getattr(camera, 'yaw', 0)
        forward_x = -math.sin(math.radians(cam_yaw))
        forward_z = -math.cos(math.radians(cam_yaw))
        
        # Build list of chunks sorted by PRIORITY (distance + STRONG facing bonus)
        chunks_by_priority = []
        for dx in range(-self.render_distance, self.render_distance + 1):
            for dz in range(-self.render_distance, self.render_distance + 1):
                dist_sq = dx * dx + dz * dz
                cx, cz = cam_cx + dx, cam_cz + dz
                
                # Calculate facing bonus: chunks in front get MUCH lower priority score
                # dot product: positive = in front, negative = behind
                if dist_sq > 0:
                    norm = math.sqrt(dist_sq)
                    dot = (dx * forward_x + dz * forward_z) / norm
                    # Facing bonus: -1 (behind) to +1 (in front)
                    # STRONG bonus: chunks behind are effectively 2x further away
                    facing_bonus = dot * norm  # Full distance as bonus (was 0.5)
                else:
                    facing_bonus = 0
                
                # Priority: lower = load first
                # In front: facing_bonus positive, so priority reduced (good)
                # Behind: facing_bonus negative, so priority increased (deferred)
                priority = dist_sq - facing_bonus * 1.5  # 1.5x multiplier for strong facing prio
                chunks_by_priority.append((priority, dist_sq, cx, cz))
        
        # Sort by priority (distance adjusted by facing)
        chunks_by_priority.sort(key=lambda x: x[0])
        
        # Convert back to simpler format for compatibility
        chunks_by_dist = [(dist_sq, cx, cz) for _, dist_sq, cx, cz in chunks_by_priority]
        
        # Determine which chunks should be loaded
        needed_terrain = set()
        needed_flora = set()
        
        for dist_sq, cx, cz in chunks_by_dist:
            dist = math.sqrt(dist_sq)
            if dist <= self.render_distance:
                needed_terrain.add((cx, cz))
            if dist <= self.flora_render_distance:
                needed_flora.add((cx, cz))
        
        # THROTTLED chunk loading - TERRAIN PRIORITY over flora!
        # Terrain mesh creation is vectorized (faster), prioritize it
        MAX_TERRAIN_LOADS_PER_FRAME = 4  # Increased: terrain is priority!
        MAX_FLORA_LOADS_PER_FRAME = 4    # Reduced: flora waits for terrain
        
        # Track startup time - delay flora until terrain has a head start
        if not hasattr(self, '_startup_time'):
            self._startup_time = 0.0
        self._startup_time += 0.016  # ~1 frame
        
        # Load new terrain chunks (in radial order, throttled)
        chunks_to_load = [(cx, cz) for _, cx, cz in chunks_by_dist 
                          if (cx, cz) in needed_terrain and (cx, cz) not in self.loaded_terrain_chunks]
        
        terrain_loaded = 0
        for cx, cz in chunks_to_load:
            if terrain_loaded >= MAX_TERRAIN_LOADS_PER_FRAME:
                break  # Defer rest to next frame
            self.load_terrain_chunk(cx, cz, chunk_manager, climate_manager)
            terrain_loaded += 1
        
        # Unload distant terrain chunks (unloading is fast, do all)
        for key in list(self.loaded_terrain_chunks - needed_terrain):
            self.unload_terrain_chunk(key[0], key[1])
        
        # Handle flora if manager provided (INCREMENTAL loading)
        # DELAY flora until terrain has loaded for 2 seconds
        if flora_manager and self._startup_time > 2.0:
            # Store climate_manager reference for biome-based flora
            self._climate_manager = climate_manager
            
            # Find chunks that need flora but don't have it yet
            flora_to_load = [(cx, cz) for _, cx, cz in chunks_by_dist 
                             if (cx, cz) in needed_flora and (cx, cz) not in self.loaded_flora_chunks]
            
            # INCREMENTAL flora loading - spread across frames to eliminate stutters
            # Also reduce flora loading if terrain still loading (terrain priority!)
            effective_flora_limit = MAX_FLORA_LOADS_PER_FRAME if terrain_loaded == 0 else 2
            flora_loaded_this_frame = 0
            
            for cx, cz in flora_to_load:
                if flora_loaded_this_frame >= effective_flora_limit:
                    break  # Defer rest to next frame
                self.load_flora_for_chunk(cx, cz, flora_manager, chunk_manager, camera.x, camera.z)
                flora_loaded_this_frame += 1
            
            # TIME-BASED GPU UPLOAD to avoid per-frame buffer recreation
            # Only upload every ~0.5 seconds to batch changes together
            if flora_loaded_this_frame > 0:
                if not hasattr(self, '_flora_upload_timer'):
                    self._flora_upload_timer = 0.0
                    self._flora_needs_upload = False
                
                self._flora_needs_upload = True
                self._flora_upload_timer += 0.016  # ~1 frame at 60fps
                
                # Upload only every 0.5 seconds OR if no more to load (finished)
                remaining_to_load = len(flora_to_load) - flora_loaded_this_frame
                if self._flora_upload_timer >= 0.5 or remaining_to_load == 0:
                    self.flora.upload_instances()
                    self._flora_upload_timer = 0.0
                    self._flora_needs_upload = False
        
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
                
                # Get size from DNA for scale variation
                size_gene = getattr(dna, 'size', None)
                if size_gene is not None:
                    if hasattr(size_gene, 'value'):
                        dna_scale = size_gene.value
                    else:
                        dna_scale = float(size_gene)
                    scale = scale * max(0.5, min(2.0, dna_scale))
                
                # Get color (may be ColorGene object or tuple)
                color = getattr(dna, 'primary_color', None)
                if color is None:
                    r, g, b = 0.6, 0.5, 0.4
                elif hasattr(color, 'r'):
                    r, g, b = color.r, color.g, color.b
                elif hasattr(color, '__getitem__'):
                    r, g, b = color[0], color[1], color[2]
                else:
                    r, g, b = 0.6, 0.5, 0.4
                
                # Add secondary color pattern influence
                secondary = getattr(dna, 'secondary_color', None)
                if secondary and hasattr(secondary, 'r'):
                    pattern = getattr(dna, 'pattern_type', 'solid')
                    if pattern in ('spotted', 'striped', 'patched'):
                        # Blend colors for patterned animals
                        blend = 0.2
                        r = r * (1 - blend) + secondary.r * blend
                        g = g * (1 - blend) + secondary.g * blend
                        b = b * (1 - blend) + secondary.b * blend
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
        self._water_level = level  # Store for optimization checks
        self.terrain.water_level = level
        self.water.water_level = level
    
    def set_weather(self, weather_type: str, intensity: float = 1.0):
        """Set weather conditions for particle effects and clouds."""
        self.weather.set_weather(weather_type, intensity)
        
        # Track stormy state for wind system (can spawn tornadoes)
        self._is_stormy = weather_type in ['storm', 'heavy_rain']
        
        # Adjust wind strength based on weather
        if weather_type == 'storm':
            self.wind.set_stormy(True)
        else:
            self.wind.set_stormy(False)
        
        # Sync cloud coverage with weather
        if weather_type == 'clear':
            self.clouds.set_weather(0.2)  # Few clouds
        elif weather_type == 'rain':
            self.clouds.set_weather(0.7, 1.5)  # Overcast, faster wind
        elif weather_type == 'heavy_rain':
            self.clouds.set_weather(0.9, 2.0)  # Very cloudy
        elif weather_type == 'storm':
            self.clouds.set_weather(1.0, 3.0)  # Full cloud cover, strong wind
        elif weather_type == 'snow':
            self.clouds.set_weather(0.6, 0.5)  # Moderate clouds, slow wind
        else:
            self.clouds.set_weather(0.4, 1.0)  # Default: partly cloudy
    
    def render(self, dt: float = 0.016):
        """
        Render the world.
        
        Args:
            dt: Delta time for animations
        """
        start = time.perf_counter()
        
        # Update wind system
        is_stormy = getattr(self, '_is_stormy', False)
        self.wind.update(dt, is_stormy=is_stormy)
        
        # Get wind uniforms and pass to renderers
        wind_dir = self.wind.direction
        wind_strength = self.wind.strength
        wind_time = self.wind.time
        tide_level = self.wind.tide_level
        has_tornado = self.wind.tornado is not None
        tornado_center = (self.wind.tornado.center_x, self.wind.tornado.center_z) if has_tornado else (0, 0)
        tornado_radius = self.wind.tornado.radius if has_tornado else 0
        tornado_strength = self.wind.tornado.strength * self.wind.tornado.intensity if has_tornado else 0
        
        # Set wind on flora
        self.flora.set_wind(wind_dir, wind_strength, wind_time,
                           has_tornado, tornado_center, tornado_radius, tornado_strength)
        
        # Check for psychedelic biome (dance mode!)
        current_biome = getattr(self, '_current_biome', None)
        self.flora.dance_mode = (current_biome == 'psychedelic')
        
        # Set wind on water
        self.water.set_wind(wind_dir, wind_strength, tide_level)
        
        # Set wind on particles
        self.wind_particles.set_wind(wind_dir, wind_strength, 
                                      has_tornado, tornado_center, tornado_radius, tornado_strength)
        
        # Clear the framebuffer (color and depth)
        # Get sky color for clear color
        sky_color = self.sky.get_sky_color()
        self.ctx.clear(sky_color[0], sky_color[1], sky_color[2], 1.0)
        
        # Render sky first (background)
        self.sky.render()
        
        # Render volumetric clouds (after sky, blended)
        self.clouds.render(dt)
        
        # Render terrain (opaque)
        self.terrain.render()
        
        # Render structures (buildings, opaque)
        self.structures.render(self._visible_structures)
        
        # Render flora
        self.flora.render(dt)
        
        # Render animals
        self.animals.render(dt)
        
        # Render water (transparent, needs blending)
        # OPTIMIZATION: Skip water if camera is very high above water level
        # Water is irrelevant at high altitudes
        cam_y = self._camera_pos[1] if hasattr(self, '_camera_pos') else 0
        water_level = getattr(self, '_water_level', 0)
        if cam_y < water_level + 500:  # Only render if within 500 units of water
            self.water.render(dt)
        
        # Render wind particles (debris)
        self.wind_particles.update(dt)
        self.wind_particles.render()
        
        # Render weather particles last (in front of everything)
        self.weather.render(dt)
        
        # Render HUD overlay (after everything else)
        self.hud.render(
            camera_x=self._camera_pos[0] if hasattr(self, '_camera_pos') else 0,
            camera_y=self._camera_pos[1] if hasattr(self, '_camera_pos') else 0,
            camera_z=self._camera_pos[2] if hasattr(self, '_camera_pos') else 0,
            fps=1000.0 / max(0.1, self.frame_stats.get('frame_time_ms', 16.67)) if self.frame_stats else 60,
            terrain_chunks=len(self.loaded_terrain_chunks),
            flora_instances=self.flora.frame_stats.get('instances_rendered', 0),
        )
        
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
        self.clouds.set_time(time)  # Sync clouds with sky
        
        # Sync sun direction and sky color with other renderers
        sun_dir = (self.sky.sun_dir.x, self.sky.sun_dir.y, self.sky.sun_dir.z)
        sky_color = self.sky.get_sky_color()
        
        self.set_lighting(sun_dir=sun_dir, fog_color=sky_color)
    
    def force_tornado(self, offset_x: float = 0, offset_z: float = 50):
        """
        Force spawn a tornado for testing.
        
        Args:
            offset_x, offset_z: Offset from camera position
        """
        self.wind.force_tornado(offset_x, offset_z)
        print(f"[RENDERER] Forced tornado spawned at camera offset ({offset_x}, {offset_z})")
    
    def get_wind_info(self) -> dict:
        """Get current wind state info for debug display."""
        return {
            'direction': self.wind.direction,
            'strength': self.wind.strength,
            'gust': self.wind.gust_factor,
            'tide': self.wind.tide_level,
            'tornado': self.wind.tornado is not None,
            'tornado_strength': self.wind.tornado.strength if self.wind.tornado else 0,
        }
    
    def cleanup(self):
        """Release all GPU resources."""
        self.sky.cleanup()
        self.clouds.cleanup()
        self.terrain.cleanup()
        self.structures.cleanup()
        self.flora.cleanup()
        self.animals.cleanup()
        self.water.cleanup()
        self.weather.cleanup()
        self.wind_particles.cleanup()
        self.hud.cleanup()
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

