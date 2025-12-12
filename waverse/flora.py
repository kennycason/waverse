"""
Alien Flora System - DNA-based procedural plants with LOD.

Plants are generated from DNA that controls:
- Growth patterns (branching, height, spread)
- Colors (trunk, leaves, glow)
- Shape (straight, curved, spiral)
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from OpenGL.GL import *


class PlantType:
    """Plant type categories."""
    GRASS = "grass"
    FERN = "fern"
    BUSH = "bush"
    TREE = "tree"
    TALL_TREE = "tall_tree"
    ALIEN = "alien"


@dataclass
class FloraDNA:
    """DNA that controls plant growth and appearance."""
    # Type
    plant_type: str = PlantType.TREE
    
    # Growth
    height: float = 5.0          # Base height (1-20)
    trunk_width: float = 0.3     # Trunk thickness (0.1-1.0)
    branch_count: int = 4        # Number of main branches (0-8)
    branch_angle: float = 0.5    # Angle of branches (0-1, 0=up, 1=horizontal)
    branch_length: float = 0.6   # Branch length relative to height (0.2-1.0)
    recursion: int = 2           # Branch recursion depth (0-4)
    
    # Shape
    trunk_curve: float = 0.0     # Trunk curvature (-1 to 1)
    spiral: float = 0.0          # Spiral factor (0-1)
    droop: float = 0.0           # Branch droop (0-1)
    frond_count: int = 0         # For ferns
    blade_count: int = 0         # For grass
    
    # Colors (RGB 0-1)
    trunk_color: Tuple[float, float, float] = (0.4, 0.25, 0.1)
    leaf_color: Tuple[float, float, float] = (0.2, 0.6, 0.3)
    glow_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (0,0,0) = no glow
    
    # Alien features
    bulb_size: float = 0.0       # Bulb/fruit size (0-1)
    tendril_count: int = 0       # Hanging tendrils (0-10)
    crystal: bool = False        # Crystalline appearance
    
    @classmethod
    def grass(cls, seed: int = None) -> "FloraDNA":
        """Generate grass tuft."""
        if seed is not None:
            np.random.seed(seed)
        
        green = 0.3 + np.random.random() * 0.4
        return cls(
            plant_type=PlantType.GRASS,
            height=0.3 + np.random.random() * 0.5,
            blade_count=5 + np.random.randint(0, 8),
            leaf_color=(0.2 + np.random.random() * 0.1, green, 0.1),
            trunk_color=(0.3, green * 0.8, 0.15),
        )
    
    @classmethod
    def fern(cls, seed: int = None) -> "FloraDNA":
        """Generate fern."""
        if seed is not None:
            np.random.seed(seed)
        
        green = 0.4 + np.random.random() * 0.3
        return cls(
            plant_type=PlantType.FERN,
            height=0.8 + np.random.random() * 1.5,
            frond_count=4 + np.random.randint(0, 6),
            droop=0.3 + np.random.random() * 0.4,
            leaf_color=(0.15, green, 0.2),
            trunk_color=(0.25, green * 0.7, 0.15),
        )
    
    @classmethod
    def bush(cls, seed: int = None) -> "FloraDNA":
        """Generate bush/shrub."""
        if seed is not None:
            np.random.seed(seed)
        
        green = 0.35 + np.random.random() * 0.35
        return cls(
            plant_type=PlantType.BUSH,
            height=1.0 + np.random.random() * 2.0,
            trunk_width=0.1 + np.random.random() * 0.2,
            branch_count=3 + np.random.randint(0, 5),
            branch_angle=0.4 + np.random.random() * 0.4,
            leaf_color=(0.2, green, 0.15),
            trunk_color=(0.35, 0.25, 0.15),
        )
    
    @classmethod
    def tree(cls, seed: int = None) -> "FloraDNA":
        """Generate normal tree."""
        if seed is not None:
            np.random.seed(seed)
        
        green = 0.3 + np.random.random() * 0.4
        return cls(
            plant_type=PlantType.TREE,
            height=4 + np.random.random() * 6,
            trunk_width=0.2 + np.random.random() * 0.3,
            branch_count=3 + np.random.randint(0, 4),
            branch_angle=0.3 + np.random.random() * 0.4,
            branch_length=0.4 + np.random.random() * 0.3,
            recursion=1 + np.random.randint(0, 2),
            leaf_color=(0.15 + np.random.random() * 0.1, green, 0.1),
            trunk_color=(0.35 + np.random.random() * 0.1, 0.22, 0.12),
        )
    
    @classmethod
    def tall_tree(cls, seed: int = None) -> "FloraDNA":
        """Generate tall/giant tree."""
        if seed is not None:
            np.random.seed(seed)
        
        green = 0.25 + np.random.random() * 0.35
        return cls(
            plant_type=PlantType.TALL_TREE,
            height=12 + np.random.random() * 10,
            trunk_width=0.5 + np.random.random() * 0.6,
            branch_count=4 + np.random.randint(0, 4),
            branch_angle=0.25 + np.random.random() * 0.3,
            branch_length=0.5 + np.random.random() * 0.3,
            recursion=2 + np.random.randint(0, 2),
            trunk_curve=np.random.random() * 0.2 - 0.1,
            leaf_color=(0.1, green, 0.08),
            trunk_color=(0.3, 0.2, 0.1),
        )
    
    @classmethod
    def alien(cls, seed: int = None) -> "FloraDNA":
        """Generate random alien plant DNA."""
        if seed is not None:
            np.random.seed(seed)
        
        # Random alien colors
        hue = np.random.random()
        sat = 0.5 + np.random.random() * 0.5
        
        # HSV to RGB for leaves
        h = hue * 6
        c = sat
        x = c * (1 - abs(h % 2 - 1))
        if h < 1: leaf = (c, x, 0)
        elif h < 2: leaf = (x, c, 0)
        elif h < 3: leaf = (0, c, x)
        elif h < 4: leaf = (0, x, c)
        elif h < 5: leaf = (x, 0, c)
        else: leaf = (c, 0, x)
        
        # Trunk color - brown/gray/purple variations
        trunk_hue = np.random.choice([0.08, 0.1, 0.75, 0.85])
        trunk = (
            0.2 + trunk_hue * 0.3,
            0.15 + np.random.random() * 0.15,
            0.1 + np.random.random() * 0.2
        )
        
        # Maybe glow
        glow = (0, 0, 0)
        if np.random.random() < 0.3:  # 30% chance of bioluminescence
            glow = (
                leaf[0] * 0.5 + 0.5,
                leaf[1] * 0.5 + 0.5,
                leaf[2] * 0.5 + 0.5
            )
        
        return cls(
            plant_type=PlantType.ALIEN,
            height=2 + np.random.random() * 15,
            trunk_width=0.1 + np.random.random() * 0.5,
            branch_count=np.random.randint(0, 8),
            branch_angle=0.2 + np.random.random() * 0.6,
            branch_length=0.3 + np.random.random() * 0.5,
            recursion=np.random.randint(1, 4),
            trunk_curve=np.random.random() * 0.6 - 0.3,
            spiral=np.random.random() * 0.5 if np.random.random() < 0.3 else 0,
            droop=np.random.random() * 0.5,
            trunk_color=trunk,
            leaf_color=leaf,
            glow_color=glow,
            bulb_size=np.random.random() * 0.5 if np.random.random() < 0.4 else 0,
            tendril_count=np.random.randint(0, 6) if np.random.random() < 0.2 else 0,
            crystal=np.random.random() < 0.1,
        )
    
    @classmethod
    def random(cls, seed: int = None) -> "FloraDNA":
        """Generate a random plant of any type."""
        if seed is not None:
            np.random.seed(seed)
        
        # Weighted distribution: lots of grass, fewer trees, rare aliens
        roll = np.random.random()
        if roll < 0.35:
            return cls.grass(seed)
        elif roll < 0.50:
            return cls.fern(seed)
        elif roll < 0.65:
            return cls.bush(seed)
        elif roll < 0.85:
            return cls.tree(seed)
        elif roll < 0.93:
            return cls.tall_tree(seed)
        else:
            return cls.alien(seed)


@dataclass
class PlantInstance:
    """A specific plant at a location."""
    x: float
    y: float  # Ground height
    z: float
    dna: FloraDNA
    scale: float = 1.0
    rotation: float = 0.0  # Y-axis rotation


class FloraManager:
    """Manages plant generation and rendering with LOD."""
    
    # LOD distances (in world units)
    LOD_FULL = 50      # Full 3D geometry
    LOD_SIMPLE = 150   # Simplified geometry
    LOD_BILLBOARD = 400  # Just colored points/simple shapes
    
    def __init__(self, world_seed: int = 42):
        self.world_seed = world_seed
        self.chunk_plants: Dict[Tuple[int, int], List[PlantInstance]] = {}
        self.display_lists: Dict[Tuple[int, int], Tuple[int, int, int]] = {}  # chunk -> (full, simple, point) lists
        
        # Pre-generate some DNA templates for variety
        self.dna_templates = [FloraDNA.random(world_seed + i) for i in range(20)]
    
    def get_plants_for_chunk(self, cx: int, cz: int, heightmap, chunk_world_x: float, 
                             chunk_world_z: float, tile_scale: float) -> List[PlantInstance]:
        """Generate or retrieve plants for a chunk."""
        key = (cx, cz)
        if key in self.chunk_plants:
            return self.chunk_plants[key]
        
        # Deterministic seed for this chunk (ensure positive and in valid range)
        chunk_seed = abs(hash((self.world_seed, cx, cz))) % (2**31)
        np.random.seed(chunk_seed)
        
        plants = []
        h, w = heightmap.shape
        
        # More plants! Mix of grass, ferns, bushes, trees
        num_plants = np.random.randint(15, 35)
        
        for _ in range(num_plants):
            # Random position in chunk
            local_x = np.random.randint(2, w - 2)
            local_z = np.random.randint(2, h - 2)
            
            # Get ground height
            ground_h = heightmap[local_z, local_x]
            
            # Only place on land (above water) and not too steep
            if ground_h < 1:  # Below water
                continue
            if ground_h > 40:  # Too high (mountain)
                continue
            
            # World position
            world_x = chunk_world_x + local_x * tile_scale
            world_z = chunk_world_z + local_z * tile_scale
            
            # Generate plant DNA based on height/biome
            plant_seed = chunk_seed + local_x * 100 + local_z
            if ground_h < 5:  # Near water - more ferns
                if np.random.random() < 0.4:
                    dna = FloraDNA.fern(plant_seed)
                else:
                    dna = FloraDNA.random(plant_seed)
            elif ground_h < 15:  # Low land - all types
                dna = FloraDNA.random(plant_seed)
            elif ground_h < 30:  # Hills - more trees, less grass
                roll = np.random.random()
                if roll < 0.5:
                    dna = FloraDNA.tree(plant_seed)
                elif roll < 0.7:
                    dna = FloraDNA.tall_tree(plant_seed)
                else:
                    dna = FloraDNA.bush(plant_seed)
            else:  # Higher - sparse, hardy plants
                if np.random.random() < 0.7:
                    dna = FloraDNA.bush(plant_seed)
                else:
                    dna = FloraDNA.grass(plant_seed)
            
            # Random variation
            scale = 0.5 + np.random.random() * 1.0
            rotation = np.random.random() * 360
            
            plants.append(PlantInstance(
                x=world_x,
                y=ground_h * 3.5,  # HEIGHT_SCALE
                z=world_z,
                dna=dna,
                scale=scale,
                rotation=rotation
            ))
        
        self.chunk_plants[key] = plants
        return plants
    
    def create_plant_display_lists(self, cx: int, cz: int, plants: List[PlantInstance]) -> Tuple[int, int, int]:
        """Create display lists for full, simple, and point LODs."""
        key = (cx, cz)
        if key in self.display_lists:
            return self.display_lists[key]
        
        # Full detail list
        full_list = glGenLists(1)
        glNewList(full_list, GL_COMPILE)
        for plant in plants:
            self._draw_plant_full(plant)
        glEndList()
        
        # Simple detail list
        simple_list = glGenLists(1)
        glNewList(simple_list, GL_COMPILE)
        for plant in plants:
            self._draw_plant_simple(plant)
        glEndList()
        
        # Point/billboard list
        point_list = glGenLists(1)
        glNewList(point_list, GL_COMPILE)
        glPointSize(4)
        glBegin(GL_POINTS)
        for plant in plants:
            glColor3f(*plant.dna.leaf_color)
            glVertex3f(plant.x, plant.y + plant.dna.height * plant.scale * 0.5, plant.z)
        glEnd()
        glEndList()
        
        self.display_lists[key] = (full_list, simple_list, point_list)
        return (full_list, simple_list, point_list)
    
    def _draw_plant_full(self, plant: PlantInstance):
        """Draw full detail plant based on type."""
        glPushMatrix()
        glTranslatef(plant.x, plant.y, plant.z)
        glRotatef(plant.rotation, 0, 1, 0)
        glScalef(plant.scale, plant.scale, plant.scale)
        
        dna = plant.dna
        
        if dna.plant_type == PlantType.GRASS:
            self._draw_grass(dna)
        elif dna.plant_type == PlantType.FERN:
            self._draw_fern(dna)
        elif dna.plant_type == PlantType.BUSH:
            self._draw_bush(dna)
        else:
            # Tree types (TREE, TALL_TREE, ALIEN)
            # Draw trunk
            glColor3f(*dna.trunk_color)
            self._draw_trunk(dna.height, dna.trunk_width, dna.trunk_curve, dna.spiral)
            
            # Draw branches
            if dna.branch_count > 0:
                self._draw_branches(dna, dna.height * 0.6, dna.recursion)
            
            # Draw canopy/leaves
            glColor3f(*dna.leaf_color)
            self._draw_canopy(dna)
            
            # Draw bulbs if present (alien plants)
            if dna.bulb_size > 0:
                self._draw_bulbs(dna)
        
        glPopMatrix()
    
    def _draw_grass(self, dna: FloraDNA):
        """Draw grass tuft."""
        glColor3f(*dna.leaf_color)
        height = dna.height
        
        glBegin(GL_TRIANGLES)
        for i in range(dna.blade_count):
            angle = (i / dna.blade_count) * 2 * math.pi + i * 0.3
            spread = 0.15
            
            # Base positions
            bx = math.cos(angle) * spread
            bz = math.sin(angle) * spread
            
            # Tip with slight curve outward
            tip_spread = spread * 1.5
            tx = math.cos(angle) * tip_spread
            tz = math.sin(angle) * tip_spread
            
            # Triangle blade
            glVertex3f(bx - 0.02, 0, bz)
            glVertex3f(bx + 0.02, 0, bz)
            glVertex3f(tx, height * (0.7 + (i % 3) * 0.15), tz)
        glEnd()
    
    def _draw_fern(self, dna: FloraDNA):
        """Draw fern with fronds."""
        height = dna.height
        
        for i in range(dna.frond_count):
            angle = (i / dna.frond_count) * 2 * math.pi
            
            glPushMatrix()
            glRotatef(math.degrees(angle), 0, 1, 0)
            glRotatef(20 + dna.droop * 40, 1, 0, 0)
            
            # Draw frond stem
            glColor3f(*dna.trunk_color)
            frond_len = height * 0.8
            
            glBegin(GL_LINES)
            glVertex3f(0, 0.1, 0)
            glVertex3f(0, 0.1, frond_len)
            glEnd()
            
            # Draw leaflets along frond
            glColor3f(*dna.leaf_color)
            glBegin(GL_TRIANGLES)
            leaflets = 6
            for j in range(leaflets):
                t = (j + 1) / (leaflets + 1)
                z = frond_len * t
                size = 0.15 * (1 - t * 0.5)  # Smaller toward tip
                
                # Left leaflet
                glVertex3f(0, 0.1, z)
                glVertex3f(-size, 0.1 + size * 0.3, z + size * 0.5)
                glVertex3f(0, 0.1, z + size)
                
                # Right leaflet
                glVertex3f(0, 0.1, z)
                glVertex3f(size, 0.1 + size * 0.3, z + size * 0.5)
                glVertex3f(0, 0.1, z + size)
            glEnd()
            
            glPopMatrix()
    
    def _draw_bush(self, dna: FloraDNA):
        """Draw bush/shrub."""
        # Multiple small stems
        glColor3f(*dna.trunk_color)
        for i in range(dna.branch_count):
            angle = (i / dna.branch_count) * 2 * math.pi
            spread = 0.2
            
            bx = math.cos(angle) * spread
            bz = math.sin(angle) * spread
            
            glBegin(GL_LINES)
            glVertex3f(bx, 0, bz)
            glVertex3f(bx * 1.5, dna.height * 0.7, bz * 1.5)
            glEnd()
        
        # Leafy dome
        glColor3f(*dna.leaf_color)
        radius = dna.height * 0.5
        
        # Draw as overlapping cones
        glBegin(GL_TRIANGLE_FAN)
        glVertex3f(0, dna.height, 0)
        segments = 10
        for i in range(segments + 1):
            angle = (i / segments) * 2 * math.pi
            x = math.cos(angle) * radius
            z = math.sin(angle) * radius
            glVertex3f(x, dna.height * 0.3, z)
        glEnd()
    
    def _draw_plant_simple(self, plant: PlantInstance):
        """Draw simplified plant based on type."""
        glPushMatrix()
        glTranslatef(plant.x, plant.y, plant.z)
        glScalef(plant.scale, plant.scale, plant.scale)
        
        dna = plant.dna
        height = dna.height
        
        if dna.plant_type == PlantType.GRASS:
            # Simple grass - just a few lines
            glColor3f(*dna.leaf_color)
            glBegin(GL_LINES)
            glVertex3f(-0.05, 0, 0)
            glVertex3f(0, height, 0)
            glVertex3f(0.05, 0, 0)
            glVertex3f(0.02, height * 0.9, 0)
            glEnd()
        elif dna.plant_type in (PlantType.FERN, PlantType.BUSH):
            # Simple bush/fern - triangle
            glColor3f(*dna.leaf_color)
            spread = height * 0.4
            glBegin(GL_TRIANGLES)
            glVertex3f(0, height, 0)
            glVertex3f(-spread, 0, -spread * 0.5)
            glVertex3f(spread, 0, spread * 0.5)
            glEnd()
        else:
            # Trees - trunk + canopy triangle
            glColor3f(*dna.trunk_color)
            w = dna.trunk_width
            glBegin(GL_QUADS)
            glVertex3f(-w, 0, 0)
            glVertex3f(w, 0, 0)
            glVertex3f(w * 0.5, height * 0.6, 0)
            glVertex3f(-w * 0.5, height * 0.6, 0)
            glEnd()
            
            glColor3f(*dna.leaf_color)
            spread = height * 0.4
            glBegin(GL_TRIANGLES)
            glVertex3f(0, height, 0)
            glVertex3f(-spread, height * 0.4, -spread)
            glVertex3f(spread, height * 0.4, spread)
            glEnd()
        
        glPopMatrix()
    
    def _draw_trunk(self, height: float, width: float, curve: float, spiral: float):
        """Draw plant trunk with optional curve/spiral."""
        segments = 8
        seg_height = height * 0.7 / segments
        
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            t = i / segments
            y = t * height * 0.7
            
            # Curve offset
            offset_x = curve * math.sin(t * math.pi) * height * 0.2
            offset_z = spiral * t * 2 * math.pi
            
            # Taper
            w = width * (1 - t * 0.6)
            
            # Two vertices for strip
            angle1 = offset_z
            angle2 = offset_z + math.pi
            
            glVertex3f(offset_x + math.cos(angle1) * w, y, math.sin(angle1) * w)
            glVertex3f(offset_x + math.cos(angle2) * w, y, math.sin(angle2) * w)
        glEnd()
    
    def _draw_branches(self, dna: FloraDNA, start_height: float, depth: int):
        """Draw recursive branches."""
        if depth <= 0 or dna.branch_count == 0:
            return
        
        for i in range(dna.branch_count):
            angle = (i / dna.branch_count) * 360
            
            glPushMatrix()
            glTranslatef(0, start_height, 0)
            glRotatef(angle, 0, 1, 0)
            glRotatef(dna.branch_angle * 90, 1, 0, 0)
            
            # Draw branch segment
            length = dna.branch_length * dna.height * 0.3
            w = dna.trunk_width * 0.4
            
            glBegin(GL_QUADS)
            glVertex3f(-w, 0, 0)
            glVertex3f(w, 0, 0)
            glVertex3f(w * 0.5, length, 0)
            glVertex3f(-w * 0.5, length, 0)
            glEnd()
            
            glPopMatrix()
    
    def _draw_canopy(self, dna: FloraDNA):
        """Draw leaf canopy."""
        height = dna.height
        spread = height * 0.35 * (1 + dna.branch_length)
        
        # Draw as layered cones
        layers = 3
        for layer in range(layers):
            layer_y = height * (0.5 + layer * 0.15)
            layer_spread = spread * (1 - layer * 0.2)
            
            glBegin(GL_TRIANGLE_FAN)
            glVertex3f(0, layer_y + height * 0.2, 0)  # Top
            
            segments = 8
            for i in range(segments + 1):
                angle = (i / segments) * 2 * math.pi
                x = math.cos(angle) * layer_spread
                z = math.sin(angle) * layer_spread
                y = layer_y - dna.droop * layer_spread * 0.3
                glVertex3f(x, y, z)
            glEnd()
    
    def _draw_bulbs(self, dna: FloraDNA):
        """Draw fruit/bulb decorations."""
        size = dna.bulb_size * 0.5
        height = dna.height
        
        # Glow color if present
        if dna.glow_color != (0, 0, 0):
            glColor3f(*dna.glow_color)
        else:
            glColor3f(dna.leaf_color[0] * 1.3, dna.leaf_color[1] * 0.8, dna.leaf_color[2] * 1.2)
        
        # Simple spheres as quads
        positions = [
            (0, height * 0.8, 0),
            (height * 0.2, height * 0.6, height * 0.1),
            (-height * 0.15, height * 0.65, -height * 0.15),
        ]
        
        for px, py, pz in positions:
            glBegin(GL_QUADS)
            glVertex3f(px - size, py - size, pz)
            glVertex3f(px + size, py - size, pz)
            glVertex3f(px + size, py + size, pz)
            glVertex3f(px - size, py + size, pz)
            glEnd()
    
    def render_chunk_flora(self, cx: int, cz: int, camera_x: float, camera_z: float,
                           heightmap, chunk_world_x: float, chunk_world_z: float,
                           tile_scale: float):
        """Render plants for a chunk with appropriate LOD."""
        # Get or generate plants
        plants = self.get_plants_for_chunk(cx, cz, heightmap, chunk_world_x, chunk_world_z, tile_scale)
        
        if not plants:
            return
        
        # Get or create display lists
        full_list, simple_list, point_list = self.create_plant_display_lists(cx, cz, plants)
        
        # Calculate distance to chunk center
        chunk_center_x = chunk_world_x + 16 * tile_scale
        chunk_center_z = chunk_world_z + 16 * tile_scale
        dist = math.sqrt((camera_x - chunk_center_x)**2 + (camera_z - chunk_center_z)**2)
        
        # Disable lighting for simpler plants
        glDisable(GL_LIGHTING)
        
        # Choose LOD based on distance
        if dist < self.LOD_FULL:
            glCallList(full_list)
        elif dist < self.LOD_SIMPLE:
            glCallList(simple_list)
        elif dist < self.LOD_BILLBOARD:
            glCallList(point_list)
        # Beyond LOD_BILLBOARD: don't render
        
        glEnable(GL_LIGHTING)
    
    def cleanup_chunk(self, cx: int, cz: int):
        """Remove cached data for a chunk."""
        key = (cx, cz)
        if key in self.display_lists:
            full_list, simple_list, point_list = self.display_lists[key]
            glDeleteLists(full_list, 1)
            glDeleteLists(simple_list, 1)
            glDeleteLists(point_list, 1)
            del self.display_lists[key]
        if key in self.chunk_plants:
            del self.chunk_plants[key]

