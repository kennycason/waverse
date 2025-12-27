"""
DNA Mesh Generator - Converts PlantDNA to detailed vertex meshes.

This replicates the visual quality of the old immediate-mode renderer
but outputs VBO-compatible vertex arrays for instanced rendering.

Key features:
- Recursive branching with DNA-seeded randomness
- Multi-segment curved/twisted trunks
- Various canopy shapes (dome, cone, umbrella, weeping, columnar)
- Leaves, flowers, and glow effects
- All geometry respects DNA parameters
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

# Type aliases
Vec3 = Tuple[float, float, float]
Color = Tuple[float, float, float]


@dataclass
class MeshData:
    """Vertex data for a mesh."""
    vertices: np.ndarray  # Nx3 positions
    normals: np.ndarray   # Nx3 normals
    colors: np.ndarray    # Nx3 colors
    
    def concatenate(self, other: 'MeshData') -> 'MeshData':
        """Combine two meshes."""
        return MeshData(
            vertices=np.vstack([self.vertices, other.vertices]) if len(self.vertices) > 0 else other.vertices,
            normals=np.vstack([self.normals, other.normals]) if len(self.normals) > 0 else other.normals,
            colors=np.vstack([self.colors, other.colors]) if len(self.colors) > 0 else other.colors,
        )
    
    @classmethod
    def empty(cls) -> 'MeshData':
        return cls(
            vertices=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
        )


class MatrixStack:
    """Simulates OpenGL matrix stack for hierarchical transforms."""
    
    def __init__(self):
        self.stack: List[np.ndarray] = [np.eye(4, dtype=np.float32)]
    
    def push(self):
        self.stack.append(self.stack[-1].copy())
    
    def pop(self):
        if len(self.stack) > 1:
            self.stack.pop()
    
    @property
    def current(self) -> np.ndarray:
        return self.stack[-1]
    
    def translate(self, x: float, y: float, z: float):
        """Apply translation."""
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        self.stack[-1] = self.stack[-1] @ T
    
    def rotate_x(self, angle_deg: float):
        """Rotate around X axis."""
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        R = np.array([
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.stack[-1] = self.stack[-1] @ R
    
    def rotate_y(self, angle_deg: float):
        """Rotate around Y axis."""
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        R = np.array([
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.stack[-1] = self.stack[-1] @ R
    
    def rotate_z(self, angle_deg: float):
        """Rotate around Z axis."""
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        R = np.array([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.stack[-1] = self.stack[-1] @ R
    
    def scale(self, sx: float, sy: float, sz: float):
        """Apply scaling."""
        S = np.array([
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.stack[-1] = self.stack[-1] @ S
    
    def transform_point(self, x: float, y: float, z: float) -> Vec3:
        """Transform a point by the current matrix."""
        p = np.array([x, y, z, 1.0], dtype=np.float32)
        tp = self.current @ p
        return (float(tp[0]), float(tp[1]), float(tp[2]))
    
    def transform_normal(self, nx: float, ny: float, nz: float) -> Vec3:
        """Transform a normal vector (ignores translation)."""
        # Use the inverse transpose of the upper 3x3 for normals
        mat3 = self.current[:3, :3]
        n = np.array([nx, ny, nz], dtype=np.float32)
        try:
            tn = np.linalg.inv(mat3).T @ n
            length = np.linalg.norm(tn)
            if length > 0.001:
                tn /= length
            return (float(tn[0]), float(tn[1]), float(tn[2]))
        except:
            return (nx, ny, nz)


class DNAMeshGenerator:
    """
    Generates detailed plant meshes from DNA.
    
    Uses a matrix stack to build hierarchical geometry matching
    the old immediate-mode renderer's visual quality.
    """
    
    def __init__(self):
        self.matrix = MatrixStack()
        self.vertices: List[Vec3] = []
        self.normals: List[Vec3] = []
        self.colors: List[Color] = []
        
        # Cached meshes by species_id
        self._cache: Dict[int, MeshData] = {}
        self._cache_max = 100
    
    def get_mesh(self, dna: Any) -> MeshData:
        """Get or generate mesh for a DNA."""
        # Get species_id for caching
        if hasattr(dna, 'species_id'):
            species_id = dna.species_id
        elif isinstance(dna, dict):
            species_id = dna.get('species_id', id(dna))
        else:
            species_id = id(dna)
        
        if species_id in self._cache:
            return self._cache[species_id]
        
        # Generate mesh
        mesh = self._generate_plant_mesh(dna)
        
        # Cache with LRU eviction
        if len(self._cache) >= self._cache_max:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[species_id] = mesh
        
        return mesh
    
    def _generate_plant_mesh(self, dna: Any) -> MeshData:
        """Generate complete mesh for a plant from DNA."""
        # Reset state
        self.matrix = MatrixStack()
        self.vertices = []
        self.normals = []
        self.colors = []
        
        # Extract DNA parameters
        params = self._extract_dna_params(dna)
        
        plant_type = params['plant_type']
        
        # Route to appropriate generator
        if plant_type == 'grass':
            self._generate_grass(params)
        elif plant_type in ('fern', 'bush', 'shrub'):
            self._generate_bush(params)
        elif plant_type == 'mushroom':
            self._generate_mushroom(params)
        elif plant_type == 'spiral':
            self._generate_spiral(params)
        elif plant_type in ('vine', 'seaweed', 'spiny_vine'):
            self._generate_vine(params)
        elif plant_type == 'octopus':
            self._generate_octopus(params)
        elif plant_type == 'tentacle':
            self._generate_tentacle(params)
        elif plant_type in ('groundcover', 'creeper', 'lichen', 'moss_pad'):
            self._generate_groundcover(params)
        elif plant_type == 'coral':
            self._generate_coral(params)
        else:
            # Default: tree
            self._generate_tree(params)
        
        # Convert to arrays
        if len(self.vertices) == 0:
            return MeshData.empty()
        
        return MeshData(
            vertices=np.array(self.vertices, dtype=np.float32),
            normals=np.array(self.normals, dtype=np.float32),
            colors=np.array(self.colors, dtype=np.float32),
        )
    
    def _extract_dna_params(self, dna: Any) -> Dict[str, Any]:
        """Extract all needed DNA parameters."""
        if isinstance(dna, dict):
            return {
                'plant_type': dna.get('plant_type', 'tree'),
                'height': self._get_gene_value(dna, 'height_gene', 1.0),
                'width': self._get_gene_value(dna, 'width_gene', 0.3),
                'trunk_color': self._dict_to_color(dna.get('trunk_color'), (0.4, 0.25, 0.15)),
                'leaf_color': self._dict_to_color(dna.get('leaf_color'), (0.2, 0.6, 0.2)),
                'flower_color': self._dict_to_color(dna.get('flower_color'), (0.9, 0.4, 0.6)),
                'branch_count': int(dna.get('branch_count', 4)),
                'branch_angle': float(dna.get('branch_angle', 0.4)),
                'branch_height': float(dna.get('branch_height', 0.6)),
                'branch_spread': float(dna.get('branch_spread', 1.0)),
                'sub_branch_chance': float(dna.get('sub_branch_chance', 0.3)),
                'recursive_depth': int(dna.get('recursive_depth', 2)),
                'leaf_density': float(dna.get('leaf_density', 0.7)),
                'leaf_size': float(dna.get('leaf_size', 0.5)),
                'canopy_shape': dna.get('canopy_shape', 'dome'),
                'canopy_spread': float(dna.get('canopy_spread', 0.5)),
                'droop': float(dna.get('droop', 0.0)),
                'asymmetry': float(dna.get('asymmetry', 0.1)),
                'trunk_segments': dna.get('trunk_segments', []),
                'spiral_factor': float(dna.get('spiral_factor', 0.0)),
                'has_flowers': dna.get('has_flowers', False),
                'flower_size': float(dna.get('flower_size', 0.0)),
                'species_id': dna.get('species_id', 0),
            }
        else:
            return {
                'plant_type': getattr(dna, 'plant_type', 'tree'),
                'height': dna.height_gene.value if hasattr(dna, 'height_gene') else 1.0,
                'width': dna.width_gene.value if hasattr(dna, 'width_gene') else 0.3,
                'trunk_color': (dna.trunk_color.r, dna.trunk_color.g, dna.trunk_color.b) if hasattr(dna, 'trunk_color') else (0.4, 0.25, 0.15),
                'leaf_color': (dna.leaf_color.r, dna.leaf_color.g, dna.leaf_color.b) if hasattr(dna, 'leaf_color') else (0.2, 0.6, 0.2),
                'flower_color': (dna.flower_color.r, dna.flower_color.g, dna.flower_color.b) if hasattr(dna, 'flower_color') else (0.9, 0.4, 0.6),
                'branch_count': getattr(dna, 'branch_count', 4),
                'branch_angle': getattr(dna, 'branch_angle', 0.4),
                'branch_height': getattr(dna, 'branch_height', 0.6),
                'branch_spread': getattr(dna, 'branch_spread', 1.0),
                'sub_branch_chance': getattr(dna, 'sub_branch_chance', 0.3),
                'recursive_depth': getattr(dna, 'recursive_depth', 2),
                'leaf_density': getattr(dna, 'leaf_density', 0.7),
                'leaf_size': getattr(dna, 'leaf_size', 0.5),
                'canopy_shape': getattr(dna, 'canopy_shape', 'dome'),
                'canopy_spread': getattr(dna, 'canopy_spread', 0.5),
                'droop': getattr(dna, 'droop', 0.0),
                'asymmetry': getattr(dna, 'asymmetry', 0.1),
                'trunk_segments': getattr(dna, 'trunk_segments', []),
                'spiral_factor': getattr(dna, 'spiral_factor', 0.0),
                'has_flowers': getattr(dna, 'has_flowers', False),
                'flower_size': getattr(dna, 'flower_size', 0.0),
                'species_id': getattr(dna, 'species_id', 0),
            }
    
    def _get_gene_value(self, d: dict, key: str, default: float) -> float:
        """Extract a Gene value from dict."""
        val = d.get(key)
        if isinstance(val, dict):
            return float(val.get('value', default))
        return float(val) if val is not None else default
    
    def _dict_to_color(self, c: Any, default: Color) -> Color:
        """Convert color dict/object to tuple."""
        if c is None:
            return default
        if isinstance(c, dict):
            return (float(c.get('r', default[0])),
                    float(c.get('g', default[1])),
                    float(c.get('b', default[2])))
        if hasattr(c, 'r'):
            return (c.r, c.g, c.b)
        return default
    
    # ==========================================================================
    # PRIMITIVE DRAWING (outputs to self.vertices/normals/colors)
    # ==========================================================================
    
    def _add_triangle(self, p1: Vec3, p2: Vec3, p3: Vec3, color: Color):
        """Add a triangle with auto-computed normal."""
        # Transform points
        tp1 = self.matrix.transform_point(*p1)
        tp2 = self.matrix.transform_point(*p2)
        tp3 = self.matrix.transform_point(*p3)
        
        # Compute normal
        v1 = np.array([tp2[0] - tp1[0], tp2[1] - tp1[1], tp2[2] - tp1[2]])
        v2 = np.array([tp3[0] - tp1[0], tp3[1] - tp1[1], tp3[2] - tp1[2]])
        n = np.cross(v1, v2)
        length = np.linalg.norm(n)
        if length > 0.001:
            n /= length
        else:
            n = np.array([0, 1, 0])
        
        for p in [tp1, tp2, tp3]:
            self.vertices.append(p)
            self.normals.append((float(n[0]), float(n[1]), float(n[2])))
            self.colors.append(color)
    
    def _add_quad(self, p1: Vec3, p2: Vec3, p3: Vec3, p4: Vec3, color: Color):
        """Add a quad as two triangles."""
        self._add_triangle(p1, p2, p3, color)
        self._add_triangle(p1, p3, p4, color)
    
    def _add_cylinder(self, radius: float, height: float, segments: int, 
                      color: Color, taper: float = 1.0):
        """Add a tapered cylinder."""
        for i in range(segments):
            a1 = (i / segments) * 2 * math.pi
            a2 = ((i + 1) / segments) * 2 * math.pi
            
            x1, z1 = math.cos(a1) * radius, math.sin(a1) * radius
            x2, z2 = math.cos(a2) * radius, math.sin(a2) * radius
            
            # Bottom edge
            self._add_quad(
                (x1, 0, z1),
                (x2, 0, z2),
                (x2 * taper, height, z2 * taper),
                (x1 * taper, height, z1 * taper),
                color
            )
    
    def _add_cone(self, radius: float, height: float, segments: int, color: Color):
        """Add a cone."""
        for i in range(segments):
            a1 = (i / segments) * 2 * math.pi
            a2 = ((i + 1) / segments) * 2 * math.pi
            
            self._add_triangle(
                (0, height, 0),
                (math.cos(a1) * radius, 0, math.sin(a1) * radius),
                (math.cos(a2) * radius, 0, math.sin(a2) * radius),
                color
            )
    
    def _add_fan(self, center: Vec3, radius: float, segments: int, 
                 color: Color, height_offset: float = 0):
        """Add a triangle fan (disc or dome base)."""
        for i in range(segments):
            a1 = (i / segments) * 2 * math.pi
            a2 = ((i + 1) / segments) * 2 * math.pi
            
            self._add_triangle(
                center,
                (math.cos(a1) * radius, center[1] + height_offset, math.sin(a1) * radius),
                (math.cos(a2) * radius, center[1] + height_offset, math.sin(a2) * radius),
                color
            )
    
    # ==========================================================================
    # PLANT TYPE GENERATORS
    # ==========================================================================
    
    def _generate_tree(self, p: Dict):
        """Generate a tree with multi-segment trunk, branches, and canopy.
        
        Uses matrix stack rotations to create meandering polygon trunks.
        Each segment's curve rotates the coordinate system, so subsequent 
        segments grow in a new direction - creating the bent polygon look.
        """
        height = min(p['height'], 5.0)  # Clamp for performance
        width = min(p['width'], 1.5)
        
        trunk_segments = p['trunk_segments']
        current_width = width * 0.15
        
        if trunk_segments and len(trunk_segments) > 0:
            # Use matrix stack to accumulate rotations (like V1's glPushMatrix approach)
            self.matrix.push()
            
            for i, seg in enumerate(trunk_segments[:5]):
                if isinstance(seg, dict):
                    seg_length = float(seg.get('length', 0.5)) * height / len(trunk_segments)
                    seg_taper = float(seg.get('taper', 0.85))
                    seg_curve = float(seg.get('curve', 0.0))
                    seg_twist = float(seg.get('twist', 0.0))
                else:
                    seg_length = getattr(seg, 'length', 0.5) * height / len(trunk_segments)
                    seg_taper = getattr(seg, 'taper', 0.85)
                    seg_curve = getattr(seg, 'curve', 0.0)
                    seg_twist = getattr(seg, 'twist', 0.0)
                
                # Add spiral factor
                seg_twist += p['spiral_factor'] * 0.3 * i
                
                # ROTATE BEFORE DRAWING - this is what creates the meandering effect!
                # The curve rotates the coordinate system, so the next segment
                # grows in a different direction
                curve_angle = seg_curve * 45  # Convert curve (-1 to 1) to degrees
                
                # Alternate the rotation axis for more interesting shapes
                if i % 2 == 0:
                    self.matrix.rotate_z(curve_angle)  # Lean left/right
                else:
                    self.matrix.rotate_x(curve_angle * 0.5)  # Lean forward/back
                
                # Apply twist rotation
                self.matrix.rotate_y(seg_twist * 30)
                
                # Draw the segment as a tapered cylinder
                self._add_cylinder(current_width, seg_length, 6, p['trunk_color'], taper=seg_taper)
                
                # Move up to the top of this segment for the next one
                self.matrix.translate(0, seg_length, 0)
                
                current_width *= seg_taper
            
            self.matrix.pop()
        else:
            # Single trunk - no segments
            self._add_cylinder(width * 0.15, height * 0.6, 6, p['trunk_color'], taper=0.7)
        
        # Draw branches
        if p['branch_count'] > 0:
            branch_start = height * p['branch_height']
            self._draw_branches(p, branch_start, height)
        
        # Draw canopy
        self._draw_canopy(p, height)
        
        # Draw flowers if present
        if p['has_flowers'] and p['flower_size'] > 0:
            self._draw_flowers(p, height)
    
    def _draw_branches(self, p: Dict, start_height: float, total_height: float):
        """Draw branches with recursive sub-branching."""
        rng = np.random.default_rng(p['species_id'] & 0xFFFFFFFF)
        
        for i in range(min(p['branch_count'], 6)):  # Limit for performance
            angle = (i / max(1, p['branch_count'])) * 360 * p['branch_spread']
            angle += p['asymmetry'] * 30 * math.sin(i * 2.5)
            
            self.matrix.push()
            
            branch_y = start_height + (i / max(1, p['branch_count'])) * (total_height * 0.4)
            self.matrix.translate(0, branch_y, 0)
            self.matrix.rotate_y(angle)
            self.matrix.rotate_x(p['branch_angle'] * 75 + p['droop'] * 20)
            
            branch_len = total_height * 0.3
            branch_w = p['width'] * 0.08
            
            self._draw_branch_recursive(
                p, branch_len, branch_w, 
                depth=0, max_depth=min(p['recursive_depth'], 2),  # Limit depth
                rng=rng
            )
            
            self.matrix.pop()
    
    def _draw_branch_recursive(self, p: Dict, length: float, width: float,
                                depth: int, max_depth: int, rng: np.random.Generator):
        """Recursively draw branches."""
        if width < 0.01 or length < 0.05 or depth > max_depth:
            return
        
        # Add curve variation
        curve = (rng.random() - 0.5) * 0.3 * (1 + depth * 0.5)
        taper = 0.7 if depth < max_depth else 0.5
        
        # Draw this branch segment
        segments = 4
        for i in range(segments):
            t1 = i / segments
            t2 = (i + 1) / segments
            y1 = t1 * length
            y2 = t2 * length
            w1 = width * (1 - t1 * (1 - taper))
            w2 = width * (1 - t2 * (1 - taper))
            x_off1 = curve * t1 * t1 * length
            x_off2 = curve * t2 * t2 * length
            
            self._add_quad(
                (-w1 + x_off1, y1, 0),
                (w1 + x_off1, y1, 0),
                (w2 + x_off2, y2, 0),
                (-w2 + x_off2, y2, 0),
                p['trunk_color']
            )
        
        # Move to end of branch
        self.matrix.translate(curve * length, length, 0)
        
        # Sub-branches
        if depth < max_depth:
            sub_count = 0
            for _ in range(2):  # Reduced from 3
                if rng.random() < p['sub_branch_chance']:
                    sub_count += 1
            
            for j in range(sub_count):
                self.matrix.push()
                
                sub_angle = (j / max(1, sub_count)) * 360 + rng.random() * 60 - 30
                self.matrix.rotate_y(sub_angle)
                self.matrix.rotate_x(25 + rng.random() * 35 + p['droop'] * 15)
                
                sub_len = length * (0.5 + rng.random() * 0.3)
                sub_wid = width * taper * (0.6 + rng.random() * 0.2)
                
                self._draw_branch_recursive(p, sub_len, sub_wid, depth + 1, max_depth, rng)
                
                self.matrix.pop()
        
        # Leaves at tips
        if depth >= max_depth - 1:
            leaf_size = p['leaf_size'] * 0.15
            num_leaves = 2 + int(rng.random() * 2)
            
            for k in range(num_leaves):
                self.matrix.push()
                self.matrix.rotate_y(k * 120 + rng.random() * 30)
                self.matrix.rotate_x(30 + rng.random() * 40)
                
                self._add_triangle(
                    (0, 0, 0),
                    (leaf_size * 0.3, leaf_size, 0),
                    (-leaf_size * 0.3, leaf_size, 0),
                    p['leaf_color']
                )
                
                self.matrix.pop()
    
    def _draw_canopy(self, p: Dict, height: float):
        """Draw tree canopy based on shape."""
        spread = height * p['canopy_spread'] * 0.5
        canopy_base = height * 0.5
        shape = p['canopy_shape']
        color = p['leaf_color']
        
        if shape == "cone":
            self.matrix.push()
            self.matrix.translate(0, canopy_base, 0)
            self._add_cone(spread, height * 0.6, 12, color)
            self.matrix.pop()
            
        elif shape == "umbrella":
            # Flat umbrella
            self.matrix.push()
            self.matrix.translate(0, height * 0.85, 0)
            self._add_fan((0, 0, 0), spread * 1.2, 12, color, -0.1)
            self.matrix.pop()
            
        elif shape == "weeping":
            # Multiple drooping layers
            for layer in range(3):
                layer_y = height * (0.9 - layer * 0.15)
                layer_spread = spread * (0.6 + layer * 0.25)
                droop_amt = p['droop'] * layer * 0.2
                
                self.matrix.push()
                self.matrix.translate(0, layer_y - droop_amt, 0)
                self._add_fan((0, 0, 0), layer_spread, 12, color)
                self.matrix.pop()
                
        elif shape == "columnar":
            # Tall narrow
            self.matrix.push()
            self.matrix.translate(0, canopy_base, 0)
            for layer in range(4):
                t = layer / 4
                layer_h = (height - canopy_base) * 0.25
                layer_w = spread * 0.4 * (1 - t * 0.15)
                self._add_cylinder(layer_w, layer_h, 8, color, taper=0.9)
                self.matrix.translate(0, layer_h, 0)
            self.matrix.pop()
            
        else:  # dome (default)
            for layer in range(3):
                layer_y = height * (0.55 + layer * 0.12)
                layer_spread = spread * (1 - layer * 0.15)
                
                self.matrix.push()
                self.matrix.translate(0, layer_y, 0)
                self._add_fan((0, height * 0.15, 0), layer_spread, 10, color)
                self.matrix.pop()
    
    def _draw_flowers(self, p: Dict, height: float):
        """Draw flowers on the plant."""
        spread = height * p['canopy_spread'] * 0.4
        size = p['flower_size'] * 0.25
        
        positions = [
            (0, height * 0.85, 0),
            (spread * 0.6, height * 0.7, spread * 0.3),
            (-spread * 0.5, height * 0.75, -spread * 0.4),
        ]
        
        for px, py, pz in positions:
            self.matrix.push()
            self.matrix.translate(px, py, pz)
            self._add_fan((0, size * 0.5, 0), size, 6, p['flower_color'])
            self.matrix.pop()
    
    # ==========================================================================
    # OTHER PLANT TYPES
    # ==========================================================================
    
    def _generate_grass(self, p: Dict):
        """Generate grass blades."""
        height = min(p['height'], 1.0)
        color = p['leaf_color']
        
        # Two crossed blades
        self._add_triangle((-0.08, 0, 0), (0.08, 0, 0), (0, height, 0), color)
        self._add_triangle((0, 0, -0.08), (0, 0, 0.08), (0, height * 0.9, 0), color)
    
    def _generate_bush(self, p: Dict):
        """Generate bush/shrub."""
        height = min(p['height'], 2.0)
        spread = height * 0.5
        color = p['leaf_color']
        
        # Multiple triangular fans
        self._add_triangle((0, height, 0), (-spread, 0, -spread * 0.5), (spread, 0, spread * 0.5), color)
        self._add_triangle((0, height * 0.9, 0), (-spread * 0.5, 0, spread), (spread * 0.5, 0, -spread), color)
        
        # Optional: add more detail based on branch_count
        for i in range(min(p['branch_count'], 4)):
            angle = (i / 4) * 2 * math.pi
            x = math.cos(angle) * spread * 0.4
            z = math.sin(angle) * spread * 0.4
            self._add_triangle(
                (x, height * 0.8, z),
                (x - spread * 0.3, 0.1, z - spread * 0.2),
                (x + spread * 0.3, 0.1, z + spread * 0.2),
                color
            )
    
    def _generate_mushroom(self, p: Dict):
        """Generate mushroom."""
        height = min(p['height'], 1.5)
        width = p['width'] * 0.5
        
        # Stem
        self._add_cylinder(width, height * 0.7, 6, p['trunk_color'], taper=0.9)
        
        # Cap
        self.matrix.push()
        self.matrix.translate(0, height * 0.6, 0)
        cap_size = height * p['canopy_spread']
        self._add_cone(cap_size, height * 0.4, 10, p['leaf_color'])
        self.matrix.pop()
    
    def _generate_spiral(self, p: Dict):
        """Generate spiral growth pattern."""
        height = min(p['height'], 3.0)
        width = p['width'] * 0.8
        color = p['trunk_color']
        
        steps = 20
        rotations = 2 + len(p.get('trunk_segments', []))
        
        for i in range(steps):
            t1 = i / steps
            t2 = (i + 1) / steps
            
            angle1 = t1 * rotations * 2 * math.pi
            angle2 = t2 * rotations * 2 * math.pi
            
            r1 = width * (1 - t1 * 0.3)
            r2 = width * (1 - t2 * 0.3)
            
            y1, y2 = t1 * height, t2 * height
            
            inner_r1 = r1 * 0.7
            inner_r2 = r2 * 0.7
            
            self._add_quad(
                (math.cos(angle1) * inner_r1, y1, math.sin(angle1) * inner_r1),
                (math.cos(angle1) * r1, y1, math.sin(angle1) * r1),
                (math.cos(angle2) * r2, y2, math.sin(angle2) * r2),
                (math.cos(angle2) * inner_r2, y2, math.sin(angle2) * inner_r2),
                color
            )
        
        # Tip
        self.matrix.push()
        self.matrix.translate(0, height, 0)
        self._add_cone(width * 0.3, width * 0.5, 6, p['leaf_color'])
        self.matrix.pop()
    
    def _generate_vine(self, p: Dict):
        """Generate vine/seaweed."""
        height = min(p['height'], 3.0)
        color = p['trunk_color']
        
        steps = 10
        width = 0.05
        
        for i in range(steps):
            t1 = i / steps
            t2 = (i + 1) / steps
            
            sway1 = math.sin(t1 * math.pi * 2) * 0.2
            sway2 = math.sin(t2 * math.pi * 2) * 0.2
            
            self._add_quad(
                (sway1 - width, t1 * height, 0),
                (sway1 + width, t1 * height, 0),
                (sway2 + width, t2 * height, 0),
                (sway2 - width, t2 * height, 0),
                color
            )
    
    def _generate_octopus(self, p: Dict):
        """Generate octopus-style radiating arms."""
        height = min(p['height'], 2.0)
        color = p['trunk_color']
        
        arm_count = max(3, p['branch_count'])
        
        for i in range(arm_count):
            angle = (i / arm_count) * 2 * math.pi
            end_x = math.cos(angle) * height * 0.5
            end_z = math.sin(angle) * height * 0.5
            
            self._add_triangle(
                (0, height * 0.3, 0),
                (end_x, 0, end_z),
                (end_x * 0.8, 0, end_z * 0.8),
                color
            )
    
    def _generate_tentacle(self, p: Dict):
        """Generate hanging tentacle."""
        height = min(p['height'], 2.0)
        color = p['trunk_color']
        
        self._add_triangle((0, height, 0), (-0.1, height * 0.5, 0), (0.1, height * 0.5, 0), color)
        self._add_triangle((0, height * 0.5, 0), (-0.1, 0, 0), (0.1, 0, 0), color)
    
    def _generate_groundcover(self, p: Dict):
        """Generate flat spreading groundcover."""
        width = min(p['width'], 1.0)
        height = min(p['height'], 0.3)
        color = p['leaf_color']
        
        # Flat disc
        self._add_fan((0, height, 0), width * 0.5, 8, color)
    
    def _generate_coral(self, p: Dict):
        """Generate coral-like structure."""
        height = min(p['height'], 1.5)
        width = p['width']
        color = p['leaf_color']
        
        # Base
        self._add_cylinder(width * 0.3, height * 0.5, 6, p['trunk_color'], taper=0.9)
        
        # Branches
        rng = np.random.default_rng(p['species_id'])
        for i in range(min(5, p['branch_count'])):
            self.matrix.push()
            
            angle = (i / 5) * 360
            self.matrix.translate(0, height * 0.4, 0)
            self.matrix.rotate_y(angle)
            self.matrix.rotate_x(30 + rng.random() * 30)
            
            branch_h = height * 0.4 * (0.7 + rng.random() * 0.3)
            self._add_cylinder(width * 0.1, branch_h, 4, color, taper=0.7)
            
            self.matrix.pop()


# Global instance for easy access
_generator = DNAMeshGenerator()


def generate_plant_mesh(dna: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate plant mesh from DNA.
    
    Returns:
        Tuple of (vertices, normals, colors) as numpy arrays.
    """
    mesh = _generator.get_mesh(dna)
    return mesh.vertices, mesh.normals, mesh.colors


def clear_mesh_cache():
    """Clear the mesh cache."""
    _generator._cache.clear()

