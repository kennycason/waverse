"""
DNA Animal Mesh Generator - Converts AnimalDNA to detailed vertex meshes.

This creates VBO-compatible vertex arrays from AnimalDNA parameters,
enabling true genetic variation in animal shapes.

Similar to dna_mesh_generator.py for plants.
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional
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
        if len(self.vertices) == 0:
            return other
        if len(other.vertices) == 0:
            return self
        return MeshData(
            vertices=np.vstack([self.vertices, other.vertices]),
            normals=np.vstack([self.normals, other.normals]),
            colors=np.vstack([self.colors, other.colors]),
        )
    
    @classmethod
    def empty(cls) -> 'MeshData':
        return cls(
            vertices=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            colors=np.zeros((0, 3), dtype=np.float32),
        )


class AnimalMeshGenerator:
    """Generates animal meshes from DNA."""
    
    def __init__(self):
        self._cache: Dict[int, MeshData] = {}
        self._cache_max = 200
    
    def get_mesh(self, dna: Any) -> MeshData:
        """Get or generate mesh for an animal DNA."""
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
        mesh = self._generate_animal_mesh(dna)
        
        # Cache with LRU eviction
        if len(self._cache) >= self._cache_max:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[species_id] = mesh
        
        return mesh
    
    def _generate_animal_mesh(self, dna: Any) -> MeshData:
        """Generate complete mesh for an animal from DNA."""
        # Extract DNA parameters
        params = self._extract_dna_params(dna)
        
        animal_type = params['animal_type']
        
        # Route to appropriate generator
        if animal_type in ('worm', 'serpent', 'centipede', 'millipede'):
            return self._generate_worm(params)
        elif animal_type in ('mammal', 'reptile', 'dinosaur', 'raptor', 'gorilla'):
            return self._generate_mammal(params)
        elif animal_type in ('insect', 'beetle', 'mantis', 'ant'):
            return self._generate_insect(params)
        elif animal_type in ('bird', 'pterosaur', 'bat', 'moth'):
            return self._generate_bird(params)
        elif animal_type in ('fish', 'shark', 'eel'):
            return self._generate_fish(params)
        elif animal_type in ('spider', 'scorpion', 'crustacean'):
            return self._generate_spider(params)
        elif animal_type in ('jellyfish', 'amoeba', 'blob'):
            return self._generate_jellyfish(params)
        else:
            # Default: generate a basic body
            return self._generate_basic_body(params)
    
    def _extract_dna_params(self, dna: Any) -> Dict:
        """Extract parameters from DNA object or dict."""
        if isinstance(dna, dict):
            return {
                'animal_type': str(dna.get('animal_type', 'mammal')).lower(),
                'body_segments': dna.get('body_segments', []),
                'limbs': dna.get('limbs', []),
                'features': dna.get('features', []),
                'base_scale': float(dna.get('base_scale', 1.0)),
                'primary_color': self._extract_color(dna.get('primary_color'), (0.5, 0.4, 0.3)),
                'secondary_color': self._extract_color(dna.get('secondary_color'), (0.4, 0.3, 0.2)),
                'pattern_type': dna.get('pattern_type', 'solid'),
            }
        else:
            animal_type = getattr(dna, 'animal_type', 'mammal')
            if hasattr(animal_type, 'lower'):
                animal_type = animal_type.lower()
            else:
                animal_type = str(animal_type).lower()
            
            primary = getattr(dna, 'primary_color', (0.5, 0.4, 0.3))
            if hasattr(primary, 'r'):
                primary = (primary.r, primary.g, primary.b)
            
            secondary = getattr(dna, 'secondary_color', (0.4, 0.3, 0.2))
            if hasattr(secondary, 'r'):
                secondary = (secondary.r, secondary.g, secondary.b)
            
            return {
                'animal_type': animal_type,
                'body_segments': getattr(dna, 'body_segments', []),
                'limbs': getattr(dna, 'limbs', []),
                'features': getattr(dna, 'features', []),
                'base_scale': getattr(dna, 'base_scale', 1.0),
                'primary_color': primary,
                'secondary_color': secondary,
                'pattern_type': getattr(dna, 'pattern_type', 'solid'),
            }
    
    def _extract_color(self, color_data, default: Color) -> Color:
        """Extract RGB from various color formats."""
        if color_data is None:
            return default
        if isinstance(color_data, dict):
            return (
                float(color_data.get('r', default[0])),
                float(color_data.get('g', default[1])),
                float(color_data.get('b', default[2])),
            )
        if hasattr(color_data, 'r'):
            return (color_data.r, color_data.g, color_data.b)
        if isinstance(color_data, (list, tuple)) and len(color_data) >= 3:
            return (float(color_data[0]), float(color_data[1]), float(color_data[2]))
        return default
    
    # =========================================================================
    # MESH GENERATORS
    # =========================================================================
    
    def _generate_worm(self, p: Dict) -> MeshData:
        """Generate a segmented worm/snake body."""
        verts, norms, colors = [], [], []
        
        num_segments = len(p['body_segments']) or 5
        num_segments = max(3, min(12, num_segments))
        base_scale = p['base_scale']
        color = p['primary_color']
        
        # Generate cylindrical segments
        segment_length = 0.15 * base_scale
        radius = 0.08 * base_scale
        
        for i in range(num_segments):
            # Taper toward tail
            taper = 1.0 - (i / num_segments) * 0.6
            r = radius * taper
            z = i * segment_length * 0.8  # Overlapping segments
            
            # Add cylinder segment
            self._add_cylinder(verts, norms, colors, 
                             (0, 0, z), r, segment_length, color, 8)
        
        return self._to_mesh_data(verts, norms, colors)
    
    def _generate_mammal(self, p: Dict) -> MeshData:
        """Generate a mammal body (head, torso, legs)."""
        verts, norms, colors = [], [], []
        
        base_scale = p['base_scale']
        color = p['primary_color']
        
        # Body (elongated ellipsoid)
        body_w = 0.15 * base_scale
        body_h = 0.12 * base_scale
        body_d = 0.25 * base_scale
        self._add_ellipsoid(verts, norms, colors, (0, body_h, 0), 
                           body_w, body_h, body_d, color, 8)
        
        # Head
        head_size = 0.08 * base_scale
        self._add_ellipsoid(verts, norms, colors, (0, body_h * 1.3, body_d * 0.8),
                           head_size, head_size * 0.8, head_size * 0.9, color, 6)
        
        # Legs (4)
        leg_radius = 0.025 * base_scale
        leg_length = 0.12 * base_scale
        leg_offsets = [
            (-body_w * 0.6, 0, body_d * 0.5),
            (body_w * 0.6, 0, body_d * 0.5),
            (-body_w * 0.6, 0, -body_d * 0.4),
            (body_w * 0.6, 0, -body_d * 0.4),
        ]
        leg_color = (color[0] * 0.9, color[1] * 0.9, color[2] * 0.9)
        for lx, ly, lz in leg_offsets:
            self._add_cylinder(verts, norms, colors,
                             (lx, ly + leg_length/2, lz), leg_radius, leg_length, leg_color, 6)
        
        # Tail
        if p.get('pattern_type') != 'none':
            tail_len = 0.1 * base_scale
            self._add_cylinder(verts, norms, colors,
                             (0, body_h, -body_d), 0.02 * base_scale, tail_len, color, 4)
        
        return self._to_mesh_data(verts, norms, colors)
    
    def _generate_insect(self, p: Dict) -> MeshData:
        """Generate an insect body (head, thorax, abdomen, 6 legs)."""
        verts, norms, colors = [], [], []
        
        base_scale = p['base_scale']
        color = p['primary_color']
        
        # Head
        head_r = 0.04 * base_scale
        self._add_ellipsoid(verts, norms, colors, (0, head_r, 0.08 * base_scale),
                           head_r, head_r * 0.8, head_r, color, 6)
        
        # Thorax
        thorax_r = 0.05 * base_scale
        self._add_ellipsoid(verts, norms, colors, (0, thorax_r, 0),
                           thorax_r * 0.8, thorax_r, thorax_r * 1.2, color, 6)
        
        # Abdomen
        abd_r = 0.06 * base_scale
        self._add_ellipsoid(verts, norms, colors, (0, abd_r, -0.1 * base_scale),
                           abd_r, abd_r * 1.2, abd_r * 1.5, color, 8)
        
        # 6 legs
        leg_r = 0.01 * base_scale
        leg_l = 0.08 * base_scale
        for i in range(3):
            z = 0.03 * base_scale - i * 0.03 * base_scale
            for side in [-1, 1]:
                lx = side * 0.04 * base_scale
                # Upper leg
                self._add_cylinder(verts, norms, colors,
                                 (lx, 0.02 * base_scale, z), leg_r, leg_l * 0.6,
                                 (color[0] * 0.8, color[1] * 0.8, color[2] * 0.8), 4)
        
        return self._to_mesh_data(verts, norms, colors)
    
    def _generate_bird(self, p: Dict) -> MeshData:
        """Generate a bird body with wings."""
        verts, norms, colors = [], [], []
        
        base_scale = p['base_scale']
        color = p['primary_color']
        
        # Body
        body_r = 0.08 * base_scale
        self._add_ellipsoid(verts, norms, colors, (0, body_r * 2, 0),
                           body_r * 0.7, body_r, body_r * 1.5, color, 8)
        
        # Head
        head_r = 0.04 * base_scale
        self._add_ellipsoid(verts, norms, colors, (0, body_r * 3, body_r * 1.5),
                           head_r, head_r * 0.9, head_r * 1.1, color, 6)
        
        # Beak (cone)
        self._add_cone(verts, norms, colors, (0, body_r * 3 - 0.01, body_r * 2),
                      0.015 * base_scale, 0.05 * base_scale, (0.8, 0.6, 0.2), 5)
        
        # Wings (flattened ellipsoids)
        wing_color = (color[0] * 0.95, color[1] * 0.95, color[2] * 0.95)
        for side in [-1, 1]:
            self._add_ellipsoid(verts, norms, colors,
                              (side * body_r * 1.5, body_r * 2.2, 0),
                              0.12 * base_scale, 0.01 * base_scale, 0.05 * base_scale,
                              wing_color, 6)
        
        # Legs
        leg_r = 0.01 * base_scale
        for side in [-1, 1]:
            self._add_cylinder(verts, norms, colors,
                             (side * 0.03 * base_scale, body_r * 0.5, 0),
                             leg_r, body_r * 1.5, (0.7, 0.5, 0.2), 4)
        
        return self._to_mesh_data(verts, norms, colors)
    
    def _generate_fish(self, p: Dict) -> MeshData:
        """Generate a fish body."""
        verts, norms, colors = [], [], []
        
        base_scale = p['base_scale']
        color = p['primary_color']
        
        # Body
        body_w = 0.05 * base_scale
        body_h = 0.1 * base_scale
        body_d = 0.2 * base_scale
        self._add_ellipsoid(verts, norms, colors, (0, 0, 0),
                           body_w, body_h, body_d, color, 10)
        
        # Tail fin
        fin_color = (color[0] * 0.9, color[1] * 0.9, color[2] * 0.9)
        self._add_ellipsoid(verts, norms, colors, (0, 0, -body_d * 0.9),
                           0.01 * base_scale, body_h * 0.8, 0.06 * base_scale, fin_color, 6)
        
        # Dorsal fin
        self._add_ellipsoid(verts, norms, colors, (0, body_h * 1.2, 0),
                           0.01 * base_scale, 0.05 * base_scale, 0.08 * base_scale, fin_color, 4)
        
        return self._to_mesh_data(verts, norms, colors)
    
    def _generate_spider(self, p: Dict) -> MeshData:
        """Generate a spider/arachnid body."""
        verts, norms, colors = [], [], []
        
        base_scale = p['base_scale']
        color = p['primary_color']
        
        # Cephalothorax
        ct_r = 0.05 * base_scale
        self._add_ellipsoid(verts, norms, colors, (0, ct_r, 0.02 * base_scale),
                           ct_r * 1.2, ct_r * 0.8, ct_r, color, 8)
        
        # Abdomen
        abd_r = 0.07 * base_scale
        self._add_ellipsoid(verts, norms, colors, (0, abd_r, -0.08 * base_scale),
                           abd_r, abd_r * 1.3, abd_r * 1.2, color, 8)
        
        # 8 legs
        leg_r = 0.008 * base_scale
        leg_l = 0.1 * base_scale
        leg_color = (color[0] * 0.7, color[1] * 0.7, color[2] * 0.7)
        for i in range(4):
            angle_offset = (i - 1.5) * 0.4
            for side in [-1, 1]:
                angle = side * (0.5 + i * 0.15)
                lx = math.sin(angle) * 0.04 * base_scale
                lz = math.cos(angle) * 0.02 * base_scale + angle_offset * 0.02
                self._add_cylinder(verts, norms, colors,
                                 (lx, 0, lz), leg_r, leg_l, leg_color, 4)
        
        return self._to_mesh_data(verts, norms, colors)
    
    def _generate_jellyfish(self, p: Dict) -> MeshData:
        """Generate a jellyfish body."""
        verts, norms, colors = [], [], []
        
        base_scale = p['base_scale']
        color = p['primary_color']
        
        # Bell (dome)
        bell_r = 0.1 * base_scale
        self._add_hemisphere(verts, norms, colors, (0, bell_r, 0),
                            bell_r, (color[0] * 0.6, color[1] * 0.6, color[2] * 0.8), 12)
        
        # Tentacles
        num_tentacles = 8
        tent_color = (color[0] * 0.8, color[1] * 0.7, color[2] * 0.9)
        for i in range(num_tentacles):
            angle = i * 2 * math.pi / num_tentacles
            tx = math.cos(angle) * bell_r * 0.7
            tz = math.sin(angle) * bell_r * 0.7
            tent_len = 0.15 * base_scale
            self._add_cylinder(verts, norms, colors,
                             (tx, -tent_len / 2, tz), 0.008 * base_scale, tent_len, tent_color, 4)
        
        return self._to_mesh_data(verts, norms, colors)
    
    def _generate_basic_body(self, p: Dict) -> MeshData:
        """Generate a basic body for unknown types."""
        verts, norms, colors = [], [], []
        
        base_scale = p['base_scale']
        color = p['primary_color']
        
        # Simple ellipsoid body
        self._add_ellipsoid(verts, norms, colors, (0, 0.1 * base_scale, 0),
                           0.1 * base_scale, 0.08 * base_scale, 0.15 * base_scale, color, 8)
        
        return self._to_mesh_data(verts, norms, colors)
    
    # =========================================================================
    # PRIMITIVE HELPERS
    # =========================================================================
    
    def _add_ellipsoid(self, verts: List, norms: List, colors: List,
                       center: Vec3, rx: float, ry: float, rz: float,
                       color: Color, segments: int = 8):
        """Add an ellipsoid to the mesh."""
        lat_steps = max(4, segments // 2)
        lon_steps = segments
        
        for lat in range(lat_steps):
            theta1 = (lat / lat_steps) * math.pi
            theta2 = ((lat + 1) / lat_steps) * math.pi
            
            for lon in range(lon_steps):
                phi1 = (lon / lon_steps) * 2 * math.pi
                phi2 = ((lon + 1) / lon_steps) * 2 * math.pi
                
                # Four corners of quad
                p1 = self._spherical_to_cart(theta1, phi1, rx, ry, rz, center)
                p2 = self._spherical_to_cart(theta1, phi2, rx, ry, rz, center)
                p3 = self._spherical_to_cart(theta2, phi1, rx, ry, rz, center)
                p4 = self._spherical_to_cart(theta2, phi2, rx, ry, rz, center)
                
                n1 = self._normalize((p1[0] - center[0], p1[1] - center[1], p1[2] - center[2]))
                n2 = self._normalize((p2[0] - center[0], p2[1] - center[1], p2[2] - center[2]))
                n3 = self._normalize((p3[0] - center[0], p3[1] - center[1], p3[2] - center[2]))
                n4 = self._normalize((p4[0] - center[0], p4[1] - center[1], p4[2] - center[2]))
                
                # Two triangles
                verts.extend([p1, p2, p3, p2, p4, p3])
                norms.extend([n1, n2, n3, n2, n4, n3])
                colors.extend([color] * 6)
    
    def _add_cylinder(self, verts: List, norms: List, colors: List,
                      center: Vec3, radius: float, height: float,
                      color: Color, segments: int = 8):
        """Add a vertical cylinder to the mesh."""
        half_h = height / 2
        
        for i in range(segments):
            angle1 = (i / segments) * 2 * math.pi
            angle2 = ((i + 1) / segments) * 2 * math.pi
            
            x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
            x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
            
            # Side quad (2 triangles)
            p1 = (center[0] + x1, center[1] - half_h, center[2] + z1)
            p2 = (center[0] + x2, center[1] - half_h, center[2] + z2)
            p3 = (center[0] + x1, center[1] + half_h, center[2] + z1)
            p4 = (center[0] + x2, center[1] + half_h, center[2] + z2)
            
            n1 = self._normalize((x1, 0, z1))
            n2 = self._normalize((x2, 0, z2))
            
            verts.extend([p1, p2, p3, p2, p4, p3])
            norms.extend([n1, n2, n1, n2, n2, n1])
            colors.extend([color] * 6)
    
    def _add_cone(self, verts: List, norms: List, colors: List,
                  base_center: Vec3, radius: float, height: float,
                  color: Color, segments: int = 8):
        """Add a cone to the mesh."""
        apex = (base_center[0], base_center[1] + height, base_center[2])
        
        for i in range(segments):
            angle1 = (i / segments) * 2 * math.pi
            angle2 = ((i + 1) / segments) * 2 * math.pi
            
            x1, z1 = math.cos(angle1) * radius, math.sin(angle1) * radius
            x2, z2 = math.cos(angle2) * radius, math.sin(angle2) * radius
            
            p1 = (base_center[0] + x1, base_center[1], base_center[2] + z1)
            p2 = (base_center[0] + x2, base_center[1], base_center[2] + z2)
            
            # Side normal (simplified)
            nx = (x1 + x2) / 2
            nz = (z1 + z2) / 2
            n = self._normalize((nx, radius / height, nz))
            
            verts.extend([p1, p2, apex])
            norms.extend([n, n, n])
            colors.extend([color] * 3)
    
    def _add_hemisphere(self, verts: List, norms: List, colors: List,
                        center: Vec3, radius: float, color: Color, segments: int = 8):
        """Add a hemisphere (dome) to the mesh."""
        lat_steps = max(2, segments // 2)
        lon_steps = segments
        
        for lat in range(lat_steps):
            theta1 = (lat / lat_steps) * (math.pi / 2)  # Only upper half
            theta2 = ((lat + 1) / lat_steps) * (math.pi / 2)
            
            for lon in range(lon_steps):
                phi1 = (lon / lon_steps) * 2 * math.pi
                phi2 = ((lon + 1) / lon_steps) * 2 * math.pi
                
                p1 = self._spherical_to_cart(theta1, phi1, radius, radius, radius, center)
                p2 = self._spherical_to_cart(theta1, phi2, radius, radius, radius, center)
                p3 = self._spherical_to_cart(theta2, phi1, radius, radius, radius, center)
                p4 = self._spherical_to_cart(theta2, phi2, radius, radius, radius, center)
                
                n1 = self._normalize((p1[0] - center[0], p1[1] - center[1], p1[2] - center[2]))
                n2 = self._normalize((p2[0] - center[0], p2[1] - center[1], p2[2] - center[2]))
                n3 = self._normalize((p3[0] - center[0], p3[1] - center[1], p3[2] - center[2]))
                n4 = self._normalize((p4[0] - center[0], p4[1] - center[1], p4[2] - center[2]))
                
                verts.extend([p1, p2, p3, p2, p4, p3])
                norms.extend([n1, n2, n3, n2, n4, n3])
                colors.extend([color] * 6)
    
    def _spherical_to_cart(self, theta: float, phi: float, 
                           rx: float, ry: float, rz: float,
                           center: Vec3) -> Vec3:
        """Convert spherical to Cartesian coordinates."""
        x = center[0] + rx * math.sin(theta) * math.cos(phi)
        y = center[1] + ry * math.cos(theta)
        z = center[2] + rz * math.sin(theta) * math.sin(phi)
        return (x, y, z)
    
    def _normalize(self, v: Vec3) -> Vec3:
        """Normalize a vector."""
        length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if length < 0.0001:
            return (0, 1, 0)
        return (v[0] / length, v[1] / length, v[2] / length)
    
    def _to_mesh_data(self, verts: List, norms: List, colors: List) -> MeshData:
        """Convert lists to MeshData."""
        if not verts:
            return MeshData.empty()
        
        return MeshData(
            vertices=np.array(verts, dtype=np.float32),
            normals=np.array(norms, dtype=np.float32),
            colors=np.array(colors, dtype=np.float32),
        )


# Global generator instance
_generator = AnimalMeshGenerator()


def generate_animal_mesh(dna: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate animal mesh from DNA.
    
    Returns:
        Tuple of (vertices, normals, colors) as numpy arrays.
    """
    mesh = _generator.get_mesh(dna)
    return mesh.vertices, mesh.normals, mesh.colors


def clear_animal_mesh_cache():
    """Clear the mesh cache."""
    _generator._cache.clear()

