"""
DNA Geometry Builder - Unified DNA-to-Mesh Generation

This module provides the core geometry generation from DNA parameters.
Both plants and animals use the same underlying primitive system,
allowing for evolution and genetic variation.

Primitives:
- Ellipsoid (sphere, egg, disc shapes)
- Cylinder (tubes, stems, trunks)
- Cone (tapered segments, spikes)
- Box (angular/crystalline)

Branching:
- Segments can have child segments
- Position/rotation relative to parent
- Recursive structure for complex shapes

Colors:
- Per-segment coloring
- Pattern support (stripes, spots, gradients)
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field


# =============================================================================
# GEOMETRY PRIMITIVES (Low-level mesh generation)
# =============================================================================

def create_ellipsoid(
    radius_x: float, radius_y: float, radius_z: float,
    segments: int = 12, rings: int = 8,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[List, List, List]:
    """Create an ellipsoid mesh centered at origin."""
    verts = []
    norms = []
    colors = []
    
    for i in range(rings):
        lat0 = math.pi * (-0.5 + float(i) / rings)
        lat1 = math.pi * (-0.5 + float(i + 1) / rings)
        
        y0 = math.sin(lat0)
        y1 = math.sin(lat1)
        r0 = math.cos(lat0)
        r1 = math.cos(lat1)
        
        for j in range(segments):
            lng0 = 2 * math.pi * j / segments
            lng1 = 2 * math.pi * (j + 1) / segments
            
            x0, z0 = math.cos(lng0), math.sin(lng0)
            x1, z1 = math.cos(lng1), math.sin(lng1)
            
            # Four corners of the quad
            p00 = (x0 * r0 * radius_x, y0 * radius_y, z0 * r0 * radius_z)
            p01 = (x1 * r0 * radius_x, y0 * radius_y, z1 * r0 * radius_z)
            p10 = (x0 * r1 * radius_x, y1 * radius_y, z0 * r1 * radius_z)
            p11 = (x1 * r1 * radius_x, y1 * radius_y, z1 * r1 * radius_z)
            
            # Normals (normalized position for ellipsoid)
            n00 = _normalize((x0 * r0, y0, z0 * r0))
            n01 = _normalize((x1 * r0, y0, z1 * r0))
            n10 = _normalize((x0 * r1, y1, z0 * r1))
            n11 = _normalize((x1 * r1, y1, z1 * r1))
            
            # Two triangles
            verts.extend([p00, p10, p11])
            verts.extend([p00, p11, p01])
            norms.extend([n00, n10, n11])
            norms.extend([n00, n11, n01])
            for _ in range(6):
                colors.append(color)
    
    return verts, norms, colors


def create_cylinder(
    radius_bottom: float, radius_top: float, height: float,
    segments: int = 12,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[List, List, List]:
    """Create a cylinder/cone mesh. If radius_top=0, it's a cone."""
    verts = []
    norms = []
    colors = []
    
    half_h = height / 2
    
    for i in range(segments):
        a0 = 2 * math.pi * i / segments
        a1 = 2 * math.pi * (i + 1) / segments
        
        x0_b = math.cos(a0) * radius_bottom
        z0_b = math.sin(a0) * radius_bottom
        x1_b = math.cos(a1) * radius_bottom
        z1_b = math.sin(a1) * radius_bottom
        
        x0_t = math.cos(a0) * radius_top
        z0_t = math.sin(a0) * radius_top
        x1_t = math.cos(a1) * radius_top
        z1_t = math.sin(a1) * radius_top
        
        # Side faces (as quads = 2 triangles)
        n0 = _normalize((math.cos(a0), (radius_bottom - radius_top) / height, math.sin(a0)))
        n1 = _normalize((math.cos(a1), (radius_bottom - radius_top) / height, math.sin(a1)))
        
        if radius_top > 0.001:
            # Full cylinder - quad sides
            verts.extend([(x0_b, -half_h, z0_b), (x1_b, -half_h, z1_b), (x0_t, half_h, z0_t)])
            verts.extend([(x1_b, -half_h, z1_b), (x1_t, half_h, z1_t), (x0_t, half_h, z0_t)])
            norms.extend([n0, n1, n0, n1, n1, n0])
            for _ in range(6):
                colors.append(color)
        else:
            # Cone - triangle sides
            verts.extend([(x0_b, -half_h, z0_b), (x1_b, -half_h, z1_b), (0, half_h, 0)])
            norms.extend([n0, n1, (0, 1, 0)])
            for _ in range(3):
                colors.append(color)
        
        # Bottom cap
        verts.extend([(0, -half_h, 0), (x1_b, -half_h, z1_b), (x0_b, -half_h, z0_b)])
        norms.extend([(0, -1, 0)] * 3)
        colors.extend([color] * 3)
        
        # Top cap (if not a cone)
        if radius_top > 0.001:
            verts.extend([(0, half_h, 0), (x0_t, half_h, z0_t), (x1_t, half_h, z1_t)])
            norms.extend([(0, 1, 0)] * 3)
            colors.extend([color] * 3)
    
    return verts, norms, colors


def create_box(
    width: float, height: float, depth: float,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[List, List, List]:
    """Create a box mesh centered at origin."""
    verts = []
    norms = []
    colors = []
    
    hw, hh, hd = width / 2, height / 2, depth / 2
    
    # 6 faces, each with 2 triangles
    faces = [
        # (v0, v1, v2, v3, normal)
        ((-hw, -hh, hd), (hw, -hh, hd), (hw, hh, hd), (-hw, hh, hd), (0, 0, 1)),   # Front
        ((hw, -hh, -hd), (-hw, -hh, -hd), (-hw, hh, -hd), (hw, hh, -hd), (0, 0, -1)), # Back
        ((-hw, hh, hd), (hw, hh, hd), (hw, hh, -hd), (-hw, hh, -hd), (0, 1, 0)),   # Top
        ((-hw, -hh, -hd), (hw, -hh, -hd), (hw, -hh, hd), (-hw, -hh, hd), (0, -1, 0)), # Bottom
        ((hw, -hh, hd), (hw, -hh, -hd), (hw, hh, -hd), (hw, hh, hd), (1, 0, 0)),   # Right
        ((-hw, -hh, -hd), (-hw, -hh, hd), (-hw, hh, hd), (-hw, hh, -hd), (-1, 0, 0)), # Left
    ]
    
    for v0, v1, v2, v3, n in faces:
        # Two triangles per face
        verts.extend([v0, v1, v2, v0, v2, v3])
        for _ in range(6):
            norms.append(n)
            colors.append(color)
    
    return verts, norms, colors


def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a 3D vector."""
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length < 0.0001:
        return (0, 1, 0)
    return (v[0]/length, v[1]/length, v[2]/length)


# =============================================================================
# SEGMENT NODE - Building block for hierarchical geometry
# =============================================================================

@dataclass
class GeometrySegment:
    """
    A segment of geometry that can have children.
    This is the fundamental building block for DNA-driven shapes.
    """
    # Shape type
    shape: str = "ellipsoid"  # ellipsoid, cylinder, cone, box
    
    # Dimensions
    size_x: float = 0.5  # Width
    size_y: float = 0.5  # Height
    size_z: float = 0.5  # Depth
    
    # Color
    color: Tuple[float, float, float] = (0.7, 0.7, 0.7)
    
    # Position relative to parent (0,0,0 = attached at parent's end)
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    
    # Rotation relative to parent (radians)
    rotation_x: float = 0.0
    rotation_y: float = 0.0
    rotation_z: float = 0.0
    
    # Child segments
    children: List["GeometrySegment"] = field(default_factory=list)
    
    # Level of detail (segments for curves)
    lod: int = 8
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "shape": self.shape,
            "size": [self.size_x, self.size_y, self.size_z],
            "color": list(self.color),
            "offset": [self.offset_x, self.offset_y, self.offset_z],
            "rotation": [self.rotation_x, self.rotation_y, self.rotation_z],
            "children": [c.to_dict() for c in self.children],
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GeometrySegment":
        """Deserialize from dictionary."""
        seg = cls()
        seg.shape = data.get("shape", "ellipsoid")
        size = data.get("size", [0.5, 0.5, 0.5])
        seg.size_x, seg.size_y, seg.size_z = size[0], size[1], size[2]
        color = data.get("color", [0.7, 0.7, 0.7])
        seg.color = (color[0], color[1], color[2])
        offset = data.get("offset", [0, 0, 0])
        seg.offset_x, seg.offset_y, seg.offset_z = offset[0], offset[1], offset[2]
        rotation = data.get("rotation", [0, 0, 0])
        seg.rotation_x, seg.rotation_y, seg.rotation_z = rotation[0], rotation[1], rotation[2]
        seg.children = [cls.from_dict(c) for c in data.get("children", [])]
        return seg


# =============================================================================
# GEOMETRY BUILDER - Converts segment trees to mesh data
# =============================================================================

class GeometryBuilder:
    """
    Builds mesh geometry from a tree of GeometrySegments.
    """
    
    @staticmethod
    def build_mesh(root: GeometrySegment) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build a complete mesh from a segment tree.
        
        Returns:
            (vertices, normals, colors) as numpy arrays
        """
        all_verts = []
        all_norms = []
        all_colors = []
        
        # Build recursively with transform stack
        GeometryBuilder._build_recursive(
            root, 
            np.eye(4),  # Identity transform
            all_verts, all_norms, all_colors
        )
        
        if not all_verts:
            # Empty mesh fallback
            return (
                np.zeros((3, 3), dtype='f4'),
                np.array([[0, 1, 0]] * 3, dtype='f4'),
                np.array([[1, 1, 1]] * 3, dtype='f4')
            )
        
        return (
            np.array(all_verts, dtype='f4'),
            np.array(all_norms, dtype='f4'),
            np.array(all_colors, dtype='f4')
        )
    
    @staticmethod
    def _build_recursive(
        segment: GeometrySegment,
        parent_transform: np.ndarray,
        all_verts: List,
        all_norms: List,
        all_colors: List
    ):
        """Recursively build geometry for a segment and its children."""
        
        # Create this segment's local transform
        local_transform = GeometryBuilder._make_transform(
            segment.offset_x, segment.offset_y, segment.offset_z,
            segment.rotation_x, segment.rotation_y, segment.rotation_z
        )
        
        # Combine with parent transform
        world_transform = parent_transform @ local_transform
        
        # Generate primitive geometry
        if segment.shape == "ellipsoid":
            verts, norms, colors = create_ellipsoid(
                segment.size_x, segment.size_y, segment.size_z,
                segment.lod, max(4, segment.lod // 2),
                segment.color
            )
        elif segment.shape == "cylinder":
            verts, norms, colors = create_cylinder(
                segment.size_x, segment.size_x,  # Same radius top/bottom
                segment.size_y,  # Height
                segment.lod,
                segment.color
            )
        elif segment.shape == "cone":
            verts, norms, colors = create_cylinder(
                segment.size_x, segment.size_z * 0.1,  # Taper to small tip
                segment.size_y,
                segment.lod,
                segment.color
            )
        elif segment.shape == "box":
            verts, norms, colors = create_box(
                segment.size_x, segment.size_y, segment.size_z,
                segment.color
            )
        else:
            # Default to ellipsoid
            verts, norms, colors = create_ellipsoid(
                segment.size_x, segment.size_y, segment.size_z,
                segment.lod, max(4, segment.lod // 2),
                segment.color
            )
        
        # Transform vertices and normals
        rot_matrix = world_transform[:3, :3]
        translation = world_transform[:3, 3]
        
        for v in verts:
            transformed = rot_matrix @ np.array(v) + translation
            all_verts.append(tuple(transformed))
        
        for n in norms:
            # Normals only need rotation, not translation
            transformed = rot_matrix @ np.array(n)
            all_norms.append(tuple(_normalize(tuple(transformed))))
        
        all_colors.extend(colors)
        
        # Calculate child attachment point (top of this segment)
        child_base_y = segment.size_y / 2 if segment.shape in ("cylinder", "cone") else segment.size_y
        child_transform = world_transform @ GeometryBuilder._make_transform(0, child_base_y, 0, 0, 0, 0)
        
        # Build children
        for child in segment.children:
            GeometryBuilder._build_recursive(
                child, child_transform,
                all_verts, all_norms, all_colors
            )
    
    @staticmethod
    def _make_transform(
        tx: float, ty: float, tz: float,
        rx: float, ry: float, rz: float
    ) -> np.ndarray:
        """Create a 4x4 transformation matrix."""
        # Rotation matrices
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        
        # Combined rotation (ZYX order)
        rot = np.array([
            [cy*cz, -cy*sz, sy, 0],
            [sx*sy*cz + cx*sz, -sx*sy*sz + cx*cz, -sx*cy, 0],
            [-cx*sy*cz + sx*sz, cx*sy*sz + sx*cz, cx*cy, 0],
            [0, 0, 0, 1]
        ])
        
        # Translation
        rot[0, 3] = tx
        rot[1, 3] = ty
        rot[2, 3] = tz
        
        return rot


# =============================================================================
# DNA TO GEOMETRY CONVERTERS
# =============================================================================

def plant_dna_to_geometry(dna: Any) -> GeometrySegment:
    """
    Convert PlantDNA to a GeometrySegment tree.
    
    This reads the DNA parameters and builds a hierarchical structure
    that can be rendered.
    """
    # Handle dict (from JSON) or actual PlantDNA object
    if isinstance(dna, dict):
        plant_type = dna.get('plant_type', 'tree')
        height = dna.get('height_gene', {}).get('value', 1.0) if isinstance(dna.get('height_gene'), dict) else dna.get('height', 1.0)
        width = dna.get('width_gene', {}).get('value', 0.5) if isinstance(dna.get('width_gene'), dict) else dna.get('width', 0.5)
        trunk_color = _dict_to_color(dna.get('trunk_color'), (0.4, 0.25, 0.15))
        leaf_color = _dict_to_color(dna.get('leaf_color'), (0.2, 0.6, 0.2))
        flower_color = _dict_to_color(dna.get('flower_color'), (0.8, 0.4, 0.6))
        branch_count = int(dna.get('branch_count', 3)) if dna.get('branch_count') is not None else 3
        branch_angle = float(dna.get('branch_angle', 0.5)) if dna.get('branch_angle') is not None else 0.5
        leaf_density = float(dna.get('leaf_density', 0.5)) if dna.get('leaf_density') is not None else 0.5
        canopy_shape = dna.get('canopy_shape', 'dome')
        canopy_spread = dna.get('canopy_spread', 1.0)
    else:
        plant_type = getattr(dna, 'plant_type', 'tree')
        height = dna.height_gene.value if hasattr(dna, 'height_gene') else 1.0
        width = dna.width_gene.value if hasattr(dna, 'width_gene') else 0.5
        trunk_color = (dna.trunk_color.r, dna.trunk_color.g, dna.trunk_color.b) if hasattr(dna, 'trunk_color') else (0.4, 0.25, 0.15)
        leaf_color = (dna.leaf_color.r, dna.leaf_color.g, dna.leaf_color.b) if hasattr(dna, 'leaf_color') else (0.2, 0.6, 0.2)
        flower_color = (dna.flower_color.r, dna.flower_color.g, dna.flower_color.b) if hasattr(dna, 'flower_color') else (0.8, 0.4, 0.6)
        branch_count = getattr(dna, 'branch_count', 3)
        branch_angle = getattr(dna, 'branch_angle', 0.5)
        leaf_density = getattr(dna, 'leaf_density', 0.5)
        canopy_shape = getattr(dna, 'canopy_shape', 'dome')
        canopy_spread = getattr(dna, 'canopy_spread', 1.0)
    
    # Build trunk segment
    trunk = GeometrySegment(
        shape="cylinder",
        size_x=width * 0.15,
        size_y=height * 0.6,
        size_z=width * 0.15,
        color=trunk_color,
        lod=8
    )
    
    # Add canopy/foliage based on plant type
    if plant_type in ('tree', 'tall_tree'):
        # Tree with spherical canopy
        canopy = GeometrySegment(
            shape="ellipsoid",
            size_x=width * canopy_spread * 0.8,
            size_y=height * 0.4,
            size_z=width * canopy_spread * 0.8,
            color=leaf_color,
            offset_y=height * 0.1,
            lod=12
        )
        trunk.children.append(canopy)
        
        # Add branches
        for i in range(min(branch_count, 6)):
            angle = (i / max(branch_count, 1)) * 2 * math.pi
            branch = GeometrySegment(
                shape="cylinder",
                size_x=width * 0.05,
                size_y=height * 0.2,
                size_z=width * 0.05,
                color=trunk_color,
                offset_x=math.cos(angle) * width * 0.1,
                offset_z=math.sin(angle) * width * 0.1,
                rotation_z=branch_angle * 0.5 * math.cos(angle),
                rotation_x=branch_angle * 0.5 * math.sin(angle),
                lod=6
            )
            # Small foliage cluster on branch
            branch.children.append(GeometrySegment(
                shape="ellipsoid",
                size_x=width * 0.2,
                size_y=width * 0.15,
                size_z=width * 0.2,
                color=leaf_color,
                lod=8
            ))
            trunk.children.append(branch)
            
    elif plant_type == 'pine' or plant_type == 'conifer':
        # Conical layers
        layers = max(3, int(leaf_density * 5))
        for i in range(layers):
            layer_y = height * 0.2 + i * height * 0.15
            layer_w = width * (0.6 - i * 0.08)
            layer = GeometrySegment(
                shape="cone",
                size_x=layer_w,
                size_y=height * 0.15,
                size_z=layer_w,
                color=(leaf_color[0] * (0.8 + i * 0.05), 
                       leaf_color[1] * (0.8 + i * 0.05), 
                       leaf_color[2] * (0.8 + i * 0.05)),
                offset_y=layer_y,
                lod=8
            )
            trunk.children.append(layer)
            
    elif plant_type in ('bush', 'shrub'):
        # Multiple overlapping spheres
        for i in range(max(3, int(leaf_density * 5))):
            angle = i * 2.4  # Golden angle-ish
            dist = width * 0.2
            sphere = GeometrySegment(
                shape="ellipsoid",
                size_x=width * 0.3,
                size_y=height * 0.25,
                size_z=width * 0.3,
                color=leaf_color,
                offset_x=math.cos(angle) * dist,
                offset_y=height * 0.1,
                offset_z=math.sin(angle) * dist,
                lod=8
            )
            trunk.children.append(sphere)
            
    elif plant_type == 'flower':
        # Petals around center
        petal_count = max(5, branch_count)
        for i in range(petal_count):
            angle = (i / petal_count) * 2 * math.pi
            petal = GeometrySegment(
                shape="ellipsoid",
                size_x=width * 0.2,
                size_y=width * 0.05,
                size_z=width * 0.1,
                color=flower_color,
                offset_x=math.cos(angle) * width * 0.15,
                offset_y=0,
                offset_z=math.sin(angle) * width * 0.15,
                rotation_x=0.3,
                lod=6
            )
            trunk.children.append(petal)
        # Center
        trunk.children.append(GeometrySegment(
            shape="ellipsoid",
            size_x=width * 0.1,
            size_y=width * 0.08,
            size_z=width * 0.1,
            color=(0.9, 0.8, 0.2),
            lod=8
        ))
        
    elif plant_type == 'mushroom':
        # Override trunk to be stem
        trunk.color = (0.95, 0.9, 0.85)
        trunk.size_y = height * 0.4
        # Cap
        cap = GeometrySegment(
            shape="ellipsoid",
            size_x=width * 0.5,
            size_y=height * 0.15,
            size_z=width * 0.5,
            color=flower_color,
            lod=12
        )
        trunk.children.append(cap)
        
    elif plant_type == 'cactus':
        trunk.color = (0.2, 0.5, 0.2)
        trunk.size_y = height * 0.7
        # Arms
        if branch_count > 1:
            for i in range(min(2, branch_count - 1)):
                side = 1 if i == 0 else -1
                arm = GeometrySegment(
                    shape="cylinder",
                    size_x=width * 0.08,
                    size_y=height * 0.3,
                    size_z=width * 0.08,
                    color=(0.2, 0.5, 0.2),
                    offset_x=side * width * 0.15,
                    offset_y=-height * 0.1,
                    rotation_z=side * 0.5,
                    lod=6
                )
                trunk.children.append(arm)
                
    elif plant_type in ('grass', 'fern'):
        # Multiple blades
        blade_count = max(3, int(leaf_density * 8))
        trunk.size_y = 0.01  # Minimal base
        for i in range(blade_count):
            angle = (i / blade_count) * 2 * math.pi + (i * 0.1)
            blade = GeometrySegment(
                shape="cylinder",
                size_x=width * 0.02,
                size_y=height * (0.5 + (i % 3) * 0.15),
                size_z=width * 0.02,
                color=leaf_color,
                offset_x=math.cos(angle) * width * 0.05,
                offset_z=math.sin(angle) * width * 0.05,
                rotation_x=math.sin(angle) * 0.2,
                rotation_z=math.cos(angle) * 0.2,
                lod=4
            )
            trunk.children.append(blade)
            
    elif plant_type == 'spiral':
        # Twisted spiral shape
        trunk.size_y = height * 0.1
        segments = max(6, int(height * 4))
        current = trunk
        for i in range(segments):
            angle = i * 0.8
            seg = GeometrySegment(
                shape="ellipsoid",
                size_x=width * 0.1 * (1 - i * 0.05),
                size_y=height * 0.08,
                size_z=width * 0.1 * (1 - i * 0.05),
                color=(
                    leaf_color[0] * (0.7 + 0.3 * math.sin(i * 0.5)),
                    leaf_color[1] * (0.7 + 0.3 * math.sin(i * 0.5 + 2)),
                    leaf_color[2] * (0.7 + 0.3 * math.sin(i * 0.5 + 4))
                ),
                offset_x=math.sin(angle) * width * 0.1,
                offset_z=math.cos(angle) * width * 0.1,
                lod=6
            )
            current.children.append(seg)
            current = seg
            
    elif plant_type == 'crystal':
        # Angular crystalline structure
        trunk.shape = "box"
        trunk.size_x = width * 0.1
        trunk.size_y = height * 0.8
        trunk.size_z = width * 0.1
        trunk.color = (0.7, 0.85, 1.0)
        # Secondary crystals
        for i in range(3):
            angle = i * 2 * math.pi / 3
            crystal = GeometrySegment(
                shape="box",
                size_x=width * 0.06,
                size_y=height * 0.4,
                size_z=width * 0.06,
                color=(0.75, 0.88, 1.0),
                offset_x=math.cos(angle) * width * 0.15,
                offset_y=-height * 0.2,
                offset_z=math.sin(angle) * width * 0.15,
                rotation_z=0.3 * math.cos(angle),
                rotation_x=0.3 * math.sin(angle),
                lod=4
            )
            trunk.children.append(crystal)
            
    elif plant_type in ('octopus', 'tentacle', 'alien'):
        # Exotic organic shapes
        trunk.shape = "ellipsoid"
        trunk.size_x = width * 0.3
        trunk.size_y = height * 0.25
        trunk.size_z = width * 0.3
        trunk.color = leaf_color
        # Tentacles
        tentacle_count = max(4, branch_count)
        for i in range(tentacle_count):
            angle = (i / tentacle_count) * 2 * math.pi
            tentacle = GeometrySegment(
                shape="cylinder",
                size_x=width * 0.04,
                size_y=height * 0.4,
                size_z=width * 0.04,
                color=flower_color,
                offset_x=math.cos(angle) * width * 0.2,
                offset_z=math.sin(angle) * width * 0.2,
                rotation_x=0.6 * math.sin(angle),
                rotation_z=0.6 * math.cos(angle),
                lod=6
            )
            # Tip
            tentacle.children.append(GeometrySegment(
                shape="ellipsoid",
                size_x=width * 0.06,
                size_y=width * 0.06,
                size_z=width * 0.06,
                color=(flower_color[0] * 1.2, flower_color[1] * 1.2, flower_color[2] * 1.2),
                lod=6
            ))
            trunk.children.append(tentacle)
    else:
        # Default: simple tree
        canopy = GeometrySegment(
            shape="ellipsoid",
            size_x=width * 0.5,
            size_y=height * 0.3,
            size_z=width * 0.5,
            color=leaf_color,
            lod=10
        )
        trunk.children.append(canopy)
    
    return trunk


def animal_dna_to_geometry(dna: Any) -> GeometrySegment:
    """
    Convert AnimalDNA to a GeometrySegment tree.
    """
    # Handle dict (from JSON) or actual AnimalDNA object
    if isinstance(dna, dict):
        animal_type = dna.get('animal_type', 'mammal')
        body_segments = dna.get('body_segments', [])
        limbs = dna.get('limbs', [])
        features = dna.get('features', [])
        base_scale = float(dna.get('base_scale', 1.0))
        primary_color = _list_to_color(dna.get('primary_color'), (0.5, 0.4, 0.3))
    else:
        animal_type = getattr(dna, 'animal_type', 'mammal')
        body_segments = getattr(dna, 'body_segments', [])
        limbs = getattr(dna, 'limbs', [])
        features = getattr(dna, 'features', [])
        base_scale = getattr(dna, 'base_scale', 1.0)
        primary_color = getattr(dna, 'primary_color', (0.5, 0.4, 0.3))
        if hasattr(primary_color, '__iter__') and not isinstance(primary_color, str):
            primary_color = tuple(primary_color)
    
    # Create root segment from first body segment
    if body_segments:
        first_seg = body_segments[0]
        if isinstance(first_seg, dict):
            size = first_seg.get('size', [0.3, 0.3, 0.3])
            color = _list_to_color(first_seg.get('color'), primary_color)
            shape = first_seg.get('shape', 'ellipsoid')
        else:
            size = getattr(first_seg, 'size', (0.3, 0.3, 0.3))
            color = getattr(first_seg, 'color', primary_color)
            shape = getattr(first_seg, 'shape', 'ellipsoid')
        
        root = GeometrySegment(
            shape=shape,
            size_x=float(size[0]) * base_scale if len(size) > 0 else 0.3,
            size_y=float(size[1]) * base_scale if len(size) > 1 else 0.3,
            size_z=float(size[2]) * base_scale if len(size) > 2 else 0.3,
            color=color,
            lod=10
        )
    else:
        # Default body
        root = GeometrySegment(
            shape="ellipsoid",
            size_x=0.3 * base_scale,
            size_y=0.2 * base_scale,
            size_z=0.25 * base_scale,
            color=primary_color,
            lod=10
        )
    
    # Add remaining body segments as a chain
    current = root
    for i, seg_data in enumerate(body_segments[1:], 1):
        if isinstance(seg_data, dict):
            size = seg_data.get('size', [0.3, 0.3, 0.3])
            color = _list_to_color(seg_data.get('color'), primary_color)
            shape = seg_data.get('shape', 'ellipsoid')
        else:
            size = getattr(seg_data, 'size', (0.3, 0.3, 0.3))
            color = getattr(seg_data, 'color', primary_color)
            shape = getattr(seg_data, 'shape', 'ellipsoid')
        
        seg = GeometrySegment(
            shape=shape,
            size_x=float(size[0]) * base_scale if len(size) > 0 else 0.3,
            size_y=float(size[1]) * base_scale if len(size) > 1 else 0.3,
            size_z=float(size[2]) * base_scale if len(size) > 2 else 0.3,
            color=color,
            offset_x=-float(size[0]) * base_scale * 0.5,  # Chain horizontally
            lod=8
        )
        current.children.append(seg)
        current = seg
    
    # Add limbs
    for limb_data in limbs:
        if isinstance(limb_data, dict):
            limb_segs = limb_data.get('segments', [])
            limb_color = _list_to_color(limb_data.get('color'), primary_color)
            attach = limb_data.get('attachment_point', [0, 0, 0])
        else:
            limb_segs = getattr(limb_data, 'segments', [])
            limb_color = getattr(limb_data, 'color', primary_color)
            attach = getattr(limb_data, 'attachment_point', [0, 0, 0])
        
        if limb_segs:
            first_limb_seg = limb_segs[0]
            if isinstance(first_limb_seg, dict):
                length = float(first_limb_seg.get('length', 0.2)) * base_scale
                thickness = float(first_limb_seg.get('thickness', 0.05)) * base_scale
            else:
                length = getattr(first_limb_seg, 'length', 0.2) * base_scale
                thickness = getattr(first_limb_seg, 'thickness', 0.05) * base_scale
            
            limb = GeometrySegment(
                shape="cylinder",
                size_x=thickness,
                size_y=length,
                size_z=thickness,
                color=limb_color if isinstance(limb_color, tuple) else tuple(limb_color),
                offset_x=float(attach[0]) * base_scale if len(attach) > 0 else 0,
                offset_y=float(attach[1]) * base_scale if len(attach) > 1 else 0,
                offset_z=float(attach[2]) * base_scale if len(attach) > 2 else 0,
                lod=6
            )
            root.children.append(limb)
    
    # Add eyes from features
    for feat in features:
        if isinstance(feat, dict):
            feat_type = feat.get('feature_type', '')
            feat_size = float(feat.get('size', 0.1)) * base_scale
            feat_color = _list_to_color(feat.get('color'), (0.1, 0.1, 0.1))
            feat_count = int(feat.get('count', 2))
        else:
            feat_type = getattr(feat, 'feature_type', '')
            feat_size = getattr(feat, 'size', 0.1) * base_scale
            feat_color = getattr(feat, 'color', (0.1, 0.1, 0.1))
            feat_count = getattr(feat, 'count', 2)
        
        if feat_type == 'eye':
            # Add eyes
            for e in range(min(feat_count, 4)):
                side = 1 if e % 2 == 0 else -1
                eye = GeometrySegment(
                    shape="ellipsoid",
                    size_x=feat_size * 0.15,
                    size_y=feat_size * 0.15,
                    size_z=feat_size * 0.15,
                    color=(0.95, 0.95, 0.95),
                    offset_x=root.size_x * 0.4,
                    offset_y=root.size_y * 0.3,
                    offset_z=side * root.size_z * 0.3,
                    lod=6
                )
                # Pupil
                eye.children.append(GeometrySegment(
                    shape="ellipsoid",
                    size_x=feat_size * 0.08,
                    size_y=feat_size * 0.08,
                    size_z=feat_size * 0.08,
                    color=feat_color if isinstance(feat_color, tuple) else tuple(feat_color),
                    offset_x=feat_size * 0.05,
                    lod=4
                ))
                root.children.append(eye)
    
    return root


def _dict_to_color(data: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert {r, g, b} dict to tuple."""
    if data is None:
        return default
    if isinstance(data, dict):
        return (
            float(data.get('r', default[0])),
            float(data.get('g', default[1])),
            float(data.get('b', default[2]))
        )
    if isinstance(data, (list, tuple)) and len(data) >= 3:
        return (float(data[0]), float(data[1]), float(data[2]))
    return default


def _list_to_color(data: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert [r, g, b] list to tuple."""
    if data is None:
        return default
    if isinstance(data, (list, tuple)) and len(data) >= 3:
        return (float(data[0]), float(data[1]), float(data[2]))
    if isinstance(data, dict):
        return _dict_to_color(data, default)
    return default

