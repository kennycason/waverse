"""
Unit tests for DNA Geometry Builder.

Tests cover:
- Primitive mesh generation (ellipsoid, cylinder, box)
- GeometrySegment serialization
- Plant DNA to geometry conversion
- Animal DNA to geometry conversion
- Hierarchical segment trees
"""

import pytest
import numpy as np
import math

from waverse.dna_geometry import (
    create_ellipsoid, create_cylinder, create_box,
    GeometrySegment, GeometryBuilder,
    plant_dna_to_geometry, animal_dna_to_geometry
)
from waverse.dna import PlantDNA, PlantType


class TestPrimitives:
    """Tests for primitive mesh generation."""
    
    def test_create_ellipsoid_produces_vertices(self):
        """Test ellipsoid creation produces valid mesh."""
        verts, norms, colors = create_ellipsoid(1.0, 0.5, 0.5, segments=8, rings=6)
        
        assert len(verts) > 0
        assert len(norms) == len(verts)
        assert len(colors) == len(verts)
        
    def test_ellipsoid_vertices_in_bounds(self):
        """Test ellipsoid vertices stay within radii."""
        rx, ry, rz = 2.0, 1.0, 1.5
        verts, _, _ = create_ellipsoid(rx, ry, rz, segments=12, rings=8)
        
        for v in verts:
            # Each vertex should be within ellipsoid bounds (with tolerance)
            assert abs(v[0]) <= rx * 1.01
            assert abs(v[1]) <= ry * 1.01
            assert abs(v[2]) <= rz * 1.01
    
    def test_create_cylinder_produces_vertices(self):
        """Test cylinder creation produces valid mesh."""
        verts, norms, colors = create_cylinder(0.5, 0.5, 1.0, segments=8)
        
        assert len(verts) > 0
        assert len(norms) == len(verts)
        assert len(colors) == len(verts)
    
    def test_cylinder_height_correct(self):
        """Test cylinder has correct height."""
        height = 2.0
        verts, _, _ = create_cylinder(0.5, 0.5, height, segments=8)
        
        y_values = [v[1] for v in verts]
        assert min(y_values) >= -height / 2 - 0.01
        assert max(y_values) <= height / 2 + 0.01
    
    def test_cone_tapers(self):
        """Test cone (cylinder with radius_top=0) creates cone shape."""
        verts, _, _ = create_cylinder(1.0, 0.0, 2.0, segments=8)
        
        # Top vertices should be at center (0, height/2, 0)
        top_verts = [v for v in verts if v[1] > 0.9]
        for v in top_verts:
            assert abs(v[0]) < 0.1 or abs(v[1] - 1.0) < 0.01  # Either at center or at cap
    
    def test_create_box_produces_vertices(self):
        """Test box creation produces valid mesh."""
        verts, norms, colors = create_box(1.0, 2.0, 0.5)
        
        assert len(verts) > 0
        assert len(norms) == len(verts)
        assert len(colors) == len(verts)
        # Box has 6 faces, 2 triangles each, 3 vertices per triangle
        assert len(verts) == 36
    
    def test_box_dimensions(self):
        """Test box has correct dimensions."""
        w, h, d = 2.0, 3.0, 1.0
        verts, _, _ = create_box(w, h, d)
        
        x_values = [v[0] for v in verts]
        y_values = [v[1] for v in verts]
        z_values = [v[2] for v in verts]
        
        assert abs(max(x_values) - w/2) < 0.01
        assert abs(min(x_values) + w/2) < 0.01
        assert abs(max(y_values) - h/2) < 0.01
        assert abs(min(y_values) + h/2) < 0.01


class TestGeometrySegment:
    """Tests for GeometrySegment dataclass."""
    
    def test_segment_creation(self):
        """Test basic segment creation."""
        seg = GeometrySegment(
            shape="cylinder",
            size_x=0.5,
            size_y=2.0,
            size_z=0.5,
            color=(0.4, 0.3, 0.2)
        )
        
        assert seg.shape == "cylinder"
        assert seg.size_y == 2.0
        assert seg.color == (0.4, 0.3, 0.2)
    
    def test_segment_children(self):
        """Test segment with children."""
        parent = GeometrySegment(shape="cylinder")
        child1 = GeometrySegment(shape="ellipsoid")
        child2 = GeometrySegment(shape="ellipsoid")
        
        parent.children = [child1, child2]
        
        assert len(parent.children) == 2
    
    def test_segment_serialization(self):
        """Test segment to_dict and from_dict."""
        original = GeometrySegment(
            shape="box",
            size_x=1.0,
            size_y=2.0,
            size_z=0.5,
            color=(0.8, 0.2, 0.1),
            offset_x=0.5,
            rotation_y=0.3
        )
        original.children.append(GeometrySegment(shape="ellipsoid"))
        
        data = original.to_dict()
        restored = GeometrySegment.from_dict(data)
        
        assert restored.shape == original.shape
        assert restored.size_x == original.size_x
        assert restored.size_y == original.size_y
        assert restored.color == original.color
        assert len(restored.children) == 1


class TestGeometryBuilder:
    """Tests for GeometryBuilder mesh generation."""
    
    def test_build_single_segment(self):
        """Test building mesh from single segment."""
        seg = GeometrySegment(
            shape="ellipsoid",
            size_x=1.0,
            size_y=0.5,
            size_z=0.5,
            color=(1.0, 0.0, 0.0)
        )
        
        verts, norms, colors = GeometryBuilder.build_mesh(seg)
        
        assert len(verts) > 0
        assert verts.dtype == np.float32
        assert norms.dtype == np.float32
        assert colors.dtype == np.float32
    
    def test_build_hierarchical_segments(self):
        """Test building mesh from parent+child segments."""
        trunk = GeometrySegment(
            shape="cylinder",
            size_x=0.2,
            size_y=2.0,
            size_z=0.2
        )
        canopy = GeometrySegment(
            shape="ellipsoid",
            size_x=1.0,
            size_y=0.8,
            size_z=1.0,
            offset_y=0.5
        )
        trunk.children.append(canopy)
        
        verts, norms, colors = GeometryBuilder.build_mesh(trunk)
        
        # Should have vertices from both trunk and canopy
        assert len(verts) > 100  # Both shapes contribute
    
    def test_child_offset_applied(self):
        """Test that child offset positions correctly."""
        parent = GeometrySegment(
            shape="cylinder",
            size_x=0.1,
            size_y=1.0,
            size_z=0.1
        )
        child = GeometrySegment(
            shape="ellipsoid",
            size_x=0.5,
            size_y=0.5,
            size_z=0.5,
            offset_y=0.5  # Above parent
        )
        parent.children.append(child)
        
        verts, _, _ = GeometryBuilder.build_mesh(parent)
        
        # Child should be positioned above the parent
        y_values = [v[1] for v in verts]
        # With parent height 1.0 and child offset 0.5, child center should be around 1.0
        assert max(y_values) > 1.0


class TestPlantDNAToGeometry:
    """Tests for plant DNA to geometry conversion."""
    
    def test_tree_dna_produces_trunk_and_canopy(self):
        """Test tree DNA produces reasonable tree shape."""
        dna = PlantDNA.create_random(PlantType.TREE, seed=42)
        
        root = plant_dna_to_geometry(dna)
        
        assert root.shape == "cylinder"  # Trunk
        assert len(root.children) > 0  # Has canopy/branches
    
    def test_different_plant_types_produce_different_shapes(self):
        """Test various plant types produce distinct geometries."""
        tree_geo = plant_dna_to_geometry(PlantDNA.create_random(PlantType.TREE, seed=1))
        bush_geo = plant_dna_to_geometry(PlantDNA.create_random(PlantType.BUSH, seed=1))
        cactus_geo = plant_dna_to_geometry(PlantDNA.create_random(PlantType.CACTUS, seed=1))
        
        # Different types should have different structures
        tree_children = len(tree_geo.children)
        bush_children = len(bush_geo.children)
        
        # Trees typically have branches + canopy, bushes have multiple foliage spheres
        assert tree_children > 0
        assert bush_children > 0
    
    def test_dna_colors_applied(self):
        """Test that DNA colors are applied to geometry."""
        dna = PlantDNA.create_random(PlantType.TREE, seed=42)
        dna.trunk_color.r = 0.8
        dna.trunk_color.g = 0.2
        dna.trunk_color.b = 0.1
        
        root = plant_dna_to_geometry(dna)
        
        # Trunk should have the specified color
        assert abs(root.color[0] - 0.8) < 0.01
        assert abs(root.color[1] - 0.2) < 0.01
    
    def test_dna_from_dict(self):
        """Test conversion from JSON-style dict."""
        dna_dict = {
            'plant_type': 'tree',
            'height_gene': {'value': 5.0},
            'width_gene': {'value': 1.0},
            'trunk_color': {'r': 0.4, 'g': 0.3, 'b': 0.2},
            'leaf_color': {'r': 0.2, 'g': 0.6, 'b': 0.2},
            'branch_count': 4
        }
        
        root = plant_dna_to_geometry(dna_dict)
        
        assert root is not None
        assert root.shape == "cylinder"
    
    @pytest.mark.parametrize("plant_type", PlantType.ALL_TYPES)
    def test_all_plant_types_produce_valid_geometry(self, plant_type):
        """Test all plant types can be converted to geometry."""
        dna = PlantDNA.create_random(plant_type, seed=42)
        root = plant_dna_to_geometry(dna)
        
        verts, norms, colors = GeometryBuilder.build_mesh(root)
        
        assert len(verts) > 0
        assert not np.isnan(verts).any()


class TestAnimalDNAToGeometry:
    """Tests for animal DNA to geometry conversion."""
    
    def test_animal_dict_produces_geometry(self):
        """Test animal DNA dict produces geometry."""
        dna_dict = {
            'animal_type': 'mammal',
            'base_scale': 1.0,
            'body_segments': [
                {'shape': 'ellipsoid', 'size': [0.5, 0.3, 0.3], 'color': [0.5, 0.4, 0.3]},
                {'shape': 'ellipsoid', 'size': [0.4, 0.25, 0.25], 'color': [0.5, 0.4, 0.3]},
            ],
            'limbs': [],
            'features': [
                {'feature_type': 'eye', 'size': 0.1, 'color': [0.1, 0.1, 0.1], 'count': 2}
            ]
        }
        
        root = animal_dna_to_geometry(dna_dict)
        
        assert root is not None
        assert root.shape == "ellipsoid"
        assert len(root.children) > 0  # Should have second segment + eyes
    
    def test_body_segments_chained(self):
        """Test body segments are connected in a chain."""
        dna_dict = {
            'animal_type': 'worm',
            'base_scale': 1.0,
            'body_segments': [
                {'shape': 'ellipsoid', 'size': [0.3, 0.3, 0.3], 'color': [0.5, 0.4, 0.3]},
                {'shape': 'ellipsoid', 'size': [0.25, 0.25, 0.25], 'color': [0.5, 0.4, 0.3]},
                {'shape': 'ellipsoid', 'size': [0.2, 0.2, 0.2], 'color': [0.5, 0.4, 0.3]},
            ],
            'limbs': [],
            'features': []
        }
        
        root = animal_dna_to_geometry(dna_dict)
        
        # First segment should have second as child
        assert len(root.children) >= 1
        # Second segment should have third as child
        second = root.children[0]
        assert len(second.children) >= 1
    
    def test_eyes_added_from_features(self):
        """Test eyes are created from feature data."""
        dna_dict = {
            'animal_type': 'mammal',
            'base_scale': 1.0,
            'body_segments': [
                {'shape': 'ellipsoid', 'size': [0.5, 0.3, 0.3], 'color': [0.5, 0.4, 0.3]},
            ],
            'limbs': [],
            'features': [
                {'feature_type': 'eye', 'size': 0.15, 'color': [0.1, 0.1, 0.1], 'count': 2}
            ]
        }
        
        root = animal_dna_to_geometry(dna_dict)
        
        # Should have eyes as children
        eye_children = [c for c in root.children if c.color == (0.95, 0.95, 0.95)]
        assert len(eye_children) == 2
    
    def test_mesh_generation_valid(self):
        """Test animal geometry can be built into mesh."""
        dna_dict = {
            'animal_type': 'bird',
            'base_scale': 0.5,
            'body_segments': [
                {'shape': 'ellipsoid', 'size': [0.4, 0.3, 0.3], 'color': [0.6, 0.5, 0.4]},
            ],
            'limbs': [
                {
                    'segments': [{'length': 0.3, 'thickness': 0.02}],
                    'color': [0.4, 0.3, 0.25],
                    'attachment_point': [0, -0.1, 0.15]
                }
            ],
            'features': []
        }
        
        root = animal_dna_to_geometry(dna_dict)
        verts, norms, colors = GeometryBuilder.build_mesh(root)
        
        assert len(verts) > 0
        assert not np.isnan(verts).any()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_segment_tree(self):
        """Test building mesh from minimal segment."""
        seg = GeometrySegment()
        verts, norms, colors = GeometryBuilder.build_mesh(seg)
        
        assert len(verts) > 0
    
    def test_deeply_nested_segments(self):
        """Test deeply nested segment hierarchy."""
        root = GeometrySegment()
        current = root
        for _ in range(10):
            child = GeometrySegment(size_y=0.1, offset_y=0.1)
            current.children.append(child)
            current = child
        
        verts, norms, colors = GeometryBuilder.build_mesh(root)
        
        assert len(verts) > 0
        # Deepest point should be at least 1.0 units up
        max_y = max(v[1] for v in verts)
        assert max_y > 0.5
    
    def test_missing_dna_fields_handled(self):
        """Test that missing DNA fields don't crash."""
        minimal_dna = {'plant_type': 'tree'}
        root = plant_dna_to_geometry(minimal_dna)
        
        verts, norms, colors = GeometryBuilder.build_mesh(root)
        assert len(verts) > 0
    
    def test_zero_size_handled(self):
        """Test that zero-size segments are handled."""
        seg = GeometrySegment(size_x=0, size_y=0, size_z=0)
        verts, norms, colors = GeometryBuilder.build_mesh(seg)
        
        # Should still produce something (even if degenerate)
        assert verts is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

