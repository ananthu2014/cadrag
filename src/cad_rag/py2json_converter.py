#!/usr/bin/env python3
"""
Python-to-JSON CAD Converter
Converts Python CAD files back to JSON format with minimal information loss.
"""

import ast
import json
import math
import re
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import sys


class UUIDGenerator:
    """Generate consistent UUIDs based on content and context"""
    
    def __init__(self):
        self.entity_counter = 0
        self.profile_counter = 0
        self.curve_counter = 0
        self.used_uuids = set()
        
    def generate_entity_uuid(self, content_hash: str, entity_type: str, index: Optional[int] = None) -> str:
        """Generate entity UUID following F..._{index} pattern"""
        if index is None:
            index = self.entity_counter
            self.entity_counter += 1
            
        # Use content hash for deterministic base
        base = hashlib.md5(f"{content_hash}_{entity_type}".encode()).hexdigest()[:15]
        # Convert to mixed case alphanumeric (observed pattern)
        base = self._to_mixed_case_alphanum(base)
        
        uuid = f"F{base}_{index}"
        
        # Ensure uniqueness
        while uuid in self.used_uuids:
            index += 1
            uuid = f"F{base}_{index}"
        
        self.used_uuids.add(uuid)
        return uuid
    
    def generate_profile_uuid(self, profile_content: str) -> str:
        """Generate profile UUID following J[A-Z][A-Z] pattern"""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        # Use content hash for some determinism
        hash_val = hash(profile_content) % (26 * 26)
        first_letter = letters[hash_val // 26]
        second_letter = letters[hash_val % 26]
        
        uuid = f"J{first_letter}{second_letter}"
        
        # Ensure uniqueness
        counter = 0
        while uuid in self.used_uuids:
            counter += 1
            first_letter = letters[(hash_val // 26 + counter) % 26]
            second_letter = letters[(hash_val % 26 + counter) % 26]
            uuid = f"J{first_letter}{second_letter}"
        
        self.used_uuids.add(uuid)
        return uuid
    
    def generate_curve_uuid(self, curve_content: str) -> str:
        """Generate curve UUID following pattern from JSON analysis"""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        # Use content hash for determinism
        hash_val = hash(curve_content) % (26 * 26)
        base_letter = letters[hash_val // 26]
        second_letter = letters[hash_val % 26]
        
        # Sometimes has 'B' suffix (observed pattern)
        suffix = "B" if self.curve_counter % 3 == 0 else ""
        
        uuid = f"J{base_letter}{second_letter}{suffix}"
        
        # Ensure uniqueness
        counter = 0
        while uuid in self.used_uuids:
            counter += 1
            base_letter = letters[(hash_val // 26 + counter) % 26]
            second_letter = letters[(hash_val % 26 + counter) % 26]
            uuid = f"J{base_letter}{second_letter}{suffix}"
        
        self.used_uuids.add(uuid)
        self.curve_counter += 1
        return uuid
    
    def _to_mixed_case_alphanum(self, hex_string: str) -> str:
        """Convert hex to mixed case alphanumeric (mimics observed pattern)"""
        result = ""
        for i, char in enumerate(hex_string):
            if char.isdigit():
                result += char
            else:
                # Mix case based on position
                if i % 2 == 0:
                    result += char.upper()
                else:
                    result += char.lower()
        return result


class GeometryRecoverer:
    """Recover geometric information from Python CAD data"""
    
    def __init__(self, tolerance: float = 1e-10):
        self.tolerance = tolerance
    
    def recover_arc_parameters(self, start: List[float], end: List[float], mid: List[float]) -> Dict[str, Any]:
        """
        Given: start, end, mid points from Python
        Recover: center, radius, start_angle, end_angle, reference_vector, normal
        """
        start = np.array(start)
        end = np.array(end)
        mid = np.array(mid)
        
        # Step 1: Calculate center and radius using perpendicular bisectors
        center, radius = self._find_circle_center_and_radius(start, end, mid)
        
        if center is None or radius is None:
            raise ValueError("Cannot determine circle from collinear points")
        
        # Step 2: Calculate angles
        def angle_from_center(point, center):
            vec = point - center
            return math.atan2(vec[1], vec[0])
        
        start_angle = angle_from_center(start, center)
        mid_angle = angle_from_center(mid, center)
        end_angle = angle_from_center(end, center)
        
        # Step 3: Determine correct angle direction (handle 2π wrap-around)
        start_angle, end_angle = self._normalize_arc_angles(start_angle, mid_angle, end_angle)
        
        # Step 4: Calculate reference vector (JSON uses start point as reference)
        reference_vector = (start - center) / radius
        
        # Step 5: Determine normal (for 2D sketches, normal is always -z)
        normal = np.array([0.0, 0.0, -1.0])
        
        return {
            "center_point": {"x": float(center[0]), "y": float(center[1]), "z": 0.0},
            "radius": float(radius),
            "start_angle": float(start_angle),
            "end_angle": float(end_angle),
            "reference_vector": {"x": float(reference_vector[0]), "y": float(reference_vector[1]), "z": 0.0},
            "normal": {"x": float(normal[0]), "y": float(normal[1]), "z": float(normal[2])}
        }
    
    def _find_circle_center_and_radius(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Calculate center and radius using perpendicular bisectors"""
        ax, ay = p1[0], p1[1]
        bx, by = p2[0], p2[1] 
        cx, cy = p3[0], p3[1]
        
        # Calculate determinants
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < self.tolerance:
            return None, None  # Points are collinear
            
        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
        
        center = np.array([ux, uy])
        radius = np.linalg.norm(center - p1)
        
        return center, radius
    
    def _normalize_arc_angles(self, start_angle: float, mid_angle: float, end_angle: float) -> Tuple[float, float]:
        """Ensure mid_angle is between start and end angles"""
        # Ensure mid_angle is between start and end
        if start_angle > end_angle:
            start_angle, end_angle = end_angle, start_angle
            
        # Handle wrap-around cases
        if not (start_angle <= mid_angle <= end_angle):
            if mid_angle < start_angle:
                mid_angle += 2 * math.pi
            elif mid_angle > end_angle:
                end_angle += 2 * math.pi
                
        return start_angle, end_angle
    
    def recover_transform_matrix(self, origin: List[float], normal: List[float], x_axis: List[float]) -> Dict[str, Any]:
        """
        Given: origin, normal, x_axis from Python
        Recover: Complete orthonormal transform matrix for JSON
        """
        origin = np.array(origin)
        normal = np.array(normal)
        x_axis = np.array(x_axis)
        
        # Normalize input vectors
        normal = normal / np.linalg.norm(normal)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Calculate y_axis using right-hand rule
        y_axis = np.cross(normal, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Verify orthogonality
        if abs(np.dot(x_axis, y_axis)) > self.tolerance:
            raise ValueError("Generated axes are not orthogonal")
        
        # JSON uses z_axis as normal
        return {
            "origin": {"x": float(origin[0]), "y": float(origin[1]), "z": float(origin[2])},
            "x_axis": {"x": float(x_axis[0]), "y": float(x_axis[1]), "z": float(x_axis[2])},
            "y_axis": {"x": float(y_axis[0]), "y": float(y_axis[1]), "z": float(y_axis[2])},
            "z_axis": {"x": float(normal[0]), "y": float(normal[1]), "z": float(normal[2])}
        }
    
    def calculate_bounding_box(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate bounding box from all entities"""
        all_points = []
        
        # Extract geometry from all entities
        for entity_id, entity_data in entities.items():
            if entity_data.get('type') == 'Sketch':
                # Extract geometry from sketch profiles
                profiles = entity_data.get('profiles', {})
                for profile_id, profile_data in profiles.items():
                    loops = profile_data.get('loops', [])
                    for loop in loops:
                        curves = loop.get('profile_curves', [])
                        for curve in curves:
                            points = self._extract_geometry_points(curve)
                            all_points.extend(points)
        
        if not all_points:
            return {
                "min_point": {"x": 0.0, "y": 0.0, "z": 0.0},
                "max_point": {"x": 0.0, "y": 0.0, "z": 0.0},
                "type": "BoundingBox3D"
            }
        
        all_points = np.array(all_points)
        min_point = np.min(all_points, axis=0)
        max_point = np.max(all_points, axis=0)
        
        return {
            "min_point": {"x": float(min_point[0]), "y": float(min_point[1]), "z": float(min_point[2])},
            "max_point": {"x": float(max_point[0]), "y": float(max_point[1]), "z": float(max_point[2])},
            "type": "BoundingBox3D"
        }
    
    def _extract_geometry_points(self, geom: Dict) -> List[List[float]]:
        """Extract points from geometry for bounding box calculation"""
        points = []
        
        geom_type = geom.get('type')
        
        if geom_type == 'Line3D':
            start_pt = geom.get('start_point', {})
            end_pt = geom.get('end_point', {})
            points.extend([
                [start_pt.get('x', 0), start_pt.get('y', 0), start_pt.get('z', 0)],
                [end_pt.get('x', 0), end_pt.get('y', 0), end_pt.get('z', 0)]
            ])
        elif geom_type == 'Arc3D':
            # Sample points along arc using center and radius
            center_pt = geom.get('center_point', {})
            radius = geom.get('radius', 0)
            start_angle = geom.get('start_angle', 0)
            end_angle = geom.get('end_angle', 0)
            
            center = [center_pt.get('x', 0), center_pt.get('y', 0), center_pt.get('z', 0)]
            
            # Sample points along arc
            num_points = 16
            if abs(end_angle - start_angle) > 1e-6:
                for i in range(num_points + 1):
                    t = i / num_points
                    angle = start_angle + t * (end_angle - start_angle)
                    x = center[0] + radius * math.cos(angle)
                    y = center[1] + radius * math.sin(angle)
                    points.append([x, y, center[2]])
            else:
                # Add start and end points
                start_pt = geom.get('start_point', {})
                end_pt = geom.get('end_point', {})
                points.extend([
                    [start_pt.get('x', 0), start_pt.get('y', 0), start_pt.get('z', 0)],
                    [end_pt.get('x', 0), end_pt.get('y', 0), end_pt.get('z', 0)]
                ])
        elif geom_type == 'Circle3D':
            center_pt = geom.get('center_point', {})
            radius = geom.get('radius', 0)
            center = [center_pt.get('x', 0), center_pt.get('y', 0), center_pt.get('z', 0)]
            
            # Add extremal points
            points.extend([
                [center[0] + radius, center[1], center[2]],
                [center[0] - radius, center[1], center[2]],
                [center[0], center[1] + radius, center[2]],
                [center[0], center[1] - radius, center[2]]
            ])
        
        return points
    
    def _sample_arc_points(self, arc: Dict, num_points: int = 16) -> List[List[float]]:
        """Sample points along arc for bounding box calculation"""
        start = np.array(arc['start'])
        end = np.array(arc['end'])
        mid = np.array(arc['mid'])
        
        # Calculate center and radius
        center, radius = self._find_circle_center_and_radius(start, end, mid)
        
        if center is None:
            return [arc['start'], arc['end'], arc['mid']]
        
        # Calculate angles
        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
        
        # Handle angle wrapping
        if end_angle < start_angle:
            end_angle += 2 * math.pi
        
        # Sample points
        angles = np.linspace(start_angle, end_angle, num_points)
        points = []
        for angle in angles:
            point = center + radius * np.array([math.cos(angle), math.sin(angle)])
            points.append([float(point[0]), float(point[1]), 0.0])
        
        return points


class MetadataHandler:
    """Handle user-editable metadata in Python files"""
    
    def __init__(self):
        self.metadata_sections = [
            '_metadata_uuids',
            '_metadata_constraints', 
            '_metadata_precision',
            '_metadata_preferences',
            '_metadata_hints'
        ]
    
    def extract_metadata(self, python_code: str) -> Dict[str, Any]:
        """Extract metadata from Python code"""
        metadata = {}
        
        try:
            # Parse the Python code
            tree = ast.parse(python_code)
            
            # Find metadata assignments
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in self.metadata_sections:
                            try:
                                # Safely evaluate the metadata value
                                metadata[target.id] = ast.literal_eval(node.value)
                            except (ValueError, SyntaxError):
                                print(f"Warning: Could not parse metadata section: {target.id}")
                                continue
        except Exception as e:
            print(f"Warning: Could not parse Python code for metadata: {e}")
            return self._get_default_metadata()
        
        return metadata
    
    def _get_default_metadata(self) -> Dict[str, Any]:
        """Return default metadata when parsing fails"""
        return {
            '_metadata_uuids': {
                'regenerate_all': True,
                'preserve_entity_uuids': {},
                'preserve_profile_uuids': {}
            },
            '_metadata_constraints': {
                'loop_orientations': {},
                'arc_directions': {}
            },
            '_metadata_precision': {
                'coordinate_precision': 15,
                'angle_precision': 16,
                'geometric_tolerance': 1e-10
            },
            '_metadata_preferences': {
                'validate_geometry': True,
                'fix_discontinuities': True,
                'preserve_original_precision': False,
                'generate_bounding_box': True
            },
            '_metadata_hints': {
                'bounding_box': {'override_calculated': False}
            }
        }
    
    def generate_metadata_template(self, reconstructed_json: Dict[str, Any]) -> str:
        """Generate metadata template for new Python files"""
        template = """
# =============================================================================
# CAD_RECONSTRUCTION_METADATA
# This section contains hints for JSON reconstruction. Users can modify these
# values to control the reconstruction process. If removed, values will be
# automatically calculated.
# =============================================================================

_metadata_uuids = {
    "regenerate_all": False,  # Set to True to generate new UUIDs
    "preserve_entity_uuids": {
"""
        
        # Generate UUID entries for all entities
        for entity_id, entity_data in reconstructed_json.get('entities', {}).items():
            if entity_data.get('type') == 'Sketch':
                template += f'        "{entity_data.get("name", "Unknown")}": "{entity_id}",\n'
            elif entity_data.get('type') == 'ExtrudeFeature':
                template += f'        "{entity_data.get("name", "Unknown")}": "{entity_id}",\n'
        
        template += """    },
    "preserve_profile_uuids": {
"""
        
        # Generate profile UUID entries
        for entity_data in reconstructed_json.get('entities', {}).values():
            if entity_data.get('type') == 'Sketch':
                for profile_id in entity_data.get('profiles', {}):
                    template += f'        "Profile_{profile_id}": "{profile_id}",\n'
        
        template += """    }
}

_metadata_constraints = {
    "loop_orientations": {
        # Auto-detected, user can override
"""
        
        # Generate constraint entries
        for entity_data in reconstructed_json.get('entities', {}).values():
            if entity_data.get('type') == 'Sketch':
                sketch_name = entity_data.get('name', 'Unknown')
                template += f'        "{sketch_name}": {{"Loop0_0": "outer"}},\n'
        
        template += """    },
    "arc_directions": {
        # Auto-detected, user can override
    }
}

_metadata_preferences = {
    "validate_geometry": True,
    "fix_discontinuities": True,
    "preserve_original_precision": False,
    "generate_bounding_box": True,
}

_metadata_hints = {
    "bounding_box": {
        "override_calculated": False,
        "min_point": {"x": 0.0, "y": 0.0, "z": 0.0},
        "max_point": {"x": 0.0, "y": 0.0, "z": 0.0},
    }
}
"""
        
        return template


class PythonASTParser:
    """Parse Python AST to extract CAD operations"""
    
    def __init__(self):
        self.uuid_gen = UUIDGenerator()
        self.geometry_recoverer = GeometryRecoverer()
        
    def parse_python_file(self, python_code: str) -> Dict[str, Any]:
        """Parse Python code to extract CAD operations"""
        
        # Parse AST
        tree = ast.parse(python_code)
        
        # Extract operations
        operations = self._extract_operations(tree)
        
        # Build entity relationships
        entities = self._build_entities(operations)
        
        # Generate sequence
        sequence = self._generate_sequence(operations)
        
        # Calculate bounding box
        bounding_box = self.geometry_recoverer.calculate_bounding_box(entities)
        
        return {
            "entities": entities,
            "properties": {"bounding_box": bounding_box},
            "sequence": sequence
        }
    
    def _extract_operations(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract CAD operations from AST"""
        operations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Check if this is a CAD operation
                if isinstance(node.value, ast.Call):
                    func_name = self._get_function_name(node.value)
                    if func_name in ['add_line', 'add_arc', 'add_circle', 'add_loop', 'add_profile', 
                                   'add_sketch', 'add_sketchplane', 'add_sketchplane_ref', 'add_extrude']:
                        
                        var_name = self._get_variable_name(node.targets[0])
                        args = self._extract_arguments(node.value)
                        
                        operations.append({
                            'type': func_name,
                            'variable': var_name,
                            'args': args,
                            'ast_node': node
                        })
        
        return operations
    
    def _get_function_name(self, call_node: ast.Call) -> Optional[str]:
        """Get function name from call node"""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        return None
    
    def _get_variable_name(self, target: ast.expr) -> Optional[str]:
        """Get variable name from assignment target"""
        if isinstance(target, ast.Name):
            return target.id
        return None
    
    def _extract_arguments(self, call_node: ast.Call) -> Dict[str, Any]:
        """Extract arguments from function call"""
        args = {}
        
        # Positional arguments
        for i, arg in enumerate(call_node.args):
            try:
                args[f'arg_{i}'] = ast.literal_eval(arg)
            except (ValueError, SyntaxError):
                # If it's a variable name, get the name
                if isinstance(arg, ast.Name):
                    args[f'arg_{i}'] = arg.id
                else:
                    args[f'arg_{i}'] = str(arg)
        
        # Keyword arguments
        for keyword in call_node.keywords:
            try:
                args[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, SyntaxError):
                # If it's a variable name, get the name
                if isinstance(keyword.value, ast.Name):
                    args[keyword.arg] = keyword.value.id
                else:
                    args[keyword.arg] = str(keyword.value)
        
        return args
    
    def _build_entities(self, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build JSON entities from operations"""
        entities = {}
        
        # Build profile ID mapping first
        self.profile_id_map = self._build_profile_id_mapping(operations)
        
        # Group operations by type
        sketches = [op for op in operations if op['type'] == 'add_sketch']
        extrudes = [op for op in operations if op['type'] == 'add_extrude']
        sketch_planes = [op for op in operations if op['type'] in ['add_sketchplane', 'add_sketchplane_ref']]
        
        # Process sketches
        for sketch_op in sketches:
            sketch_entity = self._build_sketch_entity(sketch_op, operations)
            if sketch_entity:
                entities.update(sketch_entity)
        
        # Process extrudes
        for extrude_op in extrudes:
            extrude_entity = self._build_extrude_entity(extrude_op, operations)
            if extrude_entity:
                entities.update(extrude_entity)
        
        return entities
    
    def _build_profile_id_mapping(self, operations: List[Dict[str, Any]]) -> Dict[str, str]:
        """Build mapping from profile variables to profile IDs"""
        profile_map = {}
        
        # Find all profile operations
        profile_ops = [op for op in operations if op['type'] == 'add_profile']
        
        for profile_op in profile_ops:
            profile_var = profile_op['variable']
            profile_id = self.uuid_gen.generate_profile_uuid(str(profile_op))
            profile_map[profile_var] = profile_id
        
        return profile_map
    
    def _build_sketch_entity(self, sketch_op: Dict[str, Any], all_operations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build sketch entity from operation"""
        
        # Generate UUID
        content_hash = str(hash(str(sketch_op)))
        entity_id = self.uuid_gen.generate_entity_uuid(content_hash, "Sketch")
        
        # Find associated sketch plane
        sketch_plane_var = sketch_op['args'].get('sketch_plane')
        if not sketch_plane_var:
            return None
        
        # Find sketch plane operation
        sketch_plane_op = None
        for op in all_operations:
            if op['variable'] == sketch_plane_var:
                sketch_plane_op = op
                break
        
        if not sketch_plane_op:
            return None
        
        # Build transform matrix
        transform = self._build_transform_matrix(sketch_plane_op)
        
        # Find profiles
        profile_var = sketch_op['args'].get('profile')
        profiles = self._build_profiles(profile_var, all_operations)
        
        return {
            entity_id: {
                "type": "Sketch",
                "name": sketch_op['variable'],
                "transform": transform,
                "profiles": profiles,
                "reference_plane": {}
            }
        }
    
    def _build_transform_matrix(self, sketch_plane_op: Dict[str, Any]) -> Dict[str, Any]:
        """Build transform matrix from sketch plane operation"""
        
        if sketch_plane_op['type'] == 'add_sketchplane':
            # Absolute sketch plane
            origin = sketch_plane_op['args'].get('origin', [0, 0, 0])
            normal = sketch_plane_op['args'].get('normal', [0, 0, 1])
            x_axis = sketch_plane_op['args'].get('x_axis', [1, 0, 0])
            
            try:
                return self.geometry_recoverer.recover_transform_matrix(origin, normal, x_axis)
            except Exception as e:
                print(f"Warning: Could not recover transform matrix: {e}")
                return self._default_transform_matrix()
        
        elif sketch_plane_op['type'] == 'add_sketchplane_ref':
            # Referenced sketch plane - resolve based on reference
            return self._resolve_referenced_sketch_plane(sketch_plane_op)
        
        else:
            return self._default_transform_matrix()
    
    def _resolve_referenced_sketch_plane(self, sketch_plane_op: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve referenced sketch plane transform matrix"""
        
        # Extract reference parameters
        ref_entity = sketch_plane_op['args'].get('arg_0')  # Referenced entity
        origin = sketch_plane_op['args'].get('origin', [0, 0])
        plane_type = sketch_plane_op['args'].get('type', 'sameplane')
        
        # For now, we'll implement common reference plane types
        if plane_type == 'sameplane':
            # Same plane as reference - typically XY plane
            return {
                "origin": {"x": float(origin[0]), "y": float(origin[1]), "z": 0.0},
                "x_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
                "y_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                "z_axis": {"x": 0.0, "y": 0.0, "z": 1.0}
            }
        
        elif plane_type == 'offset':
            # Offset plane from reference
            offset_distance = sketch_plane_op['args'].get('offset', 0.0)
            return {
                "origin": {"x": float(origin[0]), "y": float(origin[1]), "z": float(offset_distance)},
                "x_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
                "y_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
                "z_axis": {"x": 0.0, "y": 0.0, "z": 1.0}
            }
        
        elif plane_type == 'perpendicular':
            # Perpendicular plane
            return {
                "origin": {"x": float(origin[0]), "y": float(origin[1]), "z": 0.0},
                "x_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
                "y_axis": {"x": 0.0, "y": 0.0, "z": 1.0},
                "z_axis": {"x": 0.0, "y": -1.0, "z": 0.0}
            }
        
        else:
            # Unknown reference type - return default
            print(f"Warning: Unknown reference plane type: {plane_type}")
            return self._default_transform_matrix()
    
    def _default_transform_matrix(self) -> Dict[str, Any]:
        """Return default transform matrix"""
        return {
            "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
            "x_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
            "y_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
            "z_axis": {"x": 0.0, "y": 0.0, "z": 1.0}
        }
    
    def _build_profiles(self, profile_var: str, all_operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build profiles from profile variable"""
        profiles = {}
        
        # Find profile operation
        profile_op = None
        for op in all_operations:
            if op['variable'] == profile_var:
                profile_op = op
                break
        
        if not profile_op:
            return profiles
        
        # Generate profile UUID
        profile_id = self.uuid_gen.generate_profile_uuid(str(profile_op))
        
        # Find loops
        loops_var = profile_op['args'].get('arg_0')  # First argument is usually loops
        loops = self._build_loops(loops_var, all_operations)
        
        profiles[profile_id] = {
            "loops": loops,
            "properties": {}
        }
        
        return profiles
    
    def _build_loops(self, loops_var: str, all_operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build loops from loops variable"""
        loops = []
        
        if not loops_var:
            return loops
        
        # The Bethany code generator has a bug - it creates loops but doesn't append them to loop lists
        # We need to infer which loops belong to which profiles based on variable naming patterns
        
        # Extract profile identifier from loops_var (e.g., "Loops0" -> "0")
        if loops_var.startswith("Loops"):
            profile_id = loops_var.replace("Loops", "")
            
            # Find loop operations that match this profile pattern
            loop_ops = []
            for op in all_operations:
                if op['type'] == 'add_loop' and op['variable']:
                    # Check if the variable matches the pattern (e.g., "Loop0_0" matches profile "0")
                    var_name = op['variable']
                    if var_name.startswith(f'Loop{profile_id}_'):
                        loop_ops.append(op)
            
            # Sort by variable name to maintain order
            loop_ops.sort(key=lambda x: x['variable'])
            
            for loop_op in loop_ops:
                curves_var = loop_op['args'].get('arg_0')  # First argument is usually curves
                
                # If curves_var is not provided, infer from loop variable name
                if not curves_var:
                    loop_var = loop_op['variable']
                    if loop_var.startswith('Loop'):
                        # Convert "Loop0_0" to "Curves0_0"
                        curves_var = loop_var.replace('Loop', 'Curves')
                
                profile_curves = self._build_profile_curves(curves_var, all_operations)
                
                # Determine loop orientation
                is_outer = self._determine_loop_orientation(profile_curves, loops)
                
                loops.append({
                    "is_outer": is_outer,
                    "profile_curves": profile_curves
                })
        
        return loops
    
    def _determine_loop_orientation(self, profile_curves: List[Dict[str, Any]], existing_loops: List[Dict[str, Any]]) -> bool:
        """
        Determine if a loop is outer (True) or inner (False) based on geometry.
        Uses polygon area calculation and containment testing.
        """
        
        if not profile_curves:
            return True
        
        # If no existing loops, this is the first (outer) loop
        if not existing_loops:
            return True
        
        # Extract points from profile curves
        loop_points = self._extract_loop_points(profile_curves)
        
        if len(loop_points) < 3:
            return True
        
        # Calculate signed area using shoelace formula
        signed_area = self._calculate_signed_area(loop_points)
        
        # Check if this loop is contained within any existing outer loop
        for existing_loop in existing_loops:
            if existing_loop.get('is_outer', False):
                existing_points = self._extract_loop_points(existing_loop['profile_curves'])
                if self._is_loop_contained(loop_points, existing_points):
                    # This loop is inside an outer loop, so it's inner
                    return False
        
        # If not contained and has significant area, it's outer
        return abs(signed_area) > self.geometry_recoverer.tolerance
    
    def _extract_loop_points(self, profile_curves: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """Extract 2D points from profile curves"""
        points = []
        
        for curve in profile_curves:
            if curve.get('type') == 'Line3D':
                start_pt = curve.get('start_point', {})
                points.append((start_pt.get('x', 0), start_pt.get('y', 0)))
            elif curve.get('type') == 'Arc3D':
                start_pt = curve.get('start_point', {})
                points.append((start_pt.get('x', 0), start_pt.get('y', 0)))
                # Add sampled points along arc for better containment testing
                arc_points = self._sample_arc_points_for_containment(curve)
                points.extend(arc_points)
            elif curve.get('type') == 'Circle3D':
                center = curve.get('center_point', {})
                radius = curve.get('radius', 0)
                # Sample points around circle
                for i in range(8):
                    angle = i * 2 * math.pi / 8
                    x = center.get('x', 0) + radius * math.cos(angle)
                    y = center.get('y', 0) + radius * math.sin(angle)
                    points.append((x, y))
        
        return points
    
    def _sample_arc_points_for_containment(self, arc_curve: Dict[str, Any], num_points: int = 5) -> List[Tuple[float, float]]:
        """Sample points along an arc for containment testing"""
        points = []
        
        try:
            center = arc_curve.get('center_point', {})
            radius = arc_curve.get('radius', 0)
            start_angle = arc_curve.get('start_angle', 0)
            end_angle = arc_curve.get('end_angle', 0)
            
            # Sample points along arc
            for i in range(1, num_points):
                t = i / num_points
                angle = start_angle + t * (end_angle - start_angle)
                x = center.get('x', 0) + radius * math.cos(angle)
                y = center.get('y', 0) + radius * math.sin(angle)
                points.append((x, y))
        except:
            pass
        
        return points
    
    def _calculate_signed_area(self, points: List[Tuple[float, float]]) -> float:
        """Calculate signed area of polygon using shoelace formula"""
        if len(points) < 3:
            return 0.0
        
        area = 0.0
        n = len(points)
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return area / 2.0
    
    def _is_loop_contained(self, inner_points: List[Tuple[float, float]], 
                          outer_points: List[Tuple[float, float]]) -> bool:
        """Test if inner loop is contained within outer loop using ray casting"""
        if not inner_points or not outer_points:
            return False
        
        # Test a representative point from inner loop
        test_point = inner_points[0]
        
        # Ray casting algorithm
        x, y = test_point
        inside = False
        
        j = len(outer_points) - 1
        for i in range(len(outer_points)):
            xi, yi = outer_points[i]
            xj, yj = outer_points[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    def _build_profile_curves(self, curves_var: str, all_operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build profile curves from curves variable"""
        profile_curves = []
        
        # The Bethany code generator has a bug - it creates curves but doesn't append them to curve lists
        # We need to infer which curves belong to which loops based on variable naming patterns
        
        if not curves_var:
            return profile_curves
        
        # Extract loop identifier from curves_var (e.g., "Curves0_0" -> "0_0")
        if "_" in curves_var:
            loop_id = curves_var.replace("Curves", "")
            
            # Find curve operations that match this loop pattern
            curve_ops = []
            for op in all_operations:
                if op['type'] in ['add_line', 'add_arc', 'add_circle'] and op['variable']:
                    # Check if the variable matches the pattern (e.g., "Line0_0_0" matches loop "0_0")
                    var_name = op['variable']
                    if (var_name.startswith(f'Line{loop_id}_') or 
                        var_name.startswith(f'Arc{loop_id}_') or 
                        var_name.startswith(f'Circle{loop_id}_')):
                        curve_ops.append(op)
            
            # Sort by variable name to maintain order
            curve_ops.sort(key=lambda x: x['variable'])
            
            for curve_op in curve_ops:
                curve_data = self._build_curve_data(curve_op)
                if curve_data:
                    profile_curves.append(curve_data)
        
        return profile_curves
    
    def _build_curve_data(self, curve_op: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build curve data from curve operation"""
        
        curve_id = self.uuid_gen.generate_curve_uuid(str(curve_op))
        
        if curve_op['type'] == 'add_line':
            start = curve_op['args'].get('start', [0, 0])
            end = curve_op['args'].get('end', [0, 0])
            
            return {
                "type": "Line3D",
                "start_point": {"x": float(start[0]), "y": float(start[1]), "z": 0.0},
                "end_point": {"x": float(end[0]), "y": float(end[1]), "z": 0.0},
                "curve": curve_id
            }
        
        elif curve_op['type'] == 'add_arc':
            start = curve_op['args'].get('start', [0, 0])
            end = curve_op['args'].get('end', [0, 0])
            mid = curve_op['args'].get('mid', [0, 0])
            
            try:
                arc_params = self.geometry_recoverer.recover_arc_parameters(start, end, mid)
                return {
                    "type": "Arc3D",
                    "start_point": {"x": float(start[0]), "y": float(start[1]), "z": 0.0},
                    "end_point": {"x": float(end[0]), "y": float(end[1]), "z": 0.0},
                    "curve": curve_id,
                    **arc_params
                }
            except Exception as e:
                print(f"Warning: Could not recover arc parameters: {e}")
                return None
        
        elif curve_op['type'] == 'add_circle':
            center = curve_op['args'].get('center', [0, 0])
            radius = curve_op['args'].get('radius', 1.0)
            
            return {
                "type": "Circle3D",
                "center_point": {"x": float(center[0]), "y": float(center[1]), "z": 0.0},
                "radius": float(radius),
                "curve": curve_id,
                "normal": {"x": 0.0, "y": 0.0, "z": -1.0}
            }
        
        return None
    
    def _build_extrude_entity(self, extrude_op: Dict[str, Any], all_operations: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build extrude entity from operation"""
        
        # Generate UUID
        content_hash = str(hash(str(extrude_op)))
        entity_id = self.uuid_gen.generate_entity_uuid(content_hash, "ExtrudeFeature")
        
        # Extract extrude parameters
        operation = self._map_operation_type(extrude_op['args'].get('operation', 0))
        extent_type = self._map_extent_type(extrude_op['args'].get('type', 0))
        extent_one = float(extrude_op['args'].get('extent_one', 0.0))
        extent_two = float(extrude_op['args'].get('extent_two', 0.0))
        
        # Find associated sketch
        sketch_var = extrude_op['args'].get('sketch')
        if not sketch_var:
            return None
        
        # Find sketch entity ID
        sketch_entity_id = None
        for op in all_operations:
            if op['variable'] == sketch_var and op['type'] == 'add_sketch':
                content_hash = str(hash(str(op)))
                sketch_entity_id = self.uuid_gen.generate_entity_uuid(content_hash, "Sketch")
                break
        
        if not sketch_entity_id:
            return None
        
        # Build profiles reference using actual profile IDs
        profiles = self._build_extrude_profiles(sketch_var, sketch_entity_id, all_operations)
        
        return {
            entity_id: {
                "type": "ExtrudeFeature",
                "name": extrude_op['variable'],
                "operation": operation,
                "extent_type": extent_type,
                "extent_one": {
                    "type": "DistanceExtentDefinition",
                    "distance": {
                        "type": "ModelParameter",
                        "role": "AlongDistance",
                        "name": "none",
                        "value": extent_one
                    },
                    "taper_angle": {
                        "type": "ModelParameter",
                        "role": "TaperAngle",
                        "name": "none",
                        "value": 0.0
                    }
                },
                "extent_two": {
                    "type": "DistanceExtentDefinition",
                    "distance": {
                        "type": "ModelParameter",
                        "role": "AgainstDistance",
                        "name": "none",
                        "value": extent_two
                    },
                    "taper_angle": {
                        "type": "ModelParameter",
                        "role": "Side2TaperAngle",
                        "name": "none",
                        "value": 0.0
                    }
                },
                "start_extent": {"type": "ProfilePlaneStartDefinition"},
                "profiles": profiles
            }
        }
    
    def _build_extrude_profiles(self, sketch_var: str, sketch_entity_id: str, all_operations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Build profiles reference for extrude operations"""
        profiles = []
        
        # Find the sketch operation
        sketch_op = None
        for op in all_operations:
            if op['variable'] == sketch_var and op['type'] == 'add_sketch':
                sketch_op = op
                break
        
        if not sketch_op:
            # Fallback to default profile
            return [{"profile": "JGC", "sketch": sketch_entity_id}]
        
        # Find the profile variable from sketch operation
        profile_var = sketch_op['args'].get('profile')
        if not profile_var:
            return [{"profile": "JGC", "sketch": sketch_entity_id}]
        
        # Get the actual profile ID from mapping
        profile_id = self.profile_id_map.get(profile_var, "JGC")
        
        return [{"profile": profile_id, "sketch": sketch_entity_id}]
    
    def _map_operation_type(self, operation_code: int) -> str:
        """Map operation code to string"""
        operation_map = {
            0: "NewBodyFeatureOperation",
            1: "JoinFeatureOperation",
            2: "CutFeatureOperation",
            3: "IntersectFeatureOperation"
        }
        return operation_map.get(operation_code, "NewBodyFeatureOperation")
    
    def _map_extent_type(self, extent_code: int) -> str:
        """Map extent code to string"""
        extent_map = {
            0: "OneSideFeatureExtentType",
            1: "TwoSidesFeatureExtentType",
            2: "SymmetricFeatureExtentType"
        }
        return extent_map.get(extent_code, "OneSideFeatureExtentType")
    
    def _generate_sequence(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate sequence from operations"""
        sequence = []
        index = 0
        
        # Group operations by type and sort by variable name
        sketches = sorted([op for op in operations if op['type'] == 'add_sketch'], 
                         key=lambda x: x['variable'])
        extrudes = sorted([op for op in operations if op['type'] == 'add_extrude'], 
                         key=lambda x: x['variable'])
        
        # Interleave sketches and extrudes based on typical pattern
        max_items = max(len(sketches), len(extrudes))
        
        for i in range(max_items):
            if i < len(sketches):
                content_hash = str(hash(str(sketches[i])))
                entity_id = self.uuid_gen.generate_entity_uuid(content_hash, "Sketch")
                sequence.append({
                    "index": index,
                    "type": "Sketch",
                    "entity": entity_id
                })
                index += 1
            
            if i < len(extrudes):
                content_hash = str(hash(str(extrudes[i])))
                entity_id = self.uuid_gen.generate_entity_uuid(content_hash, "ExtrudeFeature")
                sequence.append({
                    "index": index,
                    "type": "ExtrudeFeature",
                    "entity": entity_id
                })
                index += 1
        
        return sequence
    
    def _calculate_bounding_box(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate bounding box from entities"""
        
        # Extract all geometry
        all_geometry = []
        
        for entity_data in entities.values():
            if entity_data.get('type') == 'Sketch':
                for profile_data in entity_data.get('profiles', {}).values():
                    for loop_data in profile_data.get('loops', []):
                        for curve_data in loop_data.get('profile_curves', []):
                            if curve_data.get('type') == 'Line3D':
                                all_geometry.append({
                                    'type': 'line',
                                    'start': [curve_data['start_point']['x'], curve_data['start_point']['y'], 0],
                                    'end': [curve_data['end_point']['x'], curve_data['end_point']['y'], 0]
                                })
                            elif curve_data.get('type') == 'Arc3D':
                                all_geometry.append({
                                    'type': 'arc',
                                    'start': [curve_data['start_point']['x'], curve_data['start_point']['y'], 0],
                                    'end': [curve_data['end_point']['x'], curve_data['end_point']['y'], 0],
                                    'center': [curve_data['center_point']['x'], curve_data['center_point']['y'], 0],
                                    'radius': curve_data['radius']
                                })
                            elif curve_data.get('type') == 'Circle3D':
                                all_geometry.append({
                                    'type': 'circle',
                                    'center': [curve_data['center_point']['x'], curve_data['center_point']['y'], 0],
                                    'radius': curve_data['radius']
                                })
        
        return self.geometry_recoverer.calculate_bounding_box(all_geometry)


class PythonToJSONConverter:
    """Main converter class"""
    
    def __init__(self):
        self.parser = PythonASTParser()
        self.metadata_handler = MetadataHandler()
        self.geometry_recoverer = GeometryRecoverer()
    
    def convert_file(self, input_file: str, output_file: str, add_metadata: bool = False):
        """Convert a Python file to JSON"""
        
        print(f"Converting {input_file} to {output_file}")
        
        # Read input file
        with open(input_file, 'r') as f:
            python_code = f.read()
        
        # Extract metadata
        metadata = self.metadata_handler.extract_metadata(python_code)
        
        # Parse Python code
        json_data = self.parser.parse_python_file(python_code)
        
        # Apply metadata overrides
        json_data = self._apply_metadata_overrides(json_data, metadata)
        
        # Add metadata template if requested
        if add_metadata:
            metadata_template = self.metadata_handler.generate_metadata_template(json_data)
            
            # Add metadata to Python file
            with open(input_file, 'a') as f:
                f.write(metadata_template)
        
        # Write output file
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"✅ Successfully converted to {output_file}")
    
    def _apply_metadata_overrides(self, json_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Apply metadata overrides to JSON data"""
        
        # Apply UUID overrides
        if not metadata.get('_metadata_uuids', {}).get('regenerate_all', False):
            entity_uuid_overrides = metadata.get('_metadata_uuids', {}).get('preserve_entity_uuids', {})
            profile_uuid_overrides = metadata.get('_metadata_uuids', {}).get('preserve_profile_uuids', {})
            
            # Apply entity UUID overrides
            json_data = self._apply_entity_uuid_overrides(json_data, entity_uuid_overrides)
            
            # Apply profile UUID overrides
            json_data = self._apply_profile_uuid_overrides(json_data, profile_uuid_overrides)
        
        # Apply constraint overrides
        constraints = metadata.get('_metadata_constraints', {})
        if constraints:
            json_data = self._apply_constraint_overrides(json_data, constraints)
        
        # Apply preference overrides
        preferences = metadata.get('_metadata_preferences', {})
        if preferences:
            json_data = self._apply_preference_overrides(json_data, preferences)
        
        # Apply hint overrides
        hints = metadata.get('_metadata_hints', {})
        if hints:
            json_data = self._apply_hint_overrides(json_data, hints)
        
        return json_data
    
    def _apply_entity_uuid_overrides(self, json_data: Dict[str, Any], uuid_overrides: Dict[str, str]) -> Dict[str, Any]:
        """Apply entity UUID overrides"""
        if not uuid_overrides:
            return json_data
        
        # Create mapping from entity names to new UUIDs
        entities = json_data.get('entities', {})
        new_entities = {}
        uuid_mapping = {}
        
        for old_entity_id, entity_data in entities.items():
            entity_name = entity_data.get('name', '')
            
            if entity_name in uuid_overrides:
                new_entity_id = uuid_overrides[entity_name]
                new_entities[new_entity_id] = entity_data
                uuid_mapping[old_entity_id] = new_entity_id
            else:
                new_entities[old_entity_id] = entity_data
        
        # Update sequence references
        sequence = json_data.get('sequence', [])
        for seq_item in sequence:
            old_entity_id = seq_item.get('entity')
            if old_entity_id in uuid_mapping:
                seq_item['entity'] = uuid_mapping[old_entity_id]
        
        json_data['entities'] = new_entities
        return json_data
    
    def _apply_profile_uuid_overrides(self, json_data: Dict[str, Any], uuid_overrides: Dict[str, str]) -> Dict[str, Any]:
        """Apply profile UUID overrides"""
        if not uuid_overrides:
            return json_data
        
        # Update profile IDs in sketches
        entities = json_data.get('entities', {})
        for entity_data in entities.values():
            if entity_data.get('type') == 'Sketch':
                profiles = entity_data.get('profiles', {})
                new_profiles = {}
                
                for old_profile_id, profile_data in profiles.items():
                    # Find matching override
                    new_profile_id = old_profile_id
                    for override_key, override_value in uuid_overrides.items():
                        if override_key.endswith(old_profile_id) or old_profile_id in override_key:
                            new_profile_id = override_value
                            break
                    
                    new_profiles[new_profile_id] = profile_data
                
                entity_data['profiles'] = new_profiles
        
        return json_data
    
    def _apply_constraint_overrides(self, json_data: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Apply constraint overrides"""
        
        # Apply loop orientation overrides
        loop_orientations = constraints.get('loop_orientations', {})
        if loop_orientations:
            entities = json_data.get('entities', {})
            for entity_data in entities.values():
                if entity_data.get('type') == 'Sketch':
                    sketch_name = entity_data.get('name', '')
                    if sketch_name in loop_orientations:
                        sketch_constraints = loop_orientations[sketch_name]
                        
                        # Apply loop orientation overrides
                        profiles = entity_data.get('profiles', {})
                        for profile_data in profiles.values():
                            loops = profile_data.get('loops', [])
                            for i, loop_data in enumerate(loops):
                                loop_key = f"Loop{i}_0"
                                if loop_key in sketch_constraints:
                                    orientation = sketch_constraints[loop_key]
                                    loop_data['is_outer'] = (orientation == 'outer')
        
        return json_data
    
    def _apply_preference_overrides(self, json_data: Dict[str, Any], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Apply preference overrides"""
        
        # Apply validation preferences
        if preferences.get('validate_geometry', True):
            # Validation will be handled in main conversion flow
            pass
        
        # Apply discontinuity fixing
        if preferences.get('fix_discontinuities', True):
            # Fix curve discontinuities
            json_data = self._fix_curve_discontinuities(json_data)
        
        return json_data
    
    def _apply_hint_overrides(self, json_data: Dict[str, Any], hints: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hint overrides"""
        
        # Apply bounding box overrides
        bbox_hints = hints.get('bounding_box', {})
        if bbox_hints.get('override_calculated', False):
            min_point = bbox_hints.get('min_point', {})
            max_point = bbox_hints.get('max_point', {})
            
            if min_point and max_point:
                json_data['properties']['bounding_box'] = {
                    "min_point": min_point,
                    "max_point": max_point,
                    "type": "BoundingBox3D"
                }
        
        return json_data
    
    def _fix_curve_discontinuities(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fix curve discontinuities in profiles"""
        
        entities = json_data.get('entities', {})
        for entity_data in entities.values():
            if entity_data.get('type') == 'Sketch':
                profiles = entity_data.get('profiles', {})
                for profile_data in profiles.values():
                    loops = profile_data.get('loops', [])
                    for loop_data in loops:
                        curves = loop_data.get('profile_curves', [])
                        if len(curves) > 1:
                            # Check and fix discontinuities
                            for i in range(len(curves)):
                                current_curve = curves[i]
                                next_curve = curves[(i + 1) % len(curves)]
                                
                                current_end = current_curve.get('end_point', {})
                                next_start = next_curve.get('start_point', {})
                                
                                # Check if points are close enough to be considered connected
                                if not self._points_close_enough(current_end, next_start):
                                    # Snap next curve start to current curve end
                                    next_curve['start_point'] = current_end.copy()
        
        return json_data
    
    def _points_close_enough(self, point1: Dict[str, float], point2: Dict[str, float], tolerance: float = 1e-6) -> bool:
        """Check if two points are close enough to be considered connected"""
        dx = point1.get('x', 0) - point2.get('x', 0)
        dy = point1.get('y', 0) - point2.get('y', 0)
        dz = point1.get('z', 0) - point2.get('z', 0)
        distance = math.sqrt(dx*dx + dy*dy + dz*dz)
        return distance < tolerance


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Convert Python CAD files to JSON format")
    parser.add_argument("input_file", help="Input Python file")
    parser.add_argument("output_file", help="Output JSON file")
    parser.add_argument("--add-metadata", action="store_true", 
                       help="Add metadata template to Python file")
    parser.add_argument("--validate", action="store_true",
                       help="Validate the generated JSON")
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} does not exist")
        sys.exit(1)
    
    # Convert file
    converter = PythonToJSONConverter()
    try:
        converter.convert_file(args.input_file, args.output_file, args.add_metadata)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)
    
    # Validate if requested
    if args.validate:
        print("Validating reconstruction...")
        try:
            from validation_framework import validate_reconstruction_quality
            
            # Load the generated JSON
            with open(args.output_file, 'r') as f:
                json_data = json.load(f)
            
            # Load the original Python code
            with open(args.input_file, 'r') as f:
                python_code = f.read()
            
            # Validate
            report = validate_reconstruction_quality(json_data, python_code)
            
            # Print validation report
            print("\n📋 Validation Report:")
            print(f"Overall Status: {'✅ PASSED' if report['overall_passed'] else '❌ FAILED'}")
            print(f"Levels Passed: {report['levels_passed']}/{report['total_levels']}")
            print(f"Confidence Score: {report['confidence_score']:.2f}")
            
            for level_name, result in report['detailed_results'].items():
                status = "✅" if result['passed'] else "❌"
                print(f"{status} {level_name}: {result['message']}")
                if result.get('details'):
                    print(f"   Details: {result['details']}")
            
        except ImportError:
            print("Warning: Validation framework not available")
        except Exception as e:
            print(f"Validation error: {e}")


if __name__ == "__main__":
    main()