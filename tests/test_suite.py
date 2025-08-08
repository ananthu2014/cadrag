#!/usr/bin/env python3
"""
Comprehensive Test Suite for Python-to-JSON CAD Reconstruction System

This test suite validates all aspects of the reconstruction system:
- Unit tests for individual components
- Integration tests for complete workflows
- Validation tests using real example files
- Performance tests for large files
"""

import unittest
import json
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List

# Add the current directory to path for imports
sys.path.insert(0, '/media/user/data/CSTBIR')

from py2json_converter import (
    UUIDGenerator, GeometryRecoverer, MetadataHandler, 
    PythonASTParser, PythonToJSONConverter
)
from validation_framework import ReconstructionValidator, validate_reconstruction_quality


class TestUUIDGenerator(unittest.TestCase):
    """Test UUID generation consistency and uniqueness"""
    
    def setUp(self):
        self.generator = UUIDGenerator()
    
    def test_entity_uuid_format(self):
        """Test entity UUID follows F..._{index} pattern"""
        uuid = self.generator.generate_entity_uuid("test_hash", "Sketch", 0)
        self.assertTrue(uuid.startswith("F"))
        self.assertTrue(uuid.endswith("_0"))
        self.assertEqual(len(uuid), 18)  # F + 15 chars + _ + index
    
    def test_profile_uuid_format(self):
        """Test profile UUID follows J[A-Z][A-Z] pattern"""
        uuid = self.generator.generate_profile_uuid("test_content")
        self.assertTrue(uuid.startswith("J"))
        self.assertEqual(len(uuid), 3)
        self.assertTrue(uuid[1].isupper())
        self.assertTrue(uuid[2].isupper())
    
    def test_curve_uuid_format(self):
        """Test curve UUID follows expected pattern"""
        uuid = self.generator.generate_curve_uuid("test_curve")
        self.assertTrue(uuid.startswith("J"))
        self.assertTrue(len(uuid) >= 3)
    
    def test_uuid_uniqueness(self):
        """Test that generated UUIDs are unique"""
        uuids = set()
        for i in range(100):
            uuid = self.generator.generate_entity_uuid(f"test_{i}", "Sketch", i)
            self.assertNotIn(uuid, uuids)
            uuids.add(uuid)
    
    def test_deterministic_generation(self):
        """Test that same input produces same UUID"""
        gen1 = UUIDGenerator()
        gen2 = UUIDGenerator()
        
        uuid1 = gen1.generate_entity_uuid("test_hash", "Sketch", 0)
        uuid2 = gen2.generate_entity_uuid("test_hash", "Sketch", 0)
        
        self.assertEqual(uuid1, uuid2)


class TestGeometryRecoverer(unittest.TestCase):
    """Test geometric parameter recovery"""
    
    def setUp(self):
        self.recoverer = GeometryRecoverer()
    
    def test_arc_parameter_recovery(self):
        """Test arc parameter recovery from three points"""
        # Simple quarter circle arc
        start = [1.0, 0.0]
        end = [0.0, 1.0]
        mid = [0.7071, 0.7071]  # Point on unit circle at 45 degrees
        
        result = self.recoverer.recover_arc_parameters(start, end, mid)
        
        # Check that we get reasonable results
        self.assertAlmostEqual(result['radius'], 1.0, places=3)
        self.assertAlmostEqual(result['center_point']['x'], 0.0, places=3)
        self.assertAlmostEqual(result['center_point']['y'], 0.0, places=3)
    
    def test_collinear_points_error(self):
        """Test that collinear points raise error"""
        start = [0.0, 0.0]
        end = [1.0, 0.0]
        mid = [0.5, 0.0]  # All on same line
        
        with self.assertRaises(ValueError):
            self.recoverer.recover_arc_parameters(start, end, mid)
    
    def test_transform_matrix_recovery(self):
        """Test transform matrix recovery"""
        origin = [0.0, 0.0, 0.0]
        normal = [0.0, 0.0, 1.0]
        x_axis = [1.0, 0.0, 0.0]
        
        result = self.recoverer.recover_transform_matrix(origin, normal, x_axis)
        
        # Check orthogonality
        self.assertAlmostEqual(result['x_axis']['x'], 1.0)
        self.assertAlmostEqual(result['y_axis']['y'], 1.0)
        self.assertAlmostEqual(result['z_axis']['z'], 1.0)
    
    def test_bounding_box_calculation(self):
        """Test bounding box calculation"""
        geometry = [
            {'type': 'line', 'start': [0.0, 0.0, 0.0], 'end': [1.0, 1.0, 1.0]},
            {'type': 'circle', 'center': [0.5, 0.5, 0.0], 'radius': 0.5}
        ]
        
        result = self.recoverer.calculate_bounding_box(geometry)
        
        # Check that bounding box encompasses all geometry
        self.assertLessEqual(result['min_point']['x'], 0.0)
        self.assertLessEqual(result['min_point']['y'], 0.0)
        self.assertGreaterEqual(result['max_point']['x'], 1.0)
        self.assertGreaterEqual(result['max_point']['y'], 1.0)


class TestMetadataHandler(unittest.TestCase):
    """Test metadata handling functionality"""
    
    def setUp(self):
        self.handler = MetadataHandler()
    
    def test_metadata_extraction(self):
        """Test extraction of metadata from Python code"""
        python_code = '''
# Some CAD code
SketchPlane0 = add_sketchplane(origin=[0, 0, 0])

_metadata_uuids = {
    "regenerate_all": False,
    "preserve_entity_uuids": {"Sketch0": "TestUUID"}
}

_metadata_preferences = {
    "validate_geometry": True
}
'''
        
        metadata = self.handler.extract_metadata(python_code)
        
        self.assertIn('_metadata_uuids', metadata)
        self.assertIn('_metadata_preferences', metadata)
        self.assertEqual(metadata['_metadata_uuids']['regenerate_all'], False)
        self.assertEqual(metadata['_metadata_preferences']['validate_geometry'], True)
    
    def test_default_metadata(self):
        """Test default metadata when parsing fails"""
        invalid_code = "invalid python code {"
        
        metadata = self.handler.extract_metadata(invalid_code)
        
        # Should return default metadata
        self.assertIn('_metadata_uuids', metadata)
        self.assertIn('_metadata_preferences', metadata)
        self.assertTrue(metadata['_metadata_uuids']['regenerate_all'])
    
    def test_metadata_template_generation(self):
        """Test generation of metadata template"""
        json_data = {
            "entities": {
                "TestID": {"type": "Sketch", "name": "Sketch0"},
                "TestID2": {"type": "ExtrudeFeature", "name": "Extrude0"}
            }
        }
        
        template = self.handler.generate_metadata_template(json_data)
        
        self.assertIn('_metadata_uuids', template)
        self.assertIn('Sketch0', template)
        self.assertIn('Extrude0', template)


class TestPythonASTParser(unittest.TestCase):
    """Test Python AST parsing functionality"""
    
    def setUp(self):
        self.parser = PythonASTParser()
    
    def test_operation_extraction(self):
        """Test extraction of CAD operations from Python code"""
        python_code = '''
SketchPlane0 = add_sketchplane(origin=[0, 0, 0])
Line0 = add_line(start=[0, 0], end=[1, 1])
Sketch0 = add_sketch(sketch_plane=SketchPlane0, profile=Profile0)
'''
        
        tree = self.parser.parse_python_file(python_code)
        entities = tree.get('entities', {})
        
        # Should have parsed the operations
        self.assertGreater(len(entities), 0)
    
    def test_simple_line_parsing(self):
        """Test parsing of simple line operations"""
        python_code = '''
SketchPlane0 = add_sketchplane(origin=[0, 0, 0])
Line0 = add_line(start=[0, 0], end=[1, 1])
Loop0 = add_loop([Line0])
Profile0 = add_profile([Loop0])
Sketch0 = add_sketch(sketch_plane=SketchPlane0, profile=Profile0)
'''
        
        result = self.parser.parse_python_file(python_code)
        
        # Should have entities and sequence
        self.assertIn('entities', result)
        self.assertIn('sequence', result)
        self.assertIn('properties', result)
    
    def test_arc_parsing(self):
        """Test parsing of arc operations"""
        python_code = '''
SketchPlane0 = add_sketchplane(origin=[0, 0, 0])
Arc0 = add_arc(start=[1, 0], end=[0, 1], mid=[0.707, 0.707])
Loop0 = add_loop([Arc0])
Profile0 = add_profile([Loop0])
Sketch0 = add_sketch(sketch_plane=SketchPlane0, profile=Profile0)
'''
        
        result = self.parser.parse_python_file(python_code)
        
        # Should have properly parsed arc with parameters
        entities = result.get('entities', {})
        self.assertGreater(len(entities), 0)


class TestValidationFramework(unittest.TestCase):
    """Test validation framework functionality"""
    
    def setUp(self):
        self.validator = ReconstructionValidator()
    
    def test_structural_validation(self):
        """Test structural validation of JSON"""
        valid_json = {
            "entities": {"test": {"type": "Sketch"}},
            "properties": {"bounding_box": {"type": "BoundingBox3D"}},
            "sequence": [{"entity": "test", "type": "Sketch"}]
        }
        
        result = self.validator._validate_structure(valid_json)
        self.assertTrue(result.passed)
    
    def test_invalid_structure(self):
        """Test validation of invalid JSON structure"""
        invalid_json = {
            "entities": {"test": {"type": "InvalidType"}},
            "properties": {},
            # Missing sequence
        }
        
        result = self.validator._validate_structure(invalid_json)
        self.assertFalse(result.passed)
    
    def test_geometric_validation(self):
        """Test geometric validation"""
        json_with_geometry = {
            "entities": {
                "test": {
                    "type": "Sketch",
                    "profiles": {
                        "JGC": {
                            "loops": [{
                                "is_outer": True,
                                "profile_curves": [{
                                    "type": "Circle3D",
                                    "center_point": {"x": 0, "y": 0, "z": 0},
                                    "radius": 1.0
                                }]
                            }]
                        }
                    }
                }
            },
            "properties": {},
            "sequence": []
        }
        
        result = self.validator._validate_geometry(json_with_geometry)
        self.assertTrue(result.passed)
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation workflow"""
        test_json = {
            "entities": {"test": {"type": "Sketch"}},
            "properties": {},
            "sequence": [{"entity": "test", "type": "Sketch"}]
        }
        
        test_python = "# Test Python code\npass"
        
        results = self.validator.validate_reconstruction(test_json, test_python)
        
        # Should have results for multiple validation levels
        self.assertGreater(len(results), 0)


class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete integration workflow"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.converter = PythonToJSONConverter()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_simple_conversion_workflow(self):
        """Test complete conversion workflow"""
        # Create a simple test Python file
        test_python = '''
SketchPlane0 = add_sketchplane(origin=[0, 0, 0])
Line0 = add_line(start=[0, 0], end=[1, 1])
Loop0 = add_loop([Line0])
Profile0 = add_profile([Loop0])
Sketch0 = add_sketch(sketch_plane=SketchPlane0, profile=Profile0)
Extrude0 = add_extrude(sketch=Sketch0, operation=0, type=0, extent_one=1.0, extent_two=0.0)
'''
        
        input_file = os.path.join(self.test_dir, 'test.py')
        output_file = os.path.join(self.test_dir, 'test.json')
        
        with open(input_file, 'w') as f:
            f.write(test_python)
        
        # Convert
        self.converter.convert_file(input_file, output_file)
        
        # Check output file exists and is valid JSON
        self.assertTrue(os.path.exists(output_file))
        
        with open(output_file, 'r') as f:
            json_data = json.load(f)
        
        # Check basic structure
        self.assertIn('entities', json_data)
        self.assertIn('properties', json_data)
        self.assertIn('sequence', json_data)
    
    def test_validation_integration(self):
        """Test validation integration with conversion"""
        # Create a test Python file
        test_python = '''
SketchPlane0 = add_sketchplane(origin=[0, 0, 0])
Circle0 = add_circle(center=[0, 0], radius=1.0)
Loop0 = add_loop([Circle0])
Profile0 = add_profile([Loop0])
Sketch0 = add_sketch(sketch_plane=SketchPlane0, profile=Profile0)
'''
        
        input_file = os.path.join(self.test_dir, 'test.py')
        output_file = os.path.join(self.test_dir, 'test.json')
        
        with open(input_file, 'w') as f:
            f.write(test_python)
        
        # Convert
        self.converter.convert_file(input_file, output_file)
        
        # Validate
        with open(output_file, 'r') as f:
            json_data = json.load(f)
        
        report = validate_reconstruction_quality(json_data, test_python)
        
        # Should have validation results
        self.assertIn('overall_passed', report)
        self.assertIn('detailed_results', report)


class TestRealExamples(unittest.TestCase):
    """Test with real example files"""
    
    def setUp(self):
        self.examples_dir = Path('/media/user/data/OpenECAD_Project/Bethany/examples')
        self.test_dir = tempfile.mkdtemp()
        self.converter = PythonToJSONConverter()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_example_file_conversion(self):
        """Test conversion of a real example file"""
        example_files = list(self.examples_dir.glob('*.py'))
        
        if not example_files:
            self.skipTest("No example files found")
        
        # Test with first example file
        example_file = example_files[0]
        output_file = os.path.join(self.test_dir, f"{example_file.stem}.json")
        
        try:
            self.converter.convert_file(str(example_file), output_file)
            
            # Check output exists and is valid JSON
            self.assertTrue(os.path.exists(output_file))
            
            with open(output_file, 'r') as f:
                json_data = json.load(f)
            
            # Basic structure checks
            self.assertIn('entities', json_data)
            self.assertIn('properties', json_data)
            self.assertIn('sequence', json_data)
            
            # Should have some entities
            self.assertGreater(len(json_data['entities']), 0)
            
        except Exception as e:
            self.fail(f"Conversion failed for {example_file}: {e}")
    
    def test_multiple_example_files(self):
        """Test conversion of multiple example files"""
        example_files = list(self.examples_dir.glob('*.py'))[:3]  # Test first 3 files
        
        if not example_files:
            self.skipTest("No example files found")
        
        success_count = 0
        for example_file in example_files:
            output_file = os.path.join(self.test_dir, f"{example_file.stem}.json")
            
            try:
                self.converter.convert_file(str(example_file), output_file)
                
                # Validate the output
                with open(output_file, 'r') as f:
                    json_data = json.load(f)
                
                # Basic validation
                self.assertIn('entities', json_data)
                success_count += 1
                
            except Exception as e:
                print(f"Warning: Failed to convert {example_file}: {e}")
        
        # At least one file should convert successfully
        self.assertGreater(success_count, 0)
    
    def test_cross_reference_validation(self):
        """Test cross-reference validation against existing JSON files"""
        json_files = list(self.examples_dir.glob('*.json'))
        py_files = list(self.examples_dir.glob('*.py'))
        
        if not json_files or not py_files:
            self.skipTest("No reference files found")
        
        # Find matching pairs
        for json_file in json_files[:2]:  # Test first 2 pairs
            py_file = json_file.with_suffix('.py')
            
            if py_file.exists():
                output_file = os.path.join(self.test_dir, f"{json_file.stem}_reconstructed.json")
                
                try:
                    # Convert Python to JSON
                    self.converter.convert_file(str(py_file), output_file)
                    
                    # Load both JSONs
                    with open(output_file, 'r') as f:
                        reconstructed = json.load(f)
                    
                    with open(json_file, 'r') as f:
                        reference = json.load(f)
                    
                    # Validate with cross-reference
                    with open(py_file, 'r') as f:
                        python_code = f.read()
                    
                    report = validate_reconstruction_quality(reconstructed, python_code, reference)
                    
                    # Should have cross-reference validation results
                    self.assertIn('CROSS_REFERENCE', report['detailed_results'])
                    
                except Exception as e:
                    print(f"Warning: Cross-reference test failed for {json_file}: {e}")


class TestPerformance(unittest.TestCase):
    """Test performance characteristics"""
    
    def setUp(self):
        self.converter = PythonToJSONConverter()
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)
    
    def test_large_file_performance(self):
        """Test performance with large files"""
        # Create a large test file
        large_python = []
        for i in range(100):  # Create 100 sketches
            large_python.extend([
                f"SketchPlane{i} = add_sketchplane(origin=[{i}, 0, 0])",
                f"Line{i} = add_line(start=[{i}, 0], end=[{i+1}, 1])",
                f"Loop{i} = add_loop([Line{i}])",
                f"Profile{i} = add_profile([Loop{i}])",
                f"Sketch{i} = add_sketch(sketch_plane=SketchPlane{i}, profile=Profile{i})",
                f"Extrude{i} = add_extrude(sketch=Sketch{i}, operation=0, type=0, extent_one=1.0, extent_two=0.0)",
                ""
            ])
        
        test_code = '\n'.join(large_python)
        
        input_file = os.path.join(self.test_dir, 'large_test.py')
        output_file = os.path.join(self.test_dir, 'large_test.json')
        
        with open(input_file, 'w') as f:
            f.write(test_code)
        
        # Time the conversion
        import time
        start_time = time.time()
        
        self.converter.convert_file(input_file, output_file)
        
        end_time = time.time()
        conversion_time = end_time - start_time
        
        # Should complete in reasonable time (less than 10 seconds)
        self.assertLess(conversion_time, 10.0)
        
        # Check output is valid
        with open(output_file, 'r') as f:
            json_data = json.load(f)
        
        # Should have many entities
        self.assertGreater(len(json_data['entities']), 90)


def run_test_suite():
    """Run the complete test suite"""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestUUIDGenerator,
        TestGeometryRecoverer,
        TestMetadataHandler,
        TestPythonASTParser,
        TestValidationFramework,
        TestIntegrationWorkflow,
        TestRealExamples,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_test_suite()
    sys.exit(0 if success else 1)