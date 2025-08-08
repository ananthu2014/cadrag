#!/usr/bin/env python3
"""
Test JSON to Python conversion functionality
"""

import json
import os
from bethany_lib.extrude import CADSequence
from bethany_lib.cad2code import get_cad_code

def test_json_conversion():
    """Test JSON to Python conversion with a sample file"""
    
    # Look for a JSON file to test with
    json_dir = "/media/user/Data/MultiCAD/jsonfiles"
    if not os.path.exists(json_dir):
        print(f"⚠️  JSON directory not found: {json_dir}")
        return False
    
    # Find first JSON file
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        print("⚠️  No JSON files found for testing")
        return False
    
    test_file = os.path.join(json_dir, json_files[0])
    print(f"🧪 Testing conversion with: {json_files[0]}")
    
    try:
        # Load JSON data
        with open(test_file, 'r') as f:
            json_data = json.load(f)
        print("✓ JSON loaded successfully")
        
        # Convert JSON to CADSequence
        cad_seq = CADSequence.from_dict(json_data)
        print("✓ CADSequence created successfully")
        
        # Generate Python code
        python_code = get_cad_code(cad_seq)
        print("✓ Python code generated successfully")
        print(f"  Generated code length: {len(python_code)} characters")
        
        # Show a snippet
        lines = python_code.split('\n')
        print(f"  First few lines:")
        for i, line in enumerate(lines[:5]):
            print(f"    {i+1}: {line}")
        
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing JSON to Python Conversion")
    print("=" * 40)
    
    success = test_json_conversion()
    
    if success:
        print("\n🎉 JSON conversion test passed!")
    else:
        print("\n❌ JSON conversion test failed!")