#!/usr/bin/env python3
"""
Test only the Bethany conversion functionality (no PyTorch required)
"""

import json
import os
import sys

def test_bethany_conversion():
    """Test JSON to Python conversion using Bethany libraries only"""
    
    print("🧪 Testing Bethany JSON-to-Python Conversion")
    print("=" * 50)
    
    try:
        # Import Bethany libraries
        from bethany_lib.extrude import CADSequence
        from bethany_lib.cad2code import get_cad_code
        print("✓ Bethany libraries imported successfully")
        
        # Look for JSON files
        json_dir = "/media/user/Data/MultiCAD/jsonfiles"
        if not os.path.exists(json_dir):
            print(f"⚠️  JSON directory not found: {json_dir}")
            return False
        
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')][:3]
        if not json_files:
            print("⚠️  No JSON files found")
            return False
        
        print(f"📁 Found {len(json_files)} JSON files to test")
        
        # Test conversions
        success_count = 0
        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            model_id = json_file.replace('.json', '')
            
            print(f"\n🔧 Testing: {model_id}")
            
            try:
                # Load JSON
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                print(f"  ✓ JSON loaded")
                
                # Convert to CADSequence
                cad_seq = CADSequence.from_dict(json_data)
                print(f"  ✓ CADSequence created")
                
                # Generate Python code
                python_code = get_cad_code(cad_seq)
                print(f"  ✓ Python code generated ({len(python_code)} chars)")
                
                # Show snippet
                lines = python_code.split('\n')[:3]
                print(f"  📝 Preview: {lines[0][:60]}...")
                
                success_count += 1
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n📊 Results: {success_count}/{len(json_files)} conversions successful")
        
        if success_count == len(json_files):
            print("🎉 All conversions passed! Bethany integration is working correctly.")
            return True
        elif success_count > 0:
            print("⚠️  Some conversions passed. Partial success.")
            return True
        else:
            print("❌ All conversions failed.")
            return False
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print("Make sure you're in the cad-rag environment")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_bethany_conversion()