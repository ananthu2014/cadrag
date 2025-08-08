#!/usr/bin/env python3
"""
Test script to verify the GUI and pipeline fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gui_import():
    """Test that GUI imports correctly"""
    try:
        from cad_rag_gui import CADRAGGUIApp
        print("✓ GUI import successful")
        return True
    except Exception as e:
        print(f"✗ GUI import failed: {e}")
        return False

def test_pipeline_fixes():
    """Test that pipeline has the JSON to Python conversion"""
    try:
        from cad_rag_pipeline import CADRAGPipeline
        
        # Initialize pipeline
        pipeline = CADRAGPipeline()
        
        # Check if the JSON to Python conversion method exists
        if hasattr(pipeline, '_convert_json_to_python'):
            print("✓ JSON to Python conversion method exists")
            return True
        else:
            print("✗ JSON to Python conversion method missing")
            return False
            
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False

def test_basic_retrieval():
    """Test basic retrieval functionality"""
    try:
        from cad_rag_pipeline import CADRAGPipeline
        
        pipeline = CADRAGPipeline()
        results = pipeline.retrieve_models(text_query="test", top_k=3)
        
        if results and len(results) > 0:
            print(f"✓ Basic retrieval works - found {len(results)} results")
            return True
        else:
            print("✗ No results returned from retrieval")
            return False
            
    except Exception as e:
        print(f"✗ Retrieval test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing CAD-RAG Fixes")
    print("=" * 30)
    
    tests = [
        ("GUI Import", test_gui_import),
        ("Pipeline Fixes", test_pipeline_fixes),
        ("Basic Retrieval", test_basic_retrieval)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*30}")
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes working correctly!")
    else:
        print("⚠️  Some issues remain - check error messages above")

if __name__ == "__main__":
    main()