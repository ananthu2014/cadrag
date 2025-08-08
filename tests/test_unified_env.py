#!/usr/bin/env python3
"""
Test script for unified cad-rag environment functionality
"""

def test_pytorch_imports():
    """Test PyTorch and related imports"""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        
        import torchvision
        print(f"✓ Torchvision {torchvision.__version__} imported successfully")
        
        # Test CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available with {torch.cuda.device_count()} devices")
        else:
            print("⚠️  CUDA not available")
            
        return True
    except Exception as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

def test_gemini_imports():
    """Test Gemini API imports"""
    try:
        import google.generativeai as genai
        print("✓ Google Generative AI imported successfully")
        
        from PIL import Image
        print("✓ Pillow imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Gemini imports failed: {e}")
        return False

def test_bethany_imports():
    """Test Bethany library imports"""
    try:
        from bethany_lib.extrude import CADSequence
        print("✓ Bethany CADSequence imported successfully")
        
        from bethany_lib.cad2code import get_cad_code
        print("✓ Bethany cad2code imported successfully")
        
        import numpy as np
        print(f"✓ NumPy {np.__version__} imported successfully")
        
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__} imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Bethany imports failed: {e}")
        return False

def test_cstbir_imports():
    """Test CSTBIR specific imports"""
    try:
        from interactive_cad_retrieval import InteractiveCADRetriever
        print("✓ CSTBIR InteractiveCADRetriever imported successfully")
        
        from py2json_converter import PythonToJSONConverter
        print("✓ CSTBIR PythonToJSONConverter imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ CSTBIR imports failed: {e}")
        return False

def test_pipeline_import():
    """Test the main pipeline import"""
    try:
        from cad_rag_pipeline import CADRAGPipeline
        print("✓ CADRAGPipeline imported successfully")
        
        # Try to initialize (without actually loading models)
        pipeline = CADRAGPipeline()
        print("✓ CADRAGPipeline initialized successfully")
        
        return True
    except Exception as e:
        print(f"✗ Pipeline import/init failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Unified CAD-RAG Environment")
    print("=" * 50)
    
    tests = [
        ("PyTorch Dependencies", test_pytorch_imports),
        ("Gemini API Dependencies", test_gemini_imports),
        ("Bethany Libraries", test_bethany_imports),
        ("CSTBIR Components", test_cstbir_imports),
        ("Main Pipeline", test_pipeline_import),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Unified environment is ready.")
    else:
        print("\n⚠️  Some tests failed. Check dependencies.")
    
    return all_passed

if __name__ == "__main__":
    main()