#!/usr/bin/env python3
"""
Test script for CAD-RAG integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from cad_rag_pipeline import CADRAGPipeline
        print("✓ CADRAGPipeline imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CADRAGPipeline: {e}")
        return False
    
    try:
        from cad_rag_gui import CADRAGGUIApp
        print("✓ CADRAGGUIApp imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CADRAGGUIApp: {e}")
        return False
    
    return True

def test_pipeline_initialization():
    """Test pipeline initialization"""
    print("\nTesting pipeline initialization...")
    
    try:
        from cad_rag_pipeline import CADRAGPipeline
        
        # Check if database directory exists
        db_dir = "database_embeddings"
        if not os.path.exists(db_dir):
            print(f"✗ Database directory not found: {db_dir}")
            return False
        
        print(f"✓ Database directory found: {db_dir}")
        
        # Try to initialize pipeline
        pipeline = CADRAGPipeline(database_dir=db_dir)
        print("✓ Pipeline initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    
    try:
        from cad_rag_pipeline import CADRAGPipeline
        
        pipeline = CADRAGPipeline()
        
        # Test retrieval with a simple text query
        results = pipeline.retrieve_models(text_query="cylindrical part", top_k=3)
        print(f"✓ Retrieval test passed - found {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("CAD-RAG Integration Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Pipeline Initialization", test_pipeline_initialization),
        ("Basic Functionality", test_basic_functionality)
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
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The CAD-RAG system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()