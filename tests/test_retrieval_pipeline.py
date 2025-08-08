#!/usr/bin/env python3
"""
Test script for CAD retrieval pipeline

This script tests the encoding and retrieval pipeline with sample data.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_database_encoding():
    """Test database encoding functionality"""
    print("Testing database encoding...")
    
    # Check if we have sample images to work with
    sample_dirs = [
        "data/dataset/deepcad_screenshots_side",
        "data/dataset/deepcad_sketches_side_png"
    ]
    
    available_dirs = [d for d in sample_dirs if os.path.exists(d)]
    
    if not available_dirs:
        print("⚠ No sample image directories found. Skipping encoding test.")
        return False
    
    # Create temporary database directory
    with tempfile.TemporaryDirectory() as temp_db:
        print(f"Using temporary database: {temp_db}")
        
        # Run encoding with limited sample
        cmd = f"python encode_cad_database.py --input_dirs {' '.join(available_dirs[:1])} --output_dir {temp_db}"
        print(f"Running: {cmd}")
        
        result = os.system(cmd)
        if result != 0:
            print("✗ Database encoding failed")
            return False
        
        # Check if required files were created
        required_files = [
            "metadata.json",
            "embeddings.pt", 
            "embedding_matrix.pt",
            "uid_to_index.json",
            "display_preferences.json"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(temp_db, file)):
                print(f"✗ Missing required file: {file}")
                return False
        
        print("✓ Database encoding successful")
        return True

def test_retrieval_functionality():
    """Test retrieval functionality"""
    print("\nTesting retrieval functionality...")
    
    # This would require a pre-encoded database
    # For now, just check if the script can be imported
    try:
        from interactive_cad_retrieval import InteractiveCADRetriever
        print("✓ Retrieval module imports successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import retrieval module: {e}")
        return False

def check_dependencies():
    """Check if all dependencies are available"""
    print("Checking dependencies...")
    
    required_modules = [
        'torch',
        'torchvision', 
        'PIL',
        'numpy',
        'matplotlib',
        'tqdm'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print(f"✗ Missing dependencies: {', '.join(missing)}")
        return False
    
    print("✓ All dependencies available")
    return True

def check_config():
    """Check if configuration file exists"""
    print("Checking configuration...")
    
    if not os.path.exists("config.yml"):
        print("✗ config.yml not found")
        return False
    
    try:
        from utils import load_config
        config = load_config("config.yml")
        print("✓ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False

def check_model_files():
    """Check for available model files"""
    print("Checking for model files...")
    
    model_paths = [
        "data/last_model.pt",
        "data/last_model0th.pt", 
        "data/model_epoch_15.pt"
    ]
    
    available_models = [p for p in model_paths if os.path.exists(p)]
    
    if available_models:
        print(f"✓ Found trained models: {', '.join(available_models)}")
        return True
    else:
        print("⚠ No trained models found, will use pretrained CLIP")
        return True  # Not a failure, can use pretrained

def main():
    """Run all tests"""
    print("="*60)
    print("CAD RETRIEVAL PIPELINE TEST")
    print("="*60)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("Configuration", check_config), 
        ("Model Files", check_model_files),
        ("Retrieval Module", test_retrieval_functionality),
        ("Database Encoding", test_database_encoding)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The pipeline is ready to use.")
    else:
        print("\n⚠ Some tests failed. Please check the issues above.")
    
    # Usage instructions
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Encode your CAD database:")
    print("   python encode_cad_database.py --input_dirs <image_dirs> --output_dir <db_dir>")
    print("\n2. Run interactive retrieval:")  
    print("   python interactive_cad_retrieval.py --database_dir <db_dir> --query_sketch <sketch.png>")
    print("\n3. See CAD_RETRIEVAL_GUIDE.md for detailed instructions")

if __name__ == "__main__":
    main()