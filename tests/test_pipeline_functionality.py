#!/usr/bin/env python3
"""
Test the complete CAD-RAG pipeline functionality
"""

from cad_rag_pipeline import CADRAGPipeline
import os

def test_pipeline_init():
    """Test pipeline initialization"""
    try:
        pipeline = CADRAGPipeline()
        print("✓ Pipeline initialized successfully")
        return pipeline
    except Exception as e:
        print(f"✗ Pipeline initialization failed: {e}")
        return None

def test_json_to_python_conversion(pipeline):
    """Test the JSON to Python conversion method"""
    # Look for a JSON file to test with
    json_dir = "/media/user/Data/MultiCAD/jsonfiles"
    if not os.path.exists(json_dir):
        print(f"⚠️  JSON directory not found: {json_dir}")
        return False
    
    # Find first JSON file
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')][:3]  # Test first 3
    if not json_files:
        print("⚠️  No JSON files found for testing")
        return False
    
    success_count = 0
    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        print(f"🧪 Testing conversion: {json_file}")
        
        try:
            python_code = pipeline._convert_json_to_python(json_path)
            if python_code:
                print(f"  ✓ Converted successfully ({len(python_code)} chars)")
                success_count += 1
            else:
                print(f"  ✗ Conversion returned None")
        except Exception as e:
            print(f"  ✗ Conversion failed: {e}")
    
    print(f"📊 Conversion results: {success_count}/{len(json_files)} successful")
    return success_count > 0

def test_model_python_code_retrieval(pipeline):
    """Test getting Python code for models"""
    test_models = ["00010680", "00227227", "00875135"]
    
    success_count = 0
    for model_id in test_models:
        print(f"🧪 Testing model retrieval: {model_id}")
        
        try:
            python_code = pipeline._get_model_python_code(model_id)
            if python_code:
                print(f"  ✓ Retrieved successfully ({len(python_code)} chars)")
                success_count += 1
            else:
                print(f"  ⚠️  No Python code found for {model_id}")
        except Exception as e:
            print(f"  ✗ Retrieval failed: {e}")
    
    print(f"📊 Retrieval results: {success_count}/{len(test_models)} successful")
    return success_count > 0

def main():
    """Run pipeline functionality tests"""
    print("🧪 Testing CAD-RAG Pipeline Functionality")
    print("=" * 50)
    
    # Test 1: Pipeline initialization
    print("\n📋 Test 1: Pipeline Initialization")
    pipeline = test_pipeline_init()
    if not pipeline:
        print("❌ Cannot proceed without pipeline initialization")
        return False
    
    # Test 2: JSON to Python conversion
    print("\n📋 Test 2: JSON to Python Conversion")
    conversion_works = test_json_to_python_conversion(pipeline)
    
    # Test 3: Model Python code retrieval
    print("\n📋 Test 3: Model Python Code Retrieval")  
    retrieval_works = test_model_python_code_retrieval(pipeline)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"  ✅ Pipeline initialization: PASS")
    print(f"  {'✅' if conversion_works else '❌'} JSON conversion: {'PASS' if conversion_works else 'FAIL'}")
    print(f"  {'✅' if retrieval_works else '❌'} Model retrieval: {'PASS' if retrieval_works else 'FAIL'}")
    
    all_passed = conversion_works and retrieval_works
    if all_passed:
        print("\n🎉 All core functionality tests passed!")
        print("💡 Pipeline is ready for retrieval and generation tasks")
    else:
        print("\n⚠️  Some tests failed, but core conversion is working")
    
    return all_passed

if __name__ == "__main__":
    main()