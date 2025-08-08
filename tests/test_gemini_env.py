#!/usr/bin/env python3
"""
Test script to verify Gemini environment switching works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gemini_env_switching():
    """Test that Gemini API calls work with environment switching"""
    print("Testing Gemini environment switching...")
    
    try:
        from cad_rag_pipeline import CADRAGPipeline
        
        # Initialize pipeline
        pipeline = CADRAGPipeline()
        print("✓ Pipeline initialized successfully")
        
        # Test simple prompt
        test_prompt = """You are a CAD design assistant. Generate a simple Python script that creates a cylinder.

Use the following format:
```python
# Simple cylinder creation
def create_cylinder():
    # Create cylinder with radius 5 and height 10
    cylinder = add_cylinder(radius=5, height=10)
    return cylinder
```

Generate a similar but slightly modified version:"""
        
        print("Testing Gemini API call with environment switching...")
        
        # This should use the gemini2 environment
        response = pipeline._call_gemini_api(test_prompt)
        
        print(f"✓ Gemini API call successful!")
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Gemini environment test failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline with environment switching"""
    print("\nTesting full pipeline with environment switching...")
    
    try:
        from cad_rag_pipeline import CADRAGPipeline
        
        pipeline = CADRAGPipeline()
        
        # Test retrieval (should work in current environment)
        print("Testing retrieval...")
        results = pipeline.retrieve_models(text_query="cylindrical part", top_k=3)
        print(f"✓ Retrieval successful - found {len(results)} models")
        
        if results:
            # Test generation (should switch to gemini2 environment)
            print("Testing generation with environment switching...")
            generated_code = pipeline.generate_cad_sequence(
                user_query="cylindrical part",
                selected_model=results[0],
                instructions="Make it taller"
            )
            print(f"✓ Generation successful - code length: {len(generated_code)}")
            
            # Test conversion (should work in current environment)
            print("Testing conversion...")
            json_output = pipeline.convert_to_json(generated_code)
            print(f"✓ Conversion successful - JSON keys: {list(json_output.keys())}")
            
        return True
        
    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        return False

def main():
    """Main test function"""
    print("CAD-RAG Environment Switching Test")
    print("=" * 50)
    
    tests = [
        ("Gemini Environment Switching", test_gemini_env_switching),
        ("Full Pipeline with Environment Switching", test_full_pipeline)
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
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Environment switching works correctly.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()