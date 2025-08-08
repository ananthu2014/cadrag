#!/usr/bin/env python3
"""
Test script to verify direct Gemini API call using environment path
"""

import subprocess
import tempfile
import os
import json
from cad_rag_pipeline import CADRAGPipeline

def test_gemini_direct_call():
    """Test calling Gemini API directly using environment path"""
    
    print("Testing direct Gemini API call...")
    
    # Initialize pipeline to get environment path
    pipeline = CADRAGPipeline()
    env_path = pipeline._get_gemini_env_path()
    
    if not env_path:
        print("✗ Could not find gemini2 environment path")
        return False
    
    python_path = os.path.join(env_path, "bin", "python")
    
    if not os.path.exists(python_path):
        print(f"✗ Python not found at {python_path}")
        return False
    
    print(f"✓ Using Python at: {python_path}")
    
    # Create a simple test script
    test_script = '''
import google.generativeai as genai
import json
import sys

# Test API key (first one from the list)
api_key = "AIzaSyBc1Rf5FqjYJJ6LyKh2r_HUNtZXIdOyJDw"

try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    # Simple test prompt
    prompt = "Generate a simple Python function that adds two numbers. Just return the code."
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            top_p=1.0,
            max_output_tokens=500,
        )
    )
    
    if response.text:
        print("✓ Gemini API call successful")
        print(f"Response: {response.text[:100]}...")
        
        # Write success result
        with open("test_output.json", "w") as f:
            json.dump({"success": True, "response": response.text}, f)
    else:
        print("✗ Empty response from Gemini")
        with open("test_output.json", "w") as f:
            json.dump({"success": False, "error": "Empty response"}, f)
            
except Exception as e:
    print(f"✗ Error: {e}")
    with open("test_output.json", "w") as f:
        json.dump({"success": False, "error": str(e)}, f)
'''
    
    # Write test script to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_script_path = f.name
    
    try:
        # Change to temp directory for output file
        original_cwd = os.getcwd()
        temp_dir = os.path.dirname(temp_script_path)
        os.chdir(temp_dir)
        
        # Run the test script
        result = subprocess.run(
            [python_path, temp_script_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        
        # Check if output file was created
        output_path = os.path.join(temp_dir, "test_output.json")
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                output_data = json.load(f)
            
            if output_data.get('success'):
                print("🎉 Direct Gemini API call test successful!")
                return True
            else:
                print(f"✗ API call failed: {output_data.get('error')}")
                return False
        else:
            print("✗ No output file created")
            return False
    
    finally:
        # Cleanup
        os.chdir(original_cwd)
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)
        output_path = os.path.join(temp_dir, "test_output.json")
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    test_gemini_direct_call()