#!/usr/bin/env python3
"""
Test script to verify conda environment access in subprocess
"""

import subprocess
import os

def test_conda_env_access():
    """Test if we can access gemini2 environment from subprocess"""
    
    print("Testing conda environment access...")
    
    # Test 1: Check if conda works in subprocess
    try:
        result = subprocess.run(
            ["conda", "info", "--base"], 
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"✓ Conda base path: {result.stdout.strip()}")
        else:
            print(f"✗ Failed to get conda base: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error running conda: {e}")
        return False
    
    # Test 2: Check if gemini2 environment exists
    try:
        result = subprocess.run(
            ["conda", "env", "list"], 
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            if "gemini2" in result.stdout:
                print("✓ gemini2 environment found")
            else:
                print("✗ gemini2 environment not found")
                print("Available environments:")
                print(result.stdout)
                return False
        else:
            print(f"✗ Failed to list environments: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error listing environments: {e}")
        return False
    
    # Test 3: Try to activate gemini2 environment
    try:
        conda_base = subprocess.run(
            ["conda", "info", "--base"], 
            capture_output=True, text=True, timeout=10
        ).stdout.strip()
        
        conda_script = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
        
        cmd = [
            "bash", "-c",
            f"source {conda_script} && conda activate gemini2 && python -c 'print(\"✓ Successfully activated gemini2\")'"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ gemini2 environment activation successful")
            print(f"Output: {result.stdout.strip()}")
        else:
            print(f"✗ Failed to activate gemini2: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error testing environment activation: {e}")
        return False
    
    # Test 4: Check if google-generativeai is installed in gemini2
    try:
        cmd = [
            "bash", "-c",
            f"source {conda_script} && conda activate gemini2 && python -c 'import google.generativeai; print(\"✓ google-generativeai available\")'"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✓ google-generativeai is available in gemini2")
        else:
            print(f"✗ google-generativeai not available: {result.stderr}")
            print("Please install it with: conda activate gemini2 && pip install google-generativeai")
            return False
            
    except Exception as e:
        print(f"✗ Error checking google-generativeai: {e}")
        return False
    
    print("\n🎉 All tests passed! gemini2 environment is properly configured.")
    return True

if __name__ == "__main__":
    test_conda_env_access()