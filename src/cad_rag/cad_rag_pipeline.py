#!/usr/bin/env python3
"""
CAD-RAG Pipeline Orchestrator

This module orchestrates the complete CAD-RAG workflow by integrating:
- Existing retrieval system (InteractiveCADRetriever)
- Gemini API for RAG-based generation
- Python-to-JSON conversion pipeline

Author: Claude Code
"""

import os
import json
import tempfile
import shutil
import time
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

# Import existing components
from interactive_cad_retrieval import InteractiveCADRetriever
from py2json_converter import PythonToJSONConverter

# Import Gemini API directly
import google.generativeai as genai
from PIL import Image

# Import Bethany conversion libraries
from .bethany_lib.extrude import CADSequence
from .bethany_lib.cad2code import get_cad_code

class CADRAGPipeline:
    """Main orchestrator for CAD-RAG pipeline"""
    
    def __init__(self, database_dir: str = "database_embeddings", 
                 model_path: str = "/media/user/data/CSTBIR/data/model_epoch_15.pt", config_path: str = "config.yml"):
        """
        Initialize the CAD-RAG pipeline.
        
        Args:
            database_dir: Path to encoded database directory
            model_path: Path to trained model checkpoint (optional)
            config_path: Path to configuration file
        """
        self.database_dir = database_dir
        self.model_path = model_path
        self.config_path = config_path
        
        # Initialize components
        self.retriever = None
        self.converter = PythonToJSONConverter()
        
        # Gemini API configuration
        self.api_keys = [
            "AIzaSyBc1Rf5FqjYJJ6LyKh2r_HUNtZXIdOyJDw",
            "AIzaSyBFxiGOjyGznIGw2PoEWoUJbKy8Yx0BBkU",
            "AIzaSyBDQXo5FoaE9RT_6GTdLfjdq1I2zN4e5fI",
            "AIzaSyARF7uSx2IONDPbVqDea_86uu6WevAaX8U",
            "AIzaSyDvsyXlbakMP8xJzqeaaSlv4XZJxURtu_g"
        ]
        self.current_api_key_index = 0
        
        # Initialize retriever
        self._initialize_retriever()
        
    def _initialize_retriever(self):
        """Initialize the retrieval system"""
        try:
            self.retriever = InteractiveCADRetriever(
                database_dir=self.database_dir,
                model_path=self.model_path,
                config_path=self.config_path
            )
            print("✓ Retrieval system initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize retrieval system: {e}")
            raise
    
    
    
    
    def retrieve_models(self, sketch_path: Optional[str] = None, 
                       text_query: Optional[str] = None, 
                       top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar CAD models based on sketch and/or text query.
        
        Args:
            sketch_path: Path to sketch image (optional)
            text_query: Text description (optional)
            top_k: Number of top results to return
            
        Returns:
            List of dictionaries containing model information
        """
        if not self.retriever:
            raise RuntimeError("Retrieval system not initialized")
            
        if not sketch_path and not text_query:
            raise ValueError("Must provide either sketch_path or text_query")
        
        try:
            # Encode query
            query_embedding = self.retriever.encode_query(sketch_path, text_query)
            
            # Search for similar models
            results = self.retriever.search_similar_models(query_embedding, top_k)
            
            # Format results for GUI consumption
            formatted_results = []
            for i, (model_id, score, metadata) in enumerate(results):
                formatted_results.append({
                    'rank': i + 1,
                    'model_id': model_id,
                    'similarity_score': score,
                    'metadata': metadata,
                    'image_path': metadata.get('file_path', ''),
                    'view_type': metadata.get('view_type', 'unknown')
                })
            
            print(f"✓ Retrieved {len(formatted_results)} models")
            return formatted_results
            
        except Exception as e:
            print(f"✗ Error during retrieval: {e}")
            raise
    
    def generate_cad_sequence(self, user_query: str, selected_model: Dict[str, Any], 
                            instructions: str = "", sketch_path: Optional[str] = None) -> str:
        """
        Generate modified CAD sequence using Gemini RAG approach.
        
        Args:
            user_query: User's original text query
            selected_model: Selected model from retrieval results
            instructions: Additional modification instructions
            sketch_path: Path to sketch image (optional)
            
        Returns:
            Generated Python CAD code
        """
        # Get retrieved model's Python code
        model_id = selected_model['model_id']
        retrieved_python_code = self._get_model_python_code(model_id)
        
        if not retrieved_python_code:
            raise ValueError(f"Could not find Python code for model {model_id}")
        
        # Prepare RAG prompt
        prompt = self._create_rag_prompt(
            user_query, retrieved_python_code, instructions, sketch_path
        )
        
        # Generate using Gemini API
        try:
            generated_code = self._call_gemini_api(prompt, sketch_path)
            print("✓ Successfully generated CAD sequence")
            return generated_code
            
        except Exception as e:
            print(f"✗ Error during generation: {e}")
            raise
    
    def convert_to_json(self, python_code: str) -> Dict[str, Any]:
        """
        Convert Python CAD code to JSON format.
        
        Args:
            python_code: Generated Python CAD code
            
        Returns:
            JSON representation of the CAD model
        """
        try:
            # Create temporary file for conversion
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_py:
                temp_py.write(python_code)
                temp_py_path = temp_py.name
            
            # Create temporary JSON output file
            temp_json_path = temp_py_path.replace('.py', '.json')
            
            try:
                # Convert using existing converter
                self.converter.convert_file(temp_py_path, temp_json_path)
                
                # Read and return JSON
                with open(temp_json_path, 'r') as f:
                    json_data = json.load(f)
                
                print("✓ Successfully converted to JSON")
                return json_data
                
            finally:
                # Cleanup temporary files
                for temp_file in [temp_py_path, temp_json_path]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
        
        except Exception as e:
            print(f"✗ Error during conversion: {e}")
            raise
    
    def _get_model_python_code(self, model_id: str) -> Optional[str]:
        """Get Python code for a given model ID"""
        # Try to find corresponding Python file
        possible_paths = [
            f"results/{model_id}.py",
            f"test_cad_retrieval/{model_id}.py",
            f"interactive_test/{model_id}.py"
        ]
        
        for py_path in possible_paths:
            if os.path.exists(py_path):
                with open(py_path, 'r') as f:
                    return f.read()
        
        # If not found, try to convert from JSON
        json_path = f"/media/user/Data/MultiCAD/jsonfiles/{model_id}.json"
        if os.path.exists(json_path):
            print(f"Found JSON file for {model_id}, attempting to convert to Python...")
            try:
                python_code = self._convert_json_to_python(json_path)
                if python_code:
                    return python_code
            except Exception as e:
                print(f"Failed to convert JSON to Python for {model_id}: {e}")
        
        return None
    
    def _convert_json_to_python(self, json_path: str) -> Optional[str]:
        """Convert JSON CAD file to Python code using local Bethany libraries"""
        try:
            # Load JSON data
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Convert JSON to CADSequence using Bethany library
            cad_seq = CADSequence.from_dict(json_data)
            
            # Generate Python code using Bethany library
            python_code = get_cad_code(cad_seq)
            
            model_id = os.path.basename(json_path).split('.')[0]
            print(f"✓ Successfully converted JSON to Python for {model_id}")
            return python_code
            
        except Exception as e:
            model_id = os.path.basename(json_path).split('.')[0] if json_path else "unknown"
            print(f"✗ Error converting JSON to Python for {model_id}: {e}")
            return None
    
    def _create_rag_prompt(self, user_query: str, retrieved_code: str, 
                          instructions: str, sketch_path: Optional[str] = None) -> str:
        """Create RAG prompt based on bacth_save_prompt.py template"""
        
        sketch_info = ""
        if sketch_path:
            sketch_info = f"User has provided a sketch image showing their desired design."
        
        prompt = f"""You are a CAD design assistant. The user wants to create a 3D CAD model based on their requirements.

User Query: {user_query}
{sketch_info}

Here is a similar CAD model from our database that you should use as a reference:

```python
{retrieved_code}
```

Additional Instructions: {instructions}

Please modify the reference model to match the user's requirements. Follow these guidelines:
1. Keep the same Python API structure and function calls
2. Focus on geometric changes described in the user query
3. Ensure the code is executable and syntactically correct
4. Maintain proper variable naming and structure
5. Only change geometric parameters, dimensions, and shapes as needed
6. If the user wants significant changes, adapt the structure accordingly

IMPORTANT: Your response must be ONLY valid Python code wrapped in ```python code blocks. Do not include any explanations, comments, or text outside the code block. The code must be complete and syntactically correct.

Generate the complete Python code for the modified CAD model:

```python"""

        return prompt
    
    def _call_gemini_api(self, prompt: str, sketch_path: Optional[str] = None) -> str:
        """Call Gemini API directly without subprocess"""
        
        # Iterate through API keys with fallback
        for attempt in range(len(self.api_keys)):
            try:
                # Configure API key
                api_key = self.api_keys[self.current_api_key_index]
                genai.configure(api_key=api_key)
                
                # Initialize model
                model = genai.GenerativeModel(model_name="gemini-2.5-pro")
                
                # Prepare content
                content = [prompt]
                if sketch_path and os.path.exists(sketch_path):
                    image = Image.open(sketch_path)
                    content.append(image)
                
                # Generate response
                response = model.generate_content(
                    content,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        top_p=1.0,
                        max_output_tokens=8192,
                    )
                )
                
                if response.text:
                    return self._extract_python_code(response.text)
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                print(f"API key {self.current_api_key_index + 1} failed: {e}")
                self.current_api_key_index = (self.current_api_key_index + 1) % len(self.api_keys)
                
                if attempt == len(self.api_keys) - 1:
                    raise RuntimeError("All API keys failed")
                continue
        
        raise RuntimeError("Failed to generate response")
    
    
    def _extract_python_code(self, response_text: str) -> str:
        """Extract and validate Python code from Gemini response"""
        extracted_code = None
        
        # Look for code blocks
        if "```python" in response_text:
            start = response_text.find("```python") + len("```python")
            end = response_text.find("```", start)
            if end != -1:
                extracted_code = response_text[start:end].strip()
        
        # Look for any code blocks
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            if end != -1:
                extracted_code = response_text[start:end].strip()
        
        # If no code blocks, try to extract everything
        if not extracted_code:
            extracted_code = response_text.strip()
        
        # Validate and clean the extracted code
        return self._validate_and_clean_python_code(extracted_code)
    
    def _validate_and_clean_python_code(self, code: str) -> str:
        """Validate and clean Python code to fix common issues"""
        if not code:
            raise ValueError("No code extracted from response")
        
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        # Fix common issues
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines at start
            if not cleaned_lines and not line.strip():
                continue
                
            # Fix unterminated strings by removing incomplete lines
            if line.count('"') % 2 != 0 and line.count("'") % 2 != 0:
                print(f"⚠️  Skipping line with unterminated string: {line[:50]}...")
                continue
                
            cleaned_lines.append(line)
        
        cleaned_code = '\n'.join(cleaned_lines)
        
        # Basic syntax validation
        try:
            compile(cleaned_code, '<generated_code>', 'exec')
            print("✓ Generated code syntax validation passed")
        except SyntaxError as e:
            print(f"⚠️  Syntax error in generated code: {e}")
            # Try to fix common issues
            cleaned_code = self._fix_common_syntax_errors(cleaned_code)
            
            # Try validation again
            try:
                compile(cleaned_code, '<generated_code>', 'exec')
                print("✓ Code fixed and validated successfully")
            except SyntaxError as e2:
                print(f"⚠️  Could not fix syntax errors: {e2}")
                # Return original code - let the converter handle it
        
        return cleaned_code
    
    def _fix_common_syntax_errors(self, code: str) -> str:
        """Fix common syntax errors in generated code"""
        lines = code.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Skip lines that look like incomplete strings
            if line.strip().startswith('"') and not line.strip().endswith('"'):
                if line.count('"') == 1:
                    continue
            
            # Skip lines that look like incomplete comments
            if line.strip() and not line.strip().startswith('#') and '\"' in line:
                # Try to fix escaped quotes
                line = line.replace('\\"', '"')
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def process_complete_workflow(self, sketch_path: Optional[str] = None,
                                 text_query: Optional[str] = None,
                                 instructions: str = "",
                                 top_k: int = 10) -> Dict[str, Any]:
        """
        Process complete RAG workflow from query to JSON output.
        
        Args:
            sketch_path: Path to sketch image
            text_query: Text query
            instructions: Additional instructions
            top_k: Number of retrieval results
            
        Returns:
            Dictionary containing all results
        """
        # Step 1: Retrieve similar models
        retrieval_results = self.retrieve_models(sketch_path, text_query, top_k)
        
        # Step 2: Use top result for generation
        if not retrieval_results:
            raise ValueError("No similar models found")
        
        selected_model = retrieval_results[0]  # Use top result
        
        # Step 3: Generate CAD sequence
        query_text = text_query or "User sketch query"
        generated_code = self.generate_cad_sequence(
            query_text, selected_model, instructions, sketch_path
        )
        
        # Step 4: Convert to JSON
        json_output = self.convert_to_json(generated_code)
        
        return {
            'retrieval_results': retrieval_results,
            'selected_model': selected_model,
            'generated_code': generated_code,
            'json_output': json_output
        }


if __name__ == "__main__":
    # Test the pipeline
    pipeline = CADRAGPipeline()
    
    # Test retrieval
    results = pipeline.retrieve_models(text_query="cylindrical part with holes", top_k=5)
    print(f"Found {len(results)} similar models")
    
    if results:
        # Test generation
        generated = pipeline.generate_cad_sequence(
            "cylindrical part with holes", 
            results[0], 
            "Make it slightly larger"
        )
        print(f"Generated code length: {len(generated)}")