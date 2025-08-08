#!/usr/bin/env python3
"""
CAD-RAG Basic Usage Example

This example demonstrates the core functionality of the CAD-RAG framework:
1. Initialize the pipeline
2. Retrieve similar CAD models using text and/or sketch queries
3. Generate modified CAD models using Gemini API
4. Convert between Python and JSON formats

Prerequisites:
- CAD-RAG environment activated: conda activate cad-rag
- Gemini API keys configured in cad_rag_pipeline.py
- Model weights available at data/model_epoch_15.pt
- Database embeddings available in database_embeddings/
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cad_rag import CADRAGPipeline

def basic_retrieval_example():
    """Example 1: Basic text-based retrieval"""
    print("🔍 Example 1: Basic Text-Based Retrieval")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = CADRAGPipeline(
        database_dir="database_embeddings",
        model_path="data/model_epoch_15.pt"
    )
    
    # Perform text-based retrieval
    query = "cylindrical part with holes"
    print(f"Query: '{query}'")
    
    try:
        results = pipeline.retrieve_models(
            text_query=query,
            top_k=5
        )
        
        print(f"✓ Found {len(results)} similar models:")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. Model {result['model_id']} (similarity: {result['similarity_score']:.3f})")
            
        return results
        
    except Exception as e:
        print(f"✗ Retrieval failed: {e}")
        return []

def multimodal_retrieval_example():
    """Example 2: Multi-modal retrieval with sketch and text"""
    print("\n🎨 Example 2: Multi-Modal Retrieval (Sketch + Text)")
    print("=" * 50)
    
    pipeline = CADRAGPipeline()
    
    # Example with both sketch and text
    sketch_path = "path/to/your/sketch.png"  # Replace with actual sketch path
    text_query = "mechanical part with circular features"
    
    print(f"Sketch: {sketch_path}")
    print(f"Text: '{text_query}'")
    
    try:
        # Check if sketch file exists
        if not os.path.exists(sketch_path):
            print(f"⚠️  Sketch file not found: {sketch_path}")
            print("   Falling back to text-only retrieval...")
            sketch_path = None
        
        results = pipeline.retrieve_models(
            sketch_path=sketch_path,
            text_query=text_query,
            top_k=5
        )
        
        print(f"✓ Found {len(results)} similar models:")
        for i, result in enumerate(results[:3], 1):
            print(f"  {i}. Model {result['model_id']} (similarity: {result['similarity_score']:.3f})")
            
        return results
        
    except Exception as e:
        print(f"✗ Retrieval failed: {e}")
        return []

def generation_example(retrieval_results):
    """Example 3: RAG-based CAD model generation"""
    print("\n🤖 Example 3: AI-Powered CAD Generation")
    print("=" * 50)
    
    if not retrieval_results:
        print("⚠️  No retrieval results available for generation")
        return None
    
    pipeline = CADRAGPipeline()
    
    # Use the top retrieval result as reference
    selected_model = retrieval_results[0]
    print(f"Using model {selected_model['model_id']} as reference")
    
    # Generation parameters
    user_query = "cylindrical part with holes"
    instructions = "Make it 20% larger and add two more holes symmetrically"
    
    print(f"Original query: '{user_query}'")
    print(f"Modification: '{instructions}'")
    
    try:
        print("🔄 Generating modified CAD model...")
        generated_code = pipeline.generate_cad_sequence(
            user_query=user_query,
            selected_model=selected_model,
            instructions=instructions
        )
        
        print("✓ Generation completed!")
        print(f"📝 Generated code ({len(generated_code)} characters):")
        
        # Show first few lines
        lines = generated_code.split('\n')
        for i, line in enumerate(lines[:5], 1):
            print(f"  {i}: {line}")
        
        if len(lines) > 5:
            print(f"  ... ({len(lines) - 5} more lines)")
        
        return generated_code
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return None

def conversion_example(python_code):
    """Example 4: Python to JSON conversion"""
    print("\n🔄 Example 4: Python to JSON Conversion")
    print("=" * 50)
    
    if not python_code:
        print("⚠️  No Python code available for conversion")
        return None
    
    pipeline = CADRAGPipeline()
    
    try:
        print("🔄 Converting Python code to JSON...")
        json_output = pipeline.convert_to_json(python_code)
        
        print("✓ Conversion completed!")
        print(f"📋 JSON structure:")
        
        if isinstance(json_output, dict):
            for key in list(json_output.keys())[:5]:
                print(f"  - {key}: {type(json_output[key]).__name__}")
        else:
            print(f"  Type: {type(json_output)}")
        
        return json_output
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        return None

def complete_workflow_example():
    """Example 5: Complete end-to-end workflow"""
    print("\n🌟 Example 5: Complete End-to-End Workflow")
    print("=" * 50)
    
    pipeline = CADRAGPipeline()
    
    # Parameters for complete workflow
    text_query = "mechanical bracket with mounting holes"
    instructions = "Add reinforcement ribs and increase thickness"
    
    try:
        print("🔄 Running complete workflow...")
        results = pipeline.process_complete_workflow(
            text_query=text_query,
            instructions=instructions,
            top_k=3
        )
        
        print("✓ Complete workflow finished!")
        print("📊 Results summary:")
        print(f"  - Retrieved models: {len(results['retrieval_results'])}")
        print(f"  - Selected model: {results['selected_model']['model_id']}")
        print(f"  - Generated code: {len(results['generated_code'])} chars")
        print(f"  - JSON output: {type(results['json_output'])}")
        
        return results
        
    except Exception as e:
        print(f"✗ Complete workflow failed: {e}")
        return None

def main():
    """Run all examples"""
    print("🚀 CAD-RAG Framework Examples")
    print("=" * 60)
    print("This script demonstrates the core capabilities of CAD-RAG")
    print()
    
    try:
        # Example 1: Basic retrieval
        results = basic_retrieval_example()
        
        # Example 2: Multi-modal retrieval
        multimodal_results = multimodal_retrieval_example()
        
        # Use results from first example if multimodal failed
        if multimodal_results:
            results = multimodal_results
        
        # Example 3: Generation (requires retrieval results)
        generated_code = generation_example(results)
        
        # Example 4: Conversion (requires generated code)
        json_output = conversion_example(generated_code)
        
        # Example 5: Complete workflow
        complete_results = complete_workflow_example()
        
        print("\n🎉 All examples completed!")
        print("💡 Try the GUI interface: python src/cad_rag/cad_rag_gui.py")
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n✗ Examples failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()