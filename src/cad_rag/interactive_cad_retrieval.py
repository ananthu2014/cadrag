#!/usr/bin/env python3
"""
Interactive CAD Model Retrieval System

This script performs interactive sketch+text based retrieval on encoded CAD databases.
Users can query with sketches, text, or both, then interactively select relevant results
to save along with their feature summaries.

Usage:
    python interactive_cad_retrieval.py --query_sketch sketch.png --database_dir ./db --output_dir ./results
    python interactive_cad_retrieval.py --query_text "cylindrical part" --database_dir ./db --output_dir ./results  
    python interactive_cad_retrieval.py --query_sketch sketch.png --query_text "gear" --database_dir ./db --output_dir ./results
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button

from src import clip
from utils import load_config


class InteractiveCADRetriever:
    def __init__(self, database_dir: str, model_path: str = None, config_path: str = "config.yml"):
        """
        Initialize the interactive CAD retriever.
        
        Args:
            database_dir: Path to encoded database directory
            model_path: Path to trained model checkpoint (optional)
            config_path: Path to configuration file
        """
        self.database_dir = database_dir
        self.config = load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP model
        self.model, self.preprocess = clip.load(
            self.config['model']['model_name'], 
            device=self.device, 
            jit=False
        )
        
        # Load trained weights if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading trained model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("✓ Trained model loaded successfully")
        else:
            print("Using pretrained CLIP model")
        
        self.model.eval()
        
        # Load database
        self.load_database()
        
        # Initialize selection state
        self.selected_results = []
        
    def load_database(self):
        """Load the encoded database"""
        required_files = {
            'metadata': os.path.join(self.database_dir, "metadata.json"),
            'embeddings': os.path.join(self.database_dir, "embeddings.pt"),
            'matrix': os.path.join(self.database_dir, "embedding_matrix.pt"),
            'index': os.path.join(self.database_dir, "uid_to_index.json")
        }
        
        optional_files = {
            'display': os.path.join(self.database_dir, "display_preferences.json")
        }
        
        # Check required files
        missing_files = [name for name, path in required_files.items() if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"Missing database files: {missing_files}")
        
        # Load metadata
        with open(required_files['metadata'], 'r') as f:
            self.metadata = json.load(f)
        
        # Load embeddings
        self.embeddings = torch.load(required_files['embeddings'], map_location='cpu')
        
        # Load embedding matrix for fast search
        self.embedding_matrix = torch.load(required_files['matrix'], map_location='cpu').float()
        
        # Load UID to index mapping
        with open(required_files['index'], 'r') as f:
            self.uid_to_index = json.load(f)
        
        # Load display preferences (optional, create if missing)
        if os.path.exists(optional_files['display']):
            with open(optional_files['display'], 'r') as f:
                self.display_preferences = json.load(f)
        else:
            print("⚠ Display preferences not found, creating from metadata...")
            self.display_preferences = self._create_display_preferences()
        
        print(f"✓ Database loaded: {self.metadata['total_entries']} entries, {self.metadata['total_models']} models")
        
        # JSON files directory
        self.json_dir = "/media/user/Data/MultiCAD/jsonfiles"
        self.bethany_path = "/media/user/data/OpenECAD_Project/Bethany"
        self.bethany_env = "Bethany"
    
    def _create_display_preferences(self) -> Dict:
        """Create display preferences from metadata for backwards compatibility"""
        view_priority = ['iso1', 'iso2', 'top']  # Prefer isometric views
        display_preferences = {}
        
        # Group entries by model_id
        model_views = {}
        for uid, entry in self.metadata["entries"].items():
            model_id = entry["model_id"]
            view_type = entry["view_type"]
            
            if model_id not in model_views:
                model_views[model_id] = {}
            model_views[model_id][view_type] = uid
        
        # Create display preferences
        for model_id, views in model_views.items():
            for preferred_view in view_priority:
                if preferred_view in views:
                    display_preferences[model_id] = {
                        "preferred_view": preferred_view,
                        "preferred_uid": views[preferred_view]
                    }
                    break
        
        return display_preferences
    
    def get_json_file_path(self, model_id: str) -> str:
        """Get the path to JSON file for a given model ID"""
        return os.path.join(self.json_dir, f"{model_id}.json")
    
    def convert_json_to_python(self, json_path: str, output_dir: str) -> str:
        """Convert JSON file to Python code using Bethany environment"""
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Get model ID from JSON filename
            model_id = os.path.basename(json_path).split('.')[0]
            python_output_path = os.path.join(output_dir, f"{model_id}.py")
            
            # Create temporary input directory with single JSON file
            temp_input_dir = os.path.join(output_dir, "temp_json")
            os.makedirs(temp_input_dir, exist_ok=True)
            temp_json_path = os.path.join(temp_input_dir, f"{model_id}.json")
            shutil.copy2(json_path, temp_json_path)
            
            # Construct command to run json2py.py in Bethany environment
            # Convert to absolute paths
            abs_temp_input_dir = os.path.abspath(temp_input_dir)
            abs_output_dir = os.path.abspath(output_dir)
            
            cmd = [
                "bash", "-c",
                f"source /home/user/anaconda3/etc/profile.d/conda.sh && "
                f"conda activate {self.bethany_env} && "
                f"cd {self.bethany_path} && "
                f"python json2py.py --src {abs_temp_input_dir} --num 1 --idx 0 -o {abs_output_dir}"
            ]
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Give the file system a moment to sync
            time.sleep(0.5)
            
            # Check if file exists before cleanup
            file_exists = os.path.exists(python_output_path)
            
            # Cleanup temporary directory
            shutil.rmtree(temp_input_dir, ignore_errors=True)
            
            if result.returncode == 0 and file_exists:
                return python_output_path
            else:
                print(f"Error converting JSON to Python: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"Timeout converting JSON for model {model_id}")
            return None
        except Exception as e:
            print(f"Error converting JSON for model {model_id}: {e}")
            return None
        
    def encode_query(self, sketch_path: Optional[str] = None, text_prompt: Optional[str] = None) -> torch.Tensor:
        """
        Encode query from sketch and/or text inputs.
        
        Args:
            sketch_path: Path to sketch image (optional)
            text_prompt: Text description (optional)
            
        Returns:
            Query embedding tensor
        """
        if not sketch_path and not text_prompt:
            raise ValueError("Must provide either sketch_path or text_prompt or both")
        
        embeddings = []
        
        # Encode sketch if provided
        if sketch_path:
            if not os.path.exists(sketch_path):
                raise FileNotFoundError(f"Sketch file not found: {sketch_path}")
            
            image = Image.open(sketch_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                sketch_features = self.model.encode_image(image_tensor)
                sketch_features = F.normalize(sketch_features, dim=-1)
                embeddings.append(sketch_features.cpu().squeeze(0))
            
            print(f"✓ Sketch encoded: {sketch_path}")
        
        # Encode text if provided
        if text_prompt:
            text_tokens = clip.tokenize([text_prompt]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = F.normalize(text_features, dim=-1)
                embeddings.append(text_features.cpu().squeeze(0))
            
            print(f"✓ Text encoded: '{text_prompt}'")
        
        # Combine embeddings if both provided (STBIR approach)
        if len(embeddings) == 2:
            query_embedding = (embeddings[0] + embeddings[1]) / 2
            query_embedding = F.normalize(query_embedding, dim=-1)
            print("✓ Combined sketch+text query (STBIR)")
        else:
            query_embedding = embeddings[0]
            if sketch_path:
                print("✓ Sketch-only query (SBIR)")
            else:
                print("✓ Text-only query (TBIR)")
        
        return query_embedding
    
    def search_similar_models(self, query_embedding: torch.Tensor, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar models using the query embedding.
        
        Args:
            query_embedding: Query embedding tensor
            top_k: Number of top results to return
            
        Returns:
            List of (model_id, similarity_score, metadata) tuples
        """
        # Compute similarities with all embeddings
        similarities = torch.mm(query_embedding.float().unsqueeze(0), self.embedding_matrix.T).squeeze(0)
        
        # Get top results
        top_scores, top_indices = torch.topk(similarities, min(top_k * 3, len(similarities)))  # Get more to filter by model
        
        # Group by model and keep best score per model
        model_scores = {}
        for score, idx in zip(top_scores, top_indices):
            # Find UID from index
            uid = None
            for u, i in self.uid_to_index.items():
                if i == idx.item():
                    uid = u
                    break
            
            if uid and uid in self.metadata["entries"]:
                entry = self.metadata["entries"][uid]
                model_id = entry["model_id"]
                
                # Keep best score for this model
                if model_id not in model_scores or score.item() > model_scores[model_id][0]:
                    model_scores[model_id] = (score.item(), uid, entry)
        
        # Sort by score and return top k models
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        
        results = []
        for model_id, (score, uid, entry) in sorted_models:
            results.append((model_id, score, entry))
        
        return results
    
    def display_results_interactive(self, results: List[Tuple[str, float, Dict]], 
                                  query_info: Dict) -> List[int]:
        """
        Display search results in an interactive grid and get user selections.
        
        Args:
            results: List of (model_id, similarity_score, metadata) tuples
            query_info: Information about the query
            
        Returns:
            List of selected result indices
        """
        print(f"\n{'='*60}")
        print("SEARCH RESULTS")
        print(f"{'='*60}")
        print(f"Query: {query_info}")
        print(f"Found {len(results)} similar models\n")
        
        # Display results in a simple text format first
        for i, (model_id, score, metadata) in enumerate(results, 1):
            print(f"{i:2d}. Model ID: {model_id}")
            print(f"    Similarity: {score:.4f}")
            print(f"    View: {metadata['view_type']}")
            print(f"    File: {metadata['filename']}")
            print()
        
        # Get user selection
        print("Select relevant results (e.g., '1,3,7' or 'all' or 'none'):")
        while True:
            try:
                selection = input("Your selection: ").strip().lower()
                
                if selection == 'none' or selection == '':
                    return []
                elif selection == 'all':
                    return list(range(len(results)))
                else:
                    # Parse comma-separated indices
                    indices = []
                    for item in selection.split(','):
                        idx = int(item.strip()) - 1  # Convert to 0-based
                        if 0 <= idx < len(results):
                            indices.append(idx)
                        else:
                            print(f"Warning: Index {idx + 1} is out of range")
                    
                    if indices:
                        return indices
                    else:
                        print("No valid selections. Please try again.")
                        
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas (e.g., '1,3,7')")
            except KeyboardInterrupt:
                print("\nCancelled by user")
                return []
    
    def save_selected_results(self, results: List[Tuple[str, float, Dict]], 
                            selected_indices: List[int], output_dir: str, 
                            query_name: str) -> None:
        """
        Save selected results with JSON files and Python code.
        
        Args:
            results: List of all results
            selected_indices: Indices of selected results
            output_dir: Output directory
            query_name: Base name for output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        saved_count = 0
        for i, idx in enumerate(selected_indices):
            if idx >= len(results):
                continue
                
            model_id, score, metadata = results[idx]
            rank = idx + 1  # 1-based rank
            
            # Create subdirectory for this result
            result_dir = os.path.join(output_dir, f"{query_name}-result-{rank:02d}-{model_id}")
            os.makedirs(result_dir, exist_ok=True)
            
            # Save result metadata
            result_info = {
                "rank": rank,
                "model_id": model_id,
                "similarity": score,
                "view_type": metadata.get('view_type', 'unknown'),
                "image_file": metadata.get('filename', 'unknown'),
                "image_path": metadata.get('file_path', 'unknown'),
                "query_name": query_name
            }
            
            with open(os.path.join(result_dir, "result_info.json"), 'w') as f:
                json.dump(result_info, f, indent=2)
            
            # Copy image file
            source_image = metadata['file_path']
            if os.path.exists(source_image):
                image_output = os.path.join(result_dir, f"{model_id}_image.png")
                shutil.copy2(source_image, image_output)
                print(f"✓ Saved image: {image_output}")
            else:
                print(f"⚠ Image not found: {source_image}")
                continue
            
            # Copy JSON file if available
            json_path = self.get_json_file_path(model_id)
            if os.path.exists(json_path):
                json_dest = os.path.join(result_dir, f"{model_id}.json")
                shutil.copy2(json_path, json_dest)
                print(f"✓ Saved JSON: {json_dest}")
                
                # Convert JSON to Python code
                python_path = self.convert_json_to_python(json_path, result_dir)
                if python_path:
                    print(f"✓ Generated Python code: {python_path}")
                else:
                    print(f"⚠ Failed to generate Python code for {model_id}")
            else:
                print(f"⚠ JSON file not found: {json_path}")
            
            saved_count += 1
        
        print(f"\n✓ Successfully saved {saved_count} selected results to {output_dir}")
    
    def retrieve_interactive(self, sketch_path: Optional[str] = None, 
                           text_prompt: Optional[str] = None,
                           output_dir: str = "./results", top_k: int = 10) -> None:
        """
        Main interactive retrieval workflow.
        
        Args:
            sketch_path: Path to sketch image (optional)
            text_prompt: Text description (optional)
            output_dir: Output directory for results
            top_k: Number of top results to show
        """
        print(f"\n{'='*60}")
        print("INTERACTIVE CAD MODEL RETRIEVAL")
        print(f"{'='*60}")
        
        # Encode query
        query_embedding = self.encode_query(sketch_path, text_prompt)
        
        # Search for similar models
        print(f"\nSearching for top {top_k} similar models...")
        results = self.search_similar_models(query_embedding, top_k)
        
        if not results:
            print("No similar models found.")
            return
        
        # Prepare query info for display
        query_info = []
        if sketch_path:
            query_info.append(f"Sketch: {os.path.basename(sketch_path)}")
        if text_prompt:
            query_info.append(f"Text: '{text_prompt}'")
        query_info_str = " + ".join(query_info)
        
        # Interactive selection
        selected_indices = self.display_results_interactive(results, query_info_str)
        
        if not selected_indices:
            print("No results selected. Exiting.")
            return
        
        # Generate query name for output files
        query_name = "query"
        if sketch_path:
            query_name = os.path.splitext(os.path.basename(sketch_path))[0]
        elif text_prompt:
            # Create safe filename from text
            query_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in text_prompt[:30])
        
        # Save selected results
        self.save_selected_results(results, selected_indices, output_dir, query_name)


def main():
    parser = argparse.ArgumentParser(description="Interactive CAD model retrieval with sketch+text queries")
    parser.add_argument("--database_dir", required=True,
                        help="Directory containing encoded database")
    parser.add_argument("--query_sketch", type=str, default=None,
                        help="Path to sketch image for query")
    parser.add_argument("--query_text", type=str, default=None,
                        help="Text prompt for query")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory for selected results")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained model checkpoint (optional)")
    parser.add_argument("--config_path", type=str, default="config.yml",
                        help="Path to configuration file")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top results to show")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.query_sketch and not args.query_text:
        print("Error: Must provide either --query_sketch or --query_text or both")
        sys.exit(1)
    
    if args.query_sketch and not os.path.exists(args.query_sketch):
        print(f"Error: Sketch file not found: {args.query_sketch}")
        sys.exit(1)
    
    if not os.path.exists(args.database_dir):
        print(f"Error: Database directory not found: {args.database_dir}")
        sys.exit(1)
    
    # Initialize retriever
    try:
        retriever = InteractiveCADRetriever(
            database_dir=args.database_dir,
            model_path=args.model_path,
            config_path=args.config_path
        )
    except Exception as e:
        print(f"Error initializing retriever: {e}")
        sys.exit(1)
    
    # Run interactive retrieval
    try:
        retriever.retrieve_interactive(
            sketch_path=args.query_sketch,
            text_prompt=args.query_text,
            output_dir=args.output_dir,
            top_k=args.top_k
        )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error during retrieval: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()