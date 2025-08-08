#!/usr/bin/env python3
"""
CAD-RAG GUI Application

Simple Tkinter-based interface for the CAD-RAG pipeline that allows users to:
1. Input text queries and/or upload sketches
2. View top-10 retrieval results
3. Generate modified CAD sequences
4. View and save generated Python code and JSON

Author: Claude Code
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from PIL import Image, ImageTk
from typing import List, Dict, Any, Optional
import json

from cad_rag_pipeline import CADRAGPipeline


class CADRAGGUIApp:
    """Main GUI application for CAD-RAG pipeline"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("CAD-RAG: Multi-Modal CAD Generation")
        self.root.geometry("1600x1000")
        
        # Configure default font sizes
        self.default_font = ("Arial", 12)
        self.large_font = ("Arial", 14)
        self.button_font = ("Arial", 12, "bold")
        
        # Configure default styles
        style = ttk.Style()
        style.configure("TLabel", font=self.default_font)
        style.configure("TButton", font=self.button_font, padding=10)
        style.configure("TLabelFrame.Label", font=self.large_font)
        
        # Initialize pipeline
        self.pipeline = None
        self.current_sketch_path = None
        self.retrieval_results = []
        self.selected_model_index = None
        
        # Initialize UI
        self.setup_ui()
        
        # Initialize pipeline in background
        self.init_pipeline_async()
    
    def setup_ui(self):
        """Setup the main UI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left panel - Query input
        self.setup_query_panel(main_frame)
        
        # Right panel - Results display
        self.setup_results_panel(main_frame)
        
        # Bottom panel - Generation controls
        self.setup_generation_panel(main_frame)
        
        # Status bar
        self.setup_status_bar(main_frame)
    
    def setup_query_panel(self, parent):
        """Setup the query input panel"""
        query_frame = ttk.LabelFrame(parent, text="Query Input", padding="10")
        query_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Text query
        ttk.Label(query_frame, text="Text Query:", font=self.large_font).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.text_query = tk.StringVar()
        text_entry = ttk.Entry(query_frame, textvariable=self.text_query, width=50, font=self.default_font)
        text_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Sketch upload
        ttk.Label(query_frame, text="Sketch Image:", font=self.large_font).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        sketch_frame = ttk.Frame(query_frame)
        sketch_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.sketch_label = ttk.Label(sketch_frame, text="No sketch uploaded", font=self.default_font)
        self.sketch_label.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(sketch_frame, text="Upload Sketch", 
                  command=self.upload_sketch).grid(row=0, column=1, sticky=tk.E)
        
        # Search button
        self.search_button = ttk.Button(query_frame, text="Search Similar Models", 
                                       command=self.search_models, state=tk.DISABLED)
        self.search_button.grid(row=4, column=0, pady=(15, 0))
        
        # Configure column weights
        query_frame.columnconfigure(0, weight=1)
        sketch_frame.columnconfigure(0, weight=1)
    
    def setup_results_panel(self, parent):
        """Setup the results display panel"""
        results_frame = ttk.LabelFrame(parent, text="Retrieval Results", padding="10")
        results_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Results list with scrollbar
        list_frame = ttk.Frame(results_frame)
        list_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure scrollable frame
        self.results_canvas = tk.Canvas(list_frame, height=400)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.results_canvas.yview)
        self.results_scrollable = ttk.Frame(self.results_canvas)
        
        self.results_scrollable.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        
        self.results_canvas.create_window((0, 0), window=self.results_scrollable, anchor="nw")
        self.results_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.results_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Selected model info
        self.selected_info = ttk.Label(results_frame, text="No model selected", font=self.default_font)
        self.selected_info.grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        
        # Configure weights
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
    
    def setup_generation_panel(self, parent):
        """Setup the generation controls panel"""
        gen_frame = ttk.LabelFrame(parent, text="CAD Generation", padding="10")
        gen_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5), pady=(10, 0))
        
        # Additional instructions
        ttk.Label(gen_frame, text="Additional Instructions:", font=self.large_font).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.instructions = tk.StringVar()
        instructions_entry = ttk.Entry(gen_frame, textvariable=self.instructions, width=50, font=self.default_font)
        instructions_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Generate button
        self.generate_button = ttk.Button(gen_frame, text="Generate CAD Model", 
                                         command=self.generate_model, state=tk.DISABLED)
        self.generate_button.grid(row=2, column=0, pady=(0, 15))
        
        # Output display
        ttk.Label(gen_frame, text="Generated Code:", font=self.large_font).grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        # Notebook for different output formats
        self.output_notebook = ttk.Notebook(gen_frame)
        self.output_notebook.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Python code tab
        python_frame = ttk.Frame(self.output_notebook)
        self.python_output = scrolledtext.ScrolledText(python_frame, height=18, width=60, font=("Courier", 11))
        self.python_output.pack(fill=tk.BOTH, expand=True)
        self.output_notebook.add(python_frame, text="Python Code")
        
        # JSON tab
        json_frame = ttk.Frame(self.output_notebook)
        self.json_output = scrolledtext.ScrolledText(json_frame, height=18, width=60, font=("Courier", 11))
        self.json_output.pack(fill=tk.BOTH, expand=True)
        self.output_notebook.add(json_frame, text="JSON Output")
        
        # Save buttons
        button_frame = ttk.Frame(gen_frame)
        button_frame.grid(row=5, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(button_frame, text="Save Python", 
                  command=self.save_python).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Save JSON", 
                  command=self.save_json).pack(side=tk.LEFT)
        
        # Configure weights
        gen_frame.columnconfigure(0, weight=1)
        gen_frame.rowconfigure(4, weight=1)
    
    def setup_status_bar(self, parent):
        """Setup the status bar"""
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing CAD-RAG pipeline...")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, font=self.default_font, padding=5)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def init_pipeline_async(self):
        """Initialize the pipeline in a separate thread"""
        def init_pipeline():
            try:
                self.pipeline = CADRAGPipeline()
                self.root.after(0, self.on_pipeline_ready)
            except Exception as e:
                self.root.after(0, lambda: self.on_pipeline_error(str(e)))
        
        threading.Thread(target=init_pipeline, daemon=True).start()
    
    def on_pipeline_ready(self):
        """Called when pipeline is ready"""
        self.status_var.set("Ready - Enter text query and/or upload sketch to begin")
        self.search_button.config(state=tk.NORMAL)
    
    def on_pipeline_error(self, error_msg):
        """Called when pipeline initialization fails"""
        self.status_var.set(f"Error: {error_msg}")
        messagebox.showerror("Initialization Error", f"Failed to initialize pipeline:\n{error_msg}")
    
    def upload_sketch(self):
        """Handle sketch upload"""
        file_path = filedialog.askopenfilename(
            title="Select Sketch Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")]
        )
        
        if file_path:
            self.current_sketch_path = file_path
            filename = os.path.basename(file_path)
            self.sketch_label.config(text=f"Uploaded: {filename}")
            self.status_var.set(f"Sketch uploaded: {filename}")
    
    def search_models(self):
        """Search for similar models"""
        if not self.pipeline:
            messagebox.showerror("Error", "Pipeline not initialized")
            return
        
        text_query = self.text_query.get().strip()
        
        if not text_query and not self.current_sketch_path:
            messagebox.showwarning("Warning", "Please provide either text query or sketch image")
            return
        
        # Start search in background
        self.search_button.config(state=tk.DISABLED)
        self.status_var.set("Searching for similar models...")
        
        def search_thread():
            try:
                results = self.pipeline.retrieve_models(
                    sketch_path=self.current_sketch_path,
                    text_query=text_query if text_query else None,
                    top_k=10
                )
                self.root.after(0, lambda: self.on_search_complete(results))
            except Exception as e:
                self.root.after(0, lambda: self.on_search_error(str(e)))
        
        threading.Thread(target=search_thread, daemon=True).start()
    
    def on_search_complete(self, results):
        """Handle search completion"""
        self.retrieval_results = results
        self.display_results(results)
        self.search_button.config(state=tk.NORMAL)
        self.status_var.set(f"Found {len(results)} similar models")
    
    def on_search_error(self, error_msg):
        """Handle search error"""
        self.search_button.config(state=tk.NORMAL)
        self.status_var.set(f"Search failed: {error_msg}")
        messagebox.showerror("Search Error", f"Failed to search models:\n{error_msg}")
    
    def display_results(self, results):
        """Display retrieval results"""
        # Clear previous results
        for widget in self.results_scrollable.winfo_children():
            widget.destroy()
        
        self.result_vars = []
        
        for i, result in enumerate(results):
            # Create result frame
            result_frame = ttk.Frame(self.results_scrollable)
            result_frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Radio button for selection
            var = tk.BooleanVar()
            self.result_vars.append(var)
            
            radio = ttk.Checkbutton(result_frame, variable=var, 
                                   command=lambda idx=i: self.select_model(idx))
            radio.pack(side=tk.LEFT)
            
            # Model info
            info_text = f"Rank {result['rank']}: {result['model_id']} (Score: {result['similarity_score']:.3f})"
            ttk.Label(result_frame, text=info_text, font=self.default_font).pack(side=tk.LEFT, padx=(5, 0))
            
            # Try to display image thumbnail
            self.try_display_thumbnail(result_frame, result['image_path'])
    
    def try_display_thumbnail(self, parent, image_path):
        """Try to display image thumbnail"""
        try:
            if os.path.exists(image_path):
                img = Image.open(image_path)
                img.thumbnail((96, 96))  # Larger thumbnails
                photo = ImageTk.PhotoImage(img)
                
                label = ttk.Label(parent, image=photo)
                label.image = photo  # Keep a reference
                label.pack(side=tk.RIGHT, padx=(5, 0))
        except Exception:
            pass  # Silently ignore thumbnail errors
    
    def select_model(self, index):
        """Handle model selection"""
        # Clear other selections
        for i, var in enumerate(self.result_vars):
            if i != index:
                var.set(False)
        
        self.selected_model_index = index
        selected_model = self.retrieval_results[index]
        
        self.selected_info.config(text=f"Selected: {selected_model['model_id']}")
        self.generate_button.config(state=tk.NORMAL)
    
    def generate_model(self):
        """Generate CAD model"""
        if not self.pipeline or self.selected_model_index is None:
            messagebox.showerror("Error", "No model selected")
            return
        
        selected_model = self.retrieval_results[self.selected_model_index]
        text_query = self.text_query.get().strip() or "User sketch query"
        instructions = self.instructions.get().strip()
        
        # Start generation in background
        self.generate_button.config(state=tk.DISABLED)
        self.status_var.set("Generating CAD model...")
        
        def generate_thread():
            try:
                # Generate Python code
                python_code = self.pipeline.generate_cad_sequence(
                    text_query, selected_model, instructions, self.current_sketch_path
                )
                
                # Convert to JSON
                json_data = self.pipeline.convert_to_json(python_code)
                
                self.root.after(0, lambda: self.on_generation_complete(python_code, json_data))
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.on_generation_error(error_msg))
        
        threading.Thread(target=generate_thread, daemon=True).start()
    
    def on_generation_complete(self, python_code, json_data):
        """Handle generation completion"""
        # Display Python code
        self.python_output.delete(1.0, tk.END)
        self.python_output.insert(tk.END, python_code)
        
        # Display JSON
        self.json_output.delete(1.0, tk.END)
        self.json_output.insert(tk.END, json.dumps(json_data, indent=2))
        
        self.generate_button.config(state=tk.NORMAL)
        self.status_var.set("CAD model generated successfully!")
    
    def on_generation_error(self, error_msg):
        """Handle generation error"""
        self.generate_button.config(state=tk.NORMAL)
        self.status_var.set(f"Generation failed: {error_msg}")
        messagebox.showerror("Generation Error", f"Failed to generate model:\n{error_msg}")
    
    def save_python(self):
        """Save Python code to file"""
        content = self.python_output.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No Python code to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".py",
            filetypes=[("Python files", "*.py")]
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(content)
            self.status_var.set(f"Python code saved to {file_path}")
    
    def save_json(self):
        """Save JSON data to file"""
        content = self.json_output.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Warning", "No JSON data to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(content)
            self.status_var.set(f"JSON data saved to {file_path}")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = CADRAGGUIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()