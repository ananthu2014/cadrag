"""
CAD-RAG: Multi-Modal Retrieval-Augmented Generation for CAD Design

A unified framework for CAD model retrieval and generation using:
- CLIP-based multi-modal retrieval (sketch + text)
- Gemini API for RAG-based code generation  
- Integrated CAD model conversion pipeline

Author: Claude Code
Version: 1.0.0
"""

from .cad_rag_pipeline import CADRAGPipeline
from .interactive_cad_retrieval import InteractiveCADRetriever
from .py2json_converter import PythonToJSONConverter

__version__ = "1.0.0"
__author__ = "Claude Code"
__email__ = "noreply@anthropic.com"

__all__ = [
    "CADRAGPipeline",
    "InteractiveCADRetriever", 
    "PythonToJSONConverter"
]