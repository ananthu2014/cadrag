#!/usr/bin/env python3
"""
CAD-RAG: Multi-Modal Retrieval-Augmented Generation for CAD Design
Setup script for package installation
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README_CAD_RAG.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "CAD-RAG: Multi-Modal Retrieval-Augmented Generation for CAD Design"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="cad-rag",
    version="1.0.0",
    author="Claude Code",
    author_email="noreply@anthropic.com",
    description="Multi-Modal Retrieval-Augmented Generation for CAD Design",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/anthropics/cad-rag",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gui": [
            "tkinter",  # Usually comes with Python
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cad-rag-gui=cad_rag.cad_rag_gui:main",
            "cad-rag-convert=cad_rag.py2json_converter:main",
            "cad-rag-retrieve=cad_rag.interactive_cad_retrieval:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cad_rag": [
            "bethany_lib/*.py",
            "bethany_lib/*.txt",
        ],
    },
    keywords=[
        "cad", "design", "retrieval", "generation", "ai", "clip", "gemini",
        "multi-modal", "sketch", "text", "3d", "modeling"
    ],
    project_urls={
        "Bug Reports": "https://github.com/anthropics/cad-rag/issues",
        "Source": "https://github.com/anthropics/cad-rag",
        "Documentation": "https://github.com/anthropics/cad-rag/docs",
    },
)