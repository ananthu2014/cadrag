# CAD-RAG: Multi-Modal Retrieval-Augmented Generation for CAD Design

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-red.svg)](https://pytorch.org/)

A unified framework that combines sketch-based and text-based retrieval with AI-powered CAD model generation using Google's Gemini API.

## 🌟 Features

- **Multi-Modal Retrieval**: Search CAD models using sketches, text descriptions, or both
- **AI-Powered Generation**: Use Gemini API to generate modified CAD models based on retrieved examples
- **Unified Environment**: Single Python environment for all functionality (no complex multi-environment setup)
- **CAD Format Support**: Seamless conversion between Python CAD code and JSON representations
- **GUI Interface**: User-friendly Tkinter-based graphical interface
- **CLIP-Based Embeddings**: Advanced similarity search using pre-trained CLIP models

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │    │   Sketch Input   │    │  Text Query     │
│   (GUI/CLI)     │    │   (Optional)     │    │  (Required)     │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
          ┌─────────────────────────────────────────────────┐
          │           CAD-RAG Pipeline                      │
          └─────────────────────┬───────────────────────────┘
                                │
    ┌───────────────────────────┼───────────────────────────┐
    │                           │                           │
    ▼                           ▼                           ▼
┌─────────┐              ┌─────────────┐            ┌─────────────┐
│ CLIP    │              │   Gemini    │            │   Bethany   │
│Retrieval│              │     API     │            │ CAD Library │
│ System  │              │ Generation  │            │ Conversion  │
└─────────┘              └─────────────┘            └─────────────┘
     │                           │                           │
     │ Similar Models            │ Generated Code            │ JSON Output
     ▼                           ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Final CAD Model                             │
│              (Python Code + JSON Format)                       │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **CUDA-compatible GPU** (recommended for CLIP model inference)
- **Conda** or **Mamba** package manager
- **Python 3.10+**
- **Google Gemini API Key**

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/cadrag.git
   cd cadrag
   ```

2. **Create the unified environment**:
   ```bash
   conda env create -f environment.yml
   conda activate cadrag
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

4. **Set up your Gemini API keys**:
   - Edit `src/cadrag/config.py` or set environment variables
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Usage

#### GUI Interface (Recommended)
```bash
python -m cadrag.gui
```

#### Python API
```python
from cadrag import CADRAGPipeline

# Initialize the pipeline
pipeline = CADRAGPipeline(
    database_dir="database_embeddings",
    model_path="data/model_epoch_15.pt"
)

# Retrieve similar models
results = pipeline.retrieve_models(
    text_query="cylindrical part with holes",
    top_k=10
)

# Generate modified CAD model
generated_code = pipeline.generate_cad_sequence(
    user_query="make it larger and add more holes",
    selected_model=results[0],
    instructions="increase diameter by 50%"
)

# Convert to JSON format
json_output = pipeline.convert_to_json(generated_code)
```

## 📁 Project Structure

```
cadrag/
├── src/cadrag/                     # Main package
│   ├── __init__.py                 # Package initialization
│   ├── pipeline.py                 # Main orchestrator
│   ├── gui.py                     # GUI interface
│   ├── retrieval.py               # CLIP-based retrieval
│   ├── conversion.py              # Format conversion utilities
│   ├── config.py                  # Configuration management
│   └── bethany/                   # CAD conversion library
├── tests/                         # Test suite
├── scripts/                       # Utility scripts
├── docs/                         # Documentation
├── examples/                     # Usage examples
├── data/                         # Model weights and datasets (gitignored)
├── environment.yml               # Conda environment specification
├── requirements.txt              # Pip requirements
├── setup.py                      # Package setup
├── LICENSE                       # MIT License
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🔧 Configuration

The system can be configured via:

1. **Environment variables**:
   ```bash
   export GEMINI_API_KEY="your-api-key"
   export CADRAG_DATA_DIR="/path/to/data"
   export CUDA_VISIBLE_DEVICES="0"
   ```

2. **Configuration file** (`src/cadrag/config.py`):
   ```python
   GEMINI_API_KEYS = ["key1", "key2", "key3"]  # Multiple keys for failover
   MODEL_PATH = "data/model_epoch_15.pt"
   DATABASE_DIR = "database_embeddings"
   ```

## 🧪 Testing

Run the test suite to verify your installation:

```bash
# Test environment setup
python -m pytest tests/test_environment.py

# Test CAD conversion functionality
python -m pytest tests/test_conversion.py

# Test complete pipeline
python -m pytest tests/test_pipeline.py

# Run all tests
python -m pytest tests/ -v
```

## 📚 Examples

See the [examples/](examples/) directory for:
- Basic retrieval and generation
- Batch processing scripts
- Custom model training
- API integration examples

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `python -m pytest tests/`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenECAD Bethany Project**: CAD conversion libraries
- **OpenAI CLIP**: Multi-modal embeddings
- **Google Gemini**: AI-powered code generation
- **PyTorch**: Deep learning framework

## 📞 Support

For questions, issues, or contributions:
- 🐛 [Report bugs](https://github.com/yourusername/cadrag/issues)
- 💡 [Request features](https://github.com/yourusername/cadrag/discussions)
- 📧 Contact: your-email@domain.com

## 📈 Roadmap

- [ ] Support for additional CAD formats (STEP, STL)
- [ ] Integration with popular CAD software
- [ ] Advanced sketch recognition capabilities
- [ ] Real-time collaborative design features
- [ ] Cloud deployment options

---

**⚡ Built with Claude Code** - Making CAD design accessible through AI