# CAD-RAG: Multi-Modal Retrieval-Augmented Generation for CAD Design

A unified framework that extends the CSTBIR (Composite Sketch+Text Based Image Retrieval) system with AI-powered CAD model generation using Google's Gemini API.

## 🌟 Features

### Core CAD-RAG Capabilities
- **Multi-Modal Retrieval**: Search CAD models using sketches, text descriptions, or both (based on CSTBIR)
- **AI-Powered Generation**: Use Gemini API to generate modified CAD models based on retrieved examples
- **Unified Environment**: Single Python environment for all functionality (no complex multi-environment setup)
- **CAD Format Support**: Seamless conversion between Python CAD code and JSON representations
- **GUI Interface**: User-friendly Tkinter-based graphical interface
- **CLIP-Based Embeddings**: Advanced similarity search using pre-trained CLIP models

### Enhanced CSTBIR Integration
- **Original CSTBIR**: Complete implementation of "Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions"
- **Extended Pipeline**: RAG-based generation on top of CSTBIR retrieval
- **Bethany Integration**: Seamless CAD model conversion using OpenECAD Bethany libraries

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
          │         (Built on CSTBIR)                       │
          └─────────────────────┬───────────────────────────┘
                                │
    ┌───────────────────────────┼───────────────────────────┐
    │                           │                           │
    ▼                           ▼                           ▼
┌─────────┐              ┌─────────────┐            ┌─────────────┐
│ CLIP    │              │   Gemini    │            │   Bethany   │
│Retrieval│              │     API     │            │ CAD Library │
│ System  │              │ Generation  │            │ Conversion  │
│(CSTBIR) │              │             │            │             │
└─────────┘              └─────────────┘            └─────────────┘
     │                           │                           │
     │ Similar Models            │ Generated Code            │ JSON Output
     ▼                           ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Final CAD Model                             │
│              (Python Code + JSON Format)                       │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
CAD-RAG/
├── src/cad_rag/                    # Main CAD-RAG package
│   ├── __init__.py                 # Package initialization  
│   ├── cad_rag_pipeline.py         # Main RAG orchestrator
│   ├── cad_rag_gui.py             # GUI interface
│   ├── interactive_cad_retrieval.py # CLIP-based retrieval (CSTBIR)
│   ├── py2json_converter.py        # Python-to-JSON conversion
│   └── bethany_lib/               # Integrated CAD conversion library
│       ├── __init__.py
│       ├── extrude.py            # CAD sequence handling
│       ├── cad2code.py           # JSON-to-Python conversion
│       └── ...                   # Other CAD utilities
├── tests/                         # Test suite
│   ├── test_unified_env.py       # Environment testing
│   ├── test_bethany_only.py      # CAD conversion tests
│   └── test_pipeline_functionality.py
├── scripts/                       # Utility scripts
│   └── bethany_json2py.py        # Standalone JSON-to-Python converter
├── docs/                         # Documentation
├── examples/                     # Usage examples
├── data/                         # Model weights and datasets
├── src/clip/                     # Modified CLIP implementation
├── environment_cad_rag.yml       # Unified conda environment
├── config.yml                   # Training configuration
├── dataloader.py                # CSTBIR dataset handling
├── run-normal.py                # CSTBIR training scripts
├── run-trimodal.py              # CSTBIR trimodal training
├── run-triplet.py               # CSTBIR triplet loss training
├── utils.py                     # CSTBIR utilities
├── requirements.txt             # Dependencies
└── README.md                    # This file
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
   git clone <repository-url>
   cd CAD-RAG
   ```

2. **Create the unified environment**:
   ```bash
   conda env create -f environment_cad_rag.yml
   conda activate cad-rag
   ```

3. **Set up your Gemini API keys**:
   - Edit `src/cad_rag/cad_rag_pipeline.py`
   - Replace the API keys in the `self.api_keys` list with your own Google Gemini API keys

4. **Download the model weights** (for CSTBIR):
   - Place your trained model in `data/model_epoch_15.pt`
   - Or train your own model using the CSTBIR training scripts

### Usage

#### GUI Interface (Recommended)
```bash
python src/cad_rag/cad_rag_gui.py
```

#### Python API
```python
from cad_rag import CADRAGPipeline

# Initialize the pipeline
pipeline = CADRAGPipeline(
    database_dir="database_embeddings",
    model_path="data/model_epoch_15.pt"
)

# Retrieve similar models
results = pipeline.retrieve_models(
    text_query="cylindrical part with holes",
    sketch_path="path/to/sketch.png",  # Optional
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

## 🔧 Training (CSTBIR Component)

### Training Parameters
To check and update training, model and dataset parameters see [config.yml](config.yml)

### Dataset Preparation
- Download VG images from [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html)
- Download QuickDraw Sketches from [QuickDraw Dataset](https://github.com/googlecreativelab/quickdraw-dataset/)
- Download CSTBIR dataset from [Google Drive](https://drive.google.com/drive/folders/1UgAZc5rtbO0MQ37WHS4hGQhXlqMPT6Lg?usp=sharing)

Store the downloaded dataset in the `./data/` directory.

### Training Commands

```bash
# Standard contrastive training
CUDA_VISIBLE_DEVICES=0 python run-normal.py

# Trimodal training (image, sketch, text)
CUDA_VISIBLE_DEVICES=0 python run-trimodal.py

# Triplet loss training
CUDA_VISIBLE_DEVICES=0 python run-triplet.py
```

## 🧪 Testing

Run the test suite to verify your installation:

```bash
# Test unified environment setup
python tests/test_unified_env.py

# Test CAD conversion functionality
python tests/test_bethany_only.py

# Test complete pipeline
python tests/test_pipeline_functionality.py
```

## 🔧 Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: Specify GPU devices for CLIP inference
- `CAD_RAG_DATA_DIR`: Override default data directory path

### Config Files
- `config.yml`: CSTBIR training and model configuration
- `environment_cad_rag.yml`: Unified conda environment specification

### Gemini API Configuration
Edit the API keys in `src/cad_rag/cad_rag_pipeline.py`:
```python
self.api_keys = [
    "your-gemini-api-key-1",
    "your-gemini-api-key-2",
    # Add more keys for redundancy
]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `python -m pytest tests/`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you find this work useful for your research, please consider citing both the original CSTBIR paper and this CAD-RAG extension:

```bibtex
@InProceedings{cstbir2024aaai,
    author    = {Gatti, Prajwal and Parikh, Kshitij Gopal and Paul, Dhriti Prasanna and Gupta, Manish and Mishra, Anand},
    title     = {Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions},
    booktitle = {AAAI},
    year      = {2024},
}

@software{cadrag2024,
    title     = {CAD-RAG: Multi-Modal Retrieval-Augmented Generation for CAD Design},
    author    = {Claude Code},
    year      = {2024},
    note      = {Built with Claude Code}
}
```

## 🙏 Acknowledgments

- **Original CSTBIR**: Gatti et al., "Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions"
- **OpenECAD Bethany Project**: CAD conversion libraries
- **OpenAI CLIP**: Multi-modal embeddings ([CLIP Repository](https://github.com/openai/CLIP/))
- **Google Gemini**: AI-powered code generation
- **PyTorch**: Deep learning framework

## 📞 Support

For questions, issues, or contributions:
- 🐛 [Report bugs](https://github.com/your-repo/issues)
- 💡 [Request features](https://github.com/your-repo/discussions)

---

