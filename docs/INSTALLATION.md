# CAD-RAG Installation Guide

This guide provides detailed installation instructions for the CAD-RAG framework.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows 10+
- **Python**: 3.10 or higher
- **RAM**: 8GB minimum, 16GB+ recommended
- **Storage**: 10GB free space minimum
- **GPU**: NVIDIA GPU with CUDA 11.6+ (recommended for optimal performance)

### Required Software
1. **Conda** or **Mamba** package manager
   - [Anaconda](https://docs.anaconda.com/anaconda/install/) (recommended)
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lightweight)
   - [Mamba](https://mamba.readthedocs.io/) (faster alternative)

2. **NVIDIA CUDA Toolkit** (for GPU acceleration)
   - CUDA 11.6 or compatible version
   - Verify with: `nvidia-smi`

3. **Google Gemini API Key**
   - Sign up at [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Generate API key for Gemini models

## 🚀 Quick Installation

### Option 1: Automated Installation (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd CAD-RAG

# Run the automated installation script
./scripts/install.sh

# Run with tests (optional)
./scripts/install.sh --test
```

### Option 2: Manual Installation

```bash
# Clone repository
git clone <repository-url>
cd CAD-RAG

# Create conda environment
conda env create -f environment_cad_rag.yml

# Activate environment
conda activate cad-rag

# Install package in development mode
pip install -e .
```

## 🔧 Configuration

### 1. Gemini API Keys

Edit `src/cad_rag/cad_rag_pipeline.py` and replace the placeholder API keys:

```python
self.api_keys = [
    "your-actual-gemini-api-key-1",
    "your-actual-gemini-api-key-2",
    # Add more keys for redundancy
]
```

### 2. Model Weights

Download or train the CSTBIR model weights:

- **Option A**: Use pre-trained weights
  - Place model file at `data/model_epoch_15.pt`
  
- **Option B**: Train your own model
  ```bash
  # Download CSTBIR dataset (see original README)
  # Train model
  CUDA_VISIBLE_DEVICES=0 python run-normal.py
  ```

### 3. Database Embeddings

Generate or download database embeddings:

```bash
# Generate embeddings from your CAD dataset
python encode_cad_database.py --input_dir /path/to/cad/models --output_dir database_embeddings
```

## 🧪 Verification

### Test Installation

```bash
# Activate environment
conda activate cad-rag

# Run comprehensive tests
python tests/test_unified_env.py
python tests/test_bethany_only.py
python tests/test_pipeline_functionality.py

# Test GUI (optional)
python src/cad_rag/cad_rag_gui.py
```

### Expected Output

✅ **Successful installation should show**:
- All environment tests passing
- CAD conversion working correctly
- Pipeline initialization successful
- GUI launching without errors

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA/PyTorch Issues
```bash
# Error: "CUDA out of memory" or "No CUDA device"
# Solution: 
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
# Or install CPU-only version
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### 2. Import Errors
```bash
# Error: "ModuleNotFoundError: No module named 'cad_rag'"
# Solution:
conda activate cad-rag
pip install -e .
```

#### 3. Gemini API Issues
```bash
# Error: "API key not valid"
# Solution: 
# 1. Check API key format
# 2. Verify API key is active at https://makersuite.google.com/
# 3. Check internet connection
```

#### 4. Environment Conflicts
```bash
# Error: Environment creation fails
# Solution:
conda env remove -n cad-rag  # Remove existing
conda clean --all            # Clean conda cache
conda env create -f environment_cad_rag.yml  # Recreate
```

### Performance Issues

#### GPU Not Detected
```bash
# Check GPU status
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall CUDA-compatible PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

#### Slow Inference
- **Reduce batch size** in config.yml
- **Use smaller CLIP model** variant
- **Enable mixed precision** training
- **Use CPU for smaller datasets**

## 📁 Directory Structure After Installation

```
CAD-RAG/
├── src/cad_rag/              # Main package (installed)
├── data/
│   ├── model_epoch_15.pt     # Trained model weights
│   └── dataset/              # Training dataset
├── database_embeddings/      # Pre-computed embeddings
├── tests/                    # Test suite
├── examples/                 # Usage examples
├── docs/                     # Documentation
└── environment_cad_rag.yml   # Environment specification
```

## 🚦 Next Steps

After successful installation:

1. **Test Basic Functionality**:
   ```bash
   python examples/basic_usage.py
   ```

2. **Launch GUI**:
   ```bash
   python src/cad_rag/cad_rag_gui.py
   ```

3. **Run Your First Query**:
   - Open GUI
   - Enter text query: "cylindrical part with holes"  
   - Click "Retrieve Models"
   - Select a model and click "Generate CAD"

4. **Explore Advanced Features**:
   - Multi-modal queries (sketch + text)
   - Custom generation instructions
   - Python-to-JSON conversion
   - Batch processing

## 📞 Support

If you encounter issues:

1. **Check this troubleshooting guide**
2. **Run diagnostic tests**: `python tests/test_unified_env.py`
3. **Check GPU status**: `nvidia-smi`
4. **Verify environment**: `conda list`
5. **Report issues** with error logs and system information

For additional help:
- 📖 [API Documentation](API.md)
- 💡 [Usage Examples](../examples/)
- 🐛 [Report Issues](https://github.com/your-repo/issues)

---

**Installation complete! 🎉 Ready to explore CAD-RAG!**