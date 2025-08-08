#!/bin/bash
# CAD-RAG Installation Script

set -e  # Exit on any error

echo "🚀 CAD-RAG Installation Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if conda is installed
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        print_info "Please install Anaconda or Miniconda first:"
        print_info "https://docs.anaconda.com/anaconda/install/"
        exit 1
    fi
    print_status "Conda found: $(conda --version)"
}

# Check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        print_warning "No NVIDIA GPU detected. CPU-only mode will be used."
        print_warning "Performance may be significantly slower for CLIP inference."
    fi
}

# Create conda environment
create_environment() {
    print_info "Creating conda environment 'cad-rag'..."
    
    if conda env list | grep -q "^cad-rag "; then
        print_warning "Environment 'cad-rag' already exists."
        read -p "Do you want to remove and recreate it? (y/N): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing environment..."
            conda env remove -n cad-rag
        else
            print_info "Skipping environment creation."
            return
        fi
    fi
    
    print_info "Creating new environment from environment_cad_rag.yml..."
    conda env create -f environment_cad_rag.yml
    print_status "Environment 'cad-rag' created successfully"
}

# Install package in development mode
install_package() {
    print_info "Installing CAD-RAG package in development mode..."
    
    # Activate environment and install
    eval "$(conda shell.bash hook)"
    conda activate cad-rag
    
    pip install -e .
    print_status "Package installed successfully"
}

# Setup Gemini API keys
setup_api_keys() {
    print_info "Setting up Gemini API keys..."
    print_warning "You need to manually add your Gemini API keys."
    print_info "Edit src/cad_rag/cad_rag_pipeline.py and replace the API keys in the self.api_keys list"
    print_info "Get your API keys from: https://makersuite.google.com/app/apikey"
}

# Run tests
run_tests() {
    print_info "Running installation tests..."
    
    eval "$(conda shell.bash hook)"
    conda activate cad-rag
    
    # Test environment
    print_info "Testing unified environment..."
    python tests/test_unified_env.py
    
    # Test Bethany conversion
    print_info "Testing CAD conversion..."
    python tests/test_bethany_only.py
    
    print_status "All tests completed"
}

# Main installation flow
main() {
    echo "Starting CAD-RAG installation..."
    echo
    
    # Check prerequisites
    print_info "Checking prerequisites..."
    check_conda
    check_cuda
    echo
    
    # Create environment
    create_environment
    echo
    
    # Install package
    install_package
    echo
    
    # Setup API keys
    setup_api_keys
    echo
    
    # Run tests
    if [[ "${1:-}" == "--test" ]]; then
        run_tests
        echo
    fi
    
    # Installation complete
    echo "🎉 Installation completed successfully!"
    echo
    print_info "To get started:"
    print_info "1. conda activate cad-rag"
    print_info "2. Add your Gemini API keys to src/cad_rag/cad_rag_pipeline.py"
    print_info "3. python src/cad_rag/cad_rag_gui.py"
    echo
    print_info "For more information, see README_CAD_RAG.md"
}

# Run main function
main "$@"