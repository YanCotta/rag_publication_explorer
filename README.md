# RAG Publication Explorer

A Python-based Retrieval-Augmented Generation (RAG) system for exploring and querying academic publications and documents. This project implements a complete RAG pipeline with document processing, embedding generation, vector storage, and response generation capabilities.

## 🏗️ Project Structure

This project follows the **Ready Tensor Repository Assessment Framework** at the 'Essential' level, providing a well-organized structure for machine learning and AI projects.

```
rag_publication_explorer/
├── 📁 src/                          # Source code
│   └── rag_agent/                   # Main RAG agent package
│       ├── core/                    # Core pipeline components
│       │   └── rag_pipeline.py      # Main RAG pipeline orchestrator
│       ├── data_processing/         # Document processing modules
│       │   └── document_processor.py # Document parsing and chunking
│       ├── embeddings/              # Embedding management
│       │   └── embedding_manager.py  # Text embedding generation
│       ├── retrieval/               # Document retrieval
│       │   └── vector_store.py      # Vector storage and similarity search
│       ├── generation/              # Response generation
│       │   └── llm_interface.py     # LLM interaction interface
│       ├── utils/                   # Utility functions
│       └── config/                  # Configuration management
│           └── settings.py          # Configuration loading and defaults
├── 📁 data/                         # Data directories
│   ├── raw/                         # Raw, unprocessed documents
│   ├── processed/                   # Cleaned and processed data
│   ├── embeddings/                  # Vector embeddings and indices
│   └── external/                    # External datasets and references
├── 📁 notebooks/                    # Jupyter notebooks
│   ├── exploratory/                 # Data exploration notebooks
│   ├── experiments/                 # Model experiments
│   └── demos/                       # Demonstration notebooks
│       └── rag_demo.ipynb          # Main system demonstration
├── 📁 tests/                        # Test files
│   ├── unit/                        # Unit tests
│   │   └── test_rag_pipeline.py     # RAG pipeline tests
│   ├── integration/                 # Integration tests
│   └── fixtures/                    # Test data and fixtures
├── 📁 docs/                         # Documentation
│   ├── api/                         # API documentation
│   └── user_guide/                  # User guides and tutorials
├── 📁 configs/                      # Configuration files
│   └── default.yaml                 # Default system configuration
├── 📁 requirements/                 # Dependency management
│   ├── requirements.txt             # Core dependencies
│   └── requirements-dev.txt         # Development dependencies
├── 📁 scripts/                      # Utility scripts
├── 📁 models/                       # Model storage
├── 📁 artifacts/                    # Generated artifacts
├── 📁 logs/                         # Application logs
├── main.py                          # Main application entry point
├── setup_project_structure.py      # Project setup script
├── .gitignore                       # Git ignore rules (Python optimized)
└── README.md                        # This file
```

## 🚀 Quick Start

### Option A: Automated Setup (Recommended)

Run the automated setup script that handles everything:

```bash
# Clone and navigate to the project
git clone <your-repo-url>
cd rag_publication_explorer

# Run automated setup
python3 setup.py
```

This will:
- Create a virtual environment
- Install dependencies
- Create configuration files
- Set up environment variables

### Option B: Manual Setup

#### 1. Virtual Environment Setup

**Why use virtual environments?** They isolate project dependencies, prevent conflicts, ensure reproducibility, and keep your system clean.

**Create and activate virtual environment:**

```bash
# Create virtual environment
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

**Verify activation:** Your prompt should show `(venv)`

#### 2. Install Dependencies

Choose your installation option:

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Option A: Minimal installation (quick start)
pip install -r requirements/requirements-minimal.txt

# Option B: Full installation (recommended)
pip install -r requirements/requirements.txt

# Option C: Development installation (for contributors)
pip install -r requirements/requirements-dev.txt
```

#### 3. Environment Variables

Create your `.env` file:

```bash
# Copy template
cp .env.template .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

#### 4. Configuration

Customize settings if needed:

```bash
cp configs/default.yaml configs/local.yaml
# Edit configs/local.yaml with your specific settings
```

Key configuration options:
- **Document Processing**: Chunk size, overlap, supported formats
- **Embeddings**: Model selection, device configuration
- **Vector Store**: Storage type, index paths
- **LLM**: Model selection, API keys, generation parameters

### 4. Add Documents

Place your documents in the `data/raw/` directory. Supported formats include:
- Plain text (`.txt`)
- Markdown (`.md`)
- PDF (`.pdf`) - *requires implementation*
- Word documents (`.docx`) - *requires implementation*

### 5. Run the System

```bash
# Basic usage
python main.py --query "What are the main findings in the research?"

# With custom configuration
python main.py --config configs/local.yaml --query "Your question here" --top-k 10
```

## 📊 Usage Examples

### Command Line Interface

```bash
# Basic query
python main.py --query "What is retrieval-augmented generation?"

# Advanced options
python main.py \
    --config configs/custom.yaml \
    --query "How does RAG improve language model performance?" \
    --top-k 5
```

### Programmatic Usage

```python
from src.rag_agent.core.rag_pipeline import RAGPipeline
from src.rag_agent.config.settings import load_config

# Load configuration
config = load_config('configs/default.yaml')

# Initialize pipeline
rag_pipeline = RAGPipeline(config)

# Ingest documents
rag_pipeline.ingest_documents(['data/raw/document.txt'])

# Query the system
result = rag_pipeline.query("Your question here", top_k=5)
print(result['response'])
```

### Jupyter Notebooks

Explore the system interactively using the provided notebooks:

```bash
jupyter notebook notebooks/demos/rag_demo.ipynb
```

## 🔧 Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src/rag_agent --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## 🏗️ Architecture

The RAG system consists of four main components:

1. **Document Processor**: Handles parsing and chunking of various document formats
2. **Embedding Manager**: Generates and manages text embeddings using pre-trained models
3. **Vector Store**: Stores document embeddings and performs similarity search
4. **LLM Interface**: Handles interaction with language models for response generation

### Key Features

- **Modular Design**: Each component can be independently configured and replaced
- **Extensible**: Easy to add support for new document formats and models
- **Configurable**: YAML-based configuration for different environments
- **Scalable**: Designed to handle large document collections
- **Observable**: Comprehensive logging and monitoring support

## 📝 Configuration

The system uses YAML configuration files. Key sections include:

```yaml
document_processing:
  chunk_size: 1000
  chunk_overlap: 200
  supported_formats: ['.pdf', '.txt', '.md', '.docx']

embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  batch_size: 32

vector_store:
  type: "faiss"
  index_path: "./data/embeddings/index.faiss"
  dimension: 384

llm:
  model_name: "gpt-3.5-turbo"
  max_tokens: 1000
  temperature: 0.7
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the project structure
4. Add tests for new functionality
5. Ensure code quality checks pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🆘 Support

For questions, issues, or contributions:

1. Check the documentation in the `docs/` directory
2. Look for existing issues in the GitHub repository
3. Create a new issue with detailed information about your problem
4. Review the demo notebook for usage examples

## 🔮 Future Enhancements

- [ ] Support for additional document formats (PDF, DOCX, HTML)
- [ ] Web interface using Streamlit or FastAPI
- [ ] Advanced retrieval strategies (hybrid search, re-ranking)
- [ ] Multi-modal support (images, tables)
- [ ] Evaluation metrics and benchmarking
- [ ] Docker containerization
- [ ] Cloud deployment configurations
A high-quality RAG-powered assistant that scores highly on the Ready Tensor evaluation rubrics.
