# Ready Tensor Publication Explorer

A powerful RAG (Retrieval-Augmented Generation) powered Q&A assistant designed to help users quickly find and understand information from Ready Tensor's AI/ML publication collection. This intelligent system combines advanced document processing, semantic search, and AI-generated responses to provide accurate, context-aware answers about artificial intelligence and machine learning topics.

## Overview

The Ready Tensor Publication Explorer is an innovative question-answering system that leverages state-of-the-art AI technology to make AI/ML knowledge more accessible. Built specifically for Ready Tensor's publication database, this tool addresses the challenge of quickly finding relevant information across a large collection of technical documents.

**What it does:**

- Processes and indexes Ready Tensor's AI/ML publications using advanced text chunking techniques
- Converts text into high-dimensional vector embeddings for semantic similarity search
- Uses retrieval-augmented generation to provide accurate, context-aware answers
- Presents information through an intuitive web interface that requires no technical expertise

**Why it's useful:**

- **Time-saving**: Instantly find relevant information without manually searching through dozens of publications
- **Intelligent**: Goes beyond keyword matching to understand the semantic meaning of your questions
- **Accurate**: Provides answers based on actual publication content with source attribution
- **Accessible**: Simple web interface makes AI/ML knowledge accessible to both technical and non-technical users
- **Comprehensive**: Covers the full breadth of Ready Tensor's publication library in one unified system

This project demonstrates practical application of modern AI techniques including vector databases, semantic search, and large language models, making it an excellent example of production-ready RAG implementation.

## Features

- ✅ **Answers questions using a knowledge base of Ready Tensor publications**
- 🔍 **Uses Retrieval-Augmented Generation (RAG) for accurate, context-aware responses**
- 🚀 **Built with LangChain, OpenAI, and FAISS**
- 💻 **Simple and interactive UI powered by Streamlit**
- 📚 **Source Attribution**: Every answer includes references to the specific publications used
- ⚡ **Fast Performance**: Optimized caching and vector search provide near-instantaneous responses
- 🔄 **Conversation Memory**: Maintains conversation history for context-aware follow-up questions
- 📊 **Comprehensive Coverage**: Processes and understands content from 35+ Ready Tensor publications
- 🛡️ **Robust Error Handling**: Gracefully handles edge cases and provides helpful error messages
- 📝 **Detailed Logging**: Complete audit trail of queries and responses for debugging and analysis

## Installation

Follow these step-by-step instructions to set up the Ready Tensor Publication Explorer on your local machine:

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (get one at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys))

### Step 1: Clone the Repository

```bash
git clone https://github.com/YanCotta/rag_publication_explorer.git
cd rag_publication_explorer
```

### Step 2: Create and Activate a Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

### Step 4: Create a .env File and Add Your OPENAI_API_KEY

```bash
# Create environment file from template
cp .env.template .env

# Edit the .env file and add your OpenAI API key
# Replace 'YOUR_API_KEY_HERE' with your actual OpenAI API key
OPENAI_API_KEY='your-actual-openai-api-key-here'
```

**Important**: Never commit your actual API key to version control. The .env file is already included in .gitignore for security.

### Step 5: Verify Installation

```bash
# Test the installation
python -c "import streamlit, langchain, openai, faiss; print('✅ Installation successful!')"
```

## Usage

### Starting the Application

Once installation is complete, start the Ready Tensor Publication Explorer by running:

```bash
streamlit run app.py
```

The application will automatically:

1. Load and process the Ready Tensor publication database
2. Generate embeddings and create the vector search index
3. Launch a web browser with the interactive interface
4. Display at `http://localhost:8501`

### Interacting with the Interface

The web interface provides an intuitive way to explore Ready Tensor's AI/ML knowledge base:

1. **Ask Questions**: Type your question in the text input box
2. **Get AI Answers**: Receive comprehensive answers based on the publication content
3. **View Sources**: See which specific publications informed each answer
4. **Explore Content**: Use the expandable sections to read relevant excerpts
5. **Continue Conversations**: Ask follow-up questions that build on previous responses

### Sample Questions

Here are some example questions you can ask to explore the system's capabilities:

**General Questions:**

- "What publications discuss RAG?"
- "Summarize the article on PEP8 style guides"
- "What are the main benefits of retrieval-augmented generation?"

**Technical Questions:**

- "How do I implement effective document chunking strategies?"
- "What embedding models work best for RAG applications?"
- "How can I add memory to RAG applications for better conversations?"

**Ready Tensor Specific:**

- "What does Ready Tensor recommend for machine learning project structure?"
- "What are the key evaluation metrics discussed in the publications?"
- "What coding standards and best practices are mentioned?"

### Advanced Usage

**Command Line Interface:**

The project also includes a command-line interface for advanced users:

```bash
# Process data only
python main.py --data

# Generate embeddings
python main.py --embeddings

# Test the RAG system
python main.py --test

# Complete setup
python main.py --setup
```

## Technical Architecture

### System Components

- **Data Processing**: Intelligent text chunking with overlap for context preservation
- **Embedding Generation**: OpenAI's text-embedding-ada-002 for high-quality vector representations
- **Vector Storage**: FAISS for efficient similarity search and retrieval
- **Language Model**: GPT-3.5-turbo for natural language generation
- **Web Framework**: Streamlit for responsive, interactive user interface

### Performance Specifications

- **Documents Processed**: 35 Ready Tensor publications
- **Text Chunks**: 1,200+ optimally sized chunks with semantic boundaries
- **Embedding Dimensions**: 1,536-dimensional vectors for precise semantic matching
- **Response Time**: Sub-second retrieval with 2-5 second generation time
- **Accuracy**: Context-aware responses with full source attribution

## Troubleshooting

### Common Issues

**"OpenAI API key not found"**

- Verify your .env file contains the correct API key
- Ensure the .env file is in the project root directory
- Check that your API key is valid and has sufficient credits

**"Data file not found"**

- Ensure `project_1_publications.json` exists in the `data/` directory
- Verify file permissions and accessibility

**"Import errors"**

- Activate your virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

**"Streamlit won't start"**

- Check if port 8501 is available
- Try using a different port: `streamlit run app.py --server.port 8502`

### Getting Help

- Check the application logs in `app.log` for detailed error information
- Verify all dependencies are installed correctly
- Ensure your Python version is 3.8 or higher

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions to improve the Ready Tensor Publication Explorer! Please feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

- Ready Tensor for providing the comprehensive AI/ML publication dataset
- OpenAI for the powerful embedding and language models
- The LangChain community for excellent RAG development tools
- Streamlit for the intuitive web application framework

---

**Ready Tensor Publication Explorer** - Making AI/ML knowledge accessible through intelligent question answering.
