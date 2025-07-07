"""
Main entry point for the RAG Publication Explorer.

This script provides a command-line interface to run different components
of the RAG system: data processing, embedding generation, and the Streamlit UI.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')


def run_data_processing():
    """Run the data processing pipeline."""
    print("🔄 Running data processing...")
    try:
        from data_processing.data_loader import PublicationDataLoader
        
        # Initialize data loader
        data_loader = PublicationDataLoader(chunk_size=1000, chunk_overlap=200)
        
        # Process publications file
        file_path = "data/raw/project_1_publications.json"
        chunks = data_loader.process_publications_file(file_path)
        
        print(f"✅ Data processing complete: {len(chunks)} chunks created")
        return chunks
        
    except Exception as e:
        print(f"❌ Error in data processing: {e}")
        return None


def run_embedding_generation():
    """Run the embedding generation pipeline."""
    print("🔄 Generating embeddings...")
    try:
        from data_processing.data_loader import PublicationDataLoader
        from embeddings.embedding_generator import EmbeddingGenerator
        
        # Load and process data
        data_loader = PublicationDataLoader()
        file_path = "data/raw/project_1_publications.json"
        chunks = data_loader.process_publications_file(file_path)
        
        if not chunks:
            print("❌ No chunks to process")
            return False
        
        # Generate embeddings
        embedding_generator = EmbeddingGenerator()
        embedding_generator.process_chunks_and_create_embeddings(chunks)
        
        # Save vector store
        save_path = "artifacts/vector_store"
        embedding_generator.save_vector_store(save_path)
        
        print(f"✅ Embeddings generated and saved to {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return False


def run_rag_test():
    """Test the RAG chain with sample questions."""
    print("🔄 Testing RAG chain...")
    try:
        from embeddings.embedding_generator import EmbeddingGenerator
        from rag.rag_chain import RAGChain
        
        # Load vector store
        embedding_generator = EmbeddingGenerator()
        vector_store_path = "artifacts/vector_store"
        
        if not Path(vector_store_path).exists():
            print(f"❌ Vector store not found at {vector_store_path}")
            print("Please run: python main.py --embeddings")
            return False
        
        embedding_generator.load_vector_store(vector_store_path)
        
        # Initialize RAG chain
        rag_chain = RAGChain(embedding_generator)
        
        # Test questions
        test_questions = [
            "What is RAG and how does it work?",
            "How can I add memory to RAG applications?"
        ]
        
        for question in test_questions:
            print(f"\n❓ Question: {question}")
            response = rag_chain.ask_question(question)
            print(f"🤖 Answer: {response['answer'][:200]}...")
            print(f"📚 Sources: {response['num_sources']}")
        
        print("✅ RAG chain test complete")
        return True
        
    except Exception as e:
        print(f"❌ Error testing RAG chain: {e}")
        return False


def run_streamlit_app():
    """Launch the Streamlit application."""
    print("🚀 Launching Streamlit application...")
    
    # Check prerequisites
    vector_store_path = Path("artifacts/vector_store")
    if not vector_store_path.exists():
        print("❌ Vector store not found!")
        print("Please run: python main.py --embeddings")
        return False
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    # Launch Streamlit
    import subprocess
    try:
        subprocess.run([
            "streamlit", "run", "src/ui/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        return True
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")
        return False


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="RAG Publication Explorer - AI-powered publication question answering"
    )
    
    parser.add_argument(
        "--data", 
        action="store_true", 
        help="Process publication data and create text chunks"
    )
    
    parser.add_argument(
        "--embeddings", 
        action="store_true", 
        help="Generate embeddings and create vector store"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Test the RAG chain with sample questions"
    )
    
    parser.add_argument(
        "--app", 
        action="store_true", 
        help="Launch the Streamlit web application"
    )
    
    parser.add_argument(
        "--setup", 
        action="store_true", 
        help="Run complete setup (data processing + embeddings)"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("="*60)
    print("🔍 RAG Publication Explorer")
    print("AI-powered Question Answering for Ready Tensor Publications")
    print("="*60)
    
    # Check if no arguments provided
    if not any(vars(args).values()):
        print("\nUsage Examples:")
        print("  python main.py --setup      # Run complete setup")
        print("  python main.py --data       # Process data only") 
        print("  python main.py --embeddings # Generate embeddings only")
        print("  python main.py --test       # Test RAG chain")
        print("  python main.py --app        # Launch web app")
        print("\nFor help: python main.py --help")
        return
    
    # Execute based on arguments
    if args.setup:
        print("🚀 Running complete setup...")
        chunks = run_data_processing()
        if chunks:
            run_embedding_generation()
    
    if args.data:
        run_data_processing()
    
    if args.embeddings:
        run_embedding_generation()
    
    if args.test:
        run_rag_test()
    
    if args.app:
        run_streamlit_app()


if __name__ == "__main__":
    main()
