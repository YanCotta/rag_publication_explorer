"""
Streamlit application for the RAG Publication Explorer.

This module provides a user-friendly interface for interacting with the RAG assistant,
allowing users to ask questions about AI/ML publications and receive intelligent answers.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import traceback

# Add src to path for imports
sys.path.append('src')

# Configure page
st.set_page_config(
    page_title="RAG Publication Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .error-box {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_chain():
    """
    Load the RAG chain and cache it for the session.
    This function is cached to avoid reloading the model on every interaction.
    """
    try:
        # Import required modules
        from embeddings.embedding_generator import EmbeddingGenerator
        from rag.rag_chain import RAGChain
        
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator()
        
        # Load vector store
        vector_store_path = "artifacts/vector_store"
        if not Path(vector_store_path).exists():
            st.error(f"Vector store not found at {vector_store_path}. Please run the embedding generation script first.")
            return None
            
        embedding_generator.load_vector_store(vector_store_path)
        
        # Initialize RAG chain
        rag_chain = RAGChain(embedding_generator)
        
        return rag_chain
        
    except Exception as e:
        st.error(f"Error loading RAG chain: {str(e)}")
        st.error("Please make sure you have:")
        st.error("1. Set your OPENAI_API_KEY environment variable")
        st.error("2. Generated embeddings by running the embedding script")
        st.error("3. Installed all required dependencies")
        return None


def display_sources(source_documents):
    """
    Display source documents in an organized way.
    
    Args:
        source_documents (list): List of source document metadata
    """
    if not source_documents:
        return
    
    st.subheader("📚 Sources")
    
    for i, source in enumerate(source_documents, 1):
        with st.expander(f"Source {i}: {source['title']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Author:** {source['username']}")
                st.write(f"**Preview:** {source['content_preview']}")
                
            with col2:
                similarity_score = source.get('similarity_score', 0.0)
                st.metric("Similarity", f"{similarity_score:.3f}")


def main():
    """
    Main Streamlit application function.
    """
    # Header
    st.markdown('<h1 class="main-header">🔍 RAG Publication Explorer</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the RAG Publication Explorer! This intelligent assistant can answer questions about 
    AI, machine learning, and data science based on Ready Tensor publications.
    
    **How to use:**
    1. Type your question in the text box below
    2. Click 'Ask Question' or press Enter
    3. Get intelligent answers backed by relevant sources
    """)
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This RAG (Retrieval-Augmented Generation) system combines:
        - **Document Retrieval**: Finds relevant content from publications
        - **AI Generation**: Uses OpenAI's GPT to generate comprehensive answers
        - **Source Attribution**: Shows which publications informed the answer
        """)
        
        st.header("🎯 Example Questions")
        example_questions = [
            "What is RAG and how does it work?",
            "How can I add memory to RAG applications?", 
            "What are the best practices for computer vision?",
            "How do auto-encoders work for image compression?",
            "What is CLIP and how is it used?",
            "Explain UV as a Python package manager"
        ]
        
        for i, question in enumerate(example_questions):
            if st.button(f"📝 {question}", key=f"example_{i}"):
                st.session_state.question_input = question
        
        st.header("🔧 System Status")
        
        # Check if vector store exists
        vector_store_path = Path("artifacts/vector_store")
        if vector_store_path.exists():
            st.success("✅ Vector store loaded")
        else:
            st.error("❌ Vector store not found")
        
        # Check API key
        if os.getenv('OPENAI_API_KEY'):
            st.success("✅ OpenAI API key set")
        else:
            st.error("❌ OpenAI API key not set")
    
    # Load RAG chain
    rag_chain = load_rag_chain()
    
    if rag_chain is None:
        st.stop()
    
    # Main interface
    st.header("💬 Ask Your Question")
    
    # Question input
    question = st.text_area(
        "Enter your question about AI, ML, or data science:",
        height=100,
        key="question_input",
        placeholder="e.g., How does contrastive learning work in CLIP?"
    )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit_button = st.button("🚀 Ask Question", type="primary", use_container_width=True)
    
    # Process question
    if submit_button and question.strip():
        with st.spinner("🔍 Searching for relevant information and generating answer..."):
            try:
                # Get response from RAG chain
                response = rag_chain.ask_question(question.strip())
                
                # Display question
                st.markdown('<div class="question-box">', unsafe_allow_html=True)
                st.write(f"**❓ Question:** {response['question']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display answer
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.write("**🤖 Answer:**")
                st.write(response['answer'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display sources
                if response['source_documents']:
                    display_sources(response['source_documents'])
                else:
                    st.warning("No relevant sources found for this question.")
                
                # Display metadata
                with st.expander("📊 Response Metadata", expanded=False):
                    st.write(f"**Sources used:** {response['num_sources']}")
                    if response['source_documents']:
                        st.write("**Top similarity scores:**")
                        for i, source in enumerate(response['source_documents'][:3], 1):
                            score = source.get('similarity_score', 0.0)
                            st.write(f"  {i}. {score:.3f} - {source['title']}")
                
            except Exception as e:
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.write("**❌ Error:**")
                st.write(f"An error occurred while processing your question: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show detailed error in expander for debugging
                with st.expander("🔧 Debug Information", expanded=False):
                    st.code(traceback.format_exc())
    
    elif submit_button and not question.strip():
        st.warning("⚠️ Please enter a question before submitting.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        RAG Publication Explorer | Powered by OpenAI & Ready Tensor Publications
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
