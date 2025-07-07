"""
Ready Tensor Publication Explorer

A single-file Streamlit application that implements a complete RAG (Retrieval-Augmented Generation)
system for querying AI/ML publications. This app loads publication data, creates embeddings,
builds a vector store, and provides an interactive interface for question answering.
"""

import streamlit as st
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="Ready Tensor Publication Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .source-info {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_process_data():
    """
    Load and process publication data from JSON file.
    This function is cached to ensure data is loaded only once.
    
    Returns:
        List[Document]: List of LangChain Document objects
    """
    try:
        # Check if data file exists
        data_file = Path("./data/project_1_publications.json")
        if not data_file.exists():
            # Try alternative path
            data_file = Path("./data/raw/project_1_publications.json")
            
        if not data_file.exists():
            st.error("❌ Publication data file not found! Please ensure project_1_publications.json is in the data/ folder.")
            return []
        
        logger.info(f"Loading publications data from: {data_file}")
        
        # Load JSON data
        with open(data_file, 'r', encoding='utf-8') as file:
            publications = json.load(file)
        
        if not isinstance(publications, list):
            st.error("❌ Invalid data format. Expected a list of publications.")
            return []
        
        # Process publications into documents
        documents = []
        for i, publication in enumerate(publications):
            try:
                title = publication.get('title', '').strip()
                description = publication.get('publication_description', '').strip()
                pub_id = publication.get('id', f'publication_{i}')
                username = publication.get('username', 'unknown')
                
                if not title and not description:
                    continue
                
                # Combine title and description
                combined_text = f"Title: {title}\n\nContent: {description}"
                
                # Create LangChain Document
                doc = Document(
                    page_content=combined_text,
                    metadata={
                        'id': pub_id,
                        'title': title,
                        'username': username,
                        'source': 'Ready Tensor Publications'
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Error processing publication {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(documents)} publications")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunked_documents = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunked_documents)} text chunks")
        
        return chunked_documents
        
    except Exception as e:
        logger.error(f"Error loading and processing data: {e}")
        st.error(f"❌ Error loading data: {e}")
        return []


@st.cache_resource
def create_vector_store(documents):
    """
    Create FAISS vector store from documents using OpenAI embeddings.
    This function is cached to prevent re-creating the vector store on every rerun.
    
    Args:
        documents: List of LangChain Document objects
        
    Returns:
        FAISS vector store or None if error
    """
    try:
        if not documents:
            st.error("❌ No documents available for embedding")
            return None
        
        # Check for OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("❌ OpenAI API key not found! Please set OPENAI_API_KEY in your .env file.")
            return None
        
        logger.info("Creating embeddings and vector store...")
        
        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=api_key
        )
        
        # Create FAISS vector store
        with st.spinner("🔄 Generating embeddings and creating vector store..."):
            vector_store = FAISS.from_documents(documents, embeddings)
        
        logger.info(f"Successfully created vector store with {len(documents)} documents")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        st.error(f"❌ Error creating vector store: {e}")
        return None


def create_rag_chain(vector_store):
    """
    Create and configure the RAG chain for question answering.
    
    Args:
        vector_store: FAISS vector store
        
    Returns:
        Configured RAG chain or None if error
    """
    try:
        if not vector_store:
            return None
        
        # Check for OpenAI API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("❌ OpenAI API key not found!")
            return None
        
        # Initialize ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500,
            openai_api_key=api_key
        )
        
        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Create prompt template
        system_message = """You are a helpful AI assistant that answers questions based on the provided context from Ready Tensor AI/ML publications.

Instructions:
- Use ONLY the information provided in the context to answer questions
- If the context doesn't contain enough information, say so clearly
- Provide specific, accurate, and helpful answers
- When possible, mention which publication(s) your answer is based on
- Keep your answers concise but comprehensive

Context:
{context}

Question: {input}

Answer:"""

        prompt = ChatPromptTemplate.from_template(system_message)
        
        # Create the chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, document_chain)
        
        logger.info("Successfully created RAG chain")
        return rag_chain
        
    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")
        st.error(f"❌ Error creating RAG chain: {e}")
        return None


def display_answer_with_sources(response):
    """
    Display the answer and source information in a formatted way.
    
    Args:
        response: Response from the RAG chain
    """
    if not response:
        return
    
    # Display the answer
    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    st.markdown("### 🤖 Answer")
    st.write(response.get('answer', 'No answer generated'))
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display source information
    context_docs = response.get('context', [])
    if context_docs:
        st.markdown("### 📚 Sources")
        
        # Get unique publications
        seen_publications = set()
        for i, doc in enumerate(context_docs):
            pub_title = doc.metadata.get('title', 'Unknown Title')
            pub_id = doc.metadata.get('id', f'doc_{i}')
            
            if pub_id not in seen_publications:
                seen_publications.add(pub_id)
                
                st.markdown(f'<div class="source-info">', unsafe_allow_html=True)
                st.markdown(f"**📄 {pub_title}**")
                st.markdown(f"*ID: {pub_id} | Author: {doc.metadata.get('username', 'Unknown')}*")
                
                # Show a preview of the content
                content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                with st.expander("📖 Content Preview"):
                    st.write(content_preview)
                
                st.markdown('</div>', unsafe_allow_html=True)


def main():
    """
    Main Streamlit application function.
    """
    # Header
    st.markdown('<h1 class="main-header">🔍 Ready Tensor Publication Explorer</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    Welcome to the **Ready Tensor Publication Explorer**! This AI-powered assistant helps you explore and query 
    a collection of AI/ML publications using advanced Retrieval-Augmented Generation (RAG) technology.
    
    **How it works:**
    1. 📚 Publications are loaded and processed into searchable chunks
    2. 🔍 Your questions are matched with relevant content using semantic search
    3. 🤖 AI generates accurate answers based on the retrieved information
    4. 📄 Source publications are provided for transparency
    """)
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown("""
        This RAG system uses:
        - **OpenAI embeddings** for semantic search
        - **FAISS** for vector storage
        - **GPT-3.5-turbo** for answer generation
        - **LangChain** for orchestration
        """)
        
        st.markdown("### 💡 Sample Questions")
        st.markdown("""
        - What is RAG and how does it work?
        - How can I add memory to RAG applications?
        - What are the benefits of retrieval-augmented generation?
        - How do I implement document chunking strategies?
        - What embedding models work best for RAG?
        """)
    
    # Load and process data
    with st.spinner("🔄 Loading publication data..."):
        documents = load_and_process_data()
    
    if not documents:
        st.stop()
    
    st.success(f"✅ Successfully loaded {len(documents)} document chunks!")
    
    # Create vector store
    with st.spinner("🔄 Setting up vector store..."):
        vector_store = create_vector_store(documents)
    
    if not vector_store:
        st.stop()
    
    st.success("✅ Vector store ready!")
    
    # Create RAG chain
    rag_chain = create_rag_chain(vector_store)
    
    if not rag_chain:
        st.stop()
    
    st.success("✅ RAG system initialized!")
    
    # Question input
    st.markdown("---")
    st.markdown("### 💬 Ask Your Question")
    
    user_question = st.text_input(
        "Enter your question about the publications:",
        placeholder="e.g., What is RAG and how does it work?",
        key="question_input"
    )
    
    # Session state for conversation history (optional enhancement)
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Process question
    if user_question:
        with st.spinner("🤔 Thinking..."):
            try:
                # Log the query
                logger.info(f"User query: {user_question}")
                
                # Get response from RAG chain
                response = rag_chain.invoke({"input": user_question})
                
                # Log the response
                logger.info(f"Generated response for query: {user_question[:50]}...")
                
                # Display the answer
                display_answer_with_sources(response)
                
                # Add to conversation history (optional enhancement)
                st.session_state.conversation_history.append({
                    'question': user_question,
                    'answer': response.get('answer', 'No answer generated'),
                    'sources': len(response.get('context', []))
                })
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                st.error(f"❌ Error processing your question: {e}")
    
    # Display conversation history (optional enhancement)
    if st.session_state.conversation_history:
        st.markdown("---")
        st.markdown("### 💭 Conversation History")
        
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:])):  # Show last 5
            with st.expander(f"Q{len(st.session_state.conversation_history)-i}: {conv['question'][:60]}..."):
                st.markdown(f"**Question:** {conv['question']}")
                st.markdown(f"**Answer:** {conv['answer']}")
                st.markdown(f"**Sources:** {conv['sources']} publications referenced")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit, LangChain, OpenAI, and FAISS<br>
        Ready Tensor Publication Explorer | RAG-Powered AI Assistant
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
