"""
Core RAG Pipeline implementation.
"""

import logging
from typing import List, Dict, Any

from ..data_processing.document_processor import DocumentProcessor
from ..embeddings.embedding_manager import EmbeddingManager
from ..retrieval.vector_store import VectorStore
from ..generation.llm_interface import LLMInterface


class RAGPipeline:
    """Main RAG pipeline orchestrating all components."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAG pipeline with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.document_processor = DocumentProcessor(config.get('document_processing', {}))
        self.embedding_manager = EmbeddingManager(config.get('embeddings', {}))
        self.vector_store = VectorStore(config.get('vector_store', {}))
        self.llm_interface = LLMInterface(config.get('llm', {}))
        
        self.logger.info("RAG Pipeline initialized successfully")
    
    def ingest_documents(self, document_paths: List[str]) -> None:
        """
        Ingest and process documents into the vector store.
        
        Args:
            document_paths: List of paths to documents to ingest
        """
        for doc_path in document_paths:
            self.logger.info(f"Processing document: {doc_path}")
            
            # Process document
            chunks = self.document_processor.process_document(doc_path)
            
            # Generate embeddings
            embeddings = self.embedding_manager.generate_embeddings(chunks)
            
            # Store in vector store
            self.vector_store.add_documents(chunks, embeddings)
    
    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Process a query and return relevant results.
        
        Args:
            query: User query string
            top_k: Number of top results to return
            
        Returns:
            List of relevant results with generated responses
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_query_embedding(query)
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(
            query_embedding, top_k=top_k
        )
        
        # Generate response using LLM
        response = self.llm_interface.generate_response(query, retrieved_docs)
        
        return {
            'query': query,
            'response': response,
            'retrieved_documents': retrieved_docs
        }
