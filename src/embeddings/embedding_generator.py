"""
Embedding generator module for creating and storing vector embeddings.

This module handles generating embeddings using OpenAI's text-embedding-ada-002 model
and storing them in a FAISS vector store for efficient similarity search.
"""

import logging
import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from openai import OpenAI
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    A class to handle embedding generation and FAISS vector store management.
    
    This class provides methods to generate embeddings using OpenAI's API,
    create FAISS indexes, and save/load vector stores for efficient retrieval.
    """
    
    def __init__(self, model_name: str = "text-embedding-ada-002", embedding_dim: int = 1536):
        """
        Initialize the EmbeddingGenerator.
        
        Args:
            model_name (str): OpenAI embedding model name
            embedding_dim (int): Dimension of embeddings (1536 for ada-002)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize FAISS index
        self.index = None
        self.metadata = []
        
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using OpenAI API.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Number of texts to process in each API call
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                # Call OpenAI API
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                # Extract embeddings from response
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Generated {len(batch_embeddings)} embeddings in batch")
                
                # Rate limiting - small delay between batches
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                raise
                
        logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def create_faiss_index(self, embeddings: List[List[float]]) -> faiss.Index:
        """
        Create a FAISS index from embeddings.
        
        Args:
            embeddings (List[List[float]]): List of embedding vectors
            
        Returns:
            faiss.Index: FAISS index for similarity search
        """
        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Create FAISS index (using L2 distance)
            index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Add embeddings to index
            index.add(embeddings_array)
            
            logger.info(f"Created FAISS index with {index.ntotal} vectors")
            return index
            
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}")
            raise
    
    def process_chunks_and_create_embeddings(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Process text chunks to create embeddings and build FAISS index.
        
        Args:
            chunks (List[Dict[str, Any]]): List of text chunks with metadata
        """
        try:
            # Extract texts for embedding
            texts = [chunk['chunk_text'] for chunk in chunks]
            
            logger.info(f"Generating embeddings for {len(texts)} text chunks")
            
            # Generate embeddings
            embeddings = self.generate_embeddings_batch(texts)
            
            # Create FAISS index
            self.index = self.create_faiss_index(embeddings)
            
            # Store metadata for retrieval
            self.metadata = chunks
            
            logger.info("Successfully created embeddings and FAISS index")
            
        except Exception as e:
            logger.error(f"Error processing chunks: {e}")
            raise
    
    def save_vector_store(self, save_path: str) -> None:
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            save_path (str): Directory path to save the vector store
        """
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            if self.index is None:
                raise ValueError("No FAISS index to save. Create embeddings first.")
            
            # Save FAISS index
            index_file = save_path / "faiss_index.bin"
            faiss.write_index(self.index, str(index_file))
            
            # Save metadata
            metadata_file = save_path / "metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            # Save configuration
            config_file = save_path / "config.pkl"
            config = {
                'model_name': self.model_name,
                'embedding_dim': self.embedding_dim,
                'num_vectors': self.index.ntotal
            }
            with open(config_file, 'wb') as f:
                pickle.dump(config, f)
            
            logger.info(f"Vector store saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise
    
    def load_vector_store(self, load_path: str) -> None:
        """
        Load the FAISS index and metadata from disk.
        
        Args:
            load_path (str): Directory path to load the vector store from
        """
        try:
            load_path = Path(load_path)
            
            if not load_path.exists():
                raise FileNotFoundError(f"Vector store path does not exist: {load_path}")
            
            # Load FAISS index
            index_file = load_path / "faiss_index.bin"
            if not index_file.exists():
                raise FileNotFoundError(f"FAISS index file not found: {index_file}")
            
            self.index = faiss.read_index(str(index_file))
            
            # Load metadata
            metadata_file = load_path / "metadata.pkl"
            if not metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            
            # Load configuration
            config_file = load_path / "config.pkl"
            if config_file.exists():
                with open(config_file, 'rb') as f:
                    config = pickle.load(f)
                    self.model_name = config.get('model_name', self.model_name)
                    self.embedding_dim = config.get('embedding_dim', self.embedding_dim)
            
            logger.info(f"Vector store loaded from {load_path}")
            logger.info(f"Loaded {self.index.ntotal} vectors with {len(self.metadata)} metadata entries")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using the query.
        
        Args:
            query (str): Query text to search for
            k (int): Number of similar documents to return
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with scores
        """
        try:
            if self.index is None:
                raise ValueError("No FAISS index loaded. Load or create a vector store first.")
            
            # Generate embedding for query
            query_embedding = self.generate_embeddings_batch([query])[0]
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search in FAISS index
            distances, indices = self.index.search(query_vector, k)
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):
                    result = {
                        'rank': i + 1,
                        'score': float(distance),
                        'similarity': 1 / (1 + float(distance)),  # Convert distance to similarity
                        'metadata': self.metadata[idx]
                    }
                    results.append(result)
            
            logger.debug(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error searching for similar documents: {e}")
            raise


def main():
    """
    Main function to demonstrate usage of the EmbeddingGenerator.
    """
    try:
        # Import data loader
        import sys
        sys.path.append('src')
        from data_processing.data_loader import PublicationDataLoader
        
        # Initialize components
        data_loader = PublicationDataLoader()
        embedding_generator = EmbeddingGenerator()
        
        # Load and process data
        file_path = "data/raw/project_1_publications.json"
        chunks = data_loader.process_publications_file(file_path)
        
        if not chunks:
            print("No chunks to process")
            return
        
        print(f"Processing {len(chunks)} chunks...")
        
        # Create embeddings and vector store
        embedding_generator.process_chunks_and_create_embeddings(chunks)
        
        # Save vector store
        save_path = "artifacts/vector_store"
        embedding_generator.save_vector_store(save_path)
        
        print(f"Vector store saved to {save_path}")
        
        # Test search functionality
        test_query = "What is RAG and how does it work?"
        results = embedding_generator.search_similar(test_query, k=3)
        
        print(f"\nTest search results for: '{test_query}'")
        for result in results:
            print(f"Rank {result['rank']}: {result['metadata']['title'][:50]}... (Score: {result['score']:.3f})")
            
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
