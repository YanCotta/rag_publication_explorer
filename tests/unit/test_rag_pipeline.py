"""
Unit tests for the RAG pipeline.
"""

import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rag_agent.core.rag_pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    """Test cases for RAG pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'document_processing': {
                'chunk_size': 500,
                'chunk_overlap': 100
            },
            'embeddings': {
                'model_name': 'test-model',
                'device': 'cpu'
            },
            'vector_store': {
                'type': 'faiss',
                'dimension': 384
            },
            'llm': {
                'model_name': 'test-llm',
                'max_tokens': 500
            }
        }
    
    @patch('rag_agent.core.rag_pipeline.DocumentProcessor')
    @patch('rag_agent.core.rag_pipeline.EmbeddingManager')
    @patch('rag_agent.core.rag_pipeline.VectorStore')
    @patch('rag_agent.core.rag_pipeline.LLMInterface')
    def test_pipeline_initialization(self, mock_llm, mock_vector, mock_embed, mock_doc):
        """Test RAG pipeline initialization."""
        # Initialize pipeline
        pipeline = RAGPipeline(self.config)
        
        # Verify components were initialized
        mock_doc.assert_called_once_with(self.config['document_processing'])
        mock_embed.assert_called_once_with(self.config['embeddings'])
        mock_vector.assert_called_once_with(self.config['vector_store'])
        mock_llm.assert_called_once_with(self.config['llm'])
    
    @patch('rag_agent.core.rag_pipeline.DocumentProcessor')
    @patch('rag_agent.core.rag_pipeline.EmbeddingManager')
    @patch('rag_agent.core.rag_pipeline.VectorStore')
    @patch('rag_agent.core.rag_pipeline.LLMInterface')
    def test_query_processing(self, mock_llm, mock_vector, mock_embed, mock_doc):
        """Test query processing workflow."""
        # Setup mocks
        mock_embed_instance = Mock()
        mock_vector_instance = Mock()
        mock_llm_instance = Mock()
        
        mock_embed.return_value = mock_embed_instance
        mock_vector.return_value = mock_vector_instance
        mock_llm.return_value = mock_llm_instance
        
        # Mock return values
        query_embedding = [0.1, 0.2, 0.3]
        retrieved_docs = [{'text': 'test doc', 'score': 0.9}]
        llm_response = "Test response"
        
        mock_embed_instance.generate_query_embedding.return_value = query_embedding
        mock_vector_instance.similarity_search.return_value = retrieved_docs
        mock_llm_instance.generate_response.return_value = llm_response
        
        # Initialize pipeline and process query
        pipeline = RAGPipeline(self.config)
        result = pipeline.query("test query", top_k=3)
        
        # Verify calls
        mock_embed_instance.generate_query_embedding.assert_called_once_with("test query")
        mock_vector_instance.similarity_search.assert_called_once_with(query_embedding, top_k=3)
        mock_llm_instance.generate_response.assert_called_once_with("test query", retrieved_docs)
        
        # Verify result structure
        self.assertIsInstance(result, dict)
        self.assertIn('query', result)
        self.assertIn('response', result)
        self.assertIn('retrieved_documents', result)
        self.assertEqual(result['query'], "test query")
        self.assertEqual(result['response'], llm_response)


if __name__ == '__main__':
    unittest.main()
