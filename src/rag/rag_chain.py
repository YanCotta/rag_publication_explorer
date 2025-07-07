"""
RAG Chain module for building a complete question-answering system.

This module combines the FAISS vector store, retriever, and OpenAI LLM
to create a seamless question-answering chain using LangChain.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import LLMChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomRetriever:
    """
    Custom retriever class that works with our EmbeddingGenerator.
    """
    
    def __init__(self, embedding_generator):
        """
        Initialize the custom retriever.
        
        Args:
            embedding_generator: EmbeddingGenerator instance with loaded vector store
        """
        self.embedding_generator = embedding_generator
    
    def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query (str): The query string
            k (int): Number of documents to retrieve
            
        Returns:
            List[Document]: List of relevant documents
        """
        try:
            # Search for similar documents
            results = self.embedding_generator.search_similar(query, k=k)
            
            # Convert to LangChain Document objects
            documents = []
            for result in results:
                metadata = result['metadata']
                doc = Document(
                    page_content=metadata['chunk_text'],
                    metadata={
                        'chunk_id': metadata['chunk_id'],
                        'publication_id': metadata['publication_id'],
                        'title': metadata['title'],
                        'username': metadata['username'],
                        'score': result['score'],
                        'similarity': result['similarity']
                    }
                )
                documents.append(doc)
            
            logger.debug(f"Retrieved {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []


class RAGChain:
    """
    A complete RAG (Retrieval-Augmented Generation) chain implementation.
    
    This class combines document retrieval with language model generation
    to provide accurate, context-aware answers to user questions.
    """
    
    def __init__(
        self, 
        embedding_generator, 
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize the RAG chain.
        
        Args:
            embedding_generator: EmbeddingGenerator instance with loaded vector store
            model_name (str): OpenAI model name for generation
            temperature (float): Temperature for text generation
            max_tokens (int): Maximum tokens in response
        """
        self.embedding_generator = embedding_generator
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Verify OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize retriever
        self.retriever = CustomRetriever(embedding_generator)
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Define prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Build the chain
        self.chain = self._build_chain()
        
        logger.info(f"RAG chain initialized with model: {model_name}")
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for the RAG chain.
        
        Returns:
            ChatPromptTemplate: The prompt template for question answering
        """
        template = """You are a helpful AI assistant that answers questions based on the provided context from Ready Tensor publications about AI, machine learning, and data science.

Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context provided, just say that you don't know - don't try to make up an answer.

When providing answers:
1. Be accurate and stick to the information in the context
2. Be helpful and comprehensive
3. If relevant, mention which publication or author the information comes from
4. Use a friendly and professional tone

Context:
{context}

Question: {question}

Answer:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _build_chain(self):
        """
        Build the RAG chain using LangChain components.
        
        Returns:
            The complete RAG chain
        """
        def format_docs(docs):
            """Format retrieved documents into a single context string."""
            formatted_docs = []
            for doc in docs:
                # Include title and author information
                title = doc.metadata.get('title', 'Unknown')
                username = doc.metadata.get('username', 'Unknown')
                content = doc.page_content
                
                formatted_doc = f"Publication: {title}\nAuthor: {username}\nContent: {content}\n\n"
                formatted_docs.append(formatted_doc)
            
            return "\n".join(formatted_docs)
        
        # Build the chain
        chain = (
            {"context": lambda x: format_docs(self.retriever.get_relevant_documents(x["question"])), 
             "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer from the RAG chain.
        
        Args:
            question (str): The question to ask
            
        Returns:
            Dict[str, Any]: Response containing answer and metadata
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Get relevant documents
            relevant_docs = self.retriever.get_relevant_documents(question, k=4)
            
            # Generate answer using the chain
            answer = self.chain.invoke({"question": question})
            
            # Prepare response
            response = {
                "question": question,
                "answer": answer,
                "source_documents": [
                    {
                        "title": doc.metadata.get('title', 'Unknown'),
                        "username": doc.metadata.get('username', 'Unknown'),
                        "chunk_id": doc.metadata.get('chunk_id', 'Unknown'),
                        "similarity_score": doc.metadata.get('similarity', 0.0),
                        "content_preview": doc.page_content[:200] + "..."
                    }
                    for doc in relevant_docs
                ],
                "num_sources": len(relevant_docs)
            }
            
            logger.info("Successfully generated answer")
            return response
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return {
                "question": question,
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "source_documents": [],
                "num_sources": 0
            }
    
    def get_source_summary(self, question: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Get a summary of source documents relevant to a question.
        
        Args:
            question (str): The question to search for
            k (int): Number of sources to return
            
        Returns:
            List[Dict[str, Any]]: List of source document summaries
        """
        try:
            relevant_docs = self.retriever.get_relevant_documents(question, k=k)
            
            sources = []
            for i, doc in enumerate(relevant_docs):
                source = {
                    "rank": i + 1,
                    "title": doc.metadata.get('title', 'Unknown'),
                    "username": doc.metadata.get('username', 'Unknown'),
                    "similarity_score": doc.metadata.get('similarity', 0.0),
                    "content_preview": doc.page_content[:300] + "..."
                }
                sources.append(source)
            
            return sources
            
        except Exception as e:
            logger.error(f"Error getting source summary: {e}")
            return []


def main():
    """
    Main function to demonstrate usage of the RAG chain.
    """
    try:
        # Import required modules
        import sys
        sys.path.append('src')
        from embeddings.embedding_generator import EmbeddingGenerator
        
        # Initialize embedding generator and load vector store
        embedding_generator = EmbeddingGenerator()
        vector_store_path = "artifacts/vector_store"
        
        print("Loading vector store...")
        embedding_generator.load_vector_store(vector_store_path)
        
        # Initialize RAG chain
        print("Initializing RAG chain...")
        rag_chain = RAGChain(embedding_generator)
        
        # Test questions
        test_questions = [
            "What is RAG and how does it work?",
            "How can I add memory to RAG applications?",
            "What are the best practices for computer vision projects?",
            "How do auto-encoders work for image compression?"
        ]
        
        print("\n" + "="*80)
        print("Testing RAG Chain with Sample Questions")
        print("="*80)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- Question {i} ---")
            print(f"Q: {question}")
            
            response = rag_chain.ask_question(question)
            
            print(f"A: {response['answer']}")
            print(f"\nSources used: {response['num_sources']}")
            
            if response['source_documents']:
                print("Top source:")
                top_source = response['source_documents'][0]
                print(f"  - {top_source['title']} by {top_source['username']}")
                print(f"  - Similarity: {top_source['similarity_score']:.3f}")
            
            print("-" * 80)
            
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
