#!/usr/bin/env python3
"""
Main entry point for the RAG Publication Explorer.
"""

import argparse
import logging
from pathlib import Path

from src.rag_agent.core.rag_pipeline import RAGPipeline
from src.rag_agent.config.settings import load_config


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/rag_agent.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main function to run the RAG agent."""
    parser = argparse.ArgumentParser(description='RAG Publication Explorer')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--query', type=str, required=True,
                       help='Query to search for in publications')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top results to return')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(config)
        
        # Process query
        logger.info(f"Processing query: {args.query}")
        results = rag_pipeline.query(args.query, top_k=args.top_k)
        
        # Display results
        print(f"\nTop {len(results)} results for query: '{args.query}'\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
            print("-" * 80)
            
    except Exception as e:
        logger.error(f"Error running RAG agent: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
