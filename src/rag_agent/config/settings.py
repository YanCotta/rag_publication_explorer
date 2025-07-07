"""
Configuration management for the RAG agent.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration settings.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'document_processing': {
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'supported_formats': ['.pdf', '.txt', '.md', '.docx']
        },
        'embeddings': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'device': 'cpu',
            'batch_size': 32
        },
        'vector_store': {
            'type': 'faiss',
            'index_path': 'data/embeddings/index.faiss',
            'dimension': 384
        },
        'llm': {
            'model_name': 'gpt-3.5-turbo',
            'max_tokens': 1000,
            'temperature': 0.7
        }
    }
