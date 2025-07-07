"""
Data loader module for processing publications JSON data.

This module handles loading and parsing of the project_1_publications.json file,
extracting relevant text content, and splitting it into manageable chunks for embedding.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PublicationDataLoader:
    """
    A class to handle loading and processing of publication data from JSON files.
    
    This class provides methods to load publication data, extract relevant text content,
    and split the text into chunks suitable for embedding and vector storage.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PublicationDataLoader.
        
        Args:
            chunk_size (int): Maximum number of characters per chunk
            chunk_overlap (int): Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def load_publications_data(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load publications data from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file containing publications data
            
        Returns:
            Optional[List[Dict[str, Any]]]: List of publication dictionaries or None if error
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"The file {file_path} does not exist")
                
            logger.info(f"Loading publications data from: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            if not isinstance(data, list):
                logger.error("Expected JSON file to contain a list of publications")
                raise ValueError("JSON file must contain a list of publications")
                
            logger.info(f"Successfully loaded {len(data)} publications")
            return data
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading file {file_path}: {e}")
            raise
            
    def extract_text_content(self, publications: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract title and publication_description from publications data.
        
        Args:
            publications (List[Dict[str, Any]]): List of publication dictionaries
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing extracted text content
        """
        extracted_content = []
        
        for i, publication in enumerate(publications):
            try:
                # Extract required fields
                title = publication.get('title', '').strip()
                description = publication.get('publication_description', '').strip()
                pub_id = publication.get('id', f'publication_{i}')
                username = publication.get('username', 'unknown')
                
                if not title and not description:
                    logger.warning(f"Publication {pub_id} has no title or description")
                    continue
                
                # Combine title and description
                combined_text = f"Title: {title}\n\nContent: {description}"
                
                extracted_content.append({
                    'id': pub_id,
                    'username': username,
                    'title': title,
                    'content': description,
                    'combined_text': combined_text
                })
                
                logger.debug(f"Extracted content from publication {pub_id}: {title[:50]}...")
                
            except Exception as e:
                logger.warning(f"Error extracting content from publication {i}: {e}")
                continue
                
        logger.info(f"Successfully extracted content from {len(extracted_content)} publications")
        return extracted_content
    
    def split_text_into_chunks(self, extracted_content: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Split extracted text content into chunks for embedding.
        
        Args:
            extracted_content (List[Dict[str, str]]): List of extracted text content
            
        Returns:
            List[Dict[str, Any]]: List of text chunks with metadata
        """
        all_chunks = []
        
        for content_item in extracted_content:
            try:
                # Split the combined text into chunks
                text_chunks = self.text_splitter.split_text(content_item['combined_text'])
                
                # Create chunk objects with metadata
                for i, chunk in enumerate(text_chunks):
                    chunk_data = {
                        'chunk_id': f"{content_item['id']}_chunk_{i}",
                        'publication_id': content_item['id'],
                        'username': content_item['username'],
                        'title': content_item['title'],
                        'chunk_index': i,
                        'chunk_text': chunk,
                        'chunk_length': len(chunk)
                    }
                    all_chunks.append(chunk_data)
                    
                logger.debug(f"Split publication {content_item['id']} into {len(text_chunks)} chunks")
                
            except Exception as e:
                logger.warning(f"Error splitting text for publication {content_item['id']}: {e}")
                continue
                
        logger.info(f"Created {len(all_chunks)} text chunks total")
        return all_chunks
    
    def process_publications_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Complete pipeline to load and process publications file.
        
        Args:
            file_path (str): Path to the JSON file containing publications data
            
        Returns:
            List[Dict[str, Any]]: List of processed text chunks ready for embedding
        """
        try:
            # Load data from file
            publications = self.load_publications_data(file_path)
            if not publications:
                return []
            
            # Extract text content
            extracted_content = self.extract_text_content(publications)
            if not extracted_content:
                logger.warning("No content extracted from publications")
                return []
            
            # Split into chunks
            chunks = self.split_text_into_chunks(extracted_content)
            
            logger.info(f"Processing complete: {len(chunks)} chunks ready for embedding")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in processing pipeline: {e}")
            raise


def main():
    """
    Main function to demonstrate usage of the PublicationDataLoader.
    """
    # Example usage
    data_loader = PublicationDataLoader(chunk_size=1000, chunk_overlap=200)
    
    # Path to the publications JSON file
    file_path = "data/raw/project_1_publications.json"
    
    try:
        # Process the publications file
        chunks = data_loader.process_publications_file(file_path)
        
        # Display summary
        if chunks:
            print(f"\nProcessing Summary:")
            print(f"Total chunks created: {len(chunks)}")
            print(f"Average chunk length: {sum(chunk['chunk_length'] for chunk in chunks) / len(chunks):.1f} characters")
            
            # Show first chunk as example
            if chunks:
                print(f"\nExample chunk:")
                print(f"Chunk ID: {chunks[0]['chunk_id']}")
                print(f"Publication: {chunks[0]['title']}")
                print(f"Text preview: {chunks[0]['chunk_text'][:200]}...")
        else:
            print("No chunks were created")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
