"""
MIMIC-IV-ED Data Processor
Handles loading, processing, and vector database creation from MIMIC dataset
"""
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIMICProcessor:
    """Process MIMIC-IV-ED dataset for RAG system"""
    
    def __init__(self):
        """Initialize MIMIC data processor"""
        self.data_path = Path(Config.MIMIC_DATA_PATH)
        self.vector_db_path = Path(Config.VECTOR_DB_PATH)
        self.processed_path = Path(Config.PROCESSED_DATA_PATH)
        self.embedding_model = None
        self.vector_db = None
        self.collection = None
        
        # Ensure directories exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all MIMIC CSV files into DataFrames"""
        logger.info(f"Loading MIMIC data from {self.data_path}")
        
        data = {}
        for table in Config.MIMIC_TABLES:
            csv_path = self.data_path / f"{table}.csv"
            if csv_path.exists():
                try:
                    data[table] = pd.read_csv(csv_path, low_memory=False)
                    logger.info(f"Loaded {table}: {len(data[table])} rows")
                except Exception as e:
                    logger.error(f"Error loading {table}: {e}")
            else:
                logger.warning(f"File not found: {csv_path}")
        
        return data
    
    def process_data(self) -> pd.DataFrame:
        """Process and merge MIMIC data into a unified dataset"""
        logger.info("Processing MIMIC data...")
        
        data = self.load_data()
        
        if not data:
            logger.warning("No data loaded. Using empty DataFrame.")
            return pd.DataFrame()
        
        # Start with triage data as base
        if 'triage' in data:
            processed = data['triage'].copy()
        elif 'edstays' in data:
            processed = data['edstays'].copy()
        else:
            logger.error("No base table (triage/edstays) found")
            return pd.DataFrame()
        
        # Merge with vital signs
        if 'vitalsign' in data:
            try:
                processed = processed.merge(
                    data['vitalsign'],
                    on='subject_id',
                    how='left',
                    suffixes=('', '_vital')
                )
            except Exception as e:
                logger.warning(f"Could not merge vitalsign: {e}")
        
        # Merge with diagnoses
        if 'diagnosis' in data:
            try:
                diagnosis_grouped = data['diagnosis'].groupby('subject_id')['icd_code'].apply(
                    lambda x: ', '.join(x.astype(str))
                ).reset_index(name='diagnoses')
                processed = processed.merge(
                    diagnosis_grouped,
                    on='subject_id',
                    how='left'
                )
            except Exception as e:
                logger.warning(f"Could not merge diagnosis: {e}")
        
        # Create a text representation for each case
        processed = self._create_case_text(processed)
        
        logger.info(f"Processed data: {len(processed)} cases")
        return processed
    
    def _create_case_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a text representation of each case for RAG"""
        def create_text(row):
            text_parts = []
            
            # Add demographics if available
            if 'age' in row and pd.notna(row['age']):
                text_parts.append(f"Age: {row['age']}")
            if 'gender' in row and pd.notna(row['gender']):
                text_parts.append(f"Gender: {row['gender']}")
            
            # Add vital signs
            vital_fields = ['heartrate', 'resprate', 'o2sat', 'temperature', 'sbp', 'dbp']
            for field in vital_fields:
                if field in row and pd.notna(row[field]):
                    text_parts.append(f"{field}: {row[field]}")
            
            # Add chief complaint if available
            if 'chiefcomplaint' in row and pd.notna(row['chiefcomplaint']):
                text_parts.append(f"Chief Complaint: {row['chiefcomplaint']}")
            elif 'chief_complaint' in row and pd.notna(row['chief_complaint']):
                text_parts.append(f"Chief Complaint: {row['chief_complaint']}")
            
            # Add ESI level if available
            if 'esi' in row and pd.notna(row['esi']):
                text_parts.append(f"ESI Level: {row['esi']}")
            elif 'triage_level' in row and pd.notna(row['triage_level']):
                text_parts.append(f"Triage Level: {row['triage_level']}")
            
            # Add diagnoses
            if 'diagnoses' in row and pd.notna(row['diagnoses']):
                text_parts.append(f"Diagnoses: {row['diagnoses']}")
            
            return " | ".join(text_parts)
        
        df['case_text'] = df.apply(create_text, axis=1)
        return df
    
    def create_vector_db(self, force_recreate: bool = False) -> chromadb.Collection:
        """Create or load vector database from processed data"""
        logger.info("Creating vector database...")
        
        # Initialize embedding model
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB client
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=str(self.vector_db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        collection_name = "mimic_cases"
        
        # Check if collection exists
        existing_collections = [c.name for c in client.list_collections()]
        
        if collection_name in existing_collections and not force_recreate:
            logger.info("Loading existing vector database collection...")
            self.collection = client.get_collection(name=collection_name)
            logger.info(f"Loaded collection with {self.collection.count()} cases")
            return self.collection
        
        # Create new collection
        if collection_name in existing_collections:
            client.delete_collection(name=collection_name)
        
        logger.info("Creating new vector database collection...")
        self.collection = client.create_collection(
            name=collection_name,
            metadata={"description": "MIMIC-IV-ED triage cases"}
        )
        
        # Process and add data
        processed_data = self.process_data()
        
        if processed_data.empty:
            logger.warning("No processed data available. Creating empty collection.")
            return self.collection
        
        # Chunk and embed data
        documents = []
        metadatas = []
        ids = []
        
        batch_size = 100
        total_rows = len(processed_data)
        
        for idx, row in processed_data.iterrows():
            case_text = row.get('case_text', '')
            
            if not case_text or pd.isna(case_text):
                continue
            
            # Create chunks if text is too long
            chunks = self._chunk_text(case_text)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Ensure document IDs are globally unique even when multiple rows share the same subject_id
                doc_id = f"{row.get('subject_id', 'unknown')}_{idx}_{chunk_idx}"
                
                documents.append(chunk)
                metadatas.append({
                    'subject_id': str(row.get('subject_id', '')),
                    'esi_level': str(row.get('esi', row.get('triage_level', ''))),
                    'age': str(row.get('age', '')),
                    'gender': str(row.get('gender', '')),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks)
                })
                ids.append(doc_id)
            
            # Batch insert for efficiency
            if len(documents) >= batch_size:
                embeddings = self.embedding_model.encode(documents).tolist()
                self.collection.add(
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added batch: {len(documents)} documents (Progress: {idx}/{total_rows})")
                documents = []
                metadatas = []
                ids = []
        
        # Add remaining documents
        if documents:
            embeddings = self.embedding_model.encode(documents).tolist()
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added final batch: {len(documents)} documents")
        
        logger.info(f"Vector database created with {self.collection.count()} documents")
        return self.collection
    
    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into chunks with overlap"""
        chunk_size = chunk_size or Config.CHUNK_SIZE
        overlap = overlap or Config.CHUNK_OVERLAP
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def search_similar_cases(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar cases using vector similarity
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of similar cases with metadata and similarity scores
        """
        if not query or not isinstance(query, str):
            logger.warning("Invalid query provided to search_similar_cases")
            return []
        
        top_k = top_k or Config.RAG_TOP_K
        
        try:
            if self.collection is None:
                logger.warning("Vector database not initialized. Attempting to create...")
                try:
                    self.create_vector_db()
                except Exception as e:
                    logger.error(f"Could not create vector database: {e}")
                    return []
            
            if self.collection is None:
                logger.error("Vector database collection is still None after creation attempt")
                return []
            
            # Check if collection has any documents
            collection_count = self.collection.count()
            if collection_count == 0:
                logger.warning("Vector database collection is empty. No similar cases to retrieve.")
                return []
            
            if self.embedding_model is None:
                logger.info(f"Loading embedding model: {Config.EMBEDDING_MODEL}")
                try:
                    self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
                except Exception as e:
                    logger.error(f"Error loading embedding model: {e}")
                    return []
            
            # Generate query embedding
            try:
                query_embedding = self.embedding_model.encode([query], show_progress_bar=False).tolist()[0]
            except Exception as e:
                logger.error(f"Error encoding query: {e}")
                return []
            
            # Search with proper error handling
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, collection_count)  # Don't request more than available
                )
            except Exception as e:
                logger.error(f"Error querying vector database: {e}")
                return []
            
            # Format results
            similar_cases = []
            if results and 'documents' in results and results['documents']:
                documents = results['documents'][0] if isinstance(results['documents'], list) else []
                metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else []
                distances = results.get('distances', [[]])[0] if results.get('distances') else []
                
                for i in range(len(documents)):
                    case = {
                        'document': documents[i],
                        'metadata': metadatas[i] if i < len(metadatas) else {},
                        'distance': distances[i] if i < len(distances) else 1.0
                    }
                    # Filter by similarity threshold if configured
                    similarity = 1.0 - case['distance']
                    if similarity >= Config.SIMILARITY_THRESHOLD:
                        similar_cases.append(case)
            
            logger.debug(f"Found {len(similar_cases)} similar cases above threshold")
            return similar_cases
            
        except Exception as e:
            logger.error(f"Error in search_similar_cases: {e}", exc_info=True)
            return []
    
    def normalize_text(self, text: Any) -> str:
        """Normalize text for processing"""
        if pd.isna(text) or text is None:
            return ""
        return str(text).lower().strip()
    
    def clean_text(self, text: Any) -> str:
        """Clean and normalize text"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower().strip()
        # Remove excessive whitespace
        text = ' '.join(text.split())
        return text


if __name__ == "__main__":
    # Test the processor
    processor = MIMICProcessor()
    collection = processor.create_vector_db()
    print(f"Vector database ready with {collection.count()} documents")
    
    # Test search
    test_query = "Chest pain, heart rate 110, oxygen 92"
    results = processor.search_similar_cases(test_query, top_k=3)
    print(f"\nFound {len(results)} similar cases:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {1 - result['distance']:.3f}")
        print(f"   Case: {result['document'][:100]}...")