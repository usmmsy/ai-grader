from src.rag.ingestion import ingestion_service
from src.rag.vector_store import vector_store_manager
import os

class RubricExtractorAgent:
    def __init__(self):
        self.collection_name = "rubrics"

    def ingest_rubric(self, file_path: str):
        """
        Ingests a rubric file into the vector database.
        """
        print(f"Starting ingestion for: {file_path}")
        
        try:
            # 1. Load and Chunk
            chunks = ingestion_service.process_and_chunk(file_path)
            print(f"Generated {len(chunks)} chunks.")

            # 2. Store in Vector DB
            vector_store_manager.add_documents(self.collection_name, chunks)
            
            print("Ingestion complete.")
            return True
        except Exception as e:
            print(f"Error during ingestion: {e}")
            return False

    def query_rubric(self, query: str):
        """
        Test method to query the ingested rubric.
        """
        results = vector_store_manager.query(self.collection_name, query)
        return results

rubric_extractor = RubricExtractorAgent()
