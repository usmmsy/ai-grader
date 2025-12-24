import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.config import config
import os

class VectorStoreManager:
    def __init__(self):
        self.persist_directory = config.CHROMA_DB_DIR
        self.embedding_function = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
        
        # Ensure directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)

    def get_collection(self, collection_name: str):
        """
        Returns a LangChain Chroma vector store object for a specific collection.
        """
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embedding_function,
        )

    def add_documents(self, collection_name: str, documents: list):
        """
        Adds documents to a specific collection.
        """
        vector_store = self.get_collection(collection_name)
        vector_store.add_documents(documents)
        print(f"Added {len(documents)} documents to collection '{collection_name}'")

    def query(self, collection_name: str, query_text: str, k: int = 4):
        """
        Queries the vector store.
        """
        vector_store = self.get_collection(collection_name)
        return vector_store.similarity_search(query_text, k=k)

vector_store_manager = VectorStoreManager()
