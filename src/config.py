import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    CHROMA_DB_DIR = os.path.join(os.getcwd(), "data", "chroma_db")
    RUBRICS_DIR = os.path.join(os.getcwd(), "data", "rubrics")
    SUBMISSIONS_DIR = os.path.join(os.getcwd(), "data", "submissions")
    # MODEL_NAME = "gpt-4-turbo-preview" # Or "gpt-4o"
    MODEL_NAME = "qwen3-vl:8b"  # Local model via Ollama for cost efficiency

config = Config()
