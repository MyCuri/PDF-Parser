import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# MongoDB
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE")
MONGODB_PDF_COLLECTION = "pdfs"
MONGODB_PROMPTS_COLLECTION = "prompts"

TEMP_FOLDER = "temp"
