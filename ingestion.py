from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext

import logging
import os
import qdrant_client
from dotenv import load_dotenv
import toml

# create log folder
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/ingest_log.txt", level=logging.DEBUG)

config = toml.load('config.toml')
SOURCE_DIR = config["persistance"]["source_dir"]

HF_EMBEDDING_MODEL_NAME = config["chunking"]["hf_embedding_model_name"]
CHUNK_SIZE = config["chunking"]["chunk_size"]
CHUNK_OVERLAP = config["chunking"]["chunk_overlap"]

QDRANT_URL = config["qdrant"]["url"]
QDRANT_PORT = config["qdrant"]["port"]
QDRANT_COLLECTION = config["qdrant"]["collection_name"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL_NAME)


# load the documents and create the index
splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# qdrant vectordb client
vectordb_client = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
)
# qdrant vector store
vector_store = QdrantVectorStore(client=vectordb_client, collection_name=QDRANT_COLLECTION)

# load documents
documents = SimpleDirectoryReader(SOURCE_DIR).load_data()
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# here we create the index from the documents, which persists in the vectorDB
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    transformations=[splitter]
)