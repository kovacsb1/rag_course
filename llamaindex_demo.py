import logging
import sys
import os
import toml
import qdrant_client
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    Settings
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.ollama import Ollama

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/llamalog.txt", level=logging.DEBUG)

config = toml.load('config.toml')
HF_EMBEDDING_MODEL_NAME = config["chunking"]["hf_embedding_model_name"]
SIMILARITY_TOP_K = config["retrieval"]["similarity_top_k"]
SPARSE_TOP_K = config["retrieval"]["sparse_top_k"]

OLLAMA_LLM = config["ollama"]["llm"]

QDRANT_URL = config["qdrant"]["url"]
QDRANT_PORT = config["qdrant"]["port"]
QDRANT_COLLECTION = config["qdrant"]["collection_name"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL_NAME)

# ollama
Settings.llm = Ollama(model=OLLAMA_LLM, request_timeout=360.0)


# connect to the vectorDB
vectordb_client = qdrant_client.QdrantClient(
    url=QDRANT_URL,
    port=QDRANT_PORT,
    api_key=QDRANT_API_KEY,
)

vector_store = QdrantVectorStore(client=vectordb_client, collection_name=QDRANT_COLLECTION, enable_hybrid=True)
index = VectorStoreIndex.from_vector_store(vector_store)

query_engine = index.as_query_engine(
    vector_store_query_mode="hybrid",
    similarity_top_k=SIMILARITY_TOP_K,
    sparse_top_k=SPARSE_TOP_K
    )

response = query_engine.query("What did the author do growing up?")
print(response)

print("Chunks used in creating the response:")
for n in response.source_nodes:
    print(n)