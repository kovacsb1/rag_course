import logging
import sys
import os
from dotenv import load_dotenv

from llama_index.core import (
    StorageContext,
    Settings,
    load_index_from_storage,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/llamalog.txt", level=logging.DEBUG)

load_dotenv()
PERSIST_DIR = os.environ.get("PERSIST_DIR")

HF_EMBEDDING_MODEL_NAME = os.environ.get("HF_EMBEDDING_MODEL_NAME")
OLLAMA_LLM = os.environ.get("OLLAMA_LLM")

SIMILARITY_TOP_K = int(os.environ.get("SIMILARITY_TOP_K"))

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL_NAME)

# ollama
Settings.llm = Ollama(model=OLLAMA_LLM, request_timeout=360.0)


# load the existing index
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)

response = query_engine.query("What did the author do growing up?")
print(response)

print("Chunks used in creating the response:")
for n in response.source_nodes:
    print(n)