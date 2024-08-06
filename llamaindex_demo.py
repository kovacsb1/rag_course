import logging
import sys
import os
import toml
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

config = toml.load('config.toml')
PERSIST_DIR = config["persistance"]["persist_dir"]

HF_EMBEDDING_MODEL_NAME = config["chunking"]["hf_embedding_model_name"]
SIMILARITY_TOP_K = config["retrieval"]["similarity_top_k"]

OLLAMA_LLM = config["ollama"]["llm"]

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