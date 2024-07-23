import logging
import sys
import os
from IPython.display import Markdown, display

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/llamalog.txt", level=logging.DEBUG)

PERSIST_DIR = "./storage"

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(model="phi3", request_timeout=360.0)


# load the existing index
storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
index = load_index_from_storage(storage_context)


# configure retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever
)

response = query_engine.query("What did the author do growing up?")
print(response)

print("Chunks used in creating the response:")
for n in response.source_nodes:
    print(n)