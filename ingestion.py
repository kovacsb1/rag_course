from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

import logging
import os

# create log folder
os.makedirs("logs", exist_ok=True)

logging.basicConfig(filename="logs/ingest_log.txt", level=logging.DEBUG)

PERSIST_DIR = "./storage"

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")


# load the documents and create the index
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=20,
)

documents = SimpleDirectoryReader("data").load_data()
# nodes = splitter.get_nodes_from_documents(documents)
index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
# sytore it for later
index.storage_context.persist(persist_dir=PERSIST_DIR)