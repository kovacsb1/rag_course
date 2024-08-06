from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

import logging
import os
from dotenv import load_dotenv
import toml

# create log folder
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/ingest_log.txt", level=logging.DEBUG)

config = toml.load('config.toml')
SOURCE_DIR = config["persistance"]["source_dir"]
PERSIST_DIR = config["persistance"]["persist_dir"]

HF_EMBEDDING_MODEL_NAME = config["chunking"]["hf_embedding_model_name"]
CHUNK_SIZE = config["chunking"]["chunk_size"]
CHUNK_OVERLAP = config["chunking"]["chunk_overlap"]


# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL_NAME)


# load the documents and create the index
splitter = SentenceSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

documents = SimpleDirectoryReader(SOURCE_DIR).load_data()
# nodes = splitter.get_nodes_from_documents(documents)
index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
# sytore it for later
index.storage_context.persist(persist_dir=PERSIST_DIR)