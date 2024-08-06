import chainlit as cl
from llama_index.core import (
    Settings,
    VectorStoreIndex
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore

import os
import toml
from dotenv import load_dotenv
import qdrant_client


config = toml.load('config.toml')

HF_EMBEDDING_MODEL_NAME = config["chunking"]["hf_embedding_model_name"]
SIMILARITY_TOP_K = config["retrieval"]["similarity_top_k"]
SPARSE_TOP_K = config["retrieval"]["sparse_top_k"]

OLLAMA_LLM = config["ollama"]["llm"]

QDRANT_URL = config["qdrant"]["url"]
QDRANT_PORT = config["qdrant"]["port"]
QDRANT_COLLECTION = config["qdrant"]["collection_name"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

@cl.on_chat_start
async def start():
    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL_NAME)

    # ollama
    Settings.llm = Ollama(model=OLLAMA_LLM, request_timeout=360.0)


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
    cl.user_session.set("query_engine", query_engine)
    

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    query_engine = cl.user_session.get("query_engine")

    response = query_engine.query(message.content)

    # Send a response back to the user
    await cl.Message(
        content=response.response,
    ).send()
