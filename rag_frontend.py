import chainlit as cl
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

import os
import toml
from dotenv import load_dotenv


config = toml.load('config.toml')
PERSIST_DIR = config["persistance"]["persist_dir"]

HF_EMBEDDING_MODEL_NAME = config["chunking"]["hf_embedding_model_name"]
SIMILARITY_TOP_K = config["retrieval"]["similarity_top_k"]

OLLAMA_LLM = config["ollama"]["llm"]


@cl.on_chat_start
async def start():
    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL_NAME)

    # ollama
    Settings.llm = Ollama(model=OLLAMA_LLM, request_timeout=360.0)


    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)
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
