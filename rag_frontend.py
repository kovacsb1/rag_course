import chainlit as cl
from llama_index.core import (
    Settings,
    VectorStoreIndex
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore

import os
import toml
from dotenv import load_dotenv
import qdrant_client

load_dotenv()

config = toml.load('config.toml')

HF_EMBEDDING_MODEL_NAME = config["chunking"]["hf_embedding_model_name"]
SIMILARITY_TOP_K = config["retrieval"]["similarity_top_k"]
SPARSE_TOP_K = config["retrieval"]["sparse_top_k"]

AZURE_OPENAI_MODEL= config["openai"]["model"]
AZURE_DEPLOYMENT_NAME = config["openai"]["azure_deployment"]
API_VERSION = config["openai"]["api_version"]

QDRANT_URL = config["qdrant"]["url"]
QDRANT_PORT = config["qdrant"]["port"]
QDRANT_COLLECTION = config["qdrant"]["collection_name"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

@cl.on_chat_start
async def start():
    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBEDDING_MODEL_NAME)

    # ollama
    Settings.llm = AzureOpenAI(
        model=AZURE_OPENAI_MODEL,
        deployment_name=AZURE_DEPLOYMENT_NAME,
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_version=API_VERSION,
    )

    vectordb_client = qdrant_client.QdrantClient(
        url=QDRANT_URL,
        port=QDRANT_PORT,
        api_key=QDRANT_API_KEY,
    )

    vector_store = QdrantVectorStore(client=vectordb_client, collection_name=QDRANT_COLLECTION, enable_hybrid=True)
    index = VectorStoreIndex.from_vector_store(vector_store)

    chat_engine = index.as_chat_engine(
        vector_store_query_mode="hybrid",
        similarity_top_k=SIMILARITY_TOP_K,
        sparse_top_k=SPARSE_TOP_K
    )
    cl.user_session.set("chat_engine", chat_engine)
    

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    chat_engine = cl.user_session.get("chat_engine")

    # create message object
    msg = cl.Message(content="", author="Assistant")

    res = await cl.make_async(chat_engine.stream_chat)(message.content)

    # updated message object with streaming tokens
    for token in res.response_gen:
        await msg.stream_token(token)

    print(res.source_nodes)
    await msg.send()