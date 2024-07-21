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
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


# define prompt viewing function
def display_prompt_dict(prompts_dict):
    for k, p in prompts_dict.items():
        text_md = f"**Prompt Key**: {k}<br>" f"**Text:** <br>"
        display(Markdown(text_md))
        print(p.get_template())
        display(Markdown("<br><br>"))

logging.basicConfig(filename="llamalog.txt", level=logging.DEBUG)

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(model="phi3", request_timeout=360.0)


PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # set chunking parameters
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=20,
    )
     
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
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

prompts_dict = query_engine.get_prompts()
display_prompt_dict(prompts_dict)

response = query_engine.query("What did the author do growing up?")
print(response)

print("Chunks used in creating the response:")
for n in response.source_nodes:
    print(n)