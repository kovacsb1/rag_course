# rag_course
Repository for Starschema RAG course

## How to run?
Ideally, first create a separate Python virtual environment(Python 3.12.4 is recommended), and install the required packages.
```
python -m pip install requirements.txt
```
Make sure to have a `.env` file in the project root that contains `QDRANT_API_KEY`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_ENDPOINT`

### Ingestion
The ingestion pipeline uploads hybrid embeddding vectors to the Qdrant vectorDB specified in `config.toml` [qdrant] section. This requires a running Qdrant instance at the specified IP. Also change the collection name. You can upload custom data to the data folder, or change `persistance.source_dir` in the `config.toml` file.

After all this, you can run the ingestion pipeline. There are other relevant hyperparamteres to tweak in the config file, like chunk size and embedding model.
```
python ingestion.py
```

### The RAG frontend
The fronted can be run with chainlit.
```
python -m chainlit run rag_frontend.py
```

