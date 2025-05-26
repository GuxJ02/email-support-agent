# get_embedding_function.py

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_function():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 128, "normalize_embeddings": True}
    )
