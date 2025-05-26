# insertarBD.py

import argparse
import os
import shutil
import time
import torch
from threading import Thread
from queue import Queue

from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chromaDB"
DATA_PATH   = "Data"
MAX_BATCH   = 5461  # Chroma tolera ~5 k docs por lote

def load_documents():
    docs = []
    for fn in os.listdir(DATA_PATH):
        if fn.lower().endswith(".txt"):
            docs.extend(
                TextLoader(os.path.join(DATA_PATH, fn), encoding="utf-8").load()
            )
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = 800,    # como tenías antes
        chunk_overlap   = 80,
        separators      = ["\n\n", "\n", ". "],
        length_function = len
    )
    return splitter.split_documents(docs)

def calculate_chunk_ids(chunks):
    last_pid = None
    idx = 0
    for c in chunks:
        src = c.metadata.get("source", "")
        pg  = c.metadata.get("page", 0)
        pid = f"{src}:{pg}"
        idx = idx + 1 if pid == last_pid else 0
        c.metadata["id"] = f"{pid}:{idx}"
        last_pid = pid
    return chunks

def add_to_chroma(chunks):
    chunks = calculate_chunk_ids(chunks)
    embedder = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedder
    )

    existing = set(db.get()["ids"])
    new_chunks = [c for c in chunks if c.metadata["id"] not in existing]
    print(f"Existing in DB: {len(existing)}  |  New to add: {len(new_chunks)}")

    start_time = time.time()
    q = Queue(maxsize=2)

    def consumer():
        while True:
            batch, ids, embs = q.get()
            if batch is None:
                q.task_done()
                break
            db.add_documents(batch, embeddings=embs, ids=ids)
            q.task_done()

    Thread(target=consumer, daemon=True).start()

    for i in range(0, len(new_chunks), MAX_BATCH):
        batch = new_chunks[i : i + MAX_BATCH]
        texts = [c.page_content for c in batch]
        ids   = [c.metadata["id"]   for c in batch]

        t0 = time.time()
        embs = embedder.embed_documents(texts)
        if torch.cuda.is_available():
          torch.cuda.synchronize()
        t1 = time.time()

        q.put((batch, ids, embs))
        print(f"Batch {i}-{i+len(batch)}: embed {t1-t0:.1f}s")

    q.put((None, None, None))
    q.join()

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    print(f"✅ Indexing complete in {mins}m {secs}s (Chroma persists).")

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Clear DB before indexing")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    docs   = load_documents()
    chunks = split_documents(docs)
    add_to_chroma(chunks)

if __name__ == "__main__":
    main()
