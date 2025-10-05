import os
import json
import glob
import chromadb
from sentence_transformers import SentenceTransformer
import Data_Extraction as DataExtr
from dotenv import load_dotenv

load_dotenv()

def ingest_documents(folder_path=None, persist_dir=None, state_file=None):
    import os
    folder_path = folder_path or os.path.join(os.getcwd(), "ALL_Docs")
    persist_dir = persist_dir or os.path.join(os.getcwd(), "chroma_storage")
    state_file = state_file or os.path.join(os.getcwd(), "state.json")

    os.makedirs(persist_dir, exist_ok=True)  # ensure vector DB folder exists


    # Compute hash of documents
    current_hash = DataExtr.compute_docs_hash(folder_path)

    # Check previous state
    if os.path.exists(state_file) and os.path.exists(persist_dir):
        with open(state_file, "r") as f:
            saved_hash = json.load(f).get("hash")
        if saved_hash == current_hash:
            print("✅ No changes in documents — skipping ingestion.")
            return
        else:
            print("🔄 Detected changes — re-ingesting documents.")
    else:
        print("🆕 No previous state found — performing initial ingestion.")

    # Setup ChromaDB
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete old collection if exists
    if "my_docs" in [c.name for c in client.list_collections()]:
        client.delete_collection("my_docs")

    collection = client.create_collection(name="my_docs")
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    doc_files = glob.glob(os.path.join(folder_path, "*.*"))
    doc_id = 0

    for file in doc_files:
        text = DataExtr.load_text_from_file(file)
        if not text:
            continue

        chunks = DataExtr.chunk_text(text)
        embeddings = model.encode(chunks).tolist()
        ids = [f"doc{doc_id + i}" for i in range(len(chunks))]

        collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        print(f"📄 Indexed {len(chunks)} chunks from {os.path.basename(file)}")
        doc_id += len(chunks)

    # Save new hash
    with open(state_file, "w") as f:
        json.dump({"hash": current_hash}, f)

    print("✅ Ingestion complete.\n")


def query_chromadb(user_query, persist_dir=None, top_k=5):
    persist_dir = persist_dir or os.getenv("VECTOR_DB_PATH") or os.path.join(os.getcwd(), "chroma_storage")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="my_docs")
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    query_embedding = model.encode([user_query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    top_chunks = results["documents"][0]
    return "\n".join(top_chunks)


# TESTING MODULE
if __name__ == "__main__":
    ingest_documents()
    print(query_chromadb("What documents have to be submitted with the application?"))
