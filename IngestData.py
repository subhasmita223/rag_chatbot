import os
import glob
import json
import chromadb
from sentence_transformers import SentenceTransformer
import Data_Extraction as DataExtr
import Data_Ingestion as DataIng



# ---------- Ingestion ----------

def ingest_documents(folder_path="All_Docs", persist_dir="chroma_storage", state_file="ingest_state.json"):
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

    print("✅ Ingestion complete.")


# ---------- Querying ----------

def query_chromadb(user_query, persist_dir="chroma_storage", top_k=10):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="my_docs")
    model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    query_embedding = model.encode([user_query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    print(f"\n🔍 Top {top_k} results with distance ≤ 1.1:\n")

    has_results = False
    for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0]), 1):
        if dist <= 2:
            has_results = True
            print(f"{i}. Distance: {dist:.4f}")
            print(doc + ("..." if len(doc) > 500 else ""))
            print()

    if not has_results:
        print("❌ No relevant results found with distance ≤ 1.1.")



# ---------- Main Loop ----------

if __name__ == "__main__":
    ingest_documents()

    while True:
        user_input = input("\nAsk a question (or type 'exit'): ")
        if user_input.lower() in ["exit", "quit", "e", "q"]:
            break
        query_chromadb(user_input)
