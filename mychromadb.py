import chromadb
from sentence_transformers import SentenceTransformer

# Persistent ChromaDB client (stores data to disk)
client = chromadb.PersistentClient(path="./chroma_storage")

# Create or get collection
collection = client.get_or_create_collection(name="my_knowledge_base")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents
documents = ["ChromaDB is a vector store.", "RAG stands for Retrieval-Augmented Generation."]
ids = ["doc1", "doc2"]
embeddings = model.encode(documents).tolist()

# Add documents to collection
collection.add(documents=documents, embeddings=embeddings, ids=ids)

# Query
query = "What does RAG mean?"
query_embedding = model.encode([query]).tolist()
results = collection.query(query_embeddings=query_embedding, n_results=2)

# Print results
print("Retrieved Documents:")
for doc in results["documents"][0]:
    print("-", doc)

