import Data_Ingestion as DE
import geminiAPI as GAPI

# Step 1: Make sure all documents are ingested
DE.ingest_documents()

# Step 2: Ask a question
user_query = input("Ask a question: ")

# Step 3: Retrieve relevant chunks from ChromaDB
data_chunks = DE.query_chromadb(user_query)

# Step 4: Generate response using Gemini API
response = GAPI.generate_output(user_query, data_chunks)

# Step 5: Print the bot answer
print("\nBot Answer:\n", response)
