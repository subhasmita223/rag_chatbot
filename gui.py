import streamlit as st
import os
import Data_Ingestion as DataIng  # your ingestion module
import geminiAPI as GAPI          # your chatbot module

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="RAG Chatbot")

UPLOAD_FOLDER = os.path.join(os.getcwd(), "ALL_Docs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------
# SESSION STATE SETUP
# -----------------------
if "saved_files" not in st.session_state:
    st.session_state.saved_files = []

if "documents_ingested" not in st.session_state:
    st.session_state.documents_ingested = False

if "query_to_process" not in st.session_state:
    st.session_state.query_to_process = ""

# -----------------------
# FILE UPLOAD
# -----------------------
st.header("Upload PDFs for Chatbot")

uploaded_files = st.file_uploader(
    "Upload PDFs", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if file.name not in st.session_state.saved_files:
            st.session_state.saved_files.append(file.name)

    st.success("✅ Files saved!")

# -----------------------
# DEBUG: Show uploaded files
# -----------------------
st.write("Debug: saved files in session:", st.session_state.saved_files)
st.write("Debug: Upload folder exists?", os.path.exists(UPLOAD_FOLDER))
st.write("Debug: Files in upload folder:", os.listdir(UPLOAD_FOLDER))

# -----------------------
# INGESTION
# -----------------------
if st.session_state.saved_files and not st.session_state.documents_ingested:
    with st.spinner("Ingesting documents..."):
        try:
            DataIng.ingest_documents(folder_path=UPLOAD_FOLDER)
            st.session_state.documents_ingested = True
            st.success("✅ Documents ingested!")
        except Exception as e:
            st.exception(e)

# -----------------------
# USER QUERY
# -----------------------
st.header("Ask the Chatbot")
user_input = st.text_input("Type your question here:")

if st.button("Send"):
    if not st.session_state.saved_files:
        st.warning("Please upload at least one PDF before chatting.")
    elif not st.session_state.documents_ingested:
        st.warning("Documents not ingested yet. Please wait or re-upload.")
    else:
        st.session_state.query_to_process = user_input

        # Step 1: Get top document chunks from ChromaDB
        try:
            data_chunks = DataIng.query_chromadb(user_input)
            st.write("Debug: Data Chunks:", data_chunks)
        except Exception as e:
            st.exception(e)
            data_chunks = ""

        # Step 2: Generate bot response using Gemini API
        if data_chunks:
            try:
                response = GAPI.generate_output(user_input, data_chunks)
                st.write("**Bot Response:**", response)
            except Exception as e:
                st.exception(e)

# -----------------------
# OPTIONAL: Clear session
# -----------------------
if st.button("Clear Session"):
    st.session_state.saved_files = []
    st.session_state.documents_ingested = False
    st.session_state.query_to_process = ""
    st.experimental_rerun()
