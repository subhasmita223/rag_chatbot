import streamlit as st
import os
import shutil
import Data_Ingestion as DataIng
import geminiAPI as GAPI

from streamlit.runtime.scriptrunner import RerunException, RerunData

def rerun():
    raise RerunException(RerunData())

st.set_page_config(page_title="RAG CHATBOT", layout="centered")
st.markdown("<h1 style='text-align: center;'>RAG CHATBOT</h1>", unsafe_allow_html=True)

UPLOAD_FOLDER = "Uploaded_Files"

# Initialize session state once
if "initialized" not in st.session_state:
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    st.session_state.saved_files = []
    st.session_state.chat_history = []
    st.session_state.documents_ingested = False
    st.session_state.query_to_process = None
    st.session_state.loading = False
    st.session_state.initialized = True
    st.session_state.input_text = ""

# Handle file upload
uploaded_files = st.file_uploader(
    "Upload Data Files (Drag and Drop or Click To Select)",
    accept_multiple_files=True,
)

current_filenames = [f.name for f in uploaded_files] if uploaded_files else []

# Remove deleted files from session and disk
for filename in st.session_state.saved_files[:]:
    if filename not in current_filenames:
        path_to_remove = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)
            st.session_state.saved_files.remove(filename)
            st.session_state.documents_ingested = False
        except Exception as e:
            st.warning(f"Failed to remove file {filename}: {e}")

# Save newly uploaded files
if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.saved_files:
            save_path = os.path.join(UPLOAD_FOLDER, file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            st.session_state.saved_files.append(file.name)
            st.session_state.documents_ingested = False
            st.success(f"Saved file: {file.name}")


# Show chat history
st.markdown("### Chat History")
if st.session_state.chat_history:
    for user_msg, bot_msg in st.session_state.chat_history:
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**Bot:** {bot_msg}")
        st.markdown("---")
else:
    st.info("No messages yet. Start by typing your question below.")

# Create form with columns to place input and button side by side
with st.form(key="input_form", clear_on_submit=False):
    col1, col2 = st.columns([8, 1])  # Adjust ratios as needed

    with col1:
        user_input = st.text_input(
            label="Your message",
            value=st.session_state.input_text,
            placeholder="Enter your query...",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Send")

    if submitted:
        user_input = user_input.strip()
        if not user_input:
            st.warning("Please enter a query before sending.")
        elif not st.session_state.saved_files:
            st.warning("Please upload at least one data file before chatting.")
        else:
            st.session_state.query_to_process = user_input
            st.session_state.input_text = ""
            rerun()

# Keep input_text synced to avoid auto-fill
st.session_state.input_text = user_input

# Process query outside the callback to show spinner
if st.session_state.query_to_process:
    #with st.spinner("Processing your query..."):
    try:
        if not st.session_state.documents_ingested:
            with st.spinner("Extracting and Embedding Data From Files..."):
                DataIng.ingest_documents()
                st.session_state.documents_ingested = True
        with st.spinner("Fetching Relevent Data Chunks..."):
            data = DataIng.query_chromadb(st.session_state.query_to_process)
        with st.spinner("Generating Response..."):
            bot_response = GAPI.generate_output(st.session_state.query_to_process, data)

        st.session_state.chat_history.append((st.session_state.query_to_process, bot_response))

    except Exception as e:
        st.error(f"Error: {e}")

    # Reset query_to_process AFTER processing
    st.session_state.query_to_process = None
    rerun()

# Smooth scroll to bottom of chat history
st.markdown('<div id="scroll-to-bottom"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <script>
    const elem = document.getElementById('scroll-to-bottom');
    if (elem) {
        elem.scrollIntoView({behavior: 'smooth', block: 'end'});
    }
    </script>
    """,
    unsafe_allow_html=True,
)


