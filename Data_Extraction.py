import os
import glob
import hashlib
import pdfplumber
from docx import Document
# import nltk
# from nltk.tokenize import sent_tokenize, word_tokenize
import re




# Checks Weather Files Are Modified Or Not By Calculating Their MD5 Hash - Based on Filenames and Last Modification Time

def compute_docs_hash(folder_path):
    hash_md5 = hashlib.md5()
    for file in sorted(glob.glob(os.path.join(folder_path, "*.*"))):
        stat = os.stat(file)
        hash_md5.update(f"{file}{stat.st_mtime}".encode())
    return hash_md5.hexdigest()


def extract_text_pdf(pdf_path):
    full_text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += '\n' + page_text
    return full_text



# Extract Text Data From .txt / .pdf / .docx Files

def load_text_from_file(file_path):
    ext = file_path.lower().split('.')[-1]
    text = ""

    if ext == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    elif ext == "pdf":
        text = extract_text_pdf(file_path)

    elif ext == "docx":
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text




# Create Chunks of Extracted Text Data From Files

# Downloading NLTK - Natural Language Toolkit for Chunking Sentences

# if(os.path.exists("./.venv")):
#     nltk.download('punkt', download_dir='./.venv/nltk_data', quiet=True)
#     nltk.download('punkt_tab', download_dir='./.venv/nltk_data', quiet=True)
# else:
#     nltk.download('punkt', quiet=True)
#     nltk.download('punkt_tab', quiet=True)


# Sentences and Words Tokenization Based Chunking

# def chunk_text(text, max_words_per_chunk=150):
#     sentences = sent_tokenize(text)
#     chunks = []
#     current_chunk = []
#     current_word_count = 0

#     for sentence in sentences:
#         # Clean sentence by tokenizing into words and rejoining with single spaces
#         words = word_tokenize(sentence)
#         cleaned_sentence = " ".join(words)
#         word_count = len(words)

#         if current_word_count + word_count <= max_words_per_chunk:
#             current_chunk.append(cleaned_sentence)
#             current_word_count += word_count
#         else:
#             # Save current chunk
#             chunks.append(" ".join(current_chunk))
#             # Start new chunk
#             current_chunk = [cleaned_sentence]
#             current_word_count = word_count

#     # Add last chunk
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks

def chunk_text(text):                         # split_into_qa_chunks
    # Pattern to match various question indicators
    # - Q1. / Q2. / Q. / 1. / 2. / I. / II. etc
    pattern = r'(?=\b(?:Q\d*\.|Q\.|Q\:*\-|\d{1,2}\.|[IVXLCDM]{1,4}\.)(?=\s))'

   
    chunks = re.split(pattern, text)
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks



"""Only Sentence Tokenizer"""
# def chunk_text(text, max_words_per_chunk=100):
#     sentences = sent_tokenize(text)
#     chunks = []
#     current_chunk = []

#     current_word_count = 0
#     for sentence in sentences:
#         word_count = len(sentence.split())

#         if current_word_count + word_count <= max_words_per_chunk:
#             current_chunk.append(sentence)
#             current_word_count += word_count
#         else:
#             # Save current chunk
#             chunks.append(" ".join(current_chunk))
#             # Start new chunk
#             current_chunk = [sentence]
#             current_word_count = word_count

#     # Add last chunk
#     if current_chunk:
#         chunks.append(" ".join(current_chunk))

#     return chunks





# Testing File Modules

if __name__ == "__main__":
    print("All Document Hash:", compute_docs_hash("All_Docs"), "\n\n")

    sample_text = load_text_from_file("All_Docs/oci-faq.pdf")


    chunks = chunk_text(sample_text)    

    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---")
        print(chunk)
        print()
    