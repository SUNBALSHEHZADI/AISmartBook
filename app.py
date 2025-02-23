import streamlit as st
st.set_page_config(page_title="AI Smart Book Analyzer", layout="wide")  # Must be the first Streamlit command

import torch
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF extraction
import docx2txt  # For DOCX extraction
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------
# Configuration
# ------------------------
MODEL_NAME = "ibm-granite/granite-3.1-1b-a400m-instruct"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------
# Model Loading with Caching
# ------------------------
@st.cache_resource
def load_models():
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            revision="main"
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            revision="main",
            device_map="auto" if DEVICE == "cuda" else None,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).eval()
        embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
        return tokenizer, model, embedder
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

tokenizer, model, embedder = load_models()

# ------------------------
# Text Processing Functions
# ------------------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    return splitter.split_text(text)

def extract_text(file):
    file_type = file.type
    if file_type == "application/pdf":
        try:
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        except Exception as e:
            st.error("Error processing PDF: " + str(e))
            return ""
    elif file_type == "text/plain":
        return file.read().decode("utf-8")
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            return docx2txt.process(file)
        except Exception as e:
            st.error("Error processing DOCX: " + str(e))
            return ""
    else:
        st.error("Unsupported file type: " + file_type)
        return ""

def build_index(chunks):
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# ------------------------
# Summarization and Q&A Functions
# ------------------------
def generate_summary(text):
    # Limit input text to avoid long sequences
    prompt = f"<|user|>\nSummarize the following book in a concise and informative paragraph:\n\n{text[:4000]}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.5)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary.split("<|assistant|>")[-1].strip() if "<|assistant|>" in summary else summary.strip()

def generate_answer(query, context):
    prompt = f"<|user|>\nUsing the context below, answer the following question precisely. If unsure, say 'I don't know'.\n\nContext: {context}\n\nQuestion: {query}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("<|assistant|>")[-1].strip() if "<|assistant|>" in answer else answer.strip()

# ------------------------
# Streamlit UI
# ------------------------
st.title("AI Smart Book Analyzer")
st.write("Upload a book (PDF, TXT, DOCX) to get a summary and ask questions about its content.")

uploaded_file = st.file_uploader("Upload File", type=["pdf", "txt", "docx"])

if uploaded_file:
    text = extract_text(uploaded_file)
    if text:
        st.success("File successfully processed!")
        st.write("Generating summary...")
        summary = generate_summary(text)
        st.markdown("### Book Summary")
        st.write(summary)
        
        # Process text into chunks and build FAISS index
        chunks = split_text(text)
        index = build_index(chunks)
        st.session_state.chunks = chunks
        st.session_state.index = index
        
        st.markdown("### Ask a Question about the Book:")
        query = st.text_input("Your Question:")
        if query:
            # Retrieve top 3 relevant chunks as context
            query_embedding = embedder.encode([query])
            faiss.normalize_L2(query_embedding)
            distances, indices = st.session_state.index.search(query_embedding, k=3)
            retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
            context = "\n".join(retrieved_chunks)
            answer = generate_answer(query, context)
            st.markdown("### Answer")
            st.write(answer)
