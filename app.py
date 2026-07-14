import os
import io
import time
import psutil
import numpy as np
import streamlit as st
from llama_cpp import Llama
from fastembed import TextEmbedding
from pypdf import PdfReader
from huggingface_hub import hf_hub_download

# --- 1. Hardware-Aware Configuration ---
def get_hardware_config():
    """Detects RAM and returns optimized limits."""
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    
    if total_ram_gb < 8:
        return {
            "n_ctx": 1024, "max_file_mb": 1, "max_chars": 2000, 
            "label": "Low Memory Mode (<8GB RAM)"
        }
    elif total_ram_gb <= 16:
        return {
            "n_ctx": 2048, "max_file_mb": 2, "max_chars": 4000, 
            "label": "Balanced Mode (8-16GB RAM)"
        }
    else:
        return {
            "n_ctx": 4096, "max_file_mb": 5, "max_chars": 8000, 
            "label": "High Performance Mode (>16GB RAM)"
        }

HW_CONFIG = get_hardware_config()
MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_PATH = os.path.join("models", MODEL_FILE)

# --- 2. Auto-Download Logic ---
@st.cache_resource
def ensure_model_exists():
    """Checks for the model and downloads it if missing."""
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Model not found. Downloading {MODEL_FILE} (~2GB) from HuggingFace...")
        os.makedirs("models", exist_ok=True)
        try:
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                local_dir="models",
                local_dir_use_symlinks=False
            )
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            st.stop()
    else:
        st.info(f"✅ Found existing model at {MODEL_PATH}")
    return MODEL_PATH

# --- 3. Load Models ---
@st.cache_resource
def load_models(model_path):
    llm = Llama(
        model_path=model_path,
        n_ctx=HW_CONFIG["n_ctx"], 
        n_threads=os.cpu_count(),
        verbose=False
    )
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return llm, embedder

model_path = ensure_model_exists()
llm, embedder = load_models(model_path)

# --- 4. Semantic Compression & Document Logic (Same as before, using HW_CONFIG) ---
def get_embeddings(texts):
    return np.array(list(embedder.embed(texts)))

def compress_memory(messages, current_prompt):
    recent_msgs = messages[-4:] # Fixed keep recent
    history_msgs = messages[:-4]
    if not history_msgs: return messages

    prompt_emb = get_embeddings([current_prompt])
    history_texts = [f"{m['role']}: {m['content']}" for m in history_msgs]
    history_embs = get_embeddings(history_texts)
    
    prompt_norm = prompt_emb / np.linalg.norm(prompt_emb, axis=1, keepdims=True)
    history_norm = history_embs / np.linalg.norm(history_embs, axis=1, keepdims=True)
    similarities = np.dot(prompt_norm, history_norm.T)[0]
    
    relevant_indices = np.argsort(similarities)[-3:][::-1]
    kept_history = [history_msgs[i] for i in relevant_indices]
    discard_history = [history_msgs[i] for i in range(len(history_msgs)) if i not in relevant_indices]
    
    memory_block = ""
    if discard_history:
        discard_text = "\n".join([f"{m['role']}: {m['content']}" for m in discard_history])
        summary_prompt = f"Summarize the following conversation fragments into a single, dense memory block. Focus on facts:\n\n{discard_text}\n\nSummary:"
        summary_response = llm.create_chat_completion(
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=150, temperature=0.3, stream=False
        )
        memory_block = summary_response['choices'][0]['message']['content'].strip()

    new_messages = []
    if memory_block:
        new_messages.append({"role": "system", "content": f"[Long-term Memory Context]: {memory_block}"})
    new_messages.extend(kept_history)
    new_messages.extend(recent_msgs)
    return new_messages

def process_uploaded_file(uploaded_file):
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > HW_CONFIG["max_file_mb"]:
        st.error(f"File too large ({file_size_mb:.2f} MB). Limit is {HW_CONFIG['max_file_mb']} MB for your hardware.")
        return None

    try:
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        else:
            st.error("Unsupported file type.")
            return None
            
        if len(text) > HW_CONFIG["max_chars"]:
            st.warning(f"Document truncated to {HW_CONFIG['max_chars']} chars based on your RAM.")
            text = text[:HW_CONFIG["max_chars"]] + "\n\n[... Truncated ...]"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# --- 5. Streamlit UI ---
st.set_page_config(page_title="SemanticSlidingWindow", page_icon="🧠", layout="centered")
st.title("🧠 SemanticSlidingWindow LLM")
st.caption(f"Mode: {HW_CONFIG['label']} | 100% Local & Private")

with st.sidebar:
    st.header("📄 Document Ingestion")
    
    # Hide Streamlit's default "200MB per file" text using CSS
    st.markdown("""
        <style>
        .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] > div > small {
            visibility: hidden;
        }
        </style>
    """, unsafe_allow_html=True)

    st.caption(f"App limit: {HW_CONFIG['max_file_mb']}MB to protect RAM (Widget default is 200MB)")
    
    # Updated uploader with explicit label and helpful tooltip
    uploaded_file = st.file_uploader(
        label="Upload a document",
        type=["txt", "pdf"],
        help=f"Strictly limited to {HW_CONFIG['max_file_mb']}MB to protect your CPU RAM from Out-Of-Memory errors.",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        with st.spinner("Parsing document locally..."):
            file_content = process_uploaded_file(uploaded_file)
        if file_content:
            st.success(f"✅ Loaded: {uploaded_file.name}")
            if "file_injected" not in st.session_state or st.session_state.file_injected != uploaded_file.name:
                st.session_state.messages.insert(0, {"role": "system", "content": f"User uploaded '{uploaded_file.name}'. Content:\n\n{file_content}"})
                st.session_state.messages.append({"role": "assistant", "content": f"I've read **{uploaded_file.name}**. Ask me anything about it!"})
                st.session_state.file_injected = uploaded_file.name

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm running locally on your CPU. How can I help?"}]

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_to_process = st.session_state.messages.copy()
    if len(messages_to_process) > 10:
        with st.status("🔄 Compressing context...", expanded=True) as status:
            messages_to_process = compress_memory(messages_to_process, prompt)
            status.update(label="✅ Context compressed!", state="complete")

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        metrics_placeholder = st.empty()
        full_response = ""
        start_time = time.perf_counter()
        
        stream = llm.create_chat_completion(messages=messages_to_process, max_tokens=512, temperature=0.7, stream=True, stop=["<|im_end|>"])
        for chunk in stream:
            if chunk['choices'][0]['delta'].get('content'):
                token = chunk['choices'][0]['delta']['content']
                full_response += token
                response_placeholder.markdown(full_response + "▌")
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        tps = (len(full_response) / 4.0) / elapsed if elapsed > 0 else 0
        
        response_placeholder.markdown(full_response)
        metrics_placeholder.markdown(f"⚡ *{elapsed:.2f}s | ~{tps:.1f} tokens/sec*")
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})