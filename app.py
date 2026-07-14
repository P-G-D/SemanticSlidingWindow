import os
import io
import time
import numpy as np
import streamlit as st
from llama_cpp import Llama
from fastembed import TextEmbedding
from pypdf import PdfReader

# --- Configuration ---
MODEL_PATH = "./models/qwen2.5-3b-instruct-q4_k_m.gguf"
MAX_CONTEXT_MESSAGES = 10 
KEEP_RECENT = 4           
KEEP_RELEVANT = 3         
MAX_FILE_SIZE_MB = 2
MAX_EXTRACTED_CHARS = 4000 

# --- 1. Cache Models ---
@st.cache_resource
def load_models():
    st.info("Loading local AI models into memory... (This takes ~10 seconds the first time)")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048, 
        n_threads=os.cpu_count(),
        verbose=False
    )
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    st.success("Models loaded successfully! Ready to chat.")
    return llm, embedder

llm, embedder = load_models()

# --- 2. Semantic Compression Logic ---
def get_embeddings(texts):
    return np.array(list(embedder.embed(texts)))

def compress_memory(messages, current_prompt):
    recent_msgs = messages[-KEEP_RECENT:]
    history_msgs = messages[:-KEEP_RECENT]
    
    if not history_msgs:
        return messages

    prompt_emb = get_embeddings([current_prompt])
    history_texts = [f"{m['role']}: {m['content']}" for m in history_msgs]
    history_embs = get_embeddings(history_texts)
    
    prompt_norm = prompt_emb / np.linalg.norm(prompt_emb, axis=1, keepdims=True)
    history_norm = history_embs / np.linalg.norm(history_embs, axis=1, keepdims=True)
    similarities = np.dot(prompt_norm, history_norm.T)[0]
    
    relevant_indices = np.argsort(similarities)[-KEEP_RELEVANT:][::-1]
    kept_history = [history_msgs[i] for i in relevant_indices]
    discard_history = [history_msgs[i] for i in range(len(history_msgs)) if i not in relevant_indices]
    
    memory_block = ""
    if discard_history:
        discard_text = "\n".join([f"{m['role']}: {m['content']}" for m in discard_history])
        summary_prompt = f"Summarize the following conversation fragments into a single, dense memory block. Focus on facts and user preferences:\n\n{discard_text}\n\nSummary:"
        
        summary_response = llm.create_chat_completion(
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=150,
            temperature=0.3,
            stream=False
        )
        memory_block = summary_response['choices'][0]['message']['content'].strip()

    new_messages = []
    if memory_block:
        new_messages.append({"role": "system", "content": f"[Long-term Memory Context]: {memory_block}"})
    
    new_messages.extend(kept_history)
    new_messages.extend(recent_msgs)
    return new_messages

# --- 3. Document Processing Logic ---
def process_uploaded_file(uploaded_file):
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File is too large ({file_size_mb:.2f} MB). Maximum allowed is {MAX_FILE_SIZE_MB} MB to protect CPU RAM.")
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
            
        if len(text) > MAX_EXTRACTED_CHARS:
            st.warning(f"Document truncated to {MAX_EXTRACTED_CHARS} characters to fit CPU memory constraints.")
            text = text[:MAX_EXTRACTED_CHARS] + "\n\n[... Document truncated due to local CPU context limits ...]"
            
        return text.strip()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# --- 4. Streamlit UI Setup ---
st.set_page_config(page_title="SemanticSlidingWindow", page_icon="🧠", layout="centered")

st.title("🧠 SemanticSlidingWindow LLM")
st.caption("100% Local & Private | CPU-Optimized | Dynamic Memory Compression")

# Sidebar for File Upload
with st.sidebar:
    st.header("📄 Document Ingestion")
    st.caption(f"Max size: {MAX_FILE_SIZE_MB}MB | Types: .txt, .pdf")
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        with st.spinner("Parsing document locally..."):
            file_content = process_uploaded_file(uploaded_file)
            
        if file_content:
            st.success(f"✅ Loaded: {uploaded_file.name}")
            if "file_injected" not in st.session_state or st.session_state.file_injected != uploaded_file.name:
                st.session_state.messages.insert(0, {
                    "role": "system", 
                    "content": f"The user has uploaded a document named '{uploaded_file.name}'. Here is its content for context:\n\n{file_content}"
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I have read **{uploaded_file.name}**. You can now ask me questions about it! (Note: Due to local CPU limits, I am analyzing the first {MAX_EXTRACTED_CHARS} characters)."
                })
                st.session_state.file_injected = uploaded_file.name
    else:
        st.session_state.file_injected = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am a local, privacy-focused AI assistant. Upload a `.txt` or `.pdf` file in the sidebar, and I will read it locally without sending your data to the cloud!"}
    ]

# Display chat messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- 5. Chat Input & Generation ---
if prompt := st.chat_input("Ask a question or type a message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_to_process = st.session_state.messages.copy()
    
    if len(messages_to_process) > MAX_CONTEXT_MESSAGES:
        with st.status("🔄 Compressing context window semantically...", expanded=True) as status:
            st.write("Analyzing semantic relevance of past messages...")
            messages_to_process = compress_memory(messages_to_process, prompt)
            status.update(label="✅ Context compressed! Long-term memory retained.", state="complete")

    # Generate AI Response with Streaming & Performance Metrics
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        metrics_placeholder = st.empty() # Placeholder for the tokens/sec metric
        full_response = ""
        
        # Start high-precision timer
        start_time = time.perf_counter()
        
        stream = llm.create_chat_completion(
            messages=messages_to_process,
            max_tokens=512,
            temperature=0.7,
            stream=True,
            stop=["<|im_end|>"]
        )
        
        for chunk in stream:
            if chunk['choices'][0]['delta'].get('content'):
                token = chunk['choices'][0]['delta']['content']
                full_response += token
                response_placeholder.markdown(full_response + "▌")
        
        # End timer
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        # Calculate metrics (Estimating ~4 characters per token for English text)
        estimated_tokens = max(1, len(full_response) / 4.0)
        tokens_per_sec = estimated_tokens / elapsed_time if elapsed_time > 0 else 0
        
        # Remove the blinking cursor and show final text
        response_placeholder.markdown(full_response)
        
        # Display the performance metric right below the message
        metrics_placeholder.markdown(
            f"⚡ *Generated in **{elapsed_time:.2f}s** | ~**{tokens_per_sec:.1f} tokens/sec** (Local CPU Inference)*"
        )
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})