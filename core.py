import os
import io
import psutil
import numpy as np
from llama_cpp import Llama
from fastembed import TextEmbedding
from pypdf import PdfReader
from huggingface_hub import hf_hub_download

# --- Configuration ---
MODEL_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_PATH = os.path.join("models", MODEL_FILE)

# --- 1. Hardware-Aware Configuration ---
def get_hardware_config():
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    if total_ram_gb < 8:
        return {"n_ctx": 1024, "max_file_mb": 1, "max_chars": 2000, "label": "Low Memory (<8GB)"}
    elif total_ram_gb <= 16:
        return {"n_ctx": 2048, "max_file_mb": 2, "max_chars": 4000, "label": "Balanced (8-16GB)"}
    else:
        return {"n_ctx": 4096, "max_file_mb": 5, "max_chars": 8000, "label": "High Perf (>16GB)"}

# --- 2. Auto-Download Logic ---
def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found. Downloading {MODEL_FILE} (~2GB)...")
        os.makedirs("models", exist_ok=True)
        hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir="models", local_dir_use_symlinks=False)
        print("✅ Model downloaded!")
    return MODEL_PATH

# --- 3. Load Models ---
def load_models(hw_config):
    llm = Llama(model_path=MODEL_PATH, n_ctx=hw_config["n_ctx"], n_threads=os.cpu_count(), verbose=False)
    embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return llm, embedder

# --- 4. Semantic Compression Logic ---
def compress_memory(llm, embedder, messages, current_prompt):
    recent_msgs = messages[-4:]
    history_msgs = messages[:-4]
    if not history_msgs: return messages

    prompt_emb = np.array(list(embedder.embed([current_prompt])))
    history_texts = [f"{m['role']}: {m['content']}" for m in history_msgs]
    history_embs = np.array(list(embedder.embed(history_texts)))
    
    prompt_norm = prompt_emb / np.linalg.norm(prompt_emb, axis=1, keepdims=True)
    history_norm = history_embs / np.linalg.norm(history_embs, axis=1, keepdims=True)
    similarities = np.dot(prompt_norm, history_norm.T)[0]
    
    relevant_indices = np.argsort(similarities)[-3:][::-1]
    kept_history = [history_msgs[i] for i in relevant_indices]
    discard_history = [history_msgs[i] for i in range(len(history_msgs)) if i not in relevant_indices]
    
    memory_block = ""
    if discard_history:
        discard_text = "\n".join([f"{m['role']}: {m['content']}" for m in discard_history])
        summary_prompt = f"Summarize these fragments into a dense memory block:\n\n{discard_text}\n\nSummary:"
        summary_response = llm.create_chat_completion(messages=[{"role": "user", "content": summary_prompt}], max_tokens=150, temperature=0.3, stream=False)
        memory_block = summary_response['choices'][0]['message']['content'].strip()

    new_messages = []
    if memory_block:
        new_messages.append({"role": "system", "content": f"[Long-term Memory]: {memory_block}"})
    new_messages.extend(kept_history)
    new_messages.extend(recent_msgs)
    return new_messages

# --- 5. Document Processing ---
def process_uploaded_file(uploaded_file_bytes, file_name, file_size_bytes, hw_config):
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > hw_config["max_file_mb"]:
        raise ValueError(f"File too large ({file_size_mb:.2f} MB). Limit is {hw_config['max_file_mb']} MB.")

    if file_name.endswith(".txt"):
        text = uploaded_file_bytes.decode("utf-8")
    elif file_name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded_file_bytes))
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        raise ValueError("Unsupported file type.")
        
    if len(text) > hw_config["max_chars"]:
        text = text[:hw_config["max_chars"]] + "\n\n[... Truncated due to hardware limits ...]"
    return text.strip()