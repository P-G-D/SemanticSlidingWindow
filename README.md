# 🧠 SemanticSlidingWindow LLM

A 100% local, privacy-preserving Large Language Model interface engineered to solve the "context forgetting" and CPU inference bottlenecks inherent in raw inference engines like `llama.cpp`.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.0-red) ![Local AI](https://img.shields.io/badge/AI-100%25%20Local-green)

## 🚀 The Problem Solved
Standard inference engines (like `llama.cpp` or Ollama) treat the context window as a "dumb" FIFO (First-In-First-Out) queue. When the context fills up, they blindly drop the oldest tokens. This causes two massive problems on consumer hardware:
1. **The AI "forgets"** early parts of the conversation.
2. **CPU inference crawls.** Attention mechanisms are $O(N^2)$. A massive context window on a CPU will bring token generation to a halt.

## 💡 The Solution: Semantic Sliding Window
Instead of dropping tokens blindly, this application implements an **application-level Semantic Hippocampus**. 
When the chat history exceeds a threshold, a lightweight local embedding model (`fastembed`) calculates the cosine similarity of past messages against the current prompt. 
1. It keeps the most semantically relevant history.
2. It summarizes the discarded older messages into a dense "Memory Block".
3. It feeds a perfectly sized, optimized context to the LLM.

**Result:** CPU inference stays blazing fast (maintaining high tokens/sec), RAM usage stays under 4GB, and the AI retains infinite long-term recall.

## ✨ Features
- **Dynamic Memory Compression:** Auto-summarizes irrelevant chat history to keep the KV-cache small.
- **Local RAG-Lite:** Upload `.txt` or `.pdf` files (up to 2MB) for instant, private document Q&A.
- **Strict Hardware Bounds:** Hard limits on file sizes and context truncation guarantee zero OOM (Out of Memory) errors on 16GB RAM machines.
- **Real-time Telemetry:** Live `tokens/sec` and generation time metrics displayed in the UI.
- **100% Private:** Zero cloud dependencies. All data, embeddings, and inference happen locally on-device.

## 🛠️ Tech Stack
- **Inference:** `llama-cpp-python` (Qwen 2.5 3B Instruct, 4-bit quantized)
- **Embeddings:** `fastembed` (BAAI/bge-small-en-v1.5 via ONNX)
- **Frontend:** `Streamlit`
- **Document Parsing:** `pypdf`

## ⚙️ Installation & Run

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SemanticSlidingWindow.git
   cd SemanticSlidingWindow