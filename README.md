# 🧠 SemanticSlidingWindow: Hardware-Aware Local LLM Engine

> **A privacy-first, CPU-optimized Large Language Model interface featuring dynamic semantic memory compression and hardware-adaptive resource management.**

![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python) 
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit) 
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Project Overview

Standard local inference engines (like `llama.cpp`) treat the context window as a static FIFO queue. This leads to two critical failures on consumer hardware:
1.  **Semantic Amnesia:** As the context fills, the model blindly discards early conversation history.
2.  **Inference Degradation:** Attention mechanisms scale at $O(N^2)$. Large contexts on CPUs cause token generation speeds to plummet.

**SemanticSlidingWindow** solves this by implementing an **application-level Semantic Hippocampus**. It dynamically compresses conversation history using local embeddings, ensuring the AI retains long-term recall while keeping the KV-cache small enough for blazing-fast CPU inference.

## ⚙️ Technical Architecture

### 1. Dynamic Memory Compression
Instead of truncating tokens, the engine uses a lightweight ONNX embedding model (`fastembed`) to calculate the cosine similarity between past messages and the current prompt.
*   **Relevance Filtering:** Retains the top-$K$ most semantically relevant historical messages.
*   **Dense Summarization:** Uses the LLM itself to summarize discarded fragments into a single "Memory Block," preserving factual context without bloating the token count.

### 2. Hardware-Aware Resource Management
The application utilizes `psutil` to detect system RAM and automatically configures strict operational bounds to prevent Out-Of-Memory (OOM) crashes:

| Hardware Tier | RAM Detected | Context Window (`n_ctx`) | Max File Upload | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Low-End** | < 8 GB | 1,024 tokens | 1 MB | Strict Mode |
| **Balanced** | 8 - 16 GB | 2,048 tokens | 2 MB | Standard Mode |
| **High-End** | > 16 GB | 4,096 tokens | 5 MB | Performance Mode |

### 3. Zero-Cloud Privacy Stack
*   **Inference:** `llama-cpp-python` running Qwen 2.5 3B (4-bit quantized).
*   **Embeddings:** `fastembed` (BAAI/bge-small-en-v1.5) via ONNX Runtime.
*   **Parsing:** `pypdf` for local, client-side document ingestion.

## 📊 Performance Metrics

Optimized for standard consumer CPUs (e.g., Intel i5/i7, AMD Ryzen 5/7).

*   **Throughput:** ~15–25 tokens/sec (Balanced Mode).
*   **Memory Footprint:** < 4 GB total RAM usage during active inference.
*   **Latency:** Sub-second initial response time due to optimized KV-cache sizing.

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.10+
*   Git

### Quick Start
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/SemanticSlidingWindow.git
    cd SemanticSlidingWindow
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # Windows
    .\.venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the application:**
    ```bash
    streamlit run app.py
    ```
    *Note: The application will automatically download the required Qwen 2.5 GGUF model (~2GB) from HuggingFace on the first run.*

## 💡 Key Engineering Challenges Solved

*   **The "FIFO" Problem:** Overcame the inherent limitations of raw inference engines by building a custom Python wrapper that manages context state independently of the underlying C++ backend.
*   **Windows DLL Conflicts:** Resolved complex `WinError 4551` and `torchvision` dependency conflicts by migrating to a pure-Rust/ONNX embedding stack (`fastembed`), ensuring compatibility with strict Windows Application Control policies.
*   **Hardware Variance:** Implemented dynamic configuration logic to ensure the application remains stable across devices ranging from 8GB ultrabooks to 32GB workstations.

## 📄 License

This project is licensed under the MIT License.