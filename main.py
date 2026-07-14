import os
import numpy as np
from llama_cpp import Llama
from fastembed import TextEmbedding
from rich.console import Console
from rich.panel import Panel

# --- Configuration ---
MODEL_PATH = "./models/qwen2.5-3b-instruct-q4_k_m.gguf" # Ensure this matches your file
MAX_CONTEXT_MESSAGES = 10 
KEEP_RECENT = 4           
KEEP_RELEVANT = 3         

console = Console()

# --- 1. Load Models (CPU Optimized) ---
console.print("[bold green]Loading LLM (Qwen 2.5 3B)...[/bold green]")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048, 
    n_threads=os.cpu_count(),
    verbose=False
)

console.print("[bold green]Loading Embedding Model (FastEmbed)...[/bold green]")
# FastEmbed handles the model download automatically on first run
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# --- 2. Helper Functions ---
def get_embeddings(texts):
    """Get embeddings using FastEmbed."""
    # FastEmbed returns a generator, so we convert to list then numpy
    return np.array(list(embedder.embed(texts)))

def compress_memory(messages, current_prompt):
    """
    Semantic Sliding Window: Compresses old messages while retaining relevance.
    """
    recent_msgs = messages[-KEEP_RECENT:]
    history_msgs = messages[:-KEEP_RECENT]
    
    if not history_msgs:
        return messages

    # 1. Embed current prompt and history
    prompt_emb = get_embeddings([current_prompt])
    history_texts = [f"{m['role']}: {m['content']}" for m in history_msgs]
    history_embs = get_embeddings(history_texts)
    
    # 2. Calculate cosine similarity manually (no sklearn needed)
    # Normalize vectors to unit length
    prompt_norm = prompt_emb / np.linalg.norm(prompt_emb, axis=1, keepdims=True)
    history_norm = history_embs / np.linalg.norm(history_embs, axis=1, keepdims=True)
    
    # Dot product of normalized vectors = cosine similarity
    similarities = np.dot(prompt_norm, history_norm.T)[0]
    
    # 3. Find indices of most relevant history messages
    relevant_indices = np.argsort(similarities)[-KEEP_RELEVANT:][::-1]
    
    # 4. Separate kept messages and messages to summarize
    kept_history = [history_msgs[i] for i in relevant_indices]
    discard_history = [history_msgs[i] for i in range(len(history_msgs)) if i not in relevant_indices]
    
    # 5. Summarize the discarded messages using the LLM
    if discard_history:
        discard_text = "\n".join([f"{m['role']}: {m['content']}" for m in discard_history])
        summary_prompt = f"Summarize the following conversation fragments into a single, dense memory block. Focus on facts and user preferences:\n\n{discard_text}\n\nSummary:"
        
        summary_response = llm(
            summary_prompt, 
            max_tokens=150, 
            stop=["\n"], 
            echo=False
        )
        memory_block = summary_response['choices'][0]['text'].strip()
    else:
        memory_block = ""

    # 6. Reconstruct messages
    new_messages = []
    if memory_block:
        new_messages.append({
            "role": "system", 
            "content": f"[Long-term Memory Context]: {memory_block}"
        })
    
    new_messages.extend(kept_history)
    new_messages.extend(recent_msgs)
    
    return new_messages

# --- 3. Main Chat Loop ---
def chat():
    messages = [
        {"role": "system", "content": "You are a helpful, concise local AI assistant."}
    ]
    
    console.print(Panel.fit("[bold blue]SemanticSlidingWindow LLM[/bold blue]\nType 'quit' to exit. Context auto-compresses!", border_style="blue"))
    
    while True:
        user_input = console.input("[bold cyan]You:[/bold cyan] ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        # Trigger Semantic Compression if context is getting too large
        if len(messages) > MAX_CONTEXT_MESSAGES:
            console.print("[yellow]⟳ Compressing context window semantically...[/yellow]")
            messages = compress_memory(messages, user_input)
            console.print("[green]✓ Context compressed. Memory retained.[/green]")

        # Generate response
        console.print("[bold magenta]AI:[/bold magenta] ", end="")

        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            stream=True,
            stop=["<|im_end|>"]
        )
        
        full_response = ""
        for chunk in response:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                text = delta['content']
                full_response += text
                print(text, end="", flush=True)
        print("\n")
        
        messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    chat()