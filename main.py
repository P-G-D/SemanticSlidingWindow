import time
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import core

console = Console()

def main():
    console.print(Panel.fit("[bold blue]SemanticSlidingWindow CLI[/bold blue]", border_style="blue"))
    
    # 1. Setup Environment
    hw_config = core.get_hardware_config()
    console.print(f"[dim]Hardware Mode:[/dim] {hw_config['label']}")
    
    core.ensure_model_exists()
    console.print("[green]Loading models into memory...[/green]")
    llm, embedder = core.load_models(hw_config)
    console.print("[green]Ready! Type 'quit' to exit.[/green]\n")

    messages = [{"role": "assistant", "content": "Hello! I am running locally on your CPU. How can I help?"}]
    
    # Print initial greeting
    console.print("[bold magenta]AI:[/bold magenta] Hello! I am running locally on your CPU. How can I help?")

    while True:
        user_input = console.input("\n[bold cyan]You:[/bold cyan] ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        # Trigger Semantic Compression
        if len(messages) > 10:
            console.print("[yellow]⟳ Compressing context window semantically...[/yellow]")
            messages = core.compress_memory(llm, embedder, messages, user_input)
            console.print("[green]✓ Context compressed.[/green]")

        # Generate response
        console.print("[bold magenta]AI:[/bold magenta] ", end="")
        start_time = time.perf_counter()
        
        response = llm.create_chat_completion(messages=messages, max_tokens=512, temperature=0.7, stream=True, stop=["<|im_end|>"])
        
        full_response = ""
        for chunk in response:
            if chunk['choices'][0]['delta'].get('content'):
                text = chunk['choices'][0]['delta']['content']
                full_response += text
                print(text, end="", flush=True)
                
        end_time = time.perf_counter()
        tps = (len(full_response) / 4.0) / (end_time - start_time)
        print(f"\n[dim]⚡ {end_time - start_time:.2f}s | ~{tps:.1f} tokens/sec[/dim]")
        
        messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()