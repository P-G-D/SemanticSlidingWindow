import time
import streamlit as st
import core

# --- 1. Cache Models (Web UI specific) ---
@st.cache_resource
def setup_models():
    hw_config = core.get_hardware_config()
    core.ensure_model_exists()
    llm, embedder = core.load_models(hw_config)
    return hw_config, llm, embedder

hw_config, llm, embedder = setup_models()

# --- 2. Streamlit UI Setup ---
st.set_page_config(page_title="SemanticSlidingWindow", page_icon="🧠", layout="centered")
st.title("🧠 SemanticSlidingWindow LLM")
st.caption(f"Mode: {hw_config['label']} | 100% Local & Private")

# Sidebar for File Upload
with st.sidebar:
    st.header("📄 Document Ingestion")
    st.caption(f"Max size: {hw_config['max_file_mb']}MB")
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Parsing document locally..."):
            try:
                file_content = core.process_uploaded_file(
                    uploaded_file.read(), uploaded_file.name, uploaded_file.size, hw_config
                )
                st.success(f"✅ Loaded: {uploaded_file.name}")
                if "file_injected" not in st.session_state or st.session_state.file_injected != uploaded_file.name:
                    st.session_state.messages.insert(0, {"role": "system", "content": f"User uploaded '{uploaded_file.name}'. Content:\n\n{file_content}"})
                    st.session_state.messages.append({"role": "assistant", "content": f"I've read **{uploaded_file.name}**. Ask me anything!"})
                    st.session_state.file_injected = uploaded_file.name
            except ValueError as e:
                st.error(str(e))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm running locally on your CPU. How can I help?"}]

# Display chat messages
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat Input & Generation
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    messages_to_process = st.session_state.messages.copy()
    
    if len(messages_to_process) > 10:
        with st.status("🔄 Compressing context...", expanded=True) as status:
            messages_to_process = core.compress_memory(llm, embedder, messages_to_process, prompt)
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
        tps = (len(full_response) / 4.0) / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        response_placeholder.markdown(full_response)
        metrics_placeholder.markdown(f"⚡ *{end_time - start_time:.2f}s | ~{tps:.1f} tokens/sec*")
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})