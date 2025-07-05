import streamlit as st
from src.generator import build_prompt, stream_llm,model_name
from src.retriever import retrieve_chunks
import time

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")

# Sidebar info
st.sidebar.header("System Info")
st.sidebar.markdown(f"Model : {model_name}")
st.sidebar.markdown(f"Number of chunks: 41")

# Initialize variables to store current_query,current_answer and current_sources 
if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""
if "current_sources" not in st.session_state:
    st.session_state.current_sources = []
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

# Input and submit
with st.form("query_form"):
    user_query = st.text_input("Enter your questions: ", placeholder="e.g. what is ebay?", key="input")
    submitted = st.form_submit_button("Submit")

# Clear current answer
if st.button("Reset"):
    st.session_state.current_answer = ""
    st.session_state.current_sources = []
    st.session_state.current_query = ""
    st.rerun()

# Handle submission
if submitted and user_query.strip():
    # Clear previous answer when new query is submitted
    st.session_state.current_answer = ""
    st.session_state.current_sources = []
    st.session_state.current_query = user_query
    
    with st.spinner("Retrieving context and generating response..."):
        retrieved_chunks = retrieve_chunks(user_query)
        prompt, sources = build_prompt(retrieved_chunks, user_query)
        
        # Store sources to display it
        st.session_state.current_sources = sources
        
        response_container = st.empty()
        answer = ""
        for token in stream_llm(prompt):
            answer += token
            response_container.markdown(f"**Answer:**\n{answer}")
            time.sleep(0.01)
        
        # Store final answer
        st.session_state.current_answer = answer

# Show sources toggle if we have an answer (this will persist across reruns)
if st.session_state.current_answer:
    
    show_sources = st.checkbox("Show Source Chunks", key="show_sources")
    if show_sources:
        st.markdown(f"**Answer:**\n{st.session_state.current_answer}")
        st.markdown("**Source Chunks Used:**")
        st.markdown(f"{st.session_state.current_sources}")

