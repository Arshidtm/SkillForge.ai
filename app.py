import streamlit as st
from src.extractJob import *
from src.ragPipe import *
from src.embeddings import *
from src.prompt import *
from uuid import uuid4
from streamlit_chat import message  # For chat bubbles

# Initialize LLM
llm = get_llm()

# Custom CSS for better styling
st.markdown("""
<style>
    .stTextInput>div>div>input {
        border-radius: 20px;
        padding: 10px 15px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 20px;
        padding: 10px 15px;
        min-height: 100px;
    }
    .chat-message {
        padding: 12px 16px;
        border-radius: 20px;
        margin: 8px 0;
        max-width: 70%;
    }
    .user-message {
        background-color: #0d6efd;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 5px;
    }
    .bot-message {
        background-color: #f0f2f6;
        color: black;
        margin-right: auto;
        border-bottom-left-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for job role input
with st.sidebar:
    st.title("Job Configuration")
    prompt_is_job = get_prompt_is_job()
    job_role = st.text_input("Please provide the job role", key="job_role_input")
    
    if job_role:  # Only check if field is not empty
        if is_job_role(job_role, llm=llm, prompt=prompt_is_job) != 'Yes':
            st.warning("Please provide a valid job role")
        else:
            if 'vector_store' not in st.session_state:
                with st.spinner("Fetching job data..."):
                    st.session_state.fetch_job = fetch_jobs(job_role)
                    if  not st.session_state.fetch_job:
                        st.write("No File Found")
                    else:
                        st.session_state.docs = documentation(st.session_state.fetch_job)
                        embedding = get_embedding_model()
                        st.session_state.vector_store = get_vector_store(st.session_state.docs, embedding)
                        st.success("Job data loaded successfully!")
                
            st.write("You're good to go! Ask questions in the chat.")

# Main chat area
st.title("SkillForge Career Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message">{msg["content"]}</div>', unsafe_allow_html=True)

# Only proceed if we have a valid job role and data
if 'vector_store' in st.session_state:
    retriever = get_retriever(st.session_state.vector_store)
    system_prompt = get_system_prompt()
    contextualize_q_system_prompt = get_contextualize_q_system_prompt()
    rag_chain = get_rag_chain_with_memory(llm, retriever, contextualize_q_system_prompt, system_prompt)
    
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid4())
    
    session_id = st.session_state["session_id"]
    
    if "store" not in st.session_state:
        st.session_state.store = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversational_rag_chain = get_conversational_rag_chain(rag_chain)
    
    # Chat input
    user_input = st.chat_input("Ask about skills, qualifications, or career advice...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get response
        with st.spinner("Thinking..."):
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            bot_response = response['answer']
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        # Rerun to show new messages
        st.rerun()