import streamlit as st
from src.extractJob import fetch_jobs, clean_text, documentation
from src.ragPipe import get_conversational_rag_chain, get_response, get_session_history, get_rag_chain_with_memory, is_job_role, get_llm 
from src.embeddings import get_embedding_model, get_vector_store, get_retriever
from src.prompt import get_prompt_is_job, get_system_prompt, get_contextualize_q_system_prompt
from uuid import uuid4
from streamlit_chat import message  
from src.resume_extract import extract_resume, extract_details
import spacy
import time
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# Initialize session state
def init_session_state():
    if 'init' not in st.session_state:
        # job processing
        st.session_state.fetch_job = None
        st.session_state.docs = None
        st.session_state.vectore_store = None
        st.session_state.job_role_validated = False
        
        # Resume Processing
        st.session_state.upload_file = None
        st.session_state.extracted_resume = None
        st.session_state.extracted_details = None
        st.session_state.resume_processed = False
        st.session_state.resume_analyzed = False
        
        # Chatbot
        st.session_state.messages = []        
        
        # performence
        st.session_state.last_query_time = 0
        st.session_state.api_calls = 0
        
        st.session_state.init = True

init_session_state()


# Initialize LLM and NLP
try:
    llm = get_llm()
    
    nlp = spacy.load("resume_ner_model")
except Exception as e:
    st.error(f"Model Initialization error: {str(e)}")  


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
    job_role = st.text_input(
        "Please provide the job role", 
        key="job_role_input"
        )
    
    if job_role:  
        if not st.session_state.job_role_validated:
            with st.spinner("Validating job role..."):
                if is_job_role(job_role, llm=llm, prompt=prompt_is_job) != 'Yes':
                    st.warning("Please provide a valid job role")
                else:
                    st.session_state.job_role_validated = True                    
                    with st.spinner("Fetching job data..."):
                        st.session_state.fetch_job = fetch_jobs(job_role)
                        if  not st.session_state.fetch_job:
                            st.write("No job data found")
                            st.session_state.job_role_validated = True
                        else:
                            st.session_state.docs = documentation(st.session_state.fetch_job)
                            embedding = get_embedding_model()
                            st.session_state.vector_store = get_vector_store(st.session_state.docs, embedding)
                            st.success("Job data loaded ")
                            
    if st.session_state.job_role_validated:           
        st.write("You're good to go! Ask questions in the chat.")            
        st.session_state.uploaded_file = st.file_uploader("Upload your resume")
        if st.session_state.uploaded_file and not st.session_state.resume_processed:
            with st.spinner("Extracting the resume"):
                try:
                    st.session_state.extracted_resume = extract_resume(st.session_state.uploaded_file)
                    st.session_state.extracted_details = extract_details(st.session_state.extracted_resume,nlp)
                    st.session_state.resume_processed = True
                    st.success("Resume processed")
                except Exception as e:
                    st.error(f"Error processing the resume: {str(e)}")
                            

# Main chat area
st.title("SkillForge Career Assistant")


# Display chat messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user-message">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message">{msg["content"]}</div>', unsafe_allow_html=True)
        

# Only proceed if we have a valid job role and data
if 'vector_store' in st.session_state:
    try:
        # Initialize RAG components
        retriever = get_retriever(st.session_state.vector_store)
        system_prompt = get_system_prompt()
        contextualize_q_system_prompt = get_contextualize_q_system_prompt()
        
        # Initialize rag chain with memory
        rag_chain = get_rag_chain_with_memory(
            llm, 
            retriever, 
            contextualize_q_system_prompt, 
            system_prompt
        )
        
        # Session management
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
        
        # Handle chat input
        user_input = st.chat_input("Ask about skills, qualifications, or career advice...")
        
        # Process resume if uploaded
        if (st.session_state.resume_processed and 
            not st.session_state.resume_analyzed and 
            st.session_state.extracted_details):
            
            with st.spinner("Analyzing your resume..."):
                try:
                    response = conversational_rag_chain.invoke(
                        {
                            "input": f"Analyze this resume and suggest improvements:\n{st.session_state.extracted_details} + {user_input}"
                        },
                        config={"configurable": {"session_id": session_id}}
                    )
                    bot_response = response['answer']
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": bot_response
                    })
                    st.session_state.resume_analyzed = True
                    
                except Exception as e:
                    st.error(f"Error analyzing resume: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Sorry, I couldn't analyze your resume. Please try again."
                    })      
            
        if user_input:
            # Rate limiting
            current_time = time.time()
            if current_time - st.session_state.last_query_time < 1:
                st.warning("Please wait a moment between queries")
                st.stop()
            st.session_state.last_query_time = current_time
            st.session_state.api_calls += 1
            
            # Add user message to history
            st.session_state.messages.append({
                "role": "user", 
                "content": user_input
            })
            
            # Get and display response
            with st.spinner("Researching..."):
                try:
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id},
                                "max_tokens": 1024
                                }
                    )
                    bot_response = response['answer']
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": bot_response
                    })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "I encountered an error. Please try rephrasing your question."
                    })
            
            st.rerun()

    except Exception as e:
        st.error(f"System error: {str(e)}")
        if "413" in str(e):
            st.error( "Request too large - please ask more specific questions")