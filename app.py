import logging
import streamlit as st
from PIL import Image
import io
from src.extractJob import fetch_jobs, clean_text, documentation
from src.ragPipe import (
    get_conversational_rag_chain,
    get_response,
    get_session_history,
    get_rag_chain_with_memory,
    is_job_role,
    get_llm,
)
from src.embeddings import get_embedding_model, get_vector_store, get_retriever
from src.prompt import (
    get_prompt_is_job,
    get_system_prompt,
    get_contextualize_q_system_prompt,
    get_prompt_profile_evaluator,
)
from src.profile_picture import load_yolo_model, generate_response, predict_score
from uuid import uuid4
from streamlit_chat import message
from src.resume_extract import extract_resume, extract_details
import spacy
import time
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda


# configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


if "current_module" not in st.session_state:
    st.session_state.current_module = None

if "module1" not in st.session_state:
    st.session_state.module1 = {
        "uploaded_file": None,
        "score": None,
        "image": None,
        "feedback": None,
    }

if "module2" not in st.session_state:
    st.session_state.module2 = {
        "chat_history": [],
        # Job processing
        "fetch_job": None,
        "docs": None,
        # "vector_store": None,
        "job_role_validated": False,
        # Resume processing
        "uploaded_file": None,
        "extracted_resume": None,
        "extracted_details": None,
        "resume_processed": False,
        "resume_analyzed": False,
        # Chatbot messages
        "messages": [],
        # Performance
        "last_query_time": 0,
        "api_calls": 0,
        # Session management
        # "session_id": None,
        # "store": {}
    }

with st.sidebar:
    st.header("Navigation")

    # Module selection buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "📸 Module 1", help="Profile Picture Analysis", key="module1_nav_btn"
        ):
            st.session_state.current_module = "Module 1"
    with col2:
        if st.button(
            "📊 Module 2", help="Second Module Functionality", key="module2_nav_btn"
        ):
            st.session_state.current_module = "Module 2"

# Module 1 Content
if st.session_state.current_module == "Module 1":
    st.title("👔 Profile Picture Professionalism Rater")
    try:
        llm = get_llm()
        model = load_yolo_model()
    except Exception as e:
        logger.error(f"Error loading model {str(e)}")

    # Sidebar for upload (only shown in Module 1)
    with st.sidebar:
        st.header("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a profile picture...",
            type=["jpg", "jpeg", "png", "webp"],
            key="module1_uploader",
        )

        # Store uploaded file in session state
        if uploaded_file:
            st.session_state.module1["uploaded_file"] = uploaded_file
            st.session_state.module1["image"] = Image.open(
                io.BytesIO(uploaded_file.getvalue())
            )

    # Main content area
    col1, col2 = st.columns(2)

    if st.session_state.module1["uploaded_file"]:
        with st.spinner("Analyzing professionalism..."):
            try:
                # Only predict if score doesn't exist
                # if st.session_state.module1["score"] is None:
                score, image = predict_score(
                    st.session_state.module1["uploaded_file"], model
                )
                st.session_state.module1["score"] = score
                st.session_state.module1["feedback"] = generate_response(score, llm)

                # Display results
                with col1:
                    st.image(
                        st.session_state.module1["image"],
                        caption="Your Profile Picture",
                        width=300,
                    )
                    st.subheader("Results")
                    st.metric(
                        "Professional Score", f"{st.session_state.module1['score']}/100"
                    )
                    st.progress(st.session_state.module1["score"] / 100)

                with col2:
                    st.markdown(st.session_state.module1["feedback"])

            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    else:
        # Show placeholder before upload
        with col1:
            st.image(
                "https://via.placeholder.com/300x300?text=Upload+an+image",
                caption="Preview will appear here",
                width=300,
            )
        with col2:
            st.info("ℹ️ Upload a profile picture to analyze its professionalism")
elif st.session_state.current_module == "Module 2":

    # Module 2
    # Initialize session state
    def init_session_state():
        """
        Initialize the streamlit session_state.
        """
        logger.info("Initializing the session state")
        if "init" not in st.session_state:
            # Module 2 state
            st.session_state.module2 = {
                # job processing
                "fetch_job": None,
                "docs": None,
                # "vector_store": None,
                "job_role_validated": False,
                # Resume Processing
                "uploaded_file": None,
                "extracted_resume": None,
                "extracted_details": None,
                "resume_processed": False,
                "resume_analyzed": False,
                # Chatbot messages
                "messages": [],
                "chat_history": [],
                # performance
                "last_query_time": 0,
                "api_calls": 0,
                # session management
                "session_id": None,
                "store": {},
            }
            st.session_state.init = True

    init_session_state()

    # Initialize LLM and NLP
    try:
        llm = get_llm()
        nlp = spacy.load("resume_ner_model")
        embedding = get_embedding_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error("Model Initialization Failed")
        st.error(f"Model Initialization error: {str(e)}")

    # Custom CSS for better styling
    st.markdown(
        """
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
    """,
        unsafe_allow_html=True,
    )

    # Sidebar for job role input
    with st.sidebar:
        st.title("Job Configuration")
        prompt_is_job = get_prompt_is_job()
        job_role = st.text_input("Please provide the job role", key="job_role_input")

        if job_role:
            logger.info(f"Job role input recieved {job_role}")
            if not st.session_state.module2["job_role_validated"]:
                with st.spinner("Validating job role..."):
                    if is_job_role(job_role, llm=llm, prompt=prompt_is_job) != "Yes":
                        logger.warning(f"Invalid job title provided {job_role}")
                        st.warning("Please provide a valid job role")
                    else:
                        st.session_state.module2["job_role_validated"] = True
                        logger.info(f"Job role {job_role} validated success")
                        with st.spinner("Fetching job data..."):
                            st.session_state.module2["fetch_job"] = fetch_jobs(job_role)
                            if not st.session_state.module2["fetch_job"]:
                                logger.warning("No job data  found")
                                st.write("No job data found")
                                st.session_state.module2["job_role_validated"] = True
                            else:
                                st.session_state.module2["docs"] = documentation(
                                    st.session_state.module2["fetch_job"]
                                )

                                st.session_state.module2["vector_store"] = (
                                    get_vector_store(
                                        st.session_state.module2["docs"], embedding
                                    )
                                )
                                logger.info("Job data loaded")
                                st.success("Job data loaded ")

        if st.session_state.module2["job_role_validated"]:
            st.write("You're good to go! Ask questions in the chat.")
            st.session_state.module2["uploaded_file"] = st.file_uploader(
                "Upload your resume", key="module2_file_uploader"
            )
            if (
                st.session_state.module2["uploaded_file"]
                and not st.session_state.module2["resume_processed"]
            ):
                with st.spinner("Extracting the resume"):
                    try:
                        st.session_state.module2["extracted_resume"] = extract_resume(
                            st.session_state.module2["uploaded_file"]
                        )
                        st.session_state.module2["extracted_details"] = extract_details(
                            st.session_state.module2["extracted_resume"], nlp
                        )
                        st.session_state.module2["resume_processed"] = True
                        st.success("Resume processed")
                        logger.info("resume processed success")
                    except Exception as e:
                        logger.error("Error processing the resume")
                        st.error(f"Error processing the resume: {str(e)}")

    # Main chat area
    st.title("SkillForge Career Assistant")

    # Display chat messages
    for msg in st.session_state.module2["messages"]:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-message bot-message">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
    # user_input = st.chat_input(
    #     "Ask about skills, qualifications, or career advice...",
    #     key="module2_chat_input"
    # )

    # Only proceed if we have a valid job role and data
    if "vector_store" in st.session_state.module2:
        try:
            # Initialize RAG components
            retriever = get_retriever(st.session_state.module2["vector_store"])
            system_prompt = get_system_prompt()
            contextualize_q_system_prompt = get_contextualize_q_system_prompt()

            # Initialize rag chain with memory
            rag_chain = get_rag_chain_with_memory(
                llm, retriever, contextualize_q_system_prompt, system_prompt
            )
            logger.info("RAG chain initialized")

            def context_filter(input_dict):
                if not input_dict.get("context"):
                    return {
                        "answer": "I can only answer questions about technical career skills based on my knowledge base."
                    }
                return input_dict

            filtered_rag_chain = rag_chain | RunnableLambda(context_filter)

            # Session management
            if "session_id" not in st.session_state.module2:
                st.session_state.module2["session_id"] = str(uuid4())
                logger.info(
                    f"New session created: {st.session_state.module2['session_id']}"
                )

            session_id = st.session_state.module2["session_id"]

            if "store" not in st.session_state.module2:
                st.session_state.module2["store"] = {}

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                """
                Create chat message history for session

                Args:
                    session_id (str): Unique identifier for the conversational session.

                Returns:
                    BaseChatMessageHistory: The chat history of session
                """
                if session_id not in st.session_state.module2["store"]:
                    st.session_state.module2["store"][session_id] = ChatMessageHistory()
                return st.session_state.module2["store"][session_id]

            conversational_rag_chain = get_conversational_rag_chain(filtered_rag_chain)

            # Handle chat input
            user_input = st.chat_input(
                "Ask about skills, qualifications, or career advice..."
            )

            # Process resume if uploaded
            if (
                st.session_state.module2["resume_processed"]
                and not st.session_state.module2["resume_analyzed"]
                and st.session_state.module2["extracted_details"]
            ):

                with st.spinner("Analyzing your resume..."):
                    try:
                        response = conversational_rag_chain.invoke(
                            {
                                "input": f"Analyze this resume and suggest improvements:\n{st.session_state.module2["extracted_details"]} + {user_input}"
                            },
                            config={"configurable": {"session_id": session_id}},
                        )
                        bot_response = response["answer"]
                        st.session_state.module2["messages"].append(
                            {"role": "assistant", "content": bot_response}
                        )
                        st.session_state.module2["resume_analyzed"] = True
                        logger.info("resume analyzed")

                    except Exception as e:
                        logger.error("Error analyzing resume")
                        st.error(f"Error analyzing resume: {str(e)}")
                        st.session_state.module2["messages"].append(
                            {
                                "role": "assistant",
                                "content": "Sorry, I couldn't analyze your resume. Please try again.",
                            }
                        )

            if user_input:
                # Rate limiting
                current_time = time.time()
                if current_time - st.session_state.module2["last_query_time"] < 1:
                    logger.warning("Rate limit exceeded - too frequent queries")
                    st.warning("Please wait a moment between queries")
                    st.stop()
                st.session_state.module2["last_query_time"] = current_time
                st.session_state.module2["api_calls"] += 1

                # Add user message to history
                st.session_state.module2["messages"].append(
                    {"role": "user", "content": user_input}
                )

                # Get and display response
                with st.spinner("Researching..."):
                    try:
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={
                                "configurable": {"session_id": session_id},
                                "max_tokens": 1024,
                            },
                        )
                        bot_response = response["answer"]
                        logger.info("Response generated successfully")

                        st.session_state.module2["messages"].append(
                            {"role": "assistant", "content": bot_response}
                        )
                    except Exception as e:
                        logger.error("Error generating response")
                        st.error(f"Error generating response: {str(e)}")
                        st.session_state.module2["messages"].append(
                            {
                                "role": "assistant",
                                "content": "I encountered an error. Please try rephrasing your question.",
                            }
                        )

                st.rerun()

        except Exception as e:
            logger.error("System error")
            st.error(f"System error: {str(e)}")
            if "413" in str(e):
                logger.warning("Request too large")
                st.error("Request too large - please ask more specific questions")
