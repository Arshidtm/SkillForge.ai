import logging
import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.runnables import RunnableLambda

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@st.cache_resource(ttl=3600) 
def get_llm() -> ChatGroq:
    """
    Initialize and return the Chatgroq language model.

    Returns:
        ChatGroq: An instance of the ChatGroq language model configured with llama-3.3-70b-versatile
    """
    logger.info("Initializing ChatGroq LLM with 'lama-3.3-70b-versatile' model")
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
    )
    return llm


def is_job_role(text: str, llm: ChatGroq, prompt: ChatPromptTemplate) -> str:
    """
    Determine the given text describes a job role

    Args:
        text (str): The text to analyze
        llm (ChatGroq): The LLM used for analysis
        prompt (ChatPromptTemplate): the prompt template to format the input.

    Returns:
        str: The model's response ('Yes' or 'No')
    """
    logger.info("checking text describes a job role")
    formatted_prompt = prompt.format(text=text)

    try:
        response = llm.invoke(formatted_prompt)
        return response.content
    except Exception as e:
        logger.error("Error determining the job role")
        raise





def get_rag_chain_with_memory(
    llm: ChatGroq, retriever, contextualize_q_system_prompt: str, system_prompt: str
):
    """
    creates rag cahin eith memory capabilities

    Args:
        llm (ChatGroq): The LLM
        retriever (_type_): the document retriever
        contextualize_q_system_prompt (str): prompt for contextualizing the query
        system_prompt (str): Main system prompt for chatbot

    Returns:
        A configured RAG chain with history awareness
    """
    logger.info("Creating RAG chain with memory")
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    logger.debug("creating rag chain")
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """
    create chat message history for session

    Args:
        session_id (str): Unique identifier for the conversational session.

    Returns:
        BaseChatMessageHistory: The chat history of session
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_conversational_rag_chain(rag_chain):
    """
    Add message history capabilities to a RAG chain

    Args:
        rag_chain: The base RAG chain to enhance

    Returns:
        RunnableWithMessageHistory: A RAG chain with conversation history support
    """
    logger.info("Creating the conversational chain with memory")
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


def get_response(conversational_rag_chain, input: str, session_id: str):
    """
    Get response from the conversational RAG chain.

    Args:
        conversational_rag_chain : The configured chain
        input (str): User input/query
        session_id (str): Unique Conversation session identifier

    Returns:
        The chain's response
    """
    logger.info("Getting user response")
    try:
        response = conversational_rag_chain.invoke(
            {"input": "{input}"},
            config={"configurable": {"session_id": "{session_id}"}},
        )
        logger.debug(f"response generated for session id: {session_id}")
        return response
    except Exception as e:
        logger.error(f"Error generating response {str(e)}")
        raise
