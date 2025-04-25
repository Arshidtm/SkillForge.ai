import logging
from typing import List
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name) - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@st.cache_resource(ttl=3600)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initializing and returning the embedding model.

    Returns:
        HuggingFaceEmbeddings: Initialized embedding model

    Raises:
        RuntimeaError: if model initialization fails
    """
    try:
        logger.info("Initializing the HuggingFace embedding model")
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embeddings model initialized successfully")
        return embedding_model
    except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {str(e)}")
        raise RuntimeError("Could not initialize embedding model") from e


@st.cache_resource(ttl=3600)
def get_vector_store(_text: List[Document], _embeddings: HuggingFaceEmbeddings) -> FAISS:
    """
    Create and return FAISS vector store from documents.

    Args:
        text (List[Document]): List of documents to be processed
        embeddings (HuggingFaceEmbeddings): Embedding model

    Returns:
        FAISS: Vector store containg document embedding

    Raises:
        ValueError: If input document are invalid
        RuntimeError: if vector store creation failed
    """
    try:
        if not _text or not isinstance(_text, list):
            raise ValueError("Input documents must be a non-empty list")

        logger.info("Creating vector store from documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=200
        )

        logger.info(f"splitting documents into chunks")
        text_chunks = text_splitter.split_documents(_text)
        logger.info(f"created {len(text_chunks)} text chunks")

        if not text_chunks:
            raise ValueError("No valid chunks created from documents")

        vector_store = FAISS.from_documents(text_chunks, _embeddings)

        logger.info("Created vectore store Successfully")
        return vector_store
    except ValueError as ve:
        logger.error(f"Invalid input documents: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Failed to create Vectore store: {str(e)}")
        raise RuntimeError("Vector store creation failed") from e


def get_retriever(vectore_store: FAISS):
    """
    Create and return retriever from vector store.

    Args:
        vectore_store (FAISS): Initialized Vectore store

    Returns:
        BaseRetriever: Configured retriever object

    Raises:
        ValueError: If vector store is invalid
        RuntimeError: if retriever creation failed
    """
    try:
        if not vectore_store:
            raise ValueError("Vector store cannot be None")
        logger.info("Creating retriever from vectore store")

        retriever = vectore_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )
        logger.info("Retriever created successfully")
        return retriever
    except ValueError as ve:
        logger.error(f"Invalid vector store: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Failed to create retriever: {str(e)}")
        raise RuntimeError("Retriever creation failed") from e
