import logging
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name) - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        embedding_model=HuggingFaceEmbeddings()
        logger.info("Embeddings model initialized successfully")
        return embedding_model
    except Exception as e:
        logger.error(f"Failed to initialize embeddings model: {str(e)}")
        raise RuntimeError("Could not initialize embedding model") from e
    

def get_vector_store(text,embeddings):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200)
    text_chunks=text_splitter.split_documents(text)
    vector_store=FAISS.from_documents(text_chunks,embeddings)
    return vector_store
    
    
def get_retriever(vectore_store):
    retriever=vectore_store.as_retriever(search_type='similarity',search_kwargs={'k':3})
    return retriever

