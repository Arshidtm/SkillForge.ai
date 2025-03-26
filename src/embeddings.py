from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_chunks(text): 
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(text)
    return text_chunks

def get_embedding_model(): 
    embedding_model=HuggingFaceEmbeddings('sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model

def vectorize(text_chunks,embeddings):
    db=FAISS.from_documents(text_chunks,embeddings)
    