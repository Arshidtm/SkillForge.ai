from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS



def get_embedding_model(): 
    embedding_model=HuggingFaceEmbeddings()
    return embedding_model

def get_vector_store(text,embeddings):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=200)
    text_chunks=text_splitter.split_documents(text)
    vector_store=FAISS.from_documents(text_chunks,embeddings)
    return vector_store
    
    
def get_retriever(vectore_store):
    retriever=vectore_store.as_retriever(search_type='similarity',search_kwargs={'k':3})
    return retriever

