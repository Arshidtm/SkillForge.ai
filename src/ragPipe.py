from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever


def get_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",      
        
        )
    return llm


def is_job_role(text,llm,prompt):
    formatted_prompt = prompt.format(text=text)
    
    
    response = llm.invoke(formatted_prompt)
    
    return response.content


def get_rag_chain_with_memory(llm, retriever, contextualize_q_system_prompt, system_prompt):
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ('human',"{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm,retriever,contextualize_q_prompt
        )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


store={}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_conversational_rag_chain(rag_chain):
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

def get_response(conversational_rag_chain, input, session_id):
    response = conversational_rag_chain.invoke(
        {"input": "{input}"},
        config={"configurable": {"session_id": "{session_id}"}},
    )
    return response









