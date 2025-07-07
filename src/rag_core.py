
import os
from langchain_community.document_loaders import PyPDFLoader
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv

load_dotenv()

# Get your OpenAI API key from the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def create_qa_chain(file_path: str):
    """
    This function processes an ENTIRE PDF file to create a question-answering chain.
    It is ideal for single scientific papers.
    """
    print(f"--- Creating RAG chain for entire document: {file_path} ---")
    
    # 1. Load the entire document
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # 3. Create the vector store
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(docs, embeddings)

    # 4. Create the retriever
    retriever = vector_store.as_retriever()

    # 5. Create the chat model
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    # 6. Create the prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:

    Context: {context}
    
    Question: {input}
    """)

    # 7. Create the chains
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


def create_qa_chain_for_section(file_path: str, start_page: int, end_page: int):
    """
    This function processes a SPECIFIC SECTION of a PDF to create a question-answering chain.
    It is ideal for analyzing a single chapter or paper within a larger thesis.
    """
    print(f"--- Creating RAG chain for section (Pages {start_page}-{end_page}) of: {file_path} ---")
    
    # 1. Extract text only from the specified page range
    section_text = ""
    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        start_idx = max(0, start_page - 1)
        end_idx = min(total_pages, end_page)

        if start_idx >= end_idx:
            raise ValueError("Start page is after the end page or page numbers are invalid.")

        for i in range(start_idx, end_idx):
            section_text += pdf.pages[i].extract_text() or ""
    
    if not section_text.strip():
        raise ValueError("No text could be extracted from the specified page range.")

    # 2. Split the SECTION text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents([section_text])
    
    # The rest of the pipeline is identical
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:

    Context: {context}
    
    Question: {input}
    """)
    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain