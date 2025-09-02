import streamlit as st
import os
from dotenv import load_dotenv

from data_ingestion import DocumentIndexer
from retrieval import RetrieverFactory
from rag_pipeline import RAGPipeline
from prompts import retrieval_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
import config

load_dotenv()

st.set_page_config(page_title="RAG Q&A", page_icon="\U0001f50e")
st.title("\U0001f50e RAG Q&A System")

# Initialize state for Streamlit session
if "pipeline" not in st.session_state:
    # Step 1: Build or load index
    with st.spinner("Building/loading document index..."):
        indexer = DocumentIndexer(config.PDF_PATH, config.EMBEDDING_MODEL)
        vectorstore = indexer.build_or_load_index(config.VECTORSTORE_PATH)

    # Step 2: Create retriever and LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    retriever = RetrieverFactory(vectorstore, llm).create()

    # Step 3: Create RAG pipeline
    pipeline = RAGPipeline(retriever, llm, retrieval_prompt)
    
    st.session_state.pipeline = pipeline
    st.success("RAG Pipeline is ready!")

# ----------------- Query Input -----------------
query = st.text_input("Enter your query:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Running query through RAG pipeline..."):
            answer = st.session_state.pipeline.run(query)
        st.write("### Answer:")
        st.success(answer)
