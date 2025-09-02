import os
from dotenv import load_dotenv

from data_ingestion import DocumentIndexer
from retrieval import RetrieverFactory
from rag_pipeline import RAGPipeline
from prompts import retrieval_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
import config
load_dotenv()

# # Config
# PDF_PATH = "books/report.pdf"
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# VECTORSTORE_PATH = "faiss_index"

# Step 1: Build or load index
indexer = DocumentIndexer(config.PDF_PATH, config.EMBEDDING_MODEL)
vectorstore = indexer.build_or_load_index(config.VECTORSTORE_PATH)

# Step 2: Create retriever
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
retriever = RetrieverFactory(vectorstore, llm).create()

# Step 3: Create RAG pipeline
pipeline = RAGPipeline(retriever, llm, retrieval_prompt)

# Step 4: Run queries
while True:
    query = input("\nEnter your query (or type 'exit'): ")
    if query.lower() == "exit":
        break

    answer = pipeline.run(query)
    print(f"\nAnswer: {answer}")
