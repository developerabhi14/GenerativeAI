from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


documents=[
    Document(page_content="Langchain makes it easy to work with LLMs"),
    Document(page_content="Langchain helps developers build LLM applications easily"),
    Document(page_content="Chroma is a vector database optimized for LLM based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors"),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="OpenAI provides powerful embedding models")
]
# Step 2: Initialize embedding model
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore=FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)


# enable MMR in the retriever
retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3, "lambda_mult":0.25} # k=top_results, lambda_unit=relevance=diversity balance
)

query="What is langchain?"
results=retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"\n------------------\n")
    print(doc.page_content)