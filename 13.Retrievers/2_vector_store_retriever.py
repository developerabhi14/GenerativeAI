from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# Step 1: Your source documents
documents=[
    Document(page_content="Langchain helps developers build LLM applications easily"),
    Document(page_content="Chroma is a vector database optimized for LLM based search"),
    Document(page_content="Embeddings convert text into high-dimensional vectors"),
    Document(page_content="OpenAI provides powerful embedding models")
]

# Step 2: Initialize embedding model
embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Create Chroma vector store in memory
vectorstore=Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="my_collection"
)

# Step 4: Convert vectorstore into a retriever
retriever=vectorstore.as_retriever(search_kwargs={"k":2})

query="What is chroma used for?"
results=retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--------------")
    print(doc.page_content)

print(vectorstore.similarity_search_with_score(query, k=2))
