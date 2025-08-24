from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()


# load the document
loader=TextLoader("docs.txt")
documents=loader.load()

# split the text into smaller chunks
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs=text_splitter.split_documents(documents)

# convert text embeddings and store in FAISS
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore=FAISS.from_documents(docs, embeddings)

# create a retriever (fetches relevant documents)
retriever=vectorstore.as_retriever()

# Manually retrieve relevant documents
query="What are the key takeaways from the document?"
retrieved_docs=retriever.get_relevant_documents(query)

# combine the retrieved documents into a single string
context="\n".join([doc.page_content for doc in retrieved_docs])

# initialize the model
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# manually pass retrieved text to llm
prompt=f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
response=model.invoke(prompt)

# print the answer
print("Answer:", response.content)