from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

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

# initialize the model
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Createa a RetrievalQA chain
chain=RetrievalQA.from_chain_type(llm=model, retriever=retriever)

# Manually retrieve relevant documents
query="What are the key takeaways from the document?"
response=chain.invoke(query)

# print the answer
print("Answer:", response)