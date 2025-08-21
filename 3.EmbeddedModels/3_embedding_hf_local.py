from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text="Delhi is the capital of India"

document=["Delhi is the capital of India",
          "Kolkata is the capital of West Bengal",
          "Paris is the capital of France"]

result=embeddings.embed_query(text)


print(str(result))

result_doc=embeddings.embed_documents(document)
print(str(result_doc))