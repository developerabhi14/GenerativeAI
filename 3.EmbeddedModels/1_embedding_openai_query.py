from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32) 
# dimensions parameter is optional, default is 3072 for text-embedding-3-large

result=embeddings.embed_query("Delhi is the capital of India")

print(str(result))