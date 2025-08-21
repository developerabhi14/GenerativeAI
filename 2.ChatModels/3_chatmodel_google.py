from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # temperature controls randomness of language model's output. affects creativity and variability.       

# Lower values make it more deterministic. Higher values make it more creative and varied.

result=model.invoke("What is the capital of India?")
print(result)
print(result.content)  # if you want to see answer only
