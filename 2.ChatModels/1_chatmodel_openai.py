from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0.7, max_completion_tokens=100) # temperature controls randomness of langauge model's output. affects creativity and variability.
# Lower values make it more deterministic. Higher values make it more creative and varied.   

result = model.invoke("What is the capital of India?")

print(result)

# if you want to see asnwer only
print(result.content)

