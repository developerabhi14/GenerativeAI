from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model=ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7, max_completion_tokens=100)  # temperature controls randomness of language model's output. affects creativity and variability.

# Lower values make it more deterministic. Higher values make it more creative and varied.

result=model.invoke("What is the capital of India?")
print(result)
print(result.content)  # if you want to see answer only

