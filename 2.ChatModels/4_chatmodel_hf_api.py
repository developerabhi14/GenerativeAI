from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()


hf_api_token = os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN")
llm=HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    huggingfacehub_api_token=hf_api_token
    )

model=ChatHuggingFace(llm=llm)
result=model.invoke("What is the capital of India?")

print(result)
print(result.content)  # if you want to see answer only