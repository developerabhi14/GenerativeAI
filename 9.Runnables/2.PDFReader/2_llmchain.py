# but llmchain is not deprecated

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


load_dotenv()

# initialize the model
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# create a prompt template
prompt=PromptTemplate(template="Suggest a catchy blog title about {topic}",
                        input_variables=["topic"])

# create a chain
chain=LLMChain(llm=model, prompt=prompt)

# run the chain with a specific topic
topic="Artificial Intelligence"
response=chain.invoke({"topic":topic})
# print the answer
print("Answer:", response)