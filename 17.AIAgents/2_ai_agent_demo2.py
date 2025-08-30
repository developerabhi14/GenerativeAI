from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import json
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline,HuggingFaceEndpoint
from typing import Annotated
from langchain_core.tools import InjectedToolArg
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()


api_key = os.getenv("EXCHANGE_RATE_API")

search_tool=DuckDuckGoSearchRun()

class ConversionInput(BaseModel):
    base_currency:str=Field(...,description="The base currency code (e.g. INR)")


@tool(args_schema=ConversionInput)
def get_conversion_factor(base_currency: str) -> float:
    """Return the real-time currency conversion rate from base_currency to NPR using the ExchangeRate API.
    Args should be provided as JSON, e.g. {"base_currency": "INR"}."""
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/NPR"
    response = requests.get(url)
    if response.status_code != 200:
        return {"error": f"API call failed with status {response.status_code}", "text": response.text}
    
    try:
        return response.json()
    except Exception as e:
        return {"error": f"Failed to parse JSON: {e}", "text": response.text}

model=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,  # temperature controls randomness of language model's output. affects creativity and variability.
        max_new_tokens=100  # max_new_tokens controls the maximum number of tokens to generate
    )

)

llm=ChatHuggingFace(llm=model)


prompt=hub.pull("hwchase17/react")



agent=create_react_agent(
    llm=llm,
    tools=[search_tool, get_conversion_factor],
    prompt=prompt
)

agent_executor=AgentExecutor(
    agent=agent,
    tools=[search_tool, get_conversion_factor],
    verbose=True,
    handle_parsing_errors=True
)

response=agent_executor.invoke({"input":"Find out the country whose president is Donald Trump and search for its currency and find the conversion rate of the its currency to NPR"})
print(response)