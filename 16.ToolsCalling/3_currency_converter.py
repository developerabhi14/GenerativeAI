from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import json

load_dotenv()
api_key = os.getenv("EXCHANGE_RATE_API")

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Return the real-time currency conversion rate from base_currency to target_currency using the ExchangeRate API."""
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Converts a base currency value into target currency value using a conversion rate."""
    return base_currency_value * conversion_rate


# Bind LLM with tools
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

query = "What is the conversion factor between USD and NPR and based on that, can you convert 10 usd to npr?"

messages=[HumanMessage(query)]
# Pass a single message instead of list
ai_msg = llm_with_tools.invoke(messages)

messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    if tool_call['name']=='get_conversion_factor':
        tool_msg1=get_conversion_factor.invoke(tool_call)
        conversion_rate=json.loads(tool_msg1.content)['conversion_rate']
        messages.append(tool_msg1)
    # # execute the 2nd tool
    if tool_call['name']=='convert':
        tool_call['args']['conversion_rate']=conversion_rate
        tool_msg2=convert.invoke(tool_call)
        messages.append(tool_msg2)


result=llm_with_tools.invoke(messages)
print(result)

print("\n***************\n",result.content)