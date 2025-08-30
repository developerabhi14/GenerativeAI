from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# tool create
@tool
def add (a:int, b:int)-> int:
    """addition tool"""     
    return a+b

# print(add.invoke({"a":2,"b":3}))

# tool binding

llm_with_tools=llm.bind_tools([add])

user_query="can you add 15 with 16?"
query=HumanMessage(user_query)

messages=[query]

result=llm_with_tools.invoke(messages)

messages.append(result)

tool_result=add.invoke(result.tool_calls[0])

messages.append(tool_result)

print(llm_with_tools.invoke(messages).content)