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

# tools calling (but llm does not actually call the tool, it only suggests you)
result=llm_with_tools.invoke("can you add 3 with 14")
# print(result.tool_calls[0])
argument=result.tool_calls[0]['args']
print(add.invoke(argument))

# or you can also call toolcall
print(add.invoke(result.tool_calls[0]))