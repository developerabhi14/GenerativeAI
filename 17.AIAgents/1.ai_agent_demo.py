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

load_dotenv()
search_tool=DuckDuckGoSearchRun()
# results=search_tool.invoke("arsnal vs liverpool preview", kwargs={'language':"en"})
# print(results)



llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt=hub.pull('hwchase17/react') # puss the standdard ReAct agent people (reason and act)

# create react agent manually with pulled prompt
agent=create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

# wrap it using agentexecutor
agent_executor=AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
    handle_parsing_errors=True
)

# invoke

response=agent_executor.invoke({"input":"give me 5 pointer preview for today's match between arsenal and liverpool"})
print(response)
