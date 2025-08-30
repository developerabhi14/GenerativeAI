from langchain_community.tools import DuckDuckGoSearchRun

search_tool=DuckDuckGoSearchRun()

result=search_tool.invoke("arsenal vs liverpool detailed preview")

print(result)