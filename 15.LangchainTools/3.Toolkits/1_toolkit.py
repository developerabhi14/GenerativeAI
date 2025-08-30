from langchain_core.tools import tool

# Custom tools

@tool
def add(a:int,b:int) -> int:
    """add two numbers"""
    return a+b

@tool
def sub(a:int, b:int)-> int:
    """sub two numbers"""
    return a-b


class MathToolKit:
    def get_tools(self):
        return [add, sub]
    
toolkit=MathToolKit()
tools=toolkit.get_tools()


for tl in tools:
    print(tl.name, " -> ",tl.description)