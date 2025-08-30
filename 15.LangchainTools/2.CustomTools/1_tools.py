from langchain_core.tools import tool

# step 1: create a function
def multiply(a,b):
    """multiply two numbers"""
    return a*b

# step 2: add hint types
def multiply(a:int, b:int) -> int:
    """multiply two numbers"""
    return a*b

# step 3: add tool decorator
@tool
def multiply(a:int, b:int) -> int:
    """multiply two numbers"""
    return a*b

result=multiply.invoke({"a":3, "b":6})

print(result)

print(multiply.name) # function name
print(multiply.description) # docstring
print(multiply.args) # parameters

# when interacting with llm
print(multiply.args_schema.model_json_schema())