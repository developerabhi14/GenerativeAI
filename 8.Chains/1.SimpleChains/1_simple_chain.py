from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt=PromptTemplate(template="generate 3 facts about {topic}",
                      input_variables=["topic"])

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

parser=StrOutputParser()

chain=prompt | model | parser
result=chain.invoke({"topic":"Interstellar object 3I Atlas"})
print(result)

chain.get_graph().print_ascii()