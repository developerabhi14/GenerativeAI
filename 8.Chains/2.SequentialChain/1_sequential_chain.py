from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt1=PromptTemplate(template="generate 3 detailed report on {topic}",
                      input_variables=["topic"])

prompt2=PromptTemplate(template="generate 3 point summary on following text \n {topic}",
                      input_variables=["text"])

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

parser=StrOutputParser()

chain=prompt1 | model | parser | prompt2 | model | parser
result=chain.invoke({"topic":"Ancient Aliens"})
print(result)