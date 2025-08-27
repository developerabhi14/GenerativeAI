from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
loader=TextLoader('cricket.txt', encoding='utf-8')

# initialize the model
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

promt=PromptTemplate(
    template='Write a summary for the following {poem}', input_variables=['poem']
)

parser=StrOutputParser()

docs=loader.load()

# print(docs)
# print(type(docs))
# print(docs[0])
# print(docs[0].page_content)
# print(docs[0].metadata)

chain=promt | model | parser

result=chain.invoke({'poem': docs[0].page_content})
print(result)