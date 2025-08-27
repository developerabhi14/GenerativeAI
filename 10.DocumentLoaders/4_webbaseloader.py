from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# initialize the model
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt=PromptTemplate(template='Answer the following question \n {question}, from the following {text}',input_variables=['question', 'text'])

url="https://techlekh.com/dongfeng-nammi-vigo-price-nepal/"

loader=WebBaseLoader(url)

docs=loader.load()

parser=StrOutputParser()


chain=prompt | model | parser

result=chain.invoke({'question': 'What is the product that we are talking about','text':docs[0].page_content})
print(result)