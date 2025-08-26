from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence


load_dotenv()

prompt=PromptTemplate(
    template='Write about joke about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Explain the joke \n {joke}",
    input_variables=['joke']
)
# initialize the model
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser=StrOutputParser()

chain=RunnableSequence(prompt,model, parser, prompt2, model, parser)


result=chain.invoke({'topic':'AI'})
print(result)