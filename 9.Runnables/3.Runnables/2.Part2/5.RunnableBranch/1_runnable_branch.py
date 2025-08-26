from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch


def word_count(text):
    return len(text.split())

load_dotenv()

prompt1=PromptTemplate(
    template='Write a deatailed report on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

# initialize the model
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser=StrOutputParser()


report_generator=RunnableSequence(prompt1,model, parser)
branch_chain=RunnableBranch(
    (lambda x: len(x.split())>200,RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)


final_chain=RunnableSequence(report_generator, branch_chain)
result=final_chain.invoke({'topic':'Israel vs Palestine'})
print(result)