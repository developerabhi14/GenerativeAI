from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
load_dotenv()


model1=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative']=Field(description="The sentiment of the feedback, must be either positive or negative")

parser2=PydanticOutputParser(pydantic_object=Feedback)
prompt1=PromptTemplate(template="classify the sentiment of following feedback text into positive or negative \n {feedback} \n {format_instruction}",
                        input_variables=["feedback"],
                        partial_variables={"format_instruction":parser2.get_format_instructions()}
                        )


classifier_chain=prompt1 | model1 | parser2

prompt2=PromptTemplate(template="write an appropriate response to this positive feedback \n{feedback}",
                        input_variables=["feedback"]
                        )
prompt3=PromptTemplate(template="write an appropriate response to this negative feedback \n{feedback}",
                        input_variables=["feedback"]
                        )
branch_chain=RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model1 | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model1 | parser),
    RunnableLambda(lambda x: "No valid sentiment found")
)

chain=classifier_chain | branch_chain

result=chain.invoke({"feedback":"This is a beautiful place!"})
print(result)