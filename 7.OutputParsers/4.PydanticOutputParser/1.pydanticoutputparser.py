from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate 
from langchain.output_parsers import StructuredOutputParser, ResponseSchema, PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
import os
from pydantic import BaseModel, Field

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")


class Person(BaseModel):
    name: str = Field(description="The person's full name")
    age: int=Field(gt=18, lt=20,description="The person's age, must be greater than 18")
    city: str =Field(description="The city where the person lives")


parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(template=("give me the name, age and city of a fictional {place} person\n"
                                  "Make sure the age is greater than 18.\n" 
                                    "Return the response in the following format:\n\n"
                                    "{format_instruction}\n\n"),
                                    input_variables=["place"],
                                    partial_variables={"format_instruction":parser.get_format_instructions()}
                                    )

prompt=template.invoke({"place":"Nepali"})
print(prompt)                         
chain=template | model | parser
result=chain.invoke({"place":"Nepali"})

print(result)