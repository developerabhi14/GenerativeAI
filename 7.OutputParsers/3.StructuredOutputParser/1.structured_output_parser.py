from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers import OutputFixingParser
import os

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")



schema=[
    ResponseSchema(name='fact1', description='first fact about black hole'),
    ResponseSchema(name='fact2', description='second fact about black hole'),
    ResponseSchema(name='fact3', description='third fact about black hole')
]

parser=StructuredOutputParser.from_response_schemas(schema)
safe_parser=OutputFixingParser.from_llm(llm=model, parser=parser)

template=PromptTemplate(template=("give me 3 facts about {topic}"
                                  "Return only valid JSON that follows this format:\n\n"
                                    "{format_instruction}\n\n"
                                    "Do not add extra text."),
                        input_variables=["topic"],
                        partial_variables={"format_instruction":parser.get_format_instructions()}
                        )

chain=template | model | parser
result=chain.invoke({"topic":"Black Hole"})
print(result)

# print(result.fact1)  # you can access like this also