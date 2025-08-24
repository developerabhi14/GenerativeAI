from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
load_dotenv()
hf_api_token = os.environ.get("HUGGINGFACEHUB_ACCESS_TOKEN")
llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,  # temperature controls randomness of language model's output. affects creativity and variability.
        max_new_tokens=100  # max_new_tokens controls the maximum number of tokens to generate
    )

)

model=ChatHuggingFace(llm=llm)
parser=JsonOutputParser()

template=PromptTemplate(template="give me the name, age and city of a fictional person \n {format_instruction}",
                            input_variables=[],
                            partial_variables={"format_instruction":parser.get_format_instructions()}
                            ) 

chain=template | model | parser
rseilt=chain.invoke({})
print(rseilt)

# but you cannot enforce an schema