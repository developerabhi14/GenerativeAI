from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
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

# 1st prompt (detailed report)
template1=PromptTemplate(template="Write a detailed report on {topic}",
                         input_variables=["topic"]
                         )

# 2nd prompt (short summary)
template2=PromptTemplate(template="Write a  5 line summary on the following {text}",
                         input_variables=["text"]
                         )

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser
result=chain.invoke({"topic":"Black Hole"})
print(result)