from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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

prompt1=template1.invoke({"topic":"Black Hole"})

result1=model.invoke(prompt1)
print(result1)

print("--------------------------------------------------" * 3 )
prompt2=template2.invoke({"text":result1.content})
result2=model.invoke(prompt2)
print(result2.content)