from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,  # temperature controls randomness of language model's output. affects creativity and variability.
        max_new_tokens=100  # max_new_tokens controls the maximum number of tokens to generate
    )

)

model=ChatHuggingFace(llm=llm)
result=model.invoke("What is the capital of India?")
print(result)
print(result.content)  # if you want to see answer only 