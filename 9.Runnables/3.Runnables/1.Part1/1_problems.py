import random

# problems
class NakliLLM():
    def __init__(self):
        print("NakliLLM initialized")

    def predict(self, prompt):
        response_list=[
            "This is a dummy response.",
            "NakliLLM says hello!",
            "How can I assist you today?"
        ]
        return {'response': random.choice(response_list)}

llm=NakliLLM()
prompt="Tell me a joke."

response=llm.predict(prompt)
print("Response:", response['response']) 


class NakliPromptTemplate():
    def __init__(self, template, input_variables):
        self.template=template
        self.input_variables=input_variables

    def format(self, **input_dict):
        return self.template.format(**input_dict)
    

template=NakliPromptTemplate(template="Tell me a {length}joke about {topic}.", 
                             input_variables=["length","topic"])
formatted_prompt=template.format(length="short",topic="computers")
print("Formatted Prompt:", formatted_prompt)

response=llm.predict(formatted_prompt)
print("Formatted Response:", response['response']) 


class NakliLLMChain():
    def __init__(self, llm, prompt):
        self.llm=llm
        self.prompt=prompt

    def run(self, input_dict):
        formatted_prompt=self.prompt.format(**input_dict)
        result=self.llm.predict(formatted_prompt)
        return result['response']
    
template=NakliPromptTemplate(template="Tell me a {length}joke about {topic}.", 
                             input_variables=["length","topic"])

llm=NakliLLM()
chain=NakliLLMChain(llm=llm, prompt=template)
response=chain.run({"length":"long","topic":"AI"})
print("Chain Response:", response)


