from abc import ABC, abstractmethod
import random

class Runnable(ABC):
    @abstractmethod
    def invoke(input_data):
        pass

class NakliLLM(Runnable):
    def __init__(self):
        print("NakliLLM initialized")

    def invoke(self, prompt):
        response_list=[
            "This is a dummy response.",
            "NakliLLM says hello!",
            "How can I assist you today?"
        ]
        return {'response': random.choice(response_list)}

    def predict(self, prompt):
        response_list=[
            "This is a dummy response.",
            "NakliLLM says hello!",
            "How can I assist you today?"
        ]
        return {'message':'this method is going to be deprecated','response': random.choice(response_list)}
    
class NakliPromptTemplate():
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)  # <-- unpack dict

    def format(self, input_dict):
        return self.template.format(**input_dict)  # <-- unpack dict

    
class RunnableConnect(Runnable):
    def __init__(self, runnables_list):
        self.runnables_list=runnables_list

    def invoke(self, input_data):
        current_data=input_data
        for runnable in self.runnables_list:
            current_data=runnable.invoke(current_data)
        return current_data


class NakliStrOutputParser():
    def parse(self, llm_output):
        return llm_output['response']
    
    def invoke(self, llm_output):
        return llm_output['response']
    
llm=NakliLLM()
template=NakliPromptTemplate(template="Tell me a {length}joke about {topic}.", 
                             input_variables=["length","topic"])
parser=NakliStrOutputParser()

chain=RunnableConnect([template, llm, parser])

response=chain.invoke({"length":"short","topic":"computers"})
print(response)

template1=NakliPromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

tempalate2=NakliPromptTemplate(
    template='Explain the following joke {response}',
    input_variables=['response']
)

llm=NakliLLM()
parser=NakliStrOutputParser()

chain1=RunnableConnect([template1, llm])
chain1.invoke({"topic":"AI"})

chain2=RunnableConnect([tempalate2, llm, parser])
final_chain=RunnableConnect([chain1, chain2])
final_chain.invoke({"topic":"cricket"})
print(final_chain)