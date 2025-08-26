from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableParallel, RunnableLambda


def word_count(text):
    return len(text.split())

load_dotenv()

prompt=PromptTemplate(
    template='Write exactly one joke about {topic}',
    input_variables=['topic']
)


# initialize the model
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

parser=StrOutputParser()


joke_generator=RunnableSequence(prompt,model, parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'num_words':RunnableLambda(word_count)
    })

final_chain=RunnableSequence(joke_generator, parallel_chain)
result=final_chain.invoke({'topic':'Electric Cars'})
print(result)