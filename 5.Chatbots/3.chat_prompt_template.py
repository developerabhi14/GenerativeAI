from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


chat_template=ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert.'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt=chat_template.invoke({
    'domain': 'cricket',
    'topic': 'Dusra'
})
print(prompt)