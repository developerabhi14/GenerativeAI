from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate


load_dotenv()

# initialize the model
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# create a prompt template
prompt=PromptTemplate(template="Suggest a catchy blog title about {topic}",
                        input_variables=["topic"])

# define the input
topic=input("Enter the topic you want a blog title for:")

# format the promp manually using prompt template
formatted_prompt=prompt.format(topic=topic)

# call the llm directly
response=model.invoke(formatted_prompt)

# print the response
print("Generated Blog Title:", response.content)