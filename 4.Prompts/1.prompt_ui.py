from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import load_prompt

load_dotenv()

st.header("Research Tool")
model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")
user_input=st.text_input("Enter your static  prompt")

if st.button("Static prompt Submit"):
    # static prompt passed to the model
    # but we don't use static prompt too much in parctice. Because a minor change in prompt can lead to a different result.
    # so we use dynamic prompt, which is based on user input.
    result=model.invoke(user_input)
    st.text(result.content)


paper_input=st.selectbox("Select a research paper", ["Attention is all you need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input=st.selectbox("Select explaination style", ["Beginner-friendly", "Technical", "Code-oriented", "Mathematical"])

length_input=st.selectbox("Select explaination length", ["Short(1-2 paragraphs)", "Medium(3-5 paragraphs)", "Long(detailed explanation)"])
 
template=load_prompt("template.json")

#fill the placeholders


if st.button("Dynamic prompt Submit"):
    chains= template | model
    result = chains.invoke({'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input})
    st.write(result.content)
