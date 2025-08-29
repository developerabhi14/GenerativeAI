from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

video_id="W0ZfeKPEQeM"

try:
    # if you don't care about the language, this returns the best one
    ytt_api=YouTubeTranscriptApi()
    transcript_list=ytt_api.fetch(video_id=video_id, languages=['en'])

    # flatten it to plain text
    transcript=" ".join(chunk.text for chunk in transcript_list)

except TranscriptsDisabled:
    print("No captions available for the video")


# Step 1b: Indexing(Text Splitting)
splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks=splitter.create_documents([transcript])

# print(len(chunks))
# print(chunks[0])

# Step 1c & 1d- Indexing (Embedding Generation and Storing in Vector Store)
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store=FAISS.from_documents(chunks, embedding=embedding_model)

# Step 2: Retrieval
retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

# retriever.invoke("What is the channel name?")

model=ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt=PromptTemplate(template="""
You are a helpful assistant. Answer only from the provided transcript context. If the context is insufficient, just say you don't know.
                      {context}
                      Question:{question}                      
""",
input_variables=['context', 'question'])

def format_docs(retrieved_docs):
    context_text="\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain=RunnableParallel({
    'context':retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})


parser=StrOutputParser()

main_chain=parallel_chain| prompt | model | parser 

result=main_chain.invoke("What is the video talking about?")
print(result)