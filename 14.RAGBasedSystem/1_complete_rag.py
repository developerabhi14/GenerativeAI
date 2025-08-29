from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


# Step 1a: Indexing (Document Ingestion)


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

question="What does the video say about interrupt cycle. how does it work?"

retrieved_docs=retriever.invoke(question)

context_text="\n\n".join(doc.page_content for doc in retrieved_docs)


final_prompt=prompt.invoke({"context":context_text, "question":question})

# Step 4: Generation
answer=model.invoke(final_prompt)
print(answer.content)