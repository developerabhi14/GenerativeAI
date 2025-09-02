from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def get_model():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash")


class RAGSystem():
    def document_loader(self)-> any:
        """
        Loads pdf document using a PyPDFLoader
        """
        loader=PyPDFLoader("books/report.pdf")
        docs=loader.load()
        return docs
    
    def text_splitter(self, docs: any)-> any:
        splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        result=splitter.split_documents(docs)
        return result
    
    def vectorstore(self, docs):
        embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store=FAISS.from_documents(documents=docs, embedding=embedding_model)
        multiquery_retriever=MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(searc_kwargs={"k":2}),
            llm=get_model()
        )
        return multiquery_retriever
    
prompt=PromptTemplate(template="You are a helpful assistant. Answer only from provided document. If the context is insufficient, just say you don't know \n{context} \n Question:{question} ",
                      input_variables=['context', 'question'])


class RAGRunner:
    def __init__(self):
        self.system = None
        self.retriever = None
        self.prompt = None
        self.model = None
        self.first_run = True

    def first_execution(self, query):
        self.system = RAGSystem()

        print("\n****Loading Document****")
        docs = self.system.document_loader()

        print("\n****Splitting text****")
        result = self.system.text_splitter(docs)

        print("\n****Initializing vectorstore****")
        self.retriever = self.system.vectorstore(result)

        # Save prompt and model once
        self.prompt = prompt
        self.model = get_model()

        return self.run_pipeline(query)

    def run_pipeline(self, query):
        print("\n****Invoking retriever****")
        result = self.retriever.invoke(query)

        context_text = "\n\n".join(doc.page_content for doc in result)

        print("\n****Invoking final prompt****")
        final_prompt = self.prompt.invoke({"context": context_text, "question": query})

        print("\n****Generating answer****")
        answer = self.model.invoke(final_prompt)
        print(answer.content)

    def run(self):
        while True:
            query = input("\nEnter your query (or type 'exit' to quit): ")
            if query.lower() == "exit":
                print("Exiting...")
                break

            if self.first_run:
                self.first_execution(query)
                self.first_run = False
            else:
                self.run_pipeline(query)


if __name__ == "__main__":
    runner = RAGRunner()
    runner.run()
