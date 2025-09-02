from langchain.retrievers.multi_query import MultiQueryRetriever

class RetrieverFactory:
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

    def create(self):
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.llm)
