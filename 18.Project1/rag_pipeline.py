from prompts import retrieval_prompt

class RAGPipeline:
    def __init__(self, retriever, model, prompt=retrieval_prompt):
        """
        RAG Pipeline that orchestrates retrieval + generation.

        Args:
            retriever: LangChain retriever (e.g., MultiQueryRetriever)
            model: Chat model (e.g., ChatGoogleGenerativeAI)
            prompt: PromptTemplate for RAG
        """
        self.retriever = retriever
        self.model = model
        self.prompt = prompt

    def run(self, query: str) -> str:
        """
        Execute the full RAG pipeline for a given query.

        Args:
            query (str): User question

        Returns:
            str: Final answer
        """
        # 1. Retrieve documents
        results = self.retriever.invoke(query)
        if not results:
            return "No relevant documents found."

        # 2. Build context string from retrieved docs
        context = "\n\n".join(doc.page_content for doc in results)

        # 3. Format final prompt
        final_prompt = self.prompt.format(context=context, question=query)

        # 4. Generate answer from model
        answer = self.model.invoke(final_prompt)

        return answer.content
