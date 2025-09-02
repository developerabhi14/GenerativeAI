from langchain_core.prompts import PromptTemplate

# Retrieval QA prompt (safe against injection)
retrieval_prompt = PromptTemplate(
    template=(
        "You are a retrieval-based QA assistant. "
        "Use ONLY the provided context to answer the question. "
        "If the context does not contain the answer, reply strictly with: 'I don't know.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    input_variables=["context", "question"],
)

# Optional: Summarization prompt
summarization_prompt = PromptTemplate(
    template=(
        "Summarize the following context in a concise and clear way:\n\n"
        "{context}\n\n"
        "Summary:"
    ),
    input_variables=["context"],
)

# Optional: Rewriting query for retrieval
query_rewrite_prompt = PromptTemplate(
    template=(
        "You are a query rewriter. Rewrite the user query into a clearer, more search-friendly form.\n\n"
        "Original query: {question}\n\n"
        "Rewritten query:"
    ),
    input_variables=["question"],
)
