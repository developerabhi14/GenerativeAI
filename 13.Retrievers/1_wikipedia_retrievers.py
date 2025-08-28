from langchain_community.retrievers import WikipediaRetriever

retriever=WikipediaRetriever(top_k_results=2, lang="en")

query="artificial intelligence"

docs=retriever.invoke(query)
print(len(docs))
for i, doc in enumerate(docs):
    print(f"-----Result {i+1}------")
    print(f"content: \n {doc.page_content}")
