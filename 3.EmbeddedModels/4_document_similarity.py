from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)



documents = ["Virat Kohli is a famous cricketer",
             "Sachin Tendulkar is a legendary cricketer",
             "Lionel Messi is a world-renowned footballer"
]

query='tell me about Virat Kohli'

doc_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], doc_embeddings)
print("Query:", query)
print("Similarity Scores:", similarity_scores)
print("Most similar document:", documents[similarity_scores.argmax()])
print("Similarity score:", similarity_scores.max())