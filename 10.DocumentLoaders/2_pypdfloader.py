from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv


loader=PyPDFLoader('dl-curriculum.pdf')
docs=loader.load()
# print(docs[0])
print(docs[1].metadata)

