from langchain_community.document_loaders import PyPDFLoader 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS

class DocumentIndexer:
    def __init__(self, filepath: str, embedding_model: str):
        self.filepath = filepath
        self.embedding = HuggingFaceEmbeddings(model_name=embedding_model)

    def load_and_split(self):
        docs = PyPDFLoader(self.filepath).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(docs)

    def build_or_load_index(self, persist_dir: str):
        if os.path.exists(persist_dir):
            return FAISS.load_local(persist_dir, self.embedding, allow_dangerous_deserialization=True)
        else:
            docs = self.load_and_split()
            vs = FAISS.from_documents(docs, self.embedding)
            vs.save_local(persist_dir)
            return vs
