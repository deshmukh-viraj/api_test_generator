import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = "docs/"
VECTOR_STORE_PATH = "vector_store"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_store():
    """loads documents from the 'docs' directory, splits them into chunks, creates embeddings and store it in vector store"""
    print("Starting data ingestion process...")
    
    if os.path.exists(VECTOR_STORE_PATH):
        print(f" Vector store already exists at {VECTOR_STORE_PATH}. Skipping Ingestion")
        return
    
    print(f"Loading documents from '{DOCS_PATH}'....")
    loader = DirectoryLoader(DOCS_PATH, glob="**/*.md", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()

    if not documents:
        print("No markdown documents found in the 'docs' directory")
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        return
    
    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Split documents into {len(docs)} chunks")
    
    print("Initialize Embedding Model....")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("Creating and Saving FAISS vector store...")
    vector_db = FAISS.from_documents(docs, embeddings)

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vector_db.save_local(os.path.join(VECTOR_STORE_PATH, "faiss_index"))
    print("Data ingestion complete")

    if __name__=="__main__":
        if not os.path.exists(DOCS_PATH):
            os.makedirs(DOCS_PATH)
            with open(os.path.join(DOCS_PATH, "sample_readme.mb"), 'w') as f:
                f.write("# Project README\n\nThis is a sample documentation file for the Github issue ")
            print("Created a sample 'docs' directoru and README file")

        create_vector_store()