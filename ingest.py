from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
import os
import pinecone

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '7a5457ba-c1ec-4e02-83c7-2ee0081c729d')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')


# Initializing the Pinecone
pinecone.init(
	api_key=PINECONE_API_KEY,
	environment=PINECONE_API_ENV
)
index_name = 'pincodeindex'

def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
    documents = loader.load()
    print("splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    #create embeddings here
    print("Loading sentence transformers model")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create Embeddings for Each of the Text Chunk and store it to pincone
    Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

if __name__ == "__main__":
    main()