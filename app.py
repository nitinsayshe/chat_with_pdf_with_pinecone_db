from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
import pinecone
import os
import streamlit as st
from time import sleep

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '7a5457ba-c1ec-4e02-83c7-2ee0081c729d')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter')
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def question_ans(question,docs):
    llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    chain=load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=question)
    return answer
# Initializing the Pinecone
def init_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY,environment=PINECONE_API_ENV)
    return

def main():
    st.title('Search PDF')
    question =st.text_area("Enter question")
    if st.button("Search"):
        # do similarity search
        init_pinecone()  #initalize the pincone
        docsearch = Pinecone.from_existing_index(index_name="pincodeindex", embedding=embeddings)
        docs = docsearch.similarity_search(question,k=5)
        answer = question_ans(question,docs)
        st.write(answer)
        st.write(docs)

if __name__ == "__main__":
    main()
