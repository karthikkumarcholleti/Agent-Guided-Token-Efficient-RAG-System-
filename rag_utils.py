from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os


def connect_to_vector_store():
    vector_store = Chroma(
        collection_name = "my_company_policies",
        embedding_function = OpenAIEmbeddings(
            api_key = os.getenv("OPENAI_API_KEY"),
            model = "text-embedding-3-small",
            max_retries = 3,
            request_timeout = 10
            ),
        persist_directory = "./VectorStore")
    return vector_store    



def filter_documens_content(documents):
    docs = []
    for doc in documents:
        content = doc.page_content
        docs.append([content])
    return docs



def retrieve_documents(query, k = 5):
    vector_store = connect_to_vector_store()
    results = vector_store.similarity_search(
        query = query, 
        k = k)
    return results



def get_sources(documents):
    sources = []
    for doc in documents:
        source = doc.metadata['source']
        source = source.replace("Documents/", "")
        sources.append([source])
    return sources