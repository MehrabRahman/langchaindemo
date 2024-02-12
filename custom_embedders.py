import os
import requests
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from chromadb.api.types import (
    Documents,
    EmbeddingFunction
)
from dotenv import load_dotenv

class CustomChromaEmbedder(EmbeddingFunction):
    def __init__(self):
        load_dotenv()
        self.API_TOKEN = os.environ["API_TOKEN"]
        self.API_URL = os.environ["EMBEDDING_API"]

    def __call__(self, input: Documents):
        rest_client = requests.Session()
        response = rest_client.post(
            self.API_URL, json={"inputs": input}, headers={"Authorization": f"Bearer {self.API_TOKEN}"}
        ).json()
        return response
    
    def embed_documents(self, texts):
        return self(texts)
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]
    
default_embedder = HuggingFaceHubEmbeddings(
    model = os.environ["EMBEDDING_API"],
    huggingfacehub_api_token=os.environ["API_TOKEN"]
)
    