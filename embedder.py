import os
import chromadb
import requests

from typing import Any, Dict, List, Optional

from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings
)

class CustomEmbedder(EmbeddingFunction):
    def __init__(self) -> None:
        self.API_TOKEN = ""
        self.API_URL = ""

    def __call__(self, input: Documents) -> Embeddings:
        rest_client = requests.Session()
        response = rest_client.post(
            self.API_URL, json={"inputs": input}, headers={"Authorization": f"Bearer {self.API_TOKEN}"}
        ).json()
        return response

    def embed_query(self, text: str) -> List[float]:
        print(f"CustomEmbedder.embed_query called with query {text}")
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print("CustomEmbedder.embed_documents() function called.... ")
        response = requests.post(
            "",
            headers={"Authorization": f"Bearer {self.API_TOKEN}"},
            json={
                "inputs": texts,
                "options": {"wait_for_model": True, "use_cache": True},
            },
        )
        print("CustomEmbedder.embed_function returning the following json")
        print(response.json())
        print("------------")
        return response.json()

def main():
    dataset = [
        "My cat is named Francis.",
        "I want to visit Italy.",
        "I need to download more RAM."
    ]
    metadatas = [
        {"doc_name": "testdoc"},
        {"doc_name": "testdoc"},
        {"doc_name": "testdoc"}
    ]
    ids = [
        "1",
        "2",
        "3"
    ]
    #chroma_client = chromadb.PersistentClient(path="../testdb")

    custom_embedder = CustomEmbedder()
    #chroma_client.delete_collection("test_collection")
    #collection = chroma_client.create_collection(name="test_collection", embedding_function=custom_embedder)
    
    #collection.add(
    #    documents=dataset,
    #    metadatas=metadatas,
    #    ids=ids
    #)

    # for i, dataset_id in enumerate(dataset):
    #     collection.add(
    #         documents=[dataset[i]],
    #         metadatas=[{"doc_name": "testdoc"}],
    #         ids=[str(i)]
    #     )

    #results = collection.query(
    #    query_texts=["travel"],
    #    n_results=2
    #)

    # collection.update(
    #     ids=["2"],
    #     documents="This is an update",
    #     metadatas=[{"source": "I made it up"}]
    # )

    # print(collection.peek())

    #print(results)
    

if __name__ == "__main__":
    main()