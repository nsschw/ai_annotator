from pymilvus import MilvusClient
import numpy as np
import chromadb

import logging
import abc


class DB(abc.ABC):
    @abc.abstractmethod
    def insert_data(self, path: str):
        """
        Takes data as a list of dicts which contain this projects standard keys (input, output, id, reasoning, split) and inputs them in the databases native way.
        """
        pass

    @abc.abstractmethod
    def query(self, text: str, k: int) -> list[dict]:
        """
        Takes a string as input and queries the db in its native way returning k most similar entries in the standard style
        """
        pass



class ChromaDB(DB):    
    """
    TODO:
    - [ ] Add possibility of using a custom embedding model
    """

    def __init__(self, path: str, **kwargs) -> None:
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name : str = kwargs.get("collection_name", "Demo")
        self.collection = self.client.get_or_create_collection(self.collection_name)

        
    def insert_data(self, data: list[dict]) -> None:
        # chromadb automatically tokenizes & vectorizes the documents
        documents: list[str] = [entry.pop("input") for entry in data]

        if data[0].get("id", None):
            ids: list = [entry.pop("id") for entry in data]
        else:
            ids: list = [str(f"id{i}" for i in range(len(data)))]
            logging.warning("No IDs inserted. Using the index as ID")

        self.collection.add(
            documents = documents,
            metadatas = data,
            ids = ids
        )


    def query(self, text: str, k = 3, **kwargs) -> list[dict]: 
        query_results = self.collection.query(
                query_texts=[text],
                n_results=k,
                where={"split": "train"},
            )
        
        # restructure to fit the projects general structure -> most similar doc first
        data: list[dict] = query_results["metadatas"][0]
        
        for i, example in enumerate(data):
            example["input"] = query_results["documents"][0][i]
        return data