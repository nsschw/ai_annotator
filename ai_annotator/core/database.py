import numpy as np
import chromadb
from .embedding_models import EmbeddingModel

from typing import Optional
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

    def __init__(self, path: str, embedding_model: Optional[EmbeddingModel] = None, **kwargs) -> None:
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name : str = kwargs.get("collection_name", "Demo")

        if kwargs.get("embedding_function", None):
            self.collection = self.client.get_or_create_collection(self.collection_name, embedding_function=embedding_model) 

    def insert_data(self, data: list[dict]) -> None:
        """
        Inserts a list of data entries into the database collection.
        
        Args:
            data: A list of dictionaries where each dictionary represents a data entry.
        
        Notes:
            - The "input" key in each dictionary is used as the document content and is automatically tokenized and vectorized.
            - If the "id" key is not provided in the dictionaries, a warning is logged and the index of the entry is used as the ID.
        """

        documents: list[str] = [entry.pop("input") for entry in data]

        if data[0].get("id", None):
            ids: list = [entry.pop("id") for entry in data]
        else:
            ids: list = [f"id{i}" for i in range(len(data))]
            logging.warning("No IDs inserted. Using the index as ID")

        self.collection.add(
            documents = documents,
            metadatas = data,
            ids = ids
        )
    
    def full_extraction(self) -> list[dict]:
        """
        Exports all relevant data (metadata and document)

        Returns:
            A list of dicts
        """

        output = self.collection.get(
            include=["documents", "metadatas"]
        )

        # restructure to fit the projects general structure -> most similar doc first
        data: list[dict] = output["metadatas"]
        for i, example in enumerate(data):
            example["input"] = output["documents"][i]
            example["id"] = output["ids"][i]
        return data
    
    
    def update(self, data: list[dict]):
        """
        Updates the collection with the provided data.
        Args:
            data: A list of dictionaries where each dictionary represents a document to be updated.
        
        Example:
            data = [
                {"input": "Document content 1", "id": "doc1", "metadata_key": "value1"},
                {"input": "Document content 2", "id": "doc2", "metadata_key": "value2"}
            ]
            update(data)
        """

        documents: list[str] = [entry.pop("input") for entry in data]
    
        if data[0].get("id", None):
            ids: list = [entry.pop("id") for entry in data]
        else:
            raise ValueError("ID field is missing in the data entries")
    
        self.collection.upsert(
            documents = documents,
            metadatas = data,
            ids = ids
        )


    def query(self, text: str, k = 3, split = "train") -> list[dict]: 
        """
        Queries the DB for k similar entries using the embeddings.
        
        Args:
            text: String that should be comparable to the entries in the db
            k: Amount of similar cases
            split: Split to query
        """
        query_results = self.collection.query(
                query_texts=[text],
                n_results=k,
                where={"split": split},
            )
        
        # restructure to fit the projects general structure -> most similar doc first
        data: list[dict] = query_results["metadatas"][0]
        
        for i, example in enumerate(data):
            example["input"] = query_results["documents"][0][i]
        return data