from pymilvus import MilvusClient
import numpy as np
import chromadb
import logging


class db():

    def insert_data():
        pass

    def query():
        pass



class ChromaDB:    
    """
    TODO:
    - [ ] Add possibility of using a custom embedding model
    """

    def __init__(self, path: str, **kwargs):
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name : str = kwargs.get("collection_name", "Demo")
        self.collection = self.client.get_or_create_collection(self.collection_name)

        
    def insert_data(self, data: list[dict]):
        # chromadb automatically tokenizes & vectorizes the documents
        documents: list[str] = [entry.pop("input") for entry in data]

        if data[0].get("id", None):
            ids: list = [entry.pop("id") for entry in data]
        else:
            logging.warning("No IDs inserted. Using the index as ID")
            ids: list = [str(f"id{i}" for i in range(len(data)))]

        self.collection.add(
            documents = documents,
            metadatas = data,
            ids = ids
        )

    def query(self, text: str, k = 3, **kwargs): # TODO: Find return type!!!
        doc = self.collection.query(
                query_texts=[text],
                n_results=k,
                where={"split": "train"},
            )
        
        return doc
        




class MilvusDB:
    """
    Based on MilvusLite, therefore not as fuctional
    Some functions might be weird, bc MilvusLite has a reduced functioality -> Upgrade if needed...

    """


    def __init__(self, path: str, **kwargs) -> None:        
        self.client = MilvusClient(path)
        self.collection_name : str = kwargs.get("collection_name", "Demo")

        if self.client.has_collection(collection_name=self.collection_name):
            pass
        else: 
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=768,
            )


    def insert_data(self, data: list[dict]) -> None:
        # check if consistent with available data

        # insert data
        self.client.insert(
            collection_name=self.collection_name,
            data=data)


    def add_embeddings(self, model):
        pass
         
        
        
        

    def search(self, vector: np.array, k: int=3):

        res = self.client.search(
            collection_name="demo_collection",
            data=[vector],
            limit=k,
            output_fields=["text", "subject"],
            )