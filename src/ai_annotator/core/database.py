import chromadb
from .config import AnnotationConfig
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

    def __init__(self, config: AnnotationConfig) -> None:
        self.client = chromadb.PersistentClient(path=config.db_path)
        
        if not config.embedding_model:
            self.collection = self.client.get_or_create_collection(config.collection_name)
            logging.warning("No embedding_model passed. Defaulting to ChromaDB's default model: 'all-MiniLM-L6-v2'")
        else:
            self.collection = self.client.get_or_create_collection(config.collection_name,  embedding_function=config.embedding_model)


    def insert_data(self, records: list[dict]) -> None:
        """
        Inserts a list of data records into the database collection.
        
        Args:
            data: A list of dictionaries where each dictionary represents a data record.
        
        Notes:
            - The "input" key in each dictionary is used as the document content and is automatically tokenized and vectorized.
            - If the "id" key is not provided in the dictionaries, a warning is logged and the index of the record is used as the ID.
        """

        documents: list[str] = [record.pop("input") for record in records]

        if records[0].get("id", None):
            ids: list = [record.pop("id") for record in records]
        else:
            ids: list = [f"id{i}" for i in range(len(records))]
            logging.warning("No IDs inserted. Using the index as ID")

        self.collection.add(
            documents = documents,
            metadatas = records,
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
        records: list[dict] = output["metadatas"]
        for i, record in enumerate(records):
            record["input"] = output["documents"][i]
            record["id"] = output["ids"][i]
        return records
    
    
    def update(self, records: list[dict]):
        """
        Updates the collection with the provided data.
        Args:
            data: A list of dictionaries where each dictionary represents a document to be updated.
        
        Example:
            records = [
                {"input": "Document content 1", "id": "doc1", "metadata_key": "value1"},
                {"input": "Document content 2", "id": "doc2", "metadata_key": "value2"}
            ]
        """

        documents: list[str] = [record.pop("input") for record in records]
    
        if records[0].get("id", None):
            ids: list = [record.pop("id") for record in records]
        else:
            raise ValueError("ID field is missing in the data entries")
    
        self.collection.upsert(
            documents = documents,
            metadatas = records,
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
        records: list[dict] = query_results["metadatas"][0]
        
        for i, example in enumerate(records):
            example["input"] = query_results["documents"][0][i]
        return records