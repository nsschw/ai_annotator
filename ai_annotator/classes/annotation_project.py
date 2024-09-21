import pandas as pd
from .database import MilvusDB, ChromaDB
import os
import logging

class AnnotationProject:
    """
    Class that servers as hub for annotation projects
    Connnects database, annotation model & embedding model    
    
    """

    def __init__(self, path: str, implementation = "ChromaDB") -> None:
        if implementation == "ChromaDB":
            self.db = ChromaDB(path)

        

    def add_data_from_csv(self, path: str, column_mapping_input: dict = {}, **kwargs) -> None:
        """
        - input and output are needed
        - split = training (default) 
        - id, reasoning optional
        """

        column_mapping = {"id": "id", "input":"input", "output": "output", "reasoning": "reasoning", "split": "split"}
        column_mapping.update(column_mapping_input)
        print(column_mapping)

        df_import : pd.DataFrame = pd.read_csv(path)
        data : list[dict] = []

        reasoning_available: bool = df_import.to_dict("records")[0].get(column_mapping["reasoning"], None)
        id_available: bool = df_import.to_dict("records")[0].get(column_mapping["id"], None)

        for idx, row in df_import.iterrows():
            example:dict = {}

            # needed
            example["input"] = row.get(column_mapping["input"], "TEST")
            example["output"] = row.get(column_mapping["output"], "TEST")

            # default
            example["split"] = row.get(column_mapping["split"], "train")

            # optional
            if reasoning_available:
                example["reasoning"] = row.get(column_mapping["reasoning"], None)
            if id_available:
                example["id"] = row.get(column_mapping["id"], None)

            data.append(example)
   
        self.db.insert_data(data=data)        
        logging.info("Successfully added data!")        

        # check how the following could work with melvius!!!! 
        # rather convinience functions
        # if kwargs.get("generate_reasoning") == True: 
        # if kwargs.get("generate_embeddings") == True: 


    def generate_reasonings(self, model) -> None:
        pass


    def generate_embeddings(self, model) -> None:
        pass


    def predict(self, input: any, auto_save = True) -> list[str]:
        # predict on .csv, list..., test_examples...
        pass