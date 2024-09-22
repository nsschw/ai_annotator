import pandas as pd
from .database import ChromaDB
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

        logging.info("Database initialized.")

        
    def add_data_from_csv(self, path: str, column_mapping: dict = {}, **kwargs) -> None:
        """
        - input and output are needed
        - split = training (default) 
        - id, reasoning optional
        """

        # handle column mapping
        default_column_mapping = {"id": "id", "input":"input", "output": "output", "reasoning": "reasoning", "split": "split"}
        default_column_mapping.update(column_mapping)
        column_mapping = default_column_mapping

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


    def generate_reasonings(self, model, **kwargs) -> None:
        # Gold Label-induced Reasoning: https://arxiv.org/pdf/2305.02105
        pass


    def predict(self, input: any, auto_save = True) -> list[str]:
        # predict on .csv, list..., test_examples...
        pass