import pandas as pd
from .database import MilvusDB
import os
import logging

class AnnotationProject:
    """
    Class that servers as hub for annotation projects
    
    
    """


    def __init__(self, path: str) -> None:

        if os.path.exists(path):
            logging.info("Project with this name exisits already. Loading existing project....")

        self.db = MilvusDB(path)

        # check if path/name.db exists existis
        # if yes load
        # else create


        pass


    #@classmethod
    #def load(cls, path: str) -> 'AnnotationProject':
    #
    #    project : 'AnnotationProject' = cls()
    #
    #    return project
            

    def add_data_from_csv(self, path: str, column_mapping: dict = {"id": "id", "input":"input", "output": "output", "reasoning": "reasoning", "split": "split"}, **kwargs) -> None:
        
        # input and output is needed 
        # id, reasoning optional, split optional (train = default)

        df_import : pd.DataFrame = pd.read_csv("path")
        data : list[dict] = []

        for idx, row in df_import.iterrows():
            case = {}

            # needed
            case["input"] = row.get(column_mapping["input"], "TEST")
            case["output"] = row.get(column_mapping["output"], "TEST")

            # default
            case["split"] = row.get(column_mapping["split"], "train")

            # 

            # optional            
            

        
        
        

        self.db.insert_data(data=data)
        
        # rather convinience functions
        # if kwargs.get("generate_reasoning") == True: 
        # if kwargs.get("generate_embeddings") == True: 


    def generate_reasonings(self, model) -> None:
        pass


    def generate_embeddings(self, model) -> None:
        pass


    def predict(self, input: any, auto_save = True) -> list[str]:
        # predict on .csv, list..., test_cases...
        pass