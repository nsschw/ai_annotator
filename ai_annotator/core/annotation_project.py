import pandas as pd
import os
import logging

from .database import ChromaDB
from .models import Model

class AnnotationProject:
    """
    Class that serves as hub for annotation projects
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

        reasoning_available: bool = df_import.to_dict("records")[0].get(column_mapping["reasoning"], False)
        id_available: bool = df_import.to_dict("records")[0].get(column_mapping["id"], False)

        for idx, row in df_import.iterrows():
            record: dict = {}

            # needed
            record["input"] = row.get(column_mapping["input"], KeyError)
            record["output"] = row.get(column_mapping["output"], KeyError)

            # default
            record["split"] = row.get(column_mapping["split"], "train")

            # optional
            if reasoning_available:
                record["reasoning"] = row.get(column_mapping["reasoning"], None)
            if id_available:
                record["id"] = row.get(column_mapping["id"], None)

            data.append(record)
   

        self.db.insert_data(data=data)        
        logging.info("Successfully added data!")


    def generate_reasonings(self, model: Model, task_description: str, reasoning_prompt: str = None, **kwargs) -> None:
        """
        Idea: Gold Label-induced Reasoning: https://arxiv.org/pdf/2305.02105
        
        Generates reasoning on the train split to have higher quality demonstrations
 
        Takes: model and custom prompt. Custom Prompt needs to have {input} and {output}.
        """      
        # extract data
        data = self.db.full_extract()

        # set up prompt
        if reasoning_prompt is None:
            logging.warning("No reasoning prompt was given. Falling back to default prompt")
            with open("prompts/gold_label-induced_reasoning.txt", "r") as f:
                reasoning_prompt = f.read()
        else:
            try:
                reasoning_prompt.format(output = "TEST", input = "TEST")
            except:
                raise ValueError("Invalid reasoning prompt format. Ensure it contains {input} and {output} placeholders.")
        prompt = task_description + "\n" + reasoning_prompt


        logging.info("Start generating embeddings for entries without.")
        for entry in data:
            if (entry.get("reasoning", None)) and (kwargs.get("overwrite", False) == False):
                logging.warning(f"Skipping reasoning for entry with {entry['id']} as reasoning already exists. Set overwrite = True to regenerate.")
                continue

            entry["reasoning"] = model.generate_response({"role": "user", "content": prompt.format(output = entry["output"], input = entry["input"])})

        self.db.update(data)

        
    def predict(self, input: any, **kwargs) -> list[str]:
        # predict on .csv, list..., test_examples...

        number_demonstrations: int = kwargs.get("number_demonstrations", 3)
        reasoning: bool = kwargs.get("reasoning", False)

        # check if reasoning avaiable

        #else:
        #    self.generate_reasonings()
        

        pass