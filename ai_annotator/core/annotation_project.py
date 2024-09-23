import pandas as pd
import os
import logging

from .database import ChromaDB
from .models import Model

class AnnotationProject:
    """
    Class that serves as hub for annotation projects.
    Connnects database, annotation & embedding models.
    """

    def __init__(self, path: str, implementation = "ChromaDB") -> None:

        if implementation == "ChromaDB":
            self.db = ChromaDB(path)

        logging.info("Database initialized.")

        
    def add_data_from_csv(self, path: str, column_mapping: dict = {}, **kwargs) -> None:
        """"
        Reads a CSV file and adds its data to the database.
        This function reads data from a CSV file specified by the `path` parameter,
        maps the columns according to the `column_mapping` dictionary, and inserts
        the data into the database. 
        
        Args:
            path: The file path to the CSV file to be read.
            column_mapping: dictionary mapping the default column names to the CSV column names. 
                            Change value the according column name.
        
        Returns: 
            None
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
        Queries DB to generate gold label-induced reasoning. Refer to https://arxiv.org/pdf/2305.02105 for more details. 
        
        Args:
            model: An instance of a class with a "generate_response" method.
            task_description: A description of the task for which reasoning is generated.
            reasoning_prompt: A custom prompt containing placeholders {task_description}, {input}, and {output}. 

        Kwargs:
            split: A list of splits to generate reasoning for. Default is ["train"]. 
            overwrite: A boolean indicating whether to overwrite existing reasoning. Default is False.

        Returns:
            None
        """

        # extract data
        data = self.db.full_extract()

        # set up prompt
        if reasoning_prompt is None:
            logging.warning("Reasoning prompt not provided. Using default prompt instead.")
            with open("ai_annotator/prompts/gold_label-induced_reasoning.txt", "r") as f:
                reasoning_prompt = f.read()
        else:
            try:
                reasoning_prompt.format(output = "TEST", input = "TEST", task_description = "TEST")
            except:
                raise ValueError("Invalid reasoning prompt format. Ensure it contains {task_description}, {input} and {output} placeholders.")
        
        # generate reasonings
        logging.info("Starting to generate reasoning for entries without existing reasoning.")
        for entry in data:

            # reasoning already exisits
            if (entry.get("reasoning", None)) and (kwargs.get("overwrite", False) == False):
                logging.warning(f"Skipping reasoning for entry with ID {entry['id']} because reasoning already exists. Set overwrite=True to regenerate.")
                continue
            
            # reasoning doesn't exist
            if entry.get("split") in kwargs.get("split", ["train"]):
                entry["reasoning"] = model.generate_response([{"role": "user", "content": reasoning_prompt.format(output = entry["output"], input = entry["input"], task_description=task_description)}])
                self.db.update([entry])

        logging.info("Finished generating reasonings.")
        
        
    def predict(self, input: any, **kwargs) -> list[str]:
        # predict on .csv, list..., test_examples...

        number_demonstrations: int = kwargs.get("number_demonstrations", 3)
        reasoning: bool = kwargs.get("reasoning", False)

        # check if reasoning avaiable

        #else:
        #    self.generate_reasonings()
        

        pass