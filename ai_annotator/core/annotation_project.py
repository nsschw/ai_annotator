import pandas as pd
import os
import logging
import tqdm
from typing import Optional, Union

from .database import ChromaDB
from .models import Model

class AnnotationProject:
    """
    Class that serves as hub for annotation projects.
    Connnects database, annotation & embedding models.
    """

    def __init__(self, path: str, task_description: str, implementation = "ChromaDB") -> None:

        self.reasoning_available: bool = False
        self.task_description: str = task_description

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
        
        if reasoning_available:
            self.reasoning_available = True
        else:
            self.reasoning_available = False

        logging.info("Successfully added data!")


    def generate_reasonings(self, model: Model, reasoning_prompt: str = None, **kwargs) -> None:
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
        for entry in tqdm.tqdm(data):

            # reasoning already exisits
            if (entry.get("reasoning", None)) and (kwargs.get("overwrite", False) == False):
                logging.warning(f"Skipping reasoning for entry with ID {entry['id']} because reasoning already exists. Set overwrite=True to regenerate.")
                continue
            
            # reasoning doesn't exist
            if entry.get("split") in kwargs.get("split", ["train"]):
                entry["reasoning"] = model.generate_response([{"role": "user", "content": reasoning_prompt.format(output = entry["output"], input = entry["input"], task_description=self.task_description)}])
                self.db.update([entry])

        self.reasoning_available = True
        logging.info("Finished generating reasonings.")
       
        
        
    def predict(self, model: Model, input: Optional[Union[list, str, None]] = None, **kwargs) -> list[str]:   
        """
        Generate predictions using the provided model, with optional handling for incomplete reasoning data.

        Args:
            model: The model to use for generating predictions.
            input: Input data for the prediction.
                - None: Uses a validation split for prediction if no input is provided.
                - list: A list of multiple inputs for batch prediction.
                - str: A single input for prediction.
        
        Kwargs: 
            reasoning (bool, optional): Whether to include reasoning generation. Defaults to False.
            number_demonstrations (int, optional): The number of demonstrations to use. Defaults to 3.

        Returns:
            list[str]: A list of predicted outputs based on the provided input or validation split.
        """
        
        reasoning: bool = kwargs.get("reasoning", False)

        # handle missing reasoning data
        if reasoning and not self.reasoning_available:
            while True:
                do_reasoning: str = input("Reasoning data is incomplete. Would you like to generate the missing reasonings using the current model? [y/n]: ").strip().lower()

                if do_reasoning in {"y", "n"}:
                    break
                else:
                    print("Invalid input. Please enter 'y' for yes or 'n' for no.")

            if do_reasoning == "y":
                self.generate_reasonings(model)

        # determine generation logic according to input type
        if input is None:
            return self._predict_on_valsplit(model, **kwargs)
        if isinstance(input, list):
            return self._predict_list(model, **kwargs)        
        if isinstance(input, str):
            return self._predict_single_case(model, input, **kwargs)        
        raise TypeError("Invalid input type. Expected None, list, or str.")


    def _retrieve_k_similar(self, text, k) -> list[dict]:
        """
        Retrieves k most similar entries from the db
        """
        return self.db.query(text, k)
    

    def _predict_single_case(self, model: Model, input: str, **kwargs) -> str:
        """
        Predicts a single case
        
        kwargs:
            number_demonstrations
            reasoning

        TODO:
            Style: JSON/RAW STRING...
            Repeat Task: YES / NO
            System Prompt:
        """
        
        conversation: list[dict] = []
        demonstrations: list[dict] = self._retrieve_k_similar(input, kwargs.get("number_demonstrations", 3))

        for entry in demonstrations:
            # user part
            user: str = ""
            user += self.task_description + "\n"
            user += entry["input"] 
            conversation.append({"role": "user", "content": user})

            # assistant
            assistant: str = ""
            if kwargs.get("reasoning", False):
                assistant += "Thinking: " + entry["reasoning"] + "\n"
            assistant += assistant["output"]
            conversation.append({"role": "assistant", "content": assistant})

        user_request: str = ""
        user_request += self.task_description + "\n"
        user_request += input
        conversation.append({"role": "user", "content": user_request})

        return model.generate_response(conversation)
        

    def _predict_on_val_split(self, **kwargs):
        conv = []

        pass

    def _predict_list(self, **kwargs):
        pass