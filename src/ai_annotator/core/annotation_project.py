import pandas as pd
import logging
import tqdm
from typing import Optional, Union

from .database import ChromaDB
from .config import AnnotationConfig, PathConfig

class AnnotationProject:

    def __init__(self,
                db_path: str = None,
                task_description: str = None,
                config: Optional[AnnotationConfig] = None
                ) -> None:
        """
        It is recommended to pass an AnnotationConfig, but the project can also be initialized using db_path and task_description,
        defaulting to standard models if necessary.
        """

        if (not task_description or not db_path) and not config:
            raise ValueError("Either 'config' and 'task_description' or 'db_path' must be provided.")
        
        self.config = config or AnnotationConfig(task_description = task_description, db_path=db_path)
        self.db = ChromaDB(self.config)           
        logging.info("Database initialized.")

        # tracking vars
        self.reasoning_available: bool = False

        
    def add_data_from_csv(self, path: str, column_mapping: dict = {}, split: str = "train", **kwargs) -> None:
        """"
        Reads a CSV file and adds its data to the database.

        Args:
            path: The file path to the CSV file to be read.
            column_mapping: dictionary mapping the default column names to the CSV column names. 
                            Change value the according column name.
            split: If the split is not given as a column in the CSV, this value will be used for the entries. 
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
            entry: dict = {}

            # needed
            entry["input"] = row.get(column_mapping["input"], KeyError)
            entry["output"] = row.get(column_mapping["output"], KeyError)

            # default
            entry["split"] = row.get(column_mapping["split"], split)

            # optional
            if reasoning_available:
                entry["reasoning"] = row.get(column_mapping["reasoning"], None)
            if id_available:
                entry["id"] = row.get(column_mapping["id"], None)

            data.append(entry)
   

        self.db.insert_data(data=data)        
        
        if reasoning_available:
            self.reasoning_available = True
        else:
            self.reasoning_available = False

        logging.info("Successfully added data.")


    def generate_reasoning(self, reasoning_prompt: Optional[str] = None, **kwargs) -> None:
        """
        Queries DB to generate gold label-induced reasoning. Refer to https://arxiv.org/pdf/2305.02105 for more details. 
        
        Args:
            reasoning_prompt: A custom prompt containing placeholders {task_description}, {input}, and {output}. 

        Kwargs:
            split: A list of splits to generate reasoning for. Default is ["train"]. 
            overwrite: A boolean indicating whether to overwrite existing reasoning. Default is False.
        """

        # extract data
        data = self.db.full_extraction()

        # set up prompt
        if reasoning_prompt is None:
            logging.warning("Reasoning prompt not provided. Using default prompt instead.")
            with open(PathConfig.GOLD_LABEL_PROMPT, "r") as f:
                reasoning_prompt = f.read()
        else:
            try:
                reasoning_prompt.format(output = "TEST", input = "TEST", task_description = "TEST")
            except:
                raise ValueError("Invalid reasoning prompt format. Ensure it contains {task_description}, {input} and {output} placeholders.")
        

        # generate reasoning
        for entry in tqdm.tqdm(data):

            # reasoning already exisits
            if (entry.get("reasoning", None)) and (kwargs.get("overwrite", False) == False):
                logging.warning(f"Skipping reasoning for entry with ID {entry['id']} because reasoning already exists. Set overwrite=True to regenerate.")
                continue
            
            # reasoning doesn't exist
            if entry.get("split") in kwargs.get("split", ["train"]):
                entry["reasoning"] = self.config.reasoning_model.generate([{"role": "user", "content": reasoning_prompt.format(output = entry["output"], input = entry["input"], task_description=self.config.task_description)}])
                self.db.update([entry])

        self.reasoning_available = True
        logging.info("Finished generating reasoning.")
       
        
    def predict(self, input_data: Optional[Union[list, str, None]] = None, **kwargs) -> list[str]:   
        """
        Generate predictions using the provided model, with optional handling for incomplete reasoning data.

        Args:
            input: Input data for the prediction.
                - None: Uses the test split for prediction if no input is provided.
                - list: A list of multiple inputs for batch prediction.
                - str: A single input for prediction.
        
        Kwargs: 
            use_reasoning (bool): Whether to include reasoning generation. Defaults to False.
            number_demonstrations (int): The number of demonstrations to use. Defaults to 3.

        Returns:
            list[str]: A list of predicted outputs based on the provided input or validation split.
        """
        
        use_reasoning: bool = kwargs.get("use_reasoning", False)

        # handle missing reasoning data
        if use_reasoning and not self.reasoning_available:
            while True:
                do_reasoning: str = input("Reasoning data is incomplete. Would you like to generate the missing reasonings using the current model? [y/n]: ").strip().lower()
                if do_reasoning in {"y", "n"}:
                    break
                else:
                    print("Invalid input. Please enter 'y' for yes or 'n' for no.")
            if do_reasoning == "y":
                self.generate_reasoning(self.config.model)

        # determine generation logic according to input type
        if input_data is None:
            return self._predict_on_valsplit(**kwargs)
        if isinstance(input_data, list):
            return self._predict_list(input_data, **kwargs)        
        if isinstance(input_data, str):
            return self._predict_single_case(input_data, **kwargs)        
        raise TypeError("Invalid input type. Expected None, list, or str.")


    def _retrieve_k_similar(self, text: str, k: int) -> list[dict]:
        """
        Retrieves the top k most similar entries from the database.
        The returned list is ordered by similarity - beginning with the most similar

        Args:
            text: The text for which similar entries are to be retrieved.
            k: The number of similar entries to retrieve.
        """
        
        if k == 0:
           return []
        if k < 0:
            logging.warning("The value of k is negative. No demonstrations will be retrieved.")
            return []
        
        return self.db.query(text, k)
    

    def _predict_single_case(self, input_data: str, **kwargs) -> list[str]:
        """
        Predicts a single case
        
        kwargs:
            number_demonstrations: The number of similar entries to retrieve to pass to the model as synthetic conversation.
            use_reasoning (bool): Whether the model should use the generated reasonings
        """
        
        conversation: list[dict] = []
        demonstrations: list[dict] = self._retrieve_k_similar(input_data, kwargs.get("number_demonstrations", 3))

        for entry in demonstrations:
            # user
            user: str = ""
            user += self.config.task_description + "\n"
            user += entry["input"] 
            conversation.append({"role": "user", "content": user})

            # assistant
            assistant: str = ""
            if kwargs.get("use_reasoning", False):
                assistant += "Evaluation: " + entry["reasoning"] + "\n"
            assistant += entry["output"]
            conversation.append({"role": "assistant", "content": assistant})

        user_request: str = ""
        user_request += self.config.task_description + "\n"
        user_request += input_data
        conversation.append({"role": "user", "content": user_request})

        return [self.config.annotation_model.generate(conversation)]
        

    def _predict_list(self, input_data: list[str], **kwargs) -> list[str]:
        """"
        Predicts annotations for a list of input strings.
        Args:
            inputs: A list of input strings to be annotated.
            **kwargs: Additional keyword arguments to be passed to the prediction function.
        """

        annotated_cases: list[str] = []

        for entry in tqdm.tqdm(input_data):
            annotated_cases.extend(self._predict_single_case(entry, **kwargs))

        return annotated_cases
    

    def _predict_on_val_split(self, **kwargs):
        pass