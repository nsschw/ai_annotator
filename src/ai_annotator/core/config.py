import logging
from typing import Optional
from .model import Model, OpenAIModel
from .embedding_model import EmbeddingModel
import os
import importlib.resources

class AnnotationConfig():
    """
    AnnotationConfig keeps variables that are set during the project initialization and should remain constant throughout the project.
    """
    
    def __init__(
            self,
            db_path: str,
            task_description: str,
            model: Optional[Model] = None,
            reasoning_model: Optional[Model] = None,
            annotation_model: Optional[Model] = None,
            embedding_model: Optional[EmbeddingModel] = None,
            **kwargs
            ) -> None:
        """
        Kwargs:
            collection_name (str): Name of the collection to use if multiple collections are stored in the database.
        """

        self.task_description = task_description

        # for db
        self.embedding_model = embedding_model
        self.db_path = db_path
        self.collection_name = kwargs.get("collection_name", "Demo")

        if model:
            # If a general model is provided, use it for both reasoning and annotation
            self.reasoning_model = model
            self.annotation_model = model
        
        elif reasoning_model and annotation_model:
            # If both reasoning and annotation models are provided, use them
            self.reasoning_model = reasoning_model
            self.annotation_model = annotation_model
            
        else:
            logging.warning("Neither a general model nor both specific reasoning and annotation models were provided. Defaulting to OpenAIModel 'gpt-4o-mini' for the missing ones. Make sure to set your OPENAI_API_KEY")
            self.reasoning_model = reasoning_model or OpenAIModel("gpt-4o-mini")
            self.annotation_model = annotation_model or OpenAIModel("gpt-4o-mini")


class PathConfig:
    # Get the path to the installed package
    BASE_DIR = importlib.resources.files('ai_annotator')

    # Define paths relative to the package directory
    PROMPTS_DIR = BASE_DIR / 'prompts'
    GOLD_LABEL_PROMPT = PROMPTS_DIR / 'gold_label-induced_reasoning.txt'