import logging
import abc
import numpy
from typing import Optional
import importlib


class EmbeddingModel(abc.ABC):

    @abc.abstractmethod
    def generate(documents: list[str]):
        """
        Takes a list of documents and returns an array of embeddings
        """

class HuggingFaceEmbeddingModel(EmbeddingModel):
    def __init__(self,
                 model,
                 instruction: Optional[str] = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                ) -> None:

        # import        
        sentence_transformers = importlib.import_module("sentence_transformers")

        # run init
        self.model = sentence_transformers.SentenceTransformer(model, trust_remote_code=True)
        self.instruction = instruction

    def generate(self, documents: list[str]):
        embeddings = self.model.encode(documents, prompt=self.instruction)
        return [e.tolist() for e in embeddings]
    
    def __call__(self, input: list[str]):
        return self.generate(input)