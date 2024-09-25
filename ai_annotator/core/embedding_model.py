import logging
import abc
import numpy
from typing import Optional
from sentence_transformers import SentenceTransformer



class EmbeddingModel(abc.ABC):

    @abc.abstractmethod
    def generate(documents: list[str]):
        """
        Takes a list of documents and returns an array of embeddings
        """

class HuggingFaceEmbeddingModel(EmbeddingModel):
    def __init__(self,
                 model_name,
                 instruction: Optional[str] = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
                ) -> None:
        
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.instruction = instruction

    def generate(self, documents: list[str]):
        embeddings = self.model.encode(documents, prompt=self.instruction)
        return [e.tolist() for e in embeddings]
    
    def __call__(self, input: list[str]):
        return self.generate(input)
