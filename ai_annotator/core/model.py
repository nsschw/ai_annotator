import logging
import abc
from openai import OpenAI
import pydantic


class Model(abc.ABC):

    @abc.abstractmethod
    def generate(conv:list[dict]):
        """
        Takes the commonly used input format ([{"role": "user", "content": "xyz"}) and returns only the generated output
        """


class OpenAIModel(Model):

    def __init__(self, model: str, **kwargs) -> None:
        self.client = OpenAI()
        self.model: str = model
        self.name = kwargs.get("model_name", f"OpenAI_{model}")
    
    def generate(self, conv: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conv
            )
        return response.choices[0].message.content

    def generate_structured_response(self, conv: list[dict], structure: pydantic.BaseModel):
        response = self.client.beta.chat.completions.parse(
            model = self.model,
            messages=conv,
            response_format=structure,
        )
        return response.choices[0].message.parsed