import abc
from openai import OpenAI
import pydantic
import ollama
import importlib

class Model(abc.ABC):

    @abc.abstractmethod
    def generate(conv:list[dict]):
        """
        Takes the commonly used input format ([{"role": "user", "content": "xyz"}) and returns only the generated output
        """


class OllamaModel(Model):
    def __init__(self, model, host):
        self.client = ollama.Client(host = host)
        self.model = model
        self.client.pull(model)
    
    def generate(self, conv: list[dict]) -> str:
        response: str = self.client.chat(model=self.model, messages=conv)
        return response["message"]["content"]


class OpenAIModel(Model):

    def __init__(self, model: str) -> None:
        self.client = OpenAI()
        self.model: str = model
    
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


class HuggingFaceModel(Model):

    def __init__(self, model: str):

        # import
        transformers = importlib.import_module("transformers")
        self.torch = importlib.import_module("torch")

        # run init
        self.model = transformers.AutoModelForCausalLM.from_pretrained(model, device_map="auto")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model)

    def generate(self, conv: list[dict]) -> str:

        if conv[0]["role"] == "system":
            conv = self.tokenizer.apply_chat_template(conv,  tokenize=True, return_tensors="pt", add_generation_prompt=True).to('cuda')
        else:
            conv = self.tokenizer.apply_chat_template(conv,  tokenize=True, return_tensors="pt", add_generation_prompt=True).to('cuda')

        with self.torch.no_grad():
            generated_ids = self.model.generate(conv, temperature = 1, max_new_tokens=200, do_sample=True)[0][conv.shape[-1]:]

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
