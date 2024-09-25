## LLM Annotation Studio
LLM Annotation Studio is primarily designed for prototyping and testing annotation or classification tasks using Large Language Models (LLMs). While it doesn't offer extensive functionality or customizability, it provides a streamlined solution for quick experimentation. Additionally, it serves as a wrapper for a Vector Database (currently [ChromaDB](https://github.com/chroma-core/chroma)), making it adaptable for any task that leverages retrieval-augmented generation (RAG).

It supports a range of models, including both regular LLMs and embedding models. This includes API-based options like OpenAI and Mistral, as well as locally hosted solutions such as Hugging Face and Ollama, with the possibility to easily integrate custom models. Standardized task/instruction styles and automated parsing of model outputs are also included.



## How to Use

1. **Import Necessary Modules**  
   Import the relevant classes and functions from `ai_annotator` or other libraries:
   ```python
   from ai.annotator import AnnotationProject, OllamaModel, HuggingFaceEmbeddingModel, AnnotationConfig
   from ai.annotator.evaluation.parser import parse_first_int
   ```

2. **Define Your Task**  
   Create a task description to define what you're annotating or classifying. For example:
   ```python
   task = """
   You will be given an abstract of a study. Your task is to determine whether the study is valid based on the following criteria:
   1. The study must be a meta-analysis.
   2. The study must examine the association between life satisfaction, well-being, or subjective well-being and any other variable.
   
   Structure your feedback as follows:
   
   Feedback::
   Evaluation (Your reasoning whether this is a valid article or not)
   Valid: (1 if not valid, 1 if valid)
   """
   ```

3. **Configure Models**  
   Set up the LLM and embedding models. This example shows how to use both Ollama and Hugging Face models:
   ```python
   model = OllamaModel(host="http://ollama:11434", model="llama3.1:7b")
   emb_model = HuggingFaceEmbeddingModel("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
   ```

4. **Set Project Configuration**  
   Define the configuration for the annotation project using the `AnnotationConfig` class. Specify the data path, task description, and models to use:
   ```python
   project_config = AnnotationConfig(
       db_path="SecondOrderMetaStudy",
       task_description=task,
       embedding_model=emb_model,
       model=model
   )
   ```

6. **Create Annotation Project**  
   Initialize the `AnnotationProject` with your configuration and add data from a CSV file:
   ```python
   ap = AnnotationProject(config=project_config)
   ap.add_data_from_csv("abtracts.csv", column_mapping={"input": "notes_abstract", "output": "valid_abstract"})
   ```

7. **Generate Reasoning**  
   Use a reasoning prompt to generate reasoning for each data point:
   ```python
   ap.generate_reasoning(reasoning_prompt="What are the clues that lead to: [{output}] being correct in the document: [{input}] with the task being: [{task_description}].")
   ```

8. **Run Predictions**  
   Finally, run predictions on the test dataset:
   ```python
   test_cases = ap.predict(df_test["Test_Case_1"], number_demonstrations=3, use_reasoning=True)
   ```