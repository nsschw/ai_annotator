from setuptools import setup, find_packages

setup(
    name="ai_annotator",
    version="0.1.2",
    description="Tool for rapid testing of LLMs for annotation tasks.",
    author="nsschw",
    author_email="s2nsschw@uni-trier.com",
    packages=find_packages(),  # Automatically find all packages in your directory
    install_requires=[
        "openai",
        "transformers",
        "chromadb",
        "pandas",
        "sentence_transformers",
        "pydantic",
        "ollama",
        "tqdm"
    ],
)