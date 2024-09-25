from setuptools import setup, find_packages

setup(
    name="ai_annotator",
    version="0.1.2",
    description="Tool for rapid testing of LLMs for annotation tasks.",
    author="nsschw",
    author_email="s2nsschw@uni-trier.com",
    package_dir={"": "src"},  # Tell setuptools where to find the package
    packages=find_packages(where="src"),  # Search for packages in the 'src' directory
    install_requires=[
        "openai",
        "transformers",
        "chromadb",
        "pandas",
        "sentence_transformers",
        "pydantic",
        "ollama",
        "tqdm",
    ],
)