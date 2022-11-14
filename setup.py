import sys

from setuptools import setup, find_packages

setup(
    name="btg-seq2seq",
    version="0.1",
    author='bailin',
    author_email='bailinw@mit.edu',
    description="neural btg for seq2seq learning",
    packages=find_packages(
        exclude=["*_test.py", "test_*.py", "tests"]
    ),
    install_requires=[
        "sacrebleu>=2.0.0",
        "sacremoses>=0.0.53",
        "sentencepiece>=0.1.97",
        "torch>=1.11.0",
        "transformers>=2.11.0",
        "torchtext~=0.3.1",
        "pytest~=5.4.1",
        "wandb>=0.9.4",
    ],
    entry_points={"console_scripts": ["btg=neural_btg.commands.run:main"]},
)
