# setup.py

from setuptools import setup, find_packages

setup(
    name="edakit",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "umap-learn",
    ],
    description="A library for advanced exploratory data analysis.",
    author="Venkat Lata",
    author_email="svenkatlata@gmail.com",
    url="https://github.com/svenkatlata/edakit",  # Replace with your actual URL
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires='>=3.7',
)
