from setuptools import setup, find_packages

setup(
    name="SPEED",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scanpy==1.9",
        "torch==1.13",
        "torchvision==0.14",
        "timm==0.9.12",
        "anndata",
        "numpy",
        "pandas",
        "tqdm",
        "scikit-learn",
        "scipy"
    ],
    python_requires=">=3.7",
)
