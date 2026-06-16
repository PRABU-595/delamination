from setuptools import setup, find_packages

setup(
    name="delam-ml",
    version="1.0.0",
    description="Multiscale ML Framework for Delamination Modeling in Composites",
    author="Collaborative Research Group",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "torch-geometric",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "gradio",
        "networkx",
        "gpytorch"
    ],
    entry_points={
        "console_scripts": [
            "delam-predict=predict:predict_interactive",
            "delam-train=src.training.train_mega:main"
        ]
    },
    python_requires=">=3.8",
)
