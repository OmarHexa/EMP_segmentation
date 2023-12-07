#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Binary segmentation of electron microscopy particle images",
    author="Md Omar Faruk",
    author_email="omar120cuet@gmail.com",
    url="https://github.com/OmarHexa/EMP_segmentation.git",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            # "eval_command = src.eval:main",
        ]
    },
)
