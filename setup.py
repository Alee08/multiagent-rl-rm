#!/usr/bin/env python3

from setuptools import setup, find_packages
import multiagent_rlrm

long_description = "Multi-Agent RLRM: A library that makes it easy to formulate multi-agent problems and to resolve by reinforcement learning."

setup(
    name="multiagent_rlrm",
    version="0.1",
    # version=multiagent_rlrm.__version__,
    description="Multi-Agent RLRM Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alessandro Trapasso",
    author_email="Ale.trapasso8@gmail.com",
    url="",
    packages=find_packages(),
    include_package_data=True,  # Abilita l'inclusione dei dati specificati in package_data
    package_data={
        # Specifica di includere tutti i file nella directory img
        "multiagent_rlrm.render.img": ["*"],
    },
    python_requires=">=3.8",
    install_requires=[
        "gymnasium==0.29.1",
        "pettingzoo==1.24.3",
        "unified-planning==1.1.0",
        "numpy==1.26.4",
        "pygame==2.5.2",
        "tqdm==4.66.2",
        "matplotlib==3.8.3",
    ],
    extras_require={
        "data_analysis": [
            "pandas==2.2.1",
            # "matplotlib==3.8.3",
            "seaborn==0.13.2",
            "scipy==1.12.0",
        ],
        "image_processing": ["opencv-python==4.9.0.80", "pillow==10.2.0"],
        "metrics_monitoring": ["wandb==0.16.4"],
    },
    license="APACHE",
    keywords="learning multiagent rewardmachine reinforcementlearning",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    # entry_points={},
)
