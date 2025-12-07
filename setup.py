#!/usr/bin/env python3

import shutil
from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).parent
README = BASE_DIR / "README.md"
VERSION_FILE = BASE_DIR / "multiagent_rlrm" / "__init__.py"
EGG_INFO_DIR = BASE_DIR / "multiagent_rl_rm.egg-info"

# Ensure we always rebuild metadata so the installed version matches __version__
if EGG_INFO_DIR.exists():
    shutil.rmtree(EGG_INFO_DIR)

version_namespace = {}
if VERSION_FILE.exists():
    exec(VERSION_FILE.read_text(encoding="utf-8"), version_namespace)
version = version_namespace.get("__version__", "0.0.0")

long_description = (
    README.read_text(encoding="utf-8")
    if README.exists()
    else "Multi-Agent RLRM: A library that makes it easy to formulate multi-agent problems and solve them with reinforcement learning."
)

setup(
    name="multiagent-rl-rm",
    version=version,
    description="Multi-Agent RLRM Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alessandro Trapasso",
    author_email="Ale.trapasso8@gmail.com",
    url="https://github.com/Alee08/multi-agent-rl-rm",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "multiagent_rlrm.render.img": ["*"],
    },
    python_requires=">=3.8",
    install_requires=[
        "gymnasium==0.29.1",
        "imageio==2.34.0",
        "matplotlib==3.8.3",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
        "pandas==2.2.1",
        "pettingzoo==1.24.3",
        "pillow==10.2.0",
        "pygame==2.5.2",
        "scipy==1.12.0",
        "seaborn==0.13.2",
        "tqdm==4.66.2",
        "unified-planning==1.1.0",
        "wandb==0.18.7",
    ],
    extras_require={
        "data_analysis": [
            "pandas==2.2.1",
            "seaborn==0.13.2",
            "scipy==1.12.0",
        ],
        "image_processing": ["opencv-python==4.9.0.80", "pillow==10.2.0"],
        "metrics_monitoring": ["wandb==0.18.7"],
    },
    license="Apache-2.0",
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
)
