#!/usr/bin/env python

import os
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import find_packages, setup


_PATH_ROOT = os.path.dirname(__file__)

def _load_py_module(fname, pkg="pai_lightning"):
    spec = spec_from_file_location(
        os.path.join(pkg, fname), os.path.join(_PATH_ROOT, pkg, fname)
    )
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py

setup_tools = _load_py_module("setup_tools.py")

setup(
    name="PrivateAI w/ Lightning AI: Safely train your model on a private dataset",
    version="0.0.1",
    description="Train your model by keeping data safe using PyTorch Lightning and Private AI",
    author="anandh_perumal",
    author_email="anandh@private-ai.com",
    url="https://github.com/anandhperumal3/PAILightning",
    install_requires=setup_tools._load_requirements(_PATH_ROOT),
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
)
