import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="tsboi",
    py_modules=["tsboi"],
    version="1.0",
    description="Tsboi stands for time-series boy - a library for time series forecasting with ML.",
    author="Mark Csizmadia",
    packages=find_packages(exclude=["test*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)