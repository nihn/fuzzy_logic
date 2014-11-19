import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="fuzzy_logic",
    version="0.0.1",
    author="Mateusz Moneta",
    author_email="mateusz.moneta@gmail.com",
    description="Python module for fuzzy operations.",
    license="Apache 2.0",
    keywords="fuzzy logic",
    packages=['fuzzy'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
    ],
    installation_requires=[
        'numpy',
        'matplotlib',
    ]
)
