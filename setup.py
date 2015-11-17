import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "science_concierge",
    version = "0.1",
    author = "Titipat Achakulvisut, Daniel Acuna, Tulakan Ruangrong",
    author_email = "titipat.a@u.northwestern.edu",
    description = ("a Python repository implementing Rocchio algorithm content-based suggestion based on topic distance space using Latent semantic analysis (LSA)"),
    license = "MIT",
    keywords = "Python library that implements a fast and accurate recommendation system for literature search",
    url = "http://packages.python.org/an_example_pypi_project",
    long_description=read('README.md'),
)
