import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "scholarfy",
    version = "0.0.0",
    author = "Titipat Achakulvisut, Daniel Acuna",
    author_email = "my.titipat@gmail.com",
    description = ("An demonstration of how to create, document, and publish "
                                   "to the cheese shop a5 pypi.org."),
    license = "MIT",
    keywords = "Python library that implements a fast and accurate recommendation system for literature search",
    url = "http://packages.python.org/an_example_pypi_project",
    long_description=read('README.md'),
)
