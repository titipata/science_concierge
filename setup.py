#! /usr/bin/env python
import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

DESCRIPTION = """
    a Python repository implementing Rocchio algorithm content-based suggestion 
    based on topic distance space using Latent semantic analysis (LSA)
    """

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if __name__ == "__main__":
    setup(
        name = "science_concierge",
        version = "0.1",
        author = "Titipat Achakulvisut, Daniel Acuna, Tulakan Ruangrong",
        author_email = "titipat.a@u.northwestern.edu",
        description = DESCRIPTION,
        license = "Creative Common 4.0",
        keywords = "recommendation system, Latent Semantic Analysis",
        url = "https://github.com/titipata/science_concierge",
        long_description=read('README.md'),
        packages=['science_concierge'],
    )
