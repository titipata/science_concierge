# Scholarfy: preprocess string using Porter stemmer

import re
import string
from nltk.stem.porter import PorterStemmer
from unidecode import unidecode
from sklearn.neighbors import NearestNeighbors

__all__ = ["preprocess"]

stemmer = PorterStemmer()


def preprocess(x):
    """
    Apply Snowball stemmer to string

    Parameters
    ----------
    x : input string

    """
    x = unidecode(x)
    punct = string.punctuation
    punct_re = re.compile('[{}]'.format(re.escape(punct)))

    x = x.lower()
    x = punct_re.sub(' ', x)
    new_x = []
    for token in x.split(' '):
        new_x.append(stemmer.stem(token))
    return ' '.join(new_x)
