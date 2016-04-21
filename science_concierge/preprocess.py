# Science Concierge: preprocess string using Porter stemmer

import re
import string
from unidecode import unidecode
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer


__all__ = ["preprocess"]

stemmer = PorterStemmer()
w_tokenizer = WhitespaceTokenizer()
punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))

def preprocess(text, stem=True):
    """
    Apply Snowball stemmer to string

    Parameters
    ----------
    text : input abstract of papers/posters string
    stem : apply stemmer if True, default True
    """
    text = unidecode(text).lower()
    text = punct_re.sub(' ', text) # remove punctuation
    if stem:
        text_new = [stemmer.stem(token) for token in w_tokenizer.tokenize(text)]
    else:
        text_new = w_tokenizer.tokenize(text)
    return ' '.join(text_new)
