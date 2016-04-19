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

def preprocess(text):
    """
    Apply Snowball stemmer to string

    Parameters
    ----------
    text : input string

    """
    text = unidecode(text).lower()
    text = punct_re.sub(' ', text) # remove punctuation
    text_stem = []
    for token in w_tokenizer.tokenize(text):
        text_stem.append(stemmer.stem(token))
    return ' '.join(text_stem)
