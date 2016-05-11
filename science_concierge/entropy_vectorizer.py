import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer


class LogEntropyVectorizer:
    """Log-entropy vectorizer
    adds on for scikit-learn CountVectorizer to
    calculate log-entropy term matrix

    Assume we have term i in document j can be calculated as follows
    Global entropy
        p_ij = f_ij / sum_j(f_ij)
        g_i = 1 + sum_j (p_ij * log p_ij / log n)
    log-entropy of term i in document j is
        l_ij = log(1 + f_ij) * g_i

    where
        f_ij is number of term i that appears in document j
        sum_j(f_ij) is total number of times term i occurs in
            the whole documents
        n is total number of documents
        g_i is sum of entropy across all documents j

    Parameters
    ----------
    See CountVectorizer for details explanations
    lowercase: boolean, if True, text will be considered in lowercase
    preprocessor: default None
    stop_words: None or 'english', if 'english', it will remove all
        English stop words
    ngram_range: tuple, (1,1) for unigram and (1,2) for bigram
    max_df: float range [0, 1]
    min_df: float range [0, 1]
    norm: 'l2' or None, if 'l2', it will normalize matrix using l2-norm
    smooth_idf: None for now

    example:
    >> import LogEntropyVectorizer
    >> model = LogEntropyVectorizer(norm=None, ngram_range=(1,1))
    >> docs = ['this this this book',
               'this cat good',
               'cat good shit']
    >> X = model.fit_transform(docs)

    reference:
        - https://en.wikipedia.org/wiki/Latent_semantic_indexing
        - http://webpages.ursinus.edu/akontostathis/KontostathisHICSSFinal.pdf
    """
    def __init__(self, lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern='(?u)\\b\\w\\w+\\b',
                 ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1,
                 norm='l2', smooth_idf=False):
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.max_df = max_df
        self.min_df = min_df
        self.norm = norm
        self.smooth_idf = smooth_idf

    def fit_transform(self, raw_documents):
        X = CountVectorizer(lowercase=self.lowercase, preprocessor=self.preprocessor,
                            tokenizer=self.tokenizer, stop_words=self.stop_words,
                            token_pattern=self.token_pattern, ngram_range=self.ngram_range,
                            analyzer=self.analyzer, max_df=self.max_df, min_df=self.min_df
                           ).fit_transform(raw_documents)

        n_samples, n_features = X.shape
        gf = np.ravel(X.sum(axis=0)) # count total number of each words

        if self.smooth_idf:
            n_samples += int(self.smooth_idf)
            gf += int(self.smooth_idf)

        P = (X * sp.spdiags(1./gf, diags=0, m=n_features, n=n_features)) # probability of word occurence
        p = P.data
        P.data = (p * np.log2(p) / np.log2(n_samples))
        g = 1 + np.ravel(P.sum(axis=0))
        f = np.log2(1 + X.data)
        X.data = f
        G = sp.spdiags(g, diags=0, m=n_features, n=n_features)
        L = X * G # sparse entropy matrix

        if self.norm is not None:
            L = normalize(L, norm=self.norm, copy=False)

        return L
