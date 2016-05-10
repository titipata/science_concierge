import numpy as np
import scipy.sparse as sp
from numpy import bincount
from sklearn.preprocessing import normalize

class LogEntropyVectorizer:
    """
    Class for log-entropy vectorizer,
    adds on for scikit-learn CountVectorizer
    Entropy term can be calculated as follows

        p_ij = tf_ij / gf_i
        g_i = 1 + sum (p_ij * log p_ij / log n)
    that is
        e_ij = [tf] * [idf] = log(tf_ij + 1) * g_i

    where
        gf_i is total number of times term i occurs in
            the whole collection
        tf_ij is number of term i that appear in document j
        n is total number of documents
        g_i is sum of entropy across all documents j

    example:
    >> import LogEntropyVectorizer
    >> model = LogEntropyVectorizer()
    >> docs = ['this this this book',
               'this cat good',
               'cat good shit']
    >> X = model.fit_transform(docs)

    reference: https://en.wikipedia.org/wiki/Latent_semantic_indexing
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
        P.data = 1 + (p * np.log(p) / np.log(n_samples))
        g = np.ravel(P.sum(axis=0))
        X.data = np.log(1 + X.data)
        G = sp.spdiags(g, diags=0, m=n_features, n=n_features)
        E = X * G # sparse entropy matrix

        if self.norm is not None:
            E = normalize(E, norm=self.norm, copy=False)

        return E

    def _document_frequency(X):
        """
        Count the number of non-zero values for each feature in sparse X.
        ref: https://github.com/scikit-learn/scikit-learn/blob/51a765a/sklearn/feature_extraction/text.py#L496
        """
        return bincount(X.indices, minlength=X.shape[1])
