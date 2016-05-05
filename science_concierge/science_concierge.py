import re
import numpy as np
import string
from unidecode import unidecode
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
from .vectorizer import tfidf_vectorizer, svd_vectorizer
from .assignment import build_nearest_neighbors, get_rocchio_topic

stemmer = PorterStemmer()
w_tokenizer = WhitespaceTokenizer()
punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))


class ScienceConcierge:
    """Science Concierge

    Recommendation class using Latent Semantic Analysis on list of abstracts
    Process workflow are as follows
    - Word tokenize and stemming (optional)
    - Create tf-idf matrix, unigram or bigram recommended
    - Latent Semantic Analysis (LSA) i.e. reduce dimension of using
        truncated SVD
    - Nearest neighbor assignment for recommendation

    Parameters
    ----------

    stemming: boolean, if True it will apply Porter stemmer as a preprocessor
        to , default: True
    parallel: boolean, if True multipleprocessing will used to apply preprocessing
        to abstract text, default: True
    min_df: int or float [0.0, 1.0] ignore term that appear less than min_df or has
        weight less than min_df, default: 3
    max_df: int or float [0.0, 1.0] ignore term that appear more than max_df or has
        weight greater than max_df, default: 0.8
    ngram_range: tuple, parameter for tfidf transformation
        (1, 1) for unigram, (1, 2) for bigram, default (1, 2) i.e. bigram
    n_components: int, number of components of reduced dimension vector in LSA,
        default=
    n_iter: int, iteration for LSA
    w_like: weight term for liked documents (called alpha in literature)
    w_dislike: wieght term for disliked documents
    n_recommend: number of total documents that want to be recommended, if None it will be
        set to total number of documents

    TO DO
    -----
    - update nearest neighbor model so that it allows larger scale of documents also
    - print logging output for preprocessing step

    """
    def __init__(self, stemming=True, parallel=True,
                 min_df=3, max_df=0.8, ngram_range=(1,2),
                 n_components=200, n_iter=150,
                 w_like=1.8, w_dislike=0.2, n_recommend=None,
                 save_intermediate=False):

        self.docs = None
        self.docs_preprocess = None
        self.stemming = stemming
        self.parallel = parallel
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.n_components = n_components
        self.n_iter = n_iter
        self.vectors = None
        self.w_like = w_like
        self.w_dislike = w_dislike
        self.nbrs_model = None # nearest neighbor model for recommendation
        self.n_recommend = n_recommend
        self.save_intermediate = False

    def preprocess(self, text):
        """
        Apply Snowball stemmer to string

        Parameters
        ----------
        text: str, input string
        """
        if text is None:
            text_preprocess = ''
        else:
            text = unidecode(text).lower()
            text = punct_re.sub(' ', text) # remove punctuation
            if self.stemming:
                text_preprocess = [stemmer.stem(token) for token in w_tokenizer.tokenize(text)]
            else:
                text_preprocess = w_tokenizer.tokenize(text)
        return ' '.join(text_preprocess)

    def fit(self, docs):
        """
        Create recommendation vectors and nearest neighbor model
        from list of documents

        Parameters
        ----------
        docs: list of string, list of documents' text or abstracts from papers or
            publications or posters
        """
        if not self.parallel:
            docs_preprocess = map(self.preprocess, docs)
            docs_preprocess = list(docs)
        else:
            from multiprocessing import Pool
            pool = Pool()
            docs_preprocess = pool.map(self.preprocess, docs)

        # save documents to class
        self.docs = docs
        if self.save_intermediate:
            self.docs_preprocess = docs_preprocess

        # compute tf-idf matrix and LSA vectors
        tfidf_matrix = tfidf_vectorizer(docs_preprocess,
                                        min_df=self.min_df, max_df=self.max_df,
                                        ngram_range=self.ngram_range)
        vectors = svd_vectorizer(tfidf_matrix, n_components=self.n_components,
                                 n_iter=self.n_iter)
        self.vectors = vectors

        # compute nearest neighbor model
        nbrs_model = build_nearest_neighbors(vectors, n_recommend=self.n_recommend)
        self.nbrs_model = nbrs_model

        return self

    def recommend(self, like=list(), dislike=list(), w_like=None, w_dislike=None):
        """
        Apply Rocchio algorithm to recommend related documents
        """
        if w_like:
            self.w_like = w_like
        if w_dislike:
            self.w_dislike = w_dislike

        # compute preference vector
        topic_pref = get_rocchio_topic(self.vectors, like, dislike, self.w_like, self.w_dislike)

        # do nearest neighbor to suggest related abstract with close topic
        _, index = self.nbrs_model.kneighbors(topic_pref)

        return index.flatten()
