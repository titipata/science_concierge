import logging
import re
import numpy as np
import string
from six import string_types
from unidecode import unidecode
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from .vectorizer import LogEntropyVectorizer
from .recommend import build_nearest_neighbors, get_rocchio_topic

logger = logging.getLogger('scienceconcierge')
logger.addHandler(logging.StreamHandler())

stemmer = PorterStemmer()
w_tokenizer = WhitespaceTokenizer()
punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))


def set_log_level(verbose):
    """Convenience function for setting the log level.
    Parameters
    ----------
    verbose : bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
    """
    if isinstance(verbose, bool):
        if verbose is True:
            verbose = 'INFO'
        else:
            verbose = 'WARNING'
    if isinstance(verbose, str):
        verbose = verbose.upper()
        logging_types = dict(DEBUG=logging.DEBUG, INFO=logging.INFO,
                             WARNING=logging.WARNING, ERROR=logging.ERROR,
                             CRITICAL=logging.CRITICAL)
        if verbose not in logging_types:
            raise ValueError('verbose must be of a valid type')
        verbose = logging_types[verbose]
    logger.setLevel(verbose)


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

    * parameters for preprocessing
    stemming: boolean, if True it will apply Porter stemmer as a preprocessor
        to , default: True
    parallel: boolean, if True multipleprocessing will used to apply preprocessing
        to abstract text, default: True

    * parameters for term frequenct weighting scheme
    weighting: str, options from ['count', 'tfidf', 'entropy']
    min_df: int or float [0.0, 1.0] ignore term that appear less than min_df or has
        weight less than min_df, default: 3
    max_df: int or float [0.0, 1.0] ignore term that appear more than max_df or has
        weight greater than max_df, default: 0.8
    ngram_range: tuple, parameter for tfidf transformation
        (1, 1) for unigram, (1, 2) for bigram, default (1, 2) i.e. bigram
    norm: 'l2', 'l1' or None, default: 'l2'

    * parameters for dimensionality reduction
    algorithm: str, 'arpack' or 'randomized', default 'arpack'
    n_components: int, number of components of reduced dimension vector in LSA,
        default=200
    n_iter: int, iteration for LSA

    * For recommendation
    w_like: weight term for liked documents (called alpha in literature)
    w_dislike: wieght term for disliked documents
    n_recommend: number of total documents that want to be recommended, if None it will be
        set to total number of documents

    TO DO
    -----
    - update nearest neighbor model so that it allows larger scale of documents
    - print logging output for preprocessing step

    """
    def __init__(self, stemming=True, parallel=True,
                 weighting='tfidf', strip_accents='unicode',
                 norm='l2', lowercase=True,
                 min_df=3, max_df=0.8, ngram_range=(1,2),
                 algorithm='arpack',
                 n_components=200, n_iter=150,
                 n_recommend=None, save=False,
                 verbose=False):

        self.docs = None
        self.docs_preprocess = None
        self.stemming = stemming
        self.parallel = parallel
        self.weighting = weighting
        self.strip_accents = strip_accents
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.analyzer = 'word'
        self.token_pattern = r'\w{1,}'
        self.stop_words = 'english'
        self.lowercase = lowercase
        self.norm = norm
        self.n_components = int(n_components)
        self.n_iter = int(n_iter)
        self.algorithm = algorithm
        self.vectors = None
        self.nbrs_model = None # holder for nearest neighbor model
        self.n_recommend = n_recommend
        self.save = False
        set_log_level(verbose)

    def preprocess(self, text):
        """
        Apply Porter stemmer to input string

        Parameters
        ----------
        text: str, input string

        Returns
        -------
        text_preprocess: str, output stemming string
        """
        if isinstance(text, (type(None), float)):
            text_preprocess = ''
        else:
            text = unidecode(text).lower()
            text = punct_re.sub(' ', text) # remove punctuation
            if self.stemming:
                text_preprocess = [stemmer.stem(token) for token in w_tokenizer.tokenize(text)]
            else:
                text_preprocess = w_tokenizer.tokenize(text)
            text_preprocess = ' '.join(text_preprocess)
        return text_preprocess

    def preprocess_docs(self, docs):
        """
        Preprocess string or list of strings
        """
        if isinstance(docs, string_types):
            docs = [docs]

        if self.stemming is True:
            if not self.parallel:
                docs_preprocess = list(map(self.preprocess, docs))
            else:
                from multiprocessing import Pool
                pool = Pool()
                docs_preprocess = pool.map(self.preprocess, docs)
        else:
            docs_preprocess = docs
        return docs_preprocess

    def fit_document_matrix(self, X):
        """
        Reduce dimension of sparse matrix X
        using Latent Semantic Analysis and
        build nearst neighbor model
        """
        n_components = self.n_components
        n_iter = self.n_iter
        algorithm = self.algorithm
        lsa_model = TruncatedSVD(n_components=n_components,
                                 n_iter=n_iter,
                                 algorithm=algorithm)
        # reduce dimension using Latent Semantic Analysis
        vectors = lsa_model.fit_transform(X)
        self.vectors = vectors

        # build nearest neighbor model
        nbrs_model = build_nearest_neighbors(vectors, n_recommend=self.n_recommend)
        self.nbrs_model = nbrs_model

        return self

    def fit(self, docs):
        """
        Create recommendation vectors and nearest neighbor model
        from list of documents

        Parameters
        ----------
        docs: list of string, list of documents' text or abstracts from papers or
            publications or posters
        """

        # parameters from class
        weighting = self.weighting
        strip_accents = self.strip_accents
        token_pattern = self.token_pattern
        lowercase = self.lowercase
        min_df = self.min_df
        max_df = self.max_df
        norm = self.norm
        ngram_range = self.ngram_range
        analyzer = self.analyzer
        stop_words = self.stop_words

        # preprocess text
        logger.info('preprocess documents...')
        docs_preprocess = self.preprocess_docs(docs)
        self.docs = docs
        if self.save:
            self.docs_preprocess = docs_preprocess

        # weighting documents
        if self.weighting == 'count':
            model = CountVectorizer(min_df=min_df, max_df=max_df,
                                    lowercase=lowercase,
                                    strip_accents=strip_accents, analyzer=analyzer,
                                    token_pattern=token_pattern, ngram_range=ngram_range,
                                    stop_words=stop_words)
        elif self.weighting == 'tfidf':
            model = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                    lowercase=lowercase, norm=norm,
                                    strip_accents=strip_accents, analyzer=analyzer,
                                    token_pattern=token_pattern, ngram_range=ngram_range,
                                    use_idf=True, smooth_idf=True, sublinear_tf=True,
                                    stop_words=stop_words)
        elif self.weighting == 'entropy':
            model = LogEntropyVectorizer(min_df=min_df, max_df=max_df,
                                         lowercase=lowercase, norm=norm,
                                         token_pattern=token_pattern,
                                         ngram_range=ngram_range, analyzer=analyzer,
                                         smooth_idf=False,
                                         stop_words=stop_words)
        else:
            logger.error('Choose one weighting scheme from count, tfidf or entropy')

        # text transformation and latent-semantic-analysis
        logger.info('apply %s weighting to documents...' % self.weighting)
        X = model.fit_transform(docs_preprocess)

        # fit documents matrix from sparse matrix
        logger.info('perform Latent Semantic Analysis with %d components...' % self.n_components)
        self.fit_document_matrix(X)

        return self

    def recommend(self, likes=list(), dislikes=list(), w_like=1.8, w_dislike=0.2):
        """
        Apply Rocchio algorithm and nearest neighbor to
        recommend related documents:

            x_pref = w_like * mean(x_likes) - w_dislike * mean(x_dislikes)

        see article on how to cross-validate parameters. Use recommend
        after fit method

        Parameters
        ----------
        likes: list, list of index of liked documents
        dislikes: list, list of index of disliked documents
        w_like: float, weight for liked documents, default 1.8 (from cross-validation)
        w_dislike: float, weight for disliked documents, default 0.2
            (we got 0.0 from cross-validation)
        """
        self.w_like = w_like
        self.w_dislike = w_dislike

        # compute preference vector
        topic_pref = get_rocchio_topic(self.vectors, likes, dislikes, w_like, w_dislike)

        # nearest neighbor to suggest related abstract with close topic
        _, recommend_index = self.nbrs_model.kneighbors(topic_pref)

        return recommend_index.flatten()
