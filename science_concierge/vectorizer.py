# Science Concierge: tf-idf vectorizer and dimensionality reduction

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


__all__ = ["tfidf_vectorizer", "svd_vectorizer"]


def tfidf_vectorizer(abstract_list, min_df=3, max_df=0.8,
                     ngram_range=(1, 2), return_model=False):
    """
    Transform list of abstracts to tf-idf matrix

    Parameters
    ----------
    abstract_list : list of poster abstracts

    """
    tfidf_model = TfidfVectorizer(min_df=min_df, max_df=max_df, strip_accents='unicode',
                                  analyzer='word', token_pattern=r'\w{1,}', ngram_range=ngram_range,
                                  use_idf=True, smooth_idf=True, sublinear_tf=True, stop_words='english')
    tfidf_matrix = tfidf_model.fit_transform(abstract_list)
    if return_model:
        return tfidf_matrix, tfidf_model
    else:
        return tfidf_matrix


def svd_vectorizer(tfidf_matrix, n_components=400,
                   n_iter=150, return_model=False):
    """
    Apply dimensionality reduction using truncated SVD
    or Latent Semantic Analysis (LSA) to tfidf matrix

    Parameters
    ----------
    tfidf_matrix : sparse tf-idf matrix
    n_components : number of components after dimensionality reduction
    n_iter : number of iterations

    poster_vect : truncated svd matrix, called poster vector matrix
    """
    svd_model = TruncatedSVD(n_components=n_components, n_iter=n_iter, algorithm='arpack')
    poster_vect = svd_model.fit_transform(tfidf_matrix)
    if return_model:
        return poster_vect, svd_model
    else:
        return poster_vect
