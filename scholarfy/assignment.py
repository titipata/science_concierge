import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

__all__ = ["build_nearest_neighbors", "get_schedule_rocchio"]


def build_nearest_neighbors(poster_vect):
    """
    Create nearest neighbors model using scikit-learn
    from svd matrix (also called poster vector)
    """
    n = poster_vect.shape[0]
    nbrs_model = NearestNeighbors(n_neighbors=n).fit(poster_vect)
    return nbrs_model


def get_rocchio_topic(poster_vect, like_posters=(), dislike_posters=(),
                      w_like=1.8, w_dislike=0.2):
    """
    Give poster_vector matrix as numpy array
    and list of like and dislike posters,
    return topic preference using Rocchio algorithm
    """
    n, m = poster_vect.shape

    if len(like_posters) == 0:
        topic_pref = np.zeros(m)

    if len(like_posters) == 1:
        topic_like = np.vstack(poster_vect[like] for like in like_posters)
        topic_pref = topic_like

    if len(like_posters) > 1:
        topic_like = np.vstack(poster_vect[like] for like in like_posters)
        if len(dislike_posters) > 0:
            topic_dislike = np.vstack(poster_vect[like] for dislike in dislike_posters)
        else:
            topic_dislike = np.zeros(m)
        topic_pref = w_like*topic_like.mean(0) - w_dislike*topic_dislike.mean(0)

    return np.atleast_2d(topic_pref)


def get_schedule_rocchio(nbrs_model, poster_vect, like_posters=(), dislike_posters=()):
    """
    Give list of like and dislike posters,
    return list of suggested posters (all_posters_index)
    and nearest neighbor distance (all_distances)
    """
    topic_pref = get_rocchio_topic(poster_vect, like_posters, dislike_posters)
    all_distances, all_posters_index = nbrs_model.kneighbors(topic_pref)
    return all_distances, all_posters_index
