# experiment
# select posters that are from the same human curated topic
# see if which features help the search of the topic

import scholarfy as sf
import pandas as pd
import numpy as np


def compute_node_distance(node_1, node_2):
    """
    Compute distance between two string nodes in format 'F.01.r'
    """
    node_1 = node_1.split('.')
    node_2 = node_2.split('.')
    if node_1[0] != node_2[0]:
        return 3
    elif node_1[1] != node_2[1]:
        return 2
    elif node_1[2] != node_2[2]:
        return 1
    else:
        return 0


def get_poster_same_topic(poster_idx, poster_df, n_posters=1):
    """
    Give poster number and dataframe of poster_df
    return a list random poster number that has the same human curated topic
    """
    poster_same_topic_df = poster_df[poster_df.tree == poster_df.tree.iloc[poster_idx]]
    poster_same_topic = list(poster_same_topic_df.poster_number)
    poster_idx = np.random.randint(len(poster_same_topic), size=n_posters)
    poster_number_same_topic = [poster_same_topic[i] for i in poster_idx]
    return poster_number_same_topic


def get_node_dataframe(path_to_file):
    """
    Give path to poster dataframe which has columns as follows:
        - abstract: column contains abstract of all posters
        - tree: human curated topic such as 'F.01.r', sometimes call node
        - keywords: string of keywords given from the conference
    """
    poster_df = pd.read_pickle(path_to_file)
    abstracts = list(poster_df.abstract)
    abstracts_preprocess = map(lambda abstract: sf.preprocess(abstract), abstracts)

    # poster vector or abstract vector
    tfidf_matrix = sf.tfidf_vectorizer(abstracts_preprocess)
    poster_vect = sf.svd_vectorizer(tfidf_matrix, n_components=200)
    nbrs_model = sf.build_nearest_neighbors(poster_vect)

    # keywords vector
    tfidf_matrix_kw = sf.tfidf_vectorizer(poster_df.keywords)
    keywords_vect = sf.svd_vectorizer(tfidf_matrix_kw, n_components=30)
    nbrs_model_kw = sf.build_nearest_neighbors(keywords_vect)

    result = []
    N = len(poster_df) # total number of posters
    N_trials = 1000 # number of trials
    n_suggest = 10 # number of suggested posters in experiment
    n_posters = 5 # number of posters used to predict

    for n in range(N_trials):
        poster_idx = np.random.randint(N) # randomly select one poster
        poster_idx_same_topic = get_poster_same_topic(poster_idx, poster_df, n_posters=n_posters)
        poster_likes = [poster_idx] + poster_idx_same_topic # list of posters with same topic

        for j in range(1, n_posters):
            distance, poster_idx_abs = sf.get_schedule_rocchio(nbrs_model, poster_vect, like_posters=poster_likes[0:j])
            distance, poster_idx_kw = sf.get_schedule_rocchio(nbrs_model_kw, keywords_vect, like_posters=poster_likes[0:j])
            poster_idx_random = np.random.randint(N, size=n_suggest) # random pick upall posters
            poster_list = np.vstack((np.vstack((poster_idx_abs.flatten(),
                                                poster_idx_kw.flatten()))[:, 1:1+n_suggest],
                                                poster_idx_random))

            node_distances = []
            for row in poster_list:
                node_distances.append([compute_node_distance(poster_df.tree.iloc[poster_idx], poster_df.tree.iloc[idx]) for idx in row])

            result.append([poster_idx] + list(np.array(node_distances).mean(axis=1)) + [j])

    return result

if __name__ == "__main__":
    result = get_node_dataframe('/path/to/poster_df.pickle')
    result_df = pd.DataFrame(result, columns=['poster_number', 'avg_node_distance', 'avg_node_distance_kw', 'avg_random', 'number_recommend'])
