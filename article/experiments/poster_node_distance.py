# experiment skeleton code for output figure in publication
# note that we use data provide by SfN, which you can request through the society
# from http://www.sfn.org/

import science_concierge as scc
import pandas as pd
import numpy as np

path_to_file = '' # add path to poster pickle file
poster_df = pd.read_pickle(path_to_file)
abstracts = list(poster_df.abstract)
abstracts_preprocess = map(lambda abstract: scc.preprocess(abstract), abstracts)

# poster vector or abstract vector
tfidf_matrix = scc.tfidf_vectorizer(abstracts_preprocess)
poster_vect = scc.svd_vectorizer(tfidf_matrix, n_components=200)
nbrs_model = scc.build_nearest_neighbors(poster_vect)

# keywords vector
tfidf_matrix_kw = scc.tfidf_vectorizer(poster_df.keywords)
keywords_vect = scc.svd_vectorizer(tfidf_matrix_kw, n_components=30)
nbrs_model_kw = scc.build_nearest_neighbors(keywords_vect)


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


def get_poster_different_topic(poster_idx, poster_df, n_posters=1):
    """
    Give poster number and dataframe of poster_df
    return a list of random poster number that has
    topic with node distance 1, 2, 3 respectively
    """
    node_distance_total = np.array(map(lambda x: compute_node_distance(x, poster_df.tree[poster_idx]), list(poster_df.tree)))
    idx_1 = np.where(node_distance_total == 1)[0]
    idx_2 = np.where(node_distance_total == 2)[0]
    idx_3 = np.where(node_distance_total == 3)[0]
    try:
        poster_distance_1 = [idx_1[i] for i in np.random.randint(len(idx_1), size=n_posters)]
    except:
        poster_distance_1 = []
    poster_distance_2 = [idx_2[i] for i in np.random.randint(len(idx_2), size=n_posters)]
    poster_distance_3 = [idx_3[i] for i in np.random.randint(len(idx_3), size=n_posters)]
    return poster_distance_1, poster_distance_2, poster_distance_3


def compare_node_distance():
    """
    Give path to poster dataframe which has columns as follows:
        - abstract: column contains abstract of all posters
        - tree: human curated topic such as 'F.01.r', sometimes call node
        - keywords: string of keywords given from the conference

    Compare average node distance between random selected poster,
    keywords and abstract
    """

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
            distance, poster_idx_abs = scc.get_schedule_rocchio(nbrs_model, poster_vect, like_posters=poster_likes[0:j])
            distance, poster_idx_kw = scc.get_schedule_rocchio(nbrs_model_kw, keywords_vect, like_posters=poster_likes[0:j])
            poster_idx_random = np.random.randint(N, size=n_suggest) # random pick upall posters
            poster_list = np.vstack((np.vstack((poster_idx_abs.flatten(),
                                                poster_idx_kw.flatten()))[:, 1:1+n_suggest],
                                                poster_idx_random))

            node_distances = []
            for row in poster_list:
                node_distances.append([compute_node_distance(poster_df.tree.iloc[poster_idx], poster_df.tree.iloc[idx]) for idx in row])

            result.append([poster_idx] + list(np.array(node_distances).mean(axis=1)) + [j])

    return result


def compare_dislike_poster():
    """
    Here we see the effect of beta with distance/close posters in topic space
    basically, the closer poster in human topic space and higer beta, we see that
    effect of overall suggestion getting worse
    """
    result = []
    N = len(poster_df) # total number of posters
    N_trials = 1000
    n_suggest = 10
    w_like_list = np.linspace(1.0, 2.0, num=11)

    for n in range(N_trials):
        poster_idx = np.random.randint(N) # randomly select one poster
        poster_likes = [poster_idx]
        poster_dist_1, poster_dist_2, poster_dist_3 = get_poster_different_topic(poster_idx, poster_df, n_posters=1)
        for (p, t_d) in zip([poster_dist_1, poster_dist_2, poster_dist_3], [1,2,3]):
            for w_dislike in w_dislike_list:
                distance, poster_idx_abs = get_schedule_rocchio(nbrs_model, poster_vect,
                                                                like_posters=poster_likes, dislike_posters=p,
                                                                w_like_input=1.8, w_dislike_input=w_dislike)
                poster_list = poster_idx_abs.flatten()[1:1+n_suggest]
                avg_distance = np.array([compute_node_distance(poster_df.tree.iloc[poster_idx], poster_df.tree.iloc[idx]) for idx in poster_list]).mean()
                result.append([poster_idx] + [avg_distance] + [w_dislike] + [t_d])
    result_beta_df = pd.DataFrame(result, columns=['poster_number', 'avg_distance', 'beta', 'distance'])

    # here is how to summarize the result
    # result_beta_df = pd.DataFrame(result, columns=['poster_number', 'avg_distance', 'beta', 'distance'])
    # group_beta = result_beta_df[result_beta_df.distance == 1][['avg_distance', 'beta']].groupby('beta')
    # group_beta.agg(lambda x: x.mean())
    return result_beta_df


def compare_like_dislike_relation():
    """
    Perform one like and one dislike poster experiment
    where we vary alpha (w_like) from 1 to 2 and beta (w_dislike) from 0 to 1
    return average topic distance with varying alpha and beta
    """
    result = []
    N = len(poster_df) # total number of posters
    N_trials = 1000
    n_suggest = 10
    w_dislike_list = np.linspace(0.0, 1.0, num=11)
    w_like_list = np.linspace(1.0, 2.0, num=11)

    for n in range(N_trials):
        poster_idx = np.random.randint(N) # randomly select one poster
        poster_likes = [poster_idx]
        # get random poster with distance 1, 2, 3 respectively
        poster_dist_1, poster_dist_2, poster_dist_3 = get_poster_different_topic(poster_idx, poster_df, n_posters=1)
        for w_like in w_like_list:
            for w_dislike in w_dislike_list:
                distance, poster_idx_abs = get_schedule_rocchio(nbrs_model, poster_vect,
                                                                like_posters=poster_likes, dislike_posters=poster_dist_1,
                                                                w_like_input=w_like, w_dislike_input=w_dislike)
                poster_list = poster_idx_abs.flatten()[1:1+n_suggest]
                avg_distance = np.array([compute_node_distance(poster_df.tree.iloc[poster_idx], poster_df.tree.iloc[idx]) for idx in poster_list]).mean()
                result.append([poster_idx] + [avg_distance] + [w_like] + [w_dislike])


    result_df = pd.DataFrame(result, columns=['poster_number', 'avg_distance', 'alpha', 'beta'])
    group_alpha_beta = result_df[['alpha', 'beta', 'avg_distance']].groupby(['alpha', 'beta'])
    distance_alpha_beta = group_alpha_beta.agg(lambda x: x.mean()).reset_index()

    return np.reshape(distance_alpha_beta.avg_distance, (len(w_like_list), len(w_dislike_list)))


def compare_components_vs_topic_distance():
    """
    See the relationship between number of SVD components and
    average distance of topic distance of suggested posters
    """
    # training to get poster vectors
    result = []
    poster_vect_comp = []
    N = len(poster_vect) # total number of posters
    N_trials = 1000
    n_suggest = 10
    n_posters = np.random.randint(N, size=N_trials)
    n_components_list = [50, 75, 100, 150, 200, 300, 400, 500]
    for n_c in n_components_list:
        poster_vect = scc.svd_vectorizer(tfidf_matrix, n_components=n_c)
        poster_vect_comp.append(poster_vect)

    # loop through the model
    for n_model in range(len(n_components_list)):
        nbrs_model = scc.build_nearest_neighbors(poster_vect_comp[n_model])
        for n in n_posters:
            poster_idx = n # randomly select one poster (pre-random)
            poster_idx_same_topic = get_poster_same_topic(poster_idx, poster_df, n_posters=5)
            poster_likes = [poster_idx] + poster_idx_same_topic # list of posters with same topic
            distance, poster_idx_abs = scc.get_schedule_rocchio(nbrs_model, poster_vect_comp[n_model], like_posters=poster_likes[0:1])
            poster_list = poster_idx_abs.flatten()[1:1+n_suggest]
            avg_distance = np.array([compute_node_distance(poster_df.tree.iloc[poster_idx], poster_df.tree.iloc[idx]) for idx in poster_list]).mean()
            result.append([poster_idx] + [avg_distance] + [n_components_list[n_model]])

    result_df = pd.DataFrame(result, columns=['poster_number', 'distance', 'n_components'])

    return result_df


def compare_human_topic_distance():
    """
    Perform experiment by randomly select two poster with different human curated
    distance. Then see the relationship between human distance versus topic distance
    in both keywords space and abstract topic space.
    """
    result = []
    N = len(poster_df) # total number of posters
    N_trials = 1000

    for n in range(N_trials):
        poster_idx = np.random.randint(N) # randomly select one poster
        poster_likes = [poster_idx]
        # get random poster with distance 1, 2, 3 respectively
        poster_dist_0 = get_poster_same_topic(poster_idx, poster_df)
        poster_dist_1, poster_dist_2, poster_dist_3 = get_poster_different_topic(poster_idx, poster_df, n_posters=1)

        poster_vect_abstract = np.atleast_2d(poster_vect[poster_idx])
        poster_vect_kw = np.atleast_2d(keywords_vect[poster_idx])

        for (dist, poster_dist) in zip(range(0,4), [poster_dist_0, poster_dist_1, poster_dist_2, poster_dist_3]):
            dist_abstract = np.sum((poster_vect_abstract - poster_vect[poster_dist])**2)
            dist_keyword = np.sum((poster_vect_kw - keywords_vect[poster_dist])**2)
            result.append([poster_idx] + [dist] + [dist_abstract] + [dist_keyword])

    result_df = pd.DataFrame(result, columns=['poster_number', 'human_distance',
                                              'abstact_distance', 'keyword_distance'])

    return result_df


if __name__ == "__main__":
    result = compare_node_distance('/path/to/poster_df.pickle')
    result_df = pd.DataFrame(result, columns=['poster_number', 'avg_node_distance', 'avg_node_distance_kw', 'avg_random', 'number_recommend'])
