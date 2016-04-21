import numpy as np
import pandas as pd
import science_concierge as scc
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import Word2Vec

stops = set(stopwords.words("english"))
w_tokenizer = WhitespaceTokenizer()


def remove_stop_words(abstract):
    """Tokenize and remove stop words from abstract"""
    words = w_tokenizer.tokenize(abstract)
    return [w for w in words if w not in stops]


def list2vec(word_list, word2vec_dict_bc):
    """Convert abstract list into word vector"""
    vec = [word2vec_dict_bc.value[a] for a in word_list if a in word2vec_dict_bc.value.keys()]
    return np.array(vec).mean(axis=0)


if __name__ == '__main__':
    # read dataframe pickle file with 'abstract' column
    poster_df = pd.read_pickle('poster_df.pickle')
    abstracts = list(poster_df.abstract)
    abstract_rdd = sc.parallelize(abstracts, numSlices=1000).\
        map(lambda a: scc.preprocess(a, stem=False)).\
        map(remove_stop_words)
    abstract_stem_rdd = sc.parallelize(abstracts, numSlices=1000).\
        map(lambda a: scc.preprocess(a, stem=False)).\
        collect()

    # average word vectors in abstract
    print('compute abstract vector using word vectors (takes around 40 mins)...')
    vectors_df = pd.read_json('wordvec_df.json')
    word2vec_dict = dict(zip(vectors_df.key, vectors_df.vector.map(np.array)))
    word2vec_dict_bc = sc.broadcast(word2vec_dict)
    abstract_vec_wv = np.vstack(abstract_rdd.map(lambda x: list2vec(x, word2vec_dict_bc)).collect())

    print('compute abstract vector using LSA...')
    tfidf_matrix = scc.tfidf_vectorizer(abstracts_preprocess) # convert to tf-idf matrix
    abstract_vec_lsa = scc.svd_vectorizer(tfidf_matrix, n_components=200, n_iter=150)

    print('save dataframe to pickle file...')
    poster_vect_multiple = pd.DataFrame(zip(range(len(poster_vect_wv)), abstract_vec_wv, abstract_vec_lsa),
                                        columns=['number', 'wordvec', 'lsa'])
    poster_vect_multiple.to_pickle('poster_vec_df.pickle')
