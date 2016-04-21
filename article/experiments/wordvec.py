import numpy as np
import pandas as pd
import science_concierge as scc
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import Word2Vec
conf = SparkConf().setAppName('feature_comparison').setMaster('local[8]')
sc = SparkContext(conf=conf) # spark context

stops = set(stopwords.words("english"))
w_tokenizer = WhitespaceTokenizer()


def remove_stop_words(abstract):
    """Tokenize and remove stop words from abstract"""
    words = w_tokenizer.tokenize(abstract)
    return [w for w in words if w not in stops]


if __name__ == '__main__':
    # read dataframe pickle file with 'abstract' column
    poster_df = pd.read_pickle('poster_df.pickle')
    abstracts = list(poster_df.abstract)
    abstract_rdd = sc.parallelize(abstracts, numSlices=1000).\
        map(lambda a: scc.preprocess(a, stem=False)).\
        map(remove_stop_words)

    # fit word2vec model
    word2vec = Word2Vec().setVectorSize(150)
    word2vec_model = word2vec.fit(abstract_rdd)
    vectors = word2vec_model.getVectors()
    vectors_df = pd.DataFrame([(k, vectors[k]) for k in vectors.keySet()],
                              columns=['key', 'vector'])
    print('transform vector from Spark to list...')
    vectors_df.vector = vectors_df.vector.map(list) # this takes a lot of time
    print('save dataframe of key and vector to pickle file...')
    vectors_df.to_json('wordvec.json', orient='records') # save word vector to json
