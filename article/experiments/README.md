# Experiment

this folder includes code snippets that use to run analysis and create plots in the
publication.

- `poster_node_distance.py` is for most of the plot in the paper. The data for
each plots is in `data` folder
- `wordvec.py` and `wordvec_abstract.py` is for
You can download trained word vectors from SfN 2015 using
`science_concierge.download("wordvec.json")` which has dictionary of words and
their corresponding vectors. Or `science_concierge.download("wordvec_abstract.json")`
in order to download each abstract vectors.
- `topic_comparison.py` is for feature comparison using count vectorize,
tf-idf matrix, LSA and word vectors.
