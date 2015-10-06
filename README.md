# Scholarfy

a Python repository implementing Rocchio algorithm suggestion based on topic
distance space using Latent semantic analysis (LSA)


## Usage

To clone the repository,

```bash
git clone https://github.com/titipata/scholarfy
```

## Download example data

To download Pubmed Open Access example data use `download` function,

```python
import scholarfy
scholarfy.download()
```

**Note** that you might need to insert path to Python using `sys`, i.e.

```
import sys
sys.path.insert(0, '/path/to/scholarfy/')
```

## Example on how to suggest posters

Here we can preprocess list of abstracts (`abstracts`) using `preprocess` function.
Then using `tfidf_vectorizer` to transform abstract to sparse tf-idf matrix.
Afterward, we can apply Latent Sematic Analysis or truncated SVD to tf-idf matrix.
Now, each poster will be represented as vector with dimension of `n_components`.

```python
import numpy as np
import pandas as pd
import scholarfy as sf

abstracts = ['Sciene of science ...', 'is ...', 'awesome ...']
abstracts_preprocess = map(lambda abstract: sf.preprocess(abstract), abstracts) # stemming string
tfidf_matrix = sf.tfidf_vectorizer(abstracts_preprocess) # convert to tf-idf matrix
poster_vect = sf.svd_vectorizer(tfidf_matrix, n_components=200, n_iter=150)
nbrs_model = sf.build_nearest_neighbors(poster_vect)
```

Now, we can use both trained nearest neighbor model `nbrs_model` and
truncated SVD matrix `poster_vect` to suggest other posters using function
`get_schedule_rocchio` as follows:

```python
all_distances, all_posters_index = get_schedule_rocchio(nbrs_model, poster_vect, like_posters=[10], dislike_posters=[])
```



## Dependencies

- [numpy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [unidecode](https://pypi.python.org/pypi/Unidecode)
- [nltk](http://www.nltk.org/)
- [regular expression](https://docs.python.org/2/library/re.html)
- [scikit-learn](http://scikit-learn.org/)

To install all dependencies use,

```bash
pip install -r requirements.txt
```

## Members

- [Titipat Achakulvisut](http://titipata.github.io)
- [Daniel Acuna](http://www.scienceofscience.org)
- [Tulakan Ruangrong](http://github.com/bluenex)
- [Konrad Kording](http://koerding.com/)


## License

[The MIT License (MIT)](http://choosealicense.com/licenses/mit/)

Copyright (c) 2015 Titipat Achakulvisut, Daniel E. Acuna, Tulakan Ruangrong, Konrad Kording
