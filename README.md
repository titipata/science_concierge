# Science Concierge

a Python repository implementing Rocchio algorithm content-based suggestion
based on topic distance space using Latent semantic analysis (LSA).
Science Concierge is an algorithm backend for Scholarfy
[http://www.scholarfy.net/](http://www.scholarfy.net/),
an automatic itinerary maker for conference goers.

See full article on [Arxiv](http://arxiv.org/abs/1604.01070) or full tex manuscript
[here](https://github.com/titipata/science_concierge_manuscript).


## Usage

To clone the repository,

```bash
git clone https://github.com/titipata/science_concierge
```

## Download example data

We provide `.pickle` file from Pubmed Open Acess Subset from year 2013.
The pickle file contains following columns:
`pmc, full_title, abstract, journal_title, publication_year`.
To download Pubmed Open Access example data use `download` function as follows,

```python
import science_concierge
science_concierge.download()
```

**Note** that you might need to insert path to Python using `sys` as follows

```python
import sys
sys.path.insert(0, '/path/to/science_concierge/')
```

or modify `.bash_profile` by adding these lines

```bash
export PYTHONPATH='/PATH/TO/science_concierge:$PYTHONPATH' export PYTHONPATH
```


## Example usage of Science Concierge

Here we can preprocess list of abstracts (`abstracts`) using `preprocess` function.
Then using `tfidf_vectorizer` to transform abstract to sparse tf-idf matrix.
Afterward, we can apply Latent Sematic Analysis or truncated SVD to tf-idf matrix.
Now, each poster will be represented as vector with dimension of `n_components`.

```python
import numpy as np
import pandas as pd
import science_concierge as sc

pubmed_df = pd.read_pickle('data/pubmed_example.pickle') # assuming example data is downloaded
abstracts = list(pubmed_df.abstract)
abstracts_preprocess = map(lambda abstract: sc.preprocess(abstract), abstracts) # stemming string
tfidf_matrix = sc.tfidf_vectorizer(abstracts_preprocess) # convert to tf-idf matrix
poster_vect = sc.svd_vectorizer(tfidf_matrix, n_components=200, n_iter=150)
nbrs_model = sc.build_nearest_neighbors(poster_vect)
```

Now, we can use both trained nearest neighbor model `nbrs_model` and
truncated SVD matrix `poster_vect` to suggest other posters using function
`get_schedule_rocchio` as follows:

```python
all_distances, all_posters_index = sc.get_schedule_rocchio(nbrs_model, poster_vect, like_posters=[10], dislike_posters=[])
```

where `like_posters` is a list of like poster index and `dislike_posters` is for
list of dislike posters. `all_posters_index` is the rank of suggested posters.


## Dependencies

- [numpy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [unidecode](https://pypi.python.org/pypi/Unidecode)
- [nltk](http://www.nltk.org/)
- [regular expression](https://docs.python.org/2/library/re.html)
- [scikit-learn](http://scikit-learn.org/)

To install all dependencies we provide `requirements.txt` where we can install using `pip`,

```bash
pip install -r requirements.txt
```

## Members

- [Titipat Achakulvisut](http://titipata.github.io)
- [Daniel Acuna](http://www.scienceofscience.org)
- [Tulakan Ruangrong](http://github.com/bluenex)
- [Konrad Kording](http://koerding.com/)


## License

![alt text](https://licensebuttons.net/l/by-nc-sa/3.0/88x31.png)
[Creative Commons 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

Copyright (c) 2015 Titipat Achakulvisut, Daniel E. Acuna, Tulakan Ruangrong, Konrad Kording
