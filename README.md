# Science Concierge

a Python repository for content-based recommendation
based on Latent semantic analysis (LSA) topic distance and Rocchio Algorithm.
Science Concierge is an backend algorithm for Scholarfy
[www.scholarfy.net](http://www.scholarfy.net/),
an automatic scheduler for conference.

See full article on [PLOS ONE](http://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0158423),  [Arxiv](http://arxiv.org/abs/1604.01070) or full tex manuscript and
presentation [here](https://github.com/titipata/science_concierge_manuscript). You can also see
the scale version of Scholarfy to 14.3M articles from Pubmed
at [pubmed.scholarfy.net](http://pubmed.scholarfy.net/).

## Usage

First, clone the repository.

```bash
$ git clone https://github.com/titipata/science_concierge
```

Install dependencies using `pip`,

```bash
$ pip install -r requirements.txt
```

Install the library using `setup.py`,

```bash
$ python setup.py develop install
```

## Download example data

We provide example `csv` file from Pubmed Open Acess Subset that you can download and
play with (we parsed using [pubmed_parser](https://github.com/titipata/pubmed_parser)).
Each file contains `pmc`, `pmid`, `title`, `abstract`, `publication_year` as column name.
Use `download` function to download example data,

```python
import science_concierge
science_concierge.download(['pubmed_oa_2015.csv', 'pubmed_oa_2016.csv'])
```

We provide `pubmed_oa_{year}.csv` from `{year} = 2007, ..., 2016` (**note** 2007 is
  all publications before year 2008). Alternative is to use `awscli` to download,

```bash
$ aws s3 cp s3://science-of-science-bucket/science_concierge/data/ . --recursive
```


## Example usage of Science Concierge

You can build quick recommendation by importing `ScienceConcierge` class
then use `fit` method to fit list of documents. Then use `recommend` to recommend
documents based on like or dislike documents.

```python
import pandas as pd
from science_concierge import ScienceConcierge

df = pd.read_csv('pubmed_oa_2016.csv', encoding='utf-8')
docs = list(df.abstract) # provide list of abstracts
titles = list(df.title) # titles
# select weighting from 'count', 'tfidf', or 'entropy'
recommend_model = ScienceConcierge(stemming=True, ngram_range=(1,1),
                                   weighting='entropy', norm=None,
                                   n_components=200, n_recommend=200,
                                   verbose=True)
recommend_model.fit(docs) # input list of documents or abstracts
index = recommend_model.recommend(likes=[10000], dislikes=[]) # input list of like/dislike index
docs_recommend = [titles[i] for i in index[0:10]] # recommended documents
```

## Vectorizer available

We have adds on vectorizer classes including `LogEntropyVectorizer` and
`BM25Vectorizer` for calculating documents-terms weighting from input
list of documents. Here is an example usage.

```python
from science_concierge import LogEntropyVectorizer
l_model = LogEntropyVectorizer(norm=None, ngram_range=(1,2),
                               stop_words='english', min_df=1, max_df=0.8)
X = l_model.fit_transform(docs) # where docs is list of documents
```

In this case when we have sparse matrix of documents,
we can use `fit_document_matrix` method directly.

```python
recommend_model = ScienceConcierge(n_components=200, n_recommend=200)
recommend_model.fit_document_matrix(X)
index = recommend_model.recommend(likes=[10000], dislikes=[])
```

## Dependencies

- [numpy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [unidecode](https://pypi.python.org/pypi/Unidecode)
- [nltk](http://www.nltk.org/) with white space tokenizer and Porter stemmer, <br>
  use `science_concierge.download_nltk()` to download required corpora
- [regular expression](https://docs.python.org/2/library/re.html)
- [scikit-learn](http://scikit-learn.org/)
- [cachetools](http://pythonhosted.org/cachetools/)
- [joblib](http://pythonhosted.org/joblib/)
- [pathos](https://github.com/uqfoundation/pathos)

To install all dependencies we provide `requirements.txt` where we can install using `pip`,

```bash
$ pip install -r requirements.txt
```

## Members

- [Titipat Achakulvisut](http://titipata.github.io)
- [Daniel Acuna](http://www.scienceofscience.org)
- [Tulakan Ruangrong](http://github.com/bluenex)
- [Konrad Kording](http://koerding.com/)


## License

[![License](https://licensebuttons.net/l/by-nc-sa/3.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

Copyright (c) 2015 Titipat Achakulvisut, Daniel E. Acuna, Tulakan Ruangrong, Konrad Kording
