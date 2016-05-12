# Science Concierge

a Python repository implementing Rocchio algorithm content-based suggestion
based on topic distance space using Latent semantic analysis (LSA).
Science Concierge is an algorithm backend for Scholarfy
[www.scholarfy.net](http://www.scholarfy.net/),
an automatic itinerary maker for conference goers.

See full article on [Arxiv](http://arxiv.org/abs/1604.01070) or full tex manuscript and
presentation [here](https://github.com/titipata/science_concierge_manuscript). You can also see
the scale version of Scholarfy to 14.3M articles from Pubmed
at [pubmed.scholarfy.net](http://pubmed.scholarfy.net/).


## Usage

First, clone the repository.

```bash
$ git clone https://github.com/titipata/science_concierge
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

df = pd.read_csv('pubmed_oa_2016.csv')
docs = list(df.abstract) # provide list of abstracts
recommend_model = ScienceConcierge(stemming=True, ngram_range=(1,2),
                                   n_components=200, n_recommend=200)
recommend_model.fit(docs) # input list of documents or abstracts
index = recommend_model.recommend(like=[100, 8450], dislike=[]) # index of like/dislike docs
docs_recommend = [model.docs[i] for i in index] # recommended documents
```

## Log-entropy vectorizer

We also have adds on log-entropy class for calculating log-entropy
matrix from input documents. Here is an example usage.

```python
from science_concierge import LogEntropyVectorizer
l_model = LogEntropyVectorizer(norm=None, ngram_range=(1,2))
X = l_model.fit_transform(docs) # where docs is list of documents
```


## Dependencies

- [numpy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [unidecode](https://pypi.python.org/pypi/Unidecode)
- [nltk](http://www.nltk.org/) with white space tokenizer and Porter stemmer
  use `science_concierge.download_nltk()` to download required corpora
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
