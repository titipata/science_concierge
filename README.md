# Science Concierge

a Python repository implementing Rocchio algorithm content-based suggestion
based on topic distance space using Latent semantic analysis (LSA).
Science Concierge is an algorithm backend for Scholarfy
[www.scholarfy.net](http://www.scholarfy.net/),
an automatic itinerary maker for conference goers.

See full article on [Arxiv](http://arxiv.org/abs/1604.01070) or full tex manuscript
[here](https://github.com/titipata/science_concierge_manuscript). You can also see
the scale version of Scholarfy to 14.3M articles from Pubmed
at [pubmed.scholarfy.net](http://pubmed.scholarfy.net/).


## Usage

First, clone the repository.

```bash
$ git clone https://github.com/titipata/science_concierge
```

Install repository using `setup.py`

```bash
$ python setup.py develop install
```

or you can need to insert path to Python environment using `sys` i.e.

```python
import sys; sys.path.insert(0, '/path/to/science_concierge/')
```

or directly add path to `.bash_profile`.

```bash
export PYTHONPATH='/PATH/TO/science_concierge:$PYTHONPATH'
export PYTHONPATH
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


## Example usage of Science Concierge

You can build quick recommendation by importing `ScienceConcierge` class
then use `fit` method to fit list of documents. Then use `recommend` to recommend
documents based on like or dislike documents.

```python
from science_concierge import ScienceConcierge
recommend_model = ScienceConcierge(stemming=True, ngram_range=(1,2),
                                   n_components=200, n_recommend=200)
recommend_model.fit(docs) # input list of documents or abstracts
recommend_model.recommend(like=[100, 8450], dislike=[]) # index of like/dislike docs
docs_recommend = [model.docs[i] for i in index] # recommended documents
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
