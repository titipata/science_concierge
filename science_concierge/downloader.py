# Science Concierge: Data and model downloader

import os
import urllib

__all__ = ["download"]

BUCKET_PATH = "https://s3-us-west-2.amazonaws.com/science-of-science-bucket/science_concierge/data/"

def download_files(bucket_path, file_list, download_path):
    """
    Provide path to s3 bucket, download a file list to download path
    """
    if isinstance(file_list, basestring):
        file_list = [file_list] # change to list if input is string
    if not os.path.isdir(download_path):
        os.makedirs(download_path)
    for f in file_list:
        # check if file already exists
        file_path = os.path.join(download_path, f)
        if os.path.isfile(file_path):
            print('File "%s" already exists' % f)
        else:
            print('Downloading "%s" ...' % f)
            urllib.urlretrieve(bucket_path + f, file_path)
            print('Done')


def download(file_list=["pubmed_oa_2013.pickle"]):
    """
    Downloads example data from Science Concierge S3 folder
    """
    bucket_path = BUCKET_PATH
    current_path = os.path.dirname(os.path.abspath(__file__))
    download_path = os.path.join(current_path, '..','data')
    download_files(bucket_path, file_list, download_path)


def download_nltk(corpora=['punkt']):
    """
    Download necessary NLTK copora in this case is 'punkt'
    """
    import nltk
    nltk.download(corpora)
