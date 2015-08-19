# Scholarfy: Data and model downloader

import os
import urllib

__all__ = ["download"]

def download_files(bucket_path, file_list, download_path):
    """
    Provide path to s3 bucket, download a file list to download path
    """
    if not os.path.isdir(download_path):
        os.makedirs(download_path)
    for f in file_list:
        # check if file already exists
        file_path = os.path.join(download_path, f)
        if os.path.isfile(file_path):
            print 'File "%s" already exists' % f
        else:
            print 'Downloading "%s" ...' % f
            urllib.urlretrieve(bucket_path + f, file_path)
            print 'Done'


def download():
    """This function downloads data from S3"""
    file_list = ["pubmed_data.pickle"]
    bucket_path = "https://s3-us-west-2.amazonaws.com/science-of-science-bucket/scholarfy/"
    current_path = os.path.dirname(os.path.abspath(__file__))
    download_path = os.path.join(current_path, 'pubmed_data')
    download_files(bucket_path, file_list, download_path)
