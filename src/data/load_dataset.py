import re
import os
import numpy as np

from .data_download import download_data, extract_zip_data
from src.config import RAW_DATA_DIR, DATA_URL


def load_spambase():
    """
    Load the spambase dataset from the data/raw folder
    """
    spambase = os.path.join(RAW_DATA_DIR, 'spambase.data')
    if not os.path.exists(spambase):
        file_name = DATA_URL.split('/')[-1]
        data_dir = download_data(DATA_URL, file_name)
        extract_zip_data(data_dir, RAW_DATA_DIR, delete_zip=True)
        
    data = np.loadtxt(spambase, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    y = y.astype(int)
    return X, y


def extract_feature_names():
    """
    Extract the feature names from the spambase dataset
    """
    with open(os.path.join(RAW_DATA_DIR, 'spambase.names'), 'r') as file:
        # read all lines at once
        lines = file.readlines()
        
        # extract feature names 
        feature_names = [re.sub(r':.*', '', line).strip() for line in lines if re.match(r'^[^|].*:.*', line)] 

    return feature_names