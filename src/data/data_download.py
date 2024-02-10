import os
import zipfile
import requests
from src.config import RAW_DATA_DIR


def download_data(url, filename):
    """
    Download data from a given url and save it to the data/raw folder
    Returns the path to the downloaded file
    """
    response = requests.get(url)
    with open(f'{RAW_DATA_DIR}/{filename}', 'wb') as f:
        f.write(response.content)
    return f'{RAW_DATA_DIR}/{filename}'


def extract_zip_data(file_path, extract_dir, delete_zip=False):
    """
    Extract the contents of a zip file to a given directory
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f'Extracted {file_path} to {extract_dir}')
    if delete_zip:
        os.remove(file_path)
        print(f'Deleted {file_path}')