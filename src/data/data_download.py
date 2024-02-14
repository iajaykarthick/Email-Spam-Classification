import os
import zipfile
import requests
from requests.exceptions import RequestException
from src.config import RAW_DATA_DIR


def download_data(url, filename, save_dir=RAW_DATA_DIR):
    """
    Download data from a given url and save it to the data/raw folder
    Returns the path to the downloaded file
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except RequestException as e:
        print(f'Error downloading data from {url}')
        raise e
    save_file_path = os.path.join(save_dir, filename)
    with open(save_file_path, 'wb') as f:
        f.write(response.content)
    return save_file_path


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