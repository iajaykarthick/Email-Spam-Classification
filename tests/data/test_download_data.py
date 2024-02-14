import os
import zipfile

import pytest
from requests.exceptions import RequestException

from src.config import DATA_URL
from src.data.data_download import download_data, extract_zip_data


def test_download_data(requests_mock, tmp_path):
    
    requests_mock.get(DATA_URL, content=b'file_content')
    file_name = DATA_URL.split('/')[-1]
    
    file_path = download_data(DATA_URL, file_name, save_dir=tmp_path)
    
    assert file_path == os.path.join(tmp_path, file_name)
    
    with open(file_path, 'rb') as f:
        assert f.read() == b'file_content'
        
    os.remove(file_path)
    

def test_extract_zip_data(tmp_path):

    zip_file = os.path.join(tmp_path, 'test.zip')
    with zipfile.ZipFile(zip_file, 'w') as f:
        f.writestr('test.txt', 'test_content')
        
    extract_zip_data(zip_file, tmp_path)
    
    assert os.path.exists(os.path.join(tmp_path, 'test.txt'))
    
    os.remove(os.path.join(tmp_path, 'test.txt'))
    os.remove(zip_file)
    
    
def test_download_data_failure(requests_mock, tmp_path):
    requests_mock.get(DATA_URL, status_code=500)
    file_name = DATA_URL.split('/')[-1]
    
    with pytest.raises(RequestException):
        download_data(DATA_URL, file_name, save_dir=tmp_path)
        
        
def test_extract_zip_data_no_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_zip_data('non_existent_file.zip', tmp_path)

def test_extract_zip_data_failure(tmp_path):
    non_zip_file = os.path.join(tmp_path, 'test.txt')
    with open(non_zip_file, 'w') as f:
        f.write('test_content')
        
    with pytest.raises(zipfile.BadZipFile):
        extract_zip_data(non_zip_file, tmp_path)