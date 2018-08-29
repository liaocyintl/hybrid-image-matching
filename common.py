import os
import ntpath
from datetime import datetime
import json
import numpy as np

def read_csv(path):
    f = open(path)
    lines = f.readlines()  # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
    f.close()
    lines = [line.replace("\n", "") for line in lines]
    return [line.split(",") for line in lines]

def read_lines(file):
    f = open(file)
    lines = f.readlines()  # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
    f.close()
    return [a.replace("\n", "") for a in lines]

def prepare_clean_dir(directory):
    prepare_dir(directory)
    clean_folder(directory)

def prepare_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_filename_and_postfix_from_path(path):
    name = ntpath.basename(path)
    if os.path.isdir(path):
        return name, ""
    else:
        if len(name.split(".")) == 1:
            return name, ""
        else:
            return name.split(".")[0], name.split(".")[1]

def current_datetime():
    return str(datetime.now())

class log:
    def __init__(self):
        self.a = open('run.log', 'a')


    def write(self, str):
        str += " " + current_datetime()
        print(str)
        self.a.write(str + "\n")

    def __del__(self):
        self.a.close()

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def write_json(path, data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False, cls=MyEncoder)

def load_json(path, encoding="utf-8"):
    with open(path, encoding=encoding) as data_file:
        return json.load(data_file)

def clean_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

import pickle
def save_pickle(path, obj):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, mode='rb') as f:
        return pickle.load(f)

def get_all_folders(directory):
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]

try:
    from urllib.parse import urlparse
except ImportError:
     from urlparse import urlparse

def get_hostname_from_url(url):
    parsed_uri = urlparse(url)
    return parsed_uri.hostname


def is_path_exists(path):
    return os.path.exists(path)

if __name__ == "__main__":
    print(get_hostname_from_url("http://docs.python.jp/2/library/urlparse.html"))