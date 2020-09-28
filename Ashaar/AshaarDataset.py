import io
import json
import os
import sys
import zipfile
from glob import glob

import requests
from tqdm import tqdm

from .data_classes import Poem, Poet
from .preprocessors import AshaarPreprocessor


class AshaarDataset:
    undefined = "غير محدد"
    datasets_links = dict(
        aldiwan="https://storage.googleapis.com/tnqeeb/Ashaar/aldiwan/v1/data.zip",
        aldiwanalarabi="https://storage.googleapis.com/tnqeeb/Ashaar/aldiwanalarabi/v1/data.zip",
        dctabudhabi="https://storage.googleapis.com/tnqeeb/Ashaar/dctabudhabi/v1/data.zip",
        diwany="https://storage.googleapis.com/tnqeeb/Ashaar/diwany/v1/data.zip",
        poetsgate="https://storage.googleapis.com/tnqeeb/Ashaar/poetsgate/v1/data.zip",
    )

    def __init__(
        self, dataset_name=None, dataset_target_path=None, load_from_json_files=None
    ):
        """
        dataset target path is the path where the dataset will be downloaded.
        If None,the datsaet will be downloaded to current working directory
        load from json if the user want to load from already downloaded data in a given folder path containing the json files
        """
        self.dataset_name = dataset_name
        self.dataset_target_path = (
            dataset_target_path if dataset_target_path else os.getcwd()
        )
        self.preprocessor = AshaarPreprocessor()
        if load_from_json_files:
            self.poets = self._load_from_json(load_from_json_files)
        else:
            self.poets = set()

    @classmethod
    def list_datasets(cls):
        return list(cls.datasets_links.keys())

    def download(self, target_path=None):
        dataset_url = self.datasets_links.get(self.dataset_name)
        if not dataset_url:
            raise ValueError(
                f"Wrong dataset name {self.dataset_name}. user list_datasets() to list available datasets"
            )
        self._download_dataset(dataset_url)
        self.poets = self._load_from_json(self.dataset_target_path + "/data")

    def _get_content_with_progressbar(self, request):
        totalsize = int(request.headers.get("content-length", 0))
        blocksize = 3413334
        sys.stdout.flush()
        bar = tqdm(
            total=totalsize,
            unit="iB",
            unit_scale=True,
            ncols=5,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        content = None
        for data in request.iter_content(blocksize):
            bar.update(len(data))
            if content is None:
                content = data
            else:
                content += data
        return content

    def _download_dataset(self, url):
        """
        Retrieves the dataset from web
        """
        try:
            binaries_request = requests.get(url, stream=True)
            "show the progress bar while getting content"
            content_bytes = self._get_content_with_progressbar(binaries_request)
            binzip = zipfile.ZipFile(io.BytesIO(content_bytes))
            binzip.extractall(path=self.dataset_target_path)
        except Exception as e:
            print("an error occured")
            print(e)

    def _load_from_json(self, data_folder):
        """
        convert JSON dataset files into a data object of Poet and Poem classes
        param dataset_path: the path to the data folder.
        """
        poets = list()
        for json_file in glob(f"{data_folder}/*"):
            poet = Poet()
            poet_dict = json.load(open(json_file))

            poet.name = poet_dict.get("name", self.undefined)
            poet.bio = poet_dict.get("description", self.undefined)
            poet.poems = list()
            poet.age = poet_dict.get("age", self.undefined)
            poet.url = poet_dict.get("url", self.undefined)
            poet.location = poet_dict.get("location", self.undefined)

            for poem_dict in poet_dict.get("poems", list()):
                poem = Poem()
                poem.baits = poem_dict.get("baits", self.undefined)
                poem.title = poem_dict.get("title", self.undefined)
                poem.theme = poem_dict.get("theme", self.undefined)
                poem.bahr = poem_dict.get("bahr", self.undefined)
                poem.url = poem_dict.get("url", self.undefined)
                poet.poems.append(poem)
            poets.append(poet)
        return poets

    def add_dataset(self, data_folder_path):
        """
        add a dataset to the object data
        """
        pass

    def get_poet(self, poet_name):
        preprocessed_poet_name = self.preprocessor.preprocess(poet_name)
        for poet in self.poets:
            if self.preprocessor.preprocess(poet.name) == preprocessed_poet_name:
                return poet
        return None
