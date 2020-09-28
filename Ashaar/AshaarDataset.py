import json
from glob import glob
from .data_classes import Poet, Poem


class AshaarDataset:
    undefined = "غير محدد"

    def __init__(self, dataset_name=None):
        self.dataset_name = dataset_name
        _poets = set()

    def _load_dataset(self):
        """
        Retrieves the dataset from web
        """
        pass

    def _merge_dataset(self, dataset):
        """
        Merge a given dataset to the current data 'deduplicate'
        """
        pass

    def objectify_dataset(self, data_folder):
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

    def download_datasets(self, folder_path=None):
        pass
