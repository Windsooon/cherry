import os
from .config import DATA_DIR


class Info:
    def __init__(self, lan):
        self._data_list, self._classify = self.read_files(lan)

    @property
    def data_list(self):
        return self._data_list

    @property
    def classify(self):
        return self._classify

    @classmethod
    def read_files(cls, lan):
        '''
        Read data from given file path

        :param lan: Chinese/English

        data_list:
            [
                (0, "What a lovely day"),
                (1, "I like gambling"),
                (0, "I love my dog sunkist)"
            ]
        classify: ['gamble.dat', 'normal.dat']
        '''
        data_list, classify = [], []
        file_dir_path = os.path.join(DATA_DIR, 'data/' + lan + '/data/')
        # Data files should end with .dat
        file_path = [
            os.path.join(file_dir_path, f) for f in
            os.listdir(file_dir_path) if f.endswith('.dat')]
        for i in range(len(file_path)):
            with open(file_path[i], encoding='utf-8') as f:
                for data in f.readlines():
                    data_list.append((i, data))
                # Get file name
                classify.append(
                    os.path.basename(os.path.normpath(file_path[i])))
        return data_list, classify
