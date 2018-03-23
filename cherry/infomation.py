import os
from .config import DATA_DIR, LAN_DICT


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
        # Read data from files
        dir = LAN_DICT[lan]['dir']
        type = LAN_DICT[lan]['type']
        if dir:
            tem_lst = []
            # classify = [spam, ham]
            classify = os.listdir(file_dir_path)
            # Gel files list
            for k, v in enumerate(classify):
                tem_lst.append(
                    (k, [os.path.join(file_dir_path+v, f) for f in
                     os.listdir(file_dir_path+v) if f.endswith(type)]))
            for k, v in tem_lst:
                for i in v:
                    with open(i, encoding='utf-8') as f:
                        data_list.append((k, f.read()))
        else:
            file_path = [
                os.path.join(file_dir_path, f) for f in
                os.listdir(file_dir_path) if f.endswith(type)]
            for i in range(len(file_path)):
                with open(file_path[i], encoding='utf-8') as f:
                    for data in f.readlines():
                        data_list.append((i, data))
                    # Get file name
                    classify.append(
                        os.path.basename(os.path.normpath(file_path[i])))
        return data_list, classify
