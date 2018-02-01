import os
import pickle
import random
import jieba
import numpy as np
from .config import *

BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'classify')
TEST_DATA_NUM = 30
TOPN = 0


class Classify:

    def __init__(
            self,
            lan='Chinese',
            test_num=TEST_DATA_NUM,
            topN=TOPN,
            cache=True,
            ):

        self._set_path(lan)

        if cache:
            self._load_cache()
        else:
            self.files_num = len(self.file_path)
            self._split_data(test_num)
            # topN wouldn't work with cache
            if topN:
                self._vocab_list_remove_topN(topN)
            else:
                self._vocab_list()
            self.matrix_list = self._get_vocab_matrix()
            self.vector = self._get_vector()
            # Write self.vacab_list and self.vector as cache to file
            self._write_cache()
    
    def _set_path(self, lan):
        all_data_path = os.path.join(BASE_DIR, ('all_data/' + lan + '/'))
        self.vocab_cache_path = os.path.join(all_data_path, 'cache/vocab.cache')
        self.vector_cache_path = os.path.join(all_data_path, 'cache/vector.cache')
        self.classify_cache_path = os.path.join(all_data_path, 'cache/classify.cache')
        self.stop_word_path = os.path.join(all_data_path, 'stop/stop_word.dat')
        data_path = os.path.join(all_data_path, 'data/')
        self.file_path = [data_path + x for x in os.listdir(data_path)]

    def _get_stop_word_path(self, lan):
        return os.path.join(BASE_DIR, 'data/' + lan + '/stop_word.dat')

    def _load_cache(self):
        try:
            with open(self.vocab_cache_path, 'rb') as f:
                self.vocab_list = pickle.load(f)
            with open(self.vector_cache_path, 'rb') as f:
                self.vector = pickle.load(f)
            with open(self.classify_cache_path, 'rb') as f:
                self.CLASSIFY = pickle.load(f)
        except IndexError:
                raise IOError("Can't find cache files")

    def _write_cache(self):
        with open(self.vocab_cache_path, 'wb') as f:
            pickle.dump(self.vocab_list, f)
        with open(self.vector_cache_path, 'wb') as f:
            pickle.dump(self.vector, f)
        with open(self.classify_cache_path, 'wb') as f:
            pickle.dump(self.CLASSIFY, f)

    def _read_files(self):
        '''
        Read all sentences from file given in file_path

        Set up:

        self.data_list:
            [
                "What a lovely day",
                "Free porn videos and sex movie",
                "I like gambling",
                "I love my dog sunkist"
            ]

        self.classify: [2, 0, 1, 2]

        self.data_len: 4
        '''
        self.data_list = []
        self.classify = []
        self.CLASSIFY = []
        for i in range(self.files_num):
            with open(self.file_path[i], encoding='utf-8') as f:
                for k in f.readlines():
                    # Insert all data from files in to data_list
                    self.data_list.append(k)
                    # Store this sentence belongs to which category
                    self.classify.append(i)
                # Get classify name
                self.CLASSIFY.append(
                    os.path.basename(os.path.normpath(self.file_path[i])))
        self.data_len = len(self.classify)

        with open(self.stop_word_path, encoding='utf-8') as f:
            self.stop_word_lst = [l[:-1] for l in f.readlines()]

    def _split_data(self, test_num):
        '''
        Split data into test data and train data randomly.

        type: test_num: int
        Set up:

        self.test_data:
            [
                "What a lovely day",
                "Free porn videos and sex movie",
            ]

        self.test_classify: [2, 0]

        self.train_data:
            [
                "I like gambling",
                "I love my dog sunkist"
            ]

        self.train_classify: [2, 0]
        '''
        self._read_files()
        if test_num > self.data_len - 1:
            raise IndexError("Test data should small than %s" % self.data_len)
        random_list = random.sample(range(0, self.data_len), test_num)
        # get test data
        self.test_data = [self.data_list[r] for r in random_list]
        self.test_classify = [self.CLASSIFY[self.classify[r]] for r in random_list]
        # get train data
        self.train_data = [
            self.data_list[r] for r in
            range(self.data_len) if r not in random_list]
        self.train_classify = [
            self.classify[r] for r in range(self.data_len)
            if r not in random_list]

    def _vocab_list_remove_topN(self, n):
        '''
        Remove topN most occur word

        Set up:

        self.vocab_list:
            [
                'What', 'lovely', 'day',
                'Free', 'porn', 'videos',
                'sex', 'movie', 'like',
                'gamble', 'love', 'dog', 'sunkist'
            ]
        '''
        import collections
        dic = {}
        for k in self.train_data:
            for i in jieba.cut(k):
                if i in dic:
                    dic[i] += 1
                else:
                    dic[i] = 1
        d = collections.Counter(dic)
        vocab_lst = [
            i[0] for i in d.most_common() if (len(i[0]) > 1
            and i[0] not in self.stop_word_lst)
        ]
        self.vocab_list = vocab_lst[n:]

    def _vocab_list(self):
        '''
        Get a list contain all unique non stop words belongs to train_data

        Set up:

        self._vocab_list:
            [
                'What', 'lovely', 'day',
                'Free', 'porn', 'videos',
                'sex', 'movie', 'like',
                'gamble', 'love', 'dog', 'sunkist'
            ]
        '''
        vocab_set = set()
        for k in self.train_data:
            vocab_set = vocab_set | set(jieba.cut(k))
            self.vocab_list = [
                i for i in vocab_set if (len(i) > 1
                and i not in self.stop_word_lst)
            ]

    def _sentence_to_vector(self, sentence):
        '''
        Convert strings to vector depends on vocal_list
        type sentence: strings
        '''
        return_vec = [0]*len(self.vocab_list)
        for i in jieba.cut(sentence):
            if i in self.vocab_list:
                return_vec[self.vocab_list.index(i)] += 1
        return return_vec

    def _get_vocab_matrix(self):
        '''
        Convert all sentences to vector
        '''
        return [self._sentence_to_vector(i) for i in self.train_data]

    def _get_vector(self):
        '''
        Get vector depent on different classify
        '''
        return [self.train_bayes(i) for i in range(self.files_num)]

    def train_bayes(self, index):
        '''
        Train native bayes
        type index: number of files
        '''
        num = np.ones(len(self.matrix_list[0]))
        cal, sentence = 2.0, 0.0
        # TODO: Just for in one-time in train_classify
        for i in range(len(self.train_classify)):
            if self.train_classify[i] == index:
                sentence += 1
                num += self.matrix_list[i]
                cal += sum(self.matrix_list[i])
        return np.log(num/cal), sentence/len(self.train_data)

    def _softmax(self, lst):
        '''
        Compute softmax values for each sets of scores in x.
        '''
        return np.exp(lst) / np.sum(np.exp(lst), axis=0)

    def _min_max(self, lst):
        '''
        Min-Max Normalization
        '''
        return [(x-min(lst))/(max(lst)-min(lst)) for x in lst]

    def _clean_percentage_list(self, lst):
        return list(zip(self.CLASSIFY, self._softmax(self._min_max(lst))))

    def bayes_classify(self, sentence):
        '''
        Classify sentence depend on self.vector
        type: strings: strings
        '''
        word_vector = self._sentence_to_vector(sentence)
        possibility_vector = []
        percentage_list = []
        for i in self.vector:
            # final_vector: [0, -7.3, 0, 0, -8, ...]
            final_vector = i[0] * word_vector
            # word_index: [1, 4]
            word_index = np.nonzero(final_vector)
            # non_zero_word: [明天，永远]
            non_zero_word = np.array(self.vocab_list)[word_index]
            # non_zero_vector: [-7.3, -8]
            non_zero_vector = final_vector[word_index]
            possibility_vector.append(non_zero_vector)
            percentage_list.append(sum(final_vector) + np.log(i[1]))
        possibility_array = np.array(possibility_vector)
        max_val = max(percentage_list)
        for i, j in enumerate(percentage_list):
            if j == max_val:
                max_array = possibility_array[i, :]
                left_array = np.delete(possibility_array, i, 0)
                sub_array = np.zeros(max_array.shape)
                for k in left_array:
                    sub_array += max_array - k
                return self._clean_percentage_list(percentage_list), list(zip(non_zero_word, sub_array))
