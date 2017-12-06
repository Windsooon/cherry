import os
import pickle
import random
import jieba
import numpy
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
        self.vector_cache = os.path.join(BASE_DIR, 'cache/' + lan + '/vector.cache')
        self.vocab_cache = os.path.join(BASE_DIR, 'cache/' + lan + '/vocab.cache')
        if cache:
            self._load_cache()
        else:
            self.file_path = self._get_default_path_dir(lan)
            self.files_num = len(self.file_path)
            self._split_data(test_num)
            if topN:
                self._vocab_list_remove_topN(topN)
            else:
                self._vocab_list()
            self.matrix_list = self._get_vocab_matrix()
            self.vector = self._get_vector()
            # Write self.vacab_list as cache to file
            self._write_cache()

    def _load_cache(self):
        try:
            with open(self.vector_cache, 'rb') as f:
                self.vector = pickle.load(f)
            with open(self.vocab_cache, 'rb') as f:
                self.vocab_list = pickle.load(f)
        except IOError:
                raise IOError("Can't find cache files")

    def _write_cache(self):
        with open(self.vector_cache, 'wb') as f:
            pickle.dump(self.vocab_list, f)
        with open(self.vocab_cache, 'wb') as f:
            pickle.dump(self.vector, f)
    
    def _get_default_path_dir(self, lan):
        data_dir = os.path.join(BASE_DIR, 'data/' + lan + '/large/')
        return [data_dir + x for x in os.listdir(data_dir)]

    def _read_files(self):
        '''
        Read all sentences from file given in file_path

        Set up:

        self.data_list:
            [
                "What a lovely day",
                "Free porn videos and sex movie",
                "I like to gamble",
                "I love my dog sunkist"
            ]

        self.classify: [2, 0, 1, 2]

        self.data_len: 4
        '''
        self.data_list = []
        self.classify = []
        for i in range(self.files_num):
            with open(self.file_path[i], encoding='utf-8') as f:
                for k in f.readlines():
                    # Insert all data from files in to data_list
                    self.data_list.append(k)
                    # Store this sentence belongs to which category
                    self.classify.append(i)
        self.data_len = len(self.classify)

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
                "I like to gamble",
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
        self.test_classify = [self.classify[r] for r in random_list]
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
        vocab_lst = [i[0] for i in d.most_common() if len(i[0]) > 1]
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
            self.vocab_list = [i for i in vocab_set if len(i) > 1]

    def sentence_to_vector(self, sentence):
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
        return [self.sentence_to_vector(i) for i in self.train_data]

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
        num = numpy.ones(len(self.matrix_list[0]))
        cal, sentence = 2.0, 0.0
        # TODO: Just for in one-time in train_classify
        for i in range(len(self.train_classify)):
            if self.train_classify[i] == index:
                sentence += 1
                num += self.matrix_list[i]
                cal += sum(self.matrix_list[i])
        return numpy.log(num/cal), sentence/len(self.train_data)

    def bayes_classify(self, sentence_vector):
        '''
        Classify sentence depend on self.vector
        type: sentence_vector: strings
        '''
        word_vector = self.sentence_to_vector(sentence_vector)
        percentage_list = [
            sum(i[0] * word_vector) + numpy.log(i[1])
            for i in self.vector]
        max_val = max(percentage_list)
        for i, j in enumerate(percentage_list):
            if j == max_val:
                return i, percentage_list
