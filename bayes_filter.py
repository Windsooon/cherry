import jieba
import random
import numpy

DEFAULT_FILEPATH = [
    'big/normal.dat', 'big/gamble.dat',
    'big/sex.dat', 'big/politics.dat']
TEST_DATA_NUM = 30
TOPN = 0
NORMAL = 0
GAMBLE = 1
SEX = 2
POLITICS = 3


class Bayes:

    def __init__(
            self,
            test_num=TEST_DATA_NUM,
            topN=TOPN,
            file_path=DEFAULT_FILEPATH
            ):
        self.file_path = file_path
        self.files_num = len(self.file_path)
        self._split_data(test_num)
        if topN:
            self._vocab_list_remove_topN(topN)
        else:
            self._vocab_list()
        self.matrix_list = self._get_vocab_matrix()
        self.vector = self._get_vector()

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

        self.test_classify: [2, 0]
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
        Remove topN occur word
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

        self.vocab_list:
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
        type vocab_list: list
        type sentence: strings
        convert strings to vector depends on vocal_list
        '''
        return_vec = [0]*len(self.vocab_list)
        for i in jieba.cut(sentence):
            if i in self.vocab_list:
                return_vec[self.vocab_list.index(i)] += 1
        return return_vec

    def _get_vocab_matrix(self):
        '''
        convert all sentences to vector
        '''
        return [self.sentence_to_vector(i) for i in self.train_data]

    def _get_vector(self):
        return [self.train_bayes(i) for i in range(self.files_num)]

    def train_bayes(self, index):
        '''
        type index: number of data in category
        train native bayes
        '''
        num = numpy.ones(len(self.matrix_list[0]))
        cal, sentence = 2.0, 0.0
        for i in range(len(self.train_classify)):
            if self.train_classify[i] == index:
                sentence += 1
                num += self.matrix_list[i]
                cal += sum(self.matrix_list[i])
        return numpy.log(num/cal), sentence/len(self.train_data)

    def classify_bayes(self, sentence_vector):
        '''
        '''
        word_vector = self.sentence_to_vector(sentence_vector)
        percentage_list = [
            sum(i[0] * word_vector) + numpy.log(i[1])
            for i in self.vector]
        max_val = max(percentage_list)
        for i, j in enumerate(percentage_list):
            if j == max_val:
                return i, percentage_list

    def error_rate(self):
        classify_results = []
        for i in range(len(self.test_data)):
            test_result, percentage_list = (
                self.classify_bayes(self.test_data[i]))
            classify_results.append(test_result)
            # Uncomment below to see which sentence classify wrong.
            # if test_result != self.test_classify[i]:
            #     print(self.test_data[i])
            #     print('test_result is %s' % test_result)
            #     print('true is %s' % self.test_classify[i])
            #     print('percentage_list is %s' % percentage_list)
        wrong_results = [
            i for i, j in zip(self.test_classify, classify_results) if i != j]
        return len(wrong_results)/len(self.test_data)


if __name__ == '__main__':
    a = []
    k = 20
    for i in range(k):
        bayes = Bayes()
        a.append(bayes.error_rate())
    print('The error rate is %s' % str(sum(a)/k*100)+'%')
