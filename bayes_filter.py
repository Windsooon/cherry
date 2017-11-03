import jieba
import numpy


SEX = 0
GAMBLE = 1
NORMAL = 2
POLITICS = 3


class Bayes:

    def __init__(self, path, test_num=5):
        self.path = path
        self.files_len = len(self.path)
        self.num = test_num
        self.data_set = []
        self.classify = []

    def read_files(self):
        '''
        type file_path: list
        get all data from local files
        '''
        for i in range(self.files_len):
            with open(self.path[i], encoding='utf-8') as f:
                for k in f.readlines():
                    self.data_set.append(k)
                    self.classify.append(i)
        self.data_len = len(self.classify)

    def split_data(self):
        '''
        Split data to test data and train data randomly.
        '''
        if self.num > self.data_len - 1:
            raise IndexError("Test data should small than %s" % self.data_len)
        random_list = [
            numpy.random.randint(1, self.data_len) for r in range(self.num)]
        self.test_data = [self.data_set[r] for r in random_list]
        self.test_classify = [self.classify[r] for r in random_list]
        self.train_data = [
            self.data_set[r] for r in range(self.data_len)
            if r not in random_list]
        self.train_classify = [
            self.classify[r] for r in range(self.data_len)
            if r not in random_list]

    def vocab_list(self):
        '''
        Get words list longer than one word
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

    def get_vocab_matrix(self):
        '''
        type file_path: list
        convert all sentences to vector
        '''
        matrix_list = []
        for i in self.train_data:
            matrix_list.append(self.sentence_to_vector(i))
        self.matrix_list = matrix_list

    def get_vector(self):
        return [self.train_bayes(i) for i in range(self.files_len)]

    def train_bayes(self, index):
        '''
        type count: number of data in category
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
        all_vector = self.get_vector()
        word_vector = self.sentence_to_vector(sentence_vector)
        percentage_list = [
            sum(i[0] * word_vector) + numpy.log(i[1])
            for i in all_vector]
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
            if test_result != self.test_classify[i]:
                print(self.test_data[i])
                print('test_result is %s' % test_result)
                print('true is %s' % self.test_classify[i])
                print('percentage_list is %s' % percentage_list)
        wrong_results = [
            i for i, j in zip(self.test_classify, classify_results) if i != j]
        return len(wrong_results)/len(self.test_data)


files_path = ['sex.dat', 'gamble.dat', 'normal.dat', 'politics.dat']
bayes = Bayes(files_path, 30)
bayes.read_files()
bayes.split_data()
bayes.vocab_list()
bayes.get_vocab_matrix()
print(bayes.error_rate())
