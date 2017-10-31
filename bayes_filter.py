import jieba
import numpy


SEX = 0
GAMBLE = 1
NORMAL = 2
POLITICS = 3


class Bayes:

    def __init__(self, path, num):
        self.path = path
        self.num = num
        self.data_set = []
        self.classify = []

    def read_files(self):
        '''
        type file_path: list
        get all data from local files
        '''
        for i in range(len(self.path)):
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
        return matrix_list

    def train_bayes(self):
        '''
        train native bayes
        '''
        matrix_list = self.get_vocab_matrix()
        sex_num = gamble_num = numpy.ones(len(matrix_list[0]))
        politics_num = normal_num = numpy.ones(len(matrix_list[0]))
        sex_cal = gamble_cal = normal_cal = politics_cal = 2.0
        for i in range(len(matrix_list)):
            if self.classify[i] == SEX:
                sex_num += matrix_list[i]
                sex_cal += sum(matrix_list[i])
            elif self.classify[i] == GAMBLE:
                gamble_num += matrix_list[i]
                gamble_cal += sum(matrix_list[i])
            elif self.classify[i] == NORMAL:
                normal_num += matrix_list[i]
                normal_cal += sum(matrix_list[i])
            elif self.classify[i] == POLITICS:
                politics_num += matrix_list[i]
                politics_cal += sum(matrix_list[i])
        # self.sex_vector = numpy.log(sex_num/sex_cal)
        # self.gamble_vector = numpy.log(gamble_num/gamble_cal)
        # self.normal_vector = numpy.log(normal_num/normal_cal)
        # self.politics_vector = numpy.log(politics_num/politics_cal)
        self.sex_vector = (sex_num/sex_cal)
        self.gamble_vector = (gamble_num/gamble_cal)
        self.normal_vector = (normal_num/normal_cal)
        self.politics_vector = (politics_num/politics_cal)
        print(self.sex_vector)
        print('-'*30)
        print(self.gamble_vector)
        print('-'*30)
        print(self.normal_vector)
        print('-'*30)
        print(self.politics_vector)

    def classify_bayes(self, sentence_vector):
        '''
        type word_vector: list
        sex_vector: numpy matrix
        gambel_vector: numpy matrix
        normal_vector: numpy matrix
        politics_vector: numpy matrix
        '''
        word_vector = self.sentence_to_vector(sentence_vector)
        sex_percentage = (sum(self.sex_vector * word_vector), SEX)
        gamble_percentage = (sum(self.gamble_vector * word_vector), GAMBLE)
        normal_percentage = (sum(self.normal_vector * word_vector), NORMAL)
        politics_percentage = (
                sum(self.politics_vector * word_vector), POLITICS)
        return max(
            sex_percentage, gamble_percentage,
            normal_percentage, politics_percentage)[1]

    def error_rate(self):
        classify_results = []
        for i in self.test_data:
            classify_results.append(self.classify_bayes(i))
        return [
            i for i, j in zip(self.test_classify, classify_results) if i != j]


files_path = ['sex.dat', 'gamble.dat', 'normal.dat', 'politics.dat']
bayes = Bayes(files_path, 20)
bayes.read_files()
bayes.split_data()
bayes.vocab_list()
bayes.train_bayes()
bayes.error_rate()
