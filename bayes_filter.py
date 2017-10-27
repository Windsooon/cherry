import jieba
import numpy


SEX = 0
GAMBLE = 1
NORMAL = 2
POLITICS = 3


class Bayes:

    def __init__(self, path):
        self.path = path

    def read_files(self):
        '''
        type file_path: list
        get sentences from local files
        '''
        vocab_set = set()
        classify = []
        for i in range(len(self.path)):
            with open(self.path[i], encoding='utf-8') as f:
                for k in f.readlines():
                    vocab_set = vocab_set | set(jieba.cut(k))
                    classify.append(i)
        self.classify = classify
        self.vocab_list = [i for i in vocab_set if len(i) > 1]

    def word_to_vector(self, sentence):
        '''
        type vocab_list: list
        type sentence: strings
        convert strings to vector
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
        for file in self.path:
            with open(file, encoding='utf-8') as f:
                for i in f.readlines():
                    matrix_list.append(self.word_to_vector(i))
        return matrix_list

    def train_bayes(self):
        '''
        type vocab_matrix: matrix
        type vocab_category: int
        '''
        matrix_list = self.get_vocab_matrix()
        sex_num, gamble_num = numpy.zeros(len(matrix_list[0]))
        politics_num, normal_num = numpy.zeros(len(matrix_list[0]))
        sex_cal = gamble_cal = normal_cal = politics_cal = 0.0
        for i in range(len(self.matrix_list)):
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
        sex_vector = sex_num/sex_cal
        gamble_vector = gamble_num/gamble_cal
        normal_vector = normal_num/normal_cal
        politics_vector = politics_num/politics_cal
        return sex_vector, gamble_vector, normal_vector, politics_vector


files_path = ['sex.dat', 'gamble.dat', 'normal.dat', 'politics.dat']
bayes = Bayes(files_path)
bayes.read_files()
bayes.train_bayes()
