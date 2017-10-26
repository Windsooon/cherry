import jieba
from numpy import zeros


def read_files(files_path):
    '''
    :type file_path: list
    get sentences from local files
    '''
    vocab_set = set()
    v = []
    for i in range(len(files_path)):
        with open(files_path[i], encoding='utf-8') as f:
            v.extend([i]*len(f.readlines()))
            for i in f.readlines():
                vocab_set = vocab_set | set(jieba.cut(i))
    return v, [i for i in vocab_set if len(i) > 1]


def word_to_vector(vocab_list, sentence):
    '''
    type vocab_list: list
    type sentence: strings
    convert strings to vector
    '''
    return_vec = [0]*len(vocab_list)
    for i in jieba.cut(sentence):
        if i in vocab_list:
            return_vec[vocab_list.index(i)] += 1
    return return_vec


def train_bayes(vocab_matrix, vocab_category):
    '''
    type vocab_matrix: matrix
    type vocab_category: int
    '''
    pass


files_path = ['色情.dat', '赌博.dat', '正常.dat', '政治.dat']
classify, vocab_list = read_files(files_path)
