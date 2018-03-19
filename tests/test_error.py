import unittest
import cherry
from terminaltables import AsciiTable
TEST_TIME = 1
TEST_NUM = 60


class ErrorTest(unittest.TestCase):

    def test_error_rate(self):
        e = 0
        for i in range(TEST_TIME):
            trainer = cherry.train(test_num=TEST_NUM)
            for k, data in enumerate(trainer.test_data):
                r = cherry.classify(data[1])
                if (r.percentage[0][0] !=
                        trainer.meta_classify[data[0]]):
                    e += 1
                    print(data)
                    print(r.percentage)
                    print(r.word_list)
        # print(e/(TEST_TIME * TEST_NUM))
        z = []
        z.append(['Confusion matrix'] + trainer.meta_classify)
        for i in trainer.meta_classify:
            k = [i] + [0] * len(trainer.meta_classify)
            z.append(k)
        title = 'Cherry'
        table_instance = AsciiTable(tuple(z), title)
        for i in range(1, len(trainer.meta_classify)+1):
            table_instance.justify_columns[i] = 'right'
        print(table_instance.table)
