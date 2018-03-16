import unittest
import cherry

TEST_TIME = 1
TEST_NUM = 50


class ErrorTest(unittest.TestCase):

    def test_error_rate(self):
        e = 0
        for i in range(TEST_TIME):
            trainer = cherry.train(test_num=TEST_NUM)
            for k, data in enumerate(trainer.test_data):
                r = cherry.classify(data[1])
                if (r.percentage[0][0] !=
                        trainer.meta_classify[trainer.test_data_classify[k]]):
                    e += 1
                    print(data)
                    print(r.percentage)
        print(e/(TEST_TIME * TEST_NUM))
