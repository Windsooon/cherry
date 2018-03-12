import unittest
import cherry


class ApiTest(unittest.TestCase):

    def test_classify(self):
        pass

    def test_train(self):
        train = cherry.train(
            lan='Chinese', test_num=0, test_mode=False,
            positive=cherry.config.POSITIVE, binary=True)
        print(train.test_data)
