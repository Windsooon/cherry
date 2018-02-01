import os
import shutil   
import unittest

from classify import bayes


class BayesTest(unittest.TestCase):
    def test_create_cache_files_after_first_set(self):
        self.assertTrue(
            os.path.isfile(
                os.path.join(bayes.BASE_DIR, 'all_data/Chinese/cache/vector.cache')))
        folder = os.path.join(bayes.BASE_DIR, 'all_data/Chinese/cache')
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                pass
        test_bayes = bayes.Classify(cache=False)
        self.assertTrue(
            os.path.isfile(
                os.path.join(bayes.BASE_DIR, 'all_data/Chinese/cache/vector.cache')))
        self.assertTrue(
            os.path.isfile(
                os.path.join(bayes.BASE_DIR, 'all_data/Chinese/cache/vocab.cache')))
        self.assertTrue(
            os.path.isfile(
                os.path.join(bayes.BASE_DIR, 'all_data/Chinese/cache/classify.cache')))

    def test_classify_work(self):
        test_bayes = bayes.Classify()
        percentage_list, word_list = test_bayes.bayes_classify(
            '美联储当天结束货币政策例会后发表声明说，自2017年12月以来，' + 
            '美国就业市场和经济活动继续保持稳健增长，失业率继续维持在低水平。')
        test_result = sorted(
                    percentage_list, key=lambda x: x[1], reverse=True)[0][0]
        self.assertEqual(test_result, 'normal.dat')

    def test_cache_work(self):
        test_bayes = bayes.Classify(cache=False)
        percentage_list, word_list = test_bayes.bayes_classify(
            '美联储当天结束货币政策例会后发表声明说，自2017年12月以来，' + 
            '美国就业市场和经济活动继续保持稳健增长，失业率继续维持在低水平。')
        test_result = sorted(
                    percentage_list, key=lambda x: x[1], reverse=True)[0][0]
        self.assertEqual(test_result, 'normal.dat')

    def test_data_num_correct(self):
        test_bayes = bayes.Classify(test_num=80, cache=False)
        self.assertTrue(len(test_bayes.test_data), 80)
