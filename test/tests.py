import os
import unittest
# import sys
# sys.path.insert(0, '/Users/anchuang/learn/filter/')

import bayes_filter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BayesTest(unittest.TestCase):
    def setUp(self):
        try:
            os.remove(os.path.join(BASE_DIR, 'cache/vector.cache'))
            os.remove(os.path.join(BASE_DIR, 'cache/vocab.cache'))
        except OSError:
            pass

    def test_created_cache_files(self):
        bayes_filter.BayesFilter()
        self.assertTrue(
            os.path.isfile(os.path.join(BASE_DIR, 'cache/vector.cache')))
        self.assertTrue(
            os.path.isfile(os.path.join(BASE_DIR, 'cache/vocab.cache')))

    def test_did_not_create_cache_files_when_set_false(self):
        bayes_filter.BayesFilter(cache=False)
        self.assertFalse(
            os.path.isfile(os.path.join(BASE_DIR, 'cache/vector.cache')))
        self.assertFalse(
            os.path.isfile(os.path.join(BASE_DIR, 'cache/vocab.cache')))

    def test_error_rate(self):
        '''
        test error rate
        '''
        def error_rate():
            classify_results = []
            for i in range(len(self.test_data)):
                test_result, percentage_list = (
                    self.bayes_classify(self.test_data[i]))
                classify_results.append(test_result)
                # Uncomment  to see which sentence was classified wrong.
                # if test_result != self.test_classify[i]:
                #     print(self.test_data[i])
                #     print('test_result is %s' % test_result)
                #     print('true is %s' % self.test_classify[i])
                #     print('percentage_list is %s' % percentage_list)
            wrong_results = [
                i for i, j in
                zip(self.test_classify, classify_results) if i != j]
            return len(wrong_results)/len(self.test_data)

        a = []
        k = 10
        for i in range(k):
            test_bayes_filter = bayes_filter.BayesFilter(cache=False)
            a.append(test_bayes_filter.error_rate())
        print('The error rate is %s' % "{0:.2f}".format(sum(a)/k*100)+'%')
