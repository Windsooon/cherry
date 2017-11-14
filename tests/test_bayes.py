import os
import unittest
# import sys
# sys.path.insert(0, '/Users/anchuang/learn/filter/')

from filter import spam_filter


class BayesTest(unittest.TestCase):
    def setUp(self):
        try:
            os.remove(os.path.join(spam_filter.BASE_DIR, 'cache/vector.cache'))
            os.remove(os.path.join(spam_filter.BASE_DIR, 'cache/vocab.cache'))
        except OSError:
            pass

    def test_did_not_create_cache_files_when_set_false(self):
        spam_filter.Filter(cache=False)
        self.assertTrue(
            os.path.isfile(
                os.path.join(spam_filter.BASE_DIR, 'cache/vector.cache')))
        self.assertTrue(
            os.path.isfile(
                os.path.join(spam_filter.BASE_DIR, 'cache/vocab.cache')))

    def test_error_rate(self):
        '''
        test error rate
        '''
        def error_rate(instance):
            classify_results = []
            for i in range(len(instance.test_data)):
                test_result, percentage_list = (
                    instance.bayes_classify(instance.test_data[i]))
                classify_results.append(test_result)
                # Uncomment  to see which sentence was classified wrong.
                # if test_result != self.test_classify[i]:
                #     print(self.test_data[i])
                #     print('test_result is %s' % test_result)
                #     print('true is %s' % self.test_classify[i])
                #     print('percentage_list is %s' % percentage_list)
            wrong_results = [
                i for i, j in
                zip(instance.test_classify, classify_results) if i != j]
            return len(wrong_results) / len(instance.test_data)

        a = []
        test_times = 20
        print('\nThis may takes some time')
        for i in range(test_times):
            if i % 5 == 0:
                print('Completed %s tasks, %s tasks left.' %
                      (i, test_times - i))
            test_spam_filter = spam_filter.Filter(cache=False)
            a.append(error_rate(test_spam_filter))
        print('The error rate is %s' %
              "{0:.2f}".format(sum(a) / test_times * 100) + '%')

    def test_data_num_correct(self):
        test_bayes = spam_filter.Filter(test_num=80, cache=False)
        self.assertTrue(len(test_bayes.test_data), 80)

    def test_created_cache_files(self):
        spam_filter.Filter()
        self.assertTrue(
            os.path.isfile(
                os.path.join(spam_filter.BASE_DIR, 'cache/vector.cache')))
        self.assertTrue(
            os.path.isfile(
                os.path.join(spam_filter.BASE_DIR, 'cache/vocab.cache')))

    def test_diy_dictionary(self):
        '''
        test error rate
        '''
        def error_rate(instance):
            classify_results = []
            for i in range(len(instance.test_data)):
                test_result, percentage_list = (
                    instance.bayes_classify(instance.test_data[i]))
                classify_results.append(test_result)
                # Uncomment  to see which sentence was classified wrong.
                # if test_result != self.test_classify[i]:
                #     print(self.test_data[i])
                #     print('test_result is %s' % test_result)
                #     print('true is %s' % self.test_classify[i])
                #     print('percentage_list is %s' % percentage_list)
            wrong_results = [
                i for i, j in
                zip(instance.test_classify, classify_results) if i != j]
            return len(wrong_results) / len(instance.test_data)
        a = []
        test_times = 20
        print('\nThis may takes some time')
        for i in range(test_times):
            if i % 5 == 0:
                print('Completed %s tasks, %s tasks left.' %
                      (i, test_times - i))
            test_spam_filter = spam_filter.Filter(
                cache=False, dictionary='dict.txt')
            a.append(error_rate(test_spam_filter))
        print('The error rate is %s' %
              "{0:.2f}".format(sum(a) / test_times * 100) + '%')


if __name__ == '__main__':
    unittest.main()
