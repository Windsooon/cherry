import os
import shutil   
import unittest

from classify import bayes


class BayesTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_create_cache_files_after_first_set(self):
        folder = os.path.join(bayes.BASE_DIR, 'cache')
        for the_file in os.listdir(folder):
            file_path = os.path.join(folder, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                pass
        bayes.Classify(cache=False)
        self.assertTrue(
            os.path.isfile(
                os.path.join(bayes.BASE_DIR, 'cache/Chinese/vector.cache')))
        self.assertTrue(
            os.path.isfile(
                os.path.join(bayes.BASE_DIR, 'cache/Chinese/vocab.cache')))

    def test_data_num_correct(self):
        test_bayes = bayes.Classify(test_num=80, cache=False)
        self.assertTrue(len(test_bayes.test_data), 80)

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
            return len(wrong_results)/len(instance.test_data)

        a = []
        test_times = 20
        print('\nThis may takes some time')
        for i in range(test_times):
            if i % 5 == 0:
                print('Completed %s tasks, %s tasks left.' % (i, test_times-i))
            test_bayes = bayes.Classify(cache=False)
            a.append(error_rate(test_bayes))
        print('The error rate is %s' % "{0:.2f}".format(sum(a)/test_times*100)+'%')

