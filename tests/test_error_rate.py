import unittest

from classify import bayes


class BayesTest(unittest.TestCase):
    def test_error_rate(self):
        '''
        test error rate
        '''
        def error_rate(instance):
            classify_results = []
            for i in range(len(instance.test_data)):
                percentage_list, word_list = (
                    instance.bayes_classify(instance.test_data[i]))
                test_result = sorted(
                    percentage_list, key=lambda x: x[1], reverse=True)[0][0]
                classify_results.append(test_result)
                # Uncomment to see which sentence was classified wrong.
                # if test_result != instance.test_classify[i]:
                #     print('-'*20)
                #     print(instance.test_data[i])
                #     print('test_result is %s' % test_result)
                #     print('true is %s' % instance.test_classify[i])
                #     print('percentage_list is %s' % percentage_list)
                #     print('-'*20)
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

