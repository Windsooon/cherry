# -*- coding: utf-8 -*-

"""
cherry.analysis
~~~~~~~~~~~~
This module implements the cherry Analysis.
:copyright: (c) 2018 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

import matplotlib.pyplot as plt
from terminaltables import AsciiTable
from .trainer import Trainer
from .classify import Result
from .infomation import Info
from .config import LAN_DICT, POSITIVE


class Analysis:
    def __init__(self, lan, test_time, test_num, debug, positive):
        self.lan = lan
        self.test_time = test_time
        self.test_num = test_num
        self.debug = debug
        self.positive = POSITIVE
        # Confusion matrix list
        self.cm_lst = []
        if self.positive:
            self.roc_lst = []
        self.split = LAN_DICT[self.lan]['split']
        self.dir = LAN_DICT[self.lan]['dir']
        self.type = LAN_DICT[self.lan]['type']
        self._wrong_lst = []
        self._error_rate = 0
        self._start_analysis()
        if self.positive:
            self._plot_roc()
        # Set up confustion matrix table style
        title = 'Cherry'
        self.confusion_table = AsciiTable(tuple(self.cm_lst), title)
        for i in range(1, len(self._categories)+1):
            self.confusion_table.justify_columns[i] = 'right'

    @property
    def categories(self):
        return self._categories

    @property
    def ctable(self):
        return self.confusion_table.table

    @property
    def wtable(self):
        return self.wrong_lst_table.table

    @property
    def error_rate(self):
        return "{0:.2f}".format(
            self._error_rate*100/(self.test_time*self.test_num))+'%'

    def _start_analysis(self):
        # Create deafullt confustion matrix
        info = Info(lan=self.lan)
        self._categories = info.classify

        # Create a confusiton matrix list
        self.cm_lst.append(['Confusion matrix'] + self._categories)
        for i in self._categories:
            self.cm_lst.append(['(Real)'+i] + [0] * len(self._categories))

        # Test begins
        for i in range(self.test_time):
            trainer = Trainer(
                lan=self.lan, test_num=self.test_num,
                split=self.split, dir=self.dir, type=self.type)
            for k, data in enumerate(trainer.test_data):
                r = Result(text=data[1], lan=self.lan)
                # Get positive percentage
                if self.positive:
                    if any(self.positive == k for k, v in r.percentage):
                        for k, v in r.percentage:
                            if k == self.positive:
                                positive_percentage = v
                                break
                    else:
                        error = (
                            'Please make sure POSITIVE is' +
                            ' correct in config.py')
                        raise IOError(error)
                    self.roc_lst.append((data[0], positive_percentage))
                # Comfusion matrix
                self.cm_lst[data[0]+1][self._categories.index(
                    r.percentage[0][0])+1] += 1
                if data[0] != self._categories.index(r.percentage[0][0]):
                    self._error_rate += 1
                    if self.debug:
                        print('*'*20)
                        print(
                            'real clasify is {0}'.
                            format(self._categories[data[0]]))
                        print('-'*20)
                        print('data is {0}'.format(data[1]))
                        print('-'*20)
                        print('percentage is {0}'.format(r.percentage))
                        print('-'*20)
                        print('word_list is {0}'.format(r.word_list))
                        print('-'*20)

        self.cm_lst.append(["Error rate is {0:.2f}".format(
            self._error_rate*100/(self.test_time*self.test_num))+'%'])

    def _plot_roc(self):
        '''
        Draw ROC curve
        '''
        cur = (1.0, 1.0)
        y_sum = 0.0
        num_positive = len(
            [k for k, _ in self.roc_lst if
                k == self.categories.index(POSITIVE)])
        y_step = 1/num_positive
        x_step = 1/(len(self.roc_lst)-num_positive)
        sorted_lst = sorted(self.roc_lst, key=lambda x: x[1])
        fig = plt.figure()
        fig.clf()
        ax = plt.subplot(111)
        # [0.1, 0.4, 0.5, 0.7, 0.8]
        for k, v in sorted_lst:
            if k == self.categories.index(POSITIVE):
                del_x, del_y = 0, y_step
            else:
                del_x, del_y = x_step, 0
                y_sum += cur[1]
            ax.plot([cur[0], cur[0]-del_x], [cur[1], cur[1]-del_y], c='b')
            cur = (cur[0]-del_x, cur[1]-del_y)
        ax.plot([0, 1], [0, 1], 'b--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        ax.axis([0, 1, 0, 1])
        plt.savefig('auc.png')
        self.cm_lst.append(['Auc is {0:.2f} %'.format(y_sum * x_step * 100)])
