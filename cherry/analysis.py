# -*- coding: utf-8 -*-

"""
cherry.analysis
~~~~~~~~~~~~
This module implements the cherry Analysis.
:copyright: (c) 2018 by Windson Yang
:license: MIT License, see LICENSE for more details.
"""

from terminaltables import AsciiTable
from .trainer import Trainer
from .classify import Result
from .infomation import Info
from .config import LAN_DICT


class Analysis:
    def __init__(self, **kwargs):
        self.lan = kwargs['lan']
        self.test_time = kwargs['test_time']
        self.test_num = kwargs['test_num']
        self.debug = kwargs['debug']
        self.split = LAN_DICT[self.lan]['split']
        self.dir = LAN_DICT[self.lan]['dir']
        self.type = LAN_DICT[self.lan]['type']
        self._wrong_lst = []
        self._error_rate = 0
        self._start_analysis()

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

        # Create a confusiton matrix list
        cm_lst = []
        cm_lst.append(['Confusion matrix'] + info.classify)
        for i in info.classify:
            cm_lst.append(['(Real)'+i] + [0] * len(info.classify))

        # Test begins
        for i in range(self.test_time):
            trainer = Trainer(
                lan=self.lan, test_num=self.test_num,
                split=self.split, dir=self.dir, type=self.type)
            for k, data in enumerate(trainer.test_data):
                r = Result(text=data[1], lan=self.lan)
                cm_lst[data[0]+1][info.classify.index(
                    r.percentage[0][0])+1] += 1
                if data[0] != info.classify.index(r.percentage[0][0]):
                    self._error_rate += 1
                    if self.debug:
                        print('*'*20)
                        print(
                            'real clasify is {0}'.
                            format(info.classify[data[0]]))
                        print('-'*20)
                        print('data is {0}'.format(data[1]))
                        print('-'*20)
                        print('percentage is {0}'.format(r.percentage))
                        print('-'*20)
                        print('word_list is {0}'.format(r.word_list))
                        print('-'*20)

        cm_lst.append(["Error rate is {0:.2f}".format(
            self._error_rate*100/(self.test_time*self.test_num))+'%'])
        # Set up confustion matrix table style
        title = 'Cherry'
        self.confusion_table = AsciiTable(tuple(cm_lst), title)
        for i in range(1, len(info.classify)+1):
            self.confusion_table.justify_columns[i] = 'right'
