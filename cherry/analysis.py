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


class Analysis:
    def __init__(self, **kwargs):
        self.lan = kwargs['lan']
        self.test_time = kwargs['test_time']
        self.test_num = kwargs['test_num']
        self.split = kwargs['split']
        self._error_rate = 0
        self._start_analysis()

    @property
    def cmatrix(self):
        return self.table_instance.table

    @property
    def error_rate(self):
        return "{0:.2f}".format(
            self._error_rate/self.test_time*self.test_num*100)+'%'

    def _start_analysis(self):
        # Create deafullt confustion matrix
        info = Info(lan=self.lan)

        cm_lst = []
        cm_lst.append(['Confusion matrix'] + info.classify)
        for i in info.classify:
            cm_lst.append([i] + [0] * len(info.classify))

        # Test begins
        for i in range(self.test_time):
            trainer = Trainer(
                lan=self.lan, test_num=self.test_num, split=self.split)
            for k, data in enumerate(trainer.test_data):
                r = Result(text=data[1], lan=self.lan, split=self.split)
                cm_lst[data[0]+1][info.classify.index(
                    r.percentage[0][0])+1] += 1
                if data[0] != info.classify.index(r.percentage[0][0]):
                    self._error_rate += 1
                # print(data)
                # print(r.percentage)
                # print(r.word_list)

        # Set up confustion matrix style
        title = 'Cherry'
        self.table_instance = AsciiTable(tuple(cm_lst), title)
        for i in range(1, len(info.classify)+1):
            self.table_instance.justify_columns[i] = 'right'
