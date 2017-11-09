Bayes Filter
=======================

.. image:: https://img.shields.io/pypi/v/bayes-filter.svg
    :target: https://pypi.python.org/pypi/bayes-filter

.. image:: https://img.shields.io/pypi/l/bayes-filter.svg
    :target: https://pypi.python.org/pypi/bayes-filter

.. image:: https://img.shields.io/pypi/pyversions/bayes-filter.svg
    :target: https://pypi.python.org/pypi/bayes-filter


`中文版本`_

English_Version_

.. _`中文版本`:
这个项目使用朴素贝叶斯算法来判别垃圾内容，用户输入句子会被判别成正常，色情，赌博，政治敏感四个类别。现在使用了400个训练数据，正确率大约93%

如何使用:

.. code-block:: python

    >>> from bayes import bayes_filter
    >>> test_bayes = bayes_filter.BayesFilter()
    >>> test_bayes.bayes_classify('选择轮盘游戏随机赔率，高达119倍。')
    Building prefix dict from the default dictionary ...
    Loading model from cache /var/folders/md/0251yy51045d6nknpkbn6dc80000gn/T/jieba.cache
    Loading model cost 0.969 seconds.
    Prefix dict has been built succesfully.
    (1, [-52.665796469015774, -41.781387161169008, -53.513237457719043, -56.71342538342271])

我们一开始使用了`jieba <https://github.com/fxsjy/jieba>`_ 进行分词，上面的0.969秒是分词的时间，感谢fxsjy维护如此优秀的中文分词库。这里返回了一个tuple，包含bayes判断结果的类别1（所对应的是赌博），以及对应的所有类别的相对概率，现在支持的类别有四个，用户可以自行添加数据然后进行训练

- NORMAL = 0
- GAMBLE = 1
- SEX = 2
- POLITICE = 3


.. _`English Version`:
This project uses Native Bayes algorithm to detect spam content, like normal, sex, gamble, political content. We use 400 Chinese sentences to train the algorithm and the correct rate is about 93%. Right now we only support Chinese spam content classify :<

How to use:

.. code-block:: python

    >>> from bayes import bayes_filter
    >>> test_bayes = bayes_filter.BayesFilter()
    >>> test_bayes.bayes_classify('选择轮盘游戏随机赔率，高达119倍。')
    Building prefix dict from the default dictionary ...
    Loading model from cache /var/folders/md/0251yy51045d6nknpkbn6dc80000gn/T/jieba.cache
    Loading model cost 0.969 seconds.
    Prefix dict has been built succesfully.
    (1, [-52.665796469015774, -41.781387161169008, -53.513237457719043, -56.71342538342271])

- NORMAL = 0
- GAMBLE = 1
- SEX = 2
- POLITICE = 3
