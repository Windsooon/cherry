Spam Filter
=======================

.. image:: https://img.shields.io/pypi/v/spam-filter.svg
    :target: https://pypi.python.org/pypi/spam-filter

.. image:: https://img.shields.io/pypi/l/spam-filter.svg
    :target: https://pypi.python.org/pypi/spam-filter

.. image:: https://img.shields.io/pypi/pyversions/spam-filter.svg
    :target: https://pypi.python.org/pypi/spam-filter


:Version: 0.1.6
:Download: https://pypi.python.org/pypi/spam-filter/
:Source: https://github.com/Windsooon/Spam-Filter
:Keywords: spam, filter, python, native, bayes

中文版本_

English_Version_

.. _`中文版本`:
这个项目目的是使用机器学习／人工智能来判别垃圾内容，现阶段用户输入句子会先经过分词，然后通过朴素贝叶斯模型判别成正常，色情，赌博，政治敏感四个类别。现在每个类别各使用了100个训练数据，正确率大约93%


特点
----
- 开箱即用，快速上手

  内置预训练模型以及文件缓存，开箱即用。同时使用numpy库做矩阵计算，判断速度非常快
- 准确率高

  现阶段使用了400个训练数据，正确率达到93%
  下载后可以通过运行

  .. code-block:: bash

    python -m unittest tests.test_bayes

  得到测试结果

  .. code-block:: bash

    This may takes some time
    Completed 0 tasks, 20 tasks left.
    Completed 5 tasks, 15 tasks left.
    Completed 10 tasks, 10 tasks left.
    Completed 15 tasks, 5 tasks left.
    The error rate is 6.83%
     
    测试20次，每次从数据集随机取出20个数据作为测试数据，剩下的作为训练数据。然后计算平均错误率

- 可定制

  自己可以添加修改数据源，增加训练正确率

通过pip安装：
-----------

.. code-block:: bash

   pip install spam-filter

基本使用:
--------

.. code-block:: python

    >>> from filter import spam_filter
    >>> test_bayes = spam_filter.Filter()
    >>> test_bayes.bayes_classify('选择轮盘游戏随机赔率，高达119倍。')
    Building prefix dict from the default dictionary ...
    Loading model from cache /var/folders/md/0251yy51045d6nknpkbn6dc80000gn/T/jieba.cache
    Loading model cost 0.969 seconds.
    Prefix dict has been built succesfully.
    (1, [-52.665796469015774, -41.781387161169008, -53.513237457719043, -56.71342538342271])

这里使用了内置的训练模型缓存，如果你修改了数据源的话，需要更新缓存

.. code-block:: python

    >>> from filter import spam_filter
    >>> test_bayes = spam_filter.Filter(cache=False) # 缓存文件被更新
    >>> test_bayes = spam_filter.Filter() # 将使用新数据源的缓存


我们一开始使用了 `jieba`_ 进行分词，上面的0.969秒是分词的时间（感谢fxsjy维护如此优秀的中文分词库）。返回了一个tuple，包含bayes判断结果的类别1（所对应的是赌博），以及对应的所有类别的相对概率，现在支持的类别有四个，用户可以自行添加数据然后进行训练

.. _`jieba`: https://github.com/fxsjy/jieba

- NORMAL = 0
- GAMBLE = 1
- SEX = 2
- POLITICE = 3


TO DO
-----

- 添加英文句子分类功能
- 把中文分词库分离，让用户可以自己选择分词方式
- 对长文本增加tf-idf计算词权重
- 增加SVM分类算法


.. _`english-version`:
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
