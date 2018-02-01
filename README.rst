cherry
=======================
.. image:: https://api.travis-ci.org/Sunkist-Cherry/cherry.png?branch=master
    :target: https://travis-ci.org/repositories/Sunkist-Cherry/cherry

.. image:: https://img.shields.io/pypi/v/cherry.svg
    :target: https://pypi.python.org/pypi/cherry

.. image:: https://img.shields.io/pypi/l/cherry.svg
    :target: https://pypi.python.org/pypi/cherry

.. image:: https://img.shields.io/pypi/pyversions/cherry.svg
    :target: https://pypi.python.org/pypi/cherry


:Version: 0.1.8
:Download: https://pypi.python.org/pypi/cherry/
:Source: https://github.com/Sunkist-Cherry/cherry
:Support: >=Python3.4
:Keywords: spam, filter, python, native, bayes

.. _`中文版本`:
这个项目目的是使用机器学习／人工智能来进行数据分类

例子1中的应用是判别垃圾内容，现阶段用户输入句子会先经过分词，然后通过朴素贝叶斯模型判别成正常，色情，赌博，政治敏感四个类别。现在每个类别各使用了100个训练数据，辨别准确率大约为93%。（数据內容请勿分發，传阅，出售，出租给他人）

特点
----
- 开箱即用，快速上手

  内置预训练模型以及文件缓存，开箱即用。同时使用numpy库做矩阵计算，判断速度非常快
- 准确率高

  现阶段使用了400个训练数据，准确率达到93.5%。下载后可以通过运行

  .. code-block:: bash

    python -m unittest tests.test_error_rate

  得到准确率测试结果

  .. code-block:: bash

    This may takes some time
    Completed 0 tasks, 20 tasks left.
    Completed 5 tasks, 15 tasks left.
    Completed 10 tasks, 10 tasks left.
    Completed 15 tasks, 5 tasks left.
    The error rate is 6.42%
     
    测试20次，每次从数据集随机取出20个数据作为测试数据，剩下的作为训练数据。然后计算平均错误率

- 可定制

  自己可以添加修改数据源，增加训练正确率

通过pip安装：
-----------

.. code-block:: bash

   pip install cherry

基本使用:
--------

.. code-block:: python

    >>> from classify import bayes
    >>> test_bayes = bayes.Classify()
    >>> test_bayes.bayes_classify(
        '美联储当天结束货币政策例会后发表声明说，
        自2017年12月以来，美国就业市场和经济活动
        继续保持稳健增长，失业率继续维持在低水平。')
    Building prefix dict from the default dictionary ...
    Loading model from cache /var/folders/md/0251yy51045d6nknpkbn6dc80000gn/T/jieba.cache
    Loading model cost 1.172 seconds.
    Prefix dict has been built succesfully.
    (
        [
            ('gamble.dat', 0.16622423300308523), ('normal.dat', 0.45184431202182884),
            ('politics.dat', 0.20543346471119367), ('sex.dat', 0.17649799026389221)
        ], 
        [
            ('发表声明', 1.4632451832569382), ('12', 0.076950822137048291),
            ('维持', 3.5426867249367744), ('经济', 4.1229218000749324),
            ('继续', 1.7757620767067532), ('活动', 1.750927255708719),
            ('结束', 0.36463289458882819), ('以来', -0.14619272917716231),
            ('保持', -1.3093435389828434), ('增长', 1.4632451832569382),
            ('2017', 1.4632451832569382), ('市场', 1.9864933270214866),
            ('美国', 5.8843422794122686), ('当天', 1.5810282189133229)
        ]
    )
我们使用了 `jieba`_ 进行分词，上面的1.172秒是分词的时间（感谢fxsjy维护如此优秀的中文分词库）。结果返回的是一个tuple，里面包含了两个列表，第一个列表包含的是各个类别的概率，如果要获取最高概率的类别可以用sorted函数

.. _`jieba`: https://github.com/fxsjy/jieba



.. code-block:: python

    percentage_list, word_list = test_bayes.bayes_classify(
        '美联储当天结束货币政策例会后发表声明说，
        自2017年12月以来，美国就业市场和经济活动继续保持稳健增长，
        失业率继续维持在低水平。')
    result = sorted(
        percentage_list, key=lambda x: x[1], reverse=True)[0][0]

第二个列表包含了输入句子中所有被分词的词语对应最高概率分类的概率，在这个例子里，这个列表中包含的是每个词语对句子被判断为normal.dat的影响度，可以看到，经济，美国，维持这三个词语的值最大，对句子的影响也最大。
    
    
默认使用内置的训练模型缓存，如果你修改了数据源的话，需要更新缓存

.. code-block:: python

    >>> from classify import bayes
    >>> test_bayes = bayes.Classify(cache=False) # 缓存文件被更新
    >>> test_bayes = bayes.Classify(
        '美联储当天结束货币政策例会后发表声明说，自2017年12月以来，
        美国就业市场和经济活动继续保持稳健增长，
        失业率继续维持在低水平。') # 将使用新数据源的缓存

未来功能
-----

- 添加英文句子分类功能
- 繁体字转换成简体字再训练
- 把中文分词库分离，让用户可以自己选择分词方式
- 对长文本增加tf-idf计算词权重
- 增加SVM分类算法
- 增加HMM算法


.. _`english-version`:
This project uses Native Bayes algorithm to detect spam content, like normal, sex, gamble, political content. We use 400 Chinese sentences to train the algorithm and the correct rate is about 93.5%. Right now we only support Chinese spam content classify :<

How to use:

.. code-block:: python

    >>> from classify import bayes
    >>> test_bayes = bayes.Classify()
    >>> test_bayes.bayes_classify('美联储当天结束货币政策例会后发表声明说，自2017年12月以来，美国就业市场和经济活动继续保持稳健增长，失业率继续维持在低水平。')
    Building prefix dict from the default dictionary ...
    Loading model from cache /var/folders/md/0251yy51045d6nknpkbn6dc80000gn/T/jieba.cache
    Loading model cost 1.172 seconds.
    Prefix dict has been built succesfully.
    (
        [
            ('gamble.dat', 0.16622423300308523), ('normal.dat', 0.45184431202182884),
            ('politics.dat', 0.20543346471119367), ('sex.dat', 0.17649799026389221)
        ], 
        [
            ('发表声明', 1.4632451832569382), ('12', 0.076950822137048291),
            ('维持', 3.5426867249367744), ('经济', 4.1229218000749324),
            ('继续', 1.7757620767067532), ('活动', 1.750927255708719),
            ('结束', 0.36463289458882819), ('以来', -0.14619272917716231),
            ('保持', -1.3093435389828434), ('增长', 1.4632451832569382),
            ('2017', 1.4632451832569382), ('市场', 1.9864933270214866),
            ('美国', 5.8843422794122686), ('当天', 1.5810282189133229)
        ]
    )
