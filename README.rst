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


:Version: 0.4
:Download: https://pypi.python.org/pypi/cherry/
:Source: https://github.com/Sunkist-Cherry/cherry
:Support: >=Python3.3
:Keywords: native, bayes, classify

.. _`中文版本`:

cherry使用贝叶斯模型算法对文本进行分类（目前支持中英文），并提供混淆矩阵用作数据分析

特点
------

- 内置预训练模型缓存，开箱即用。使用numpy库做矩阵计算，判断速度极快。

- 准确率高，例子一使用1000个训练数据，多元分类（正常，赌博，色情，政治）下准确率达到96%，二元分类下准确率达到98%。

- 容易定制，只需要把需要分类的数据放在不同的文件用作训练，就能得到目标分类器，支持自定义分词算法。

- 新增测试功能，能够获取测试后的混淆矩阵，以及被错误分类的结果。

安装
--------

.. code-block:: bash

   pip install cherry

基本使用
------------

.. code-block:: python

    >>> import cherry
    >>> result = cherry.classify('她们对计算机很有热情，也希望学习到数据分析，网络爬虫，人工智能等方面的知识，从而运用在她们工作上')
    Building prefix dict from the default dictionary ...
    Loading model from cache /var/folders/md/0251yy51045d6nknpkbn6dc80000gn/T/jieba.cache
    Loading model cost 0.899 seconds.
    Prefix dict has been built succesfully.
    >>> result.percentage
    [('normal.dat', 0.837), ('politics.dat', 0.108), ('gamble.dat', 0.053), ('sex.dat', 0.002)]
    >>> result.word_list
    [('工作', 7.0784042046861373), ('学习', 4.2613376272953198), ('方面', 3.795076414904381), ('希望', 2.1552995125795613), ('人工智能', 1.1353997980863895), ('网络', 0.41148095885968772), ('从而', 0.27235358073104443), ('数据分析', 0.036787509418279463), ('热情', 0.036787509418278574), ('她们', -4.660672209426675)]

默认中文使用 `jieba`_ 分词，上面的0.899秒是它载入模型的时间（感谢fxsjy维护如此优秀的中文分词库）。结果返回的是一个Result对象，Result的percentage属性显示了对应数据每个类别的概率，正常句子的概率为83.7%，政治敏感的概率为10.8%，赌博的概率为5%，色情的概率为0.2%。

.. code-block:: bash

    [('normal.dat', 0.837), ('politics.dat', 0.108), ('gamble.dat', 0.053), ('sex.dat', 0.002)]
    
result的word_list属性显示的是句子的有效部分（这里的有效部分**根据分词函数划分**，中文默认情况下，要求在结巴分词结果中词语长度大于1，不在stop_word列表中，并且在其他训练数据中出现过这个词）对划分类别的影响程度。
    
.. code-block:: bash

    [('工作', 7.0784042046861373), ('学习', 4.2613376272953198), ('方面', 3.795076414904381), ('希望', 2.1552995125795613), ('人工智能', 1.1353997980863895), ('网络', 0.41148095885968772), ('从而', 0.27235358073104443), ('数据分析', 0.036787509418279463), ('热情', 0.036787509418278574), ('她们', -4.660672209426675)]

在上面的例子中。句子被正确分类成正常类别。影响度最大的词语分别是“工作”，“学习”，“方面”

英文的话只需要加上语言参数：（训练数据为正常邮件以及垃圾邮件）

.. code-block:: python

    >>> result = cherry.classify(lan='English', text='Yeah, I got one of Tumblr’s you-may-have-unwittingly-interacted-with-propaganda-blogs emails too. And like everyone else, I kind of shrugged because really, what am I supposed to do about that now')
    >>> result.percentage
    >>> [('ham', 0.795), ('spam', 0.205)]
    >>> result.word_list
    [('everyone', 0.85148562324955179), ('like', 0.82516831493217779), ('kind', 0.65081492778740113), ('got', 0.53303189213101732), ('else', 0.53303189213101732), ('one', 0.19882980404434303), ('now', -0.38717273906427518), ('emails', -1.364088092754864)]


.. _`jieba`: https://github.com/fxsjy/jieba


定制
-------

默认支持中英文分类，如果需要支持其他语言，可以参考config.py中的LAN_DICT，每个语言接受3个参数，

- dir

  + 数据是否单独存放在一个文件中（参加英文数据/data/English/data/）还是存放在同一个文件（参照中文数据/data/Chinese/data/）

    
.. code-block:: bash

    .
    ├── Chinese
    │   ├── cache
    │   │   ├── classify.cache
    │   │   ├── vector.cache
    │   │   └── vocab_list.cache
    │   ├── data
    │   │   ├── gamble.dat
    │   │   ├── normal.dat
    │   │   ├── politics.dat
    │   │   └── sex.dat
    │   └── stop_word.dat
    └── English
        ├── cache
        │   ├── classify.cache
        │   ├── vector.cache
        │   └── vocab_list.cache
        ├── data
        │   ├── ham
        │   │   ├── 0001.1999-12-10.farmer.ham.txt
        │   │   ├── 0002.1999-12-13.farmer.ham.txt
        │   ├── spam
        │   │   ├── 0003.1999-12-10.farmer.ham.txt
        │   │   ├── 0004.1999-12-13.farmer.ham.txt

- type

  + 数据文件后缀，例如.dat，.txt。

- split

  + 分词函数，需要返回一个列表，包含分词后的每个词语，并添加在config文件中。

测试
-------

  由于测试数据包含敏感内容，如果用户想进行测试，可以通过Google dirve下载 `test_data`_ 然后放在对应语言文件夹。
  
.. _`test_data`: https://drive.google.com/file/d/1OtbY7RCjkoQWYb0fHIOTBcJfgDlW5Tjz/view?usp=sharing
  
git clone仓库之后运行

.. code-block:: bash

  >>> python runanalysis.py -h

  usage: runanalysis.py [-h] [-l LANGUAGE] [-t TEST_TIME] [-n NUM] [-d]

    Native bayes testing.

    optional arguments:
      -h, --help            show this help message and exit
      -l LANGUAGE, --language LANGUAGE
                            Which language's dataset we will use
      -t TEST_TIME, --test_time TEST_TIME
                            How many times we split data for testing
      -n NUM, --num NUM     How many test data we need every time
      -d                    Show wrong classified data

runanalysis.py是测试脚本，默认从中文数据中随机选取60个数据做为测试数据，剩下的数据用作训练数据。重复10次：

.. code-block:: bash

  >>> python runanalysis.py -t 10

  +Cherry---------------+------------+---------+------------+--------------+
  | Confusion matrix    | gamble.dat | sex.dat | normal.dat | politics.dat |
  +---------------------+------------+---------+------------+--------------+
  | (Real)gamble.dat    |        141 |       0 |          0 |            0 |
  | (Real)sex.dat       |          0 |     165 |          0 |            0 |
  | (Real)normal.dat    |          3 |       8 |        118 |           11 |
  | (Real)politics.dat  |          0 |       0 |          2 |          152 |
  | Error rate is 4.00% |            |         |            |              |
  +---------------------+------------+---------+------------+--------------+

得到混淆矩阵以及准确率，如上图。混淆矩阵可以了解哪些数据被错误分类了，如上图，大部分被错误分类的都是正常的数据。如果把正常类别看成阳性，可以看到

查准率(precision)为

.. math::

    (118+2) / 120= 98%

查全率(recall)

.. math::

    118 / (3+8+11+118)= 84%

注意事项
--------
- 输入句子需转换成简体中文

未来功能
--------

- 增加Adaboost算法

