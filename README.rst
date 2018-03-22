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
:Keywords: spam, filter, classify, native, bayes

.. _`中文版本`:

这个项目目的是使用机器学习／人工智能进行数据分类。

特点
------

- 内置预训练模型缓存，开箱即用。使用numpy库做矩阵计算，判断速度极快

- 准确率高，例子一使用1000个训练数据，多元分类下准确率达到96%，二元分类下准确率达到98%。

- 容易定制，只需要把需要分类的数据放在不同的文件用作训练，就能得到目标分类器，支持自定义以及分词算法

- 增加了混淆矩阵，以及错误数据分类输出模式，可以根据被错误分类的结果来调整分类数据

通过pip安装：
---------------

.. code-block:: bash

   pip install cherry

基本使用:
------------

.. code-block:: python

    >>> import cherry
    >>> result = cherry.classify('理查德费曼')
    Building prefix dict from the default dictionary ...
    Loading model from cache /var/folders/md/0251yy51045d6nknpkbn6dc80000gn/T/jieba.cache
    Loading model cost 1.172 seconds.
    Prefix dict has been built succesfully.
    >>> r.percentage
    >>> r.word_list

我们使用了 `jieba`_ 进行分词，上面的1.172秒是分词的时间（感谢fxsjy维护如此优秀的中文分词库）。结果返回的是一个Result对象，Result对象的percentage属性显示了对应数据每个类别的概率，word_list属性显示的是句子的有效部分（这里的有效部分根据分词函数来划分，中文的话，结巴分词结果中长度大于1，不在stop_word列表中，并且在其他训练数据中出现过这个词）的对划分类别的影响程度。在上面的例子中。

.. _`jieba`: https://github.com/fxsjy/jieba

测试
-------

  由于测试数据包含敏感内容，如果用户想进行测试，可以下载 `test_data`_ 然后放在'data/Chinese/data/'文件夹下面。
  
.. _`test_data`: https://drive.google.com/file/d/1eP_dWZnmjBrYcmCoPETSRzmmqCHBGUfZ/view?usp=sharing
  
git clone仓库之后运行

.. code-block:: bash

  >>> python runanalysis.py -h

  usage: runanalysis.py [-h] [-l LANGUAGE] [-s SPLIT] [-t TEST_TIME] [-n NUM]
                      [-d]

    Native bayes testing.

    optional arguments:
      -h, --help            show this help message and exit
      -l LANGUAGE, --language LANGUAGE
                            Which language's dataset we will use
      -s SPLIT, --split SPLIT
                            Split function to tokenizer data
      -t TEST_TIME, --test_time TEST_TIME
                            How many times we split data for testing
      -n NUM, --num NUM     How many test data we need every time
      -d                    Show wrong classified data

runanalysis是测试脚本，默认从数据中随机选取60个数据做为测试数据，剩下的数据用作训练数据。如果你需要进行10次训练和测试（大约需要2分钟），运行： 

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

得到混淆矩阵以及准确率，如上图。混淆矩阵可以了解哪些数据被错误分类了，如上图，大部分被错误分类的都是正常的数据，可以看到查准率非常高(118+2)/120=98%，不过查全率较低118/(3+8+11+118)=84%

.. code-block:: python

    percentage_list, word_list = test_bayes.bayes_classify(
        '美联储当天结束货币政策例会后发表声明说，
        自2017年12月以来，美国就业市场和经济活动继续保持稳健增长，
        失业率继续维持在低水平。')
    result = sorted(
        percentage_list, key=lambda x: x[1], reverse=True)[0][0]

第二个列表包含了输入句子中所有被分词的词语对应最高概率分类的概率，在这个例子里，这个列表中包含的是每个词语对句子被判断为normal.dat的影响度，可以看到，经济，美国，维持这三个词语的值最大，对句子的影响也最大。

注意事项
--------
- 输入句子需转换成简体中文

未来功能
--------

- 增加Adaboost算法

