## 中文文档

### 目录

- [特性](#特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [例子](#例子)
- [定制](#定制)
  - [数据集](#数据集)
  - [设置](#设置)
  - [训练](#训练)
  - [分类](#分类)
- [高级用法](advance-usage)
  - [效果](#效果)
  - [搜索](#搜索)
  - [可视化](#可视化)

### 特性

- **无需机器学习知识，开箱即用，定制简单**

    cherry 自带三个预训练模型，预训练模型分类只需一行代码。使用自己的数据集进行定制训练也只需要十行代码。同时 cherry 支持自定义分词算法，分类算法以及 stop words 词库。
    
- **高精确率，召回率**

    在小型数据集（4个类别 共 1000条 数据）中达到 96% 精确率以及召回率。在大型数据集中（9个类别 共 5万条 数据）达到 97% 精确率以及召回率。

- **支持多种自定义算法**

	定制模式下，支持 sklearn 中所有[特征工程函数](https://scikit-learn.org/stable/modules/feature_extraction.html)以及所有[分类器](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)。可以通过 `search()` 找出特定数据集的最优算法以及参数。
	
- **可视化**

    简易绘制学习曲线图像，判断模型是否过拟合或者欠拟合。


### 安装

    pip install cherry
    
### 快速开始

预训练模型包含 3个 数据集，分别是：

1. 垃圾 / 正常 (`model='spam'`, 包含约 5000条 英文邮件)
2. 赌博 / 色情 / 敏感 / 正常 (`model='harmful'`，包含约 1000条 中文句子)
3. 彩票 / 财经 / 房产 / 家居 / 科技 / 社会 / 体育 / 游戏 / 娱乐 (`model=news`，包含约 45000条 中文新闻)

使用预训练模型分类非常简单，只需要直接调用 `classify()` 方法，可以指定两个参数，`text` 是由需要分类的数据组成的列表，`models` 参数是 `spam`, `harmful`, `news` 三者之一。

	>>> res = cherry.classify(model='harmful', text=['她们对计算机很有热情，也希望学习到数据分析，网络爬虫，人工智能等方面的知识，从而运用在她们工作上'])
    >>> res.word_list
    [(2, '她们'), (1, '网络'), (1, '热情'), (1, '方面'), (1, '数据分析'), (1, '希望'), (1, '工作'), (1, '学习'), (1, '从而')]
    >>> res.probability
    # 返回结果分别对应 赌博，正常，政治，色情四个类别的概率
    array([[4.43336608e-03, 9.95215198e-01, 3.51419231e-04, 1.68657851e-08]])

返回的 `res` 对象包含 `word_list` 以及 `probability`。其中 `word_list` 包含待分类文本中前 20个 词语（按出现频率降序排列），`probability` 包含对应类别索引下的概率（索引与每一条训练数据最后的类别一致）

### 定制
有两种方法使用自定义数据集。第一，你可以直接把数据以及对应参数直接传给 `train()` API，具体方式请参考[训练](#训练)。第二，以文件类型进行训练，你的数据集需要包含

1. 数据集 `data.xxx`（需命名为 `data`开头，任意后缀的文件）
2. 停止词 `stop_words.xxx`（任意后缀，需命名为 `stop_words`开头，任意后缀的文件，停止词可直接拷贝 cherry 例子中自带的停止词）

#### 1. 数据集
1. 在 data 文件夹内新建 `your_data_name` 文件夹，你的模型所有数据以及生成的缓存都会存放在此文件夹中。
2. 把 `data.xxx` 以及 `stop_words.xxx` 放在文件夹中（可参考例子）。`your_data_name` 用作给 cherry 分辨不同的模型。
3. 数据集中每一行代表一条数据，数据结束后需要添加 ',' 以及对应的分类类别（不需要空格），例如：

    > 这是一条正常数据,0
    > 
    > 这是赌博相关数据,1

    cherry 会提取每一行数据最后的类别作为标签进行训练。

#### 2. 停止词



### 设置
在开始训练前，你可以自定义分词函数，cherry 默认使用 jieba 进行中文分词，你也可以使用其他第三方库或者自行实现。此函数接受输入文档，并返回分词后词语组成的列表。它位于 `base.py` 中的 `tokenizer()`

	# base.py
	
	def tokenizer(text):
    '''
    You can use your own tokenizer function here, by default,
    this function only work for chinese
    '''
        # For English:
        # from nltk.tokenize import word_tokenize
        # return [t.lower() for t in word_tokenize(text) if len(t) > 1]
        return [t for t in jieba.cut(text) if len(t) > 1]

### 训练

	>>> cherry.train(model='your_data_name')
	
训练就是那么简单，你也可以把测试数据传到 `train()` 函数进行训练，这里以 sklearn 中的 iris 数据集为例

    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> x_data, y_data = iris.data, iris.target
    >>> cherry.train(model='your_data_name', x_data=x_data, y_data=y_data)
    
注意，你依然需要新建 `your_data_name` 文件夹用来存放缓存文件，如果你熟悉 `sklearn`，你也可以自定义特征函数以及分类函数，具体使用方法可以参考 [API](#api)

### 分类
训练完之后，cherry 会在 `your_data_name` 下生成训练模型缓存，调用 `classify()` 就能直接使用模型进行分类了，

	>>> res = cherry.classify(model='harmful', text=['她们对计算机很有热情，也希望学习到数据分析，网络爬虫，人工智能等方面的知识，从而运用在她们工作上'])
    >>> res.word_list
    [(2, '她们'), (1, '网络'), (1, '热情'), (1, '方面'), (1, '数据分析'), (1, '希望'), (1, '工作'), (1, '学习'), (1, '从而')]
    >>> res.probability
    # 返回结果分别对应 赌博，正常，政治，色情四个类别的概率
    array([[4.43336608e-03, 9.95215198e-01, 3.51419231e-04, 1.68657851e-08]])

### 效果
`performance()` 方法能够计算出输入模型分成 `n_splits` 份后交叉检验得出的分数

	>>> cherry.performance(model='harmful', n_splits=5)
	
	              precision    recall  f1-score   support

           0       0.98      1.00      0.99        44
           1       0.96      0.88      0.92        52
           2       0.90      0.96      0.93        49
           3       1.00      1.00      1.00        45

    accuracy                           0.96       190
    macro avg       0.96      0.96     0.96       190
    weighted avg    0.96      0.96     0.96       190

### 可视化
使用 `display()` API 可以得出不同特征函数以及分类器下的学习曲线，以下以此为默认 `MNB, SGD, RandomForest` 方法的学习曲线图像

	>>> cherry.display(model='harmful', clf_method='SGD')
	
<img src="https://raw.githubusercontent.com/Windsooon/cherry/master/imgs/MNB.png" alt="" height="500">
<img src="https://raw.githubusercontent.com/Windsooon/cherry/master/imgs/SGD.png" alt="" height="500">
<img src="https://raw.githubusercontent.com/Windsooon/cherry/master/imgs/RandomForest.png" alt="" height="500">
	
### 搜索
通过把传人需要搜索的参数范围，可以得出最佳参数。

    >>> parameters = {
    ...     'clf__alpha': [0.1, 0.5, 1],
    ...     'clf__fit_prior': [True, False]
    ... }

    >>> cherry.search('harmful', parameters)
    
    score is 0.9199693815090905
    clf__alpha: 0.1
    clf__fit_prior: True
    

### API

**def train(model, vectorizer=None, vectorizer\_method=None, clf=None, clf\_method=None, x\_data=None, y\_data=None):**

- model (string)
    
    使用的训练模型，默认模型包括 `harmful`, `news` 和 `spam`

- vectorizer (sklearn object)

    特征函数，可以直接传入 sklearn 的 [特征函数](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text)，默认使用 `CountVectorizer()`，
 
   > 短文本建议使用 `CountVectorizer()`，长文本建议使用 `TfidfVectorizer()`，大型数据集可以使用 `HashingVectorizer()` 节省内存。
   
- vectorizer_method (string)

    cherry 支持使用缩写来设置特征函数（仅当 `vectorizer` 为 `None` 时会被调用），'Count' 会调用 `CountVectorizer(tokenizer=tokenizer, stop_words=get_stop_words(model))`，`tokenizer` 对应 `base.py` 中定义的分词函数。`Tfidf` 对应 `TfidfVectorizer`，`Hashing` 对应 `HashingVectorizer`。
    
- clf (sklearn object)

    分类函数，可以直接传入 sklearn 的 [分类函数](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)，默认使用 `MultinomialNB()`。
 
- clf_method (string)

    cherry 支持使用缩写来设置常用分类函数（包含常用参数）（仅当 `clf` 为 `None` 时会被调用），`MNB` 会调用 `MultinomialNB(alpha=0.1)`, `SGD` 对应 `SGDClassifier`, `RandomForest` 对应 `RandomForestClassifier`, `AdaBoost` 对应 `AdaBoostClassifier`。

- x_data (numpy array)

    `train()` 支持直接传入文本数据进行训练，`x_data` 包含全部训练文本。

- y_data (numpy array)

    `train()` 支持直接传入文本数据进行训练，`y_data` 包含全部训练文本对应的类别。
    

**def classify(text, model, N=20):**

- text (list)
    
    需要训练的文本列表内容
    
- model
    
    使用的训练模型，默认模型包括 `harmful`, `news` 和 `spam`
    
- N

	`word_list`属性 中 输出的词语数量，默认为 20
	
**返回对象的属性**

- word_list

    包含待分类文本中前 N个 词语
    

- probability

    输入文本的对应的每个类别的概率，例如
    
        array([[4.43336608e-03, 9.95215198e-01, 3.51419231e-04, 1.68657851e-08]])
	
**def performance(
        model, vectorizer=None, vectorizer\_method=None,
        clf=None, clf\_method=None, x\_data=None,
        y\_data=None, n\_splits=5, output='Stdout')**
        
- model (string)
    
    使用的训练模型，默认模型包括 `harmful`, `news` 和 `spam`

- vectorizer (sklearn object)

    特征函数，可以直接传入 sklearn 的 [特征函数](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text)，默认使用 `CountVectorizer()`，
 
   > 短文本建议使用 `CountVectorizer()`，长文本建议使用 `TfidfVectorizer()`，大型数据集可以使用 `HashingVectorizer()` 节省内存。
   
- vectorizer_method (string)

    cherry 支持使用缩写来设置特征函数（仅当 `vectorizer` 为 `None` 时会被调用），'Count' 会调用 `CountVectorizer(tokenizer=tokenizer, stop_words=get_stop_words(model))`，`tokenizer` 对应 `base.py` 中定义的分词函数。`Tfidf` 对应 `TfidfVectorizer`，`Hashing` 对应 `HashingVectorizer`。
    
- clf (sklearn object)

    分类函数，可以直接传入 sklearn 的 [分类函数](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)，默认使用 `MultinomialNB()`。
 
- clf_method (string)

    cherry 支持使用缩写来设置常用分类函数（包含常用参数）（仅当 `clf` 为 `None` 时会被调用），`MNB` 会调用 `MultinomialNB(alpha=0.1)`, `SGD` 对应 `SGDClassifier`, `RandomForest` 对应 `RandomForestClassifier`, `AdaBoost` 对应 `AdaBoostClassifier`。

- x_data (numpy array)

    `train()` 支持直接传入文本数据进行训练，`x_data` 包含全部训练文本。

- y_data (numpy array)

    `train()` 支持直接传入文本数据进行训练，`y_data` 包含全部训练文本对应的类别。
    
- n_splits (int)

    使用 Kfold 进行交叉检验时，K 的值。
    
- output (file path)

    输出结果，默认输出到终端，这里可以指定输出的文件名。
    
**def search(model, parameters, vectorizer=None, vectorizer\_method=None,
        clf=None, clf\_method=None, x\_data=None, y\_data=None, method='RandomizedSearchCV', cv=3, iid=False, n_jobs=1)**
        
- model (string)
    
    使用的训练模型，默认模型包括 `harmful`, `news` 和 `spam`

- vectorizer (sklearn object)

    特征函数，可以直接传入 sklearn 的 [特征函数](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text)，默认使用 `CountVectorizer()`，
 
   > 短文本建议使用 `CountVectorizer()`，长文本建议使用 `TfidfVectorizer()`，大型数据集可以使用 `HashingVectorizer()` 节省内存。
   
- vectorizer_method (string)

    cherry 支持使用缩写来设置特征函数（仅当 `vectorizer` 为 `None` 时会被调用），'Count' 会调用 `CountVectorizer(tokenizer=tokenizer, stop_words=get_stop_words(model))`，`tokenizer` 对应 `base.py` 中定义的分词函数。`Tfidf` 对应 `TfidfVectorizer`，`Hashing` 对应 `HashingVectorizer`。
    
- clf (sklearn object)

    分类函数，可以直接传入 sklearn 的 [分类函数](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)，默认使用 `MultinomialNB()`。
 
- clf_method (string)

    cherry 支持使用缩写来设置常用分类函数（包含常用参数）（仅当 `clf` 为 `None` 时会被调用），`MNB` 会调用 `MultinomialNB(alpha=0.1)`, `SGD` 对应 `SGDClassifier`, `RandomForest` 对应 `RandomForestClassifier`, `AdaBoost` 对应 `AdaBoostClassifier`。

- x_data (numpy array)

    `train()` 支持直接传入文本数据进行训练，`x_data` 包含全部训练文本。

- y_data (numpy array)

    `train()` 支持直接传入文本数据进行训练，`y_data` 包含全部训练文本对应的类别。
    
- method (string)

    可以选择 'RandomizedSearchCV' 或者 'GridSearchCV' 方法
    
- cv (int)

    使用 Kfold 进行交叉检验时，K 的值。
    
- iid （boolean)

    默认为 False，返回 Kfold 检验的平均分数，如果为 真，则在此基础上，按照不同类别样本数进行加权，详情可以参考[这里](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

- n_jobs (int)

	运行 `search()` 时使用的处理器数量


**def display(
        model, vectorizer=None, vectorizer_method=None,
        clf=None, clf_method=None, x_data=None, y_data=None)**

- model (string)
    
    使用的训练模型，默认模型包括 `harmful`, `news` 和 `spam`

- vectorizer (sklearn object)

    特征函数，可以直接传入 sklearn 的 [特征函数](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text)，默认使用 `CountVectorizer()`，
 
   > 短文本建议使用 `CountVectorizer()`，长文本建议使用 `TfidfVectorizer()`，大型数据集可以使用 `HashingVectorizer()` 节省内存。
   
- vectorizer_method (string)

    cherry 支持使用缩写来设置特征函数（仅当 `vectorizer` 为 `None` 时会被调用），'Count' 会调用 `CountVectorizer(tokenizer=tokenizer, stop_words=get_stop_words(model))`，`tokenizer` 对应 `base.py` 中定义的分词函数。`Tfidf` 对应 `TfidfVectorizer`，`Hashing` 对应 `HashingVectorizer`。
    
- clf (sklearn object)

    分类函数，可以直接传入 sklearn 的 [分类函数](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)，默认使用 `MultinomialNB()`。
 
- clf_method (string)

    cherry 支持使用缩写来设置常用分类函数（包含常用参数）（仅当 `clf` 为 `None` 时会被调用），`MNB` 会调用 `MultinomialNB(alpha=0.1)`, `SGD` 对应 `SGDClassifier`, `RandomForest` 对应 `RandomForestClassifier`, `AdaBoost` 对应 `AdaBoostClassifier`。

- x_data (numpy array)

    `train()` 支持直接传入文本数据进行训练，`x_data` 包含全部训练文本。

- y_data (numpy array)

    `train()` 支持直接传入文本数据进行训练，`y_data` 包含全部训练文本对应的类别。

	
### FAQ

TODO
