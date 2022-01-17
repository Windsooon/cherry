# Cherry

![logo](https://github.com/Windsooon/cherry/blob/master/imgs/text.png?raw=true)

[![image](https://travis-ci.org/Windsooon/cherry.svg?branch=master)](https://travis-ci.org/Windsooon/cherry)
[![image](https://img.shields.io/pypi/v/cherry.svg)](https://pypi.python.org/pypi/cherry)
[![image](https://img.shields.io/pypi/l/cherry.svg)](https://pypi.python.org/pypi/cherry)
[![image](https://img.shields.io/pypi/pyversions/cherry.svg)](https://pypi.python.org/pypi/cherry)

## Cherry - Text classification with no machine learning knowledge needed

|  Cherry   | Windson  |
|  ----     | ----  |
| Download  | https://pypi.python.org/pypi/cherry |
| Source    | https://github.com/Windsooon/cherry |
| Keywords  | machine learning, text classification |


## Document

- [Feature](#feature)
- [Requirements](#requirements)
- [Install](#install)
- [Built in model](#built-in-model)
- [Quickstart](#quick-start)
- [Example](#example)
- [API](#api)
  - [Performance](#performance)
  - [Search](#search)
  - [Display](#display)

## Feature

#### Text classification in five minutes

Even though you had never learned about machine learning. You can use Cherry to train your text classification model in 5 minutes with over 80% accuracy (check out the example models). Cherry also provides extra features for users who want to improve their model.

#### Easy to optimize and optimize performance

  Cherry provide [performence()](#performance) and [display()](#display) api to help you debug and improve your model.

## Requirements

    - Python (above 3.6)

## Installation

Install using `pip`
    
    pip install cherry
    # Cherry use nltk for text tokenizer 
    pip install nltk
    # After install nltk, You need to download punkt
    >>> import nltk
    >>> nltk.download('punkt')

or clone the project from github.

    git clone git@github.com:Windsooon/cherry.git

## Built in model

Cherry has three built in models,`newsgroups`,  `review` and `email`:

- [The 20 Newsgroups dataset](http://qwone.com/~jason/20Newsgroups/)
    
    These datasets contain 11,315 news. they were organized into 20 different newsgroups, each corresponding to one of the below topic:

    - alt.atheism, comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware
    - comp.sys.mac.hardware, comp.windows.x, misc.forsale, rec.autos
    - rec.motorcycles, rec.sport.baseball, rec.sport.hockey, sci.crypt
    - sci.electronics, sci.med, sci.space, soc.religion.christian
    - talk.politics.guns, talk.politics.mideast, talk.politics.misc, talk.religion.misc
        
- [Comics & Graphic book review](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home)

    These datasets contain 108,463 reviews from the Goodreads book review website, Every book review also has rating from 0 point to 5 points.
	   
- [SMS Spam Collection](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)

    These datasets contain 5,578 SMS messages manually extracted from the Grumbletext Web site and randomly chosen ham messages of the NUS SMS Corpus (NSC).

## Quick Start

### Use built-in model

In the [Comics & Graphic book review](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) datasets, every book review also has rating from 0 point to 5 points. If you want to predict rating based on the book review:

> This is an extremely entertaining and often insightful collection by Nobel physicist Richard Feynman drawn from slices of his life experiences. Some might believe that the telling of a physicist’s life would be droll fare for anyone other than a fellow scientist, but in this instance, nothing could be further from the truth.

After finish [Installation](#installation), in your project path run `cherry.train('review')`
    
    # You only need to run this line of code at the first time.
    # This line of code will:
    # 1. Download `review` datasets from remote server (User in China may need use VPN)
    # 2. Train datasets using default settings ([Countvectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) and [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html))
    >>> cherry.train('review')

Then you can use `classify()` to predict the rating now.

    >>> res = cherry.classify('review', text='This is an extremely entertaining and often
    insightful collection by Nobel physicist Richard Feynman drawn from slices of his life
    experiences. Some might believe that the telling of a physicist’s life would be droll
    fare for anyone other than a fellow scientist, but in this instance, nothing could be
    further from the truth.')

The return `res` is a Classify object has two built-in method. `get_probability()` will return an array contains the probability of each category. The order of the return array depend on category name, in this case would be 0, 1, 2, 3, 4. We can see that there is 99.63% (9.96313288e-01) this review will be classify as 4 point.

    # The probability of this review had been rating as 4 points is 99.6%
	>>> res.get_probability()
	array([[6.99908424e-11, 2.48677319e-11, 6.17978214e-06, 3.39472694e-03,
        9.96313288e-01, 2.85805135e-04]])
	
Another method `get_word_list()` return a list that contains words that Cherry use for classifying. 

    >>> res.get_word_list()
    [[(2, 'physicist'), (2, 'life'), (1, 'truth'), (1, 'telling'), (1, 'slices'), (1, 'scientist'), (1, 'richard'), (1, 'nobel'), (1, 'instance'), (1, 'insightful'), (1, 'feynman'), (1, 'fellow'), (1, 'fare'), (1, 'extremely'), (1, 'experiences'), (1, 'entertaining'), (1, 'droll'), (1, 'drawn'), (1, 'collection'), (1, 'believe')]]

As you can see, some of the words in the review didin't show up here. There are two reasons for this 1) The training data didn't contain that word. For instance, The word `Backend` and `Engineer` never show up in training data. So the model don't know how to classify these words. 2) the word is a [stop word](https://en.wikipedia.org/wiki/Stop_words).

### Use your own dataset
Create a folder `your_model_name `under datasets in project path like this:

    ├── project path
    │   ├── datasets
    |   │   ├── your_model_name
    |   │   │   ├── category1
    |   |   │     ├── file_1
    |   |   │     ├── file_2
    |   |   │     ├── …
    |   │   │   ├── category2
    |   |   │     ├── file_10
    |   |   │     ├── file_11
    |   |   │     ├── …

Train you dataset:

    # By default, encoding will be utf-8,
    # You only need to run `train` at the first time
    >>> cherry.train('your_model_name', encoding='your_encoding')
    # Classify text, `text` can be a list of text too.
    >>> res = cherry.classify('your_model_name', text='text to be classified')

## Example
    
Let's build an email classifier from sketch, cherry will use this model to predict an email is spam or not.

### Project setup

    mkdir tutorial
    cd tutorial

    # Create a virtual environment to isolate our package dependencies locally
    python3 -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`

    # Install cherry and nltk
    pip install cherry
    pip install nltk
    >>> import nltk
    >>> nltk.download('punkt')

    # Create a new folder for email dataset
    mkdir -p datasets/email_tutorial

### Prepare dataset

1. Download the datasets from [SMS Spam Collection v. 1](http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/) then unzip it and put it inside `tutorial/datasets/email_tutorial` folder, now you got a file named `SMSSpamCollection.txt` which contains lots of emails.
2. Create a folder name `ham` and `spam` inside `email_tutorial` dir.
3. Create a script `email.py` in the same folder using code below to extract the email content and group them by category. every file would only contain text.

        import os
        import json

        ham_counter = 0
        spam_counter = 0

        with open('SMSSpamCollection.txt', 'r') as f:
            for line in f.readlines():
                if line.startswith('ham'):
                    ham_counter += 1
                    with open(os.path.join('ham', str(ham_counter)), 'w') as nf:
                            _, text = line.split('ham', 1)
                            nf.write(text.strip())
                else:
                    spam_counter += 1
                    with open(os.path.join('spam', str(spam_counter)), 'w') as nf:
                            _, text = line.split('spam', 1)
                            nf.write(text.strip())

4. Now your folder structure should look like this:

        tutorial
           ├── dataset
           │   ├── email_tutorial
           |   |   ├── email.py
           |   |   ├── SMSSpamCollection.txt
           │   │   ├── ham
           │   │   ├── spam
 
5. Run `python email.py`
6. Delete `SMSSpamCollection.txt` and `email.py`
7. Back to the path of `tutorial`, Like `cd path_to/tutorial`
6. Train the email model:
        
        >>> import cherry
        >>> cherry.train('email_tutorial', encoding='latin1')

7. Inside `email_tutorial` folder you can find `clf.pkz`, `ve.pkz`, `email_tutorial.pkz` which Cherry will use them for classify later.

         >>> res = cherry.classify('email_tutorial', 'Thank you for your interest in cherry! We wanted to let you'
              'know we received your application for Backend Engineer, and we are delighted that you'
              'would consider joining our team.')
         # 99.9% is a ham email
         >>> res.get_probability()
         array([[9.99985571e-01, 1.44288379e-05]])
         >>> res.get_word_list()
         [[(1, 'wanted'), (1, 'thank'), (1, 'team'), (1, 'received'), (1, 'let'),
         (1, 'joining'), (1, 'consider'), (1, 'application')]]

8. If you want to know good your model did, you can use [performance()]() which will use k-fold cross validation (By default, K equals to 10):
   
 		 >>> res = cherry.performance('email_tutorial', encoding='latin1', output='files')
         >>> res.get_score()
 		  
   The report will be save in `report` files, you can find the precision, recall, and f1-score.

                     precision    recall  f1-score   support

               0       0.99      1.00      0.99       485
               1       0.97      0.95      0.96        73

        accuracy                           0.99       558
       macro avg       0.98      0.97      0.98       558
    weighted avg       0.99      0.99      0.99       558
 	
   If you want to know which text had been clasiify wrong:
 	
 	     >>> res = cherry.performance('email_tutorial', encoding='latin1')
         >>> res.get_score()
         Text: Dhoni have luck to win some big title.so we will win:) has been classified as: 1 should be: 0
         Text: Back 2 work 2morro half term over! Can U C me 2nite 4 some sexy passion B4 I have 2 go back? Chat NOW 09099726481 Luv DENA Calls Â£1/minMobsmoreLKPOBOX177HP51FL has been classified as: 0 should be: 1
         Text: Latest News! Police station toilet stolen, cops have nothing to go on! has been classified as: 0 should be: 1
         ...

9. To display the graph, you can use

        >>> res.display('email_tutorial', encoding='latin1')

   ![img](https://github.com/Windsooon/cherry/blob/master/imgs/display.png?raw=true)

10. If you want to improve your model, you can use search method. 

        >>> parameters = {'clf__alpha': [0.1, 0.5, 1],'clf__fit_prior': [True, False]}
        >>> cherry.search('email_tutorial', parameters)
 
### API

#### def train(model, language='English', preprocessing=None, categories=None, encoding='utf-8', vectorizer=None, vectorizer\_method='Count', clf=None, clf\_method='MNB', x_data=None, y_data=None)

- model (String)
    
    The name of the model, you can use build-in models `email`, `review` and `newsgroups`, or pass
    the folder name of your dataset.

- language (String)

    The language of the training dataset. Cherry supports `English` and `Chinese`.

- preprocessing (function)

    The function will be called once for every input data before training.

- categories (List)

    Specify the training directory, for instance ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc'].

- encoding (String)
    
    The encoding of the dataset.

- vectorizer (Sklearn object)

    Feature extraction function use to convert the data into vertcor，by default is `CountVectorizer()`.
    you can pass [different feature extraction function](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_extraction.text) from Sklearn.
 
   > For some long texts you can use `TfidfVectorizer()`，If you need to save memory you can use  `HashingVectorizer()`, (get\_word\_list() function wouldn't work at this case)
   
- vectorizer_method (String)

    Cherry supports shortcut to set up feature extraction function when `vectorizer` is `None`.
    `Count` corresponds to `CountVectorizer(tokenizer=tokenizer, stop_words=get_stop_words(model))`,
    `Tfidf` corresponds to `TfidfVectorizer` and `Hashing` corresponds to `HashingVectorizer`.
    
- clf (Sklearn object)

    Classify function, by default is `MultinomialNB()`. You can pass [classify function](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html) from Sklearn.
 
- clf_method (String)

    Cherry supports shortcut to set up classify function when `clf` is `None`,
    `MNB` corresponds to `MultinomialNB(alpha=0.1)`,
    `SGD` corresponds to `SGDClassifier`,
    `RandomForest` corresponds to `RandomForestClassifier`,
    `AdaBoost` corresponds to `AdaBoostClassifier`.

- x_data (numpy array)

    training text data, if `x_data` and `y_data` is None, cherry will try to find the text files data in `model`

- y_data (numpy array)

    correspond labels data, if `x_data` and `y_data` is None, cherry will try to find the text files data in `model`

#### def classify(model, text)

- model (String)

    The name of the model, you can use build-in models `email`, `review` and `newsgroups`, or pass
    the folder name of your dataset.

- text (List / String)

    the text to be classify.

#### def performance(model, language='English', preprocessing=None, categories=None, encoding='utf-8', vectorizer=None, vectorizer\_method='Count', clf=None, clf\_method='MNB', x\_data=None, y\_data=None, n\_splits=10, output='Stdout')

Just as same as `train()` API

- n_splits (Integer)

    number of folds. Must be at least 2.

- output ('Stdout' or 'Files')

    'Stdout' will print the scores to standerd output and 'Files' will store the scores into a local file named 'report'.

#### def search(model, parameters, language='English', preprocessing=None, categories=None, encoding='utf-8', vectorizer=None, vectorizer\_method='Count', clf=None, clf\_method='MNB', x\_data=None, y\_data=None, method='RandomizedSearchCV', cv=3, n\_jobs=-1):

##### TODO

#### def display(model, language='English', preprocessing=None, categories=None, encoding='utf-8', vectorizer=None, vectorizer\_method='Count', clf=None, clf\_method='MNB', x\_data=None, y\_data=None)

Just as same as `train()` API

### Tests

    >>> python runtests.py
