3.1 加载20新闻组数据集
--------------

这些文件保存在对象的属性`data`中，执行`tweety.data`命令这些数据就会被加载进来，同样引用文件名称`filenames`也具有同样的效果，如

    >>> len(twenty_train.data)
    2257
    >>> len(twenty_train.filenames)
    2257

现在，让我们看看第一个被加载进的文件的第一行内容：

    >>> print("\n".join(twenty_train.data[0].split("\n")[:3]))
    From: sd345@city.ac.uk (Michael Collier)
    Subject: Converting images to HP LaserJet III?
    Nntp-Posting-Host: hampton
    >>> print(twenty_train.target_names[twenty_train.target[0]])
    comp.graphics
监督学习算法要求在训练集中的每篇文档都有对应的类别号。在`20newsgroups`案例中，类别标号是新闻组 `newsgroup`的名称，同时，它还与存放该文件的文件名称一样。
出于计算运算速率和空间开销的考虑，`scikit-learn`以整数数组的形式加载目标属性，该数组中元素值与`target_names`列表中类标名称的索引是一一对应的关系。每一个样本的类标都储存在`tweety_train`对象的`target`属性中。

    >>> twenty_train.target[:10]
    array([1, 1, 3, 3, 3, 3, 3, 2, 2, 2])
我们通过类标号也可以反向获取类别名称：

    >>> for t in twenty_train.target[:10]:
    ... print(twenty_train.target_names[t])
    ...
    comp.graphics
    comp.graphics
    soc.religion.christian
    soc.religion.christian
    soc.religion.christian
    soc.religion.christian
    soc.religion.christian
    sci.med
    sci.med
    sci.med
你可能注意到了，样本已经被随机混洗（`shuffle`）了（伪随机种子方式），这极大地方便了你选择第一个样本集进行快速地训练数据模型，同时也方便了在未使用所有数据集训练模型情况下对数据模型总体的认识。

3.2 从文本文件中析取特征变量
================

为了使机器学习能在文本文档上工作，我们必须把文本内容转换成数值特征向量。
## 3.2.1 词袋模型 ##
词袋模型最直观表现：
1. 只要训练数据集的任意文档中任意单词出现，就为其分配一个固定值。
2. 对于每一篇文档 `#i`，统计每个单词`W`出现的次数，并保存在`X[i, j]`，作为特征 `#j`的值，`j`为单词`W`在字典中索引。

词袋中的`n_features`是语料库中不同单词的个数，该数值常常大于100,000。
如果`n_samples`=10000,那么以`float32`类型储存`X`数组需要4GB的内存空间，这在现代计算机上是不可忍受的。
幸运的是，在`X`中大部分值是0，因为对于给定的任意一篇文档所使用的不同词都不会高于上千个。鉴于这个原因，我们有把握说，词袋是高维稀疏数据集，我们可以仅仅储存向量中非0特征值。
`scipy.sparse` 就是处理高维稀疏矩阵的非常好的数据结构，并且`scikit-learn`也有对此结构的内建支持。
## 3.2.2使用scikit-learn标记文本 ##
高水平组件的操作包含了文本处理、标记和过滤停用词等操作，这使得很容易构建一个特征字典及把文本转成特征向量。

    >>> from sklearn.feature_extraction.text import CountVectorizer
    >>> count_vect = CountVectorizer()
    >>> X_train_counts = count_vect.fit_transform(twenty_train.data)
    >>> X_train_counts.shape
    (2257, 35788)
CountVectorizer支持对使用N元分词语法分成词数的统计（N-grams）或对序列字符数计数。一旦对模型拟合后，我们便成功地便构建了特征字典的索引切片。

    >>> count_vect.vocabulary_.get(u'algorithm')
    4690
词汇表单词的索引值与该词在整个训练语料库中出现频数相关。

## 3.2.3 从出现与否到出现频数 ##

对单词作出现与否的统计是一件很好的开始，但是这存在一个问题：对应一篇更长的文档词出现次数的均值总比短文档要高，即便是在讨论同一个话题。
为了避免潜在的差异性，我们统计每个单词出现次数与语料库文档中单词总个数之比，便会生成一个新的特征变量，即频繁项集：`tf`。
还需要考虑的问题是，对于一个很高的`tf`值我们需要降低它在语料库文档中出现的权重，`tf`大代表的信息可能不如仅在语料库部分文档中出现次数较少的单词代表的信息多，如，“是”、“的”、“我”等词。
这种操作称为逆文档率`tf-idf`。
`tf`与`tf-idf`计算如下：

    >>> from sklearn.feature_extraction.text import TfidfTransformer
    >>> tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    >>> X_train_tf = tf_transformer.transform(X_train_counts)
    >>> X_train_tf.shape
    (2257, 35788)
在上面的实例代码中，我们首先使用`fit`函数来拟合数据得到估计器，第二，使用`transform`函数把统计矩阵转换成`tf-idf`表达式。这两步结合在一起使用也能获取最终相同的结果，而且速度还要快些，即使用`fit_transform`函数，下面这个例子与上面等价：

    >>> tfidf_transformer = TfidfTransformer()
    >>> X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    >>> X_train_tfidf.shape
    (2257, 35788)
## 3.3 训练分类器 ##
现在，我们已经获取到了文档词的特征向量，之后便能训练分类器来预测一个新的文档所属类别。以贝叶斯分类器开始，它提供了很多能够完成下面任务的基本曲线函数，同时scikit-learn还容纳了贝叶斯分类的许多变量参数，其中对文档单词处理最适合是`multinomial`变量。

    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
在尝试预测新文档输出结果前，我们需要把前面从训练样本中析取特征的操作应用在新文档上进行析取特征。
在`transformers`变换对象上调用`transform`方法与调用`fit_transform`的区别在于：在前面，我们已经对训练集样本进行了拟合操作，现在也就不再需要`fit`方法拟合了，即只需调用`transform`方法。

    >>> docs_new = ['God is love', 'OpenGL on the GPU is fast']
    >>> X_new_counts = count_vect.transform(docs_new)
    >>> X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    >>> predicted = clf.predict(X_new_tfidf)
    >>> for doc, category in zip(docs_new, predicted):
    ... print('%r => %s' % (doc, twenty_train.target_names[category]))
    ...
    'God is love' => soc.religion.christian
    'OpenGL on the GPU is fast' => comp.graphics
## 3.4 构建pipline管道 ##
为了能使向量对象、变换对象、分类对象在一起工作，scikit-learn提供了`Pipeline`管道，它工作起来如多个复合分类器对象一样。

    >>> from sklearn.pipeline import Pipeline
    >>> text_clf = Pipeline([('vect', CountVectorizer()),
    ... ('tfidf', TfidfTransformer()),
    ... ('clf', MultinomialNB()),
    ... ])
`vect`、`tfidf`和`clf`（classifer）名称是任意取的，我们还能在下面的网格搜索部分再看到它们的身影。现在，我们使用简单的命令训练模型：

    >>> text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
## 3.5 在测试集上对模型评估 ##
评估模型的预测准确度是相当容易的。

    >>> import numpy as np
    >>> twenty_test = fetch_20newsgroups(subset='test',
    ... categories=categories, shuffle=True, random_state=42)
    >>> docs_test = twenty_test.data
    >>> predicted = text_clf.predict(docs_test)
    >>> np.mean(predicted == twenty_test.target)
    0.834...
还不错嘛，我们取得了83.4%的准确度。尝试使用线性支持向量机（SVM）模型是否有更好的结果，该模型广受好评，是最好的文本分类算法之一，尽管与贝叶斯分类器相比有些不高效。我们可以通过把一个新的不同的分类器对象追加到`Pipeline`管道对象中达到修改模型学习对象的目的。

    >>> from sklearn.linear_model import SGDClassifier
    >>> text_clf = Pipeline([('vect', CountVectorizer()),
    ... ('tfidf', TfidfTransformer()),
    ... ('clf', SGDClassifier(loss='hinge', penalty='l2',
    ... alpha=1e-3, n_iter=5, random_state=42)),
    ... ])
    >>> _ = text_clf.fit(twenty_train.data, twenty_train.target)
    >>> predicted = text_clf.predict(docs_test)
    >>> np.mean(predicted == twenty_test.target)
    0.912...
对于更深次的结果分析，scikit-learn提供了很多实用功能，如：

    >>> from sklearn import metrics
    >>> print(metrics.classification_report(twenty_test.target, predicted,
    ... target_names=twenty_test.target_names))
    ...
    precision recall f1-score support
    alt.atheism 0.95 0.81 0.87 319
    comp.graphics 0.88 0.97 0.92 389
    sci.med 0.94 0.90 0.92 396
    soc.religion.christian 0.90 0.95 0.93 398
    avg / total 0.92 0.91 0.91 1502
    >>> metrics.confusion_matrix(twenty_test.target, predicted)
    array([[258, 11, 15, 35],
    [ 4, 379, 3, 3],
    [ 5, 33, 355, 3],
    [ 5, 10, 4, 379]])
正如预期的一样，混淆矩阵显示来自`20newsgroups`的`athesim`和`christian`是更凌乱的与`computer graphics`相比。
## 3.6 使用网格搜索优化参数 ##
在`TfidfTransformer`类中内建了很多参数，如`use_idf`，而且，分类器对象也倾向于使用更多的参数，如，`MultinomialNB`类包含了一个平滑参数`alpha`参数，`SGDClasifier`也有惩罚项（penalty）参数`alpha`和在对象函数可配置的损失函数及惩罚项（penalty items）
代替多个组件链对参数的调整操作，通过对网格可能值穷举来搜索最佳参数值是可能的。我们使用单一词或二元分词语法，并对每个线性SVM分类器赋予0.01或0.001的惩罚项系数试验所有的分类器。

    >>> from sklearn.grid_search import GridSearchCV
    >>> parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
    ... 'tfidf__use_idf': (True, False),
    ... 'clf__alpha': (1e-2, 1e-3),
    ... }
很显然，这种搜索的运算开销是巨大的。如果我们有多CPU内核，我们通过对`n_jobs`参数设置来告诉网格搜索对象能够使用的CPU核数，使它尝试并行计算由八个参数组成的联合体。如果给`n_jobs`参数赋值`-1`，那么机器上所有可用CPU内核将全部投入使用。

    >>> gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
网格搜索对象工作方式与scikit-learn其它模型一样。让我们看看该网格搜索对象在较小的训练数据集中运算速率大小。

    >>> gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
在`GridSearchCV`对象上调用`fit`方法的结果是一个分类器对象，我们能够使用该对象进行预测操作。

    >>> twenty_train.target_names[gs_clf.predict(['God is love'])]
    'soc.religion.christian'
它是一个相当大、复杂的对象，不过，我们可以通过检查对象的`grid_scores_`属性来获取最佳的参数，该属性返回一个参数与分数对（`parameters/score`）的列表。为得到最佳得分的属性，进行如下操作：

    >>> best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    >>> for param_name in sorted(parameters.keys()):
    ... print("%s: %r" % (param_name, best_parameters[param_name]))
    ...
    clf__alpha: 0.001
    tfidf__use_idf: True
    vect__ngram_range: (1, 1)
    >>> score
    0.900...
## 3.6.1 练习 ##
为了练习方便而不破坏原有文件，我们复制'`skeletons`'目录里内容到新目录'`workspace`'。

    % cp -r skeletons workspace
在ipython中执行如下命令：

    %run workspace/exercise_XX_script.py arg1 arg2 arg3
如果抛出异常，请使用`%debug`命令进行调试。
## 扩展 ##
1. 在`CountVectorizer`类中多次尝试练习`analyzer` 和1token` `normalisation`操作。
2. 如果没有类标号，尝试使用`Clustering`
3. 如果每篇文档具有多个类标签，那么翻阅`Multiclass and multilabel` 部分
4. 对于隐藏的语义分析，请使用`Truncated SVD`。
5. 好好使用 `Out-of-core Classification`类从不适合存入计算机内存数据中学习。
6. 你应该好好看看 与`CountVectorizer`类相比，储存效率高的`Hashing Vectorizer`类。
