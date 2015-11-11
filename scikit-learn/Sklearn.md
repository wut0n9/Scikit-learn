1.2 加载样例数据集
导入数据集datasets 模块

    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> digits = datasets.load_digits()

数据集是一个类字典对象，它包含了所有的数据及关于数据的元数据。数据储存在data 属性中，是n_samples, n_features数组形式。对于监督学习，预测目标变量数据储存在target属性中。
例如，在digits数据集里，通过digits.data 访问被用来做样本数据分类特征变量：

    >>> print(digits.data)
    [[ 0. 0. 5. ..., 0. 0. 0.]
    [ 0. 0. 0. ..., 10. 0. 0.]
    [ 0. 0. 0. ..., 16. 9. 0.]
    ...,
    [ 0. 0. 1. ..., 6. 0. 0.]
    [ 0. 0. 2. ..., 12. 0. 0.]
    [ 0. 0. 10. ..., 12. 1. 0.]]

并且，digits.target与每个我们尝试学习的数字图片相对应。

    >>> digits.target
    array([0, 1, 2, ..., 8, 9, 8])

数组的形状shape
数据总是二维数据，shape为(n_sample, n_features)，虽然原始数组有不同的形状shape。在这个digits案例中，每个原始样本都是(8, 8)的shape形状，并且可以使用以下方式访问到：

    >>> digits.images[0]
    array([[ 0., 0., 5., 13., 9., 1., 0., 0.],
    [ 0., 0., 13., 15., 10., 15., 5., 0.],
    [ 0., 3., 15., 2., 0., 11., 8., 0.],
    [ 0., 4., 12., 0., 0., 8., 8., 0.],
    [ 0., 5., 8., 0., 0., 9., 8., 0.],
    [ 0., 4., 11., 0., 1., 12., 7., 0.],
    [ 0., 2., 14., 5., 10., 12., 0., 0.],
    [ 0., 0., 6., 13., 10., 0., 0., 0.]])

1.3 学习与预测
在这个digits数据集案例中，我们的任务是给定一张图片来预测其代表的数字。给定所有10种可能的类（数字为0-9）的样本，在其拟合出一个能够预测未知样本所属类别的模拟器。
在scikit-learn中，一个分类模拟器（估计器estimator）是Python对象，实现了fit(x, y)和predict(T)方法。
在该例子里，估计器是实现 support vector classification的sklearn.svm.SVC类。
一个估计器的构造需要参数设定来调整模型，但在第一次操作，我们仅仅把估计器看成黑盒：

    >>> from sklearn import svm
    >>> clf = svm.SVC(gamma=0.001, C=100.)

##选择模型参数 ##
该例子中，我们手动设置gamma值。gamma值可以使用网格搜索和交叉验证（grid search 和 cross validation）工具自动设定。
之后，调用估计器对象clf,当做分类器对象。把训练集传给fit方法，也即是从模型中学习了。对于训练集，其允许我们使用所有的数据集除了剩下一个作为测试集用的。使用Python用法[:-1]选出数据集作为训练集，它将产生一个包含所有数据集的除了digits.data最后一项的新的数组。而余下的作为测试集：

    >>> clf.fit(digits.data[:-1], digits.target[:-1])
    SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
    gamma=0.001, kernel='rbf', max_iter=-1, probability=False,
    random_state=None, shrinking=True, tol=0.001, verbose=False)

激动人心的时刻到了，现在，你可以预测新值了。我们询问估计器在数据集中未被用来训练模型的最后一项代表的数字为多少：

    >>> clf.predict(digits.data[-1])
    array([8])

1.4 模型持久化
使用Python的Pickle模块保存我们的scikit-learn模型。

    >>> from sklearn import svm
    >>> from sklearn import datasets
    >>> clf = svm.SVC()
    >>> iris = datasets.load_iris()
    >>> X, y = iris.data, iris.target
    >>> clf.fit(X, y)
    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
    >>> import pickle
    >>> s = pickle.dumps(clf)
    >>> clf2 = pickle.loads(s)
    >>> clf2.predict(X[0])
    array([0])
    >>> y[0]
    0

在scikit特定的任务中，使用joblib's代替pickle更令人兴奋（joblib.dump& joblib.load），joblib在大数据处理效率更高。

    >>> from sklearn.externals import joblib
    >>> joblib.dump(clf, 'filename.pkl')

之后，你可以这样加载序列化的对象（pickle对象）或在另外的Python进程中加载：

    >>> clf = joblib.load('filename.pkl')

**注意**
jiblib.dump返回的是文件名列表。每个包含clf对象的numpy数组被序列化为单独的文件，当使用joblib.load加载时务必保证所有文件在同一目录下。
2.1.1数据集
scikit-learn处理一个或更多的**二维**数据集，他们被理解为高维样本集的列表。第一轴是samples轴，第二轴为features轴。（言外之意，样本集是二维以上的必须reshape，以下介绍）
一个简单例子：

    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> data = iris.data
    >>> data.shape
    (150, 4)

这说明在iris中有150个样本集，每一个样本集都有4个特征，细节在iris.DESCR中描述。
当数据集初始时不是(n_samples, n_features)形状时，我们需要对其预处理。
**n_samples表示样本个数，n_features表示每个样本的属性个数**
一个重塑数据集形状的例子：

    >>> digits = datasets.load_digits()
    >>> digits.images.shape
    (1797, 8, 8)

为了在scikit中使用该三维数据集，必须将8x8的图片变换成长度为64的一个特征向量。

    >>> data = digits.images.reshape((digits.images.shape[0], -1))

2.1.2 估计器对象Estimators objects
需拟合数据：一个估计器是任何从数据集中学习的对象，他可能是分类器、回归器、聚类算法或一个从原始数据中析取、过滤出特征的变换对象。
所有的估计器对象都提供了一个fit方法，且接受一个数据集（通常来说是二维数组）

    >>> estimator.fit(data)

估计器参数：估计器所有参数当在初始化对象时被设置，或者通过其属性设置。

    >>> estimator = Estimator(param1=1, param2=2)
    >>> estimator.param1
    1

估计参数：在拟合数据的同时，估计器也在计算模型的估计参数，其值保存在以下划线结尾的估计对象属性里

    >>> estimator.estimated_param_

2.2 监督学习：从高维样本集预测输出变量。
**监督学习解决问题**
监督学习在于学习两个集合间的联系：观察集X与另外一个数据集y，我们成为目标集。大多数情况下，y为数组长度为1的n_samples。
所有监督学习都提供了fit(X, y)方法拟合模型，同时，predict方法在未知数据集上预测，返回所属类别。
2.2.1最近邻与高维曲线
iris数据集是分类任务，也即通过他们的petal和sepal的宽度和长度标记三种不同的irises类型（Setosa, Versicolor, Virginica）.
图片1

    >>> import numpy as np
    >>> from sklearn import datasets
    >>> iris = datasets.load_iris()
    >>> iris_X = iris.data
    >>> iris_y = iris.target
    >>> np.unique(iris_y)
    array([0, 1, 2])

*np.unique相当于set集合的作用*
KNN（最近邻分类）
KNN可能是最简单的分类器：给定一个新的X_test，在训练集上找与其最近的特征向量。
一个例子：
*Split iris data in train and test data
A random permutation, to split the data randomly*

    >>> np.random.seed(0)
    >>> indices = np.random.permutation(len(iris_X))
    >>> iris_X_train = iris_X[indices[:-10]]
    >>> iris_y_train = iris_y[indices[:-10]]
    >>> iris_X_test = iris_X[indices[-10:]]
    >>> iris_y_test = iris_y[indices[-10:]]
    >>> # Create and fit a nearest-neighbor classifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> knn = KNeighborsClassifier()
    >>> knn.fit(iris_X_train, iris_y_train)
    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    metric_params=None, n_neighbors=5, p=2, weights='uniform')
    >>> knn.predict(iris_X_test)
    array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])
    >>> iris_y_test
    array([1, 1, 1, 0, 0, 0, 2, 1, 2, 0])

***np.random.permutation方法***
高维曲线
在上例中，如果数据仅一个特征描述，且特征值在0-1间，那么新的数据会比1/n远。因此，KNN决策规则是非常高效的，只要1/n足够小与分类特征变量数量相比。
2.2.2 线性模型
Diabets 数据集
该数据集包含10个生理变量（年龄、性别、体重、血压等）和一年后的病情指标，该任务是提供生理指标预测病情。

    >>> diabetes = datasets.load_diabetes()
    >>> diabetes_X_train = diabetes.data[:-20]
    >>> diabetes_X_test = diabetes.data[-20:]
    >>> diabetes_y_train = diabetes.target[:-20]
    >>> diabetes_y_test = diabetes.target[-20:]

线性回归
**LinearRegression**，是最简单的回归形式，通过调整参数来求满足最小二乘的拟合直线。
图片2



    >>> from sklearn import linear_model
    >>> regr = linear_model.LinearRegression()
    >>> regr.fit(diabetes_X_train, diabetes_y_train)
    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
    >>> print(regr.coef_)
    [ 0.30349955 -237.63931533 510.53060544 327.73698041 -814.13170937
    492.81458798 102.84845219 184.60648906 743.51961675 76.09517222]
    >>> # The mean square error
    >>> np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2)
    2004.56760268...
    >>> # Explained variance score: 1 is perfect prediction
    >>> # and 0 means that there is no linear relationship
    >>> # between X and Y.
    >>> regr.score(diabetes_X_test, diabetes_y_test)
    0.5850753022690...



**并不是所有的估计器对象都有score方法？**错误信息：query data dimension must match training data dimension
收缩
如果每维 包含的数据很少，那么噪点就会包含很多的变量

    import numpy as np
    from sklearn import linear_model
    import pylab as pl
    x = np.c_[.5, 1].T
    y = [.5, 1]
    test = np.c_[0, 2].T
    regr = linear_model.LinearRegression()
    np.random.seed(0)
    print x
    print np.random.normal(size=(2, 1))
    for _ in range(6):
        this_x = .1*np.random.normal(size=(2, 1)) + x
        regr.fit(this_x, y)
        pl.plot(test, regr.predict(test))
        pl.scatter(this_x, y, s=3)

图3
在高维数组上统计学习解决方案是收缩回归系数为0：从样本集中 两个都是很可能不相关的，这就是**Ridge**回归

    import numpy as np
    from sklearn import linear_model
    import pylab as pl
    x = np.c_[.5, 1].T
    y = [.5, 1]
    test = np.c_[0, 2].T
    regr = linear_model.Ridge()
    np.random.seed(0)
    print x
    print np.random.normal(size=(2, 1))
    for _ in range(6):
        this_x = .1*np.random.normal(size=(2, 1)) + x
        regr.fit(this_x, y)
        pl.plot(test, regr.predict(test))
        pl.scatter(this_x, y, s=3)
这是一个偏差/方差权衡的例子：ridge的alpha越大，偏差**句就越大，而方差会越小。
我们选择合适**alpha**最小化left out error ， 这次使用diabetes数据集，而非Python语法构造的。

    >>> alphas = np.logspace(-4, -1, 6)
    >>> from __future__ import print_function
    >>> print([regr.set_params(alpha=alpha
    ... ).fit(diabetes_X_train, diabetes_y_train,
    ... ).score(diabetes_X_test, diabetes_y_test) for alpha in alphas])
    [0.5851110683883..., 0.5852073015444..., 0.5854677540698..., 0.5855512036503..., 0.5830717085554..
图4
正如所见，特征2具有很大的系数，但其对y带来的信息很少与特征1相比。
为了改善此问题，我们仅选择感兴趣的特征，无信息的特征设置为0.**Ridge**回归能够降低无信息的贡献，但不是将他们设置为0。还有一个方法是**Lasso**是将一些特征设置为0.这些方法称为稀疏矩阵法。

    from sklearn import datasets
    import numpy as np
    diabetes = datasets.load_diabetes()
    diabetes_x_train = diabetes.data[:-20]
    diabetes_y_train = diabetes.data[:-20]
    diabetes_x_test = diabetes.data[-20:]
    diabetes_y_test = diabetes.data[-20:]
    from sklearn import linear_model
    import pylab as pl
    regr = linear_model.Lasso()
    alphas = np.logspace(-4, -1, 6)
    scores = [regr.set_params(alpha=alpha).fit(diabetes_x_train, diabetes_y_train)
             .score(diabetes_x_test, diabetes_y_test) for alpha in alphas]
    best_alpha = alphas[scores.index(max(scores))]
    regr.alpha = best_alpha
    regr.fit(diabetes_x_train, diabetes_y_train)
    print regr.coef_
这个程序特别赞，

 1. 创建一个线性回归分类器，使用set_params参数进行alpha参数的设定。好，为分类对象设定好alpha之后评估，
 2. `alphas[scores.index(max(scores))]`。
 3. 选出score评分最高的一组，也就是模型拟合最好的，据此找到最优alpha ，最后打印各特征变量的系数

    regr.coef_ 特征变量系数
**分类**
对于分类问题，正如为iris打标记，线性回归不是一个好的方法，因为它给离决策边界远的点太多的权重，有个线性方式是拟合**sigmoid**函数或者**logistic**函数
     ``logistic = linear_model.LogisticRegression(C=1e5)
        logistic.fit(iris_x_train, iris_y_train)``

*多类分类*
如果多个类需要预测，这被称为 一对多分类，我们使用投票的方式选出最终结果。
*收缩与稀疏logistic回归*

    LogisticRegression(C=100000.0, class_weight=None, 
    dual=False,    fit_intercept=True, intercept_scaling=1,max_iter=100,
    multi_class='ovr',penalty='l2',random_state=None，
    solver='liblinear', tol=0.0001, verbose=0)

正如所见，C参数是用来控制LogisticRegreesion规则的数量，该值越大，结果中规则就越小，penalty=12赋给Shrinkage（非稀疏系数），penalty=11赋给稀疏系数
**SVM**
线性SVMs
支持向量机属于判断家族成员：在一组样本中构建一个平面，使得类间距离最大。规则的设置由C参数决定：C值越小，计算边界时使用的样本量越大，也即产生更多的规则，C值越小，说明样本离分割面（线）越近，即规则少。
图2.2.3_1
支持向量机分为SVC支持向量分类和SVR（支持向量回归）

        from sklearn import svm
        svc = svm.SVC(kernel='linear')
        svc.fit(iris_x_train, iris_y_train)


**注意**数据规整化
对于一些估计器，包括SVMs，数据集中的每一个特征都具有单位标准差，这很重要对一个好的预测结果。
使用内核
在特征空间中类号并不总是线性。解决方案是构造一个多项式决策函数，而非使用线性的。这通过kernel参数设定。

    >>> svc = svm.SVC(kernel='linear')

Polynomial kernel

    >>> svc = svm.SVC(kernel='poly',... degree=3)
    >>> # degree: polynomial degree
RBF kernel

    >>> svc = svm.SVC(kernel='rbf')
    >>> # gamma: inverse of size of
    >>> # radial kernel
图2.2.3_Using Kernel

2.3 模型选择：估计器和参数的选择
------------------
正如所见，每个估计器都提供score方法，用来评估模型的在新数据上拟合质量。

    >>> from sklearn import datasets, svm
    >>> digits = datasets.load_digits()
    >>> X_digits = digits.data
    >>> y_digits = digits.target
    >>> svc = svm.SVC(C=1, kernel='linear')
    >>> svc.fit(X_digits[:-100], y_digits[:-100]).score(X_digits[-100:], y_digits[-100:])
    0.97999999999999
为了得到更好的预测准确率，我们能够用*folds*分隔。
*we can
successively split the data in folds that we use for training and testing:*

    import numpy as np
    x_fold = np.array_split(x_digits, 3)
    y_fold = np.array_split(y_digits, 3)
    scores = list()
    for i in range(3):
        x_train = list(x_fold)
        x_test = x_train.pop(i)
        x_train = np.concatenate(x_train)
        y_train = list(y_fold)
        y_test = y_train.pop(i)
        y_train = np.concatenate(y_train)
        scores.append(svc.fit(x_train, y_train).score(x_test, y_test))
    print(scores)

[0.93489148580968284, 0.95659432387312182, 0.93989983305509184]
这被称为KFold交叉验证。
**注意**

 1. array_split方法：把一个数组等分成或近似等分成多个子数组部分。如果不能等分数组，该方法也不会抛出异常。
 2. pop方法是列表对象方法。剔除列表中由参数指定索引的元素。
 3. concatenate：Numpy顶层方法，按指定轴串接数组序列，返回新的数组。
2.3.2交叉验证生成器
上面提到的在训练集和测试集上分割数据的方式写起来很厌倦的，因此，scikit-learn提供了交叉验证生成器来生成分片（索引）列表。

    `import numpy as np
    from numpy import *
    from sklearn import cross_validation
    k_fold = cross_validation.KFold(n=6, n_folds=3)
    for train_indices, test_indices in k_fold:
          print 'Train %s | Test %s ' % (train_indices, test_indices)`

Train [2 3 4 5] | Test [0 1] 
Train [0 1 4 5] | Test [2 3] 
Train [0 1 2 3] | Test [4 5] 

kflod对象中包含着分割好的训练集和测试集，以(train, test)元祖列表形式保存。
**K阶交叉验证**
是指把数据集等分成K部分，每次试验时取k-1部分用来训练数据集，余下的作为测试集。
*因为是监督学习，每个样本都被标记了类号，因此在拟合方法fit中前两个参数代表的样本条数是相等的。*
交叉验证的应用：

    kfold = cross_validation.KFold(len(x_digits), n_folds=3)
    [svc.fit(x_digits[train], y_digits[train]).score(x_digits[test], y_digits[test]) for train, test in kfold]
使用cross_val_score方法计算估计器的得分：

    cross_validation.cross_val_score(svc, x_digits, y_digits, cv=kfold, n_jobs=-1)
n_jobs=-1表示该运算调度计算机的所有的CPU。
该参数在win系统上运行有问题。
**交叉验证生成器**
 4. KFold(n, k)
把数据集分成K阶，k-1阶用于训练数据集，留下的作为测试。
  StratifiedKFold(y, k)   
It preserves the class ratios / label
distribution within each fold.
LeaveOneOut (留一验证,LOOCV)
假设dataset中有n个样本，那LOOCV也就是n-CV，意思是每个样本单独作为一次测试集，剩余n-1个样本则做为训练集。
LeaveOneLabelOut
使用类标号分组样本集
2.3.3 网格搜索和交叉验证估计器
网格搜索
sklearn提供了一个对象，针对给定的数据集，该估计器通过一个参数的设定能够 使得在拟合模型的同时评估模型质量，并且选择为该参数选择合适的值以最大化交叉验证得分。该对象接受估计器（模拟器）对象，而且还提供了估计器的方法。

`from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn import datasets
from sklearn import svm
svc = svm.SVC(kernel='linear', C=1)
Cs = np.logspace(-6, -1, 10)
digits = datasets.load_digits()
x_digits = digits.data
y_digits = digits.target
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs))
clf.fit(x_digits[:1000], y_digits[:1000])
print clf.best_score_
print clf.best_estimator_.C
print clf.score(x_digits[-1000:], y_digits[-1000:])`

估计器重要的属性

    best_score_ 保存模型拟合最高得分
    best_estimator_.C   最好估计器对象的C参数的大小
默认GridSearchCV使用3阶交叉验证。但是，当它检测到接收的对象为分类器，而非回归对象时，它会用3阶分层。
*以上段话的解释应该是GridSearchCV对象本身蕴含着交叉验证，如果再把该对象传给交叉验证的方法cross_val_score，那么cross_val_score方法就会自动地交叉验证（默认三次试验）*
嵌套交叉验证
`print cross_validation.cross_val_score(clf, x_digits, y_digits)`
[ 0.93853821  0.96327212  0.94463087]
二阶交叉验证循环在并行计算上性能很好：一个是使用GridSearchCV估计器设定gamma参数，另一个是通过cross_val_score评估估计器预测能力。其在新数据上运算返回结果是稳定的。
交叉验证估计器
为交叉验证设置合适的参数可以提高运算性能。这就是为什么sklearn提供Cross-validation:evaluate estimator performance 估计器自动设置参数。

    from sklearn.grid_search import GridSearchCV
    import numpy as np
    from sklearn import datasets
    from sklearn import svm, cross_validation, linear_model
    
    lasso = linear_model.LassoCV()
    diabetes = datasets.load_diabetes()
    x_diabetes = diabetes.data
    y_diabetes = diabetes.target
    lasso.fit(x_diabetes, y_diabetes)
    print lasso.alpha_

0.0122918950875
这个估计器的调用与其“同行”a相似。
**重要属性**
alpha_
2.4 非监督学习：seeking representations of the data
2.4.1 聚类：数据集分组
k-means 聚类
注意，这存在很多的聚类标准和相关算法，但k-means是最简单的。

    from sklearn import cluster, datasets
    iris = datasets.load_iris()
    x_iris = iris.data
    y_iris = iris.target
    k_means = cluster.KMeans(n_clusters=3)
    k_means.fit(x_iris)
    print k_means.labels_[::10]
    print y_iris[::10]
重要属性 labels_  保存着由聚类算法打的类标记
分层凝聚聚类：Ward（Hierarchical agglomerative clustering: Ward ）
凝聚和分裂层次聚类分别使用自底向上和自顶向下策略把对象组织到层次结构中。
凝聚算法从每一个对象都是一个簇开始，迭代地合并，形成更大的簇。
分裂层次聚类算法与其相反，它开始时令所有给定的对象形成一个簇，迭代地分裂，形成一个更小的簇。


    Connectivity-constrained clustering
    With agglomerative clustering, it is possible to specify which samples can be clustered together by giving a connectivity graph. Graphs in the scikit are represented by their adjacency matrix. Often, a sparse matrix is used. This
    can be useful, for instance, to retrieve connected regions (sometimes also referred to as connected components) whenclustering an image:
    from sklearn.feature_extraction.image import grid_to_graph
    from sklearn.cluster import AgglomerativeClustering
    
    ###### #生成数据
    lena = sp.misc.lena()
    
    lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
    X = np.reshape(lena, (-1, 1))
    
    connectivity = grid_to_graph(*lena.shape)
    
    print("Compute structured hierarchical clustering...")
    st = time.time()
    n_clusters = 15 # number of regions
    ward = AgglomerativeClustering(n_clusters=n_clusters,
    linkage='ward', connectivity=connectivity).fit(X)
    label = np.reshape(ward.labels_, lena.shape)
    print("Elapsed time: ", time.time() - st)
    print("Number of pixels: ", label.size)
    print("Number of clusters: ", np.unique(label).size)
    

特征聚类（Feature agglomeration）
我们已经知道稀疏化常被用来降低维灾难，也就是样本量不足与特征变量相比。一种合成特征的方式是特征凝聚。这种方法可以通过聚类在特征方向来实现的，换句话说聚类转置数据。

    >>> digits = datasets.load_digits()
    >>> images = digits.images
    >>> X = np.reshape(images, (len(images), -1))
    >>> connectivity = grid_to_graph(*images[0].shape)
    >>> agglo = cluster.FeatureAgglomeration(connectivity=connectivity,
    ... n_clusters=32)
    >>> agglo.fit(X)
    FeatureAgglomeration(affinity='euclidean', compute_full_tree='auto',...
    >>> X_reduced = agglo.transform(X)
    >>> X_approx = agglo.inverse_transform(X_reduced)
    >>> images_approx = np.reshape(X_approx, images.shape)

2.5 估计器放在集成一起
2.5.1 Pipeline
我们知道一些估计器是用来变换数据，另一些用来预测变量的。我们也能创建一个组合的估计器。
**文本处理**
3.2
20newsgroups文本

    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    from sklearn.datasets import fetch_20newsgroups
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, 
                                     random_state=42)
    print twenty_train.target_names
通过fetch_20newsgroups方法的categories关键字参数限制获取数据的类别，减少计算开销。
重要属性：
target_names 类别的名称
data  数据集保存在data属性中
filename  样本所在各个文件的名称



=======
>>>>>>> origin/master

