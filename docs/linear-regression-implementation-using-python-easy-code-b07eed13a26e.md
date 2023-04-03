# 使用 Python(简单代码)实现线性回归

> 原文：<https://medium.com/analytics-vidhya/linear-regression-implementation-using-python-easy-code-b07eed13a26e?source=collection_archive---------9----------------------->

在这篇文章中，我将向您介绍实施线性回归模型的基本技术，该模型使用由
提出的算法来预测值和/或分类问题，该算法由亚塞尔·s·阿布-穆斯塔法、马利克·马格东-伊斯梅尔、宣天利和他们在《从数据中学习》一书中提出。

我将一步一步地解释每一段代码。这是代码

```
**import** **numpy** **as** **np**

**class** **LinearRegression**:

    **def** __init__(self):
        self.weights = 0

    **def** fit(self, X, y):
        X = np.insert(X.T, 0, 1, axis=0)
        X_cross = np.matmul(np.linalg.pinv(np.matmul(X, X.T)), X)
        self.weights = np.matmul(X_cross, y)
        **return** self.weights

    **def** predict(self, x):
        y_pred = np.sign(np.dot(self.weights.T, x))
        **return** y_pred

    **def** error(self, X, y):
        error = 0
        X = np.insert(X.T, 0, 1, axis=0)
        **for** i, x **in** enumerate(X.T):
            **if** self.predict(x)!=y[i]:
                error += 1
        **return** error/len(y)
```

首先，我必须导入 **numpy** 库，因为我需要一些 **numpy** 函数来实现代码。

```
**import** **numpy** **as** **np**
```

我定义了一个名为“LinearRegression”的类，以初始化开始，这是一些类函数所需要的。我给 **self.weights** 参数赋值 0(不是数组)，因为它将被进一步的函数修改，这些函数将训练我们的模型，最终返回数组值。此外，不限于将 0 分配给初始权重，可以分配任何随机数。

```
**class** **LinearRegression**:

    **def** __init__(self):
        self.weights = 0
```

接下来是 **fit()** 函数，它返回训练数据后的最终权重。它将需要参数 X 和 y，因为它将基于训练数据找到权重，即 X=X_train 和 y=y_train。因此，当您想要拟合数据时，请发送这个特定函数的 X_train 和 y_train 值。然后，我为 X 的转置的每一列(X.T[0，0]=0，X.T[0，1]=0，X.T[0，2]=0，…，X.T[0，n]=0)插入 0 作为第一个元素。

```
**def** fit(self, X, y):
        X = np.insert(X.T, 0, 1, axis=0)
```

按照《从数据中学习》一书中的算法，我正在寻找矩阵 **X_cross** ，这是寻找权重所必需的。寻找 **X_cross** 的公式如下

```
X_cross = (X^T*X)^(-1)*X^T
```

我使用 **pinv** 函数来确保我们的乘积矩阵是可逆的，因为仅仅使用 **inv** 函数将会抛出矩阵不可逆的异常。 **Pinv** 函数是求矩阵逆的通用求逆方法，即使矩阵不可逆(非常强大的工具)。

```
X_cross = np.matmul(np.linalg.pinv(np.matmul(X, X.T)), X)
```

然后，找到两个矩阵的乘积，我将它赋给 self.weights 变量，这是 mx1 数组，其中 m 是 X_train 矩阵中的行数。最后，退货。

```
self.weights = np.matmul(X_cross, y)
        **return** self.weights
```

接下来，我将使用用于分类问题的符号函数。因此，如果您不想对预测值进行分类，只需删除 np.dot(self.weights.T，x)前面的符号项。为了对事物进行分类，我正在寻找 self.weights 的值和 X_test 的每个点的值的点积。x 参数是小写的，因为它表示一个单点，这意味着我的函数 predict()只预测特定点的符号，然后返回给误差函数。

```
**def** predict(self, x):
        y_pred = np.sign(np.dot(self.weights.T, x))
        **return** y_pred
```

下一个函数，误差函数，是针对分类问题的。与 **sklearn** 库中的 **accuracy_metric** 函数相同。它需要 X_test 和 y_test。最初，我将误差定义为零。然后，再次在矩阵 X_test^T.循环通过新定义的矩阵 x 的行的开头插入 1 的行，我通过调用 **self.predict()** 函数并检查我的预测是否等于实际的 y_test 值来预测点 x 的值，这是矩阵的行。如果不是，则将误差变量加 1。毕竟，我将返回平均误差。

```
**def** error(self, X, y):
        error = 0
        X = np.insert(X.T, 0, 1, axis=0)
        **for** i, x **in** enumerate(X.T):
            **if** self.predict(x)!=y[i]:
                error += 1
        **return** error/len(y)
```