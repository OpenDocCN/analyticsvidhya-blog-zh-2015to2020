# “感知器训练算法”简单的 Python 实现

> 原文：<https://medium.com/analytics-vidhya/perceptron-training-algorithm-easy-python-implementation-e96192a709b1?source=collection_archive---------23----------------------->

我给大家分享一个叫“感知器”的基础训练算法的代码，逐行讲解。

代码如下:

```
**class** **Perceptron**:
    **def** __init__(self, X):
        self.weights = np.zeros(shape=len(X[0])+1)

    **def** fit(self, X, y, iters=10): **while** iters>0:
            **for** i, x **in** enumerate(X):
                **if** self.predict(x, self.weights)!=y[i]:
                    x = np.insert(x, 0, 1)
                    self.weights = self.weights + y[i]*x
            iters -= 1
        **return** self.weights

    **def** predict(self, x, w):
        **return** np.sign(np.dot(np.transpose(w[1:]),x) + w[0])
```

首先，我们必须定义一个名为**感知器**的类，并对其进行必要的初始化。它需要一些预先随机初始化的权重(在我的例子中它们都是 0)。 **X** 是一个训练模型，根据这个模型我们将向量**的形状自权重**设置为 **X +1** 的特征个数:

```
**class** **Perceptron**:
    **def** __init__(self, X):
        self.weights = np.zeros(shape=len(X[0])+1)
```

然后我们实现“**拟合**函数。它包含 **while** 循环，也可以使用错误定义来实现。您可以设置自己的计数器，让循环迭代任意多次( **iters** )。在 **while** 循环中，我们为循环创建**，为此我们**枚举 X** 来访问每一行 **X** 对应的 **y** 。如果预测函数在给定的迭代中返回一个不等于 **y** 的数(0 或 1 ),那么我们必须通过向 **X** 的每一行插入 1 并将其与 **y** 的乘积加到先前的**自权重上来更新**自权重**。如果它等于，那么我们什么都不做，自权重保持不变。最后，在我们完成所有迭代之后，我们返回这些**自重**。****

```
 **def** fit(self, X, y, iters=10): **while** iters>0:
            **for** i, x **in** enumerate(X):
                **if** self.predict(x, self.weights)!=y[i]:
                    x = np.insert(x, 0, 1)
                    self.weights = self.weights + y[i]*x
            iters -= 1
        **return** self.weights
```

“预测”功能来了。它的作用是找到一个相应的 **y** (称之为预测的 y) ，这是从训练我们的模型中找到的。我们返回**自重**去掉第一个元素和 X(**X**的某一行)和**自重**的第一个元素的点积之和的符号(0 或 1)。

```
**def** predict(self, x, w):
    **return** np.sign(np.dot(np.transpose(w[1:]),x) + w[0])
```

感谢阅读！！！