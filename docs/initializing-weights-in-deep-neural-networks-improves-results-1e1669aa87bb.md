# 初始化深度神经网络中的权重可以改善结果

> 原文：<https://medium.com/analytics-vidhya/initializing-weights-in-deep-neural-networks-improves-results-1e1669aa87bb?source=collection_archive---------27----------------------->

![](img/9783855e31a84d7a09f255c14ef155c2.png)

维克多·弗雷塔斯在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

深层神经网络是核心，许多层乘法(和其他运算)层层叠加。网络越深，我们就有越多的机会进行更多的计算，从而改进预测。那么，我们应该仅仅创造可以计算任何东西的超级深度网络吗？理论上..也许吧。实际上，它并不真正起作用。

也就是说，在某种程度上，将网做得更深是有好处的，但要充分利用它，我们需要小心一些。事实证明，图层输入的平均值为 0 且标准差为 1 非常重要。在第一个输入中，这很容易做到，只需归一化(减去平均值并除以标准差)，但随着数据经过一些运算，可能是与权重相乘，然后是激活函数(例如 ReLU)，平均值和标准差(std)都会发生变化。让我们看看这个:

*我将使用奇妙的 fastai 库，因为这篇文章的灵感来自于 [fastai 的](https://www.fast.ai/)伟大作品

```
# import what we need
from fastai import datasets
import gzip, pickle
import torch
from torch import tensor# we will use the MNIST dataset for this demonstrationMNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'
```

让我们创建一个获取数据并将其映射到张量的函数。

```
def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train, y_train, x_valid, y_valid))x_train, y_train, x_valid, y_valid = get_data()# Lets see what we've got:
x_train.shape, y_train.shape, x_valid.shape, y_valid.shape---> (torch.Size([50000, 784]),  torch.Size([50000]),  torch.Size([10000, 784]),  torch.Size([10000]))
```

现在我们有了一个训练集和一个验证集，训练集是 50k x 784 矩阵，验证集是 10k。我没有详细说明，因为我想你知道 MNIST 数据集，但更多细节请参考[这个](https://en.wikipedia.org/wiki/MNIST_database)。

查看训练集的均值和标准差，我们看到它没有均值 0 和标准差 1(为什么会有？)

```
x_train.mean(), x_train.std()---> (tensor(0.1304), tensor(0.3073))
```

我们得到的平均值为 0.13，标准差为 0.3。这不是我们想要的，但是我们可以用一个简单的规范化函数轻松地解决这个问题。让我们创建一个，然后在数据上激活它。

```
def normalize(data, mean, std):
    return (data - mean)/stdx_train = normalize(x_train, x_train.mean(), x_train.std())
x_valid = normalize(x_valid, x_train.mean(), x_train.std())# check the new mean and std
x_train.mean(), x_train.std()---> (tensor(0.0001), tensor(1.))
```

太好了，我们得到了平均值 0 和标准差 1，这正是我们想要的。**注意我们使用了与* ***相同的*** *列车数据的均值和标准差作为验证数据。如果数据没有被相同的统计数据标准化，训练数据的训练将不会预测验证数据。*

我们说完了吗？不完全是。由于我们有更多的层，我们对数据做更多的操作。甚至在第一次简单的矩阵乘法之后，我们可以看到均值和标准差，再次不是我们所需要的。

```
# A few convenience variables
n, m = x_train.shape
c = y_train.max() + 1
n, m, c---> 50000, 784, tensor(10)# number of hidden 
nh = 50# random weights matrix
w1 = torch.randn(m, nh)# biases
b1 = torch.zeros(nh)w1.shape, b1.shape
---> (torch.Size([784, 50]), torch.Size([50]))
```

我们有了随机权重矩阵和偏差向量，让我们创建并运行线性乘法运算。

```
# linear operation
def linear(x, w, b):
    return x@w + b# x_valid is normalized
r1 = lin(x_valid, w1, b1)r1.mean(), r1.std()
---> (tensor(1.7274), tensor(26.3374))
```

正如我们所看到的，我们对数据进行了规范化，这很好，但在一次操作后，它又偏离了轨道。我们的平均值为 1.7，标准差为 26.3。不行，这样学习效率不高。我们要做的是使用一种技术来解决这个问题，这种技术深受一位名叫何的研究员的启发。简单而有效(在某种程度上，很快会有更多关于这方面的内容)。

```
import math
w1 = torch.randn(m, nh)/math.sqrt(m)
b1 = torch.zeros(nh)
```

我们在这里所做的，就是将权重除以行数的平方根。我们的体重除以 784。仅仅这个简单的技巧，就能带来巨大的不同。让我们来看看实际情况:

```
# calculate a linear operation with the new w1, b1
r2 = lin(x_valid, w1, b1)r2.mean(), r2.std
---> (tensor(-0.0807), tensor(1.0222))
```

和..成功了！这非常接近 0 和 1。还不错。

不过……这个线性操作层印象不是很深刻吧？它甚至没有激活功能，这意味着它无法学习非线性数据，而这些数据可能会学习。让我们添加一个激活函数，然后，我们将使用 ReLU。

```
def relu(x):
    return x.clamp_min(0.)t = relu(linear(x_valid, w1, b1))t.mean(), t.std()---> (tensor(0.3671), tensor(0.5580))
```

正如我们所见..问题又回来了。执行 relu 操作后，mean 不为 0，std 不为 1。我们如何解决这个问题？基于[这篇论文](https://arxiv.org/abs/1502.01852)，对我们的初始化做了一个微小但重要的改变。发生的情况是，当 ReLU 用 0 替换任何负数时，我们不仅丢失了所有负的绝对数字(它们应该保持平均值不上升)，我们还每次都将方差减半。如果我们有 8 层，方差变成(1/2)⁸，这使得梯度，从而学习，是不可能的。

**关于 ReLU 定义的注释。这可以用很多方法来实现，但是，如果有一种方法使用 Pytorch 方法来实现它，这通常意味着它会快得多，因为它很有可能不是用 Python 实现的。

因此，要解决这个问题，我们要做的就是用(2/m)替换(/m)。在上面放一个 2。

```
# note the 2/m
w1 = torch.randn(m, nh)*math.sqrt(2/m)
```

这一点至关重要:

```
t2 = relu(linear(x_valid, w1, b1))t.mean(), t.std()---> (tensor(0.5414), tensor(0.8305))
```

虽然还不完美，但已经好多了。我们可以看到 std 越来越接近 1。我们仍然有一个问题。如前所述，我们用 0 代替了所有的负数，所以这有点过于简单(虽然在我的测试中效果很好),只是从平均值中减去了 0.5。让我们稍微修改一下 ReLU:

```
def relu(x):
    return x.clamp_min(0.) - 0.5t = relu(lin(x_valid, w1, b1))t.mean(), t.std()---> (tensor(-0.0025), tensor(0.7597))
```

现在好多了！这个非常简单但非常重要的技巧让学习变得更有效率，没有它(或任何足够好的替代方法)很难得到 STOA 结果。希望你学到了重要的东西，如果有任何问题/评论，请随时联系我们！

Bakugan@gmail.com 或在[领英](https://www.linkedin.com/in/chen-margalit-4b1a67117/)