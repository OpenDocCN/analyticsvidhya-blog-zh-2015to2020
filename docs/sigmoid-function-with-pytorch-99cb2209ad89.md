# PyTorch 的 Sigmoid 函数

> 原文：<https://medium.com/analytics-vidhya/sigmoid-function-with-pytorch-99cb2209ad89?source=collection_archive---------2----------------------->

![](img/c7073437d5a071bebbd440222a032b9f.png)

PyTorch

## 在这篇文章中，我将告诉你如何使用 PyTorch 计算 sigmoid(激活)函数。

# Sigmoid 函数

首先，我们需要知道什么是 Sigmoid 函数。Sigmoid 函数在分类器算法中非常常用于计算概率。它总是返回一个介于 0 和 1 之间的值，这是一个事物的概率。

点击阅读更多关于 sigmoid 函数的详细信息[。](http://mathworld.wolfram.com/SigmoidFunction.html)

![](img/52cd1615c4b9508eca4e4023c839c731.png)

Sigmoid 函数

# PyTorch

![](img/4d1545082e1b6dcc4338c6f0d5c66c92.png)

PyTorch 徽标

[PyTorch](https://pytorch.org/) 是由脸书 AI 团队开发的深度学习框架。所有深度学习框架都有一个称为张量的主干。您可以将张量视为矩阵或向量，即一维张量是矩阵，二维张量是矩阵，三维张量是具有 3 个索引的数组，即三维张量中的 RGB 颜色代码。我们可以有 n 维张量。让我们看看我们将如何计算激活(带 PyTorch 的 sigmoid 函数)。

PyTorch 张量可以加、乘、减等，就像 Numpy 数组一样。一般来说，使用 PyTorch 张量与使用 Numpy 数组的方式非常相似。

# 让我们编码并理解它

让我们使用 PyTorch 的内置方法生成一些随机数据。

我们使用 [torch.manual.seed()](https://pytorch.org/docs/stable/torch.html#torch.manual_seed) ， [torch.randn()](https://pytorch.org/docs/stable/torch.html#torch.randn) ， [torch.randn_like()](https://pytorch.org/docs/stable/torch.html#torch.randn_like) 生成随机数据。

```
import torch # we import torch(i.e pyTorch)#Lets Generate some Random Datatorch.manual.seed(7) #set the seed
features = torch.randn((1,5))
#torch.randn takes tuple of dimensions and returns tensor of
#that dimension, here 1x5 vectorweights = torch.randn_like(features) 
#randn_like takes a tensor and return a random tensor of that sizebias = torch.randn((1,1))
```

现在我们已经生成了随机数据，让我们对 sigmoid 函数进行编码

```
def sigmoid(x):
   """Sigmoid Activation Function
      Arguments:
      x.torch.tensor
      Returns
      Sigmoid(x.torch.tensor)
   """
   return 1 / (1+torch.exp(x))
   #remember formula of sigmoid
   #function, torch.exp returns tensor of exp of all values in it>>> print(weights)
  tensor([[-0.8948, -0.3556,  1.2324,  0.1382, -1.6822]])
>>>print(features)
  tensor([[-0.1468,  0.7861,  0.9468, -1.1143,  1.6908]])
>>>print(bias)
  tensor([[0.3177]])
```

**第一种方法**

我们用 sigmoid(w1x1+w2x2+…)计算 sigmoid 函数。+wnxn+b)即我们将权重和特征相乘，这是一个元素接一个元素的乘法，即 w1x1 + w2x2 +…。+wnxn 并在结果中添加偏差。我们将整个结果发送给激活函数，答案存储在 y 中。

```
y = activation(torch.sum(features * weight)+bias)>>>print(y)
  tensor([[0.1595]])
```

**第二种方法**

第二种方法是矩阵乘法。我们使用 torch.matmul()或 torch.mm()方法将我们的特征向量与权重向量相乘。

我们在 PyTorch 中一般用 [torch.mm()](https://pytorch.org/docs/stable/torch.html#torch.mm) 或者 [torch.matmul()](https://pytorch.org/docs/stable/torch.html#torch.matmul) 。这里我们简单地将两个向量相乘，并加入偏差。然后我们把它发送给激活函数，得到结果。这种方法有一个技巧，我们将会看到，当我们以后处理复杂的神经网络时，这将会给我们很大的帮助。让我们编码它，看看事情。我们将使用先前声明的变量和函数。

```
>>> print(weights)
  tensor([[-0.8948, -0.3556,  1.2324,  0.1382, -1.6822]])
>>>print(features)
  tensor([[-0.1468,  0.7861,  0.9468, -1.1143,  1.6908]])
>>>print(bias)
  tensor([[0.3177]])y = activation(torch.mm(features , weight) + bias)
print(y)
```

看起来很简单，对吧？但是这里有一个真正的问题。我们的两个向量都是 5 x 1 维的，这意味着它们不能相乘，也就是说，要乘向量/矩阵，第一个向量的列数必须等于第二个向量的行数，在我们的例子中，这是不可能的。它返回以下错误。

```
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-13-15d592eb5279> in <module>()
----> y=activation(torch.mm(features, weights)+ bias))RuntimeError: size mismatch, m1: [1 x 5], m2: [1 x 5] at /Users/soumith/minicondabuild3/conda-bld/pytorch_1524590658547/work/aten/src/TH/generic/THTensorMath.c:2033
```

这是一个非常常见的错误，也是需要处理的最重要的错误之一。我们必须重塑我们的张量，使它们能够繁殖。我们必须重塑我们的权重向量，以获得正确的维度。幸运的是，我们有一些函数要处理。我们可以使用[weights . shape()](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.reshape)或者 [weights.resize()](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.resize) 或者 [weights.view(](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view) )。你可以在官方文档中看到它们之间的细微差别。我们将使用 weights.view()

```
>>>weights.view(5,1)
tensor([[-0.8948],
        [-0.3556],
        [ 1.2324],
        [ 0.1382],
        [-1.6822]])
```

让我们现在编码

```
y = activation(torch.mm(features , weights.view(5,1))+ bias)>>>print(y)
  tensor([[0.1595]])
```

这就是我们在 PyTorch 中计算激活函数的方法。当我们的神经网络中有多层即隐藏层时，它甚至更强大。

想了解更多关于 PyTorch 深度学习的知识？

> 【PyTorch 深度学习简介

对这个领域完全陌生？从这里开始。

> [‘机器学习的初学者学习路径](/machine-learning-digest/beginners-learning-path-for-machine-learning-5a7fb90f751a)