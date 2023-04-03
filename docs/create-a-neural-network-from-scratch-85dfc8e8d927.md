# 从头开始创建神经网络

> 原文：<https://medium.com/analytics-vidhya/create-a-neural-network-from-scratch-85dfc8e8d927?source=collection_archive---------34----------------------->

> 在这项工作中，我将只使用 python 库 NumPy 创建一个基本的神经网络。

# **第一步:**

让我们首先创建我们的自变量或输入特征集和相应的因变量或标签

input_set = np.array([[0，1，0]，

[0,0,1],

[1,0,0],

[1,1,0],

[1,1,1],

[0,1,1],

[0,1,0]])

#因变量

labels = np.array([[1，

0,

0,

1,

1,

0,

1]])

labels = labels . shape(7，1)#将标签转换为矢量

# 第二步:

定义超参数；我们将使用 numpy 的 **random.seed** 函数，这样无论何时执行代码，我们都可以获得相同的随机值。

随机种子(42)

权重= np.random.rand(3，1)

bias = np.random.rand(1)

lr = 0.05 #学习率

# 第三步:

定义**激活函数**及其导数:我们的激活函数是 **sigmoid** 函数

def sigmoid (x): # sigmoid 函数

return 1/(1+np.exp(-x))

def sigmoid _ derivative(x):# sigmoid 函数的导数

返回 sigmoid(x)*(1-sigmoid(x))

# 第四步:

现在是训练我们的人工神经网络模型的时候了，我们将对我们的数据**训练算法 25，000 次**，因此我们的纪元将是 25，000。

对于范围内的纪元(25000):

输入=输入集

XW = np.dot(输入，权重)+偏差

z =形(XW)

误差= z-标注

print (error.sum())

dcost =错误

dpred = sigmoid_derivative(z)

z_del = dcost*dpred

输入=输入集。T

权重=权重 lr * np.dot(输入，z_del)

对于 z_del 中的数字:

偏差=偏差—lr *数量

# 第五步:

做出**预测**；是时候做一些预测了。让我们试试**【1，0，0】**

single_pt = np.array([1，0，0])

结果= sigmoid(np.dot(single_pt，weights) + bias)

打印(结果)

# output = [0.02262959]

如你所见，**输出**更接近于 0 而不是 1，因此它被归类为 0。让我们用**【0，1，0】**再试一次

single_pt = np.array([0，1，0])

结果= sigmoid(np.dot(single_pt，weights) + bias)

打印(结果)

# output = [0.98778682]

如您所见，输出更接近于 1，而不是 0，因此它被归类为 1。

# 这种模式的缺点是:

所设计的 ANN 将不能对非线性可分数据进行分类。