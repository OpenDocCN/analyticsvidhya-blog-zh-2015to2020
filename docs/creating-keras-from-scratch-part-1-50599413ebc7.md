# 从头开始创建 Keras

> 原文：<https://medium.com/analytics-vidhya/creating-keras-from-scratch-part-1-50599413ebc7?source=collection_archive---------18----------------------->

## 第一部分

![](img/b40d46785985130b5a6ad28d21c09325.png)

# 简介:

我认为，太多的机器学习工程师在没有太多了解什么是机器学习的情况下，就直接用 Keras 和 Tensorflow 等库应用机器学习模型。这方面的一个很好的练习是回去尝试从头创建一个机器学习算法。我选择更深入一层，试图从头开始重新创建 Keras 提供的功能。

# 目标:

这个程序的目标是创建一个灵活的用户界面来创建神经网络。为了简单起见，程序将只包括最简单的层类型。

# 概念:

我的程序的结构非常类似于装配线。程序的每一部分都充当装配线上的工人。产品将从代码的每一部分传递到中央处理器，中央处理器进行所有的计算，并将相关信息发送回来。赋予每个程序这些“装配线”属性的最佳方式是将整个模型表示为一个类，并将每种类型的层表示为一个子类。

# 浏览代码:

**步骤 1|先决条件:**

```
import numpy
from matplotlib import pyplot as pltdef sigmoid(x):
    return 1/(1+np.exp(-x))def sigmoid_p(x):
    return sigmoid(x)*(1 -sigmoid(x))def relu(x):
    return np.maximum(x, 0)def relu_p(x):
    return np.heaviside(x, 0)def tanh(x):
    return np.tanh(x)def tanh_p(x):
    return 1.0 - np.tanh(x)**2def deriv_func(z,function):
    if function == sigmoid:
        return sigmoid_p(z)
    elif function == relu:
        return relu_p(z)
    elif function == tanh:
        return tanh_p(z)
```

我导入了 Numpy 来进行矩阵操作，导入了 Matplotlib 来绘制损耗随时间的变化。激活功能描述如下。还有另一个函数，它将一个值和一个激活函数作为输入，并返回导数值。

**步骤 2|创建主神经网络类:**

```
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.loss = []
    def add(self,layer_function):
        self.layers.append(layer_function)

    def initialize_weights(self):
        for layer in self.layers:
            index = self.layers.index(layer)
            weights = layer.initialize_weights(self.layers,index)
            self.weights.append(weights)

    def propagate(self,X):
        As,Zs = [],[]
        input_data = X
        for layer in self.layers:
            a,z = layer.propagate(input_data)
            As.append(a)
            Zs.append(z)
            input_data = a
        return As,Zs
```

这是定义所有其他类和函数的类。这是操作所有其他工人的产品并返回相关信息的工人。因此，它可以访问嵌套类中包含的所有变量。

**步骤 3|创建感知器类:**

```
class Perceptron:
        def __init__(self,nodes,input_shape= None,activation = None):
            self.nodes = nodes
            self.input_shape = input_shape
            self.activation = activation
        def initialize_weights(self,layers,index):
            if self.input_shape:
                self.weights = np.random.randn(self.input_shape[-1],self.nodes)
            else:
                self.weights = np.random.randn(layers[index-1].weights.shape[-1],self.nodes)
            return self.weights
        def propagate(self,input_data):
            z = np.dot(input_data,self.weights)
            if self.activation:
                a = self.activation(z)
            else:
                a = z
            return a,z
        def network_train(self,gradient):
            self.weights += gradient
```

感知器在更大的网络中充当迷你神经网络。训练函数非常简单，因为不需要外部导数。

**第 4 步|定义培训:**

```
def train(self,X,y,iterations):
        loss = []
        for i in range(iterations):
            As,Zs = self.propagate(X)
            loss.append(np.square(sum(y - As[-1])))
            As.insert(0,X)
            g_wm = [0] * len(self.layers)
            for i in range(len(g_wm)):
                pre_req = (y-As[-1])*2
                a_1 = As[-(i+2)]
                z_index = -1
                w_index = -1
                if i == 0:
                    range_value = 1
                else:
                    range_value = 2*i
                for j in range(range_value):
                    if j% 2 == 0:
                        pre_req = pre_req * sigmoid_p(Zs[z_index])
                        z_index -= 1
                    else:
                        pre_req = np.dot(pre_req,self.weights[w_index].T)
                        w_index -= 1
                gradient = np.dot(a_1.T,pre_req)
                g_wm[-(i+1)] = gradient
                for i in range(len(self.layers)):
                    self.layers[i].network_train(g_wm[i])
        return loss
```

我创建的训练函数可能不是最简洁的，但它是健壮的，因此网络可以保持其灵活性。

# 完整源代码:

```
import numpy
from matplotlib import pyplot as pltdef sigmoid(x):
    return 1/(1+np.exp(-x))def sigmoid_p(x):
    return sigmoid(x)*(1 -sigmoid(x))def relu(x):
    return np.maximum(x, 0)def relu_p(x):
    return np.heaviside(x, 0)def tanh(x):
    return np.tanh(x)def tanh_p(x):
    return 1.0 - np.tanh(x)**2def deriv_func(z,function):
    if function == sigmoid:
        return sigmoid_p(z)
    elif function == relu:
        return relu_p(z)
    elif function == tanh:
        return tanh_p(z)class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.loss = []
    def add(self,layer_function):
        self.layers.append(layer_function)

    def initialize_weights(self):
        for layer in self.layers:
            index = self.layers.index(layer)
            weights = layer.initialize_weights(self.layers,index)
            self.weights.append(weights)

    def propagate(self,X):
        As,Zs = [],[]
        input_data = X
        for layer in self.layers:
            a,z = layer.propagate(input_data)
            As.append(a)
            Zs.append(z)
            input_data = a
        return As,Zs

    def train(self,X,y,iterations):
        loss = []
        for i in range(iterations):
            As,Zs = self.propagate(X)
            loss.append(np.square(sum(y - As[-1])))
            As.insert(0,X)
            g_wm = [0] * len(self.layers)
            for i in range(len(g_wm)):
                pre_req = (y-As[-1])*2
                a_1 = As[-(i+2)]
                z_index = -1
                w_index = -1
                if i == 0:
                    range_value = 1
                else:
                    range_value = 2*i
                for j in range(range_value):
                    if j% 2 == 0:
                        pre_req = pre_req * sigmoid_p(Zs[z_index])
                        z_index -= 1
                    else:
                        pre_req = np.dot(pre_req,self.weights[w_index].T)
                        w_index -= 1
                gradient = np.dot(a_1.T,pre_req)
                g_wm[-(i+1)] = gradient
                for i in range(len(self.layers)):
                    self.layers[i].network_train(g_wm[i])
        return loss

    class Perceptron:
        def __init__(self,nodes,input_shape= None,activation = None):
            self.nodes = nodes
            self.input_shape = input_shape
            self.activation = activation
        def initialize_weights(self,layers,index):
            if self.input_shape:
                self.weights = np.random.randn(self.input_shape[-1],self.nodes)
            else:
                self.weights = np.random.randn(layers[index-1].weights.shape[-1],self.nodes)
            return self.weights
        def propagate(self,input_data):
            z = np.dot(input_data,self.weights)
            if self.activation:
                a = self.activation(z)
            else:
                a = z
            return a,z
        def network_train(self,gradient):
            self.weights += gradient

model = NeuralNetwork()Perceptron = model.PerceptronX = np.array([[0,1,1],[1,1,0],[1,0,1]])
y = np.array([[0],[1],[1]])
model.add(Perceptron(5,input_shape = (None,3),activation = sigmoid))
model.add(Perceptron(10,activation = sigmoid))
model.add(Perceptron(10,activation = sigmoid))
model.add(Perceptron(1,activation = sigmoid))
model.initialize_weights()
loss = model.train(X,y,1000)
plt.plot(loss)
```

我希望你从这篇文章中学到了一些东西！请随意使用这段代码来加深您对机器学习的理解！