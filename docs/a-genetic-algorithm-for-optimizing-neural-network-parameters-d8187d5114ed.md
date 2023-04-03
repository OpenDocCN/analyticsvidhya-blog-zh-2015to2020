# 用 Python 优化神经网络参数的遗传算法

> 原文：<https://medium.com/analytics-vidhya/a-genetic-algorithm-for-optimizing-neural-network-parameters-d8187d5114ed?source=collection_archive---------4----------------------->

![](img/2b9823ec91dc7087262b0078c0fa299b.png)

照片由[克莱门特·H](https://unsplash.com/@clemhlrdt?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

人工神经网络是一种受监督的机器学习算法，在诸如语音和图像识别、时间序列预测、机器翻译软件等各种领域的应用中非常流行。它们在研究中很有用，因为它们能够解决随机问题，这通常允许对极其复杂的问题进行近似求解。

然而，定义理想的网络架构是非常困难的。没有明确的规则规定中间层有多少个神经元，或者有多少层，或者这些神经元之间的连接应该如何实现。为了解决这类问题，本文介绍了如何使用遗传算法在 Python 中自动找到好的神经网络架构。

首先，你需要安装 [scikit-learn 包](https://scikit-learn.org/stable/)。一个简单有效的数据挖掘和数据分析工具。

对于混合算法的训练，我们将使用鸢尾花类(Setosa，Virginica 和 Versicolor)的数据库。

![](img/d87fe1b16e4a3c71b0380f0bd289d897.png)

```
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from random import randint
import random
from sklearn.metrics import mean_absolute_error as maeiris = [datasets.load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris)()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

现在我们可以开始构建遗传算法了。群体中的个体由激活、求解器和隐藏层中的神经元数量组成——这里的神经网络有两个隐藏层。下面的代码显示了一个填充初始化的示例。人口规模由 size_mlp 定义。

```
def inicialization_populacao_mlp(size_mlp):
  pop =  [[]]*size_mlp
  activation = ['identity','logistic', 'tanh', 'relu']
  solver = ['lbfgs','sgd', 'adam']
  pop = [[random.choice(activation), random.choice(solver),  randint(2,100), randint(2,100)] for i in range(0, size_mlp)]
  return pop
```

交叉算子是一种用于结合两个亲本的信息来产生新个体的算子。目的是增加遗传变异并提供更好的选择。这里使用的重组是单点杂交。

```
def crossover_mlp(mother_1, mother_2):
  child = [mother_1[0], mother_2[1], mother_1[2], mother_2[3]]    
  return child
```

为了进一步增加遗传可变性和避免局部极小值，使用的另一个算子是突变。突变的概率由 prob_mut 定义。

```
def mutation_mlp(child, prob_mut):
 for c in range(0, len(child)):
 if np.random.rand() > prob_mut:
 k = randint(2,3)
 child[c][k] = int(child[c][k]) + randint(1, 10)
 return child
```

这样的例子是分类任务，适应度函数是根据神经网络的精度计算的，在这种情况下，遗传算法的目标是最大化神经网络的精度。

```
def function_fitness_mlp(pop, X_train, y_train, X_test, y_test): 
    fitness = []
    j = 0
    for w in pop:
        clf = MLPClassifier(learning_rate_init=0.09, activation=w[0], solver = w[1], alpha=1e-5, hidden_layer_sizes=(int(w[2]), int(w[3])),  max_iter=1000, n_iter_no_change=80)try:
            clf.fit(X_train, y_train)
            f = accuracy_score(clf.predict(X_test), y_test)fitness.append([f, clf, w])
        except:
            pass
    return fitness#
```

最后，构造了遗传算法的主体。

```
def ag_mlp(X_train, y_train, X_test, y_test, num_epochs = 10, size_mlp=10, prob_mut=0.8):
    pop = inicializacao_populacao_mlp(size_mlp)
    fitness = function_fitness_mlp(pop,  X_train, y_train, X_test, y_test)
    pop_fitness_sort = np.array(list(reversed(sorted(fitness,key=lambda x: x[0]))))for j in range(0, num_epochs):
        length = len(pop_fitness_sort)
        #seleciona os pais
        parent_1 = pop_fitness_sort[:,2][:length//2]
        parent_2 = pop_fitness_sort[:,2][length//2:]#cruzamento
        child_1 = [cruzamento_mlp(parent_1[i], parent_2[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = [cruzamento_mlp(parent_2[i], parent_1[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = mutacao_mlp(child_2, prob_mut)

        #calcula o fitness dos filhos para escolher quem vai passar pra próxima geração
        fitness_child_1 = function_fitness_mlp(child_1,X_train, y_train, X_test, y_test)
        fitness_child_2 = function_fitness_mlp(child_2, X_train, y_train, X_test, y_test)
        pop_fitness_sort = np.concatenate((pop_fitness_sort, fitness_child_1, fitness_child_2))
        sort = np.array(list(reversed(sorted(pop_fitness_sort,key=lambda x: x[0]))))

        #seleciona individuos da proxima geração
        pop_fitness_sort = sort[0:size_mlp, :]
        best_individual = sort[0][1]

    return best_individual
```

享受你的编码！

要下载代码，点击[这里](https://github.com/luaffjk/ga-mlp)。另外，如果你喜欢在 Medium 上阅读更多类似的东西，考虑注册会员来支持我和成千上万的其他作家。或者你可以给我买一杯 [*这里的*](https://www.buymeacoffee.com/luanagoncaz) *代替。祝你愉快:)*