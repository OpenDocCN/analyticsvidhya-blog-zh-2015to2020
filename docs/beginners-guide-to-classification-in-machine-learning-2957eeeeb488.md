# 机器学习分类初学者指南

> 原文：<https://medium.com/analytics-vidhya/beginners-guide-to-classification-in-machine-learning-2957eeeeb488?source=collection_archive---------14----------------------->

![](img/272833b2babb894bb41d177d7baed79a.png)

**分类**接受监督学习。它指定数据元素所属的类，最适合在输出具有有限值和离散值时使用。在本文中，我将比较一些流行的分类模型，如 CART、感知器、逻辑回归、神经网络和随机森林。

## 资料组

为简单起见，我使用了一个包含 100 多个实例和 9 个特征的小型生育数据集:

*   执行分析的季节
*   年龄
*   儿童疾病
*   事故或严重创伤
*   外科手术
*   去年的高烧
*   饮酒
*   吸烟习惯
*   每天坐着的时间

使用的数据集可以在[这里](https://archive.ics.uci.edu/ml/datasets/Fertility)找到。

在加载数据之前，我们需要导入这些库:

```
import pandas as pd
import numpy as np
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

之后，我们可以通过运行以下命令来读取数据:

```
path= '<path-to-file>'
data = pd.read_csv(path)
```

## 预处理

在现实世界中，您几乎总是需要对数据进行预处理和规范化。然而，我们的数据集已经被规范化了(使用标签编码)。例如，冬季、春季、夏季和秋季表示为-1、-0.33、0.33 和 1。唯一需要预处理的部分是最后一列，即输出。“N”需要转换为 1，“O”需要转换为 0。这可以通过运行以下命令来完成:

```
data.Output.replace(('N', 'O'), (1, 0), inplace=True)
```

接下来，我们需要初始化 X 轴和 Y 轴。“输出”列将是我们的 Y 轴，其余的功能将构成 X 轴。在此之后，数据将分为训练和测试。最常见的比例是 70:30。这里，X_train 和 Y_train 将包含数据集的 70 %, X _ test 和 Y_test 将包含剩余的 30%。

```
Y = data['Output']
X = data.iloc[:,:-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
test_size=30)
```

## 不同模型的分析

导入这些库:

```
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
```

## 感知器

感知器是一个单层神经网络。它是一种线性分类器，即基于将一组权重与特征向量相结合的线性预测函数进行预测的分类算法。

```
ppn = Perceptron(max_iter=100, eta0=0.5)
ppn.fit(X_train, Y_train)y_pred = ppn.predict(X_test)
print(accuracy_score(Y_test, y_pred))accuracy=accuracy_score(Y_test,y_pred)print('Accuracy: %.2f'%(accuracy*100))
```

这里，`max_iter`指的是对训练数据的最大通过/迭代次数，而`eta0`指的是更新所乘以的常数。该模型的准确率为 **83.33%**

注:应用 L2 和弹性网正则化后，结果保持不变，而 L1 正则化的精度降低到 73.33。您可以通过在代码的第一行添加另一个参数`penalty='l1/l2/elasticnet'`来验证这一点。

## 逻辑回归

逻辑回归是二元分类问题(具有两个输出值的问题)的常用方法。它用于描述数据，并解释一个因变量与一个或多个标称变量、序数变量、区间变量或比率级自变量之间的关系。

```
lg_reg = LogisticRegression()
lg_reg.fit(X_train, Y_train)y_pred = lg_reg.predict(X_test)accuracy=accuracy_score(Y_test,y_pred)
print('Accuracy: %.2f'%(accuracy*100))
```

我们不需要为这个模型使用任何额外的参数。代码的前两行将调用逻辑回归函数并训练数据。下一行预测 X_test 的输出。其准确率为 **86.67%**

注意:L1 和 L2 正则化对模型的准确性没有影响，而 elasticnet 正则化是不可能的(因为数据集太小)。

## 购物车决策树

决策树是最强大和最流行的分类和预测工具。决策树是类似树结构的流程图，其中每个内部节点表示对属性的测试，每个分支表示测试的结果，每个叶节点保存一个类标签。

```
classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)y_pred = classifier.predict(X_test)classifier = DecisionTreeClassifier(max_leaf_nodes=60)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)accuracy=accuracy_score(Y_test,y_pred)
print('Accuracy: %.2f'%(accuracy*100))
```

这里的`max_leaf_nodes`指的是以最佳优先的方式种植树木。最佳节点被定义为杂质的相对减少。数字 60 可以是任何数字，也可以是零。该模型的准确率为 **76.67%**

## 随机森林

随机森林的运行方式是在训练时构建大量决策树，并输出作为各个树的类(分类)或均值预测(回归)模式的类。

```
classifier= RandomForestClassifier(n_estimators=100, criterion= 'gini') 
classifier.fit(X_train,Y_train)y_pred= classifier.predict(X_test)

accuracy=accuracy_score(Y_test,y_pred)
print('Accuracy: %.2f'%(accuracy*100))
```

`n_estimators`指的是森林中的树木数量，`criterion`指的是衡量一次分裂质量的函数。这可以是基尼指数或熵(在这种情况下，两者都产生相同的准确性)。该模型的准确率为 **80%**

## 神经网络

为了创建神经网络，我们将使用 TensorFlow 后端。为此，我们需要以下库:

```
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
```

神经网络是一系列算法，通过模拟人脑运行方式的过程，努力识别一组数据中的潜在关系。他们可以适应不断变化的输入；因此，网络无需重新设计输出标准即可生成最佳结果。

```
model= Sequential()
model.add(Dense(9, input_dim=9, activation='relu')) 
model.add(Dense(7,activation='relu'))
model.add(Dense(2,activation='softmax')) model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])y_test_cat=to_categorical(Y_test)
y_train_cat=to_categorical(Y_train)
model.fit(X_train, y_train_cat,epochs=100,batch_size=10)
```

这里，数字 9 指的是数据集特征的数量，数字 7 指的是神经元的数量，数字 2 指的是数据集的可能输出，而`epochs`指的是所有训练向量被使用一次来更新权重的次数的度量。

```
_,accuracy=model.evaluate(X_test,y_test_cat)
print('Accuracy: %.2f'%(accuracy*100))
```

该模型的准确率为 **86.67%**

## 结论

在通过五种不同的分类模型训练和测试数据集之后，观察到线性回归和神经网络具有最高的准确性(86.67%)，其次是感知器(83.33%)、随机森林(80%)和 CART 决策树(76.67%)