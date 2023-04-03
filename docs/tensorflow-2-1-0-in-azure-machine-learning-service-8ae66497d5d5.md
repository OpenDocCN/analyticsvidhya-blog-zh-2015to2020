# Azure 机器学习服务中的 Tensorflow 2.1.0

> 原文：<https://medium.com/analytics-vidhya/tensorflow-2-1-0-in-azure-machine-learning-service-8ae66497d5d5?source=collection_archive---------30----------------------->

![](img/e2be0edf6b9f8c7c15394f5fb35939b9.png)

# 构建一个简单的 Hello world 张量流模型来测试张量流版本 2.1.0

# 首先将 pip 版本升级到最新版本

```
!pip install --upgrade pip
```

# 现在检查 tensorflow 版本，以确保我们有正确的版本

```
import tensorflow as tf print(tf.__version__)
```

如果遇到问题，请检查依赖包的错误并进行 pip 安装。当我创建计算实例时，我的版本是 2.1.0

# 安装 tensorflow

```
pip install --upgrade tensorflow
```

在这里可以找到最新版本的 tensorflow 软件包。

[https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip)

以上是设置必要的包。

现在是时候开始编码了。

# Hello world 示例

导入 tenforflow 包。

```
import tensorflow as tf
```

加载数据集并拆分用于训练和测试。

```
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() x_train, x_test = x_train / 255.0, x_test / 255.0
```

现在构建深度神经网络架构

```
model = tf.keras.models.Sequential([ tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2), tf.keras.layers.Dense(10) ])
```

现在训练模型

```
predictions = model(x_train[:1]).numpy() predictions
```

将类转换为概率

```
tf.nn.softmax(predictions).numpy()
```

损失。SparseCategoricalCrossentropy loss 接受一个 logits 向量和一个真实索引，并返回每个示例的标量损失。

```
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

这个未经训练的模型给出的概率接近随机(每类 1/10)，所以初始损失应该接近-tf.log(1/10) ~= 2.3。

```
loss_fn(y_train[:1], predictions).numpy()model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
```

Model.fit 方法调整模型参数以最小化损失:

```
model.fit(x_train, y_train, epochs=5)
```

用测试数据评估模型

```
model.evaluate(x_test, y_test, verbose=2)
```

如果希望模型返回概率，可以包装定型模型，并将 softmax 附加到它:

```
probability_model = tf.keras.Sequential([ model, tf.keras.layers.Softmax() ])
```

显示输出

```
probability_model(x_test[:5])
```

样本结束。

*最初发表于*[T5【https://github.com】](https://github.com/balakreshnan/mlops/blob/master/tensorflow/tensorflowtest.md)*。*