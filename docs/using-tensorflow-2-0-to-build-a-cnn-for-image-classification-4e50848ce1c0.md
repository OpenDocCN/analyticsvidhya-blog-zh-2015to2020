# 使用 Tensorflow 2.0 构建用于图像分类的 CNN😎

> 原文：<https://medium.com/analytics-vidhya/using-tensorflow-2-0-to-build-a-cnn-for-image-classification-4e50848ce1c0?source=collection_archive---------6----------------------->

**目标受众:熟悉 Python，对神经网络有基本了解的 ML 爱好者。我绝不是专家，想分享我学到的东西！**

TensorFlow 版本于 2019 年 9 月(差不多 1 年前)发布，促进了机器学习模型的创建和使用。即使有一些使用 TensorFlow 版的经验，我发现在 MNIST 基准数据集上实现一个准确率约为 98.5%的工作模型有一个陡峭的学习曲线。在这里，我想分享我如何在 Python 3 中用 TensorFlow 2.0 实现了一个卷积神经网络，以便帮助那些可能正在学习曲线中挣扎的人。

# 安装方法:

下面是我使用的硬件和软件规格，它们导致了一个工作安装和一个工作模型:

![](img/43c90b68370a1517e7df9227f93ed53f.png)

由[麦斯威尔·尼尔森](https://unsplash.com/@maxcodes?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

*   操作系统:Windows 10
*   IDE: PyCharm 2019.3.5
*   Python 3.6.7
*   张量流 2.3.0
*   NVIDIA GeForce GTX1050
*   NVIDIA 驱动程序版本 451.67
*   CUDA 10.1(更新 2)
*   CUPTI 10.1(自带 CUDA)
*   cuDNN v7.6.5

之前在 Tensorflow v1.0 (TF1)中，有一个单独的 GPU tensorflow 包，而在 TF2，有一个针对 CPU、GPU 和多 GPU tensorflow 版本的全局包。用你的 Python 包管理器安装 TF2 (我在 PyCharm 中使用了 pip)。👍

TF2 应该默认使用你的 CPU，但是如果你有一个[有效的 NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) 并且想像我一样使用它，那么你就需要再遵循几个[安装步骤](https://www.tensorflow.org/install/gpu)。你必须安装各自的 [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) 、 [CUPTI](https://developer.nvidia.com/cuda-toolkit-archive) 和 [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) 库。TF 网站保存了这些库的兼容版本列表[这里](https://www.tensorflow.org/install/source#linux)。

检查 TF2 是否已成功安装在您的系统上:

```
import tensorflow as tf
print(tf.constant('Hello from TensorFlow ' + tf.__version__))
```

对于 GPU 构建，请检查您是否有一个 GPU，并为其安装了 TF2:

```
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())
```

如果这些行如预期运行，那么恭喜你🎉，您就可以在 TensorFlow 2.0 中建立模型了。如果你在安装过程中有问题(我肯定有)，请在下面评论，我会尽力提供反馈。

# 构建模型框架:

现在让我们来看看我们将用来在 TF2 构建定制机器学习(ML)模型的主干。注意，这个框架使用了 Keras 包，这是一个高级 TensorFlow 包装器，使得定制 ML 模型(尤其是 CNN 的)更加容易。这里我用 32c3p 2–32c3p 2–32c5s 2-D128-D10 架构做了一个 CNN 图像分类器:

```
import os
# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout

class Image_CNN(Model):
  def __init__(self):
    super().__init__()
    self.conv1 = Conv2D(32, 3, input_shape=(..., 3), strides=1, activation='relu')
    self.conv2 = Conv2D(32, 3, strides=1, activation='relu')
    self.conv3 = Conv2D(32, 5, strides=2, activation='relu')

    self.pool1 = MaxPool2D(pool_size=(2,2))
    self.batchnorm = BatchNormalization()
    self.dropout40 = Dropout(rate=0.4)

    self.flatten = Flatten()
    self.d128 = Dense(128, activation='relu')
    self.d10softmax = Dense(10, activation='softmax')

  def call(self, x, training=False):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool1(x)
    x = self.conv3(x)
    x = self.batchnorm(x)
    if training:
        x = self.dropout40(x, training=training)

    x = self.flatten(x)
    x = self.d128(x)
    x = self.d10softmax(x)

    return x
```

如你所见，我们导入了 CNN 的构建块，并在我们的“Image_CNN”类中实现了它们。自定义模型继承自 Keras 模型，必须包含 2 个方法(函数):“__init__”和“call”。“__init__”方法定义了层,“call”方法定义了在急切执行下使用的层的顺序和数量，这是 TF2 最大的新特性之一。

**注意#1** :为了使你的 ML 模型完全定制化，你应该定义你的模型的每一个构建块(层),而不是使用预先打包的 Keras 层。

**注意#2** :在这个 CNN 中，我在卷积步骤后添加了一个漏层。这不仅有助于训练神经网络，它应该只用于训练(我们不想删除我们的任何测试结果)。

# 定义训练和测试步骤:

我们定义了将为每个数据点和每个时期执行的训练函数和测试函数。

```
@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)
```

每个函数都由一个 tensorflow 装饰函数包装，该函数将 Python 函数转换为静态 tensorflow 图。因为 TF2 使用急切执行，这些函数的逐行求值可能会很慢，所以我们将函数转换成一个静态图来加速代码执行。👍

您还会注意到，在“train_step”方法中，我们使用了 tf。TF2 的另一个新特色。这也是向 TensorFlow 的急切执行方法转变的结果。因为我们的模型急切地执行(而不是作为静态图形)，我们需要在它们运行时跟踪每一层的梯度，我们使用 GradientTape 来完成这项工作。这些梯度然后被馈送到所选择的优化器中，以通过最小化损失函数来继续学习过程。🧠

# 准备数据集并运行模型:

![](img/a44dad8c82db9612dbefb062830eb91f.png)

由[拍摄的亚历山大·辛恩](https://unsplash.com/@swimstaralex?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我们现在准备训练和测试我们的神经网络！在这里，我们加载了经典的手写数字 MNIST 数据集，它可以直接从 tf.keras.datasets 导入。有 60，000 个训练图像和 10，000 个测试图像，每个图像的维数为 28x28。然后使用 tf.data.Dataset 将这些图像以 32 个一批的方式输入到我们的模型中。

```
if __name__ == '__main__':
    import time
    start_time = time.time()

    # Load MNIST images and normalize pixel range to 0-1.
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension.
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    # Set up a data pipeline for feeding the training and testing data into the model.
    shuff_size = int(0.25 * len(y_train))
    batch_size = 32
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(shuff_size).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # Instantiate our neural network model from the predefined class. Also define the loss function and optimizer.
    model = Image_CNN()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    # Define the metrics for loss and accuracy.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Run and iterate model over epochs
    EPOCHS = 5
    for epoch in range(EPOCHS):

      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()

      # Train then test the model
      for images, labels in train_ds:
        train_step(images, labels)
      for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

      # Print results
      template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
      print(template.format(epoch + 1,
                            train_loss.result(),
                            train_accuracy.result() * 100,
                            test_loss.result(),
                            test_accuracy.result() * 100))
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
```

因为我们正在执行多标签图像分类，所以我们的精度度量将是分类交叉熵，并且我们将在 5 个时期内训练我们的 CNN。让我们看看我们做得怎么样:

```
Epoch 1, Loss: 1.5532550811767578, Accuracy: 92.288330078125, Test Loss: 1.4825093746185303, Test Accuracy: 98.0Epoch 2, Loss: 1.4951773881912231, Accuracy: 96.80332946777344, Test Loss: 1.4800351858139038, Test Accuracy: 98.12999725341797Epoch 3, Loss: 1.488990306854248, Accuracy: 97.30999755859375, Test Loss: 1.4788316488265991, Test Accuracy: 98.2699966430664Epoch 4, Loss: 1.4862409830093384, Accuracy: 97.55833435058594, Test Loss: 1.4759035110473633, Test Accuracy: 98.58000183105469Epoch 5, Loss: 1.484948992729187, Accuracy: 97.66667175292969, Test Loss: 1.475019931793213, Test Accuracy: 98.63999938964844time elapsed: 26.20s
```

我们已经成功安装了 TF2，创建了 CNN 图像分类器，并使用 GPU 在 30 秒内实现了良好的测试准确性！

如果你对我们在✌的过程有任何问题和评论，欢迎在下面发帖

```
import os
# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, BatchNormalization, Dropout

class Image_CNN(Model):
  def __init__(self):
    super().__init__()
    self.conv1 = Conv2D(32, 3, input_shape=(..., 3), strides=1, activation='relu')
    self.conv2 = Conv2D(32, 3, strides=1, activation='relu')
    self.conv3 = Conv2D(32, 5, strides=2, activation='relu')

    self.pool1 = MaxPool2D(pool_size=(2,2))
    self.batchnorm = BatchNormalization()
    self.dropout40 = Dropout(rate=0.4)

    self.flatten = Flatten()
    self.d128 = Dense(128, activation='relu')
    self.d10softmax = Dense(10, activation='softmax')

  def call(self, x, training=False):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    x = self.pool1(x)
    x = self.conv3(x)
    x = self.batchnorm(x)
    if training:
        x = self.dropout40(x, training=training)

    x = self.flatten(x)
    x = self.d128(x)
    x = self.d10softmax(x)

    return x

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=True)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = model(images, training=False)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

if __name__ == '__main__':
    import time
    start_time = time.time()

    # Load MNIST images and normalize pixel range to 0-1.
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension.
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    # Set up a data pipeline for feeding the training and testing data into the model.
    shuff_size = int(0.25 * len(y_train))
    batch_size = 32
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(shuff_size).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # Instantiate our neural network model from the predefined class. Also define the loss function and optimizer.
    model = Image_CNN()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    # Define the metrics for loss and accuracy.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Run and iterate model over epochs
    EPOCHS = 5
    for epoch in range(EPOCHS):

      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()

      # Train then test the model
      for images, labels in train_ds:
        train_step(images, labels)
      for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

      # Print results
      template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
      print(template.format(epoch + 1,
                            train_loss.result(),
                            train_accuracy.result() * 100,
                            test_loss.result(),
                            test_accuracy.result() * 100))
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
```