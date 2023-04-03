# 用 TensorFlow 处理机器学习数据

> 原文：<https://medium.com/analytics-vidhya/processing-data-for-machine-learning-with-tensorflow-9119a0d45954?source=collection_archive---------22----------------------->

![](img/456457648cc6e81df5a99d7445a02e0b.png)

凯文·Ku 在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

# 一步一步地将你的数据集转换成张量流

当使用 TensorFlow 训练数据时，看到显示形状或数据类型的错误是不正确的，这是非常令人困惑的。这是我的笔记，试图以一种简单的方式组织 tf 数据集，用于电影评论分类。

在本文中，我将处理[大型电影评论数据集](https://homl.info/imdb)，并训练一个 Keras.models.Sequential 模型，这是一个简单的层堆栈模型。

我的步骤:

> 1.加载数据集
> 
> 2.为输入创建 tf.data.Dataset
> 
> 3.创建文本矢量化图层(包括标记化和填充)
> 
> 4.创建单词包
> 
> 5.创建模型
> 
> 6.健身和训练模型

# 加载数据集

从检查 zip 文件中有哪些文件开始，我们可以使用 os.walk(filepath)。然后我们会有这样的东西:

加载数据

> /root/。keras/datasets/aclImdb[' IMDB . vocab '，' imdbEr.txt '，' README']
> 
> /root/。keras/datasets/aclImdb/train[' labeled bow . feat '，' unsupBow.feat '，' urls_neg.txt '，' urls_unsup.txt '，' urls_pos.txt']

所有文件都在其文件夹下的列表中。我们将使用这 4 个文件夹下的评论，分别是正面和负面语义的训练和测试数据集。

> /root/。keras/datasets/aclImdb/train/pos
> 
> /root/。keras/datasets/aclImdb/train/neg
> 
> /root/。keras/datasets/aclImdb/test/pos
> 
> /root/。keras/datasets/aclImdb/train/pos

4 条路径

制作 4 条路径。它们都包含 12500 条评论。(12500, 12500, 12500, 12500)

## **使用 tf.data.TextLineDataset 创建 TensorFlow 数据集**

为了简化，我在这里不做函数。我们可以直接把路径列表放入 tf.data.TexLineDataset，记得把路径变成字符串格式。我们是这样做的:

> 1.将路径传递到 tf.data.TexLineDataset()并生成 6 个 tf.data.Dataset
> 
> 2.使用 tf.data.Dataset.map()的方法将标注添加到数据集中。0 表示阴性，1 表示阳性。
> 
> 3.使用 tf.data.Dataset.concatenate()组合 neg 和 pos
> 
> 4.洗牌训练集，批处理，并预取所有三个集合。(我们现在也可以跳过 prefetcf)

*<【预取数据集】形状:((无，)，(无，))，类型:(tf.string，tf.int32) >*

## 处理评论

我们创建一个简单的 tf.constant 来确保我们的步骤是正确的，然后创建一个函数。

预处理单词和填充

**预处理后 _word(X_example)对于简单的例子，应该是这样的:**

> <tf.tensor: shape="(3," dtype="string," numpy="array([[b”It’s”," b="">'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'。 ！!'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b '<pad>' !'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b '<pad>' b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b' <pad>'，b</pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></pad></tf.tensor:>

# 创建文本矢量化图层

> 1.按词频获取所有词汇
> 
> 2.制作单词和索引的张量来创建初始化器
> 
> 3.做一张桌子

词频:使用前 1000 个单词作为词汇表

词频

## 接下来，我们将输出一个 max_size 最常用单词的列表(现在设置为 1000)，确保<pad>是第一个。</pad>

## 创建文本矢量层

在词汇表中查找每个单词的索引

在我们的模型中使用它之前，我们必须创建文本向量类并对其进行修改。

# 一袋单词

这也是我们需要添加到模型中的一层。这可以让你有一个字数的总结。举个例子，

tf.constant([[1，3，1，0，0]，[2，2，0，0，0]])

我们计算发生的次数并得到这个(不是输出)

[[ 0:2 , 1: 2 , 2:0 , 3:1 ] , [ 0:3 , 1:0 , 2:2 , 3: 0 ] ]

去掉 0 ( <pad>，所以是[[2。, 0., 1.] , [0., 2., 0.,]]</pad>

稍后创建一个类添加到模型中，我们将在模型中传递我们的输入。

一袋单词

总结和训练模型

基于张量流的电影评论分类

如果不微调参数，我们可以得到大约 0.73 的精度。

```
Epoch 1/5
782/782 [==============================] - 12s 15ms/step - loss: 0.1737 - accuracy: 0.9498 - val_loss: 0.6392 - val_accuracy: 0.7236
Epoch 2/5
782/782 [==============================] - 12s 15ms/step - loss: 0.1060 - accuracy: 0.9794 - val_loss: 0.7092 - val_accuracy: 0.7214
Epoch 3/5
782/782 [==============================] - 12s 15ms/step - loss: 0.0605 - accuracy: 0.9944 - val_loss: 0.7724 - val_accuracy: 0.7258
Epoch 4/5
782/782 [==============================] - 12s 15ms/step - loss: 0.0327 - accuracy: 0.9989 - val_loss: 0.8467 - val_accuracy: 0.7179
Epoch 5/5
782/782 [==============================] - 12s 15ms/step - loss: 0.0177 - accuracy: 0.9998 - val_loss: 0.9208 - val_accuracy: 0.7253<tensorflow.python.keras.callbacks.History at 0x7fb437eb2d68>
```

下一篇文章，我将尝试总结我在使用 TensorFlow 和 Keras 处理用于训练的图像数据方面的笔记。

*参考:使用 Scikit-Learn、Keras 和 TensorFlow 进行机器实践学习:构建智能系统的概念、工具和技术第二版*