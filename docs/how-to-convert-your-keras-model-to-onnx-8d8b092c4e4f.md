# 如何将 Keras 模型转换为 ONNX

> 原文：<https://medium.com/analytics-vidhya/how-to-convert-your-keras-model-to-onnx-8d8b092c4e4f?source=collection_archive---------0----------------------->

## 将 Keras 模型转换为 ONNX 格式，并提供一些说明

> 被困在付费墙后面？点击[该好友链接](/@clh0524/how-to-convert-your-keras-model-to-onnx-8d8b092c4e4f?source=friends_link&sk=75278939091222853d4949fcf5bf2899)进入:)

![](img/d324547e180a0cbd017336237ff3240b.png)![](img/351f1d27c6506b3e8b4facee859e1b6e.png)

# 直觉

我喜欢 Keras 的简单。用 10 分钟左右的时间，我就可以用优雅的代码，用它的顺序或函数式 API，建立一个深度学习模型。然而，Keras 总是以非常慢的速度加载它的模型。此外，我不得不使用另一个深度学习框架，因为系统约束或者仅仅是我的老板告诉我使用那个框架。虽然我可以用其他人的脚本将我的 Keras 模型转换到其他框架，但我仍然将我的模型转换到 ONNX，以尝试它在 AI 工具中声称的互操作性。

# ONNX 是什么？

[ONNX](https://onnx.ai/) 是“开放式神经网络交换”的缩写。ONNX 的目标是成为一种表示深度学习模型的开放格式，以便我们可以轻松地在框架之间移动模型，它是由脸书和微软创建的。

# 将 Keras 模型转换为 ONNX

1.  从 [my GitHub](https://github.com/Cuda-Chen/keras2onnx-example) 下载示例代码
2.  点击从[下载预训练重量](https://drive.google.com/file/d/1ouJ8xZzi6x2cEkojS3DC1Wy77zjBGP1c/view)
3.  键入以下命令进行设置

```
$ conda create -n keras2onnx-example python=3.6 pip
$ conda activate keras2onnx-example
$ pip install -r requirements.txt
```

4.运行此命令将预训练的 Keras 模型转换为 ONNX

```
$ python convert_keras_to_onnx.py
```

5.使用 ONNX 运行时运行此命令进行推理

```
$ python main.py 3_001_0.bmp
```

它最终应该输出以下消息:

```
...
3_001_0.bmp
    1.000  3
    0.000  37
    0.000  42
    0.000  14
    0.000  17
```

# 一些解释

`convert_keras_to_onnx.py`将 Keras `.h5`模型转换为 ONNX 格式，即`.onnx`。它的代码如下所示:

将 Keras 模型转换为 ONNX 有几点:

1.  记得导入`onnx`和`keras2onnx`包。
2.  `keras2onnx.convert_keras()`函数将 keras 模型转换为 ONNX 对象。
3.  `onnx.save_model()`功能是将 ONNX 对象保存到`.onnx`文件中。

`main.py`利用 ONNX 模型推断鱼类图像。我把代码粘贴到这里:

推理时有一些要点:

1.  记得导入`onnxruntime`包。
2.  `onnxruntime.InferenceSession()`函数加载 ONNX 模型。
3.  第 34 行的`run()`函数预测图像并返回预测结果。此外，`sess.run(None, feed)[0]`将第 0 个元素作为 numpy 矩阵返回。
4.  `np.squeeze(pred_onnx)`将 numpy 矩阵压缩成 numpy 向量，即去掉第 0 维，这样就可以得到每一类的概率。

# 推理时间

## 总推理时间(负载模型+推理)

一些报告称 ONNX 在模式加载时间和推理时间上都运行得更快。因此，本节我在笔记本电脑上做了一个推理时间实验。

该实验的硬件如下所示:

*   CPU:酷睿 i5–3230m
*   内存:16GB

软件和软件包如下所示:

*   操作系统:CentOS 7.6
*   编程语言:Python 3.6
*   包装
*   keras 版本 2.2.4
*   tensorflow 版本 1.13.1
*   onnxruntime

每个推理运行三次，以消除其他因素的误差，例如上下文切换。

对于使用 Keras 的推理，我的计算机运行时会产生以下结果:

```
$ time python resnet50_predict.py 3_001_0.bmp
# run for three times
...
real    0m37.801s
user    0m37.254s
sys 0m1.590s
...
real    0m35.558s
user    0m35.838s
sys 0m1.362s
...
real    0m36.444s
user    0m36.542s
sys 0m1.418s
```

推理时间约为(37.081+35.58+36.444)/3 = 36.37 秒(四舍五入到小数点后第二位)。

相反，ONNX 运行时的推理显示:

```
$ time python main.py
# run three times
...
real    0m2.576s
user    0m2.919s
sys 0m0.759s
...
real    0m2.530s
user    0m2.931s
sys 0m0.700s
...
real    0m2.560s
user    0m2.944s
sys 0m0.710s
```

哇哦。多么巨大的进步啊！推理时间约为(2.576+2.530+2.560)/3 = 2.56 秒。

> *用 Keras 推断的代码可以在我的 GitHub repo* *上找到* [*。*](https://github.com/Cuda-Chen/fish-classifier/tree/master/cnn)

## 推理时间(仅推理)

> 编辑:我的一个朋友说我应该只测试 Keras 和 ONNX 之间的推理时间，因为我们实际上只加载一次模型。因此，我将只测试 Keras 和 ONNX 之间的推理时间，并将它分成两部分:
> 
> *1。keras(tensor flow 由* `*pip*` *安装)v.s. ONNX*
> 
> *2。Keras(装有 TensorFlow 由* `*conda*` *安装)v.s. ONNX*

当然，我写`comparison.py`是为了做对比测试，如下图所示:

你可以看到我运行每种推理方法 10 次，取平均时间，我运行`comparison.py`三次以减少错误。

## keras(tensor flow 由`pip`安装)v.s. ONNX

对比如下所示:

```
$ python comparison.py
...
Keras inferences with 0.8759469270706177 second in average
ONNX inferences with 0.3100883007049561 second in average
...
Keras inferences with 0.8891681671142578 second in average
ONNX inferences with 0.313812255859375 second in average
...
Keras inferences with 0.9052883148193359 second in average
ONNX inferences with 0.3306725025177002 second in average
```

我们发现 Keras 推理需要(0.88+0.87+0.91)/3 = 0.87 秒，而 ONNX 推理需要(0.31+0.31+0.33)/3 = 0.32 秒。ONNX 和 Keras 之间的加速比为 0.87/0.32 = 2.72 倍。

## keras(tensor flow 由`conda`安装)v.s. ONNX

等一下！`pip install tensorflow`安装 TensorFlow，不优化英特尔处理器。所以让我们先移除 TensorFlow，然后通过`conda`安装它(我安装的版本是`1.13.1`)。

然后再次运行`comparison.py`:

```
$ python comparison.py
...
Keras inferences with 0.9810404300689697 second in average
ONNX inferences with 0.604683232307434 second in average
...
Keras inferences with 0.8862279415130615 second in average
ONNX inferences with 0.6059059381484986 second in average
...
Keras inferences with 0.9496192932128906 second in average
ONNX inferences with 0.5927849292755127 second in average
```

我们发现 Keras 的推理时间为(0.98+0.89+0.95)/3 = 0.94 秒。与 ONNX 相比，它的推理时间为(0.60+0.61+0.59)/3 = 0.6 秒。这是 0.94/0.6 = 1.57 倍的加速。有趣的是，通过`conda`安装 TensorFlow 后，Keras 和 ONNX 都变得更慢。

# 结论

在这篇文章中，我将介绍 ONNX，并展示如何将 Keras 模型转换为 ONNX 模型。我还演示了如何使用 ONNX 模型进行预测。希望你喜欢这篇文章！

> 如果你有什么想法和问题要分享，请联系我[**clh 960524【at】Gmail . com**](http://clh960524@gmail.com)。另外，你可以查看我的 [GitHub 库](https://github.com/Cuda-Chen)中的其他作品。如果你像我一样对机器学习、图像处理和并行计算充满热情，请随时[在 LinkedIn](https://www.linkedin.com/in/lu-hsuan-chen-78071b171/) 上添加我。