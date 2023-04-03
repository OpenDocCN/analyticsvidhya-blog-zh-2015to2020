# 在 MNIST 数据集上使用张量流的 Softmax 分类器及示例代码

> 原文：<https://medium.com/analytics-vidhya/softmax-classifier-using-tensorflow-on-mnist-dataset-with-sample-code-6538d0783b84?source=collection_archive---------1----------------------->

![](img/e5264bdb22924bcab6578dc9d1d4a25e.png)

**安装张量流**

```
!pip install tensorflow
```

## 正在加载 Mnist 数据集

每个 MNIST 数据点都有两个部分:一个手写数字的图像和一个相应的标签。我们称图像为“x”，标签为“y”。训练集和测试集都包含图像及其对应的…