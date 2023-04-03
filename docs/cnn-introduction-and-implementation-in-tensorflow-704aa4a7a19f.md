# 卷积神经网络介绍及其在 TensorFlow 中的实现

> 原文：<https://medium.com/analytics-vidhya/cnn-introduction-and-implementation-in-tensorflow-704aa4a7a19f?source=collection_archive---------10----------------------->

# 简介:

卷积神经网络是深度神经网络，通常设计用于处理图像数据集。当我们在处理像素时，如果我们将所有像素扁平化后直接馈入一个全连通网络，泛化就变得极其困难。想象一下，如果我们有一个数据集，其中每个图像的大小为 R . g . b(360 * 360 ),那么我们将有 388800 个像素(输入向量)的单个图像提供给多层感知器。因此，有必要强调…