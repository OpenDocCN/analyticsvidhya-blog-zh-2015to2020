# 盗梦网络及其衍生产品

> 原文：<https://medium.com/analytics-vidhya/inception-network-and-its-derivatives-e31b14388bf9?source=collection_archive---------5----------------------->

在传统的图像分类模型中，每一层都从前面的层中提取信息，以获得有用的信息。但是，每种图层类型提取不同种类的信息。5x5 卷积内核的输出告诉我们不同于 3x3 卷积内核的输出，3x 3 卷积内核告诉我们不同于 max-pooling 内核的输出，等等。但是，您如何确定转换是否提供了有用的信息呢？