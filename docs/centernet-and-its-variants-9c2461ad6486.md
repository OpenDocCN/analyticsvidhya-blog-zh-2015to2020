# CenterNet 及其变体

> 原文：<https://medium.com/analytics-vidhya/centernet-and-its-variants-9c2461ad6486?source=collection_archive---------2----------------------->

CenterNet 是一个基于点的目标检测框架，它可以很容易地扩展到多个计算机视觉任务，包括目标跟踪、实例分割、人体姿态估计、3d 目标检测、动作检测、人-目标交互检测等。

CenterNet 不是将预定义的锚点分类到对象中并回归相应的边界框形状，而是将对象视为点，并直接回归对象的中心点和相应的属性，例如边界框的大小、偏移量、深度，甚至边界框的形状…