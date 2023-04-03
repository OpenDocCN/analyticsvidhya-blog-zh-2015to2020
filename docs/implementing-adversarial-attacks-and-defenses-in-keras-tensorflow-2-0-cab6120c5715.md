# 在 Keras & Tensorflow 2.0 中实现对抗性攻击和防御

> 原文：<https://medium.com/analytics-vidhya/implementing-adversarial-attacks-and-defenses-in-keras-tensorflow-2-0-cab6120c5715?source=collection_archive---------2----------------------->

## 从 98%的准确率到 20%

最先进的图像分类对自动驾驶汽车至关重要；一个错误的分类就可能导致人命损失。对抗性攻击是一种对图像进行难以察觉的改变的方法，这种改变会导致看似强大的图像分类技术对图像进行一致的错误分类。

在本文中，我们将介绍实现对抗性攻击的基础知识，以及我们如何防御这些攻击。