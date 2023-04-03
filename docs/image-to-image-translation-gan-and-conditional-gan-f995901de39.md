# 图像到图像的翻译:GAN 和条件 GAN

> 原文：<https://medium.com/analytics-vidhya/image-to-image-translation-gan-and-conditional-gan-f995901de39?source=collection_archive---------6----------------------->

许多计算机视觉和图像处理问题需要将输入图像转换成相应的输出图像。CNN 正被用于此目的，并且它们正成为解决许多图像预测问题的常用工具。CNN 学习最小化损失函数，尽管学习过程是自动的，但在设计有效损失时需要大量的人工努力。简单来说，我们仍然需要告诉 CNN 我们想要最小化什么。

例如，如果我们要求 CNN 最小化预测之间的欧几里德距离…