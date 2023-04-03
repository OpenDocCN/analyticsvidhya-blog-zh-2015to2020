# CycleGAN 的美丽

> 原文：<https://medium.com/analytics-vidhya/the-beauty-of-cyclegan-c51c153493b8?source=collection_archive---------2----------------------->

## 将马翻译成斑马背后的直觉和数学

*本文假设您已经对 gan 的工作原理有了很好的理解。如果你想快速复习，或者对 GANs 完全陌生，我有一篇关于它们的文章*[](/analytics-vidhya/implementing-a-gan-in-keras-d6c36bc6ab5f)**。**

*CycleGAN 的目标很简单，学习某个数据集 *X* 和另一个数据集 *Y* 之间的映射。例如， *X* 可能是马图像的数据集，而 *Y* 可能是斑马图像的数据集。*