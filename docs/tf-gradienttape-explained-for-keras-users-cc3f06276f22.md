# tf。为 Keras 用户解释 GradientTape

> 原文：<https://medium.com/analytics-vidhya/tf-gradienttape-explained-for-keras-users-cc3f06276f22?source=collection_archive---------0----------------------->

## TF 2.0 中高级优化的必备指南

![](img/7fc604dd30961d84387be40d3d04c3c0.png)

改编自[这里](https://www.hillsdale.edu/wp-content/uploads/2017/01/Physics-Blackboard-Equations.jpg)和[这里](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/718px-Tensorflow_logo.svg.png)

*本文提供了* `tf.GradientTape` *内置方法以及如何使用它们的教程。如果您已经对* `tf.GradientTape` *有了深入的了解，并且正在寻找更高级的用途，请随意跳到* ***【高级用途】*** 一节