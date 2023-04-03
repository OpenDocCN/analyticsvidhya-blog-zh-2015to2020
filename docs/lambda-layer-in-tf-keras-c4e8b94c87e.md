# tf.keras 中的 Lambda 图层

> 原文：<https://medium.com/analytics-vidhya/lambda-layer-in-tf-keras-c4e8b94c87e?source=collection_archive---------2----------------------->

![](img/47cd290ccfb6cdacc98201169f73c59a.png)

羚羊峡谷。图片提供:Erica Li[https://unsplash.com/photos/s8GQ-CoTq6A](https://unsplash.com/photos/s8GQ-CoTq6A)

# 介绍

当你需要在前一层上做一些操作，但不想给它增加任何可训练的权重时，Lambda 层是有用的。 [Lambda 图层](https://l.facebook.com/l.php?u=https%3A%2F%2Fkeras.io%2Flayers%2Fcore%2F%3Ffbclid%3DIwAR2vUiR_v5HPiGgJnmicgz2kXKL8DCAbHxg9ZEjmLXMDfbXF0dXmmMUJ-3E%23lambda&h=AT174P7LbTMMgal6JQaE_CPGIvNupO0X-1VAxNFeNr-1T5calM7IcE5S2TC1L-ovNOdFhR7P-P8s_aG9QwdvNXOaBEwEz7Y0uMHwzzpAF_qv7mBb4U8ZvbuQa1he0A)是一种定制图层进行简单运算的简单方法。假设你想添加你自己的激活函数(不是内置的 Keras)到一个层。然后，您首先需要定义一个函数，它将把前一层的输出作为…