# 感知器网络，需要平滑函数和 sigmoid 神经元

> 原文：<https://medium.com/analytics-vidhya/network-of-perceptrons-the-need-for-a-smooth-function-and-sigmoid-neuron-9a42fc5ac97f?source=collection_archive---------19----------------------->

单个感知器只能表示线性可分的布尔函数。在做任何事情之前，让我们先非正式地定义什么是线性可分函数和线性不可分函数。

> 一个函数 **f(y) = w.x + b** 被称为是线性可分的，如果存在一条线(或一个平面，如果它是三维的)将一组点**(XJ，yi)** 与另一组点 **(xj，yj)** 分开，使得所有的点**(, yi)**位于这条线上或线上，而点 **(xj，yj)** 位于这条线以下。