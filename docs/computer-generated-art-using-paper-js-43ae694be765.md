# 使用纸 JS 的生成艺术

> 原文：<https://medium.com/analytics-vidhya/computer-generated-art-using-paper-js-43ae694be765?source=collection_archive---------18----------------------->

![](img/db39a347be334d822edc09fb036ba79f.png)

## [点击此处观看演示](http://karanmhatre.com/auto-art)

## [在这里找到 GitHub 上的代码](https://github.com/karanmhatre1/auto-art)

# 纸质 JS 基础

[Paper JS](http://paperjs.org) 是一个运行在 HTML5 画布之上的开源矢量图形脚本框架。意思是，在 HTML5 画布上制作元素的简单方法。

我建议从[纸质 JS 教程](http://paperjs.org/tutorials/)开始，了解他们使用的各种组件，如点、大小、矩形和圆形。

# 艺术灵感

![](img/9d451208cb659e9a310d84930c27e0b2.png)

[Pinterest 上的心情板](https://in.pinterest.com/karanmhatre2000/auto-art/)

有了足够的艺术和技术知识，让我们开始编程吧。

# 成分

## 半圆

![](img/d3bcae8dc89173de34427fa079151d9a.png)

不是半圆，但我是这么叫的。

## 椭圆形

![](img/a83a523be63e1fae9ad49e980fd5d621.png)

## 线条图案

![](img/47a87688e2e7660ae9b3aa719efd4baa.png)

线

![](img/9a0bfea7740b3d0428c5927a21478ec1.png)

让他们动摇

## 环形路径

![](img/f2915802ab4a87c685fc4ccc26fc4825.png)

# 动态选择

> Art = Math.random()

当制作算法生成的艺术品时，随机数生成是你最好的朋友。

## 元素数量

随机选择 1-5 之间的一个数字来决定要显示的元素数量。此外，选择元素的类型(圆、半圆、线型或圆形路径)

## 颜色

从预先定义的调色板中选择颜色。我为每个元素选择一种颜色。

![](img/8f6aa176de4bf36315613f7cdc638e87.png)

调色板

## **尺寸**

我需要确保尺寸不会太大或太小。经过测试，我测试了一些最佳值。

## **位置**

为画布边界内的每个元素选择一个随机位置。

## **旋转**

除了**线条图案外，随机旋转应用于每一个。**旋转值从【90，180，-90，-180】中选择。

# 把它们放在一起

1.  决定我们想要显示多少元素，
2.  为每个元素选择大小、位置、颜色和旋转。
3.  在画布上添加元素。

![](img/284a58952314860fcf58c5871383f3b6.png)

请随意在[演示页面](http://karanmhatre.com/auto-art)上测试该算法。把你在 me@karanmhatre.com 的作品寄给我。

在 Github 上叉一下。