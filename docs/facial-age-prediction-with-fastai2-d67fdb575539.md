# 用 Fastai2 进行面部年龄预测

> 原文：<https://medium.com/analytics-vidhya/facial-age-prediction-with-fastai2-d67fdb575539?source=collection_archive---------15----------------------->

不久前，年龄预测应用在 iOS 手机用户中相当流行。在这篇文章中，我们将创建一个深度学习模型，根据人脸图像预测一个人的年龄。

让我们开始吧…！！

对于这个模型，我们将使用 Fastai2 库。它包含了 NLP、推荐系统和计算机视觉所需的所有子库。对于这个计算机视觉任务，我们将使用视觉子库。

```
!pip install fastai2 -q
from fastai2.vision.all import *
from fastai2.basics import *
```