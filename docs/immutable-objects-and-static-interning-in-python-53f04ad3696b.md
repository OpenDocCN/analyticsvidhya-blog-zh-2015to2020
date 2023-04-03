# python 中的不可变对象和静态实习。

> 原文：<https://medium.com/analytics-vidhya/immutable-objects-and-static-interning-in-python-53f04ad3696b?source=collection_archive---------17----------------------->

# `apple pie" is not "apple pie"`为什么

![](img/17c0348e1992a5ccd8266a19c9fa08cd.png)

照片由[丹尼尔·阿克谢诺夫](https://unsplash.com/@danwhale?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 拍摄

当你开始使用 python 中的`is`操作符时，你会被它迷住。看完下面的例子后，你应该会质疑现实，想知道发生了什么，并问自己一个`apple`是否是一个`apple`？