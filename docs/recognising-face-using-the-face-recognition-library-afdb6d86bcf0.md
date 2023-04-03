# 使用“人脸识别”库识别人脸

> 原文：<https://medium.com/analytics-vidhya/recognising-face-using-the-face-recognition-library-afdb6d86bcf0?source=collection_archive---------9----------------------->

## 用不到 50 行代码列出你自己的已知人员名单。

![](img/aa5b31e68e3c9848518a5e00e6e789de.png)

来源于[亚马逊认知](https://d1.awsstatic.com/product-marketing/Rekognition/Image%20for%20celebrity%20recognition_v3.2264009c637a0ee8cf02b75fd82bb30aa34073eb.jpg)

让我们实现一个真实的人脸识别系统！一个系统可以从我们自己的已知人员列表中识别人脸。我们将使用构建在`dlib`之上的`face_recognition`库【1】。