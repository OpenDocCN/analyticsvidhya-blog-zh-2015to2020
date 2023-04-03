# 使用深度学习在血液样本图像中检测疟疾

> 原文：<https://medium.com/analytics-vidhya/malaria-detection-in-blood-sample-images-using-deep-learning-b28c736c827c?source=collection_archive---------10----------------------->

## 组块学习

![](img/a8e14d98ae67455a41a633588cbe3f20.png)

来源:维基百科

***疟疾*** *是一种蚊媒传染病，影响人类和其他动物。疟疾引起的症状通常包括发烧、疲劳、呕吐和头痛。在严重的情况下，它会导致皮肤发黄、癫痫发作、昏迷或死亡。症状通常在被受感染的蚊子叮咬 10 到 15 天后出现。*

**传播区域:**

![](img/bc38648053d27d4c30369babcde744e9.png)

来源:必应搜索

**诊断:**

![](img/a7e36a5b69e7c67a3fcffa0dbc6a8328.png)

来源:维基百科

上面的图像为我收集血液图像样本铺平了道路，因为医生收集病人的血液样本是为了检测疾病。

因此，废话不多说，直接进入编码，今天我们将通过一个快速代码来了解血液样本图像中的疟疾检测，从在 Kaggle 数据集中找到这些数据，到构建一个简单而强大的分类器，再到血液图像中的寄生样本和未感染样本。

***继续之前注意:我一定会推荐你使用 Kaggle 内核而不是 Google Colab。为什么？见下文。***

![](img/196c82dbcee2ca75aaea42519d9fb63b.png)

来源:Kaggle 内核(类型->！nvidia-smi) 16 GBs 的 GPU Ram 和 Nvidia Tesla P100

![](img/d0c2fc91ca38353946b4e6269638b467.png)

来源:Kaggle 内核(类型->！NVIDIA-SMI)11gb 的 GPU Ram 和特斯拉 K80

在 GPU 内存方面，P100 肯定大于 K80，这有助于 CNN 训练大型模型。

***代码:***

![](img/9d1b76af4b10624cbdddd462fc02a738.png)

1.  [加载]Kaggle 内核中的数据(数据集在 ka ggle 数据集中可用)

![](img/8c17892bfe35cefce2dd89d900ac5abd.png)

*   [默认]内核加载时很少导入，如果链接，则显示数据集。

![](img/d6dcdfe055aea3196c2e8cfcdc4854b1.png)

*   加载构建分类器所需的[库]。

![](img/87ea30015dcd9b24708f766568a2357b.png)

2.数据集开始加载并描述它们。png 格式图片]。

![](img/164376a4500c6bbcc08eca745c261267.png)

3.[可视化]寄生样本图像

![](img/a3e1e0e7837495d1adab3d234d39a64c.png)

未受感染的样本图像

![](img/d4c8f8a1ab35ee604dd050750634f46c.png)

4.已将图像调整为 w-64、h-64；否则它可能会遇到内存错误(它发生在我 128，128 ),并将其转换为 Keras 图像数组格式。

![](img/ba1c3a49561ca07eed6254e4d0a731d7.png)

*   查看调整大小后的图像。

![](img/ac695742b6f5f8adb5796eb87a449b34.png)

5.为训练过程拆分数据集。

![](img/5952cf0142a12d18ef3e842446e8c1b7.png)

6.导入预训练(VGG16，19，ResNet，Inception Net，RetinaNet)模型，这样做是为了节省一些时间，并取得了良好的效果，因为我们知道预训练模型已经在大型数据集上进行了训练，获得了很好的洞察力，这将有助于更好地理解新数据集中的模式。

![](img/0761bf2ff22017404f96f99ce9549f3c.png)

*   移除顶层(因为它们是在 1000 个 ImageNet 类上训练的，并将其修改为我们用于两个类)。

![](img/a50bf4545193ad9278ae2a00d8dddac0.png)![](img/33cc397c9bb97b666c370bc25f9a22de.png)

*   在调用带参数的模型时，结果也是摘要。

![](img/7b2d522777848b0c190468e582d6d324.png)

*   设置损失函数、优化器和精确度指标

![](img/d53a3a2ea13b42d0d800270160ad0bc4.png)

7.开始训练吧，看，我们达到了非常好的准确率，大约 96%。

![](img/fe11f5e5f84de99b09824c2a38657749.png)![](img/df814a44aea3bcda2121d55ceb7931ef.png)

*   学习曲线也符合训练值

![](img/be2c690843a780d92d66df73e37a7061.png)

*   数据集的测试结果已经出来了。

***代码链接:***

*https://www . ka ggle . com/Susan T4 learning/malaria detection-in-blood samples？scriptVersionId=22550587*

*希望您对此感兴趣，在接下来的文章中，我们将介绍其他医学图像格式，并了解这些挑战以及解决这些挑战的不同 CNN 架构。*

**坚持学习！！！**