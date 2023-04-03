# 通过使用机器学习和分段来预测验证码——创建一个基于光学字符识别(OCR)的算法。

> 原文：<https://medium.com/analytics-vidhya/predicting-captchas-by-using-machine-learning-and-segmentation-make-an-optical-character-53fae2bbf747?source=collection_archive---------16----------------------->

图像处理对于获取图像中包含的信息非常重要，因此它可以进一步用于数据建模，这些数据可以用于空间研究、信息安全(CAPTCHA-区分计算机和人类的全自动公共图灵测试)、语言处理、OCR 扫描和预测准确的数据。

> 神秘感永远不在于我们看到了什么，而在于深入发现它是什么，它的目的是什么。

对于基于文本的图像处理，我们需要遵循的步骤是至关重要的，并且与这些预测的准确度成正比。

![](img/64883423a96aea7db8d259682af714e8.png)

基于文本的图像预测的处理流程图

关于数据集

数据集来自[威廉米、罗德里戈&罗萨斯、奥拉西奥。(2013).](https://www.researchgate.net/publication/248380891_captcha_dataset)

对于基于 OCR 的机器学习算法来说，这是一个很好的数据集。

以下是从图像数据中获取结果的步骤。

1.  阅读文件

![](img/92e77f5fc5371f826d17d1e7d9b3be5a.png)![](img/1a509cba97b11f1f648dc3c47874ce77.png)

2)获取验证码中的唯一字符并分配标签。

![](img/cae65a9e2ef19126f84b1d161697d56f.png)![](img/8a10e585bbf68a32e8cbebf2f4c83de2.png)

3)使用 OpenCV 进行分词和灰度缩放

![](img/5272ea69436d2a34f0ff79c4986b87c7.png)

4)模糊和阈值处理

**自适应阈值** →处理照明中出现的空间变化(根据像素强度的差异，帮助从背景中移除所需的前景图像对象。

**OTSU →** 这包括迭代通过阈值，并计算阈值两侧像素级别的扩散度，即落在前景或背景中的像素

![](img/14948d127abb7a6ac49eac977b67bf75.png)

5)形态转化

![](img/cd44a07b9870d55a90a131cd2dda99dc.png)

6)侵蚀和膨胀

![](img/2194839171aeae80d33892f784403f78.png)![](img/d78c9b8211b1989aff4d3634ace9d729.png)![](img/1fcada318aacf58b12b61f64b75e583e.png)

7)限制每个字符

![](img/f0f07db9d38cb4f6c1b96336b5e3b596.png)

8)拆分数据以进行训练和测试

![](img/972eebf6511c4f38af19e3394d7dc7fc.png)![](img/4fcbc593d7499d65bab7206293ea90d2.png)![](img/8908bcdb8ec39b7fd49b102b881a798a.png)![](img/796f88d00d380ae3cdcf74b611f46e33.png)![](img/9a84934b561cfe10d520f7a04753f136.png)![](img/796f88d00d380ae3cdcf74b611f46e33.png)

学习 15 个时代

![](img/6fe4ffdf41098e075cf931dda0567ac6.png)

显示每个字符预测结果的混淆矩阵

![](img/8a474dec324131e9a02c5cb04504b41d.png)

应用卷积神经网络预测验证码

![](img/15ff67a26a33d96e205c91cab52c0ba4.png)![](img/a65e1324eae0fc66eaff3d78475e140d.png)

连续的历元提供了更好的准确度

![](img/e793ff4ab790a82e5545a37a9b4a3c4f.png)

绘制验证码的图形以查看字符的预测结果。

![](img/29fc25c1d553a0f13ed3bcf9eabb794f.png)

预测验证码的绘图结果(能够预测)

![](img/41816efde4ec0e8ca2e087cccd863a84.png)

预测验证码的准确率约为 93%。

**参考文献**

1.  dataset-[https://www . research gate . net/publication/248380891 _ captcha _ dataset](https://www.researchgate.net/publication/248380891_captcha_dataset)
2.  分词-[https://towards data science . com/fast-word-Segmentation-for-noise-text-2c 2c 41 f 9 E8 da](https://towardsdatascience.com/fast-word-segmentation-for-noisy-text-2c2c41f9e8da)
3.  阈值处理-[https://www . research gate . net/publication/228350447 _ OCR _ based _ thresholding](https://www.researchgate.net/publication/228350447_OCR_based_thresholding)
4.  [https://medium . com/spinor/a-straight-introduction-to-image-thresholding-using-python-f1 c 085 f 02d 5e](/spinor/a-straightforward-introduction-to-image-thresholding-using-python-f1c085f02d5e)
5.  包围盒-【https://nanonets.com/blog/deep-learning-ocr/ 