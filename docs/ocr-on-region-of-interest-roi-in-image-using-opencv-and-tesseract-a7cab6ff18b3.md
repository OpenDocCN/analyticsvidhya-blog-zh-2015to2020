# 使用 OpenCV 和 Tesseract 对图像中感兴趣区域进行光学字符识别

> 原文：<https://medium.com/analytics-vidhya/ocr-on-region-of-interest-roi-in-image-using-opencv-and-tesseract-a7cab6ff18b3?source=collection_archive---------0----------------------->

![](img/489b7e3930fed00f62243d7aa42507ff.png)

安妮·斯普拉特在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的照片

在本帖中，我们将使用 OpenCV 对图像的选定区域应用 OCR。在这篇博客结束时，你将能够在输入图像上应用**自动方向校正**，**选择感兴趣的区域**，**对所选区域**应用 OCR。

这篇博客基于 Python 3.x，我假设你已经安装了 Pytesseract 和 OpenCV