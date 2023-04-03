# 使用 Python 和 OpenCV 进行高斯模糊

> 原文：<https://medium.com/analytics-vidhya/gaussian-blurring-with-python-and-opencv-ba8429eb879b?source=collection_archive---------0----------------------->

# 介绍

在这里，我们将讨论图像噪声，如何将它添加到图像中，以及如何使用 OpenCV 通过高斯模糊来最小化噪声。我们有一点数学要复习，但是用 Python 实现它并不完全必要。如果你只是想要代码，跳到底部找一个 TL；博士

# 高斯模糊

在图像处理中，高斯模糊被用于减少图像中的噪声量。用 OpenCV 在 Python 中实现图像的高斯模糊非常简单…