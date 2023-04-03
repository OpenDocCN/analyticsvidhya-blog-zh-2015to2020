# CNN 中的卷积、填充、步幅和池化

> 原文：<https://medium.com/analytics-vidhya/convolution-padding-stride-and-pooling-in-cnn-13dc1f3ada26?source=collection_archive---------0----------------------->

# **卷积运算**

卷积是一种数学运算，用于从图像中提取特征。卷积由图像核定义。图像内核只不过是一个小矩阵。很多时候，一个 3x3 的核矩阵是很常见的。

在下图中，绿色矩阵是原始图像，黄色移动矩阵称为核，用于学习原始图像的不同特征。内核首先水平移动，然后下移，再移动…