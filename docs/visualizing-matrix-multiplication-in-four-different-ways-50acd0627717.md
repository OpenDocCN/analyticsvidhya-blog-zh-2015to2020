# 以四种不同的方式可视化矩阵乘法

> 原文：<https://medium.com/analytics-vidhya/visualizing-matrix-multiplication-in-four-different-ways-50acd0627717?source=collection_archive---------18----------------------->

![](img/ee5889b2a28d1783bf79d61129840cf3.png)

图片来源:谷歌

如果你正在从任何一本教科书中学习机器学习理论，我敢打赌你在某个时候已经在矩阵乘法中困惑了。当一个人不理解或不虔诚地执行常规乘法来理解步骤之间的内在逻辑时，感觉非常糟糕。

以下四种方法肯定会帮助你减少学习涉及矩阵乘法的理论的工作量:

遵循的矩阵符号是:

![](img/6c1e465b50ebdb37289e4945aa3dd1b9.png)![](img/e9d3db74df734d8ca4921a818b8685c1.png)

这里 c 代表列，r 代表行

# **1。单独的行/列和矩阵**

![](img/4c5abc3c77216c93e626b09a6dd58a2e.png)

AB 的列= A x 的相应列

AB 的行= A x B 的相应行

# 2.矩阵的行/列的线性组合

![](img/884d5f9d360c71cb453fc1efa1e76254.png)

为了更好地理解，请一次查看 AB 的每一列/行。

![](img/21f29db16e5dc1be132f625fbd6b1929.png)

AB 的列= A 的列的线性组合，系数作为 B 的对应列

AB 行= B 行的线性组合，系数为 A 的相应行

# 3.矩阵的线性组合

![](img/73ca1ffede70f6d6bd7d1344b22a5a6a.png)

注意， ***a_ci*** 和 ***b_ri*** 的相乘给出了一个形状为 ***(m x s)*** *的矩阵。*

# 4.传统矩阵乘法

![](img/fc39e244d01d0f195e791049064b7c9a.png)

# **例如:**

让

![](img/4244ab27a9b8217b032462a6f7de1bce.png)

然后通过矩阵的线性组合计算 AB

![](img/acc4ee764d52dbf920a94d3551940c7b.png)

另外，请注意使用前两种方法对 AB 的第一列进行的计算

![](img/68e2a9160cd57af3f4e744891dbdb0d1.png)

类似地，可以使用前两种方法计算 AB 的第一行

![](img/b6adfeaf651e82b616ce23fed967bccb.png)

我希望，这种看待矩阵乘法的新方法能启发对数学证明的理解。

如果你开始喜欢矩阵乘法，请评论！

**检查第 2 部分**

[](/analytics-vidhya/visualizing-matrix-multiplication-in-four-different-ways-part-2-ed96cea120c1) [## 以四种不同方式可视化矩阵乘法—第 2 部分

### 如何在头脑中进行矩阵乘法运算？可视化矩阵乘法。matrix 背后的直觉是什么…

medium.com](/analytics-vidhya/visualizing-matrix-multiplication-in-four-different-ways-part-2-ed96cea120c1)