# 机器学习的矩阵演算

> 原文：<https://medium.com/analytics-vidhya/matrix-calculus-for-machine-learning-e0262f0eaa8e?source=collection_archive---------0----------------------->

由于机器学习处理更高维度的数据，用一元和二元微积分知识理解算法既麻烦又缓慢。如果有人问 *x* 的导数，你会毫不犹豫地说出它的 2 *x* ，而不用运用第一原理——可微性的定义。在这里，我将提供一些技巧和窍门来执行矩阵计算，就像 *x .* 的微分一样

![](img/37429d20c21bd4c31e90d37b34ed968b.png)

矩阵计算

# **矩阵演算的符号**

在学习阅读任何语言之前，了解字母是很重要的，对吗？所以拜托，花点时间去掌握下面的记法，这样会更容易构造单词(恒等式)，然后是句子(更复杂的方程)。

![](img/c4e6de218be75bdb15ea573b63f70cc7.png)

向量符号中的多元函数

**注:粗体变量是向量**。

![](img/83547b885deaaa17eeb85132bd5563aa.png)

f 作为 m 个函数的向量

# **6 种常见的矩阵导数**

![](img/dc99fc19ec0f39269193e373666b8419.png)

没有提到矩阵与向量和矩阵的微分，因为它可能产生二维以上的矩阵。

**矩阵微分的两种布局:分子布局和分母布局**

![](img/1d37491faf4b9c0e99755f0efb681948.png)

分子布局~分母布局

![](img/316719c9508a1cce254ed3550cb4d1dd.png)![](img/19c21cc00ded91cf4aa802771f9ea6ce.png)

注意矩阵的维数

![](img/4e67131de490743aacfd9e1a7de4c7fd.png)

变得困惑？让我为你简化它。

**经验法则:**分子布局的维度等同于分子和分母的转置。

分母布局的维度等同于分子的分母和转置。

现在，再次回到上面的等式来理解这个技巧。

为了进一步的讨论，我们将遵循分子布局，如果你遵循分母布局，然后只需转置结果，这将在大多数情况下工作。

# 重要关系

现在，我们一步一步地构建我们的基础，这样在本文结束时，您就可以像玩魔术一样进行矩阵运算了！

**恒等式 1:** 可变矢量与恒定矢量

![](img/8bd49e93590c3e2281cc6d541aa05936.png)

**恒等式 2:** 变量向量与常数矩阵

![](img/c03e0da15d6ae266dd08c4846e83d5b7.png)

如果你不仅仅是通过看方程来理解的话，请访问我的博客查看矩阵乘法的可视化:

[](/analytics-vidhya/visualizing-matrix-multiplication-in-four-different-ways-50acd0627717) [## 以四种不同的方式可视化矩阵乘法

### 如果你正在从任何教科书中学习机器学习理论，我敢打赌你在某些时候对矩阵乘法感到困惑…

medium.com](/analytics-vidhya/visualizing-matrix-multiplication-in-four-different-ways-50acd0627717) 

**恒等式 3:** 紫外线倍增法则

1.  对于矢量

![](img/9a3593fb78fb94b8d2e0afd0a97a8caf.png)

2.对于矩阵

![](img/6cc0335a557323e1c894282153bcaf73.png)

**恒等式 4:** 复合函数的微分

![](img/5d05e0127afdb61f8d017488d11ba558.png)

**恒等式 5:** 矩阵的逆矩阵的微分

![](img/15912b5cc24eff2f341a6e314b7b3af5.png)

**恒等式 6:** 矩阵行列式的微分

1.  **按标量**

![](img/2d0a3e8c215698c45d1dd1f9659d0fdb.png)

它类似于标量微分

2.**通过矢量**

将上述结果与标量和向量的微分结果相结合。

![](img/676664f5ae6de89cc21e081cd20a6960.png)

3.**通过矩阵**

将上述结果与标量与矩阵微分的结果相结合。

![](img/3315fe6a7a6157d3ffee4c92814a531d.png)

**推论**

**行列式与矩阵本身的微分**

![](img/19c56d087a2478397854951018df5146.png)

注意转置矩阵是如何再次转置的

以上三个结果可能只高出一个公式:

![](img/eb2334ff4b66894e32a00f1d0e13a054.png)

**恒等式 7:** 矩阵的微分痕迹

我们可以使用如上所示的单个表单来表示所有这三种情况

![](img/2413b028505260c652b1fdba21435eb7.png)

# **一些重要且常用的结果**

**带变量向量的二次表达式**

![](img/0d10ce50c3bc58a487c92c9603de5272.png)

如果 A 是对称矩阵，那么，

![](img/a17fc35f96a6f64a04433bde4115cc20.png)

**带有 x 转置的线性表达式**

![](img/1399201da6b063541d8f70a0514b498e.png)

# **与机器学习相关的示例**

我们知道逻辑回归使对数损失函数最小化。现在，为了实现梯度下降，我们需要对数损失 w.r.t 参数向量的微分。

![](img/312c0be803fa65a5b8faea5771f8fdab.png)

**乙状结肠功能的微分**

![](img/c0342012987cb163508bdcbbf431b7f6.png)

**测井曲线损失 w.r.t .参数向量的微分**

![](img/6fd7564946dc5c438ab3f0745eaf4a0e.png)

不确定对角线是如何进入画面的？让我展示给你看。

![](img/279eae7800dfea745a7b079a9582b8a7.png)

这可以简化为

![](img/fd7e223524ef0531d89f447bff1b15d7.png)

人们可以通过记住以下逻辑来避免这种冗长繁琐的想象:

**元素式函数的微分可以像标量一样进行**

![](img/6883c0bcd91a6341ab7c186ba456f2e6.png)

**练习**

尝试对数损失函数的二阶导数。

![](img/d7e59f8b2ba8fb2dc48635e27b99b538.png)

**提示:**

你需要以下身份来快速完成。甚至对数损失函数的一阶导数也可以简单地通过使用**哈达玛乘积(逐元素乘积)**的恒等式来完成

**奖金身份:**

![](img/03c871b768aa597976bcee6e49986744.png)

如果你觉得上面的文章有用，请分享对我作品的喜爱，它激励我为你写更多高质量的内容:)

**参考文献:**

1.  [https://medium . com/analytics-vid hya/visualizing-matrix-multi-in-four-different-ways-50 ACD 0627717](/analytics-vidhya/visualizing-matrix-multiplication-in-four-different-ways-50acd0627717)
2.  矩阵微分:Randal J. Barnes 美国明尼苏达州明尼阿波利斯大学土木工程系
3.  深度学习需要的矩阵演算:特伦斯·帕尔和杰瑞米·霍华德
4.  黑客帝国食谱:[http://matrixcookbook.com](http://matrixcookbook.com)
5.  hada mard _ product:h[ttps://en . Wikipedia . org/wiki/hada mard _ product _(matrix)](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))