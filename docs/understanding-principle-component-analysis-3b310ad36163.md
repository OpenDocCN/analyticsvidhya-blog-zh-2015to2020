# 理解主成分分析

> 原文：<https://medium.com/analytics-vidhya/understanding-principle-component-analysis-3b310ad36163?source=collection_archive---------20----------------------->

![](img/8fbb24ca6aff7ac18d894bcdb2ce5080.png)

法国普罗旺斯凡尔登峡谷

主成分分析(PCA)广泛应用于机器学习和数据科学。PCA 在低维空间中找到模型数据的表示，而不会丢失大量信息。这种数据压缩过程可用于数据可视化和分析，以及加速机器学习算法和流水线。

让我们了解 PCA 是如何工作的，我们有了模型的数据:

![](img/0a8cfb3d93cf2f6fe719d659f34396f7.png)

在哪里

![](img/39113f4134750f07e0f93c74d0262023.png)

该模型有 k 个数据点，每一个都是一个 l 维向量，我们希望在 m 维的低维空间中找到数据的表示

We start by normalising the data across the l dimensions, for every dimension j we compute the mean across the data points:

![](img/32716620c7ef700a26c123943f8204d2.png)

And for every dimension j and data point i instead of taking:

![](img/bc961c7a08fffb6652b00a7d0a7fb4ba.png)

we take the normalised value:

![](img/87c04b5b2e9c0932e99626a0b7b4702e.png)

For simplicity we continue using the original notation:

![](img/0a8cfb3d93cf2f6fe719d659f34396f7.png)

for the normalised data points.

We define the data matrix X, the columns are our data points and the rows hold the l dimensions of the data:

![](img/578fc604f9762cacc1b05542d38636c8.png)

For every pair of dimensions i,j =1,2,…,l we compute the covariance:

![](img/16eb8723b846c82bac59d8b364000729.png)

and define the data covariance matrix COV(X):

![](img/e997c6310a0f203d38d06e47ee66c1ef.png)

For example entry (1,2) is computed by:

![](img/b3775969e8d0a1e21dd907dfa444331a.png)

using the mean values of the first and second dimensions:

![](img/273f364f8e496b24f851eb4ddd2c7bd8.png)

Notice that COV(X) is symmetric since for every i,j=1,2,…,l we have:

![](img/b99ea8e90afca0d08a2aa60be0bfee35.png)

Next we compute the eigenvalues:

![](img/e0425a8b857b9204b1fec4b66693240e.png)

and eigenvectors of the covariance matrix:

![](img/eeebb1305ba4f9a091ab59fea4d2f84d.png)

where

![](img/a3567027e9f0e29d75341ad61fe26f09.png)

**，这些特征向量被称为数据的主分量，代表数据中显著方差的方向**，主分量 e1 在方差最大的方向，主分量 e2 在第二大的方向，依此类推。

例如，假设我们的数据具有以下形式:

![](img/876d3b736dcb22890d88a43627507dff.png)

那么主要分量 e1、e2 将是:

![](img/51aea71efe2c886f51f5d4345e796a7f.png)

可以看出，e1 捕获数据中最大变化量的方向，e2 捕获第二大变化量的方向。

协方差矩阵的特征值也包含关于数据的有价值的信息，它们代表数据在主分量方向上的方差。

在下图中，我们在主分量 e1、e2 的方向上有一对向量 v1、v2。相应地，可以看出，v1、v2 的幅度捕捉到了数据的扩展。

![](img/5b09f2b5c75fcce9fb573210bdb18a49.png)

现在我们有了协方差矩阵的特征值和特征向量，我们可以选择一个较低的维数 m < 1，并定义特征向量矩阵 E。E 的行是捕获数据中最大方差的 m 个特征向量，通过取对应于 m 个最大特征值的特征向量来选择这些特征向量。

![](img/55a04b7408d93a2f97b281443b417c5b.png)

我们将 X 乘以 E，并将我们的数据从 l 维空间投影到由 m 个主分量构成的较低的 m 维空间

![](img/2351e9e8c40fcc0a91ee441b78fd8245.png)

为了便于记记，我们写下

![](img/2c68aadb343a395139c47a0a191fca07.png)

通过忠实于我们最初的符号，我们可以把 EX 写成

![](img/7941806bc9a17f40a2c83fe5bef7b28f.png)

我们的多维数据集是

![](img/19dabe325ef720438a8360dd0a7745bb.png)

在哪里

![](img/4c1837203d5cd644a1d2e83b9f664df8.png)

我们完事了。

请注意，对于每个 i=1，2，…，k，矩阵 EX(1)中的列 I 包含数据点 xi 在 m 个主分量方向上的投影，我们在矩阵 EX(2)中的符号捕捉到了这一点，k 个 m 维数据点是列，m 个主分量在行中表示。