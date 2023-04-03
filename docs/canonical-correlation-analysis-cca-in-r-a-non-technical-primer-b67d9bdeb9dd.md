# R 中的典型相关分析(CCA ):非技术入门

> 原文：<https://medium.com/analytics-vidhya/canonical-correlation-analysis-cca-in-r-a-non-technical-primer-b67d9bdeb9dd?source=collection_archive---------3----------------------->

![](img/72d8c595baa0d04c63b12813caafccd8.png)

照片由[克莱门特·H](https://unsplash.com/@clemhlrdt?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄

在这篇文章中，介绍了使用 R 统计编程环境的典型相关分析(CCA ),并对结果输出进行了相关解释。

**什么是典型相关分析？**

典型相关分析是一类多元统计分析技术，用于同时分析对象(相似实验单元)的多个测量值。

CCA 和 PCA 之间有很小的区别；

*   CCA 用于同时关联在相同实验单元上测量或观察的几个度量因变量和几个度量自变量。
*   另一方面，主成分分析通常通过一些初始变量的线性组合来减少单一数据集的数据维数。
*   该技术是多重相关分析的扩展，通常适用于多元回归分析方法适用的相同情况。
*   在我们开始 R 中的分析之前，让我们回顾一些与典型相关分析相关的概念。

**典型相关分析的目的**

1.  ***数据简化***
    典型相关分析提供了一种利用这些变量的
    线性组合来解释两组变量之间关系的方法。

****典型相关的性质****

**典型相关对于响应变量( *y 的*)和解释变量( *x 的*)的尺度变化是不变的。
换句话说，在分析中改变两组感兴趣变量的测量尺度，例如，从英寸到厘米，不会干扰随后的典型相关性。

同样，第一典型相关( *r₁* )是 *Y* 和 *X* 的线性函数之间的最大相关。
换句话说， *r₁* 超过了任何$Y$与任何 *X* 之间的(绝对)简单相关，或者任何 *Y* 与所有 *X 的*之间的多重相关，或者任何 *X* 与所有 *Y 的*之间的多重相关。**

****使用 R 的典型相关分析****

**现在，让我们看看如何在 r 中执行 CCA。**

**在本节中，我们将使用从 [UCI 机器学习库](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)中获得的红酒质量数据集。初始数据集共有 12 个变量的 1599 个观察值。数据仅用于说明目的。**

**数据被清除了 5 个变量，剩下的变量被分成两组(在这里获得数据[)；](https://github.com/okutse/data/blob/master/trial_data.csv)**

*   **set 1( *ide* )由 3 个变量`chlorides`、`free.sulfur.dioxide`和`total.sulfur.dioxide`组成**
*   **set 2 ( *酸度*)由 4 个变量`fixed.acidity`、`volatile.acidity`和`density`组成**

**在这个分析中，我们的兴趣是确定在解释两组变量之间的关联时有意义的维数(规范变量)。**

**接下来，我们加载 R 中的数据，分成 2 组；**

```
#loading the data into R;
pole=read.csv("C:/Users/Amos Okutse/Desktop/trial_data.csv", header=**TRUE**)
ide<-pole[,1:3]
acidity<-pole[,4:7]
```

**接下来，我们加载所有需要的包以便于分析，并使用`CCA`包中的`matcor()`来检查集合间和集合内的关联。**

```
#loading the required packages
**library**(lme4)
**library**(CCA) #facilitates canonical correlation analysis
**library**(CCP) #facilitates checking the significance of the canonical variates
#checking the between and within set associations
cormat<**-matcor**(ide,acidity)
#extracting the within study correlations for set 1 and set 2 and #between set cor
round(cormat$Ycor, 4)
                 fixed.acidity volatile.acidity citric.acid density
fixed.acidity           1.0000          -0.2561      0.6717  0.6680
volatile.acidity       -0.2561           1.0000     -0.5525  0.0220
citric.acid             0.6717          -0.5525      1.0000  0.3649
density                 0.6680           0.0220      0.3649  1.0000round(cormat$Xcor, 4)
chlorides free.sulfur.dioxide total.sulfur.dioxide
chlorides               1.0000              0.0056               0.0474
free.sulfur.dioxide     0.0056              1.0000               0.6677
total.sulfur.dioxide    0.0474              0.6677               1.0000
```

**这两个集合之间的关联可以提取为:**

```
#between set associations
 cormat<-matcor(ide,acidity)
 round(cormat$XYcor, 4)##                      chlorides free.sulfur.dioxide total.sulfur.dioxide
 ## chlorides               1.0000              0.0056               0.0474
 ## free.sulfur.dioxide     0.0056              1.0000               0.6677
 ## total.sulfur.dioxide    0.0474              0.6677               1.0000
 ## fixed.acidity           0.0937             -0.1538              -0.1132
 ## volatile.acidity        0.0613             -0.0105               0.0765
 ## citric.acid             0.2038             -0.0610               0.0355
 ## density                 0.2006             -0.0219               0.0713
 ##                      fixed.acidity volatile.acidity citric.acid density
 ## chlorides                   0.0937           0.0613      0.2038  0.2006
 ## free.sulfur.dioxide        -0.1538          -0.0105     -0.0610 -0.0219
 ## total.sulfur.dioxide       -0.1132           0.0765      0.0355  0.0713
 ## fixed.acidity               1.0000          -0.2561      0.6717  0.6680
 ## volatile.acidity           -0.2561           1.0000     -0.5525  0.0220
 ## citric.acid                 0.6717          -0.5525      1.0000  0.3649
 ## density                     0.6680           0.0220      0.3649  1.0000
```

**在下一段代码中，我们将获得典型相关，然后从中提取原始的典型系数。然后将对原始规范系数进行解释。**

```
#obtaining the canonical correlations
 can_cor1=**cc**(ide,acidity)
 can_cor1$cor## [1] 0.45361635 0.20703957 0.06092621
```

**原始标准系数可以从`can_cor1`中获得:**

```
#raw canonical coefficients
 **can_cor1**[3:4]## $xcoef
 ##                              [,1]         [,2]         [,3]
 ## chlorides            -15.40227312  6.022271042 -13.39830038
 ## free.sulfur.dioxide    0.04039044 -0.081896302  -0.09040281
 ## total.sulfur.dioxide  -0.02578993 -0.004526871   0.03142581
 ## 
 ## $ycoef
 ##                          [,1]         [,2]          [,3]
 ## fixed.acidity       0.6300078    0.7382125    0.04156297
 ## volatile.acidity   -3.7269496    2.7250334    5.25358495
 ## citric.acid        -6.6286891    0.6789206    0.75975488
 ## density          -382.1226185 -319.7329865 -344.27947229
```

**原始规范系数的解释遵循类似于线性回归模型系数的解释的方式。**

**例如，考虑一组变量`acidity`。假设我们想要解释`fixed.acidity`对正在讨论的集合的第一个规范变量的影响，解释如下:
当其他变量保持不变时，`fixed.acidity`增加一个单位将导致`acidity`变量集合的第一个规范变量的值增加 0.63 个单位。
此外，`volatile acidity`将导致酸度变量组的第二维增加约 2.72 个单位。**

**在下面的代码块中，我们将实现`comput`函数来计算变量和规范变量之间的相关性(以及规范维度上的变量负载)。**

**通常，规范维数与较小集合中的变量数相同。然而，在解释两组变量之间的关系时有意义的规范维数可能小于较小数据集中的变量数。在这种情况下，有 3 个维度。**

```
**#computes the canonical loadings**
 **can_cor2**=**comput**(ide,acidity,can_cor1)
 **can_cor2**[3:6] **#displays the canonical loadings**## $corr.X.xscores
 ##                            [,1]       [,2]       [,3]
 ## chlorides            -0.7627757  0.2716167 -0.5868540
 ## free.sulfur.dioxide  -0.1479687 -0.9544958 -0.2589267
 ## total.sulfur.dioxide -0.6006467 -0.7074330  0.3725079
 ## 
 ## $corr.Y.xscores
 ##                        [,1]       [,2]        [,3]
 ## fixed.acidity    -0.0368851 0.17516149 -0.03066069
 ## volatile.acidity -0.1137480 0.01498495  0.05033043
 ## citric.acid      -0.2036616 0.10471705 -0.03413442
 ## density          -0.2151756 0.06505414 -0.03208948
 ## 
 ## $corr.X.yscores
 ##                             [,1]       [,2]        [,3]
 ## chlorides            -0.34600754  0.0562354 -0.03575479
 ## free.sulfur.dioxide  -0.06712102 -0.1976184 -0.01577542
 ## total.sulfur.dioxide -0.27246318 -0.1464666  0.02269549
 ## 
 ## $corr.Y.yscores
 ##                         [,1]       [,2]       [,3]
 ## fixed.acidity    -0.08131343 0.84602907 -0.5032430
 ## volatile.acidity -0.25075819 0.07237725  0.8260885
 ## citric.acid      -0.44897316 0.50578277 -0.5602585
 ## density          -0.47435584 0.31421115 -0.5266942
```

**为了获得尺寸的统计显著性，我们将使用加载的`CCP`包。代码和输出如下:**

```
**#test of canonical dimensions**
 rho=can_cor1$cor
**##defining the number of observations, no of variables in first set,
 #and number of variables in second set**
 n=**dim**(ide)[1]
 p=**length**(ide)
 q=**length**(acidity)
 **##Calculating the F approximations using different test statistics
 #using wilks test statistic**
 **p.asym**(rho,n,p,q,tstat="Wilks")## Wilks' Lambda, using F-approximation (Rao's F):
 ##               stat    approx df1      df2      p.value
 ## 1 to 3:  0.7573653 38.878017  12 4212.328 0.000000e+00
 ## 2 to 3:  0.9535817 12.770396   6 3186.000 2.675637e-14
 ## 3 to 3:  0.9962880  2.969489   2 1594.000 5.161358e-02
```

**以上结果来自维尔克检验统计。根据所实施的测试统计，这些结果可能略有不同。**

**在上面的输出中，第一个测试确定从 1 到 3 的组合维度是否显著。由于 p 值小于α = 0.05 的显著性水平，因此所有 3 个维度都具有统计学显著性(F= 11.72，p = .00)。**

**类似地，第二个测试确定维度 2 和维度 3 组合的显著性。自 p <0.05, it follows that the dimensions are statistically significant.**

**Lastly, the last test determines the significance of the 3ʳᵈ dimension, which is not statistically significant in this case since p> 0.05。**

****使用 R 计算标准化规范系数****

**当变量之间的标准偏差存在较大差异时，最佳做法通常是执行标准化程序，以帮助或简化变量之间的比较。**

**在 R 中，标准系数的标准化过程如下进行；**

```
#standardizing the first set of canonical coefficients(ide)
 std_coef1<-**diag**(**sqrt**(**diag**(**cov**(ide))))
 std_coef1**%*%**can_cor1$xcoef##            [,1]       [,2]       [,3]
 ## [1,] -0.7249126  0.2834400 -0.6305951
 ## [2,]  0.4224903 -0.8566482 -0.9456276
 ## [3,] -0.8483682 -0.1489129  1.0337622
```

**标准化第二组标准系数；**

```
##standardizing the coeficents of the second set (acidity)
 std_coef2<-**diag**(**sqrt**(**diag**(**cov**(acidity))))
 std_coef2**%*%**can_cor1$ycoef##            [,1]       [,2]        [,3]
 ## [1,]  1.0969042  1.2852991  0.07236513
 ## [2,] -0.6673465  0.4879437  0.94070537
 ## [3,] -1.2912762  0.1322545  0.14800111
 ## [4,] -0.7211930 -0.6034429 -0.64977034
```

****解释标准化的标准系数****

**该系数的解释遵循标准化回归系数的解释。例如，在变量的`acidity`集合中，当模型中的所有其他变量保持不变时，`fixed.acidity` 值增加一个单位将导致第一个规范变量的标准偏差增加 1.096 个单位。**

****使用典型相关时的注意事项****

*   **所涉及的样本量应该相对较大。该技术最适用于较大的样本量。**
*   **变量应该遵循多元正态分布，因为典型相关假设 X 和 Y 的联合分布是多元正态的。**
*   **所用的样本应能代表相关人群。**

****延伸阅读:****

**[1] Afifi，a .，May，s .，和 Clark，V. A. (2003 年)。*计算机辅助多元分析*。CRC 出版社。**

**[2] González，I .，Déjean，s .，Martin，P. G .，和 Baccini，A. (2008 年)。CCA:扩展典型相关分析的 R 包。*统计软件杂志*， *23* (12)，1–14。**

**[3]科尔特斯，p .，塞尔代拉，a .，阿尔梅达，f .，马托斯，t .，和赖斯，J. (2009 年)。通过物理化学特性的数据挖掘建立葡萄酒偏好模型。*决策支持系统*， *47* (4)，547–553。**