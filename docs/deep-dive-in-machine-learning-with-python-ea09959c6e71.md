# 使用 Python 深入研究机器学习

> 原文：<https://medium.com/analytics-vidhya/deep-dive-in-machine-learning-with-python-ea09959c6e71?source=collection_archive---------14----------------------->

## 第十四部分:初始数据分析(IDA)及实例

![](img/02bd40b78f1ac01681fb5e136be81226.png)

图片[链接](https://www.finereport.com/en/data-analysis/comparison-of-data-analysis-tools-excel-r-python-and-bi.html)

欢迎来到另一个深入研究 Python 机器学习的博客，在上一个 [**博客**](/analytics-vidhya/deep-dive-in-machine-learning-with-python-64bcbe0b1b40) 中，我们将触及不同层次数据分析的基础。

在今天的博客中，我们将通过使用**自闭症谱系障碍**(儿童)数据集来深入研究初始数据分析(IDA)的核心概念。感谢数据集创建者和 [**UCI ML 知识库**](https://archive.ics.uci.edu/ml/datasets/Autistic+Spectrum+Disorder+Screening+Data+for+Children++) 提供此数据集。

## 导入所需的 python 库

![](img/5d18a34730a16cd34646fd265ee20f23.png)

**Python 包**

## 加载 ARFF(属性关系文件格式)数据集文件

![](img/fc4aae61c98ecc75483c283f6080213c.png)

**加载的 ASD 数据集**

# 步骤 1:更改字符编码

如果你参考上面的图像(即 ***加载 ASD 数据集*** )，那么你会发现**‘b’**与每个数据值相关联。这意味着数据是以字节为单位的，因此我们需要改变字符编码。

![](img/925ad2d2425fa445520db02781f14499.png)

**字符编码**

## 浏览数据集

![](img/2609b2c2a6b6f98e8dbda08e99aaa9bc.png)

在这里，我们了解了数据集中要素的数量，并显示了前 5 条记录。

# 步骤 2:数据类型处理

![](img/76273d6c17553ceda29d0cfd21ff1cb1.png)

**特征数据类型**

## 步骤 2.1:“年龄”转换为 dtype“INT”

![](img/9094b8b57c6a00a8b38376dfa103ea3d.png)

因此，在将' **AGE** '变量的数据类型从 FLOAT 转换为 INT 之前，我们需要填充它的空值。因此，用 0 替换空值，然后处理这 4 条记录。

![](img/f04d059fb9eac46f7057e10792106d9e.png)

**填充‘年龄’中的空值**

![](img/03a7aa7f28f2db79ef0d0749f78654a2.png)

**年龄数据类型转换为整数**

## 步骤 2.2:将“性别”标记为 dtype“INT”(1 表示 m(即男性)，0 表示 f(即女性))

![](img/81cf65b0d66234a5c99e1f81db302640.png)

**男性和女性的数量**

![](img/62d4b8e423081cd454b2d425c384d380.png)

**性别编码为 0 和 1**

## 步骤 2.3:将“先天性黄疸”标记为 dtype“INT”(1 对应于“是”，0 对应于“否”)

![](img/9c7814f463a3db0dec35946f5e5e91ff.png)

**贴标签前**

![](img/90082a540f0a5d924c1ff0ef68ba78b6.png)

**贴标签后**

## 步骤 2.4:将“FAMILY_MEMBER_WITH_PDD”标记为 dtype“INT”(1 对应于“yes”，0 对应于“no”)

![](img/892add221c6a4200e20a44820abbf188.png)

**贴标签前**

![](img/460591b4053b0c9593c25619b9ae5824.png)

**贴标签后**

## 步骤 2.5:将“USED_SCREENING_APP_BEFORE”标记为 dtype“INT”(1 表示“是”，0 表示“否”)

![](img/03e4f041c41384b718d632c0e705a931.png)

**贴标签前**

![](img/5ea98e32ad7b3e0f7debd5b278e30178.png)

**贴标签后**

## 步骤 2.6:将“筛选问题”变量的数据类型转换为“INT”

![](img/d854fe90675ab53411717405d01b70d3.png)

## 步骤 2.7:将“SCREENING_SCORE”标记为 dtype“INT”

![](img/5895084c4917a1623232575779bb0a3f.png)

**数据类型转换前**

![](img/ef8138e10480553d06ff81a3e0c93b27.png)

**数据类型转换后**

## 步骤 2.8:将“ASD_Label”标记为 dtype“INT”(1 对应于“yes”，0 对应于“no”)

![](img/2836b0dcf535ad1f1cd741b5e8c9d966.png)

**贴标签前**

![](img/677d079c502f5b33bdbd6562ac00c24a.png)

**贴标签后**

## 步骤 2.9:标准化“WHOS _ 完成 _ 测试”的数据

![](img/8bd0a8f8f7d4942f59af7a8fa2229532.png)

**标准化前**

![](img/d17031f1702d8cf560760105e81a8360.png)

**标准化后**

# 第一手清洁数据框

![](img/197c2a3059b016403bee6622f313c05b.png)![](img/bf6cf98c8f7125f3620111a03106296f.png)![](img/afa7ad3cc1a3b82f97ef53f51e3ec743.png)

礼貌 **WWE** 和**新的一天**

恭喜你，我们的博客到此结束。总而言之，我们涵盖了**初始数据分析(IDA)** 的前两个阶段。

请关注我们即将发布的帖子，我们将努力填补'**种族**和'**WHOS _ 完成 _ 测试**'中缺失的值。并且，建立我们的第一个机器学习回归模型来预测' **AGE** '中的缺失值。

> 如果你想下载这个博客的 Jupyter 笔记本，请访问下面的 GitHub 库:
> 
> [**https://github.com/Rajesh-ML-Engg/Autism_Spectrum_Disorder**](https://github.com/Rajesh-ML-Engg/Autism_Spectrum_Disorder)

***谢谢大家，学习愉快！！***

***博客-15:*** [***初始数据分析-二***](/@Rajesh_ML_Engg/deep-dive-in-machine-learning-with-python-4d4d8ab37f07)