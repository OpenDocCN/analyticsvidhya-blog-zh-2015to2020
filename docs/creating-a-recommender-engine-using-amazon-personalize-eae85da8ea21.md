# 使用 Amazon Personalize 创建推荐引擎

> 原文：<https://medium.com/analytics-vidhya/creating-a-recommender-engine-using-amazon-personalize-eae85da8ea21?source=collection_archive---------10----------------------->

在这篇博客文章中，我将带你了解如何使用亚马逊提供的最新托管机器学习服务之一，名为“[个性化](https://aws.amazon.com/personalize/)”。

我用一些模拟数据来训练一个神经网络，当输入特定的用户 id 时，它最终会提供推荐的产品。

这篇博文将解释我的方法。让我们开始吧！

# 数据清理和准备

## 连接数据

提供的数据在 5 个单独的 CSV 中。为了创建一个大型 CSV，我使用了 Pandas 数据帧并执行了完全外部连接。只要两个表之间有一个匹配的属性，完全外连接就简单地接受所有属性的联合。注意，AWS Personalize 只接受一个
单一 csv 文件，所以这一步是必需的。

[https://gist . github . com/TD shah/78 ea a87 CBC 3505 de 05 f 18 da 6 D5 C9 e 29d . js](https://gist.github.com/tdshah/78eaa87cbc3505de05f18da6d5c9e29d.js)

Personalize 要求以下特性的名称与指定的名称完全一致，因此我重命名了相应的列，以匹配“USER_ID”、“TIMESTAMP”和“ITEM_ID”。这三个特征是使用 Personalize 训练的任何模型的最低要求，因为它们详细描述了 a)谁购买了该物品 b)购买了什么物品 c)购买该物品的时间。

## 时间戳转换

对于“时间戳”列，个性化需要 Unix 时间戳。我通过应用执行转换的 lambda 函数将所需的列转换为 Unix 时间。

[https://gist . github . com/TD shah/29 CB 3 f 6b 18 a 29 c 667 DD 5208 b 16 a 127 DD . js](https://gist.github.com/tdshah/29cb3f6b18a29c667dd5208b16a127dd.js)

## 其他数据转换

我有一列数据“是”和“否”，我通过应用执行转换的 lambda 函数将它们转换为 0 和 1。

[https://gist . github . com/TD shah/0782 d0 a9 A8 bb a9 c 37 e 940 A0 c 62 DCA 3c . js](https://gist.github.com/tdshah/0782d0a9a8bba9c37e940a0c62cdca3c.js)

接下来，我手工删除了对创建模型没有直接帮助的特性(主要是像 first_name，last_name 这样的字符串)。还有一个“zipcode”属性，我把它去掉了。邮政编码本来可以使用，但就其现状而言，它被视为一个连续变量，尽管它应该被视为分类变量(数字较高的邮政编码并不意味着什么)。要解决这个问题，您可以创建一个区域映射(例如，西海岸邮政编码=“1”)。中西部邮政编码= "2 "等。这个映射会将邮政编码转换成一个连续变量，因为您可以比较邮政编码
区域的数量。

# 模型准备

## 特征选择

我使用了 3 种不同的[方法](https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e)(单变量选择、特征重要性、相关矩阵)来执行特征选择，以查看哪些特征对模型最重要。

单变量选择-

[https://gist . github . com/TD shah/b 6 f 999 BF 94 f 646 DC 113 DD 0 ab 86 a6 df 86 . js](https://gist.github.com/tdshah/b6f999bf94f646dc113dd0ab86a6df86.js)

功能重要性-

[https://gist . github . com/TD shah/d 8 e 09 b 22 DC 85478783101 c0d 4 f 325 f 91 . js](https://gist.github.com/tdshah/d8e09b22dc85478783101c0d4f325f91.js)

相关矩阵-

[https://gist . github . com/TD shah/ce 79 df 7489 BFA 1 fee 63 c 27664 F2 BC 7 e 3 . js](https://gist.github.com/tdshah/ce79df7489bfa1fee63c27664f2bc7e3.js)

## 超参数优化

我在[亚马逊 SageMaker](https://aws.amazon.com/sagemaker/) 中使用超参数
调优工作执行了超参数优化(HPO)。我在 AWS Personalize 中选择的解决方案(机器学习模型)“hrnn 元数据”有 3 个超参数:recency_mask(布尔值)、hidden_dimension(值的范围)、bptt(值的范围)。HPO 有助于确定为超参数设置什么起始值，之后个性化可以根据需要执行自动调整来更改超参数。

# 训练模型

我将清理后的 CSV 上传到 S3 存储桶中，并创建了适当的存储桶访问策略+ IAM 角色，用于个性化访问亚马逊 S3 资源。

在亚马逊个性化控制台**，

1.  创建了一个数据集组，我将 CSV 数据集上传到其中。我选择了
    “用户-商品交互数据集组类型”，因为我的数据集包含用户信息和商品信息，我的最终目标是找到这些属性之间的关系&为特定用户提供产品推荐
2.  创建了一个新的 JSON 格式的模式，它与我的数据集中的属性相匹配
3.  创建了从提供的 S3 文件路径导入数据集的导入作业
4.  创建的解决方案(机器学习模型)。请参见上文所选的解决方案类型。一旦创建了解决方案，就可以捕获描述模型性能的各种度量:标准化的折扣累积、精度、平均倒数排名
5.  进一步的参数调整应该有助于提高模型的性能
6.  创建了一个活动，允许您通过输入用户 id 并获得推荐项目 id 的列表来测试训练模型

最后，可以通过 AWS Personalize
“get recommendations”API 调用推荐引擎。这就是将结果返回到 web 应用程序或其他用户界面的方式。

*   *这些步骤也可以通过[Python SDK(boto 3)I](https://docs.aws.amazon.com/personalize/latest/dg/aws-personalize-set-up-sdks.html)n 而不是控制台来执行

Tanmay Shah，黑客行业博客撰稿人