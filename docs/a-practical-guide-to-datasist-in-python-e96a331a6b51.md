# Python 数据列表实用指南

> 原文：<https://medium.com/analytics-vidhya/a-practical-guide-to-datasist-in-python-e96a331a6b51?source=collection_archive---------10----------------------->

用于数据分析和可视化的 Datasist。

![](img/e909acc5dd44a1a3b071884d4c48b684.png)

Datasist 是一个开源 python 库，它提供了用于简单数据分析、可视化以及有效构建和管理数据科学项目的函数和方法。

在本文中，我将向您介绍使用名为“Datasist”的 python 库进行数据分析、数据可视化和解释的世界。

# **目录**

1.  安装数据表
2.  使用 Datasist structdata
3.  使用数据列表的特征工程
4.  使用 Datasist 实现数据可视化

不要浪费太多时间，让我们开始学习 Datasist 实用指南。

**安装数据表**

Datasist 是一个适用于 linux、MacOS 和 Windows 的 python 包，可以像任何其他 python 包一样安装。

如果激活了现有的 Python 环境，可以使用以下命令安装 datasist:

![](img/66add10549133277d748a4bddc40958f.png)

要在虚拟环境中安装 datasist，可以使用 Anaconda 包。

注意:你必须安装[巨蟒](http://www.anaconda.com)。

确认您已经安装了 Anaconda。您需要使用以下命令:

![](img/3b9cea5a843a930ec7bb499a8deb4d05.png)

现在开始创建一个新的虚拟环境，并安装最新版本的 Python。再说 Python 3.5 及以上。您将使用以下命令:

![](img/471198ebb806c0763e1186a329aa81c4.png)

下一步是你的环境。您将使用以下命令:

![](img/ea678ca9510429e19e24fcaa45ca73a7.png)

现在，使用以下命令安装 datasist:

![](img/50bdc0b58e0e9983cded010469c2f4b5.png)

下一步是使用以下命令测试您的安装:

![](img/264ed6b74d26b7748aadade5dff35e21.png)

现在我们已经完成了安装。让我们进行单子上的下一项议程。

**使用数据表结构数据**

你想象过吗？什么是数据列表？。嗯，我会说:Datasist 是一个 python 库，它使数据分析、可视化、清理变得容易，为数据科学家在原型开发期间做准备。

注意:我们将使用 Jupyter 笔记本，我们将使用尼日利亚数据科学 2020 hackhaton 数据集。准备好这一切后。让我们开始行动吧。

最后，打开 jupyter 笔记本，导入库和数据集，如下所示。

![](img/8401ed72098678cf89264193bcaf07f8.png)

structdata 模块有许多功能可以处理 Pandas DataFrame 格式的结构化数据。因此，您可以轻松地操作和分析数据帧。让我们深入研究一下可用的功能。

1.  描述:我们都知道熊猫也有描述的功能。让我们来看看 Datasist 描述方法。

![](img/22a578805c49bc212580a8df38cac3c5.png)

Lol，datastruct 模块显示以下输出:

(a)前五个数据点。

![](img/7002c213676e47dd8d61b105dda19cd9.png)

(b)随机五个数据点。

![](img/0fc4312ee256ab1d88aff8791bbf298e.png)

(c)最后五个数据点

![](img/d438cb375b1d309a1f6887e27f22b140.png)

(d)数据集的形状、数据集的大小

![](img/ff0973eadce96c31c9bbc86395e2ffb8.png)

(e)数据类型

![](img/3adf3167d6f9b07d9e840884bceb4647.png)

(f)数据集中的数字特征

![](img/3fa6e80ad32cb9ec5907ac908685c5ce.png)

(g)数据集中的分类特征

![](img/88b0fa0a1be8837ed84408791a95e5cc.png)

(h)栏目的统计说明

![](img/fef7ec4f8c29562ca27c03c7f62f71f0.png)

(一)分类特征描述

![](img/bc907230cd64baeb1996397987f4dac5.png)

(j)分类特征的唯一类别计数

![](img/58918108490548718d428336f3ba2ae2.png)

(k)数据中缺少数值

![](img/81844410c2ee2d21c44636e4bd76bf62.png)

是不是很神奇？。让我们来看看 datastruct 模块功能的另一个方面。

2.check_train_test_set:要使用这个函数，必须通过()train_df 和 test_df)函数。公共索引(申请者 ID)和两个数据集中可用的任何特征。让我们看一看。

![](img/f234c6c1f26be75f3c9740499eef0ef1.png)

现在，让我们看看我们的输出:

![](img/7beee93b3c98ed8da3ac152c5b3fbf56.png)

3.display_missing values:可以检查数据集中缺少的值。我们去看看。

![](img/4d441168caf100c57615dca35496cf40.png)

让我们看看输出给了我们什么:

![](img/6a03adec4b86700060d9f7111cedf520.png)

很神奇吧？

4.get_cats_feats 和 get_num_feats:这个函数用于检索分类和数字特征，并以列表的形式给出它们的输出。

让我们使用这个函数，看看它是什么样子的:

![](img/172417170a517566961d8ad67c14b8c4.png)

你看到了，对吧？。让我们检查数字特征。

![](img/af994b925a51773f3169aae7f6d9abfc.png)

这看起来很有趣，似乎很容易。

5.get_unique_counts:在一个分类特性中，你可以使用函数来获得唯一的类。下面我们来看看。

![](img/83f759c4cdb336eee6a655bfa33e34ee.png)

你看到这是怎么回事了吗？啊？非常简单明了。

6.join_train_test:您可以使用 function(join_train_and_test)函数来连接训练集和测试集。让我们做一些事情:

![](img/97de0bffeb1391469855b4caa7af4e5a.png)

datasist 中的 structdata 模块也有更多的功能。您可以阅读 datasist API 文档来了解更多信息。

**使用数据表的特征工程**

特征工程是使用数据的领域知识来创建使机器学习算法工作的特征的过程。它是从原始数据中提取重要特征并将其转换为适合机器学习的格式的行为。

现在，让我们探索一下 datasist 的一些特性工程模块。

1.  drop_missing:函数删除缺少值的要素。下面我们来看看:

![](img/b19b9d58efb4e406c79d406e2fd09f49.png)

让我们看看输出结果如何:

![](img/493855ef7c4f25012c6bb8d85224630c.png)![](img/7ff7bf72b2ecf827f9b35b0e6a5744a3.png)

2.drop_redundant:这个函数删除没有变化的特征。让我们创建一个新的数据集。让我们来看看:

![](img/5fe78b86e43086295ef496b9b2444d2f.png)

让我们看看输出是什么样的:

![](img/c7b7eb0e018385f740a281800ad4f6a4.png)

现在，检查数据集。你会发现它是多余的，这意味着它从头到尾都有相同的类。我们将使用该函数删除该列。检查以下内容:

![](img/97d95f4f6c0e1226dcb9f4029a4bd378.png)

输出是这样的:

![](img/1a8511b2a3c0eef4590dda4254c8fc86.png)

不对不对，酷。你看到了吗？

3.convert_dtypes:该函数自动获取 DataFrame 中没有在其正确类型上表示的特性。让我们来看看:

![](img/9c68c7ae8ef3149ad44613fcccf26639.png)

输出给出:

![](img/3185ba4672b817d8d74c3649a6718b87.png)

现在，让我们看看数据类型。

![](img/0a61ac4341e56d0e68c2018c7549adda.png)

注:年龄特征假设为整数。通过使用 convert_dtype 函数。它会自动修复。让我们来看看:

![](img/64b8af90ce63a3a00abf0639b5448cea.png)

4.fill_missing_cats:这个函数自动填充缺失的值。让我们检查一下:

![](img/9fc9943cfd11ee970cec6ce351fa84d0.png)![](img/69ed00dbcb71f6ad3f0c76097d025ba6.png)

5.fill_missing_values:此函数适用于数字特征，您可以指定填充策略(平均值、中值和众数)。让我们检查一下:

![](img/579e3d9ee6fee903841cbf69ca830e6d.png)

酷，你看到了吗？很简单。让我们深入研究下一个问题，即使用 datasist 实现可视化。

## **使用 datasist 进行可视化**

在开始行动之前，让我们重新导入之前使用的数据集。

![](img/dcf4c17542ebc5b21f7d8b8ffb9f9716.png)

输出是这样的:

![](img/3ae34cf6f511adcb5d7d1562b1a94985.png)

现在，我们将把我们的可视化分成两部分，即:

分类特征的可视化

(b)数字特征可视化

**类别特征的可视化**

分类特征的可视化包括:紫线图、计数图、箱线图等。现在，让我们开始一个接一个地接触它们。

*   countplot:这制作了一个所有分类特征的柱状图，以显示它们的类别计数。让我们检查一下:

![](img/410220425013ef04bbe1833cc7e1a3a9.png)

输出是这样的:

![](img/474daf2ede47c70203a05a954a78733f.png)![](img/01cb3e5742dc650eb3eef091f234d6b3.png)![](img/351b5ecd74064f8f59eb79c0743c09cc.png)

*   boxplot:针对分类目标列绘制所有数字特征的盒状图。看一看:

![](img/293adf47c6d8ef24563e0f6d1e2d92b5.png)![](img/6f32a4c4e5112fa741ec02407149433f.png)

*   catbox:用于针对分类目标绘制数据集中所有分类特征的并排条形图。让我们检查一下:

![](img/c5533367f9e40a51eabf176fdd0d8cbc.png)

让我们看看输出:

![](img/38807af23b092e80ca41a2f6abd5dcb7.png)

**数字特征可视化**

数字特征的可视化包括散点图、直方图、kde 图等。让我们开始行动吧:

*   plot_missing:该函数可用于可视化数据集中缺失的值。让我们来看看:

![](img/4d20899267cb04b67895e6552abbcfe0.png)

现在，让我们看看输出:

![](img/4bbcb4496b8f4cce3981077b9f648b43.png)

现在，我们完成了教程。现在，您将能够使用 datasist 解决数据分析和数据可视化的问题，例如(pandas、seaborn、matplotlib 和许多其他工具)。

注:本教程的笔记本可在[这里](https://github.com/AdesinaA/Introduction-to-datasist)获得。

感谢阅读。请随时在以下社交媒体上联系我: [Twitter](https://www.twitter.com/AdesinaAbdulra9) 、 [Linkedin](https://www.linkedin.com/in/adesinaabdulrahman\) 、 [Gmail](https://adesinaabdulrahman16@gmail.com) 、[脸书](https://www.facebook.com/adesinaabdulrahman16)。