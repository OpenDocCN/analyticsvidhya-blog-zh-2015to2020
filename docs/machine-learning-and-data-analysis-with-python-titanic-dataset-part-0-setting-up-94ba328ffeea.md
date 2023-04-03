# 使用 Python 进行机器学习和数据分析，Titanic 数据集:第 0 部分，设置

> 原文：<https://medium.com/analytics-vidhya/machine-learning-and-data-analysis-with-python-titanic-dataset-part-0-setting-up-94ba328ffeea?source=collection_archive---------15----------------------->

这是用 Python 进行机器学习和数据分析系列文章的第 0 部分，是关于真实世界的例子，来自 Kaggle 的泰坦尼克号灾难数据集。这将是一系列视频，我将向您展示如何使用 Python、Pandas 和 SciKit Learn 进行机器学习和数据分析，解决现实世界的问题。在这个系列中，我将一步一步地介绍如何开始解决这样的问题。从数据探索和可视化、特征工程开始，然后建立模型进行预测。

第 0 部分将带您了解如何开始，包括设置环境和下载必要的数据。

如果你通过视频学得更好，我还做了一个视频展示所有的步骤，所以你可以照着做:

从 Kaggle 下载数据集:

[](https://www.kaggle.com/c/titanic/data) [## 泰坦尼克号:机器从灾难中学习

### 从这里开始！预测泰坦尼克号上的生存并熟悉 ML 基础知识

www.kaggle.com](https://www.kaggle.com/c/titanic/data) 

转到数据，单击全部下载。将它放在容易找到的地方并安装依赖项。我们将使用 Jupyter Notebook，我将向您展示如何设置该环境。

最简单的方法是下载 Anaconda:

[](https://www.anaconda.com/distribution/) [## Anaconda Python/R 发行版-免费下载

### 开源的 Anaconda 发行版是在…上执行 Python/R 数据科学和机器学习的最简单的方法

www.anaconda.com](https://www.anaconda.com/distribution/) 

一定要下载 Python 3.7 版本(我假设你有 Python 并且熟悉 Python 的一些基础知识，如果没有，在这里下载:[https://www.python.org/downloads/](https://www.python.org/downloads/))。

从你的应用程序文件夹启动 Anaconda Navigator，你会在 Jupyter Notebook 旁边看到一个安装按钮。点击安装，然后启动。

在 Jupyter Notebook 中，您可以看到根目录下的所有文件和文件夹。转到存储下载文件夹的位置。首先，您应该只能看到 3 个文件:

*   性别 _ 提交. csv
*   train.csv
*   test.csv

这些是我们将要处理的文件。train.csv 文件将包含我们稍后将用作训练数据的内容。test.csv 文件将与 train.csv 几乎完全相同，除了它将缺少一列，这是我们试图预测的基础真值。gender_submission.csv 文件是一个示例提交文件，其中幸存的列被硬编码为 0，表示该乘客的性别是男性，如果该乘客是女性，则为 1。当我们想要提交我们的预测时，我们必须把它做成一个相同格式的 csv 文件。我稍后会谈到如何做到这一点(或者如果你检查我的 YouTube 播放列表，视频现在应该已经准备好了)。

现在我们要创建一个新文件，通过点击右上角的“新建”标签创建一个新的 Python3 文件。给它起个名字。

运行以下命令，确保您已经安装了我们将使用的库:

*   进口熊猫作为 pd
*   导入 sklearn

如果您没有这些库，您可以使用 pip 下载它们。

在本系列的第 1 部分中，我将向您展示如何通过可视化来理解数据集:

[](/@qinchen.wang.personal/machine-learning-and-data-analysis-with-python-titanic-dataset-part-1-visualization-8a6e80732dd3) [## 用 Python 进行机器学习和数据分析，泰坦尼克号数据集:第 1 部分，可视化

### 每一个伟大的机器学习和数据科学项目都是从定义问题开始的:你必须处理哪些数据…

medium.com](/@qinchen.wang.personal/machine-learning-and-data-analysis-with-python-titanic-dataset-part-1-visualization-8a6e80732dd3)