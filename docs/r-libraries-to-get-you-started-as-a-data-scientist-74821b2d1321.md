# r 库，帮助您成为一名数据科学家

> 原文：<https://medium.com/analytics-vidhya/r-libraries-to-get-you-started-as-a-data-scientist-74821b2d1321?source=collection_archive---------30----------------------->

如果您是数据科学领域的初学者，并且对学习如何使用 R 编程来解决数据科学问题感兴趣，本文提供了一个必须了解的 R 库列表，它将帮助您入门。r 是数据科学中最流行的编程语言之一，与 Python、Java 和 Scala 齐名。首先我会推荐安装 RStudio，R 编程的 IDE(集成开发环境)。您将在 RStudio 中获得所有必需的功能，如控制台、代码编辑器、调试器以及用于绘制数据集和查看历史的工具。

如果你是许多试图自学数据科学的人之一，你需要访问数据集。Kaggle 有大量有趣的数据集可供你使用。现在您已经安装了 RStudio 并选择了数据集，让我们来谈谈一些重要的 R 库，让您开始学习。

![](img/77c80500c6ef19feb3f3337082b344aa.png)

由[马库斯·斯皮斯克](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍摄的照片

# **ggplot2**

ggplot2 是图形文法的实现。ggplot2 是 r 中主要的数据可视化库。可视化是数据科学和分析的重要组成部分。在 ggplot2 的帮助下，您可以轻松地为数值和分类数据绘制单变量或多变量的静态图。您还可以按大小、颜色和符号对数据进行分组。为了安装和加载 ggplot2，在 R 控制台中分别运行以下命令:

> install.packages("ggplot2 ")
> 
> 库(ggplot2)

# **数据表**

数据帧主要用于在 R 中存储任何类型的数据。数据表是一个 R 包，它提供了数据帧的增强版本。数据表中的两个增强是速度和更清晰的语法。数据表能够比数据帧更快地处理连接、索引、赋值和分组。这是因为数据帧不必要地复制了整个数据。因此，当数据集很大(如超过 10 GBs)时，建议使用数据表。

> install.packages("data.table ")
> 
> 库(数据.表格)

# Dplyr

Dplyr 是基本的数据操作 R 包。它以动词的形式提供函数，通常与 group_by()函数结合使用，以执行以下类型的数据操作:

mutate()select()filter()summary()array()

以我个人的经验，dplyr 是数据突变最常用的包之一。您可以用同样的方式安装这个软件包:

> install.packages("dplyr ")
> 
> 图书馆(dplyr)

# Tidyr

顾名思义，R 中的这个包是用来整理/清理数据的。该软件包最适合于每行代表一个观察值、列代表一个特征/变量的数据。Tidyr 在清理数据时非常方便，它使用了“fill()”和“replace_na()”这样的函数，前者填充缺失的单元格，后者用您选择的值替换缺失的值。Tidyr 的一些最重要的函数是 gather()、separate()和 spread()。

> install.package("tidyr ")
> 
> 图书馆(tidyr)

![](img/18f17d52c4176026aa5e9905bbbd09cf.png)

[安东](https://unsplash.com/@uniqueton?utm_source=medium&utm_medium=referral)在 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上拍照

# 脱字号

Caret 是分类和回归训练的缩写。Caret 提供了在分类和回归问题中训练机器学习模型的函数。Caret 帮助您简化以下步骤:

**数据预处理**:预处理()函数帮助检查缺失数据

**数据分割:**将数据分割成训练集和测试集，例如，使用 createDataPartition()函数

**训练模型:** Caret 提供了大量各种各样的机器学习算法。你可以在这里看一看[http://topepo.github.io/caret/available-models.html](http://topepo.github.io/caret/available-models.html)

Caret 的其他重要功能是特征选择、参数调整和变量重要性估计，以及您可以创建自己的模型。

要安装 caret，您可以使用以下命令:

> install.packages("caret "，dependencies=c("Depends "，" Suggests "))

还有其他几个重要的 R 包要知道，但这是另一个故事了。