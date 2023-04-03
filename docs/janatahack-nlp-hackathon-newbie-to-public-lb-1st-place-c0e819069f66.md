# JanataHack NLP 黑客马拉松:公共 LB 第一名的新手

> 原文：<https://medium.com/analytics-vidhya/janatahack-nlp-hackathon-newbie-to-public-lb-1st-place-c0e819069f66?source=collection_archive---------19----------------------->

隔离时间还在继续，有效地度过一天是我脑海中唯一的目标。我偶然发现了 JanataHack NLP 黑客马拉松，想试一试。问题陈述是为了预测，评论者是否根据评论文本和其他信息推荐了特定的游戏。

这个问题引起了我的兴趣，尽管我以前在 NLP 方面没有任何经验。

![](img/62ba4eefab19daf6335bb30e88d3ae4f.png)

由[卡拉·里维拉](https://unsplash.com/@karla_rivera?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)在 [Unsplash](https://unsplash.com/t/athletics?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) 上拍摄

在关于 NLP 的文章中，我读到了 N-grams、TF-IDF 和许多其他传统的 NLP 技术。然后我偶然发现了杰瑞米·霍华德的 [fastai 讲座视频](https://course.fast.ai/videos/?lesson=4)，他在视频中谈到使用 fastai 采用深度学习方法来解决 NLP 问题，也强调了迁移学习的使用。

我们得到了 64 个游戏的数据集，其中包括 review_texts.csv(包含游戏评论)和 game_overview.csv(包含每个游戏的概述)。

# 第一次

我们开始尝试在 WikiText-103 数据集上预处理的 LSTM 模型上应用 Jeremy 的迁移学习技术。使用 game_overview.csv 将 LSTM 模型微调为语言模型，并在训练数据集上训练为分类模型。这种方法让我们获得了 **~0.83 磅**(排行榜分数)。在实验过程中，局部 CV(交叉验证分数)与 LB 密切相关。

# 第二次

为了进一步提高我们的 LB 排名，我们学习了预先训练的模型架构，如 BERT、GPT-2、罗伯塔、XLM、DistilBert、XLNet、T5、CTRL 等。通过[拥抱面部变形金刚](https://github.com/huggingface/transformers)。

[**BERT**](https://arxiv.org/abs/1810.04805)**是一个双向转换器，用于对大量未标记的文本数据进行预训练，以学习一种可用于微调特定机器学习任务的语言表示。**

**使用预先训练好的 BERT(基本无壳)，并遵循 fastai 的一次拟合循环方法，这很快使我们获得了 **~0.91 磅**，这比我们之前的成绩有了巨大的提高。**

# **最后阶段**

**[**罗伯塔**](https://arxiv.org/pdf/1907.11692.pdf) **。【RoBERTa 是在脸书大学推出的稳健优化的 BERT 方法，它是对 BERT 的再培训，具有改进的培训方法、1000%以上的数据和计算能力。重要的是，RoBERTa 使用 160 GB 的文本进行预训练，包括 16GB 的书籍语料库和 BERT 中使用的英语维基百科。附加数据包括 [CommonCrawl 新闻数据集](http://web.archive.org/web/20190904092526/http://commoncrawl.org/2016/10/news-dataset-available/)(6300 万篇文章，76 GB)，网络文本语料库(38 GB)，以及来自 CommonCrawl 的故事(31 GB)。****

**换成 RoBERTa (base)后提升了约 0.01 磅，我们的体重为 **~0.925 磅**。LB 仍然与 CV 完全相关。**

**RoBERTa(大号)大幅提升了约 0.015 磅，将我们推至约 0.9397 磅(T11)。这是我们最好的单一模型得分。
我们的最佳 LB 分数([公众第一名](https://datahack.analyticsvidhya.com/contest/janatahack-nlp-hackathon/#LeaderBoard))为 **0.94235** 是 4 个 RoBERTa(大)模特的合奏。**

> **是什么让我们更上一层楼……**

*   **我们没有使用训练集中的所有训练信息，只是将“user_reviews”列输入到模型中。使用所有的培训信息可能会有所帮助。**
*   **为更多的时代而训练。我们训练每个罗伯塔最多 6 个时期。**
*   **不同型号的组合。我们只集合了基于 RoBERTa 的模型。**

**这是我们为黑客马拉松处理上述问题陈述的整体方法。[这里是](https://github.com/nainci/JanataHack-NLP-Hackathon)GitHub 回购。**

**作为初学者，我们欢迎您的所有建议和评论。请在下面留下评论，以便进一步讨论。**