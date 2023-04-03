# 概率校准要点(带代码)

> 原文：<https://medium.com/analytics-vidhya/probability-calibration-essentials-with-code-6c446db74265?source=collection_archive---------1----------------------->

在[的上一篇文章](/@rajneeshtiwari_22870/fallacy-of-using-auc-as-a-model-selection-criteria-d6bf50f4de0d)中，我描述了 AUC 指标的工作原理。如果你还没有通读这篇文章，这里有链接([链接](/@rajneeshtiwari_22870/fallacy-of-using-auc-as-a-model-selection-criteria-d6bf50f4de0d))。简单回顾一下，我详细讨论了 AUC 作为模型选择指标的用法，并快速查看了这样做的一些谬误。

在这篇文章中，我将展开同样的主题，并强调概率校准的概念。

需要记住的一点是 ***概率校准只有在我们对概率*** 感兴趣的时候才有用。一个这样的例子是，如果指标…