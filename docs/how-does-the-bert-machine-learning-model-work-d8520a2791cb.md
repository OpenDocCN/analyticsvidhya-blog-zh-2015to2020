# 伯特模型是如何工作的？

> 原文：<https://medium.com/analytics-vidhya/how-does-the-bert-machine-learning-model-work-d8520a2791cb?source=collection_archive---------11----------------------->

BERT 已经在 GitHub 上[开源，也上传到了](http://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) [TF Hub](https://tfhub.dev/) 。

我认为理解它的最好方法是玩它的代码。GitHub 上的[自述文件详细描述了它是什么以及它是如何工作的:](https://github.com/google-research/bert/blob/master/README.md)

**BERT—B**I directional**E**n coder**R**presentations from[Ttransformers](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)是一种预先训练语言表示的方法，这意味着我们在大型文本语料库(如维基百科)上训练一个通用的“语言理解”模型，然后将该模型用于我们关心的下游 NLP 任务(如问答)。伯特胜过…