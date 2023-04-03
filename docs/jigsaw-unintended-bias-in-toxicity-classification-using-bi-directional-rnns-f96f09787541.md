# 双向 RNNs 在毒性分类中的 Jigsaw 非预期偏差

> 原文：<https://medium.com/analytics-vidhya/jigsaw-unintended-bias-in-toxicity-classification-using-bi-directional-rnns-f96f09787541?source=collection_archive---------16----------------------->

## EDA，最大限度地减少无意的偏见。

![](img/856d74a4163677999e5b774320659e3f.png)

照片由 [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral) 上的 [Prateek Katyal](https://unsplash.com/@prateekkatyal?utm_source=medium&utm_medium=referral) 拍摄

## 概观

由[竖锯](https://jigsaw.google.com/)和[谷歌](http://www.google.com)组成的对话人工智能团队在 [Kaggle](https://www.kaggle.com/) 举办了一场比赛，以检测评论中的毒性，并最大限度地减少文本/短语中的意外偏见，如男性、女性…