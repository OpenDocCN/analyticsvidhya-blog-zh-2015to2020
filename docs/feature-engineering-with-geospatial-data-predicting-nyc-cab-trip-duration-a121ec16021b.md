# 使用地理空间数据的要素工程:预测纽约市出租车出行持续时间

> 原文：<https://medium.com/analytics-vidhya/feature-engineering-with-geospatial-data-predicting-nyc-cab-trip-duration-a121ec16021b?source=collection_archive---------3----------------------->

由于新冠肺炎，移动数据最近变得越来越受欢迎，所以我想研究一个涉及地理空间数据的预测问题。我决定参加[纽约市出租车出行持续时间 Kaggle 竞赛](https://www.kaggle.com/c/nyc-taxi-trip-duration)，其目标是在给定主要地理空间和时间特征的情况下，预测纽约市出租车出行的持续时间。

使用 LightGBM 模型，我能够获得 0.38109 的 RMSLE 分数，这将使我在[公共排行榜](https://www.kaggle.com/c/nyc-taxi-trip-duration/leaderboard#score)的 1254 个条目中排名第 177 位(但 Kaggle 不会晚发布…