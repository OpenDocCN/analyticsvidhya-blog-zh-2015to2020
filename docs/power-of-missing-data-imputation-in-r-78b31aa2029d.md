# 简单随机抽样中缺失数据填补技术的比较，Mean 与 MICE 随机森林填补

> 原文：<https://medium.com/analytics-vidhya/power-of-missing-data-imputation-in-r-78b31aa2029d?source=collection_archive---------3----------------------->

# 1.介绍

缺失数据是几乎所有数据集中都会出现的一种常见现象。如果大部分数据丢失，这可能是一个严重的问题。统计学的经验法则是，如果丢失的数据少于 5%,那么具有丢失值的行将从统计分析中删除。假设 30%的数据至少有一列缺少值，该怎么办？此外，错过的类型也很重要…