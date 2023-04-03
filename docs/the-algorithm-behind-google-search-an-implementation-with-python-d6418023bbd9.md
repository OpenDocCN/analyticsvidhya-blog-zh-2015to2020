# 谷歌搜索背后的算法:用 Python 实现

> 原文：<https://medium.com/analytics-vidhya/the-algorithm-behind-google-search-an-implementation-with-python-d6418023bbd9?source=collection_archive---------4----------------------->

![](img/f12288405e0dcec0e8515d0848755c43.png)

PageRank (PR)是谷歌搜索使用的一种算法，用于根据网站的重要性在其搜索引擎结果中对其进行排名。但是“重要性”是什么意思呢？在这种情况下，根据谷歌的说法，它意味着一个页面的向内链接的数量和质量，事实上:

> “PageRank 的工作原理是通过计算一个页面的链接数量和质量来确定一个页面的链接质量……