# 用单词嵌入生成(简单的)纵横字谜

> 原文：<https://medium.com/analytics-vidhya/generating-simple-crosswords-with-word-embeddings-108aa2d2f0d7?source=collection_archive---------30----------------------->

单词嵌入很酷，如果你不相信我，看看[他们能做的所有很酷的事情](/search?q=word%20embeddings)！

纵横字谜也很酷，如果你不相信我，可以问问[比尔·克林顿、乔恩·斯图尔特和本·伯恩斯](https://www.imdb.com/title/tt0492506/)。

虽然生成 NYT 质量的网格和线索超出了这个简单练习的范围，但我们可以生成你会在 NYT 的“[迷你](https://www.nytimes.com/crosswords/game/mini)”中找到的那种谜题，当你有几分钟空闲时可以填充 5×5 的网格。