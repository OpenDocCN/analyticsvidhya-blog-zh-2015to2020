# 图形卷积直觉

> 原文：<https://medium.com/analytics-vidhya/graph-convolution-intuition-9416c0f51167?source=collection_archive---------10----------------------->

TD；LR-给定一个用邻接矩阵表示的图形，它不能捕获卷积所需的空间相关性。这篇文章抓住了傅立叶域如何帮助卷积的直觉

图形表示的先决条件—[https://medium . com/analytics-vid hya/unbox ai-introduction-to-Graph-machine-learning-e4b 88514258 c](/analytics-vidhya/unboxai-introduction-to-graph-machine-learning-e4b88514258c)

# 为什么卷积

在我们深入研究图形卷积之前，让我们了解卷积在空间域中的作用——标准答案是“它捕捉潜在的空间……