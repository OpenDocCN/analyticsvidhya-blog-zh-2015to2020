# ε贪婪算法的多臂 Bandit 分析

> 原文：<https://medium.com/analytics-vidhya/multi-armed-bandit-analysis-of-epsilon-greedy-algorithm-8057d7087423?source=collection_archive---------4----------------------->

ε贪婪算法是决策科学背后的关键算法之一，体现了探索与利用之间的平衡。勘探与开发之间的两难境地可以简单地定义为:

*   剥削:根据你对环境的了解，选择具有最佳平均回报的选项/行动。
*   探索:认识到你对不同选项的了解可能是有限的，并选择参与那些可能会显示自己具有高价值的选项。